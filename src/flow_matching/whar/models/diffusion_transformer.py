import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from flow_matching.supervised.odes_sdes import ConditionalVectorField


# -------------------------------------------------
# Sinusoidal frequency embedding for timesteps
# -------------------------------------------------
def timestep_sinusoidal_embedding(timesteps: Tensor, time_dim: int) -> Tensor:
    """
    timesteps: Tensor of shape (B, ...) (any shape, e.g., (B, 1, 1, 1))
    time_dim: int, must be even
    returns: Tensor of shape (B, time_dim) by flattening extra dims
    """
    assert time_dim % 2 == 0
    half = time_dim // 2
    device = timesteps.device

    # flatten to (B, 1)
    timesteps = timesteps.view(timesteps.shape[0], -1).float()  # (B, 1)

    # compute frequencies
    freqs = torch.exp(
        -math.log(10000.0) * torch.arange(half, device=device) / half
    )  # (half,)

    # compute args (B, half)
    args = timesteps * freqs.unsqueeze(0)  # broadcasting

    # concatenate sin and cos
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)  # (B, time_dim)

    return emb


def pos_embed_2d(gh: int, gw: int, embed_dim: int, device: torch.device) -> Tensor:
    # --- build 2D sinusoidal positional embeddings dynamically ---
    coords_h = torch.arange(gh, device=device)
    coords_w = torch.arange(gw, device=device)
    gy, gx = torch.meshgrid(coords_h, coords_w, indexing="ij")  # (gh, gw)
    coords = torch.stack([gx, gy], dim=-1).float()  # (gh, gw, 2)

    omega = torch.exp(
        torch.arange(embed_dim // 4, device=device).float()
        * -(math.log(10000.0) / (embed_dim // 4))
    )  # (embed_dim//4,)

    out_x = coords[..., 0:1] * omega  # (gh, gw, embed_dim//4)
    out_y = coords[..., 1:2] * omega  # (gh, gw, embed_dim//4)

    pos_emb = torch.cat(
        [torch.sin(out_x), torch.cos(out_x), torch.sin(out_y), torch.cos(out_y)],
        dim=-1,
    )  # (gh, gw, embed_dim)

    pos_emb = pos_emb.permute(2, 0, 1).unsqueeze(0)  # (1, embed_dim, gh, gw)

    return pos_emb


# -------------------------------------------------
# Adaptive LayerNorm (no affine)
# -------------------------------------------------
class AdaptiveLayerNorm(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.ln = nn.LayerNorm(dim, elementwise_affine=False)

    def forward(self, x: Tensor, gamma: Tensor, beta: Tensor) -> Tensor:
        return self.ln(x) * (1 + gamma.unsqueeze(1)) + beta.unsqueeze(1)


# -------------------------------------------------
# FeedForward
# -------------------------------------------------
class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, num_layers: int = 2) -> None:
        super().__init__()
        layers = [nn.Linear(dim, hidden_dim)]

        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
        layers.append(nn.Linear(hidden_dim, dim))

        self.layers = nn.ModuleList(layers)

    def forward(self, x: Tensor) -> Tensor:
        for i, layer in enumerate(self.layers):
            x = layer(x)
            x = F.silu(x) if i < len(self.layers) - 1 else x
        return x


# -------------------------------------------------
# AdaLN-Zero Block
# -------------------------------------------------
class AdaLNZeroBlock(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        cond_dim: int,
        mlp_mult: int = 4,
        mlp_layers: int = 2,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim

        self.cond_proj = nn.Sequential(nn.SiLU(), nn.Linear(cond_dim, 6 * embed_dim))
        nn.init.zeros_(self.cond_proj[-1].weight)  # type: ignore
        nn.init.zeros_(self.cond_proj[-1].bias)  # type: ignore

        self.ln1 = AdaptiveLayerNorm(embed_dim)
        self.mhsa = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.mhsa_out = nn.Linear(embed_dim, embed_dim)

        self.ln2 = AdaptiveLayerNorm(embed_dim)
        self.mlp = FeedForward(embed_dim, embed_dim * mlp_mult, mlp_layers)

    def forward(self, x: Tensor, cond: Tensor) -> Tensor:
        g1, b1, a1, g2, b2, a2 = self.cond_proj(cond).split(self.embed_dim, dim=1)

        h = self.ln1(x, g1, b1)
        h = self.mhsa(h, h, h, need_weights=False)[0]
        x = x + a1.unsqueeze(1) * self.mhsa_out(h)

        h = self.ln2(x, g2, b2)
        h = self.mlp(h)
        x = x + a2.unsqueeze(1) * self.mlp(h)

        return x


# -------------------------------------------------
# DiT Model with Conv2d patch embedding
# -------------------------------------------------
class DiT(ConditionalVectorField):
    def __init__(
        self,
        channels: int = 18,
        patch_size: int = 4,
        embed_dim: int = 384,
        time_dim: int = 256,
        num_heads: int = 6,
        num_blocks: int = 12,
        mlp_mult: int = 4,
        mlp_layers: int = 2,
        num_classes: int = 6,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.time_dim = time_dim

        self.patch_embed = nn.Conv2d(channels, embed_dim, patch_size, patch_size)

        # embeddings
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, embed_dim), nn.SiLU(), nn.Linear(embed_dim, embed_dim)
        )
        self.class_embed = nn.Embedding(num_classes, embed_dim)

        # transformer blocks
        self.blocks = nn.ModuleList(
            [
                AdaLNZeroBlock(
                    embed_dim,
                    num_heads,
                    cond_dim=embed_dim,
                    mlp_mult=mlp_mult,
                    mlp_layers=mlp_layers,
                )
                for _ in range(num_blocks)
            ]
        )

        # final norm
        self.final_cond_proj = nn.Sequential(
            nn.SiLU(), nn.Linear(embed_dim, 2 * embed_dim)
        )
        self.final_ln = AdaptiveLayerNorm(embed_dim)
        self.decoder = nn.ConvTranspose2d(
            embed_dim, channels, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x: Tensor, t: Tensor, y: Tensor) -> Tensor:
        B, C, H, W = x.shape
        gh, gw = H // self.patch_size, W // self.patch_size  # grid size

        # --- patchify ---
        z = self.patch_embed(x)  # (B, embed_dim, gh, gw)
        # print(f"z shape: {z.shape}")

        # --- positional embedding ---
        z = z + pos_embed_2d(gh, gw, self.embed_dim, x.device)  # (B, embed_dim, gh, gw)
        # print(f"z shape + pos: {z.shape}")

        # --- flatten for transformer ---
        z = z.flatten(2).transpose(1, 2)  # (B, gh*gw, embed_dim)
        # print(f"z shape + flatten: {z.shape}")

        # --- time & class embeddings ---
        # print(t.shape)
        t_emb = timestep_sinusoidal_embedding(t, self.time_dim)
        # print(t_emb.shape)
        t_emb = self.time_mlp(t_emb)
        c_emb = self.class_embed(y)
        cond = t_emb + c_emb  # (B, embed_dim)

        # --- transformer blocks ---
        for block in self.blocks:
            z = block(z, cond)  # (B, gh*gw, embed_dim)

        # --- final conditioning ---
        g, b = self.final_cond_proj(cond).split(self.embed_dim, dim=1)
        z = self.final_ln(z, g, b)

        # --- reshape and decode ---
        z = z.transpose(1, 2).reshape(B, self.embed_dim, gh, gw)
        # (B, embed_dim, gh, gw)
        z = self.decoder(z)  # (B, channels, H, W)

        return z


# -------------------------------------------------
# Smoke test
# -------------------------------------------------
if __name__ == "__main__":
    B, C, H, W, p = 2, 18, 32, 32, 2
    num_classes = 6
    model = DiT(
        channels=C,
        patch_size=p,
        embed_dim=384,
        num_heads=6,
        num_blocks=12,
        num_classes=num_classes,
    )
    x = torch.randn(B, C, H, W)
    t = torch.randint(0, 1000, (B,))
    y = torch.randint(0, num_classes, (B,))
    out = model(x, t, y)
    print("in shape:", x.shape)
    print("out shape:", out.shape)
