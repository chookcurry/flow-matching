import math
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

from flow_matching.supervised.odes_sdes import ConditionalVectorField


# -----------------------------
# Positional & time embeddings
# -----------------------------
class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, embedding_dim: int) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim

    def forward(self, t: Tensor) -> Tensor:
        """
        t: Tensor of shape [B] or [B, 1]
        Returns: [B, embedding_dim]
        """
        if t.dim() == 2:
            t = t.squeeze(-1)
        half_dim = self.embedding_dim // 2
        emb_scale = -math.log(10000.0) / max(half_dim - 1, 1)
        freqs = torch.exp(torch.arange(half_dim, device=t.device) * emb_scale)
        args = t[:, None] * freqs[None, :]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if emb.shape[-1] < self.embedding_dim:
            # pad if odd dim requested
            emb = nn.functional.pad(emb, (0, self.embedding_dim - emb.shape[-1]))
        return emb


def build_2d_sincos_pos_embed(h: int, w: int, dim: int, device=None) -> Tensor:
    """Create 2D sine-cosine positional embeddings as in ViT/MAE.
    Returns: [H*W, dim]
    """
    assert dim % 4 == 0, "positional embedding dim must be divisible by 4"
    device = device or torch.device("cpu")

    grid_y = torch.arange(h, device=device, dtype=torch.float32)
    grid_x = torch.arange(w, device=device, dtype=torch.float32)
    grid = torch.meshgrid(grid_y, grid_x, indexing="ij")  # [2, H, W]
    grid_stack = torch.stack(grid, dim=0)  # [2, H, W]

    dim_half = dim // 2
    dim_quarter = dim // 4

    def get_embed(pos, d_model):
        omega = torch.arange(d_model // 2, device=device, dtype=torch.float32)
        omega = 1.0 / (10000 ** (omega / (d_model // 2)))
        out = pos.flatten()[:, None] * omega[None, :]
        return torch.cat([torch.sin(out), torch.cos(out)], dim=1)

    emb_y = get_embed(grid_stack[0], dim_half)  # [H*W, dim/2]
    emb_x = get_embed(grid_stack[1], dim_half)  # [H*W, dim/2]
    pos = torch.cat([emb_y[:, : dim_quarter * 2], emb_x[:, : dim_quarter * 2]], dim=1)
    # ensure exact dim
    if pos.shape[1] < dim:
        pos = nn.functional.pad(pos, (0, dim - pos.shape[1]))
    elif pos.shape[1] > dim:
        pos = pos[:, :dim]
    return pos  # [H*W, dim]


# -----------------------------
# AdaLayerNorm (FiLM-style)
# -----------------------------
class AdaLayerNorm(nn.Module):
    """LayerNorm with learned affine from conditioning vector.
    Implements x -> ln(x) * (1 + gamma) + beta, where [gamma, beta] are linear in context.
    """

    def __init__(self, hidden_dim: int, cond_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.to_gamma_beta = nn.Linear(cond_dim, hidden_dim * 2)
        nn.init.zeros_(self.to_gamma_beta.weight)
        nn.init.zeros_(self.to_gamma_beta.bias)

    def forward(self, x: Tensor, cond: Tensor) -> Tensor:
        # x: [B, N, D], cond: [B, cond_dim]
        g, b = self.to_gamma_beta(cond).chunk(2, dim=-1)
        return self.norm(x) * (1 + g.unsqueeze(1)) + b.unsqueeze(1)


class TransformerBlock(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        n_heads: int,
        mlp_ratio: float = 4.0,
        cond_dim: int = 0,
        attn_dropout: float = 0.0,
        resid_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.ada_ln_1 = (
            AdaLayerNorm(hidden_dim, cond_dim)
            if cond_dim > 0
            else nn.LayerNorm(hidden_dim)
        )
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=n_heads,
            dropout=attn_dropout,
            batch_first=True,
        )
        self.drop_path_attn = nn.Dropout(resid_dropout)

        self.ada_ln_2 = (
            AdaLayerNorm(hidden_dim, cond_dim)
            if cond_dim > 0
            else nn.LayerNorm(hidden_dim)
        )
        mlp_hidden = int(hidden_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, mlp_hidden),
            nn.GELU(),
            nn.Linear(mlp_hidden, hidden_dim),
        )
        self.drop_path_mlp = nn.Dropout(resid_dropout)

        self.cond_dim = cond_dim

    def forward(self, x: Tensor, cond: Optional[Tensor] = None) -> Tensor:
        # x: [B, N, D]
        if self.cond_dim > 0:
            x_norm = self.ada_ln_1(x, cond)
        else:
            x_norm = self.ada_ln_1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm, need_weights=False)
        x = x + self.drop_path_attn(attn_out)

        if self.cond_dim > 0:
            x_norm = self.ada_ln_2(x, cond)
        else:
            x_norm = self.ada_ln_2(x)
        x = x + self.drop_path_mlp(self.mlp(x_norm))
        return x


# -----------------------------
# Flow Transformer Backbone
# -----------------------------
class FlowTransformerBackbone(ConditionalVectorField):
    """
    Transformer denoiser operating on global tokens with 2D sinusoidal positional embeddings.

    Input latent shape: [B, C, H, W]
    Tokens: N = H*W, each projected from C -> hidden_dim.
    Conditioning: time embedding (sinusoidal) and class label embedding → combined context used in AdaLayerNorm per block.
    Number of transformer blocks is configurable.
    """

    def __init__(
        self,
        num_classes: int,
        latent_channels: int = 20,
        time_emb_dim: int = 64,
        hidden_dim: int = 64,
        num_heads: int = 4,
        num_blocks: int = 4,
        mlp_ratio: float = 4.0,
        attn_dropout: float = 0.0,
        resid_dropout: float = 0.0,
        use_class_condition: bool = True,
    ) -> None:
        super().__init__()
        self.latent_channels = latent_channels
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_blocks = num_blocks
        self.use_class_condition = use_class_condition

        # Time & label conditioning
        self.time_embed = SinusoidalTimeEmbedding(time_emb_dim)
        cond_in_dim = time_emb_dim
        if use_class_condition:
            self.label_embed = nn.Embedding(num_classes, hidden_dim)
            cond_in_dim += hidden_dim
        self.cond_mlp = nn.Sequential(
            nn.Linear(cond_in_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Input and output projections
        self.in_proj = nn.Linear(latent_channels, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, latent_channels)

        # Transformer blocks
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    hidden_dim=hidden_dim,
                    n_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    cond_dim=hidden_dim,
                    attn_dropout=attn_dropout,
                    resid_dropout=resid_dropout,
                )
                for _ in range(num_blocks)
            ]
        )

        # Final norm before projecting back
        self.final_ln = nn.LayerNorm(hidden_dim)

        # Cache for positional embeddings per (H, W)
        self._pos_cache: dict = {}

    @torch.no_grad()
    def _get_pos_embed(self, H: int, W: int, device) -> Tensor:
        key = (H, W, device)
        if key not in self._pos_cache:
            pos = build_2d_sincos_pos_embed(H, W, self.hidden_dim, device=device)
            self._pos_cache[key] = pos
        return self._pos_cache[key]

    def forward(self, x: Tensor, t: Tensor, y: Optional[Tensor] = None) -> Tensor:
        """
        x: [B, C=latent_channels, H, W]
        t: [B] or [B, 1]
        y: [B] (class indices) or None if unconditional
        Returns: velocity/score field with same shape as x
        """
        B, C, H, W = x.shape
        assert C == self.latent_channels, (
            f"Expected {self.latent_channels} channels, got {C}"
        )

        # Tokens: [B, N, C]
        tokens = x.permute(0, 2, 3, 1).reshape(B, H * W, C)
        tokens = self.in_proj(tokens)  # [B, N, D]

        # Add 2D positional embeddings (fixed sin-cos)
        pos = self._get_pos_embed(H, W, x.device)  # [N, D]
        tokens = tokens + pos.unsqueeze(0)

        # Build conditioning vector [B, hidden_dim]
        t_emb = self.time_embed(t.view(B))  # [B, time_emb_dim]
        if self.use_class_condition and y is not None:
            y_emb = self.label_embed(y)  # [B, hidden_dim]
            cond = torch.cat([t_emb, y_emb], dim=-1)
        else:
            cond = t_emb
        cond = self.cond_mlp(cond)  # [B, hidden_dim]

        # Transformer stack
        h = tokens
        for blk in self.blocks:
            h = blk(h, cond)

        h = self.final_ln(h)
        out = self.out_proj(h)  # [B, N, C]
        out = out.reshape(B, H, W, C).permute(0, 3, 1, 2).contiguous()
        return out  # velocity field
