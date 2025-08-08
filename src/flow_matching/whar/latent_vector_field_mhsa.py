import torch
import torch.nn as nn
import math
from torch import Tensor

from flow_matching.supervised.odes_sdes import ConditionalVectorField


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, embedding_dim: int) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim

    def forward(self, t: Tensor) -> Tensor:
        half_dim = self.embedding_dim // 2
        emb = torch.exp(
            torch.arange(half_dim, device=t.device) * -math.log(10000) / (half_dim - 1)
        )
        emb = t.unsqueeze(1) * emb  # [B, half_dim]
        return torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)  # [B, embedding_dim]


class MHSAOverTime(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.mha = nn.MultiheadAttention(
            embed_dim=dim, num_heads=num_heads, batch_first=True
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        # x: [B, C, H, W]
        B, C, H, W = x.shape

        x_ = x.permute(0, 2, 3, 1).reshape(B * H, W, C)  # (B*H, W, C)

        x_norm = self.norm(x_)
        attn_out, _ = self.mha(x_norm, x_norm, x_norm)
        x_ = x_ + attn_out

        x_ = x_.view(B, H, W, C).permute(0, 3, 1, 2)  # back to [B, C, H, W]

        return x_


class MHSAOverFreq(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.mha = nn.MultiheadAttention(
            embed_dim=dim, num_heads=num_heads, batch_first=True
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        # x: [B, C, H, W]
        B, C, H, W = x.shape

        x_ = x.permute(0, 3, 2, 1).reshape(B * W, H, C)  # (B*W, H, C)

        x_norm = self.norm(x_)
        attn_out, _ = self.mha(x_norm, x_norm, x_norm)
        x_ = x_ + attn_out

        x_ = x_.view(B, W, H, C).permute(0, 3, 2, 1)  # back to [B, C, H, W]

        return x_


class AxialMHSABlockSimple(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.mhsa_time = MHSAOverTime(dim, num_heads)
        self.mhsa_freq = MHSAOverFreq(dim, num_heads)

        self.norm = nn.LayerNorm(dim)

        self.ff = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, x):
        x = self.mhsa_time(x)
        x = self.mhsa_freq(x)

        x_ = x.permute(0, 2, 3, 1).contiguous()  # (B, H, W, C)

        x_ = self.norm(x_)
        ff_out = self.ff(x_)
        x_ = x_ + ff_out

        x = x_.permute(0, 3, 1, 2).contiguous()

        return x


class FlowNetBackboneAxial(ConditionalVectorField):
    def __init__(
        self,
        num_classes: int,
        latent_channels: int = 64,
        time_emb_dim: int = 128,
        num_heads=8,
    ) -> None:
        super().__init__()
        self.time_embed = SinusoidalTimeEmbedding(time_emb_dim)
        self.fc_time = nn.Sequential(
            nn.Linear(time_emb_dim, latent_channels * 2),  # scale + shift for time
        )

        self.label_embed = nn.Embedding(num_classes + 1, 128)
        self.fc_label = nn.Sequential(
            nn.Linear(128, latent_channels * 2),  # scale + shift for label
        )

        self.encoder = nn.Sequential(
            AxialMHSABlockSimple(latent_channels, num_heads),
            AxialMHSABlockSimple(latent_channels, num_heads),
        )
        self.decoder = nn.Sequential(
            AxialMHSABlockSimple(latent_channels, num_heads),
            nn.Conv2d(latent_channels, latent_channels, kernel_size=3, padding=1),
        )

    def forward(self, x: Tensor, t: Tensor, y: Tensor) -> Tensor:
        B, C, H, W = x.shape

        time_emb = self.time_embed(t.view(B))  # [B, time_emb_dim]
        time_mod = self.fc_time(time_emb)  # [B, 2*C]
        scale_time = time_mod[:, :C].view(B, C, 1, 1)
        shift_time = time_mod[:, C:].view(B, C, 1, 1)

        label_emb = self.label_embed(y)  # [B, 128]
        label_mod = self.fc_label(label_emb)  # [B, 2*C]
        scale_label = label_mod[:, :C].view(B, C, 1, 1)
        shift_label = label_mod[:, C:].view(B, C, 1, 1)

        # Combine time and label FiLM: scale and shift multiply/add
        scale = 1 + scale_time + scale_label
        shift = shift_time + shift_label

        x_mod = x * scale + shift

        h = self.encoder(x_mod)
        v = self.decoder(h)
        return v
