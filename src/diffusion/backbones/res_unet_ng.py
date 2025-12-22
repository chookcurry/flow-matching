import math
from typing import List, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from diffusion.backbones.backbone import Backbone


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        assert dim % 2 == 0
        self.half_dim = dim // 2

    def forward(self, t: Tensor) -> Tensor:
        t = t.view(-1, 1)
        freqs = torch.exp(
            -math.log(10000)
            * torch.arange(0, self.half_dim, dtype=torch.float32, device=t.device)
            / self.half_dim
        )
        angles = t * freqs
        return torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)


class AdaGroupNorm(nn.Module):
    def __init__(self, num_groups: int, num_channels: int, cond_dim: int) -> None:
        super().__init__()
        self.group_norm = nn.GroupNorm(num_groups, num_channels, affine=False, eps=1e-6)
        self.linear = nn.Linear(cond_dim, 2 * num_channels)

        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x: Tensor, cond: Tensor) -> Tensor:
        gamma, beta = self.linear(cond).chunk(2, dim=1)
        gamma = gamma.view(gamma.shape[0], gamma.shape[1], 1, 1)
        beta = beta.view(beta.shape[0], beta.shape[1], 1, 1)
        x = self.group_norm(x) * (1 + gamma) + beta
        return x


class AttentionBlock(nn.Module):
    def __init__(self, channels: int, num_heads: int = 4):
        super().__init__()
        self.num_heads = num_heads
        self.norm = nn.GroupNorm(8, channels)
        self.qkv = nn.Linear(channels, channels * 3, bias=False)
        self.proj = nn.Linear(channels, channels)

    def forward(self, x: Tensor) -> Tensor:
        B, C, H, W = x.shape
        h = self.norm(x)
        h = h.view(B, C, -1).permute(0, 2, 1)

        qkv = self.qkv(h).view(B, -1, 3, self.num_heads, C // self.num_heads)
        q, k, v = qkv.unbind(2)

        q, k, v = map(lambda t: t.transpose(1, 2), (q, k, v))
        out = F.scaled_dot_product_attention(q, k, v)

        out = out.transpose(1, 2).reshape(B, -1, C)
        out = self.proj(out).permute(0, 2, 1).view(B, C, H, W)
        return x + out


class ResBlock(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, cond_dim: int, dropout: float = 0.1
    ) -> None:
        super().__init__()
        self.norm1 = AdaGroupNorm(8, in_channels, cond_dim)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.norm2 = AdaGroupNorm(8, out_channels, cond_dim)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)

        self.shortcut = (
            nn.Conv2d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: Tensor, cond: Tensor) -> Tensor:
        h: Tensor = self.conv1(F.silu(self.norm1(x, cond)))
        h = self.conv2(self.dropout(F.silu(self.norm2(h, cond))))
        h = h + self.shortcut(x)
        return h


# --- Updated Structural Blocks ---


class EncoderBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, cond_dim: int) -> None:
        super().__init__()
        self.res = ResBlock(in_channels, out_channels, cond_dim)
        self.down = nn.Conv2d(out_channels, out_channels, 3, stride=2, padding=1)

    def forward(self, x: Tensor, cond: Tensor) -> Tuple[Tensor, Tensor]:
        skip = self.res(x, cond)
        x = self.down(skip)
        return x, skip


class DecoderBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, cond_dim: int) -> None:
        super().__init__()
        # The ResBlock handles the concatenation logic
        self.res = ResBlock(in_channels + out_channels, out_channels, cond_dim)
        # We still keep the convolution for smoothness after upscaling
        self.conv_up = nn.Conv2d(out_channels, out_channels, 3, padding=1)

    def forward(self, x: Tensor, skip: Tensor, cond: Tensor) -> Tensor:
        # 1. Dynamically upsample x to match the skip connection's spatial size
        # This handles cases like 9x9 -> 17x17 perfectly.
        x = F.interpolate(x, size=skip.shape[2:], mode="nearest")

        # 2. Concatenate along the channel dimension
        x = torch.cat([x, skip], dim=1)

        # 3. Process with ResBlock
        x = self.res(x, cond)

        # 4. Final convolution to refine the upsampled features
        x = self.conv_up(x)
        return x


class ResUnet(Backbone):
    def __init__(
        self,
        in_channels: int,
        channels: List[int],
        num_classes: int,
        t_dim: int = 256,
        y_dim: int = 256,
        cond_dim: int = 512,
    ) -> None:
        super().__init__()

        self.t_embedder = SinusoidalTimeEmbedding(t_dim)
        self.y_embedder = nn.Embedding(num_classes + 1, y_dim)
        self.cond_mlp = nn.Sequential(
            nn.Linear(t_dim + y_dim, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim),
        )

        self.stem = nn.Conv2d(in_channels, channels[0], kernel_size=3, padding=1)

        self.encoders = nn.ModuleList([])
        for i in range(len(channels) - 1):
            self.encoders.append(EncoderBlock(channels[i], channels[i + 1], cond_dim))

        self.bridge_res1 = ResBlock(channels[-1], channels[-1], cond_dim)
        self.bridge_attn = AttentionBlock(channels[-1])
        self.bridge_res2 = ResBlock(channels[-1], channels[-1], cond_dim)

        self.decoders = nn.ModuleList([])
        for i in reversed(range(len(channels) - 1)):
            self.decoders.append(DecoderBlock(channels[i + 1], channels[i], cond_dim))

        self.head = nn.Sequential(
            nn.GroupNorm(8, channels[0]),
            nn.SiLU(),
            nn.Conv2d(channels[0], in_channels, 3, padding=1),
        )

    def forward(self, x: Tensor, t: Tensor, y: Tensor) -> Tensor:
        t_emb = self.t_embedder(t)
        y_emb = self.y_embedder(y)
        cond = self.cond_mlp(torch.cat([t_emb, y_emb], dim=1))

        x = self.stem(x)

        skips = []
        for encoder in self.encoders:
            x, skip = encoder(x, cond)
            skips.append(skip)

        x = self.bridge_res1(x, cond)
        x = self.bridge_attn(x)
        x = self.bridge_res2(x, cond)

        for decoder in self.decoders:
            x = decoder(x, skips.pop(), cond)

        x = self.head(x)
        return x
