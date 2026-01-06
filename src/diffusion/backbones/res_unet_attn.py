import math
from typing import List, Tuple

import torch
from torch import Tensor, nn

from diffusion.backbones.backbone import Backbone


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()

        assert dim % 2 == 0
        self.half_dim = dim // 2

    def forward(self, t: Tensor) -> Tensor:
        # t: (B,) or (B, 1)

        t = t.view(-1, 1)
        # (B, 1)

        # Compute frequencies: [1, 10000^(2i/d)]
        freqs = torch.exp(
            -math.log(10000)
            * torch.arange(0, self.half_dim, dtype=torch.float32)
            / self.half_dim
        ).to(t.device)
        # (half_dim,)

        angles = t * freqs * 2 * math.pi
        # (B, half_dim)

        emb = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
        # (B, dim)

        return emb


class AdaGroupNorm(nn.Module):
    def __init__(self, num_groups: int, num_channels: int, cond_dim: int) -> None:
        super().__init__()

        self.group_norm = nn.GroupNorm(num_groups, num_channels, affine=False, eps=1e-6)
        self.linear = nn.Linear(cond_dim, 2 * num_channels)
        # outputs scale (γ) and shift (β)
        # Initialize to do nothing at start (γ ≈ 1, β ≈ 0)

        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x: Tensor, cond: Tensor) -> Tensor:
        # x: (B, C, H, W)
        # cond: (B, cond_dim)

        x = self.group_norm(x)
        # (B, C, H, W)

        gamma, beta = self.linear(cond).chunk(2, dim=1)
        # (B, C)

        gamma = gamma.unsqueeze(-1).unsqueeze(-1)
        beta = beta.unsqueeze(-1).unsqueeze(-1)
        # (B, C, 1, 1)

        out: Tensor = x * (1 + gamma) + beta
        # (B, C, H, W)

        return out


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, cond_dim: int) -> None:
        super().__init__()

        self.norm = AdaGroupNorm(8, in_channels, cond_dim)
        self.act = nn.SiLU(inplace=True)
        self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=1)

    def forward(self, x: Tensor, cond: Tensor) -> Tensor:
        # x: (B, C, H, W)
        # cond: (B, cond_dim)

        x = self.norm(x, cond)
        x = self.act(x)
        x = self.conv(x)
        # (B, C, H, W)

        return x


class ResBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, cond_dim: int) -> None:
        super().__init__()

        self.shortcut = (
            nn.Conv2d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else nn.Identity()
        )

        self.conv1 = ConvBlock(in_channels, out_channels, cond_dim)
        self.conv2 = ConvBlock(out_channels, out_channels, cond_dim)

        if in_channels == out_channels:
            nn.init.zeros_(self.conv2.conv.weight)
            if self.conv2.conv.bias is not None:
                nn.init.zeros_(self.conv2.conv.bias)

    def forward(self, x: Tensor, cond: Tensor) -> Tensor:
        # x: (B, C, H, W)
        # cond: (B, cond_dim)

        res = self.shortcut(x)
        x = self.conv1(x, cond)
        x = self.conv2(x, cond)
        x += res
        # (B, C, H, W)

        return x


class AttentionBlock(nn.Module):
    def __init__(self, channels: int, num_heads: int = 8) -> None:
        super().__init__()

        self.norm = nn.GroupNorm(8, channels, eps=1e-6)
        self.attention = nn.MultiheadAttention(
            embed_dim=channels,
            num_heads=num_heads,
            batch_first=True,
        )

    def forward(self, x: Tensor) -> Tensor:
        # x: (B, C, H, W)
        B, C, H, W = x.shape

        # Normalize
        x_norm = self.norm(x)

        # Reshape to sequence: (B, C, H, W) -> (B, H*W, C)
        x_seq = x_norm.view(B, C, H * W).transpose(1, 2)

        # Apply self-attention
        attn_out, _ = self.attention(x_seq, x_seq, x_seq)
        # (B, H*W, C)

        # Reshape back: (B, H*W, C) -> (B, C, H, W)
        attn_out = attn_out.transpose(1, 2).view(B, C, H, W)

        # Residual connection
        out = x + attn_out

        return out


class Upsample(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, cond_dim: int):
        super().__init__()

        # self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.upsample = nn.Upsample(
            scale_factor=2, mode="bilinear", align_corners=False
        )
        # self.upsample = nn.ConvTranspose2d(
        #     in_channels, out_channels, kernel_size=4, stride=2, padding=1
        # )

        self.conv = ConvBlock(in_channels, out_channels, cond_dim)

    def forward(self, x: Tensor, cond: Tensor) -> Tensor:
        # x: (B, C, H/, W/2)

        x = self.upsample(x)
        x = self.conv(x, cond)
        # (B, C, H, W)

        return x


class EncoderBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        cond_dim: int,
        use_attention: bool = False,
    ) -> None:
        super().__init__()

        self.res1 = ResBlock(in_channels, out_channels, cond_dim)
        self.attention = AttentionBlock(out_channels) if use_attention else None
        self.res2 = ResBlock(out_channels, out_channels, cond_dim)
        self.downsample = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=2, padding=1
        )

    def forward(self, x: Tensor, cond: Tensor) -> Tuple[Tensor, Tensor]:
        # x: (B, in_channels, H, W)
        # cond: (B, cond_dim)

        skip = self.res1(x, cond)
        if self.attention is not None:
            skip = self.attention(skip)
        skip = self.res2(skip, cond)
        # (B, out_channels, H, W)

        x = self.downsample(skip)
        # (B, out_channels, H/2, W/2)

        return x, skip


class DecoderBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        cond_dim: int,
        use_attention: bool = False,
    ) -> None:
        super().__init__()

        self.upsample = Upsample(in_channels, in_channels, cond_dim)

        self.res1 = ResBlock(2 * in_channels, out_channels, cond_dim)
        self.attention = AttentionBlock(out_channels) if use_attention else None
        self.res2 = ResBlock(out_channels, out_channels, cond_dim)

    def forward(self, x: Tensor, skip: Tensor, cond: Tensor) -> Tensor:
        # x: (B, in_channels, H/2, W/2)
        # skip: (B, in_channels, H, W)
        # cond: (B, cond_dim)

        x = self.upsample(x, cond)
        # (B, in_channels, H, W)

        if x.shape[-2:] != skip.shape[-2:]:
            x = nn.functional.interpolate(
                x, size=skip.shape[-2:], mode="bilinear", align_corners=False
            )

        x = torch.cat([x, skip], dim=1)
        # (B, 2 * in_channels, H, W)

        x = self.res1(x, cond)
        if self.attention is not None:
            x = self.attention(x)
        x = self.res2(x, cond)
        # (B, out_channels, H, W)

        return x


class BridgeBlock(nn.Module):
    def __init__(
        self, channels: int, cond_dim: int, use_attention: bool = False
    ) -> None:
        super().__init__()

        self.res1 = ResBlock(channels, channels, cond_dim)
        self.attention = AttentionBlock(channels) if use_attention else None
        self.res2 = ResBlock(channels, channels, cond_dim)

    def forward(self, x: Tensor, cond: Tensor) -> Tensor:
        # x: (B, channels, H, W)
        # cond: (B, cond_dim)

        x = self.res1(x, cond)
        if self.attention is not None:
            x = self.attention(x)
        x = self.res2(x, cond)
        # (B, channels, H, W)

        return x


class Conditioner(nn.Module):
    def __init__(self, num_classes: int, t_dim: int, y_dim: int, cond_dim: int) -> None:
        super().__init__()

        self.t_embedder = SinusoidalTimeEmbedding(t_dim)
        self.y_embedder = nn.Embedding(num_classes + 1, y_dim)

        self.mlp = nn.Sequential(
            nn.Linear(t_dim + y_dim, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim),
        )

    def forward(self, t: Tensor, y: Tensor) -> Tensor:
        # t: (B)
        # y: (B)

        t_embed = self.t_embedder(t)
        y_embed = self.y_embedder(y)
        cond = torch.cat([t_embed, y_embed], dim=1)
        cond = self.mlp(cond)
        # (B, cond_dim)

        return cond


class BiTimeConditioner(nn.Module):
    def __init__(self, num_classes: int, t_dim: int, y_dim: int, cond_dim: int) -> None:
        super().__init__()

        self.t_embedder = SinusoidalTimeEmbedding(t_dim)
        self.y_embedder = nn.Embedding(num_classes + 1, y_dim)

        self.mlp = nn.Sequential(
            nn.Linear(2 * t_dim + y_dim, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim),
        )

    def forward(self, s: Tensor, t: Tensor, y: Tensor) -> Tensor:
        # s: (B)
        # t: (B)
        # y: (B)

        s_embed = self.t_embedder(s)
        t_embed = self.t_embedder(t)
        y_embed = self.y_embedder(y)
        cond = torch.cat([s_embed, t_embed, y_embed], dim=1)
        cond = self.mlp(cond)
        # (B, cond_dim)

        return cond


class ResUnet(Backbone):
    def __init__(
        self,
        in_channels: int,
        channel_dims: List[int],
        use_attention: List[bool],
        num_classes: int,
        t_dim: int,
        y_dim: int,
        cond_dim: int,
    ) -> None:
        super().__init__()

        self.start = nn.Conv2d(in_channels, channel_dims[0], kernel_size=3, padding=1)
        self.conditioner = Conditioner(num_classes, t_dim, y_dim, cond_dim)

        self.encoders = nn.ModuleList(
            [
                EncoderBlock(
                    channel_dims[i], channel_dims[i + 1], cond_dim, use_attention[i]
                )
                for i in range(len(channel_dims) - 1)
            ]
        )

        self.bridge = BridgeBlock(channel_dims[-1], cond_dim, use_attention[-1])

        self.decoders = nn.ModuleList(
            [
                DecoderBlock(
                    channel_dims[i], channel_dims[i - 1], cond_dim, use_attention[i - 1]
                )
                for i in range(len(channel_dims) - 1, 0, -1)
            ]
        )

        self.norm = nn.GroupNorm(8, channel_dims[0], eps=1e-6)
        self.act = nn.SiLU(inplace=True)
        self.final = nn.Conv2d(channel_dims[0], in_channels, kernel_size=3, padding=1)

    def forward(self, x: Tensor, t: Tensor, y: Tensor) -> Tensor:
        # x: (B, in_channels, H, W)
        # t: (B)
        # y: (B)

        cond = self.conditioner(t, y)
        # (B, cond_dim)

        x = self.start(x)
        skips: List[Tensor] = []

        for encoder in self.encoders:
            x, skip = encoder(x, cond)
            skips.append(skip)

        x = self.bridge(x, cond)

        for decoder in self.decoders:
            x = decoder(x, skips.pop(), cond)

        x = self.norm(x)
        x = self.act(x)
        x = self.final(x)
        # (B, in_channels, H, W)

        return x


class BiTimeResUnet(nn.Module):
    def __init__(
        self,
        in_channels: int,
        channel_dims: List[int],
        use_attention: List[bool],
        num_classes: int,
        t_dim: int,
        y_dim: int,
        cond_dim: int,
    ) -> None:
        super().__init__()

        self.start = nn.Conv2d(in_channels, channel_dims[0], kernel_size=3, padding=1)
        self.conditioner = BiTimeConditioner(num_classes, t_dim, y_dim, cond_dim)

        self.encoders = nn.ModuleList(
            [
                EncoderBlock(
                    channel_dims[i], channel_dims[i + 1], cond_dim, use_attention[i]
                )
                for i in range(len(channel_dims) - 1)
            ]
        )

        self.bridge = BridgeBlock(channel_dims[-1], cond_dim, use_attention[-1])

        self.decoders = nn.ModuleList(
            [
                DecoderBlock(
                    channel_dims[i], channel_dims[i - 1], cond_dim, use_attention[i - 1]
                )
                for i in range(len(channel_dims) - 1, 0, -1)
            ]
        )

        self.norm = nn.GroupNorm(8, channel_dims[0], eps=1e-6)
        self.act = nn.SiLU(inplace=True)
        self.final = nn.Conv2d(channel_dims[0], in_channels, kernel_size=3, padding=1)

    def forward(self, x: Tensor, s: Tensor, t: Tensor, y: Tensor) -> Tensor:
        # x: (B, in_channels, H, W)
        # s: (B)
        # t: (B)
        # y: (B)

        cond = self.conditioner(s, t, y)
        # (B, cond_dim)

        x = self.start(x)
        skips: List[Tensor] = []

        for encoder in self.encoders:
            x, skip = encoder(x, cond)
            skips.append(skip)

        x = self.bridge(x, cond)

        for decoder in self.decoders:
            x = decoder(x, skips.pop(), cond)

        x = self.norm(x)
        x = self.act(x)
        x = self.final(x)
        # (B, in_channels, H, W)

        return x


if __name__ == "__main__":
    model = ResUnet(
        in_channels=18,
        channel_dims=[32, 64, 128, 256],
        use_attention=[False, False, True, True],
        num_classes=6,
        t_dim=16,
        y_dim=16,
        cond_dim=64,
    )
    input = torch.randn(1, 18, 64, 64)
    t = torch.rand(1)
    y = torch.randint(0, 6, (1,))
    out = model(input, t, y)
    print(out.shape)
