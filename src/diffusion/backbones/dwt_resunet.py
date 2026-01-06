from typing import List, Tuple

import torch
from torch import Tensor, nn

from diffusion.backbones.res_unet import Conditioner

# --- Core Interaction Blocks ---


class AdaGroupNorm1d(nn.Module):
    """AdaGN adapted for (B, C, Coeffs, T) input."""

    def __init__(self, num_groups: int, num_channels: int, cond_dim: int) -> None:
        super().__init__()
        self.group_norm = nn.GroupNorm(num_groups, num_channels, affine=False, eps=1e-6)
        self.linear = nn.Linear(cond_dim, 2 * num_channels)

    def forward(self, x: Tensor, cond: Tensor) -> Tensor:
        B, C, Coeffs, T = x.shape
        # Flatten Coeffs into Time for GN
        x_reshaped = x.view(B, C, -1)
        x_norm = self.group_norm(x_reshaped)
        x = x_norm.view(B, C, Coeffs, T)

        gamma, beta = self.linear(cond).chunk(2, dim=1)
        gamma = gamma.view(B, C, 1, 1)
        beta = beta.view(B, C, 1, 1)

        return x * (1 + gamma) + beta


class DWTInteractionBlock(nn.Module):
    """Pointwise (Cross-C/B) + Depthwise (Temporal) logic."""

    def __init__(
        self, in_channels: int, out_channels: int, num_coeffs: int, cond_dim: int
    ) -> None:
        super().__init__()
        self.num_coeffs = num_coeffs
        self.norm = AdaGroupNorm1d(8, in_channels, cond_dim)
        self.act = nn.SiLU(inplace=True)

        # Pointwise Mixer: (B, C*B, T)
        self.mixer = nn.Conv1d(in_channels * num_coeffs, out_channels * num_coeffs, 1)
        # Depthwise Temporal: Evolution over T
        self.temporal = nn.Conv1d(
            out_channels * num_coeffs,
            out_channels * num_coeffs,
            3,
            padding=1,
            groups=out_channels * num_coeffs,
        )

    def forward(self, x: Tensor, cond: Tensor) -> Tensor:
        B_size, C, Coeff, T = x.shape
        x = self.norm(x, cond)
        x = self.act(x)

        x = x.view(B_size, C * Coeff, T)
        x = self.mixer(x)
        x = self.temporal(x)

        return x.view(B_size, -1, Coeff, T)


class DWTResBlock(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, num_coeffs: int, cond_dim: int
    ) -> None:
        super().__init__()
        self.shortcut = (
            nn.Conv1d(in_channels * num_coeffs, out_channels * num_coeffs, 1)
            if in_channels != out_channels
            else nn.Identity()
        )
        self.block1 = DWTInteractionBlock(
            in_channels, out_channels, num_coeffs, cond_dim
        )
        self.block2 = DWTInteractionBlock(
            out_channels, out_channels, num_coeffs, cond_dim
        )

    def forward(self, x: Tensor, cond: Tensor) -> Tensor:
        b, c, coeffs, t = x.shape
        res = self.shortcut(x.view(b, c * coeffs, t)).view(b, -1, coeffs, t)
        x = self.block1(x, cond)
        x = self.block2(x, cond)
        return x + res


class DWTAttentionBlock(nn.Module):
    """
    Attends across the combined Coefficient and Time dimensions.
    Sequence length = num_coeffs * T
    """

    def __init__(self, channels: int, num_coeffs: int, num_heads: int = 8) -> None:
        super().__init__()
        self.num_coeffs = num_coeffs
        self.norm = nn.GroupNorm(8, channels, eps=1e-6)
        self.attention = nn.MultiheadAttention(
            embed_dim=channels,
            num_heads=num_heads,
            batch_first=True,
        )

    def forward(self, x: Tensor) -> Tensor:
        # x: (B, C, Coeffs, T)
        B, C, Coeffs, T = x.shape

        # 1. Normalize across channels
        x_norm = self.norm(x.view(B, C, -1)).view(B, C, Coeffs, T)

        # 2. Reshape to Sequence: (B, C, Coeffs, T) -> (B, Coeffs*T, C)
        # Here 'C' (Channels) acts as the embedding vector
        x_seq = x_norm.permute(0, 2, 3, 1).reshape(B, Coeffs * T, C)

        # 3. Apply Multi-head Self Attention
        attn_out, _ = self.attention(x_seq, x_seq, x_seq)

        # 4. Reshape back to (B, C, Coeffs, T)
        attn_out = attn_out.view(B, Coeffs, T, C).permute(0, 3, 1, 2)

        return x + attn_out


# --- U-Net Structural Blocks ---


class DWTEncoderBlock(nn.Module):
    def __init__(
        self, in_c: int, out_c: int, num_b: int, cond_d: int, use_attn: bool
    ) -> None:
        super().__init__()
        self.res1 = DWTResBlock(in_c, out_c, num_b, cond_d)
        self.attn = DWTAttentionBlock(out_c, num_b) if use_attn else None
        self.res2 = DWTResBlock(out_c, out_c, num_b, cond_d)
        # Downsample Time only
        self.down = nn.Conv2d(
            out_c, out_c, kernel_size=(1, 3), stride=(1, 2), padding=(0, 1)
        )

    def forward(self, x: Tensor, cond: Tensor) -> Tuple[Tensor, Tensor]:
        x = self.res1(x, cond)
        if self.attn:
            x = self.attn(x)
        skip = self.res2(x, cond)
        x = self.down(skip)
        return x, skip


class DWTDecoderBlock(nn.Module):
    def __init__(
        self, in_c: int, out_c: int, num_b: int, cond_d: int, use_attn: bool
    ) -> None:
        super().__init__()
        self.up = nn.Upsample(scale_factor=(1, 2), mode="bilinear", align_corners=False)
        self.res1 = DWTResBlock(in_c * 2, out_c, num_b, cond_d)
        self.attn = DWTAttentionBlock(out_c, num_b) if use_attn else None
        self.res2 = DWTResBlock(out_c, out_c, num_b, cond_d)

    def forward(self, x: Tensor, skip: Tensor, cond: Tensor) -> Tensor:
        x = self.up(x)
        if x.shape[-1] != skip.shape[-1]:
            x = nn.functional.interpolate(x, size=skip.shape[-2:], mode="bilinear")

        x = torch.cat([x, skip], dim=1)
        x = self.res1(x, cond)
        if self.attn:
            x = self.attn(x)
        return self.res2(x, cond)


# --- Final Backbone ---


class DWTResUnet(nn.Module):
    def __init__(
        self,
        in_channels: int,  # Sensors (C)
        num_coeffs: int,  # DWT Bands (B)
        channel_dims: List[int],
        use_attention: List[bool],
        num_classes: int,
        cond_dim: int = 512,
    ) -> None:
        super().__init__()
        self.num_coeffs = num_coeffs
        self.conditioner = Conditioner(
            num_classes, t_dim=128, y_dim=128, cond_dim=cond_dim
        )

        self.start = nn.Conv2d(in_channels, channel_dims[0], kernel_size=3, padding=1)

        # Encoders
        self.encoders = nn.ModuleList(
            [
                DWTEncoderBlock(
                    channel_dims[i],
                    channel_dims[i + 1],
                    num_coeffs,
                    cond_dim,
                    use_attention[i],
                )
                for i in range(len(channel_dims) - 1)
            ]
        )

        # Bridge
        mid_dim = channel_dims[-1]
        self.bridge_res1 = DWTResBlock(mid_dim, mid_dim, num_coeffs, cond_dim)
        self.bridge_attn = (
            DWTAttentionBlock(mid_dim, num_coeffs) if use_attention[-1] else None
        )
        self.bridge_res2 = DWTResBlock(mid_dim, mid_dim, num_coeffs, cond_dim)

        # Decoders
        self.decoders = nn.ModuleList(
            [
                DWTDecoderBlock(
                    channel_dims[i],
                    channel_dims[i - 1],
                    num_coeffs,
                    cond_dim,
                    use_attention[i - 1],
                )
                for i in range(len(channel_dims) - 1, 0, -1)
            ]
        )

        self.final = nn.Conv2d(channel_dims[0], in_channels, kernel_size=3, padding=1)

    def forward(self, x: Tensor, t: Tensor, y: Tensor) -> Tensor:
        # x: (Batch, C, B, T)
        cond = self.conditioner(t, y)

        x = self.start(x)
        skips = []

        for enc in self.encoders:
            x, skip = enc(x, cond)
            skips.append(skip)

        x = self.bridge_res1(x, cond)
        if self.bridge_attn:
            x = self.bridge_attn(x)
        x = self.bridge_res2(x, cond)

        for dec in self.decoders:
            x = dec(x, skips.pop(), cond)

        return self.final(x)


if __name__ == "__main__":
    # Example: 18 Sensors, 4 DWT Coefficients, 3 Levels of U-Net
    model = DWTResUnet(
        in_channels=18,
        num_coeffs=4,
        channel_dims=[64, 128, 256],
        use_attention=[False, True, True],
        num_classes=5,
    )

    x = torch.randn(1, 18, 4, 256)  # (B, C, Coeffs, Time)
    t = torch.tensor([500])
    y = torch.tensor([2])

    out = model(x, t, y)
    print(f"Final output shape: {out.shape}")  # (1, 18, 4, 256)
