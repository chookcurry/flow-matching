import math
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from flow_matching.supervised.odes_sdes import ConditionalVectorField


class FourierEncoder(nn.Module):
    """
    Based on https://github.com/lucidrains/denoising-diffusion-pytorch/blob/main/denoising_diffusion_pytorch/karras_unet.py#L183
    """

    def __init__(self, dim: int):
        super().__init__()
        assert dim % 2 == 0
        self.half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(1, self.half_dim))

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
        - t: (bs, 1, 1, 1)
        Returns:
        - embeddings: (bs, dim)
        """
        t = t.view(-1, 1)  # (bs, 1)
        freqs = t * self.weights * 2 * math.pi  # (bs, half_dim)
        sin_embed = torch.sin(freqs)  # (bs, half_dim)
        cos_embed = torch.cos(freqs)  # (bs, half_dim)
        return torch.cat([sin_embed, cos_embed], dim=-1) * math.sqrt(2)  # (bs, dim)


class WHARResidualBlock(nn.Module):
    def __init__(self, num_channels: int, emb_dim: int, gn_groups: int = 8):
        super().__init__()

        # Convs
        self.conv1 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1)

        # GroupNorms
        groups = min(gn_groups, num_channels)
        self.norm1 = nn.GroupNorm(groups, num_channels)
        self.norm2 = nn.GroupNorm(groups, num_channels)

        # Fusion MLP for (t, y) -> emb_dim
        # We'll concatenate t and y (shape: [B, emb_dim] each) -> 2*emb_dim -> emb_dim
        self.fuse = nn.Sequential(
            nn.Linear(emb_dim * 2, emb_dim),
            nn.SiLU(),  # nonlinearity
            nn.Linear(emb_dim, emb_dim),
            nn.SiLU(),
        )

        # Single FiLM layer that produces gamma and beta for feature-wise affine
        self.film = nn.Linear(emb_dim, num_channels * 2)

        # init last conv (conv2) to zero so residual starts as identity (stability trick)
        nn.init.zeros_(self.conv2.weight)
        if self.conv2.bias is not None:
            nn.init.zeros_(self.conv2.bias)

    def forward(self, x: Tensor, t: Tensor, y: Tensor) -> Tensor:
        """
        x: [B, C, H, W]
        t: [B, emb_dim]
        y: [B, emb_dim]
        """
        res = x

        # first conv with normalization + nonlinearity
        h = self.norm1(x)
        h = F.silu(h)
        h = self.conv1(h)

        # fuse embeddings and produce FiLM parameters
        fused = torch.cat([t, y], dim=1)  # [B, 2*emb_dim]
        fused = self.fuse(fused)  # [B, emb_dim]
        gamma, beta = self.film(fused).chunk(2, dim=1)
        gamma = gamma[:, :, None, None]
        beta = beta[:, :, None, None]

        # apply FiLM
        h = gamma * h + beta

        # second conv with normalization + nonlinearity
        h = self.norm2(h)
        h = F.silu(h)
        h = self.conv2(h)

        # residual scaling (stabilize deep residual stacks)
        out = (h + res) / math.sqrt(2.0)
        out = F.silu(out)
        return out


class WHAREncoder(nn.Module):
    def __init__(
        self, channels_in: int, channels_out: int, emb_dim: int, num_blocks: int
    ):
        super().__init__()

        # Use ModuleList so parameters are registered
        self.blocks = nn.ModuleList(
            [WHARResidualBlock(channels_in, emb_dim) for _ in range(num_blocks)]
        )

        # downsample: keep same spatial height, halve width (like your original stride=(1,2))
        self.downsample = nn.Conv2d(
            channels_in, channels_out, kernel_size=3, stride=(1, 2), padding=1
        )

    # def pad_to_even(self, x: Tensor) -> Tensor:
    #     h, w = x.shape[-2:]
    #     pad_h = h % 2
    #     pad_w = w % 2
    #     if pad_h or pad_w:
    #         x = F.pad(x, (0, pad_w, 0, pad_h))
    #     return x

    def pad_to_even(self, x: Tensor, mode: str = "reflect") -> Tensor:
        h, w = x.shape[-2:]
        pad_h = (2 - h % 2) % 2
        pad_w = (2 - w % 2) % 2
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode=mode)
        return x

    def forward(self, x: Tensor, t: Tensor, y: Tensor) -> Tuple[Tensor, Tensor]:
        for block in self.blocks:
            x = block(x, t, y)

        skip_con = x.clone()

        x = self.pad_to_even(x)
        x = self.downsample(x)

        return x, skip_con


class WHARMidcoder(nn.Module):
    def __init__(self, channels_in: int, emb_dim: int, num_blocks: int):
        super().__init__()

        # Use ModuleList
        self.blocks = nn.ModuleList(
            [WHARResidualBlock(channels_in, emb_dim) for _ in range(num_blocks)]
        )

    def forward(self, x: Tensor, t: Tensor, y: Tensor) -> Tensor:
        for block in self.blocks:
            x = block(x, t, y)
        return x


class WHARDecoder(nn.Module):
    def __init__(
        self, channels_in: int, channels_out: int, emb_dim: int, num_blocks: int
    ):
        """
        channels_in: number of channels entering the decoder (from bottleneck)
        channels_out: number of channels to produce after reduce (and matching skip_con channels)
        Typical usage: channels_in == 2 * channels_out for symmetric U-Net channel lists.
        """
        super().__init__()

        # Upsample: nearest neighbor + conv (preferred to avoid checkerboard)
        self.upsample = nn.Sequential(
            nn.Upsample(scale_factor=(1, 2), mode="nearest"),
            nn.Conv2d(channels_in, channels_out, kernel_size=3, padding=1),
        )

        # reduce conv expects concatenated channels as in original design.
        # As in your original code: concat of upsample_out (channels_out) + skip_con (channels_out)
        # gives in_channels = 2 * channels_out, which we expect to equal channels_in.
        # We'll assert that to help catch mismatched channel lists.
        assert channels_in == 2 * channels_out, (
            f"Expected channels_in == 2 * channels_out for symmetric channels (got {channels_in} vs {channels_out})"
        )

        self.reduce = nn.Conv2d(channels_in, channels_out, kernel_size=3, padding=1)

        # blocks after reduction
        self.blocks = nn.ModuleList(
            [WHARResidualBlock(channels_out, emb_dim) for _ in range(num_blocks)]
        )

    def crop_to_match(self, x: Tensor, target: Tensor) -> Tensor:
        _, _, h, w = target.shape
        return x[..., :h, :w]

    def forward(self, x: Tensor, t: Tensor, y: Tensor, skip_con: Tensor) -> Tensor:
        x = self.upsample(x)
        x = self.crop_to_match(x, skip_con)

        x = torch.cat(
            [x, skip_con], dim=1
        )  # expected channels: channels_out + skip_channels (skip_channels == channels_out)
        x = self.reduce(x)
        for block in self.blocks:
            x = block(x, t, y)
        return x


class WHARUnet(ConditionalVectorField):
    def __init__(
        self,
        in_channels: int,
        channels: List[int],
        num_blocks: int,
        emb_dim: int,
        num_classes: int,
    ):
        super().__init__()

        self.t_embedder = FourierEncoder(emb_dim)
        self.y_embedder = nn.Embedding(num_classes + 1, emb_dim)
        self.init_conv = nn.Conv2d(in_channels, channels[0], kernel_size=3, padding=1)

        # Encoders: ModuleList
        self.encoders = nn.ModuleList(
            [
                WHAREncoder(channels[i], channels[i + 1], emb_dim, num_blocks)
                for i in range(len(channels) - 1)
            ]
        )

        # Midcoder
        self.midcoder = WHARMidcoder(channels[-1], emb_dim, num_blocks)

        # Decoders: ModuleList (mirror order)
        self.decoders = nn.ModuleList(
            [
                WHARDecoder(channels[i], channels[i - 1], emb_dim, num_blocks)
                for i in range(len(channels) - 1, 0, -1)
            ]
        )

        self.out_conv = nn.Conv2d(channels[0], in_channels, kernel_size=3, padding=1)

    def forward(self, x: Tensor, t: Tensor, y: Tensor) -> Tensor:
        # If user left t_embedder as Identity, assume t already embedded.
        t_emb = self.t_embedder(t)
        y_emb = self.y_embedder(y)

        x = self.init_conv(x)

        skip_cons = []

        for encoder in self.encoders:
            x, skip_con = encoder(x, t_emb, y_emb)
            skip_cons.append(skip_con)

        x = self.midcoder(x, t_emb, y_emb)

        for decoder in self.decoders:
            skip_con = skip_cons.pop()
            x = decoder(x, t_emb, y_emb, skip_con)

        x = self.out_conv(x)
        return x
