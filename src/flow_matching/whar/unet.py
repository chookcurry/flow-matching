from typing import List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from flow_matching.supervised.models import ConditionalVectorField, FourierEncoder


class WHARResidualBlock(nn.Module):
    def __init__(self, num_channels: int, emb_dim: int):
        super(WHARResidualBlock, self).__init__()

        self.conv1 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1)

        self.film_t = nn.Linear(emb_dim, num_channels * 2)  # for γ and β
        self.film_y = nn.Linear(emb_dim, num_channels * 2)  # for γ and β

    def forward(self, x: Tensor, t: Tensor, y: Tensor) -> Tensor:
        res = x.clone()

        x = self.conv1(x)

        gamma_t, beta_t = self.film_t(t).chunk(2, 1)
        gamma_t, beta_t = gamma_t[:, :, None, None], beta_t[:, :, None, None]
        x = gamma_t * x + beta_t

        gamma_y, beta_y = self.film_y(y).chunk(2, 1)
        gamma_y, beta_y = gamma_y[:, :, None, None], beta_y[:, :, None, None]
        x = gamma_y * x + beta_y

        x = F.relu(x)

        x = self.conv2(x)

        x += res
        x = F.relu(x)

        return x


class WHAREncoder(nn.Module):
    def __init__(
        self,
        channels_in: int,
        channels_out: int,
        emb_dim: int,
        num_blocks: int,
    ):
        super(WHAREncoder, self).__init__()

        self.blocks = [
            WHARResidualBlock(channels_in, emb_dim) for _ in range(num_blocks)
        ]

        self.downsample = nn.Conv2d(
            channels_in, channels_out, kernel_size=3, stride=(1, 2), padding=1
        )

    def pad_to_even(self, x: Tensor) -> Tensor:
        h, w = x.shape[-2:]
        pad_h = h % 2
        pad_w = w % 2
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h))
        return x

    def forward(self, x: Tensor, t: Tensor, y: Tensor) -> Tuple[Tensor, Tensor]:
        for block in self.blocks:
            x = block(x, t, y)

        skip_con = x.clone()

        x = self.pad_to_even(x)
        x = self.downsample(x)

        return x, skip_con


class WHARMidcoder(nn.Module):
    def __init__(
        self,
        channels_in: int,
        emb_dim: int,
        num_blocks: int,
    ):
        super(WHARMidcoder, self).__init__()

        self.blocks = [
            WHARResidualBlock(channels_in, emb_dim) for _ in range(num_blocks)
        ]

    def forward(self, x: Tensor, t: Tensor, y: Tensor) -> Tensor:
        for block in self.blocks:
            x = block(x, t, y)
        return x


class WHARDecoder(nn.Module):
    def __init__(
        self,
        channels_in: int,
        channels_out: int,
        emb_dim: int,
        num_blocks: int,
    ):
        super(WHARDecoder, self).__init__()

        # self.upsample = nn.ConvTranspose2d(
        #     channels_in,
        #     channels_out,
        #     kernel_size=3,
        #     stride=2,
        #     padding=1,
        #     # output_padding=1,
        # )

        self.upsample = nn.Sequential(
            nn.Upsample(scale_factor=(1, 2), mode="bilinear"),
            nn.Conv2d(channels_in, channels_out, kernel_size=3, padding=1),
        )

        self.reduce = nn.Conv2d(channels_in, channels_out, kernel_size=3, padding=1)

        self.blocks = [
            WHARResidualBlock(channels_out, emb_dim) for _ in range(num_blocks)
        ]

    def crop_to_match(self, x: Tensor, target: Tensor) -> Tensor:
        _, _, h, w = target.shape
        return x[..., :h, :w]

    def forward(self, x: Tensor, t: Tensor, y: Tensor, skip_con: Tensor) -> Tensor:
        x = self.upsample(x)

        x = self.crop_to_match(x, skip_con)

        x = torch.cat([x, skip_con], dim=1)
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
        super(WHARUnet, self).__init__()

        self.t_embedder = FourierEncoder(emb_dim)
        self.y_embedder = nn.Embedding(num_classes + 1, emb_dim)

        self.init_conv = nn.Conv2d(in_channels, channels[0], kernel_size=3, padding=1)

        self.encoders = nn.ModuleList(
            [
                WHAREncoder(channels[i], channels[i + 1], emb_dim, num_blocks)
                for i in range(len(channels) - 1)
            ]
        )

        self.midcoder = WHARMidcoder(channels[-1], emb_dim, num_blocks)

        self.decoders = nn.ModuleList(
            [
                WHARDecoder(channels[i], channels[i - 1], emb_dim, num_blocks)
                for i in range(len(channels) - 1, 0, -1)
            ]
        )

        self.out_conv = nn.Conv2d(channels[0], in_channels, kernel_size=3, padding=1)

    def forward(self, x: Tensor, t: Tensor, y: Tensor) -> Tensor:
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
