import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple

from flow_matching.mnist.vector_field import FourierEncoder
from flow_matching.supervised.odes_sdes import ConditionalVectorField


# -----------------------------------------
# Residual Block (Unchanged except structure)
# -----------------------------------------
class MNISTLatentResidualBlock(nn.Module):
    def __init__(self, num_channels: int, emb_dim: int):
        super().__init__()

        self.conv1 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1)

        self.film_t = nn.Linear(emb_dim, num_channels * 2)
        self.film_y = nn.Linear(emb_dim, num_channels * 2)

    def forward(
        self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor
    ) -> torch.Tensor:
        res = x

        x = self.conv1(x)
        gamma_t, beta_t = self.film_t(t).chunk(2, dim=1)
        x = gamma_t[:, :, None, None] * x + beta_t[:, :, None, None]

        gamma_y, beta_y = self.film_y(y).chunk(2, dim=1)
        x = gamma_y[:, :, None, None] * x + beta_y[:, :, None, None]

        x = F.relu(x)
        x = self.conv2(x)
        x += res
        return F.relu(x)


# -----------------------------------------
# Encoder (NO DOWNSAMPLING)
# -----------------------------------------
class MNISTLatentEncoder(nn.Module):
    def __init__(self, channels: int, emb_dim: int, num_blocks: int):
        super().__init__()
        self.blocks = nn.ModuleList(
            [MNISTLatentResidualBlock(channels, emb_dim) for _ in range(num_blocks)]
        )

    def forward(
        self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        for block in self.blocks:
            x = block(x, t, y)
        return x, x.clone()  # skip connection


# -----------------------------------------
# Midcoder
# -----------------------------------------
class MNISTLatentMidcoder(nn.Module):
    def __init__(self, channels: int, emb_dim: int, num_blocks: int):
        super().__init__()
        self.blocks = nn.ModuleList(
            [MNISTLatentResidualBlock(channels, emb_dim) for _ in range(num_blocks)]
        )

    def forward(
        self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor
    ) -> torch.Tensor:
        for block in self.blocks:
            x = block(x, t, y)
        return x


# -----------------------------------------
# Decoder (NO UPSAMPLING)
# -----------------------------------------
class MNISTLatentDecoder(nn.Module):
    def __init__(self, channels: int, emb_dim: int, num_blocks: int):
        super().__init__()
        self.reduce = nn.Conv2d(channels * 2, channels, kernel_size=3, padding=1)
        self.blocks = nn.ModuleList(
            [MNISTLatentResidualBlock(channels, emb_dim) for _ in range(num_blocks)]
        )

    def forward(
        self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor, skip_con: torch.Tensor
    ) -> torch.Tensor:
        x = torch.cat([x, skip_con], dim=1)
        x = self.reduce(x)
        for block in self.blocks:
            x = block(x, t, y)
        return x


# -----------------------------------------
# UNet (Final Vector Field)
# -----------------------------------------
class MNISTLatentUnet(ConditionalVectorField):
    def __init__(
        self,
        in_channels: int = 8,
        channels: List[int] = [8, 8],  # no channel change
        num_blocks: int = 2,
        emb_dim: int = 64,
        num_classes: int = 10,
    ):
        super().__init__()

        self.t_embedder = FourierEncoder(emb_dim)
        self.y_embedder = nn.Embedding(num_classes + 1, emb_dim)

        self.init_conv = nn.Conv2d(in_channels, channels[0], kernel_size=3, padding=1)

        self.encoders = nn.ModuleList(
            [
                MNISTLatentEncoder(channels[i], emb_dim, num_blocks)
                for i in range(len(channels))
            ]
        )

        self.midcoder = MNISTLatentMidcoder(channels[-1], emb_dim, num_blocks)

        self.decoders = nn.ModuleList(
            [
                MNISTLatentDecoder(channels[i], emb_dim, num_blocks)
                for i in reversed(range(len(channels)))
            ]
        )

        self.out_conv = nn.Conv2d(channels[0], in_channels, kernel_size=3, padding=1)

    def forward(
        self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor
    ) -> torch.Tensor:
        t_emb = self.t_embedder(t)
        y_emb = self.y_embedder(y)

        x = self.init_conv(x)
        skip_connections = []

        for encoder in self.encoders:
            x, skip = encoder(x, t_emb, y_emb)
            skip_connections.append(skip)

        x = self.midcoder(x, t_emb, y_emb)

        for decoder in self.decoders:
            skip = skip_connections.pop()
            x = decoder(x, t_emb, y_emb, skip)

        return self.out_conv(x)
