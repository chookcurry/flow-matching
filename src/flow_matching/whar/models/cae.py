from torch import nn, Tensor
from typing import Tuple
import torch.nn.functional as F


class ConditionalGroupNorm(nn.Module):
    def __init__(self, num_features: int, num_groups: int, embedding_dim: int):
        super().__init__()
        self.num_features = num_features
        self.num_groups = num_groups

        self.groupnorm = nn.GroupNorm(num_groups, num_features, affine=False)
        self.embed_to_gamma = nn.Linear(embedding_dim, num_features)
        self.embed_to_beta = nn.Linear(embedding_dim, num_features)

    def forward(self, x: Tensor, embed: Tensor) -> Tensor:
        normalized = self.groupnorm(x)

        gamma = self.embed_to_gamma(embed).unsqueeze(-1).unsqueeze(-1)
        beta = self.embed_to_beta(embed).unsqueeze(-1).unsqueeze(-1)

        return gamma * normalized + beta


class CondResidualBlock(nn.Module):
    def __init__(self, channels: int, num_groups: int, embedding_dim: int):
        super().__init__()
        self.norm1 = ConditionalGroupNorm(channels, num_groups, embedding_dim)
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.norm2 = ConditionalGroupNorm(channels, num_groups, embedding_dim)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU()

    def forward(self, x: Tensor, embed: Tensor) -> Tensor:
        residual = x

        out = self.norm1(x, embed)
        out = self.relu(out)
        out = self.conv1(out)

        out = self.norm2(out, embed)
        out = self.relu(out)
        out = self.conv2(out)

        return out + residual


class ResidualBlock(nn.Module):
    def __init__(self, channels: int, num_groups: int):
        super().__init__()
        self.norm1 = nn.GroupNorm(num_groups, channels, affine=True)
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(num_groups, channels, affine=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU()

    def forward(self, x: Tensor) -> Tensor:
        residual = x

        out = self.norm1(x)
        out = self.relu(out)
        out = self.conv1(out)

        out = self.norm2(out)
        out = self.relu(out)
        out = self.conv2(out)

        return out + residual


class Encoder(nn.Module):
    def __init__(self, latent_channels: int, num_classes: int, embedding_dim: int = 32):
        super().__init__()
        self.embed = nn.Embedding(num_classes, embedding_dim)

        self.conv1 = nn.Conv2d(18, 32, kernel_size=4, stride=2, padding=1)
        self.resblock1 = nn.Sequential(
            CondResidualBlock(32, 4, embedding_dim),
            CondResidualBlock(32, 4, embedding_dim),
        )

        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1)
        self.resblock2 = nn.Sequential(
            CondResidualBlock(64, 8, embedding_dim),
            CondResidualBlock(64, 8, embedding_dim),
        )

        self.conv3 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)
        self.resblock3 = nn.Sequential(
            CondResidualBlock(128, 16, embedding_dim),
            CondResidualBlock(128, 16, embedding_dim),
        )

        self.z_proj = nn.Conv2d(128, latent_channels, kernel_size=3, padding=1)

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        embed = self.embed(y)

        h = self.conv1(x)
        for block in self.resblock1:
            h = block(h, embed)

        h = self.conv2(h)
        for block in self.resblock2:
            h = block(h, embed)

        h = self.conv3(h)
        for block in self.resblock3:
            h = block(h, embed)

        z = self.z_proj(h)
        z = F.tanh(z)

        return z


class Decoder(nn.Module):
    def __init__(self, latent_channels: int):
        super().__init__()

        self.conv1 = nn.Conv2d(latent_channels, 128, kernel_size=3, padding=1)
        self.resblock1 = nn.Sequential(ResidualBlock(128, 16), ResidualBlock(128, 16))

        self.convT1 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.resblock2 = nn.Sequential(ResidualBlock(64, 8), ResidualBlock(64, 8))

        self.convT2 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)
        self.resblock3 = nn.Sequential(ResidualBlock(32, 4), ResidualBlock(32, 4))

        self.convT3 = nn.ConvTranspose2d(32, 18, kernel_size=4, stride=2, padding=1)

    def forward(self, z: Tensor) -> Tensor:
        h = self.conv1(z)
        for block in self.resblock1:
            h = block(h)

        h = self.convT1(h)
        for block in self.resblock2:
            h = block(h)

        h = self.convT2(h)
        for block in self.resblock3:
            h = block(h)

        h = self.convT3(h)
        h = F.tanh(h)

        return h


class SpectrogramCAE(nn.Module):
    def __init__(
        self, latent_channels: int = 64, num_classes: int = 10, embedding_dim: int = 32
    ):
        super().__init__()
        self.encoder = Encoder(latent_channels, num_classes, embedding_dim)
        self.decoder = Decoder(latent_channels)

    def encode(self, x: Tensor, y: Tensor) -> Tensor:
        z = self.encoder(x, y)
        return z

    def decode(self, z: Tensor) -> Tensor:
        x = self.decoder(z)
        return x

    def forward(self, x: Tensor, y: Tensor) -> Tuple[Tensor, Tensor]:
        z = self.encode(x, y)
        recon = self.decode(z)
        return recon, z
