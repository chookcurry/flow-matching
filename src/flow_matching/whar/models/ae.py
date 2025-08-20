from torch import nn, Tensor
from typing import Tuple

from flow_matching.latent.ae import Autoencoder


class ResidualBlock(nn.Module):
    def __init__(self, channels: int, num_groups: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.GroupNorm(num_groups, channels),
            nn.ReLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups, channels),
            nn.ReLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.block(x) + x


class Encoder(nn.Module):
    def __init__(self, in_channels: int, latent_channels: int):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=4, stride=2, padding=1),
            ResidualBlock(32, 4),
            ResidualBlock(32, 4),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            ResidualBlock(64, 8),
            ResidualBlock(64, 8),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            ResidualBlock(128, 16),
            ResidualBlock(128, 16),
            nn.Conv2d(128, latent_channels, kernel_size=3, padding=1),
            nn.Tanh(),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.layers(x)


class Decoder(nn.Module):
    def __init__(self, in_channels: int, latent_channels: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(latent_channels, 128, kernel_size=3, padding=1),
            ResidualBlock(128, 16),
            ResidualBlock(128, 16),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            ResidualBlock(64, 8),
            ResidualBlock(64, 8),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            ResidualBlock(32, 4),
            ResidualBlock(32, 4),
            nn.ConvTranspose2d(32, in_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),
        )

    def forward(self, z: Tensor) -> Tensor:
        return self.layers(z)


class SpectrogramAE(Autoencoder):
    def __init__(self, in_channels: int = 18, latent_channels: int = 64):
        super().__init__()
        self.encoder = Encoder(in_channels, latent_channels)
        self.decoder = Decoder(in_channels, latent_channels)

    def encode(self, x: Tensor) -> Tensor:
        return self.encoder(x)

    def decode(self, z: Tensor) -> Tensor:
        return self.decoder(z)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        z = self.encode(x)
        recon = self.decode(z)
        return recon, z
