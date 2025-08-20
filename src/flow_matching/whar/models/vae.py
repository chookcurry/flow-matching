import torch
from torch import nn
from torch import Tensor
from typing import Tuple

from flow_matching.latent.vae import VAE


class Encoder(nn.Module):
    def __init__(self, latent_channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(18, 32, kernel_size=4, stride=2, padding=1),  # -> [B, 32, 16, 16]
            nn.GroupNorm(4, 32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),  # -> [B, 64, 8, 8]
            nn.GroupNorm(8, 64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # -> [B, 128, 4, 4]
            nn.GroupNorm(16, 128),
            nn.ReLU(),
        )

        self.mu_layer = nn.Conv2d(128, latent_channels, kernel_size=3, padding=1)
        self.logvar_layer = nn.Conv2d(128, latent_channels, kernel_size=3, padding=1)
        # -> [B, 64, 4, 4]

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        h = self.net(x)
        mu = self.mu_layer(h)
        logvar = self.logvar_layer(h)
        return mu, logvar


class Decoder(nn.Module):
    def __init__(self, latent_channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(latent_channels, 128, kernel_size=3, padding=1),
            nn.GroupNorm(16, 128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            # -> [B, 64, 8, 8]
            nn.GroupNorm(8, 64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            # -> [B, 32, 16, 16]
            nn.GroupNorm(4, 32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 18, kernel_size=4, stride=2, padding=1),
            # -> [B, 18, 32, 32]
        )

    def forward(self, z: Tensor) -> Tensor:
        return self.net(z)


# class Decoder(nn.Module):
#     def __init__(self, latent_channels: int):
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.Conv2d(latent_channels, 128, kernel_size=3, padding=1),
#             nn.GroupNorm(16, 128),
#             nn.ReLU(),
#             nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
#             nn.Conv2d(128, 64, kernel_size=3, padding=1),
#             # -> [B, 64, 8, 8]
#             nn.GroupNorm(8, 64),
#             nn.ReLU(),
#             nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
#             nn.Conv2d(64, 32, kernel_size=3, padding=1),
#             # -> [B, 32, 16, 16]
#             nn.GroupNorm(4, 32),
#             nn.ReLU(),
#             nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
#             nn.Conv2d(32, 18, kernel_size=3, padding=1),
#             # -> [B, 18, 32, 32]
#         )

#     def forward(self, z: Tensor) -> Tensor:
#         return self.net(z)


class SpectrogramVAE(VAE):
    def __init__(self, latent_channels: int = 64):
        super().__init__()
        self.encoder = Encoder(latent_channels)
        self.decoder = Decoder(latent_channels)

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def encode(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

    def decode(self, z: Tensor) -> Tensor:
        return self.decoder(z)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        z, mu, logvar = self.encode(x)
        recon = self.decode(z)
        return recon, mu, logvar
