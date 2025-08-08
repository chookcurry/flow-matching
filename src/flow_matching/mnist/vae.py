from typing import Tuple
import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F

from flow_matching.latent.vae import VAE

LATENT_CHANNELS = 8
LATENT_H = 4
LATENT_W = 4


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 4, stride=2, padding=1),  # 16x16
            nn.GroupNorm(4, 32),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2, padding=1),  # 8x8
            nn.GroupNorm(8, 64),
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),  # 4x4
            nn.GroupNorm(8, 128),
            nn.ReLU(),
        )
        self.mu_layer = nn.Conv2d(128, LATENT_CHANNELS, kernel_size=3, padding=1)
        self.logvar_layer = nn.Conv2d(128, LATENT_CHANNELS, kernel_size=3, padding=1)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        h = self.net(x)  # B x 128 x 4 x 4
        mu = self.mu_layer(h)  # B x 32 x 4 x 4
        logvar = self.logvar_layer(h)
        return mu, logvar


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(LATENT_CHANNELS, 128, 3, padding=1),
            nn.GroupNorm(8, 128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),  # 8x8
            nn.GroupNorm(8, 64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),  # 16x16
            nn.GroupNorm(4, 32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1),  # 32x32
            nn.GroupNorm(4, 16),
            nn.ReLU(),
            nn.Conv2d(16, 1, kernel_size=3, padding=1),
            nn.Sigmoid(),  # MNIST is [0,1]
        )

    def forward(self, z: Tensor) -> Tensor:
        return self.net(z)


def vae_loss_binary(recon_x: Tensor, x: Tensor, mu: Tensor, logvar: Tensor):
    recon_loss = F.binary_cross_entropy(recon_x, x, reduction="sum")
    # KL divergence between N(mu, sigma) and N(0,1)
    kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kl_div


class MNISTVAE(VAE):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

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
