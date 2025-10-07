from torch import nn, Tensor
from typing import Tuple, Optional
import torch

from diffusion.architectures.latent.autoencoder import AEC, CAE, CAEC, AE


class ConditionalGroupNorm(nn.Module):
    def __init__(self, num_features: int, num_groups: int, embedding_dim: int):
        super().__init__()
        self.groupnorm = nn.GroupNorm(num_groups, num_features, affine=False)
        self.embed_to_gamma = nn.Linear(embedding_dim, num_features)
        self.embed_to_beta = nn.Linear(embedding_dim, num_features)

    def forward(self, x: Tensor, embed: Tensor) -> Tensor:
        normalized = self.groupnorm(x)
        gamma = self.embed_to_gamma(embed).unsqueeze(-1).unsqueeze(-1)
        beta = self.embed_to_beta(embed).unsqueeze(-1).unsqueeze(-1)
        return gamma * normalized + beta


class ResidualBlock(nn.Module):
    def __init__(
        self,
        c: int,
        num_groups: int,
        dropout: float = 0.0,
        residual_scale: float = 1.0,
        embedding_dim: Optional[int] = None,
    ):
        super().__init__()

        self.norm1 = (
            ConditionalGroupNorm(c, num_groups, embedding_dim)
            if embedding_dim is not None
            else nn.GroupNorm(num_groups, c, affine=True)
        )
        self.conv1 = nn.Conv2d(c, c, kernel_size=3, padding=1)

        self.norm2 = (
            ConditionalGroupNorm(c, num_groups, embedding_dim)
            if embedding_dim is not None
            else nn.GroupNorm(num_groups, c, affine=True)
        )
        self.conv2 = nn.Conv2d(c, c, kernel_size=3, padding=1)

        self.activation = nn.SiLU()
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

        self.residual_scale = residual_scale
        self.conditional = embedding_dim is not None

    def forward(self, x: Tensor, embed: Optional[Tensor] = None) -> Tensor:
        residual = x

        out = self.norm1(x, embed) if self.conditional else self.norm1(x)
        out = self.activation(out)
        out = self.dropout(out)
        out = self.conv1(out)

        out = self.norm2(out, embed) if self.conditional else self.norm2(out)
        out = self.activation(out)
        out = self.dropout(out)
        out = self.conv2(out)

        return residual + self.residual_scale * out


class BaseEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_channels_latent: int,
        num_classes: int | None,
        embedding_dim: int | None,
        dropout: float = 0.0,
        residual_scale: float = 1.0,
    ):
        super().__init__()

        self.embed = (
            nn.Embedding(num_classes, embedding_dim)
            if num_classes is not None and embedding_dim is not None
            else None
        )

        def block(c, groups):
            return nn.ModuleList(
                [
                    ResidualBlock(c, groups, dropout, residual_scale, embedding_dim),
                    ResidualBlock(c, groups, dropout, residual_scale, embedding_dim),
                ]
            )

        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=4, stride=2, padding=1)
        self.resblock1 = block(32, 4)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1)
        self.resblock2 = block(64, 8)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)
        self.resblock3 = block(128, 16)

        self.z_proj = nn.Conv2d(128, num_channels_latent, kernel_size=3, padding=1)

        self.conditional = embedding_dim is not None

    def forward(self, x: Tensor, y: Optional[Tensor] = None) -> Tensor:
        embed = self.embed(y) if self.embed is not None else None

        h = self.conv1(x)
        for layer in self.resblock1:
            h = layer(h, embed) if self.conditional else layer(h)

        h = self.conv2(h)
        for layer in self.resblock2:
            h = layer(h, embed) if self.conditional else layer(h)

        h = self.conv3(h)
        for layer in self.resblock3:
            h = layer(h, embed) if self.conditional else layer(h)

        z: Tensor = torch.tanh(self.z_proj(h))
        return z


class BaseDecoder(nn.Module):
    def __init__(
        self,
        out_c: int,
        num_channels_latent: int,
        num_classes: int | None,
        embedding_dim: int | None,
        dropout: float = 0.0,
        residual_scale: float = 1.0,
    ):
        super().__init__()

        self.embed = (
            nn.Embedding(num_classes, embedding_dim)
            if num_classes is not None and embedding_dim is not None
            else None
        )

        def block(c, groups):
            return nn.ModuleList(
                [
                    ResidualBlock(c, groups, dropout, residual_scale, embedding_dim),
                    ResidualBlock(c, groups, dropout, residual_scale, embedding_dim),
                ]
            )

        self.conv1 = nn.Conv2d(num_channels_latent, 128, kernel_size=3, padding=1)
        self.resblock1 = block(128, 16)

        self.convT1 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.resblock2 = block(64, 8)

        self.convT2 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)
        self.resblock3 = block(32, 4)

        self.convT3 = nn.ConvTranspose2d(32, out_c, kernel_size=4, stride=2, padding=1)
        self.conditional = embedding_dim is not None

    def forward(self, z: Tensor, y: Optional[Tensor] = None) -> Tensor:
        embed = self.embed(y) if self.embed is not None else None

        h = self.conv1(z)
        for layer in self.resblock1:
            h = layer(h, embed) if self.conditional else layer(h)

        h = self.convT1(h)
        for layer in self.resblock2:
            h = layer(h, embed) if self.conditional else layer(h)

        h = self.convT2(h)
        for layer in self.resblock3:
            h = layer(h, embed) if self.conditional else layer(h)

        recon: Tensor = torch.tanh(self.convT3(h))
        return recon


# === Specializations ===


class SpectrogramAE(AE):
    def __init__(self, num_channels_spect: int, num_channels_latent: int):
        super().__init__()
        self.encoder = BaseEncoder(num_channels_spect, num_channels_latent, None, None)
        self.decoder = BaseDecoder(num_channels_spect, num_channels_latent, None, None)

    def encode(self, x: Tensor) -> Tensor:
        return self.encoder(x)

    def decode(self, z: Tensor) -> Tensor:
        return self.decoder(z)

    def forward(self, x: Tensor, y: Tensor | None = None) -> Tuple[Tensor, Tensor]:
        z = self.encode(x)
        recon = self.decode(z)
        return recon, z


class SpectrogramCAE(CAE):
    def __init__(
        self,
        num_channels_spect: int,
        num_channels_latent: int,
        num_classes: int,
        embedding_dim: int,
    ):
        super().__init__()
        self.encoder = BaseEncoder(
            num_channels_spect, num_channels_latent, num_classes, embedding_dim
        )
        self.decoder = BaseDecoder(
            num_channels_spect, num_channels_latent, num_classes, None
        )

    def encode(self, x: Tensor, y: Tensor) -> Tensor:
        return self.encoder(x, y)

    def decode(self, z: Tensor) -> Tensor:
        return self.decoder(z)

    def forward(self, x: Tensor, y: Tensor) -> Tuple[Tensor, Tensor]:
        z = self.encode(x, y)
        recon = self.decode(z)
        return recon, z


class SpectrogramAEC(AEC):
    def __init__(
        self,
        num_channels_spect: int,
        num_channels_latent: int,
        num_classes: int,
        embedding_dim: int,
    ):
        super().__init__()
        self.encoder = BaseEncoder(
            num_channels_spect, num_channels_latent, num_classes, None
        )
        self.decoder = BaseDecoder(
            num_channels_spect, num_channels_latent, num_classes, embedding_dim
        )

    def encode(self, x: Tensor) -> Tensor:
        return self.encoder(x)

    def decode(self, z: Tensor, y: Tensor) -> Tensor:
        return self.decoder(z, y)

    def forward(self, x: Tensor, y: Tensor) -> Tuple[Tensor, Tensor]:
        z = self.encode(x)
        recon = self.decode(z, y)
        return recon, z


class SpectrogramCAEC(CAEC):
    def __init__(
        self,
        num_channels_spect: int,
        num_channels_latent: int,
        num_classes: int,
        embedding_dim: int,
    ):
        super().__init__()
        self.encoder = BaseEncoder(
            num_channels_spect, num_channels_latent, num_classes, embedding_dim
        )
        self.decoder = BaseDecoder(
            num_channels_spect, num_channels_latent, num_classes, embedding_dim
        )

    def encode(self, x: Tensor, y: Tensor) -> Tensor:
        return self.encoder(x, y)

    def decode(self, z: Tensor, y: Tensor) -> Tensor:
        return self.decoder(z, y)

    def forward(self, x: Tensor, y: Tensor) -> Tuple[Tensor, Tensor]:
        z = self.encode(x, y)
        recon = self.decode(z, y)
        return recon, z
