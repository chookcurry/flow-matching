from torch import nn, Tensor
from typing import Tuple, Optional
import torch

from flow_matching.architectures.autoencoder import AEC, CAE, CAEC, AE


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
        num_downsamples: int,
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

        self.convs = nn.ModuleList()
        self.res_blocks = nn.ModuleList()

        channels = [in_channels] + [32 * 2**i for i in range(num_downsamples)]
        # e.g. 18, 32, 64, 128 for num_downsamples = 3

        groups = [4 * 2**i for i in range(num_downsamples)]
        # e.g. 4, 8, 16 for num_downsamples = 3

        for i in range(num_downsamples):
            self.convs.append(
                nn.Conv2d(
                    channels[i], channels[i + 1], kernel_size=4, stride=2, padding=1
                )
            )
            self.res_blocks.append(block(channels[i + 1], groups[i]))

        self.z_proj = nn.Conv2d(
            channels[-1], num_channels_latent, kernel_size=3, padding=1
        )

        self.conditional = embedding_dim is not None

    def forward(self, x: Tensor, y: Optional[Tensor] = None) -> Tensor:
        embed = self.embed(y) if self.embed is not None else None

        h = x
        for conv, block in zip(self.convs, self.res_blocks):
            h = conv(h)
            for layer in block:  # type: ignore
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
        num_downsamples: int,
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

        self.convs = nn.ModuleList()
        self.res_blocks = nn.ModuleList()

        channels = [32 * 2**i for i in range(num_downsamples - 1, -1, -1)] + [out_c]
        # e.g. 128, 64, 32, 18 for num_downsamples = 3

        groups = [4 * 2**i for i in range(num_downsamples - 1, -1, -1)]
        # e.g. 16, 8, 4 for num_downsamples = 3

        self.z_proj = nn.Conv2d(
            num_channels_latent, channels[0], kernel_size=3, padding=1
        )

        for i in range(num_downsamples):
            self.res_blocks.append(block(channels[i], groups[i]))
            self.convs.append(
                nn.ConvTranspose2d(
                    channels[i], channels[i + 1], kernel_size=4, stride=2, padding=1
                )
            )

        self.conditional = embedding_dim is not None

    def forward(self, z: Tensor, y: Optional[Tensor] = None) -> Tensor:
        embed = self.embed(y) if self.embed is not None else None

        h = self.z_proj(z)
        for conv, block in zip(self.convs, self.res_blocks):
            for layer in block:  # type: ignore
                h = layer(h, embed) if self.conditional else layer(h)
            h = conv(h)

        recon: Tensor = torch.tanh(h)
        return recon


# === Specializations ===


class SpectrogramAE(AE):
    def __init__(
        self, num_channels_spect: int, num_channels_latent: int, num_downsamples: int
    ):
        super().__init__()
        self.encoder = BaseEncoder(
            num_channels_spect, num_channels_latent, None, None, num_downsamples
        )
        self.decoder = BaseDecoder(
            num_channels_spect, num_channels_latent, None, None, num_downsamples
        )

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
        num_downsamples: int,
    ):
        super().__init__()
        self.encoder = BaseEncoder(
            num_channels_spect,
            num_channels_latent,
            num_classes,
            embedding_dim,
            num_downsamples,
        )
        self.decoder = BaseDecoder(
            num_channels_spect,
            num_channels_latent,
            num_classes,
            None,
            num_downsamples,
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
        num_downsamples: int,
    ):
        super().__init__()
        self.encoder = BaseEncoder(
            num_channels_spect,
            num_channels_latent,
            num_classes,
            None,
            num_downsamples,
        )
        self.decoder = BaseDecoder(
            num_channels_spect,
            num_channels_latent,
            num_classes,
            embedding_dim,
            num_downsamples,
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
        num_downsamples: int,
    ):
        super().__init__()
        self.encoder = BaseEncoder(
            num_channels_spect,
            num_channels_latent,
            num_classes,
            embedding_dim,
            num_downsamples,
        )
        self.decoder = BaseDecoder(
            num_channels_spect,
            num_channels_latent,
            num_classes,
            embedding_dim,
            num_downsamples,
        )

    def encode(self, x: Tensor, y: Tensor) -> Tensor:
        return self.encoder(x, y)

    def decode(self, z: Tensor, y: Tensor) -> Tensor:
        return self.decoder(z, y)

    def forward(self, x: Tensor, y: Tensor) -> Tuple[Tensor, Tensor]:
        z = self.encode(x, y)
        recon = self.decode(z, y)
        return recon, z
