from torch import nn, Tensor
from typing import Tuple
import torch.nn.functional as F

from diffusion.architectures.latent.autoencoder import AEC, CAE, CAEC, AE


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

        out: Tensor = gamma * normalized + beta

        return out


class ResidualBlock(nn.Module):
    def __init__(self, c: int, num_groups: int):
        super().__init__()

        self.norm1 = nn.GroupNorm(num_groups, c, affine=True)
        self.conv1 = nn.Conv2d(c, c, kernel_size=3, padding=1)

        self.norm2 = nn.GroupNorm(num_groups, c, affine=True)
        self.conv2 = nn.Conv2d(c, c, kernel_size=3, padding=1)

        self.relu = nn.ReLU()

    def forward(self, x: Tensor) -> Tensor:
        residual = x

        out: Tensor = self.norm1(x)
        out = self.relu(out)
        out = self.conv1(out)

        out = self.norm2(out)
        out = self.relu(out)
        out = self.conv2(out)

        return out + residual


class ConditionalResidualBlock(nn.Module):
    def __init__(self, c: int, num_groups: int, embedding_dim: int):
        super().__init__()

        self.norm1 = ConditionalGroupNorm(c, num_groups, embedding_dim)
        self.conv1 = nn.Conv2d(c, c, kernel_size=3, padding=1)

        self.norm2 = ConditionalGroupNorm(c, num_groups, embedding_dim)
        self.conv2 = nn.Conv2d(c, c, kernel_size=3, padding=1)

        self.relu = nn.ReLU()

    def forward(self, x: Tensor, embed: Tensor) -> Tensor:
        residual = x

        out: Tensor = self.norm1(x, embed)
        out = self.relu(out)
        out = self.conv1(out)

        out = self.norm2(out, embed)
        out = self.relu(out)
        out = self.conv2(out)

        return out + residual


class Encoder(nn.Module):
    def __init__(self, in_c: int, num_channels_latent: int):
        super().__init__()

        self.conv1 = nn.Conv2d(in_c, 32, kernel_size=4, stride=2, padding=1)
        self.resblock1 = nn.Sequential(ResidualBlock(32, 4), ResidualBlock(32, 4))

        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1)
        self.resblock2 = nn.Sequential(ResidualBlock(64, 8), ResidualBlock(64, 8))

        self.conv3 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)
        self.resblock3 = nn.Sequential(ResidualBlock(128, 16), ResidualBlock(128, 16))
        self.z_proj = nn.Conv2d(128, num_channels_latent, kernel_size=3, padding=1)

    def forward(self, x: Tensor) -> Tensor:
        h = self.conv1(x)
        for block in self.resblock1:
            h = block(h)

        h = self.conv2(h)
        for block in self.resblock2:
            h = block(h)

        h = self.conv3(h)
        for block in self.resblock3:
            h = block(h)

        z: Tensor = self.z_proj(h)
        z = F.tanh(z)

        return z


class ConditionalEncoder(nn.Module):
    def __init__(
        self, in_c: int, num_channels_latent: int, num_classes: int, embedding_dim: int
    ):
        super().__init__()

        self.embed = nn.Embedding(num_classes, embedding_dim)

        self.conv1 = nn.Conv2d(in_c, 32, kernel_size=4, stride=2, padding=1)
        self.resblock1 = nn.Sequential(
            ConditionalResidualBlock(32, 4, embedding_dim),
            ConditionalResidualBlock(32, 4, embedding_dim),
        )

        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1)
        self.resblock2 = nn.Sequential(
            ConditionalResidualBlock(64, 8, embedding_dim),
            ConditionalResidualBlock(64, 8, embedding_dim),
        )

        self.conv3 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)
        self.resblock3 = nn.Sequential(
            ConditionalResidualBlock(128, 16, embedding_dim),
            ConditionalResidualBlock(128, 16, embedding_dim),
        )

        self.z_proj = nn.Conv2d(128, num_channels_latent, kernel_size=3, padding=1)

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

        z: Tensor = self.z_proj(h)
        z = F.tanh(z)

        return z


class Decoder(nn.Module):
    def __init__(self, out_c: int, num_channels_latent: int):
        super().__init__()

        self.conv1 = nn.Conv2d(num_channels_latent, 128, kernel_size=3, padding=1)
        self.resblock1 = nn.Sequential(ResidualBlock(128, 16), ResidualBlock(128, 16))

        self.convT1 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.resblock2 = nn.Sequential(ResidualBlock(64, 8), ResidualBlock(64, 8))

        self.convT2 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)
        self.resblock3 = nn.Sequential(ResidualBlock(32, 4), ResidualBlock(32, 4))

        self.convT3 = nn.ConvTranspose2d(32, out_c, kernel_size=4, stride=2, padding=1)

    def forward(self, z: Tensor) -> Tensor:
        h: Tensor = self.conv1(z)
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


class ConditionalDecoder(nn.Module):
    def __init__(
        self, out_c: int, num_channels_latent: int, num_classes: int, embedding_dim: int
    ):
        super().__init__()

        self.embed = nn.Embedding(num_classes, embedding_dim)

        self.conv1 = nn.Conv2d(num_channels_latent, 128, kernel_size=3, padding=1)
        self.resblock1 = nn.Sequential(
            ConditionalResidualBlock(128, 16, embedding_dim),
            ConditionalResidualBlock(128, 16, embedding_dim),
        )

        self.convT1 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.resblock2 = nn.Sequential(
            ConditionalResidualBlock(64, 8, embedding_dim),
            ConditionalResidualBlock(64, 8, embedding_dim),
        )

        self.convT2 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)
        self.resblock3 = nn.Sequential(
            ConditionalResidualBlock(32, 4, embedding_dim),
            ConditionalResidualBlock(32, 4, embedding_dim),
        )

        self.convT3 = nn.ConvTranspose2d(32, out_c, kernel_size=4, stride=2, padding=1)

    def forward(self, z: Tensor, y: Tensor) -> Tensor:
        embed = self.embed(y)

        h: Tensor = self.conv1(z)
        for block in self.resblock1:
            h = block(h, embed)

        h = self.convT1(h)
        for block in self.resblock2:
            h = block(h, embed)

        h = self.convT2(h)
        for block in self.resblock3:
            h = block(h, embed)

        h = self.convT3(h)
        h = F.tanh(h)

        return h


class SpectrogramAE(AE):
    def __init__(self, num_channels_spect: int, num_channels_latent: int):
        super().__init__()

        self.encoder = Encoder(num_channels_spect, num_channels_latent)
        self.decoder = Decoder(num_channels_spect, num_channels_latent)

    def encode(self, x: Tensor) -> Tensor:
        z: Tensor = self.encoder(x)
        return z

    def decode(self, z: Tensor) -> Tensor:
        x: Tensor = self.decoder(z)
        return x

    def forward(self, x: Tensor, y: Tensor) -> Tuple[Tensor, Tensor]:
        z: Tensor = self.encode(x)
        recon: Tensor = self.decode(z)
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

        self.encoder = ConditionalEncoder(
            num_channels_spect, num_channels_latent, num_classes, embedding_dim
        )

        self.decoder = Decoder(num_channels_spect, num_channels_latent)

    def encode(self, x: Tensor, y: Tensor) -> Tensor:
        z: Tensor = self.encoder(x, y)
        return z

    def decode(self, z: Tensor) -> Tensor:
        x: Tensor = self.decoder(z)
        return x

    def forward(self, x: Tensor, y: Tensor) -> Tuple[Tensor, Tensor]:
        z: Tensor = self.encode(x, y)
        recon: Tensor = self.decode(z)
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

        self.encoder = Encoder(num_channels_spect, num_channels_latent)

        self.decoder = ConditionalDecoder(
            num_channels_spect, num_channels_latent, num_classes, embedding_dim
        )

    def encode(self, x: Tensor) -> Tensor:
        z: Tensor = self.encoder(x)
        return z

    def decode(self, z: Tensor, y: Tensor) -> Tensor:
        x: Tensor = self.decoder(z, y)
        return x

    def forward(self, x: Tensor, y: Tensor) -> Tuple[Tensor, Tensor]:
        z: Tensor = self.encode(x)
        recon: Tensor = self.decode(z, y)
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

        self.encoder = ConditionalEncoder(
            num_channels_spect, num_channels_latent, num_classes, embedding_dim
        )

        self.decoder = ConditionalDecoder(
            num_channels_spect, num_channels_latent, num_classes, embedding_dim
        )

    def encode(self, x: Tensor, y: Tensor) -> Tensor:
        z: Tensor = self.encoder(x, y)
        return z

    def decode(self, z: Tensor, y: Tensor) -> Tensor:
        x: Tensor = self.decoder(z, y)
        return x

    def forward(self, x: Tensor, y: Tensor) -> Tuple[Tensor, Tensor]:
        z: Tensor = self.encode(x, y)
        recon: Tensor = self.decode(z, y)
        return recon, z
