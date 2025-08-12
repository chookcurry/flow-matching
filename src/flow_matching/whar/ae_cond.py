from torch import nn, Tensor
from typing import Tuple
import torch.nn.functional as F

from flow_matching.latent.ae import CondAutoencoder


class ConditionalGroupNorm(nn.Module):
    def __init__(self, num_features: int, num_groups: int, embedding_dim: int):
        super().__init__()
        self.num_features = num_features
        self.num_groups = num_groups
        self.groupnorm = nn.GroupNorm(num_groups, num_features, affine=False)

        # MLPs to produce scale (gamma) and shift (beta) from class embedding
        self.embed_to_gamma = nn.Linear(embedding_dim, num_features)
        self.embed_to_beta = nn.Linear(embedding_dim, num_features)

    def forward(self, x: Tensor, embed: Tensor) -> Tensor:
        normalized = self.groupnorm(x)
        gamma = self.embed_to_gamma(embed).unsqueeze(-1).unsqueeze(-1)
        beta = self.embed_to_beta(embed).unsqueeze(-1).unsqueeze(-1)
        return gamma * normalized + beta


class ResidualBlock(nn.Module):
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


class Encoder(nn.Module):
    def __init__(self, latent_channels: int, num_classes: int, embedding_dim: int = 32):
        super().__init__()
        self.embed = nn.Embedding(num_classes, embedding_dim)

        self.conv1 = nn.Conv2d(18, 32, kernel_size=4, stride=2, padding=1)  # downsample
        self.resblock1 = nn.Sequential(
            ResidualBlock(32, 4, embedding_dim), ResidualBlock(32, 4, embedding_dim)
        )

        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1)  # downsample
        self.resblock2 = nn.Sequential(
            ResidualBlock(64, 8, embedding_dim), ResidualBlock(64, 8, embedding_dim)
        )

        self.conv3 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)
        # 64, 128, kernel_size=3, stride=1, padding=1 # downsample
        self.resblock3 = nn.Sequential(
            ResidualBlock(128, 16, embedding_dim), ResidualBlock(128, 16, embedding_dim)
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


# class Decoder(nn.Module):
#     def __init__(self, latent_channels: int, num_classes: int, embedding_dim: int = 32):
#         super().__init__()
#         self.embed = nn.Embedding(num_classes, embedding_dim)

#         self.conv1 = nn.Conv2d(latent_channels, 128, kernel_size=3, padding=1)
#         self.resblock1 = nn.Sequential(
#             ResidualBlock(128, 16, embedding_dim), ResidualBlock(128, 16, embedding_dim)
#         )

#         # PixelShuffle upsample 8x8 -> 16x16
#         self.conv1_up = nn.Conv2d(128, 64 * 4, kernel_size=3, padding=1)
#         self.pixel_shuffle1 = nn.PixelShuffle(upscale_factor=2)
#         self.resblock2 = nn.Sequential(
#             ResidualBlock(64, 8, embedding_dim), ResidualBlock(64, 8, embedding_dim)
#         )

#         # PixelShuffle upsample 16x16 -> 32x32
#         self.conv2_up = nn.Conv2d(64, 32 * 4, kernel_size=3, padding=1)
#         self.pixel_shuffle2 = nn.PixelShuffle(upscale_factor=2)
#         self.resblock3 = nn.Sequential(
#             ResidualBlock(32, 4, embedding_dim), ResidualBlock(32, 4, embedding_dim)
#         )

#         # Final PixelShuffle upsample to output
#         self.conv3_up = nn.Conv2d(32, 18 * 4, kernel_size=3, padding=1)
#         self.pixel_shuffle3 = nn.PixelShuffle(upscale_factor=2)

#     def forward(self, z: Tensor, y: Tensor) -> Tensor:
#         embed = self.embed(y)

#         h = self.conv1(z)
#         for block in self.resblock1:
#             h = block(h, embed)

#         h = self.conv1_up(h)
#         h = self.pixel_shuffle1(h)
#         for block in self.resblock2:
#             h = block(h, embed)

#         h = self.conv2_up(h)
#         h = self.pixel_shuffle2(h)
#         for block in self.resblock3:
#             h = block(h, embed)

#         h = self.conv3_up(h)
#         h = self.pixel_shuffle3(h)

#         h = F.tanh(h)
#         return h


# class Decoder(nn.Module):
#     def __init__(self, latent_channels: int, num_classes: int, embedding_dim: int = 32):
#         super().__init__()
#         self.embed = nn.Embedding(num_classes, embedding_dim)

#         self.conv1 = nn.Conv2d(latent_channels, 128, kernel_size=3, padding=1)
#         self.resblock1 = nn.Sequential(
#             ResidualBlock(128, 16, embedding_dim), ResidualBlock(128, 16, embedding_dim)
#         )

#         # Upsample 8x8 -> 16x16
#         self.upsample1 = nn.Upsample(
#             scale_factor=2, mode="bilinear", align_corners=False
#         )
#         self.conv1_up = nn.Conv2d(128, 64, kernel_size=3, padding=1)
#         self.resblock2 = nn.Sequential(
#             ResidualBlock(64, 8, embedding_dim), ResidualBlock(64, 8, embedding_dim)
#         )

#         # Upsample 16x16 -> 32x32
#         self.upsample2 = nn.Upsample(
#             scale_factor=2, mode="bilinear", align_corners=False
#         )
#         self.conv2_up = nn.Conv2d(64, 32, kernel_size=3, padding=1)
#         self.resblock3 = nn.Sequential(
#             ResidualBlock(32, 4, embedding_dim), ResidualBlock(32, 4, embedding_dim)
#         )

#         # Final upsample to get output size (if needed)
#         self.upsample3 = nn.Upsample(
#             scale_factor=2, mode="bilinear", align_corners=False
#         )
#         self.conv3_up = nn.Conv2d(32, 18, kernel_size=3, padding=1)

#     def forward(self, z: Tensor, y: Tensor) -> Tensor:
#         embed = self.embed(y)

#         h = self.conv1(z)
#         for block in self.resblock1:
#             h = block(h, embed)

#         h = self.upsample1(h)
#         h = self.conv1_up(h)
#         for block in self.resblock2:
#             h = block(h, embed)

#         h = self.upsample2(h)
#         h = self.conv2_up(h)
#         for block in self.resblock3:
#             h = block(h, embed)

#         h = self.upsample3(h)
#         h = self.conv3_up(h)

#         h = F.tanh(h)

#         return h


class Decoder(nn.Module):
    def __init__(self, latent_channels: int, num_classes: int, embedding_dim: int = 32):
        super().__init__()
        self.embed = nn.Embedding(num_classes, embedding_dim)

        self.conv1 = nn.Conv2d(latent_channels, 128, kernel_size=3, padding=1)
        self.resblock1 = nn.Sequential(
            ResidualBlock(128, 16, embedding_dim), ResidualBlock(128, 16, embedding_dim)
        )

        self.convT1 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.resblock2 = nn.Sequential(
            ResidualBlock(64, 8, embedding_dim), ResidualBlock(64, 8, embedding_dim)
        )

        self.convT2 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)
        self.resblock3 = nn.Sequential(
            ResidualBlock(32, 4, embedding_dim), ResidualBlock(32, 4, embedding_dim)
        )

        self.convT3 = nn.ConvTranspose2d(
            32, 18, kernel_size=4, stride=2, padding=1
        )  # 32, 18, kernel_size=3, stride=1, padding=1)

    def forward(self, z: Tensor, y: Tensor) -> Tensor:
        embed = self.embed(y)

        h = self.conv1(z)
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


class CondSpectrogramAE(CondAutoencoder):
    def __init__(
        self, latent_channels: int = 64, num_classes: int = 10, embedding_dim: int = 32
    ):
        super().__init__()
        self.encoder = Encoder(latent_channels, num_classes, embedding_dim)
        self.decoder = Decoder(latent_channels, num_classes, embedding_dim)

    def encode(self, x: Tensor, y: Tensor) -> Tensor:
        z = self.encoder(x, y)
        return z

    def decode(self, z: Tensor, y: Tensor) -> Tensor:
        return self.decoder(z, y)

    def forward(self, x: Tensor, y: Tensor) -> Tuple[Tensor, Tensor]:
        z = self.encode(x, y)
        recon = self.decode(z, y)
        return recon, z
