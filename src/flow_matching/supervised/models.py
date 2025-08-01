from abc import ABC, abstractmethod
from typing import List
import torch
from torch import nn
import math
import torch.nn.functional as F


class ConditionalVectorField(nn.Module, ABC):
    """
    MLP-parameterization of the learned vector field u_t^theta(x)
    """

    @abstractmethod
    def forward(
        self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
        - x: (bs, c, h, w)
        - t: (bs, 1, 1, 1)
        - y: (bs,)
        Returns:
        - u_t^theta(x|y): (bs, c, h, w)
        """
        pass


class FourierEncoder(nn.Module):
    """
    Based on https://github.com/lucidrains/denoising-diffusion-pytorch/blob/main/denoising_diffusion_pytorch/karras_unet.py#L183
    """

    def __init__(self, dim: int):
        super().__init__()
        assert dim % 2 == 0
        self.half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(1, self.half_dim))

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
        - t: (bs, 1, 1, 1)
        Returns:
        - embeddings: (bs, dim)
        """
        t = t.view(-1, 1)  # (bs, 1)
        freqs = t * self.weights * 2 * math.pi  # (bs, half_dim)
        sin_embed = torch.sin(freqs)  # (bs, half_dim)
        cos_embed = torch.cos(freqs)  # (bs, half_dim)
        return torch.cat([sin_embed, cos_embed], dim=-1) * math.sqrt(2)  # (bs, dim)


class ResidualLayer(nn.Module):
    def __init__(self, channels: int, time_embed_dim: int, y_embed_dim: int):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.SiLU(),
            nn.BatchNorm2d(channels),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
        )
        self.block2 = nn.Sequential(
            nn.SiLU(),
            nn.BatchNorm2d(channels),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
        )
        # Converts (bs, time_embed_dim) -> (bs, channels)
        self.time_adapter = nn.Sequential(
            nn.Linear(time_embed_dim, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, channels),
        )
        # Converts (bs, y_embed_dim) -> (bs, channels)
        self.y_adapter = nn.Sequential(
            nn.Linear(y_embed_dim, y_embed_dim),
            nn.SiLU(),
            nn.Linear(y_embed_dim, channels),
        )

    def forward(
        self, x: torch.Tensor, t_embed: torch.Tensor, y_embed: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
        - x: (bs, c, h, w)
        - t_embed: (bs, t_embed_dim)
        - y_embed: (bs, y_embed_dim)
        """
        res = x.clone()  # (bs, c, h, w)

        # Initial conv block
        x = self.block1(x)  # (bs, c, h, w)

        # Add time embedding
        t_embed = self.time_adapter(t_embed).unsqueeze(-1).unsqueeze(-1)
        # (bs, c, 1, 1)
        x = x + t_embed

        # Add y embedding (conditional embedding)
        y_embed = self.y_adapter(y_embed).unsqueeze(-1).unsqueeze(-1)  # (bs, c, 1, 1)
        x = x + y_embed

        # Second conv block
        x = self.block2(x)  # (bs, c, h, w)

        # Add back residual
        x = x + res  # (bs, c, h, w)

        return x


class Encoder(nn.Module):
    def __init__(
        self,
        channels_in: int,
        channels_out: int,
        num_residual_layers: int,
        t_embed_dim: int,
        y_embed_dim: int,
    ):
        super().__init__()
        self.res_blocks = nn.ModuleList(
            [
                ResidualLayer(channels_in, t_embed_dim, y_embed_dim)
                for _ in range(num_residual_layers)
            ]
        )
        self.downsample = nn.Conv2d(
            channels_in, channels_out, kernel_size=3, stride=2, padding=1
        )

    def forward(
        self, x: torch.Tensor, t_embed: torch.Tensor, y_embed: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
        - x: (bs, c_in, h, w)
        - t_embed: (bs, t_embed_dim)
        - y_embed: (bs, y_embed_dim)
        """
        # Pass through residual blocks: (bs, c_in, h, w) -> (bs, c_in, h, w)
        for block in self.res_blocks:
            x = block(x, t_embed, y_embed)

        # Downsample: (bs, c_in, h, w) -> (bs, c_out, h // 2, w // 2)
        x = self.downsample(x)

        return x


class Midcoder(nn.Module):
    def __init__(
        self,
        channels: int,
        num_residual_layers: int,
        t_embed_dim: int,
        y_embed_dim: int,
    ):
        super().__init__()
        self.res_blocks = nn.ModuleList(
            [
                ResidualLayer(channels, t_embed_dim, y_embed_dim)
                for _ in range(num_residual_layers)
            ]
        )

    def forward(
        self, x: torch.Tensor, t_embed: torch.Tensor, y_embed: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
        - x: (bs, c, h, w)
        - t_embed: (bs, t_embed_dim)
        - y_embed: (bs, y_embed_dim)
        """
        # Pass through residual blocks: (bs, c, h, w) -> (bs, c, h, w)
        for block in self.res_blocks:
            x = block(x, t_embed, y_embed)

        return x


class Decoder(nn.Module):
    def __init__(
        self,
        channels_in: int,
        channels_out: int,
        num_residual_layers: int,
        t_embed_dim: int,
        y_embed_dim: int,
    ):
        super().__init__()
        self.upsample = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear"),
            nn.Conv2d(channels_in, channels_out, kernel_size=3, padding=1),
        )
        self.res_blocks = nn.ModuleList(
            [
                ResidualLayer(channels_out, t_embed_dim, y_embed_dim)
                for _ in range(num_residual_layers)
            ]
        )

    def forward(
        self, x: torch.Tensor, t_embed: torch.Tensor, y_embed: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
        - x: (bs, c, h, w)
        - t_embed: (bs, t_embed_dim)
        - y_embed: (bs, y_embed_dim)
        """
        # Upsample: (bs, c_in, h, w) -> (bs, c_out, 2 * h, 2 * w)
        x = self.upsample(x)

        # Pass through residual blocks: (bs, c_out, h, w) -> (bs, c_out, 2 * h, 2 * w)
        for block in self.res_blocks:
            x = block(x, t_embed, y_embed)

        return x


class MNISTUNet(ConditionalVectorField):
    def __init__(
        self,
        channels: List[int],
        num_residual_layers: int,
        t_embed_dim: int,
        y_embed_dim: int,
    ):
        super().__init__()
        # Initial convolution: (bs, 1, 32, 32) -> (bs, c_0, 32, 32)
        self.init_conv = nn.Sequential(
            nn.Conv2d(1, channels[0], kernel_size=3, padding=1),
            nn.BatchNorm2d(channels[0]),
            nn.SiLU(),
        )

        # Initialize time embedder
        self.time_embedder = FourierEncoder(t_embed_dim)

        # Initialize y embedder
        self.y_embedder = nn.Embedding(num_embeddings=11, embedding_dim=y_embed_dim)

        # Encoders, Midcoders, and Decoders
        encoders = []
        decoders = []
        for curr_c, next_c in zip(channels[:-1], channels[1:]):
            encoders.append(
                Encoder(curr_c, next_c, num_residual_layers, t_embed_dim, y_embed_dim)
            )
            decoders.append(
                Decoder(next_c, curr_c, num_residual_layers, t_embed_dim, y_embed_dim)
            )
        self.encoders = nn.ModuleList(encoders)
        self.decoders = nn.ModuleList(reversed(decoders))

        self.midcoder = Midcoder(
            channels[-1], num_residual_layers, t_embed_dim, y_embed_dim
        )

        # Final convolution
        self.final_conv = nn.Conv2d(channels[0], 1, kernel_size=3, padding=1)

    def forward(
        self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
        - x: (bs, 1, 32, 32)
        - t: (bs, 1, 1, 1)
        - y: (bs,)
        Returns:
        - u_t^theta(x|y): (bs, 1, 32, 32)
        """
        # Embed t and y
        t_embed = self.time_embedder(t)  # (bs, time_embed_dim)
        y_embed = self.y_embedder(y)  # (bs, y_embed_dim)

        # Initial convolution
        x = self.init_conv(x)  # (bs, c_0, 32, 32)

        residuals = []

        # Encoders
        for encoder in self.encoders:
            x = encoder(
                x, t_embed, y_embed
            )  # (bs, c_i, h, w) -> (bs, c_{i+1}, h // 2, w //2)
            residuals.append(x.clone())

        # Midcoder
        x = self.midcoder(x, t_embed, y_embed)

        # Decoders
        for decoder in self.decoders:
            res = residuals.pop()  # (bs, c_i, h, w)
            x = x + res
            x = decoder(
                x, t_embed, y_embed
            )  # (bs, c_i, h, w) -> (bs, c_{i-1}, 2 * h, 2 * w)

        # Final convolution
        x = self.final_conv(x)  # (bs, 1, 32, 32)

        return x


class SimpleUNet(ConditionalVectorField):
    def __init__(
        self,
        in_channels=3,
        base_channels=32,
        out_channels=3,
        num_classes=10,
        t_embed_dim=64,
        y_embed_dim=64,
    ):
        super().__init__()

        # Time and label embeddings
        self.time_embedder = FourierEncoder(t_embed_dim)
        self.y_embedder = nn.Embedding(num_classes + 1, y_embed_dim)

        self.time_adapter = nn.Sequential(
            nn.Linear(t_embed_dim, t_embed_dim),
            nn.SiLU(),
            nn.Linear(t_embed_dim, base_channels * 2),
        )
        self.y_adapter = nn.Sequential(
            nn.Linear(y_embed_dim, y_embed_dim),
            nn.SiLU(),
            nn.Linear(y_embed_dim, base_channels * 2),
        )

        # Encoder
        self.enc_conv1 = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1)
        self.enc_conv2 = nn.Conv2d(
            base_channels, base_channels, kernel_size=3, padding=1
        )
        self.pool = nn.MaxPool2d(
            kernel_size=(2, 2), stride=(2, 2)
        )  # H: 4 → 2, W: 50 → 25

        # Bottleneck
        self.bottleneck_conv1 = nn.Conv2d(
            base_channels, base_channels * 2, kernel_size=3, padding=1
        )
        self.bottleneck_conv2 = nn.Conv2d(
            base_channels * 2, base_channels * 2, kernel_size=3, padding=1
        )

        # Decoder
        self.up = nn.Upsample(scale_factor=(2, 2), mode="bilinear", align_corners=False)
        self.dec_conv1 = nn.Conv2d(
            base_channels * 2 + base_channels, base_channels, kernel_size=3, padding=1
        )
        self.dec_conv2 = nn.Conv2d(
            base_channels, base_channels, kernel_size=3, padding=1
        )

        # Final output layer
        self.final_conv = nn.Conv2d(base_channels, out_channels, kernel_size=1)

    def forward(self, x, t, y):
        """
        Args:
        - x: (bs, 3, 4, 50)
        - t: (bs, 1, 1, 1)
        - y: (bs,)
        Returns:
        - out: (bs, out_channels, 4, 50)
        """
        # Get embeddings
        t_embed = self.time_embedder(t)  # (bs, t_embed_dim)
        y_embed = self.y_embedder(y)  # (bs, y_embed_dim)
        t_proj = self.time_adapter(t_embed).unsqueeze(-1).unsqueeze(-1)  # (bs, c, 1, 1)
        y_proj = self.y_adapter(y_embed).unsqueeze(-1).unsqueeze(-1)  # (bs, c, 1, 1)

        # Encoder
        x1 = F.relu(self.enc_conv1(x))  # (bs, base, 4, 50)
        x1 = F.relu(self.enc_conv2(x1))  # (bs, base, 4, 50)
        x_pooled = self.pool(x1)  # (bs, base, 2, 25)

        # Bottleneck
        x2 = F.relu(self.bottleneck_conv1(x_pooled))  # (bs, base*2, 2, 25)
        x2 = F.relu(self.bottleneck_conv2(x2))  # (bs, base*2, 2, 25)

        # Add time and label embedding
        x2 = x2 + t_proj + y_proj  # (bs, base*2, 2, 25)

        # Decoder
        x_up = self.up(x2)  # (bs, base*2, 4, 50)
        x_cat = torch.cat([x_up, x1], dim=1)  # (bs, base*3, 4, 50)
        x3 = F.relu(self.dec_conv1(x_cat))  # (bs, base, 4, 50)
        x3 = F.relu(self.dec_conv2(x3))  # (bs, base, 4, 50)

        return self.final_conv(x3)  # (bs, out_channels, 4, 50)
