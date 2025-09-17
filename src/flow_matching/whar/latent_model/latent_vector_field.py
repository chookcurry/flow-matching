import torch
import torch.nn as nn
from torch import Tensor
import math

from flow_matching.supervised.odes_sdes import ConditionalVectorField


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, embedding_dim: int) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim

    def forward(self, t: Tensor) -> Tensor:
        """
        t: Tensor of shape [B, 1] or [B]
        Returns: [B, embedding_dim]
        """
        half_dim = self.embedding_dim // 2
        emb = torch.exp(
            torch.arange(half_dim, device=t.device) * -math.log(10000) / (half_dim - 1)
        )
        emb = t.unsqueeze(1) * emb  # [B, half_dim]
        return torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)  # [B, embedding_dim]


class FlowNetBackbone(ConditionalVectorField):
    def __init__(
        self, num_classes: int, latent_channels: int = 64, time_emb_dim: int = 128
    ) -> None:
        super().__init__()
        self.time_embed = SinusoidalTimeEmbedding(time_emb_dim)
        self.fc_time = nn.Sequential(
            nn.Linear(time_emb_dim, 128),
            nn.ReLU(),
            nn.Linear(128, latent_channels),  # To match latent channels for FiLM
        )

        def conv_block(in_ch, out_ch):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
                nn.GroupNorm(8, out_ch),
                nn.ReLU(),
            )

        # Label embedding
        self.label_embed = nn.Embedding(num_classes, 128)
        self.fc_label = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, latent_channels),
        )

        self.encoder = nn.Sequential(
            conv_block(latent_channels, 128),
            conv_block(128, 128),
        )

        self.decoder = nn.Sequential(
            conv_block(128, 128),
            nn.Conv2d(
                128, latent_channels, kernel_size=3, padding=1
            ),  # Predict velocity
        )

    def forward(self, x: Tensor, t: Tensor, y: Tensor) -> Tensor:
        """
        z: Tensor of shape [B, C=64, 4, 4]
        t: Tensor of shape [B] or [B, 1]
        Returns: velocity field, same shape as z
        """
        B, C, H, W = x.shape
        time_emb = self.time_embed(t.view(B))  # [B, time_emb_dim]
        scale_shift = self.fc_time(time_emb).view(B, C, 1, 1)  # [B, C, 1, 1]

        # FiLM-like modulation (scale only)
        x_mod = x + scale_shift

        h = self.encoder(x_mod)
        v = self.decoder(h)
        return v  # velocity field
