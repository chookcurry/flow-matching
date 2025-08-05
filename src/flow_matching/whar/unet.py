from typing import List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from flow_matching.mnist.unet import FourierEncoder
from flow_matching.supervised.odes_sdes import ConditionalVectorField


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


class WHARResidualBlock(nn.Module):
    def __init__(self, num_channels: int, emb_dim: int):
        super(WHARResidualBlock, self).__init__()

        self.conv1 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1)

        self.film_t = nn.Linear(emb_dim, num_channels * 2)  # for γ and β
        self.film_y = nn.Linear(emb_dim, num_channels * 2)  # for γ and β

    def forward(self, x: Tensor, t: Tensor, y: Tensor) -> Tensor:
        res = x.clone()

        x = self.conv1(x)

        gamma_t, beta_t = self.film_t(t).chunk(2, 1)
        gamma_t, beta_t = gamma_t[:, :, None, None], beta_t[:, :, None, None]
        x = gamma_t * x + beta_t

        gamma_y, beta_y = self.film_y(y).chunk(2, 1)
        gamma_y, beta_y = gamma_y[:, :, None, None], beta_y[:, :, None, None]
        x = gamma_y * x + beta_y

        x = F.relu(x)

        x = self.conv2(x)

        x += res
        x = F.relu(x)

        return x


class WHAREncoder(nn.Module):
    def __init__(
        self,
        channels_in: int,
        channels_out: int,
        emb_dim: int,
        num_blocks: int,
    ):
        super(WHAREncoder, self).__init__()

        self.blocks = [
            WHARResidualBlock(channels_in, emb_dim) for _ in range(num_blocks)
        ]

        self.downsample = nn.Conv2d(
            channels_in, channels_out, kernel_size=3, stride=(1, 2), padding=1
        )

    def pad_to_even(self, x: Tensor) -> Tensor:
        h, w = x.shape[-2:]
        pad_h = h % 2
        pad_w = w % 2
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h))
        return x

    def forward(self, x: Tensor, t: Tensor, y: Tensor) -> Tuple[Tensor, Tensor]:
        for block in self.blocks:
            x = block(x, t, y)

        skip_con = x.clone()

        x = self.pad_to_even(x)
        x = self.downsample(x)

        return x, skip_con


class WHARMidcoder(nn.Module):
    def __init__(
        self,
        channels_in: int,
        emb_dim: int,
        num_blocks: int,
    ):
        super(WHARMidcoder, self).__init__()

        self.blocks = [
            WHARResidualBlock(channels_in, emb_dim) for _ in range(num_blocks)
        ]

    def forward(self, x: Tensor, t: Tensor, y: Tensor) -> Tensor:
        for block in self.blocks:
            x = block(x, t, y)
        return x


class WHARDecoder(nn.Module):
    def __init__(
        self,
        channels_in: int,
        channels_out: int,
        emb_dim: int,
        num_blocks: int,
    ):
        super(WHARDecoder, self).__init__()

        # self.upsample = nn.ConvTranspose2d(
        #     channels_in,
        #     channels_out,
        #     kernel_size=3,
        #     stride=2,
        #     padding=1,
        #     # output_padding=1,
        # )

        self.upsample = nn.Sequential(
            nn.Upsample(scale_factor=(1, 2), mode="bilinear"),
            nn.Conv2d(channels_in, channels_out, kernel_size=3, padding=1),
        )

        self.reduce = nn.Conv2d(channels_in, channels_out, kernel_size=3, padding=1)

        self.blocks = [
            WHARResidualBlock(channels_out, emb_dim) for _ in range(num_blocks)
        ]

    def crop_to_match(self, x: Tensor, target: Tensor) -> Tensor:
        _, _, h, w = target.shape
        return x[..., :h, :w]

    def forward(self, x: Tensor, t: Tensor, y: Tensor, skip_con: Tensor) -> Tensor:
        x = self.upsample(x)

        x = self.crop_to_match(x, skip_con)

        x = torch.cat([x, skip_con], dim=1)
        x = self.reduce(x)
        for block in self.blocks:
            x = block(x, t, y)
        return x


class WHARUnet(ConditionalVectorField):
    def __init__(
        self,
        in_channels: int,
        channels: List[int],
        num_blocks: int,
        emb_dim: int,
        num_classes: int,
    ):
        super(WHARUnet, self).__init__()

        self.t_embedder = FourierEncoder(emb_dim)
        self.y_embedder = nn.Embedding(num_classes + 1, emb_dim)

        self.init_conv = nn.Conv2d(in_channels, channels[0], kernel_size=3, padding=1)

        self.encoders = nn.ModuleList(
            [
                WHAREncoder(channels[i], channels[i + 1], emb_dim, num_blocks)
                for i in range(len(channels) - 1)
            ]
        )

        self.midcoder = WHARMidcoder(channels[-1], emb_dim, num_blocks)

        self.decoders = nn.ModuleList(
            [
                WHARDecoder(channels[i], channels[i - 1], emb_dim, num_blocks)
                for i in range(len(channels) - 1, 0, -1)
            ]
        )

        self.out_conv = nn.Conv2d(channels[0], in_channels, kernel_size=3, padding=1)

    def forward(self, x: Tensor, t: Tensor, y: Tensor) -> Tensor:
        t_emb = self.t_embedder(t)
        y_emb = self.y_embedder(y)

        x = self.init_conv(x)

        skip_cons = []

        for encoder in self.encoders:
            x, skip_con = encoder(x, t_emb, y_emb)
            skip_cons.append(skip_con)

        x = self.midcoder(x, t_emb, y_emb)

        for decoder in self.decoders:
            skip_con = skip_cons.pop()
            x = decoder(x, t_emb, y_emb, skip_con)

        x = self.out_conv(x)

        return x
