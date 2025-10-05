import math
from typing import Tuple
import torch
import torch.nn as nn
from torch import Tensor

from flow_matching.supervised.odes_sdes import Backbone


class FourierEncoder(nn.Module):
    def __init__(self, dim: int):
        super().__init__()

        assert dim % 2 == 0
        self.half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(1, self.half_dim))

    def forward(self, t: Tensor) -> Tensor:
        # t: (B,) or (B,1) or (B,1,1,1)

        t = t.view(-1, 1)
        freqs = t * self.weights * 2 * math.pi  # (B, half_dim)
        sin_embed = torch.sin(freqs)
        cos_embed = torch.cos(freqs)
        return torch.cat([sin_embed, cos_embed], dim=-1) * math.sqrt(2.0)  # (B, dim)


class FiLMHead(nn.Module):
    def __init__(self, cond_dim: int, n_channels: int, hidden: int | None = None):
        super().__init__()
        h = hidden or max(n_channels, cond_dim)

        self.net = nn.Sequential(
            nn.Linear(cond_dim, h), nn.ReLU(), nn.Linear(h, 2 * n_channels)
        )

        self.n_channels = n_channels

    def forward(self, cond: Tensor) -> Tuple[Tensor, Tensor]:
        # cond: (B, cond_dim)

        gammas: Tensor
        betas: Tensor

        gammas, betas = self.net(cond).chunk(2, dim=-1)  # (B, C), (B, C)
        # Optional: start near identity

        return gammas, betas


# ========= Residual block with FiLM =========
class ResidualFiLMBlock(nn.Module):
    def __init__(self, n_channels: int, film_head: FiLMHead, dropout: float = 0.0):
        super().__init__()
        self.conv1 = nn.Conv2d(n_channels, n_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(n_channels, n_channels, 3, padding=1)
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.film_head = film_head
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: Tensor, cond: Tensor):
        residual = x
        out = self.conv1(x)
        out = self.act(out)
        # FiLM after first conv
        gammas, betas = self.film_head(cond)  # (B, C), (B, C)
        gammas = gammas.unsqueeze(-1).unsqueeze(-1)  # (B, C, 1, 1)
        betas = betas.unsqueeze(-1).unsqueeze(-1)
        out = gammas * out + betas

        out = self.dropout(out)
        out = self.conv2(out)
        out = out + residual
        return self.act(out)


# ========= Full model =========
class FiLMNetMultiBlock(Backbone):
    """
    Input:  (B, 24, 4, 4)
    Output: (B, 24, 4, 4)
    Conditioning: label y (categorical) + time t (continuous via Fourier)
    Each residual block has its own FiLM head (different γ, β).
    """

    def __init__(
        self,
        in_channels: int = 24,
        hidden: int = 64,
        num_blocks: int = 5,
        y_classes: int = 10,
        y_dim: int = 16,
        t_fourier_dim: int = 32,
        cond_dim: int = 64,
        dropout: float = 0.0,
    ):
        super().__init__()

        # embeddings
        self.embed_y = nn.Embedding(y_classes + 1, y_dim)
        self.embed_t = FourierEncoder(t_fourier_dim)

        # shared conditioning backbone (small MLP)
        self.cond_backbone = nn.Sequential(
            nn.Linear(y_dim + t_fourier_dim, cond_dim),
            nn.ReLU(),
            nn.Linear(cond_dim, cond_dim),
            nn.ReLU(),
        )

        # (optional) learned block index embedding to let blocks specialize
        self.block_idx_emb = nn.Embedding(num_blocks, cond_dim)

        # stem / head
        self.in_conv = nn.Conv2d(in_channels, hidden, 3, padding=1)
        self.out_conv = nn.Conv2d(hidden, in_channels, 3, padding=1)

        # per-block FiLM heads + blocks
        blocks = []
        for b in range(num_blocks):
            film_head = FiLMHead(cond_dim=cond_dim, n_channels=hidden)
            block = ResidualFiLMBlock(hidden, film_head, dropout=dropout)
            blocks.append(block)
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x: Tensor, t: Tensor, y: Tensor):
        """
        x: (B, 24, 4, 4)
        y: (B,)  long
        t: (B,) or (B,1) or (B,1,1,1)  float
        """
        # conditioning vector
        y_emb = self.embed_y(y)  # (B, y_dim)
        t_emb = self.embed_t(t)  # (B, t_fourier_dim)
        cond = torch.cat([y_emb, t_emb], dim=-1)  # (B, y_dim + t_dim)
        cond = self.cond_backbone(cond)  # (B, cond_dim)

        h = self.in_conv(x)  # (B, hidden, 4, 4)

        # run residual FiLM blocks, add per-block index embedding
        for idx, block in enumerate(self.blocks):
            cond_b = cond + self.block_idx_emb.weight[idx]  # (B, cond_dim)
            h = block(h, cond_b)

        out = self.out_conv(h)  # (B, 24, 4, 4)
        return out


# ======== tiny usage example ========
if __name__ == "__main__":
    B = 8
    x = torch.randn(B, 24, 4, 4)
    y = torch.randint(0, 10, (B,))
    t = torch.rand(B) * 1.0  # any continuous range

    model = FiLMNetMultiBlock(
        in_channels=24,
        hidden=64,
        num_blocks=5,
        y_classes=10,
        y_dim=16,
        t_fourier_dim=32,
        cond_dim=64,
        dropout=0.05,
    )
    out = model(x, y, t)
    print(out.shape)  # torch.Size([8, 24, 4, 4])
