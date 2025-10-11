import torch
import torch.nn as nn
from enum import Enum
from typing import Tuple


# ============================================================
# Core Blocks
# ============================================================


class FeedForward(nn.Sequential):
    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.0):
        super().__init__(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )


class TransformerBlock(nn.Module):
    def __init__(
        self, dim: int, heads: int = 4, dropout: float = 0.0, mlp_dim: int = 128
    ):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=dim, num_heads=heads, dropout=dropout, batch_first=True
        )
        self.ff = FeedForward(dim, mlp_dim, dropout)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.attn(self.norm1(x), self.norm1(x), self.norm1(x))
        x = x + attn_out
        x = x + self.ff(self.norm2(x))
        return x


class SelfAttentionInteraction(nn.Module):
    """Self-attention across channels."""

    def __init__(self, sensor_channel: int, n_channels: int):
        super().__init__()
        self.attn = nn.MultiheadAttention(n_channels, num_heads=1, batch_first=True)
        self.gamma = nn.Parameter(torch.tensor([0.0]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: [B, C, F]
        out, _ = self.attn(x, x, x)
        return x + self.gamma * out


class WeightedAggregation(nn.Module):
    """Softmax-weighted aggregation across channels."""

    def __init__(self, sensor_channel: int, hidden_dim: int):
        super().__init__()
        self.score = nn.Linear(hidden_dim, 1)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: [B, C, F]
        weights = self.softmax(self.score(x).squeeze(-1))  # [B, C]
        context = torch.einsum("bcf,bc->bf", x, weights)
        return context


# ============================================================
# Enums + Registries (compact)
# ============================================================


class CrossChannelType(Enum):
    ATTENTION = "attn"
    TRANSFORMER = "transformer"
    IDENTITY = "identity"
    FCINTER = "fcinter"


CROSSCHANNEL_MAP = {
    CrossChannelType.ATTENTION: SelfAttentionInteraction,
    CrossChannelType.TRANSFORMER: lambda c, f: TransformerBlock(f),
    CrossChannelType.IDENTITY: lambda c, f: nn.Identity(),
    CrossChannelType.FCINTER: lambda c, f: nn.Sequential(
        nn.Linear(c, c), nn.ReLU(), nn.Linear(f, f)
    ),
}


class CrossChannelAggType(Enum):
    FILTER = "filter"
    NAIVE = "naive"
    FC = "fc"


CROSSCHANNEL_AGG_MAP = {
    CrossChannelAggType.FILTER: WeightedAggregation,
    CrossChannelAggType.NAIVE: WeightedAggregation,
    CrossChannelAggType.FC: lambda c, f: nn.Linear(c, f),
}


class TemporalType(Enum):
    GRU = "gru"
    LSTM = "lstm"
    ATTENTION = "attn"
    TRANSFORMER = "transformer"
    IDENTITY = "identity"
    CONV = "conv"


TEMPORAL_MAP = {
    TemporalType.GRU: lambda c, f: nn.GRU(f, f, batch_first=True),
    TemporalType.LSTM: lambda c, f: nn.LSTM(f, f, batch_first=True),
    TemporalType.ATTENTION: SelfAttentionInteraction,
    TemporalType.TRANSFORMER: lambda c, f: TransformerBlock(f),
    TemporalType.IDENTITY: lambda c, f: nn.Identity(),
    TemporalType.CONV: lambda c, f: nn.Sequential(
        nn.Conv1d(f, f, kernel_size=5, padding="same"), nn.ReLU()
    ),
}


class TemporalAggType(Enum):
    NAIVE = "naive"
    TNAIVE = "tnaive"
    FC = "fc"
    IDENTITY = "identity"


TEMPORAL_AGG_MAP = {
    TemporalAggType.NAIVE: WeightedAggregation,
    TemporalAggType.TNAIVE: WeightedAggregation,
    TemporalAggType.FC: lambda c, f: nn.Linear(c * f, f),
    TemporalAggType.IDENTITY: lambda c, f: nn.Identity(),
}


# ============================================================
# Main Model
# ============================================================


class TinyHAR_Model(nn.Module):
    def __init__(
        self,
        input_shape: Tuple[int, int, int, int],
        number_class: int,
        filter_num: int,
        nb_conv_layers: int = 4,
        cross_channel_interaction_type: CrossChannelType = CrossChannelType.ATTENTION,
        cross_channel_aggregation_type: CrossChannelAggType = CrossChannelAggType.FILTER,
        temporal_info_interaction_type: TemporalType = TemporalType.GRU,
        temporal_info_aggregation_type: TemporalAggType = TemporalAggType.FC,
        dropout: float = 0.1,
    ):
        super().__init__()

        C_in = input_shape[3]

        # PART 1 , ============= Channel-wise Feature Extraction =============================
        convs = []
        for i in range(nb_conv_layers):
            stride = (2, 1) if i % 2 == 0 else (1, 1)
            convs.append(
                nn.Sequential(
                    nn.Conv2d(1 if i == 0 else filter_num, filter_num, (5, 1), stride),
                    nn.ReLU(),
                    nn.BatchNorm2d(filter_num),
                )
            )
        self.layers_conv = nn.Sequential(*convs)

        down_len = self.get_down_len(input_shape)

        # PART 2 , ================ Cross Channel Interaction =================================
        self.channel_interaction = CROSSCHANNEL_MAP[cross_channel_interaction_type](
            C_in, filter_num
        )

        # PART 3 , =============== Cross Channel Fusion =======================================
        if cross_channel_aggregation_type == CrossChannelAggType.FC:
            self.channel_fusion = CROSSCHANNEL_AGG_MAP[cross_channel_aggregation_type](
                C_in * filter_num, 2 * filter_num
            )
        else:
            self.channel_fusion = CROSSCHANNEL_AGG_MAP[cross_channel_aggregation_type](
                C_in, 2 * filter_num
            )

        # PART 4 , ============= Temporal Interaction =========================================
        self.temporal_interaction = TEMPORAL_MAP[temporal_info_interaction_type](
            C_in, 2 * filter_num
        )

        # PART 5 , ================= Temporal Aggregation =====================================
        if temporal_info_aggregation_type == TemporalAggType.FC:
            self.flatten = nn.Flatten()
            self.temporal_fusion = TEMPORAL_AGG_MAP[temporal_info_aggregation_type](
                down_len, 2 * filter_num
            )
        else:
            self.temporal_fusion = TEMPORAL_AGG_MAP[temporal_info_aggregation_type](
                C_in, 2 * filter_num
            )

        self.dropout = nn.Dropout(dropout)

        # PART 6 , =================== Prediction ============================================
        self.prediction = nn.Linear(2 * filter_num, number_class)

    def get_down_len(self, input_shape: Tuple[int, int, int, int]) -> int:
        x = torch.rand(input_shape)
        x = self.layers_conv(x)
        return x.shape[2]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # B F L C
        x = self.layers_conv(x)  # -> [B, F*, L*, C]
        x = x.permute(0, 3, 2, 1)  # -> [B, C, L*, F*]

        # cross-channel interaction
        x = torch.stack(
            [self.channel_interaction(x[:, :, t, :]) for t in range(x.shape[2])], dim=2
        )
        x = self.dropout(x)

        # cross-channel fusion
        if isinstance(self.channel_fusion, nn.Linear):  # FC fusion
            B, C, L, F = x.shape
            x = x.permute(0, 2, 1, 3).reshape(B, L, -1)
            x = self.channel_fusion(x)
        else:
            B, C, L, F = x.shape
            x = torch.stack(
                [self.channel_fusion(x[:, :, t, :]) for t in range(L)], dim=1
            )

        # temporal interaction
        if isinstance(self.temporal_interaction, (nn.GRU, nn.LSTM)):
            x, _ = self.temporal_interaction(x)
        else:
            x = self.temporal_interaction(x)

        # temporal aggregation
        if hasattr(self, "flatten"):
            x = self.flatten(x)
            x = self.temporal_fusion(x)
        else:
            x = self.temporal_fusion(x)

        return self.prediction(x)
