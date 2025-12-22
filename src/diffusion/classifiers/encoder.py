from abc import ABC, abstractmethod
from typing import Dict, Tuple

import torch
import torch.nn as nn
from torch import Tensor


class Encoder(ABC, nn.Module):
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def encode(self, x: Tensor) -> Tensor:
        pass


@torch.no_grad()
def compute_features(
    samples_synth: Dict[int, Tensor], samples_real: Dict[int, Tensor], encoder: Encoder
) -> Tuple[Dict[int, Tensor], Dict[int, Tensor]]:
    encoder.eval()

    assert len(samples_synth) == len(samples_real)

    features_synth = {key: encoder(value) for key, value in samples_synth.items()}
    features_real = {key: encoder(samples_real[key]) for key in samples_real}

    return features_synth, features_real
