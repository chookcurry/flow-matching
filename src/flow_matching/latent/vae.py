from torch import nn
from torch import Tensor
from typing import Tuple
from abc import ABC, abstractmethod


class VAE(nn.Module, ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def decode(self, z: Tensor) -> Tensor:
        # recon
        pass

    @abstractmethod
    def encode(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        # z, mu, logvar
        pass

    @abstractmethod
    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        # recon, mu, logvar
        pass
