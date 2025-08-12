from torch import nn
from torch import Tensor
from typing import Tuple
from abc import ABC, abstractmethod


class Autoencoder(nn.Module, ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def decode(self, z: Tensor) -> Tensor:
        # recon
        pass

    @abstractmethod
    def encode(self, x: Tensor) -> Tensor:
        # z,
        pass

    @abstractmethod
    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        # recon, z
        pass


class CondAutoencoder(nn.Module, ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def decode(self, z: Tensor, y: Tensor) -> Tensor:
        # recon
        pass

    @abstractmethod
    def encode(self, x: Tensor, y: Tensor) -> Tensor:
        # z
        pass

    @abstractmethod
    def forward(self, x: Tensor, y: Tensor) -> Tuple[Tensor, Tensor]:
        # recon, z
        pass
