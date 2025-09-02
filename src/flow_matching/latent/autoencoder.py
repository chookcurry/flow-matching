from torch import nn
from torch import Tensor
from typing import Tuple
from abc import ABC, abstractmethod


class AE(nn.Module, ABC):
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def decode(self, z: Tensor) -> Tensor:
        # recon
        pass

    @abstractmethod
    def encode(self, x: Tensor) -> Tensor:
        # z
        pass

    @abstractmethod
    def forward(self, x: Tensor, y: Tensor) -> Tuple[Tensor, Tensor]:
        # recon, z
        pass


class CAE(nn.Module, ABC):
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def decode(self, z: Tensor) -> Tensor:
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


class AEC(nn.Module, ABC):
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def decode(self, z: Tensor, y: Tensor) -> Tensor:
        # recon
        pass

    @abstractmethod
    def encode(self, x: Tensor) -> Tensor:
        # z
        pass

    @abstractmethod
    def forward(self, x: Tensor, y: Tensor) -> Tuple[Tensor, Tensor]:
        # recon, z
        pass


class CAEC(nn.Module, ABC):
    def __init__(self) -> None:
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


class VAE(nn.Module, ABC):
    def __init__(self) -> None:
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


class CVAEC(nn.Module, ABC):
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def decode(self, z: Tensor, y: Tensor) -> Tensor:
        # recon
        pass

    @abstractmethod
    def encode(self, x: Tensor, y: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        # z, mu, logvar
        pass

    @abstractmethod
    def forward(self, x: Tensor, y: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        # recon, mu, logvar
        pass
