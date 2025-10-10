from abc import ABC, abstractmethod

from torch import Tensor
import torch


class Generator(ABC):
    @abstractmethod
    @torch.no_grad()
    def generate(self, y: Tensor, x0: Tensor | None = None) -> Tensor:
        pass
