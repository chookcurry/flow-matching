from abc import ABC, abstractmethod
from typing import Tuple

from torch import Tensor
import torch


class Generator(ABC):
    @abstractmethod
    @torch.no_grad()
    def sample_prior(
        self, num_samples: int, shape: Tuple[int, ...], device: torch.device
    ) -> Tensor:
        pass

    @abstractmethod
    @torch.no_grad()
    def generate(
        self, y: Tensor, x0: Tensor | None = None, guidance_scale: float | None = None
    ) -> Tensor:
        pass
