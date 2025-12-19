from abc import ABC, abstractmethod
from typing import Tuple

import torch
from torch import Tensor


class Generator(ABC):
    @abstractmethod
    @torch.no_grad()
    def sample_prior(
        self,
        num_samples: int,
        shape: Tuple[int, ...],
        device: torch.device,
        y: Tensor | None = None,
    ) -> Tensor:
        pass

    @abstractmethod
    @torch.no_grad()
    def generate(
        self, y: Tensor, x0: Tensor | None = None, guidance_scale: float | None = None
    ) -> Tensor:
        pass
