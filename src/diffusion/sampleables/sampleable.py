from abc import ABC, abstractmethod
from typing import Optional, Tuple
import torch
from torch import nn


class Sampleable(ABC):
    @abstractmethod
    def sample(
        self, num_samples: int, class_label: int | None = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # samples: (B, ...)
        # labels: (B, label_dim)
        pass


class IsotropicGaussian(nn.Module, Sampleable):
    def __init__(self, shape: Tuple[int, ...], std: float = 1.0):
        super().__init__()

        self.shape = shape
        self.std = std
        self.dummy = nn.Buffer(torch.zeros(1))

    def sample(
        self, num_samples: int, class_label: int | None = None
    ) -> Tuple[torch.Tensor, None]:
        samples = self.std * torch.randn(num_samples, *self.shape).to(self.dummy.device)
        return samples, None
