from abc import ABC, abstractmethod

import torch
from torch import Tensor, nn


class ODE(ABC):
    @abstractmethod
    def drift_coeff(self, x_t: Tensor, t: Tensor, y: Tensor) -> Tensor:
        # x_t: (B, C, H, W)
        # t: (B, 1, 1, 1)
        # y: (B, 1, 1, 1)
        # drift: (B, C, H, W)
        pass


class SDE(ABC):
    @abstractmethod
    def drift_coeff(self, x_t: Tensor, t: Tensor, y: Tensor) -> Tensor:
        # x_t: (B, C, H, W)
        # t: (B, 1, 1, 1)
        # y: (B, 1, 1, 1)
        # drift: (B, C, H, W)
        pass

    @abstractmethod
    def diffusion_coeff(self, x_t: Tensor, t: Tensor, y: Tensor) -> Tensor:
        # x_t: (B, C, H, W)
        # t: (B, 1, 1, 1)
        # y: (B, 1, 1, 1)
        # diffusion_coeff: (B, C, H, W)
        pass


class Backbone(nn.Module, ABC):
    @abstractmethod
    def forward(self, x: Tensor, t: Tensor, y: Tensor) -> Tensor:
        # x: (B, C, H, W)
        # t: (B, 1, 1, 1)
        # y: (B, 1, 1, 1)
        # vf: (B, C, H, W)
        pass


class GuidedNeuralODE(ODE):
    def __init__(self, vf: Backbone, null_class: int, scale: float = 1.0):
        self.vf = vf
        self.null_class = null_class
        self.scale = scale

    def drift_coeff(self, x_t: Tensor, t: Tensor, y: Tensor) -> Tensor:
        # x_t: (B, C, H, W)
        # t: (B, 1, 1, 1)
        # y: (B, 1, 1, 1)

        unguided_y = torch.ones_like(y) * self.null_class
        unguided_vf: Tensor = self.vf(x_t, t, unguided_y)
        # (B, C, H, W)

        guided_vf: Tensor = self.vf(x_t, t, y)
        # (B, C, H, W)

        drift = (1 - self.scale) * unguided_vf + self.scale * guided_vf
        # (B, C, H, W)

        return drift
