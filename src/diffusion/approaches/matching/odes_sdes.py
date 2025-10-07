from abc import ABC, abstractmethod

import torch
from torch import Tensor

from diffusion.architectures.backbone import Backbone


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

    def linear_sigma(self, t: Tensor) -> Tensor:
        return 1 - t

    def tent_sigma(self, t: Tensor) -> Tensor:
        return t * (1 - t)


class GuidedNeuralODE(ODE):
    def __init__(self, vf: Backbone, null_class: int, scale: float):
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


class GuidedNeuralSDE(SDE):
    def __init__(
        self,
        vf: Backbone,
        score_fn: Backbone,
        null_class: int,
        scale: float,
    ):
        self.vf = vf
        self.score_fn = score_fn
        self.null_class = null_class
        self.scale = scale
        self.sigma = self.tent_sigma

    def drift_coeff(self, x_t: Tensor, t: Tensor, y: Tensor) -> Tensor:
        # x_t: (B, C, H, W)
        # t: (B, 1, 1, 1)
        # y: (B, 1, 1, 1)

        unguided_y = torch.ones_like(y) * self.null_class
        unguided_vf: Tensor = self.vf(x_t, t, unguided_y)
        unguided_score: Tensor = self.score_fn(x_t, t, unguided_y)
        # (B, C, H, W)

        guided_vf: Tensor = self.vf(x_t, t, y)
        guided_score: Tensor = self.score_fn(x_t, t, y)
        # (B, C, H, W)

        vf = (1 - self.scale) * unguided_vf + self.scale * guided_vf
        score = (1 - self.scale) * unguided_score + self.scale * guided_score
        drift = vf + 0.5 * self.sigma(t) ** 2 * score
        # (B, C, H, W)

        return drift

    def diffusion_coeff(self, x_t: Tensor, t: Tensor, y: Tensor) -> Tensor:
        # x_t: (B, C, H, W)
        # t: (B, 1, 1, 1)
        # y: (B, 1, 1, 1)

        return self.sigma(t) * torch.randn_like(x_t)
