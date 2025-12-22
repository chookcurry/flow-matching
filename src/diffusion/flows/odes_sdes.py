from abc import ABC, abstractmethod
from typing import Callable

import torch
from torch import Tensor

from diffusion.backbones.backbone import Backbone


class ODE(ABC):
    @abstractmethod
    def drift_coeff(
        self, x_t: Tensor, t: Tensor, y: Tensor, guidance_scale: float | None = None
    ) -> Tensor:
        # x_t: (B, C, H, W)
        # t: (B, 1, 1, 1)
        # y: (B, 1, 1, 1)
        # drift: (B, C, H, W)
        pass


class SDE(ABC):
    @abstractmethod
    def drift_coeff(
        self, x_t: Tensor, t: Tensor, y: Tensor, guidance_scale: float | None = None
    ) -> Tensor:
        # x_t: (B, C, H, W)
        # t: (B, 1, 1, 1)
        # y: (B, 1, 1, 1)
        # drift: (B, C, H, W)
        pass

    @abstractmethod
    def diffusion_coeff(self, x_t: Tensor, t: Tensor) -> Tensor:
        # x_t: (B, C, H, W)
        # t: (B, 1, 1, 1)
        # y: (B, 1, 1, 1)
        # diffusion_coeff: (B, C, H, W)
        pass


class GuidedNeuralODE(ODE):
    def __init__(self, vf: Backbone, null_class: int):
        self.vf = vf
        self.null_class = null_class

    def drift_coeff(
        self, x_t: Tensor, t: Tensor, y: Tensor, guidance_scale: float | None = None
    ) -> Tensor:
        # x_t: (B, C, H, W)
        # t: (B, 1, 1, 1)
        # y: (B, 1, 1, 1)

        unguided_y = self.null_class * torch.ones_like(y)
        unguided_vf: Tensor = self.vf(x_t, t, unguided_y)
        # (B, C, H, W)

        if guidance_scale is not None:
            guided_vf: Tensor = self.vf(x_t, t, y)
            # (B, C, H, W)

            drift = (1 - guidance_scale) * unguided_vf + guidance_scale * guided_vf
            # (B, C, H, W)
        else:
            drift = unguided_vf
            # (B, C, H, W)

        return drift


def linear_sigma(t: Tensor) -> Tensor:
    return torch.ones_like(t) - t


def tent_sigma(t: Tensor) -> Tensor:
    return t * (torch.ones_like(t) - t)


class GuidedNeuralSDE(SDE):
    def __init__(
        self,
        vf: Backbone,
        score_fn: Backbone,
        null_class: int,
        sigma: Callable[[Tensor], Tensor] = tent_sigma,
    ):
        self.vf = vf
        self.score_fn = score_fn

        self.null_class = null_class
        self.sigma = sigma

    def drift_coeff(
        self, x_t: Tensor, t: Tensor, y: Tensor, guidance_scale: float | None = None
    ) -> Tensor:
        # x_t: (B, C, H, W)
        # t: (B, 1, 1, 1)
        # y: (B, 1, 1, 1)

        unguided_y = torch.ones_like(y) * self.null_class
        unguided_vf: Tensor = self.vf(x_t, t, unguided_y)
        unguided_score: Tensor = self.score_fn(x_t, t, unguided_y)
        # (B, C, H, W)

        if guidance_scale is not None:
            guided_vf: Tensor = self.vf(x_t, t, y)
            guided_score: Tensor = self.score_fn(x_t, t, y)
            # (B, C, H, W)

            vf = (1 - guidance_scale) * unguided_vf + guidance_scale * guided_vf
            score = (
                1 - guidance_scale
            ) * unguided_score + guidance_scale * guided_score
            # (B, C, H, W)
        else:
            vf = unguided_vf
            score = unguided_score
            # (B, C, H, W)

        drift = vf + 0.5 * self.sigma(t) ** 2 * score

        return drift

    def diffusion_coeff(self, x_t: Tensor, t: Tensor) -> Tensor:
        # x_t: (B, C, H, W)
        # t: (B, 1, 1, 1)

        return self.sigma(t) * torch.randn_like(x_t)
