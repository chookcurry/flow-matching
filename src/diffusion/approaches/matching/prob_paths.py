import torch
from abc import ABC, abstractmethod
from typing import Tuple

from torch.func import vmap, jacrev
from torch import Tensor, nn

from diffusion.sampleables.sampleable import IsotropicGaussian
from diffusion.sampleables.sampleable import Sampleable
from diffusion.architectures.backbones.backbone import Backbone


class CondProbPath(nn.Module, ABC):
    def __init__(self, p_simple: Sampleable, p_data: Sampleable) -> None:
        super().__init__()

        self.p_simple = p_simple
        self.p_data = p_data

    @abstractmethod
    def sample_cond_path(self, z: Tensor, t: Tensor, y: Tensor | None = None) -> Tensor:
        # z: (B, C, H, W)
        # t: (B, 1, 1, 1)
        # x: (B, C, H, W)
        pass

    @abstractmethod
    def cond_vf(self, x: Tensor, z: Tensor, t: Tensor) -> Tensor:
        # x: (B, C, H, W)
        # z: (B, C, H, W)
        # t: (B, 1, 1, 1)
        # vf: (B, C, H, W)
        pass

    @abstractmethod
    def cond_score(self, x: Tensor, z: Tensor, t: Tensor) -> Tensor:
        # x: (B, C, H, W)
        # z: (B, C, H, W)
        # t: (B, 1, 1, 1)
        # score: (B, C, H, W)
        pass


class Alpha(ABC):
    def __init__(self) -> None:
        # Check alpha_t(0) = 0, alpha_t(1) = 1
        assert torch.allclose(self(torch.zeros(1, 1, 1, 1)), torch.zeros(1, 1, 1, 1))
        assert torch.allclose(self(torch.ones(1, 1, 1, 1)), torch.ones(1, 1, 1, 1))

    @abstractmethod
    def __call__(self, t: Tensor) -> Tensor:
        # (B, 1, 1, 1)
        pass

    def dt(self, t: Tensor) -> Tensor:
        # (B, 1, 1, 1)

        t = t.unsqueeze(1)
        dt: Tensor = vmap(jacrev(self))(t)
        dt = dt.view(-1, 1, 1, 1)
        # (B, 1, 1, 1)

        return dt


class Beta(ABC):
    def __init__(self) -> None:
        # Check beta_0 = 1, beta_1 = 0
        assert torch.allclose(self(torch.zeros(1, 1, 1, 1)), torch.ones(1, 1, 1, 1))
        assert torch.allclose(self(torch.ones(1, 1, 1, 1)), torch.zeros(1, 1, 1, 1))

    @abstractmethod
    def __call__(self, t: Tensor) -> Tensor:
        # (B, 1, 1, 1)
        pass

    def dt(self, t: Tensor) -> Tensor:
        # (B, 1, 1, 1)

        t = t.unsqueeze(1)
        dt: Tensor = vmap(jacrev(self))(t)
        dt = dt.view(-1, 1, 1, 1)
        # (B, 1, 1, 1)

        return dt


class LinearAlpha(Alpha):
    def __call__(self, t: Tensor) -> Tensor:
        # (B, 1, 1, 1)

        alpha_t = t
        # (B, 1, 1, 1)

        return alpha_t

    def dt(self, t: Tensor) -> Tensor:
        # (B, 1, 1, 1)

        dt = torch.ones_like(t)
        # (B, 1, 1, 1)

        return dt


class LinearBeta(Beta):
    def __call__(self, t: Tensor) -> Tensor:
        # (B, 1, 1, 1)

        beta_t: Tensor = 1 - t
        # (B, 1, 1, 1)

        return beta_t

    def dt(self, t: Tensor) -> Tensor:
        # (B, 1, 1, 1)

        dt = -torch.ones_like(t)
        # (B, 1, 1, 1)

        return dt


class ScoreFromVectorFieldForGaussianProbPath(Backbone):
    def __init__(
        self, vf: Backbone, alpha: Alpha = LinearAlpha(), beta: Beta = LinearBeta()
    ) -> None:
        super().__init__()
        self.vf = vf
        self.alpha = alpha
        self.beta = beta

    def forward(self, x: Tensor, t: Tensor, y: Tensor) -> Tensor:
        pred: Tensor = self.vf(x, t, y)
        numerator = self.alpha(t) * pred - self.alpha.dt(t) * x
        denominator = self.beta(t) ** 2 * self.alpha.dt(t) - self.alpha(
            t
        ) * self.beta.dt(t) * self.beta(t)
        return numerator / denominator


class GaussianCondProbPath(CondProbPath):
    def __init__(
        self,
        p_data: Sampleable,
        p_simple_shape: Tuple[int, ...],
        alpha: Alpha = LinearAlpha(),
        beta: Beta = LinearBeta(),
    ):
        p_simple = IsotropicGaussian(shape=p_simple_shape)
        super().__init__(p_simple, p_data)

        self.alpha = alpha
        self.beta = beta

    def sample_cond_path(self, z: Tensor, t: Tensor, y: Tensor | None = None) -> Tensor:
        # z: (B, C, H, W)
        # t: (B, 1, 1, 1)

        start, _ = self.p_simple.sample(z.shape[0], None)
        x = self.alpha(t) * z + self.beta(t) * start
        # (B, C, H, W)

        return x

    def cond_vf(self, x: Tensor, z: Tensor, t: Tensor) -> Tensor:
        # x: (B, C, H, W)
        # z: (B, C, H, W)
        # t: (B, 1, 1, 1)

        alpha_t = self.alpha(t)
        beta_t = self.beta(t)
        dt_alpha_t = self.alpha.dt(t)
        dt_beta_t = self.beta.dt(t)
        # (B, 1, 1, 1)

        vf = (dt_alpha_t - dt_beta_t / beta_t * alpha_t) * z + dt_beta_t / beta_t * x
        # (B, C, H, W)

        return vf

    def cond_score(self, x: Tensor, z: Tensor, t: Tensor) -> Tensor:
        # x: (B, C, H, W)
        # z: (B, C, H, W)
        # t: (B, 1, 1, 1)

        alpha_t = self.alpha(t)
        beta_t = self.beta(t)
        # (B, 1, 1, 1)

        score = (alpha_t * z - x) / beta_t**2
        # (B, C, H, W)

        return score


# class TestGaussianCondProbPath(CondProbPath):
#     def __init__(
#         self,
#         num_classes: int,
#         p_data: Sampleable,
#         p_simple_shape: Tuple[int, ...],
#         alpha: Alpha,
#         beta: Beta,
#     ):
#         p_simple = ConditionalGaussian(num_classes, p_simple_shape)
#         # ConditionalGaussianHypersphere(num_classes, p_simple_shape)
#         super().__init__(p_simple, p_data)

#         self.alpha = alpha
#         self.beta = beta

#     def sample_cond_path(self, z: Tensor, t: Tensor, y: Tensor | None = None) -> Tensor:
#         # z: (B, C, H, W)
#         # t: (B, 1, 1, 1)
#         # y: (B)

#         assert y is not None

#         start, _ = self.p_simple.sample(z.shape[0], y)
#         x = self.alpha(t) * z + self.beta(t) * start
#         # (B, C, H, W)

#         return x

#     def cond_vf(self, x: Tensor, z: Tensor, t: Tensor) -> Tensor:
#         # x: (B, C, H, W)
#         # z: (B, C, H, W)
#         # t: (B, 1, 1, 1)

#         alpha_t = self.alpha(t)
#         beta_t = self.beta(t)
#         dt_alpha_t = self.alpha.dt(t)
#         dt_beta_t = self.beta.dt(t)
#         # (B, 1, 1, 1)

#         vf = (dt_alpha_t - dt_beta_t / beta_t * alpha_t) * z + dt_beta_t / beta_t * x
#         # (B, C, H, W)

#         return vf

#     def cond_score(self, x: Tensor, z: Tensor, t: Tensor) -> Tensor:
#         # x: (B, C, H, W)
#         # z: (B, C, H, W)
#         # t: (B, 1, 1, 1)

#         alpha_t = self.alpha(t)
#         beta_t = self.beta(t)
#         # (B, 1, 1, 1)

#         score = (alpha_t * z - x) / beta_t**2
#         # (B, C, H, W)

#         return score
