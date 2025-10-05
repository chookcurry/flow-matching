from abc import ABC, abstractmethod
from typing import Tuple
from torch import Tensor, nn, randn_like

from flow_matching.supervised.sampleables import IsotropicGaussian
from flow_matching.supervised.alphas_betas import Alpha, Beta
from flow_matching.supervised.sampleables import Sampleable


class CondProbPath(nn.Module, ABC):
    def __init__(self, p_simple: Sampleable, p_data: Sampleable) -> None:
        super().__init__()

        self.p_simple = p_simple
        self.p_data = p_data

    def sample_marginal_path(self, t: Tensor) -> Tensor:
        # (B, 1, 1, 1)

        B = t.shape[0]

        # Sample conditioning variable z ~ p(z)
        z, _ = self.sample_cond_var(B)
        # (B, C, H, W)

        # Sample conditional probability path x ~ p_t(x|z)
        x = self.sample_cond_path(z, t)
        # (B, C, H, W)

        return x

    @abstractmethod
    def sample_cond_var(self, B: int) -> Tuple[Tensor, Tensor | None]:
        # z: (B, C, H, W)
        # y: (B, y_dim)
        pass

    @abstractmethod
    def sample_cond_path(self, z: Tensor, t: Tensor) -> Tensor:
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


class GaussianCondProbPath(CondProbPath):
    def __init__(
        self,
        p_data: Sampleable,
        p_simple_shape: Tuple[int, ...],
        alpha: Alpha,
        beta: Beta,
    ):
        p_simple = IsotropicGaussian(shape=p_simple_shape)
        super().__init__(p_simple, p_data)

        self.alpha = alpha
        self.beta = beta

    def sample_cond_var(self, B: int) -> Tuple[Tensor, Tensor | None]:
        z, y = self.p_simple.sample(B)
        # (B, C, H, W), (B, y_dim)

        return z, y

    def sample_cond_path(self, z: Tensor, t: Tensor) -> Tensor:
        # z: (B, C, H, W)
        # t: (B, 1, 1, 1)

        x = self.alpha(t) * z + self.beta(t) * randn_like(z)
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
