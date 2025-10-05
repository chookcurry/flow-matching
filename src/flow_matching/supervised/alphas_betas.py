from abc import ABC, abstractmethod

import torch
from torch import Tensor
from torch.func import vmap, jacrev


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
