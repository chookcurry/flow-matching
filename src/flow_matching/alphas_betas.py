from abc import ABC, abstractmethod

import torch
from torch.func import vmap, jacrev


class Alpha(ABC):
    def __init__(self) -> None:
        # Check alpha_t(0) = 0
        assert torch.allclose(self(torch.zeros(1, 1)), torch.zeros(1, 1))
        # Check alpha_1 = 1
        assert torch.allclose(self(torch.ones(1, 1)), torch.ones(1, 1))

    @abstractmethod
    def __call__(self, t: torch.Tensor) -> torch.Tensor:
        """
        Evaluates alpha_t. Should satisfy: self(0.0) = 0.0, self(1.0) = 1.0.
        Args:
            - t: time (num_samples, 1)
        Returns:
            - alpha_t (num_samples, 1)
        """
        pass

    def dt(self, t: torch.Tensor) -> torch.Tensor:
        """
        Evaluates d/dt alpha_t.
        Args:
            - t: time (num_samples, 1)
        Returns:
            - d/dt alpha_t (num_samples, 1)
        """
        t = t.unsqueeze(1)  # (num_samples, 1, 1)
        dt: torch.Tensor = vmap(jacrev(self))(t)  # (num_samples, 1, 1, 1, 1)
        return dt.view(-1, 1)


class Beta(ABC):
    def __init__(self) -> None:
        # Check beta_0 = 1
        assert torch.allclose(self(torch.zeros(1, 1)), torch.ones(1, 1))
        # Check beta_1 = 0
        assert torch.allclose(self(torch.ones(1, 1)), torch.zeros(1, 1))

    @abstractmethod
    def __call__(self, t: torch.Tensor) -> torch.Tensor:
        """
        Evaluates alpha_t. Should satisfy: self(0.0) = 1.0, self(1.0) = 0.0.
        Args:
            - t: time (num_samples, 1)
        Returns:
            - beta_t (num_samples, 1)
        """
        pass

    def dt(self, t: torch.Tensor) -> torch.Tensor:
        """
        Evaluates d/dt beta_t.
        Args:
            - t: time (num_samples, 1)
        Returns:
            - d/dt beta_t (num_samples, 1)
        """
        t = t.unsqueeze(1)  # (num_samples, 1, 1)
        dt: torch.Tensor = vmap(jacrev(self))(t)  # (num_samples, 1, 1, 1, 1)
        return dt.view(-1, 1)


class LinearAlpha(Alpha):
    """
    Implements alpha_t = t
    """

    def __call__(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            - t: time (num_samples, 1)
        Returns:
            - alpha_t (num_samples, 1)
        """
        return t

    def dt(self, t: torch.Tensor) -> torch.Tensor:
        """
        Evaluates d/dt alpha_t.
        Args:
            - t: time (num_samples, 1)
        Returns:
            - d/dt alpha_t (num_samples, 1)
        """
        return torch.ones_like(t)


class SquareRootBeta(Beta):
    """
    Implements beta_t = rt(1-t)
    """

    def __call__(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            - t: time (num_samples, 1)
        Returns:
            - beta_t (num_samples, 1)
        """
        return torch.sqrt(1 - t)

    def dt(self, t: torch.Tensor) -> torch.Tensor:
        """
        Evaluates d/dt alpha_t.
        Args:
            - t: time (num_samples, 1)
        Returns:
            - d/dt alpha_t (num_samples, 1)
        """
        dt: torch.Tensor = -0.5 / (torch.sqrt(1 - t) + 1e-4)
        return dt
