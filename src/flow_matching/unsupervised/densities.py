from abc import ABC, abstractmethod
from typing import Optional
import numpy as np
from sklearn.datasets import make_circles, make_moons  # type: ignore
import torch
from torch.distributions import (
    MultivariateNormal,
    Distribution,
    MixtureSameFamily,
    Categorical,
)


class Sampleable(ABC):
    """
    Distribution which can be sampled from
    """

    @property
    @abstractmethod
    def dim(self) -> int:
        """
        Returns:
            - Dimensionality of the distribution
        """
        pass

    @abstractmethod
    def sample(self, num_samples: int) -> torch.Tensor:
        """
        Args:
            - num_samples: the desired number of samples
        Returns:
            - samples: shape (batch_size, dim)
        """
        pass


class Density(ABC):
    """
    Distribution with tractable density
    """

    @abstractmethod
    def log_density(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns the log density at x.
        Args:
            - x: shape (batch_size, dim)
        Returns:
            - log_density: shape (batch_size, 1)
        """
        pass


class Gaussian(torch.nn.Module, Sampleable, Density):
    """
    Multivariate Gaussian distribution
    """

    def __init__(self, mean: torch.Tensor, cov: torch.Tensor):
        """
        mean: shape (dim,)
        cov: shape (dim,dim)
        """
        super().__init__()

        self.mean: torch.Tensor
        self.cov: torch.Tensor

        self.register_buffer("mean", mean)
        self.register_buffer("cov", cov)

    @property
    def dim(self) -> int:
        return self.mean.shape[0]

    @property
    def distribution(self) -> Distribution:
        return MultivariateNormal(self.mean, self.cov, validate_args=False)

    def sample(self, num_samples: int) -> torch.Tensor:
        return self.distribution.sample((num_samples,))

    def log_density(self, x: torch.Tensor) -> torch.Tensor:
        return self.distribution.log_prob(x).view(-1, 1)

    @classmethod
    def isotropic(cls, dim: int, std: float) -> "Gaussian":
        mean = torch.zeros(dim)
        cov = torch.eye(dim) * std**2
        return cls(mean, cov)


class GaussianMixture(torch.nn.Module, Sampleable, Density):
    """
    Two-dimensional Gaussian mixture model, and is a Density and a Sampleable. Wrapper around torch.distributions.MixtureSameFamily.
    """

    def __init__(
        self,
        means: torch.Tensor,  # nmodes x data_dim
        covs: torch.Tensor,  # nmodes x data_dim x data_dim
        weights: torch.Tensor,  # nmodes
    ):
        """
        means: shape (nmodes, 2)
        covs: shape (nmodes, 2, 2)
        weights: shape (nmodes, 1)
        """
        super().__init__()
        self.nmodes = means.shape[0]

        self.means: torch.Tensor
        self.covs: torch.Tensor
        self.weights: torch.Tensor

        self.register_buffer("means", means)
        self.register_buffer("covs", covs)
        self.register_buffer("weights", weights)

    @property
    def dim(self) -> int:
        return self.means.shape[1]

    @property
    def distribution(self) -> Distribution:
        return MixtureSameFamily(
            mixture_distribution=Categorical(probs=self.weights, validate_args=False),
            component_distribution=MultivariateNormal(
                loc=self.means,
                covariance_matrix=self.covs,
                validate_args=False,
            ),
            validate_args=False,
        )

    def log_density(self, x: torch.Tensor) -> torch.Tensor:
        return self.distribution.log_prob(x).view(-1, 1)

    def sample(self, num_samples: int) -> torch.Tensor:
        return self.distribution.sample(torch.Size((num_samples,)))

    @classmethod
    def random_2D(
        cls,
        nmodes: int,
        std: float,
        scale: float = 10.0,
        x_offset: float = 0.0,
        seed: float = 0.0,
    ) -> "GaussianMixture":
        torch.manual_seed(seed)
        means = (torch.rand(nmodes, 2) - 0.5) * scale + x_offset * torch.Tensor(
            [1.0, 0.0]
        )
        covs = torch.diag_embed(torch.ones(nmodes, 2)) * std**2
        weights = torch.ones(nmodes)
        return cls(means, covs, weights)

    @classmethod
    def symmetric_2D(
        cls, nmodes: int, std: float, scale: float = 10.0, x_offset: float = 0.0
    ) -> "GaussianMixture":
        angles = torch.linspace(0, 2 * np.pi, nmodes + 1)[:nmodes]
        means = (
            torch.stack([torch.cos(angles), torch.sin(angles)], dim=1) * scale
            + torch.Tensor([1.0, 0.0]) * x_offset
        )
        covs = torch.diag_embed(torch.ones(nmodes, 2) * std**2)
        weights = torch.ones(nmodes) / nmodes
        return cls(means, covs, weights)


class MoonsSampleable(Sampleable):
    """
    Implementation of the Moons distribution using sklearn's make_moons
    """

    def __init__(
        self,
        device: torch.device,
        noise: float = 0.05,
        scale: float = 5.0,
        offset: Optional[torch.Tensor] = None,
    ):
        """
        Args:
            noise: Standard deviation of Gaussian noise added to the data
            scale: How much to scale the data
            offset: How much to shift the samples from the original distribution (2,)
        """
        self.noise = noise
        self.scale = scale
        self.device = device
        if offset is None:
            offset = torch.zeros(2)
        self.offset = offset.to(device)

    @property
    def dim(self) -> int:
        return 2

    def sample(self, num_samples: int) -> torch.Tensor:
        """
        Args:
            num_samples: Number of samples to generate
        Returns:
            torch.Tensor: Generated samples with shape (num_samples, 3)
        """
        samples, _ = make_moons(
            n_samples=num_samples,
            noise=self.noise,
            random_state=None,  # Allow for random generation each time
        )
        return (
            self.scale * torch.from_numpy(samples.astype(np.float32)).to(self.device)
            + self.offset
        )


class CirclesSampleable(Sampleable):
    """
    Implementation of concentric circle distribution using sklearn's make_circles
    """

    def __init__(
        self,
        device: torch.device,
        noise: float = 0.05,
        scale: float = 5.0,
        offset: Optional[torch.Tensor] = None,
    ) -> None:
        """
        Args:
            noise: standard deviation of Gaussian noise added to the data
        """
        self.noise = noise
        self.scale = scale
        self.device = device
        if offset is None:
            offset = torch.zeros(2)
        self.offset = offset.to(device)

    @property
    def dim(self) -> int:
        return 2

    def sample(self, num_samples: int) -> torch.Tensor:
        """
        Args:
            num_samples: number of samples to generate
        Returns:
            torch.Tensor: shape (num_samples, 3)
        """
        samples, _ = make_circles(
            n_samples=num_samples, noise=self.noise, factor=0.5, random_state=None
        )
        return (
            self.scale * torch.from_numpy(samples.astype(np.float32)).to(self.device)
            + self.offset
        )


class CheckerboardSampleable(Sampleable):
    """
    Checkboard-esque distribution
    """

    def __init__(
        self, device: torch.device, grid_size: int = 3, scale: float = 5.0
    ) -> None:
        """
        Args:
            noise: standard deviation of Gaussian noise added to the data
        """
        self.grid_size = grid_size
        self.scale = scale
        self.device = device

    @property
    def dim(self) -> int:
        return 2

    def sample(self, num_samples: int) -> torch.Tensor:
        """
        Args:
            num_samples: number of samples to generate
        Returns:
            torch.Tensor: shape (num_samples, 3)
        """
        grid_length = 2 * self.scale / self.grid_size
        samples = torch.zeros(0, 2).to(self.device)
        while samples.shape[0] < num_samples:
            # Sample num_samples
            new_samples = (
                (torch.rand(num_samples, 2).to(self.device) - 0.5) * 2 * self.scale
            )
            x_mask = (
                torch.floor((new_samples[:, 0] + self.scale) / grid_length) % 2 == 0
            )  # (bs,)
            y_mask = (
                torch.floor((new_samples[:, 1] + self.scale) / grid_length) % 2 == 0
            )  # (bs,)
            accept_mask = torch.logical_xor(~x_mask, y_mask)
            samples = torch.cat([samples, new_samples[accept_mask]], dim=0)
        return samples[:num_samples]
