from abc import ABC, abstractmethod
from typing import Optional, Tuple

import torch
from torch import Tensor, nn


class Sampleable(ABC):
    @abstractmethod
    def sample(
        self, num_samples: int, y: Tensor | None = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # y: (B)
        pass


class IsotropicGaussian(nn.Module, Sampleable):
    def __init__(self, shape: Tuple[int, ...], mean: float = 0.0, std: float = 1.0):
        super().__init__()

        self.shape = shape
        self.mean = mean
        self.std = std

        self.dummy = nn.Buffer(torch.zeros(1))

    def sample(
        self, num_samples: int, y: Tensor | None = None
    ) -> Tuple[torch.Tensor, None]:
        eps = torch.randn(num_samples, *self.shape, device=self.dummy.device)
        samples = self.mean + self.std * eps
        return samples, None


# class ConditionalGaussian(nn.Module, Sampleable):
#     def __init__(
#         self, num_classes: int, shape: Tuple[int, ...], mean_scale: float = 3.0
#     ):
#         super().__init__()
#         self.num_classes = num_classes
#         self.shape = shape

#         # distinct mean vectors for each class
#         self.means = nn.Parameter(torch.randn(num_classes, *shape) * mean_scale)
#         self.std = 1.0
#         self.dummy = nn.Buffer(torch.zeros(1))

#     def sample(
#         self, num_samples: int, y: Tensor | None = None
#     ) -> Tuple[Tensor, Tensor | None]:
#         if y is None:
#             y = torch.randint(
#                 low=0,
#                 high=self.num_classes,
#                 size=(num_samples,),
#                 device=self.dummy.device,
#             )

#         eps = torch.randn(num_samples, *self.shape, device=self.dummy.device)
#         mean = self.means.to(self.dummy.device)[y]  # (B, *shape)
#         samples = mean + self.std * eps

#         return samples, y


# class ConditionalGaussianHypersphere(nn.Module, Sampleable):
#     def __init__(self, num_classes: int, shape: Tuple[int, ...], radius: float = 3.0):
#         super().__init__()

#         self.shape = shape
#         self.num_classes = num_classes
#         self.means = self._make_hyperspherical_means(num_classes, shape, radius)
#         self.means = self.means.view(num_classes, *shape)
#         self.std = 1.0
#         self.dummy = nn.Buffer(torch.zeros(1))

#     def _make_hyperspherical_means(
#         self, num_classes: int, shape: Tuple[int, ...], radius: float
#     ) -> Tensor:
#         dim = int(torch.tensor(shape).prod().item())

#         # Use random orthogonal directions, normalize to radius
#         x = torch.randn(num_classes, dim)
#         x = radius * x / x.norm(dim=-1, keepdim=True)
#         return x

#     def sample(
#         self, num_samples: int, y: Tensor | None = None
#     ) -> Tuple[Tensor, Tensor | None]:
#         if y is None:
#             y = torch.randint(
#                 low=0,
#                 high=self.num_classes,
#                 size=(num_samples,),
#                 device=self.dummy.device,
#             )

#         eps = torch.randn(num_samples, *self.shape, device=self.dummy.device)
#         mean = self.means.to(self.dummy.device)[y]  # (B, *shape)
#         samples = mean + self.std * eps

#         return samples, y
