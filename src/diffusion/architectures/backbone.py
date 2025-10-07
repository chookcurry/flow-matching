from abc import ABC, abstractmethod
from torch import Tensor, nn


class Backbone(nn.Module, ABC):
    @abstractmethod
    def forward(self, x: Tensor, t: Tensor, y: Tensor) -> Tensor:
        # x: (B, C, H, W)
        # t: (B, 1, 1, 1)
        # y: (B, 1, 1, 1)
        # vf: (B, C, H, W)
        pass
