from abc import ABC, abstractmethod
import torch.nn as nn
from torch import Tensor


class Encoder(ABC, nn.Module):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def encode(self, x: Tensor) -> Tensor:
        pass
