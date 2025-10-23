import torch.nn as nn
from torch import Tensor

from diffusion.architectures.classifiers.encoder import Encoder


class MNISTClassifier(Encoder):
    def __init__(self, in_c: int = 1, num_classes: int = 10):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(in_c, 32, 3, 1, 1),  # 32x32 -> 32x32
            nn.ReLU(),
            nn.MaxPool2d(2),  # 32x32 -> 16x16
            nn.Conv2d(32, 64, 3, 1, 1),  # 16x16 -> 16x16
            nn.ReLU(),
            nn.MaxPool2d(2),  # 16x16 -> 8x8
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 128),  # fix: 64*8*8
        )

        self.head = nn.Linear(128, num_classes)

    def encode(self, x: Tensor) -> Tensor:
        return self.encoder(x)

    def forward(self, x: Tensor) -> Tensor:
        return self.head(self.encoder(x))
