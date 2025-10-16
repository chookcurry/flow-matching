import torch.nn as nn
from torch import Tensor

from diffusion.architectures.classifiers.mnist_classifier import Encoder


class WISDMClassifier(Encoder):
    def __init__(self, in_c: int = 6, num_classes: int = 10):
        super().__init__()

        # Encoder: extracts features
        self.encoder = nn.Sequential(
            nn.Conv2d(in_c, 32, kernel_size=3, stride=1, padding=1),  # 32x26 -> 32x26
            nn.ReLU(),
            nn.MaxPool2d(2),  # 32x26 -> 16x13
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),  # 16x13 -> 16x13
            nn.ReLU(),
            nn.MaxPool2d(2),  # 16x13 -> 8x6
            nn.Flatten(),
            nn.Linear(64 * 8 * 6, 128),  # flattened feature size
        )

        # Head: classification layer
        self.head = nn.Linear(128, num_classes)

    # Feature extractor
    def encode(self, x: Tensor) -> Tensor:
        return self.encoder(x)

    # Full forward pass
    def forward(self, x: Tensor) -> Tensor:
        features = self.encode(x)
        return self.head(features)
