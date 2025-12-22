import torch.nn as nn
from torch import Tensor

from diffusion.classifiers.encoder import Encoder


class WISDMClassifier(Encoder):
    def __init__(self, in_c: int, num_classes: int, size: int):
        super().__init__()

        # Encoder: extracts features
        self.encoder = nn.Sequential(
            nn.Conv2d(in_c, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
        )

        self.feature_extractor = nn.Linear(size, 128)

        # Head: classification layer
        self.head = nn.Linear(128, num_classes)

    # Feature extractor
    def encode(self, x: Tensor) -> Tensor:
        x = self.feature_extractor(self.encoder(x))
        return x

    # Full forward pass
    def forward(self, x: Tensor) -> Tensor:
        features = self.encode(x)
        x = self.head(features)
        return x
