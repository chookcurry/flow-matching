import torch.nn as nn
from torch import Tensor

from diffusion.classifiers.encoder import Encoder

# class WISDMClassifier(Encoder):
#     def __init__(self, in_c: int, num_classes: int, size: int):
#         super().__init__()
#         # Encoder: extracts features
#         self.encoder = nn.Sequential(
#             nn.Conv2d(in_c, 32, kernel_size=3, stride=1, padding=1),
#             nn.ReLU(),
#             nn.MaxPool2d(2),
#             nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
#             nn.ReLU(),
#             nn.MaxPool2d(2),
#             nn.Flatten(),
#         )
#         self.feature_extractor = nn.Linear(size, 128)
#         # Head: classification layer
#         self.head = nn.Linear(128, num_classes)
#     # Feature extractor
#     def encode(self, x: Tensor) -> Tensor:
#         x = self.feature_extractor(self.encoder(x))
#         return x
#     # Full forward pass
#     def forward(self, x: Tensor) -> Tensor:
#         features = self.encode(x)
#         x = self.head(features)
#         return x


class WISDMClassifier(Encoder):  # Inheriting from nn.Module or your custom Encoder
    def __init__(self, in_c: int, num_classes: int, size: int):
        super().__init__()

        # Encoder: Deeper feature extraction with Batch Norm
        self.encoder = nn.Sequential(
            # Block 1: 32x32 -> 16x16
            nn.Conv2d(in_c, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.2),
            # Block 2: 16x16 -> 8x8
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.3),
            # Block 3: 8x8 -> 4x4
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),  # Output size: 128 * 4 * 4 = 2048
        )

        # Updated size logic: For 32x32 input, flatten results in 128 * 4 * 4
        # Note: 'size' parameter should match the flattened output (default 2048)
        self.feature_extractor = nn.Sequential(
            nn.Linear(size, 256), nn.ReLU(), nn.Dropout(0.5)
        )

        # Head: classification layer
        self.head = nn.Linear(256, num_classes)

    def encode(self, x: Tensor) -> Tensor:
        x = self.encoder(x)
        x = self.feature_extractor(x)
        return x

    def forward(self, x: Tensor) -> Tensor:
        features = self.encode(x)
        x = self.head(features)
        return x
