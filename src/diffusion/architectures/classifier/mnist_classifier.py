from abc import ABC, abstractmethod
import torch
import torch.nn as nn
from typing import Dict
from torch import Tensor

from diffusion.data.sampleables import Sampleable
from diffusion.training.trainer import Trainer


class Encoder(ABC, nn.Module):
    @abstractmethod
    def encode(self, x: Tensor) -> Tensor:
        pass


# ----------------------
# CNN Model
# ----------------------
class SimpleCNN(Encoder):
    def __init__(self, num_classes: int = 10):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, 1, 1),  # 32x32 -> 32x32
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


# ----------------------
# Trainer Class
# ----------------------
class MNISTClassifierTrainer(Trainer):
    def __init__(
        self,
        classifier: nn.Module,
        train_data: Sampleable,
        val_data: Sampleable,
        num_classes: int = 10,
        lr: float = 1e-3,
        batch_size: int = 64,
        val_num_samples: int = 1000,
    ):
        super().__init__(classifier)

        self.train_data = train_data
        self.val_data = val_data
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.val_num_samples = val_num_samples

        self.optimizer = self.get_optimizer(lr)
        self.criterion = nn.CrossEntropyLoss()

    def get_train_loss(self, batch_size: int, device: torch.device) -> Tensor:
        x, y = self.train_data.sample(batch_size)
        assert y is not None
        x, y = x.to(device), y.to(device)

        logits = self.model(x)
        loss = self.criterion(logits, y)
        return loss

    @torch.no_grad()
    def get_val_metrics(self, device: torch.device) -> Dict[str, float]:
        self.model.eval()

        # Sample one batch
        x, y = self.val_data.sample(self.val_num_samples)
        assert y is not None
        x, y = x.to(device), y.to(device)

        # Forward pass
        logits: Tensor = self.model(x)
        loss: Tensor = self.criterion(logits, y)

        # Compute accuracy
        _, preds = logits.max(1)
        accuracy = preds.eq(y).float().mean().item()

        # Return metrics
        metrics = {
            "loss": loss.item(),
            "accuracy": accuracy,
        }

        return metrics
