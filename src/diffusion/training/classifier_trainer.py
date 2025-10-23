import torch
from diffusion.sampleables.sampleable import Sampleable
from diffusion.training.trainer import Trainer
from torch import nn
from torch import Tensor


class ClassifierTrainer(Trainer):
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
    def get_val_loss(self, batch_size: int, device: torch.device) -> Tensor:
        self.model.eval()

        # Sample one batch
        x, y = self.val_data.sample(batch_size)
        assert y is not None
        x, y = x.to(device), y.to(device)

        # Forward pass
        logits: Tensor = self.model(x)
        # loss: Tensor = self.criterion(logits, y)

        # Compute accuracy
        _, preds = logits.max(1)
        accuracy = preds.eq(y).float().mean()

        return accuracy
