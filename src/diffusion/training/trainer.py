from abc import ABC, abstractmethod
from typing import Dict
import torch
from tqdm import tqdm
from torch import Tensor
from torch import nn
from wandb import Run
from torch.optim import Optimizer, AdamW

from diffusion.utils.utils import AverageMeter, model_size_b, MiB
from diffusion.utils.logging import logger


class Trainer(ABC):
    def __init__(self, model: nn.Module):
        super().__init__()

        self.model = model
        self.optimizer = self.get_optimizer()

    def get_optimizer(self, lr: float = 1e-3) -> Optimizer:
        return AdamW(self.model.parameters(), lr=lr)

    def train(
        self,
        num_epochs: int,
        steps_per_epoch: int,
        batch_size: int,
        device: torch.device,
        lr: float = 1e-3,
        validate: bool = True,
        run: Run | None = None,
    ) -> None:
        size_b = model_size_b(self.model)
        logger.info(f"Training model with size: {size_b / MiB:.3f} MiB")

        self.model.to(device)
        optimizer = (
            self.optimizer
            if self.optimizer.param_groups[0]["lr"] == lr
            else self.get_optimizer(lr)
        )

        # Early stopping setup
        # best_val_loss = float("inf")
        # current_val_loss = float("inf")
        # best_model_state = self.model.state_dict()
        # patience_counter = 0

        losses = AverageMeter()

        for epoch in range(num_epochs):
            self.model.train()
            losses.reset()

            pbar = tqdm(range(steps_per_epoch))
            pbar.set_description(f"Epoch {epoch}/{num_epochs}")

            for _ in pbar:
                optimizer.zero_grad()
                loss = self.get_train_loss(batch_size=batch_size, device=device)

                run.log({"train/loss": loss.item()}) if run else None
                losses.update(loss.item())
                pbar.set_postfix(loss=f"{losses.avg:.6f}")

                loss.backward()
                optimizer.step()

            if not validate:
                continue

            self.model.eval()
            metrics = self.get_val_metrics(device)
            logger.info([f"{key}: {value:.6f}" for key, value in metrics.items()])
            run.log({f"val/{k}": v for k, v in metrics.items()}) if run else None

        #     current_val_loss = float(loss.item())
        #     if current_val_loss < best_val_loss:
        #         best_val_loss = current_val_loss
        #         best_model_state = self.model.state_dict()
        #         patience_counter = 0
        #     else:
        #         patience_counter += 1

        #     if patience_counter >= patience:
        #         logger.info(f"Early stopping triggered at epoch {epoch}/{num_epochs}")
        #         break

        # self.model.load_state_dict(best_model_state)
        # return best_model_state

    @abstractmethod
    def get_train_loss(self, batch_size: int, device: torch.device) -> Tensor:
        pass

    @abstractmethod
    @torch.no_grad()
    def get_val_metrics(self, device: torch.device) -> Dict[str, float]:
        pass


def sample_time_uniform(batch_size: int) -> Tensor:
    return torch.rand(batch_size, 1, 1, 1)


def sample_time_logit_normal(batch_size: int) -> Tensor:
    return torch.sigmoid(torch.normal(0.0, 0.6, size=(batch_size, 1, 1, 1)))
