from abc import ABC, abstractmethod
from typing import Any, Dict
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
        val_batch_size: int = 500,
        patience: int = 5,
        run: Run | None = None,
    ) -> dict[str, Any]:
        size_b = model_size_b(self.model)
        logger.info(f"Training model with size: {size_b / MiB:.3f} MiB")

        self.model.to(device)
        optimizer = (
            self.optimizer
            if self.optimizer.param_groups[0]["lr"] == lr
            else self.get_optimizer(lr)
        )

        # Early stopping setup
        best_val_loss = float("inf")
        current_val_loss = float("inf")
        best_model_state = self.model.state_dict()
        patience_counter = 0

        train_losses = AverageMeter()

        for epoch in range(num_epochs):
            self.model.train()
            train_losses.reset()

            # setup progress bar
            pbar = tqdm(range(steps_per_epoch))
            pbar.set_description(f"Epoch {epoch}/{num_epochs}")

            # run tra
            for _ in pbar:
                optimizer.zero_grad()
                train_loss = self.get_train_loss(batch_size, device)

                run.log({"train/loss": train_loss.item()}) if run else None
                train_losses.update(train_loss.item())
                pbar.set_postfix(train_loss=f"{train_losses.avg:.6f}")

                train_loss.backward()
                optimizer.step()

            if not validate:
                continue

            self.model.eval()

            # validate
            val_loss = self.get_val_loss(val_batch_size, device)
            logger.info(f"val loss: {val_loss}")
            run.log({"val/loss": val_loss.item()}) if run else None

            # update when val loss improves
            current_val_loss = float(val_loss.item())
            if current_val_loss < best_val_loss:
                best_val_loss = current_val_loss
                best_model_state = self.model.state_dict()
                patience_counter = 0
            else:
                patience_counter += 1

            # early stopping
            if patience_counter >= patience:
                logger.info(f"Early stopping triggered at epoch {epoch}/{num_epochs}")
                break

            # val_metrics = self.get_val_metrics(device)
            # logger.info([f"{key}: {value:.6f}" for key, value in val_metrics.items()])
            # run.log({f"val/{k}": v for k, v in val_metrics.items()}) if run else None

        self.model.load_state_dict(best_model_state)
        return best_model_state

    @abstractmethod
    def get_train_loss(self, batch_size: int, device: torch.device) -> Tensor:
        pass

    @abstractmethod
    @torch.no_grad()
    def get_val_loss(self, batch_size: int, device: torch.device) -> Tensor:
        pass

    # @abstractmethod
    # @torch.no_grad()
    # def get_val_metrics(self, device: torch.device) -> Dict[str, float]:
    #     pass
