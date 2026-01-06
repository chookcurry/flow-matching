from abc import ABC, abstractmethod
from typing import Any, List

import matplotlib.pyplot as plt
import torch
from torch import Tensor, nn
from torch.optim import AdamW
from tqdm import tqdm
from wandb import Run

from diffusion.utils.logging import logger
from diffusion.utils.utils import AverageMeter, MiB, model_size_b


class Trainer(ABC):
    def __init__(self, model: nn.Module):
        super().__init__()

        self.model = model

    def train(
        self,
        num_epochs: int,
        steps_per_epoch: int,
        batch_size: int,
        device: torch.device,
        lr: float = 1e-3,
        validate: bool = True,
        val_batch_size: int | None = None,
        patience: int | None = 5,
        run: Run | None = None,
        plot_path: str | None = None,
    ) -> dict[str, Any]:
        size_b = model_size_b(self.model)
        logger.info(f"Training model with size: {size_b / MiB:.3f} MiB")

        self.model.to(device)
        optimizer = AdamW(self.model.parameters(), lr=lr)

        # Early stopping setup
        best_val_loss = float("inf")
        current_val_loss = float("inf")
        best_model_state = self.model.state_dict()
        patience_counter = 0

        meter = AverageMeter()
        train_losses: List[float] = []
        val_losses: List[float] = []

        for epoch in range(num_epochs):
            self.model.train()
            meter.reset()

            # setup progress bar
            pbar = tqdm(range(steps_per_epoch))
            pbar.set_description(f"Epoch {epoch}/{num_epochs}")

            # loop
            for _ in pbar:
                optimizer.zero_grad()
                train_loss = self.get_train_loss(batch_size, device)

                run.log({"train/loss": train_loss.item()}) if run else None
                meter.update(train_loss.item())
                pbar.set_postfix(train_loss=f"{meter.avg:.6f}")
                train_losses.append(train_loss.item())

                train_loss.backward()
                optimizer.step()

            if not validate:
                continue

            self.model.eval()

            # validate
            val_loss = self.get_val_loss(val_batch_size or batch_size, device)

            # update when val loss improves
            current_val_loss = float(val_loss.item())

            if current_val_loss < best_val_loss:
                best_val_loss = current_val_loss
                best_model_state = self.model.state_dict()
                patience_counter = 0
            else:
                patience_counter += 1

            logger.info(f"val loss: {val_loss}, best val loss: {best_val_loss}")
            run.log({"val/loss": val_loss.item()}) if run else None
            val_losses.append(val_loss.item())

            if plot_path is not None:
                # inflate val losses to match train losses length for plotting
                inflated_val_losses = []
                for v in val_losses:
                    inflated_val_losses.extend([v] * steps_per_epoch)

                plt.figure(figsize=(10, 5))
                plt.plot(train_losses, label="Train Loss")
                plt.plot(inflated_val_losses, label="Validation Loss")
                plt.xlabel("Iterations")
                plt.ylabel("Loss")
                plt.title("Training and Validation Loss Over Time")
                plt.legend()
                plt.yscale("log")
                plt.savefig(plot_path)
                plt.close()

            # early stopping
            if patience is not None and patience_counter >= patience:
                logger.info(
                    f"Early stopping triggered at epoch {epoch}/{num_epochs} with best val loss: {best_val_loss}"
                )
                break

            # val_metrics = self.get_val_metrics(device)
            # logger.info([f"{key}: {value:.6f}" for key, value in val_metrics.items()])
            # run.log({f"val/{k}": v for k, v in val_metrics.items()}) if run else None

        return best_model_state

    @abstractmethod
    def get_train_loss(self, batch_size: int, device: torch.device) -> Tensor:
        pass

    @abstractmethod
    def get_val_loss(self, batch_size: int, device: torch.device) -> Tensor:
        pass

    # @abstractmethod
    # @torch.no_grad()
    # def get_val_metrics(self, device: torch.device) -> Dict[str, float]:
    #     pass
