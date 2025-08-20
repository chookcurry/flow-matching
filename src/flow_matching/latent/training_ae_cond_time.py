from typing import Any, Tuple
import torch
from tqdm import tqdm
from torch import Tensor
from aim import Run
from torch.utils.data import DataLoader

from flow_matching.latent.ae import CondAutoencoder
from flow_matching.supervised.training import MiB, model_size_b


class CondAETimeTrainer:
    def __init__(
        self,
        model: CondAutoencoder,
        train_loader: DataLoader,
        val_loader: DataLoader,
        eta: float,
        null_class: int,
        loss_fn=torch.nn.MSELoss(),
        track: bool = False,
    ):
        super().__init__()

        assert eta > 0 and eta < 1

        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.loss_fn = loss_fn
        self.eta = eta
        self.null_class = null_class

        self.run = (
            Run(log_system_params=False, system_tracking_interval=None)
            if track
            else None
        )

    def get_optimizer(self, lr: float):
        return torch.optim.Adam(self.model.parameters(), lr=lr)

    def get_train_loss(
        self, batch: Tuple[Tensor, Tensor], device: torch.device
    ) -> Tensor:
        y, x = batch
        x, y = x.to(device), y.to(device)

        mask = torch.rand(y.shape[0]) < self.eta
        y[mask] = self.null_class

        recon, _ = self.model(x, y)
        loss = self.loss_fn(recon, x)

        return loss

    def get_val_metrics(
        self, batch: Tuple[Tensor, Tensor], device: torch.device
    ) -> Tuple[Tensor, Tensor, Tensor]:
        y, x = batch
        x, y = x.to(device), y.to(device)

        recon, _ = self.model(x, y)
        loss = self.loss_fn(recon, x)

        x = x.detach().cpu()
        recon = recon.detach().cpu()

        mse = ((x - recon) ** 2).sum()
        mae = (x - recon).abs().sum()

        return loss, mse, mae

    def train(
        self, num_epochs: int, device: torch.device, lr: float = 1e-3, **kwargs: Any
    ) -> None:
        # Report model size
        size_b = model_size_b(self.model)
        print(f"Training model with size: {size_b / MiB:.3f} MiB")

        # Start
        self.model.to(device)
        optimizer = self.get_optimizer(lr)
        self.model.train()

        for epoch in range(num_epochs):
            self.model.train()

            # Train loop
            pbar = tqdm(self.train_loader)
            for batch in pbar:
                optimizer.zero_grad()
                loss = self.get_train_loss(batch, device)

                if self.run:
                    self.run.track(loss.item(), name="train_loss")

                loss.backward()
                optimizer.step()
                pbar.set_description(f"Epoch {epoch}, Loss: {loss.item():.3f}")

            # Finish
            self.model.eval()

            # Validation
            val_losses = []
            val_mses = []
            val_maes = []

            pbar = tqdm(self.val_loader)
            for batch in pbar:
                loss, mse, mae = self.get_val_metrics(batch, device)

                val_losses.append(loss)
                val_mses.append(mse)
                val_maes.append(mae)

            val_loss = torch.stack(val_losses).mean().item()
            val_mse = torch.stack(val_mses).mean().item()
            val_mae = torch.stack(val_maes).mean().item()

            print(f"Loss: {val_loss:.3f}, MSE: {val_mse:.3f}, MAE: {val_mae:.3f}")

            if self.run:
                self.run.track(val_loss, name="val_loss")
                self.run.track(val_mse, name="val_mse")
                self.run.track(val_mae, name="val_mae")
