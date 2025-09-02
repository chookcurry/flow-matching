from typing import Dict, List, Tuple
import numpy as np
import torch
from tqdm import tqdm
from torch import Tensor, vmap
from aim import Run  # type: ignore
from torch.utils.data import DataLoader
from torch.optim import Optimizer, Adam

from flow_matching.latent.autoencoder import AE, AEC, CAE, CAEC
from flow_matching.utils.utils import MiB, model_size_b
from flow_matching.whar.ae_losses import ae_log_mag_phase, ae_mse, ae_spect_conv
from flow_matching.whar.supcon import loss_supcon
from flow_matching.whar.stft import (
    compress_stft,
    decompress_stft,
    istft_transform,
    stft_transform,
)


def transform(x: Tensor) -> Tensor:
    x = stft_transform(x)
    x = compress_stft(x)

    C, RI, F, T = x.shape
    x = x.reshape(C * RI, F, T)

    return x


def detransform(x: Tensor) -> Tensor:
    C, F, T = x.shape
    x = x.reshape(C // 2, 2, F, T)

    x = decompress_stft(x)
    x = istft_transform(x)

    return x


def loss_fn(x: Tensor, y: Tensor, z: Tensor, recon: Tensor) -> Tensor:
    return ae_mse(recon, x) + 0.01 * loss_supcon(z.view(z.shape[0], -1), y)


def collate_fn(batch: List[Tuple[Tensor, Tensor]]) -> Tuple[Tensor, Tensor]:
    x_list = []
    y_list = []

    for y, x in batch:
        x = transform(x)

        x_list.append(x)
        y_list.append(y)

    x_stack = torch.stack(x_list)
    y_stack = torch.stack(y_list)

    return x_stack, y_stack


class CAETrainer:
    def __init__(
        self,
        model: AE | CAE | AEC | CAEC,
        train_loader: DataLoader[Tuple[Tensor, Tensor]],
        val_loader: DataLoader[Tuple[Tensor, Tensor]],
        eta: float,
        null_class: int,
        track: bool = False,
    ) -> None:
        super().__init__()

        assert eta > 0 and eta < 1

        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.eta = eta
        self.null_class = null_class

        self.detransform = detransform
        self.loss_fn = loss_fn
        self.train_loader.collate_fn = collate_fn
        self.val_loader.collate_fn = collate_fn

        self.run = (
            Run(log_system_params=False, system_tracking_interval=None)
            if track
            else None
        )

        self.optimizer = self.get_optimizer()

    def get_optimizer(self, lr: float = 1e-3) -> Optimizer:
        return Adam(self.model.parameters(), lr=lr)

    def get_train_loss(
        self, batch: Tuple[Tensor, Tensor], device: torch.device
    ) -> Tensor:
        x, y = batch
        x, y = x.to(device), y.to(device)

        mask = torch.rand(y.shape[0]) < self.eta
        y[mask] = self.null_class

        recon, z = self.model(x, y)
        loss = loss_fn(x, y, z, recon)

        return loss

    @torch.no_grad()
    def get_val_metrics(
        self, batch: Tuple[Tensor, Tensor], device: torch.device
    ) -> Dict[str, float]:
        x, y = batch
        x, y = x.to(device), y.to(device)

        recon, z = self.model(x, y)
        loss = loss_fn(x, y, z, recon)

        x = x.detach().cpu()
        recon = recon.detach().cpu()

        mse = ae_mse(recon, x)
        log_mag_phase = ae_log_mag_phase(recon, x)
        spect_conv = ae_spect_conv(recon, x)

        x_time = vmap(self.detransform)(x)
        recon_time = vmap(self.detransform)(recon)

        time_mse = ((x_time - recon_time) ** 2).sum()
        time_mae = (x_time - recon_time).abs().sum()

        return {
            "loss": loss.item(),
            "mse": mse.item(),
            "log_mag_phase": log_mag_phase.item(),
            "spect_conv": spect_conv.item(),
            "time_mse": time_mse.item(),
            "time_mae": time_mae.item(),
        }

    def train(self, num_epochs: int, device: torch.device, lr: float = 1e-3) -> None:
        # Report model size
        size_b = model_size_b(self.model)
        print(f"Training model with size: {size_b / MiB:.3f} MiB")

        # Start
        self.model.to(device)
        optimizer = (
            self.optimizer
            if self.optimizer.param_groups[0]["lr"] == lr
            else self.get_optimizer(lr)
        )

        for epoch in range(num_epochs):
            # Train loop
            self.model.train()

            pbar = tqdm(self.train_loader)
            for batch in pbar:
                optimizer.zero_grad()
                loss = self.get_train_loss(batch, device)

                if self.run:
                    self.run.track(loss.item(), name="train/loss")

                if loss.isnan():
                    continue

                loss.backward()
                optimizer.step()
                pbar.set_description(f"Epoch {epoch}, Loss: {loss.item():.3f}")

            # val loop
            self.model.eval()

            metrics_lists: Dict[str, List[float]] = {}

            pbar = tqdm(self.val_loader)
            for batch in pbar:
                metrics = self.get_val_metrics(batch, device)

                for k, v in metrics.items():
                    metrics_lists.setdefault(k, []).append(v)

            metrics = {k: float(np.mean(v)) for k, v in metrics_lists.items()}

            print([f"{key}: {value:.3f}" for key, value in metrics.items()])

            if self.run:
                for k, v in metrics.items():
                    self.run.track(v, name=f"val/{k}")
