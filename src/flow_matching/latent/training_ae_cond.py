from typing import Any, Callable, Tuple
import torch
from tqdm import tqdm
from torch import Tensor, vmap
from aim import Run
from torch.utils.data import DataLoader

from flow_matching.latent.ae import CondAutoencoder
from flow_matching.supervised.training import MiB, model_size_b
from flow_matching.whar.ae_losses import (
    ae_log_mag,
    ae_log_mag_phase,
    ae_mse,
    ae_spect_conv,
)
from flow_matching.whar.stft import compress_stft, decompress_stft, stft_transform


def default_collate_fn(batch: Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tensor]:
    transformed_batch = []

    for y, x in batch:
        x_transformed = stft_transform(x)
        x_transformed = compress_stft(x_transformed)
        C, RI, F, T = x_transformed.shape
        x_transformed = x_transformed.reshape(C * RI, F, T)
        transformed_batch.append((x_transformed, y))

    x_stack = torch.stack([x for x, _ in transformed_batch])
    y_stack = torch.stack([y for _, y in transformed_batch])

    return x_stack, y_stack


class CondAETrainer:
    def __init__(
        self,
        model: CondAutoencoder,
        train_loader: DataLoader,
        val_loader: DataLoader,
        loss_fn: Callable[[Tensor, Tensor], Tensor],
        track: bool = False,
    ):
        super().__init__()
        self.model = model

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.train_loader.collate_fn = default_collate_fn
        self.val_loader.collate_fn = default_collate_fn

        self.loss_fn = loss_fn

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
        x, y = batch
        x, y = x.to(device), y.to(device)
        recon, _ = self.model(x, y)
        loss = self.loss_fn(recon, x)
        return loss

    def get_val_metrics(
        self, batch: Tuple[Tensor, Tensor], device: torch.device
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        x, y = batch
        x, y = x.to(device), y.to(device)

        recon, _ = self.model(x, y)
        loss = self.loss_fn(recon, x)

        x = x.detach().cpu()
        recon = recon.detach().cpu()

        x = vmap(decompress_stft)(x)
        recon = vmap(decompress_stft)(recon)

        mse = ae_mse(recon, x)
        log_mag = ae_log_mag(recon, x)
        log_mag_phase = ae_log_mag_phase(recon, x)
        spect_conv = ae_spect_conv(recon, x)

        return loss, mse, log_mag, log_mag_phase, spect_conv

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
            val_log_mags = []
            val_log_mag_phases = []
            val_spect_convs = []

            pbar = tqdm(self.val_loader)
            for batch in pbar:
                loss, mse, log_mag, log_mag_phase, spect_conv = self.get_val_metrics(
                    batch, device
                )

                val_losses.append(loss)
                val_mses.append(mse)
                val_log_mags.append(log_mag)
                val_log_mag_phases.append(log_mag_phase)
                val_spect_convs.append(spect_conv)

            val_loss = torch.stack(val_losses).mean().item()
            val_mse = torch.stack(val_mses).mean().item()
            val_log_mag = torch.stack(val_log_mags).mean().item()
            val_log_mag_phase = torch.stack(val_log_mag_phases).mean().item()
            val_spect_conv = torch.stack(val_spect_convs).mean().item()

            print(
                f"Loss: {val_loss:.3f}, MSE: {val_mse:.3f}, Log Mag: {val_log_mag:.3f}, Log Mag Phase: {val_log_mag_phase:.3f}, Spect Conv: {val_spect_conv:.3f}"
            )

            if self.run:
                self.run.track(val_loss, name="val_loss")
                self.run.track(val_mse, name="val_mse")
                self.run.track(val_log_mag, name="val_log_mag")
                self.run.track(val_log_mag_phase, name="val_log_mag_phase")
                self.run.track(val_spect_conv, name="val_spect_conv")
