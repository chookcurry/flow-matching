from typing import Any, List, Tuple
import torch
from tqdm import tqdm
from torch import Tensor, vmap
from aim import Run
from torch.utils.data import DataLoader

from flow_matching.supervised.training import MiB, model_size_b
from flow_matching.whar.ae_losses import (
    ae_log_mag,
    ae_log_mag_phase,
    ae_mse,
    ae_spect_conv,
)
from flow_matching.whar.models.vae_cond import CondSpectrogramVAE
from flow_matching.whar.stft import (
    compress_stft,
    decompress_stft,
    istft_transform,
    stft_transform,
)
from flow_matching.whar.vae_losses import vae_mse


def default_collate_fn(batch: List[Tuple[Tensor, Tensor]]) -> Tuple[Tensor, Tensor]:
    x_list = []
    y_list = []

    for y, x in batch:
        x = stft_transform(x)
        x = compress_stft(x)

        C, RI, F, T = x.shape
        x = x.reshape(C * RI, F, T)

        x_list.append(x)
        y_list.append(y)

    x_stack = torch.stack(x_list)
    y_stack = torch.stack(y_list)

    return x_stack, y_stack


class CondVAETrainer:
    def __init__(
        self,
        model: CondSpectrogramVAE,
        train_loader: DataLoader,
        val_loader: DataLoader,
        eta: float,
        null_class: int,
        track: bool = False,
    ):
        super().__init__()

        assert eta > 0 and eta < 1

        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.eta = eta
        self.null_class = null_class

        self.train_loader.collate_fn = default_collate_fn
        self.val_loader.collate_fn = default_collate_fn

        self.run = (
            Run(log_system_params=False, system_tracking_interval=None)
            if track
            else None
        )

    def get_optimizer(self, lr: float):
        return torch.optim.Adam(self.model.parameters(), lr=lr)

    def get_train_loss(
        self, batch: Tuple[Tensor, Tensor], device: torch.device, beta: float
    ) -> Tensor:
        x, y = batch
        x, y = x.to(device), y.to(device)

        mask = torch.rand(y.shape[0]) < self.eta
        y[mask] = self.null_class

        recon, mu, logvar = self.model(x, y)
        loss = vae_mse(recon, x, mu, logvar)

        return loss

    @torch.no_grad()
    def get_val_metrics(
        self, batch: Tuple[Tensor, Tensor], device: torch.device, beta: float
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        x, y = batch
        x, y = x.to(device), y.to(device)

        _, mu, logvar = self.model.encode(x, y)
        recon = self.model.decode(mu, y)
        loss = vae_mse(recon, x, mu, logvar)

        x = x.detach().cpu()
        recon = recon.detach().cpu()

        mse = ae_mse(recon, x)
        log_mag = ae_log_mag(recon, x)
        log_mag_phase = ae_log_mag_phase(recon, x)
        spect_conv = ae_spect_conv(recon, x)

        B, C, H, W = x.shape

        x_time = vmap(decompress_stft)(x.reshape(B, C // 2, 2, H, W))
        recon_time = vmap(decompress_stft)(recon.reshape(B, C // 2, 2, H, W))

        x_time = vmap(istft_transform)(x_time)
        recon_time = vmap(istft_transform)(recon_time)

        time_mse = ((x_time - recon_time) ** 2).sum()
        time_mae = (x_time - recon_time).abs().sum()

        return loss, mse, log_mag, log_mag_phase, spect_conv, time_mse, time_mae

    def train(
        self, num_epochs: int, device: torch.device, lr: float = 1e-3, **kwargs: Any
    ) -> None:
        beta = kwargs["beta"]

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
                loss = self.get_train_loss(
                    batch, device, beta=beta or epoch / num_epochs
                )

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
            val_time_mses = []
            val_time_maes = []

            pbar = tqdm(self.val_loader)
            for batch in pbar:
                loss, mse, log_mag, log_mag_phase, spect_conv, time_mse, time_mae = (
                    self.get_val_metrics(batch, device, beta=beta or epoch / num_epochs)
                )

                val_losses.append(loss)
                val_mses.append(mse)
                val_log_mags.append(log_mag)
                val_log_mag_phases.append(log_mag_phase)
                val_spect_convs.append(spect_conv)
                val_time_mses.append(time_mse)
                val_time_maes.append(time_mae)

            val_loss = torch.stack(val_losses).mean().item()
            val_mse = torch.stack(val_mses).mean().item()
            val_log_mag = torch.stack(val_log_mags).mean().item()
            val_log_mag_phase = torch.stack(val_log_mag_phases).mean().item()
            val_spect_conv = torch.stack(val_spect_convs).mean().item()
            val_time_mse = torch.stack(val_time_mses).mean().item()
            val_time_mae = torch.stack(val_time_maes).mean().item()

            print(
                f"Loss: {val_loss:.3f}, MSE: {val_mse:.3f}, Log Mag: {val_log_mag:.3f}, Log Mag Phase: {val_log_mag_phase:.3f}, Spect Conv: {val_spect_conv:.3f}, Time MSE: {val_time_mse:.3f}, Time MAE: {val_time_mae:.3f}"
            )

            if self.run:
                self.run.track(val_loss, name="val_loss")
                self.run.track(val_mse, name="val_mse")
                self.run.track(val_log_mag, name="val_log_mag")
                self.run.track(val_log_mag_phase, name="val_log_mag_phase")
                self.run.track(val_spect_conv, name="val_spect_conv")
                self.run.track(val_time_mse, name="val_time_mse")
                self.run.track(val_time_mae, name="val_time_mae")
