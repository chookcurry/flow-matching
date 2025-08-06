from typing import Any, Tuple
from aim import Run
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from flow_matching.evaluation.f1 import f1_score, precision_recall_knn
from flow_matching.evaluation.kid import kernel_inception_distance_polynomial
from flow_matching.supervised.training import MiB, model_size_b

LATENT_CHANNELS = 8
LATENT_H = 4
LATENT_W = 4


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 4, stride=2, padding=1),  # 16x16
            nn.GroupNorm(4, 32),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2, padding=1),  # 8x8
            nn.GroupNorm(8, 64),
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),  # 4x4
            nn.GroupNorm(8, 128),
            nn.ReLU(),
        )
        self.mu_layer = nn.Conv2d(128, LATENT_CHANNELS, kernel_size=3, padding=1)
        self.logvar_layer = nn.Conv2d(128, LATENT_CHANNELS, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.net(x)  # B x 128 x 4 x 4
        mu = self.mu_layer(h)  # B x 32 x 4 x 4
        logvar = self.logvar_layer(h)
        return mu, logvar


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(LATENT_CHANNELS, 128, 3, padding=1),
            nn.GroupNorm(8, 128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),  # 8x8
            nn.GroupNorm(8, 64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),  # 16x16
            nn.GroupNorm(4, 32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1),  # 32x32
            nn.GroupNorm(4, 16),
            nn.ReLU(),
            nn.Conv2d(16, 1, kernel_size=3, padding=1),
            nn.Sigmoid(),  # MNIST is [0,1]
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class MNIST_VAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decoder(z)
        return recon, mu, logvar


def vae_loss(
    recon_x: torch.Tensor, x: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor
):
    recon_loss = F.binary_cross_entropy(recon_x, x, reduction="sum")
    # KL divergence between N(mu, sigma) and N(0,1)
    kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kl_div


class VAETrainer:
    def __init__(
        self,
        model: MNIST_VAE,
        train_loader: DataLoader,
        val_loader: DataLoader,
        track: bool = False,
    ):
        super().__init__()
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.run = (
            Run(log_system_params=False, system_tracking_interval=None)
            if track
            else None
        )

    def get_optimizer(self, lr: float):
        return torch.optim.Adam(self.model.parameters(), lr=lr)

    def train(
        self, num_epochs: int, device: torch.device, lr: float = 1e-3, **kwargs: Any
    ) -> None:
        # Report model size
        size_b = model_size_b(self.model)
        print(f"Training model with size: {size_b / MiB:.3f} MiB")

        # Start
        self.model.to(device)
        opt = self.get_optimizer(lr)
        self.model.train()

        for epoch in range(num_epochs):
            self.model.train()

            # Train loop
            pbar = tqdm(self.train_loader)
            for idx, batch in enumerate(pbar):
                opt.zero_grad()
                loss = self.get_train_loss(batch, device)

                if self.run:
                    self.run.track(loss.item(), name="train_loss")

                loss.backward()
                opt.step()
                pbar.set_description(f"Epoch {epoch}, Loss: {loss.item():.3f}")

            # Finish
            self.model.eval()

            # Validation
            val_losses = []
            val_kids = []
            val_precisions = []
            val_recalls = []
            val_f1s = []

            pbar = tqdm(self.val_loader)
            for batch in pbar:
                loss, kid, precision, recall, f1 = self.get_val_metrics(batch, device)

                val_losses.append(loss)
                val_kids.append(kid)
                val_precisions.append(precision)
                val_recalls.append(recall)
                val_f1s.append(f1)

            val_loss = torch.stack(val_losses).mean()
            val_kid = torch.stack(val_kids).mean()
            val_precision = torch.stack(val_precisions).mean()
            val_recall = torch.stack(val_recalls).mean()
            val_f1 = torch.stack(val_f1s).mean()

            print(
                f"Epoch {epoch}, Loss: {val_loss.item():.3f}, KID: {val_kid.item():.3f}, Precision: {val_precision.item():.3f}, Recall: {val_recall.item():.3f}, F1: {val_f1.item():.3f}"
            )

            if self.run:
                self.run.track(val_loss.item(), name="val_loss")
                self.run.track(val_kid.item(), name="val_kid")
                self.run.track(val_precision.item(), name="val_precision")
                self.run.track(val_recall.item(), name="val_recall")
                self.run.track(val_f1.item(), name="val_f1")

    def get_train_loss(
        self, batch: Tuple[torch.Tensor, torch.Tensor], device: torch.device
    ) -> torch.Tensor:
        x, _ = batch
        recon, mu, logvar = self.model(x.to(device))
        return vae_loss(recon, x, mu, logvar)

    def get_val_metrics(
        self, batch: Tuple[torch.Tensor, torch.Tensor], device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x, _ = batch
        x1, x2 = torch.split(x, x.shape[0] // 2, dim=0)

        recon, mu, logvar = self.model(x1.to(device))

        loss = vae_loss(recon, x1, mu, logvar)
        kid = kernel_inception_distance_polynomial(x2, recon)
        precision, recall = precision_recall_knn(x2, recon)
        f1 = f1_score(precision, recall)

        return loss, kid, precision, recall, f1
