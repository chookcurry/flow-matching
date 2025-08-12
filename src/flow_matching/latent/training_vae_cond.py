from typing import Any, Callable, Tuple
import torch
from tqdm import tqdm
from torch import Tensor
from aim import Run
from torch.utils.data import DataLoader

from flow_matching.evaluation.f1 import f1_score, precision_recall_knn
from flow_matching.evaluation.kid import kernel_inception_distance_polynomial_biased
from flow_matching.latent.vae import CondVAE
from flow_matching.supervised.training import MiB, model_size_b
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


class CondVAETrainer:
    def __init__(
        self,
        model: CondVAE,
        train_loader: DataLoader,
        val_loader: DataLoader,
        loss_fn: Callable[[Tensor, Tensor, Tensor, Tensor], Tensor],
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
        recon, mu, logvar = self.model(x.to(device), y.to(device))
        loss = self.loss_fn(recon, x, mu, logvar)
        return loss

    def get_val_metrics(
        self, batch: Tuple[Tensor, Tensor], device: torch.device
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        x, y = batch
        x1, x2 = torch.split(x, x.shape[0] // 2, dim=0)
        y1, y2 = torch.split(y, y.shape[0] // 2, dim=0)

        recon1, mu1, logvar1 = self.model(x1.to(device), y1.to(device))

        # x1 = decompress_stft(x1)
        # recon1 = decompress_stft(recon1)
        loss = self.loss_fn(recon1, x1, mu1, logvar1) * 30

        x2 = decompress_stft(x2)
        recon1 = decompress_stft(recon1)
        kid = kernel_inception_distance_polynomial_biased(x2, recon1)
        precision, recall = precision_recall_knn(x2, recon1)
        f1 = f1_score(precision, recall)

        return loss, kid, precision, recall, f1

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

            val_loss = torch.stack(val_losses).mean().item()
            val_kid = torch.stack(val_kids).mean().item()
            val_precision = torch.stack(val_precisions).mean().item()
            val_recall = torch.stack(val_recalls).mean().item()
            val_f1 = torch.stack(val_f1s).mean().item()

            print(
                f"Loss: {val_loss:.3f}, KID: {val_kid:.3f}, Precision: {val_precision:.3f}, Recall: {val_recall:.3f}, F1: {val_f1:.3f}"
            )

            if self.run:
                self.run.track(val_loss, name="val_loss")
                self.run.track(val_kid, name="val_kid")
                self.run.track(val_precision, name="val_precision")
                self.run.track(val_recall, name="val_recall")
                self.run.track(val_f1, name="val_f1")
