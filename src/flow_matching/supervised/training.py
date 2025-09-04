from abc import ABC, abstractmethod
from typing import Dict
import torch
from tqdm import tqdm
from torch import Tensor
from wandb import Run
from torch.optim import Optimizer, Adam

from flow_matching.supervised.odes_sdes import ConditionalVectorField
from flow_matching.supervised.prob_paths import ConditionalProbabilityPath
from flow_matching.utils.utils import model_size_b, MiB
from flow_matching.utils.logging import logger


def sample_time_uniform(batch_size: int) -> torch.Tensor:
    return torch.rand(batch_size, 1, 1, 1)


def sample_time_logit_normal(batch_size: int) -> torch.Tensor:
    return torch.sigmoid(torch.normal(0.0, 0.6, size=(batch_size, 1, 1, 1)))


class Trainer(ABC):
    def __init__(self, model: ConditionalVectorField):
        super().__init__()
        self.model = model
        self.optimizer = self.get_optimizer()

    @abstractmethod
    def get_train_loss(self, batch_size: int, device: torch.device) -> Tensor:
        pass

    @abstractmethod
    @torch.no_grad()
    def get_val_metrics(self, device: torch.device) -> Dict[str, float]:
        pass

    def get_optimizer(self, lr: float = 1e-3) -> Optimizer:
        return Adam(self.model.parameters(), lr=lr)

    def train(
        self,
        num_epochs: int,
        device: torch.device,
        batch_size: int,
        lr: float = 1e-3,
        val_every_n_epochs: int = 1000,
        run: Run | None = None,
    ) -> None:
        # Report model size
        size_b = model_size_b(self.model)
        logger.info(f"Training model with size: {size_b / MiB:.3f} MiB")

        # Start
        self.model.to(device)
        optimizer = (
            self.optimizer
            if self.optimizer.param_groups[0]["lr"] == lr
            else self.get_optimizer(lr)
        )

        # Train loop
        pbar = tqdm(range(num_epochs))
        for epoch in pbar:
            self.model.train()

            optimizer.zero_grad()
            loss = self.get_train_loss(batch_size=batch_size, device=device)

            run.log({"train/loss": loss.item()}) if run else None
            pbar.set_description(f"Epoch {epoch}, loss: {loss.item():.3f}")

            loss.backward()
            optimizer.step()

            if epoch % val_every_n_epochs == 0:
                self.model.eval()

                metrics = self.get_val_metrics(device)

                run.log({"val/" + k: v for k, v in metrics.items()}) if run else None
                logger.info(
                    f"Epoch {epoch},", *[f"{k}: {v:.3f}" for k, v in metrics.items()]
                )

        # Finish
        self.model.eval()


class FlowTrainer(Trainer):
    def __init__(
        self,
        path: ConditionalProbabilityPath,
        model: ConditionalVectorField,
        eta: float,
        null_class: int,
    ):
        super().__init__(model)

        assert eta > 0 and eta < 1

        self.eta = eta
        self.path = path
        self.null_class = null_class
        self.sample_time = sample_time_uniform

    def get_val_metrics(self, device: torch.device) -> Dict[str, float]:
        return {}

    def get_train_loss(self, batch_size: int, device: torch.device) -> Tensor:
        # Step 1: Sample z,y from p_data
        batch_z, batch_y = self.path.p_data.sample(batch_size)
        assert batch_y is not None

        batch_z, batch_y = batch_z.to(device), batch_y.to(device)

        # Step 2: Set each label to null class with probability eta
        mask = torch.rand(batch_size) < self.eta
        batch_y[mask] = self.null_class

        # Step 3: Sample t and x
        batch_t = self.sample_time(batch_size).to(device)
        batch_x = self.path.sample_conditional_path(batch_z, batch_t)

        # Step 4: Regress and output loss
        pred = self.model(batch_x, batch_t, batch_y)
        ref = self.path.conditional_vector_field(batch_x, batch_z, batch_t)

        return torch.mean((pred - ref) ** 2)
