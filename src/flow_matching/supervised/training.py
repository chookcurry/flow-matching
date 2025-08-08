from abc import ABC, abstractmethod
from typing import Any, Callable
import torch
from tqdm import tqdm
from torch import Tensor, nn
from aim import Run

from flow_matching.supervised.odes_sdes import ConditionalVectorField
from flow_matching.supervised.prob_paths import ConditionalProbabilityPath


MiB = 1024**2


def model_size_b(model: nn.Module) -> int:
    size = 0
    for param in model.parameters():
        size += param.nelement() * param.element_size()
    for buf in model.buffers():
        size += buf.nelement() * buf.element_size()
    return size


def sample_time_uniform(batch_size: int) -> torch.Tensor:
    return torch.rand(batch_size, 1, 1, 1)


def sample_time_logit_normal(batch_size: int) -> torch.Tensor:
    return torch.sigmoid(torch.normal(0.0, 0.6, size=(batch_size, 1, 1, 1)))


class Trainer(ABC):
    def __init__(self, model: nn.Module, track: bool = False):
        super().__init__()
        self.model = model
        self.run = (
            Run(log_system_params=False, system_tracking_interval=None)
            if track
            else None
        )

    @abstractmethod
    def get_train_loss(self, batch_size: int) -> Tensor:
        pass

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
        optimizer = self.get_optimizer(lr)
        self.model.train()

        # Train loop
        pbar = tqdm(range(num_epochs))
        for epoch in pbar:
            optimizer.zero_grad()
            loss = self.get_train_loss(**kwargs)

            if self.run:
                self.run.track(loss.item(), name="loss")

            loss.backward()
            optimizer.step()
            pbar.set_description(f"Epoch {epoch}, loss: {loss.item():.3f}")

        # Finish
        self.model.eval()


class FlowTrainer(Trainer):
    def __init__(
        self,
        path: ConditionalProbabilityPath,
        model: ConditionalVectorField,
        eta: float,
        null_class: int,
        track: bool = False,
        sample_time: Callable[[int], Tensor] = sample_time_logit_normal,
    ):
        super().__init__(model, track)

        assert eta > 0 and eta < 1

        self.eta = eta
        self.path = path
        self.null_class = null_class
        self.sample_time = sample_time

    def get_train_loss(self, batch_size: int) -> Tensor:
        # Step 1: Sample z,y from p_data
        batch_z, batch_y = self.path.p_data.sample(batch_size)
        assert batch_y is not None

        # Step 2: Set each label to null class with probability eta
        mask = torch.rand(batch_size) < self.eta
        batch_y[mask] = self.null_class

        # Step 3: Sample t and x
        batch_t = self.sample_time(batch_size)
        batch_x = self.path.sample_conditional_path(batch_z, batch_t)

        # Step 4: Regress and output loss
        pred = self.model(batch_x, batch_t, batch_y)
        ref = self.path.conditional_vector_field(batch_x, batch_z, batch_t)

        return torch.mean((pred - ref) ** 2)
