from abc import ABC, abstractmethod
from typing import Any
import torch
from tqdm import tqdm

from flow_matching.unsupervised.model import MLPScore, MLPVectorField
from flow_matching.unsupervised.prob_paths import ConditionalProbabilityPath


class Trainer(ABC):
    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model

    @abstractmethod
    def get_train_loss(self, batch_size: int) -> torch.Tensor:
        pass

    def get_optimizer(self, lr: float) -> torch.optim.Optimizer:
        return torch.optim.Adam(self.model.parameters(), lr=lr)

    def train(
        self, num_epochs: int, device: torch.device, lr: float = 1e-3, **kwargs: Any
    ) -> None:
        # Start
        self.model.to(device)
        opt = self.get_optimizer(lr)
        self.model.train()

        # Train loop
        pbar = tqdm(enumerate(range(num_epochs)))
        for idx, epoch in pbar:
            opt.zero_grad()
            loss = self.get_train_loss(**kwargs)
            loss.backward()
            opt.step()
            pbar.set_description(f"Epoch {idx}, loss: {loss.item()}")

        # Finish
        self.model.eval()


class ConditionalFlowMatchingTrainer(Trainer):
    def __init__(
        self, path: ConditionalProbabilityPath, model: MLPVectorField, **kwargs: Any
    ):
        super().__init__(model, **kwargs)
        self.path = path

    def get_train_loss(self, batch_size: int) -> torch.Tensor:
        batch_z = self.path.p_data.sample(batch_size)
        batch_t = torch.rand(batch_size, 1)
        batch_x = self.path.sample_conditional_path(batch_z, batch_t)

        pred = self.model(batch_x, batch_t)
        ref = self.path.conditional_vector_field(batch_x, batch_z, batch_t)

        return torch.mean((pred - ref) ** 2)


class ConditionalScoreMatchingTrainer(Trainer):
    def __init__(
        self, path: ConditionalProbabilityPath, model: MLPScore, **kwargs: Any
    ) -> None:
        super().__init__(model, **kwargs)
        self.path = path

    def get_train_loss(self, batch_size: int) -> torch.Tensor:
        batch_z = self.path.p_data.sample(batch_size)
        batch_t = torch.rand(batch_size, 1)
        batch_x = self.path.sample_conditional_path(batch_z, batch_t)

        pred = self.model(batch_x, batch_t)
        ref = self.path.conditional_score(batch_x, batch_z, batch_t)

        return torch.mean((pred - ref) ** 2)
