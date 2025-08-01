from abc import ABC, abstractmethod
from typing import Any
import torch
from tqdm import tqdm
from torch import nn
from aim import Run

from flow_matching.supervised.odes_sdes import ConditionalVectorField
from flow_matching.supervised.prob_paths import ConditionalProbabilityPath


MiB = 1024**2


def model_size_b(model: nn.Module) -> int:
    """
    Returns model size in bytes. Based on https://discuss.pytorch.org/t/finding-model-size/130275/2
    Args:
    - model: self-explanatory
    Returns:
    - size: model size in bytes
    """
    size = 0
    for param in model.parameters():
        size += param.nelement() * param.element_size()
    for buf in model.buffers():
        size += buf.nelement() * buf.element_size()
    return size


class Trainer(ABC):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
        self.run = Run()

    @abstractmethod
    def get_train_loss(self, batch_size: int) -> torch.Tensor:
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
        opt = self.get_optimizer(lr)
        self.model.train()

        # Train loop
        pbar = tqdm(range(num_epochs))
        for epoch in pbar:
            opt.zero_grad()
            loss = self.get_train_loss(**kwargs)

            self.run.track(loss.item(), name="loss")

            loss.backward()
            opt.step()
            pbar.set_description(f"Epoch {epoch}, loss: {loss.item():.3f}")

        # Finish
        self.model.eval()


class CFGTrainer(Trainer):
    def __init__(
        self,
        path: ConditionalProbabilityPath,
        model: ConditionalVectorField,
        eta: float,
        null_class: int,
        **kwargs: Any,
    ):
        assert eta > 0 and eta < 1
        super().__init__(model, **kwargs)
        self.eta = eta
        self.path = path
        self.null_class = null_class

    def get_train_loss(self, batch_size: int) -> torch.Tensor:
        # Step 1: Sample z,y from p_data
        batch_z, batch_y = self.path.p_data.sample(batch_size)
        assert batch_y is not None

        # Step 2: Set each label to 10 (i.e., null) with probability eta
        mask = torch.rand(batch_size) < self.eta
        batch_y[mask] = self.null_class  # Set to null label

        # Step 3: Sample t and x
        # batch_t = torch.rand(batch_size, 1, 1, 1)
        mu = 0.0  # logit(0.5)
        sigma = 0.6  # controls concentration around 0.5
        logit_t = torch.normal(mean=mu, std=sigma, size=(batch_size, 1, 1, 1))
        batch_t = torch.sigmoid(logit_t)  # Now t is in [0,1], peaking near 0.5
        batch_x = self.path.sample_conditional_path(batch_z, batch_t)

        # Step 4: Regress and output loss
        pred = self.model(batch_x, batch_t, batch_y)
        ref = self.path.conditional_vector_field(batch_x, batch_z, batch_t)

        return torch.mean((pred - ref) ** 2)
