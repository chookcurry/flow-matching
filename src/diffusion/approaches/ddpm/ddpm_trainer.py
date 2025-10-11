from typing import Dict

import torch
from torch import Tensor
import torch.nn.functional as F

from diffusion.architectures.backbones.backbone import Backbone
from diffusion.training.trainer import Trainer
from diffusion.approaches.ddpm.backward_process import BackwardProcess, ForwardProcess
from diffusion.sampleables.sampleable import Sampleable


class DDPMTrainer(Trainer):
    def __init__(
        self,
        dataset: Sampleable,
        val_dataset: Sampleable,
        forward_process: ForwardProcess,
        backward_process: BackwardProcess,
        backbone: Backbone,
        num_classes: int,
        y_drop_prob: float = 0.2,
        num_val_samples: int = 2000,
    ):
        super().__init__(backbone)

        assert 0 < y_drop_prob < 1

        self.dataset = dataset
        self.val_dataset = val_dataset
        self.forward_process = forward_process
        self.backward_process = backward_process
        self.backbone = backbone
        self.num_classes = num_classes
        self.null_class = num_classes
        self.y_drop_prob = y_drop_prob
        self.num_val_samples = num_val_samples

    def get_train_loss(self, batch_size: int, device: torch.device) -> Tensor:
        # Step 1: Sample x, y from p_data
        batch_x, batch_y = self.dataset.sample(batch_size)
        assert batch_y is not None
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)

        # Step 2: Set each label to null class with probability eta
        mask = torch.rand(batch_size, device=device) < self.y_drop_prob
        batch_y[mask] = self.null_class

        # Compute DDPM loss: E_t[ || ε - ε_θ(x_t, t, y) ||² ]
        t = torch.randint(
            low=0,
            high=self.forward_process.timesteps,
            size=(batch_size,),
            device=batch_x.device,
        )

        # Diffuse x_0 to x_t
        noise = torch.randn_like(batch_x)
        x_t = self.forward_process.q_sample(batch_x, t, noise)

        # Predict noise and compute MSE
        eps_pred = self.backbone(x_t, t, batch_y)
        loss = F.mse_loss(eps_pred, noise)

        return loss

    @torch.no_grad()
    def get_val_metrics(self, device: torch.device) -> Dict[str, float]:
        # Step 1: Sample x, y from p_data
        batch_x, batch_y = self.val_dataset.sample(self.num_val_samples)
        assert batch_y is not None
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)

        # Step 2: Set each label to null class with probability eta
        mask = torch.rand(self.num_val_samples, device=device) < self.y_drop_prob
        batch_y[mask] = self.null_class

        # Compute DDPM loss: E_t[ || ε - ε_θ(x_t, t, y) ||² ]
        t = torch.randint(
            low=0,
            high=self.forward_process.timesteps,
            size=(self.num_val_samples,),
            device=batch_x.device,
        )

        # Diffuse x_0 to x_t
        noise = torch.randn_like(batch_x)
        x_t = self.forward_process.q_sample(batch_x, t, noise)

        # Predict noise and compute MSE
        eps_pred = self.backbone(x_t, t, batch_y)
        val_loss = F.mse_loss(eps_pred, noise)

        return {"val_loss": val_loss.item()}
