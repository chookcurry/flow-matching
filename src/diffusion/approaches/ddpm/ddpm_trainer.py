from typing import Dict

import torch
from torch import Tensor
import torch.nn.functional as F

from diffusion.architectures.backbone import Backbone
from diffusion.training.trainer import Trainer
from diffusion.evaluation.f1 import f1_score, precision_recall_knn
from diffusion.evaluation.kid import kernel_inception_distance_poly
from diffusion.approaches.ddpm.backward_process import BackwardProcess, ForwardProcess
from diffusion.data.sampleables import Sampleable


class DDPMTrainer(Trainer):
    def __init__(
        self,
        dataset: Sampleable,
        backbone: Backbone,
        num_classes: int,
        timesteps: int = 30,
        y_drop_prob: float = 0.1,
        guidance_scale: float = 3.0,
        num_samples: int = 1000,
    ):
        super().__init__(backbone)

        assert 0 < y_drop_prob < 1

        self.dataset = dataset
        self.backbone = backbone
        self.num_classes = num_classes
        self.null_class = num_classes
        self.guidance_scale = guidance_scale
        self.y_drop_prob = y_drop_prob
        self.num_samples = num_samples

        self.device = next(backbone.parameters()).device
        self.forward_process = ForwardProcess(timesteps, self.device)
        self.backward_process = BackwardProcess(
            backbone, self.forward_process, self.null_class
        )

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
        return F.mse_loss(eps_pred, noise)

    @torch.no_grad()
    def get_val_metrics(self, device: torch.device) -> Dict[str, float]:
        # # Sample all data and conditions at once
        # x, y = self.dataset.sample(self.num_samples)
        # assert y is not None
        # x, y = x.to(device), y.to(device)

        # # Generate samples with guidance
        # samples = self.backward_process.sample(
        #     batch_size=self.num_samples,
        #     shape=x.shape[1:],
        #     y=y,
        #     guidance_scale=self.guidance_scale,
        # )

        # # Compute evaluation metrics
        # kid = kernel_inception_distance_poly(samples, x)
        # precision, recall = precision_recall_knn(samples, x)
        # f1 = f1_score(precision, recall)

        # return {
        #     "kid": kid.item(),
        #     "precision": precision.item(),
        #     "recall": recall.item(),
        #     "f1": f1.item(),
        # }
