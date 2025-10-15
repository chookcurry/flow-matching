from typing import Callable, Dict

import torch
from torch import Tensor
from diffusion.architectures.backbones.backbone import Backbone
from diffusion.approaches.matching.prob_paths import CondProbPath
from diffusion.training.trainer import Trainer


def sample_time_uniform(batch_size: int) -> Tensor:
    return torch.rand(batch_size, 1, 1, 1)


def sample_time_logit_normal(batch_size: int) -> Tensor:
    return torch.sigmoid(torch.normal(0.0, 0.6, size=(batch_size, 1, 1, 1)))


class FlowTrainer(Trainer):
    def __init__(
        self,
        path: CondProbPath,
        val_path: CondProbPath,
        backbone: Backbone,
        null_class: int,
        y_drop_prob: float = 0.2,
        sample_time: Callable[[int], Tensor] = sample_time_uniform,
    ):
        super().__init__(backbone)

        assert 0 < y_drop_prob < 1

        self.path = path
        self.val_path = val_path
        self.backbone = backbone
        self.null_class = null_class
        self.y_drop_prob = y_drop_prob
        self.sample_time = sample_time

    def _get_loss(
        self, path: CondProbPath, batch_size: int, device: torch.device
    ) -> Tensor:
        # Step 1: Sample x, y from p_data
        batch_x, batch_y = path.p_data.sample(batch_size)
        assert batch_y is not None
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)

        # Step 2: Set each label to null class with probability eta
        mask = torch.rand(batch_size, device=device) < self.y_drop_prob
        batch_y[mask] = self.null_class

        # Step 3: Sample t and conditional path
        batch_t = self.sample_time(batch_size).to(device)
        batch_xt = path.sample_cond_path(batch_x, batch_t, batch_y)

        # Step 4: Regress and output loss
        pred = self.model(batch_xt, batch_t, batch_y)
        ref = path.cond_vf(batch_xt, batch_x, batch_t)
        loss = torch.mean((pred - ref) ** 2)

        return loss

    def get_train_loss(self, batch_size: int, device: torch.device) -> Tensor:
        return self._get_loss(self.path, batch_size, device)

    @torch.no_grad()
    def get_val_loss(self, batch_size: int, device: torch.device) -> Tensor:
        return self._get_loss(self.val_path, batch_size, device)
