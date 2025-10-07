from typing import Dict

import torch
from diffusion.evaluation.f1 import f1_score, precision_recall_knn
from diffusion.evaluation.kid import kernel_inception_distance_poly
from diffusion.approaches.matching.odes_sdes import GuidedNeuralODE, Backbone
from diffusion.approaches.matching.prob_paths import CondProbPath
from diffusion.approaches.matching.simulators import EulerSimulator
from diffusion.training.trainer import Trainer


class FlowTrainer(Trainer):
    def __init__(
        self,
        path: CondProbPath,
        backbone: Backbone,
        num_classes: int,
        y_drop_prob: float = 0.2,
        guidance_scale: float = 3.0,
        num_timesteps: int = 10,
        num_samples: int = 1000,
    ):
        super().__init__(backbone)

        assert 0 < y_drop_prob < 1

        self.path = path
        self.num_classes = num_classes
        self.null_class = num_classes
        self.y_drop_prob = y_drop_prob
        self.guidance_scale = guidance_scale
        self.num_timesteps = num_timesteps
        self.num_samples = num_samples

    def get_train_loss(self, batch_size: int, device: torch.device) -> torch.Tensor:
        # Step 1: Sample x, y from p_data
        batch_x, batch_y = self.path.p_data.sample(batch_size)
        assert batch_y is not None
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)

        # Step 2: Set each label to null class with probability eta
        mask = torch.rand(batch_size, device=device) < self.y_drop_prob
        batch_y[mask] = self.null_class

        # Step 3: Sample t and conditional path
        batch_t = self.sample_time_uniform(batch_size).to(device)
        batch_xt = self.path.sample_cond_path(batch_x, batch_t)

        # Step 4: Regress and output loss
        pred = self.model(batch_xt, batch_t, batch_y)
        ref = self.path.cond_vf(batch_xt, batch_x, batch_t)

        return torch.mean((pred - ref) ** 2)

    @torch.no_grad()
    def get_val_metrics(self, device: torch.device) -> Dict[str, float]:
        ode = GuidedNeuralODE(self.model, self.null_class, self.guidance_scale)
        simulator = EulerSimulator(ode)

        # Time steps shared for all samples
        ts = (
            torch.linspace(0, 1, self.num_timesteps)
            .view(1, -1, 1, 1, 1)
            .expand(self.num_samples, -1, 1, 1, 1)
            .to(device)
        )

        # Sample all data and conditions at once
        x, y = self.path.p_data.sample(self.num_samples)
        assert y is not None
        x, y = x.to(device), y.to(device)

        # Sample simple prior and simulate
        x0, _ = self.path.p_simple.sample(self.num_samples)
        x0 = x0.to(device)
        x1 = simulator.simulate(x0, ts, y)

        # Compute overall metrics directly
        kid = kernel_inception_distance_poly(x1, x)
        precision, recall = precision_recall_knn(x1, x)
        f1 = f1_score(precision, recall)

        metrics = {
            "kid": kid.item(),
            "precision": precision.item(),
            "recall": recall.item(),
            "f1": f1.item(),
        }

        return metrics
