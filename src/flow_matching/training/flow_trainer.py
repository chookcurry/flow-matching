from typing import Any, Dict, List

import numpy as np
import torch
from flow_matching.evaluation.f1 import f1_score, precision_recall_knn
from flow_matching.evaluation.kid import kernel_inception_distance_poly_biased
from flow_matching.supervised.odes_sdes import GuidedNeuralODE, Backbone
from flow_matching.supervised.prob_paths import CondProbPath
from flow_matching.supervised.simulators import RK4Simulator
from flow_matching.training.training import Trainer, sample_time_uniform


class FlowTrainer(Trainer):
    def __init__(
        self,
        path: CondProbPath,
        model: Backbone,
        eta: float,
        null_class: int,
        num_classes: int,
        guidance_scale: float = 1.0,
        num_timesteps: int = 10,
        num_samples: int = 40,
    ):
        super().__init__(model)

        assert 0 < eta < 1

        self.path = path
        self.eta = eta
        self.null_class = null_class
        self.num_classes = num_classes
        self.guidance_scale = guidance_scale
        self.num_timesteps = num_timesteps
        self.num_samples = num_samples

        self.sample_time = sample_time_uniform

    def get_train_loss(self, batch_size: int, device: torch.device) -> torch.Tensor:
        # Step 1: Sample x, y from p_data
        batch_x, batch_y = self.path.p_data.sample(batch_size)
        assert batch_y is not None

        batch_x, batch_y = batch_x.to(device), batch_y.to(device)

        # Step 2: Set each label to null class with probability eta
        mask = torch.rand(batch_size, device=device) < self.eta
        batch_y[mask] = self.null_class

        # Step 3: Sample t and conditional path
        batch_t = self.sample_time(batch_size).to(device)
        batch_xt = self.path.sample_cond_path(batch_x, batch_t)

        # Step 4: Regress and output loss
        pred = self.model(batch_xt, batch_t, batch_y)
        ref = self.path.cond_vf(batch_xt, batch_x, batch_t)

        return torch.mean((pred - ref) ** 2)

    @torch.no_grad()
    def get_val_metrics(self, device: torch.device) -> Any:
        ode = GuidedNeuralODE(
            self.model, scale=self.guidance_scale, null_class=self.null_class
        )

        simulator = RK4Simulator(ode)

        ts = (
            torch.linspace(0, 1, self.num_timesteps)
            .view(1, -1, 1, 1, 1)
            .expand(self.num_samples, -1, 1, 1, 1)
            .to(device)
        )

        metrics_lists: Dict[str, List[float]] = {}

        for label in range(self.num_classes):
            batch_x, batch_y = self.path.p_data.sample(self.num_samples, label)
            assert batch_y is not None

            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            batch_x0, _ = self.path.p_simple.sample(self.num_samples)
            batch_x0 = batch_x0.to(device)
            batch_x1 = simulator.simulate(batch_x0, ts, batch_y)

            kid = kernel_inception_distance_poly_biased(batch_x1, batch_x)
            precision, recall = precision_recall_knn(batch_x1, batch_x)
            f1 = f1_score(precision, recall)

            metrics_lists.setdefault("kid", []).append(kid.item())
            metrics_lists.setdefault("precision", []).append(precision.item())
            metrics_lists.setdefault("recall", []).append(recall.item())
            metrics_lists.setdefault("f1", []).append(f1.item())

        metrics = {k: np.mean(v) for k, v in metrics_lists.items()}

        return metrics
