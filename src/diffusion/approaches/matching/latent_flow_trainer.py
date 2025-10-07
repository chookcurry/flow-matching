from typing import Any, Callable, Dict, List

import numpy as np
import torch
from torch import Tensor
from diffusion.evaluation.f1 import f1_score, precision_recall_knn
from diffusion.evaluation.kid import kernel_inception_distance_poly_biased
from diffusion.architectures.latent.autoencoder import AE, AEC, CAE, CAEC
from diffusion.approaches.matching.odes_sdes import GuidedNeuralODE, Backbone
from diffusion.approaches.matching.prob_paths import CondProbPath
from diffusion.approaches.matching.simulators import RK4Simulator
from diffusion.training.trainer import Trainer, sample_time_uniform


class LatentFlowTrainer(Trainer):
    def __init__(
        self,
        path: CondProbPath,
        model: Backbone,
        ae: AE | CAE | AEC | CAEC,
        eta: float,
        null_class: int,
        num_classes: int,
        guidance_scale: float = 2.0,
        num_timesteps: int = 100,
        num_samples: int = 40,
        sample_time: Callable[[int], Tensor] = sample_time_uniform,
    ):
        super().__init__(model)

        assert eta > 0 and eta < 1

        self.path = path
        self.ae = ae
        self.eta = eta
        self.null_class = null_class
        self.num_classes = num_classes
        self.guidance_scale = guidance_scale
        self.num_timesteps = num_timesteps
        self.num_samples = num_samples
        self.sample_time = sample_time

        for param in self.ae.parameters():
            param.requires_grad = False

    def get_train_loss(self, batch_size: int, device: torch.device) -> torch.Tensor:
        # Step 1: Sample z, y from p_data
        batch_z, batch_y = self.path.p_data.sample(batch_size)
        assert batch_y is not None

        batch_z, batch_y = batch_z.to(device), batch_y.to(device)

        # encode z to latent space
        with torch.no_grad():
            batch_z = (
                self.ae.encode(batch_z)
                if isinstance(self.ae, (AE, AEC))
                else self.ae.encode(batch_z, batch_y)
            )

        # Step 2: Set each label to null class with probability eta
        mask = torch.rand(batch_size) < self.eta
        batch_y[mask] = self.null_class

        # Step 3: Sample t and x
        batch_t = self.sample_time(batch_size).to(device)
        batch_x = self.path.sample_cond_path(batch_z, batch_t)

        # Step 4: Regress and output loss
        pred = self.model(batch_x, batch_t, batch_y)
        ref = self.path.cond_vf(batch_x, batch_z, batch_t)

        return torch.mean((pred - ref) ** 2)

    @torch.no_grad()
    def get_val_metrics(
        self,
        device: torch.device,
    ) -> Any:
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
            batch_z, batch_y = self.path.p_data.sample(self.num_samples, label)
            assert batch_y is not None

            batch_z, batch_y = batch_z.to(device), batch_y.to(device)

            batch_z = (
                self.ae.encode(batch_z)
                if isinstance(self.ae, (AE, AEC))
                else self.ae.encode(batch_z, batch_y)
            )

            batch_x0, _ = self.path.p_simple.sample(self.num_samples)
            batch_x0 = batch_x0.to(device)
            batch_x1 = simulator.simulate(batch_x0, ts, batch_y)

            kid = kernel_inception_distance_poly_biased(batch_x1, batch_z)
            precision, recall = precision_recall_knn(batch_x1, batch_z)
            f1 = f1_score(precision, recall)

            metrics_lists["kid"].append(kid.item())
            metrics_lists["precision"].append(precision.item())
            metrics_lists["recall"].append(recall.item())
            metrics_lists["f1"].append(f1.item())

        metrics = {k: np.mean(v) for k, v in metrics_lists.items()}

        return metrics
