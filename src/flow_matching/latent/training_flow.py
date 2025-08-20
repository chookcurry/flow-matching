from typing import Any, Callable, List

import torch
from torch import Tensor
from flow_matching.evaluation.f1 import f1_score, precision_recall_knn
from flow_matching.evaluation.kid import kernel_inception_distance_polynomial_biased
from flow_matching.latent.ae import CondAutoencoder
from flow_matching.supervised.odes_sdes import CFGVectorFieldODE, ConditionalVectorField
from flow_matching.supervised.prob_paths import ConditionalProbabilityPath
from flow_matching.supervised.simulators import EulerSimulator, RK4Simulator
from flow_matching.supervised.training import (
    Trainer,
    sample_time_uniform,
    sample_time_logit_normal,
)


class LatentFlowTrainer(Trainer):
    def __init__(
        self,
        path: ConditionalProbabilityPath,
        model: ConditionalVectorField,
        ae: CondAutoencoder,
        eta: float,
        null_class: int,
        num_classes: int,
        sample_time: Callable[[int], Tensor] = sample_time_logit_normal,
        track: bool = False,
    ):
        super().__init__(model, track)

        assert eta > 0 and eta < 1

        self.path = path
        self.ae = ae
        self.eta = eta
        self.null_class = null_class
        self.num_classes = num_classes
        self.sample_time = sample_time

        # freeze autoencoder
        for param in self.ae.parameters():
            param.requires_grad = False

    def get_train_loss(self, batch_size: int, device: torch.device) -> torch.Tensor:
        # Step 1: Sample z, y from p_data
        batch_z, batch_y = self.path.p_data.sample(batch_size)
        assert batch_y is not None

        batch_z, batch_y = batch_z.to(device), batch_y.to(device)

        # encode z to latent space
        with torch.no_grad():
            batch_z = self.ae.encode(batch_z, batch_y)

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

    @torch.no_grad()
    def get_val_metrics(
        self,
        device: torch.device,
        guidance_scale: float = 2.0,
        num_timesteps: int = 100,
        num_samples: int = 40,
    ) -> Any:
        ode = CFGVectorFieldODE(
            self.model, guidance_scale=guidance_scale, null_class=self.null_class
        )

        simulator = RK4Simulator(ode)

        ts = (
            torch.linspace(0, 1, num_timesteps)
            .view(1, -1, 1, 1, 1)
            .expand(num_samples, -1, 1, 1, 1)
            .to(device)
        )

        kids = []
        precisions = []
        recalls = []
        f1s = []

        for label in range(self.num_classes):
            batch_z, batch_y = self.path.p_data.sample(num_samples, label)
            assert batch_y is not None

            batch_z, batch_y = batch_z.to(device), batch_y.to(device)

            # encode z to latent space
            batch_z = self.ae.encode(batch_z, batch_y)

            batch_x0, _ = self.path.p_simple.sample(num_samples)
            batch_x0 = batch_x0.to(device)

            assert batch_x0.shape == batch_z.shape, (
                f"{batch_x0.shape} != {batch_z.shape}"
            )

            batch_x1 = simulator.simulate(batch_x0, ts, batch_y)

            kid = kernel_inception_distance_polynomial_biased(batch_x1, batch_z)
            precision, recall = precision_recall_knn(batch_x1, batch_z)
            f1 = f1_score(precision, recall)

            kids.append(kid)
            precisions.append(precision)
            recalls.append(recall)
            f1s.append(f1)

        kid = torch.stack(kids).mean()
        precision = torch.stack(precisions).mean()
        recall = torch.stack(recalls).mean()
        f1 = torch.stack(f1s).mean()

        metrics = {
            "kid": kid.item(),
            "precision": precision.item(),
            "recall": recall.item(),
            "f1": f1.item(),
        }

        return metrics
