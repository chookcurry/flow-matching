from typing import Callable

import torch
from torch import Tensor
from flow_matching.latent.vae import VAE
from flow_matching.supervised.odes_sdes import ConditionalVectorField
from flow_matching.supervised.prob_paths import ConditionalProbabilityPath
from flow_matching.supervised.training import Trainer, sample_time_logit_normal


class LatentFlowTrainer(Trainer):
    def __init__(
        self,
        path: ConditionalProbabilityPath,
        model: ConditionalVectorField,
        vae: VAE,
        eta: float,
        null_class: int,
        sample_time: Callable[[int], Tensor] = sample_time_logit_normal,
    ):
        super().__init__(model)

        assert eta > 0 and eta < 1

        self.path = path
        self.vae = vae
        self.eta = eta
        self.null_class = null_class
        self.sample_time = sample_time

        # freeze vae
        for param in self.vae.parameters():
            param.requires_grad = False

    def get_train_loss(self, batch_size: int) -> torch.Tensor:
        # Step 1: Sample z, y from p_data
        batch_z, batch_y = self.path.p_data.sample(batch_size)
        assert batch_y is not None

        # encode z to latent space
        with torch.no_grad():
            batch_z, _, _ = self.vae.encode(batch_z)

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
