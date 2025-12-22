from typing import Tuple

import torch
from torch import Tensor

from diffusion.backbones.backbone import Backbone
from diffusion.flows.odes_sdes import GuidedNeuralSDE
from diffusion.flows.prob_paths import (
    CondProbPath,
    ScoreFromVectorFieldForGaussianProbPath,
)
from diffusion.flows.simulators import EulerMaruyamaSimulator
from diffusion.generation.generator import Generator


class ScoreGenerator(Generator):
    def __init__(
        self,
        path: CondProbPath,
        backbone: Backbone,
        num_timesteps: int,
        null_class: int,
        device: torch.device,
    ) -> None:
        self.path = path
        self.num_timesteps = num_timesteps
        self.device = device

        score = ScoreFromVectorFieldForGaussianProbPath(backbone)

        self.sde = GuidedNeuralSDE(backbone, score, null_class)
        self.simulator = EulerMaruyamaSimulator(self.sde)

    def sample_prior(
        self,
        num_samples: int,
        shape: Tuple[int, ...],
        device: torch.device,
        y: Tensor | None = None,
    ) -> Tensor:
        x0, _ = self.path.p_simple.sample(num_samples, y)
        return x0

    def generate(
        self, y: Tensor, x0: Tensor | None = None, guidance_scale: float | None = None
    ) -> Tensor:
        num_samples = y.shape[0]

        if x0 is None:
            x0, _ = self.path.p_simple.sample(num_samples)
            x0 = x0.to(self.device)

        ts = (
            torch.linspace(0, 1, self.num_timesteps)
            .view(1, -1, 1, 1, 1)
            .expand(num_samples, -1, 1, 1, 1)
            .to(self.device)
        )

        x1 = self.simulator.simulate(x0, ts, y, guidance_scale)

        return x1
