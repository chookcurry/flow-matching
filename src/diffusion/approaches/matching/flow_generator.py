from torch import Tensor
import torch

from diffusion.evaluation.generator import Generator
from diffusion.approaches.matching.alphas_betas import (
    LinearAlpha,
    LinearBeta,
    ScoreFromVectorFieldForGaussianProbPath,
)
from diffusion.approaches.matching.odes_sdes import (
    GuidedNeuralODE,
    Backbone,
    GuidedNeuralSDE,
)
from diffusion.approaches.matching.prob_paths import CondProbPath
from diffusion.approaches.matching.simulators import (
    EulerMaruyamaSimulator,
    EulerSimulator,
)


class FlowGenerator(Generator):
    def __init__(
        self,
        path: CondProbPath,
        backbone: Backbone,
        num_timesteps: int,
        null_class: int,
        guidance_scale: float,
        device: torch.device,
    ) -> None:
        self.path = path
        self.num_timesteps = num_timesteps
        self.device = device

        self.ode = GuidedNeuralODE(backbone, null_class, guidance_scale)
        self.simulator = EulerSimulator(self.ode)

    def generate(self, y: Tensor, x0: Tensor | None = None) -> Tensor:
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

        x1 = self.simulator.simulate(x0, ts, y)

        return x1


class ScoreGenerator(Generator):
    def __init__(
        self,
        path: CondProbPath,
        backbone: Backbone,
        num_timesteps: int,
        null_class: int,
        guidance_scale: float,
        device: torch.device,
    ) -> None:
        self.path = path
        self.num_timesteps = num_timesteps
        self.device = device

        score = ScoreFromVectorFieldForGaussianProbPath(
            backbone, LinearAlpha(), LinearBeta()
        )

        self.sde = GuidedNeuralSDE(backbone, score, null_class, guidance_scale)
        self.simulator = EulerMaruyamaSimulator(self.sde)

    def generate(self, y: Tensor, x0: Tensor | None = None) -> Tensor:
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

        x1 = self.simulator.simulate(x0, ts, y)

        return x1
