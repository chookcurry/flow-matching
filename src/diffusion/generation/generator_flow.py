import torch
from torch import Tensor

from diffusion.backbones.backbone import Backbone
from diffusion.flows.odes_sdes import GuidedNeuralODE
from diffusion.flows.prob_paths import CondProbPath
from diffusion.flows.simulators import EulerSimulator
from diffusion.generation.generator import Generator


class FlowGenerator(Generator):
    def __init__(
        self,
        path: CondProbPath,
        backbone: Backbone,
        num_timesteps: int,
        null_class: int,
        device: torch.device,
    ) -> None:
        self.path = path
        self.backbone = backbone
        self.num_timesteps = num_timesteps
        self.device = device

        self.ode = GuidedNeuralODE(backbone, null_class)
        self.simulator = EulerSimulator(self.ode)

    def sample_prior(self, num_samples: int, y: Tensor | None = None) -> Tensor:
        x0, _ = self.path.p_simple.sample(num_samples, y)
        return x0

    def generate(
        self, y: Tensor, x0: Tensor | None = None, guidance_scale: float | None = None
    ) -> Tensor:
        self.backbone.eval()

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
