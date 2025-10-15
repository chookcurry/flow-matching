from typing import Tuple
import torch
from torch import Tensor
from tqdm import tqdm

from diffusion.approaches.ddpm.backward_process import BackwardProcess, ForwardProcess
from diffusion.architectures.backbones.backbone import Backbone
from diffusion.evaluation.generator import Generator


class DDPMGenerator(Generator):
    def __init__(
        self,
        backbone: Backbone,
        device: torch.device,
        timesteps: int,
        null_class: int,
        shape: Tuple[int, ...],
    ):
        self.backbone = backbone
        self.device = device
        self.shape = shape

        self.forward_process = ForwardProcess(timesteps, device)
        self.backward_process = BackwardProcess(
            backbone, self.forward_process, null_class
        )

    def sample_prior(
        self,
        num_samples: int,
        shape: Tuple[int, ...],
        device: torch.device,
        y: Tensor | None = None,
    ) -> Tensor:
        return torch.randn((num_samples, *shape), device=device)

    def generate(
        self, y: Tensor, x0: Tensor | None = None, guidance_scale: float | None = None
    ) -> Tensor:
        x = (
            torch.randn((y.shape[0], *self.shape), device=y.device)
            if x0 is None
            else x0
        )

        for step in tqdm(reversed(range(self.backward_process.timesteps))):
            t = torch.full((y.shape[0],), step, device=y.device)
            x = self.backward_process.p_sample(x, t, y, guidance_scale)

        return x
