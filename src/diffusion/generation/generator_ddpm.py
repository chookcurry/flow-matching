from typing import Tuple

import torch
from torch import Tensor
from tqdm import tqdm

from diffusion.backbones.backbone import Backbone
from diffusion.ddpm.backward import BackwardProcess
from diffusion.ddpm.forward import ForwardProcess
from diffusion.generation.generator import Generator


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
            backbone, self.forward_process, null_class, shape, device
        )

    def sample_prior(
        self,
        num_samples: int,
        y: Tensor | None = None,
    ) -> Tensor:
        return torch.randn((num_samples, *self.shape), device=self.device)

    def generate(
        self, y: Tensor, x0: Tensor | None = None, guidance_scale: float | None = None
    ) -> Tensor:
        self.backbone.eval()

        x = (
            torch.randn((y.shape[0], *self.shape), device=y.device)
            if x0 is None
            else x0
        )

        for step in tqdm(reversed(range(self.backward_process.timesteps))):
            t = torch.full((y.shape[0],), step, device=y.device)
            x = self.backward_process.p_sample(x, t, y, guidance_scale)

        return x
