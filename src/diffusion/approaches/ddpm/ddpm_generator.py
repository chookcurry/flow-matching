from typing import Tuple
import torch
from torch import Tensor
from tqdm import tqdm

from diffusion.approaches.ddpm.backward_process import BackwardProcess, ForwardProcess
from diffusion.architectures.backbone import Backbone
from diffusion.evaluation.generator import Generator


class DDPMGenerator(Generator):
    def __init__(
        self,
        backbone: Backbone,
        device: torch.device,
        timesteps: int,
        null_class: int,
        shape: Tuple[int, ...],
        guidance_scale: float,
    ):
        self.backbone = backbone
        self.device = device
        self.shape = shape
        self.guidance_scale = guidance_scale

        self.forward_process = ForwardProcess(timesteps, device)
        self.backward_process = BackwardProcess(
            backbone, self.forward_process, null_class
        )

    def generate(self, y: Tensor, x0: Tensor | None = None) -> Tensor:
        x = (
            torch.randn((y.shape[0], *self.shape), device=y.device)
            if x0 is None
            else x0
        )

        for step in tqdm(reversed(range(self.backward_process.timesteps))):
            t = torch.full((y.shape[0],), step, device=y.device)
            x = self.backward_process.p_sample(x, t, y, self.guidance_scale)

        return x
