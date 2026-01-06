import torch
from torch import Tensor

from diffusion.backbones.backbone import Backbone
from diffusion.flows.prob_paths import CondProbPath
from diffusion.generation.generator import Generator


class FlowMapGenerator(Generator):
    def __init__(
        self,
        path: CondProbPath,
        backbone: Backbone,
        num_timesteps: int,  # Number of steps K [cite: 136, 234]
        null_class: int,  # Label used for unconditional generation
        device: torch.device,
    ) -> None:
        self.path = path
        self.backbone = backbone
        self.num_timesteps = num_timesteps
        self.null_class = null_class
        self.device = device

    def sample_prior(self, num_samples: int, y: Tensor | None = None) -> Tensor:
        # Sample from the base distribution (typically Gaussian) [cite: 110, 131]
        x0, _ = self.path.p_simple.sample(num_samples, y)
        return x0

    def generate(
        self,
        y: Tensor,
        x0: Tensor | None = None,
        guidance_scale: float | None = None,
    ) -> Tensor:
        self.backbone.eval()
        num_samples = y.shape[0]

        if x0 is None:
            x0, _ = self.path.p_simple.sample(num_samples)
            x0 = x0.to(self.device)

        # Create time steps
        ts = torch.linspace(0, 1, self.num_timesteps + 1).to(self.device)
        y_null = torch.full_like(y, self.null_class)

        xt = x0

        with torch.no_grad():
            for i in range(self.num_timesteps):
                # Broadcast s and t to batch size
                s = ts[i].view(1, 1, 1, 1).expand(num_samples, -1, -1, -1)
                t = ts[i + 1].view(1, 1, 1, 1).expand(num_samples, -1, -1, -1)

                # Calculate the step size (t - s)
                dt = t - s

                if guidance_scale is None or guidance_scale == 1.0:
                    # Model predicts 'v', we calculate x + dt * v
                    v_pred = self.backbone(xt, s, t, y)
                    xt = xt + dt * v_pred
                else:
                    # CFG on the velocity field v, NOT the position
                    v_cond = self.backbone(xt, s, t, y)
                    v_uncond = self.backbone(xt, s, t, y_null)

                    # Extrapolate velocity
                    v_cfg = v_uncond + guidance_scale * (v_cond - v_uncond)

                    # Apply step
                    xt = xt + dt * v_cfg

        return xt
