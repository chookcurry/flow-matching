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
        guidance_scale: float | None = None,  # w > 1.0 enhances class attributes
    ) -> Tensor:
        num_samples = y.shape[0]

        if x0 is None:
            # Initialize from base distribution [cite: 110, 131]
            x0, _ = self.path.p_simple.sample(num_samples)
            x0 = x0.to(self.device)

        # Discretization points for multistep generation [cite: 136, 233]
        ts = torch.linspace(0, 1, self.num_timesteps + 1).to(self.device)

        # Create null labels for the unconditional pass
        y_null = torch.full_like(y, self.null_class)

        xt = x0

        with torch.no_grad():
            for i in range(self.num_timesteps):
                s = ts[i].view(1, 1, 1, 1).expand(num_samples, -1, -1, -1)
                t = ts[i + 1].view(1, 1, 1, 1).expand(num_samples, -1, -1, -1)

                if guidance_scale is None or guidance_scale == 1.0:
                    # Simple conditional forward pass [cite: 118, 412]
                    xt = self.backbone(xt, s, t, y)
                else:
                    # Classifier-Free Guidance Logic
                    # 1. Conditional prediction X(x, s, t, y)
                    x_cond = self.backbone(xt, s, t, y)

                    # 2. Unconditional prediction X(x, s, t, null)
                    x_uncond = self.backbone(xt, s, t, y_null)

                    # 3. Linear extrapolation [cite: 35, 397, 413]
                    # xt = x_uncond + w * (x_cond - x_uncond)
                    xt = x_uncond + guidance_scale * (x_cond - x_uncond)

        return xt
