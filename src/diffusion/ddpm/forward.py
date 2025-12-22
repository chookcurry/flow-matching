import math

import torch
import torch.nn.functional as F
from torch import Tensor

from diffusion.ddpm.utils import extract


def cosine_beta_schedule(timesteps: int, s: float = 0.008) -> Tensor:
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float32)

    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])

    return torch.clip(betas, 0, 0.999).float()


# ------------------------------------------------------------
# 1️⃣ Forward Process — defines q(x_t | x_0) and schedules
# ------------------------------------------------------------


class ForwardProcess:
    def __init__(self, timesteps: int, device: torch.device):
        self.timesteps = timesteps

        # Compute beta schedule
        self.betas = cosine_beta_schedule(timesteps).to(device)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)

        # Precompute useful terms for q_sample
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1 - self.alphas_cumprod)

        # Precompute posterior variance (sigma_t^2)
        self.post_var = (
            self.betas * (1 - self.alphas_cumprod_prev) / (1 - self.alphas_cumprod)
        )

        # Coefficient for x_0 term in posterior mean
        self.posterior_mean_coef1 = (
            torch.sqrt(self.alphas_cumprod_prev) * self.betas
        ) / (1 - self.alphas_cumprod)

        # Coefficient for x_t term in posterior mean
        self.posterior_mean_coef2 = (
            torch.sqrt(self.alphas) * (1 - self.alphas_cumprod_prev)
        ) / (1 - self.alphas_cumprod)

    @torch.no_grad()
    def q_sample(self, x_start: Tensor, t: Tensor, noise: Tensor) -> Tensor:
        sqrt_ac = extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_om_ac = extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)

        # sample x_t = √ᾱ_t * x₀ + √(1-ᾱ_t) * ε
        return sqrt_ac * x_start + sqrt_om_ac * noise
