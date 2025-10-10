import math
import torch
import torch.nn.functional as F
from torch import Tensor
from diffusion.architectures.backbone import Backbone


def cosine_beta_schedule(timesteps: int, s: float = 0.008) -> Tensor:
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float32)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999).float()


def extract(a: Tensor, t: Tensor, x_shape: tuple) -> Tensor:
    out = a.cpu().gather(0, t.cpu().long()).to(a.device)
    return out.view((t.shape[0],) + (1,) * (len(x_shape) - 1))


# ------------------------------------------------------------
# 1️⃣ Forward Process — defines q(x_t | x_0) and schedules
# ------------------------------------------------------------
# sample x_t = √ᾱ_t * x₀ + √(1-ᾱ_t) * ε
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
        return sqrt_ac * x_start + sqrt_om_ac * noise
        # return (
        #     extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        #     + extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        # )


# ------------------------------------------------------------
# 2️⃣ Backward Process — defines p(x_{t-1} | x_t)
# ------------------------------------------------------------
class BackwardProcess:
    def __init__(self, backbone: Backbone, forward: ForwardProcess, null_class: int):
        self.backbone = backbone
        self.forward = forward
        self.timesteps = forward.timesteps
        self.null_class = null_class

    def p_sample(
        self, x_t: Tensor, t: Tensor, y: Tensor, guidance_scale: float = 1.0
    ) -> Tensor:
        b = x_t.shape[0]

        # Pre-extract values once for reuse
        sqrt_ac = extract(self.forward.sqrt_alphas_cumprod, t, x_t.shape)
        sqrt_om_ac = extract(self.forward.sqrt_one_minus_alphas_cumprod, t, x_t.shape)
        post_var = extract(self.forward.post_var, t, x_t.shape)
        coef1 = extract(self.forward.posterior_mean_coef1, t, x_t.shape)
        coef2 = extract(self.forward.posterior_mean_coef2, t, x_t.shape)

        # ------------------------------------------------
        # Classifier-free guidance: compute once efficiently
        # ------------------------------------------------
        if guidance_scale == 1.0:
            eps_pred = self.backbone(x_t, t, y)
        else:
            if y.dtype in (torch.int64, torch.int32):
                uncond = torch.full_like(y, self.null_class)
            else:
                uncond = torch.zeros_like(y)

            # Concatenate once, not multiple times
            x_in = torch.cat([x_t, x_t], dim=0)
            t_in = torch.cat([t, t], dim=0)
            cond_in = torch.cat([uncond, y], dim=0)

            eps_all = self.backbone(x_in, t_in, cond_in)
            eps_uncond, eps_cond = eps_all.chunk(2, dim=0)
            eps_pred = eps_uncond + guidance_scale * (eps_cond - eps_uncond)

        # ------------------------------------------------
        # Predict x₀ efficiently
        # ------------------------------------------------
        x0_pred = (x_t - sqrt_om_ac * eps_pred) / sqrt_ac
        x0_pred.clamp_(-1.0, 1.0)  # in-place clamp for efficiency

        # ------------------------------------------------
        # Compute posterior mean (DDPM)
        # ------------------------------------------------
        posterior_mean = coef1 * x0_pred + coef2 * x_t

        # ------------------------------------------------
        # Add noise if t != 0
        # ------------------------------------------------
        noise = torch.randn_like(x_t)
        nonzero_mask = (t != 0).view(b, *([1] * (x_t.ndim - 1)))
        return posterior_mean + nonzero_mask * torch.sqrt(post_var) * noise
