import math
import torch
import torch.nn.functional as F
from torch import Tensor
from tqdm import tqdm
from diffusion.architectures.backbone import Backbone


def cosine_beta_schedule(timesteps: int, s: float = 0.008) -> Tensor:
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float32)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999).float()


class DDPM:
    def __init__(
        self,
        backbone: Backbone,
        device: torch.device,
        timesteps: int = 1000,
        unconditional_label: int = -1,
    ):
        self.backbone = backbone
        self.device = device
        self.timesteps = timesteps
        self.unconditional_label = unconditional_label

        self.betas = cosine_beta_schedule(timesteps).to(self.device)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1 - self.alphas_cumprod)
        self.posterior_variance = (
            self.betas * (1 - self.alphas_cumprod_prev) / (1 - self.alphas_cumprod)
        )

    def _extract(self, a: Tensor, t: Tensor, x_shape: tuple) -> Tensor:
        out = a.cpu().gather(0, t.cpu().long()).to(a.device)
        return out.view((t.shape[0],) + (1,) * (len(x_shape) - 1))

    def q_sample(self, x_start: Tensor, t: Tensor, noise: Tensor) -> Tensor:
        return self._extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start.to(
            self.device
        ) + self._extract(
            self.sqrt_one_minus_alphas_cumprod, t, x_start.shape
        ) * noise.to(self.device)

    def training_losses(self, x_start: Tensor, y: Tensor) -> Tensor:
        b = x_start.shape[0]
        t = torch.randint(0, self.timesteps, (b,), device=x_start.device)
        noise = torch.randn_like(x_start, dtype=torch.float32, device=x_start.device)
        x_t = self.q_sample(x_start, t, noise)

        eps_pred = self.backbone(x_t, t, y)
        return F.mse_loss(eps_pred, noise)

    @torch.no_grad()
    def p_sample(
        self, x_t: Tensor, t: Tensor, y: Tensor, guidance_scale: float = 1.0
    ) -> Tensor:
        b = x_t.shape[0]

        if guidance_scale == 1.0:
            eps_pred = self.backbone(x_t, t, y)
        else:
            if y.dtype in (torch.int64, torch.int32):
                uncond = torch.full_like(y, self.unconditional_label)
            else:
                uncond = torch.zeros_like(y)
            x_in = torch.cat([x_t, x_t], dim=0)
            t_in = torch.cat([t, t], dim=0)
            cond_in = torch.cat([uncond, y], dim=0)
            eps_uncond, eps_cond = self.backbone(x_in, t_in, cond_in).chunk(2, 0)
            eps_pred = eps_uncond + guidance_scale * (eps_cond - eps_uncond)

        x0_pred = (
            x_t
            - self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape) * eps_pred
        ) / self._extract(self.sqrt_alphas_cumprod, t, x_t.shape)
        x0_pred = torch.clamp(x0_pred, -1, 1)

        posterior_mean = (
            self._extract(self.betas, t, x_t.shape) * x0_pred
            + self._extract(self.alphas, t, x_t.shape) * x_t
        )

        noise = torch.randn_like(x_t, dtype=torch.float32, device=x_t.device)
        nonzero_mask = (t != 0).float().view((b,) + (1,) * (len(x_t.shape) - 1))

        return (
            posterior_mean
            + nonzero_mask
            * torch.sqrt(self._extract(self.posterior_variance, t, x_t.shape))
            * noise
        )

    @torch.no_grad()
    def sample(
        self, batch_size: int, shape: tuple, y: Tensor, guidance_scale: float = 1.0
    ) -> Tensor:
        x = torch.randn((batch_size, *shape), device=self.device)

        for t_ in tqdm(reversed(range(self.timesteps))):
            t = torch.full((batch_size,), t_, device=self.device, dtype=torch.long)
            x = self.p_sample(x, t, y, guidance_scale)
        return x
