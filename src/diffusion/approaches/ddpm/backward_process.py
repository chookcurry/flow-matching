import torch
from torch import Tensor
from diffusion.approaches.ddpm.forward_process import ForwardProcess
from diffusion.approaches.ddpm.utils import extract
from diffusion.architectures.backbones.backbone import Backbone


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
        self, x_t: Tensor, t: Tensor, y: Tensor, guidance_scale: float | None = None
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
        if guidance_scale is None:
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
