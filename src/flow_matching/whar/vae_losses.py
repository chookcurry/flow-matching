from torch import Tensor
import torch
import torch.nn.functional as F

from flow_matching.whar.ae_losses import get_real_imag


def vae_mse(
    recon_x: Tensor,
    x: Tensor,
    mu: Tensor,
    logvar: Tensor,
    beta: float = 0.1,  # 0.1
) -> Tensor:
    recon_real, recon_imag = get_real_imag(recon_x)
    x_real, x_imag = get_real_imag(x)

    # Reconstruction loss (MSE on real/imag)
    mse_real = F.mse_loss(recon_real, x_real, reduction="mean")
    mse_imag = F.mse_loss(recon_imag, x_imag, reduction="mean")

    kl_div = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

    loss = mse_real + mse_imag + beta * kl_div

    return loss
