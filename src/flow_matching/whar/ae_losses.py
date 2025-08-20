from typing import Tuple
from torch import Tensor
import torch
import torch.nn.functional as F


def get_real_imag(x: Tensor) -> Tuple[Tensor, Tensor]:
    B, C, H, W = x.shape

    # Reshape to [B, C//2, 2, H, W] to split real/imag
    x = x.view(B, C // 2, 2, H, W)

    # Real and imaginary parts
    real = x[:, :, 0]
    imag = x[:, :, 1]

    return real, imag


def ae_mse(recon_x: Tensor, x: Tensor) -> Tensor:
    recon_real, recon_imag = get_real_imag(recon_x)
    x_real, x_imag = get_real_imag(x)

    # Reconstruction loss (MSE on real/imag)
    mse_real = F.mse_loss(recon_real, x_real, reduction="mean")
    mse_imag = F.mse_loss(recon_imag, x_imag, reduction="mean")

    return mse_real + mse_imag


def ae_log_mag(recon_x: Tensor, x: Tensor) -> Tensor:
    recon_real, recon_imag = get_real_imag(recon_x)
    x_real, x_imag = get_real_imag(x)

    eps = 1e-8

    # Compute magnitude, add epsilon for numerical stability
    recon_power = recon_real**2 + recon_imag**2 + eps
    x_power = x_real**2 + x_imag**2 + eps

    log_mag = F.l1_loss(
        0.5 * torch.log(recon_power + eps),
        0.5 * torch.log(x_power + eps),
        reduction="mean",
    )

    return log_mag


def ae_log_mag_phase(recon_x: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    recon_real, recon_imag = get_real_imag(recon_x)
    x_real, x_imag = get_real_imag(x)

    eps = 1e-8

    # Compute magnitude, add epsilon for numerical stability
    recon_power = recon_real**2 + recon_imag**2 + eps
    x_power = x_real**2 + x_imag**2 + eps

    log_mag = F.l1_loss(
        0.5 * torch.log(recon_power + eps),
        0.5 * torch.log(x_power + eps),
        reduction="mean",
    )

    # Compute phase
    recon_phase = torch.atan2(recon_imag, recon_real)
    x_phase = torch.atan2(x_imag, x_real)

    # Phase loss (cosine similarity)
    phase_dist = torch.mean(1 - torch.cos(recon_phase - x_phase))

    # Normalize by total elements
    return log_mag + phase_dist


def phase_loss_weighted(
    recon_x: torch.Tensor, x: torch.Tensor, eps=1e-8, delta=1e-8
) -> torch.Tensor:
    # Separate real and imaginary parts
    recon_real, recon_imag = get_real_imag(recon_x)
    x_real, x_imag = get_real_imag(x)

    # Magnitude
    x_mag = torch.sqrt(x_real**2 + x_imag**2 + eps)
    recon_phase = torch.atan2(recon_imag, recon_real)
    x_phase = torch.atan2(x_imag, x_real)

    # Phase difference cosine distance
    phase_cos = torch.cos(recon_phase - x_phase)
    phase_term = 1.0 - torch.clamp(phase_cos, -1.0, 1.0)

    # Weight by magnitude of target
    weight = x_mag / (x_mag.mean() + delta)
    weighted_phase_loss = (weight * phase_term).mean()

    return weighted_phase_loss


def ae_spect_conv(recon_x: Tensor, x: Tensor) -> Tensor:
    recon_real, recon_imag = get_real_imag(recon_x)
    x_real, x_imag = get_real_imag(x)

    B = recon_x.shape[0]

    # Compute magnitude, add epsilon for numerical stability
    recon_mag = torch.sqrt(recon_real**2 + recon_imag**2 + 1e-8)
    x_mag = torch.sqrt(x_real**2 + x_imag**2 + 1e-8)

    # Spectral convergence per sample
    # Flatten C,H,W to compute per-sample L2 norms
    diff_norm = torch.norm((recon_mag - x_mag).reshape(B, -1), dim=1)
    ref_norm = torch.norm(x_mag.reshape(B, -1), dim=1) + 1e-8
    sc_per_sample = diff_norm / ref_norm

    return sc_per_sample.mean()
