from torch import Tensor
import torch
import torch.nn.functional as F


def vae_loss_mse(
    recon_x: Tensor, x: Tensor, mu: Tensor, logvar: Tensor, beta: float = 0
) -> Tensor:
    B, C, H, W = recon_x.shape

    # Reshape to [B, C//2, 2, H, W] to split real/imag
    recon_x = recon_x.view(B, C // 2, 2, H, W)
    x = x.view(B, C // 2, 2, H, W)

    # Real and imaginary parts
    recon_real = recon_x[:, :, 0]
    recon_imag = recon_x[:, :, 1]
    x_real = x[:, :, 0]
    x_imag = x[:, :, 1]

    # Reconstruction loss (MSE on real/imag)
    recon_loss_real = F.mse_loss(recon_real, x_real, reduction="mean")
    recon_loss_imag = F.mse_loss(recon_imag, x_imag, reduction="mean")
    kl_div = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

    loss = recon_loss_real + recon_loss_imag + beta * kl_div

    return loss


def vae_loss_log_mag(recon_x: Tensor, x: Tensor, mu: Tensor, logvar: Tensor) -> Tensor:
    B, C, H, W = recon_x.shape

    # Reshape to [B, C//2, 2, H, W] to split real/imag
    recon_x = recon_x.view(B, C // 2, 2, H, W)
    x = x.view(B, C // 2, 2, H, W)

    # Real and imaginary parts
    recon_real = recon_x[:, :, 0]
    recon_imag = recon_x[:, :, 1]
    x_real = x[:, :, 0]
    x_imag = x[:, :, 1]

    # Compute magnitude, add epsilon for numerical stability
    recon_mag = torch.sqrt(recon_real**2 + recon_imag**2 + 1e-8)
    x_mag = torch.sqrt(x_real**2 + x_imag**2 + 1e-8)

    # Log-magnitude loss
    log_mag_loss = F.l1_loss(
        torch.log1p(recon_mag), torch.log1p(x_mag), reduction="mean"
    )

    kl_div = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

    loss = log_mag_loss + kl_div

    return loss


def vae_loss_log_mag_phase(
    recon_x: torch.Tensor, x: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor
) -> torch.Tensor:
    B, C, H, W = recon_x.shape

    # Reshape to [B, C//2, 2, H, W] to split real/imag
    recon_x = recon_x.view(B, C // 2, 2, H, W)
    x = x.view(B, C // 2, 2, H, W)

    # Real and imaginary parts
    recon_real = recon_x[:, :, 0]
    recon_imag = recon_x[:, :, 1]
    x_real = x[:, :, 0]
    x_imag = x[:, :, 1]

    # Compute magnitude, add epsilon for numerical stability
    recon_mag = torch.sqrt(recon_real**2 + recon_imag**2 + 1e-8)
    x_mag = torch.sqrt(x_real**2 + x_imag**2 + 1e-8)

    # Compute phase
    recon_phase = torch.atan2(recon_imag, recon_real)
    x_phase = torch.atan2(x_imag, x_real)

    # Log-magnitude loss (L1)
    log_mag_loss = F.l1_loss(
        torch.log1p(recon_mag), torch.log1p(x_mag), reduction="mean"
    )

    # Phase loss (cosine similarity)
    phase_diff = recon_phase - x_phase
    phase_loss = torch.mean(1 - torch.cos(phase_diff))

    # KL divergence term
    kl_div = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

    # Normalize by total elements
    loss = (log_mag_loss + phase_loss) + kl_div

    return loss


def vae_loss_spect_conv(
    recon_x: Tensor, x: Tensor, mu: Tensor, logvar: Tensor
) -> Tensor:
    B, C, H, W = recon_x.shape

    # Reshape to [B, C//2, 2, H, W] to split real/imag
    recon_x = recon_x.view(B, C // 2, 2, H, W)
    x = x.view(B, C // 2, 2, H, W)

    # Real and imaginary parts
    recon_real = recon_x[:, :, 0]
    recon_imag = recon_x[:, :, 1]
    x_real = x[:, :, 0]
    x_imag = x[:, :, 1]

    # Compute magnitude, add epsilon for numerical stability
    recon_mag = torch.sqrt(recon_real**2 + recon_imag**2 + 1e-8)
    x_mag = torch.sqrt(x_real**2 + x_imag**2 + 1e-8)

    # Spectral convergence per sample
    # Flatten C,H,W to compute per-sample L2 norms
    diff_norm = torch.norm((recon_mag - x_mag).reshape(B, -1), dim=1)
    ref_norm = torch.norm(x_mag.reshape(B, -1), dim=1) + 1e-8
    sc_per_sample = diff_norm / ref_norm
    spect_conv = sc_per_sample.mean()

    kl_div = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

    loss = spect_conv + kl_div

    return loss
