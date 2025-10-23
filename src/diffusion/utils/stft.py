import math
from matplotlib import pyplot as plt
import torch
from torch import Tensor


def stft_transform(x: Tensor, n_fft: int = 63, hop_length: int = 4) -> Tensor:
    # (time, channels)

    stft = torch.stft(
        x.permute(1, 0),
        n_fft=n_fft,
        hop_length=hop_length,
        return_complex=True,
        window=torch.hann_window(n_fft).to(x.device),
    )  # (channels, freq_bins, time_steps)

    spect = torch.stack([stft.real, stft.imag], dim=1)
    # (channels, 2, freq_bins, time_steps)

    return spect


def istft_transform(
    stft_separated: Tensor, n_fft: int = 63, hop_length: int = 4, length: int = 128
) -> Tensor:
    # (channels, 2, freq_bins, time_steps)

    real = stft_separated[:, 0]
    imag = stft_separated[:, 1]
    stft_complex = torch.complex(real, imag)
    # (channels, freq_bins, time_steps)

    waveform = torch.istft(
        stft_complex,
        n_fft=n_fft,
        hop_length=hop_length,
        window=torch.hann_window(n_fft).to(stft_separated.device),
        length=length,
    )  # (channels, time)

    return waveform.permute(1, 0)  # (time, channels)


def compress_stft(
    stft_separated: Tensor, alpha: float = 0.6, beta: float = 1.0
) -> Tensor:
    # Apply amplitude transformation to STFT coefficients:
    # c_tilde = beta * |c|^alpha * exp(i * angle(c))

    # (channels, 2, freq_bins, time_steps)
    real = stft_separated[:, 0]
    imag = stft_separated[:, 1]
    c = torch.complex(real, imag)

    magnitude = torch.abs(c)
    phase = torch.angle(c)

    # Compress magnitude
    magnitude_compressed = magnitude**alpha

    # Normalize to [0, 1] by dividing by max, then scale by beta
    max_val = magnitude_compressed.max()
    if max_val > 0:
        magnitude_compressed = (magnitude_compressed / max_val) * beta

    # Rebuild complex coefficients
    c_tilde = magnitude_compressed * torch.exp(1j * phase)

    # Separate real and imag parts again
    real_tilde = c_tilde.real
    imag_tilde = c_tilde.imag

    return torch.stack([real_tilde, imag_tilde], dim=1)


def decompress_stft(
    stft_compressed: Tensor, alpha: float = 0.6, beta: float = 1.0
) -> Tensor:
    # Inverse amplitude transformation to recover original STFT coefficients:
    # c = (|c_tilde| / beta)^(1/alpha) * exp(i * angle(c_tilde))

    # (channels, 2, freq_bins, time_steps)

    real = stft_compressed[:, 0]
    imag = stft_compressed[:, 1]
    c_tilde = torch.complex(real, imag)

    magnitude_tilde = torch.abs(c_tilde)
    phase_tilde = torch.angle(c_tilde)

    # Avoid division by zero
    magnitude = (magnitude_tilde / beta).clamp(min=1e-8) ** (1 / alpha)

    c = magnitude * torch.exp(1j * phase_tilde)

    real_orig = c.real
    imag_orig = c.imag

    return torch.stack([real_orig, imag_orig], dim=1)


def compress_stft_log(stft_separated: Tensor) -> Tensor:
    """
    Fully invertible log-based STFT compression.
    No HF damping here.
    """
    real = stft_separated[:, 0]
    imag = stft_separated[:, 1]
    c = torch.complex(real, imag)

    magnitude = torch.abs(c)
    phase = torch.angle(c)

    # Log compression
    magnitude_compressed = torch.log1p(magnitude)

    # Rebuild complex coefficients
    c_tilde = magnitude_compressed * torch.exp(1j * phase)
    return torch.stack([c_tilde.real, c_tilde.imag], dim=1)


def decompress_stft_log(stft_compressed: Tensor) -> Tensor:
    """
    Inverse of compress_stft_invertible
    """
    real = stft_compressed[:, 0]
    imag = stft_compressed[:, 1]
    c_tilde = torch.complex(real, imag)

    magnitude_tilde = torch.abs(c_tilde)
    phase_tilde = torch.angle(c_tilde)

    magnitude = torch.expm1(magnitude_tilde)

    c = magnitude * torch.exp(1j * phase_tilde)
    return torch.stack([c.real, c.imag], dim=1)


def compress_stft_tanh(stft_separated: Tensor, scale: float = 1.0) -> Tensor:
    """
    Compress STFT magnitudes to [-1, 1] using tanh, fully invertible without per-sample stats.

    Args:
        stft_separated: Tensor (channels, 2, freq_bins, time_steps)
        scale: Controls compression strength

    Returns:
        Compressed STFT tensor in [-1,1]
    """
    real = stft_separated[:, 0]
    imag = stft_separated[:, 1]
    c = torch.complex(real, imag)

    magnitude = torch.abs(c)
    phase = torch.angle(c)

    # Compress magnitude to [-1,1]
    magnitude_scaled = torch.tanh(scale * magnitude)

    # Rebuild complex
    c_scaled = magnitude_scaled * torch.exp(1j * phase)
    return torch.stack([c_scaled.real, c_scaled.imag], dim=1)


def decompress_stft_tanh(stft_compressed: Tensor, scale: float = 1.0) -> Tensor:
    real = stft_compressed[:, 0]
    imag = stft_compressed[:, 1]
    c_scaled = torch.complex(real, imag)

    magnitude_scaled = torch.abs(c_scaled).clamp(-0.999, 0.999)
    # clamp to avoid atanh issues
    phase_scaled = torch.angle(c_scaled)

    magnitude = (
        (1 / scale) * 0.5 * torch.log((1 + magnitude_scaled) / (1 - magnitude_scaled))
    )
    c = magnitude * torch.exp(1j * phase_scaled)

    return torch.stack([c.real, c.imag], dim=1)


def plot_spectrogram_grid(stft_separated: torch.Tensor) -> None:
    """
    Plot a single-row grid of STFT magnitude spectrograms with a shared colorbar.

    Args:
        stft_separated (Tensor): STFT tensor of shape
            (channels, 2, freq_bins, time_steps) or (channels*2, freq_bins, time_steps)
            where [:, 0] = real, [:, 1] = imag
    """

    # Allow both (C,2,H,W) and flattened (C*2,H,W)
    if stft_separated.ndim == 3:
        C_RI, H, W = stft_separated.shape
        RI = 2
        if C_RI % RI != 0:
            raise ValueError(
                f"Expected channel dimension divisible by {RI}, got {C_RI}"
            )
        C = C_RI // RI
        stft_separated = stft_separated.view(C, RI, H, W)
    elif stft_separated.ndim == 4:
        C = stft_separated.shape[0]
    else:
        raise ValueError(
            f"Invalid tensor shape {stft_separated.shape}, expected (C,2,H,W) or (C*2,H,W)"
        )

    # Compute magnitude
    real = stft_separated[:, 0]
    imag = stft_separated[:, 1]
    magnitude = torch.sqrt(real**2 + imag**2)

    # Create a single-row grid
    fig, axs = plt.subplots(1, C, figsize=(4 * C, 3), sharey=True)
    axs = axs if C > 1 else [axs]

    # Find global vmin/vmax for consistent color scaling
    vmin = magnitude.min().item()
    vmax = magnitude.max().item()

    # Plot each channel
    for i, ax in enumerate(axs):
        im = ax.imshow(
            magnitude[i].detach().cpu().numpy(),
            origin="lower",
            aspect="auto",
            cmap="magma",
            vmin=vmin,
            vmax=vmax,
        )
        ax.set_title(f"Ch {i + 1}", fontsize=9)
        ax.set_xlabel("Time frames")
        if i == 0:
            ax.set_ylabel("Frequency bins")
        else:
            ax.set_ylabel("")

    # Add a single colorbar for all subplots
    cbar = fig.colorbar(im, ax=axs, orientation="vertical", fraction=0.02, pad=0.02)  # type: ignore
    cbar.set_label("Magnitude", rotation=270, labelpad=15)

    # plt.tight_layout()
    plt.show()
