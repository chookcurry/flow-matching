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
):
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
    """
    Apply amplitude transformation to STFT coefficients:
    c_tilde = beta * |c|^alpha * exp(i * angle(c))

    Args:
        stft_separated: Tensor (channels, 2, freq_bins, time_steps) with real and imag parts.
        alpha: Compression exponent in (0, 1].
        beta: Scaling factor for normalization.

    Returns:
        Transformed STFT tensor of the same shape.
    """
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
    """
    Inverse amplitude transformation to recover original STFT coefficients:
    c = (|c_tilde| / beta)^(1/alpha) * exp(i * angle(c_tilde))

    Args:
        stft_compressed: Tensor (channels, 2, freq_bins, time_steps) with compressed real and imag parts.
        alpha: Compression exponent used during compression.
        beta: Scaling factor used during compression.

    Returns:
        Decompressed STFT tensor of the same shape.
    """
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


def plot_spectrogram_grid(stft_separated: Tensor) -> None:
    # (channels, 2, freq_bins, time_steps)

    assert stft_separated.ndim == 4
    channels = stft_separated.shape[0]
    assert channels == 9

    real = stft_separated[:, 0]
    imag = stft_separated[:, 1]
    magnitude = torch.sqrt(real**2 + imag**2)

    _, axs = plt.subplots(3, 3, figsize=(6, 6))
    for i, ax in enumerate(axs.flatten()):
        ax.imshow(
            magnitude[i].detach().cpu().numpy(),
            origin="lower",
            aspect="equal",
            cmap="magma",
        )
        ax.axis("off")

    plt.tight_layout()
    plt.show()
