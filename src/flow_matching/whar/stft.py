from matplotlib import pyplot as plt
import torch


def stft_transform(x, n_fft=63, hop_length=4):
    """
    Args:
        x: Tensor of shape [time, channels]
    Returns:
        Tensor of shape [channels, 2, freq_bins, time_steps]
    """
    stft = torch.stft(
        x.permute(1, 0),
        n_fft=n_fft,
        hop_length=hop_length,
        return_complex=True,
        window=torch.hann_window(n_fft),
    )
    return torch.stack([stft.real, stft.imag], dim=1)


def istft_transform(stft_separated, n_fft=63, hop_length=4, length=128):
    """
    Args:
        stft_separated: Tensor of shape [channels, 2, freq_bins, time_steps]
    Returns:
        Tensor of shape [time, channels]
    """
    real = stft_separated[:, 0]
    imag = stft_separated[:, 1]
    stft_complex = torch.complex(real, imag)
    return torch.istft(
        stft_complex,
        n_fft=n_fft,
        hop_length=hop_length,
        window=torch.hann_window(n_fft),
        length=length,
    ).permute(1, 0)


def plot_spectrogram_grid(stft_separated: torch.Tensor, title="STFT Magnitude"):
    """
    Plots magnitude spectrograms for each input channel in a 3x3 grid without colorbars or axes.

    Args:
        stft_separated: Tensor of shape [channels, 2, freq_bins, time_steps]
                        Assumes channels == 9 for a 3x3 grid.
    """
    assert stft_separated.ndim == 4, "Expected shape [channels, 2, freq, time]"
    channels = stft_separated.shape[0]
    assert channels == 9, "Expected exactly 9 channels for 3x3 grid"

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
