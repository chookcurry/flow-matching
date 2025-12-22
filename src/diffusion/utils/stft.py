import torch
from matplotlib import pyplot as plt
from matplotlib.image import AxesImage
from scipy.signal import butter, filtfilt  # type: ignore
from torch import Tensor

N_FFT = 64
HOP_LENGTH = 3
WIN_LENGTH = N_FFT
WINDOW = torch.hann_window(N_FFT)
ALPHA = 0.6
BETA = 1.0
FS = 20


def transform(x: Tensor) -> Tensor:
    # (time, channels)

    x = highpass_filter(x, FS)
    stft = transform_stft(x)
    stft = compress_stft(stft)
    stft = merge_stft_channels(stft)
    # (channels * 2, freq_bins, time_steps)

    return stft


def detransform(stft: Tensor, x_length: int) -> Tensor:
    # (channels * 2, freq_bins, time_steps)

    stft = unmerge_stft_channels(stft)
    stft = decompress_stft(stft)
    x = transform_istft(stft, x_length)
    # (time, channels)

    return x


def merge_stft_channels(stft: Tensor) -> Tensor:
    # (channels, 2, freq_bins, time_steps)

    C, RI, F, T = stft.shape
    stft = stft.view(C * RI, F, T)
    # (channels * 2, freq_bins, time_steps)

    return stft


def unmerge_stft_channels(stft: Tensor) -> Tensor:
    # (channels * 2, freq_bins, time_steps)

    CRI, F, T = stft.shape
    stft = stft.view(CRI // 2, 2, F, T)
    # (channels, 2, freq_bins, time_steps)

    return stft


def transform_stft(x: Tensor) -> Tensor:
    # (time, channels)

    x = x.permute(1, 0)
    # (channels, time)

    stft = torch.stft(
        input=x,
        center=True,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        win_length=WIN_LENGTH,
        return_complex=True,
        window=WINDOW.to(x.device),
    )  # (channels, freq_bins, time_steps)

    stft = torch.stack([stft.real, stft.imag], dim=1)
    # (channels, 2, freq_bins, time_steps)

    return stft


def transform_istft(stft: Tensor, x_length: int) -> Tensor:
    # (channels, 2, freq_bins, time_steps)

    stft = torch.complex(stft[:, 0], stft[:, 1])
    # (channels, freq_bins, time_steps)

    x = torch.istft(
        stft,
        center=True,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        win_length=WIN_LENGTH,
        window=WINDOW.to(stft.device),
        length=x_length,
    )  # (channels, time)

    x = x.permute(1, 0)
    # (time, channels)

    return x


def compress_stft(stft: Tensor) -> Tensor:
    # (channels, 2, freq_bins, time_steps)
    # Apply amplitude transformation to STFT coefficients:
    # c_tilde = beta * |c|^alpha * exp(i * angle(c))

    stft = torch.complex(stft[:, 0], stft[:, 1])
    # (channels, freq_bins, time_steps)

    magnitude = torch.abs(stft)
    phase = torch.angle(stft)
    mag_compressed = BETA * (magnitude**ALPHA)
    real_tilde = mag_compressed * torch.cos(phase)
    imag_tilde = mag_compressed * torch.sin(phase)
    # (channels, freq_bins, time_steps)

    stft = torch.stack([real_tilde, imag_tilde], dim=1)
    # (channels, 2, freq_bins, time_steps)

    return stft


def decompress_stft(stft: Tensor) -> Tensor:
    # (channels, 2, freq_bins, time_steps)
    # Inverse amplitude transformation to recover original STFT coefficients:
    # c = (|c_tilde| / beta)^(1/alpha) * exp(i * angle(c_tilde))

    real = stft[:, 0]
    imag = stft[:, 1]
    mag_compressed = torch.sqrt(real**2 + imag**2)
    phase = torch.atan2(imag, real)
    magnitude = (mag_compressed / BETA) ** (1 / ALPHA)
    # (channels, freq_bins, time_steps)

    stft = torch.polar(magnitude, phase)
    # (channels, freq_bins, time_steps)

    stft = torch.stack([stft.real, stft.imag], dim=1)
    # (channels, 2, freq_bins, time_steps)

    return stft


def highpass_filter(x: Tensor, fs: int, cutoff: float = 0.1, order: int = 5) -> Tensor:
    # (time, channels)

    # Adjust cutoff based on your actual sampling rate
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    filter = butter(order, normal_cutoff, btype="high", analog=False)

    assert isinstance(filter, tuple) and len(filter) == 2
    b, a = filter

    # data expected to be (Time, Channels)
    x = torch.tensor(
        filtfilt(b, a, x.detach().cpu().numpy(), axis=0).copy(),
        dtype=torch.float32,
        device=x.device,
    )
    # (time, channels)

    return x


def plot_spects(stft: torch.Tensor) -> None:
    # (channels, 2, freq_bins, time_steps)
    # or # (channels * 2, freq_bins, time_steps)

    stft = stft if stft.ndim == 4 else unmerge_stft_channels(stft)
    # (channels, 2, freq_bins, time_steps)

    C = stft.shape[0]
    mag = torch.sqrt(stft[:, 0] ** 2 + stft[:, 1] ** 2).detach().cpu().numpy()
    # (channels, freq_bins, time_steps)

    # Create a single-row grid
    fig, axs = plt.subplots(1, C, figsize=(4 * C, 3), sharey=True)
    axs = axs if C > 1 else [axs]

    # Find global vmin/vmax for consistent color scaling
    vmin = mag.min()
    vmax = mag.max()

    im: AxesImage | None = None

    # Plot each channel
    for i, ax in enumerate(axs):
        im = ax.imshow(
            mag[i],
            origin="lower",
            aspect="auto",
            cmap="magma",
            vmin=vmin,
            vmax=vmax,
        )
        ax.set_title(f"Ch {i + 1}", fontsize=9)
        ax.set_xlabel("Time frames")
        ax.set_ylabel("Frequency bins" if i == 0 else None)

    # Add a single colorbar for all subplots
    if im is not None:
        cbar = fig.colorbar(im, ax=axs, orientation="vertical", fraction=0.02, pad=0.02)
        cbar.set_label("Magnitude", rotation=270, labelpad=15)

    # plt.tight_layout()
    plt.show()
