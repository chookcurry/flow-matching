# Several plotting utility functions
from typing import Any, Optional, Tuple

from matplotlib import pyplot as plt
from matplotlib.axes import Axes
import numpy as np
import torch
import seaborn as sns
import matplotlib.colors as mcolors

from flow_matching.unsupervised.densities import Density, Sampleable


def hist2d_samples(
    samples: torch.Tensor,
    ax: Optional[Axes] = None,
    bins: int = 200,
    scale: float = 5.0,
    percentile: float = 99,
    **kwargs: Any,
) -> None:
    H, xedges, yedges = np.histogram2d(
        samples[:, 0],
        samples[:, 1],
        bins=bins,
        range=[[-scale, scale], [-scale, scale]],
    )

    # Determine color normalization based on the 99th percentile
    cmax = float(np.percentile(H, percentile))
    cmin = 0.0
    norm = mcolors.Normalize(vmax=cmax, vmin=cmin)

    assert ax is not None

    # Plot using imshow for more control
    extent = (xedges[0], xedges[-1], yedges[0], yedges[-1])
    ax.imshow(H.T, extent=extent, origin="lower", norm=norm, **kwargs)


def hist2d_sampleable(
    sampleable: Sampleable,
    num_samples: int,
    ax: Optional[Axes] = None,
    bins: int = 200,
    scale: float = 5.0,
    percentile: int = 99,
    **kwargs: Any,
) -> None:
    assert sampleable.dim == 2
    if ax is None:
        ax = plt.gca()
    samples = sampleable.sample(num_samples).detach().cpu()  # (ns, 2)
    hist2d_samples(samples, ax, bins, scale, percentile, **kwargs)


def scatter_sampleable(
    sampleable: Sampleable, num_samples: int, ax: Optional[Axes] = None, **kwargs: Any
) -> None:
    assert sampleable.dim == 2
    if ax is None:
        ax = plt.gca()
    samples = sampleable.sample(num_samples)  # (ns, 2)
    ax.scatter(samples[:, 0].cpu(), samples[:, 1].cpu(), **kwargs)


def kdeplot_sampleable(
    sampleable: Sampleable, num_samples: int, ax: Optional[Axes] = None, **kwargs: Any
) -> None:
    assert sampleable.dim == 2
    if ax is None:
        ax = plt.gca()
    samples = sampleable.sample(num_samples)  # (ns, 2)
    sns.kdeplot(
        x=samples[:, 0].cpu().numpy(), y=samples[:, 1].cpu().numpy(), ax=ax, **kwargs
    )


def imshow_density(
    device: torch.device,
    density: Density,
    x_bounds: Tuple[float, float],
    y_bounds: Tuple[float, float],
    bins: int,
    ax: Optional[Axes] = None,
    x_offset: float = 0.0,
    **kwargs: Any,
) -> None:
    if ax is None:
        ax = plt.gca()
    x_min, x_max = x_bounds
    y_min, y_max = y_bounds
    x = torch.linspace(x_min, x_max, bins).to(device) + x_offset
    y = torch.linspace(y_min, y_max, bins).to(device)
    X, Y = torch.meshgrid(x, y)
    xy = torch.stack([X.reshape(-1), Y.reshape(-1)], dim=-1)
    log_prob = density.log_density(xy).reshape(bins, bins).T
    ax.imshow(
        log_prob.cpu(), extent=(x_min, x_max, y_min, y_max), origin="lower", **kwargs
    )


def contour_density(
    device: torch.device,
    density: Density,
    bins: int,
    scale: float,
    ax: Optional[Axes] = None,
    x_offset: float = 0.0,
    **kwargs: Any,
) -> None:
    if ax is None:
        ax = plt.gca()
    x = torch.linspace(-scale + x_offset, scale + x_offset, bins).to(device)
    y = torch.linspace(-scale, scale, bins).to(device)
    X, Y = torch.meshgrid(x, y)
    xy = torch.stack([X.reshape(-1), Y.reshape(-1)], dim=-1)
    log_prob = density.log_density(xy).reshape(bins, bins).T
    ax.contour(log_prob.cpu(), origin="lower", **kwargs)
