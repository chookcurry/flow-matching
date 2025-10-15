from typing import Tuple
import torch
from torch import Tensor


def density_coverage_knn(
    real_feats: Tensor, gen_feats: Tensor, k: int = 3, batch_size: int = 300
) -> Tuple[Tensor, Tensor]:
    """
    Compute Density and Coverage metrics between real and generated features.
    Based on Kynkäänniemi et al., 'Improved Precision and Recall' and 'Density & Coverage'.

    Args:
        real_feats (Tensor): Real feature embeddings, shape [N, D].
        gen_feats (Tensor): Generated feature embeddings, shape [M, D].
        k (int): Number of neighbors for kNN radius.
        batch_size (int): Batch size for distance computation (for memory efficiency).

    Returns:
        (density, coverage) as Tensors
    """
    assert real_feats.shape[1:] == gen_feats.shape[1:], "Feature dims must match"
    N = real_feats.shape[0]
    M = gen_feats.shape[0]

    real_feats = real_feats.view(N, -1)
    gen_feats = gen_feats.view(M, -1)

    # 1️⃣ Compute real-to-real kNN distances (same as for precision)
    dists_rr = compute_pairwise_distances(real_feats, real_feats)
    dists_rr.fill_diagonal_(float("inf"))
    kth_vals_real, _ = dists_rr.topk(k, largest=False, dim=1)
    r_real = kth_vals_real[:, -1]  # radius per real sample [N]

    # 2️⃣ DENSITY: how many real neighborhoods each fake sample is inside
    total_density = 0.0
    for i in range(0, gen_feats.size(0), batch_size):
        gen_batch = gen_feats[i : i + batch_size]  # [B, D]
        dists = compute_pairwise_distances(gen_batch, real_feats)  # [B, N]
        within = dists <= r_real.unsqueeze(0)  # [B, N]
        total_density += within.float().sum().item()  # count all inside
    density = total_density / (k * M)

    # 3️⃣ COVERAGE: fraction of real samples covered by at least one fake sample
    covered = torch.zeros(N, dtype=torch.bool, device=real_feats.device)
    for i in range(0, real_feats.size(0), batch_size):
        real_batch = real_feats[i : i + batch_size]
        dists = compute_pairwise_distances(real_batch, gen_feats)  # [B, M]
        within = dists <= r_real[i : i + batch_size].unsqueeze(1)
        covered[i : i + batch_size] = within.any(dim=1)
    coverage = covered.float().mean()

    return torch.tensor(density), coverage


def compute_pairwise_distances(x: Tensor, y: Tensor) -> Tensor:
    """Compute Euclidean distance matrix between x and y."""
    return torch.cdist(x, y, p=2)
