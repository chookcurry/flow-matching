from typing import Tuple
import torch


def compute_pairwise_distances(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    # x: [N, D], y: [M, D]
    return torch.cdist(x, y, p=2)  # Euclidean


def precision_recall_knn(
    real_feats: torch.Tensor, gen_feats: torch.Tensor, k=3, batch_size=1000
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute precision and recall in feature space using kNN radii.

    Args:
        real_feats (Tensor): Real features, shape (N, D)
        gen_feats (Tensor): Generated features, shape (M, D)
        k (int): Number of neighbors to use (default: 3)
        batch_size (int): Chunk size for memory-efficient computation

    Returns:
        (float, float): Precision, Recall
    """

    real_feats = real_feats.view(real_feats.size(0), -1)
    gen_feats = gen_feats.view(gen_feats.size(0), -1)

    # Compute k-th NN distance from real to real
    dists_rr = compute_pairwise_distances(real_feats, real_feats)
    dists_rr[torch.eye(dists_rr.size(0)).bool()] = float("inf")  # ignore self
    kth_vals_real, _ = dists_rr.topk(k, largest=False, dim=1)  # [N, k]
    r_real = kth_vals_real[:, -1]  # [N] radius of kNN ball

    # Compute k-th NN distance from gen to gen
    dists_gg = compute_pairwise_distances(gen_feats, gen_feats)
    dists_gg[torch.eye(dists_gg.size(0)).bool()] = float("inf")
    kth_vals_gen, _ = dists_gg.topk(k, largest=False, dim=1)
    r_gen = kth_vals_gen[:, -1]

    # Precision: how many gen samples fall within real kNN balls
    precision_count = torch.zeros(1)
    for i in range(0, gen_feats.size(0), batch_size):
        gen_batch = gen_feats[i : i + batch_size]
        dists = compute_pairwise_distances(gen_batch, real_feats)  # [B, N]
        within = dists <= r_real.unsqueeze(0)  # [B, N]
        precision_count += (within.any(dim=1)).float().sum().item()

    precision = precision_count / gen_feats.size(0)

    # Recall: how many real samples fall within gen kNN balls
    recall_count = torch.zeros(1)
    for i in range(0, real_feats.size(0), batch_size):
        real_batch = real_feats[i : i + batch_size]
        dists = compute_pairwise_distances(real_batch, gen_feats)  # [B, M]
        within = dists <= r_gen.unsqueeze(0)  # [B, M]
        recall_count += (within.any(dim=1)).float().sum().item()

    recall = recall_count / real_feats.size(0)

    return precision, recall


def f1_score(precision: torch.Tensor, recall: torch.Tensor) -> torch.Tensor:
    if precision + recall == torch.zeros(1):
        return torch.zeros(1)
    return 2 * (precision * recall) / (precision + recall)
