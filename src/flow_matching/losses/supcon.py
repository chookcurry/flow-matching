import torch
import torch.nn.functional as F
from torch import Tensor


def loss_supcon(x: Tensor, y: Tensor, temp: float = 0.07) -> Tensor:
    # x: (batch_size, latent_dim)
    # y: (batch_size,)

    device = x.device
    batch_size = x.shape[0]

    # Normalize embeddings
    x = F.normalize(x, dim=1)

    # Cosine similarity matrix [N, N]
    sim_matrix = torch.matmul(x, x.T) / temp

    # Mask to remove self-comparisons
    self_mask = torch.eye(batch_size, dtype=torch.bool, device=device)

    # Positive mask: same label & not self
    y = y.contiguous().view(-1, 1)
    pos_mask = torch.eq(y, y.T) & ~self_mask

    # --- log-probabilities (log-softmax) ---
    # subtract row max for stability
    logits_max, _ = torch.max(sim_matrix, dim=1, keepdim=True)
    logits = sim_matrix - logits_max.detach()

    # denominator: exp(logits) over all but self
    exp_logits = torch.exp(logits) * (~self_mask).float()
    log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-12)

    # --- average log-likelihood over positives ---
    pos_counts = pos_mask.sum(1)
    mean_log_prob_pos = (pos_mask.float() * log_prob).sum(1) / (pos_counts + 1e-12)

    # only anchors with positives should contribute
    valid = pos_counts > 0
    loss = -(mean_log_prob_pos[valid].mean())

    return loss
