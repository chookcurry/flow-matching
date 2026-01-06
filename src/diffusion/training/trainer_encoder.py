import torch
from torch import Tensor, nn
from torch.nn import functional as F

from diffusion.classifiers.encoder import Encoder
from diffusion.sampleables.sampleable import Sampleable
from diffusion.training.trainer import Trainer


class EncoderTrainer(Trainer):
    def __init__(
        self,
        encoder: Encoder,
        train_data: Sampleable,
        val_data: Sampleable,
        num_classes: int,
        temperature: float = 0.07,
    ):
        super().__init__(encoder)

        self.train_data = train_data
        self.val_data = val_data
        self.num_classes = num_classes
        self.criterion = SupConLoss(temperature)

    def get_train_loss(self, batch_size: int, device: torch.device) -> Tensor:
        return self._get_loss(self.train_data, batch_size, device)

    @torch.no_grad()
    def get_val_loss(self, batch_size: int, device: torch.device) -> Tensor:
        return self._get_loss(self.val_data, batch_size, device)

    def _get_loss(
        self, data: Sampleable, batch_size: int, device: torch.device
    ) -> Tensor:
        x, y = data.sample(batch_size)
        assert y is not None
        x, y = x.to(device), y.to(device)

        assert isinstance(self.model, Encoder)
        embeddings = self.model.encode(x)
        loss: Tensor = self.criterion(embeddings, y)

        return loss


class SupConLoss(nn.Module):
    """
    Supervised Contrastive Loss (Khosla et al., 2020)
    https://arxiv.org/abs/2004.11362
    """

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, features: Tensor, labels: Tensor) -> Tensor:
        # Normalize feature embeddings
        features = F.normalize(features, dim=1)
        batch_size = features.size(0)

        # Compute similarity matrix
        sim_matrix = torch.div(torch.matmul(features, features.T), self.temperature)

        # Mask self-similarity
        mask = torch.eye(batch_size, dtype=torch.bool, device=features.device)
        sim_matrix = sim_matrix.masked_fill(mask, -1e9)

        # Create positive mask (same class)
        labels = labels.contiguous().view(-1, 1)
        pos_mask = torch.eq(labels, labels.T).to(features.device)
        pos_mask = pos_mask & ~mask

        # Log-softmax over similarity
        log_prob = sim_matrix - torch.logsumexp(sim_matrix, dim=1, keepdim=True)

        # Mean log-likelihood over positive pairs
        mean_log_prob_pos = (pos_mask * log_prob).sum(1) / pos_mask.sum(1).clamp(min=1)

        # Final loss
        loss = -mean_log_prob_pos.mean()
        return loss
