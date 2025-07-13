from typing import List, Type
import torch

from flow_matching.alphas_betas import Alpha, Beta


def build_mlp(
    dims: List[int], activation: Type[torch.nn.Module] = torch.nn.SiLU
) -> torch.nn.Module:
    mlp: List[torch.nn.Module] = []
    for idx in range(len(dims) - 1):
        mlp.append(torch.nn.Linear(dims[idx], dims[idx + 1]))
        if idx < len(dims) - 2:
            mlp.append(activation())
    return torch.nn.Sequential(*mlp)


class MLPVectorField(torch.nn.Module):
    """
    MLP-parameterization of the learned vector field u_t^theta(x)
    """

    def __init__(self, dim: int, hiddens: List[int]):
        super().__init__()
        self.dim = dim
        self.net = build_mlp([dim + 1] + hiddens + [dim])

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
        - x: (bs, dim)
        Returns:
        - u_t^theta(x): (bs, dim)
        """
        xt = torch.cat([x, t], dim=-1)
        pred: torch.Tensor = self.net(xt)
        return pred


class MLPScore(torch.nn.Module):
    """
    MLP-parameterization of the learned score field
    """

    def __init__(self, dim: int, hiddens: List[int]):
        super().__init__()
        self.dim = dim
        self.net = build_mlp([dim + 1] + hiddens + [dim])

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
        - x: (bs, dim)
        Returns:
        - s_t^theta(x): (bs, dim)
        """
        xt = torch.cat([x, t], dim=-1)
        pred: torch.Tensor = self.net(xt)
        return pred


class ScoreFromVectorField(torch.nn.Module):
    """
    Parameterization of score via learned vector field (for the special case of a Gaussian conditional probability path)
    """

    def __init__(self, vector_field: MLPVectorField, alpha: Alpha, beta: Beta) -> None:
        super().__init__()
        self.vector_field = vector_field
        self.alpha = alpha
        self.beta = beta

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
        - x: (bs, dim)
        Returns:
        - score: (bs, dim)
        """
        pred: torch.Tensor = self.vector_field(x, t)
        numerator = self.alpha(t) * pred - self.alpha.dt(t) * x
        denominator = self.beta(t) ** 2 * self.alpha.dt(t) - self.alpha(
            t
        ) * self.beta.dt(t) * self.beta(t)
        return numerator / denominator
