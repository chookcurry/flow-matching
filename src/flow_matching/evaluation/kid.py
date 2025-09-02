import torch
from torch import Tensor


def kernel_inception_distance_poly(x: Tensor, y: Tensor, degree: int = 3) -> Tensor:
    # polynomial kernel: K(x, y) = (xᵀy / d + 1)^degree

    B = x.size(0)
    assert x.shape == y.shape, "x and y must have the same shape"

    # Flatten spatial dimensions
    x = x.view(B, -1)
    y = y.view(B, -1)
    d = x.size(1)

    # Compute dot products
    xx = (x @ x.t()) / d + 1
    yy = (y @ y.t()) / d + 1
    xy = (x @ y.t()) / d + 1

    # Apply polynomial kernel
    K_xx = xx.pow(degree)
    K_yy = yy.pow(degree)
    K_xy = xy.pow(degree)

    # Remove diagonals (unbiased estimate)
    sum_K_xx = (K_xx.sum() - K_xx.diag().sum()) / (B * (B - 1))
    sum_K_yy = (K_yy.sum() - K_yy.diag().sum()) / (B * (B - 1))
    sum_K_xy = K_xy.sum() / (B * B)

    return sum_K_xx + sum_K_yy - 2 * sum_K_xy


def kernel_inception_distance_poly_biased(
    x: Tensor, y: Tensor, degree: int = 3
) -> Tensor:
    # polynomial kernel: K(x, y) = (xᵀy / d + 1)^degree

    B = x.size(0)
    assert x.shape == y.shape, "x and y must have the same shape"

    # Flatten spatial dimensions
    x = x.view(B, -1)
    y = y.view(B, -1)
    d = x.size(1)

    # Compute dot products
    xx = (x @ x.t()) / d + 1
    yy = (y @ y.t()) / d + 1
    xy = (x @ y.t()) / d + 1

    # Apply polynomial kernel
    K_xx = xx.pow(degree)
    K_yy = yy.pow(degree)
    K_xy = xy.pow(degree)

    # Remove diagonals (unbiased estimate)
    sum_K_xx = K_xx.sum() / (B * B)
    sum_K_yy = K_yy.sum() / (B * B)
    sum_K_xy = K_xy.sum() / (B * B)

    return sum_K_xx + sum_K_yy - 2 * sum_K_xy


def kernel_inception_distance_rbf(x: Tensor, y: Tensor, alpha: float = 0.001) -> Tensor:
    # RBF kernel: K(x, y) = exp(-alpha * ||x - y||^2)

    B = x.size(0)
    assert x.shape == y.shape, "x and y must have the same shape"

    # Flatten spatial dimensions: B x (C * H * W)
    x = x.view(B, -1)
    y = y.view(B, -1)

    # Compute dot products
    xx = torch.mm(x, x.t())  # [B, B]
    yy = torch.mm(y, y.t())  # [B, B]
    xy = torch.mm(x, y.t())  # [B, B]

    # Compute squared norms
    rx = xx.diag().unsqueeze(0).expand_as(xx)
    ry = yy.diag().unsqueeze(0).expand_as(yy)

    # Compute kernels
    K_xx = torch.exp(-alpha * (rx.t() + rx - 2 * xx))  # [B, B]
    K_yy = torch.exp(-alpha * (ry.t() + ry - 2 * yy))  # [B, B]
    K_xy = torch.exp(-alpha * (rx.t() + ry - 2 * xy))  # [B, B]

    # Remove diagonals (unbiased estimate)
    sum_K_xx = (K_xx.sum() - K_xx.diag().sum()) / (B * (B - 1))
    sum_K_yy = (K_yy.sum() - K_yy.diag().sum()) / (B * (B - 1))
    sum_K_xy = K_xy.sum() / (B * B)

    # MMD² (KID)
    return sum_K_xx + sum_K_yy - 2 * sum_K_xy
