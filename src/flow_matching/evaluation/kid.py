import torch


def kernel_inception_distance_polynomial(
    x: torch.Tensor, y: torch.Tensor, degree=3
) -> torch.Tensor:
    """
    Compute KID using a polynomial kernel: K(x, y) = (xᵀy / d + 1)^degree

    Args:
        x (torch.Tensor): Generated features (B, C, H, W) or (B, D)
        y (torch.Tensor): Real features (B, C, H, W) or (B, D)
        degree (int): Degree of the polynomial kernel

    Returns:
        torch.Tensor: KID estimate using polynomial kernel
    """
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


def kernel_inception_distance_polynomial_biased(
    x: torch.Tensor, y: torch.Tensor, degree=3
) -> torch.Tensor:
    """
    Compute KID using a polynomial kernel: K(x, y) = (xᵀy / d + 1)^degree

    Args:
        x (torch.Tensor): Generated features (B, C, H, W) or (B, D)
        y (torch.Tensor): Real features (B, C, H, W) or (B, D)
        degree (int): Degree of the polynomial kernel

    Returns:
        torch.Tensor: KID estimate using polynomial kernel
    """
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


def kernel_inception_distance_rbf(x: torch.Tensor, y: torch.Tensor, alpha=0.001):
    """
    Compute Kernel Inception Distance (KID) using an RBF kernel.

    Args:
        x (torch.Tensor): Generated features of shape (B, C, H, W) or (B, D)
        y (torch.Tensor): Real features of shape (B, C, H, W) or (B, D)
        alpha (float): RBF kernel bandwidth coefficient

    Returns:
        torch.Tensor: Scalar KID value (MMD^2 estimate)
    """

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
