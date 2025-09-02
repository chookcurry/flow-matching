import torch.nn as nn


MiB = 1024**2


def model_size_b(model: nn.Module) -> int:
    size = 0
    for param in model.parameters():
        size += param.nelement() * param.element_size()
    for buf in model.buffers():
        size += buf.nelement() * buf.element_size()
    return size


def model_size_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
