from torch import Tensor


def extract(a: Tensor, t: Tensor, x_shape: tuple) -> Tensor:
    out = a.cpu().gather(0, t.cpu().long()).to(a.device)
    return out.view((t.shape[0],) + (1,) * (len(x_shape) - 1))
