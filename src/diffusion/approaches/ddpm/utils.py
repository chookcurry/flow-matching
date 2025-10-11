from torch import Tensor


def extract(a: Tensor, t: Tensor, x_shape: tuple) -> Tensor:
    out = a.gather(dim=0, index=t.int())
    return out.view((t.shape[0],) + (1,) * (len(x_shape) - 1))
