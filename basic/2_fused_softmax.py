import torch
from torch import Tensor
import triton
import triton.language as tl
from triton.runtime import driver

# use safe softmax
def native_softmax(x: Tensor) -> Tensor:
    x_max = x.max(dim=1)[0]
    z = x - x_max[:, None]
    numerator = z.exp()
    denominator = z.sum(dim=1)
    ret = numerator / denominator[:, None]
    return ret

def torch_softmax(x: Tensor) -> Tensor:
    return x.softmax(dim=1)

