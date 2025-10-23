import torch
import triton
import triton.language as tl
from torch import Tensor

@triton.jit
def _seeded_dropout(
    x_ptr,
    output_ptr,
    n_elements: tl.constexpr,
    p: tl.constexpr,
    seed: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    random = tl.rand(seed, offsets)
    x_keep = random > p
    output = tl.where(x_keep, x / (1 - p), 0.0)
    tl.store(output_ptr + offsets, output, mask=mask)

def seed_dropout(x: Tensor, p: float, seed: int):
    output = torch.empty_like(x)
    assert  x.is_contiguous()
    n_elements = x.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']), )
    _seeded_dropout[grid](x, output, n_elements, p, seed, BLOCK_SIZE=1024)
    return output


def run():
    N = 1_000_000
    p = 0.5
    seed = 0
    x = torch.randn((N, )).float().cuda()
    _ = seed_dropout(x, p, seed)

if __name__ == "__main__":
    run()