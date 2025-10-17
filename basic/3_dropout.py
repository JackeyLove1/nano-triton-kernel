import torch
import triton
import triton.language as tl
from torch import Tensor
import tabulate

@triton.jit
def _dropout(
        x_ptr, # input ptr
        x_keep_ptr, # pointer to a mask of 0s and 1s
        output_ptr,
        n_elements,
        p,
        BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    block_start = BLOCK_SIZE * pid
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)
    x_keep = tl.load(x_keep_ptr + offsets, mask=mask)

    out = tl.where(x_keep, x / (1 - p), 0.0)
    tl.store(output_ptr + offsets, out, mask=mask)

def dropout_kernel(x: Tensor, x_keep: Tensor, p: float) -> Tensor:
    output = torch.empty_like(x)
    assert x.is_contiguous()
    n_elements = x.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']), )
    _dropout[grid](x, x_keep, output, n_elements, p, BLOCK_SIZE=1024)
    return output

def run_dropout():
    size=(10, )
    x = torch.randn(size=size).cuda()
    p = 0.5
    x_keep = (torch.randn(size=size, ) > p).to(torch.int32).cuda()
    output = dropout_kernel(x, x_keep, p)
    print(tabulate.tabulate([
        ["input"] + x.tolist(),
        ["keep mask"] + x_keep.tolist(),
        ["output"] + output.tolist(),
    ]))

if __name__ == "__main__":
    run_dropout()