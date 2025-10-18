import torch
import triton
import triton.language as tl
import torch.functional as F

def torch_solve(input: torch.Tensor, output: torch.Tensor, N: int):
    output.copy_(F.leaky_relu(input))

@triton.jit
def leaky_relu_kernel(
    input,
    output,
    n_elements,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(input + offsets, mask=mask, other=0.0)
    x = tl.where(x > 0, x , 0.01 * x)
    tl.store(output + offsets, x, mask=mask)

# input, output are tensors on the GPU
def solve(input: torch.Tensor, output: torch.Tensor, N: int):
    BLOCK_SIZE = 256
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    leaky_relu_kernel[grid](
        input,
        output,
        N,
        BLOCK_SIZE
    )