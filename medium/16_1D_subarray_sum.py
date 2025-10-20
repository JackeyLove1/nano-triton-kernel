'''
Implement a program that computes the sum of a subarray of 32-bit integers. You are given an input array input of length N, and two indices S and E. S and E are inclusive, 0-based start and end indices — compute the sum of input[S..E].

Implementation Requirements
Use only native features (external libraries are not permitted)
The solve function signature must remain unchanged
The final result must be stored in the output variable
'''

import torch
import triton
import triton.language as tl

def torch_solve(input: torch.Tensor, output: torch.Tensor, N: int, S: int, E: int):
    output.copy_(input[S:E+1].sum())

# input, output are tensors on the GPU
@triton.jit
def subarray_sum_kernel(
        input, output,
        N: tl.constexpr,
        S: tl.constexpr,
        E: tl.constexpr,
        BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = (offsets < N) & (S <= offsets) & (offsets <= E)

    x = tl.load(input + offsets, mask=mask, other=0.0)
    sum = tl.sum(x, axis=0)
    tl.atomic_add(output, sum)


def solve(input: torch.Tensor, output: torch.Tensor, N: int, S: int, E: int):
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(N, BLOCK_SIZE), )
    subarray_sum_kernel[grid](
        input, output, N, S, E, BLOCK_SIZE
    )

def run():
    N = 100_000
    S = 1
    E = 3
    input = torch.tensor([1,2,1,3,4]).float().cuda()
    output = torch.tensor([0]).float().cuda()
    solve(input, output, N, S, E)
    print(f"Sum: {output}, expected: {6}")

if __name__ == "__main__":
    run()