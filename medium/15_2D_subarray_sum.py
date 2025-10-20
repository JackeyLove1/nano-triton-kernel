'''
Implement a program that computes the sum of a 2D subarray of 32-bit integers. You are given an input 2D array input of length N x M, and two row indices S_ROW and E_ROW and two column indices S_COL and E_COL. S_ROW, E_ROW, S_COL and E_COL are inclusive, 0-based start and end indices — compute the sum of input[S_ROW..E_ROW][S_COL..E_COL].

Implementation Requirements
Use only native features (external libraries are not permitted)
The solve function signature must remain unchanged
The final result must be stored in the output variable
'''

import torch
import triton
import triton.language as tl


def torch_solve(input: torch.Tensor, output: torch.Tensor, N: int, M: int, S_ROW: int, E_ROW: int, S_COL: int, E_COL: int):
    torch.sum(input[S_ROW:E_ROW+1, S_COL:E_COL+1], dim=(0,1), out=output)

# input, output are tensors on the GPU
@triton.jit
def subsum_kernel(
        input, output,
        M: tl.constexpr,
        N: tl.constexpr,
        S_ROW: tl.constexpr,
        E_ROW: tl.constexpr,
        S_COL: tl.constexpr,
        E_COL: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    off_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    off_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_m = (S_ROW <= off_m[:, None]) & (off_m[:, None] <= E_ROW) & (off_m[:, None] < M)
    mask_n = (S_COL <= off_n[None, :]) & (off_n[None, :] <= E_COL) & (off_n[None, :] < N)
    mask = mask_m & mask_n

    input_ptr = input + off_m[:, None] * N + off_n[None, :]

    x = tl.load(input_ptr, mask=mask, other=0.0).to(tl.int64) # avoid overflow
    block_sum = tl.sum(tl.sum(x, axis=1), axis=0)
    tl.atomic_add(output, block_sum, sem='relaxed')


def solve(input: torch.Tensor, output: torch.Tensor, M: int, N: int, S_ROW: int, E_ROW: int, S_COL: int, E_COL: int):
    BLOCK_M = 128
    BLOCK_N = 64
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    subsum_kernel[grid](
        input, output,
        M,N,S_ROW,E_ROW,S_COL,E_COL,
        BLOCK_M, BLOCK_N
    )

def run():
    M, N = 2, 3
    S_ROW = 0
    E_ROW = 1
    S_COL = 1
    E_COL = 2
    input = torch.tensor([[1,2,3],[4,5,6]]).float().cuda()
    output1 = torch.tensor([0]).float().cuda()
    output2 = torch.tensor([0]).float().cuda()
    torch_solve(input, output1, M, N, S_ROW, E_ROW, S_COL, E_COL)
    solve(input, output2, M, N, S_ROW, E_ROW, S_COL, E_COL)
    print(f"output:{output1}")
    print(f"output:{output2}")

if __name__ == "__main__":
    run()

