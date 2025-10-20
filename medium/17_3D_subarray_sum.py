'''
Implement a program that computes the sum of a 3D subarray of 32-bit integers. You are given an input 3D array input of length N x M x K, and two depth indices S_DEP and E_DEP and two row indices S_ROW and E_ROW and two column indices S_COL and E_COL. S_DEP, E_DEP, S_ROW, E_ROW, S_COL and E_COL are inclusive, 0-based start and end indices — compute the sum of input[S_DEP..E_DEP][S_ROW..E_ROW][S_COL..E_COL].

Implementation Requirements
Use only native features (external libraries are not permitted)
The solve function signature must remain unchanged
The final result must be stored in the output variable
'''

import torch
import triton
import triton.language as tl

# input, output are tensors on the GPU
def torch_solve(input: torch.Tensor, output: torch.Tensor, N: int, M: int, K: int, S_DEP: int, E_DEP: int, S_ROW: int, E_ROW: int, S_COL: int, E_COL: int):
    torch.sum(input[S_DEP:E_DEP+1, S_ROW:E_ROW+1, S_COL:E_COL+1], dim=(0,1,2), out=output)


# input, output are tensors on the GPU
# input, output are tensors on the GPU
@triton.jit
def subsum_kernel(
        input, output,
        M: tl.constexpr,
        N: tl.constexpr,
        K: tl.constexpr,
        S_DEP: tl.constexpr,
        E_DEP: tl.constexpr,
        S_ROW: tl.constexpr,
        E_ROW: tl.constexpr,
        S_COL: tl.constexpr,
        E_COL: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_k = tl.program_id(2)

    off_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    off_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    off_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)

    mask_m = (S_DEP <= off_m[:, None, None]) & (off_m[:, None, None] <= E_DEP) & (off_m[:, None, None] < M)
    mask_n = (S_ROW <= off_n[None, :, None]) & (off_n[None, :, None] <= E_ROW) & (off_n[None, :, None] < N)
    mask_k = (S_COL <= off_k[None, None, :]) & (off_k[None, None, :] <= E_COL) & (off_k[None, None, :] < K)
    mask = mask_m & mask_n & mask_k

    input_ptr = input + off_m[:, None, None] * (N * K) + off_n[None, :, None] * K + off_k[None, None, :]

    x = tl.load(input_ptr, mask=mask, other=0).to(tl.int64) # avoid overflow
    block_sum = tl.sum(tl.sum(tl.sum(x, axis=2), axis=1), axis=0)
    tl.atomic_add(output, block_sum, sem='relaxed')


def solve(input: torch.Tensor, output: torch.Tensor, M: int, N: int, K: int, S_DEP: int, E_DEP: int, S_ROW: int, E_ROW: int, S_COL: int, E_COL: int):
    BLOCK_M = 64
    BLOCK_N = 32
    BLOCK_K = 32
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N), triton.cdiv(K, BLOCK_K))
    subsum_kernel[grid](
        input, output, M, N, K, S_DEP, E_DEP, S_ROW, E_ROW, S_COL, E_COL, BLOCK_M, BLOCK_N, BLOCK_K
    )

def run():
    input = torch.tensor([[[1, 2, 3], [4, 5, 1]], [[1, 1, 1], [2, 2, 2]]]).int().cuda()
    output = torch.tensor([0]).int().cuda()
    M, N, K = 2, 2, 3
    S_DEP, E_DEP = 0, 1
    S_ROW, E_ROW = 0, 1
    S_COL, E_COL = 1, 2
    solve(input, output, M, N, K, S_DEP,E_DEP,S_ROW,E_ROW,S_COL,E_COL)
    print(f"output: {output}, expected:{17}")

if __name__ == "__main__":
    run()