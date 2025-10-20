'''
Write a GPU program that counts the number of elements with the integer value p in an 3D array of 32-bit integers. The program should count the number of elements with p in an 3D array. You are given an input 3D array input of length N x M x K and integer p.

Implementation Requirements
Use only native features (external libraries are not permitted)
The solve function signature must remain unchanged
The final result must be stored in the output variable
'''

import torch
import triton
import triton.language as tl

# input, output are tensors on the GPU
# 3D version
@triton.jit
def subsum_kernel(
        input, output,
        M: tl.constexpr,
        N: tl.constexpr,
        K: tl.constexpr,
        P: tl.constexpr,
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

    mask_m = off_m[:, None, None] < M
    mask_n = off_n[None, :, None] < N
    mask_k = off_k[None, None, :] < K
    mask = mask_m & mask_n & mask_k

    input_ptr = input + off_m[:, None, None] * (N * K) + off_n[None, :, None] * K + off_k[None, None, :]

    x = tl.load(input_ptr, mask=mask, other=0)
    eq = (x == P).to(tl.int32)
    eq = tl.where(mask, eq, 0)
    sum = tl.sum(tl.sum(tl.sum(eq, axis=2), axis=1), axis=0)
    tl.atomic_add(output, sum, sem='relaxed')


def solve_3d(input: torch.Tensor, output: torch.Tensor, M: int, N: int, K: int, P: int):
    BLOCK_M = 64
    BLOCK_N = 32
    BLOCK_K = 32
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N), triton.cdiv(K, BLOCK_K))
    subsum_kernel[grid](
        input, output, M, N, K, P, BLOCK_M, BLOCK_N, BLOCK_K
    )

# 1D version

@triton.jit
def count_eq_kernel(x_ptr, out_ptr,
                    n_elements,
                    P: tl.constexpr,
                    BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    x = tl.load(x_ptr + offs, mask=mask, other=0)

    # 注意：把布尔转成 int32 再求和更稳妥
    eq = (x == P)
    eq_i32 = tl.where(mask & eq, 1, 0).to(tl.int32)

    s = tl.sum(eq_i32, axis=0)  # 标量
    tl.atomic_add(out_ptr, s, sem='relaxed')


def solve(input: torch.Tensor, output: torch.Tensor, M: int, N: int, K: int, P: int):
    assert input.is_cuda and output.is_cuda
    x = input.contiguous().view(-1).int()   # 展平
    n = M * N * K
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n, BLOCK_SIZE),)
    # 可根据硬件/问题规模调整 warps
    count_eq_kernel[grid](x, output, n, P, BLOCK_SIZE, num_warps=4)

def run():
    input = torch.tensor([[[1, 2, 3],[4, 5, 1]],[[1, 1, 1],[2, 2, 2]]]).int().cuda()
    output1 = torch.tensor([0]).int().cuda()
    output2 = torch.tensor([0]).int().cuda()
    M, N, K = 2, 2, 3
    P = 1
    solve(input, output1, M, N, K, P)
    solve_3d(input, output2, M, N, K, P)
    print(f"output1: {output1},output2:{output2},expected:{5}")

if __name__ == "__main__":
    run()