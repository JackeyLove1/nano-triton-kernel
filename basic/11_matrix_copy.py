'''
Implement a program that copies an N x N matrix of 32-bit floating point numbers from input array A to output array B on the GPU. The program should perform a direct element-wise copyso that Bi,j= Ai,j for all valid indices.
Implementation Requirements
External libraries are not permitted
The solve function signature must remain unchanged
e The final result must be stored in matrix B
'''
import torch
import triton
import triton.language as tl
from torch import Tensor

def torch_solve(A: torch.Tensor, B: torch.Tensor, N: int):
    B.copy_(A)

@triton.jit
def matrix_copy_kernel_block(
        A_ptr, B_ptr,
        N,
        stride_ar,stride_ac,
        stride_br,stride_bc,
        BLOCK_SIZE: tl.constexpr
):
    pid_row = tl.program_id(0)
    pid_col = tl.program_id(1)
    row_ = pid_row * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    col_ = pid_col * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    a_tile_ptrs = A_ptr + row_[:, None] * stride_ar + col_[None, :] * stride_ac
    b_tile_ptrs = B_ptr + row_[:, None] * stride_br + col_[None, :] * stride_bc

    mask = (row_[:, None] < N) & (col_[None, :] < N)
    vals = tl.load(a_tile_ptrs, mask=mask, other=0.0)
    tl.store(b_tile_ptrs, vals, mask=mask)

# a, b are tensors on the GPU
def solve_block(a: torch.Tensor, b: torch.Tensor, N: int):
    BLOCK_SIZE = 256
    grid = (triton.cdiv(N, BLOCK_SIZE), triton.cdiv(N, BLOCK_SIZE))

    matrix_copy_kernel_block[grid](
        a, b,
        N,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        BLOCK_SIZE
    )

@triton.jit
def matrix_copy_kernel_1d(
        A_ptr, B_ptr,
        N: tl.constexpr,
        BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    vals = tl.load(A_ptr + offsets, mask=mask, other=0.0)
    tl.store(B_ptr + offsets,vals, mask=mask)


def solve(a: torch.Tensor, b: torch.Tensor, N: int):
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(N*N, BLOCK_SIZE), )
    matrix_copy_kernel_1d[grid](
        a, b, N*N, BLOCK_SIZE
    )

def run_kernel():
    N = 2048
    a = torch.randn((N, N)).cuda()
    b = torch.randn((N, N)).cuda()
    solve_block(a, b, N)
    solve(a, b, N)


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['matrix_size'],
        x_vals=[2 ** i for i in range(6, 15, 1)],  # 64x64 to 2048x2048
        x_log=True,
        line_arg='provider',
        line_vals=['torch', 'solve_block', 'solve_1d'],
        line_names=['PyTorch', 'Triton Block (2D)', 'Triton 1D'],
        styles=[('green', '-'), ('blue', '-'), ('red', '-')],
        ylabel='GB/s',
        plot_name='matrix-copy-performance',
        args={},
    ))
def benchmark(matrix_size, provider):
    """
    Benchmark matrix copy operations with different implementations.
    matrix_size: N where the matrix is N x N
    """
    N = matrix_size
    a = torch.randn((N, N), device='cuda', dtype=torch.float32)
    b = torch.randn((N, N), device='cuda', dtype=torch.float32)
    quantiles = [0.5, 0.2, 0.8]
    
    if provider == 'torch':
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: torch_solve(a, b, N), 
            quantiles=quantiles
        )
    elif provider == 'solve_block':
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: solve_block(a, b, N), 
            quantiles=quantiles
        )
    elif provider == 'solve_1d':
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: solve(a, b, N), 
            quantiles=quantiles
        )
    
    # 计算吞吐量: 读一次 + 写一次 = 2 * N * N * 4 bytes (float32)
    gbps = lambda ms: 2 * N * N * 4 / ms * 1e-6
    return gbps(ms), gbps(max_ms), gbps(min_ms)


if __name__ == "__main__":
    # Uncomment to run basic kernel test
    # run_kernel()
    
    # Run benchmark
    benchmark.run(print_data=True, show_plots=False)

'''
matrix-copy-performance:
   matrix_size     PyTorch  Triton Block (2D)   Triton 1D
0         64.0    8.000000           1.777778    6.400000
1        128.0   16.000000           6.095238   25.600001
2        256.0  102.400003          17.655172   85.333330
3        512.0  256.000001          66.064517  256.000001
4       1024.0  512.000001         186.181817  481.882344
5       2048.0  630.153853         306.242991  630.153853
6       4096.0  672.164101         300.623852  675.628891
7       8192.0  683.556697         291.271116  683.556697
8      16384.0  687.365465         289.182578  688.493782
'''