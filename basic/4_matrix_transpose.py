'''
Write a program that transposes a matrix of 32-bit floating point numbers on a GPU. The transpose of a matrix switches its rows and columns. Given a matrix A
 of dimensions rows x cols, the transpose A.T will have dimensions cols x rows. All matrices are stored in row-major format.

Implementation Requirements
Use only native features (external libraries are not permitted)
The solve function signature must remain unchanged
The final result must be stored in the matrix output
'''

import torch
import triton
import triton.language as tl
from torch import Tensor

def matrix_transpose_torch(input: Tensor, output: Tensor):
    output.copy_(input.T)

# 1D dimension version
@triton.jit
def matrix_transpose_kernel_1d(
        input, output,
        rows, cols,
        stride_ir, stride_ic,
        stride_or, stride_oc
):
    pid_row = tl.program_id(axis=0)
    pid_col = tl.program_id(axis=1)

    mask = (pid_row < rows) & (pid_col < cols)

    input_offset = pid_row * stride_ir + pid_col * stride_ic
    input_value = tl.load(input + input_offset, mask=mask, other=0)

    output_offsets = pid_row * stride_oc + pid_col * stride_or
    tl.store(output + output_offsets,input_value,mask=mask)


# input, output are tensors on the GPU
def solve_1d(input: torch.Tensor, output: torch.Tensor, rows: int, cols: int):
    stride_ir, stride_ic = cols, 1
    stride_or, stride_oc = rows, 1

    grid = (rows, cols)
    matrix_transpose_kernel_1d[grid](
        input, output,
        rows, cols,
        stride_ir, stride_ic,
        stride_or, stride_oc
    )


@triton.jit
def matrix_transpose_kernel_triton_block(
        input, output,
        rows, cols,
        stride_ir, stride_ic,
        stride_or, stride_oc,
        BLOCK_SIZE: tl.constexpr
):
    pid_row = tl.program_id(axis=0)
    pid_col = tl.program_id(axis=1)

    rows_ = pid_row * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    cols_ = pid_col * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    input_mask = (rows_[:, None] < rows) & (cols_[None, :] < cols)
    output_mask = (cols_[:, None] < cols) & (rows_[None,:] < rows)

    input_offsets = input + rows_[:, None] * stride_ir + cols_[None, :] * stride_ic
    output_offsets = output + cols_[:, None] * stride_or + rows_[None, :] * stride_oc

    block = tl.load(input_offsets, mask=input_mask)
    tl.store(output_offsets, tl.trans(block), mask=output_mask)

# input, output are tensors on the GPU
def solve_triton_block(input: torch.Tensor, output: torch.Tensor, rows: int, cols: int):
    stride_ir, stride_ic = cols, 1
    stride_or, stride_oc = rows, 1

    BLOCK_SIZE = 64
    grid = (triton.cdiv(rows, BLOCK_SIZE), triton.cdiv(cols, BLOCK_SIZE))
    matrix_transpose_kernel_triton[grid](
        input, output,
        rows, cols,
        stride_ir, stride_ic,
        stride_or, stride_oc,
        BLOCK_SIZE=BLOCK_SIZE
    )


# 性能基准测试
@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['size'],
        x_vals=[2 ** i for i in range(8, 15, 1)],
        x_log=True,
        line_arg='provider',
        line_vals=['triton_1d', 'torch', 'triton_block'],
        line_names=['Triton 1D', 'Torch', 'Triton Block'],
        styles=[('blue', '-'), ('green', '-'), ('red', '-')],
        ylabel='GB/s',
        plot_name='matrix-transpose-performance',
        args={},
    ))
def benchmark(size, provider):
    # 创建方阵 (size x size)
    input_matrix = torch.rand(size, size, device='cuda', dtype=torch.float32)
    output_matrix = torch.empty_like(input_matrix)
    
    quantiles = [0.5, 0.2, 0.8]
    if provider == 'torch':
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: matrix_transpose_torch(input_matrix, output_matrix), 
            quantiles=quantiles
        )
    if provider == 'triton_1d':
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: solve_1d(input_matrix, output_matrix, size, size),
            quantiles=quantiles
        )
    if provider == 'triton_block':
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: solve_triton_block(input_matrix, output_matrix, size, size),
            quantiles=quantiles
        )
    
    # 计算带宽：2次内存操作(读+写) × 矩阵元素个数 × 元素大小 / 时间
    gbps = lambda ms: 2 * input_matrix.numel() * input_matrix.element_size() / ms * 1e-6
    return gbps(ms), gbps(max_ms), gbps(min_ms)

if __name__ == '__main__':
    benchmark.run(print_data=True, show_plots=False)

'''
matrix-transpose-performance:
      size  Triton 1D       Torch  Triton Block
0    256.0  10.448980   85.333330     85.333330
1    512.0  13.044586  227.555548    292.571425
2   1024.0  13.191626  372.363633    546.133347
3   2048.0  13.271770  414.784823    642.509816
4   4096.0  13.283875  278.876596    675.628891
5   8192.0  13.168765  259.291787    682.666643
6  16384.0  13.171907  262.472095    684.672534
'''