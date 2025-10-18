import torch
import triton
import triton.language as tl

# input, output are tensors on the GPU
def torch_solve(input: torch.Tensor, output: torch.Tensor, N: int, M: int, K: int):
    output.copy_((input.flatten() == K).sum(dim=0))

@triton.jit
def count_array_1d(
        input_ptr, output_ptr,
        N, M,
        K: tl.constexpr,
        BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N * M
    values = tl.load(input_ptr + offsets, mask=mask)
    matches = tl.where(values == K, 1, 0)
    count = tl.sum(matches, axis=0)
    if count != 0:
        tl.atomic_add(output_ptr, count, sem="relaxed")

def solve_1d(input: torch.Tensor, output: torch.Tensor, N: int, M: int, K: int):
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(N * M, BLOCK_SIZE), )
    count_array_1d[grid](input, output, N, M, K, BLOCK_SIZE)

@triton.jit
def count_array_block(
        input_ptr, output_ptr,
        N, M,
        K,
        stride_an, stride_am,
        BLOCK_SIZE: tl.constexpr
):
    pid_n = tl.program_id(0)
    pid_m = tl.program_id(1)
    rows_ = pid_n * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    cols_ = pid_m * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    input_tile_ptr = input_ptr + rows_[:, None] * stride_an + cols_[None, :] * stride_am
    mask = (rows_[:, None] < N) & (cols_[None, :] < M)
    values = tl.load(input_tile_ptr, mask=mask, other=0.0)
    matches = values == K
    block_count = tl.sum(matches, axis=1) # [Block]
    block_count = tl.sum(block_count, axis=0) # scalar

    if block_count != 0:
        tl.atomic_add(output_ptr, block_count, sem="relaxed")

def solve_block(input: torch.Tensor, output: torch.Tensor, N: int, M: int, K: int):
    BLOCK_SIZE = 64
    grid = (triton.cdiv(N, BLOCK_SIZE), triton.cdiv(M, BLOCK_SIZE))
    count_array_block[grid](
        input, output, N, M, K, input.stride(0), input.stride(1), BLOCK_SIZE
    )


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['total_size'],
        x_vals=[2 ** i for i in range(16, 25, 1)],
        x_log=True,
        line_arg='provider',
        line_vals=['triton_1d', 'triton_block', 'torch'],
        line_names=['Triton 1D', 'Triton Block', 'PyTorch'],
        styles=[('blue', '-'), ('green', '-'), ('red', '-')],
        ylabel='GB/s',
        plot_name='2d-array-count-performance',
        args={},
    ))
def benchmark(total_size, provider):
    # Create 2D matrix with size sqrt(total_size) x sqrt(total_size)
    size = int(total_size ** 0.5)
    N, M = size, size
    K = 5
    
    input_data = torch.randint(0, 10, (N, M), device='cuda', dtype=torch.int32)
    output = torch.zeros(1, device='cuda', dtype=torch.int32)
    
    quantiles = [0.5, 0.2, 0.8]
    
    if provider == 'torch':
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: torch_solve(input_data, output, N, M, K), 
            quantiles=quantiles
        )
    elif provider == 'triton_1d':
        output.zero_()
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: solve_1d(input_data, output, N, M, K), 
            quantiles=quantiles
        )
    elif provider == 'triton_block':
        output.zero_()
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: solve_block(input_data, output, N, M, K), 
            quantiles=quantiles
        )
    
    # Calculate throughput in GB/s
    # Reading N*M elements of int32 (4 bytes each)
    gbps = lambda ms: N * M * 4 / ms * 1e-6
    return gbps(ms), gbps(max_ms), gbps(min_ms)

if __name__ == '__main__':
    benchmark.run(print_data=True, show_plots=False)

'''
2d-array-count-performance:
   total_size   Triton 1D  Triton Block     PyTorch
0     65536.0   25.600001     51.200001   10.039216
1    131072.0   85.315101     73.127231   28.438367
2    262144.0  102.400003    146.285712   48.761905
3    524288.0  170.630202    255.945313   75.835648
4   1048576.0  256.000001    341.333321   83.591839
5   2097152.0  372.284088    409.512510   95.235465
6   4194304.0  468.114273    512.000001  108.503311
7   8388608.0  590.288282    606.685184  117.422939
8  16777216.0  661.979817    655.360017  122.268655
'''


