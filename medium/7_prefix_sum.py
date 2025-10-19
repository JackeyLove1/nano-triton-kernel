'''
Write a CUDA program that computes the prefix sum (cumulative sum) of an array of 32-bit floating point numbers. For an input array [a, b, c, d, ...], the prefix sum is [a, a+b, a+b+c, a+b+c+d, ...].

Implementation Requirements
Use only CUDA native features (external libraries are not permitted)
The solve function signature must remain unchanged
The result must be stored in the output array
'''

import torch
import triton
import triton.language as tl

def torch_solve(input: torch.Tensor, output: torch.Tensor, N: int):
    output.copy_(input.cumsum(dim=-1))


# -----------------------------------------------------------
@triton.jit
def prefix_sum_stage1(
        input, output, block_sum_ptr,
        N: tl.constexpr,
        BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N

    block_data = tl.load(input + offsets, mask=mask, other=0)
    block_cusum = tl.cumsum(block_data, axis=0)
    tl.store(output+offsets,block_cusum, mask=mask)

    block_sum = tl.sum(block_data, axis=0)
    tl.store(block_sum_ptr + pid, block_sum)

@triton.jit
def prefix_sum_stage2(
        input, output, block_sum_ptr,
        N: tl.constexpr,
        BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    if pid > 0:
        block_data = tl.load(output + offsets, mask=mask, other=0.0)
        block_prev_sum = tl.load(block_sum_ptr + pid - 1)
        block_prefix_sum = block_data + block_prev_sum
        tl.store(output + offsets,block_prefix_sum, mask=mask)


def solve(data: torch.Tensor, output: torch.Tensor, N: int):
    BLOCK_SIZE = 1024
    GRID_SIZE = triton.cdiv(N, BLOCK_SIZE)
    grid = ((GRID_SIZE, ))

    block_sum_ptr = torch.empty(GRID_SIZE, dtype=data.dtype, device=data.device)

    # stage1: calculate the block cusum
    prefix_sum_stage1[grid](
        data, output, block_sum_ptr, N, BLOCK_SIZE
    )

    block_sum_ptr.cumsum_(dim=0)

    # stage2 calculate the data cusum
    prefix_sum_stage2[grid](
        data, output, block_sum_ptr, N, BLOCK_SIZE
    )

def run():
    # 创建测试数据
    N = 100_000  # 测试一个不被 1024 整除的数来验证掩码的正确性
    input_tensor = torch.randn(N, dtype=torch.float32, device='cuda')

    # 分配输出张量
    output_triton = torch.empty_like(input_tensor)
    output_torch = torch.empty_like(input_tensor)

    # 执行 Triton 计算
    print("Running Triton implementation...")
    solve(input_tensor, output_triton, N)

    # 执行 PyTorch 参考计算
    print("Running PyTorch reference implementation...")
    torch_solve(input_tensor, output_torch, N)

    # 验证结果
    print("Verifying results...")
    is_correct = torch.allclose(output_triton, output_torch, atol=1e-4, rtol=1e-4)
    print(f"Are the results correct? {'✅ Yes' if is_correct else '❌ No'}")

    # 打印一些结果进行目视检查
    print("\nInput tensor (first 10 elements):")
    print(input_tensor[:10])
    print("\nTriton output (first 10 elements):")
    print(output_triton[:10])
    print("\nPyTorch output (first 10 elements):")
    print(output_torch[:10])

# -----------------------------------------------------------
def prefix_sum(x: torch.Tensor) -> torch.Tensor:
    """Wrapper function for prefix sum computation"""
    N = x.numel()
    output = torch.empty_like(x, dtype=x.dtype, device=x.device)
    solve(x, output, N)
    return output

@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['size'],
        x_vals=[2 ** i for i in range(12, 28, 1)],
        x_log=True,
        line_arg='provider',  # 参数名称，其值对应于绘图中的不同线条
        line_vals=['triton', 'torch'],  # `line_arg` 的可能值
        line_names=['Triton', 'Torch'],  # 线条的标签名称
        styles=[('blue', '-'), ('green', '-')],  # 线条样式
        ylabel='GB/s',  # y 轴标签名称
        plot_name='prefix-sum-performance',  # 绘图名称
        args={},  # 不在 `x_names` 和 `y_name` 中的函数参数值
    ))
def benchmark(size, provider):
    x = torch.rand(size, device='cuda', dtype=torch.float)
    quantiles = [0.5, 0.2, 0.8]
    if provider == 'torch':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: torch_solve(x, torch.empty_like(x), x.numel()), quantiles=quantiles)
    if provider == 'triton':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: prefix_sum(x), quantiles=quantiles)
    # GB/s = (input + output) * element_size / time_in_ms * 1e-6
    gbps = lambda ms: 2 * x.numel() * x.element_size() / ms * 1e-6
    return gbps(ms), gbps(max_ms), gbps(min_ms)

if __name__ == "__main__":
    run()
    print("================= performence ==================")
    benchmark.run(print_data=True, show_plots=False)

'''
prefix-sum-performance:
           size      Triton       Torch
0        4096.0    2.909091    3.555555
1        8192.0    6.400000    7.111111
2       16384.0   12.800000   14.222222
3       32768.0   25.600001   28.444443
4       65536.0   46.545454   46.545454
5      131072.0   85.333330   93.090908
6      262144.0  157.538463  157.538463
7      524288.0  256.000001  256.000001
8     1048576.0  327.680008  303.407407
9     2097152.0  282.482752  292.571425
10    4194304.0  309.132077  318.135927
11    8388608.0  326.049747  329.326630
12   16777216.0  334.367358  336.082050
13   33554432.0  339.564767  338.701010
14   67108864.0  341.778349  341.111268
15  134217728.0  342.783915  341.666983
'''