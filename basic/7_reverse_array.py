'''
Implement a program that reverses an array of 32-bit floating point numbers in-place. The program should perform an in-place reversal of input.

Implementation Requirements
Use only native features (external libraries are not permitted)
The solve function signature must remain unchanged
The final result must be stored back in input
'''
import torch
import triton
import triton.language as tl
from torch import Tensor

# torch base benchmark
def torch_solve(input: torch.Tensor, N: int):
    return input.flip(dims=[0])


@triton.jit
def reverse_kernel(
        input,
        N,
        BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    left_offsets = BLOCK_SIZE * pid + tl.arange(0, BLOCK_SIZE)
    right_offsets = N - left_offsets - 1
    half = N // 2
    mask = left_offsets < half

    left_ptr = input + left_offsets
    right_ptr = input + right_offsets

    left_value = tl.load(left_ptr, mask=mask, other=0.0)
    right_value = tl.load(right_ptr, mask=mask, other=0.0)

    tl.store(left_ptr, right_value, mask=mask)
    tl.store(right_ptr, left_value, mask=mask)


# input is a tensor on the GPU
def solve(input: torch.Tensor, N: int):
    BLOCK_SIZE = 1024
    n_blocks = triton.cdiv(N // 2, BLOCK_SIZE)
    grid = (n_blocks,)

    reverse_kernel[grid](
        input,
        N,
        BLOCK_SIZE
    )
    return input


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
        plot_name='reverse-array-performance',  # 绘图名称。也用作保存绘图的文件名
        args={},  # 不在 `x_names` 和 `y_name` 中的函数参数值
    ))
def benchmark(size, provider):
    input_data = torch.rand(size, device='cuda', dtype=torch.float)
    quantiles = [0.5, 0.2, 0.8]
    if provider == 'torch':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: torch_solve(input_data.clone(), size), quantiles=quantiles)
    if provider == 'triton':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: solve(input_data.clone(), size), quantiles=quantiles)
    # 反向操作需要读取所有元素和写入所有元素，所以是 2 * numel()
    gbps = lambda ms: 2 * input_data.numel() * input_data.element_size() / ms * 1e-6
    return gbps(ms), gbps(max_ms), gbps(min_ms)


if __name__ == '__main__':
    benchmark.run(print_data=True, show_plots=False)

'''
reverse-array-performance:
           size      Triton       Torch
0        4096.0    4.571429    4.571429
1        8192.0   10.666666   10.666666
2       16384.0   21.333333   18.285714
3       32768.0   36.571428   36.571428
4       65536.0   73.142856   64.000000
5      131072.0  128.000000  128.000000
6      262144.0  204.800005  204.800005
7      524288.0  292.571425  292.571425
8     1048576.0  341.333321  273.066674
9     2097152.0  315.076927  292.571425
10    4194304.0  327.680008  318.135927
11    8388608.0  334.367358  330.989909
12   16777216.0  337.814445  336.946021
13   33554432.0  339.564767  341.333321
14   67108864.0  340.446760  342.896010
15  134217728.0  340.446760  343.795416
'''