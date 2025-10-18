'''
Sigmoid Linear Unit
# Implement the SiLU (Sigmoid Linear Unit) activation function forward pass for 1D input vectors
Given an input tensor of shape ([N]) where (N) is the number of elements, compute the output using the elementwise formula.

**SiLU is defined as:**
$\sigma(x) = \frac{1}{1 + e^{-x}}$
$\text{SiLU}(x) = x \cdot \sigma(x)$

Implementation Requirements
* Use only native features (external libraries are not permitted).
* The `solve` function signature must remain unchanged.
* The final result must be stored in the `output` tensor.
'''

import torch
import triton
import triton.language as tl

def torch_solve(input: torch.Tensor, output: torch.Tensor, N: int):
    output.copy_(torch.nn.functional.silu(input))

@triton.jit
def silu_kernel(input_ptr, output_ptr,
                n_elements:tl.constexpr,
                BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    y = x * tl.sigmoid(x)
    tl.store(output_ptr + offsets, y, mask=mask)


# input, output are tensors on the GPU
def solve(input: torch.Tensor, output: torch.Tensor, N: int):
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    silu_kernel[grid](input, output, N, BLOCK_SIZE)

# wrapper function for benchmark
def silu_triton(x: torch.Tensor) -> torch.Tensor:
    """Triton implementation of SiLU"""
    BLOCK_SIZE = 1024
    output = torch.empty_like(x, dtype=x.dtype, device=x.device)
    assert x.is_cuda
    n_elements = x.numel()
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    silu_kernel[grid](x, output, n_elements, BLOCK_SIZE)
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
        plot_name='silu-performance',  # 绘图名称
        args={},  # 不在 `x_names` 和 `y_name` 中的函数参数值
    ))
def benchmark(size, provider):
    x = torch.randn(size, device='cuda', dtype=torch.float)
    quantiles = [0.5, 0.2, 0.8]
    if provider == 'torch':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: torch.nn.functional.silu(x), quantiles=quantiles)
    if provider == 'triton':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: silu_triton(x), quantiles=quantiles)
    # 访存字节数：读取输入(N*4) + 写出输出(N*4) = 2*N*4 字节
    gbps = lambda ms: 2 * x.numel() * x.element_size() / ms * 1e-6
    return gbps(ms), gbps(max_ms), gbps(min_ms)

if __name__ == '__main__':
    benchmark.run(print_data=True, show_plots=False)

'''
silu-performance:
           size      Triton       Torch
0        4096.0    4.000000    6.400000
1        8192.0   12.800000    9.142857
2       16384.0   32.000000   25.600001
3       32768.0   51.200001   39.384613
4       65536.0  102.400003   78.769226
5      131072.0  203.527946  146.285712
6      262144.0  292.571425  256.000001
7      524288.0  455.111095  409.600010
8     1048576.0  512.000001  512.000001
9     2097152.0  585.142849  564.965503
10    4194304.0  630.153853  630.153853
11    8388608.0  661.979817  655.360017
12   16777216.0  672.164101  672.164101
13   33554432.0  680.893520  679.129534
14   67108864.0  685.343787  683.556697
15  134217728.0  686.690249  686.690249
'''