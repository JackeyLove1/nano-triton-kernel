'''
Write a CUDA program that performs parallel reduction on an array of 32-bit floating point numbers to compute their sum. The program should take an input array and produce a single output value containing the sum of all elements.

Implementation Requirements
Use only CUDA native features (external libraries are not permitted)
The solve function signature must remain unchanged
The final result must be stored in the output variable
'''

import torch
import triton
import triton.language as tl


def torch_solve(input: torch.Tensor, output: torch.Tensor, N: int):
    output.copy_(input.sum(dim=0))

@triton.jit
def reduce_sum_kernel(
        input_ptr, output_ptr,
        N: tl.constexpr,
        BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N

    x = tl.load(input_ptr + offsets, mask=mask, other=0)
    y = tl.sum(x, axis=0)
    tl.atomic_add(output_ptr, y, sem="relaxed")

# input, output are tensors on the GPU
def solve(input: torch.Tensor, output: torch.Tensor, N: int):
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(N, BLOCK_SIZE), )
    reduce_sum_kernel[grid](input, output, N, BLOCK_SIZE)

# wrapper function for performance testing
def reduce_sum(x: torch.Tensor) -> torch.Tensor:
    """Compute sum of input tensor using Triton kernel"""
    assert x.is_cuda, "Input tensor must be on GPU"
    output = torch.zeros(1, dtype=x.dtype, device=x.device)
    N = x.numel()
    solve(x, output, N)
    return output

@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['size'],
        x_vals=[2 ** i for i in range(12, 28, 1)],
        x_log=True,
        line_arg='provider',
        line_vals=['triton', 'torch'],
        line_names=['Triton', 'Torch'],
        styles=[('blue', '-'), ('green', '-')],
        ylabel='GB/s',
        plot_name='reduction-performance',
        args={},
    ))
def benchmark(size, provider):
    x = torch.rand(size, device='cuda', dtype=torch.float32)
    quantiles = [0.5, 0.2, 0.8]
    
    if provider == 'torch':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: x.sum(), quantiles=quantiles)
    if provider == 'triton':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: reduce_sum(x), quantiles=quantiles)
    
    # Calculate GB/s: read x.numel() elements, each of element_size bytes
    gbps = lambda ms: x.numel() * x.element_size() / ms * 1e-6
    return gbps(ms), gbps(max_ms), gbps(min_ms)

if __name__ == '__main__':
    benchmark.run(print_data=True, show_plots=False)

'''
reduction-performance:
           size      Triton       Torch
0        4096.0    2.285714    2.666667
1        8192.0    5.333333    5.333333
2       16384.0   10.666666    8.000000
3       32768.0   21.333333   12.800000
4       65536.0   36.571428   17.066667
5      131072.0   73.142856   56.888887
6      262144.0  128.000000  102.400003
7      524288.0  204.800005  170.666661
8     1048576.0  292.571425  273.066674
9     2097152.0  390.095241  356.173905
10    4194304.0  496.484845  468.114273
11    8388608.0  585.142849  564.965503
12   16777216.0  648.871301  636.271854
13   33554432.0  686.240841  672.164101
14   67108864.0  706.498312  699.050661
15  134217728.0  716.240451  699.983980
'''