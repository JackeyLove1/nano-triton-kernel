'''
Implement a program that performs the Rectified Linear Unit (ReLU) activation function on a vector of 32-bit floating point numbers. The ReLU function sets all negative values to zero and leaves positive values unchanged:
ReLU(x) = max(0, x)
Implementation Requirements
External libraries are not permitted
The solve function signature must remain unchanged
The final result must be stored in output
'''

import torch
import triton
import triton.language as tl

# torch relu
def torch_solve(input: torch.Tensor, output: torch.Tensor, N: int):
    output.copy_(input.relu_())

@triton.jit
def relu_kernel(input, output, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(input + offsets, mask=mask, other=0.0)
    x = tl.where(x > 0, x, 0.0)
    tl.store(output + offsets, x, mask=mask)


# input, output are tensors on the GPU
def solve(input: torch.Tensor, output: torch.Tensor, N: int):
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    relu_kernel[grid](input, output, N, BLOCK_SIZE)

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
        plot_name='relu-performance',
        args={},
    ))
def benchmark(size, provider):
    x = torch.rand(size, device='cuda', dtype=torch.float)
    output = torch.empty_like(x)
    quantiles = [0.5, 0.2, 0.8]
    if provider == 'torch':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: torch_solve(x, output, size), quantiles=quantiles)
    if provider == 'triton':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: solve(x, output, size), quantiles=quantiles)
    gbps = lambda ms: 2 * x.numel() * x.element_size() / ms * 1e-6
    return gbps(ms), gbps(max_ms), gbps(min_ms)

if __name__ == '__main__':
    benchmark.run(print_data=True, show_plots=False)

'''
relu-performance:
           size      Triton       Torch
0        4096.0    8.000000    4.571429
1        8192.0   16.000000   10.666666
2       16384.0   32.000000   18.285714
3       32768.0   51.848103   36.571428
4       65536.0  102.400003   73.142856
5      131072.0  170.666661  128.000000
6      262144.0  292.571425  204.800005
7      524288.0  455.111095  315.076927
8     1048576.0  512.000001  327.680008
9     2097152.0  585.142849  303.407407
10    4194304.0  642.509816  321.254908
11    8388608.0  661.979817  332.670049
12   16777216.0  675.628891  338.687332
13   33554432.0  682.666643  341.333321
14   67108864.0  685.343787  342.896010
15  134217728.0  687.140246  343.457587
'''