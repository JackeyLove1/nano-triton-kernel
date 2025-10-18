import torch
import triton
import triton.language as tl
import torch.functional as F

def torch_solve(input: torch.Tensor, output: torch.Tensor, N: int):
    output.copy_(torch.nn.LeakyReLU(0.1)(input))

@triton.jit
def leaky_relu_kernel(
    input,
    output,
    n_elements,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(input + offsets, mask=mask, other=0.0)
    x = tl.where(x > 0, x , 0.01 * x)
    tl.store(output + offsets, x, mask=mask)

# input, output are tensors on the GPU
def solve(input: torch.Tensor, output: torch.Tensor, N: int):
    BLOCK_SIZE = 256
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    leaky_relu_kernel[grid](
        input,
        output,
        N,
        BLOCK_SIZE
    )

@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['size'],
        x_vals=[2 ** i for i in range(12, 28, 1)],
        x_log=True,
        line_arg='provider',  # Argument name whose value corresponds to a different line in the plot
        line_vals=['triton', 'torch'],  # Possible values for `line_arg`
        line_names=['Triton', 'Torch'],  # Label name for the lines
        styles=[('blue', '-'), ('green', '-')],  # Line styles
        ylabel='GB/s',  # Label name for the y-axis
        plot_name='leaky-relu-performance',  # Name for the plot
        args={},  # Values for function arguments not in `x_names` and `y_name`
    ))
def benchmark(size, provider):
    x = torch.rand(size, device='cuda', dtype=torch.float)
    output = torch.empty_like(x)
    quantiles = [0.5, 0.2, 0.8]
    if provider == 'torch':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: torch_solve(x, output, size), quantiles=quantiles)
    if provider == 'triton':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: solve(x, output, size), quantiles=quantiles)
    # Leaky ReLU: read 1 input tensor + write 1 output tensor = 2 memory operations
    gbps = lambda ms: 2 * x.numel() * x.element_size() / ms * 1e-6
    return gbps(ms), gbps(max_ms), gbps(min_ms)

if __name__ == '__main__':
    benchmark.run(print_data=True, show_plots=False)

'''
leaky-relu-performance:
           size      Triton       Torch
0        4096.0    8.000000    4.571429
1        8192.0   16.000000   10.666666
2       16384.0   32.000000   21.333333
3       32768.0   51.200001   36.571428
4       65536.0  102.400003   73.142856
5      131072.0  170.666661  128.000000
6      262144.0  292.571425  204.800005
7      524288.0  409.600010  315.076927
8     1048576.0  512.000001  303.407407
9     2097152.0  585.142849  303.407407
10    4194304.0  630.153853  321.254908
11    8388608.0  655.360017  332.670049
12   16777216.0  672.164101  338.687332
13   33554432.0  677.374664  341.333321
14   67108864.0  683.556697  343.120420
15  134217728.0  687.140246  343.682733
'''