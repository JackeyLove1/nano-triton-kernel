'''
# Swish-Gated Linear Unit (SwiGLU) — Forward Pass (1D)

Implement the SwiGLU activation function forward pass for **1D input vectors**.
Given an input tensor of shape **[N]** (where *N* is the number of elements), compute the output using the elementwise formula.
Both the input and output tensors must be of type **`float32`**.

## Definition

1. **Split** input (x) into two halves: (x_1) and (x_2).
2. **Compute SiLU** on the first half:
   $$
   \operatorname{SiLU}(x_1)=x_1 \cdot \sigma(x_1), \qquad
   \sigma(x)=\frac{1}{1+e^{-x}}
   $$
3. **Compute SwiGLU output**:
   $\operatorname{SwiGLU}(x_1,x_2)=\operatorname{SiLU}(x_1)\cdot x_2$

## Implementation Requirements

* Use only native features (no external libraries).
* The `solve` function signature must remain unchanged.
* The final result must be stored in the `output` tensor.
'''

import torch
import triton
import triton.language as tl

def torch_solve(input: torch.Tensor, output: torch.Tensor, N: int):
    x1, x2 = input[:int(N / 2)], input[int(N / 2):]
    torch.mul(x1 * torch.nn.functional.sigmoid(x1), x2, out=output)

@triton.jit
def swiglu(
    input_ptr, output_ptr, N: tl.constexpr, BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    H = N // 2
    mask = offsets < H
    x_1 = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    x_2 = tl.load(input_ptr + offsets + H, mask=mask, other=0.0)
    # x_silu = x_1 * tl.sigmoid(x_1)
    x_silu = 1 / (1 + tl.exp(-x_1)) * x_1
    y = x_silu * x_2
    tl.store(output_ptr + offsets, y, mask=mask)

# input, output are tensors on the GPU
def solve(input: torch.Tensor, output: torch.Tensor, N: int):
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(N//2, BLOCK_SIZE),)
    swiglu[grid](
        input, output, N, BLOCK_SIZE=BLOCK_SIZE
    )

# for testing performance
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
        plot_name='swiglu-performance',  # Name for the plot
        args={},  # Values for function arguments not in `x_names` and `y_name`
    ))
def benchmark(size, provider):
    input_data = torch.rand(size, device='cuda', dtype=torch.float)
    output = torch.empty(size // 2, device='cuda', dtype=torch.float)
    quantiles = [0.5, 0.2, 0.8]
    if provider == 'torch':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: torch_solve(input_data, output, size), quantiles=quantiles)
    if provider == 'triton':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: solve(input_data, output, size), quantiles=quantiles)
    # GB/s = (input_size + input_size + output_size) * element_size / time_in_ms
    # For SwiGLU: read 2*N floats (N//2 for x1, N//2 for x2), write N//2 floats
    gbps = lambda ms: 2 * input_data.numel() * input_data.element_size() / ms * 1e-6
    return gbps(ms), gbps(max_ms), gbps(min_ms)

if __name__ == '__main__':
    benchmark.run(print_data=True, show_plots=False)

'''
swiglu-performance:
           size      Triton       Torch
0        4096.0    8.000000    3.362890
1        8192.0   16.000000    7.111111
2       16384.0   25.600001   12.190476
3       32768.0   51.200001   26.859017
4       65536.0  102.400003   56.888887
5      131072.0  170.666661   93.090908
6      262144.0  339.564767  157.538463
7      524288.0  455.111095  256.000001
8     1048576.0  630.153853  341.333321
9     2097152.0  744.727267  303.407407
10    4194304.0  819.200021  318.135927
11    8388608.0  873.813292  327.680008
12   16777216.0  897.753389  336.082050
13   33554432.0  910.222190  340.889477
14   67108864.0  916.587429  344.021006
15  134217728.0  919.803550  345.323898
'''