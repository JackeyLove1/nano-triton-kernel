'''
# Softmax on GPU — Problem Statement

Write a program that computes the softmax function for an array of 32-bit floating-point numbers on a GPU. The softmax function is defined as follows:
For an input array ( \mathbf{x} ) of length ( n ), the softmax of ( \mathbf{x} ), denoted ( \sigma(\mathbf{x}) ), is an array of length ( n ) where the ( i )-th element is:
$$\sigma(\mathbf{x})*i ;=; \frac{e^{x_i}}{\sum*{j=1}^{n} e^{x_j}}$$
Your solution should handle potential overflow issues by using the “max trick”: **subtract the maximum value of the input array from each element before exponentiation.**

## Implementation Requirements
* Use only native features (**external libraries are not permitted**).
* The `solve` function signature **must remain unchanged**.
* The final result **must be stored** in the array `output`.
'''

import torch
import triton
import triton.language as tl

# input, output are tensors on the GPU
def torch_solve(input: torch.Tensor, output: torch.Tensor, N: int):
    max = torch.max(input)
    torch.exp(input - max, out=output)
    sum = torch.sum(output)
    torch.div(output, sum, out=output)

@triton.jit
def max_kernel(
        input, output,
        N: tl.constexpr,
        BLOCK_SIZE: tl.constexpr
):
    input = input.to(tl.pointer_type(tl.float32))
    output = output.to(tl.pointer_type(tl.float32))
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N

    # Use -inf for OOB to avoid contaminating block max when values are negative
    x = tl.load(input + offsets, mask=mask, other=-float('inf'))
    x_max = tl.max(x, axis=0)
    # Write per-program max to output[pid]; reduce on host later
    tl.store(output + pid, x_max)

@triton.jit
def exp_sum_kernel(
        input, output,
        input_max,
        N: tl.constexpr,
        BLOCK_SIZE: tl.constexpr
):
    input = input.to(tl.pointer_type(tl.float32))
    output = output.to(tl.pointer_type(tl.float32))
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N

    # Use -inf for OOB so exp(-inf) -> 0 and doesn't affect the sum
    x = tl.load(input + offsets, mask=mask, other=-float('inf'))
    x = x - input_max
    x_exp = tl.exp(x)
    x_sum = tl.sum(x_exp, axis=0)
    tl.atomic_add(output, x_sum)



@triton.jit
def softmax_kernel(
    input, output,
    input_max,
    softmax_sum,
    N: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    input = input.to(tl.pointer_type(tl.float32))
    output = output.to(tl.pointer_type(tl.float32))
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N

    # Use -inf for OOB to keep intermediate exp values at 0
    x = tl.load(input + offsets, mask=mask, other=-float('inf'))
    x = x - input_max
    x_exp = tl.exp(x)
    y = x_exp / softmax_sum
    tl.store(output + offsets, y, mask=mask)


# input, output are tensors on the GPU
def solve(input: torch.Tensor, output: torch.Tensor, N: int):
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(N, BLOCK_SIZE), )
    # Stage 1: per-program maxima, then reduce on host
    block_max = torch.empty(grid[0], device=input.device, dtype=torch.float32)
    max_kernel[grid](input, block_max, N, BLOCK_SIZE)
    input_max = torch.max(block_max)
    softmax_sum = torch.tensor(0, device=input.device, dtype=torch.float32)
    exp_sum_kernel[grid](input, softmax_sum, input_max.item(), N, BLOCK_SIZE)
    softmax_kernel[grid](input, output, input_max.item(), softmax_sum.item(), N, BLOCK_SIZE)

def softmax_triton(x: torch.Tensor) -> torch.Tensor:
    """Wrapper function for Triton softmax implementation"""
    N = x.shape[0]
    output = torch.empty_like(x, dtype=x.dtype, device=x.device)
    solve(x, output, N)
    return output

# ---------------------------------------------

@triton.jit
def reduce_max_kernel(input_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    values = tl.load(input_ptr + offsets, mask=mask, other=-float('inf'))
    tl.atomic_max(output_ptr, tl.max(values, axis=0))

@triton.jit
def reduce_sum_kernel(input_ptr, max_val_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    max_val = tl.load(max_val_ptr)
    values = tl.load(input_ptr + offsets, mask=mask, other=-float('inf'))
    exp_values = tl.exp(values - max_val)
    tl.atomic_add(output_ptr, tl.sum(exp_values, axis=0))

@triton.jit
def softmax_write_kernel(input_ptr, output_ptr, max_val_ptr, sum_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    max_val = tl.load(max_val_ptr)
    denom = tl.load(sum_ptr)

    values = tl.load(input_ptr + offsets, mask=mask, other=-float('inf'))
    exp_values = tl.exp(values - max_val)
    softmax = exp_values / denom
    tl.store(output_ptr + offsets, softmax, mask=mask)


# input, output are tensors on the GPU
def solve2(input: torch.Tensor, output: torch.Tensor, N: int):
    BLOCK_SIZE = 1024
    num_blocks = (N + BLOCK_SIZE - 1) // BLOCK_SIZE

    # Step 1: Block-wise max reduction
    global_max = torch.full((), -float('inf'), device='cuda', dtype=torch.float32)
    reduce_max_kernel[(num_blocks,)](
        input, global_max, N,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    # Step 2: Block-wise sum of exp(x - max)
    global_sum = torch.full((), 0, device='cuda', dtype=torch.float32)
    reduce_sum_kernel[(num_blocks,)](
        input, global_max, global_sum, N,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    # Step 3: Final softmax write
    softmax_write_kernel[(num_blocks,)](
        input, output,
        global_max, global_sum, N,
        BLOCK_SIZE=BLOCK_SIZE,
    )

def softmax_triton2(x: torch.Tensor) -> torch.Tensor:
    """Wrapper function for Triton softmax implementation"""
    N = x.shape[0]
    output = torch.empty_like(x, dtype=x.dtype, device=x.device)
    solve2(x, output, N)
    return output

# ------------------------------------------------------
@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['size'],
        x_vals=[2 ** i for i in range(10, 24, 1)],
        x_log=True,
        line_arg='provider',
        line_vals=['torch', 'triton', 'triton2'],
        line_names=['Torch', 'Triton', 'Triton2'],
        styles=[('blue', '-'), ('green', '-'), ('red', '-')],
        ylabel='GB/s',
        plot_name='softmax-performance',
        args={},
    ))
def benchmark(size, provider):
    x = torch.randn(size, device='cuda', dtype=torch.float32)
    quantiles = [0.5, 0.2, 0.8]
    if provider == 'torch':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: torch.softmax(x, dim=0), quantiles=quantiles)
    if provider == 'triton':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: softmax_triton(x), quantiles=quantiles)
    if provider == 'triton2':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: softmax_triton2(x), quantiles=quantiles)
    # 2 * x.numel() * x.element_size() bytes (read input + write output)
    gbps = lambda ms: 2 * x.numel() * x.element_size() / ms * 1e-6
    return gbps(ms), gbps(max_ms), gbps(min_ms)


def run():
    N = 2 ** 10
    input = torch.randn(N, device='cuda', dtype=torch.float32)
    output1 = torch.zeros_like(input)
    output2 = torch.zeros_like(input)
    torch_solve(input, output1, N)
    solve(input, output2, N)
    errors = (output1 - output2).flatten().sum()
    print(f"sum errors: {errors}")
    # print(f"output1: {output1}")
    # print(f"output2: {output2}")

if __name__ == "__main__":
    benchmark.run(print_data=True, show_plots=False)

'''
softmax-performance:
         size      Torch      Triton     Triton2
0      1024.0   1.142857    0.051282    0.666667
1      2048.0   3.200000    0.102564    1.452482
2      4096.0   5.333333    0.237037    2.909091
3      8192.0   9.142857    0.484848    5.333333
4     16384.0  14.222222    0.969238   10.666666
5     32768.0  18.285714    1.939394   21.333333
6     65536.0  22.260869    3.878788   42.666665
7    131072.0  25.600001    7.585185   73.142856
8    262144.0  28.054793   15.170371  136.533337
9    524288.0  29.467627   29.681159  227.555548
10  1048576.0  27.769491   56.888887  273.066674
11  2097152.0  26.214401   97.796678  273.066674
12  4194304.0  26.447135  161.418717  306.242991
13  8388608.0  26.565060  221.405396  329.326630
'''