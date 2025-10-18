'''
Implement a program that performs a 1D convolution operation. Given an input array and a kernel (filter), compute the convolved output. The convolution should be performed with a "valid" boundary condition, meaning the kernel is only applied where it fully overlaps with the input.

The input consists of two arrays:

input: A 1D array of 32-bit floating-point numbers.
kernel: A 1D array of 32-bit floating-point numbers representing the convolution kernel.
The output should be written to the output array, which will have a size of input_size - kernel_size + 1.
'''

import torch
import triton
import triton.language as tl
import torch.nn.functional as F
from triton.language import dtype


# input, kernel, output are tensors on the GPU
def torch_1d(input: torch.Tensor, kernel: torch.Tensor, output: torch.Tensor, input_size: int, kernel_size: int):
    """
    Computes 1D convolution using PyTorch's native function.
    'output' is an out-parameter to store the result.
    """
    output.copy_(F.conv1d(input.view(1,1,input_size), kernel.view(1,1,kernel_size), stride=1, padding=0).flatten())

@triton.jit
def conv1d_kernel_1(
        input, kernel, output,
        input_size, kernel_size,
        BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    output_size = input_size - kernel_size + 1
    mask = offsets < output_size

    acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for k in range(kernel_size):
        k_val = tl.load(kernel + k)
        input_ =tl.load(input + offsets + k, mask=mask, other=0.0)
        acc += k_val * input_
    tl.store(output + offsets, acc, mask=mask)



# input, kernel, output are tensors on the GPU
def solve_1(input: torch.Tensor, kernel: torch.Tensor, output: torch.Tensor, input_size: int, kernel_size: int):
    BLOCK_SIZE = 1024
    n_blocks = triton.cdiv(input_size - kernel_size + 1, BLOCK_SIZE)
    grid = (n_blocks,)

    conv1d_kernel_1[grid](
        input, kernel, output,
        input_size, kernel_size,
        BLOCK_SIZE
    )

# --------------------------------------------

@triton.jit
def conv1d_kernel_2(
        input, kernel, output,
        input_size, kernel_size,
        BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    output_size = input_size - kernel_size + 1
    mask = offsets < output_size

    acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for k in range(kernel_size):
        k_val = tl.load(kernel + k, cache_modifier='.ca') # cache in L2 and global memory
        input_ =tl.load(input + offsets + k, mask=mask, other=0.0)
        acc += k_val * input_
    tl.store(output + offsets, acc, mask=mask, cache_modifier='.cs')

# input, kernel, output are tensors on the GPU
def solve_2(input: torch.Tensor, kernel: torch.Tensor, output: torch.Tensor, input_size: int, kernel_size: int):
    BLOCK_SIZE = 512
    n_blocks = triton.cdiv(input_size - kernel_size + 1, BLOCK_SIZE)
    grid = (n_blocks,)

    conv1d_kernel_2[grid](
        input, kernel, output,
        input_size, kernel_size,
        BLOCK_SIZE
    )

@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['input_size'],
        x_vals=[2 ** i for i in range(10, 20, 1)],
        x_log=True,
        line_arg='provider',
        line_vals=['torch', 'triton1', 'triton2'],
        line_names=['Torch', 'Triton1', 'Triton2'],
        styles=[('blue', '-'), ('green', '-'), ('red', '-')],
        ylabel='GFLOP/s',
        plot_name='1d-convolution-performance',
        args={'kernel_size': 32},
    ))
def benchmark(input_size, kernel_size, provider):
    input_data = torch.randn((input_size,), dtype=torch.float, device='cuda')
    kernel = torch.randn((kernel_size,), dtype=torch.float, device='cuda')
    output_size = input_size - kernel_size + 1
    quantiles = [0.5, 0.2, 0.8]
    
    if provider == 'torch':
        output = torch.zeros((output_size,), dtype=torch.float, device='cuda')
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: torch_1d(input_data, kernel, output, input_size, kernel_size),
            quantiles=quantiles
        )
    if provider == 'triton1':
        output = torch.zeros((output_size,), dtype=torch.float, device='cuda')
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: solve_1(input_data, kernel, output, input_size, kernel_size),
            quantiles=quantiles
        )

    if provider == 'triton2':
        output = torch.zeros((output_size,), dtype=torch.float, device='cuda')
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: solve_2(input_data, kernel, output, input_size, kernel_size),
            quantiles=quantiles
        )
    
    # 每个输出元素需要 kernel_size 次乘法和 kernel_size-1 次加法
    # 总计算量：output_size * kernel_size * 2 FLOPs
    gflops = lambda ms: output_size * kernel_size * 2 / ms * 1e-6
    return gflops(ms), gflops(min_ms), gflops(max_ms)

def run():
    input_size, kernel_size = 2048, 32
    input_data = torch.randn((input_size, ), dtype=torch.float, device='cuda')
    kernel = torch.randn((kernel_size, ), dtype=torch.float, device='cuda')
    max_errors = 1e-3

    output1 = torch.zeros((input_size - kernel_size + 1,), dtype=torch.float, device='cuda')
    torch_1d(input_data, kernel, output1, input_size, kernel_size)

    output2 = torch.zeros((input_size - kernel_size + 1,), dtype=torch.float, device='cuda')
    solve_2(input_data, kernel, output2, input_size, kernel_size)

    # 验证正确性
    error = torch.abs(output1 - output2).max().item()
    print(f"Maximum error between Triton and PyTorch: {error}")
    assert error < max_errors, f"Error {error} exceeds threshold {max_errors}"
    print("✓ Correctness verification passed!")

if __name__ == "__main__":
    benchmark.run(print_data=True, show_plots=False)

'''
1d-convolution-performance:
   input_size        Torch      Triton1      Triton2
0      1024.0     6.206250     8.866071    12.412500
1      2048.0    15.757813    18.008928    21.010416
2      4096.0    29.944754    36.294642    42.343749
3      8192.0    56.673609    72.866070    85.010414
4     16384.0   113.562496   146.008927   170.343744
5     32768.0   227.340270   292.294639   341.010405
6     65536.0   409.406260   511.757814   584.866063
7    131072.0   630.004815   819.006271   910.006912
8    262144.0   862.213854  1170.147305  1170.147305
9    524288.0  1023.939456  1424.611381  1424.611381
'''