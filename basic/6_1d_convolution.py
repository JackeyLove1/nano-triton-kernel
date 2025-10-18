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
def conv1d_kernel(
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
def solve(input: torch.Tensor, kernel: torch.Tensor, output: torch.Tensor, input_size: int, kernel_size: int):
    BLOCK_SIZE = 1024
    n_blocks = triton.cdiv(input_size - kernel_size + 1, BLOCK_SIZE)
    grid = (n_blocks,)

    conv1d_kernel[grid](
        input, kernel, output,
        input_size, kernel_size,
        BLOCK_SIZE
    )

def run():
    input_size, kernel_size = 2048, 32
    input_data = torch.randn((input_size, ), dtype=torch.float, device='cuda')
    kernel = torch.randn((kernel_size, ), dtype=torch.float, device='cuda')
    max_errors = 1e-3

    output1 = torch.zeros((input_size - kernel_size + 1,), dtype=torch.float, device='cuda')
    torch_1d(input_data, kernel,output1,input_size, kernel_size)

    output2 = torch.zeros((input_size - kernel_size + 1,), dtype=torch.float, device='cuda')
    solve(input_data, kernel, output2, input_size, kernel_size)


if __name__ == "__main__":
    run()