'''
Implement a program that computes the maximum sum of any contiguous subarray of length exactly window_size. You are given an array input of length N consisting of 32-bit signed integers, and an integer window_size.

Implementation Requirements
Use only native features (external libraries are not permitted)
The solve function signature must remain unchanged
The final result must be stored in the output variable
'''

import torch
import triton
import triton.language as tl

def torch_solve(input: torch.Tensor, output: torch.Tensor, N: int, window_size: int):
    out = torch.cumsum(input,dim=0)
    out = torch.max(out[window_size:N] - out[0:N-window_size])
    output.copy_(out)

# input, output are tensors on the GPU

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

@triton.jit
def max_windows_sum_kernel(
        input, output,
        N: tl.constexpr,
        window_size: tl.constexpr,
        BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = (offsets +  window_size) < N

    window_start = tl.load(input + offsets, mask=mask, other=0.0)
    window_end = tl.load(input + offsets + window_size, mask=mask, other=-float('inf'))
    diff = window_end - window_start
    max_window_sum = tl.max(diff, axis=0)
    tl.atomic_max(output, max_window_sum)

def solve1(data: torch.Tensor, output: torch.Tensor, N: int, window_size: int):
    BLOCK_SIZE = 1024
    GRID_SIZE = triton.cdiv(N, BLOCK_SIZE)
    grid = ((GRID_SIZE, ))

    block_sum_ptr = torch.empty(GRID_SIZE, dtype=data.dtype, device=data.device)
    prefix_sum = torch.zeros_like(data)

    # stage1: calculate the block cusum
    prefix_sum_stage1[grid](
        data, prefix_sum, block_sum_ptr, N, BLOCK_SIZE
    )

    block_sum_ptr.cumsum_(dim=0)

    # stage2 calculate the data cusum
    prefix_sum_stage2[grid](
        data, prefix_sum, block_sum_ptr, N, BLOCK_SIZE
    )

    # stage3 calculate max window size
    max_windows_sum_kernel[grid](
        prefix_sum, output, N, window_size, BLOCK_SIZE
    )

# ===========================

@triton.jit
def max_subarray_sum_kernel1(
        input, output,
        N: tl.constexpr,
        window_size: tl.constexpr,
        BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    valid_starts = N - window_size + 1
    base = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    base_mask = base < valid_starts

    acc = tl.zeros((BLOCK_SIZE, ), dtype=tl.float32)
    for k in range(window_size):
        offset = base + k
        value = tl.load(input + offset, mask=base_mask, other=0.0)
        acc += value
    acc = tl.where(base_mask, acc, -float('inf'))
    window_max = tl.max(acc, axis=0)
    tl.atomic_max(output, window_max, sem='relaxed')

def solve(input: torch.Tensor, output: torch.Tensor, N: int, window_size: int):
    BLOCK_SIZE = 1024
    valid_start = N - window_size + 1
    grid = (triton.cdiv(valid_start, BLOCK_SIZE), )
    output.fill_(-float('inf'))
    max_subarray_sum_kernel1[grid](
        input, output, N, window_size, BLOCK_SIZE
    )

def run():
    N = 100_000
    window_size = 100
    data = torch.randn(N, device='cuda', dtype=torch.float32)
    output1 = torch.zeros(1, device='cuda', dtype=torch.float32)
    output2 = torch.zeros_like(output1)
    output3 = torch.zeros_like(output1)
    torch_solve(data, output1, N, window_size)
    solve1(data, output2, N, window_size)
    solve(data, output3, N, window_size)
    print(f"output1: {output1}")
    print(f"output2: {output2}")
    print(f"output3: {output3}")
    print(f"equal: {torch.allclose(output1, output2)}")

def run2():
    N = 4
    window_size = 3
    data = torch.tensor([-1,-4,-2,1]).cuda()
    output1 = torch.zeros(1, device='cuda', dtype=torch.float32)
    output2 = torch.zeros_like(output1)
    output3 = torch.zeros_like(output1)
    torch_solve(data, output1, N, window_size)
    solve1(data, output2, N, window_size)
    solve(data, output3, N, window_size)
    print(f"output1: {output1}")
    print(f"output2: {output2}")
    print(f"output3: {output3}")
    print(f"equal: {torch.allclose(output1, output2)}")

if __name__ == "__main__":
    run()
    run2()