'''
Write a GPU program that computes the histogram of an array of 32-bit integers. The histogram should count the number of occurrences of each integer value in the range [0, num_bins). You are given an input array input of length N and the number of bins num_bins.

The result should be an array of integers of length num_bins, where each element represents the count of occurrences of its corresponding index in the input array.

Implementation Requirements
Use only native features (external libraries are not permitted)
The solve function signature must remain unchanged
The final result must be stored in the histogram array.
'''

import torch
import triton
import triton.language as tl


def torch_solve(input: torch.Tensor, histogram: torch.Tensor, N: int, num_bins: int):
    histogram.copy_(torch.histc(input, bins=num_bins))


# input, histogram are tensors on the GPU
@triton.jit
def histogram_kernel1(
        input, output,
        N: tl.constexpr,
        num_bins: tl.constexpr,
        BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    x = tl.load(input + offsets, mask=mask, other=-1)
    # Use tl.where to set invalid values to 0 (which won't contribute to histogram)
    # Then use atomic_add with scatter pattern
    valid_mask = (x >= 0) & (x < num_bins)
    # Cast x to int32 for indexing
    indices = x.to(tl.int32)
    # Use atomic_add with scatter: add 1 to output[indices] for each valid element
    tl.atomic_add(output + indices, 1, mask=valid_mask)


def solve1(input: torch.Tensor, histogram: torch.Tensor, N: int, num_bins: int):
    BLOCK_SIZE = 4096
    grid = (triton.cdiv(N, BLOCK_SIZE), )
    histogram_kernel1[grid](
        input, histogram, N, num_bins, BLOCK_SIZE
    )

# ---------------------------------------------------------

# @triton.jit
# def hist_kernel(
#     in_ptr,
#     hist_ptr,
#     N,
#     n_bins,
#     BLOCK_SIZE: tl.constexpr,
#     BLOCK_BIN: tl.constexpr,
# ):
#     pid = tl.program_id(0)
#     off = tl.arange(0, BLOCK_SIZE) + pid * BLOCK_SIZE
#     msk = off < N
#
#     data = tl.load(in_ptr + off, mask=msk)
#     hist = tl.histogram(data, BLOCK_BIN, mask=msk)
#
#     out_off = tl.arange(0, BLOCK_BIN)
#     out_msk = out_off < n_bins
#     tl.atomic_add(hist_ptr + out_off, hist, mask=out_msk & (hist > 0))
#
# # input, histogram are tensors on the GPU
# def solve(input: torch.Tensor, histogram: torch.Tensor, N: int, num_bins: int):
#     BLOCK_SIZE = 4096
#     grid = (triton.cdiv(N, BLOCK_SIZE),)
#     hist_kernel[grid](input, histogram, N, num_bins, BLOCK_SIZE, triton.next_power_of_2(num_bins))

# ---------------------------------------------------------

def run():
    N = 10_000
    num_bins = 10
    import random
    input = torch.tensor([random.randint(0, 10) for _ in range(N)]).cuda()
    histogram1 = torch.zeros(num_bins).cuda()
    histogram2 = torch.zeros(num_bins).cuda()
    solve1(input, histogram1, N, num_bins)
    # solve(input, histogram2, N, num_bins)
    print(f"histogram: {histogram1}")
    print(f"histogram: {histogram2}")

if __name__ == "__main__":
    run()
