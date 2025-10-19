'''
Write a program that sorts an array of 32-bit floating-point numbers in ascending order. You are free to choose any sorting algorithm.

Implementation Requirements
Use only native features (external libraries are not permitted)
The solve function signature must remain unchanged
The sorted result must be stored back in the input data array
'''

import torch
import triton
import triton.language as tl

# data is a tensor on the GPU
def torch_solve(data: torch.Tensor, N: int):
    data.sort(dim=0)

# -----------------------------
# Triton kernel: one "bitonic step"
# -----------------------------
@triton.jit
def _bitonic_step(x_ptr, n, j, k,  # pointers & scalar params
                  BLOCK: tl.constexpr):
    """
    One stage of a bitonic network:
      For each index i, compare with i^j and conditionally swap
      depending on whether the current subsequence is ascending or descending.

    We only handle pairs once (when (i^j) > i) to avoid double-writes.
    """
    pid = tl.program_id(axis=0)
    idx = pid * BLOCK + tl.arange(0, BLOCK)

    in_bounds = idx < n
    partner = idx ^ j
    partner_in_bounds = partner < n

    # Only perform compare-swap once per pair to avoid races
    do_pair = in_bounds & partner_in_bounds & (partner > idx)

    # Load both sides of the pair
    a = tl.load(x_ptr + idx, mask=do_pair, other=0.0)
    b = tl.load(x_ptr + partner, mask=do_pair, other=0.0)

    # Ascending if the k-th bit of i is 0 (classic bitonic rule)
    ascending = (idx & k) == 0

    # Swap if order is wrong for the current (ascending/descending) run
    swap = (a > b) == ascending

    new_a = tl.where(swap, b, a)
    new_b = tl.where(swap, a, b)

    # Store back
    tl.store(x_ptr + idx, new_a, mask=do_pair)
    tl.store(x_ptr + partner, new_b, mask=do_pair)


def _next_power_of_two(x: int) -> int:
    if x <= 1:
        return 1
    return 1 << (x - 1).bit_length()



# data is a tensor on the GPU
def solve(data: torch.Tensor, N: int):
    """
    Triton-based in-place sort for a 1D float32 tensor `data` of length >= N.
    - Sorts ascending
    - Uses a global bitonic network (O(N log^2 N))
    - No external libraries; only PyTorch + Triton
    """
    # Quick exits
    if N <= 1:
        return
    assert data.is_cuda, "Input tensor must be on GPU"
    assert data.dtype == torch.float32, "Expected float32 data"
    assert data.dim() == 1, "Expected a 1D tensor"
    assert N <= data.numel(), "N exceeds data length"

    device = data.device
    BLOCK = 1024

    # Round up to next power of two for bitonic network
    M = _next_power_of_two(N)

    # Work buffer (length M). Fill with +inf, then copy first N values.
    work = torch.empty(M, device=device, dtype=data.dtype)
    work.fill_(float('inf'))
    work[:N].copy_(data[:N])

    # Launch parameters
    grid = (triton.cdiv(M, BLOCK),)

    # Bitonic stages: k = 2,4,8,...,M ; j halves inside each k
    # (classic bitonic network structure)
    for k in [1 << p for p in range(1, M.bit_length() + 1)]:  # 2..M
        j = k >> 1
        while j > 0:
            _bitonic_step[grid](work, M, j, k, BLOCK=BLOCK)
            j >>= 1

    # Write back the sorted first N results into the original tensor
    data[:N].copy_(work[:N])

# ========================================
# Benchmark Testing Code
# ========================================
@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['size'],
        x_vals=[2 ** i for i in range(12, 24, 1)],  # 4K to 8M elements
        x_log=True,
        line_arg='provider',  # Argument name whose value corresponds to a different line in the plot
        line_vals=['triton', 'torch'],  # Possible values for `line_arg`
        line_names=['Triton (Bitonic)', 'PyTorch (Native)'],  # Label name for the lines
        styles=[('blue', '-'), ('green', '-')],  # Line styles
        ylabel='GB/s',  # Label name for the y-axis
        plot_name='sorting-performance',  # Name for the plot
        args={},  # Values for function arguments not in `x_names` and `y_name`
    ))
def benchmark(size, provider):
    """
    Benchmark sorting performance comparing Triton vs PyTorch implementations.
    
    Performance is measured in GB/s (throughput).
    Each benchmark creates a random float32 tensor and sorts it.
    """
    data = torch.randn(size, device='cuda', dtype=torch.float32)
    quantiles = [0.5, 0.2, 0.8]  # median, 20th percentile, 80th percentile
    
    if provider == 'torch':
        # Clone data for PyTorch test
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: torch_solve(data.clone(), size), 
            quantiles=quantiles
        )
    if provider == 'triton':
        # Clone data for Triton test
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: solve(data.clone(), size), 
            quantiles=quantiles
        )
    
    # Calculate throughput: GB/s
    # Total bytes = number of elements * bytes per element
    gbps = lambda ms: data.numel() * data.element_size() / ms * 1e-6
    return gbps(ms), gbps(max_ms), gbps(min_ms)


if __name__ == '__main__':
    benchmark.run(print_data=True, show_plots=False)