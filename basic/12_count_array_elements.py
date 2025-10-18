'''
Write a GPU program that counts the number of elements with the integer value k in an array of 32-bit integers. The program should count the number of elements with k in an array. You are given an input array input of length N and integer k.

Implementation Requirements
Use only native features (external libraries are not permitted)
The solve function signature must remain unchanged
The final result must be stored in the output variable
'''

import torch
import triton
import triton.language as tl

# input, output are tensors on the GPU
def torch_solve(input: torch.Tensor, output: torch.Tensor, N: int, K: int):
    output.copy_((input == K).sum(dim=-1))


@triton.jit
def count_equal_kernel(input_ptr, output_ptr, N, K, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    x = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    x = tl.where(x == K, 1, 0)
    cnt = tl.sum(x.to(tl.int32),axis=0)
    tl.atomic_add(output_ptr, cnt)


# input, output are tensors on the GPU
def solve(input: torch.Tensor, output: torch.Tensor, N: int, K: int):
    BLOCK_SIZE = 1024
    grid = lambda meta: (triton.cdiv(N, meta['BLOCK_SIZE']),)
    count_equal_kernel[grid](input, output, N, K, BLOCK_SIZE=BLOCK_SIZE)

def kernel_test():
    N, K = 10_000, 1
    import random
    input = torch.tensor([random.randint(0, 10) for _ in range(N)], dtype=torch.int32).cuda()
    output = torch.zeros((1, ), dtype=torch.int32).cuda()
    solve(input, output, N, K)


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['size'],
        x_vals=[2 ** i for i in range(8, 28, 1)],
        x_log=True,
        line_arg='provider',
        line_vals=['triton', 'torch'],
        line_names=['Triton', 'Torch'],
        styles=[('blue', '-'), ('green', '-')],
        ylabel='GB/s',
        plot_name='count-array-elements-performance',
        args={},
    ))
def benchmark(size, provider):
    K = 5
    input = torch.randint(0, 10, (size,), device='cuda', dtype=torch.int32)
    output = torch.zeros((1,), device='cuda', dtype=torch.int32)
    quantiles = [0.5, 0.2, 0.8]
    
    if provider == 'torch':
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: torch_solve(input, output, size, K), 
            quantiles=quantiles
        )
    if provider == 'triton':
        output.zero_()  # Reset output for each run
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: solve(input, output, size, K), 
            quantiles=quantiles
        )
    
    gbps = lambda ms: input.numel() * input.element_size() / ms * 1e-6
    return gbps(ms), gbps(max_ms), gbps(min_ms)


if __name__ == "__main__":
    # kernel_test()  # Uncomment to run quick test
    benchmark.run(print_data=True, show_plots=False)

'''
count-array-elements-performance:
           size      Triton       Torch
0         256.0    0.166667    0.083333
1         512.0    0.250000    0.181818
2        1024.0    0.800000    0.266667
3        2048.0    2.000000    0.666667
4        4096.0    2.000000    1.333333
5        8192.0    6.400000    2.000000
6       16384.0   16.000000    4.266667
7       32768.0   16.000000    8.000000
8       65536.0   42.666665    9.142857
9      131072.0  102.400003   26.947370
10     262144.0  102.400003   53.894739
11     524288.0  227.555548   60.235293
12    1048576.0  341.333321   83.591839
13    2097152.0  356.173905   97.523810
14    4194304.0  496.484845  104.356686
15    8388608.0  606.814814  117.448029
16   16777216.0  648.871301  122.497196
17   33554432.0  686.240841  124.949481
18   67108864.0  706.587597  125.577967
19  134217728.0  717.220268  126.915516
'''