'''
RMS Normalization: Triton implementation with performance benchmarking
Reference: Root Mean Square Layer Normalization
'''

import torch
import triton
import triton.language as tl
from torch import Tensor


def torch_rms_norm(input: Tensor, gamma: Tensor, beta: Tensor, eps: float) -> Tensor:
    """PyTorch reference implementation of RMS Norm"""
    rms = (input.pow(2).mean(dim=-1, keepdim=True) + eps).sqrt()
    x_rms = input / rms
    output = gamma * x_rms + beta
    return output


def torch_solve(input: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor,
          output: torch.Tensor, N: int, eps: float):
    rms = (input.pow(2).mean() + eps).sqrt()
    x_rms = input / rms
    output.copy_(gamma * x_rms + beta)

# input, output are tensors on the GPU
@triton.jit
def sum_kernel(
        input, block_sum,
        N: tl.constexpr,
        BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N

    x = tl.load(input + offsets, mask=mask, other=0.0).to(tl.float32)
    sum = tl.sum(x * x, axis=0)
    tl.store(block_sum + pid, sum)

@triton.jit
def rms_kernel(
        input, output, rms,
        gamma: tl.constexpr,
        beta: tl.constexpr,
        N: tl.constexpr,
        BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N

    x = tl.load(input + offsets, mask=mask, other=0.0).to(tl.float32)
    rms = tl.load(rms)
    x_i = x / rms
    y = gamma * x_i + beta
    tl.store(output + offsets, y, mask=mask)

def solve(input: torch.Tensor, gamma: float, beta: float,
          output: torch.Tensor, N: int, eps: float):
    BLOCK_SIZE = 1024
    GRID_SIZE = triton.cdiv(N, BLOCK_SIZE)
    grid = (GRID_SIZE, )
    block_sum = torch.zeros(GRID_SIZE, dtype=input.dtype,device=input.device)
    sum_kernel[grid](
        input, block_sum, N, BLOCK_SIZE
    )
    rms = (block_sum.sum(dim=0) / N + eps).sqrt()
    # print(f"rms: {rms}")
    rms_kernel[grid](
        input, output, rms, gamma, beta, N, BLOCK_SIZE
    )

def rms_norm(input: Tensor, gamma: float = 1.0, beta: float = 0.0, 
             eps: float = 1e-5) -> Tensor:
    """Triton implementation wrapper for RMS Norm"""
    assert input.is_cuda, "Input must be on GPU"
    N = input.numel()
    output = torch.empty_like(input, dtype=input.dtype, device=input.device)
    solve(input, gamma, beta, output, N, eps)
    return output

@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['size'],
        x_vals=[2 ** i for i in range(10, 24, 1)],
        x_log=True,
        line_arg='provider',
        line_vals=['triton', 'torch'],
        line_names=['Triton', 'Torch'],
        styles=[('blue', '-'), ('green', '-')],
        ylabel='GB/s',
        plot_name='rms-norm-performance',
        args={},
    ))
def benchmark(size, provider):
    x = torch.randn(size, device='cuda', dtype=torch.float32)
    gamma = 1.0
    beta = 0.0
    eps = 1e-5
    quantiles = [0.5, 0.2, 0.8]
    
    if provider == 'torch':
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: torch.nn.functional.layer_norm(x, (x.shape[-1],), weight=None, bias=None, eps=eps),
            quantiles=quantiles
        )
    if provider == 'triton':
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: rms_norm(x, gamma, beta, eps),
            quantiles=quantiles
        )
    
    # Calculate GB/s: (input + output) / time
    gbps = lambda ms: 2 * x.numel() * x.element_size() / ms * 1e-6
    return gbps(ms), gbps(max_ms), gbps(min_ms)

def run():
     input = torch.tensor([1.0, 2.0, 3.0, 4.0]).float().cuda()
     N = 4
     gamma = 1.0
     beta = 0.0
     eps = 1e-5
     output = torch.zeros_like(input)
     solve(input, gamma, beta, output, N, eps)
     print(f"output: {output}, expected: [0.36514813, 0.73029625, 1.0954444, 1.4605925]")

if __name__ == "__main__":
    # Run correctness test
    run()
    
    # Run benchmark
    print("\n" + "="*50)
    print("Running RMS Norm Benchmark...")
    print("="*50)
    benchmark.run(print_data=True, show_plots=False)

'''
rms-norm-performance:
         size      Triton     Torch
0      1024.0    0.429530  1.158371
1      2048.0    1.145414  2.534654
2      4096.0    2.267996  3.864151
3      8192.0    4.481401  5.305699
4     16384.0    9.002198  5.818182
5     32768.0   16.286283  6.473331
6     65536.0   32.251967  6.918919
7    131072.0   61.478426  7.156147
8    262144.0  113.777774  7.297995
9    524288.0  195.922280  7.359668
10  1048576.0  228.348430  6.262997
11  2097152.0  274.496336  6.287030
12  4194304.0  312.355084  6.294729
13  8388608.0  334.367358  6.292028

'''