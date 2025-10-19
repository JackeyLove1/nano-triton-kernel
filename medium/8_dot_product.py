'''
Here’s the content extracted from your image in **Markdown format**:

## Implement a CUDA program that computes the dot product of two vectors containing 32-bit floating point numbers
The dot product is the sum of the products of the corresponding elements of two vectors.
Mathematically, the dot product of two vectors **A** and **B** of length *n* is defined as:

$$A \cdot B = \sum_{i=0}^{n-1} A_i \cdot B_i = A_0 \cdot B_0 + A_1 \cdot B_1 + \dots + A_{n-1} \cdot B_{n-1}$$

## Implementation Requirements

* Use only CUDA native features (external libraries are not permitted)
* The `solve` function signature must remain unchanged
* The final result must be stored in the `output` variable
'''

import torch
import triton
import triton.language as tl

def torch_solve(A: torch.Tensor, B: torch.Tensor, result: torch.Tensor, N: int):
    result.copy_(torch.dot(A, B))

# a, b, result are tensors on the GPU
@triton.jit
def dot_kernel(
        a, b, result,
        N: tl.constexpr,
        BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    data_a = tl.load(a + offsets, mask=mask, other=0.0)
    data_b = tl.load(b + offsets, mask=mask, other=0.0)
    block_sum = tl.sum(data_a * data_b,axis=0)
    tl.atomic_add(result, block_sum, sem="relaxed")


def solve(a: torch.Tensor, b: torch.Tensor, result: torch.Tensor, N: int):
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(N, BLOCK_SIZE), )
    dot_kernel[grid](a, b, result, N, BLOCK_SIZE)

@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['size'],
        x_vals=[2 ** i for i in range(12, 28, 1)],
        x_log=True,
        line_arg='provider',
        line_vals=['triton', 'torch'],
        line_names=['Triton', 'Torch'],
        styles=[('blue', '-'), ('green', '-')],
        ylabel='GB/s',
        plot_name='dot-product-performance',
        args={},
    ))
def benchmark(size, provider):
    a = torch.randn(size, device='cuda', dtype=torch.float32)
    b = torch.randn(size, device='cuda', dtype=torch.float32)
    quantiles = [0.5, 0.2, 0.8]
    
    if provider == 'torch':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: torch.dot(a, b), quantiles=quantiles)
    if provider == 'triton':
        def triton_dot():
            result = torch.zeros(1, device='cuda', dtype=torch.float32)
            solve(a, b, result, size)
            return result
        ms, min_ms, max_ms = triton.testing.do_bench(triton_dot, quantiles=quantiles)
    
    gbps = lambda ms: 2 * a.numel() * a.element_size() / ms * 1e-6
    return gbps(ms), gbps(max_ms), gbps(min_ms)

def run():
    N = 100_000
    a = torch.randn(N, device='cuda', dtype=torch.float32)
    b = torch.randn(N, device='cuda', dtype=torch.float32)
    result1 = torch.zeros(1, device='cuda', dtype=torch.float32)
    result2 = torch.zeros(1, device='cuda', dtype=torch.float32)
    solve(a, b, result1, N)
    torch_solve(a, b, result2, N)
    errors = (result2 - result1).sum()
    equal = torch.allclose(result1, result2)
    print(f"result1: {result1}")
    print(f"result2: {result2}")
    print(f"equal: {equal}, errors:{errors}")


if __name__ == "__main__":
    benchmark.run(print_data=True, show_plots=True)

'''
dot-product-performance:
           size      Triton       Torch
0        4096.0    4.571429    4.571429
1        8192.0   10.666666    9.142857
2       16384.0   21.333333   18.285714
3       32768.0   36.571428   32.000000
4       65536.0   73.142856   56.888887
5      131072.0  128.000000  102.400003
6      262144.0  204.800005  170.666661
7      524288.0  292.571425  273.066674
8     1048576.0  390.095241  372.363633
9     2097152.0  496.484845  468.114273
10    4194304.0  585.142849  564.965503
11    8388608.0  648.871301  636.271854
12   16777216.0  686.240841  675.628891
13   33554432.0  706.498339  699.050661
14   67108864.0  716.240451  710.417370
15  134217728.0  720.670781  717.220268
'''