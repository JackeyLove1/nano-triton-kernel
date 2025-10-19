'''
## General Matrix Multiplication (GEMM) Implementation

Implement a basic General Matrix Multiplication (GEMM).
Given matrix **A** of dimensions **M × K**, matrix **B** of dimensions **K × N**, input/output matrix **C** of dimensions **M × N**, and scalar multipliers **α** and **β**, compute the operation:
$C = α · (A × B) + β · C_{initial}$
The input matrices **A**, **B**, and the initial state of **C** contain 16-bit floating-point numbers (**FP16 / half**).
All matrices are stored in row-major order.
The scalars **α** and **β** are 32-bit floats.

## Implementation Requirements

* Use only native features (external libraries other than WMMA are not permitted).
* The `solve` function signature must remain unchanged.
* Accumulation during multiplication should use **FP32** for better precision before converting the final result to **FP16**.
* The final result must be stored back into matrix **C** as **half**.
'''

import torch
import triton
import triton.language as tl


def torch_solve(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, M: int, N: int, K: int, alpha: float, beta: float):
    output = alpha * torch.matmul(A.float().view(M, K), B.float().view(K, N)) + beta * C.float().view(M, N)
    # output = output.view(-1, 1).squeeze(1).to(torch.float16)
    C.copy_(output)

# a, b, c are tensors on the GPU
@triton.jit
def gemm_kernel(
        a, # [M, K]
        b, # [K, N]
        c, # [M, N]
        M: tl.constexpr,
        N: tl.constexpr,
        K: tl.constexpr,
        alpha: tl.constexpr,
        beta: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    off_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    off_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    off_k = tl.arange(0, BLOCK_K)

    stride_am = K  # A[m, k] = A[m * K + k]
    stride_ak = 1
    stride_bk = N  # B[k, n] = B[k * N + n]
    stride_bn = 1
    stride_cm = N  # C[m, n] = C[m * N + n]
    stride_cn = 1

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    a_ptr = a + (off_m[:, None] * stride_am + off_k[None, :] * stride_ak)
    b_ptr = b + (off_k[:, None] * stride_bk + off_n[None, :] * stride_bn)

    for k0 in range(0, K, BLOCK_K):
        a_mask = (off_m[:, None] < M) & ((k0 + off_k[None, :]) < K)
        b_mask = ((k0 + off_k[:, None]) < K) & (off_n[None, :] < N)

        a_tile = tl.load(a_ptr, mask=a_mask, other=0.0) # [BLOCK_M, BLOCK_K]
        b_tile = tl.load(b_ptr, mask=b_mask, other=0.0) # [BLOCK_K, BLOCK_N]

        acc += tl.dot(a_tile, b_tile, out_dtype=tl.float32, allow_tf32=False) # [BLOCK_M, BLOCK_N]

        # load next tile in A and B
        a_ptr += BLOCK_K * stride_ak
        b_ptr += BLOCK_K * stride_bk

    c_ptr = c + (off_m[:, None] * stride_cm + off_n[None, :] * stride_cn)
    c_mask = (off_m[:, None] < M) & (off_n[None, :] < N)
    c_init_tile = tl.load(c_ptr, mask=c_mask, other=0.0).to(tl.float32)

    out_tile = alpha * acc + beta * c_init_tile
    tl.store(c_ptr, out_tile.to(tl.float16) ,mask=c_mask)

def solve(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor,
          M: int, N: int, K: int, alpha: float, beta: float):
    BLOCK_M = 128
    BLOCK_N = 128
    BLOCK_K = 32

    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    gemm_kernel[grid](
        a, b, c,
        M, N, K,
        float(alpha), float(beta),
        BLOCK_M, BLOCK_N, BLOCK_K
    )

def run():
    M, N, K = 10_000, 5_000, 2_000
    a = torch.randn((M, K), device='cuda', dtype=torch.float16)
    b = torch.randn((K, N), device='cuda', dtype=torch.float16)
    c1 = torch.zeros((M, N), device='cuda', dtype=torch.float16)
    c2 = torch.zeros_like(c1)
    alpha = 5
    beta = 3
    solve(a, b, c1, M, N, K, alpha, beta)
    torch_solve(a,b,c2,M,N,K,alpha,beta)
    print(f"equal: {torch.allclose(c1, c2)}")

if __name__ == "__main__":
    print("run compile test")
    run()

    print("run performance test")
    import math
    @triton.testing.perf_report(
        triton.testing.Benchmark(
            x_names=['size'],
            x_vals=[2 ** i for i in range(7, 13)],  # 128 .. 4096
            x_log=True,
            line_arg='provider',
            line_vals=['triton', 'torch'],
            line_names=['Triton', 'Torch'],
            styles=[('blue', '-'), ('green', '-')],
            ylabel='TFLOPS',
            plot_name='gemm-performance',
            args={'alpha': 1.0, 'beta': 0.0},
        )
    )
    def benchmark(size, provider, alpha=1.0, beta=0.0):
        M = N = K = size
        a = torch.randn((M, K), device='cuda', dtype=torch.float16)
        b = torch.randn((K, N), device='cuda', dtype=torch.float16)
        c = torch.zeros((M, N), device='cuda', dtype=torch.float16)
        quantiles = [0.5, 0.2, 0.8]
        if provider == 'torch':
            ms, min_ms, max_ms = triton.testing.do_bench(
                lambda: torch_solve(a, b, c, M, N, K, alpha, beta), quantiles=quantiles
            )
        if provider == 'triton':
            ms, min_ms, max_ms = triton.testing.do_bench(
                lambda: solve(a, b, c, M, N, K, alpha, beta), quantiles=quantiles
            )
        def tflops(ms):
            # 2*M*N*K FLOPs per GEMM
            return (2.0 * M * N * K) / (ms * 1e-3) / 1e12
        return tflops(ms), tflops(max_ms), tflops(min_ms)

    benchmark.run(print_data=True, show_plots=True)

'''
gemm-performance:
     size     Triton      Torch
0   128.0   0.372364   0.170667
1   256.0   2.340571   1.170286
2   512.0  10.922666   5.349878
3  1024.0  46.603377   9.986438
4  2048.0  60.567564  15.768061
5  4096.0  50.419883  18.523009
'''