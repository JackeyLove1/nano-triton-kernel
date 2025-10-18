'''
# Matrix Multiplication on GPU (32-bit Float)

Write a program that multiplies two matrices of 32-bit floating point numbers on a GPU.
Given matrix **A** of dimensions **M × N** and matrix **B** of dimensions **N × K**, compute the product matrix **C = A × B**, which will have dimensions **M × K**.
All matrices are stored in **row-major** format.

## Implementation Requirements

* Use only native features (external libraries are not permitted).
* The `solve` function signature must remain unchanged.
* The final result must be stored in matrix **C**.

'''
import torch
import triton
import triton.language as tl


def torch_solve(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, M: int, N: int, K: int):
    torch.matmul(A, B, out=C)


# a: [M, N], b: [N, K] -> c: [M, K]
@triton.jit
def matrix_multiplication_kernel_1(
        a, b, c,
        M, N, K,
        stride_am, stride_an,
        stride_bn, stride_bk,
        stride_cm, stride_ck,
        BLOCK: tl.constexpr = 128,
):
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)
    if (pid_m >= M) or (pid_k >= K):
        return

    # 用 float64 累加，降低舍入误差
    acc = tl.zeros((), dtype=tl.float64)

    for n0 in range(0, N, BLOCK):
        offs = n0 + tl.arange(0, BLOCK)
        mask = offs < N

        a_ptrs = a + pid_m * stride_am + offs * stride_an
        b_ptrs = b + offs * stride_bn + pid_k * stride_bk

        a_tile = tl.load(a_ptrs, mask=mask, other=0.0)
        b_tile = tl.load(b_ptrs, mask=mask, other=0.0)

        acc += tl.sum(a_tile * b_tile, axis=0)

    c_ptr = c + pid_m * stride_cm + pid_k * stride_ck
    tl.store(c_ptr, acc.to(tl.float32))

def solve_1(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, M: int, N: int, K: int):
    # Row-major strides (elements, not bytes)
    stride_am, stride_an = N, 1       # A[m, n] = A_ptr + m*N + n
    stride_bn, stride_bk = K, 1       # B[n, k] = B_ptr + n*K + k
    stride_cm, stride_ck = K, 1       # C[m, k] = C_ptr + m*K + k


    # Launch one program per output element (matches the kernel design above)
    grid = (M, K)
    matrix_multiplication_kernel_1[grid](
        a, b, c,
        M, N, K,
        stride_am, stride_an,
        stride_bn, stride_bk,
        stride_cm, stride_ck,
    )


@triton.jit
def matrix_multiplication_kernel(
        a, b, c,
        M, N, K,
        stride_am, stride_an,
        stride_bn, stride_bk,
        stride_cm, stride_ck,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
        gsz: tl.constexpr
):
    hw_pid_m = tl.program_id(0)
    hw_pid_k = tl.program_id(1)

    num_programs_m = tl.num_programs(0)
    num_programs_k = tl.num_programs(1)

    logical_pid_m, logical_pid_k = tl.swizzle2d(hw_pid_m, hw_pid_k, num_programs_m, num_programs_k, gsz)

    range_m = logical_pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    range_k = logical_pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    range_n = tl.arange(0, BLOCK_N)

    offset_a = a + range_m[:, None] * stride_am + range_n[None, :] * stride_an
    offset_b = b + range_n[:, None] * stride_bn + range_k[None, :] * stride_bk

    accumulator = tl.zeros((BLOCK_M, BLOCK_K), dtype=tl.float32)

    for current_n in range(0, N, BLOCK_N):
        updated_range_n = range_n + current_n

        mask_a = (range_m[:, None] < M) & (updated_range_n[None, :] < N)
        mask_b = (updated_range_n[:, None] < N) & (range_k[None, :] < K)

        a_data = tl.load(offset_a, mask=mask_a)
        b_data = tl.load(offset_b, mask=mask_b)

        accumulator += tl.dot(a_data, b_data, input_precision="ieee")

        offset_a += BLOCK_N * stride_an
        offset_b += BLOCK_N * stride_bn

    mask_c = (range_m[:, None] < M) & (range_k < K)
    tl.store(c + range_m[:, None] * stride_cm + range_k[None, :] * stride_ck, accumulator, mask=mask_c)


# a, b, c are tensors on the GPU
def solve(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, M: int, N: int, K: int):
    stride_am, stride_an = N, 1
    stride_bn, stride_bk = K, 1
    stride_cm, stride_ck = K, 1

    BLOCK_M = 128
    BLOCK_N = 64
    BLOCK_K = 64

    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(K, BLOCK_K))

    matrix_multiplication_kernel[grid](
        a, b, c,
        M, N, K,
        stride_am, stride_an,
        stride_bn, stride_bk,
        stride_cm, stride_ck,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        gsz=8
    )


# ==================== Accuracy Testing ====================

def matmul_accuracy_test():
    """Test matrix multiplication accuracy with different matrix sizes"""
    print("\n" + "="*60)
    print("Matrix Multiplication Accuracy Testing")
    print("="*60)
    
    test_configs = [
        (32, 32, 32),
        (64, 64, 64),
        (128, 128, 128),
        (256, 256, 256),
        (512, 512, 512),
        (128, 256, 64),  # Non-square matrices
        (256, 512, 128),
        (1024, 512, 256),
    ]
    
    all_passed = True
    
    for M, N, K in test_configs:
        # Create random input matrices on GPU
        A = torch.randn((M, N), dtype=torch.float32, device='cuda')
        B = torch.randn((N, K), dtype=torch.float32, device='cuda')
        
        # Compute reference result using PyTorch
        C_ref = torch.matmul(A, B)
        
        # Compute result using Triton implementation
        C_triton = torch.empty((M, K), dtype=torch.float32, device='cuda')
        solve(A, B, C_triton, M, N, K)
        
        # Compute relative error
        abs_error = torch.abs(C_triton - C_ref)
        max_abs_error = torch.max(abs_error).item()
        rel_error = max_abs_error / (torch.max(torch.abs(C_ref)).item() + 1e-8)
        
        # Check if result is correct (with some tolerance for floating point errors)
        passed = rel_error < 1e-4
        all_passed = all_passed and passed
        
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status} | M={M:4d}, N={N:4d}, K={K:4d} | "
              f"Max Abs Error: {max_abs_error:.2e} | Relative Error: {rel_error:.2e}")
    
    print("="*60 + "\n")
    return all_passed


# ==================== Performance Benchmark ====================

def matmul_triton(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """Triton implementation of matrix multiplication"""
    M, N = A.shape
    N_B, K = B.shape
    assert N == N_B, f"Incompatible dimensions: A({M}, {N}) @ B({N_B}, {K})"
    
    C = torch.empty((M, K), dtype=A.dtype, device=A.device)
    solve(A, B, C, M, N, K)
    return C


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['M', 'N', 'K'],
        x_vals=[
            (128, 128, 128),
            (256, 256, 256),
            (512, 512, 512),
            (1024, 1024, 1024),
            (2048, 2048, 2048),
            (4096, 4096, 4096),
            (8192, 8192, 8192),
            # (2**15, 2**15, 2**15),
            # (2**18, 2**18, 2**18),
            # (2**21, 2**21, 2**21),
        ],
        x_log=False,
        line_arg='provider',
        line_vals=['triton', 'torch'],
        line_names=['Triton', 'Torch'],
        styles=[('blue', '-'), ('green', '-')],
        ylabel='TFLOPS',
        plot_name='matmul-performance',
        args={},
    ))
def benchmark(M, N, K, provider):
    A = torch.randn((M, N), dtype=torch.float32, device='cuda')
    B = torch.randn((N, K), dtype=torch.float32, device='cuda')
    
    quantiles = [0.5, 0.2, 0.8]
    
    if provider == 'torch':
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: torch.matmul(A, B), 
            quantiles=quantiles
        )
    elif provider == 'triton':
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: matmul_triton(A, B),
            quantiles=quantiles
        )
    
    # Calculate TFLOPS: 2 * M * N * K FLOPs in ms milliseconds
    # 1 TFLOP = 1e12 FLOPs
    tflops = lambda ms: 2 * M * N * K / (ms * 1e9)
    
    return tflops(ms), tflops(max_ms), tflops(min_ms)


if __name__ == '__main__':
    # Run accuracy tests first
    test_passed = matmul_accuracy_test()
    
    if test_passed:
        print("All accuracy tests passed! Running performance benchmark...")
        benchmark.run(print_data=True, show_plots=False)
    else:
        print("Some accuracy tests failed. Skipping performance benchmark.")

'''
matmul-performance:
        M       N       K     Triton      Torch
0   128.0   128.0   128.0   0.178087   0.455111
1   256.0   256.0   256.0   0.936229   2.978909
2   512.0   512.0   512.0   4.766255  10.922666
3  1024.0  1024.0  1024.0  10.866073  16.384000
4  2048.0  2048.0  2048.0  13.096968  21.788593
5  4096.0  4096.0  4096.0  13.306010  20.876922
6  8192.0  8192.0  8192.0  13.467731  21.004750
'''