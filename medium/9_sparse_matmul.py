'''
## Implement a CUDA program that performs sparse matrix-vector multiplication

Given a sparse matrix **A** of dimensions *M × N* and a dense vector **x** of length *N*, compute the product vector
$y = A \times x$
which will have length *M*. **A** is stored in row-major order.
`nnz` is the number of non-zero elements in **A**.

Mathematically, the operation is defined as:
$$y_i = \sum_{j=0}^{N-1} A_{ij} \cdot x_j \quad \text{for } i = 0, 1, \dots, M - 1$$
The matrix **A** is approximately **60 – 70% sparse**.

## Implementation Requirements
* Use only CUDA native features (external libraries are not permitted)
* The `solve` function signature must remain unchanged
* The final result must be stored in vector **y**
'''

import torch
import triton
import triton.language as tl

# A, x, y are tensors on the GPU
def torch_solve(A: torch.Tensor, x: torch.Tensor, y: torch.Tensor, M: int, N: int, nnz: int):
    result = torch.matmul(A.view(M, N), x.view(N, 1))
    y.copy_(result.view(M))

# =========================================================

@triton.jit
def spmv_kernel(
        A, x, y,
        M: tl.constexpr,
        N: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr
):
    pid_m = tl.program_id(0)

    off_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    off_n = tl.arange(0, BLOCK_N)

    acc = tl.zeros((BLOCK_M, ), dtype=tl.float32)

    for n in range(0, triton.cdiv(N, BLOCK_N)):
        current_n_off = n * BLOCK_N + off_n

        a_ptr = A + (off_m[:, None] * N) + current_n_off[None, :]
        a_mask = (off_m[:, None] < M) & (current_n_off[None, :] < N)

        x_ptr = x + current_n_off
        x_mask = current_n_off < N

        a_data = tl.load(a_ptr, mask=a_mask, other=0.0) # [BLOCK_M, BLOCK_N]
        x_data = tl.load(x_ptr, mask=x_mask, other=0.0) # [BLOCK_N]

        acc += tl.sum(a_data * x_data[None, :], axis=1) # [BLOCK_M]

    out_mask = off_m < M
    tl.store(y + off_m, acc, mask=out_mask)



def solve(A: torch.Tensor, x: torch.Tensor, y: torch.Tensor, M: int, N: int, nnz: int):
    BLOCK_M, BLOCK_N = 64, 64
    grid = ((triton.cdiv(M, BLOCK_M), ))
    spmv_kernel[grid](
        A, x, y, M, N, BLOCK_M, BLOCK_N
    )

@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['M', 'N'],
        x_vals=[(512, 256), (1024, 512), (2048, 1024), (4096, 2048), (8192, 4096), (10000, 5000)],
        x_log=False,
        line_arg='provider',  # 参数名称，其值对应于绘图中的不同线条
        line_vals=['triton', 'torch'],  # `line_arg` 的可能值
        line_names=['Triton', 'Torch'],  # 线条的标签名称
        styles=[('blue', '-'), ('green', '-')],  # 线条样式
        ylabel='GFLOPS',  # y 轴标签名称
        plot_name='sparse-matmul-performance',  # 绘图名称
        args={},  # 不在 `x_names` 中的函数参数值
    ))
def benchmark(M, N, provider):
    nnz = (M * N) // 4  # 保持约75%稀疏度的数据
    
    quantiles = [0.5, 0.2, 0.8]
    
    # 创建测试数据
    A = torch.randn((M, N), device='cuda', dtype=torch.float32)
    x = torch.randn(N, device='cuda', dtype=torch.float32)
    
    if provider == 'torch':
        y = torch.zeros(M, device='cuda', dtype=torch.float32)
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: torch_solve(A, x, y, M, N, nnz), 
            quantiles=quantiles
        )
    elif provider == 'triton':
        y = torch.zeros(M, device='cuda', dtype=torch.float32)
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: solve(A, x, y, M, N, nnz), 
            quantiles=quantiles
        )
    
    # 计算GFLOPS (稀疏矩阵乘法的计算量为 2 * M * N，但考虑稀疏性)
    # 对于密集计算：M * N * 2 浮点操作（乘法和加法）
    gflops = lambda ms: 2 * M * N / ms * 1e-9
    return gflops(ms), gflops(max_ms), gflops(min_ms)

def run():
    M, N, nnz = 10_000, 5_000, 1000
    A = torch.randn((M, N), device='cuda', dtype=torch.float32)
    x = torch.randn(N, device='cuda', dtype=torch.float32)
    y1 = torch.zeros(M, device='cuda', dtype=torch.float32)
    y2 = torch.zeros_like(y1)
    torch_solve(A, x, y1, M, N, nnz)
    solve(A, x, y2, M, N, nnz)
    # print(f"y1:{y1}")
    # print(f"y2:{y2}")
    print(f"equal: {torch.allclose(y1, y2)}")

if __name__ == "__main__":
    # 运行正确性测试
    run()
    print("\n" + "="*60 + "\n")
    # 运行性能基准测试
    benchmark.run(print_data=True, show_plots=True)


'''
sparse-matmul-performance:
         M       N    Triton     Torch
0    512.0   256.0  0.036571  0.028444
1   1024.0   512.0  0.113778  0.093091
2   2048.0  1024.0  0.204800  0.195048
3   4096.0  2048.0  0.297891  0.287439
4   8192.0  4096.0  0.343120  0.336082
5  10000.0  5000.0  0.318099  0.342654
'''