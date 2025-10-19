'''
# 2D Convolution on the GPU — Problem Statement (Markdown Extraction)

Write a program that performs a 2D convolution operation on the GPU. Given an input matrix and a kernel (filter), compute the convolved output. The convolution should be performed with a **"valid"** boundary condition, meaning the kernel is only applied where it fully overlaps with the input.

## Inputs

* `input`: A 2D matrix of 32-bit floating-point numbers, represented as a 1D array in row-major order.
* `kernel`: A 2D kernel (filter) of 32-bit floating-point numbers, also represented as a 1D array in row-major order.

## Output

Write the result to the `output` matrix (also a 1D array in row-major order).

### Output dimensions

* `output_rows = input_rows - kernel_rows + 1`
* `output_cols = input_cols - kernel_cols + 1`

## Convolution definition

[
\text{output}[i][j] ;=; \sum_{m=0}^{\text{kernel_rows}-1} \sum_{n=0}^{\text{kernel_cols}-1}
\text{input}[,i+m,][,j+n,] \times \text{kernel}[,m,][,n,]
]

## Implementation Requirements

* Use only native features (external libraries are not permitted).
* The `solve` function signature must remain unchanged.
* The final result must be stored in the array `output`.

'''

import torch
import triton
import triton.language as tl
import torch.nn.functional as F


def torch_solve(input: torch.Tensor, kernel: torch.Tensor, output: torch.Tensor,
          input_rows: int, input_cols: int, kernel_rows: int, kernel_cols: int):
    # NCHW -> batch_size , channels, height, width
    input = input.reshape(1, 1, input_rows, input_cols)
    kernel = kernel.reshape(1, 1, kernel_rows, kernel_cols)
    output_rows = input_rows - kernel_rows + 1
    output_cols = input_cols - kernel_cols + 1
    conv_result = F.conv2d(input, kernel, stride=1, padding=0).reshape(output_rows, output_cols)
    output.copy_(conv_result)

# input, kernel, output are tensors on the GPU
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32}, num_warps=4, num_stages=2),
    ],
    key=['OR', 'OC']
)
@triton.jit
def conv2d_kernel(
        input_ptr, kernel, output_ptr,
        IR: tl.constexpr, # input_rows
        IC: tl.constexpr, # input_cols
        KR:tl.constexpr, # kernel_rows
        KC: tl.constexpr, # kernel_cols
        OR: tl.constexpr, # output rows
        OC: tl.constexpr, # output cols
        BLOCK_M: tl.constexpr, # rows_stride
        BLOCK_N: tl.constexpr  # cols_ stride
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # calculate the output rows and cols
    oi = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    oj = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    o_mask = (oi[:, None] < OR) & (oj[None, :] < OC)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # row-major: idx = r * W + c
    base_input_idx = oi[:, None] * IC + oj[None, :]
    for km in range(KR):
        row_offset = km * IC
        for kn in range(KC):
            # kernel value
            k_val = tl.load(kernel + km * KC + kn)

            # input data offset
            input_ptr_offset = base_input_idx + row_offset + kn
            x = tl.load(input_ptr + input_ptr_offset, mask=o_mask, other=0)
            # if tl.program_id(0) == 0 and tl.program_id(1) == 0:
            #     tl.static_print("k_val shape: ", k_val.shape)
            #     tl.static_print("x shape: ", x.shape)

            acc += k_val * x
    # tl.device_print("acc:", acc)
    output_index = oi[:, None] * OC + oj[None, :]
    tl.store(output_ptr + output_index, acc, mask=o_mask)



def solve(input: torch.Tensor,
          kernel: torch.Tensor,
          output: torch.Tensor,
          input_rows: int, input_cols: int,
          kernel_rows: int, kernel_cols: int):
    assert input.is_cuda and kernel.is_cuda and output.is_cuda
    IR, IC = input_rows, input_cols
    KR, KC = kernel_rows, kernel_cols
    OR, OC = input_rows - kernel_rows + 1, input_cols - kernel_cols + 1
    assert OR > 0 and OC > 0

    def grid(meta):
        BM, BN = meta["BLOCK_M"], meta["BLOCK_N"]
        return (
            triton.cdiv(OR, BM),
            triton.cdiv(OC, BN)
        )

    conv2d_kernel[grid](
        input, kernel, output,
        IR, IC,
        KR, KC,
        OR, OC,
    )

def run():
    input = torch.arange(1, 10).view(3, 3).cuda()
    kernel = torch.tensor([[0,1],[1,0]]).cuda()
    output = torch.zeros((2,2)).cuda()
    solve(input, kernel, output,
          input.shape[0], input.shape[1],
          kernel.shape[0], kernel.shape[1])
    print(f"output:{output}")



@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['input_size'],  # 输入矩阵大小 (N x N)
        x_vals=[2**i for i in range(5, 12)],
        x_log=True,
        line_arg='provider',  # 区分不同实现的参数
        line_vals=['triton', 'torch'],  # Triton实现 vs PyTorch实现
        line_names=['Triton', 'Torch'],
        styles=[('blue', '-'), ('green', '-')],
        ylabel='GB/s',  # 吞吐量
        plot_name='conv2d-performance',
        args={},
    ))
def benchmark(input_size, provider):
    """
    Benchmark 2D convolution performance.
    
    Args:
        input_size: Size of the input matrix (input_size x input_size)
        provider: 'triton' for Triton implementation or 'torch' for PyTorch implementation
    
    Returns:
        (gbps_median, gbps_max, gbps_min): Performance metrics in GB/s
    """
    # 固定kernel大小为3x3
    kernel_size = 3
    device = 'cuda'
    dtype = torch.float32
    
    # 创建输入、kernel和输出张量
    input_tensor = torch.randn(input_size, input_size, device=device, dtype=dtype)
    kernel_tensor = torch.randn(kernel_size, kernel_size, device=device, dtype=dtype)
    
    output_size = input_size - kernel_size + 1
    output_tensor = torch.zeros(output_size, output_size, device=device, dtype=dtype)
    
    quantiles = [0.5, 0.2, 0.8]
    
    if provider == 'torch':
        # PyTorch实现
        def torch_bench():
            output = torch.zeros(output_size, output_size, device=device, dtype=dtype)
            torch_solve(input_tensor, kernel_tensor, output,
                       input_size, input_size, kernel_size, kernel_size)
            return output
        
        ms, min_ms, max_ms = triton.testing.do_bench(torch_bench, quantiles=quantiles)
    
    elif provider == 'triton':
        # Triton实现
        def triton_bench():
            output = torch.zeros(output_size, output_size, device=device, dtype=dtype)
            solve(input_tensor, kernel_tensor, output,
                  input_size, input_size, kernel_size, kernel_size)
            return output
        
        ms, min_ms, max_ms = triton.testing.do_bench(triton_bench, quantiles=quantiles)
    
    # 计算吞吐量 (GB/s)
    # 数据量 = input + kernel + output (bytes)
    total_bytes = (input_tensor.numel() + kernel_tensor.numel() + output_tensor.numel()) * dtype_size(dtype)
    gbps = lambda ms: total_bytes / ms * 1e-6
    
    return gbps(ms), gbps(max_ms), gbps(min_ms)


def dtype_size(dtype):
    """Get the size of a torch dtype in bytes."""
    if dtype == torch.float32:
        return 4
    elif dtype == torch.float16:
        return 2
    elif dtype == torch.float64:
        return 8
    else:
        return 4  # default to float32 size


if __name__ == "__main__":
    print("============= run compile test ==============")
    run()

    print("============= run performance test ===========")
    benchmark.run(print_data=True, show_plots=True)

'''
conv2d-performance:
   input_size      Triton       Torch
0        32.0    0.755078    0.943848
1        64.0    2.822798    3.881348
2       128.0   11.459162   14.005642
3       256.0   39.080830   50.805079
4       512.0  145.717911  136.003389
5      1024.0  255.501588  181.690015
6      2048.0  167.877183  205.887108
'''