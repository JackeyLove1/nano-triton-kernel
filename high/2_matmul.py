'''
z-order computing swizzling

pid_n=0  pid_n=1  pid_n=2  pid_n=3
      ┌───────┬───────┬───────┬───────┐
pid_m=0 │ pid=0 │ pid=4 │ pid=8 │ pid=12│  ┐
pid_m=1 │ pid=1 │ pid=5 │ pid=9 │ pid=13│  │
pid_m=2 │ pid=2 │ pid=6 │ pid=10│ pid=14│  ├─ Group 0
pid_m=3 │ pid=3 │ pid=7 │ pid=11│ pid=15│  ┘
      ├───────┼───────┼───────┼───────┤
pid_m=4 │ pid=16│ pid=20│ pid=24│ pid=28│  ┐
pid_m=5 │ pid=17│ pid=21│ pid=25│ pid=29│  │
pid_m=6 │ pid=18│ pid=22│ pid=26│ pid=30│  ├─ Group 1
pid_m=7 │ pid=19│ pid=23│ pid=27│ pid=31│  ┘
      └───────┴───────┴───────┴───────┘

help understand code:
def calculate(
        pid: int,
        num_pid_m: int,
        num_pid_n: int,
        GROUP_SIZE_M: int,
):
    pid = pid
    num_pid_m = num_pid_m
    num_pid_n = num_pid_n
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    group_offset = pid % num_pid_in_group
    print(f"group offset: {group_offset}")
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (group_offset % group_size_m)
    pid_n = group_offset // group_size_m
    print(f" pid_m: {pid_m}, pid_N: {pid_n}")
'''

import torch
import triton
import triton.language as tl
from torch import Tensor

def get_autotune_config():
    return [
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64, 'GROUP_SIZE_M': 8}, num_stages=3,
                      num_warps=8),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32, 'BLOCK_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32, 'BLOCK_K': 32, 'GROUP_SIZE_M': 8}, num_stages=5,
                      num_warps=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_SIZE_M': 8}, num_stages=5,
                      num_warps=2),
        # Good config for fp8 inputs.
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 128, 'GROUP_SIZE_M': 8}, num_stages=3,
                      num_warps=8),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 128, 'GROUP_SIZE_M': 8}, num_stages=3,
                      num_warps=8),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64, 'BLOCK_K': 128, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 128, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 128, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 64, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32, 'BLOCK_K': 64, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4)
    ]


'''
Kernel for computing the matmul C = A x B.
A has shape (M, K), B has shape (K, N) and C has shape (M, N)
'''
@triton.autotune(
    configs=get_autotune_config(),
    key=['M', 'N', 'K']
)
@triton.jit
def matmul_kernel(
        a_ptr, b_ptr, c_ptr,
        M, N, K,
        stride_am, stride_ak,
        stride_bk, stride_bn,
        stride_cm, stride_cn,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
        GROUP_SIZE_M: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    group_offset = pid % num_pid_in_group
    first_pid_m = GROUP_SIZE_M * group_id
    group_size_m = tl.minimum(GROUP_SIZE_M, num_pid_m - first_pid_m)
    pid_m = first_pid_m + (group_offset % group_size_m)
    pid_n = group_offset // group_size_m

    off_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    off_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    off_k = tl.arange(0, BLOCK_K)

    mask_m = off_m[:, None] < M
    mask_n = off_n[None, :] < N

    a_ptrs = a_ptr + (off_m[:, None] * stride_am + off_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (off_k[:, None] * stride_bk + off_n[None, :] * stride_bn)

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # hints
    tl.multiple_of(off_k, 8) # good for fp16 tensor cores when BLOCK_K % 8 == 0

    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_left = K - k * BLOCK_K
        k_mask_row = (off_k[None, :] < k_left)  # shape (1, BLOCK_K) for A
        k_mask_col = (off_k[:, None] < k_left)  # shape (BLOCK_K, 1) for B

        a = tl.load(a_ptrs, mask=mask_m & k_mask_row, other=0.0)
        b = tl.load(b_ptrs, mask=k_mask_col & mask_n, other=0.0)

        accumulator += tl.dot(a, b)

        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    c = accumulator.to(tl.float16)

    off_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    off_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = c_ptr + (off_cm[:, None] * stride_cm + off_cn[None, :] * stride_cn)
    c_mask = mask_m & mask_n
    tl.store(c_ptrs, c, mask=c_mask)

def matmul(a: Tensor, b: Tensor) -> Tensor:
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    assert a.is_contiguous(), "Matrix A must be contiguous"
    M, K = a.shape
    K, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=torch.float16)
    # BLOCK_M = 64
    # BLOCK_N = 64
    # BLOCK_K = 32
    # GROUP_SIZE_M = 8
    # grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N), )
    grid = lambda META: (triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']),)
    matmul_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
    )
    return c

def run():
    torch.manual_seed(0)
    a = torch.randn((1024, 512), device='cuda', dtype=torch.float16)
    b = torch.randn((512, 1024), device='cuda', dtype=torch.float16)
    triton_output = matmul(a, b)
    torch_output = torch.matmul(a, b)
    print(f"triton_output_with_fp16_inputs={triton_output}")
    print(f"torch_output_with_fp16_inputs={torch_output}")
    if torch.allclose(triton_output, torch_output, atol=0.125, rtol=0):
        print("✅ Triton and Torch match")
    else:
        print("❌ Triton and Torch differ")


if __name__ == "__main__":
    run()