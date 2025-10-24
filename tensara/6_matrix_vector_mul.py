import torch
import triton
import triton.language as tl

@triton.jit
def _spmv_kernel_2d(
        a_ptr, b_ptr, c_ptr,
        M: tl.constexpr,
        K: tl.constexpr,
        BLOCK_SIZE: tl.constexpr
):
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)

    if pid_m >= M:
        return

    off_k = pid_k * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = off_k < K

    a_offset = pid_m * K + off_k
    b_offset = off_k

    a = tl.load(a_ptr + a_offset, mask=mask, other=0.0)
    b = tl.load(b_ptr + b_offset, mask=mask, other=0.0)
    partial = tl.sum(a * b, axis=0)
    # tl.device_print("c: ", c)

    tl.atomic_add(c_ptr + pid_m, partial)

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 128},  num_warps=2),
        triton.Config({"BLOCK_SIZE": 256},  num_warps=4),
        triton.Config({"BLOCK_SIZE": 512},  num_warps=4),
        triton.Config({"BLOCK_SIZE": 1024}, num_warps=8),
        triton.Config({"BLOCK_SIZE": 2048}, num_warps=8),
    ],
    key=["M", "K"],
)
@triton.jit
def _spmv_kernel_1d(
        a_ptr, b_ptr, c_ptr,
        M: tl.constexpr,
        K: tl.constexpr,
        BLOCK_SIZE: tl.constexpr
):
    pid_m = tl.program_id(0)

    if pid_m >= M:
        return

    row_base = pid_m * K
    acc = tl.zeros((), dtype=tl.float32)

    for k0 in tl.static_range(0, K, BLOCK_SIZE):
        offset = k0 + tl.arange(0, BLOCK_SIZE)
        mask = offset < K
        a = tl.load(a_ptr + row_base + offset, mask=mask, other=0.0)
        b = tl.load(b_ptr + offset, mask=mask, other=0.0)
        acc += tl.sum(a * b, axis=0)

    tl.store(c_ptr + pid_m, acc)

# Note: input_a, input_b, output_c are all float32 device tensors
def solution_2d(input_a, input_b, output_c, m: int, k: int):
    BLOCK_SIZE = 1024
    def grid(META):
        return (
            m,
            triton.cdiv(k, BLOCK_SIZE),
        )
    _spmv_kernel_2d[grid](input_a, input_b, output_c, m, k, BLOCK_SIZE)

# Note: input_a, input_b, output_c are all float32 device tensors
def solution(input_a, input_b, output_c, m: int, k: int):
    def grid(META):
        return (
            m,
            triton.cdiv(k, META['BLOCK_SIZE']),
        )
    _spmv_kernel_1d[grid](input_a, input_b, output_c, m, k)

if __name__ == "__main__":
    M = 1024
    K = 10_000
    a = torch.randn((M, K)).float().cuda()
    b = torch.randn((K, )).float().cuda()
    c1 = torch.zeros((M, )).float().cuda()
    solution(a, b, c1, M, K)
    c2 = torch.matmul(a, b)
    print(f"c1: {c1}")
    print(f"c2: {c2}")
    print(f"close: {torch.allclose(c1, c2, rtol=0.15)}")