import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 128},  num_warps=2),
        triton.Config({"BLOCK_SIZE": 256},  num_warps=4),
        triton.Config({"BLOCK_SIZE": 512},  num_warps=4),
        triton.Config({"BLOCK_SIZE": 1024}, num_warps=8),
        triton.Config({"BLOCK_SIZE": 2048}, num_warps=8),
    ],
    key=["N", "K"],
)
@triton.jit
def _1d_conv_kernel(
        a_ptr, b_ptr, c_ptr,
        N: tl.constexpr,
        K: tl.constexpr,
        BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    out_mask = offsets < N

    radius = K // 2  # K is odd by contract
    acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)

    # Unrolled loop
    for k in tl.static_range(0, K):
        a_idx = offsets + (k - radius)
        in_bounds = (a_idx >= 0) & (a_idx < N) & out_mask
        k_val = tl.load(b_ptr + k)
        input_val = tl.load(a_ptr + a_idx, mask=in_bounds, other=0.0)
        acc += k_val * input_val
    tl.store(c_ptr + offsets, acc,  mask=out_mask)


# Note: A, B, C are all float32 device tensors
def solution(A, B, C, N: int, K: int):
    grid = lambda META: (triton.cdiv(N, META["BLOCK_SIZE"]),)
    _1d_conv_kernel[grid](A,B,C,N,K)
