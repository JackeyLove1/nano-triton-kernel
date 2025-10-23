import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 32,  "BLOCK_N": 32},  num_warps=4),
        triton.Config({"BLOCK_M": 64,  "BLOCK_N": 64},  num_warps=8),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64},  num_warps=8),
        triton.Config({"BLOCK_M": 32,  "BLOCK_N": 64},  num_warps=4),
    ],
    key=["M", "N"],
)
@triton.jit
def _relu_kernel(
        input_ptr, output_ptr,
        alpha: tl.constexpr,
        M: tl.constexpr,
        N: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    off_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]  # (BM, 1)
    off_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)[None, :]  # (1, BN)

    mask = (off_m < M) & (off_n < N)
    offsets = off_m * N + off_n  # row-major: row * N + col

    x = tl.load(input_ptr + offsets, mask=mask, other=0.0)

    y = tl.where(x > 0, x, alpha * x)
    tl.store(output_ptr + offsets, y, mask=mask)



# Note: A, B, C are all float32 device tensors
def solution(input, alpha: float, output, n: int, m: int):
    def grid(meta):
        return (
            triton.cdiv(m, meta["BLOCK_M"]),
            triton.cdiv(n, meta["BLOCK_N"]),
        )
    _relu_kernel[grid](input, output, alpha, m, n)

