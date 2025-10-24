import triton
import triton.language as tl

@triton.jit
def _gelu_erf(x):
    x32 = x.to(tl.float32)
    inv_sqrt2 = 0.7071067811865476  # 1/sqrt(2)
    return 0.5 * x32 * (1.0 + tl.erf(x32 * inv_sqrt2))

@triton.jit
def tanh_stable(z):
    # tanh(z) = 2*sigmoid(2z) - 1，避免直接 exp(2z) 溢出
    return 2.0 / (1.0 + tl.exp(-2.0 * z)) - 1.0

@triton.jit
def _gelu_approx(x):
    x32 = x.to(tl.float32)
    k = 0.7978845608028654  # sqrt(2/pi)
    inner = k * (x32 + 0.044715 * x32 * x32 * x32)
    return 0.5 * x32 * (1.0 + tanh_stable(inner))

@triton.jit
def _gelu_kernel(
        input_ptr, output_ptr,
        M, N,
        BLOCK_SIZE: tl.constexpr
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    if pid_m >= M:
        return

    row_base = pid_m * N
    offsets = pid_n * BLOCK_SIZE +  tl.arange(0, BLOCK_SIZE)
    mask = offsets < N

    x = tl.load(input_ptr + row_base + offsets, mask=mask, other=0.0)
    y = _gelu_approx(x)

    tl.store(output_ptr + row_base + offsets, y, mask=mask)

# Note: input, output are all float32 device tensors
def solution(input, output, n: int, m: int):
    BLOCK_SIZE = 1024
    def grid(META):
        return (
            m,
            triton.cdiv(n, BLOCK_SIZE)
        )
    _gelu_kernel[grid](input, output, m, n, BLOCK_SIZE)


if __name__ == "__main__":
    M = 10_000
    N = 5_000
    import torch
    input = torch.randn((M, N)).float().cuda()
    output = torch.zeros_like(input)
    solution(input, output, N, M)