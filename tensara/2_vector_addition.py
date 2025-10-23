import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
    triton.Config({"BLOCK_SIZE":256}),
    triton.Config({"BLOCK_SIZE":512}),
    triton.Config({"BLOCK_SIZE":1024}),
    triton.Config({"BLOCK_SIZE":2048}),
    ],
    key=['N']
)
@triton.jit
def _vector_add_kernel(
        a_ptr, b_ptr, c_ptr,
        N: tl.constexpr,
        BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    a = tl.load(a_ptr + offsets, mask=mask, other=0.0)
    b = tl.load(b_ptr + offsets, mask=mask, other=0.0)
    c = a + b
    tl.store(c_ptr + offsets, c, mask=mask)


# Note: d_input1, d_input2, d_output are all float32 device tensors
def solution(d_input1, d_input2, d_output, n: int):
    grid = lambda META: (triton.cdiv(n, META["BLOCK_SIZE"]), )
    _vector_add_kernel[grid](
        d_input1, d_input2, d_output, n
    )

if __name__ == "__main__":
    N = 1_000_000
    a = torch.rand(N).float().cuda()
    b = torch.rand(N).float().cuda()
    c = torch.empty_like(a)
    solution(a, b, c, N)