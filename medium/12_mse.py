'''
## Mean Squared Error (MSE)

Implement a CUDA program to calculate the **Mean Squared Error (MSE)** between predicted values and target values.
Given two arrays of equal length, `predictions` and `targets`, compute:
$$\text{MSE} = \frac{1}{N} \sum_{i=1}^{N} (\text{predictions}_i - \text{targets}_i)^2$$
where **N** is the number of elements in each array.

### Implementation Requirements
* External libraries are not permitted.
* The `solve` function signature must remain unchanged.
* The final result must be stored in the `mse` variable.
'''
from itertools import accumulate

import torch
import triton
import triton.language as tl
import torch.nn.functional as F

def torch_solve(predictions: torch.Tensor, targets: torch.Tensor, mse: torch.Tensor, N: int):
    mse.copy_(F.mse_loss(predictions, targets))

# predictions, targets, mse are tensors on the GPU

@triton.jit
def mse_kernel(
    predictions, targets, mse,
    N: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    pids = tl.program_id(0)
    offsets = pids * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N

    p = tl.load(predictions + offsets, mask=mask, other=0.0)
    t = tl.load(targets + offsets, mask=mask, other=0)
    diff = p - t
    sq = diff * diff
    block_sum = tl.sum(sq, axis=0)
    tl.atomic_add(mse, block_sum, sem='relaxed')


def solve(predictions: torch.Tensor, targets: torch.Tensor, mse: torch.Tensor, N: int):
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(N, BLOCK_SIZE), )
    accumulate = torch.zeros(1, dtype=predictions.dtype, device=predictions.device)
    mse_kernel[grid](
        predictions, targets, accumulate, N, BLOCK_SIZE
    )
    mse.copy_(accumulate / N)

def run():
    N = 10_000
    predictions = torch.randn(N, dtype=torch.float32, device='cuda')
    targets = torch.randn(N, dtype=torch.float32, device='cuda')
    mse1 = torch.zeros(1, dtype=torch.float32, device='cuda')
    mse2 = torch.zeros_like(mse1)
    solve(predictions, targets, mse1, N)
    torch_solve(predictions, targets, mse2, N)
    print(f"mse1: {mse1}")
    print(f"mse2: {mse2}")
    print(f"equal:{torch.allclose(mse1, mse2)}")

if __name__ == "__main__":
    run()