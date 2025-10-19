'''
# CUDA Softmax Attention — Problem Statement
Implement a CUDA program that computes the **softmax attention** operation for a given set of matrices.
Given:
* Query matrix ( \mathbf{Q} ) of size ( M \times d )
* Key matrix ( \mathbf{K} ) of size ( N \times d )
* Value matrix ( \mathbf{V} ) of size ( N \times d )
compute the output matrix using:
$$
\operatorname{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V})
= \operatorname{softmax}!\left(\frac{\mathbf{Q}\mathbf{K}^{\top}}{\sqrt{d}}\right)\mathbf{V},
$$
where the **softmax is applied row-wise**.
---
## Implementation Requirements

* Use **only CUDA native features** (external libraries are not permitted).
* The `solve` function signature **must remain unchanged**.
* The final result must be stored in the **output** matrix `output`.

'''

import torch
import triton
import triton.language as tl
import math

# Q, K, V, output are tensors on the GPU
def torch_solve(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, output: torch.Tensor,
          M: int, N: int, d: int):
    torch.matmul(torch.nn.functional.softmax(torch.div(Q @ K.T, math.sqrt(d))), V, out=output)

