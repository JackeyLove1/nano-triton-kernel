'''
Implement a GPU program that, given a 1D array input of 32-bit floating point numbers of length N, selects the k largest elements and writes them in descending order to the output array of length k.

Implementation Requirements
External libraries are not permitted
The solve function signature must remain unchanged
The final result must be stored in the output array
'''

import torch
import triton
import triton.language as tl

def torch_solve(input: torch.Tensor, output: torch.Tensor, N: int, k: int):
    torch.topk(input, k, sorted=True, out=(output, torch.zeros_like(output, dtype=torch.long)))

# input, output are tensors on the GPU
def solve(input: torch.Tensor, output: torch.Tensor, N: int, k: int):
    pass