'''
Implement a 2D max pooling operation for image/feature map downsampling. The program should take an input tensor and produce an output tensor by applying max pooling with specified kernel size, stride, and padding.

Implementation Requirements
External libraries are not permitted
The solve function signature must remain unchanged
The final result must be stored in tensor output
'''

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

def torch_solve(input: torch.Tensor, output: torch.Tensor,
          N: int, C: int, H: int, W: int,
          kernel_size: int, stride: int, padding:int):
   out = F.max_pool2d(
      input=input.view(N, C, H, W),
      kernel_size=kernel_size,
      stride=stride,
      padding=padding
   )
   output.copy_(out.flatten())

# ==============================================================================


def solve(input, output, N, C, H, W, kernel_size, stride, padding):
   pass