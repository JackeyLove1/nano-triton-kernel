'''
Write a program to invert the colors of an image. The image is represented as a 1D array of RGBA (Red, Green, Blue, Alpha) values, where each component is an 8-bit unsigned integer

Color inversion is performed by subtracting each color component (R, G, B) from 255. The Alpha component should remain unchanged.

The input array image will contain width * height * 4 elements. The first 4 elements represent the RGBA values of the top-left pixel, the next 4 elements represent the pixel to its right, and so on.
'''

import torch
import triton
import triton.language as tl
from torch import Tensor

# image is a tensor on the GPU
def invert_torch(image: torch.Tensor, width: int, height: int):
    device = image.device
    v1 = torch.tensor([-1, -1, -1, 1], device=device).repeat(width * height)
    v2 = torch.tensor([255, 255, 255, 0], device=device).repeat(width * height)
    image.copy_((image * v1 + v2).flatten())


@triton.jit
def invert_kernel(
        image,
        width, height,
        BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < width * height

    r_offset = offsets * 4 + 0
    g_offset = offsets * 4 + 1
    b_offset = offsets * 4 + 2

    r_value = tl.load(image + r_offset, mask=mask, other=0)
    g_value = tl.load(image + g_offset, mask=mask, other=0)
    b_value = tl.load(image + b_offset, mask=mask, other=0)


    tl.store(image + r_offset, 255 - r_value, mask=mask)
    tl.store(image + g_offset, 255 - g_value, mask=mask)
    tl.store(image + b_offset, 255 - b_value, mask=mask)


# image is a tensor on the GPU
def solve(image: torch.Tensor, width: int, height: int):
    BLOCK_SIZE = 1024
    n_pixels = width * height
    grid = (triton.cdiv(n_pixels, BLOCK_SIZE),)

    invert_kernel[grid](
        image,
        width, height,
        BLOCK_SIZE
    )