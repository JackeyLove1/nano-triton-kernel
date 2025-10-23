import torch
import triton
import triton.language as tl
from torch import Tensor

'''
$$y = \frac{x - \mathrm{E}[x]}{\sqrt{\mathrm{Var}(x) + \epsilon}} * w + b$$
'''
@triton.jit
def _layer_norm_fwd_fused(
    X,  # pointer to the input 输入指针
    Y,  # pointer to the output 输出指针
    W,  # pointer to the weights 权重指针
    B,  # pointer to the biases 偏差指针
    Mean,  # pointer to the mean 均值指针
    Rstd,  # pointer to the 1/std 1/std 指针
    stride,  # how much to increase the pointer when moving by 1 row 指针移动一行应该增加多少
    N,  # number of columns in X X 的列数
    eps,  # epsilon to avoid division by zero 用于避免除以 0 的 epsilon
    BLOCK_SIZE: tl.constexpr,
):
    # Map the program id to the row of X and Y it should compute.
    # 映射程序 id 到对应计算的 X 和 Y 的行
    row = tl.program_id(0)
    Y += row * stride
    X += row * stride
    # Compute mean
    # 计算均值
    mean = 0
    _mean = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        a = tl.load(X + cols, mask=cols < N, other=0.).to(tl.float32)
        _mean += a
    mean = tl.sum(_mean, axis=0) / N
    # Compute variance
    # 计算方差
    _var = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        x = tl.load(X + cols, mask=cols < N, other=0.).to(tl.float32)
        x = tl.where(cols < N, x - mean, 0.)
        _var += x * x
    var = tl.sum(_var, axis=0) / N
    rstd = 1 / tl.sqrt(var + eps)
    # Write mean / rstd
    # 写入 mean / rstd
    tl.store(Mean + row, mean)
    tl.store(Rstd + row, rstd)
    # Normalize and apply linear transformation
    # 归一化并应用线性变换
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        w = tl.load(W + cols, mask=mask)
        b = tl.load(B + cols, mask=mask)
        x = tl.load(X + cols, mask=mask, other=0.).to(tl.float32)
        x_hat = (x - mean) * rstd
        y = x_hat * w + b
        # Write output
        tl.store(Y + cols, y, mask=mask)

