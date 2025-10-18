# Nano Triton Kernel

A collection of high-performance GPU operators implemented in **Triton** with performance benchmarks against PyTorch implementations.

## Overview

This repository provides educational implementations of common deep learning operators using the **Triton** programming framework. Each operator includes:
- **Triton kernel implementation** - Efficient GPU code using Triton's DSL
- **PyTorch baseline** - Standard PyTorch reference implementation
- **Benchmarking suite** - Automatic performance comparison using `triton.testing.Benchmark`

## Why Triton?

Triton is a language and compiler for writing highly efficient custom CUDA kernels without low-level GPU programming. It offers:
- **Productivity**: Write kernels in Python-like syntax
- **Performance**: Optimizations comparable to hand-tuned CUDA code
- **Portability**: Works across different GPU architectures

## Operators

| File | Operation | Type |
|------|-----------|------|
| `1_vector_add.py` | Element-wise addition | Arithmetic |
| `2_fused_softmax.py` | Fused softmax computation | Activation |
| `3_dropout.py` | Dropout regularization | Regularization |
| `4_matrix_transpose.py` | Matrix transpose | Linear Algebra |
| `5_color_inversion.py` | Image color inversion (RGBA) | Image Processing |
| `6_1d_convolution.py` | 1D convolution | Convolution |
| `7_reverse_array.py` | In-place array reversal | Utility |
| `8_relu.py` | ReLU activation | Activation |
| `9_leaky_relu.py` | Leaky ReLU activation | Activation |
| `10_rainbow_table.py` | FNV1a hash with iterations | Hashing |

## Installation

```bash
pip install torch triton
```

## Usage

Run any benchmark to compare Triton vs PyTorch performance:

```bash
cd basic/
python 1_vector_add.py
```

## Example Output

```
vector-add-performance:
           size      Triton       Torch
0        4096.0    9.600000   12.000000
1        8192.0   24.000000   24.000000
...
15  134217728.0  691.672797  693.502633
```

## Benchmark Metrics

- **GB/s** (Vector operations): Throughput in gigabytes per second
- **GFLOP/s** (Convolution): Giga floating-point operations per second

## Key Features

✅ **Correctness Verified** - Output validated against PyTorch implementations  
✅ **Multiple Implementations** - Compare different optimization strategies  
✅ **Comprehensive Benchmarks** - Test across various input sizes  
✅ **Clean Code** - Well-commented and educational

## Learning Resources

Each kernel demonstrates important Triton concepts:
- **Memory tiling & block-level operations**
- **Masking & boundary conditions**
- **Cache modifiers & memory hierarchy optimization**
- **Grid & block coordination**

## References

- [Triton Documentation](https://triton-lang.org/)
- [Triton GitHub](https://github.com/openai/triton)
- [Triton Tutorial](https://triton-lang.org/main/getting-started/tutorials/02-fused-softmax.html)

---

*Educational repository for learning GPU programming with Triton*
