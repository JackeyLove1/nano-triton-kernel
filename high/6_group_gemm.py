# Copyright (c) 2023 NVIDIA Corporation & Affiliates. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining
# a copy of this software and associated documentation files
# (the "Software"), to deal in the Software without restriction,
# including without limitation the rights to use, copy, modify, merge,
# publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
# IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
# CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
# SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


import torch


import triton
import triton.language as tl




@triton.autotune(
    configs=[
        triton.Config({
            'BLOCK_SIZE_M': 128,
            'BLOCK_SIZE_N': 128,
            'BLOCK_SIZE_K': 32,
            'NUM_SM': 84,
        }),
        triton.Config({
            'BLOCK_SIZE_M': 128,
            'BLOCK_SIZE_N': 128,
            'BLOCK_SIZE_K': 32,
            'NUM_SM': 128,
        }),
        triton.Config({
            'BLOCK_SIZE_M': 64,
            'BLOCK_SIZE_N': 64,
            'BLOCK_SIZE_K': 32,
            'NUM_SM': 84,
        }),
        triton.Config({
            'BLOCK_SIZE_M': 64,
            'BLOCK_SIZE_N': 64,
            'BLOCK_SIZE_K': 32,
            'NUM_SM': 128,
        }),
    ],
    key=['group_size'],
)
@triton.jit
def grouped_matmul_kernel(
    # device tensor of matrices pointers
    # 设备张量矩阵指针
    group_a_ptrs,
    group_b_ptrs,
    group_c_ptrs,
    # device tensor of gemm sizes. its shape is [group_size, 3]
    # 设备张量的 GEMM（General Matrix Multiply）大小。其形状为 [group_size, 3]
    # dim 0 is group_size, dim 1 is the values of <M, N, K> of each gemm
    # 第 0 维是 group_size，第 1 维是每个 GEMM 的 <M, N, K> 值
    group_gemm_sizes,
    # device tensor of leading dimension sizes. its shape is [group_size, 3]
    # 设备张量的主导维度大小。其形状为 [group_size, 3]
    # dim 0 is group_size, dim 1 is the values of <lda, ldb, ldc> of each gemm
    # 第 0 维是 group_size，第 1 维是每个 GEMM 的 <lda, ldb, ldc> 值
    g_lds,
    # number of gemms
    # gemms 数量
    group_size,
    # number of virtual SM
    # 虚拟 SM 数量
    NUM_SM: tl.constexpr,
    # tile sizes
    # tile 大小
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    tile_idx = tl.program_id(0)
    last_problem_end = 0
    for g in range(group_size):
        # get the gemm size of the current problem
        # 得到当前问题的 gemm 大小
        gm = tl.load(group_gemm_sizes + g * 3)
        gn = tl.load(group_gemm_sizes + g * 3 + 1)
        gk = tl.load(group_gemm_sizes + g * 3 + 2)
        num_m_tiles = tl.cdiv(gm, BLOCK_SIZE_M)
        num_n_tiles = tl.cdiv(gn, BLOCK_SIZE_N)
        num_tiles = num_m_tiles * num_n_tiles
        # iterate through the tiles in the current gemm problem
        # 迭代当前 GEMM 问题中的 tiles
        while (tile_idx >= last_problem_end and tile_idx < last_problem_end + num_tiles):
            # pick up a tile from the current gemm problem
            # 从当前 GEMM 问题选择一个 title
            k = gk
            lda = tl.load(g_lds + g * 3)
            ldb = tl.load(g_lds + g * 3 + 1)
            ldc = tl.load(g_lds + g * 3 + 2)
            a_ptr = tl.load(group_a_ptrs + g).to(tl.pointer_type(tl.float16))
            b_ptr = tl.load(group_b_ptrs + g).to(tl.pointer_type(tl.float16))
            c_ptr = tl.load(group_c_ptrs + g).to(tl.pointer_type(tl.float16))
            # figure out tile coordinates
            # 确定 title 坐标
            tile_idx_in_gemm = tile_idx - last_problem_end
            tile_m_idx = tile_idx_in_gemm // num_n_tiles
            tile_n_idx = tile_idx_in_gemm % num_n_tiles


            # do regular gemm here
            # 此处进行常规 gemm
            offs_am = tile_m_idx * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
            offs_bn = tile_n_idx * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
            offs_k = tl.arange(0, BLOCK_SIZE_K)
            a_ptrs = a_ptr + offs_am[:, None] * lda + offs_k[None, :]
            b_ptrs = b_ptr + offs_k[:, None] * ldb + offs_bn[None, :]
            accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
            for kk in range(0, tl.cdiv(k, BLOCK_SIZE_K)):
                # hint to Triton compiler to do proper loop pipelining
                # 提示 Triton 编译器进行适当的循环流水线处理
                tl.multiple_of(a_ptrs, [16, 16])
                tl.multiple_of(b_ptrs, [16, 16])
                # assume full tile for now
                # 现在假设完整的 tile
                a = tl.load(a_ptrs)
                b = tl.load(b_ptrs)
                accumulator += tl.dot(a, b)
                a_ptrs += BLOCK_SIZE_K
                b_ptrs += BLOCK_SIZE_K * ldb
            c = accumulator.to(tl.float16)


            offs_cm = tile_m_idx * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
            offs_cn = tile_n_idx * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
            c_ptrs = c_ptr + ldc * offs_cm[:, None] + offs_cn[None, :]


            # assumes full tile for now
            # 现在假设完整的 tile
            tl.store(c_ptrs, c)


            # go to the next tile by advancing NUM_SM
            # 通过增加 NUM_SM 来进入下一个 tile
            tile_idx += NUM_SM


        # get ready to go to the next gemm problem
        # 准备进入下一个 gemm 问题
        last_problem_end = last_problem_end + num_tiles




def group_gemm_fn(group_A, group_B):
    device = torch.device('cuda')
    assert len(group_A) == len(group_B)
    group_size = len(group_A)


    A_addrs = []
    B_addrs = []
    C_addrs = []
    g_sizes = []
    g_lds = []
    group_C = []
    for i in range(group_size):
        A = group_A[i]
        B = group_B[i]
        assert A.shape[1] == B.shape[0]
        M, K = A.shape
        K, N = B.shape
        C = torch.empty((M, N), device=device, dtype=A.dtype)
        group_C.append(C)
        A_addrs.append(A.data_ptr())
        B_addrs.append(B.data_ptr())
        C_addrs.append(C.data_ptr())
        g_sizes += [M, N, K]
        g_lds += [A.stride(0), B.stride(0), C.stride(0)]


    # note these are device tensors
    # 注意这些是设备张量
    d_a_ptrs = torch.tensor(A_addrs, device=device)
    d_b_ptrs = torch.tensor(B_addrs, device=device)
    d_c_ptrs = torch.tensor(C_addrs, device=device)
    d_g_sizes = torch.tensor(g_sizes, dtype=torch.int32, device=device)
    d_g_lds = torch.tensor(g_lds, dtype=torch.int32, device=device)
    # we use a fixed number of CTA, and it's auto-tunable
    # 我们使用固定数量的 CTA（线程块），并且它是自动可调节的
    grid = lambda META: (META['NUM_SM'], )
    grouped_matmul_kernel[grid](
        d_a_ptrs,
        d_b_ptrs,
        d_c_ptrs,
        d_g_sizes,
        d_g_lds,
        group_size,
    )


    return group_C




group_m = [1024, 1024, 512, 128]
group_n = [1024, 512, 1024, 128]
group_k = [1024, 2048, 256, 128]
group_A = []
group_B = []
assert len(group_m) == len(group_n)
assert len(group_n) == len(group_k)
group_size = len(group_m)
for i in range(group_size):
    M = group_m[i]
    N = group_n[i]
    K = group_k[i]
    A = torch.rand((M, K), device="cuda", dtype=torch.float16)
    B = torch.rand((K, N), device="cuda", dtype=torch.float16)
    group_A.append(A)
    group_B.append(B)


tri_out = group_gemm_fn(group_A, group_B)
ref_out = [torch.matmul(a, b) for a, b in zip(group_A, group_B)]
for i in range(group_size):
    assert torch.allclose(ref_out[i], tri_out[i], atol=1e-2, rtol=0)




# only launch the kernel, no tensor preparation here to remove all overhead
# 只启动内核，这里不进行张量准备，以移除所有开销。
def triton_perf_fn(a_ptrs, b_ptrs, c_ptrs, sizes, lds, group_size):
    grid = lambda META: (META['NUM_SM'], )
    grouped_matmul_kernel[grid](
        a_ptrs,
        b_ptrs,
        c_ptrs,
        sizes,
        lds,
        group_size,
    )




def torch_perf_fn(group_A, group_B):
    for a, b in zip(group_A, group_B):
        torch.matmul(a, b)




@triton.testing.perf_report(
    triton.testing.Benchmark(
        # argument names to use as an x-axis for the plot
        # 用作绘图 x 轴的参数名称
        x_names=['N'],
        x_vals=[2**i for i in range(7, 11)],  # different possible values for `x_name` `x_name` 可能的不同取值
        line_arg='provider',
        # argument name whose value corresponds to a different line in the plot 参数名称，其值对应绘图中的不同线条
        # possible values for `line_arg``
        # `line_arg` 的可能取值
        line_vals=['cublas', 'triton'],
        # label name for the lines
        # 线条的标签名称
        line_names=["cuBLAS", "Triton"],
        # line styles
        # 线条样式
        styles=[('green', '-'), ('blue', '-')],
        ylabel="runtime(ms)",  # label name for the y-axis y 轴标签名称
        plot_name="group-gemm-performance",
        # name for the plot. Used also as a file name for saving the plot.
        # 绘图的名称。同时也作为保存绘图的文件名使用。
        args={},
    ))
def benchmark(N, provider):
    group_size = 4
    group_A = []
    group_B = []
    A_addrs = []
    B_addrs = []
    C_addrs = []
    g_sizes = []
    g_lds = []
    group_C = []
    for i in range(group_size):
        A = torch.rand((N, N), device="cuda", dtype=torch.float16)
        B = torch.rand((N, N), device="cuda", dtype=torch.float16)
        C = torch.empty((N, N), device="cuda", dtype=torch.float16)
        group_A.append(A)
        group_B.append(B)
        group_C.append(C)
        A_addrs.append(A.data_ptr())
        B_addrs.append(B.data_ptr())
        C_addrs.append(C.data_ptr())
        g_sizes += [N, N, N]
        g_lds += [N, N, N]


    d_a_ptrs = torch.tensor(A_addrs, device="cuda")
    d_b_ptrs = torch.tensor(B_addrs, device="cuda")
    d_c_ptrs = torch.tensor(C_addrs, device="cuda")
    d_g_sizes = torch.tensor(g_sizes, dtype=torch.int32, device="cuda")
    d_g_lds = torch.tensor(g_lds, dtype=torch.int32, device="cuda")


    quantiles = [0.5, 0.2, 0.8]
    if provider == 'cublas':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: torch_perf_fn(group_A, group_B), quantiles=quantiles)
    if provider == 'triton':
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: triton_perf_fn(d_a_ptrs, d_b_ptrs, d_c_ptrs, d_g_sizes, d_g_lds, group_size), quantiles=quantiles)
    return ms, max_ms, min_ms




benchmark.run(show_plots=True, print_data=True)

'''
group-gemm-performance:
        N    cuBLAS    Triton
0   128.0  0.014336  0.008192
1   256.0  0.017408  0.010240
2   512.0  0.053248  0.027648
3  1024.0  0.190464  0.162816
'''

'''
好的，我们来详细解析一下分组GEMM（Grouped GEMM）的概念、作用以及您提供的这段 Triton 代码是如何实现它的。

---

### 1. 分组GEMM是什么？(What is Grouped GEMM?)

**分组GEMM** (Grouped General Matrix Multiplication) 是一种对多个独立的矩阵乘法问题进行批处理（batch processing）的计算操作。

与常规的**批处理GEMM (Batched GEMM)** 的关键区别在于：
* **Batched GEMM**: 要求批次中所有的矩阵乘法具有**完全相同**的维度 (M, N, K)。例如，一次性计算16个 `[128, 256] x [256, 512]` 的矩阵乘法。
* **Grouped GEMM**: 允许批次中的每个矩阵乘法具有**不同**的维度 (M, N, K)。例如，一次性计算下面三个独立的GEMM：
    * Problem 1: `C₁[100, 200] = A₁[100, 300] x B₁[300, 200]`
    * Problem 2: `C₂[50, 80] = A₂[50, 120] x B₂[120, 80]`
    * Problem 3: `C₃[400, 150] = A₃[400, 50] x B₃[50, 150]`

简单来说，分组GEMM就是将一堆“尺寸不一”的矩阵乘法任务打包在一起，交给GPU一次性高效完成。

### 2. 分组GEMM有什么作用？(What is its purpose?)

在现代深度学习，尤其是大语言模型（LLM）推理中，分组GEMM非常重要，其主要作用是**提高GPU利用率和吞吐量**。

这在以下场景中尤其关键：

1.  **混合专家模型 (Mixture-of-Experts, MoE)**:
    * 在MoE模型（如Mixtral）中，输入的每个token会被路由到少数几个“专家”（Expert，通常是FFN/MLP）。
    * 因为路由是动态的，所以每个专家接收到的token数量（即batch size）是不一样的。
    * 这意味着在同一个推理步骤中，你需要为每个专家执行一个矩阵乘法，而这些乘法的M维度（batch size）各不相同。使用分组GEMM可以高效地处理这些不同尺寸的计算任务。

2.  **处理不同序列长度的批处理请求 (Batched Inference with Varying Sequence Lengths)**:
    * 当多个用户同时向模型发送请求时，他们的输入序列长度通常是不同的。
    * 传统的做法是将所有输入填充（padding）到最长的序列长度，这会造成大量的无效计算和内存浪费。
    * 使用分组GEMM，可以将每个请求的计算（例如在Attention或MLP层中）看作一个独立的GEMM问题，并将这些尺寸不同的问题组合成一个group来执行，从而避免了不必要的padding。

**核心优势**: 分组GEMM通过将许多小的、异构的计算任务捆绑在一起，使得GPU的计算核心（SMs）能够持续保持繁忙状态，避免了因任务尺寸小或需要等待同步而产生的空闲时间，从而显著提升了整体的计算效率。

---

### 3. 为什么会使用CTA？以及代码分析

这里的“CTA”是CUDA编程中的一个核心概念，指的是 **Cooperative Thread Array**，在NVIDIA的文档中也常被称为 **Thread Block**（线程块）。

* **CTA/Thread Block**: 一组线程，它们被调度到GPU上的同一个流式多处理器（Streaming Multiprocessor, SM）上执行。同一个CTA内的线程可以高效地通过共享内存（Shared Memory）进行通信和同步。
* **在Triton中**: `triton.jit` 装饰的函数（即一个kernel）会被编译成CUDA代码。我们通过 `tl.program_id(axis)` 来启动和区分不同的程序实例。**每一个Triton的程序实例（program instance）通常就对应一个CTA/Thread Block**。

所以，您的问题“为什么会使用CTA？”可以理解为“**这段代码是如何组织和分配CTA来并行解决这些不同尺寸的GEMM问题的？**”

#### 代码并行策略分析

这段代码采用了一种非常聪明且高效的并行策略，可以称之为 **"虚拟化网格 (Virtualized Grid)"** 或 **"问题空间切片 (Problem-Space Tiling)"**。

让我们一步步分解它的逻辑：

1.  **全局Tile索引 (`tile_idx`)**:
    * `tile_idx = tl.program_id(0)`
    * 代码启动了 `NUM_SM` 个程序实例（CTA）。每个CTA被分配了一个从 `0` 到 `NUM_SM - 1` 的唯一ID。这个 `tile_idx` 是一个**全局的、跨越所有GEMM问题的tile索引**。
    * 想象一下，我们将所有GEMM问题的输出矩阵C在逻辑上“拼接”起来，然后将这个巨大的虚拟矩阵划分成一个个 `BLOCK_SIZE_M x BLOCK_SIZE_N` 大小的tile。`tile_idx` 就是这些tile的全局编号。

2.  **遍历问题以找到自己的任务 (`for g in range(group_size)`)**:
    * 这个 `for` 循环**不是**让一个CTA按顺序解决所有问题。
    * 相反，它的作用是让**每一个CTA**确定自己被分配的 `tile_idx` 到底属于哪一个GEMM问题 (`g`)。

3.  **核心判断逻辑 (`while` 循环)**:
    * `num_tiles = num_m_tiles * num_n_tiles`: 计算当前问题 `g` 总共有多少个tile。
    * `while (tile_idx >= last_problem_end and tile_idx < last_problem_end + num_tiles)`: 这是关键。它检查当前CTA的 `tile_idx` 是否落在第 `g` 个问题的tile区间内。`last_problem_end` 记录了前面 `g-1` 个问题总共占用了多少个tile。
    * 如果条件满足，那么这个CTA就知道它的任务是计算第 `g` 个GEMM问题中的某一个tile。

4.  **计算Tile并执行GEMM**:
    * 一旦CTA确定了它属于问题 `g`，它就会加载该问题对应的矩阵指针 (`a_ptr`, `b_ptr`) 和维度 (`lda`, `k` 等)。
    * `tile_idx_in_gemm = tile_idx - last_problem_end`: 计算出这是当前问题内部的第几个tile。
    * `tile_m_idx = ...`, `tile_n_idx = ...`: 从内部tile索引计算出具体的二维坐标。
    * 接下来的部分就是标准的Triton GEMM实现：分块加载A和B，在累加器中计算点积，最后写回C。

5.  **工作窃取/动态调度 (`tile_idx += NUM_SM`)**:
    * 这是实现高效率和负载均衡的**神来之笔**。
    * 当一个CTA完成了它当前的tile计算后，它不是直接退出。
    * 它通过 `tile_idx += NUM_SM` 将自己的全局索引向前跳跃一个“步长”，这个步长等于启动的CTA总数。
    * 然后 `while` 循环会继续判断这个新的 `tile_idx` 属于哪个任务。这样，一个完成了小任务的CTA可以立即去“窃取”或“认领”一个还未被计算的tile，无论这个tile属于哪个GEMM问题。
    * 这种机制确保了即使某些GEMM问题很大，而另一些很小，所有CTA也能保持繁忙，共同完成所有工作，实现了出色的**负载均衡 (Load Balancing)**。

### 总结

* **分组GEMM** 是一种用于高效处理一批**尺寸不同**的独立矩阵乘法任务的技术，广泛应用于LLM推理中的MoE和变长序列处理等场景。
* 您提供的Triton代码通过一种**虚拟化网格**的策略来调度**CTA (Triton中的`program_id`实例)**。
* 它不为每个GEMM问题静态分配CTA，而是创建一个全局的tile池，让所有CTA动态地从中领取任务。
* 通过 `tile_idx += NUM_SM` 的**步进（grid-stride）**方式，实现了高效的**工作窃取**和**负载均衡**，确保GPU资源被最大化利用，即使在处理高度异构的任务时也能保持高吞吐量。
'''