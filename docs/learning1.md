下面给你一条**由浅入深、直达贡献 PR** 的学习路线，只挑“必要基础 + 高命中”内容，并为每一环配上质量较高的 **YouTube** 学习资源（可直接看/跳学）。你可以把它当作 4 大板块：**编译基础 → LLVM/MLIR → GPU并行与CUDA/HIP → TVM/TensorIR/TileLang 实战**，再加上**性能分析与贡献流程**补全闭环。

---

# 0. 速览（建立整体图）

* **目标心智图**：前端 DSL（TileLang/Python）→ 降到 TVM TensorIR → Pass 优化/调度 → 目标后端（CUDA/HIP…）→ 配合分析器调优。
* 建议先看 1～2 个关于 **TVM/TensorIR/MLC** 的总览视频，形成“全栈编译”轮廓：

  * *Machine Learning Compilation* 总览（MLC 课程 Ep1、Ep2/3/4）([YouTube][1])
  * TVM 总体介绍（FCRC 教程 1/9）与 TensorIR 抽象讲解 ([YouTube][2])
  * TileLang 官方文档：快速理解“tile 一等抽象”与 API 名词（非视频，但建议配读）([tilelang.com][3])

---

# 1. 编译原理（够用就好）

> 目标：能看懂“词法/语法/IR/优化 pass”的基本术语；明白“前端—中端—后端”的职责边界。

**推荐视频**

* Stanford **CS143 Compilers**（Alex Aiken）整套（理论+实践平衡，条理清晰）([YouTube][4])
* Stanford Online 课程页（同源扩展材料）([online.stanford.edu][5])

> 取用策略：看完**词法/语法/IR**与**数据流优化**相关章节即可；把“代码生成后端”章节与 LLVM/MLIR 结合到下一步一起看。

---

# 2. LLVM / MLIR（理解 IR 与 Pass 生态）

> 目标：理解现代编译器 IR 设计（SSA）、Pass 管线、Dialect 思想；能读懂/跟跑简单 Pass。

**推荐视频**

* **EuroLLVM Tutorial**：LLVM 一小时快速巡游（架构与工具链）([YouTube][6])
* **How to build a compiler with LLVM & MLIR** 系列（入门到上手写点东西）([YouTube][7])
* 讲解 LLVM 工具与 IR 的入门视频（可作补充）([YouTube][8])

---

# 3. GPU 并行基础与 CUDA/HIP（执行模型与内存层次）

> 目标：吃透“线程/warp/block/grid”“共享/寄存器/全局内存”“访存合并/占用率/隐藏延迟”等关键概念；能写/读简单 CUDA/HIP kernel。

**并行体系与 GPU 架构**

* **CMU 15-418/618 并行课**（经典，讲内存层次/工作划分很透）([YouTube][9])
* Stanford CS149（Kayvon 关于一致性/内存层次的相关讲）([YouTube][10])

**CUDA 入门与实践**

* 官方/社区 **CUDA 教程播放列表**（任选其一做快速实操）([YouTube][11])
* NVIDIA Developer 官方频道（长期更新工具/最佳实践）([YouTube][12])

**HIP/ROCm（便于后续跨后端观念）**

* AMD 官方 **ROCm/HIP Workshop** 与 **rocprof/roctracer** 入门 ([YouTube][13])

---

# 4. 性能分析与剖析工具（为内核调优与回归测试做准备）

> 目标：学会用 Nsight Compute/Systems、rocprof 定位瓶颈（带宽、占用、访存效率、指令级细节）。

**NVIDIA Nsight**

* Intro to **Nsight Compute/Systems**（官方/优质教程）([YouTube][14])
* Nsight Developer Tools 专题播放列表与工具导航页（总入口）([YouTube][15])

**AMD ROCm**

* **rocprof/Profiler** 基础与工作坊录播 ([YouTube][16])

---

# 5. TVM / TensorIR（TileLang 的“中台”）

> 目标：能读写简单 TensorIR；理解调度/算子张量化、自动调优思路；知道 Relay ↔ TensorIR 的分层。

**核心资源**

* **TensorIR 专题与创建教程**（TVM 官方文档/讲座）([Apache TVM][17])
* **TVMCon / TVM 教程与大会播放列表**（量大但可按需检索：量化/自动调优/新后端整合）([YouTube][18])
* **MLC-AI 课程播放列表**（陈天奇主讲，系统讲解编译抽象与自动化优化）([YouTube][19])

---

# 6. Triton（类 TileLang 的块化编程范式，对照理解）

> 目标：把“按 tile 写 kernel + 编译器补齐细节”的设计吃透，便于与 TileLang 的抽象做迁移/对比。

**推荐视频**

* **Triton 官方频道/大会**（会议、社区月会、Enable Triton on Intel GPUs 等）([YouTube][20])
* 入门到手写 softmax/matmul 的教程播放列表（代码驱动）([YouTube][21])
* *Triton Internals* 分享（理解后端/Pass 对 TileLang 迁移有帮助）([YouTube][22])

---

# 7. TileLang 专项（从用户到贡献者）

> 目标：能看懂/运行官方示例，读 IR 降低路径，定位可以贡献的小切口（文档、小优化、算子/后端适配点）。

**资料与路径**

* **TileLang 官方文档（Get Started/Overview/API）**：先跑 GEMM/Attention 示例，理解 `T.Kernel/T.Pipelined/T.copy/T.gemm` 等原语的 IR 降级行为。([tilelang.com][23])
* **TileLang 仓库（README/Quick Start/示例）**：对照源代码与生成 IR，梳理前端→TIR→后端的关节点。([GitHub][24])

> 做法建议：
>
> 1. 选 **GEMM/FlashAttention** 之一跑通 → 打印/保存生成的 TIR；
> 2. 改 1～2 个参数（tile 大小/流水线级数）→ 结合 Nsight/rocprof 对比性能指标；
> 3. 从文档修缮或小型 bugfix 切入第一单 PR（降低集成风险）。

---

# 8. 进阶选修（算子库与硬件特性，助力高阶优化）

* **CUTLASS 概念/实践**（理解 Tensor Core 路径与 tile/布局代数）([YouTube][25])
* **更多 GPU 性能清单**（实战化 checklist/案例）([YouTube][26])

---

## 学习顺序（最短可行路径）

1. 看 **MLC Ep1 + TVM 概览**（总览 1–2 小时）→ **TensorIR 基础** 小节。([YouTube][1])
2. 并行/GPU 基础任选一套（CMU 15-418 的并行模型 + 一套短篇 CUDA 入门）。([YouTube][9])
3. 学会 **Nsight/rocprof** 的基本用法（定位瓶颈所必需）。([YouTube][14])
4. 过一遍 **Triton 入门**（对照 TileLang 的“tile范式”）。([YouTube][21])
5. 回到 **TileLang 文档与示例**，跑通 GEMM/Attention，打印 TIR，对照优化效果。([tilelang.com][23])
6. 最后补 **编译原理/LLVM-MLIR**（为读 Pass/改代码打下 IR 素养）。([YouTube][4])

---

### 小贴士（直奔贡献）

* **先文档与小修补**：降低合入摩擦（API docstring、错误信息、边界处理）；
* **加单元/基准**：任何改动都配 **最小复现 + 测试**，并用 Nsight/rocprof 贴出关键指标；
* **对齐生态语言**：PR 描述里明确 *before/after* 性能、TIR 差异截图或摘要；
* **参考 TVMCon/TritonCon** 议题挑选选题方向（后端支持、张量化、流水线/访存优化等）。([YouTube][27])

需要的话，我可以把上面清单打包成一个**可勾选的周计划表（含视频链接与练习）**，并附上“打印 TIR、用 Nsight 采样、写基准”的最小脚手架。

[1]: https://www.youtube.com/watch?v=Oc_wVXdnrrM&utm_source=chatgpt.com "Machine Learning Compilation: Episode 1 / Overview"
[2]: https://www.youtube.com/watch?v=bqLfplA977o&utm_source=chatgpt.com "TVM Tutorial at FCRC [1/9]: TVM Overview"
[3]: https://tilelang.com/get_started/overview.html?utm_source=chatgpt.com "The Tile Language: A Brief Introduction"
[4]: https://www.youtube.com/playlist?list=PLEAYkSg4uSQ3yc_zf_f1GOxl5CZo0LVBb&utm_source=chatgpt.com "Compilers - Alex Aiken | Stanford"
[5]: https://online.stanford.edu/courses/soe-ycscs1-compilers?utm_source=chatgpt.com "Compilers | Course - Stanford Online"
[6]: https://www.youtube.com/watch?v=7GHXDEIMGIY&utm_source=chatgpt.com "2023 EuroLLVM - Tutorial: A whirlwind tour of the LLVM ..."
[7]: https://www.youtube.com/playlist?list=PLlONLmJCfHTo9WYfsoQvwjsa5ZB6hjOG5&utm_source=chatgpt.com "How to build a compiler with LLVM and MLIR"
[8]: https://www.youtube.com/watch?v=Lvc8qx8ukOI&utm_source=chatgpt.com "Programming Language with LLVM [1/20] Introduction to ..."
[9]: https://www.youtube.com/playlist?list=PLMDSb3PWPnvivPLXHM9SlZLljrO9unIAW&utm_source=chatgpt.com "CMU 15-418＼618 Parallel Computer Architecture and ..."
[10]: https://www.youtube.com/watch?v=J7v_ubArrno&utm_source=chatgpt.com "Stanford CS149 I Parallel Computing I 2023 I Lecture 19 ..."
[11]: https://www.youtube.com/playlist?list=PLKK11Ligqititws0ZOoGk3SW-TZCar4dK&utm_source=chatgpt.com "CUDA Tutorials"
[12]: https://www.youtube.com/nvidiadeveloper?utm_source=chatgpt.com "NVIDIA Developer"
[13]: https://www.youtube.com/playlist?list=PLx15eYqzJifcJ0AKvmygLt-Rw75SiL-Km&utm_source=chatgpt.com "AMD ROCm Training"
[14]: https://www.youtube.com/watch?v=dUDGO66IadU&utm_source=chatgpt.com "Intro to NVIDIA Nsight Systems | CUDA Developer Tools"
[15]: https://www.youtube.com/playlist?list=PL5B692fm6--ukF8S7ul5NmceZhXLRv_lR&utm_source=chatgpt.com "Boost CUDA Development with Nsight Developer Tools"
[16]: https://www.youtube.com/watch?v=AKenglkAziA&utm_source=chatgpt.com "Introduction to ROCm Profiler -AMD Profiling workshop - Day ..."
[17]: https://tvm.apache.org/docs/deep_dive/tensor_ir/index.html?utm_source=chatgpt.com "TensorIR — tvm 0.22.dev0 documentation"
[18]: https://www.youtube.com/playlist?list=PLR4pm7mU3ROmrk9rimv6nRinKb0YKBx0f&utm_source=chatgpt.com "TVM Conference 2018"
[19]: https://www.youtube.com/playlist?list=PLFxzvDFotCitb0dOv5SpNdK6t3Uu7tBRo&utm_source=chatgpt.com "MLC-AI"
[20]: https://www.youtube.com/%40Triton-openai?utm_source=chatgpt.com "Triton"
[21]: https://www.youtube.com/playlist?list=PLSXcJOyFhmS-qb_CF-GLhkWxSmi-ftbPO&utm_source=chatgpt.com "Intro to Triton Series (coding Triton language kernels for ..."
[22]: https://www.youtube.com/watch?v=njgow_zaJMw&utm_source=chatgpt.com "Lecture 29: Triton Internals"
[23]: https://tilelang.com/?utm_source=chatgpt.com "Tile Language 0.1.6.post1 documentation"
[24]: https://github.com/tile-ai/tilelang?utm_source=chatgpt.com "tile-ai/tilelang: Domain-specific language designed ..."
[25]: https://www.youtube.com/watch?v=PWWOGrLZtZg&utm_source=chatgpt.com "CUTLASS: A CUDA C++ Template Library for Accelerating ..."
[26]: https://www.youtube.com/watch?v=SGhfUhlowB4&utm_source=chatgpt.com "Lecture 8: CUDA Performance Checklist"
[27]: https://www.youtube.com/watch?v=pSF3uRhuk-k&utm_source=chatgpt.com "[TVMCon23] UMA Universal Modular Accelerator Interface"
