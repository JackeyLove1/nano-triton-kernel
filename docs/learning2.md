### TileLang 贡献源码的基础知识学习路线

要贡献 TileLang 源码（一个基于 TVM 的 DSL，用于 AI 内核优化），您需要从编程、数学和编译/GPU 基础入手。这些知识帮助您理解瓦片抽象、IR 变换和硬件优化。假设您有基本计算机科学背景，以下是按逻辑顺序的起点（从易到难，预计 4-8 周）。每个部分包括为什么重要、学习时长建议，以及精选 YouTube 视频（基于流行度和完整性）。

#### 1. **Python 编程基础（1-2 周）**
   - **为什么重要**：TileLang 用 Python-like 语法编写内核，依赖 NumPy/PyTorch 等库进行测试和调试。掌握 OOP、函数式编程和调试是起点。
   - **学习重点**：变量、循环、函数、类、模块导入、错误处理。
   - **推荐视频**：
     - **Python Full Course for Beginners [2025]** by Bro Code（4 小时，全程实践导向，适合零基础）。链接：https://www.youtube.com/watch?v=K5KVEU3aaeQ
     - **Python for Beginners (Full Course) | #100DaysOfCode** by CodeWithHarry（播放列表，100 天挑战式，包含项目）。链接：https://www.youtube.com/playlist?list=PLu0W_9lII9agwh1XjRt242xIpHhPT2llg

#### 2. **线性代数基础（1 周）**
   - **为什么重要**：AI 内核（如 GEMM、FlashAttention）涉及矩阵运算、向量空间和 SVD 等；TileLang 的优化依赖这些数学抽象。
   - **学习重点**：向量/矩阵运算、特征值、奇异值分解（SVD），并联想到 ML（如 PCA）。
   - **推荐视频**：
     - **Linear Algebra for Machine Learning** by Jon Krohn（播放列表，48 集，专为 ML 设计，结合代码示例）。链接：https://www.youtube.com/playlist?list=PLRDl2inPrWQW1QSWhBU0ki-jq_uElkh2a
     - **Mathematics for Machine Learning: Linear Algebra** by Imperial College London（免费课程，理论+应用，适合初学者）。链接：https://www.youtube.com/watch?v=tVQZvJwi-ec

#### 3. **编译原理基础（1-2 周）**
   - **为什么重要**：TileLang 扩展 TVM 的 IR（中间表示），涉及词法/语法分析、优化传递（如循环瓦片化）和代码生成。您需理解编译流程以调试 JIT。
   - **学习重点**：词法分析、解析、IR、优化、代码生成；简单实现一个 mini-compiler。
   - **推荐视频**：
     - **Compiler Design / Principles of Compiler Design (FULL COURSE)** by Gate Smashers（播放列表，全面覆盖，从介绍到优化）。链接：https://www.youtube.com/playlist?list=PLENQMW_c1dimxHUu6KjuBC2rOlAaoLozF
     - **Compilers (2020-21) - Introduction** by Imperial College London（系列讲座，学术深度，结合现代工具）。链接：https://www.youtube.com/watch?v=iojJeCGC9nY

#### 4. **GPU 编程基础（CUDA/HIP，1 周）**
   - **为什么重要**：TileLang 针对 GPU 后端（如 NVIDIA/AMD），需懂并行线程、内存层次和内核启动，以优化 TileLang 的 PTX/HIP 输出。
   - **学习重点**：CUDA 模型、线程块、共享内存、简单内核编写（e.g., 向量加法）。
   - **推荐视频**：
     - **CUDA Programming Course – High-Performance Computing** by NVIDIA Developer（官方，实践导向，包含代码）。链接：https://www.youtube.com/watch?v=86FAWCzIe_4
     - **CUDA Tutorials** by CUDA By Example（播放列表，从基础到高级，易上手）。链接：https://www.youtube.com/playlist?list=PLKK11Ligqititws0ZOoGk3SW-TZCar4dK

#### 5. **Apache TVM 基础（1 周）**
   - **为什么重要**：TileLang 是 TVM 的 DSL 扩展，直接用 TVM API 定义 prim_func 和调度；这是贡献的桥梁。
   - **学习重点**：TVM Python API、Relay IR、AutoTVM 调优、简单模型编译。
   - **推荐视频**：
     - **How to use Apache TVM to optimize your ML models** by OctoML（1 小时，端到端示例）。链接：https://www.youtube.com/watch?v=VK6cuGRH230
     - **An Introduction to Apache TVM** by Leandro Nunes（安装+ API 演示，适合入门）。链接：https://www.youtube.com/watch?v=J3KB_QEMIlI

| 基础知识 | 预计时长 | 为什么优先级高 | 练习建议 |
|----------|----------|----------------|----------|
| Python | 1-2 周 | 语法入口 | 写简单脚本测试 NumPy 矩阵 |
| 线性代数 | 1 周 | 数学核心 | 用 Python 实现矩阵乘法 |
| 编译原理 | 1-2 周 | IR 理解 | 构建 toy lexer/parser |
| GPU 编程 | 1 周 | 硬件优化 | 编译 CUDA 内核基准 |
| TVM | 1 周 | TileLang 基础 | 编译 ResNet 到 CUDA |

完成这些后，直接进入我之前提供的 TileLang 学习路线（从环境准备起步）。边学边练：用 Jupyter Notebook 记录笔记，并在 GitHub 上 fork TVM/TileLang 实验。遇到问题，参考官方 Discord 或 Stack Overflow。如果您有特定弱点（如数学），可以优先那个部分！