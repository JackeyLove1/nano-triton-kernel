import triton
import triton.language as tl

# input:  1D float32 device tensor，长度 H
# output: 1D float32 device tensor，长度按公式 H_out = floor((H + 2P - k)/S + 1)
def solution(input, kernel_size: int, stride: int, padding: int, output, H: int):
    # 计算输出长度
    H_out = (H + 2 * padding - kernel_size) // stride + 1
    if H_out <= 0:
        raise ValueError("H_out <= 0：请检查 H、kernel_size、stride、padding 的取值是否合理。")

    # Triton 中为了向量化窗口求和，需要给一个“最大窗口尺寸”的编译期常量
    K_MAX = 256
    assert kernel_size <= K_MAX, f"kernel_size={kernel_size} 超过 K_MAX={K_MAX}"

    @triton.jit
    def avg_pool1d_kernel(x_ptr, out_ptr,
                          H, k, s, p, H_out,
                          BLOCK: tl.constexpr, K_MAX: tl.constexpr):
        # 每个 program 负责 BLOCK 个输出位置
        pid = tl.program_id(0)
        i = pid * BLOCK + tl.arange(0, BLOCK)                 # [BLOCK]
        o_mask = i < H_out

        # 该输出位置对应的输入起点 = i * stride - padding
        start = i * s - p                                      # [BLOCK]

        # 第二维度是窗口内的 m=0..K_MAX-1
        m = tl.arange(0, K_MAX)                                # [K_MAX]
        idx = start[:, None] + m[None, :]                      # [BLOCK, K_MAX]

        # 合法取数：窗口未越界 && m<k && i 在有效输出范围
        k_mask = (m[None, :] < k)
        in_mask = (idx >= 0) & (idx < H)
        load_mask = k_mask & in_mask & o_mask[:, None]

        vals = tl.load(x_ptr + idx, mask=load_mask, other=0.0) # [BLOCK, K_MAX]
        acc = tl.sum(vals, axis=1)                             # [BLOCK]
        avg = acc / tl.maximum(k, 1)                           # 防御性除法

        tl.store(out_ptr + i, avg, mask=o_mask)

    # 选择每个程序块处理多少个输出元素
    BLOCK = 256
    grid = lambda meta: ((H_out + BLOCK - 1) // BLOCK,)

    avg_pool1d_kernel[grid](input, output,
                            H, kernel_size, stride, padding, H_out,
                            BLOCK=BLOCK, K_MAX=K_MAX)
