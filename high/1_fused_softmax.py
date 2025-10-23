import torch
import triton
import triton.language as tl
from triton.runtime import driver


def naive_softmax(x):
    """Compute row-wise softmax of X using native pytorch
    使用原生 PyTorch 计算 X 的逐行 softmax


    We subtract the maximum element in order to avoid overflows. Softmax is invariant to
    this shift.
    我们减去最大元素以避免溢出。Softmax 对于这种偏移是不变的。
    """
    # read  MN elements ; write M  elements
    # 读取 MN 个元素；写入 M 个元素
    x_max = x.max(dim=1)[0]
    # read MN + M elements ; write MN elements
    # 读取 MN + M 个元素；写入 MN 个元素
    z = x - x_max[:, None]
    # read  MN elements ; write MN elements
    # 读取 MN 个元素；写入 MN 个元素
    numerator = torch.exp(z)
    # read  MN elements ; write M  elements
    # 读取 MN 个元素；写入 M 个元素
    denominator = numerator.sum(dim=1)
    # read MN + M elements ; write MN elements
    # 读取 MN + M 个元素；写入 MN 个元素
    ret = numerator / denominator[:, None]
    # in total: read 5MN + 2M elements ; wrote 3MN + 2M elements
    # 总计：读取 5MN + 2M 个元素；写入 3MN + 2M 个元素
    return ret


@triton.jit
def softmax_kernel(output_ptr, input_ptr, input_row_stride, output_row_stride, n_rows, n_cols, BLOCK_SIZE: tl.constexpr,
                   num_stages: tl.constexpr):
    # starting row of the program
    # 程序起始行
    row_start = tl.program_id(0)
    row_step = tl.num_programs(0)
    for row_idx in tl.range(row_start, n_rows, row_step, num_stages=num_stages):
        # The stride represents how much we need to increase the pointer to advance 1 row
        # 步长表示我们需要对指针增加多少以推进 1 行
        row_start_ptr = input_ptr + row_idx * input_row_stride
        # The block size is the next power of two greater than n_cols, so we can fit each
        # 块大小是大于 n_cols 的下一个二的幂，因此我们可以适配
        # row in a single block
        # 单个块中的行
        col_offsets = tl.arange(0, BLOCK_SIZE)
        input_ptrs = row_start_ptr + col_offsets
        # Load the row into SRAM, using a mask since BLOCK_SIZE may be > than n_cols
        # 将行加载到 SRAM 中，使用掩码，因为 BLOCK_SIZE 可能大于 n_cols
        mask = col_offsets < n_cols
        row = tl.load(input_ptrs, mask=mask, other=-float('inf'))
        # Subtract maximum for numerical stability
        # 为了数值稳定性而减去最大值
        row_minus_max = row - tl.max(row, axis=0)
        # Note that exponentiation in Triton is fast but approximate (i.e., think __expf in CUDA)
        # 请注意，Triton 中的指数运算速度很快，但是是近似的（例如，类似于 CUDA 中的 __expf）。
        numerator = tl.exp(row_minus_max)
        denominator = tl.sum(numerator, axis=0)
        softmax_output = numerator / denominator
        # Write back output to DRAM
        # 将输出写回 DRAM
        output_row_start_ptr = output_ptr + row_idx * output_row_stride
        output_ptrs = output_row_start_ptr + col_offsets
        tl.store(output_ptrs, softmax_output, mask=mask)
        
def run():
    torch.manual_seed(0)
    x = torch.randn(1823, 781).float().cuda()

def get_device():
    device = torch.cuda.current_device()
    properties = driver.active.utils.get_device_properties(device)
    print(f"device: {device}")
    print(f"properties: {properties}")


if __name__ == "__main__":
    get_device()