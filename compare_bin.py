from pathlib import Path
import numpy as np


# ======================= 配置 =======================
PROJECT_ROOT = Path("/home/xhy/projects/yolo_obb_dfl_tensorcore")

BATCH = 1
M = 64          # DFL 输出通道
K = 64          # 最后 1×1 Conv 的输入通道
HEIGHT = 80
WIDTH = 80

ATOL = 1e-4
RTOL = 1e-5
# ====================================================


INPUT_DIR = PROJECT_ROOT / "input_case"
OUTPUT_DIR = PROJECT_ROOT / "output_case"

# 合并导出脚本生成的四个文件
A_PATH = INPUT_DIR / "p3_a_dfl_raw.bin"
WEIGHT_PATH = INPUT_DIR / "p3_w_dfl_raw.bin"
BIAS_PATH = INPUT_DIR / "p3_bias_dfl_raw.bin"
GOLDEN_PATH = INPUT_DIR / "p3_dfl_raw.bin"

# 自写 CUDA kernel 的输出
CUDA_PATH = OUTPUT_DIR / "output_dfl_p3.bin"


def load_bin(path, expected_count):
    """读取连续 float32 二进制文件，并验证元素数。"""
    if not path.exists():
        raise FileNotFoundError(f"找不到文件：{path}")

    data = np.fromfile(path, dtype=np.float32)

    if data.size != expected_count:
        raise RuntimeError(
            f"文件大小不匹配：{path}\n"
            f"期望元素数：{expected_count}\n"
            f"实际元素数：{data.size}\n"
            f"期望字节数：{expected_count * 4}\n"
            f"实际字节数：{data.size * 4}"
        )

    return data


def main():
    spatial = HEIGHT * WIDTH

    # A: [B, K, H, W]
    a = load_bin(
        A_PATH,
        BATCH * K * spatial
    ).reshape(BATCH, K, HEIGHT, WIDTH)

    # 原始 PyTorch weight 是 [M, K, 1, 1]；
    # 对 1×1 Conv 而言，展平后可直接看成 [M, K]。
    weight = load_bin(
        WEIGHT_PATH,
        M * K
    ).reshape(M, K)

    bias = load_bin(
        BIAS_PATH,
        M
    )

    # Golden C 和 CUDA C 都是 [B, M, H, W]
    golden = load_bin(
        GOLDEN_PATH,
        BATCH * M * spatial
    ).reshape(BATCH, M, HEIGHT, WIDTH)

    cuda_out = load_bin(
        CUDA_PATH,
        BATCH * M * spatial
    ).reshape(BATCH, M, HEIGHT, WIDTH)

    # ---------- 全量比较 ----------
    abs_diff = np.abs(cuda_out - golden)
    close = np.isclose(cuda_out, golden, atol=ATOL, rtol=RTOL)

    max_abs = float(abs_diff.max())
    mean_abs = float(abs_diff.mean())
    failed_count = int((~close).sum())

    worst_flat = int(np.argmax(abs_diff))
    b, m, h, w = np.unravel_index(worst_flat, abs_diff.shape)

    # Golden 很小时相对误差没有参考价值，仅打印观察。
    relative = abs_diff / np.maximum(np.abs(golden), 1e-8)
    max_relative = float(relative.max())

    print("========== Compare Result ==========")
    print(f"CUDA file         : {CUDA_PATH}")
    print(f"Golden file       : {GOLDEN_PATH}")
    print(f"shape             : {cuda_out.shape}")
    print(f"total elements    : {cuda_out.size}")
    print(f"exactly equal     : {np.array_equal(cuda_out, golden)}")
    print(f"max abs error     : {max_abs:.8e}")
    print(f"mean abs error    : {mean_abs:.8e}")
    print(f"max relative error: {max_relative:.8e}")
    print(f"failed elements   : {failed_count}")
    print(f"tolerance         : atol={ATOL}, rtol={RTOL}")

    print("\n========== Worst Element ==========")
    print(f"index [B,C,H,W]   : [{b}, {m}, {h}, {w}]")
    print(f"CUDA value        : {cuda_out[b, m, h, w]:.8f}")
    print(f"Golden value      : {golden[b, m, h, w]:.8f}")
    print(f"abs error         : {abs_diff[b, m, h, w]:.8e}")

    # ---------- 用 A/W/bias 在 CPU 手算最差元素 ----------
    # C[b,m,h,w] = bias[m] + sum_k(W[m,k] * A[b,k,h,w])
    manual = np.float32(bias[m])

    for k_index in range(K):
        product = np.float32(
            weight[m, k_index] * a[b, k_index, h, w]
        )
        manual = np.float32(manual + product)

    n = h * WIDTH + w

    print("\n========== Manual Check ==========")
    print(f"matrix C[m,n]     : C[{m}, {n}]")
    print(f"CPU manual        : {manual:.8f}")
    print(f"CUDA              : {cuda_out[b, m, h, w]:.8f}")
    print(f"Golden            : {golden[b, m, h, w]:.8f}")
    print(
        f"|manual - CUDA|   : "
        f"{abs(manual - cuda_out[b, m, h, w]):.8e}"
    )
    print(
        f"|manual - Golden| : "
        f"{abs(manual - golden[b, m, h, w]):.8e}"
    )

    if failed_count == 0:
        print("\nPASS: CUDA 输出与模型 Golden 一致。")
    else:
        print("\nFAIL: 存在超出容差的元素。")


if __name__ == "__main__":
    main()