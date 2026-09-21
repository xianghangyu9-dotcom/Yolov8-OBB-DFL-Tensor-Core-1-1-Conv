from pathlib import Path
import numpy as np


# ======================= 配置 =======================
PROJECT_ROOT = Path("/home/xhy/projects/yolo_obb_dfl_tensorcore")

BATCH = 1
M = 64
K = 64
HEIGHT = 80
WIDTH = 80

# WMMA 的浮点累加顺序可能与 PyTorch 不同。
# 先用这个容差；若正确性通过，再记录实际 max abs error。
ATOL = 1e-3
RTOL = 1e-3
# ====================================================


INPUT_DIR = PROJECT_ROOT / "input_case_v3"
OUTPUT_DIR = PROJECT_ROOT / "output_case_v3"

# V3 / WMMA 的输入：FP16
A_PATH = INPUT_DIR / "p3_a_fp16.bin"
WEIGHT_PATH = INPUT_DIR / "p3_w_fp16.bin"

# bias、Golden、CUDA 输出：FP32
BIAS_PATH = INPUT_DIR / "p3_bias_fp32.bin"
GOLDEN_PATH = INPUT_DIR / "p3_golden_fp16acc_fp32.bin"
CUDA_PATH = OUTPUT_DIR / "output_dfl_p3.bin"


def load_bin(path: Path, dtype, expected_count: int) -> np.ndarray:
    """读取二进制张量，并按 dtype 检查文件大小。"""
    if not path.exists():
        raise FileNotFoundError(f"找不到文件：{path}")

    dtype = np.dtype(dtype)
    data = np.fromfile(path, dtype=dtype)

    if data.size != expected_count:
        raise RuntimeError(
            f"文件元素数量不匹配：{path}\n"
            f"dtype：{dtype}\n"
            f"期望元素数：{expected_count}\n"
            f"实际元素数：{data.size}\n"
            f"期望字节数：{expected_count * dtype.itemsize}\n"
            f"实际字节数：{data.size * dtype.itemsize}"
        )

    return data


def main():
    spatial = HEIGHT * WIDTH

    # A: [B, K, H, W]，FP16
    a_fp16 = load_bin(
        A_PATH,
        np.float16,
        BATCH * K * spatial
    ).reshape(BATCH, K, HEIGHT, WIDTH)

    # W: [M, K, 1, 1]，FP16
    # 对 1×1 Conv 展平看成 [M, K]。
    weight_fp16 = load_bin(
        WEIGHT_PATH,
        np.float16,
        M * K
    ).reshape(M, K)

    # bias: [M]，FP32
    bias_fp32 = load_bin(
        BIAS_PATH,
        np.float32,
        M
    )

    # Golden 与 CUDA 输出：均为 [B, M, H, W] FP32
    golden = load_bin(
        GOLDEN_PATH,
        np.float32,
        BATCH * M * spatial
    ).reshape(BATCH, M, HEIGHT, WIDTH)

    cuda_out = load_bin(
        CUDA_PATH,
        np.float32,
        BATCH * M * spatial
    ).reshape(BATCH, M, HEIGHT, WIDTH)

    # ---------- 全量比较 ----------
    abs_diff = np.abs(cuda_out - golden)
    close_mask = np.isclose(
        cuda_out,
        golden,
        atol=ATOL,
        rtol=RTOL
    )

    max_abs = float(abs_diff.max())
    mean_abs = float(abs_diff.mean())
    failed_count = int((~close_mask).sum())

    worst_flat = int(np.argmax(abs_diff))
    b, m, h, w = np.unravel_index(worst_flat, abs_diff.shape)

    relative_diff = abs_diff / np.maximum(np.abs(golden), 1e-8)
    max_relative = float(relative_diff.max())

    print("========== V3 WMMA Compare Result ==========")
    print(f"A file            : {A_PATH}")
    print(f"W file            : {WEIGHT_PATH}")
    print(f"Golden file       : {GOLDEN_PATH}")
    print(f"CUDA file         : {CUDA_PATH}")
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

    # ---------- CPU 手算最差元素 ----------
    # 注意：
    # 输入 A/W 本身是 FP16；
    # WMMA 的目标是 FP16 输入、FP32 累加。
    manual = np.float32(bias_fp32[m])

    for k_index in range(K):
        a_value = np.float32(a_fp16[b, k_index, h, w])
        w_value = np.float32(weight_fp16[m, k_index])

        manual = np.float32(
            manual + np.float32(a_value * w_value)
        )

    n = h * WIDTH + w

    print("\n========== FP16 Input Manual Check ==========")
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
        print("\nPASS: WMMA CUDA 输出与 FP16 专用 Golden 一致。")
    else:
        print("\nFAIL: 存在超出容差的元素。")


if __name__ == "__main__":
    main()