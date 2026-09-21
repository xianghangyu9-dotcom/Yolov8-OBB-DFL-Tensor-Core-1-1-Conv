from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


# ==================== 配置 ====================
PROJECT_ROOT = Path("/home/xhy/projects/yolo_obb_dfl_tensorcore")

BATCH = 1
M = 64
K = 64
HEIGHT = 80
WIDTH = 80

DEVICE = "cuda:0"
# ==============================================


FP32_DIR = PROJECT_ROOT / "input_case"
V2_DIR = PROJECT_ROOT / "input_case_v3"
V2_DIR.mkdir(exist_ok=True)

A_FP32_PATH = FP32_DIR / "p3_a_dfl_raw.bin"
W_FP32_PATH = FP32_DIR / "p3_w_dfl_raw.bin"
BIAS_FP32_PATH = FP32_DIR / "p3_bias_dfl_raw.bin"


def load_bin(path, dtype, expected_count):
    if not path.exists():
        raise FileNotFoundError(f"文件不存在：{path}")

    data = np.fromfile(path, dtype=dtype)

    if data.size != expected_count:
        raise RuntimeError(
            f"文件大小不匹配：{path}\n"
            f"期望元素数：{expected_count}\n"
            f"实际元素数：{data.size}"
        )

    return data


def save_bin(path, array):
    array = np.ascontiguousarray(array)
    array.tofile(path)

    print(
        f"{path.name:32s} "
        f"shape={list(array.shape)}, "
        f"dtype={array.dtype}, "
        f"bytes={array.nbytes}"
    )


def main():
    spatial = HEIGHT * WIDTH

    # ---------- 1. 读取已验证的 FP32 A/W/bias ----------
    a_fp32_np = load_bin(
        A_FP32_PATH,
        np.float32,
        BATCH * K * spatial
    ).reshape(BATCH, K, HEIGHT, WIDTH)

    w_fp32_np = load_bin(
        W_FP32_PATH,
        np.float32,
        M * K
    ).reshape(M, K, 1, 1)

    bias_fp32_np = load_bin(
        BIAS_FP32_PATH,
        np.float32,
        M
    )

    # ---------- 2. A/W 量化为 FP16 ----------
    # 这两个 FP16 文件将作为 V3 WMMA kernel 的真实输入。
    a_fp16 = torch.from_numpy(a_fp32_np).to(torch.float16)
    w_fp16 = torch.from_numpy(w_fp32_np).to(torch.float16)

    a_fp16_np = a_fp16.numpy()
    w_fp16_np = w_fp16.numpy()

    save_bin(V2_DIR / "p3_a_fp16.bin", a_fp16_np)
    save_bin(V2_DIR / "p3_w_fp16.bin", w_fp16_np)
    save_bin(V2_DIR / "p3_bias_fp32.bin", bias_fp32_np)

    # ---------- 3. 生成 V3 专用 Golden ----------
    # 禁止 TF32，避免它混入参考计算。
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")

    # 重点：
    # A/W 先转成 FP16，再转回 FP32 参与卷积。
    # 所以输入量化与 WMMA 一致，但累加器和输出保持 FP32。
    a_reference = a_fp16.float().to(DEVICE)
    w_reference = w_fp16.float().to(DEVICE)
    bias_reference = torch.from_numpy(bias_fp32_np).to(DEVICE)

    with torch.inference_mode():
        golden = F.conv2d(
            a_reference,
            w_reference,
            bias_reference,
            stride=1,
            padding=0
        )

    golden_np = golden.float().cpu().contiguous().numpy()

    save_bin(
        V2_DIR / "p3_golden_fp16acc_fp32.bin",
        golden_np
    )

    # ---------- 4. 打印 FP16 量化误差 ----------
    a_quant_error = np.max(
        np.abs(a_fp16.float().numpy() - a_fp32_np)
    )

    w_quant_error = np.max(
        np.abs(w_fp16.float().numpy() - w_fp32_np)
    )

    print("\n========== FP16 Quantization ==========")
    print(f"A max quantization error: {a_quant_error:.8e}")
    print(f"W max quantization error: {w_quant_error:.8e}")

    print("\nV3 数据生成完成。")
    print(f"输出目录：{V2_DIR}")


if __name__ == "__main__":
    main()