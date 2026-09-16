from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO

# 脚本放在项目根目录时，输出到项目/input_case。
PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_ROOT / "input_case"
OUTPUT_DIR.mkdir(exist_ok=True)
WEIGHTS = PROJECT_ROOT / "yolov8n-obb.pt"
IMAGE = PROJECT_ROOT / "test.jpg"

IMGSZ = 640
DEVICE = "cuda:0"

# 关闭 TF32，让 PyTorch Golden 更适合与你的 float32 CUDA V0 对比。
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


def save_bin(filename, tensor):
    """保存连续 float32 数据；不写文件头。"""
    array = tensor.detach().float().cpu().contiguous().numpy()
    path = OUTPUT_DIR / filename
    array.tofile(path)

    print(f"{filename:24s} shape={list(array.shape)}, bytes={array.nbytes}")


# ---------- 1. 图片变成模型输入 [1, 3, 640, 640] ----------
bgr = cv2.imread(IMAGE)

if bgr is None:
    raise RuntimeError(f"无法读取图片：{IMAGE}")

bgr = cv2.resize(bgr, (IMGSZ, IMGSZ))
rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

image_np = rgb.transpose(2, 0, 1).astype(np.float32) / 255.0
image_np = np.ascontiguousarray(image_np)

image = torch.from_numpy(image_np).unsqueeze(0).to(DEVICE)


# ---------- 2. 加载模型，选择 P3 最后一个 1×1 DFL Conv ----------
model = YOLO(WEIGHTS).model.to(DEVICE).eval()

head = model.model[-1]

# P3 的 DFL 分支：
# cv2[0] = 两个 3×3 Conv + 最后一个 1×1 Conv
target_conv = head.cv2[0][-1]

print("target conv:", target_conv)

if not isinstance(target_conv, torch.nn.Conv2d):
    raise RuntimeError("head.cv2[0][-1] 不是 Conv2d，请先打印 head.cv2[0] 检查模型结构。")

if target_conv.kernel_size != (1, 1):
    raise RuntimeError("当前选中的层不是 1×1 Conv。")


# ---------- 3. hook 同时截获最后 1×1 Conv 的输入 A 和输出 C ----------
saved = {}


def save_conv_input_and_output(module, inputs, output):
    # inputs[0]：进入最后 1×1 Conv 前的真实输入 A
    # output：最后 1×1 Conv 的真实输出 C
    saved["a"] = inputs[0].detach().clone()
    saved["c"] = output.detach().clone()


hook = target_conv.register_forward_hook(save_conv_input_and_output)

try:
    with torch.inference_mode():
        _ = model(image)
finally:
    hook.remove()


if "a" not in saved or "c" not in saved:
    raise RuntimeError("hook 没有捕获到 A/C，请检查模型是否成功前向推理。")


# ---------- 4. 同一次模型状态下取得 W、bias ----------
a = saved["a"]                   # [1, K, 80, 80]
c_golden = saved["c"]            # [1, 64, 80, 80]

weight = target_conv.weight      # [64, K, 1, 1]
bias = target_conv.bias          # [64]

if bias is None:
    raise RuntimeError("目标 Conv 没有 bias；当前 CUDA kernel 需要相应修改。")


# ---------- 5. 保存给 CUDA 使用和对比的四个文件 ----------
save_bin("p3_a_dfl_raw.bin", a)
save_bin("p3_w_dfl_raw.bin", weight)
save_bin("p3_bias_dfl_raw.bin", bias)
save_bin("p3_dfl_raw.bin", c_golden)

print("\n导出完成。")
print("CUDA 输入：A、W、bias")
print("Golden 输出：p3_dfl_raw.bin")