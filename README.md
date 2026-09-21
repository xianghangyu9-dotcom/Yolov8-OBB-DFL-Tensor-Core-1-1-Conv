# 性能测试方法

- GPU：NVIDIA GeForce RTX 5060
- CUDA：13.3
- 编译模式：Release，`CMAKE_CUDA_ARCHITECTURES=120`
- 算子：YOLOv8-OBB P3 DFL 分支最后一层 `1×1 Conv`
- 数学形式：`C[M, N] = W[M, K] × A[K, N] + bias[M]`
- 固定 shape：`M=64`，`K=64`，`N=80×80=6400`
- 测试方式：CUDA Event，100 次 warmup，1000 次迭代
- 计时范围：仅 GPU kernel/库调用，不包含文件读取、Host-to-Device、Device-to-Host 拷贝和 cuBLAS handle 创建

# 手写 kernel 版本说明

| 版本 | 精度 | 核心优化 |
|---|---|---|
| V0 | FP32 | Global Memory 读取 + 寄存器 `4×4` 输出微块 |
| V1 | FP32 | Shared Memory tiling，复用 W/A tile |
| V2 | FP32 | Shared Memory 双缓冲 + `float4` 向量化 Global-to-Shared 读取 |
| V3 | FP16 输入/权重，FP32 累加 | WMMA Tensor Core；`64×128×32` CTA tile；`2×4` warp grid；每 warp 计算 `2×2` 个 WMMA fragment；`float4` 搬运、Shared Memory bias epilogue |

# 手写 kernel 延迟结果

| 版本 | 第 1 次 | 第 2 次 | 第 3 次 | 平均延迟 | 相对 V0 延迟比 |
|---|---:|---:|---:|---:|---:|
| V0 | 66.0789 us | 65.8671 us | 65.8043 us | **65.9168 us** | 1.00× |
| V1 | 18.4550 us | 18.9105 us | 18.5537 us | **18.6397 us** | **3.54×** |
| V2 | 11.3472 us | 11.1521 us | 10.9927 us | **11.1640 us** | **5.90×** |
| V3 | 10.8079 us | 10.6403 us | 10.4060 us | **10.6181 us** | **6.21×** |

> 注：V0~V2 使用 FP32 输入和 FP32 Golden；V3 使用 FP16 A/W 输入、FP32 accumulate 与对应 FP16 Golden。因此 V3 相对 V0 的延迟比仅表示端到端算子延迟变化，不应解释为完全相同数值精度下的纯优化收益。

# cuBLAS 参考比较

cuBLAS 使用与 V3 相同的 FP16 A/W、FP32 输出和 FP32 accumulate 配置。

| 实现 | 第 1 次 | 第 2 次 | 第 3 次 | 平均延迟 | 相对 V3 延迟比 |
|---|---:|---:|---:|---:|---:|
| V3 fused WMMA | 10.8079 us | 10.6403 us | 10.4060 us | **10.6181 us** | **1.00×** |
| cuBLAS GEMM-only | 12.7657 us | 12.8273 us | 13.1128 us | **12.9019 us** | 1.22× |
| cuBLAS GEMM + 独立 bias kernel | 20.0748 us | 20.6858 us | 19.9522 us | **20.2376 us** | 1.91× |

# 性能分析

- V1 相比 V0 延迟降低约 **71.7%**。Shared Memory 降低了 W/A tile 的重复 Global Memory 读取。
- V2 相比 V1 延迟降低约 **40.1%**。收益主要来自 `float4` 向量化读取和 Shared Memory 双缓冲。
- V3 相比 V2 延迟降低约 **4.9%**。在 `K=64` 的小 K 场景中，Tensor Core 流水深度有限，WMMA fragment、Shared Memory 输出暂存和 bias epilogue 也会引入额外开销。
- V3 比 cuBLAS GEMM-only 快约 **17.7%**。该结果仅针对当前固定 shape，反映了固定 tile、无边界处理和定制访存路径的价值。
- cuBLAS + 独立 bias kernel 比 cuBLAS GEMM-only 多约 **7.34 us**。额外的 kernel launch 以及 C 的 Global Memory 读写开销明显。
- V3 在同一 kernel 内完成 GEMM、bias 融合与最终写回，因此比 cuBLAS + 独立 bias kernel 快约 **47.5%**。

# 正确性验证

- V0、V1、V2：与模型同次推理导出的 FP32 Golden 输出比较。
- V3：使用 FP16 A/W 输入，与“FP16 输入、FP32 accumulate”的 Golden 输出比较。
- cuBLAS + bias：与 V3 使用同一份 FP16 Golden 输出比较。
- 所有纳入性能表的版本均通过输出比较脚本的容差检查。

> 该项目是基于真实 YOLOv8-OBB DFL 分支固定 shape 的 CUDA 算子优化原型，用于研究访存优化、warp tiling、Shared Memory 双缓冲、WMMA Tensor Core 与算子融合；不试图替代 TensorRT、cuDNN、CUTLASS 或 cuBLASLt 的通用高性能实现。