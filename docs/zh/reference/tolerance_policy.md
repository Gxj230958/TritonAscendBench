# 容差策略

> English: [../../reference/tolerance_policy.md](../../reference/tolerance_policy.md)

> 权威打分模型：`docs/DESIGN.md §4`。GPU-Triton → 910B NPU-Triton 迁移；无 AscendC 基线。此处的容差类仅对**正确性**进行门控；S2 等价性使用严格的逐位精确(bit-exact)，而非这些容差类。


> 机器可读形式：`ascendbench/policies/tolerance_policy.yaml`（由 `eval.py` 加载）

每个任务的 `meta.yaml.tolerance_class` 引用下列容差类之一。对于浮点输出，runner 使用 `torch.allclose(out_c, out_o, atol=cls.atol, rtol=cls.rtol)`；整数、fp8 和逐位精确(bit-exact)类会路由到 dtype 专属的比较器——完整的分派逻辑以及 C2/S2/上限(ceiling)标准请参见 [`correctness_model.md`](correctness_model.md)。下表仅为示意；`ascendbench/policies/tolerance_policy.yaml` 才是完整的权威容差类列表（它还定义了诸如 `elementwise_exact_or_inf_safe` 等类）。

## 容差类

| Class | dtype | atol | rtol | Notes |
|-------|-------|------|------|-------|
| `elementwise` | FP16/BF16 | 1e-3 | 1e-3 | 逐元素算子的默认类（silu、gelu、bias） |
| `vector_reduce` | FP16/BF16 | 1e-3 | 1e-3 | RMSNorm / LayerNorm / softmax（行内 reduce） |
| `cube_matmul_fp16` | FP16/BF16 → FP32 acc | 5e-3 | 1e-3 | 带 FP32 累加器的 GEMM |
| `int8_w8a8` | INT8 → FP16 dequant | 5e-2 | 1e-2 | W8A8 INT8 矩阵乘，预期的量化误差 |
| `int4_w4a8` | INT4 → INT8 dequant | 5e-2 | 1e-2 | W4A8 分组矩阵乘 |
| `fp8_software_emulated` | FP8 ↔ FP16 cast | 1e-1 | 5e-2 | 910B 无 FP8 cube → 在 Roofline 中通过 vector compute + HBM 建模（无 FP8-cube π）；不影响打分资格（打分资格完全由 `eligibility.py` 定义） |
| `recurrent_chunk` | FP16/BF16 | 5e-3 | 5e-3 | Mamba / GDN / KDA chunk 流水线 |
| `sampling_rng` | int / float | 0 | 0 | 当 seed 匹配时要求逐位精确(bit-exact) |
| `atomic_scatter` | FP32 | 1e-4 | 1e-4 | 非确定性；重跑 3 次以验证一致性 |
| `cumsum_scan` | FP16/BF16/INT | 1e-3 | 1e-3 | 前缀扫描累积 |
| `attention_flash` | FP16/BF16 | 5e-3 | 1e-3 | 在线 softmax + 矩阵乘融合 |
| `mrope` | FP16/BF16 | 1e-3 | 1e-3 | mRoPE 位置嵌入 |

## 特殊规则

- **NaN/Inf 一致性**（子项 ⑤）：
  `out_c` 中 NaN/Inf 的位置必须与 `out_o` 完全一致。如果 `out_o[i]` 为 NaN，则 `out_c[i]` 也必须为 NaN。

- **逐 dtype 哨兵值**：
  - 浮点 dtypes：`NaN`
  - int8：`INT8_MIN`（-128）
  - int16/32：对应的 MIN
  - bool：交替掩码 `0xAA`

- **容差升级**（跨 oracle 漂移）：
  如果 PyTorch oracle 参考实现与冻结的 GPU-Triton kernel 之间的差异在标准容差下未通过，该任务**不会**自动升级。负责人(owner)必须排查并决定采取以下哪种方式：
  1. 收紧/放宽容差（并在 `contract.md` 中记录有据可依的理由）
  2. 修复 oracle / wrapper
  3. 将任务标记为 `BLOCKED`——这是一个硬性发布阻塞；它会保留在 `|S|` 中，且**不会**从分母中移除（不允许子集逃逸，`docs/DESIGN.md §4.4`）

## 权威来源

- 唯一可信来源：`ascendbench/policies/tolerance_policy.yaml`（本文件的机器可读对应物）。
- 每个任务的 `meta.yaml.tolerance_class` 必须是 `ascendbench/policies/tolerance_policy.yaml` 中某个键（完整权威）。新增容差类需要：
  1. 在 PR 描述中给出理由
  2. 至少一位 Phase 2 owner 的代码评审
  3. 同时更新本 `.md` 与该 `.yaml`
