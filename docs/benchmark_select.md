# GPU Triton → 华为 910B NPU Triton-Ascend 迁移 Benchmark：100 题精选（v3.1 — scored 100）

> **v3.1 errata（2026-05-14）**：v3 原 L3 共 22 题（其中 L3-07 SDPA 私有进 appendix 非 scored）→ scored = 99。
> 用户要求严格 scored 100，补 **L3-23 `merge_16x16_to_32x32_inverse_kernel`**（vLLM+SGLang, CV, FLA chunk 维度变换）作公开可验证 L3 题。
> **新分布**：L1 40 + L2 38 + L3 23（22 scored + 1 appendix L3-07）= **scored 100 + appendix 1 = 共 101**。
> L3-07 SDPA 保留为 appendix（不进主榜分母，与 codex round 4 R10 决策一致）。



> 选题日期：2026-05-13（v3 修订 — 对齐 KernelBench 风格，补充 protocol / task 规范 / 路线图）
> 候选池：`/data1/code/agentic-rl/kernel/docs/npu/npu triton-ascend.csv`（294 个算子）
> 源码对照：`/data1/code/agentic-rl/kernel/infer/`（vllm / sglang / vllm-ascend / sgl-kernel-npu）
> KernelBench 对照：`/data1/code/agentic-rl/kernel/KernelBench/`（L1 100 / L2 100 / L3 50 / L4 20，共 270 题）
> 选题方法：Claude 与 codex 并行独立选题 → 合并 → 基于 910B 硬件能力与 triton-ascend 当前进展深度优化 → 对照 KernelBench 补 protocol/规范/路线图
> 目标平台：**华为昇腾 Atlas 800I A2 / 910B（即 Atlas A2 系列）**，搭配 triton-ascend 3.2.0 (x86) / 3.5.0 (ARM) + CANN 8.5.0+

## 0. 本 Benchmark 的定位 — Migration vs Generation（v3 新增）

KernelBench 是 **Generation Benchmark**：给 PyTorch `class Model(nn.Module)`，让 LLM/工程师**从零生成** CUDA/Triton kernel；用 `fast_p` 衡量"正确 + 速度 ≥ p × PyTorch eager"的比例。

本 benchmark 是 **Migration Benchmark**：给定 vLLM/SGLang 已有的 **GPU Triton kernel**（已被生产验证），把它**移植**到 910B 的 `triton-ascend`，衡量"在 NPU 上能跑、跑对、跑得快"。两个 benchmark 的目标域不同：

| 维度 | KernelBench | 本 benchmark |
|---|---|---|
| 起点 | PyTorch reference (`class Model`) | 已有 GPU Triton kernel (`@triton.jit`) |
| 终点 | CUDA / Triton / DSL kernel | NPU triton-ascend kernel |
| 测什么能力 | **生成**正确高效 kernel 的能力 | **移植**已有 kernel 到新硬件的可行性与性能 |
| 算子来源 | 通用 (CV conv + activation + norm + 标准 attention) | vLLM/SGLang 真实推理路径 (MLA / Fused MoE / Mamba chunk / paged FA / 量化 GEMM) |
| 难度梯度 | 单算子 → fusion → 完整子模块 → HF 模型 | 纯 VV → CV 混合 → 完整 fused 子系统 |
| Baseline | PyTorch eager wall-clock | GPU Triton 原 kernel + AscendC native（若存在） |
| 单一指标 | `fast_p`（正确 + ≥ p× speedup vs PyTorch） | `migration_score`（正确率 × 相对 GPU triton 的速度保留率，定义见 §10） |

**两个 benchmark 互补**：用 KernelBench 测 "910B 上 triton-ascend 从 PyTorch 直接生成多好的 kernel"；用本 benchmark 测 "910B 上 vLLM/SGLang 真实推理路径能否跑通且性能合格"。

## 1. 平台对齐与关键事实

### 1.1 Atlas 系列与 910B 的关系（[confirmed]）

| Atlas 平台 | 芯片代号 | 对应 CSV 标签 | 状态 |
|---|---|---|---|
| **Atlas A2 系列**（800I A2 / 800T A2 / 300I A2） | **910B / 910B4** | CSV 未直接标记 | **本 benchmark 目标平台** |
| Atlas A3 系列（800I A3 / 900 A3 SuperPoD） | 910C / 910D | "A3 独有" / "A3/A5 共有" | 下一代 |
| Atlas A5 / Ascend 950 系列 | 950PR / 950DT | "A5 独有" / "A3/A5 共有" | 2026 Q1 起，FP8/FP4 cube 算术 |

**关键认知**：CSV 是从 **A3/A5 视角**编排的算子规划清单（170 个 A3/A5 共有 + 1 个 A3 独有 + 74 个 A5 独有），**910B（A2）不在主标签里**。绝大多数 "A3/A5 共有" 算子在 910B 上能跑（同一 ISA 家族向下兼容），但 "A5 独有" 中依赖 **FP8 cube 算术** 或 **新指令集** 的算子在 910B 上跑不了。

### 1.2 910B 硬件 dtype 支持矩阵（[confirmed]）

| dtype | Cube 算术 | Vector 算术 | 存储/搬运 | 在 910B 上推荐路径 |
|---|---|---|---|---|
| FP16 | ✅ 原生 | ✅ 原生 | ✅ | 主路径 |
| BF16 | ✅ 原生 | ✅ 原生 | ✅ | 主路径 |
| FP32 | ❌（Cube fp32 累加 OK） | ✅ 原生 | ✅ | 累加器 |
| INT8 | ✅ 原生 | ✅ 原生 | ✅ | W8A8 量化主路径 |
| **FP8** (E4M3/E5M2) | ❌ 无算术 | ❌ 无算术 | ✅ 仅存储 + `tt.fp_to_fp` 软件 cast | **不进入 benchmark 性能比较**，仅做正确性 + 软件模拟 stress |
| FP4 / NF4 / MXFP4 | ❌（要等 A5 / 950） | ❌ | ✅ 仅搬运 | 不在 benchmark |
| W4A8 / W4A4_LAOS_DYNAMIC | ✅ 通过 dequant-in-GEMM 转 INT8 cube | — | ✅ | **910B 重点量化路径**（vLLM-Ascend 已支持） |
| W8A8 / W8A8C8 / W8A8SC | ✅ INT8 cube | — | ✅ | **最成熟** |
| W8A16 | ✅ FP16/BF16 cube | — | ✅ | 主流量化 |
| INT8 KV cache (C8) | — | ✅ | ✅ | 长上下文优化 |

**结论**：本 v2 版本 **删除所有 FP8 cube 算术依赖型算子**（性能无法在 910B 真实测量），FP8 仅保留 1 题作为 `tt.fp_to_fp` 软件 cast + fused vector 算术的 stress test 代表。

### 1.3 910B 上 triton-ascend 关键约束（[confirmed]）

| 约束 | 数值 / 状态 | 对 kernel 设计的影响 |
|---|---|---|
| **UB (Unified Buffer)** | **192 KB / AI core**（Atlas A2 系列） | 单 kernel tile 总大小（输入 + 输出 + 中间）≤ ~180 KB；MLA / Flash Attention 的 BLOCK_M × BLOCK_K × dtype_size 必须严格控制 |
| Triton Python API 覆盖 | ~85%（2025-06 起持续提升） | 部分高级 API（async copy / TMA / wgmma）不支持，需重写 |
| `tl.atomic_add` | BF16 / FP16 dtype 当前不支持；INT/FP32 OK；Atomic 类 API 2025-08 起在改进 | MoE token combine / scatter_add 类算子可能要降级为多 stage VV 实现 |
| `tl.dot` (cube matmul) | 二维或三维（batch BMM），内积维度对齐 16；BF16 矩阵乘有限制 | 小矩阵或不规则 K 维必须 padding；group GEMM 要 padding 到 16 倍数 |
| `tl.scan` / `tl.sort` | 2025-09 起支持非连续内存访问 | Mamba / FLA 的 chunk cumsum 可用，但 stride 非连续场景要近期 release |
| 异步 copy / Pipeline / TMA | ❌ 不支持 | 双缓冲 / pipeline 优化要手动拆 stage |
| `tt.fp_to_fp` FP8 cast | ✅ 2025 Q3 扩展，但**软件实现**（位操作） | FP8 quant/dequant 在 910B 上能跑，性能远低于硬件 cast |
| `scatter_ub_to_out` (UB → GM) | ✅ 2025 Q3 新增 | EP scatter / MoE permute 类算子可用 |
| Triton 版本 | x86: 3.2.0；ARM: 3.5.0；路线图 2026 全栈升 3.5 | Triton 3.5 之后 API 才完整 |

### 1.4 910B 上已有 native AscendC / triton 实现（作为正确性 oracle）

**vLLM-Ascend（triton-ascend 实现）**：
- `rope_optimization` / `mrope_fusion`
- `penalty_*` / `temperature` / `topk_softmax` / `min_p` / `bincount`
- `causal_conv1d`（triton-ascend 版）
- `chunk_gated_delta_rule`（Qwen3-Next 路径）
- `l2norm`

**vLLM-Ascend（AscendC native op）**：
- `npu_fused_infer_attention_score` (FIA，主 attention)
- `npu_flash_attention_unpad`
- `npu_fusion_attention`（已被 A2/A3 上的 `_npu_flash_attention_unpad` 替代）
- `transpose_kv_cache_by_block` (GQA 加速)
- `GMM` (grouped matmul, MoE)
- `npu_recurrent_gated_delta_rule`

**sgl-kernel-npu（AscendC native op，sgl-project/sgl-kernel-npu）**：
- MLA + paged KV cache decode
- GQA + Decode Attention
- RMSNorm / Fused Add+RMSNorm+Bias / Split QKV+RMSNorm+RoPE
- SwiGLU + Quantized SwiGLU INT8
- LoRA: BGMV / SGMV / SGEMMV expand+shrink
- Causal Conv1D (Mamba SSM)
- Speculative decoding (build_tree / verify_tree)
- Lightning Indexer (sparse top-K, NSA)
- Triangular matrix inverse (`tri_inv`)
- Batch MatMul transpose
- DeepEP-Ascend (token dispatch / combine, fused Deep MoE)

**意义**：上述 native AscendC 实现可作为本 benchmark 中对应 triton kernel 的 **正确性 oracle**（精度对齐）以及 **性能上限对照**（triton-ascend 跑出来应 ≤ AscendC native 的 1.5×）。

## 2. v2 版本相对 v1 的修订记录

v2 基于 §1 的硬件 / 软件事实做以下 7 处修订（题数维持 40/38/22 = 100）：

| 修订 | 类型 | 算子 | 理由 |
|---|---|---|---|
| ❌ 删 | L1-01 | `_act_quant_kernel`（FP8 act quant） | 910B 无 FP8 算术 + 无硬件 FP8 cast；triton-ascend `tt.fp_to_fp` 软件位操作模拟，性能数据失真 |
| ❌ 删 | L1-11 | `_trtllm_prefill_attn_kvfp8_dequant`（FP8 KV dequant） | 同上 |
| ❌ 删 | L1-12 | `_per_token_group_quant_fp8` | 同上 |
| ❌ 删 | L1-14 | `_per_token_quant_int8` | 与 L1-13 `_per_token_group_quant_int8` 形态高度重复（per-token vs per-token-group 在 NPU vector reduce 上差异极小） |
| ❌ 删 | L2-12 | `scaled_mm_kernel`（FP8 scaled MM） | 910B Cube 无 FP8 算术；INT8 等价路径已由 L2-10 `_w8a8_block_int8_matmul` 与 L2-11 `_w8a8_triton_block_scaled_mm` 覆盖 |
| ❌ 删 | L2-16 | `fp8_gemm_group_triton_kernel`（CSV 标记 **A5 独有**） | A5 平台才支持 FP8 cube grouped GEMM；910B 不能跑 |
| ❌ 删 | L3-12 | `_fbgemm_grouped_gemm_fp8_rowwise`（CSV 标 **A5 独有** + FBGEMM） | 双重不适配：A5 独有 + FBGEMM 不在 vLLM/SGLang 路径 |
| ➕ 增 | L1 | `gelu_and_mul_kernel`（SGLang, VV） | Gemma / GPT-OSS MLP 路径，与 L1-20 `silu_and_mul_kernel` 形成 GELU vs SiLU 对照；910B 上 vector exp/tanh 实现成熟 |
| ➕ 增 | L1 | `merge_state_kernel`（SGLang, VV） | SGLang flash-decoding 状态合并，与 L1-28 vLLM `merge_attn_states_kernel` 互补（双源 chunked prefill 合并） |
| ➕ 增 | L1 | `transform_index_page_table_decode_kernel`（SGLang, VV） | DeepSeek-V3 / V4 decode 阶段 page table 变换，与 L1-34 `create_flashmla_kv_indices_triton` 配对 |
| ➕ 增 | L1 | `compute_problem_sizes_w4a8_kernel`（SGLang, VV） | **W4A8 MoE 路径核心元数据**；vLLM-Ascend 已支持 W4A8 / W4A4_LAOS_DYNAMIC，是 910B 的重点量化路径 |
| ➕ 增 | L2 | `_silu_and_mul_post_quant_kernel`（SGLang, VV） | INT8 fused silu+mul+per-token-quant，与 L2-36/37/38 Cutlass W4A8 MoE 三件套配套完整覆盖 |
| ➕ 增 | L2 | `post_reorder_triton_kernel_for_cutlass_moe`（SGLang, VV） | Cutlass W4A8 MoE 输出 reorder，与 L2-37 `pre_reorder_triton_kernel_for_cutlass_moe` 完整成对 |
| ➕ 增 | L3 | `_fwd_kernel_alibi`（vLLM, CV） | ALiBi attention 完整版（BLOOM / MPT / Falcon / Baichuan 路径），与 L3-04 `_fwd_kernel`（RoPE）形成 ALiBi vs RoPE 完整对照 |
| 🔁 保留 + 重标 | L1-38 | `_silu_mul_fp8_quant_deep_gemm`（DeepSeek FP8 silu+mul+quant） | **唯一保留的 FP8 算子**，作为 triton-ascend `tt.fp_to_fp` 软件 FP8 cast + fused vector 算术的 stress test 代表；标注 "仅作正确性比对 + 软件模拟性能参考，不进入硬件能力评估" |

## 3. L1 — 基础单算子（40 题，纯 VV / 单 stage）

| # | 算子名 | 代码仓 | CV/VV | 选择理由 | NPU 难度 | 源码定位 |
|---:|---|---|---|---|---|---|
| L1-01 | `_bias_kernel` | vLLM | VV | Llama/Qwen projection 后 bias 广播，最小 elementwise 基准 | 低 | vLLM 通用 |
| L1-02 | `_compute_slot_mappings_kernel` | vLLM | VV | vLLM paged KV cache prefill/decode slot 映射 | 中低：整数索引 + mask | `vllm/v1/worker/utils.py` |
| L1-03 | `_convert_req_index_to_global_index_kernel` | vLLM | VV | vLLM V1 batched decode request → global 索引转换 | 中低：索引 gather | `vllm/v1/worker/` |
| L1-04 | `_copy_page_indices_kernel` | vLLM | VV | paged attention page-table copy | 低：纯 copy | vLLM v1 attention |
| L1-05 | `_gather_block_tables_kernel` | vLLM | VV | Llama/Qwen/DeepSeek paged FA block table gather | 中低：scatter-gather | `vllm/v1/attention/ops/` |
| L1-06 | `reshape_and_cache_kernel_flash` | vLLM | VV | vLLM FA KV cache 主写入 | 中低：layout store | `vllm/v1/attention/ops/triton_reshape_and_cache_flash.py` |
| L1-07 | `_dequantize_k_cache_fast_kernel` | SGLang | VV | SGLang INT8/MXFP8 KV decode fast path | 中低：VV dequant | `sglang/srt/layers/attention/` |
| L1-08 | `_dequantize_k_cache_paged_kernel` | SGLang | VV | 长上下文 paged KV dequant | 中：paged gather + dequant | `sglang/srt/layers/attention/` |
| L1-09 | `_quantize_k_cache_fast_kernel` | SGLang | VV | KV cache 压缩写入；与 dequant 配对验证 INT8 KV cache (C8) | 中低 | `sglang/srt/layers/attention/` |
| L1-10 | `_per_token_group_quant_int8` | vLLM | VV | Llama/Qwen INT8 group quant — **910B W8A8 主路径** | 中：group reduce | `vllm/model_executor/layers/quantization/utils/int8_utils.py` |
| L1-11 | `awq_dequantize_kernel` | vLLM | VV | AWQ 权重解包（与 AWQ GEMM 分层） | 中：unpack + scale | `vllm/model_executor/layers/quantization/awq_triton.py` |
| L1-12 | `_rms_norm_kernel` | vLLM | VV | Llama/Qwen/DeepSeek 主干 RMSNorm，行内 reduce 核心 | 中：行内 reduce | vLLM elementwise |
| L1-13 | `layer_norm_fwd_kernel` | vLLM | VV | GPT / 多模态 LayerNorm，mean+var 统计 | 中：mean/var reduce | vLLM |
| L1-14 | `_layer_norm_fwd_1pass_kernel` | vLLM、SGLang | VV | Mamba/GDN/FLA gated norm | 中：单 pass reduce | `vllm/model_executor/layers/fla/` |
| L1-15 | `act_and_mul_kernel` | SGLang | VV | SGLang 通用 act*gate | 低：elementwise | `sglang/srt/layers/activation.py` |
| L1-16 | `silu_and_mul_kernel` | SGLang | VV | SwiGLU = Llama / Qwen / DeepSeek FFN 主路径 | 中低：exp + mul | `sglang/srt/layers/activation.py` |
| L1-17 | `gelu_and_mul_kernel` | SGLang | VV | **v2 新增**：Gemma / GPT-OSS MLP 路径，与 silu_and_mul 形成 GELU vs SiLU 对照 | 中低：tanh/erf + mul | `sglang/srt/layers/activation.py` |
| L1-18 | `fused_softcap_kernel` | SGLang | VV | Gemma/Qwen logits softcap | 低：tanh/scale | `sglang/srt/layers/logits_processor.py` |
| L1-19 | `_topk_log_softmax_kernel` | vLLM | VV | top-k logprobs 输出 | 中：reduce + top-k | vLLM sample |
| L1-20 | `_log_softmax_kernel` | vLLM | VV | dense logprobs 输出 | 中：vocab reduce | vLLM sample |
| L1-21 | `_gumbel_sample_kernel` | vLLM | VV | vLLM 随机采样主路径（RNG + argmax） | 中：RNG + reduce | `vllm/v1/worker/gpu/sample/gumbel.py` |
| L1-22 | `_min_p_kernel` | vLLM | VV | min-p sampling 线上常用过滤 | 中低：阈值过滤 | vLLM sample |
| L1-23 | `_penalties_and_temperature_kernel` | vLLM | VV | decode 前 penalty + temperature | 低：elementwise | vLLM sample |
| L1-24 | `_rejection_sample_kernel` | vLLM | VV | EAGLE / spec decode 接受拒绝 | 中低：分支 + mask | vLLM spec_decode |
| L1-25 | `merge_attn_states_kernel` | vLLM | VV | chunked prefill / flash-decoding 合并 KV-split 注意力中间状态 | 中：softmax 状态合并 | `vllm/v1/attention/ops/triton_merge_attn_states.py` |
| L1-26 | `merge_state_kernel` | SGLang | VV | **v2 新增**：SGLang flash-decoding 状态合并（与 L1-25 vLLM 互补，双源对照） | 中：softmax 状态合并 | `sglang/srt/layers/attention/triton_ops/merge_state.py` |
| L1-27 | `_apply_grammar_bitmask_kernel` | vLLM | VV | vLLM v1 grammar/JSON mode 约束采样 | 中：bitmask + logits | `vllm/v1/worker/gpu/structured_outputs.py` |
| L1-28 | `pack_bitmatrix` | vLLM | VV | guided decode bitmask 打包，与 L1-27 配对 | 中：位操作打包 | `vllm/v1/worker/gpu/structured_outputs.py` |
| L1-29 | `_prepare_eagle_inputs_kernel` | vLLM | VV | EAGLE draft 输入准备 | 中低：索引 pack | vLLM spec_decode |
| L1-30 | `_prepare_mrope_positions_kernel` | vLLM | VV | Qwen2-VL / Qwen3-VL mRoPE 位置准备 | 低：整数向量 | vLLM mrope |
| L1-31 | `_triton_mrope_forward` | vLLM | VV | Qwen / Qwen-VL mRoPE apply（vLLM-Ascend **已有 triton-ascend 实现**作 oracle） | 中：rotary vector | vLLM mrope |
| L1-32 | `create_flashinfer_kv_indices_triton` | SGLang | VV | SGLang FlashInfer paged decode KV index | 中低：索引构造 | `sglang/srt/layers/attention/` |
| L1-33 | `create_flashmla_kv_indices_triton` | SGLang | VV | DeepSeek-V3/V4 FlashMLA decode KV index | 中低：MLA 索引 | `sglang/srt/layers/attention/` |
| L1-34 | `transform_index_page_table_decode_kernel` | SGLang | VV | **v2 新增**：DeepSeek-V3/V4 decode 阶段 page table 变换，与 L1-33 FlashMLA 配套 | 中：索引变换 | `sglang/srt/mem_cache/` |
| L1-35 | `compute_src2dst_triton_kernel` | SGLang | VV | SGLang MoE token permutation 索引计算 | 中：索引映射 | `sglang/srt/layers/moe/ep_moe/kernels.py` |
| L1-36 | `compute_seg_indptr_triton_kernel` | SGLang | VV | varlen / MoE 分段 offset 构造 | 中：prefix 元数据 | `sglang/srt/layers/moe/ep_moe/kernels.py` |
| L1-37 | `compute_problem_sizes_w4a8_kernel` | SGLang | VV | **v2 新增**：W4A8 MoE 路径核心元数据（910B 重点量化路径） | 中：shape 元数据 | `sglang/srt/layers/moe/cutlass_w4a8_moe.py` |
| L1-38 | `_silu_mul_fp8_quant_deep_gemm` | vLLM | VV | **v2 唯一保留的 FP8 算子**：作为 triton-ascend `tt.fp_to_fp` 软件 FP8 cast + fused vector 算术的 stress test；DeepSeek FP8 MoE/MLP 路径代表；**性能不进入硬件能力评估** | 中：软件 FP8 cast | `vllm/model_executor/layers/quantization/utils/fp8_utils.py` |
| L1-39 | `_state_passing_fwd_kernel` | vLLM、SGLang | VV | Mamba / SSD chunk 间 state passing | 中：跨 chunk 状态 | `vllm/model_executor/layers/mamba/ops/` |
| L1-40 | `_chunk_cumsum_fwd_kernel` | vLLM、SGLang | VV | Mamba / FLA chunk cumsum 基础 scan | 中：prefix scan（triton-ascend 2025-09 起支持非连续访问） | `vllm/model_executor/layers/fla/ops/` |

## 4. L2 — 复合 fused / CV 混合（38 题）

| # | 算子名 | 代码仓 | CV/VV | 选择理由 | NPU 难度 | 源码定位 |
|---:|---|---|---|---|---|---|
| L2-01 | `_bmm_chunk_fwd_kernel` | vLLM、SGLang | CV | Mamba2 / SSD chunk prefill BMM stage | 中：cube BMM | `vllm/model_executor/layers/mamba/ops/` |
| L2-02 | `_chunk_scan_fwd_kernel` | vLLM、SGLang | CV | Mamba / SSD chunk scan + matmul | 中高：CV + scan | `vllm/model_executor/layers/mamba/ops/ssd_chunk_scan.py` |
| L2-03 | `_chunk_state_fwd_kernel` | vLLM | CV | Mamba 固定长度 state 构造 | 中：cube + mask | `vllm/model_executor/layers/mamba/ops/ssd_chunk_state.py` |
| L2-04 | `_chunked_lora_expand_kernel` | SGLang | CV | SGLang LoRA chunked prefill expand | 中：LoRA cube | `sglang/srt/lora/` |
| L2-05 | `_chunked_lora_shrink_kernel` | SGLang | CV | LoRA shrink，与 expand 配对 | 中：小 K matmul | `sglang/srt/lora/` |
| L2-06 | `_qkv_lora_b_kernel` | SGLang | CV | Llama / Qwen LoRA QKV B projection | 中：cube + layout | `sglang/srt/lora/` |
| L2-07 | `_gate_up_lora_b_kernel` | SGLang | CV | FFN gate / up LoRA B | 中：cube + slice | `sglang/srt/lora/` |
| L2-08 | `_sgemm_lora_a_kernel` | SGLang | CV | SGLang prefill LoRA A（sgl-kernel-npu 已有 SGEMMV native 实现作 oracle） | 中：小矩阵 cube | `sglang/srt/lora/` |
| L2-09 | `_sgemm_lora_b_kernel` | SGLang | CV | LoRA B 写回 hidden | 中：小 K/N tile | `sglang/srt/lora/` |
| L2-10 | `_w8a8_block_int8_matmul` | vLLM | CV | **910B W8A8 主路径**：INT8 block matmul（Llama/Qwen 量化） | 中高：cube + scale | `vllm/model_executor/layers/quantization/utils/int8_utils.py` |
| L2-11 | `_w8a8_triton_block_scaled_mm` | vLLM | CV | W8A8 block scaled MM 核心（INT8 / BF16 路径） | 中高：cube + block scale | `vllm/model_executor/layers/quantization/compressed_tensors/triton_scaled_mm.py` |
| L2-12 | `awq_gemm_kernel` | vLLM | CV | AWQ 量化 GEMM 主算子（dequant-in-GEMM） | 中高：cube + unpack | `vllm/model_executor/layers/quantization/awq_triton.py` |
| L2-13 | `bmm_kernel` | vLLM | CV | attention / MLA / 小投影通用 BMM | 中：cube BMM | vLLM |
| L2-14 | `matmul_kernel_persistent` | vLLM | CV | persistent matmul，验证 triton-ascend 对 persistent tile 调度的支持（910B 当前不支持 persistent，需评估退化为 non-persistent） | 中高：persistent cube | `vllm/model_executor/layers/batch_invariant.py` |
| L2-15 | `fused_dual_residual_rmsnorm_kernel` | SGLang | VV | SGLang transformer block residual + RMSNorm fused（sgl-kernel-npu Fused Add+RMSNorm+Bias 作 oracle） | 中：多输入 reduce | `sglang/srt/layers/elementwise.py` |
| L2-16 | `fused_rmsnorm_kernel` | SGLang | VV | decode 高频小 kernel | 中：reduce + cast | `sglang/srt/layers/elementwise.py` |
| L2-17 | `layer_norm_gated_fwd_kernel` | vLLM、SGLang | VV | FLA / GDN / Mamba gated norm | 中：norm + gate | `vllm/model_executor/layers/fla/ops/` |
| L2-18 | `fused_gdn_gating_kernel` | vLLM、SGLang | VV | GDN / linear attention gating | 中：非线性 + gate | `vllm/model_executor/layers/fla/ops/` |
| L2-19 | `fused_sigmoid_gating_delta_rule_update_kernel` | SGLang | VV | Gated DeltaNet decode update | 中：recurrent update | `sglang/srt/layers/attention/fla/` |
| L2-20 | `_causal_conv1d_update_kernel` | vLLM | VV | Mamba decode conv state 更新 | 中：状态更新 | `vllm/model_executor/layers/mamba/ops/causal_conv1d.py` |
| L2-21 | `_causal_conv1d_fwd_kernel` | vLLM、SGLang | VV | Mamba prefill causal conv（vLLM-Ascend **已有 triton-ascend 实现**作 oracle） | 中：windowed vector | 同上 |
| L2-22 | `_linear_attn_decode_kernel` | vLLM | VV | Linear attention decode recurrent reduce | 中：recurrent reduce | `vllm/model_executor/layers/lightning_attn.py` |
| L2-23 | `_fwd_kv_parallel` | vLLM | CV | Lightning attention KV parallel stage | 中高：cube + KV tile | `vllm/model_executor/layers/lightning_attn.py` |
| L2-24 | `_fwd_kv_reduce` | vLLM | VV | Lightning attention 跨 tile reduce | 中：跨 tile reduce | 同上 |
| L2-25 | `_fwd_kernel_ep_gather` | vLLM、SGLang | VV | MoE expert parallel token gather | 中：EP gather | `sglang/srt/layers/moe/ep_moe/kernels.py` |
| L2-26 | `_fwd_kernel_ep_scatter_1` | vLLM、SGLang | VV | MoE EP output scatter（依赖 triton-ascend 2025-Q3 新增 `scatter_ub_to_out`） | 中：EP scatter | 同上 |
| L2-27 | `deepep_permute_triton_kernel` | SGLang | VV | DeepSeek MoE / DeepEP token dispatch permute | 中：不规则搬运 | 同上 |
| L2-28 | `deepep_post_reorder_triton_kernel` | SGLang | VV | DeepEP 专家输出 reorder | 中：scatter-gather | 同上 |
| L2-29 | `fused_moe_router_cudacore_kernel` | SGLang | VV | DeepSeek / Qwen-MoE 纯 vector router | 中：top-k / router | `sglang/srt/layers/moe/router.py` |
| L2-30 | `fused_moe_router_tensorcore_kernel` | SGLang | CV | cube router，与 cudacore 对照 | 高：cube router | 同上 |
| L2-31 | `moe_align_block_size_stage2` | SGLang | VV | MoE block alignment（只选 stage2 代表） | 中：prefix + pad | `sglang/srt/layers/moe/ep_moe/kernels.py` |
| L2-32 | `experts_combine_kernel` | SGLang | VV | MoE top-k expert 加权 combine | 中：reduce combine | `sglang/srt/layers/moe/` |
| L2-33 | `_fwd_grouped_kernel_stage1_rope` | SGLang | CV | DeepSeek-V3/V4 MLA stage1 + RoPE fused | 高：MLA + rope cube | `sglang/srt/layers/attention/triton_ops/rocm_mla_decode_rope.py` |
| L2-34 | `silu_mul_static_tensorwise_quant_triton_kernel_for_cutlass_moe` | SGLang | VV | Cutlass W4A8 MoE silu+mul+per-tensor-quant fused | 中：fused 量化 | `sglang/srt/layers/moe/cutlass_w4a8_moe.py` |
| L2-35 | `pre_reorder_triton_kernel_for_cutlass_moe` | SGLang | VV | Cutlass W4A8 MoE 入口 reorder | 中：scatter-gather | 同上 |
| L2-36 | `post_reorder_triton_kernel_for_cutlass_moe` | SGLang | VV | **v2 新增**：Cutlass W4A8 MoE 输出 reorder，与 L2-35 完整成对 | 中：scatter-gather | 同上 |
| L2-37 | `_silu_and_mul_post_quant_kernel` | SGLang | VV | **v2 新增**：INT8 fused silu+mul+per-token-quant，与 L2-34 静态 quant 形成动态 vs 静态对照 | 中：fused 量化 | `sglang/srt/layers/activation.py` |
| L2-38 | `concat_and_cast_mha_k_kernel` | SGLang | VV | MHA K cache 拼接 + dtype cast，DeepSeek-V3 K 缓存路径 | 中：拼接 + cast | `sglang/srt/mem_cache/` |

## 5. L3 — 完整子系统 / 高复杂度（23 题 = 22 scored + 1 appendix，v3.1 errata 补 L3-23）

| # | 算子名 | 代码仓 | CV/VV | 选择理由 | NPU 难度 | 源码定位 |
|---:|---|---|---|---|---|---|
| L3-01 | `kernel_unified_attention_3d` | vLLM | CV | vLLM chunked prefill / decode unified attention 高维入口 | 高：多 stage tile | `vllm/v1/attention/ops/` |
| L3-02 | `kernel_paged_attention_2d` | vLLM | CV | Llama / Qwen / DeepSeek paged decode 服务主路径（910B 实际线上用 AscendC FIA，本题测 triton-ascend 编译器能力上限） | 高：paged KV + cube | 同上 |
| L3-03 | `kernel_unified_attention_2d` | vLLM | CV | vLLM 2D chunked prefill 形态，与 L3-01 3D 互补 | 高：多 stage tile | `vllm/v1/attention/ops/chunked_prefill_paged_decode.py` |
| L3-04 | `_fwd_kernel` | vLLM | CV | Triton Flash Attention prefill 基础 kernel（long context） | 高：softmax + cube | `vllm/v1/attention/ops/prefix_prefill.py` |
| L3-05 | `_fwd_kernel_alibi` | vLLM | CV | **v2 新增**：ALiBi attention 完整版（BLOOM / MPT / Falcon / Baichuan 路径），与 L3-04 RoPE 对照 | 高：ALiBi 偏置 + softmax + cube | 同上 |
| L3-06 | `_fwd_grouped_kernel_stage1` | vLLM | CV | grouped / MLA attention stage1（DeepSeek-V3/V4 类） | 高：grouped tile | `sglang/srt/layers/attention/triton_ops/decode_attention.py` |
| L3-07 | `sdpa (attention+mask)` | 字节中台编译 | 未标 | 保留作为完整 SDPA + mask + softmax + matmul 融合形态参考 | 高：mask + softmax | 无公开源码 |
| L3-08 | `jagged_dense_flash_attention_fwd_kernel` | FBGEMM | CV | **保留**作为唯一 jagged FA 代表（910B 评估变长 attention 能力，仅作正确性参考） | 高：jagged shape | FBGEMM 原仓 |
| L3-09 | `fused_moe_kernel` | vLLM、SGLang | CV | DeepSeek / Qwen-MoE 主专家 GEMM（routing + top-k + grouped GEMM + scale） | 高：grouped MoE cube | `vllm/model_executor/layers/fused_moe/fused_moe.py` |
| L3-10 | `fused_moe_kernel_gptq_awq` | vLLM、SGLang | CV | 量化 MoE 专家路径（GPTQ / AWQ dequant-in-GEMM） | 高：量化 MoE | 同上 |
| L3-11 | `expert_triton_kernel` | vLLM | CV | vLLM MoE 单专家执行 | 高：专家 tile | `vllm/model_executor/layers/fused_moe/` |
| L3-12 | `moe_mmk` | vLLM | CV | MoE mixed-matrix kernel | 高：动态 grouped GEMM | `vllm/model_executor/layers/fused_moe/` |
| L3-13 | `chunk_gated_delta_rule_fwd_kernel_h_blockdim64` | vLLM、SGLang | CV | Gated DeltaNet / GDN full chunk fwd（vLLM-Ascend **已有 triton-ascend 实现**作 oracle） | 高：chunk recurrent + cube | `vllm/model_executor/layers/fla/ops/chunk.py` |
| L3-14 | `fused_recurrent_gated_delta_rule_fwd_kernel` | vLLM、SGLang | VV | GDN recurrent fwd 长序列（sgl-kernel-npu **已有 AscendC 实现**作 oracle） | 高：循环依赖 | `vllm/model_executor/layers/fla/ops/fused_recurrent.py` |
| L3-15 | `fused_recurrent_gated_delta_rule_update_fwd_kernel` | SGLang | VV | SGLang decode 单 token recurrent state 更新 | 高：state update | `sglang/srt/layers/attention/fla/` |
| L3-16 | `chunk_fwd_kernel_o` | vLLM、SGLang | CV | Mamba / FLA chunk 输出阶段 | 高：chunk output | `vllm/model_executor/layers/fla/ops/chunk.py` |
| L3-17 | `chunk_gla_fwd_kernel_o` | vLLM | CV | GLA / linear attention 输出阶段 | 高：GLA tile | `vllm/model_executor/layers/fla/ops/` |
| L3-18 | `chunk_scaled_dot_kkt_fwd_kernel` | vLLM、SGLang | CV | GDN / KDA KKT scaled-dot chunk 核心 | 高：KKT cube | `vllm/model_executor/layers/fla/ops/kda.py` |
| L3-19 | `chunk_kda_scaled_dot_kkt_fwd_kernel_intra_sub_inter` | vLLM、SGLang | CV | KDA inter-sub 跨块依赖 | 高：跨块依赖 | 同上 |
| L3-20 | `recompute_w_u_fwd_kernel` | vLLM、SGLang | CV | GDN / KDA full pipeline W/U recompute | 高：recompute + cube | 同上 |
| L3-21 | `solve_tril_16x16_kernel` | vLLM、SGLang | VV | GDN / KDA chunk 16x16 三角求解（sgl-kernel-npu `tri_inv` AscendC 作 oracle） | 中高：小矩阵依赖 | `vllm/model_executor/layers/fla/ops/solve_tril.py` |
| L3-22 | `chunk_local_cumsum_vector_kernel` | vLLM、SGLang | CV | Mamba / FLA chunk-local cumsum CV 完整版 | 高：vector scan + cube | `vllm/model_executor/layers/fla/ops/` |
| **L3-23** | `merge_16x16_to_32x32_inverse_kernel` | vLLM、SGLang | CV | **v3.1 errata 新增**：FLA chunk 块大小变换的 inverse op（16×16 → 32×32），是 GDN/KDA chunk pipeline 完整子系统的真实组件（vLLM Apache 2.0 公开可验证）；与 L3-21 `solve_tril_16x16_kernel` 配对覆盖 chunk 小矩阵运算路径 | 高：cube + 不规则维度变换 + UB tile 设计 | `vllm/model_executor/layers/fla/ops/` + `sglang/srt/layers/attention/fla/` |

## 6. 覆盖矩阵（v2）

| 类别 | L1 | L2 | L3 | 合计 |
|---|---:|---:|---:|---:|
| Attention / MLA / paged decode / Flash Attention | 3 | 3 (Lightning + MLA-rope) | 8 | 14 |
| MoE / EP / 路由 / dispatch / Cutlass W4A8 MoE | 3 (含 W4A8 元数据) | 11 (含 Cutlass W4A8 三件套) | 4 | 18 |
| Norm / 激活 / residual fusion | 5 (含 GELU + SiLU 对照) | 4 | 0 | 9 |
| Quant / dequant / 量化 matmul | 2 (INT8 group + AWQ dequant) | 5 (W8A8 INT8 + AWQ + Cutlass W4A8 fused quant) | 0 | 7 |
| Sampling / spec decode / logits / structured outputs | 10 (含 grammar bitmask + pack_bitmatrix) | 0 | 0 | 10 |
| KV cache 索引 / page table | 6 (含 DeepSeek-V3 page table 变换) | 0 | 0 | 6 |
| LoRA | 0 | 6 | 0 | 6 |
| State-space / Mamba / GDN / GLA / KDA / linear attention | 4 | 7 | 9 | 20 |
| Cube GEMM 基础 / persistent | 0 | 3 | 0 | 3 |
| Jagged / unstructured | 0 | 0 | 1 | 1 |
| RoPE / 位置 | 2 | 0 | 0 | 2 |
| Attention 状态合并（chunked / flash-decoding） | 2 (vLLM + SGLang 双源) | 0 | 0 | 2 |
| FP8 软件模拟 stress test | 1 (silu_mul_fp8 唯一保留) | 0 | 0 | 1 |
| ALiBi attention | 0 | 0 | 1 | 1 |
| **合计** | **40** | **38** | **22** | **100** |

## 7. 显式排除（v2 更新版）

| 排除类别 | 数量 | 主因 |
|---|---:|---|
| **FP8 cube 算术依赖**（scaled_mm、fp8_gemm_group、_fbgemm_grouped_gemm_fp8、_per_token_group_quant_fp8 等） | 6+ | **910B Cube 单元无 FP8 算术指令**；软件模拟性能不能反映硬件能力 |
| FP4 / NF4 / MXFP4 cube 类（`_kernel_dequantize_mx4`、`_kernel_rms_quantize_mx4_unpack`、`_kernel_silu_quantize_mx4_unpack` 等） | 5+ | 910B 仅支持 FP4/NF4 存储，无 cube 算术；要等 A5 / 950 |
| A5 独有算子（CSV "A5独有" 中依赖 A5 新指令的算子） | 多 | 910B 平台不支持 |
| 训练后向（`_bwd_*` / `*_backward*`） | 全部 | 用户场景 agentic RL rollout / 推理优先，policy update 用普通 PyTorch |
| FBGEMM jagged FA / matmul fp8 row/block 系列 | 12+ | `infer/vllm` / `infer/sglang` 全部 0 匹配；与 vLLM/SGLang 推理路径无关；只保留 1 个 jagged FA fwd 代表（L3-08，仅作变长 attention 参考） |
| FBGEMM quant/dequant fp8 row/block | 10+ | 同上 + 910B 无 FP8 cube |
| FBGEMM jagged 元数据（jagged_to_dense / array_jagged_bmm 等） | 10+ | jagged tensor 是 PyTorch FBGEMM 推荐系统概念，与 LLM 推理无关 |
| 私有"字节中台编译"未标代码仓 | 27/30 | 不可验证源码；只保留 `sdpa (attention+mask)` 作完整 attention 形态参考 |
| 私有"字节智创"（`_binned_copy*`、`_padded_copy_wgrad`） | 3 | 同上；功能已覆盖 |
| 同族 stage 重复（`moe_align_block_size_stage1/3/4`、`_fwd_kernel_ep_scatter_2`、`_fwd_diag_kernel`、`_fwd_none_diag_kernel`、`l2norm_fwd_kernel1/2`、`layer_norm_gated_fwd_kernel1`） | 8+ | 同 fused op 的 stage 变体只选代表 |
| 重复量化形态（`_per_token_quant_int8` 与 `_per_token_group_quant_int8` 形态接近） | 1 | v2 新增去重：保留 group 版 |
| 调试 / launch helper（`_num_nans_kernel`、`_compute_pid`、`hf_weight_loading`、`_assertion_rewrite`、`_operator_dispatch`） | 5 | 对 NPU 算子迁移代表性不足 |
| 训练专用 `chunk_bwd_*` / `prepare_wy_repr_bwd` / `layer_norm_gated_bwd` 等 | 6+ | 训练后向 |
| 字节中台 `_*_forward_backward_diff(fp32)` 三件套 | 3 | 训练用 diff 算子 |

**总排除约 120+ 项**（294 个候选 - 100 选中 - 部分 candidates 自然分布）。

## 8. 910B Benchmark 推进建议（v2 更新）

### 8.1 执行优先级

1. **Phase 1（L1 前 20 题）**：纯 elementwise / reduce / index gather / RMSNorm / SwiGLU / GELU / sampling — 验证 triton-ascend 基础 VV 编译路径，~2 周
2. **Phase 2（L1 21-40）**：W8A8/W4A8 量化 quant、KV cache 索引、structured outputs、Mamba state-passing — 验证整型路径 + 软件 FP8 cast 边界，~2 周
3. **Phase 3（L2 1-20）**：LoRA + W8A8 INT8 matmul + AWQ + LightningAttention + MoE EP — 验证 cube 单元 + scatter_ub_to_out，~3 周
4. **Phase 4（L2 21-38）**：Mamba conv1d + GDN gating + Cutlass W4A8 MoE 五件套 + MLA-rope — 验证 fused 模式，~3 周
5. **Phase 5（L3 1-12 attention/MoE）**：完整 Flash Attention / Paged Decode / Unified Attention / Fused MoE — triton-ascend 编译器能力上限测试，~4 周
6. **Phase 6（L3 13-22 Mamba/GDN/KDA）**：完整 chunk pipeline + ALiBi — 最终 stress test，~4 周

### 8.2 正确性参考标准

| 算子类 | atol / rtol | 备注 |
|---|---|---|
| FP16 / BF16 elementwise + reduce | `atol=1e-3, rtol=1e-3` | 标准 |
| INT8 / W8A8 cube matmul | `atol=5e-2, rtol=1e-2` | INT8 量化误差预期 |
| FP8 软件模拟 (L1-38 唯一保留) | `atol=1e-1, rtol=5e-2` | **仅做正确性参考**，性能数据失真 |
| Mamba / FLA chunk recurrent | `atol=5e-3, rtol=5e-3` | 跨 chunk 累加 |
| ALiBi attention | `atol=1e-3, rtol=1e-3` | bias 加和有限 |

### 8.3 性能指标

报告以下指标，并标注是否在 910B 真实硬件 cube/vector 上执行：

- `kernel_time_ms`（中位数 + p99）
- `memory_bandwidth_gb_s`（实际 / 理论峰值；910B HBM 峰值 ~1.6 TB/s）
- `compute_utilization`（INT8 TOPS / FP16 TFLOPS / peak；910B INT8 ≈ 512 TOPS, FP16 ≈ 256 TFLOPS）
- `cube_util`（仅 CV 算子）
- `ub_usage_kb`（应 ≤ 192 KB）

### 8.4 NPU 难点预警

| 题 | 预警内容 |
|---|---|
| L1-38 `_silu_mul_fp8_quant_deep_gemm` | **唯一 FP8 算子**，依赖 `tt.fp_to_fp` 软件位操作模拟；性能数据**不进入硬件能力评估**，仅作正确性参考 |
| L2-14 `matmul_kernel_persistent` | 910B 当前不支持 persistent kernel；评估能否退化为 non-persistent + iteration |
| L2-30 `fused_moe_router_tensorcore_kernel` / L3-09 `fused_moe_kernel` / L3-10 `fused_moe_kernel_gptq_awq` | Grouped GEMM 在 910B 上要 cube tile padding（16 倍数），关注 padding cost |
| L3-04 / L3-05 `_fwd_kernel*` / L3-09 `fused_moe_kernel` | 完整 online softmax + matmul fused，UB ≤ 192 KB 限制下要重新设计 BLOCK_M × BLOCK_K |
| L3-13 / L3-16 / L3-17 / L3-18 / L3-19 / L3-20 / L3-22 | Mamba / GDN / KDA chunk pipeline 涉及不规则 block 协作 + recurrent 依赖，是 triton-ascend 真正难关 |
| L2-26 `_fwd_kernel_ep_scatter_1` | 依赖 triton-ascend 2025-Q3 新增 `scatter_ub_to_out` API；老版本 triton-ascend 不可用 |
| L2-31 `moe_align_block_size_stage2` / L3-09/10 中的 atomic 操作 | `tl.atomic_add` 在 BF16 / FP16 上不支持；需用 FP32 累加或拆 stage 实现 |

### 8.5 复用现有 NPU 实现作 oracle

| Benchmark 题 | 已有 NPU 实现 | 路径 |
|---|---|---|
| L1-31 `_triton_mrope_forward` | vLLM-Ascend triton-ascend | `infer/vllm-ascend/vllm_ascend/ops/triton/` |
| L1-39/40 `_state_passing_fwd_kernel` / `_chunk_cumsum_fwd_kernel` | vLLM-Ascend triton + sgl-kernel-npu AscendC | `infer/vllm-ascend/` + `infer/sglang/sgl-kernel-npu/csrc/` |
| L2-08/09 `_sgemm_lora_a/b_kernel` | sgl-kernel-npu SGEMMV AscendC | `infer/sglang/sgl-kernel-npu/python/sgl_kernel_npu/` |
| L2-15 `fused_dual_residual_rmsnorm_kernel` | sgl-kernel-npu Fused Add+RMSNorm AscendC | 同上 `csrc/` |
| L2-21 `_causal_conv1d_fwd_kernel` | vLLM-Ascend triton + sgl-kernel-npu AscendC | `infer/vllm-ascend/` + `infer/sglang/sgl-kernel-npu/csrc/` |
| L3-02 `kernel_paged_attention_2d` / L3-06 `_fwd_grouped_kernel_stage1` | sgl-kernel-npu MLA + paged KV AscendC | `infer/sglang/sgl-kernel-npu/csrc/lightning_indexer/` 等 |
| L3-13 `chunk_gated_delta_rule_fwd_kernel_h_blockdim64` | vLLM-Ascend triton-ascend (Qwen3-Next 路径) | `infer/vllm-ascend/vllm_ascend/ops/triton/fla/sigmoid_gating.py` |
| L3-14 `fused_recurrent_gated_delta_rule_fwd_kernel` | sgl-kernel-npu AscendC `recurrent_gated_delta_rule` | `infer/sglang/sgl-kernel-npu/csrc/recurrent_gated_delta_rule/` |
| L3-21 `solve_tril_16x16_kernel` | sgl-kernel-npu AscendC `tri_inv` | `infer/sglang/sgl-kernel-npu/csrc/tri_inv/` |

跑 benchmark 时先对比 triton-ascend 输出与上述 native 实现，再对比 GPU triton 输出，三方都对齐了再做性能比较。

## 9. v2 文档变更总结

| 维度 | v1 | v2 |
|---|---|---|
| 目标平台说明 | 笼统说 910B | 明确：Atlas A2 = 910B；CSV 是 A3/A5 视角，与 910B 有 gap 并明示 |
| 硬件 dtype 矩阵 | 无 | 新增完整 dtype 支持矩阵（FP8/FP4 软件模拟标注） |
| triton-ascend 约束 | 仅在 NPU 难点中零散提及 | 新增 §1.3 系统化约束清单（UB 192 KB / atomic dtype / dot 维度 / scan / TMA） |
| Native AscendC 实现作 oracle | 仅在第 9 节末尾简提 | 新增 §1.4 与 §8.5 完整 oracle 对照表 |
| FP8 算子 | L1 4 个 + L2 2 个 + L3 1 个 = 7 个 | 保留 1 个（L1-38，明示"性能不进入硬件评估"），删 6 个 |
| W4A8 / Cutlass MoE | L2 仅有 2 个（L2-36/37 silu_mul + pre_reorder） | L1 + L2 完整覆盖（L1-37 compute_problem_sizes + L2-34/35/36/37/38 五件套 + 元数据） |
| ALiBi attention | 缺 | L3-05 新增（BLOOM/MPT/Falcon 路径） |
| GELU MLP 路径 | 缺 | L1-17 新增（Gemma/GPT-OSS） |
| 双源 attention 状态合并 | 仅 vLLM `merge_attn_states` | L1-25/26 加入 SGLang `merge_state_kernel` |
| DeepSeek-V3 page table 变换 | 缺 | L1-34 新增（与 FlashMLA KV index 配套） |
| INT8 量化重复去重 | per-token + per-token-group 都选 | 删 per-token 版，保留 group 版 |
| 显式排除分类 | 8 类 | 13 类（新增 FP8 cube、FP4/MXFP4、A5 独有指令依赖等独立分类） |
| 推进 Phase | 笼统 5 段 | 6 个 Phase 各带预期周数 |


## 10. Benchmark Protocol — 对齐 KernelBench 的科学性要求（v3 新增）

KernelBench 的科学性核心在于 **可复现 + 可比较 + 防 reward-hacking**。本 benchmark 沿用同一框架，针对 Migration 场景调整。

### 10.1 Baseline 与单一指标

**三层 baseline**（按优先级）：

1. **Primary**：GPU Triton 原 kernel（来自 `infer/vllm/` 或 `infer/sglang/`），在 H100 / A100 / H800 上跑出 `t_gpu_triton`
2. **Secondary**：AscendC native 实现（来自 `infer/sgl-kernel-npu/` 或 `infer/vllm-ascend/`），在 910B 上跑出 `t_ascendc`，作为 NPU 性能上限参考
3. **Tertiary**：PyTorch eager（910B + torch_npu），跑出 `t_torch_eager`，作为正确性 oracle 兜底

**单一 benchmark 指标 — `migration_score`**：

```
migration_score(p) = (# tasks with correctness=True AND t_npu_triton ≤ p × t_ascendc) / N_total
```

类比 KernelBench `fast_p`：
- `migration_score(1.0)` = NPU triton 跑出来 ≥ AscendC native 的速度（"到达硬件能力上限"）
- `migration_score(1.5)` = NPU triton ≤ 1.5× AscendC time（"性能可接受"）
- `migration_score(∞)` = 单纯正确率（NPU triton 能跑且数值对齐）
- 若该题没有 AscendC 实现，fallback 用 `t_torch_eager` 作 denominator

**辅助指标**：
- `correctness_rate` = 正确题数 / 总题数（独立于性能）
- `gpu_keep_rate` = `t_gpu_triton / t_npu_triton`（NPU 跑到 GPU 多少比例的速度；< 1 表示 NPU 比 GPU 慢）
- 按 L1/L2/L3 拆开报，避免单一汇总掩盖梯度差异

### 10.2 Correctness Protocol（防 reward-hacking 关键）

借鉴 KernelBench `eval_kernel_against_ref()`：

| 项 | 设定 |
|---|---|
| 随机输入次数 `n_correctness` | **5 次**，每次独立 seed |
| Seed 范围 | `seed = 42 + task_id × 1000 + trial_id`（固定可复现） |
| 输入分布 | float: `torch.rand` (U[0,1])；int: `torch.randint`；mask: 50% True + 边界 5% / 95% / 全 True / 全 False 共 5 个 case |
| 容差 | FP16/BF16: `atol=1e-3, rtol=1e-3`；INT8/W8A8: `atol=5e-2, rtol=1e-2`；FP8 软件模拟: `atol=1e-1, rtol=5e-2`；recurrent 长 chunk: `atol=5e-3, rtol=5e-3` |
| 对比函数 | `torch.allclose(npu_out, gpu_out, atol, rtol)`，跨 device 通过 `.cpu()` 后对比 |
| 数值溢出检查 | 任一输入 trial 出 NaN/Inf → 标记 `correctness=False` |

**Reward-hacking 防御**（受 KernelBench `kernel_static_checker` + 对抗测试启发）：

| 风险 | 防御 |
|---|---|
| 复用 PyTorch reference 的中间张量 | NPU triton kernel 启动前 `torch.npu.synchronize()` + clear cache；输入 tensor 独立 clone |
| 修改输入做"对齐" | 跑完检查 `input_tensor` 内容 hash 与初始 hash 相同 |
| 非默认 NPU stream 偷计算 | 强制单 stream + sync before timing |
| 速度过快（> 3× AscendC native） | **自动 flag**，人工 review；KernelBench 经验：> 2× 大概率是 bug 或 hack |
| 编译期常量替换 | static check：kernel 源码不能含 task-specific magic number |
| 触发 NPU 异步 copy 但不等 | `npu.synchronize()` 必在 timing 之外但 result 读取之前 |

### 10.3 Performance Protocol（对齐 KernelBench timing.py）

| 项 | 设定 |
|---|---|
| Warmup 次数 | **10 次**（NPU JIT + cache 预热） |
| Measurement trial 次数 | **100 次** |
| Timing 方法 | 主：`torch.npu.Event(enable_timing=True)` start/end（类 cuda_event）；辅：`time.perf_counter` host time E2E |
| 统计量 | **中位数 + p99 + std**（KernelBench 推荐中位数避免 outlier） |
| Frequency lock | NPU 频率锁到 nominal（避免热降频）；BIOS power profile = performance |
| 内存清理 | 每 10 trial `torch.npu.empty_cache()` + `synchronize()` |
| GPU baseline 同样规则 | H100/A100 上同 protocol 跑 GPU triton，得 `t_gpu_triton` |

### 10.4 Adversarial 单元测试（受 KernelBench `test_eval_adversarial.py` 启发）

每个 task 应有至少 3 个 adversarial 输入：

1. **极端 shape**：(BS=1, seq=1) / (BS=1, seq=very_large) / (BS=large, seq=1)
2. **边界数值**：含 ±inf / NaN / underflow 极小值
3. **mask 边界**：全 True / 全 False / 棋盘格 / 第一行/列

跑通 adversarial 才算 `robust_correctness=True`，是 `correctness=True` 的更严格版本。

---

## 11. Task 文件规范 — 对齐 KernelBench 的可执行性（v3 新增）

KernelBench 每题是一个独立 Python 文件，包含 `class Model(nn.Module)` + `get_inputs()` + `get_init_inputs()`。本 benchmark 沿用同款规范，但 reference 是 GPU triton kernel：

### 11.1 Task 文件目录结构（计划）

```
docs/npu/tasks/
├── level1/
│   ├── L1_01_bias.py
│   ├── L1_02_compute_slot_mappings.py
│   ├── ...
│   └── L1_40_chunk_cumsum_fwd.py
├── level2/
│   ├── L2_01_bmm_chunk_fwd.py
│   ├── ...
├── level3/
│   ├── ...
└── shared/
    ├── pytorch_oracles.py   # 用于交叉验证的 PyTorch 等价实现
    ├── ascendc_oracles.py   # 来自 sgl-kernel-npu / vllm-ascend 的 AscendC oracle 引用
    └── shape_configs.py     # 每题的 small/medium/large shape 配置
```

### 11.2 单题文件模板

```python
"""
L1-01: _bias_kernel (vLLM, VV)
来源: vllm/vllm/model_executor/layers/.../bias.py
功能: y = x + bias 广播加法
难度: 低（连续 elementwise）
NPU 难点: 无；最小 baseline 题
"""
import torch
import triton
import triton.language as tl

# --- GPU Triton reference (来自 vLLM 原 kernel) ---
@triton.jit
def _bias_kernel(...):
    ...  # 原 GPU 实现，保持原样

# --- PyTorch oracle (兜底) ---
class PyTorchRef(torch.nn.Module):
    def forward(self, x, bias):
        return x + bias

# --- Shape 配置 (small/medium/large) ---
SHAPE_CONFIGS = [
    dict(name="small",  bs=1,   seq=128,  hidden=4096,  dtype=torch.bfloat16),
    dict(name="medium", bs=8,   seq=2048, hidden=4096,  dtype=torch.bfloat16),
    dict(name="large",  bs=32,  seq=8192, hidden=16384, dtype=torch.bfloat16),
]

def get_inputs(config):
    bs, seq, hidden, dt = config["bs"], config["seq"], config["hidden"], config["dtype"]
    torch.manual_seed(42)
    x    = torch.rand(bs, seq, hidden, dtype=dt)
    bias = torch.rand(hidden, dtype=dt)
    return [x, bias]

def get_init_inputs():
    return []  # bias_kernel 无 init 参数

# --- Adversarial cases ---
ADVERSARIAL = [
    dict(bs=1, seq=1, hidden=4096),       # extreme small
    dict(bs=1, seq=65536, hidden=4096),   # long seq
    dict(bs=128, seq=1, hidden=4096),     # large batch decode
]

# --- Metadata ---
META = dict(
    level=1, idx=1, name="bias_kernel", repo="vLLM",
    dtype_class="VV", category="elementwise",
    npu_difficulty="low",
    ascendc_oracle=None,         # 无 native AscendC 实现
    pytorch_oracle="PyTorchRef",
    expect_speedup_threshold=1.0,  # NPU triton ≥ 1.0× PyTorch eager 才算合格
)
```

### 11.3 每题强制元数据字段

| 字段 | 类型 | 说明 |
|---|---|---|
| `level` | int (1/2/3) | 难度分级 |
| `idx` | int | 题号 |
| `name` | str | 算子名 |
| `repo` | str | 来源（vLLM / SGLang / FBGEMM / 私有） |
| `dtype_class` | str | VV / CV |
| `category` | str | attention / moe / norm / quant / sampling / kv_cache / lora / state_space / activation 之一 |
| `npu_difficulty` | str | low / medium / high |
| `ascendc_oracle` | str or None | 若有 AscendC native 实现，给路径 |
| `pytorch_oracle` | str | PyTorch ref class 名 |
| `expect_speedup_threshold` | float | 该题预期最低 speedup（用于 fast_p 计算） |
| `requires_triton_ascend_features` | list[str] | 该题依赖哪些 triton-ascend 特性（`scatter_ub_to_out` / `tt.fp_to_fp` / `tl.scan_noncontiguous` 等） |

---

## 12. V3 路线图 — 从文档到可运行 Benchmark（v3 新增）

当前 v3 仅完成 "**选题 + protocol 规范**"。从 KernelBench 完整度看，还差 4 个 Phase：

| Phase | 任务 | 产物 | 工作量估计 |
|---|---|---|---|
| **A. Task 文件生成** | 把 100 题逐一从 `infer/vllm` / `infer/sglang` 抽取 GPU triton kernel 源码，写成 §11 模板格式的独立 Python 文件 | `docs/npu/tasks/level{1,2,3}/L?_??_*.py` × 100 | 2-3 周（自动化脚本 + 人工审核） |
| **B. Evaluator Harness** | 复用 KernelBench `src/kernelbench/eval.py` + `timing.py` 的 90% 代码，把 `cuda_event` 改成 `torch.npu.Event`，把 GPU device 选择改成 NPU | `npu_kernel_bench/` 包，含 `eval.py` / `timing.py` / `run_and_check.py` | 1-2 周 |
| **C. Baseline 数据采集** | 三层 baseline 各跑一遍：(1) H100 上 GPU triton；(2) 910B 上 AscendC native（仅约 1/3 的题有 native 实现）；(3) 910B 上 torch eager | `results/h100_gpu_triton/` / `results/910b_ascendc/` / `results/910b_torch_eager/` JSON 数据 | 1 周（自动化 + 人工 sanity） |
| **D. NPU Triton Kernel 跑通 + 报告** | 工程师/agent 把每题的 NPU triton 实现写出来，跑 evaluator，出 `migration_score` 报告 | `results/910b_npu_triton/` + Markdown 报告 | 主体工作，4-8 周（取决于 triton-ascend 编译器实际能力） |
| **E. 开源化（可选）** | HuggingFace dataset 格式发布（参考 KernelBench HF dataset），社区贡献 NPU triton 实现 | HF dataset + leaderboard | 1-2 周 |

**建议先做 Phase A + B**（生成 task 文件 + harness），这是真正把"选题"变成"benchmark"的关键。其他 Phase 可以后续推进。

---

## 13. 与 KernelBench 的差距矩阵（v3 新增）

按 4 个维度系统对照，已补的标 ✅，本 v3 文档已说明但需后续实现的标 📋，仍是差距的标 ⚠️：

### 13.1 覆盖面

| 维度 | KernelBench | 本 benchmark v3 | 状态 |
|---|---|---|---|
| 任务总数 | 270 (L1 100 + L2 100 + L3 50 + L4 20) | 100 (L1 40 + L2 38 + L3 22) | ⚠️ 题数较少；但目标域更窄（仅 LLM 推理路径） |
| CV / Conv 类 | 38 个 conv 变体（standard / transposed / depthwise / 1D-3D / dilated / strided / grouped） | 仅 `causal_conv1d`（Mamba 路径） | ⚠️ **缺失**；vLLM/SGLang 主推理路径不用 conv（多模态 visual encoder 走 native PyTorch / Ascend cube）；如未来要扩多模态 visual prefill，建议补 conv2d 作 supplement |
| Standalone activation | 14 个（ReLU / GELU / Sigmoid / Tanh / Swish / SELU / HardSigmoid / Softplus / Softsign / ELU / HardTanh / Swish / Mish 等） | 仅 silu_and_mul / act_and_mul / gelu_and_mul（均 fused） | ⚠️ 缺独立激活基线；但 vLLM/SGLang 真实路径都是 fused activation |
| Norm | 8 个（BatchNorm / InstanceNorm / GroupNorm / RMSNorm / FrobeniusNorm / L1Norm / L2Norm / LayerNorm） | 4 个（RMSNorm / LayerNorm / 1pass / L2Norm 在 L2 fused） | ✅ 主流覆盖；BatchNorm/InstanceNorm/GroupNorm 在 LLM 推理罕用 |
| Reduction / cumsum | 7 个 reduction + 5 个 cumsum 变体 | 2 个（state_passing + chunk_cumsum）；prefix scan 主要在 Mamba 路径 | ✅ 已覆盖 LLM 关键场景 |
| Loss / 训练专用 | 6 个（MSE/CE/Huber/KL/Triplet/Hinge） | 0 | ✅ 故意排除（用户场景 = inference + RL rollout，loss 走 PyTorch） |
| 标准 attention | 1 (`ScaledDotProductAttention`) | 8（含 paged / chunked / MLA / ALiBi / SDPA / unified 2D/3D / grouped） | ✅ **本 benchmark 显著优于** KernelBench |
| Fused MoE / EP / MLA | 0 | 18 | ✅ **本 benchmark 独有** |
| State-space (Mamba/GDN/KDA) | 2 (`Mamba2ReturnY` / `Mamba2ReturnFinalState`，仅 L3 完整模型) | 20 (含 chunk/recurrent/scan 各级) | ✅ **本 benchmark 显著优于** |
| 量化 GEMM (W4A8/W8A8/AWQ) | 0 | 9 | ✅ **本 benchmark 独有** |
| LoRA | 0 | 6 | ✅ 本 benchmark 独有 |
| Sampling / spec decode / structured output | 0 | 10 | ✅ 本 benchmark 独有 |
| 完整模型架构（L3） | 50（ResNet/VGG/MobileNet/EfficientNet/ViT/Swin/RNN/LSTM/GRU/MiniGPT/UNet/NetVlad/Mamba2） | 0 | ⚠️ **缺失**；本 benchmark 测的是 kernel 级别，完整模型由 vLLM/SGLang serving benchmark 覆盖 |
| 完整 HF 模型（L4） | 20（GPT-Neo/OPT/BART/BigBird/Electra/Reformer/GPT-2 在不同 bs×seq）| 0 | ⚠️ 同上 |

**覆盖面结论**：在 **LLM 推理路径**深度上**显著优于** KernelBench；在 **CV / 完整模型架构**广度上不及 KernelBench。这是本 benchmark **定位为 Migration Benchmark** 的合理 trade-off。

### 13.2 科学性 / Protocol

| 维度 | KernelBench | 本 benchmark v3 | 状态 |
|---|---|---|---|
| 单一汇总指标 | `fast_p` | `migration_score(p)` | ✅ §10.1 已定义 |
| Correctness trial 数 | `n_correctness` (5 次) | 5 次 + 5 adversarial | ✅ §10.2 |
| 性能 warmup / trial | 10 / 100 | 10 / 100 | ✅ §10.3 |
| Timing 方法 | cuda_event / Triton do_bench / host_time | torch.npu.Event + host_time | ✅ §10.3 |
| 统计量 | 中位数（避免 outlier） | 中位数 + p99 + std | ✅ |
| Seed 固定 | ✅ | ✅ `seed = 42 + task_id × 1000 + trial_id` | ✅ |
| 输入分布规范 | torch.rand 标准 | torch.rand + mask 边界 + adversarial 5 case | ✅ |
| 容差按 dtype 分档 | ✅ | ✅ §10.2（FP16/INT8/FP8/recurrent 分档） | ✅ |
| Reward-hacking 防御 | static checker + adversarial 单元测试 | §10.2 列出 6 类 + flag > 3× speedup | ✅ |
| Frequency lock / 热降频 | ✅ | ✅ §10.3 | ✅ |

**科学性结论**：v3 已**对齐 KernelBench**。

### 13.3 可执行性

| 维度 | KernelBench | 本 benchmark v3 | 状态 |
|---|---|---|---|
| 每题独立 task 文件 | ✅ 单 Python 文件 | 📋 §11 给规范 + 模板，未生成文件 | 📋 Phase A |
| Evaluator harness | ✅ `src/kernelbench/eval.py` | 📋 §12 Phase B（可复用 KernelBench 90% 代码） | 📋 |
| 多 shape 配置 | KernelBench L4 有；L1-L3 单 shape | 📋 §11.2 模板含 small/medium/large 三档 + 3 adversarial | 📋 |
| Modal / 云上跑 | ✅ Modal 镜像 | ⚠️ 910B 需 Docker + CANN 8.5 + Atlas 800I A2 实机 | ⚠️ 硬件门槛 |
| HF dataset 化 | ✅ | 📋 §12 Phase E（可选） | 📋 |
| Baseline 数据 | ✅ H100/A100/L40S 等多卡 timing | 📋 §12 Phase C：H100 GPU triton + 910B AscendC + 910B torch eager 三层 | 📋 |

**可执行性结论**：v3 给出**完整规范 + 路线图**，但实际 task 文件 / harness / baseline **尚未生成**。

### 13.4 可对比性

| 维度 | KernelBench | 本 benchmark v3 | 状态 |
|---|---|---|---|
| 与其他 benchmark 关系说明 | 较弱 | ✅ §0 明确 Migration vs Generation 定位 | ✅ |
| Leaderboard / 公开排名 | 在做（v0.1 仍在迭代） | 📋 Phase E | 📋 |
| 跨硬件可比 | ✅（H100 / A100 / L40S 等） | 📋 当前仅 910B；未来可扩 910C / 950 | 📋 |
| 不同 LLM 实现可比 | ✅ | 📋 取决于 task 文件落地后 | 📋 |

---

## 14. v3 相对 v2 的变更总结

| 章节 | v2 | v3 | 变更 |
|---|---|---|---|
| §0 定位 | 无 | **新增**：Migration vs Generation Benchmark 对照 | 明确本 benchmark 与 KernelBench 的不同任务范畴 |
| §1-§9 选题主体 | 完整 | **完全保留** | v3 不改变 100 题选择 |
| §10 Protocol | 无（仅 §8 推进建议中零散提及）| **新增**：baseline / migration_score / correctness / performance / adversarial 完整 protocol | 对齐 KernelBench `eval.py` + `timing.py` + adversarial 测试规范 |
| §11 Task 文件规范 | 无 | **新增**：目录结构 + 单题模板 + 元数据字段 | 对齐 KernelBench 每题独立 Python 文件的格式 |
| §12 路线图 | 无 | **新增**：Phase A-E 实施路径 | 把"选题文档"变成"可运行 benchmark" |
| §13 差距矩阵 | 无 | **新增**：覆盖面 / 科学性 / 可执行性 / 可对比性 4 维度对照 | 让差距与价值都透明 |
| §14 变更总结 | 无 | **新增**：本表 | 历史追踪 |

**核心洞察**：v3 的最大价值不在改选题，而在**把本 benchmark 从"选题清单"升级为"可执行 benchmark 规范"**。100 题选择的合理性已在 v2 完成；v3 补足的是 KernelBench 那种 "科学性 + 可执行性 + 可对比性" 框架。

## 15. 文件清单

- 候选池 CSV：`/data1/code/agentic-rl/kernel/docs/npu/npu triton-ascend.csv`
- KernelBench 路径：`/data1/code/agentic-rl/kernel/KernelBench/`
- 历史版本：v1（合并 codex 初版）、v2（基于 910B 硬件能力深度优化）
- **v3 本文档**：`/data1/code/agentic-rl/kernel/docs/npu/benchmark_select.md`
- 关键外部参考：
  - KernelBench v0.1：https://github.com/ScalingIntelligence/KernelBench
  - KernelBench HF dataset：https://huggingface.co/datasets/ScalingIntelligence/KernelBench
  - triton-ascend 主仓：https://github.com/triton-lang/triton-ascend
  - triton-ascend 镜像：https://github.com/Ascend/triton-ascend
  - triton-ascend-ops 算子样例库：https://github.com/Ascend/triton-ascend-ops
  - 在线文档：https://ascend.github.io/triton-ascend/
  - vLLM-Ascend：https://github.com/vllm-project/vllm-ascend
  - sgl-kernel-npu：https://github.com/sgl-project/sgl-kernel-npu

---
