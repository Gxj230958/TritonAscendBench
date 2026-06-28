# Kernel 选型方案 — TritonAscendBench 题库

## 1. 概览

题库收录 **101** 个 kernel，分三级：**L1 36 · L2 35 · L3 30**。全部取自 **vLLM / SGLang 现网生产推理路径**中的真实 `@triton.jit` kernel —— 迁移基准的源对象是生产算子本身，而非为基准构造的合成算子。

| 维度 | 构成 |
|------|------|
| 来源仓 | vLLM 61 · SGLang 40 |
| 计算类型 | VV（向量 / 逐元素）66 · CV（含 cube matmul）35 |
| 迁移难度 | medium 71 · high 24 · low 6 |

## 2. 选型原则

1. **生产路径优先**：只收 vLLM / SGLang 推理路径在用的 kernel；训练后向（`_bwd_*` / `*_backward`）一律不收（目标场景是 agentic RL rollout / 推理）。
2. **分级分层**：按迁移复杂度分 L1 / L2 / L3（§3），覆盖从单算子到完整子系统。
3. **910B 硬件适配**：依 910B 硬件 dtype 矩阵与 `triton-ascend` 能力裁剪（§5）——无法在 910B 真实测量的算子（FP8 cube 算术等）不进题库。
4. **去同质冗余**：同一 fused op 的 stage 变体、形态高度重复的量化算子只保留代表。
5. **覆盖完整性**：按功能族做覆盖（§4），关键路径成对收录（GELU vs SiLU、ALiBi vs RoPE、quantize vs dequant、vLLM vs SGLang 双源状态合并）。

## 3. 分级定义

| 级别 | 定义 | 典型形态 |
|------|------|---------|
| **L1** | 基础单算子，纯 VV / 单 stage | bias / norm / 激活 / sampling / KV 索引 / 量化 quant |
| **L2** | 复合 fused / CV 混合 | LoRA matmul / W8A8 INT8 GEMM / MoE 路由+permute / Mamba conv1d / GDN gating |
| **L3** | 完整子系统 / 高复杂度 | 完整 Flash/Paged/Unified Attention · Fused MoE · Mamba/GDN/KDA chunk pipeline |

难度随级别递增：L1 以 medium/low 为主，L3 几乎全 high —— chunk recurrent + online-softmax fused 是 `triton-ascend` 迁移的真正难关（§5）。

## 4. 功能覆盖

题库按算子类别覆盖推理全路径（类别取自各题 `meta.yaml`）：

| 类别 | 题数 | 类别 | 题数 |
|------|-----:|------|-----:|
| moe（MoE / EP / 路由 / Cutlass W4A8） | 20 | elementwise（激活 / 状态合并） | 6 |
| attention（MLA / paged / Flash） | 16 | normalization（含 norm / residual fusion） | 7 |
| ssm（Mamba / GDN / GLA / KDA） | 18 | other | 4 |
| gemm（Cube GEMM / LoRA matmul） | 9 | convolution（causal conv1d） | 3 |
| quantization（量化 / dequant / 量化 matmul） | 9 | spec_decode | 1 |
| sampling（spec decode / logits / structured） | 8 | | |

关键成对覆盖：GELU↔SiLU 激活、ALiBi↔RoPE attention、quantize↔dequant KV cache、W4A8 Cutlass MoE 前/后 reorder 成对、vLLM↔SGLang 双源 chunked-prefill 状态合并。

## 5. 910B 硬件适配准则

选型受 910B（Atlas A2 系列）硬件与 `triton-ascend` 能力直接约束。

**dtype**：FP16 / BF16 / INT8 在 cube + vector 均原生（主路径与 W8A8 / W4A8 量化路径）；**FP8 无 cube/vector 算术**（仅存储 + `tt.fp_to_fp` 软件 cast）；FP4 / NF4 仅存储。因此 **FP8 cube 算术依赖型算子不进题库**（性能无法在 910B 真实测量），仅保留一个 fused FP8 软件 cast 算子作正确性 / stress 代表。

**`triton-ascend` 关键约束**：

| 约束 | 状态 | 选型影响 |
|------|------|---------|
| UB（Unified Buffer） | 192 KB / AI core | Flash / MLA 的 BLOCK_M×BLOCK_K×dtype 严格受限 |
| `tl.dot`（cube matmul） | 内积维对齐 16；BF16 有限制 | group GEMM 须 padding 到 16 倍数 |
| `tl.atomic_add` | BF16 / FP16 不支持 | MoE combine / scatter_add 可能降级为多 stage VV |
| 异步 copy / Pipeline / TMA | 不支持 | 双缓冲 / pipeline 须手动拆 stage |
| `tl.scan` / `tl.sort` 非连续访存 | 受版本限制 | Mamba / FLA chunk cumsum 依赖较新 release |

## 6. 排除准则

下列类别不进题库：

- **FP8 cube 算术依赖**（`scaled_mm` / `fp8_gemm_group` / `_per_token_group_quant_fp8` 等）：910B Cube 无 FP8 算术指令，软件模拟不反映硬件能力。
- **FP4 / NF4 / MXFP4 cube 类**：910B 仅支持其存储，无 cube 算术。
- **A5 平台独有指令算子**：910B 不支持。
- **训练后向**（`_bwd_*` / `*_backward` / `chunk_bwd_*`）：目标场景是推理 / rollout。
- **FBGEMM 全系列**（jagged FA / fp8 row-block / jagged 元数据）：与 vLLM / SGLang 推理路径无关，整批不收；变长 attention 由 vLLM/SGLang 在途算子覆盖。
- **未公开源码的私有算子**：不可验证上游来源。
- **同族 stage 重复**（`moe_align_block_size_stage*`、`l2norm_fwd_kernel*` 等）：同一 fused op 的 stage 变体只取代表。
- **调试 / launch helper**（`_num_nans_kernel` / `_compute_pid` 等）：对算子迁移代表性不足。

## 7. 完整题目清单

> 字段取自各题 `tasks/level*/<id>/meta.yaml`（task_id / 仓 / VV·CV / 类别 / 迁移难度 / 上游路径）。

### L1

| task_id | 仓 | 类型 | 类别 | 迁移难度 | 上游路径 |
|---------|----|------|------|---------|----------|
| `L1_01_bias` | vLLM | VV | sampling | medium | `vllm/v1/worker/gpu/sample/logit_bias.py` |
| `L1_02_compute_slot_mappings` | vLLM | VV | attention | medium | ``vllm/v1/worker/utils.py`` |
| `L1_03_convert_req_index_to_global_index` | vLLM | VV | convolution | medium | ``vllm/v1/worker/`` |
| `L1_04_copy_page_indices` | vLLM | VV | attention | low | `vLLM v1 attention` |
| `L1_05_gather_block_tables` | vLLM | VV | moe | medium | ``vllm/v1/attention/ops/`` |
| `L1_06_reshape_and_cache_kernel_flash` | vLLM | VV | attention | medium | ``vllm/v1/attention/ops/triton_reshape_and_cache_flash.py`` |
| `L1_07_dequantize_k_cache_fast` | SGLang | VV | quantization | medium | ``sglang/srt/layers/attention/`` |
| `L1_08_dequantize_k_cache_paged` | SGLang | VV | attention | medium | ``sglang/srt/layers/attention/`` |
| `L1_09_quantize_k_cache_fast` | SGLang | VV | quantization | medium | ``sglang/srt/layers/attention/`` |
| `L1_10_per_token_group_quant_int8` | vLLM | VV | quantization | medium | ``vllm/model_executor/layers/quantization/utils/int8_utils.py`` |
| `L1_11_awq_dequantize` | vLLM | VV | quantization | medium | ``vllm/model_executor/layers/quantization/awq_triton.py`` |
| `L1_12_rms_norm` | vLLM | VV | norm | medium | `vllm/model_executor/layers/batch_invariant.py` |
| `L1_13_fused_softcap` | SGLang | VV | elementwise | low | ``sglang/srt/layers/logits_processor.py`` |
| `L1_14_merge_attn_states` | vLLM | VV | attention | medium | ``vllm/v1/attention/ops/triton_merge_attn_states.py`` |
| `L1_15_merge_state` | SGLang | VV | elementwise | medium | ``sglang/srt/layers/attention/triton_ops/merge_state.py`` |
| `L1_16_apply_grammar_bitmask` | vLLM | VV | sampling | medium | ``vllm/v1/worker/gpu/structured_outputs.py`` |
| `L1_17_prepare_eagle_inputs` | vLLM | VV | spec_decode | medium | `vLLM spec_decode` |
| `L1_18_act_quant` | SGLang | VV | quantization | low | `python/sglang/srt/layers/attention/nsa/triton_kernel.py` |
| `L1_19_triton_mrope_forward` | vLLM | VV | attention | medium | `vLLM mrope` |
| `L1_20_create_flashinfer_kv_indices_triton` | SGLang | VV | attention | medium | ``sglang/srt/layers/attention/`` |
| `L1_21_create_flashmla_kv_indices_triton` | SGLang | VV | attention | medium | ``sglang/srt/layers/attention/`` |
| `L1_22_transform_index_page_table_decode` | SGLang | VV | attention | medium | ``sglang/srt/mem_cache/`` |
| `L1_23_compute_src2dst_triton` | SGLang | VV | moe | medium | ``sglang/srt/layers/moe/ep_moe/kernels.py`` |
| `L1_24_compute_seg_indptr_triton` | SGLang | VV | moe | medium | ``sglang/srt/layers/moe/ep_moe/kernels.py`` |
| `L1_25_compute_problem_sizes_w4a8` | SGLang | VV | quantization | medium | ``sglang/srt/layers/moe/cutlass_w4a8_moe.py`` |
| `L1_26_state_passing_fwd` | vLLM | VV | ssm | medium | ``vllm/model_executor/layers/mamba/ops/`` |
| `L1_27_chunk_cumsum_fwd` | vLLM | VV | ssm | medium | `vllm/model_executor/layers/mamba/ops/ssd_chunk_state.py` |
| `L1_28_fused_rmsnorm` | SGLang | VV | normalization | medium | ``sglang/srt/layers/elementwise.py`` |
| `L1_29_fused_gdn_gating` | vLLM | VV | ssm | medium | ``vllm/model_executor/layers/fla/ops/`` |
| `L1_30_deepep_permute_triton` | SGLang | VV | moe | medium | ``sglang/srt/layers/moe/ep_moe/kernels.py`` |
| `L1_31_deepep_post_reorder_triton` | SGLang | VV | moe | medium | ``sglang/srt/layers/moe/ep_moe/kernels.py`` |
| `L1_32_experts_combine` | SGLang | VV | moe | medium | ``sglang/srt/layers/moe/`` |
| `L1_33_pre_reorder_triton_kernel_for_cutlass_moe` | SGLang | VV | moe | medium | ``sglang/srt/layers/moe/cutlass_w4a8_moe.py`` |
| `L1_34_post_reorder_triton_kernel_for_cutlass_moe` | SGLang | VV | moe | medium | ``sglang/srt/layers/moe/cutlass_w4a8_moe.py`` |
| `L1_35_concat_and_cast_mha_k` | SGLang | VV | elementwise | medium | ``sglang/srt/mem_cache/`` |
| `L1_36_gemma_qkv_rmsnorm` | SGLang | VV | normalization | medium | ``sglang/srt/layers/gemma4_fused_ops.py`` |

### L2

| task_id | 仓 | 类型 | 类别 | 迁移难度 | 上游路径 |
|---------|----|------|------|---------|----------|
| `L2_01_layer_norm_fwd` | vLLM | VV | normalization | medium | `vLLM` |
| `L2_02_layer_norm_fwd_1pass` | vLLM | VV | normalization | medium | ``vllm/model_executor/layers/fla/`` |
| `L2_03_act_and_mul` | SGLang | VV | elementwise | low | ``sglang/srt/layers/activation.py`` |
| `L2_04_silu_and_mul` | SGLang | VV | elementwise | low | `python/sglang/srt/layers/elementwise.py` |
| `L2_05_gelu_and_mul` | SGLang | VV | elementwise | medium | ``sglang/srt/layers/activation.py`` |
| `L2_06_topk_log_softmax` | vLLM | VV | sampling | medium | `vLLM sample` |
| `L2_07_log_softmax` | vLLM | VV | sampling | medium | `vLLM sample` |
| `L2_08_gumbel_sample` | vLLM | VV | sampling | medium | ``vllm/v1/worker/gpu/sample/gumbel.py`` |
| `L2_09_min_p` | vLLM | VV | sampling | medium | `vLLM sample` |
| `L2_10_penalties_and_temperature` | vLLM | VV | sampling | low | `vllm/v1/worker/gpu/sample/penalties.py` |
| `L2_11_rejection_sample` | vLLM | VV | sampling | medium | `vllm/v1/sample/rejection_sampler.py` |
| `L2_12_pack_bitmatrix` | vLLM | VV | other | medium | ``vllm/v1/worker/gpu/structured_outputs.py`` |
| `L2_13_silu_mul_fp8_quant_deep_gemm` | vLLM | VV | moe | medium | ``vllm/model_executor/layers/fused_moe/experts/batched_deep_gemm_moe.py`` |
| `L2_14_bmm_chunk_fwd` | vLLM | CV | gemm | medium | ``vllm/model_executor/layers/mamba/ops/`` |
| `L2_15_chunked_lora_shrink` | SGLang | CV | gemm | medium | ``sglang/srt/lora/`` |
| `L2_16_sgemm_lora_a` | SGLang | CV | gemm | medium | ``sglang/srt/lora/`` |
| `L2_17_w8a8_block_int8_matmul` | vLLM | CV | moe | high | `vllm/model_executor/layers/quantization/utils/int8_utils.py` |
| `L2_18_w8a8_triton_block_scaled_mm` | vLLM | CV | quantization | medium | ``vllm/model_executor/layers/quantization/compressed_tensors/triton_scaled_mm.py`` |
| `L2_19_awq_gemm` | vLLM | CV | gemm | medium | ``vllm/model_executor/layers/quantization/awq_triton.py`` |
| `L2_20_bmm` | vLLM | CV | gemm | medium | `vLLM` |
| `L2_21_matmul_kernel_persistent` | vLLM | CV | gemm | medium | ``vllm/model_executor/layers/batch_invariant.py`` |
| `L2_22_fused_dual_residual_rmsnorm` | SGLang | VV | normalization | medium | ``sglang/srt/layers/elementwise.py`` |
| `L2_23_layer_norm_gated_fwd` | vLLM | VV | normalization | medium | ``vllm/model_executor/layers/fla/ops/`` |
| `L2_24_linear_attn_decode` | vLLM | VV | attention | medium | ``vllm/model_executor/layers/lightning_attn.py`` |
| `L2_25_fwd_kernel_ep_gather` | vLLM | VV | moe | medium | ``sglang/srt/layers/moe/ep_moe/kernels.py`` |
| `L2_26_fwd_kernel_ep_scatter_1` | vLLM | VV | moe | medium | ``sglang/srt/layers/moe/ep_moe/kernels.py`` |
| `L2_27_fused_moe_router_cudacore` | SGLang | VV | moe | medium | ``sglang/srt/layers/moe/router.py`` |
| `L2_28_moe_align_block_size_stage2` | SGLang | VV | moe | medium | ``sglang/srt/layers/moe/ep_moe/kernels.py`` |
| `L2_29_silu_mul_static_tensorwise_quant_triton_kernel_for_cutlass_moe` | SGLang | VV | moe | medium | ``sglang/srt/layers/moe/cutlass_w4a8_moe.py`` |
| `L2_30_silu_and_mul_post_quant` | SGLang | VV | quantization | medium | ``sglang/srt/layers/activation.py`` |
| `L2_31_scaled_mm` | vLLM | CV | quantization | medium | ``vllm/model_executor/layers/quantization/compressed_tensors/triton_scaled_mm.py`` |
| `L2_32_correct_attn_cp_out` | vLLM | VV | attention | high | `vllm/v1/attention/ops/common.py` |
| `L2_33_moe_mmk` | vLLM | CV | moe | high | ``vllm/model_executor/layers/fused_moe/`` |
| `L2_34_solve_tril_16x16` | vLLM | VV | ssm | medium | ``vllm/model_executor/layers/fla/ops/solve_tril.py`` |
| `L2_35_chunk_local_cumsum_vector` | vLLM | CV | ssm | high | ``vllm/model_executor/layers/fla/ops/`` |

### L3

| task_id | 仓 | 类型 | 类别 | 迁移难度 | 上游路径 |
|---------|----|------|------|---------|----------|
| `L3_01_chunk_scan_fwd` | vLLM | CV | ssm | medium | ``vllm/model_executor/layers/mamba/ops/ssd_chunk_scan.py`` |
| `L3_02_chunk_state_fwd` | vLLM | CV | ssm | medium | ``vllm/model_executor/layers/mamba/ops/ssd_chunk_state.py`` |
| `L3_03_chunked_lora_expand` | SGLang | CV | gemm | medium | ``sglang/srt/lora/`` |
| `L3_04_qkv_lora_b` | SGLang | CV | attention | medium | ``sglang/srt/lora/`` |
| `L3_05_gate_up_lora_b` | SGLang | CV | gemm | medium | ``sglang/srt/lora/`` |
| `L3_06_sgemm_lora_b` | SGLang | CV | gemm | medium | ``sglang/srt/lora/`` |
| `L3_07_fused_sigmoid_gating_delta_rule_update` | SGLang | VV | ssm | medium | ``sglang/srt/layers/attention/fla/`` |
| `L3_08_causal_conv1d_update` | vLLM | VV | convolution | medium | ``vllm/model_executor/layers/mamba/ops/causal_conv1d.py`` |
| `L3_09_causal_conv1d_fwd` | vLLM | VV | convolution | medium | ``vllm/model_executor/layers/mamba/ops/causal_conv1d.py`` |
| `L3_10_fused_moe_router_tensorcore` | SGLang | CV | moe | high | ``sglang/srt/layers/moe/router.py`` |
| `L3_11_fwd_grouped_kernel_stage1_rope` | SGLang | CV | attention | high | ``sglang/srt/layers/attention/triton_ops/rocm_mla_decode_rope.py`` |
| `L3_12_selective_state_update` | vLLM | VV | ssm | medium | ``vllm/model_executor/layers/mamba/ops/mamba_ssm.py`` |
| `L3_13_kernel_unified_attention_3d` | vLLM | CV | attention | high | `vllm/v1/attention/ops/triton_unified_attention.py` |
| `L3_14_kernel_paged_attention_2d` | vLLM | CV | attention | high | `vllm/v1/attention/ops/triton_unified_attention.py` |
| `L3_15_chunk_state_varlen` | SGLang | CV | ssm | high | `python/sglang/srt/layers/attention/mamba/ops/ssd_chunk_state.py` |
| `L3_16_fwd` | SGLang | CV | attention | high | `python/sglang/srt/layers/attention/triton_ops/prefill_attention.py` |
| `L3_17_fwd_kernel_alibi` | vLLM | CV | other | high | `python/sglang/srt/layers/attention/triton_ops/prefill_attention.py` |
| `L3_18_fwd_grouped_kernel_stage1` | vLLM | CV | other | high | ``sglang/srt/layers/attention/triton_ops/decode_attention.py`` |
| `L3_19_fused_moe` | vLLM | CV | moe | high | ``vllm/model_executor/layers/fused_moe/fused_moe.py`` |
| `L3_20_fused_moe_kernel_gptq_awq` | vLLM | CV | moe | high | ``vllm/model_executor/layers/fused_moe/fused_moe.py`` |
| `L3_21_expert_triton` | vLLM | CV | moe | high | ``vllm/model_executor/layers/fused_moe/`` |
| `L3_22_chunk_gated_delta_rule_fwd_kernel_h_blockdim64` | vLLM | CV | ssm | high | ``vllm/model_executor/layers/fla/ops/chunk.py`` |
| `L3_23_fused_recurrent_gated_delta_rule_fwd` | vLLM | VV | ssm | high | ``vllm/model_executor/layers/fla/ops/fused_recurrent.py`` |
| `L3_24_fused_recurrent_gated_delta_rule_update_fwd` | SGLang | VV | ssm | high | ``sglang/srt/layers/attention/fla/`` |
| `L3_25_chunk_fwd_kernel_o` | vLLM | CV | ssm | high | ``vllm/model_executor/layers/fla/ops/chunk.py`` |
| `L3_26_chunk_gla_fwd_kernel_o` | vLLM | CV | ssm | high | ``vllm/model_executor/layers/fla/ops/`` |
| `L3_27_chunk_scaled_dot_kkt_fwd` | vLLM | CV | ssm | high | ``vllm/model_executor/layers/fla/ops/kda.py`` |
| `L3_28_chunk_kda_scaled_dot_kkt_fwd_kernel_intra_sub_inter` | vLLM | CV | ssm | high | ``vllm/model_executor/layers/fla/ops/kda.py`` |
| `L3_29_recompute_w_u_fwd` | vLLM | CV | other | high | ``vllm/model_executor/layers/fla/ops/kda.py`` |
| `L3_30_merge_16x16_to_32x32_inverse` | vLLM | CV | ssm | high | ``vllm/model_executor/layers/fla/ops/` + `sglang/srt/layers/attention/fla/`` |
