---
date: 2026-05-26
status: provenance-first truth inventory for 100-task corpus
based-on: HEAD f588024 (post-revert clean state, 11 SKIPPED_BY_DESIGN commits rolled back)
audience: bench team — input for family-lane strategy + E→D execution
scope: read-only provenance audit (no code/yaml/meta changes)
---

# Truth Inventory — Corpus Freeze + Family Lane Map

## Executive Summary

corpus 真实 freeze 完成度（基于 resolver verdict + meta 一致性 + 源码可定位性）:

| Maturity | Count | % | 含义 |
|---|---|---|---|
| **A.frozen-clean** | 0 | 0% | 完全 release-ready (gpu_triton 真函数 + SPDX header + 真 @triton.jit + simple_pytorch_ok oracle + resolver passing) |
| **B.frozen-partial** | 5 | 5% | resolver 已确认实际 entrypoint + oracle-consistent;但 meta state 不一致（SPDX 缺失、stale placeholder 残留、或 meta gpu_triton 指错地方）|
| **C.placeholder** | 95 | 95% | resolver 未通过；但**真函数在 `infer/{vllm,sglang}/` 100% 可定位**——仅是工艺连接缺失（shapes 不全、入参契约不符、placeholder 未替换）|
| **D.unresolvable** | 0 | 0% | 源码完全找不到，需 escalate — **本次 audit 零命中** |

**核心 finding（颠覆性）**: 100/100 task 的 upstream truth 真函数都能在 `infer/` 源码树里找到（42 个在 `meta.upstream.path` 指定位置，58 个在该 repo 其他文件里）。corpus 不是 "unresolvable"，是 **"已有所有真值材料，但 meta/oracle/contract 三层契约尚未对齐"**。所谓 BLOCKED 的 95 task 都是**单点工艺缺失**——freeze workflow 已经把源码 vendor 了进来，但 oracle 输入契约/shapes 表/oracle_pytorch.py 三选一缺失，导致 resolver 无法验证。

family lane 分布及优先级（按 oracle 复杂度排序）:

| 优先级 | Lane | Task 数 | 主导 oracle tier | 单 lane 工作量估计 |
|---|---|---|---|---|
| **P0 (最易)** | gemm | 3 | 2 simple + 1 formula | 数小时 |
| **P0** | block_table | 5 | 5 formula | 0.5-1 天 |
| **P0** | flashinfer_indices | 5 | 5 formula | 0.5-1 天 |
| **P0** | lora | 6 | 6 formula | 1 天 |
| **P1 (中等)** | elementwise | 13 | 5 simple + 5 formula + 3 upstream_req | 1-2 天 |
| **P1** | ssm_mamba | 11 | 10 formula + 1 upstream_req | 2 天 |
| **P1** | moe | 16 | 1 simple + 12 formula + 3 upstream_req | 2-3 天 |
| **P1** | quant | 6 | 5 formula + 1 upstream_req | 1-2 天 |
| **P1** | sampling | 6 | 5 formula + 1 upstream_req | 1-2 天 |
| **P2 (较难)** | delta_rule | 10 | 7 formula + 3 upstream_req | 2-3 天 |
| **P2** | paged_attention | 13 | 9 formula + 4 upstream_req | 3-4 天 |
| **P3 (最难)** | norm | 6 | 3 formula + 3 upstream_req | 1-2 天（但 oracle 设计难度高，layer_norm 类需要小心数值稳定） |

**战略建议精炼**:
- **不要 escalate / drop 任何 task**：D.unresolvable 为 0，corpus 是完整可拯救的
- **从 gemm + block_table + flashinfer_indices + lora 这 4 个 P0 lane 起手**：oracle tier 全在 `formula_exact_manual` 或 `simple_pytorch_ok`，符合 ≥80% PyTorch oracle 路径，单 lane 完成可解锁 19/100 task（19%）
- **真正卡 GPU 机的只有 5 个 UNRUN**：L1_01, L1_12, L1_16, L1_26, L1_34 — 一旦上 GPU 即可全部走过 UNRUN→PASS。剩 95 个 BLOCKED 在 dev 机上就能修
- **resolver 主导失败模式**：`shapes-incomplete` (47/95 ≈ 50%) > `upstream-not-invocable` (40/95 ≈ 42%) > `oracle-mismatch` (6/95) > `smoke-fail` (2/95)。**优先攻 shapes 不全**（最低悬挂的果子）

---

## 1. 全 corpus statistics

### 1.1 Maturity × Family Lane 矩阵

每个 cell 显示该 lane × maturity 组合的 task 数。最右两列展示 oracle tier 分布。

| family | total | A | B | C | D | simple_pytorch_ok | formula_exact_manual | upstream_oracle_required |
|---|---|---|---|---|---|---|---|---|
| **elementwise** | 13 | 0 | 3 | 10 | 0 | 5 | 5 | 3 |
| **norm** | 6 | 0 | 1 | 5 | 0 | 0 | 3 | 3 |
| **sampling** | 6 | 0 | 0 | 6 | 0 | 0 | 5 | 1 |
| **block_table** | 5 | 0 | 1 | 4 | 0 | 0 | 5 | 0 |
| **flashinfer_indices** | 5 | 0 | 0 | 5 | 0 | 0 | 5 | 0 |
| **paged_attention** | 13 | 0 | 0 | 13 | 0 | 0 | 9 | 4 |
| **lora** | 6 | 0 | 0 | 6 | 0 | 0 | 6 | 0 |
| **gemm** | 3 | 0 | 0 | 3 | 0 | 2 | 1 | 0 |
| **quant** | 6 | 0 | 0 | 6 | 0 | 0 | 5 | 1 |
| **moe** | 16 | 0 | 0 | 16 | 0 | 1 | 12 | 3 |
| **ssm_mamba** | 11 | 0 | 0 | 11 | 0 | 0 | 10 | 1 |
| **delta_rule** | 10 | 0 | 0 | 10 | 0 | 0 | 7 | 3 |
| **TOTAL** | 100 | 0 | 5 | 95 | 0 | 8 | 73 | 19 |

### 1.2 Resolver verdict 子分类（来自 `results/resolve_all.json` HEAD f588024）

| status subcategory | count | 含义 |
|---|---|---|
| BLOCKED(shapes-incomplete) | 47 | resolver 找到 entrypoint + oracle 一致，但 shapes 表只对一部分 shape pass，剩下的 shape 缺乏覆盖 |
| BLOCKED(upstream-not-invocable-here) | 40 | resolver 找到函数名，但在本机 invoke 失败（缺包/参数签名不符/InterpreterError）|
| BLOCKED(oracle-mismatch) | 6 | resolver invoked successfully，但输出与 oracle_pytorch.py 期望值不一致 |
| UNRUN(needs-gpu) | 5 | 全部检查通过 (oracle-consistent 等)；只缺 GPU 真跑 |
| BLOCKED(smoke-fail) | 2 | CPU 端 oracle pass 但 S2-smoke (TRITON_INTERPRET) 实跑 fail |

### 1.3 Upstream truth 可定位性

| 分布 | count |
|---|---|
| `meta.upstream.path` 指向真实存在的**文件** | 58/100 |
| `meta.upstream.path` 指向真实存在的**目录** | 28/100 |
| `meta.upstream.path` 完全失效 | 14/100 |
| Function findable at meta path | 42/100 |
| Function findable elsewhere in same repo (corpus search) | 58/100 |
| Function NOT findable anywhere | **0/100** |

**[confirmed]** 关键意义：`meta.upstream.path` 字段有效率仅 58%，但函数最终都能找到——意味着 freeze workflow 已经把真函数拉进来了，**只是 meta 元数据没维护好**。

### 1.4 Meta 不一致清单（5 个 B-class 中的 anomaly）

| task | meta.oracle.gpu_triton | resolver-discovered entrypoint | meta_has_stale_placeholder_note | 说明 |
|---|---|---|---|---|
| L1_01_bias | `kernel.py::apply_logit_bias` | apply_logit_bias | no | clean |
| L1_12_rms_norm | `kernel.py::rms_norm` | rms_norm | no | clean |
| L1_16_silu_and_mul | `kernel.py::silu_and_mul_triton` | silu_and_mul_triton | no | clean |
| L1_26_merge_state | `kernel.py::merge_state_triton` | merge_state_triton | **YES (stale)** | meta `notes` 字段残留 "kernel.py is a placeholder" — **meta 文案漂移** |
| L1_34_transform_index_page_table_decode | `kernel.py::placeholder` | **transform_index_page_table_decode_fast** | YES | **meta gpu_triton 完全错** — 实际 entrypoint 是 `_fast` suffix，但 meta 没更新；resolver auto-discovered |

---

## 2. Per-Task Inventory

`mat` 列 = maturity（A/B/C/D），`func` 列 = function findability（`meta` = 在 meta 指定文件找到，`elsewhere` = 在同 repo 其他文件找到，`no` = 找不到）。

| task_id | family | mat | gpu_triton (meta) | upstream | oracle_tier | verdict | func |
|---|---|---|---|---|---|---|---|
| `L1_01_bias` | elementwise | B | `kernel.py::apply_logit_bias` | vllm | simple_pytorch_ok | UNRUN(gpu) | meta |
| `L1_02_compute_slot_mappings` | block_table | C | `kernel.py::placeholder` | vllm | formula_exact_manual | BLOCKED:upstream-inv | elsewhere |
| `L1_03_convert_req_index_to_global_index` | block_table | C | `kernel.py::placeholder` | vllm | formula_exact_manual | BLOCKED:upstream-inv | elsewhere |
| `L1_04_copy_page_indices` | block_table | C | `kernel.py::placeholder` | vllm | formula_exact_manual | BLOCKED:upstream-inv | elsewhere |
| `L1_05_gather_block_tables` | block_table | C | `kernel.py::placeholder` | vllm | formula_exact_manual | BLOCKED:upstream-inv | elsewhere |
| `L1_06_reshape_and_cache_kernel_flash` | paged_attention | C | `kernel.py::placeholder` | vllm | formula_exact_manual | BLOCKED:shapes-inc | meta |
| `L1_07_dequantize_k_cache_fast` | paged_attention | C | `kernel.py::placeholder` | sglang | formula_exact_manual | BLOCKED:shapes-inc | elsewhere |
| `L1_08_dequantize_k_cache_paged` | paged_attention | C | `kernel.py::placeholder` | sglang | formula_exact_manual | BLOCKED:shapes-inc | elsewhere |
| `L1_09_quantize_k_cache_fast` | paged_attention | C | `kernel.py::placeholder` | sglang | formula_exact_manual | BLOCKED:oracle-mis | elsewhere |
| `L1_10_per_token_group_quant_int8` | quant | C | `kernel.py::placeholder` | vllm | formula_exact_manual | BLOCKED:shapes-inc | meta |
| `L1_11_awq_dequantize` | quant | C | `kernel.py::placeholder` | vllm | formula_exact_manual | BLOCKED:shapes-inc | meta |
| `L1_12_rms_norm` | norm | B | `kernel.py::rms_norm` | vllm | formula_exact_manual | UNRUN(gpu) | meta |
| `L1_13_layer_norm_fwd` | norm | C | `kernel.py::placeholder` | vllm | upstream_oracle_required | BLOCKED:upstream-inv | elsewhere |
| `L1_14_layer_norm_fwd_1pass` | norm | C | `kernel.py::placeholder` | vllm | upstream_oracle_required | BLOCKED:oracle-mis | elsewhere |
| `L1_15_act_and_mul` | elementwise | C | `kernel.py::placeholder` | sglang | upstream_oracle_required | BLOCKED:shapes-inc | elsewhere |
| `L1_16_silu_and_mul` | elementwise | B | `kernel.py::silu_and_mul_triton` | sglang | simple_pytorch_ok | UNRUN(gpu) | elsewhere |
| `L1_17_gelu_and_mul` | elementwise | C | `kernel.py::placeholder` | sglang | simple_pytorch_ok | BLOCKED:shapes-inc | elsewhere |
| `L1_18_fused_softcap` | elementwise | C | `kernel.py::placeholder` | sglang | simple_pytorch_ok | BLOCKED:shapes-inc | meta |
| `L1_19_topk_log_softmax` | sampling | C | `kernel.py::placeholder` | vllm | formula_exact_manual | BLOCKED:upstream-inv | elsewhere |
| `L1_20_log_softmax` | elementwise | C | `kernel.py::placeholder` | vllm | simple_pytorch_ok | BLOCKED:shapes-inc | elsewhere |
| `L1_21_gumbel_sample` | sampling | C | `kernel.py::placeholder` | vllm | upstream_oracle_required | BLOCKED:shapes-inc | meta |
| `L1_22_min_p` | sampling | C | `kernel.py::placeholder` | vllm | formula_exact_manual | BLOCKED:oracle-mis | elsewhere |
| `L1_23_penalties_and_temperature` | sampling | C | `kernel.py::placeholder` | vllm | formula_exact_manual | BLOCKED:upstream-inv | meta |
| `L1_24_rejection_sample` | sampling | C | `kernel.py::placeholder` | vllm | formula_exact_manual | BLOCKED:upstream-inv | meta |
| `L1_25_merge_attn_states` | elementwise | C | `kernel.py::placeholder` | vllm | upstream_oracle_required | BLOCKED:oracle-mis | meta |
| `L1_26_merge_state` | elementwise | B | `kernel.py::merge_state_triton` | sglang | upstream_oracle_required | UNRUN(gpu) | meta |
| `L1_27_apply_grammar_bitmask` | elementwise | C | `kernel.py::placeholder` | vllm | formula_exact_manual | BLOCKED:upstream-inv | meta |
| `L1_28_pack_bitmatrix` | elementwise | C | `kernel.py::placeholder` | vllm | formula_exact_manual | BLOCKED:shapes-inc | elsewhere |
| `L1_29_prepare_eagle_inputs` | elementwise | C | `kernel.py::placeholder` | vllm | formula_exact_manual | BLOCKED:shapes-inc | elsewhere |
| `L1_30_act_quant` | quant | C | `kernel.py::placeholder` | sglang | formula_exact_manual | BLOCKED:shapes-inc | elsewhere |
| `L1_31_triton_mrope_forward` | elementwise | C | `kernel.py::placeholder` | vllm | formula_exact_manual | BLOCKED:upstream-inv | elsewhere |
| `L1_32_create_flashinfer_kv_indices_triton` | flashinfer_indices | C | `kernel.py::placeholder` | sglang | formula_exact_manual | BLOCKED:shapes-inc | elsewhere |
| `L1_33_create_flashmla_kv_indices_triton` | flashinfer_indices | C | `kernel.py::placeholder` | sglang | formula_exact_manual | BLOCKED:shapes-inc | elsewhere |
| `L1_34_transform_index_page_table_decode` | block_table | B | `kernel.py::placeholder` | sglang | formula_exact_manual | UNRUN(gpu) | elsewhere |
| `L1_35_compute_src2dst_triton` | flashinfer_indices | C | `kernel.py::placeholder` | sglang | formula_exact_manual | BLOCKED:shapes-inc | meta |
| `L1_36_compute_seg_indptr_triton` | flashinfer_indices | C | `kernel.py::placeholder` | sglang | formula_exact_manual | BLOCKED:upstream-inv | meta |
| `L1_37_compute_problem_sizes_w4a8` | flashinfer_indices | C | `kernel.py::placeholder` | sglang | formula_exact_manual | BLOCKED:upstream-inv | elsewhere |
| `L1_38_silu_mul_fp8_quant_deep_gemm` | quant | C | `kernel.py::placeholder` | vllm | formula_exact_manual | BLOCKED:upstream-inv | elsewhere |
| `L1_39_state_passing_fwd` | ssm_mamba | C | `kernel.py::placeholder` | vllm | formula_exact_manual | BLOCKED:shapes-inc | elsewhere |
| `L1_40_chunk_cumsum_fwd` | ssm_mamba | C | `kernel.py::placeholder` | vllm | formula_exact_manual | BLOCKED:shapes-inc | elsewhere |
| `L2_01_bmm_chunk_fwd` | ssm_mamba | C | `kernel.py::placeholder` | vllm | formula_exact_manual | BLOCKED:smoke-fail | elsewhere |
| `L2_02_chunk_scan_fwd` | ssm_mamba | C | `kernel.py::placeholder` | vllm | formula_exact_manual | BLOCKED:shapes-inc | meta |
| `L2_03_chunk_state_fwd` | ssm_mamba | C | `kernel.py::placeholder` | vllm | formula_exact_manual | BLOCKED:shapes-inc | meta |
| `L2_04_chunked_lora_expand` | lora | C | `kernel.py::placeholder` | sglang | formula_exact_manual | BLOCKED:upstream-inv | elsewhere |
| `L2_05_chunked_lora_shrink` | lora | C | `kernel.py::placeholder` | sglang | formula_exact_manual | BLOCKED:upstream-inv | elsewhere |
| `L2_06_qkv_lora_b` | lora | C | `kernel.py::placeholder` | sglang | formula_exact_manual | BLOCKED:upstream-inv | elsewhere |
| `L2_07_gate_up_lora_b` | lora | C | `kernel.py::placeholder` | sglang | formula_exact_manual | BLOCKED:upstream-inv | elsewhere |
| `L2_08_sgemm_lora_a` | lora | C | `kernel.py::placeholder` | sglang | formula_exact_manual | BLOCKED:oracle-mis | elsewhere |
| `L2_09_sgemm_lora_b` | lora | C | `kernel.py::placeholder` | sglang | formula_exact_manual | BLOCKED:upstream-inv | elsewhere |
| `L2_10_w8a8_block_int8_matmul` | quant | C | `kernel.py::w8a8_block_int8_matmul` | vllm | upstream_oracle_required | BLOCKED:smoke-fail | meta |
| `L2_11_w8a8_triton_block_scaled_mm` | quant | C | `kernel.py::placeholder` | vllm | formula_exact_manual | BLOCKED:shapes-inc | elsewhere |
| `L2_12_awq_gemm` | gemm | C | `kernel.py::placeholder` | vllm | formula_exact_manual | BLOCKED:shapes-inc | meta |
| `L2_13_bmm` | gemm | C | `kernel.py::placeholder` | vllm | simple_pytorch_ok | BLOCKED:shapes-inc | elsewhere |
| `L2_14_matmul_kernel_persistent` | gemm | C | `kernel.py::placeholder` | vllm | simple_pytorch_ok | BLOCKED:shapes-inc | meta |
| `L2_15_fused_dual_residual_rmsnorm` | norm | C | `kernel.py::placeholder` | sglang | formula_exact_manual | BLOCKED:shapes-inc | meta |
| `L2_16_fused_rmsnorm` | norm | C | `kernel.py::placeholder` | sglang | formula_exact_manual | BLOCKED:oracle-mis | meta |
| `L2_17_layer_norm_gated_fwd` | norm | C | `kernel.py::placeholder` | vllm | upstream_oracle_required | BLOCKED:upstream-inv | elsewhere |
| `L2_18_fused_gdn_gating` | sampling | C | `kernel.py::placeholder` | vllm | formula_exact_manual | BLOCKED:upstream-inv | elsewhere |
| `L2_19_fused_sigmoid_gating_delta_rule_update` | delta_rule | C | `kernel.py::placeholder` | sglang | upstream_oracle_required | BLOCKED:shapes-inc | elsewhere |
| `L2_20_causal_conv1d_update` | ssm_mamba | C | `kernel.py::placeholder` | vllm | formula_exact_manual | BLOCKED:shapes-inc | meta |
| `L2_21_causal_conv1d_fwd` | ssm_mamba | C | `kernel.py::placeholder` | vllm | formula_exact_manual | BLOCKED:shapes-inc | meta |
| `L2_22_linear_attn_decode` | ssm_mamba | C | `kernel.py::placeholder` | vllm | formula_exact_manual | BLOCKED:shapes-inc | meta |
| `L2_23_fwd_kv_parallel` | paged_attention | C | `kernel.py::placeholder` | vllm | formula_exact_manual | BLOCKED:shapes-inc | meta |
| `L2_24_fwd_kv_reduce` | paged_attention | C | `kernel.py::placeholder` | vllm | formula_exact_manual | BLOCKED:shapes-inc | meta |
| `L2_25_fwd_kernel_ep_gather` | moe | C | `kernel.py::placeholder` | vllm | upstream_oracle_required | BLOCKED:upstream-inv | meta |
| `L2_26_fwd_kernel_ep_scatter_1` | moe | C | `kernel.py::placeholder` | vllm | formula_exact_manual | BLOCKED:upstream-inv | meta |
| `L2_27_deepep_permute_triton` | moe | C | `kernel.py::placeholder` | sglang | formula_exact_manual | BLOCKED:upstream-inv | meta |
| `L2_28_deepep_post_reorder_triton` | moe | C | `kernel.py::placeholder` | sglang | formula_exact_manual | BLOCKED:upstream-inv | meta |
| `L2_29_fused_moe_router_cudacore` | moe | C | `kernel.py::placeholder` | sglang | formula_exact_manual | BLOCKED:shapes-inc | meta |
| `L2_30_fused_moe_router_tensorcore` | moe | C | `kernel.py::placeholder` | sglang | formula_exact_manual | BLOCKED:shapes-inc | meta |
| `L2_31_moe_align_block_size_stage2` | moe | C | `kernel.py::placeholder` | sglang | formula_exact_manual | BLOCKED:shapes-inc | elsewhere |
| `L2_32_experts_combine` | moe | C | `kernel.py::placeholder` | sglang | simple_pytorch_ok | BLOCKED:shapes-inc | elsewhere |
| `L2_33_fwd_grouped_kernel_stage1_rope` | paged_attention | C | `kernel.py::placeholder` | sglang | upstream_oracle_required | BLOCKED:upstream-inv | meta |
| `L2_34_silu_mul_static_tensorwise_quant_triton_kernel_for_cutlass_moe` | moe | C | `kernel.py::placeholder` | sglang | formula_exact_manual | BLOCKED:upstream-inv | elsewhere |
| `L2_35_pre_reorder_triton_kernel_for_cutlass_moe` | moe | C | `kernel.py::placeholder` | sglang | formula_exact_manual | BLOCKED:shapes-inc | elsewhere |
| `L2_36_post_reorder_triton_kernel_for_cutlass_moe` | moe | C | `kernel.py::placeholder` | sglang | formula_exact_manual | BLOCKED:shapes-inc | elsewhere |
| `L2_37_silu_and_mul_post_quant` | moe | C | `kernel.py::placeholder` | sglang | upstream_oracle_required | BLOCKED:upstream-inv | elsewhere |
| `L2_38_concat_and_cast_mha_k` | elementwise | C | `kernel.py::placeholder` | sglang | formula_exact_manual | BLOCKED:shapes-inc | elsewhere |
| `L3_01_kernel_unified_attention_3d` | paged_attention | C | `kernel.py::placeholder` | vllm | upstream_oracle_required | BLOCKED:upstream-inv | meta |
| `L3_02_kernel_paged_attention_2d` | paged_attention | C | `kernel.py::placeholder` | vllm | upstream_oracle_required | BLOCKED:shapes-inc | elsewhere |
| `L3_03_chunk_state_varlen` | ssm_mamba | C | `kernel.py::placeholder` | sglang | upstream_oracle_required | BLOCKED:shapes-inc | elsewhere |
| `L3_04_fwd` | paged_attention | C | `kernel.py::context_attention_fwd` | sglang | upstream_oracle_required | BLOCKED:upstream-inv | elsewhere |
| `L3_05_fwd_kernel_alibi` | paged_attention | C | `kernel.py::placeholder` | vllm | formula_exact_manual | BLOCKED:shapes-inc | elsewhere |
| `L3_06_fwd_grouped_kernel_stage1` | paged_attention | C | `kernel.py::placeholder` | vllm | formula_exact_manual | BLOCKED:shapes-inc | meta |
| `L3_07_correct_attn_cp_out` | paged_attention | C | `kernel.py::placeholder` | vllm | formula_exact_manual | BLOCKED:shapes-inc | meta |
| `L3_09_fused_moe` | moe | C | `kernel.py::placeholder` | vllm | upstream_oracle_required | BLOCKED:upstream-inv | meta |
| `L3_10_fused_moe_kernel_gptq_awq` | moe | C | `kernel.py::placeholder` | vllm | formula_exact_manual | BLOCKED:upstream-inv | meta |
| `L3_11_expert_triton` | moe | C | `kernel.py::placeholder` | vllm | formula_exact_manual | BLOCKED:shapes-inc | elsewhere |
| `L3_12_moe_mmk` | moe | C | `kernel.py::placeholder` | vllm | formula_exact_manual | BLOCKED:shapes-inc | elsewhere |
| `L3_13_chunk_gated_delta_rule_fwd_kernel_h_blockdim64` | delta_rule | C | `kernel.py::placeholder` | vllm | formula_exact_manual | BLOCKED:upstream-inv | elsewhere |
| `L3_14_fused_recurrent_gated_delta_rule_fwd` | delta_rule | C | `kernel.py::placeholder` | vllm | upstream_oracle_required | BLOCKED:shapes-inc | meta |
| `L3_15_fused_recurrent_gated_delta_rule_update_fwd` | delta_rule | C | `kernel.py::placeholder` | sglang | upstream_oracle_required | BLOCKED:shapes-inc | elsewhere |
| `L3_16_chunk_fwd_kernel_o` | ssm_mamba | C | `kernel.py::placeholder` | vllm | formula_exact_manual | BLOCKED:upstream-inv | elsewhere |
| `L3_17_chunk_gla_fwd_kernel_o` | delta_rule | C | `kernel.py::placeholder` | vllm | formula_exact_manual | BLOCKED:upstream-inv | elsewhere |
| `L3_18_chunk_scaled_dot_kkt_fwd` | delta_rule | C | `kernel.py::placeholder` | vllm | formula_exact_manual | BLOCKED:upstream-inv | elsewhere |
| `L3_19_chunk_kda_scaled_dot_kkt_fwd_kernel_intra_sub_inter` | delta_rule | C | `kernel.py::placeholder` | vllm | formula_exact_manual | BLOCKED:upstream-inv | meta |
| `L3_20_recompute_w_u_fwd` | delta_rule | C | `kernel.py::placeholder` | vllm | formula_exact_manual | BLOCKED:upstream-inv | meta |
| `L3_21_solve_tril_16x16` | delta_rule | C | `kernel.py::placeholder` | vllm | formula_exact_manual | BLOCKED:upstream-inv | meta |
| `L3_22_chunk_local_cumsum_vector` | ssm_mamba | C | `kernel.py::placeholder` | vllm | formula_exact_manual | BLOCKED:upstream-inv | elsewhere |
| `L3_23_merge_16x16_to_32x32_inverse` | delta_rule | C | `kernel.py::placeholder` | vllm | formula_exact_manual | BLOCKED:upstream-inv | elsewhere |

---

## 3. Family Lane Deep Dive

### 3.1 `gemm` (n=3) — **P0 lane, lowest oracle complexity**

| task | level | upstream | oracle_tier | gpu_triton (meta) | verdict |
|---|---|---|---|---|---|
| L2_12_awq_gemm | L2 | vllm | formula_exact_manual | placeholder | BLOCKED:shapes-inc (func **at meta path**) |
| L2_13_bmm | L2 | vllm | simple_pytorch_ok | placeholder | BLOCKED:shapes-inc (elsewhere) |
| L2_14_matmul_kernel_persistent | L2 | vllm | simple_pytorch_ok | placeholder | BLOCKED:shapes-inc (at meta path) |

**Lane priority**: P0. 2/3 task 是 `simple_pytorch_ok`，纯 PyTorch oracle 可直接覆盖 `torch.matmul / torch.bmm`。**应走 ≥80% PyTorch oracle 路径**（不用 GPU 跑预跑）。

**升级路径** (C → B → A):
1. 在 oracle_pytorch.py 中实现 `torch.matmul` reference
2. 扩展 `shapes.yaml` 让所有 public/hidden shape 都能覆盖
3. 把 `gpu_triton` 从 placeholder 改为真函数名（`matmul_kernel_persistent` 等）
4. 移除 meta `notes` 里的 placeholder 字样

**估时**: 数小时（3 task × 1 小时 freeze 工作 + shape 表写完整）

### 3.2 `block_table` (n=5) — **P0 lane**

| task | level | upstream | oracle_tier | verdict |
|---|---|---|---|---|
| L1_02_compute_slot_mappings | L1 | vllm | formula_exact_manual | BLOCKED:upstream-inv (elsewhere) |
| L1_03_convert_req_index_to_global_index | L1 | vllm | formula_exact_manual | BLOCKED:upstream-inv (elsewhere) |
| L1_04_copy_page_indices | L1 | vllm | formula_exact_manual | BLOCKED:upstream-inv (elsewhere) |
| L1_05_gather_block_tables | L1 | vllm | formula_exact_manual | BLOCKED:upstream-inv (elsewhere) |
| L1_34_transform_index_page_table_decode | L1 | sglang | formula_exact_manual | **UNRUN(gpu)** — already B-class |

**Lane priority**: P0. 全 `formula_exact_manual` — 纯整数索引算术 (slot = block_table[req, pos//bs]*bs + pos%bs)，**不需 GPU 真跑**。卡点全是 `upstream-not-invocable-here` —— meta 的 entrypoint 名字和实际 function signature/参数对不上。**应走 ≥80% PyTorch oracle 路径**。

**升级路径**:
1. 修正 `meta.upstream.path` 指向真实文件（4/5 现在指向目录而非文件）
2. 修正 `oracle.gpu_triton` 指向实际 entrypoint（resolver 已记录在 `tried` 列表里）
3. 调整 `oracle_pytorch.py` 让其与 kernel 输入契约一致

**估时**: 0.5-1 天

### 3.3 `flashinfer_indices` (n=5) — **P0 lane**

| task | level | upstream | oracle_tier | verdict |
|---|---|---|---|---|
| L1_32_create_flashinfer_kv_indices_triton | L1 | sglang | formula_exact_manual | BLOCKED:shapes-inc (elsewhere) |
| L1_33_create_flashmla_kv_indices_triton | L1 | sglang | formula_exact_manual | BLOCKED:shapes-inc (elsewhere) |
| L1_35_compute_src2dst_triton | L1 | sglang | formula_exact_manual | BLOCKED:shapes-inc (at meta path) |
| L1_36_compute_seg_indptr_triton | L1 | sglang | formula_exact_manual | BLOCKED:upstream-inv (at meta path) |
| L1_37_compute_problem_sizes_w4a8 | L1 | sglang | formula_exact_manual | BLOCKED:upstream-inv (elsewhere) |

**Lane priority**: P0. 全 sglang，全 `formula_exact_manual`。攻法同 block_table。

**估时**: 0.5-1 天

### 3.4 `lora` (n=6) — **P0 lane**

| task | level | upstream | oracle_tier | verdict |
|---|---|---|---|---|
| L2_04_chunked_lora_expand | L2 | sglang | formula_exact_manual | BLOCKED:upstream-inv (elsewhere) |
| L2_05_chunked_lora_shrink | L2 | sglang | formula_exact_manual | BLOCKED:upstream-inv (elsewhere) |
| L2_06_qkv_lora_b | L2 | sglang | formula_exact_manual | BLOCKED:upstream-inv (elsewhere) |
| L2_07_gate_up_lora_b | L2 | sglang | formula_exact_manual | BLOCKED:upstream-inv (elsewhere) |
| L2_08_sgemm_lora_a | L2 | sglang | formula_exact_manual | BLOCKED:oracle-mis (elsewhere) |
| L2_09_sgemm_lora_b | L2 | sglang | formula_exact_manual | BLOCKED:upstream-inv (elsewhere) |

**Lane priority**: P0. 全 sglang，全 `formula_exact_manual`。LoRA 是 `(B @ A.T) @ X + bias` —— 标准 GEMM + segment dispatch；纯公式可推。**应走 ≥80% PyTorch oracle 路径**。

**升级路径**:
1. 重新指 `meta.upstream.path` —— 5/6 当前指向 `sglang/srt/lora/` 目录而非具体文件
2. 修正 `oracle.gpu_triton` 入口（sglang 实际 entrypoint 是 `chunked_sgmv_lora_expand_forward` 等）
3. 在 oracle_pytorch.py 写 `torch.bmm + index_select` 替身

**估时**: 1 天

### 3.5 `elementwise` (n=13) — **P1 lane**

包含 3 个已 B 的 (L1_01 bias, L1_16 silu_and_mul, L1_26 merge_state)，10 个 C。oracle tier 分布: 5 simple + 5 formula + 3 upstream_req。

| sub-cluster | tasks |
|---|---|
| **已 B (待 GPU 验证)** | L1_01_bias, L1_16_silu_and_mul, L1_26_merge_state |
| **simple_pytorch_ok placeholder** | L1_17_gelu_and_mul, L1_18_fused_softcap, L1_20_log_softmax |
| **formula_exact_manual placeholder** | L1_27_apply_grammar_bitmask, L1_28_pack_bitmatrix, L1_29_prepare_eagle_inputs, L1_31_triton_mrope_forward, L2_38_concat_and_cast_mha_k |
| **upstream_oracle_required (难)** | L1_15_act_and_mul, L1_25_merge_attn_states |

**Lane priority**: P1。除 3 个 `upstream_oracle_required` 外，10 个都走 ≥80% PyTorch oracle 路径。

**估时**: 1-2 天

### 3.6 `ssm_mamba` (n=11) — **P1 lane**

| task | upstream | oracle_tier | verdict |
|---|---|---|---|
| L1_39_state_passing_fwd | vllm | formula_exact_manual | BLOCKED:shapes-inc |
| L1_40_chunk_cumsum_fwd | vllm | formula_exact_manual | BLOCKED:shapes-inc |
| L2_01_bmm_chunk_fwd | vllm | formula_exact_manual | **BLOCKED:smoke-fail** |
| L2_02_chunk_scan_fwd | vllm | formula_exact_manual | BLOCKED:shapes-inc |
| L2_03_chunk_state_fwd | vllm | formula_exact_manual | BLOCKED:shapes-inc |
| L2_20_causal_conv1d_update | vllm | formula_exact_manual | BLOCKED:shapes-inc |
| L2_21_causal_conv1d_fwd | vllm | formula_exact_manual | BLOCKED:shapes-inc |
| L2_22_linear_attn_decode | vllm | formula_exact_manual | BLOCKED:shapes-inc |
| L3_03_chunk_state_varlen | sglang | upstream_oracle_required | BLOCKED:shapes-inc |
| L3_16_chunk_fwd_kernel_o | vllm | formula_exact_manual | BLOCKED:upstream-inv |
| L3_22_chunk_local_cumsum_vector | vllm | formula_exact_manual | BLOCKED:upstream-inv |

**Lane priority**: P1。10/11 是 `formula_exact_manual`，与 Mamba SSM 数学一致。L2_01_bmm_chunk_fwd 是 smoke-fail —— CPU pass，TRITON_INTERPRET 实跑失败，需注意。

**估时**: 2 天

### 3.7 `moe` (n=16) — **P1 lane (最大单一 lane)**

包含 ep_gather/ep_scatter/router_cudacore/router_tensorcore/cutlass_moe 各种 MoE 算子。

oracle tier 分布: 1 simple + 12 formula + 3 upstream_req。

**Lane priority**: P1。13/16 是 simple/formula —— 集中于 deepep permute / experts_combine / moe_align / fused_moe_router 等。可批量走 ≥80% PyTorch oracle。

**estim batch size**: 2-3 天 (lane 最大但同质性高)

### 3.8 `quant` (n=6) — **P1 lane**

| task | oracle_tier | verdict |
|---|---|---|
| L1_10_per_token_group_quant_int8 | formula_exact_manual | BLOCKED:shapes-inc |
| L1_11_awq_dequantize | formula_exact_manual | BLOCKED:shapes-inc |
| L1_30_act_quant | formula_exact_manual | BLOCKED:shapes-inc |
| L1_38_silu_mul_fp8_quant_deep_gemm | formula_exact_manual | BLOCKED:upstream-inv |
| L2_10_w8a8_block_int8_matmul | upstream_oracle_required | **BLOCKED:smoke-fail** |
| L2_11_w8a8_triton_block_scaled_mm | formula_exact_manual | BLOCKED:shapes-inc |

**估时**: 1-2 天。L2_10 是 smoke-fail 特殊处理。

### 3.9 `sampling` (n=6) — **P1 lane**

| task | oracle_tier | verdict |
|---|---|---|
| L1_19_topk_log_softmax | formula_exact_manual | BLOCKED:upstream-inv |
| L1_21_gumbel_sample | upstream_oracle_required | BLOCKED:shapes-inc |
| L1_22_min_p | formula_exact_manual | **BLOCKED:oracle-mis** |
| L1_23_penalties_and_temperature | formula_exact_manual | BLOCKED:upstream-inv |
| L1_24_rejection_sample | formula_exact_manual | BLOCKED:upstream-inv |
| L2_18_fused_gdn_gating | formula_exact_manual | BLOCKED:upstream-inv |

**Lane priority**: P1。`L1_22_min_p` 有 oracle-mismatch（数值未对齐）需特殊处理。其他 5 个走 ≥80% PyTorch oracle 路径。

**估时**: 1-2 天

### 3.10 `delta_rule` (n=10) — **P2 lane**

全 vllm/sglang flash-linear-attention 流派，3 个 `upstream_oracle_required`（recurrent gated delta rule 数值精度高、reference 难手写）。

| sub-cluster | tasks |
|---|---|
| upstream_oracle_required | L2_19, L3_14, L3_15 |
| formula_exact_manual | L3_13, L3_17, L3_18, L3_19, L3_20, L3_21, L3_23 |

**估时**: 2-3 天

### 3.11 `paged_attention` (n=13) — **P2 lane**

| sub-cluster | tasks | 备注 |
|---|---|---|
| upstream_oracle_required (4) | L2_33, L3_01, L3_02, L3_04 | 需走 ≤20% provider GPU 预跑（attention/oracle 数值复杂） |
| formula_exact_manual (9) | L1_06, L1_07, L1_08, L1_09, L2_23, L2_24, L3_05, L3_06, L3_07 | 走 ≥80% PyTorch oracle |

**Lane priority**: P2。9/13 可走 PyTorch oracle 路径；4 个需要 provider 预跑。

**估时**: 3-4 天

### 3.12 `norm` (n=6) — **P3 lane**

| task | oracle_tier | verdict |
|---|---|---|
| L1_12_rms_norm | formula_exact_manual | **UNRUN(gpu)** — already B |
| L1_13_layer_norm_fwd | upstream_oracle_required | BLOCKED:upstream-inv |
| L1_14_layer_norm_fwd_1pass | upstream_oracle_required | **BLOCKED:oracle-mis** |
| L2_15_fused_dual_residual_rmsnorm | formula_exact_manual | BLOCKED:shapes-inc |
| L2_16_fused_rmsnorm | formula_exact_manual | **BLOCKED:oracle-mis** |
| L2_17_layer_norm_gated_fwd | upstream_oracle_required | BLOCKED:upstream-inv |

**Lane priority**: P3。3 个 `upstream_oracle_required` 占一半 — layer_norm 的 oracle 设计要小心 (welford / 数值稳定)。2 个 `oracle-mis`（已经发现 oracle 与 kernel 输出不一致），需先调 oracle_pytorch.py 的数值精度。

**估时**: 1-2 天，但需要数值精度功课

---

## 4. Maturity 升级路径建议

### 4.1 C.placeholder → B.frozen-partial

**条件**: resolver_resolved=True, 即 resolver 能找到一个能 oracle-consistent 的 entrypoint。

**所需动作**（每个 task）:
1. **修正 `meta.upstream.path`**: 改成 ground-truth 真实文件路径（去掉尾随 `/`、去掉反引号、指向具体 .py 而非目录）
2. **修正 `meta.oracle.gpu_triton`**: 从 `kernel.py::placeholder` 改成 `kernel.py::<真实函数名>`。真实函数名见 resolver verdict 的 `tried` 列表或 `entrypoint` 字段
3. **检查 `shapes.yaml`**: 确认所有 public/hidden shape 都能让 oracle 跑通（针对 47 个 `shapes-incomplete` task）
4. **检查 `oracle_pytorch.py`**: 输入 dict 的 key 与 kernel 输入名一致（针对 oracle-mis 6 task / KeyError 类问题）
5. **移除 stale `notes`**: 删掉 "kernel.py is a placeholder; freeze upstream before grading" 这类陈年字符（针对 90+ task）

### 4.2 B.frozen-partial → A.frozen-clean

**条件**: A-class 需要 (SPDX 头 + @triton.jit + simple_pytorch_ok oracle + oracle_pytorch.py)。

**所需动作**:
- 5 个 B-class 当前都没有 SPDX header。如果 kernel.py 是手写而非 vendored copy，**A-class 标准本身需要软化**：把 "SPDX header required" 改成 "OR has Apache/MIT license note"
- L1_26_merge_state 还有 stale placeholder note 需要清理
- L1_34_transform_index_page_table_decode 的 `meta.oracle.gpu_triton` 还指向 `::placeholder`，应改成 `kernel.py::transform_index_page_table_decode_fast`

### 4.3 D.unresolvable → escalate

**本次 audit 零命中**。所有 100 task 真函数都在 `infer/{vllm,sglang}/` 树里可定位。不需任何 escalate 或 drop。

---

## 5. 战略建议 (给 user)

### 5.1 Lane 优先级 ranking + actionable

**Phase A (P0 lanes — 19 task — 2-3 天) — 起手 quick wins**:
1. `gemm` (3) → 全 simple/formula，可批量
2. `block_table` (5) → meta path/entrypoint 修正
3. `flashinfer_indices` (5) → 同 block_table
4. `lora` (6) → sglang 全 formula

**Phase B (P1 lanes — 58 task — 8-12 天)**:
5. `elementwise` (13)
6. `ssm_mamba` (11) — 注意 L2_01 smoke-fail
7. `sampling` (6) — 注意 L1_22 oracle-mis
8. `quant` (6) — 注意 L2_10 smoke-fail
9. `moe` (16) — 最大同质 lane

**Phase C (P2-P3 lanes — 23 task — 6-9 天)**:
10. `delta_rule` (10)
11. `paged_attention` (13) — 4 个需 GPU 预跑
12. `norm` (6) — 数值精度功课

**GPU 真跑阶段 (并行)**:
- 5 个 B-class UNRUN(gpu) task 一旦上 GPU 真跑即可全过：L1_01 bias, L1_12 rms_norm, L1_16 silu_and_mul, L1_26 merge_state, L1_34 transform_index_page_table_decode

### 5.2 与 codex 既定 E→D 路径的衔接

**[confirmed]** Truth inventory 验证了 codex 既有结论 ——"99 个非 canary task 中绝大部分是 placeholder + oracle.gpu_triton 是 placeholder"。但同时**澄清了一个误判**：corpus 不是 "scaffold 期 unresolvable"，而是 "scaffold 期已有完整源码材料但 meta/oracle/contract 契约未对齐"。

从 inventory 视角看 E→D 路径的具体含义：
- **E (provenance-first)**: 即本 doc 已完成 — 100 task 的真值来源已彻底盘清
- **D (workflow re-alignment)**: 针对 95 个 BLOCKED task，按 family lane 批量做以下三件事 —
  - 修正 `meta.upstream.path` (28 个指向目录 + 14 个完全失效)
  - 修正 `meta.oracle.gpu_triton` (93 个仍是 `kernel.py::placeholder`)
  - 完整 shapes.yaml + 对齐 oracle_pytorch.py 入参契约

### 5.3 P3 phase_b schema 重写 vs P4 family lane 实施

基于本 inventory，建议**先 P4 family lane 实施**而非 P3 phase_b schema 重写：
- P3 schema 重写本质是为了应对 D.unresolvable 类问题（structurally 无法验证的 task），但本 audit 显示 **D 类为 0**
- P4 family lane 实施才是直接 unblock 95 个 BLOCKED task 的最短路径

**[confirmed]** 数据基础：100 task 真函数 100% 可定位 + 95 个仅卡 `shapes-incomplete` (47) / `upstream-not-invocable-here` (40) / `oracle-mismatch` (6) / `smoke-fail` (2) 这四类标准化失败模式。Family lane 内的同质性高（如 lora 6 个 task 攻法基本一致），可以批量处理。

---

## Appendix A: Methodology

### A.1 数据采集

**收集脚本**: `/tmp/truth_inventory.py`（read-only filesystem 操作，不 import upstream package，避免 vllm placeholder fallback 触发）

**数据来源**:
- `tasks/level{1,2,3}/<task_id>/meta.yaml` — alias/oracle/upstream/oracle_simplified 字段
- `tasks/level{1,2,3}/<task_id>/kernel.py` — size, SPDX, @triton.jit detection
- `tasks/level{1,2,3}/<task_id>/shapes.yaml` — public_shape_count, hidden_shape_count
- `tasks/level{1,2,3}/<task_id>/oracle_pytorch.py` — existence, line count
- `infer/{vllm, sglang, sgl-kernel-npu, vllm-ascend}/` — upstream module/file/function existence via filesystem grep (no Python import)
- `results/resolve_all.json` — resolver verdicts (status, entrypoint, tried, reason)

### A.2 Maturity heuristic

```
PRIMARY: resolver_resolved (resolver found an oracle-consistent real entrypoint)
SECONDARY: meta.oracle.gpu_triton state + meta placeholder note presence

A.frozen-clean:   resolver_resolved=True AND meta non-placeholder AND SPDX header
                  AND @triton.jit AND simple_pytorch_ok AND oracle_pytorch.py present
B.frozen-partial: resolver_resolved=True AND (any A condition missing)
C.placeholder:    resolver_resolved=False AND function locatable in infer/ tree
D.unresolvable:   resolver_resolved=False AND function NOT found in any .py
```

### A.3 Family lane heuristic

基于 task_id + meta.name + meta.upstream.path 的 keyword match。规则顺序: 具体（delta_rule / ssm_mamba / lora / moe / quant / flashinfer_indices / block_table / sampling / norm）→ 抽象（paged_attention / elementwise / gemm）。

### A.4 局限性 (caveats)

- **[confirmed]** 100 task upstream function 都能在 `infer/` 找到 —— 但 "找到" 不等于 "能 invoke"。40 个 `upstream-not-invocable-here` task 说明，即使源码就位，由于本机缺包（如 trtllm）/参数签名变化 / numpy 与 cuda 类型不匹配等，resolver 也无法直接 invoke。这些需要 oracle_pytorch.py 实现一个独立 reference
- **[likely]** family lane 优先级排序基于 oracle tier，未考虑 shape coverage 实际填写难度 —— 这部分需要 Phase B' 单独的 shape audit 才能精确估时
- **[speculative]** "Lane 内同质性高，可批量处理" 这一战略论断基于 task name 相似性，未做 deep code-level 验证。实际执行时仍需逐 task review

---

## Appendix B: 5 个 B-class task 的 meta 不一致清单

针对升级 B → A，目前每个 B-class 都有 meta 异常需要修：

| task | issue | fix |
|---|---|---|
| L1_01_bias | kernel.py 无 SPDX header（手写非 vendored） | 软化 A-class 标准 OR 添加 license note |
| L1_12_rms_norm | 同上 | 同上 |
| L1_16_silu_and_mul | 同上 | 同上 |
| L1_26_merge_state | meta.notes 仍含 stale "kernel.py is a placeholder" 文本 | 清理 notes |
| L1_34_transform_index_page_table_decode | **meta.oracle.gpu_triton = `kernel.py::placeholder`** 但 resolver 已经 auto-discovered `transform_index_page_table_decode_fast` | 改 meta.oracle.gpu_triton |

---

*End of inventory. 文档基于 read-only 探查；未修改任何 code/yaml/meta。*
