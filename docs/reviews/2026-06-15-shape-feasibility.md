---
title: TAB shape/dtype feasibility (Unit 1 -- kernel-source-derived, reproducible)
date: 2026-06-15
tool: scripts/corpus/shape_feasibility_report.py --regen
---

# Shape / dtype feasibility report

Per-task dtype- and shape-flexibility derived from the KERNEL SOURCE + `param_constraints`
+ launcher asserts (NOT `meta.dtype`). Heuristic + evidence-backed: every verdict cites the
tokens it matched so it is spot-checkable. Regenerate with `--regen` (deterministic).

Tasks: **101**

## dtype_class distribution

| class | count | meaning |
|---|---|---|
| `multi_dtype` | 10 | kernel dispatches on / accepts >=2 float dtypes -- Unit-4 dtype-sweep candidate |
| `single_dtype` | 47 | one float dtype token; likely locked |
| `quant_locked` | 20 | fp8/int8 quant; dtype not freely swappable |
| `unknown` | 24 | no clear dtype signal (manual check) |

## shape_class distribution

| class | count | meaning |
|---|---|---|
| `loose` | 42 | bound-only dim constraints; free to scale |
| `medium` | 42 | a few structural constraints |
| `tight` | 17 | many structural constraints / launcher asserts; needs a per-family lattice |

> These counts are DERIVED (reproducible), not hand-asserted. The expansion (Unit 2/3/4)
> validates every candidate shape by a real build_inputs + oracle dry-run regardless of class.

## Per-task

| task | level | dtype_class | shape_class | #shapes | launcher | evidence |
|------|-------|-------------|-------------|---------|----------|----------|
| L1_01_bias | level1 | unknown | medium | 9 | - | 3 bound dim constraint(s); 1 tensor_invariant(s) |
| L1_02_compute_slot_mappings | level1 | unknown | medium | 9 | Y | 3 bound dim constraint(s); 1 tensor_invariant(s) |
| L1_03_convert_req_index_to_global_index | level1 | unknown | medium | 9 | - | 4 bound dim constraint(s); 1 tensor_invariant(s) |
| L1_04_copy_page_indices | level1 | unknown | medium | 9 | Y | 2 bound dim constraint(s); 1 tensor_invariant(s) |
| L1_05_gather_block_tables | level1 | unknown | medium | 9 | Y | 3 bound dim constraint(s); 1 tensor_invariant(s) |
| L1_06_reshape_and_cache_kernel_flash | level1 | quant_locked | tight | 9 | Y | quant tokens: ['fp8']; 6 bound dim constraint(s); 4 tensor_invariant(s) |
| L1_07_dequantize_k_cache_fast | level1 | multi_dtype | loose | 9 | - | float dtype tokens: ['bfloat16', 'float32']; 1 bound dim constraint(s) |
| L1_08_dequantize_k_cache_paged | level1 | multi_dtype | medium | 9 | - | float dtype tokens: ['bfloat16', 'float32']; 2 bound dim constraint(s); 1 tensor_invariant(s) |
| L1_09_quantize_k_cache_fast | level1 | quant_locked | tight | 9 | Y | float dtype tokens: ['bfloat16', 'float32']; quant tokens: ['fp8']; 1 bound dim constraint(s); 1 structural dim constraint(s): ['D == 512']; 7 launcher assert(s) |
| L1_10_per_token_group_quant_int8 | level1 | quant_locked | medium | 9 | Y | float dtype tokens: ['float32']; quant tokens: ['int8']; 2 bound dim constraint(s); 1 structural dim constraint(s): ['D % 8 == 0']; 1 launcher assert(s) |
| L1_11_awq_dequantize | level1 | unknown | medium | 9 | - | 2 bound dim constraint(s); 1 structural dim constraint(s): ['N % 64 == 0'] |
| L1_12_rms_norm | level1 | single_dtype | loose | 9 | - | float dtype tokens: ['float32']; 2 bound dim constraint(s) |
| L1_13_fused_softcap | level1 | unknown | loose | 9 | - | 2 bound dim constraint(s) |
| L1_14_merge_attn_states | level1 | unknown | loose | 9 | - | 3 bound dim constraint(s) |
| L1_15_merge_state | level1 | unknown | loose | 9 | - | 3 bound dim constraint(s) |
| L1_16_apply_grammar_bitmask | level1 | unknown | medium | 9 | Y | 4 bound dim constraint(s); 1 tensor_invariant(s) |
| L1_17_prepare_eagle_inputs | level1 | unknown | medium | 9 | Y | 3 bound dim constraint(s); 2 tensor_invariant(s) |
| L1_18_act_quant | level1 | single_dtype | medium | 9 | - | float dtype tokens: ['float32']; 2 bound dim constraint(s); 1 structural dim constraint(s): ['D % 8 == 0'] |
| L1_19_triton_mrope_forward | level1 | unknown | loose | 9 | - | 2 bound dim constraint(s) |
| L1_20_create_flashinfer_kv_indices_triton | level1 | unknown | medium | 9 | Y | 2 bound dim constraint(s); 2 tensor_invariant(s) |
| L1_21_create_flashmla_kv_indices_triton | level1 | unknown | medium | 9 | Y | 2 bound dim constraint(s); 1 tensor_invariant(s) |
| L1_22_transform_index_page_table_decode | level1 | unknown | loose | 9 | - | 2 bound dim constraint(s) |
| L1_23_compute_src2dst_triton | level1 | quant_locked | medium | 9 | Y | quant tokens: ['fp8']; 1 bound dim constraint(s); 1 tensor_invariant(s) |
| L1_24_compute_seg_indptr_triton | level1 | quant_locked | loose | 9 | Y | quant tokens: ['fp8']; 2 bound dim constraint(s) |
| L1_25_compute_problem_sizes_w4a8 | level1 | quant_locked | loose | 9 | Y | quant tokens: ['fp8']; 2 bound dim constraint(s) |
| L1_26_state_passing_fwd | level1 | single_dtype | loose | 9 | - | float dtype tokens: ['float32']; 4 bound dim constraint(s) |
| L1_27_chunk_cumsum_fwd | level1 | single_dtype | loose | 9 | Y | float dtype tokens: ['float32']; 2 bound dim constraint(s); 2 launcher assert(s) |
| L1_28_fused_rmsnorm | level1 | single_dtype | loose | 9 | - | float dtype tokens: ['float32']; 2 bound dim constraint(s) |
| L1_29_fused_gdn_gating | level1 | single_dtype | loose | 9 | - | float dtype tokens: ['float32']; 2 bound dim constraint(s) |
| L1_30_deepep_permute_triton | level1 | quant_locked | loose | 9 | Y | float dtype tokens: ['float32']; quant tokens: ['fp8']; 2 bound dim constraint(s); 2 launcher assert(s) |
| L1_31_deepep_post_reorder_triton | level1 | quant_locked | loose | 9 | Y | quant tokens: ['fp8']; 2 bound dim constraint(s) |
| L1_32_experts_combine | level1 | unknown | loose | 9 | Y | 2 bound dim constraint(s); 2 launcher assert(s) |
| L1_33_pre_reorder_triton_kernel_for_cutlass_moe | level1 | quant_locked | loose | 9 | Y | float dtype tokens: ['float32']; quant tokens: ['fp8']; 2 bound dim constraint(s) |
| L1_34_post_reorder_triton_kernel_for_cutlass_moe | level1 | quant_locked | loose | 9 | Y | float dtype tokens: ['float32']; quant tokens: ['fp8']; 2 bound dim constraint(s) |
| L1_35_concat_and_cast_mha_k | level1 | multi_dtype | medium | 9 | Y | float dtype tokens: ['bfloat16', 'float16', 'float32']; 4 bound dim constraint(s); 3 launcher assert(s) |
| L1_36_gemma_qkv_rmsnorm | level1 | single_dtype | medium | 9 | Y | float dtype tokens: ['float32']; 4 bound dim constraint(s); 1 structural dim constraint(s): ['QH % KVH == 0'] |
| L2_01_layer_norm_fwd | level2 | single_dtype | loose | 9 | - | float dtype tokens: ['float32']; 2 bound dim constraint(s) |
| L2_02_layer_norm_fwd_1pass | level2 | single_dtype | loose | 9 | - | float dtype tokens: ['float32']; 2 bound dim constraint(s) |
| L2_03_act_and_mul | level2 | quant_locked | tight | 9 | Y | float dtype tokens: ['float32']; quant tokens: ['fp8', 'int8']; 3 bound dim constraint(s); 1 structural dim constraint(s): ['D % 2 == 0']; 3 launcher assert(s) |
| L2_04_silu_and_mul | level2 | single_dtype | loose | 9 | - | float dtype tokens: ['float32']; 2 bound dim constraint(s) |
| L2_05_gelu_and_mul | level2 | single_dtype | medium | 9 | - | float dtype tokens: ['float32']; 2 bound dim constraint(s); 1 structural dim constraint(s): ['D % 2 == 0'] |
| L2_06_topk_log_softmax | level2 | single_dtype | loose | 9 | - | float dtype tokens: ['float32']; 3 bound dim constraint(s) |
| L2_07_log_softmax | level2 | unknown | loose | 9 | - | 2 bound dim constraint(s) |
| L2_08_gumbel_sample | level2 | multi_dtype | medium | 9 | Y | float dtype tokens: ['float32', 'float64']; 3 bound dim constraint(s); 1 tensor_invariant(s); 3 launcher assert(s) |
| L2_09_min_p | level2 | single_dtype | medium | 9 | - | float dtype tokens: ['float32']; 3 bound dim constraint(s); 1 tensor_invariant(s) |
| L2_10_penalties_and_temperature | level2 | single_dtype | medium | 9 | Y | float dtype tokens: ['float32']; 3 bound dim constraint(s); 1 tensor_invariant(s) |
| L2_11_rejection_sample | level2 | unknown | tight | 9 | Y | 2 bound dim constraint(s); 3 tensor_invariant(s) |
| L2_12_pack_bitmatrix | level2 | unknown | loose | 9 | Y | 3 bound dim constraint(s) |
| L2_13_silu_mul_fp8_quant_deep_gemm | level2 | quant_locked | tight | 9 | Y | float dtype tokens: ['float32']; quant tokens: ['fp8']; 3 bound dim constraint(s); 1 structural dim constraint(s): ['D % 128 == 0']; 6 launcher assert(s) |
| L2_14_bmm_chunk_fwd | level2 | multi_dtype | medium | 9 | - | float dtype tokens: ['bfloat16', 'float16', 'float32']; 3 bound dim constraint(s); 2 tensor_invariant(s) |
| L2_15_chunked_lora_shrink | level2 | single_dtype | medium | 9 | Y | float dtype tokens: ['float32']; 4 bound dim constraint(s); 2 tensor_invariant(s); 1 launcher assert(s) |
| L2_16_sgemm_lora_a | level2 | single_dtype | medium | 9 | Y | float dtype tokens: ['float32']; 4 bound dim constraint(s); 2 tensor_invariant(s); 1 launcher assert(s) |
| L2_17_w8a8_block_int8_matmul | level2 | quant_locked | tight | 9 | - | float dtype tokens: ['bfloat16', 'float16', 'float32']; quant tokens: ['int8', 'w8a8']; 3 bound dim constraint(s); 2 structural dim constraint(s): ['D % 8 == 0', 'N % 8 == 0'] |
| L2_18_w8a8_triton_block_scaled_mm | level2 | quant_locked | tight | 9 | Y | float dtype tokens: ['bfloat16', 'float16', 'float32']; quant tokens: ['int8']; 3 bound dim constraint(s); 2 structural dim constraint(s): ['D % 8 == 0', 'N % 8 == 0']; 7 launcher assert(s) |
| L2_19_awq_gemm | level2 | unknown | medium | 9 | - | 3 bound dim constraint(s); 1 structural dim constraint(s): ['N % 32 == 0'] |
| L2_20_bmm | level2 | multi_dtype | loose | 9 | - | dtype-dispatch config dict keyed by ['bfloat16', 'float16', 'float32']; float dtype tokens: ['bfloat16', 'float16', 'float32']; 4 bound dim constraint(s) |
| L2_21_matmul_kernel_persistent | level2 | multi_dtype | loose | 9 | - | dtype-dispatch config dict keyed by ['bfloat16', 'float16', 'float32']; float dtype tokens: ['bfloat16', 'float16', 'float32']; 3 bound dim constraint(s) |
| L2_22_fused_dual_residual_rmsnorm | level2 | single_dtype | loose | 9 | - | float dtype tokens: ['float32']; 2 bound dim constraint(s) |
| L2_23_layer_norm_gated_fwd | level2 | single_dtype | loose | 9 | - | float dtype tokens: ['float32']; 2 bound dim constraint(s) |
| L2_24_linear_attn_decode | level2 | single_dtype | loose | 9 | - | float dtype tokens: ['float32']; 3 bound dim constraint(s) |
| L2_25_fwd_kernel_ep_gather | level2 | single_dtype | medium | 9 | Y | float dtype tokens: ['float32']; 4 bound dim constraint(s); 1 tensor_invariant(s); 1 launcher assert(s) |
| L2_26_fwd_kernel_ep_scatter_1 | level2 | unknown | loose | 9 | Y | 1 bound dim constraint(s) |
| L2_27_fused_moe_router_cudacore | level2 | single_dtype | loose | 9 | - | float dtype tokens: ['float32']; 2 bound dim constraint(s) |
| L2_28_moe_align_block_size_stage2 | level2 | unknown | loose | 9 | Y | 2 bound dim constraint(s) |
| L2_29_silu_mul_static_tensorwise_quant_triton_kernel_for_cutlass_moe | level2 | quant_locked | tight | 9 | Y | float dtype tokens: ['float32']; quant tokens: ['e4m3', 'fp8']; 3 bound dim constraint(s); 1 structural dim constraint(s): ['D % G == 0']; 7 launcher assert(s) |
| L2_30_silu_and_mul_post_quant | level2 | quant_locked | medium | 9 | Y | float dtype tokens: ['float32']; quant tokens: ['e4m3', 'fp8']; 3 bound dim constraint(s); 4 launcher assert(s) |
| L2_31_scaled_mm | level2 | quant_locked | loose | 9 | Y | float dtype tokens: ['bfloat16', 'float32']; quant tokens: ['int8', 'w8a8']; 3 bound dim constraint(s); 1 launcher assert(s) |
| L2_32_correct_attn_cp_out | level2 | unknown | loose | 9 | Y | 5 bound dim constraint(s); 2 launcher assert(s) |
| L2_33_moe_mmk | level2 | single_dtype | medium | 9 | Y | float dtype tokens: ['float32']; 4 bound dim constraint(s); 1 tensor_invariant(s); 2 launcher assert(s) |
| L2_34_solve_tril_16x16 | level2 | single_dtype | tight | 9 | - | float dtype tokens: ['float32']; 3 bound dim constraint(s); 2 structural dim constraint(s): ['T % 16 == 0', 'BT == 16'] |
| L2_35_chunk_local_cumsum_vector | level2 | single_dtype | tight | 9 | - | float dtype tokens: ['float32']; 4 bound dim constraint(s); 4 tensor_invariant(s) |
| L3_01_chunk_scan_fwd | level3 | single_dtype | tight | 9 | Y | float dtype tokens: ['float32']; 7 bound dim constraint(s); 1 structural dim constraint(s): ['H % G == 0']; 6 launcher assert(s) |
| L3_02_chunk_state_fwd | level3 | single_dtype | medium | 9 | Y | float dtype tokens: ['float32']; 4 bound dim constraint(s); 4 launcher assert(s) |
| L3_03_chunked_lora_expand | level3 | single_dtype | medium | 9 | Y | float dtype tokens: ['float32']; 4 bound dim constraint(s); 2 tensor_invariant(s) |
| L3_04_qkv_lora_b | level3 | single_dtype | medium | 9 | Y | float dtype tokens: ['float32']; 4 bound dim constraint(s); 2 tensor_invariant(s); 2 launcher assert(s) |
| L3_05_gate_up_lora_b | level3 | single_dtype | medium | 9 | Y | float dtype tokens: ['float32']; 4 bound dim constraint(s); 2 tensor_invariant(s); 2 launcher assert(s) |
| L3_06_sgemm_lora_b | level3 | single_dtype | medium | 9 | Y | float dtype tokens: ['float32']; 4 bound dim constraint(s); 2 tensor_invariant(s) |
| L3_07_fused_sigmoid_gating_delta_rule_update | level3 | single_dtype | loose | 9 | Y | float dtype tokens: ['float32']; 3 bound dim constraint(s) |
| L3_08_causal_conv1d_update | level3 | single_dtype | loose | 9 | Y | float dtype tokens: ['float32']; 4 bound dim constraint(s) |
| L3_09_causal_conv1d_fwd | level3 | single_dtype | loose | 9 | Y | float dtype tokens: ['float32']; 3 bound dim constraint(s); 1 launcher assert(s) |
| L3_10_fused_moe_router_tensorcore | level3 | single_dtype | medium | 9 | Y | float dtype tokens: ['float32']; 2 bound dim constraint(s); 4 launcher assert(s) |
| L3_11_fwd_grouped_kernel_stage1_rope | level3 | single_dtype | loose | 9 | Y | float dtype tokens: ['float32']; 5 bound dim constraint(s) |
| L3_12_selective_state_update | level3 | multi_dtype | medium | 9 | Y | float dtype tokens: ['float16', 'float32']; 5 bound dim constraint(s); 1 structural dim constraint(s): ['H % G == 0'] |
| L3_13_kernel_unified_attention_3d | level3 | quant_locked | tight | 9 | Y | float dtype tokens: ['float32']; quant tokens: ['fp8', 'int8']; 4 bound dim constraint(s); 4 tensor_invariant(s); 1 launcher assert(s) |
| L3_14_kernel_paged_attention_2d | level3 | quant_locked | medium | 9 | Y | float dtype tokens: ['float32']; quant tokens: ['fp8']; 6 bound dim constraint(s); 1 tensor_invariant(s); 1 launcher assert(s) |
| L3_15_chunk_state_varlen | level3 | single_dtype | tight | 9 | Y | float dtype tokens: ['float32']; 5 bound dim constraint(s); 3 tensor_invariant(s); 1 launcher assert(s) |
| L3_16_fwd | level3 | single_dtype | tight | 9 | - | float dtype tokens: ['float32']; 4 bound dim constraint(s); 3 tensor_invariant(s) |
| L3_17_fwd_kernel_alibi | level3 | single_dtype | loose | 9 | Y | float dtype tokens: ['float32']; 3 bound dim constraint(s) |
| L3_18_fwd_grouped_kernel_stage1 | level3 | single_dtype | medium | 10 | Y | float dtype tokens: ['float32']; 8 bound dim constraint(s); 1 structural dim constraint(s): ['H % E == 0'] |
| L3_19_fused_moe | level3 | multi_dtype | medium | 9 | Y | float dtype tokens: ['bfloat16', 'float16', 'float32']; 4 bound dim constraint(s); 2 tensor_invariant(s); 2 launcher assert(s) |
| L3_20_fused_moe_kernel_gptq_awq | level3 | quant_locked | tight | 9 | Y | float dtype tokens: ['bfloat16', 'float32']; quant tokens: ['awq', 'int4']; 5 bound dim constraint(s); 2 tensor_invariant(s); 5 launcher assert(s) |
| L3_21_expert_triton | level3 | multi_dtype | medium | 9 | Y | float dtype tokens: ['bfloat16', 'float16', 'float32']; 4 bound dim constraint(s); 1 tensor_invariant(s); 2 launcher assert(s) |
| L3_22_chunk_gated_delta_rule_fwd_kernel_h_blockdim64 | level3 | single_dtype | medium | 6 | - | float dtype tokens: ['float32']; 7 bound dim constraint(s); 1 structural dim constraint(s): ['K % 64 == 0'] |
| L3_23_fused_recurrent_gated_delta_rule_fwd | level3 | single_dtype | loose | 10 | Y | float dtype tokens: ['float32']; 4 bound dim constraint(s) |
| L3_24_fused_recurrent_gated_delta_rule_update_fwd | level3 | single_dtype | loose | 10 | Y | float dtype tokens: ['float32']; 4 bound dim constraint(s) |
| L3_25_chunk_fwd_kernel_o | level3 | single_dtype | medium | 6 | - | float dtype tokens: ['float32']; 7 bound dim constraint(s); 1 structural dim constraint(s): ['H % Hg == 0'] |
| L3_26_chunk_gla_fwd_kernel_o | level3 | single_dtype | medium | 9 | Y | float dtype tokens: ['float32']; 5 bound dim constraint(s); 1 structural dim constraint(s): ['L % C == 0'] |
| L3_27_chunk_scaled_dot_kkt_fwd | level3 | single_dtype | medium | 6 | - | float dtype tokens: ['float32']; 6 bound dim constraint(s); 1 structural dim constraint(s): ['H % Hg == 0'] |
| L3_28_chunk_kda_scaled_dot_kkt_fwd_kernel_intra_sub_inter | level3 | single_dtype | loose | 9 | Y | float dtype tokens: ['float32']; 6 bound dim constraint(s); 1 launcher assert(s) |
| L3_29_recompute_w_u_fwd | level3 | unknown | tight | 9 | Y | 6 bound dim constraint(s); 2 structural dim constraint(s): ['L % C == 0', 'H % 2 == 0'] |
| L3_30_merge_16x16_to_32x32_inverse | level3 | single_dtype | tight | 9 | - | float dtype tokens: ['float32']; 3 bound dim constraint(s); 2 structural dim constraint(s): ['T % 32 == 0', 'BT == 32'] |
