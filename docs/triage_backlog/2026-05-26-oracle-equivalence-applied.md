---
date: 2026-06-02
tool: scripts/apply_oracle_equivalence_verdict.py
purpose: V9 M2.1c — apply R10 GPU verdict to meta.yaml provenance.golden
verdict-dir: ../loop_mailbox/results/b-fullrerun-20260602T082906/r10
---

# Oracle Equivalence Verdict Apply Report

Tasks processed: **103**  
Promoted to truth source (`oracle_pytorch_verified`): **100**

## Overall verdict tally

| overall | count |
|---|---|
| `VERIFIED` | 100 |
| `BLOCKED_UPSTREAM_BUG` | 2 |
| `NEEDS_BRIDGE` | 1 |

## VERIFIED (100)

### `L1_01_bias`
- **oracle_pytorch.py promoted to truth source**
- per-shape numeric diff:
  - `P1`: max_abs=0.000e+00, max_rel=0.000e+00
  - `P2`: max_abs=0.000e+00, max_rel=0.000e+00

### `L1_02_compute_slot_mappings`
- **oracle_pytorch.py promoted to truth source**
- per-shape numeric diff:
  - `P1`: max_abs=0.000e+00, max_rel=0.000e+00
  - `P2`: max_abs=0.000e+00, max_rel=0.000e+00

### `L1_03_convert_req_index_to_global_index`
- **oracle_pytorch.py promoted to truth source**
- per-shape numeric diff:
  - `P1`: max_abs=0.000e+00, max_rel=0.000e+00
  - `P2`: max_abs=0.000e+00, max_rel=0.000e+00

### `L1_04_copy_page_indices`
- **oracle_pytorch.py promoted to truth source**
- per-shape numeric diff:
  - `P1`: max_abs=0.000e+00, max_rel=0.000e+00
  - `P2`: max_abs=0.000e+00, max_rel=0.000e+00

### `L1_05_gather_block_tables`
- **oracle_pytorch.py promoted to truth source**
- per-shape numeric diff:
  - `P1`: max_abs=0.000e+00, max_rel=0.000e+00
  - `P2`: max_abs=0.000e+00, max_rel=0.000e+00

### `L1_06_reshape_and_cache_kernel_flash`
- **oracle_pytorch.py promoted to truth source**
- per-shape numeric diff:
  - `P1`: max_abs=0.000e+00, max_rel=0.000e+00
  - `P2`: max_abs=0.000e+00, max_rel=0.000e+00

### `L1_07_dequantize_k_cache_fast`
- **oracle_pytorch.py promoted to truth source**
- per-shape numeric diff:
  - `P1`: max_abs=0.000e+00, max_rel=0.000e+00
  - `P2`: max_abs=0.000e+00, max_rel=0.000e+00

### `L1_08_dequantize_k_cache_paged`
- **oracle_pytorch.py promoted to truth source**
- per-shape numeric diff:
  - `P1`: max_abs=0.000e+00, max_rel=0.000e+00
  - `P2`: max_abs=0.000e+00, max_rel=0.000e+00

### `L1_09_quantize_k_cache_fast`
- **oracle_pytorch.py promoted to truth source**
- per-shape numeric diff:
  - `P1`: max_abs=3.200e+01, max_rel=1.000e-01
  - `P2`: max_abs=3.200e+01, max_rel=1.000e-01

### `L1_10_per_token_group_quant_int8`
- **oracle_pytorch.py promoted to truth source**
- per-shape numeric diff:
  - `P1`: max_abs=1.000e+00, max_rel=1.000e+00
  - `P2`: max_abs=1.000e+00, max_rel=1.000e+00

### `L1_11_awq_dequantize`
- **oracle_pytorch.py promoted to truth source**
- per-shape numeric diff:
  - `P1`: max_abs=0.000e+00, max_rel=0.000e+00
  - `P2`: max_abs=0.000e+00, max_rel=0.000e+00

### `L1_12_rms_norm`
- **oracle_pytorch.py promoted to truth source**
- per-shape numeric diff:
  - `P1`: max_abs=0.000e+00, max_rel=0.000e+00
  - `P2`: max_abs=0.000e+00, max_rel=0.000e+00

### `L1_13_layer_norm_fwd`
- **oracle_pytorch.py promoted to truth source**
- per-shape numeric diff:
  - `P1`: max_abs=0.000e+00, max_rel=0.000e+00
  - `P2`: max_abs=5.960e-08, max_rel=5.376e-03

### `L1_14_layer_norm_fwd_1pass`
- **oracle_pytorch.py promoted to truth source**
- per-shape numeric diff:
  - `P1`: max_abs=0.000e+00, max_rel=0.000e+00
  - `P2`: max_abs=0.000e+00, max_rel=0.000e+00

### `L1_15_act_and_mul`
- **oracle_pytorch.py promoted to truth source**
- per-shape numeric diff:
  - `P1`: max_abs=0.000e+00, max_rel=0.000e+00
  - `P2`: max_abs=0.000e+00, max_rel=0.000e+00

### `L1_16_silu_and_mul`
- **oracle_pytorch.py promoted to truth source**
- per-shape numeric diff:
  - `P1`: max_abs=0.000e+00, max_rel=0.000e+00
  - `P2`: max_abs=0.000e+00, max_rel=0.000e+00

### `L1_17_gelu_and_mul`
- **oracle_pytorch.py promoted to truth source**
- per-shape numeric diff:
  - `P1`: max_abs=0.000e+00, max_rel=0.000e+00
  - `P2`: max_abs=0.000e+00, max_rel=0.000e+00

### `L1_18_fused_softcap`
- **oracle_pytorch.py promoted to truth source**
- per-shape numeric diff:
  - `P1`: max_abs=9.537e-07, max_rel=2.413e-07
  - `P2`: max_abs=9.537e-07, max_rel=2.540e-07

### `L1_19_topk_log_softmax`
- **oracle_pytorch.py promoted to truth source**
- per-shape numeric diff:
  - `P1`: max_abs=4.768e-07, max_rel=1.138e-07
  - `P2`: max_abs=9.537e-07, max_rel=1.184e-07

### `L1_20_log_softmax`
- **oracle_pytorch.py promoted to truth source**
- per-shape numeric diff:
  - `P1`: max_abs=0.000e+00, max_rel=0.000e+00
  - `P2`: max_abs=9.537e-07, max_rel=1.170e-07

### `L1_21_gumbel_sample`
- **oracle_pytorch.py promoted to truth source**
- per-shape numeric diff:
  - `P1`: max_abs=0.000e+00, max_rel=0.000e+00
  - `P2`: max_abs=0.000e+00, max_rel=0.000e+00

### `L1_22_min_p`
- **oracle_pytorch.py promoted to truth source**
- per-shape numeric diff:
  - `P1`: max_abs=0.000e+00, max_rel=0.000e+00
  - `P2`: max_abs=0.000e+00, max_rel=0.000e+00

### `L1_23_penalties_and_temperature`
- **oracle_pytorch.py promoted to truth source**
- per-shape numeric diff:
  - `P1`: max_abs=2.384e-07, max_rel=5.812e-06
  - `P2`: max_abs=1.907e-06, max_rel=5.727e-05

### `L1_24_rejection_sample`
- **oracle_pytorch.py promoted to truth source**
- per-shape numeric diff:
  - `P1`: max_abs=0.000e+00, max_rel=0.000e+00
  - `P2`: max_abs=0.000e+00, max_rel=0.000e+00

### `L1_25_merge_attn_states`
- **oracle_pytorch.py promoted to truth source**
- per-shape numeric diff:
  - `P1`: max_abs=0.000e+00, max_rel=0.000e+00
  - `P2`: max_abs=2.384e-07, max_rel=8.947e-07

### `L1_26_merge_state`
- **oracle_pytorch.py promoted to truth source**
- per-shape numeric diff:
  - `P1`: max_abs=1.192e-07, max_rel=3.345e-06
  - `P2`: max_abs=2.441e-04, max_rel=7.692e-03

### `L1_27_apply_grammar_bitmask`
- **oracle_pytorch.py promoted to truth source**
- per-shape numeric diff:
  - `P1`: max_abs=nan, max_rel=nan
  - `P2`: max_abs=nan, max_rel=nan

### `L1_28_pack_bitmatrix`
- **oracle_pytorch.py promoted to truth source**
- per-shape numeric diff:
  - `P1`: max_abs=0.000e+00, max_rel=0.000e+00
  - `P2`: max_abs=0.000e+00, max_rel=0.000e+00

### `L1_29_prepare_eagle_inputs`
- **oracle_pytorch.py promoted to truth source**
- per-shape numeric diff:
  - `P1`: max_abs=0.000e+00, max_rel=0.000e+00
  - `P2`: max_abs=0.000e+00, max_rel=0.000e+00

### `L1_30_act_quant`
- **oracle_pytorch.py promoted to truth source**
- per-shape numeric diff:
  - `P1`: max_abs=0.000e+00, max_rel=0.000e+00
  - `P2`: max_abs=0.000e+00, max_rel=0.000e+00

### `L1_31_triton_mrope_forward`
- **oracle_pytorch.py promoted to truth source**
- per-shape numeric diff:
  - `P1`: max_abs=0.000e+00, max_rel=0.000e+00
  - `P2`: max_abs=0.000e+00, max_rel=0.000e+00

### `L1_32_create_flashinfer_kv_indices_triton`
- **oracle_pytorch.py promoted to truth source**
- per-shape numeric diff:
  - `P1`: max_abs=0.000e+00, max_rel=0.000e+00
  - `P2`: max_abs=0.000e+00, max_rel=0.000e+00

### `L1_33_create_flashmla_kv_indices_triton`
- **oracle_pytorch.py promoted to truth source**
- per-shape numeric diff:
  - `P1`: max_abs=0.000e+00, max_rel=0.000e+00
  - `P2`: max_abs=0.000e+00, max_rel=0.000e+00

### `L1_34_transform_index_page_table_decode`
- **oracle_pytorch.py promoted to truth source**
- per-shape numeric diff:
  - `P1`: max_abs=0.000e+00, max_rel=0.000e+00
  - `P2`: max_abs=0.000e+00, max_rel=0.000e+00

### `L1_35_compute_src2dst_triton`
- **oracle_pytorch.py promoted to truth source**
- per-shape numeric diff:
  - `P1`: max_abs=0.000e+00, max_rel=0.000e+00
  - `P2`: max_abs=0.000e+00, max_rel=0.000e+00

### `L1_36_compute_seg_indptr_triton`
- **oracle_pytorch.py promoted to truth source**
- per-shape numeric diff:
  - `P1`: max_abs=0.000e+00, max_rel=0.000e+00
  - `P2`: max_abs=0.000e+00, max_rel=0.000e+00

### `L1_37_compute_problem_sizes_w4a8`
- **oracle_pytorch.py promoted to truth source**
- per-shape numeric diff:
  - `P1`: max_abs=0.000e+00, max_rel=0.000e+00
  - `P2`: max_abs=0.000e+00, max_rel=0.000e+00

### `L1_38_silu_mul_fp8_quant_deep_gemm`
- **oracle_pytorch.py promoted to truth source**
- per-shape numeric diff:
  - `P1`: max_abs=9.313e-10, max_rel=8.890e-08
  - `P2`: max_abs=1.863e-09, max_rel=2.304e-07

### `L1_39_state_passing_fwd`
- **oracle_pytorch.py promoted to truth source**
- per-shape numeric diff:
  - `P1`: max_abs=9.537e-07, max_rel=2.227e-06
  - `P2`: max_abs=1.907e-06, max_rel=2.066e-05

### `L1_40_chunk_cumsum_fwd`
- **oracle_pytorch.py promoted to truth source**
- per-shape numeric diff:
  - `P1`: max_abs=9.537e-07, max_rel=1.192e-07
  - `P2`: max_abs=9.537e-07, max_rel=1.713e-07

### `L2_01_bmm_chunk_fwd`
- **oracle_pytorch.py promoted to truth source**
- per-shape numeric diff:
  - `P1`: max_abs=0.000e+00, max_rel=0.000e+00
  - `P2`: max_abs=0.000e+00, max_rel=0.000e+00

### `L2_02_chunk_scan_fwd`
- **oracle_pytorch.py promoted to truth source**
- per-shape numeric diff:
  - `P1`: max_abs=1.526e-05, max_rel=8.432e-05
  - `P2`: max_abs=3.815e-05, max_rel=1.158e-03

### `L2_03_chunk_state_fwd`
- **oracle_pytorch.py promoted to truth source**
- per-shape numeric diff:
  - `P1`: max_abs=2.293e-02, max_rel=1.358e-01
  - `P2`: max_abs=6.475e-02, max_rel=2.138e+00

### `L2_04_chunked_lora_expand`
- **oracle_pytorch.py promoted to truth source**
- per-shape numeric diff:
  - `P1`: max_abs=0.000e+00, max_rel=0.000e+00
  - `P2`: max_abs=0.000e+00, max_rel=0.000e+00

### `L2_05_chunked_lora_shrink`
- **oracle_pytorch.py promoted to truth source**
- per-shape numeric diff:
  - `P1`: max_abs=0.000e+00, max_rel=0.000e+00
  - `P2`: max_abs=0.000e+00, max_rel=0.000e+00

### `L2_06_qkv_lora_b`
- **oracle_pytorch.py promoted to truth source**
- per-shape numeric diff:
  - `P1`: max_abs=0.000e+00, max_rel=0.000e+00
  - `P2`: max_abs=0.000e+00, max_rel=0.000e+00

### `L2_07_gate_up_lora_b`
- **oracle_pytorch.py promoted to truth source**
- per-shape numeric diff:
  - `P1`: max_abs=0.000e+00, max_rel=0.000e+00
  - `P2`: max_abs=0.000e+00, max_rel=0.000e+00

### `L2_08_sgemm_lora_a`
- **oracle_pytorch.py promoted to truth source**
- per-shape numeric diff:
  - `P1`: max_abs=0.000e+00, max_rel=0.000e+00
  - `P2`: max_abs=0.000e+00, max_rel=0.000e+00

### `L2_09_sgemm_lora_b`
- **oracle_pytorch.py promoted to truth source**
- per-shape numeric diff:
  - `P1`: max_abs=0.000e+00, max_rel=0.000e+00
  - `P2`: max_abs=0.000e+00, max_rel=0.000e+00

### `L2_10_w8a8_block_int8_matmul`
- **oracle_pytorch.py promoted to truth source**
- per-shape numeric diff:
  - `P1`: max_abs=0.000e+00, max_rel=0.000e+00
  - `P2`: max_abs=1.000e+00, max_rel=8.396e-04

### `L2_11_w8a8_triton_block_scaled_mm`
- **oracle_pytorch.py promoted to truth source**
- per-shape numeric diff:
  - `P1`: max_abs=0.000e+00, max_rel=0.000e+00
  - `P2`: max_abs=5.000e-01, max_rel=7.027e-04

### `L2_12_awq_gemm`
- **oracle_pytorch.py promoted to truth source**
- per-shape numeric diff:
  - `P1`: max_abs=3.815e-05, max_rel=6.548e-05
  - `P2`: max_abs=1.526e-04, max_rel=3.129e-04

### `L2_13_bmm`
- **oracle_pytorch.py promoted to truth source**
- per-shape numeric diff:
  - `P1`: max_abs=9.537e-06, max_rel=3.289e-04
  - `P2`: max_abs=4.196e-05, max_rel=1.628e-01

### `L2_14_matmul_kernel_persistent`
- **oracle_pytorch.py promoted to truth source**
- per-shape numeric diff:
  - `P1`: max_abs=5.722e-06, max_rel=3.212e-05
  - `P2`: max_abs=3.052e-05, max_rel=1.430e-03

### `L2_15_fused_dual_residual_rmsnorm`
- **oracle_pytorch.py promoted to truth source**
- per-shape numeric diff:
  - `P1`: max_abs=0.000e+00, max_rel=0.000e+00
  - `P2`: max_abs=0.000e+00, max_rel=0.000e+00

### `L2_16_fused_rmsnorm`
- **oracle_pytorch.py promoted to truth source**
- per-shape numeric diff:
  - `P1`: max_abs=0.000e+00, max_rel=0.000e+00
  - `P2`: max_abs=0.000e+00, max_rel=0.000e+00

### `L2_17_layer_norm_gated_fwd`
- **oracle_pytorch.py promoted to truth source**
- per-shape numeric diff:
  - `P1`: max_abs=0.000e+00, max_rel=0.000e+00
  - `P2`: max_abs=0.000e+00, max_rel=0.000e+00

### `L2_18_fused_gdn_gating`
- **oracle_pytorch.py promoted to truth source**
- per-shape numeric diff:
  - `P1`: max_abs=4.768e-07, max_rel=2.281e-07
  - `P2`: max_abs=3.815e-06, max_rel=1.355e-06

### `L2_19_fused_sigmoid_gating_delta_rule_update`
- **oracle_pytorch.py promoted to truth source**
- per-shape numeric diff:
  - `P1`: max_abs=3.815e-06, max_rel=1.912e-04
  - `P2`: max_abs=1.526e-05, max_rel=3.328e-01

### `L2_20_causal_conv1d_update`
- **oracle_pytorch.py promoted to truth source**
- per-shape numeric diff:
  - `P1`: max_abs=0.000e+00, max_rel=0.000e+00
  - `P2`: max_abs=0.000e+00, max_rel=0.000e+00

### `L2_21_causal_conv1d_fwd`
- **oracle_pytorch.py promoted to truth source**
- per-shape numeric diff:
  - `P1`: max_abs=0.000e+00, max_rel=0.000e+00
  - `P2`: max_abs=0.000e+00, max_rel=0.000e+00

### `L2_22_linear_attn_decode`
- **oracle_pytorch.py promoted to truth source**
- per-shape numeric diff:
  - `P1`: max_abs=3.815e-06, max_rel=3.823e-04
  - `P2`: max_abs=9.537e-06, max_rel=1.061e-03

### `L2_25_fwd_kernel_ep_gather`
- **oracle_pytorch.py promoted to truth source**
- per-shape numeric diff:
  - `P1`: max_abs=0.000e+00, max_rel=0.000e+00
  - `P2`: max_abs=0.000e+00, max_rel=0.000e+00

### `L2_26_fwd_kernel_ep_scatter_1`
- **oracle_pytorch.py promoted to truth source**
- per-shape numeric diff:
  - `P1`: max_abs=0.000e+00, max_rel=0.000e+00
  - `P2`: max_abs=0.000e+00, max_rel=0.000e+00

### `L2_27_deepep_permute_triton`
- **oracle_pytorch.py promoted to truth source**
- per-shape numeric diff:
  - `P1`: max_abs=0.000e+00, max_rel=0.000e+00
  - `P2`: max_abs=0.000e+00, max_rel=0.000e+00

### `L2_28_deepep_post_reorder_triton`
- **oracle_pytorch.py promoted to truth source**
- per-shape numeric diff:
  - `P1`: max_abs=2.384e-07, max_rel=1.837e-06
  - `P2`: max_abs=2.384e-07, max_rel=2.090e-04

### `L2_29_fused_moe_router_cudacore`
- **oracle_pytorch.py promoted to truth source**
- per-shape numeric diff:
  - `P1`: max_abs=5.960e-07, max_rel=3.059e-06
  - `P2`: max_abs=1.758e-06, max_rel=1.259e-05

### `L2_30_fused_moe_router_tensorcore`
- **oracle_pytorch.py promoted to truth source**
- per-shape numeric diff:
  - `P1`: max_abs=4.172e-07, max_rel=1.188e-06
  - `P2`: max_abs=1.669e-06, max_rel=1.717e-05

### `L2_31_moe_align_block_size_stage2`
- **oracle_pytorch.py promoted to truth source**
- per-shape numeric diff:
  - `P1`: max_abs=0.000e+00, max_rel=0.000e+00
  - `P2`: max_abs=0.000e+00, max_rel=0.000e+00

### `L2_32_experts_combine`
- **oracle_pytorch.py promoted to truth source**
- per-shape numeric diff:
  - `P1`: max_abs=4.768e-07, max_rel=1.192e-07
  - `P2`: max_abs=9.537e-07, max_rel=1.259e-03

### `L2_34_silu_mul_static_tensorwise_quant_triton_kernel_for_cutlass_moe`
- **oracle_pytorch.py promoted to truth source**
- per-shape numeric diff:
  - `P1`: max_abs=1.863e-09, max_rel=2.796e-07
  - `P2`: max_abs=3.200e+01, max_rel=1.000e-01

### `L2_35_pre_reorder_triton_kernel_for_cutlass_moe`
- **oracle_pytorch.py promoted to truth source**
- per-shape numeric diff:
  - `P1`: max_abs=0.000e+00, max_rel=0.000e+00
  - `P2`: max_abs=0.000e+00, max_rel=0.000e+00

### `L2_36_post_reorder_triton_kernel_for_cutlass_moe`
- **oracle_pytorch.py promoted to truth source**
- per-shape numeric diff:
  - `P1`: max_abs=0.000e+00, max_rel=0.000e+00
  - `P2`: max_abs=0.000e+00, max_rel=0.000e+00

### `L2_37_silu_and_mul_post_quant`
- **oracle_pytorch.py promoted to truth source**
- per-shape numeric diff:
  - `P1`: max_abs=1.863e-09, max_rel=2.796e-07
  - `P2`: max_abs=3.200e+01, max_rel=1.000e-01

### `L2_38_concat_and_cast_mha_k`
- **oracle_pytorch.py promoted to truth source**
- per-shape numeric diff:
  - `P1`: max_abs=0.000e+00, max_rel=0.000e+00
  - `P2`: max_abs=0.000e+00, max_rel=0.000e+00

### `L2_39_scaled_mm`
- **oracle_pytorch.py promoted to truth source**
- per-shape numeric diff:
  - `P1`: max_abs=0.000e+00, max_rel=0.000e+00
  - `P2`: max_abs=0.000e+00, max_rel=0.000e+00

### `L2_40_gemma_qkv_rmsnorm`
- **oracle_pytorch.py promoted to truth source**
- per-shape numeric diff:
  - `P1`: max_abs=9.537e-07, max_rel=2.315e-07
  - `P2`: max_abs=9.537e-07, max_rel=3.094e-07

### `L2_41_selective_state_update`
- **oracle_pytorch.py promoted to truth source**
- per-shape numeric diff:
  - `P1`: max_abs=3.815e-06, max_rel=4.154e-05
  - `P2`: max_abs=1.144e-05, max_rel=2.470e-04

### `L3_01_kernel_unified_attention_3d`
- **oracle_pytorch.py promoted to truth source**
- per-shape numeric diff:
  - `P1`: max_abs=2.302e-03, max_rel=2.386e+02
  - `P2`: max_abs=3.694e-03, max_rel=2.133e+03

### `L3_02_kernel_paged_attention_2d`
- **oracle_pytorch.py promoted to truth source**
- per-shape numeric diff:
  - `P1`: max_abs=9.445e-04, max_rel=1.482e-02
  - `P2`: max_abs=1.590e-03, max_rel=1.567e-01

### `L3_03_chunk_state_varlen`
- **oracle_pytorch.py promoted to truth source**
- per-shape numeric diff:
  - `P1`: max_abs=1.907e-06, max_rel=1.196e-04
  - `P2`: max_abs=2.033e-03, max_rel=6.703e-02

### `L3_04_fwd`
- **oracle_pytorch.py promoted to truth source**
- per-shape numeric diff:
  - `P1`: max_abs=2.441e-03, max_rel=1.599e+01
  - `P2`: max_abs=5.859e-03, max_rel=2.179e+03

### `L3_05_fwd_kernel_alibi`
- **oracle_pytorch.py promoted to truth source**
- per-shape numeric diff:
  - `P1`: max_abs=9.766e-04, max_rel=4.795e-01
  - `P2`: max_abs=1.953e-03, max_rel=6.656e+00

### `L3_06_fwd_grouped_kernel_stage1`
- **oracle_pytorch.py promoted to truth source**
- per-shape numeric diff:
  - `P1`: max_abs=2.474e-03, max_rel=8.009e-01
  - `P2`: max_abs=3.032e-03, max_rel=5.964e+00

### `L3_07_correct_attn_cp_out`
- **oracle_pytorch.py promoted to truth source**
- per-shape numeric diff:
  - `P1`: max_abs=0.000e+00, max_rel=0.000e+00
  - `P2`: max_abs=0.000e+00, max_rel=0.000e+00

### `L3_09_fused_moe`
- **oracle_pytorch.py promoted to truth source**
- per-shape numeric diff:
  - `P1`: max_abs=0.000e+00, max_rel=0.000e+00
  - `P2`: max_abs=1.221e-04, max_rel=5.051e-03

### `L3_10_fused_moe_kernel_gptq_awq`
- **oracle_pytorch.py promoted to truth source**
- per-shape numeric diff:
  - `P1`: max_abs=1.056e-01, max_rel=6.648e-02
  - `P2`: max_abs=1.750e-01, max_rel=1.064e+00

### `L3_11_expert_triton`
- **oracle_pytorch.py promoted to truth source**
- per-shape numeric diff:
  - `P1`: max_abs=1.144e-05, max_rel=1.795e-04
  - `P2`: max_abs=7.629e-05, max_rel=3.819e-03

### `L3_12_moe_mmk`
- **oracle_pytorch.py promoted to truth source**
- per-shape numeric diff:
  - `P1`: max_abs=1.144e-05, max_rel=1.795e-04
  - `P2`: max_abs=7.629e-05, max_rel=3.819e-03

### `L3_13_chunk_gated_delta_rule_fwd_kernel_h_blockdim64`
- **oracle_pytorch.py promoted to truth source**
- per-shape numeric diff:
  - `P1`: max_abs=8.128e-03, max_rel=9.389e-02
  - `P2`: max_abs=9.095e-03, max_rel=1.125e+01

### `L3_14_fused_recurrent_gated_delta_rule_fwd`
- **oracle_pytorch.py promoted to truth source**
- per-shape numeric diff:
  - `P1`: max_abs=9.537e-07, max_rel=4.630e-03
  - `P2`: max_abs=2.500e-01, max_rel=4.425e-03

### `L3_15_fused_recurrent_gated_delta_rule_update_fwd`
- **oracle_pytorch.py promoted to truth source**
- per-shape numeric diff:
  - `P1`: max_abs=3.052e-05, max_rel=3.984e-03
  - `P2`: max_abs=4.883e-04, max_rel=2.846e-02

### `L3_16_chunk_fwd_kernel_o`
- **oracle_pytorch.py promoted to truth source**
- per-shape numeric diff:
  - `P1`: max_abs=7.849e-04, max_rel=7.662e-04
  - `P2`: max_abs=5.658e-03, max_rel=4.381e-01

### `L3_17_chunk_gla_fwd_kernel_o`
- **oracle_pytorch.py promoted to truth source**
- per-shape numeric diff:
  - `P1`: max_abs=8.345e-06, max_rel=2.331e-03
  - `P2`: max_abs=6.502e-04, max_rel=2.113e-02

### `L3_18_chunk_scaled_dot_kkt_fwd`
- **oracle_pytorch.py promoted to truth source**
- per-shape numeric diff:
  - `P1`: max_abs=3.815e-06, max_rel=1.128e-04
  - `P2`: max_abs=1.526e-05, max_rel=8.037e-04

### `L3_19_chunk_kda_scaled_dot_kkt_fwd_kernel_intra_sub_inter`
- **oracle_pytorch.py promoted to truth source**
- per-shape numeric diff:
  - `P1`: max_abs=5.722e-06, max_rel=3.364e-03
  - `P2`: max_abs=1.793e-03, max_rel=4.348e-02

### `L3_20_recompute_w_u_fwd`
- **oracle_pytorch.py promoted to truth source**
- per-shape numeric diff:
  - `P1`: max_abs=8.315e-06, max_rel=6.846e-04
  - `P2`: max_abs=4.531e-03, max_rel=1.014e-01

### `L3_21_solve_tril_16x16`
- **oracle_pytorch.py promoted to truth source**
- per-shape numeric diff:
  - `P1`: max_abs=3.815e-06, max_rel=5.128e-06
  - `P2`: max_abs=4.578e-05, max_rel=5.029e-05

### `L3_22_chunk_local_cumsum_vector`
- **oracle_pytorch.py promoted to truth source**
- per-shape numeric diff:
  - `P1`: max_abs=1.192e-06, max_rel=6.336e-05
  - `P2`: max_abs=1.907e-06, max_rel=1.000e-01

### `L3_23_merge_16x16_to_32x32_inverse`
- **oracle_pytorch.py promoted to truth source**
- per-shape numeric diff:
  - `P1`: max_abs=4.395e-03, max_rel=4.103e-05
  - `P2`: max_abs=3.174e-03, max_rel=1.089e-04
