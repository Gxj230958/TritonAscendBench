# Full GPU oracle-equivalence run — bc2full1 (103 tasks, real CUDA)

**Date 2026-06-01. Real GPU run via daemon; harness = the H1/H3-fixed R10.a+R10.b.**

## Tally

| verdict | n |
|---|---:|
| VERIFIED | 62 |
| FAILED | 22 |
| MIXED | 6 |
| NEEDS_BRIDGE | 5 |
| SIGNATURE_MISMATCH | 4 |
| LAUNCH_BLOCKED | 2 |
| RUNTIME_ERROR | 1 |
| WRAPPER_CONTRACT_MISMATCH | 1 |

**62/103 GPU-VERIFIED** (oracle-equiv baseline was 10 in meta). All 3 new tasks (L2_39/L2_40/L2_41) VERIFIED. The non-VERIFIED are the bench's NEEDS-INVESTIGATION / BLOCKED output to root-cause (oracle-bug vs upstream-bug vs dtype/harness).

## Per-task verdict + worst max_rel

| task | verdict | worst max_rel |
|---|---|---|
| L1_01_bias | FAILED | - |
| L1_02_compute_slot_mappings | VERIFIED | 0.0 |
| L1_03_convert_req_index_to_global_index | VERIFIED | 0.0 |
| L1_04_copy_page_indices | VERIFIED | 0.0 |
| L1_05_gather_block_tables | VERIFIED | 0.0 |
| L1_06_reshape_and_cache_kernel_flash | VERIFIED | 0.0 |
| L1_07_dequantize_k_cache_fast | VERIFIED | 0.0 |
| L1_08_dequantize_k_cache_paged | RUNTIME_ERROR | - |
| L1_09_quantize_k_cache_fast | FAILED | 0.1 |
| L1_10_per_token_group_quant_int8 | VERIFIED | 1.0 |
| L1_11_awq_dequantize | NEEDS_BRIDGE | - |
| L1_12_rms_norm | VERIFIED | 2.409e-07 |
| L1_13_layer_norm_fwd | LAUNCH_BLOCKED | - |
| L1_14_layer_norm_fwd_1pass | FAILED | 0.0003806 |
| L1_15_act_and_mul | SIGNATURE_MISMATCH | - |
| L1_16_silu_and_mul | VERIFIED | 3.519e-07 |
| L1_17_gelu_and_mul | NEEDS_BRIDGE | - |
| L1_18_fused_softcap | VERIFIED | 2.54e-07 |
| L1_19_topk_log_softmax | VERIFIED | 1.184e-07 |
| L1_20_log_softmax | VERIFIED | 1.17e-07 |
| L1_21_gumbel_sample | VERIFIED | 0.0 |
| L1_22_min_p | FAILED | - |
| L1_23_penalties_and_temperature | FAILED | 5.727e-05 |
| L1_24_rejection_sample | FAILED | 0.0 |
| L1_25_merge_attn_states | FAILED | - |
| L1_26_merge_state | VERIFIED | 0.003741 |
| L1_27_apply_grammar_bitmask | VERIFIED | - |
| L1_28_pack_bitmatrix | VERIFIED | 0.0 |
| L1_29_prepare_eagle_inputs | VERIFIED | 0.0 |
| L1_30_act_quant | NEEDS_BRIDGE | - |
| L1_31_triton_mrope_forward | VERIFIED | 6.275e-06 |
| L1_32_create_flashinfer_kv_indices_triton | VERIFIED | 0.0 |
| L1_33_create_flashmla_kv_indices_triton | VERIFIED | 0.0 |
| L1_34_transform_index_page_table_decode | VERIFIED | 0.0 |
| L1_35_compute_src2dst_triton | VERIFIED | 0.0 |
| L1_36_compute_seg_indptr_triton | VERIFIED | 0.0 |
| L1_37_compute_problem_sizes_w4a8 | VERIFIED | 0.0 |
| L1_38_silu_mul_fp8_quant_deep_gemm | VERIFIED | 2.276e-07 |
| L1_39_state_passing_fwd | VERIFIED | 2.066e-05 |
| L1_40_chunk_cumsum_fwd | VERIFIED | 1.971e-07 |
| L2_01_bmm_chunk_fwd | FAILED | 0.2137 |
| L2_02_chunk_scan_fwd | FAILED | 21180.0 |
| L2_03_chunk_state_fwd | VERIFIED | 2.074 |
| L2_04_chunked_lora_expand | FAILED | 0.7617 |
| L2_05_chunked_lora_shrink | FAILED | 0.02226 |
| L2_06_qkv_lora_b | VERIFIED | 0.188 |
| L2_07_gate_up_lora_b | FAILED | 1.925 |
| L2_08_sgemm_lora_a | FAILED | 0.02226 |
| L2_09_sgemm_lora_b | MIXED | 9.816 |
| L2_10_w8a8_block_int8_matmul | VERIFIED | 0.0008396 |
| L2_11_w8a8_triton_block_scaled_mm | VERIFIED | 0.0007027 |
| L2_12_awq_gemm | NEEDS_BRIDGE | - |
| L2_13_bmm | FAILED | 145.0 |
| L2_14_matmul_kernel_persistent | LAUNCH_BLOCKED | - |
| L2_15_fused_dual_residual_rmsnorm | VERIFIED | 0.0001599 |
| L2_16_fused_rmsnorm | VERIFIED | 2.49e-07 |
| L2_17_layer_norm_gated_fwd | SIGNATURE_MISMATCH | - |
| L2_18_fused_gdn_gating | VERIFIED | 2.369e-06 |
| L2_19_fused_sigmoid_gating_delta_rule_update | VERIFIED | 0.009829 |
| L2_20_causal_conv1d_update | WRAPPER_CONTRACT_MISMATCH | - |
| L2_21_causal_conv1d_fwd | VERIFIED | 0.002108 |
| L2_22_linear_attn_decode | VERIFIED | 0.001061 |
| L2_23_fwd_kv_parallel | SIGNATURE_MISMATCH | - |
| L2_24_fwd_kv_reduce | SIGNATURE_MISMATCH | - |
| L2_25_fwd_kernel_ep_gather | VERIFIED | 0.0001528 |
| L2_26_fwd_kernel_ep_scatter_1 | VERIFIED | 0.0 |
| L2_27_deepep_permute_triton | VERIFIED | 0.0 |
| L2_28_deepep_post_reorder_triton | VERIFIED | 0.000209 |
| L2_29_fused_moe_router_cudacore | VERIFIED | 7.712e-06 |
| L2_30_fused_moe_router_tensorcore | FAILED | 6000000000000.0 |
| L2_31_moe_align_block_size_stage2 | VERIFIED | 0.0 |
| L2_32_experts_combine | VERIFIED | 0.001259 |
| L2_33_fwd_grouped_kernel_stage1_rope | NEEDS_BRIDGE | - |
| L2_34_silu_mul_static_tensorwise_quant_triton_kernel_for_cutlass_moe | MIXED | 0.1 |
| L2_35_pre_reorder_triton_kernel_for_cutlass_moe | VERIFIED | 0.0 |
| L2_36_post_reorder_triton_kernel_for_cutlass_moe | VERIFIED | 7.459e-05 |
| L2_37_silu_and_mul_post_quant | MIXED | 0.1 |
| L2_38_concat_and_cast_mha_k | VERIFIED | 0.0 |
| L2_39_scaled_mm | VERIFIED | 0.0 |
| L2_40_gemma_qkv_rmsnorm | VERIFIED | 3.094e-07 |
| L2_41_selective_state_update | VERIFIED | 0.000247 |
| L3_01_kernel_unified_attention_3d | VERIFIED | 422.8 |
| L3_02_kernel_paged_attention_2d | VERIFIED | 0.1567 |
| L3_03_chunk_state_varlen | MIXED | 21.89 |
| L3_04_fwd | FAILED | - |
| L3_05_fwd_kernel_alibi | VERIFIED | 0.01611 |
| L3_06_fwd_grouped_kernel_stage1 | VERIFIED | 0.6431 |
| L3_07_correct_attn_cp_out | VERIFIED | 3.868e-07 |
| L3_09_fused_moe | FAILED | 0.5253 |
| L3_10_fused_moe_kernel_gptq_awq | FAILED | 1.064 |
| L3_11_expert_triton | FAILED | 6.371 |
| L3_12_moe_mmk | FAILED | 6.371 |
| L3_13_chunk_gated_delta_rule_fwd_kernel_h_blockdim64 | FAILED | 264.9 |
| L3_14_fused_recurrent_gated_delta_rule_fwd | VERIFIED | 0.005107 |
| L3_15_fused_recurrent_gated_delta_rule_update_fwd | VERIFIED | 0.01518 |
| L3_16_chunk_fwd_kernel_o | FAILED | 36.49 |
| L3_17_chunk_gla_fwd_kernel_o | MIXED | 26.66 |
| L3_18_chunk_scaled_dot_kkt_fwd | VERIFIED | 0.004593 |
| L3_19_chunk_kda_scaled_dot_kkt_fwd_kernel_intra_sub_inter | VERIFIED | 0.03349 |
| L3_20_recompute_w_u_fwd | MIXED | 0.1014 |
| L3_21_solve_tril_16x16 | VERIFIED | 5.029e-05 |
| L3_22_chunk_local_cumsum_vector | VERIFIED | 0.1 |
| L3_23_merge_16x16_to_32x32_inverse | VERIFIED | 0.0001089 |

## FAILED/MIXED triage buckets

- near-tolerance (<0.05; precision/dtype boundary): L1_14, L1_23, L1_24, L2_05, L2_08
- large-divergence (>=0.05; real mismatch, root-cause needed): L2_30(6e12), L2_02(2e4), L3_13(265), L2_13(145), L3_16(36), L3_17(27), L3_03(22), L2_09(9.8), L3_11/12(6.4), L2_07(1.9), L3_10(1.1), L2_04(0.76), L3_09(0.53), L2_01(0.21), L3_20(0.10), L1_09(0.10), L2_34(0.10), L2_37(0.10)
- many large-divergence are GEMM/dot/SSM-chunk (the dtype-sensitive tl.dot tasks).

## Next phase (NOT done — honest)
Per-task root-cause each FAILED/MIXED: is OUR oracle wrong (fix oracle, re-run), the upstream kernel buggy (honest BLOCKED, like L2_23/24), or a dtype/harness issue? Each with codex co-review (no self-classification). NEEDS_BRIDGE/SIGNATURE_MISMATCH = shapes/schema gaps to bridge.