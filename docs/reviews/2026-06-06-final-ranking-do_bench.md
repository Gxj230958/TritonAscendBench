# Final ranking — do_bench device-only timing on NVIDIA H200 (2026-06-06)

Source: brankall5 (do_bench) vs brankall4 (whole-operator host-wall). All 101 tasks ran their real triton kernel via the trusted-operator path (`oracle_equivalence_gpu_runner._verify_one_task`, TAB_RANK_TIMING=1).

- **95 RANKED + 6 FLOOR = 101/101 have a result**

- timing_method: {'do_bench': 101} (0 fallback)

- cost_model: {'BLOCKED': 41, 'PASS': 54, 'FLOOR_DOMINATED': 6}

- s_t do_bench/host-wall: median 2.86x, max 7.72x, min 0.71x

`e = t_ref/t_npu` (Roofline); s_t = geomean per-shape e; best_e = highest per-shape e (kernel-saturated shape). `*` = cost_model not ncu-xcheck-PASS. do_bench: continuous launches no per-iter sync + L2 flush (docs/reviews/2026-06-06-device-timing-methodology.md).

| # | task | overall | corr | shapes | s_t (do_bench) | s_t (host-wall) | gain | best_e | cost_model |
|---|------|---------|------|--------|----------------|-----------------|------|--------|------------|
| 1 | L3_20_recompute_w_u_fwd | RANKED | CEIL | 9/9 | 1.24e-01 | 5.70e-02 | 2.2x | 56.7% | BLOCKED * |
| 2 | L3_13_chunk_gated_delta_rule_fwd_kernel_h_blockdim64 | RANKED | CEIL | 6/6 | 9.43e-02 | 3.13e-02 | 3.0x | 34.9% | BLOCKED * |
| 3 | L3_17_chunk_gla_fwd_kernel_o | RANKED | VERIF | 9/9 | 7.98e-02 | 2.56e-02 | 3.1x | 36.3% | BLOCKED * |
| 4 | L2_13_bmm | RANKED | VERIF | 9/9 | 7.57e-02 | 4.21e-02 | 1.8x | 53.6% | PASS |
| 5 | L3_16_chunk_fwd_kernel_o | RANKED | CEIL | 6/6 | 4.68e-02 | 1.19e-02 | 3.9x | 25.2% | PASS |
| 6 | L3_22_chunk_local_cumsum_vector | RANKED | VERIF | 9/9 | 3.70e-02 | 7.13e-03 | 5.2x | 75.5% | BLOCKED * |
| 7 | L2_22_linear_attn_decode | RANKED | VERIF | 9/9 | 3.63e-02 | 1.62e-02 | 2.2x | 66.8% | BLOCKED * |
| 8 | L3_04_fwd | RANKED | CEIL | 9/9 | 3.47e-02 | 1.79e-02 | 1.9x | 13.9% | BLOCKED * |
| 9 | L3_11_expert_triton | RANKED | VERIF | 9/9 | 3.11e-02 | 1.68e-02 | 1.9x | 35.2% | PASS |
| 10 | L1_26_merge_state | RANKED | CEIL | 9/9 | 2.68e-02 | 1.00e-02 | 2.7x | 80.6% | PASS |
| 11 | L3_12_moe_mmk | RANKED | VERIF | 9/9 | 2.64e-02 | 1.08e-02 | 2.4x | 16.7% | PASS |
| 12 | L3_02_kernel_paged_attention_2d | RANKED | VERIF | 9/9 | 1.81e-02 | 1.59e-02 | 1.1x | 20.3% | BLOCKED * |
| 13 | L3_19_chunk_kda_scaled_dot_kkt_fwd_kernel_intra_sub_inter | RANKED | VERIF | 9/9 | 1.17e-02 | 4.02e-03 | 2.9x | 6.2% | PASS |
| 14 | L2_38_concat_and_cast_mha_k | RANKED | VERIF | 9/9 | 1.16e-02 | 3.42e-03 | 3.4x | 55.5% | BLOCKED * |
| 15 | L3_15_fused_recurrent_gated_delta_rule_update_fwd | RANKED | CEIL | 10/10 | 1.13e-02 | 7.15e-03 | 1.6x | 19.8% | PASS |
| 16 | L2_34_silu_mul_static_tensorwise_quant_triton_kernel_for_cutlass_moe | RANKED | VERIF | 9/9 | 9.32e-03 | 4.07e-03 | 2.3x | 11.1% | PASS |
| 17 | L1_38_silu_mul_fp8_quant_deep_gemm | RANKED | MIXED | 4/5 | 9.20e-03 | 5.70e-03 | 1.6x | 20.2% | BLOCKED * |
| 18 | L1_36_compute_seg_indptr_triton | RANKED | VERIF | 9/9 | 7.45e-03 | 5.69e-03 | 1.3x | 19.6% | BLOCKED * |
| 19 | L3_09_fused_moe | RANKED | CEIL | 9/9 | 7.21e-03 | 7.35e-03 | 1.0x | 29.0% | BLOCKED * |
| 20 | L2_41_selective_state_update | RANKED | VERIF | 9/9 | 7.09e-03 | 1.40e-03 | 5.0x | 21.0% | PASS |
| 21 | L2_32_experts_combine | RANKED | VERIF | 9/9 | 6.57e-03 | 2.22e-03 | 3.0x | 30.0% | PASS |
| 22 | L2_01_bmm_chunk_fwd | RANKED | CEIL | 9/9 | 6.07e-03 | 1.48e-03 | 4.1x | 9.5% | PASS |
| 23 | L3_18_chunk_scaled_dot_kkt_fwd | RANKED | VERIF | 6/6 | 5.44e-03 | 1.00e-03 | 5.4x | 2.9% | BLOCKED * |
| 24 | L1_18_fused_softcap | RANKED | VERIF | 9/9 | 4.86e-03 | 1.70e-03 | 2.9x | 32.0% | BLOCKED * |
| 25 | L1_16_silu_and_mul | RANKED | VERIF | 9/9 | 4.05e-03 | 1.27e-03 | 3.2x | 36.7% | BLOCKED * |
| 26 | L2_37_silu_and_mul_post_quant | RANKED | VERIF | 9/9 | 3.74e-03 | 1.63e-03 | 2.3x | 4.4% | BLOCKED * |
| 27 | L2_12_awq_gemm | RANKED | VERIF | 9/9 | 3.56e-03 | 1.16e-03 | 3.1x | 6.5% | PASS |
| 28 | L2_14_matmul_kernel_persistent | RANKED | VERIF | 9/9 | 3.22e-03 | 1.61e-03 | 2.0x | 6.0% | PASS |
| 29 | L2_40_gemma_qkv_rmsnorm | RANKED | CEIL | 9/9 | 3.06e-03 | 1.53e-03 | 2.0x | 6.1% | PASS |
| 30 | L1_21_gumbel_sample | RANKED | VERIF | 9/9 | 3.01e-03 | 1.03e-03 | 2.9x | 5.2% | PASS |
| 31 | L1_34_transform_index_page_table_decode | RANKED | VERIF | 9/9 | 2.78e-03 | 1.08e-03 | 2.6x | 1.6% | PASS |
| 32 | L1_09_quantize_k_cache_fast | RANKED | VERIF | 9/9 | 2.53e-03 | 4.73e-04 | 5.3x | 4.2% | PASS |
| 33 | L3_05_fwd_kernel_alibi | RANKED | VERIF | 9/9 | 2.27e-03 | 2.30e-03 | 1.0x | 5.6% | PASS |
| 34 | L1_19_topk_log_softmax | RANKED | VERIF | 9/9 | 2.22e-03 | 7.80e-04 | 2.8x | 5.5% | PASS |
| 35 | L2_28_deepep_post_reorder_triton | RANKED | VERIF | 9/9 | 1.89e-03 | 5.96e-04 | 3.2x | 6.8% | PASS |
| 36 | L1_22_min_p | RANKED | VERIF | 9/9 | 1.85e-03 | 8.26e-04 | 2.2x | 10.5% | PASS |
| 37 | L2_11_w8a8_triton_block_scaled_mm | RANKED | VERIF | 9/9 | 1.81e-03 | 6.72e-04 | 2.7x | 1.5% | PASS |
| 38 | L2_10_w8a8_block_int8_matmul | RANKED | VERIF | 9/9 | 1.74e-03 | 5.64e-04 | 3.1x | 1.8% | PASS |
| 39 | L1_17_gelu_and_mul | RANKED | VERIF | 9/9 | 1.74e-03 | 7.64e-04 | 2.3x | 26.8% | PASS |
| 40 | L1_14_layer_norm_fwd_1pass | RANKED | VERIF | 9/9 | 1.71e-03 | 2.38e-04 | 7.2x | 0.6% | BLOCKED * |
| 41 | L3_23_merge_16x16_to_32x32_inverse | RANKED | VERIF | 9/9 | 1.62e-03 | 3.75e-04 | 4.3x | 2.3% | PASS |
| 42 | L3_01_kernel_unified_attention_3d | RANKED | VERIF | 9/9 | 1.53e-03 | 1.57e-03 | 1.0x | 1.5% | BLOCKED * |
| 43 | L2_33_fwd_grouped_kernel_stage1_rope | RANKED | VERIF | 9/9 | 1.47e-03 | 3.52e-04 | 4.2x | 0.7% | BLOCKED * |
| 44 | L2_17_layer_norm_gated_fwd | RANKED | CEIL | 9/9 | 1.40e-03 | 3.14e-04 | 4.5x | 0.6% | BLOCKED * |
| 45 | L2_39_scaled_mm | RANKED | VERIF | 9/9 | 1.40e-03 | 4.47e-04 | 3.1x | 0.6% | BLOCKED * |
| 46 | L2_15_fused_dual_residual_rmsnorm | RANKED | CEIL | 9/9 | 1.33e-03 | 3.03e-04 | 4.4x | 0.5% | PASS |
| 47 | L1_13_layer_norm_fwd | RANKED | VERIF | 9/9 | 1.27e-03 | 1.72e-04 | 7.4x | 0.4% | BLOCKED * |
| 48 | L3_21_solve_tril_16x16 | RANKED | VERIF | 9/9 | 1.22e-03 | 2.76e-04 | 4.4x | 14.4% | PASS |
| 49 | L1_15_act_and_mul | RANKED | VERIF | 9/9 | 1.18e-03 | 3.87e-04 | 3.0x | 15.2% | PASS |
| 50 | L2_29_fused_moe_router_cudacore | RANKED | VERIF | 9/9 | 1.16e-03 | 2.61e-04 | 4.4x | 3.0% | PASS |
| 51 | L1_23_penalties_and_temperature | RANKED | VERIF | 9/9 | 1.11e-03 | 5.55e-04 | 2.0x | 10.6% | PASS |
| 52 | L1_08_dequantize_k_cache_paged | RANKED | VERIF | 9/9 | 1.08e-03 | 1.40e-04 | 7.7x | 1.1% | BLOCKED * |
| 53 | L3_03_chunk_state_varlen | RANKED | CEIL | 9/9 | 1.06e-03 | 1.15e-03 | 0.9x | 2.6% | BLOCKED * |
| 54 | L2_36_post_reorder_triton_kernel_for_cutlass_moe | RANKED | VERIF | 9/9 | 1.06e-03 | 2.96e-04 | 3.6x | 4.4% | PASS |
| 55 | L2_19_fused_sigmoid_gating_delta_rule_update | RANKED | CEIL | 9/9 | 1.06e-03 | 1.49e-03 | 0.7x | 4.7% | PASS |
| 56 | L2_25_fwd_kernel_ep_gather | RANKED | VERIF | 9/9 | 1.04e-03 | 2.42e-04 | 4.3x | 3.6% | PASS |
| 57 | L2_03_chunk_state_fwd | RANKED | VERIF | 9/9 | 9.61e-04 | 9.01e-04 | 1.1x | 10.4% | BLOCKED * |
| 58 | L3_10_fused_moe_kernel_gptq_awq | RANKED | VERIF | 9/9 | 9.49e-04 | 9.25e-04 | 1.0x | 0.8% | PASS |
| 59 | L2_30_fused_moe_router_tensorcore | RANKED | VERIF | 9/9 | 8.17e-04 | 2.15e-04 | 3.8x | 1.7% | PASS |
| 60 | L1_20_log_softmax | RANKED | VERIF | 9/9 | 8.10e-04 | 2.13e-04 | 3.8x | 1.6% | PASS |
| 61 | L1_39_state_passing_fwd | RANKED | VERIF | 9/9 | 7.62e-04 | 1.48e-04 | 5.1x | 2.9% | BLOCKED * |
| 62 | L2_16_fused_rmsnorm | RANKED | VERIF | 9/9 | 7.50e-04 | 2.12e-04 | 3.5x | 0.3% | BLOCKED * |
| 63 | L2_07_gate_up_lora_b | RANKED | CEIL | 9/9 | 7.19e-04 | 6.00e-04 | 1.2x | 5.2% | PASS |
| 64 | L1_30_act_quant | RANKED | VERIF | 9/9 | 6.06e-04 | 1.09e-04 | 5.5x | 0.8% | PASS |
| 65 | L2_35_pre_reorder_triton_kernel_for_cutlass_moe | RANKED | VERIF | 9/9 | 6.02e-04 | 1.51e-04 | 4.0x | 2.3% | PASS |
| 66 | L1_12_rms_norm | RANKED | CEIL | 9/9 | 5.95e-04 | 1.64e-04 | 3.6x | 0.2% | BLOCKED * |
| 67 | L1_25_merge_attn_states | RANKED | VERIF | 9/9 | 5.91e-04 | 1.61e-04 | 3.7x | 1.7% | PASS |
| 68 | L1_03_convert_req_index_to_global_index | RANKED | VERIF | 9/9 | 5.38e-04 | 1.24e-04 | 4.3x | 0.8% | BLOCKED * |
| 69 | L1_10_per_token_group_quant_int8 | RANKED | VERIF | 9/9 | 4.76e-04 | 9.59e-05 | 5.0x | 0.4% | PASS |
| 70 | L1_11_awq_dequantize | RANKED | VERIF | 9/9 | 4.50e-04 | 1.17e-04 | 3.9x | 0.9% | PASS |
| 71 | L2_20_causal_conv1d_update | RANKED | VERIF | 9/9 | 4.17e-04 | 7.21e-05 | 5.8x | 0.5% | BLOCKED * |
| 72 | L2_27_deepep_permute_triton | RANKED | VERIF | 9/9 | 3.79e-04 | 1.33e-04 | 2.9x | 1.6% | PASS |
| 73 | L3_14_fused_recurrent_gated_delta_rule_fwd | RANKED | CEIL | 10/10 | 3.61e-04 | 1.79e-04 | 2.0x | 0.1% | BLOCKED * |
| 74 | L2_21_causal_conv1d_fwd | RANKED | CEIL | 9/9 | 3.10e-04 | 3.96e-04 | 0.8x | 1.4% | PASS |
| 75 | L3_07_correct_attn_cp_out | RANKED | VERIF | 9/9 | 2.64e-04 | 7.91e-05 | 3.3x | 0.7% | PASS |
| 76 | L1_05_gather_block_tables | RANKED | VERIF | 9/9 | 2.37e-04 | 2.02e-04 | 1.2x | 1.3% | BLOCKED * |
| 77 | L2_09_sgemm_lora_b | RANKED | VERIF | 9/9 | 2.02e-04 | 1.82e-04 | 1.1x | 0.9% | PASS |
| 78 | L1_07_dequantize_k_cache_fast | RANKED | VERIF | 9/9 | 2.02e-04 | 2.80e-05 | 7.2x | 0.1% | BLOCKED * |
| 79 | L2_08_sgemm_lora_a | RANKED | VERIF | 9/9 | 1.78e-04 | 1.45e-04 | 1.2x | 0.5% | PASS |
| 80 | L1_01_bias | RANKED | VERIF | 9/9 | 1.72e-04 | 4.39e-05 | 3.9x | 0.2% | BLOCKED * |
| 81 | L2_04_chunked_lora_expand | RANKED | VERIF | 9/9 | 1.68e-04 | 1.38e-04 | 1.2x | 0.7% | PASS |
| 82 | L2_05_chunked_lora_shrink | RANKED | VERIF | 9/9 | 1.65e-04 | 1.25e-04 | 1.3x | 0.5% | PASS |
| 83 | L1_31_triton_mrope_forward | RANKED | VERIF | 9/9 | 1.55e-04 | 3.97e-05 | 3.9x | 0.1% | BLOCKED * |
| 84 | L1_06_reshape_and_cache_kernel_flash | RANKED | VERIF | 9/9 | 1.53e-04 | 5.57e-05 | 2.7x | 0.1% | PASS |
| 85 | L2_02_chunk_scan_fwd | RANKED | CEIL | 9/9 | 1.29e-04 | 1.25e-04 | 1.0x | 0.0% | PASS |
| 86 | L3_06_fwd_grouped_kernel_stage1 | RANKED | VERIF | 10/10 | 1.28e-04 | 1.38e-04 | 0.9x | 0.3% | PASS |
| 87 | L2_06_qkv_lora_b | RANKED | VERIF | 9/9 | 8.08e-05 | 8.00e-05 | 1.0x | 0.1% | PASS |
| 88 | L1_33_create_flashmla_kv_indices_triton | RANKED | VERIF | 9/9 | 6.89e-05 | 2.57e-05 | 2.7x | 0.1% | BLOCKED * |
| 89 | L1_27_apply_grammar_bitmask | RANKED | VERIF | 9/9 | 6.66e-05 | 1.91e-05 | 3.5x | 0.2% | BLOCKED * |
| 90 | L1_32_create_flashinfer_kv_indices_triton | RANKED | VERIF | 9/9 | 6.64e-05 | 5.84e-05 | 1.1x | 0.1% | BLOCKED * |
| 91 | L2_18_fused_gdn_gating | RANKED | VERIF | 9/9 | 5.82e-05 | 4.45e-05 | 1.3x | 0.1% | BLOCKED * |
| 92 | L2_31_moe_align_block_size_stage2 | RANKED | VERIF | 9/9 | 4.12e-05 | 4.01e-05 | 1.0x | 0.0% | PASS |
| 93 | L1_02_compute_slot_mappings | RANKED | VERIF | 9/9 | 2.38e-05 | 1.75e-05 | 1.4x | 0.0% | BLOCKED * |
| 94 | L2_26_fwd_kernel_ep_scatter_1 | RANKED | VERIF | 9/9 | 1.19e-05 | 1.04e-05 | 1.1x | 0.0% | BLOCKED * |
| 95 | L1_40_chunk_cumsum_fwd | RANKED | VERIF | 9/9 | 5.04e-06 | 4.24e-06 | 1.2x | 0.0% | BLOCKED * |
| 96 | L1_04_copy_page_indices | FLOOR | VERIF | 0/9 | - | 1.08e-05 | - | 0.0% | FLOOR_DOMINATED * |
| 97 | L1_24_rejection_sample | FLOOR | VERIF | 0/9 | - | 1.76e-05 | - | 0.1% | FLOOR_DOMINATED * |
| 98 | L1_28_pack_bitmatrix | FLOOR | VERIF | 0/9 | - | 6.61e-07 | - | 0.0% | FLOOR_DOMINATED * |
| 99 | L1_29_prepare_eagle_inputs | FLOOR | VERIF | 0/9 | - | 7.32e-06 | - | 0.0% | FLOOR_DOMINATED * |
| 100 | L1_35_compute_src2dst_triton | FLOOR | VERIF | 0/9 | - | 4.43e-06 | - | 0.0% | FLOOR_DOMINATED * |
| 101 | L1_37_compute_problem_sizes_w4a8 | FLOOR | VERIF | 0/9 | - | 2.82e-06 | - | 0.0% | FLOOR_DOMINATED * |
