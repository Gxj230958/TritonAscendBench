# Clean set-relation triage (R10.a, codex-endorsed discriminator)
date: 2026-05-29
source: scripts/oracle_pyref_precompute.py --all (集合包含关系, 非 50% 阈值)

## SIGNATURE_MISMATCH 34 (wrapper required param 不在 shapes -> Phase2 自动补目标)

- `L1_02_compute_slot_mappings` missing: ['num_tokens_padded', 'query_start_loc']
- `L1_05_gather_block_tables` missing: ['num_reqs_padded']
- `L1_08_dequantize_k_cache_paged` missing: ['page_table_1_flattened', 'quant_k_cache']
- `L1_15_act_and_mul` missing: ['config', 'down_input', 'gateup_output']
- `L1_21_gumbel_sample` missing: ['apply_temperature', 'expanded_idx_mapping', 'logits', 'pos', 'seed', 'temperature']
- `L1_23_penalties_and_temperature` missing: ['expanded_local_pos', 'output_bin_counts']
- `L1_24_rejection_sample` missing: ['sampling_metadata', 'target_logits']
- `L1_27_apply_grammar_bitmask` missing: ['grammar_bitmask', 'grammar_req_ids', 'input_batch']
- `L1_28_pack_bitmatrix` missing: ['num_local_experts', 'topk_weights']
- `L1_29_prepare_eagle_inputs` missing: ['input_batch', 'input_buffers', 'last_sampled', 'max_num_reqs', 'next_prefill_tokens']
- `L1_37_compute_problem_sizes_w4a8` missing: ['problem_sizes2']
- `L1_39_state_passing_fwd` missing: ['last_chunk_indices']
- `L1_40_chunk_cumsum_fwd` missing: ['cu_chunk_seqlens']
- `L2_02_chunk_scan_fwd` missing: ['cu_chunk_seqlens', 'out', 'seq_idx']
- `L2_03_chunk_state_fwd` missing: ['cu_chunk_seqlens']
- `L2_11_w8a8_triton_block_scaled_mm` missing: ['block_size']
- `L2_17_layer_norm_gated_fwd` missing: ['bias', 'weight']
- `L2_18_fused_gdn_gating` missing: ['b']
- `L2_19_fused_sigmoid_gating_delta_rule_update` missing: ['b', 'initial_state_indices', 'initial_state_source', 'softplus_threshold']
- `L2_21_causal_conv1d_fwd` missing: ['conv_states', 'query_start_loc']
- `L2_22_linear_attn_decode` missing: ['slot_idx']
- `L2_23_fwd_kv_parallel` missing: ['kv_caches', 'slope_rate', 'slot_idx']
- `L2_24_fwd_kv_reduce` missing: ['kv_caches', 'slope_rate', 'slot_idx']
- `L2_25_fwd_kernel_ep_gather` missing: ['expert_map', 'output_tensor', 'recv_topk_ids', 'recv_topk_weight']
- `L2_28_deepep_post_reorder_triton` missing: ['hidden_size', 'num_local_experts', 'num_tokens', 'routed_scaling_factor', 'topk_ids']
- `L2_31_moe_align_block_size_stage2` missing: ['expert_ids', 'num_tokens_post_pad', 'sorted_token_ids']
- `L2_35_pre_reorder_triton_kernel_for_cutlass_moe` missing: ['a1_scales', 'gateup_input', 'hidden_size', 'num_local_experts', 'num_tokens', 'topk_ids']
- `L2_36_post_reorder_triton_kernel_for_cutlass_moe` missing: ['hidden_size', 'num_local_experts', 'num_tokens', 'routed_scaling_factor', 'topk_ids']
- `L2_37_silu_and_mul_post_quant` missing: ['output_scale']
- `L3_02_kernel_paged_attention_2d` missing: ['k_scale', 'kv_cache_dtype', 'max_query_len', 'max_seq_len', 'output', 'query_start_loc', 'v_scale', 'value']
- `L3_03_chunk_state_varlen` missing: ['chunk_states']
- `L3_07_correct_attn_cp_out` missing: ['ctx']
- `L3_14_fused_recurrent_gated_delta_rule_fwd` missing: ['initial_state']
- `L3_15_fused_recurrent_gated_delta_rule_update_fwd` missing: ['initial_state_indices', 'initial_state_source']

## NEEDS_BRIDGE resolver-returned-None 16 (resolver 找不到入口 -> Phase1 目标)

- `L1_04_copy_page_indices`
- `L1_07_dequantize_k_cache_fast`
- `L1_32_create_flashinfer_kv_indices_triton`
- `L1_33_create_flashmla_kv_indices_triton`
- `L1_35_compute_src2dst_triton`
- `L2_06_qkv_lora_b`
- `L2_07_gate_up_lora_b`
- `L2_26_fwd_kernel_ep_scatter_1`
- `L2_27_deepep_permute_triton`
- `L2_33_fwd_grouped_kernel_stage1_rope`
- `L3_05_fwd_kernel_alibi`
- `L3_06_fwd_grouped_kernel_stage1`
- `L3_09_fused_moe`
- `L3_10_fused_moe_kernel_gptq_awq`
- `L3_11_expert_triton`
- `L3_12_moe_mmk`

## NEEDS_BRIDGE oracle-extra-keys 15 (oracle 读 wrapper 没有的 key -> 逐个判断 oracle vs wrapper)

- `L1_09_quantize_k_cache_fast` extra_in_oracle: ['fp8_max', 'group_size']
- `L1_11_awq_dequantize` extra_in_oracle: ['group_size']
- `L1_14_layer_norm_fwd_1pass` extra_in_oracle: ['residual']
- `L1_17_gelu_and_mul` extra_in_oracle: ['approximate']
- `L1_30_act_quant` extra_in_oracle: ['block_size', 'fp8_max']
- `L1_38_silu_mul_fp8_quant_deep_gemm` extra_in_oracle: ['fp8_max', 'group_size']
- `L2_04_chunked_lora_expand` extra_in_oracle: ['lora_ranks', 'scalings', 'seg_indptr', 'weight_indices']
- `L2_05_chunked_lora_shrink` extra_in_oracle: ['lora_ranks', 'seg_indptr', 'weight_indices']
- `L2_08_sgemm_lora_a` extra_in_oracle: ['lora_ranks', 'seg_indptr', 'weight_indices']
- `L2_09_sgemm_lora_b` extra_in_oracle: ['base_output', 'lora_ranks', 'scalings', 'seg_indptr', 'weight_indices']
- `L2_12_awq_gemm` extra_in_oracle: ['group_size']
- `L2_15_fused_dual_residual_rmsnorm` extra_in_oracle: ['residual_a', 'residual_b', 'weight']
- `L2_16_fused_rmsnorm` extra_in_oracle: ['residual']
- `L2_38_concat_and_cast_mha_k` extra_in_oracle: ['out_dtype']
- `L3_01_kernel_unified_attention_3d` extra_in_oracle: ['alibi_slopes', 'scale']
