---
date: 2026-05-26
tool: scripts/oracle_pyref_precompute.py
purpose: V9 M2.1a (dev) — R10 stage-1 oracle PyTorch reference precompute
---

# Oracle PyRef Precompute Audit

Tasks audited: **100**  

## Status distribution

| status | count |
|---|---|
| `SIGNATURE_MISMATCH` | 52 |
| `ALIGNED_READY` | 30 |
| `NEEDS_BRIDGE` | 17 |
| `RUNTIME_ERROR` | 1 |

## ALIGNED_READY (30)

### `L1_16_silu_and_mul` (level1)

- kernel required (1): `hidden_states`
- oracle inputs[..] (1): `hidden_states`
- oracle outputs (1): `output`
- shapes inputs (1): `hidden_states`
- saved ref artifacts:
  - `r10_pyref/ref_P1.safetensors` sha256=`417aef8ca63593fc...`
  - `r10_pyref/ref_P2.safetensors` sha256=`c20cb812862d59ab...`

### `L1_20_log_softmax` (level1)

- kernel required (1): `input`
- kernel optional (1): `dim`
- oracle inputs[..] (2): `dim, input`
- oracle outputs (1): `output`
- shapes inputs (2): `dim, input`
- extra-in-oracle (oracle reads non-kernel param): `['dim']`
- saved ref artifacts:
  - `r10_pyref/ref_P1.safetensors` sha256=`7ca706511e20e019...`
  - `r10_pyref/ref_P2.safetensors` sha256=`7b3bcec3a57b08a4...`

### `L1_19_topk_log_softmax` (level1)

- kernel required (2): `logits, token_ids`
- oracle inputs[..] (2): `logits, token_ids`
- oracle outputs (1): `logprobs`
- shapes inputs (3): `logits, num_logprobs, token_ids`
- saved ref artifacts:
  - `r10_pyref/ref_P1.safetensors` sha256=`6a9b2c8984f60485...`
  - `r10_pyref/ref_P2.safetensors` sha256=`dd6b84db2f838557...`

### `L1_12_rms_norm` (level1)

- kernel required (2): `input, weight`
- kernel optional (1): `eps`
- oracle inputs[..] (3): `eps, input, weight`
- oracle outputs (1): `output`
- shapes inputs (3): `eps, input, weight`
- extra-in-oracle (oracle reads non-kernel param): `['eps']`
- saved ref artifacts:
  - `r10_pyref/ref_P1.safetensors` sha256=`d82c65db35c0f6fb...`
  - `r10_pyref/ref_P2.safetensors` sha256=`9bebfb6fea31ce25...`

### `L1_13_layer_norm_fwd` (level1)

- kernel required (4): `x, weight, bias, eps`
- kernel optional (6): `z, out, group_size, norm_before_gate, is_rms_norm, activation`
- oracle inputs[..] (4): `bias, eps, weight, x`
- oracle outputs (1): `output`
- shapes inputs (4): `bias, eps, weight, x`
- saved ref artifacts:
  - `r10_pyref/ref_P1.safetensors` sha256=`eff1e65a6b128128...`
  - `r10_pyref/ref_P2.safetensors` sha256=`556a898917b4dedd...`

### `L1_10_per_token_group_quant_int8` (level1)

- kernel required (2): `x, group_size`
- kernel optional (2): `eps, dtype`
- oracle inputs[..] (3): `eps, group_size, x`
- oracle outputs (2): `q, scale`
- shapes inputs (3): `eps, group_size, x`
- extra-in-oracle (oracle reads non-kernel param): `['eps']`
- saved ref artifacts:
  - `r10_pyref/ref_P1.safetensors` sha256=`e409848f1515507d...`
  - `r10_pyref/ref_P2.safetensors` sha256=`20d3608b65f17048...`

### `L1_06_reshape_and_cache_kernel_flash` (level1)

- kernel required (8): `key, value, key_cache, value_cache, slot_mapping, kv_cache_dtype, k_scale, v_scale`
- oracle inputs[..] (5): `key, key_cache, slot_mapping, value, value_cache`
- oracle outputs (2): `key_cache, value_cache`
- shapes inputs (8): `k_scale, key, key_cache, kv_cache_dtype, slot_mapping, v_scale, value, value_cache`
- saved ref artifacts:
  - `r10_pyref/ref_P1.safetensors` sha256=`2ad61d6c3731e9a3...`
  - `r10_pyref/ref_P2.safetensors` sha256=`709d87ff1275a013...`

### `L1_22_min_p` (level1)

- kernel required (3): `logits, expanded_idx_mapping, min_p`
- oracle inputs[..] (3): `expanded_idx_mapping, logits, min_p`
- oracle outputs (3): `expanded_idx_mapping, logits, min_p`
- shapes inputs (3): `expanded_idx_mapping, logits, min_p`
- saved ref artifacts:
  - `r10_pyref/ref_P1.safetensors` sha256=`68b752d51cb42641...`
  - `r10_pyref/ref_P2.safetensors` sha256=`e8631636aa8bae1a...`

### `L1_25_merge_attn_states` (level1)

- kernel required (5): `output, prefix_output, prefix_lse, suffix_output, suffix_lse`
- kernel optional (3): `output_lse, prefill_tokens_with_context, output_scale`
- oracle inputs[..] (4): `prefix_lse, prefix_output, suffix_lse, suffix_output`
- oracle outputs (2): `output, output_lse`
- shapes inputs (5): `output, prefix_lse, prefix_output, suffix_lse, suffix_output`
- saved ref artifacts:
  - `r10_pyref/ref_P1.safetensors` sha256=`e18e4e428da15cd4...`
  - `r10_pyref/ref_P2.safetensors` sha256=`e095c076b765bd90...`

### `L1_34_transform_index_page_table_decode` (level1)

- kernel required (2): `page_table, topk_indices`
- kernel optional (2): `result, page_size`
- oracle inputs[..] (3): `page_size, page_table, topk_indices`
- oracle outputs (1): `result`
- shapes inputs (3): `page_size, page_table, topk_indices`
- extra-in-oracle (oracle reads non-kernel param): `['page_size']`
- saved ref artifacts:
  - `r10_pyref/ref_P1.safetensors` sha256=`24646a63a527db21...`
  - `r10_pyref/ref_P2.safetensors` sha256=`e14362d8cd9677af...`

### `L1_26_merge_state` (level1)

- kernel required (4): `prefix_output, prefix_lse, suffix_output, suffix_lse`
- kernel optional (2): `output, output_lse`
- oracle inputs[..] (4): `prefix_lse, prefix_output, suffix_lse, suffix_output`
- oracle outputs (2): `output, output_lse`
- shapes inputs (4): `prefix_lse, prefix_output, suffix_lse, suffix_output`
- saved ref artifacts:
  - `r10_pyref/ref_P1.safetensors` sha256=`e6d389864326073a...`
  - `r10_pyref/ref_P2.safetensors` sha256=`1968d66c18a236b7...`

### `L1_31_triton_mrope_forward` (level1)

- kernel required (8): `q, k, cos, sin, mrope_section, head_size, rotary_dim, mrope_interleaved`
- oracle inputs[..] (7): `cos, head_size, k, mrope_section, q, rotary_dim, sin`
- oracle outputs (2): `k, q`
- shapes inputs (8): `cos, head_size, k, mrope_interleaved, mrope_section, q, rotary_dim, sin`
- saved ref artifacts:
  - `r10_pyref/ref_P1.safetensors` sha256=`1c4a957edeb10a05...`
  - `r10_pyref/ref_P2.safetensors` sha256=`f51cc3ee06936e39...`

### `L2_14_matmul_kernel_persistent` (level2)

- kernel required (2): `a, b`
- kernel optional (1): `bias`
- oracle inputs[..] (3): `a, b, bias`
- oracle outputs (1): `C`
- shapes inputs (4): `a, b, bias, input`
- extra-in-oracle (oracle reads non-kernel param): `['bias']`
- saved ref artifacts:
  - `r10_pyref/ref_P1.safetensors` sha256=`007481f1160c77b5...`
  - `r10_pyref/ref_P2.safetensors` sha256=`4c0a64d95d24719e...`

### `L2_13_bmm` (level2)

- kernel required (2): `a, b`
- oracle inputs[..] (2): `a, b`
- oracle outputs (1): `C`
- shapes inputs (3): `a, b, input`
- saved ref artifacts:
  - `r10_pyref/ref_P1.safetensors` sha256=`c81d7e97bee1ba6d...`
  - `r10_pyref/ref_P2.safetensors` sha256=`9742ac0f8c4df4f8...`

### `L2_10_w8a8_block_int8_matmul` (level2)

- kernel required (5): `A, B, As, Bs, block_size`
- kernel optional (1): `output_dtype`
- oracle inputs[..] (6): `A, As, B, Bs, block_size, output_dtype`
- oracle outputs (1): `C`
- shapes inputs (5): `A, As, B, Bs, block_size`
- missing-in-oracle (oracle reads non-shapes key): `['output_dtype']`
- extra-in-oracle (oracle reads non-kernel param): `['output_dtype']`
- saved ref artifacts:
  - `r10_pyref/ref_P1.safetensors` sha256=`3cc0a8086922082d...`
  - `r10_pyref/ref_P2.safetensors` sha256=`6eb0bec65b25cdb3...`

### `L2_20_causal_conv1d_update` (level2)

- kernel required (3): `x, conv_state, weight`
- kernel optional (10): `bias, activation, conv_state_indices, num_accepted_tokens, query_start_loc, max_query_len, null_block_id, block_idx_last_scheduled_token, initial_state_idx, validate_data`
- oracle inputs[..] (5): `activation, bias, conv_state, weight, x`
- oracle outputs (2): `conv_state, output`
- shapes inputs (5): `activation, bias, conv_state, weight, x`
- extra-in-oracle (oracle reads non-kernel param): `['activation', 'bias']`
- saved ref artifacts:
  - `r10_pyref/ref_P1.safetensors` sha256=`13f07f8b14d8d671...`
  - `r10_pyref/ref_P2.safetensors` sha256=`a3c26935557f3e57...`

### `L1_01_bias` (level1)

- kernel required (11): `logits, expanded_idx_mapping, pos, num_allowed_token_ids, allowed_token_ids, num_logit_bias, logit_bias_token_ids, logit_bias, min_lens, num_stop_token_ids, stop_token_ids`
- oracle inputs[..] (11): `allowed_token_ids, expanded_idx_mapping, logit_bias, logit_bias_token_ids, logits, min_lens, num_allowed_token_ids, num_logit_bias, num_stop_token_ids, pos, stop_token_ids`
- oracle outputs (11): `allowed_token_ids, expanded_idx_mapping, logit_bias, logit_bias_token_ids, logits, min_lens, num_allowed_token_ids, num_logit_bias, num_stop_token_ids, pos, stop_token_ids`
- shapes inputs (11): `allowed_token_ids, expanded_idx_mapping, logit_bias, logit_bias_token_ids, logits, min_lens, num_allowed_token_ids, num_logit_bias, num_stop_token_ids, pos, stop_token_ids`
- saved ref artifacts:
  - `r10_pyref/ref_P1.safetensors` sha256=`05b01911032fdf5d...`
  - `r10_pyref/ref_P2.safetensors` sha256=`ecb096d8fa633ce3...`

### `L2_30_fused_moe_router_tensorcore` (level2)

- kernel required (7): `x, router_weight, topk, moe_softcapping, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K`
- kernel optional (1): `correction_bias`
- oracle inputs[..] (5): `correction_bias, moe_softcapping, router_weight, topk, x`
- oracle outputs (2): `topk_ids, topk_weights`
- shapes inputs (8): `BLOCK_SIZE_K, BLOCK_SIZE_M, BLOCK_SIZE_N, correction_bias, moe_softcapping, router_weight, topk, x`
- extra-in-oracle (oracle reads non-kernel param): `['correction_bias']`
- saved ref artifacts:
  - `r10_pyref/ref_P1.safetensors` sha256=`f43813b515770a0e...`
  - `r10_pyref/ref_P2.safetensors` sha256=`a336bf85e9983129...`

### `L2_29_fused_moe_router_cudacore` (level2)

- kernel required (4): `x, router_weight, topk, moe_softcapping`
- kernel optional (1): `correction_bias`
- oracle inputs[..] (5): `correction_bias, moe_softcapping, router_weight, topk, x`
- oracle outputs (2): `topk_ids, topk_weights`
- shapes inputs (7): `BLOCK_SIZE_M, BLOCK_SIZE_N, correction_bias, moe_softcapping, router_weight, topk, x`
- extra-in-oracle (oracle reads non-kernel param): `['correction_bias']`
- saved ref artifacts:
  - `r10_pyref/ref_P1.safetensors` sha256=`f43813b515770a0e...`
  - `r10_pyref/ref_P2.safetensors` sha256=`a336bf85e9983129...`

### `L2_32_experts_combine` (level2)

- kernel required (2): `moe_hidden_states, mlp_hidden_states`
- kernel optional (1): `output_buffer`
- oracle inputs[..] (2): `mlp_hidden_states, moe_hidden_states`
- oracle outputs (1): `output`
- shapes inputs (6): `hidden_states, mlp_hidden_states, moe_hidden_states, residual, softcap_const, x`
- saved ref artifacts:
  - `r10_pyref/ref_P1.safetensors` sha256=`ebc1bba730114a52...`
  - `r10_pyref/ref_P2.safetensors` sha256=`e7d0d20c12ae68ab...`

### `L2_34_silu_mul_static_tensorwise_quant_triton_kernel_for_cutlass_moe` (level2)

- kernel required (4): `input, output, masked_m, scale`
- oracle inputs[..] (2): `input, scale`
- oracle outputs (1): `output`
- shapes inputs (7): `down_output, input, masked_m, output, problem_sizes1, scale, x`
- saved ref artifacts:
  - `r10_pyref/ref_P1.safetensors` sha256=`2ec8eacd7e589b2d...`
  - `r10_pyref/ref_P2.safetensors` sha256=`4aa01fac16effe28...`

### `L2_01_bmm_chunk_fwd` (level2)

- kernel required (4): `a, b, chunk_size, cu_chunk_seqlens`
- kernel optional (2): `causal, output_dtype`
- oracle inputs[..] (5): `a, b, causal, chunk_size, cu_chunk_seqlens`
- oracle outputs (1): `output`
- shapes inputs (5): `a, b, causal, chunk_size, cu_chunk_seqlens`
- extra-in-oracle (oracle reads non-kernel param): `['causal']`
- saved ref artifacts:
  - `r10_pyref/ref_P1.safetensors` sha256=`f9205bf0ae1f420c...`
  - `r10_pyref/ref_P2.safetensors` sha256=`b9c779c92893a4d2...`

### `L3_16_chunk_fwd_kernel_o` (level3)

- kernel required (4): `q, k, v, h`
- kernel optional (6): `g, scale, cu_seqlens, chunk_indices, chunk_size, core_attn_out`
- oracle inputs[..] (6): `g, h, k, q, scale, v`
- oracle outputs (1): `output`
- shapes inputs (6): `g, h, k, q, scale, v`
- extra-in-oracle (oracle reads non-kernel param): `['g', 'scale']`
- saved ref artifacts:
  - `r10_pyref/ref_P1.safetensors` sha256=`36c6b7eefd20ea40...`
  - `r10_pyref/ref_P2.safetensors` sha256=`f3be9766cc658b9f...`

### `L3_17_chunk_gla_fwd_kernel_o` (level3)

- kernel required (7): `q, v, g, A, h, o, scale`
- kernel optional (2): `cu_seqlens, chunk_size`
- oracle inputs[..] (6): `A, g, h, q, scale, v`
- oracle outputs (1): `output`
- shapes inputs (7): `A, g, h, o, q, scale, v`
- saved ref artifacts:
  - `r10_pyref/ref_P1.safetensors` sha256=`b57020844cec16cd...`
  - `r10_pyref/ref_P2.safetensors` sha256=`852aa99cd416e3d9...`

### `L3_18_chunk_scaled_dot_kkt_fwd` (level3)

- kernel required (1): `k`
- kernel optional (6): `g, beta, cu_seqlens, chunk_indices, chunk_size, output_dtype`
- oracle inputs[..] (3): `beta, g, k`
- oracle outputs (1): `output`
- shapes inputs (3): `beta, g, k`
- extra-in-oracle (oracle reads non-kernel param): `['beta', 'g']`
- saved ref artifacts:
  - `r10_pyref/ref_P1.safetensors` sha256=`90240e2b58a72f30...`
  - `r10_pyref/ref_P2.safetensors` sha256=`b3f6b708795715ee...`

### `L3_22_chunk_local_cumsum_vector` (level3)

- kernel required (2): `g, chunk_size`
- kernel optional (5): `reverse, cu_seqlens, chunk_indices, head_first, output_dtype`
- oracle inputs[..] (2): `g, reverse`
- oracle outputs (1): `output`
- shapes inputs (3): `chunk_size, g, reverse`
- extra-in-oracle (oracle reads non-kernel param): `['reverse']`
- saved ref artifacts:
  - `r10_pyref/ref_P1.safetensors` sha256=`6ff5aec6666f08fd...`
  - `r10_pyref/ref_P2.safetensors` sha256=`75f05137866c54f9...`

### `L3_19_chunk_kda_scaled_dot_kkt_fwd_kernel_intra_sub_inter` (level3)

- kernel required (2): `q, k`
- kernel optional (6): `gk, beta, scale, cu_seqlens, chunk_size, output_dtype`
- oracle inputs[..] (5): `beta, gk, k, q, scale`
- oracle outputs (1): `output`
- shapes inputs (5): `beta, gk, k, q, scale`
- extra-in-oracle (oracle reads non-kernel param): `['beta', 'gk', 'scale']`
- saved ref artifacts:
  - `r10_pyref/ref_P1.safetensors` sha256=`ec748368320d3d6e...`
  - `r10_pyref/ref_P2.safetensors` sha256=`2e86ef60db75bdbe...`

### `L3_20_recompute_w_u_fwd` (level3)

- kernel required (6): `k, v, beta, g_cumsum, A, cu_seqlens`
- kernel optional (1): `chunk_indices`
- oracle inputs[..] (5): `A, beta, g_cumsum, k, v`
- oracle outputs (2): `u, w`
- shapes inputs (6): `A, beta, cu_seqlens, g_cumsum, k, v`
- saved ref artifacts:
  - `r10_pyref/ref_P1.safetensors` sha256=`61b6d56790f33b3a...`
  - `r10_pyref/ref_P2.safetensors` sha256=`63cd059ccb0b9c63...`

### `L3_04_fwd` (level3)

- kernel required (7): `q, k, v, o, b_start_loc, b_seq_len, max_input_len`
- kernel optional (2): `is_causal, sm_scale`
- oracle inputs[..] (7): `b_seq_len, b_start_loc, is_causal, k, q, sm_scale, v`
- oracle outputs (1): `o`
- shapes inputs (9): `b_seq_len, b_start_loc, is_causal, k, max_input_len, o, q, sm_scale, v`
- extra-in-oracle (oracle reads non-kernel param): `['is_causal', 'sm_scale']`
- saved ref artifacts:
  - `r10_pyref/ref_P1.safetensors` sha256=`9688db7fbe1e646a...`
  - `r10_pyref/ref_P2.safetensors` sha256=`d4f94624322634a1...`

### `L3_13_chunk_gated_delta_rule_fwd_kernel_h_blockdim64` (level3)

- kernel required (3): `k, w, u`
- kernel optional (9): `g, gk, initial_state, output_final_state, chunk_size, save_new_value, cu_seqlens, chunk_indices, chunk_offsets`
- oracle inputs[..] (5): `g, initial_state, k, u, w`
- oracle outputs (1): `output`
- shapes inputs (4): `g, k, u, w`
- missing-in-oracle (oracle reads non-shapes key): `['initial_state']`
- extra-in-oracle (oracle reads non-kernel param): `['g', 'initial_state']`
- saved ref artifacts:
  - `r10_pyref/ref_P1.safetensors` sha256=`0b31ca14a9d8a183...`
  - `r10_pyref/ref_P2.safetensors` sha256=`a6630ae65956abe9...`

## SIGNATURE_MISMATCH (52)

### `L1_05_gather_block_tables` (level1)

- kernel required (9): `batch_idx_to_req_idx, src_block_table_ptrs, dst_block_table_ptrs, block_table_strides, num_blocks_ptr, num_blocks_stride, num_reqs, max_num_blocks, BLOCK_SIZE`
- oracle inputs[..] (2): `block_table, idx_mapping`
- oracle outputs (1): `output`
- shapes inputs (2): `block_table, idx_mapping`
- **missing in shapes**: `['BLOCK_SIZE', 'batch_idx_to_req_idx', 'block_table_strides', 'dst_block_table_ptrs', 'max_num_blocks', 'num_blocks_ptr', 'num_blocks_stride', 'num_reqs', 'src_block_table_ptrs']`
- extra-in-oracle (oracle reads non-kernel param): `['block_table', 'idx_mapping']`
- **suggested fix**: shapes.yaml.inputs is missing ['BLOCK_SIZE', 'batch_idx_to_req_idx', 'block_table_strides', 'dst_block_table_ptrs', 'max_num_blocks', 'num_blocks_ptr', 'num_blocks_stride', 'num_reqs', 'src_block_table_ptrs'] that kernel `_gather_block_tables_kernel` requires; add or rename keys.

### `L1_02_compute_slot_mappings` (level1)

- kernel required (14): `max_num_tokens, idx_mapping, query_start_loc, pos, block_table_ptrs, block_table_strides, block_sizes, slot_mappings_ptr, slot_mappings_stride, cp_rank, CP_SIZE, CP_INTERLEAVE, PAD_ID, TRITON_BLOCK_SIZE`
- oracle inputs[..] (4): `block_size, block_table, idx_mapping, positions`
- oracle outputs (1): `slot_mapping`
- shapes inputs (5): `block_size, block_table, idx_mapping, max_num_tokens, positions`
- **missing in shapes**: `['CP_INTERLEAVE', 'CP_SIZE', 'PAD_ID', 'TRITON_BLOCK_SIZE', 'block_sizes', 'block_table_ptrs', 'block_table_strides', 'cp_rank', 'pos', 'query_start_loc', 'slot_mappings_ptr', 'slot_mappings_stride']`
- extra-in-oracle (oracle reads non-kernel param): `['block_size', 'block_table', 'positions']`
- **suggested fix**: shapes.yaml.inputs is missing ['CP_INTERLEAVE', 'CP_SIZE', 'PAD_ID', 'TRITON_BLOCK_SIZE', 'block_sizes', 'block_table_ptrs', 'block_table_strides', 'cp_rank', 'pos', 'query_start_loc', 'slot_mappings_ptr', 'slot_mappings_stride'] that kernel `_compute_slot_mappings_kernel` requires; add or rename keys.

### `L1_04_copy_page_indices` (level1)

- kernel required (5): `kv_cache, block_tables_prefill, k_scale, v_scale, dequant_dtype`
- oracle inputs[..] (3): `block_table, page_size, seq_lens`
- oracle outputs (1): `kv_indices`
- shapes inputs (3): `block_table, page_size, seq_lens`
- **missing in shapes**: `['block_tables_prefill', 'dequant_dtype', 'k_scale', 'kv_cache', 'v_scale']`
- extra-in-oracle (oracle reads non-kernel param): `['block_table', 'page_size', 'seq_lens']`
- **suggested fix**: shapes.yaml.inputs is missing ['block_tables_prefill', 'dequant_dtype', 'k_scale', 'kv_cache', 'v_scale'] that kernel `trtllm_prefill_attn_kvfp8_dequant` requires; add or rename keys.

### `L1_07_dequantize_k_cache_fast` (level1)

- kernel required (1): `quant_k_cache`
- kernel optional (1): `group_size`
- oracle inputs[..] (4): `group_size, nope_q, nope_s, rope`
- oracle outputs (1): `output`
- shapes inputs (4): `group_size, nope_q, nope_s, rope`
- **missing in shapes**: `['quant_k_cache']`
- extra-in-oracle (oracle reads non-kernel param): `['group_size', 'nope_q', 'nope_s', 'rope']`
- **suggested fix**: shapes.yaml.inputs is missing ['quant_k_cache'] that kernel `_dequantize_k_cache_fast` requires; add or rename keys.

### `L1_08_dequantize_k_cache_paged` (level1)

- kernel required (2): `quant_k_cache, page_table_1_flattened`
- kernel optional (1): `group_size`
- oracle inputs[..] (5): `group_size, nope_q, nope_s, page_table, rope`
- oracle outputs (1): `output`
- shapes inputs (5): `group_size, nope_q, nope_s, page_table, rope`
- **missing in shapes**: `['page_table_1_flattened', 'quant_k_cache']`
- extra-in-oracle (oracle reads non-kernel param): `['group_size', 'nope_q', 'nope_s', 'page_table', 'rope']`
- **suggested fix**: shapes.yaml.inputs is missing ['page_table_1_flattened', 'quant_k_cache'] that kernel `dequantize_k_cache_paged` requires; add or rename keys.

### `L1_15_act_and_mul` (level1)

- kernel required (3): `gateup_output, down_input, config`
- kernel optional (5): `topk_ids, expert_ids, down_moe_use_tma, activation, swiglu_limit`
- oracle inputs[..] (2): `act, x`
- oracle outputs (1): `output`
- shapes inputs (3): `act, input, x`
- **missing in shapes**: `['config', 'down_input', 'gateup_output']`
- extra-in-oracle (oracle reads non-kernel param): `['act', 'x']`
- **suggested fix**: shapes.yaml.inputs is missing ['config', 'down_input', 'gateup_output'] that kernel `act_and_mul_triton` requires; add or rename keys.

### `L1_18_fused_softcap` (level1)

- kernel required (2): `full_logits, final_logit_softcapping`
- oracle inputs[..] (2): `cap, x`
- oracle outputs (1): `output`
- shapes inputs (2): `cap, x`
- **missing in shapes**: `['final_logit_softcapping', 'full_logits']`
- extra-in-oracle (oracle reads non-kernel param): `['cap', 'x']`
- **suggested fix**: shapes.yaml.inputs is missing ['final_logit_softcapping', 'full_logits'] that kernel `fused_softcap` requires; add or rename keys.

### `L1_21_gumbel_sample` (level1)

- kernel required (6): `logits, expanded_idx_mapping, temperature, seed, pos, apply_temperature`
- kernel optional (2): `output_processed_logits, output_processed_logits_col`
- **missing in shapes**: `['apply_temperature', 'expanded_idx_mapping', 'logits', 'pos', 'seed', 'temperature']`
- **suggested fix**: shapes.yaml.inputs is missing ['apply_temperature', 'expanded_idx_mapping', 'logits', 'pos', 'seed', 'temperature'] that kernel `gumbel_sample` requires; add or rename keys.

### `L1_24_rejection_sample` (level1)

- kernel required (8): `draft_token_ids, num_draft_tokens, max_spec_len, cu_num_draft_tokens, draft_probs, target_logits, bonus_token_ids, sampling_metadata`
- kernel optional (2): `synthetic_mode, synthetic_conditional_rates`
- oracle inputs[..] (9): `bonus_token_ids, cu_num_draft_tokens, draft_probs, draft_token_ids, is_greedy, max_spec_len, recovered_token_ids, target_probs, uniform_probs`
- oracle outputs (1): `output_token_ids`
- shapes inputs (11): `bonus_token_ids, cu_num_draft_tokens, draft_probs, draft_token_ids, is_greedy, max_spec_len, num_draft_tokens, recovered_token_ids, target_probs, uniform_probs, x`
- **missing in shapes**: `['sampling_metadata', 'target_logits']`
- extra-in-oracle (oracle reads non-kernel param): `['is_greedy', 'recovered_token_ids', 'target_probs', 'uniform_probs']`
- **suggested fix**: shapes.yaml.inputs is missing ['sampling_metadata', 'target_logits'] that kernel `rejection_sample` requires; add or rename keys.

### `L1_23_penalties_and_temperature` (level1)

- kernel required (9): `logits, expanded_idx_mapping, token_ids, expanded_local_pos, repetition_penalty, frequency_penalty, presence_penalty, prompt_bin_mask, output_bin_counts`
- oracle inputs[..] (7): `expanded_idx_mapping, frequency_penalty, logits, output_token_counts, presence_penalty, prompt_bin_mask, repetition_penalty`
- oracle outputs (1): `logits`
- shapes inputs (9): `all_token_ids, expanded_idx_mapping, frequency_penalty, logits, output_token_counts, presence_penalty, prompt_bin_mask, repetition_penalty, token_ids`
- **missing in shapes**: `['expanded_local_pos', 'output_bin_counts']`
- extra-in-oracle (oracle reads non-kernel param): `['output_token_counts']`
- **suggested fix**: shapes.yaml.inputs is missing ['expanded_local_pos', 'output_bin_counts'] that kernel `apply_penalties` requires; add or rename keys.

### `L1_28_pack_bitmatrix` (level1)

- kernel required (3): `topk_ids, topk_weights, num_local_experts`
- oracle inputs[..] (2): `num_experts, topk_ids`
- oracle outputs (1): `bitmatrix`
- shapes inputs (2): `num_experts, topk_ids`
- **missing in shapes**: `['num_local_experts', 'topk_weights']`
- extra-in-oracle (oracle reads non-kernel param): `['num_experts']`
- **suggested fix**: shapes.yaml.inputs is missing ['num_local_experts', 'topk_weights'] that kernel `make_routing_data` requires; add or rename keys.

### `L1_27_apply_grammar_bitmask` (level1)

- kernel required (7): `logits_ptr, logits_stride, logits_indices_ptr, bitmask_ptr, bitmask_stride, vocab_size, BLOCK_SIZE`
- oracle inputs[..] (3): `bitmask, logits, logits_indices`
- oracle outputs (1): `logits`
- shapes inputs (3): `bitmask, logits, logits_indices`
- **missing in shapes**: `['BLOCK_SIZE', 'bitmask_ptr', 'bitmask_stride', 'logits_indices_ptr', 'logits_ptr', 'logits_stride', 'vocab_size']`
- extra-in-oracle (oracle reads non-kernel param): `['bitmask', 'logits', 'logits_indices']`
- **suggested fix**: shapes.yaml.inputs is missing ['BLOCK_SIZE', 'bitmask_ptr', 'bitmask_stride', 'logits_indices_ptr', 'logits_ptr', 'logits_stride', 'vocab_size'] that kernel `_apply_grammar_bitmask_kernel` requires; add or rename keys.

### `L1_29_prepare_eagle_inputs` (level1)

- kernel required (9): `last_token_indices, current_draft_step, input_buffers, input_batch, num_sampled, num_rejected, last_sampled, next_prefill_tokens, max_num_reqs`
- oracle inputs[..] (3): `last_token_indices, num_rejected, num_sampled`
- oracle outputs (1): `next_input_indices`
- shapes inputs (4): `current_draft_step, last_token_indices, num_rejected, num_sampled`
- **missing in shapes**: `['input_batch', 'input_buffers', 'last_sampled', 'max_num_reqs', 'next_prefill_tokens']`
- **suggested fix**: shapes.yaml.inputs is missing ['input_batch', 'input_buffers', 'last_sampled', 'max_num_reqs', 'next_prefill_tokens'] that kernel `prepare_eagle_inputs` requires; add or rename keys.

### `L1_32_create_flashinfer_kv_indices_triton` (level1)

- kernel required (12): `q, k, v, key_cache, value_cache, slot_mapping, pos, cos_sin, k_scale, v_scale, is_neox, flash_layout`
- kernel optional (7): `apply_scale, offs, q_out, k_out, output_zeros, zeros_out, swa_slot_mapping`
- oracle inputs[..] (3): `block_table, page_size, seq_lens`
- oracle outputs (2): `kv_indices, kv_indptr`
- shapes inputs (8): `block_table, extend_seq_lens, input_emb, k, key, page_size, q, seq_lens`
- **missing in shapes**: `['cos_sin', 'flash_layout', 'is_neox', 'k_scale', 'key_cache', 'pos', 'slot_mapping', 'v', 'v_scale', 'value_cache']`
- extra-in-oracle (oracle reads non-kernel param): `['block_table', 'page_size', 'seq_lens']`
- **suggested fix**: shapes.yaml.inputs is missing ['cos_sin', 'flash_layout', 'is_neox', 'k_scale', 'key_cache', 'pos', 'slot_mapping', 'v', 'v_scale', 'value_cache'] that kernel `fused_qk_rope_reshape_and_cache` requires; add or rename keys.

### `L1_33_create_flashmla_kv_indices_triton` (level1)

- kernel required (12): `q, k, v, key_cache, value_cache, slot_mapping, pos, cos_sin, k_scale, v_scale, is_neox, flash_layout`
- kernel optional (7): `apply_scale, offs, q_out, k_out, output_zeros, zeros_out, swa_slot_mapping`
- oracle inputs[..] (3): `block_table, page_size, seq_lens`
- oracle outputs (2): `kv_indices, kv_indptr`
- shapes inputs (10): `block_table, extend_seq_lens, input_emb, k, key, page_size, q, seq_lens, v, value`
- **missing in shapes**: `['cos_sin', 'flash_layout', 'is_neox', 'k_scale', 'key_cache', 'pos', 'slot_mapping', 'v_scale', 'value_cache']`
- extra-in-oracle (oracle reads non-kernel param): `['block_table', 'page_size', 'seq_lens']`
- **suggested fix**: shapes.yaml.inputs is missing ['cos_sin', 'flash_layout', 'is_neox', 'k_scale', 'key_cache', 'pos', 'slot_mapping', 'v_scale', 'value_cache'] that kernel `fused_qk_rope_reshape_and_cache` requires; add or rename keys.

### `L1_35_compute_src2dst_triton` (level1)

- kernel required (1): `topk_ids`
- oracle inputs[..] (2): `num_toks, reorder_ids`
- oracle outputs (1): `src2dst`
- shapes inputs (6): `input_tensor, masked_m, num_toks, recv_x, reorder_ids, x`
- **missing in shapes**: `['topk_ids']`
- extra-in-oracle (oracle reads non-kernel param): `['num_toks', 'reorder_ids']`
- **suggested fix**: shapes.yaml.inputs is missing ['topk_ids'] that kernel `cutlass_w4_run_moe_ep_preproess` requires; add or rename keys.

### `L1_36_compute_seg_indptr_triton` (level1)

- kernel required (5): `topk_ids, num_local_experts, hidden_states, top_k, block_shape`
- kernel optional (1): `output_dtype`
- oracle inputs[..] (2): `num_experts, topk_ids`
- oracle outputs (1): `seg_indptr`
- shapes inputs (9): `down_output, hidden_states, input, masked_m, num_experts, num_local_experts, output, problem_sizes1, topk_ids`
- **missing in shapes**: `['block_shape', 'top_k']`
- extra-in-oracle (oracle reads non-kernel param): `['num_experts']`
- **suggested fix**: shapes.yaml.inputs is missing ['block_shape', 'top_k'] that kernel `moe_ep_deepgemm_preprocess` requires; add or rename keys.

### `L1_37_compute_problem_sizes_w4a8` (level1)

- kernel required (6): `masked_m, problem_sizes1, problem_sizes2, n, k, num_experts`
- oracle inputs[..] (4): `k, masked_m, n, num_experts`
- oracle outputs (2): `problem_sizes1, problem_sizes2`
- shapes inputs (7): `input, k, masked_m, n, num_experts, problem_sizes1, x`
- **missing in shapes**: `['problem_sizes2']`
- **suggested fix**: shapes.yaml.inputs is missing ['problem_sizes2'] that kernel `compute_problem_sizes_w4a8` requires; add or rename keys.

### `L1_39_state_passing_fwd` (level1)

- kernel required (3): `states, dA_cumsum, last_chunk_indices`
- kernel optional (2): `initial_states, out_dtype`
- oracle inputs[..] (3): `dA_cumsum, initial_states, states`
- oracle outputs (1): `output`
- shapes inputs (2): `dA_cumsum, states`
- **missing in shapes**: `['last_chunk_indices']`
- missing-in-oracle (oracle reads non-shapes key): `['initial_states']`
- extra-in-oracle (oracle reads non-kernel param): `['initial_states']`
- **suggested fix**: shapes.yaml.inputs is missing ['last_chunk_indices'] that kernel `_state_passing_fwd` requires; add or rename keys.

### `L1_40_chunk_cumsum_fwd` (level1)

- kernel required (4): `dt, A, chunk_size, cu_chunk_seqlens`
- kernel optional (3): `dt_bias, dt_softplus, dt_limit`
- oracle inputs[..] (5): `A, dt, dt_bias, dt_limit, dt_softplus`
- oracle outputs (2): `dA_cumsum, dt_out`
- shapes inputs (4): `A, chunk_size, dt, dt_softplus`
- **missing in shapes**: `['cu_chunk_seqlens']`
- missing-in-oracle (oracle reads non-shapes key): `['dt_bias', 'dt_limit']`
- extra-in-oracle (oracle reads non-kernel param): `['dt_bias', 'dt_limit', 'dt_softplus']`
- **suggested fix**: shapes.yaml.inputs is missing ['cu_chunk_seqlens'] that kernel `_chunk_cumsum_fwd` requires; add or rename keys.

### `L2_03_chunk_state_fwd` (level2)

- kernel required (5): `B, x, dt, dA_cumsum, cu_chunk_seqlens`
- kernel optional (2): `states, states_in_fp32`
- oracle inputs[..] (4): `B, dA_cumsum, dt, x`
- oracle outputs (1): `states`
- shapes inputs (4): `B, dA_cumsum, dt, x`
- **missing in shapes**: `['cu_chunk_seqlens']`
- **suggested fix**: shapes.yaml.inputs is missing ['cu_chunk_seqlens'] that kernel `_chunk_state_fwd` requires; add or rename keys.

### `L2_02_chunk_scan_fwd` (level2)

- kernel required (9): `cb, x, dt, dA_cumsum, C, states, cu_chunk_seqlens, out, seq_idx`
- kernel optional (3): `D, z, initial_states`
- oracle inputs[..] (8): `C, D, cb, dA_cumsum, dt, states, x, z`
- oracle outputs (1): `output`
- shapes inputs (8): `C, D, cb, dA_cumsum, dt, states, x, z`
- **missing in shapes**: `['cu_chunk_seqlens', 'out', 'seq_idx']`
- extra-in-oracle (oracle reads non-kernel param): `['D', 'z']`
- **suggested fix**: shapes.yaml.inputs is missing ['cu_chunk_seqlens', 'out', 'seq_idx'] that kernel `_chunk_scan_fwd` requires; add or rename keys.

### `L2_11_w8a8_triton_block_scaled_mm` (level2)

- kernel required (5): `A, B, As, Bs, block_size`
- kernel optional (1): `output_dtype`
- oracle inputs[..] (6): `A, As, B, Bs, group_k, group_n`
- oracle outputs (1): `C`
- shapes inputs (8): `A, As, B, Bs, group_k, group_n, input, x`
- **missing in shapes**: `['block_size']`
- extra-in-oracle (oracle reads non-kernel param): `['group_k', 'group_n']`
- **suggested fix**: shapes.yaml.inputs is missing ['block_size'] that kernel `w8a8_triton_block_scaled_mm` requires; add or rename keys.

### `L2_17_layer_norm_gated_fwd` (level2)

- kernel required (4): `x, g, weight, bias`
- kernel optional (6): `activation, eps, residual, out_dtype, residual_dtype, is_rms_norm`
- oracle inputs[..] (6): `b, eps, g, residual, w, x`
- oracle outputs (1): `output`
- shapes inputs (8): `b, eps, g, k, q, residual, w, x`
- **missing in shapes**: `['bias', 'weight']`
- extra-in-oracle (oracle reads non-kernel param): `['b', 'eps', 'residual', 'w']`
- **suggested fix**: shapes.yaml.inputs is missing ['bias', 'weight'] that kernel `layer_norm_gated_fwd` requires; add or rename keys.

### `L2_18_fused_gdn_gating` (level2)

- kernel required (4): `A_log, a, b, dt_bias`
- kernel optional (2): `beta, threshold`
- oracle inputs[..] (5): `A_log, a, beta, dt_bias, threshold`
- oracle outputs (1): `output`
- shapes inputs (5): `A_log, a, beta, dt_bias, threshold`
- **missing in shapes**: `['b']`
- extra-in-oracle (oracle reads non-kernel param): `['beta', 'threshold']`
- **suggested fix**: shapes.yaml.inputs is missing ['b'] that kernel `fused_gdn_gating` requires; add or rename keys.

### `L2_19_fused_sigmoid_gating_delta_rule_update` (level2)

- kernel required (11): `A_log, a, dt_bias, softplus_beta, softplus_threshold, q, k, v, b, initial_state_source, initial_state_indices`
- kernel optional (9): `scale, use_qk_l2norm_in_kernel, cu_seqlens, is_kda, disable_state_update, intermediate_states_buffer, intermediate_state_indices, cache_steps, retrieve_parent_token`
- oracle inputs[..] (10): `A_log, a, beta, dt_bias, k, q, softplus_beta, softplus_threshold, state, v`
- oracle outputs (2): `output, state`
- shapes inputs (9): `A_log, a, beta, dt_bias, k, q, softplus_beta, state, v`
- **missing in shapes**: `['b', 'initial_state_indices', 'initial_state_source', 'softplus_threshold']`
- missing-in-oracle (oracle reads non-shapes key): `['softplus_threshold']`
- extra-in-oracle (oracle reads non-kernel param): `['beta', 'state']`
- **suggested fix**: shapes.yaml.inputs is missing ['b', 'initial_state_indices', 'initial_state_source', 'softplus_threshold'] that kernel `fused_sigmoid_gating_delta_rule_update` requires; add or rename keys.

### `L2_21_causal_conv1d_fwd` (level2)

- kernel required (5): `x, weight, bias, conv_states, query_start_loc`
- kernel optional (12): `cache_indices, has_initial_state, activation, pad_slot_id, null_block_id, block_idx_first_scheduled_token, block_idx_last_scheduled_token, initial_state_idx, num_computed_tokens, block_size_to_align, metadata, validate_data`
- oracle inputs[..] (4): `activation, bias, weight, x`
- oracle outputs (1): `output`
- shapes inputs (4): `activation, bias, weight, x`
- **missing in shapes**: `['conv_states', 'query_start_loc']`
- extra-in-oracle (oracle reads non-kernel param): `['activation']`
- **suggested fix**: shapes.yaml.inputs is missing ['conv_states', 'query_start_loc'] that kernel `causal_conv1d_fn` requires; add or rename keys.

### `L2_22_linear_attn_decode` (level2)

- kernel required (6): `q, k, v, kv_caches, slope_rate, slot_idx`
- kernel optional (1): `BLOCK_SIZE`
- oracle inputs[..] (5): `k, kv_caches, q, slope_rate, v`
- oracle outputs (2): `kv_cache, output`
- shapes inputs (5): `k, kv_caches, q, slope_rate, v`
- **missing in shapes**: `['slot_idx']`
- **suggested fix**: shapes.yaml.inputs is missing ['slot_idx'] that kernel `linear_decode_forward_triton` requires; add or rename keys.

### `L2_23_fwd_kv_parallel` (level2)

- kernel required (6): `q, k, v, kv_caches, slope_rate, slot_idx`
- kernel optional (1): `BLOCK_SIZE`
- oracle inputs[..] (4): `k, q, slope, v`
- oracle outputs (1): `output`
- shapes inputs (4): `k, q, slope, v`
- **missing in shapes**: `['kv_caches', 'slope_rate', 'slot_idx']`
- extra-in-oracle (oracle reads non-kernel param): `['slope']`
- **suggested fix**: shapes.yaml.inputs is missing ['kv_caches', 'slope_rate', 'slot_idx'] that kernel `linear_decode_forward_triton` requires; add or rename keys.

### `L2_24_fwd_kv_reduce` (level2)

- kernel required (6): `q, k, v, kv_caches, slope_rate, slot_idx`
- kernel optional (1): `BLOCK_SIZE`
- oracle inputs[..] (4): `k, q, slope, v`
- oracle outputs (1): `output`
- shapes inputs (4): `k, q, slope, v`
- **missing in shapes**: `['kv_caches', 'slope_rate', 'slot_idx']`
- extra-in-oracle (oracle reads non-kernel param): `['slope']`
- **suggested fix**: shapes.yaml.inputs is missing ['kv_caches', 'slope_rate', 'slot_idx'] that kernel `linear_decode_forward_triton` requires; add or rename keys.

### `L2_25_fwd_kernel_ep_gather` (level2)

- kernel required (6): `input_tensor, recv_topk_ids, recv_topk_weight, input_index, expert_map, output_tensor`
- oracle inputs[..] (2): `input_index, input_tensor`
- oracle outputs (1): `output`
- shapes inputs (4): `input_index, input_tensor, recv_x, recv_x_scale`
- **missing in shapes**: `['expert_map', 'output_tensor', 'recv_topk_ids', 'recv_topk_weight']`
- **suggested fix**: shapes.yaml.inputs is missing ['expert_map', 'output_tensor', 'recv_topk_ids', 'recv_topk_weight'] that kernel `ep_gather` requires; add or rename keys.

### `L2_26_fwd_kernel_ep_scatter_1` (level2)

- kernel required (10): `recv_x, recv_x_scale, recv_topk, num_recv_tokens_per_expert, expert_map, expert_start_loc, output_tensor, output_tensor_scale, m_indices, output_index`
- oracle inputs[..] (3): `dst_index, out_rows, recv_x`
- oracle outputs (1): `output`
- shapes inputs (5): `dst_index, input_tensor, out_rows, recv_x, recv_x_scale`
- **missing in shapes**: `['expert_map', 'expert_start_loc', 'm_indices', 'num_recv_tokens_per_expert', 'output_index', 'output_tensor', 'output_tensor_scale', 'recv_topk']`
- extra-in-oracle (oracle reads non-kernel param): `['dst_index', 'out_rows']`
- **suggested fix**: shapes.yaml.inputs is missing ['expert_map', 'expert_start_loc', 'm_indices', 'num_recv_tokens_per_expert', 'output_index', 'output_tensor', 'output_tensor_scale', 'recv_topk'] that kernel `ep_scatter` requires; add or rename keys.

### `L2_27_deepep_permute_triton` (level2)

- kernel required (5): `x, x_scale, masked_m, output_scale, output`
- oracle inputs[..] (3): `src2dst, topk, x`
- oracle outputs (1): `output`
- shapes inputs (9): `down_output, input, masked_m, output, problem_sizes1, src2dst, topk, x, x_scale`
- **missing in shapes**: `['output_scale']`
- extra-in-oracle (oracle reads non-kernel param): `['src2dst', 'topk']`
- **suggested fix**: shapes.yaml.inputs is missing ['output_scale'] that kernel `fp8_per_token_to_per_tensor_quant_triton` requires; add or rename keys.

### `L2_28_deepep_post_reorder_triton` (level2)

- kernel required (10): `down_output, output, src2dst, topk_ids, topk_weights, num_local_experts, topk, num_tokens, hidden_size, routed_scaling_factor`
- oracle inputs[..] (4): `src2dst, topk, topk_weights, x`
- oracle outputs (1): `output`
- shapes inputs (10): `down_output, input, masked_m, output, problem_sizes1, src2dst, topk, topk_weights, x, x_scale`
- **missing in shapes**: `['hidden_size', 'num_local_experts', 'num_tokens', 'routed_scaling_factor', 'topk_ids']`
- extra-in-oracle (oracle reads non-kernel param): `['x']`
- **suggested fix**: shapes.yaml.inputs is missing ['hidden_size', 'num_local_experts', 'num_tokens', 'routed_scaling_factor', 'topk_ids'] that kernel `post_reorder_for_cutlass_moe` requires; add or rename keys.

### `L2_31_moe_align_block_size_stage2` (level2)

- kernel required (6): `topk_ids, num_experts, block_size, sorted_token_ids, expert_ids, num_tokens_post_pad`
- oracle inputs[..] (3): `block_size, num_experts, topk_ids`
- oracle outputs (3): `expert_ids, num_tokens_post_pad, sorted_token_ids`
- shapes inputs (3): `block_size, num_experts, topk_ids`
- **missing in shapes**: `['expert_ids', 'num_tokens_post_pad', 'sorted_token_ids']`
- **suggested fix**: shapes.yaml.inputs is missing ['expert_ids', 'num_tokens_post_pad', 'sorted_token_ids'] that kernel `moe_align_block_size_triton` requires; add or rename keys.

### `L2_33_fwd_grouped_kernel_stage1_rope` (level2)

- kernel required (15): `q, k_buffer, v_buffer, att_out, k_pe_tokens_out, kv_lora_rank, cos_sin_cache, positions, rotary_dim, kv_indptr, kv_indices, num_kv_splits, sm_scale, logit_cap, use_rope`
- kernel optional (1): `is_neox_style`
- oracle inputs[..] (7): `cos, k_buffer, q, rope_dim, scale, sin, v_buffer`
- oracle outputs (1): `output`
- shapes inputs (5): `att_out, k_buffer, q, scale, v_buffer`
- **missing in shapes**: `['cos_sin_cache', 'k_pe_tokens_out', 'kv_indices', 'kv_indptr', 'kv_lora_rank', 'logit_cap', 'num_kv_splits', 'positions', 'rotary_dim', 'sm_scale', 'use_rope']`
- missing-in-oracle (oracle reads non-shapes key): `['cos', 'rope_dim', 'sin']`
- extra-in-oracle (oracle reads non-kernel param): `['cos', 'rope_dim', 'scale', 'sin']`
- **suggested fix**: shapes.yaml.inputs is missing ['cos_sin_cache', 'k_pe_tokens_out', 'kv_indices', 'kv_indptr', 'kv_lora_rank', 'logit_cap', 'num_kv_splits', 'positions', 'rotary_dim', 'sm_scale', 'use_rope'] that kernel `_decode_grouped_att_m_fwd_rope` requires; add or rename keys.

### `L2_35_pre_reorder_triton_kernel_for_cutlass_moe` (level2)

- kernel required (9): `input, gateup_input, src2dst, topk_ids, a1_scales, num_local_experts, topk, num_tokens, hidden_size`
- oracle inputs[..] (3): `input, src2dst, topk`
- oracle outputs (1): `gateup_input`
- shapes inputs (7): `down_output, input, masked_m, output, src2dst, topk, x`
- **missing in shapes**: `['a1_scales', 'gateup_input', 'hidden_size', 'num_local_experts', 'num_tokens', 'topk_ids']`
- **suggested fix**: shapes.yaml.inputs is missing ['a1_scales', 'gateup_input', 'hidden_size', 'num_local_experts', 'num_tokens', 'topk_ids'] that kernel `pre_reorder_for_cutlass_moe` requires; add or rename keys.

### `L2_36_post_reorder_triton_kernel_for_cutlass_moe` (level2)

- kernel required (10): `down_output, output, src2dst, topk_ids, topk_weights, num_local_experts, topk, num_tokens, hidden_size, routed_scaling_factor`
- oracle inputs[..] (4): `down_output, src2dst, topk, topk_weights`
- oracle outputs (1): `output`
- shapes inputs (9): `down_output, input, input_tensor, masked_m, output, src2dst, topk, topk_weights, x`
- **missing in shapes**: `['hidden_size', 'num_local_experts', 'num_tokens', 'routed_scaling_factor', 'topk_ids']`
- **suggested fix**: shapes.yaml.inputs is missing ['hidden_size', 'num_local_experts', 'num_tokens', 'routed_scaling_factor', 'topk_ids'] that kernel `post_reorder_for_cutlass_moe` requires; add or rename keys.

### `L2_37_silu_and_mul_post_quant` (level2)

- kernel required (5): `input, output, output_scale, quant_group_size, masked_m`
- kernel optional (1): `scale_ue8m0`
- oracle inputs[..] (4): `fp8_max, input, masked_m, quant_group_size`
- oracle outputs (2): `output, output_scale`
- shapes inputs (7): `fp8_max, input, masked_m, output, problem_sizes1, quant_group_size, x`
- **missing in shapes**: `['output_scale']`
- extra-in-oracle (oracle reads non-kernel param): `['fp8_max']`
- **suggested fix**: shapes.yaml.inputs is missing ['output_scale'] that kernel `silu_and_mul_masked_post_quant_fwd` requires; add or rename keys.

### `L3_02_kernel_paged_attention_2d` (level3)

- kernel required (14): `query, key, value, output, kv_cache_dtype, key_cache, value_cache, block_table, query_start_loc, seq_lens, max_seq_len, max_query_len, k_scale, v_scale`
- kernel optional (7): `alibi_slopes, sliding_window, sm_scale, output_scale, sinks, is_block_table_ptr, causal`
- oracle inputs[..] (8): `alibi_slopes, block_table, key_cache, query, scale, seq_lens, sliding_window, value_cache`
- oracle outputs (1): `output`
- shapes inputs (7): `block_table, key, key_cache, query, scale, seq_lens, value_cache`
- **missing in shapes**: `['k_scale', 'kv_cache_dtype', 'max_query_len', 'max_seq_len', 'output', 'query_start_loc', 'v_scale', 'value']`
- missing-in-oracle (oracle reads non-shapes key): `['alibi_slopes', 'sliding_window']`
- extra-in-oracle (oracle reads non-kernel param): `['alibi_slopes', 'scale', 'sliding_window']`
- **suggested fix**: shapes.yaml.inputs is missing ['k_scale', 'kv_cache_dtype', 'max_query_len', 'max_seq_len', 'output', 'query_start_loc', 'v_scale', 'value'] that kernel `chunked_prefill_paged_decode` requires; add or rename keys.

### `L3_03_chunk_state_varlen` (level3)

- kernel required (6): `B, x, dt, dA_cumsum, cu_seqlens, chunk_states`
- kernel optional (1): `initial_states`
- oracle inputs[..] (6): `B, cu_seqlens, dA_cumsum, dt, initial_states, x`
- oracle outputs (1): `states`
- shapes inputs (5): `B, cu_seqlens, dA_cumsum, dt, x`
- **missing in shapes**: `['chunk_states']`
- missing-in-oracle (oracle reads non-shapes key): `['initial_states']`
- extra-in-oracle (oracle reads non-kernel param): `['initial_states']`
- **suggested fix**: shapes.yaml.inputs is missing ['chunk_states'] that kernel `chunk_state_varlen` requires; add or rename keys.

### `L3_06_fwd_grouped_kernel_stage1` (level3)

- kernel required (12): `q, k_buffer, v_buffer, att_out, Req_to_tokens, B_Seqlen, num_kv_splits, sm_scale, page_size, logit_cap, k_scale, v_scale`
- kernel optional (1): `is_mla`
- oracle inputs[..] (7): `cos, k_buffer, q, rope_dim, scale, sin, v_buffer`
- oracle outputs (1): `output`
- shapes inputs (5): `att_out, k_buffer, q, scale, v_buffer`
- **missing in shapes**: `['B_Seqlen', 'Req_to_tokens', 'k_scale', 'logit_cap', 'num_kv_splits', 'page_size', 'sm_scale', 'v_scale']`
- missing-in-oracle (oracle reads non-shapes key): `['cos', 'rope_dim', 'sin']`
- extra-in-oracle (oracle reads non-kernel param): `['cos', 'rope_dim', 'scale', 'sin']`
- **suggested fix**: shapes.yaml.inputs is missing ['B_Seqlen', 'Req_to_tokens', 'k_scale', 'logit_cap', 'num_kv_splits', 'page_size', 'sm_scale', 'v_scale'] that kernel `_decode_grouped_att_m_fwd` requires; add or rename keys.

### `L3_07_correct_attn_cp_out` (level3)

- kernel required (2): `x, lengths`
- kernel optional (3): `pad_value, block_t, block_d`
- oracle inputs[..] (4): `cp_rank, is_lse_base_on_e, lses, out`
- oracle outputs (1): `output`
- shapes inputs (5): `cp_rank, is_lse_base_on_e, lses, out, x`
- **missing in shapes**: `['lengths']`
- extra-in-oracle (oracle reads non-kernel param): `['cp_rank', 'is_lse_base_on_e', 'lses', 'out']`
- **suggested fix**: shapes.yaml.inputs is missing ['lengths'] that kernel `pack_seq_triton` requires; add or rename keys.

### `L3_05_fwd_kernel_alibi` (level3)

- kernel required (14): `q, k, v, o, kv_cache_dtype, k_cache, v_cache, b_loc, b_start_loc, b_seq_len, max_seq_len, max_input_len, k_scale, v_scale`
- kernel optional (8): `alibi_slopes, sliding_window, sm_scale, skip_decode, fp8_out_scale, sinks, is_block_table_ptr, causal`
- oracle inputs[..] (5): `alibi_slopes, k, q, scale, v`
- oracle outputs (1): `output`
- shapes inputs (6): `alibi_slopes, k, o, q, scale, v`
- **missing in shapes**: `['b_loc', 'b_seq_len', 'b_start_loc', 'k_cache', 'k_scale', 'kv_cache_dtype', 'max_input_len', 'max_seq_len', 'v_cache', 'v_scale']`
- extra-in-oracle (oracle reads non-kernel param): `['alibi_slopes', 'scale']`
- **suggested fix**: shapes.yaml.inputs is missing ['b_loc', 'b_seq_len', 'b_start_loc', 'k_cache', 'k_scale', 'kv_cache_dtype', 'max_input_len', 'max_seq_len', 'v_cache', 'v_scale'] that kernel `context_attention_fwd` requires; add or rename keys.

### `L3_09_fused_moe` (level3)

- kernel required (18): `A, B, C, A_scale, B_scale, topk_weights, sorted_token_ids, expert_ids, num_tokens_post_padded, mul_routed_weight, top_k, config, compute_type, use_fp8_w8a8, use_int8_w8a8, use_int8_w8a16, use_int4_w4a16, per_channel_quant`
- kernel optional (2): `block_shape, B_bias`
- oracle inputs[..] (7): `a, apply_weight, b, block_size, expert_ids, sorted_token_ids, topk_weights`
- oracle outputs (1): `output`
- shapes inputs (7): `a, apply_weight, b, block_size, expert_ids, sorted_token_ids, topk_weights`
- **missing in shapes**: `['A', 'A_scale', 'B', 'B_scale', 'C', 'compute_type', 'config', 'mul_routed_weight', 'num_tokens_post_padded', 'per_channel_quant', 'top_k', 'use_fp8_w8a8', 'use_int4_w4a16', 'use_int8_w8a16', 'use_int8_w8a8']`
- extra-in-oracle (oracle reads non-kernel param): `['a', 'apply_weight', 'b', 'block_size']`
- **suggested fix**: shapes.yaml.inputs is missing ['A', 'A_scale', 'B', 'B_scale', 'C', 'compute_type', 'config', 'mul_routed_weight', 'num_tokens_post_padded', 'per_channel_quant', 'top_k', 'use_fp8_w8a8', 'use_int4_w4a16', 'use_int8_w8a16', 'use_int8_w8a8'] that kernel `invoke_fused_moe_triton_kernel` requires; add or rename keys.

### `L3_11_expert_triton` (level3)

- kernel required (13): `A, B, C, expert_num_tokens, compute_type, A_scale, B_scale, B_zp, use_fp8_w8a8, use_int8_w8a16, use_int4_w4a16, config, per_act_token_quant`
- kernel optional (1): `block_shape`
- oracle inputs[..] (3): `a, b, expert_num_tokens`
- oracle outputs (1): `output`
- shapes inputs (3): `a, b, expert_num_tokens`
- **missing in shapes**: `['A', 'A_scale', 'B', 'B_scale', 'B_zp', 'C', 'compute_type', 'config', 'per_act_token_quant', 'use_fp8_w8a8', 'use_int4_w4a16', 'use_int8_w8a16']`
- extra-in-oracle (oracle reads non-kernel param): `['a', 'b']`
- **suggested fix**: shapes.yaml.inputs is missing ['A', 'A_scale', 'B', 'B_scale', 'B_zp', 'C', 'compute_type', 'config', 'per_act_token_quant', 'use_fp8_w8a8', 'use_int4_w4a16', 'use_int8_w8a16'] that kernel `invoke_moe_batched_triton_kernel` requires; add or rename keys.

### `L3_10_fused_moe_kernel_gptq_awq` (level3)

- kernel required (16): `A, B, C, B_scale, B_zp, topk_weights, sorted_token_ids, expert_ids, num_tokens_post_padded, mul_routed_weight, top_k, config, compute_type, use_int8_w8a16, use_int4_w4a16, block_shape`
- oracle inputs[..] (7): `a, apply_weight, b, block_size, expert_ids, sorted_token_ids, topk_weights`
- oracle outputs (1): `output`
- shapes inputs (7): `a, apply_weight, b, block_size, expert_ids, sorted_token_ids, topk_weights`
- **missing in shapes**: `['A', 'B', 'B_scale', 'B_zp', 'C', 'block_shape', 'compute_type', 'config', 'mul_routed_weight', 'num_tokens_post_padded', 'top_k', 'use_int4_w4a16', 'use_int8_w8a16']`
- extra-in-oracle (oracle reads non-kernel param): `['a', 'apply_weight', 'b', 'block_size']`
- **suggested fix**: shapes.yaml.inputs is missing ['A', 'B', 'B_scale', 'B_zp', 'C', 'block_shape', 'compute_type', 'config', 'mul_routed_weight', 'num_tokens_post_padded', 'top_k', 'use_int4_w4a16', 'use_int8_w8a16'] that kernel `invoke_fused_moe_wna16_triton_kernel` requires; add or rename keys.

### `L3_14_fused_recurrent_gated_delta_rule_fwd` (level3)

- kernel required (7): `q, k, v, g, beta, scale, initial_state`
- kernel optional (5): `inplace_final_state, cu_seqlens, ssm_state_indices, num_accepted_tokens, use_qk_l2norm_in_kernel`
- oracle inputs[..] (7): `beta, g, initial_state, k, q, scale, v`
- oracle outputs (2): `final_state, output`
- shapes inputs (6): `beta, g, k, q, scale, v`
- **missing in shapes**: `['initial_state']`
- missing-in-oracle (oracle reads non-shapes key): `['initial_state']`
- **suggested fix**: shapes.yaml.inputs is missing ['initial_state'] that kernel `fused_recurrent_gated_delta_rule_fwd` requires; add or rename keys.

### `L3_12_moe_mmk` (level3)

- kernel required (13): `A, B, C, expert_num_tokens, compute_type, A_scale, B_scale, B_zp, use_fp8_w8a8, use_int8_w8a16, use_int4_w4a16, config, per_act_token_quant`
- kernel optional (1): `block_shape`
- oracle inputs[..] (3): `a, b, expert_num_tokens`
- oracle outputs (1): `output`
- shapes inputs (3): `a, b, expert_num_tokens`
- **missing in shapes**: `['A', 'A_scale', 'B', 'B_scale', 'B_zp', 'C', 'compute_type', 'config', 'per_act_token_quant', 'use_fp8_w8a8', 'use_int4_w4a16', 'use_int8_w8a16']`
- extra-in-oracle (oracle reads non-kernel param): `['a', 'b']`
- **suggested fix**: shapes.yaml.inputs is missing ['A', 'A_scale', 'B', 'B_scale', 'B_zp', 'C', 'compute_type', 'config', 'per_act_token_quant', 'use_fp8_w8a8', 'use_int4_w4a16', 'use_int8_w8a16'] that kernel `invoke_moe_batched_triton_kernel` requires; add or rename keys.

### `L3_15_fused_recurrent_gated_delta_rule_update_fwd` (level3)

- kernel required (8): `q, k, v, g, beta, scale, initial_state_source, initial_state_indices`
- kernel optional (8): `use_qk_l2norm_in_kernel, cu_seqlens, disable_state_update, disable_output_calculation, intermediate_states_buffer, intermediate_state_indices, cache_steps, retrieve_parent_token`
- oracle inputs[..] (7): `beta, g, initial_state, k, q, scale, v`
- oracle outputs (2): `final_state, output`
- shapes inputs (6): `beta, g, k, q, scale, v`
- **missing in shapes**: `['initial_state_indices', 'initial_state_source']`
- missing-in-oracle (oracle reads non-shapes key): `['initial_state']`
- extra-in-oracle (oracle reads non-kernel param): `['initial_state']`
- **suggested fix**: shapes.yaml.inputs is missing ['initial_state_indices', 'initial_state_source'] that kernel `fused_recurrent_gated_delta_rule_update_fwd` requires; add or rename keys.

### `L3_21_solve_tril_16x16` (level3)

- kernel required (10): `A, Ai, cu_seqlens, chunk_indices, T, H, BT, USE_TMA, IS_VARLEN, DOT_PRECISION`
- oracle inputs[..] (1): `A`
- oracle outputs (1): `output`
- shapes inputs (1): `A`
- **missing in shapes**: `['Ai', 'BT', 'DOT_PRECISION', 'H', 'IS_VARLEN', 'T', 'USE_TMA', 'chunk_indices', 'cu_seqlens']`
- **suggested fix**: shapes.yaml.inputs is missing ['Ai', 'BT', 'DOT_PRECISION', 'H', 'IS_VARLEN', 'T', 'USE_TMA', 'chunk_indices', 'cu_seqlens'] that kernel `solve_tril_16x16_kernel` requires; add or rename keys.

### `L3_23_merge_16x16_to_32x32_inverse` (level3)

- kernel required (10): `A, Ai, cu_seqlens, chunk_indices, T, H, BT, USE_TMA, IS_VARLEN, DOT_PRECISION`
- oracle inputs[..] (1): `A`
- oracle outputs (1): `output`
- shapes inputs (1): `A`
- **missing in shapes**: `['Ai', 'BT', 'DOT_PRECISION', 'H', 'IS_VARLEN', 'T', 'USE_TMA', 'chunk_indices', 'cu_seqlens']`
- **suggested fix**: shapes.yaml.inputs is missing ['Ai', 'BT', 'DOT_PRECISION', 'H', 'IS_VARLEN', 'T', 'USE_TMA', 'chunk_indices', 'cu_seqlens'] that kernel `merge_16x16_to_32x32_inverse_kernel` requires; add or rename keys.

## NEEDS_BRIDGE (17)

### `L1_09_quantize_k_cache_fast` (level1)

- kernel required (2): `k_nope, k_rope`
- kernel optional (1): `group_size`
- oracle inputs[..] (4): `fp8_max, group_size, k_nope, k_rope`
- oracle outputs (3): `nope_q, nope_s, rope`
- shapes inputs (4): `fp8_max, group_size, k_nope, k_rope`
- extra-in-oracle (oracle reads non-kernel param): `['fp8_max', 'group_size']`
- **suggested fix**: oracle reads ['fp8_max', 'group_size', 'k_nope', 'k_rope'] including ['fp8_max'] which kernel `_quantize_k_cache_fast` does not have (kernel required=['k_nope', 'k_rope'], optional=['group_size']); bridge build_inputs to derive kernel inputs from oracle inputs OR rewrite oracle.

### `L1_11_awq_dequantize` (level1)

- kernel required (3): `qweight, scales, zeros`
- kernel optional (2): `block_size_x, block_size_y`
- oracle inputs[..] (4): `group_size, qweight, scales, zeros`
- oracle outputs (1): `output`
- shapes inputs (5): `group_size, input, qweight, scales, zeros`
- extra-in-oracle (oracle reads non-kernel param): `['group_size']`
- **suggested fix**: oracle reads ['group_size', 'qweight', 'scales', 'zeros'] including ['group_size'] which kernel `awq_dequantize_triton` does not have (kernel required=['qweight', 'scales', 'zeros'], optional=['block_size_x', 'block_size_y']); bridge build_inputs to derive kernel inputs from oracle inputs OR rewrite oracle.

### `L1_14_layer_norm_fwd_1pass` (level1)

- kernel required (4): `x, weight, bias, eps`
- kernel optional (5): `z, out, group_size, norm_before_gate, is_rms_norm`
- oracle inputs[..] (5): `bias, eps, residual, weight, x`
- oracle outputs (1): `output`
- shapes inputs (5): `bias, eps, residual, weight, x`
- extra-in-oracle (oracle reads non-kernel param): `['residual']`
- **suggested fix**: oracle reads ['bias', 'eps', 'residual', 'weight', 'x'] including ['residual'] which kernel `_layer_norm_fwd` does not have (kernel required=['bias', 'eps', 'weight', 'x'], optional=['group_size', 'is_rms_norm', 'norm_before_gate', 'out', 'z']); bridge build_inputs to derive kernel inputs from oracle inputs OR rewrite oracle.

### `L1_17_gelu_and_mul` (level1)

- kernel required (1): `hidden_states`
- kernel optional (3): `scales, quantize, out`
- oracle inputs[..] (2): `approximate, x`
- oracle outputs (1): `output`
- shapes inputs (6): `approximate, hidden_states, residual, softcap_const, weight1, x`
- extra-in-oracle (oracle reads non-kernel param): `['approximate', 'x']`
- **suggested fix**: oracle reads ['approximate', 'x'] including ['approximate', 'x'] which kernel `gelu_and_mul_triton` does not have (kernel required=['hidden_states'], optional=['out', 'quantize', 'scales']); bridge build_inputs to derive kernel inputs from oracle inputs OR rewrite oracle.

### `L1_30_act_quant` (level1)

- kernel required (1): `x`
- kernel optional (2): `block_size, scale_fmt`
- oracle inputs[..] (3): `block_size, fp8_max, x`
- oracle outputs (2): `q, scale`
- shapes inputs (2): `block_size, x`
- missing-in-oracle (oracle reads non-shapes key): `['fp8_max']`
- extra-in-oracle (oracle reads non-kernel param): `['block_size', 'fp8_max']`
- **suggested fix**: oracle reads ['block_size', 'fp8_max', 'x'] including ['fp8_max'] which kernel `act_quant` does not have (kernel required=['x'], optional=['block_size', 'scale_fmt']); bridge build_inputs to derive kernel inputs from oracle inputs OR rewrite oracle.

### `L1_38_silu_mul_fp8_quant_deep_gemm` (level1)

- kernel required (2): `y, tokens_per_expert`
- kernel optional (3): `num_parallel_tokens, group_size, quant_scale_fmt`
- oracle inputs[..] (4): `fp8_max, group_size, tokens_per_expert, y`
- oracle outputs (2): `output, scale`
- shapes inputs (4): `fp8_max, group_size, tokens_per_expert, y`
- extra-in-oracle (oracle reads non-kernel param): `['fp8_max', 'group_size']`
- **suggested fix**: oracle reads ['fp8_max', 'group_size', 'tokens_per_expert', 'y'] including ['fp8_max'] which kernel `persistent_masked_m_silu_mul_quant` does not have (kernel required=['tokens_per_expert', 'y'], optional=['group_size', 'num_parallel_tokens', 'quant_scale_fmt']); bridge build_inputs to derive kernel inputs from oracle inputs OR rewrite oracle.

### `L2_04_chunked_lora_expand` (level2)

- kernel required (6): `x, weights, batch_info, slice_offsets, max_slice_size, base_output`
- oracle inputs[..] (7): `base_output, lora_ranks, scalings, seg_indptr, weight_indices, weights, x`
- oracle outputs (1): `output`
- shapes inputs (10): `base_output, batch_info, lora_ranks, max_slice_size, scalings, seg_indptr, slice_offsets, weight_indices, weights, x`
- extra-in-oracle (oracle reads non-kernel param): `['lora_ranks', 'scalings', 'seg_indptr', 'weight_indices']`
- **suggested fix**: oracle reads ['base_output', 'lora_ranks', 'scalings', 'seg_indptr', 'weight_indices', 'weights', 'x'] including ['lora_ranks', 'scalings', 'seg_indptr', 'weight_indices'] which kernel `chunked_sgmv_lora_expand_forward` does not have (kernel required=['base_output', 'batch_info', 'max_slice_size', 'slice_offsets', 'weights', 'x'], optional=[]); bridge build_inputs to derive kernel inputs from oracle inputs OR rewrite oracle.

### `L2_05_chunked_lora_shrink` (level2)

- kernel required (4): `x, weights, batch_info, num_slices`
- oracle inputs[..] (5): `lora_ranks, seg_indptr, weight_indices, weights, x`
- oracle outputs (1): `output`
- shapes inputs (8): `batch_info, lora_ranks, num_slices, scalings, seg_indptr, weight_indices, weights, x`
- extra-in-oracle (oracle reads non-kernel param): `['lora_ranks', 'seg_indptr', 'weight_indices']`
- **suggested fix**: oracle reads ['lora_ranks', 'seg_indptr', 'weight_indices', 'weights', 'x'] including ['lora_ranks', 'seg_indptr', 'weight_indices'] which kernel `chunked_sgmv_lora_shrink_forward` does not have (kernel required=['batch_info', 'num_slices', 'weights', 'x'], optional=[]); bridge build_inputs to derive kernel inputs from oracle inputs OR rewrite oracle.

### `L2_06_qkv_lora_b` (level2)

- kernel required (5): `x, qkv_lora_b, batch_info, output_offset, max_qkv_out_dim`
- kernel optional (2): `base_output, n_slices`
- oracle inputs[..] (7): `base_output, lora_ranks, scalings, seg_indptr, weight_indices, weights, x`
- oracle outputs (1): `output`
- shapes inputs (9): `batch_info, lora_ranks, max_qkv_out_dim, output_offset, qkv_lora_b, scalings, seg_indptr, weight_indices, x`
- missing-in-oracle (oracle reads non-shapes key): `['base_output', 'weights']`
- extra-in-oracle (oracle reads non-kernel param): `['base_output', 'lora_ranks', 'scalings', 'seg_indptr', 'weight_indices', 'weights']`
- **suggested fix**: oracle reads ['base_output', 'lora_ranks', 'scalings', 'seg_indptr', 'weight_indices', 'weights', 'x'] including ['lora_ranks', 'scalings', 'seg_indptr', 'weight_indices', 'weights'] which kernel `qkv_lora_b_fwd` does not have (kernel required=['batch_info', 'max_qkv_out_dim', 'output_offset', 'qkv_lora_b', 'x'], optional=['base_output', 'n_slices']); bridge build_inputs to derive kernel inputs from oracle inputs OR rewrite oracle.

### `L2_07_gate_up_lora_b` (level2)

- kernel required (4): `x, gate_up_lora_b, batch_info, output_dim`
- kernel optional (1): `base_output`
- oracle inputs[..] (7): `base_output, lora_ranks, scalings, seg_indptr, weight_indices, weights, x`
- oracle outputs (1): `output`
- shapes inputs (8): `batch_info, gate_up_lora_b, lora_ranks, output_dim, scalings, seg_indptr, weight_indices, x`
- missing-in-oracle (oracle reads non-shapes key): `['base_output', 'weights']`
- extra-in-oracle (oracle reads non-kernel param): `['base_output', 'lora_ranks', 'scalings', 'seg_indptr', 'weight_indices', 'weights']`
- **suggested fix**: oracle reads ['base_output', 'lora_ranks', 'scalings', 'seg_indptr', 'weight_indices', 'weights', 'x'] including ['lora_ranks', 'scalings', 'seg_indptr', 'weight_indices', 'weights'] which kernel `gate_up_lora_b_fwd` does not have (kernel required=['batch_info', 'gate_up_lora_b', 'output_dim', 'x'], optional=['base_output']); bridge build_inputs to derive kernel inputs from oracle inputs OR rewrite oracle.

### `L2_09_sgemm_lora_b` (level2)

- kernel required (3): `x, weights, batch_info`
- kernel optional (1): `base_output`
- oracle inputs[..] (7): `base_output, lora_ranks, scalings, seg_indptr, weight_indices, weights, x`
- oracle outputs (1): `output`
- shapes inputs (7): `batch_info, lora_ranks, scalings, seg_indptr, weight_indices, weights, x`
- missing-in-oracle (oracle reads non-shapes key): `['base_output']`
- extra-in-oracle (oracle reads non-kernel param): `['base_output', 'lora_ranks', 'scalings', 'seg_indptr', 'weight_indices']`
- **suggested fix**: oracle reads ['base_output', 'lora_ranks', 'scalings', 'seg_indptr', 'weight_indices', 'weights', 'x'] including ['lora_ranks', 'scalings', 'seg_indptr', 'weight_indices'] which kernel `sgemm_lora_b_fwd` does not have (kernel required=['batch_info', 'weights', 'x'], optional=['base_output']); bridge build_inputs to derive kernel inputs from oracle inputs OR rewrite oracle.

### `L2_08_sgemm_lora_a` (level2)

- kernel required (3): `x, weights, batch_info`
- kernel optional (1): `stack_num`
- oracle inputs[..] (5): `lora_ranks, seg_indptr, weight_indices, weights, x`
- oracle outputs (1): `output`
- shapes inputs (7): `batch_info, lora_ranks, scalings, seg_indptr, weight_indices, weights, x`
- extra-in-oracle (oracle reads non-kernel param): `['lora_ranks', 'seg_indptr', 'weight_indices']`
- **suggested fix**: oracle reads ['lora_ranks', 'seg_indptr', 'weight_indices', 'weights', 'x'] including ['lora_ranks', 'seg_indptr', 'weight_indices'] which kernel `sgemm_lora_a_fwd` does not have (kernel required=['batch_info', 'weights', 'x'], optional=['stack_num']); bridge build_inputs to derive kernel inputs from oracle inputs OR rewrite oracle.

### `L2_12_awq_gemm` (level2)

- kernel required (5): `input, qweight, scales, qzeros, split_k_iters`
- kernel optional (3): `block_size_m, block_size_n, block_size_k`
- oracle inputs[..] (5): `group_size, input, qweight, qzeros, scales`
- oracle outputs (1): `C`
- shapes inputs (6): `group_size, input, qweight, qzeros, scales, split_k_iters`
- extra-in-oracle (oracle reads non-kernel param): `['group_size']`
- **suggested fix**: oracle reads ['group_size', 'input', 'qweight', 'qzeros', 'scales'] including ['group_size'] which kernel `awq_gemm_triton` does not have (kernel required=['input', 'qweight', 'qzeros', 'scales', 'split_k_iters'], optional=['block_size_k', 'block_size_m', 'block_size_n']); bridge build_inputs to derive kernel inputs from oracle inputs OR rewrite oracle.

### `L2_15_fused_dual_residual_rmsnorm` (level2)

- kernel required (5): `x, residual, weight1, weight2, eps`
- kernel optional (1): `autotune`
- oracle inputs[..] (5): `eps, residual_a, residual_b, weight, x`
- oracle outputs (2): `hidden, output`
- shapes inputs (10): `eps, hidden_states, residual, residual_a, residual_b, softcap_const, weight, weight1, weight2, x`
- extra-in-oracle (oracle reads non-kernel param): `['residual_a', 'residual_b', 'weight']`
- **suggested fix**: oracle reads ['eps', 'residual_a', 'residual_b', 'weight', 'x'] including ['residual_a', 'residual_b', 'weight'] which kernel `fused_dual_residual_rmsnorm` does not have (kernel required=['eps', 'residual', 'weight1', 'weight2', 'x'], optional=['autotune']); bridge build_inputs to derive kernel inputs from oracle inputs OR rewrite oracle.

### `L2_16_fused_rmsnorm` (level2)

- kernel required (3): `x, weight, eps`
- kernel optional (2): `autotune, inplace`
- oracle inputs[..] (4): `eps, residual, weight, x`
- oracle outputs (2): `hidden, output`
- shapes inputs (8): `eps, hidden_states, residual, softcap_const, weight, weight1, weight2, x`
- extra-in-oracle (oracle reads non-kernel param): `['residual']`
- **suggested fix**: oracle reads ['eps', 'residual', 'weight', 'x'] including ['residual'] which kernel `fused_rmsnorm` does not have (kernel required=['eps', 'weight', 'x'], optional=['autotune', 'inplace']); bridge build_inputs to derive kernel inputs from oracle inputs OR rewrite oracle.

### `L2_38_concat_and_cast_mha_k` (level2)

- kernel required (3): `k, k_nope, k_rope`
- oracle inputs[..] (3): `k_nope, k_rope, out_dtype`
- oracle outputs (1): `k`
- shapes inputs (11): `extend_seq_lens, input_emb, k, k_nope, k_rope, key, out_dtype, q, seq_lens, v, value`
- extra-in-oracle (oracle reads non-kernel param): `['out_dtype']`
- **suggested fix**: oracle reads ['k_nope', 'k_rope', 'out_dtype'] including ['out_dtype'] which kernel `concat_and_cast_mha_k_triton` does not have (kernel required=['k', 'k_nope', 'k_rope'], optional=[]); bridge build_inputs to derive kernel inputs from oracle inputs OR rewrite oracle.

### `L3_01_kernel_unified_attention_3d` (level3)

- kernel required (16): `q, k, v, out, cu_seqlens_q, max_seqlen_q, seqused_k, max_seqlen_k, softmax_scale, causal, window_size, block_table, softcap, q_descale, k_descale, v_descale`
- kernel optional (15): `seq_threshold_3D, num_par_softmax_segments, softmax_segm_output, softmax_segm_max, softmax_segm_expsum, alibi_slopes, output_scale, qq_bias, sinks, mm_prefix_range, use_alibi_sqrt, kv_quant_mode, k_scale_cache, v_scale_cache, chunk_lookback`
- oracle inputs[..] (9): `alibi_slopes, causal, cu_seqlens_q, k, q, scale, seqused_k, softcap, v`
- oracle outputs (1): `output`
- shapes inputs (16): `block_table, causal, cu_seqlens_q, k, k_descale, max_seqlen_k, max_seqlen_q, out, q, q_descale, seqused_k, softcap, softmax_scale, v, v_descale, window_size`
- missing-in-oracle (oracle reads non-shapes key): `['alibi_slopes', 'scale']`
- extra-in-oracle (oracle reads non-kernel param): `['alibi_slopes', 'scale']`
- **suggested fix**: oracle reads ['alibi_slopes', 'causal', 'cu_seqlens_q', 'k', 'q', 'scale', 'seqused_k', 'softcap', 'v'] including ['scale'] which kernel `unified_attention` does not have (kernel required=['block_table', 'causal', 'cu_seqlens_q', 'k', 'k_descale', 'max_seqlen_k', 'max_seqlen_q', 'out', 'q', 'q_descale', 'seqused_k', 'softcap', 'softmax_scale', 'v', 'v_descale', 'window_size'], optional=['alibi_slopes', 'chunk_lookback', 'k_scale_cache', 'kv_quant_mode', 'mm_prefix_range', 'num_par_softmax_segments', 'output_scale', 'qq_bias', 'seq_threshold_3D', 'sinks', 'softmax_segm_expsum', 'softmax_segm_max', 'softmax_segm_output', 'use_alibi_sqrt', 'v_scale_cache']); bridge build_inputs to derive kernel inputs from oracle inputs OR rewrite oracle.

## RUNTIME_ERROR (1)

### `L1_03_convert_req_index_to_global_index` (level1)

- kernel required (3): `req_id, block_table, token_indices`
- kernel optional (7): `BLOCK_SIZE, NUM_TOPK_TOKENS, BLOCK_N, HAS_PREFILL_WORKSPACE, prefill_workspace_request_ids, prefill_workspace_starts, return_valid_counts`
- oracle inputs[..] (4): `BLOCK_SIZE, block_table, req_id, token_indices`
- oracle outputs (1): `out`
- shapes inputs (6): `BLOCK_N, BLOCK_SIZE, NUM_TOPK_TOKENS, block_table, req_id, token_indices`
- extra-in-oracle (oracle reads non-kernel param): `['BLOCK_SIZE']`
- runtime error: `oracle.run(P1): IndexError: index 3 is out of bounds for dimension 0 with size 2`
