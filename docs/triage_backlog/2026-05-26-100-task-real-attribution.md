---
date: 2026-05-26
topic: 100-task real attribution (independent of any categorization)
scope: raw data per task — resolver verdict + kernel signatures + oracle/shapes diff + cross-checks
method: pure grep + AST parse + resolver verdict; NO subjective categorization
---

# 100-Task Real Attribution (Ground Truth Data)

## 0. Method

- Parsed `kernel.py` via Python AST (top-level defs + class methods + decorator counts)
- Parsed `oracle_pytorch.py` via AST (`inputs["X"]` accesses + `run()` return dict keys)
- Parsed `shapes.yaml` `inputs:` block keys
- Cross-referenced `results/resolve_all.json` real reason + tried candidates
- Extracted Python exception class from resolver reason (no judgment call)
- **No categorization** — facts only; user makes the call.

## 1. Aggregate Counts

### Verdict distribution

| Verdict | Count |
|---|---|
| `shapes-incomplete` | 47 |
| `upstream-not-invocable-here` | 40 |
| `oracle-mismatch` | 6 |
| `PASS` | 5 |
| `smoke-fail` | 2 |

### Real error type (from resolver reason)

| Error type | Count |
|---|---|
| `ShapesIncompleteError` | 38 |
| `AttributeError` | 13 |
| `ExprError` | 9 |
| `AssertionError` | 8 |
| `RuntimeError` | 6 |
| `InterpreterError` | 5 |
| `oracle-consistent` | 4 |
| `oracle-mismatch` | 4 |
| `ModuleNotFoundError` | 3 |
| `NotImplementedError` | 2 |
| `ValueError` | 2 |
| `KeyError` | 1 |
| `upstream-not-invocable-here` | 1 |
| `ZeroDivisionError` | 1 |

### Structural facts

- meta.name NOT a callable in kernel.py: **0/100**
- meta.name NOT in resolver's tried list (non-PASS only): **90/95**
- kernel.py uses `@triton.heuristics`: **13/100**  
  Tasks: L1_13_layer_norm_fwd, L1_14_layer_norm_fwd_1pass, L2_17_layer_norm_gated_fwd, L3_13_chunk_gated_delta_rule_fwd_kernel_h_blockdim64, L3_14_fused_recurrent_gated_delta_rule_fwd, L3_16_chunk_fwd_kernel_o, L3_17_chunk_gla_fwd_kernel_o, L3_18_chunk_scaled_dot_kkt_fwd, L3_19_chunk_kda_scaled_dot_kkt_fwd_kernel_intra_sub_inter, L3_20_recompute_w_u_fwd, L3_21_solve_tril_16x16, L3_22_chunk_local_cumsum_vector, L3_23_merge_16x16_to_32x32_inverse
- oracle_pytorch.py is placeholder: **1/100**

### Non-stdlib imports in kernel.py (potential pip-dep gaps)

- `__future__`: 2 tasks
- `base`: 1 tasks
- `chunk_delta_h`: 3 tasks
- `collections.abc`: 9 tasks
- `cumsum`: 3 tasks
- `einops`: 5 tasks
- `flashinfer`: 1 tasks
- `fused_recurrent`: 3 tasks
- `index`: 10 tasks
- `json`: 4 tasks
- `l2norm`: 3 tasks
- `logging`: 12 tasks
- `mamba_ssm`: 3 tasks
- `numpy`: 7 tasks
- `op`: 12 tasks
- `packaging`: 2 tasks
- `prefix_prefill`: 1 tasks
- `pytest`: 1 tasks
- `transformers.activations`: 1 tasks
- `utils`: 8 tasks
- `yarn_scaling_rope`: 1 tasks

## 2. Per-Task Raw Data

Columns: `task_id | verdict | err_type | meta.name | meta.name∈kernel | meta.name∈tried |
`tried[0] | tried_count | oracle inputs | shapes inputs | oracle-shapes | shapes-oracle |
`kernel triton_jit | heuristics | kernel imports`

Sort: verdict_class then task_id.

### PASS (5 tasks)

| task_id | err | meta.name | nKn | nTr | tried[0] | nTried | oracle in | shapes in | o−s | s−o | jit/heur/at | imports |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `L1_01_bias` | oracle-consistent | `_bias_kernel` | ✓ | ✗ | `—` | 0 | `allowed_token_ids, expanded_idx_mapping, logit_bias, logit_b` | `allowed_token_ids, expanded_idx_mapping, logit_bia` | `—` | `—` | 1/0/0 | `—` |
| `L1_12_rms_norm` | oracle-consistent | `_rms_norm_kernel` | ✓ | ✗ | `—` | 0 | `eps, input, weight` | `eps, input, weight` | `—` | `—` | 1/0/0 | `—` |
| `L1_16_silu_and_mul` | oracle-consistent | `silu_and_mul_kernel` | ✓ | ✗ | `—` | 0 | `hidden_states` | `hidden_states` | `—` | `—` | 1/0/0 | `—` |
| `L1_26_merge_state` | oracle-consistent | `merge_state_kernel` | ✓ | ✗ | `—` | 0 | `prefix_lse, prefix_output, suffix_lse, suffix_output` | `prefix_lse, prefix_output, suffix_lse, suffix_outp` | `—` | `—` | 1/0/0 | `typing` |
| `L1_34_transform_index_page_table_decod` | — | `transform_index_page_table_dec` | ✓ | ✗ | `transform_index_page_table_d` | 1 | `page_size, page_table, topk_indices` | `page_size, page_table, topk_indices` | `—` | `—` | 1/0/0 | `typing` |

### shapes-incomplete (47 tasks)

| task_id | err | meta.name | nKn | nTr | tried[0] | nTried | oracle in | shapes in | o−s | s−o | jit/heur/at | imports |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `L1_06_reshape_and_cache_kernel_flash` | ShapesIncompleteEr | `reshape_and_cache_kernel_flash` | ✓ | ✗ | `triton_reshape_and_cache_fla` | 3 | `key, key_cache, slot_mapping, value, value_cache` | `key, key_cache, slot_mapping, value, value_cache` | `—` | `—` | 3/0/0 | `—` |
| `L1_07_dequantize_k_cache_fast` | ShapesIncompleteEr | `_dequantize_k_cache_fast_kerne` | ✓ | ✗ | `_dequantize_k_cache_fast` | 2 | `group_size, nope_q, nope_s, rope` | `group_size, nope_q, nope_s, rope` | `—` | `—` | 2/0/0 | `—` |
| `L1_08_dequantize_k_cache_paged` | ShapesIncompleteEr | `_dequantize_k_cache_paged_kern` | ✓ | ✗ | `dequantize_k_cache_paged` | 2 | `group_size, nope_q, nope_s, page_table, rope` | `group_size, nope_q, nope_s, page_table, rope` | `—` | `—` | 2/0/0 | `—` |
| `L1_10_per_token_group_quant_int8` | InterpreterError | `_per_token_group_quant_int8` | ✓ | ✗ | `per_token_group_quant_int8` | 3 | `eps, group_size, x` | `eps, group_size, x` | `—` | `—` | 3/0/0 | `functools, json, logging, os, ` |
| `L1_11_awq_dequantize` | AssertionError | `awq_dequantize_kernel` | ✓ | ✗ | `awq_dequantize_triton` | 2 | `group_size, qweight, scales, zeros` | `group_size, input, qweight, scales, zeros` | `—` | `input` | 2/0/0 | `—` |
| `L1_15_act_and_mul` | ShapesIncompleteEr | `act_and_mul_kernel` | ✓ | ✗ | `act_and_mul_triton` | 5 | `act, x` | `act, input, x` | `—` | `input` | 9/0/0 | `__future__, functools, collect` |
| `L1_17_gelu_and_mul` | ValueError | `gelu_and_mul_kernel` | ✓ | ✗ | `gelu_and_mul_triton` | 6 | `approximate, x` | `approximate, hidden_states, residual, softcap_cons` | `—` | `hidden_states, residual, softcap_const, ` | 6/0/0 | `typing` |
| `L1_18_fused_softcap` | ShapesIncompleteEr | `fused_softcap_kernel` | ✓ | ✗ | `fused_softcap` | 1 | `cap, x` | `cap, x` | `—` | `—` | 1/0/0 | `dataclasses, logging, typing` |
| `L1_20_log_softmax` | ShapesIncompleteEr | `_log_softmax_kernel` | ✓ | ✗ | `log_softmax` | 5 | `dim, input` | `dim, input` | `—` | `—` | 6/0/0 | `math, os, collections.abc, typ` |
| `L1_21_gumbel_sample` | ShapesIncompleteEr | `_gumbel_sample_kernel` | ✓ | ✗ | `gumbel_sample` | 2 | `—` | `—` | `—` | `—` | 4/0/0 | `—` |
| `L1_28_pack_bitmatrix` | ShapesIncompleteEr | `pack_bitmatrix` | ✓ | ✗ | `make_routing_data` | 1 | `num_experts, topk_ids` | `num_experts, topk_ids` | `—` | `—` | 1/0/0 | `utils` |
| `L1_29_prepare_eagle_inputs` | ShapesIncompleteEr | `_prepare_eagle_inputs_kernel` | ✓ | ✗ | `prepare_eagle_inputs` | 3 | `last_token_indices, num_rejected, num_sampled` | `current_draft_step, last_token_indices, num_reject` | `—` | `current_draft_step` | 3/0/0 | `typing` |
| `L1_30_act_quant` | ShapesIncompleteEr | `_act_quant_kernel` | ✓ | ✗ | `act_quant` | 2 | `block_size, fp8_max, x` | `block_size, x` | `fp8_max` | `—` | 2/0/0 | `typing` |
| `L1_32_create_flashinfer_kv_indices_tri` | ShapesIncompleteEr | `create_flashinfer_kv_indices_t` | ✓ | ✗ | `fused_qk_rope_reshape_and_ca` | 5 | `block_table, page_size, seq_lens` | `block_table, extend_seq_lens, input_emb, k, key, p` | `—` | `extend_seq_lens, input_emb, k, key, q` | 11/0/0 | `—` |
| `L1_33_create_flashmla_kv_indices_trito` | ShapesIncompleteEr | `create_flashmla_kv_indices_tri` | ✓ | ✗ | `fused_qk_rope_reshape_and_ca` | 5 | `block_table, page_size, seq_lens` | `block_table, extend_seq_lens, input_emb, k, key, p` | `—` | `extend_seq_lens, input_emb, k, key, q, v` | 11/0/0 | `—` |
| `L1_35_compute_src2dst_triton` | ShapesIncompleteEr | `compute_src2dst_triton_kernel` | ✓ | ✗ | `cutlass_w4_run_moe_ep_prepro` | 6 | `num_toks, reorder_ids` | `input_tensor, masked_m, num_toks, recv_x, reorder_` | `—` | `input_tensor, masked_m, recv_x, x` | 21/0/0 | `logging` |
| `L1_39_state_passing_fwd` | ShapesIncompleteEr | `_state_passing_fwd_kernel` | ✓ | ✗ | `_state_passing_fwd` | 1 | `dA_cumsum, initial_states, states` | `dA_cumsum, states` | `initial_states` | `—` | 1/0/1 | `—` |
| `L1_40_chunk_cumsum_fwd` | ShapesIncompleteEr | `_chunk_cumsum_fwd_kernel` | ✓ | ✗ | `_chunk_cumsum_fwd` | 2 | `A, dt, dt_bias, dt_limit, dt_softplus` | `A, chunk_size, dt, dt_softplus` | `dt_bias, dt_limit` | `chunk_size` | 2/0/2 | `mamba_ssm` |
| `L2_02_chunk_scan_fwd` | ShapesIncompleteEr | `_chunk_scan_fwd_kernel` | ✓ | ✗ | `_chunk_scan_fwd` | 1 | `C, D, cb, dA_cumsum, dt, states, x, z` | `C, D, cb, dA_cumsum, dt, states, x, z` | `—` | `—` | 1/0/1 | `packaging` |
| `L2_03_chunk_state_fwd` | ShapesIncompleteEr | `_chunk_state_fwd_kernel` | ✓ | ✗ | `_chunk_state_fwd` | 2 | `B, dA_cumsum, dt, x` | `B, dA_cumsum, dt, x` | `—` | `—` | 2/0/2 | `mamba_ssm` |
| `L2_11_w8a8_triton_block_scaled_mm` | ShapesIncompleteEr | `_w8a8_triton_block_scaled_mm` | ✓ | ✗ | `w8a8_triton_block_scaled_mm` | 4 | `A, As, B, Bs, group_k, group_n` | `A, As, B, Bs, group_k, group_n, input, x` | `—` | `input, x` | 5/0/0 | `functools, json, os, collectio` |
| `L2_12_awq_gemm` | AssertionError | `awq_gemm_kernel` | ✓ | ✗ | `awq_gemm_triton` | 2 | `group_size, input, qweight, qzeros, scales` | `group_size, input, qweight, qzeros, scales, split_` | `—` | `split_k_iters` | 2/0/0 | `—` |
| `L2_13_bmm` | AssertionError | `bmm_kernel` | ✓ | ✗ | `bmm_batch_invariant` | 5 | `a, b` | `a, b, input` | `—` | `input` | 6/0/0 | `math, os, collections.abc, typ` |
| `L2_14_matmul_kernel_persistent` | NotImplementedErro | `matmul_kernel_persistent` | ✓ | ✗ | `matmul_persistent` | 5 | `a, b, bias` | `a, b, bias, input` | `—` | `input` | 6/0/0 | `math, os, collections.abc, typ` |
| `L2_15_fused_dual_residual_rmsnorm` | ShapesIncompleteEr | `fused_dual_residual_rmsnorm_ke` | ✓ | ✗ | `fused_dual_residual_rmsnorm` | 6 | `eps, residual_a, residual_b, weight, x` | `eps, hidden_states, residual, residual_a, residual` | `—` | `hidden_states, residual, softcap_const, ` | 6/0/0 | `typing` |
| `L2_19_fused_sigmoid_gating_delta_rule_` | ShapesIncompleteEr | `fused_sigmoid_gating_delta_rul` | ✓ | ✗ | `fused_sigmoid_gating_delta_r` | 1 | `A_log, a, beta, dt_bias, k, q, softplus_beta, softplus_thres` | `A_log, a, beta, dt_bias, k, q, softplus_beta, stat` | `softplus_threshold` | `—` | 1/0/0 | `typing` |
| `L2_20_causal_conv1d_update` | InterpreterError | `_causal_conv1d_update_kernel` | ✓ | ✗ | `causal_conv1d_update` | 2 | `activation, bias, conv_state, weight, x` | `activation, bias, conv_state, weight, x` | `—` | `—` | 2/0/0 | `numpy` |
| `L2_21_causal_conv1d_fwd` | ShapesIncompleteEr | `_causal_conv1d_fwd_kernel` | ✓ | ✗ | `causal_conv1d_fn` | 2 | `activation, bias, weight, x` | `activation, bias, weight, x` | `—` | `—` | 2/0/0 | `numpy` |
| `L2_22_linear_attn_decode` | ShapesIncompleteEr | `_linear_attn_decode_kernel` | ✓ | ✗ | `linear_decode_forward_triton` | 1 | `k, kv_caches, q, slope_rate, v` | `k, kv_caches, q, slope_rate, v` | `—` | `—` | 5/0/0 | `einops` |
| `L2_23_fwd_kv_parallel` | ShapesIncompleteEr | `_fwd_kv_parallel` | ✓ | ✗ | `linear_decode_forward_triton` | 1 | `k, q, slope, v` | `k, q, slope, v` | `—` | `—` | 5/0/0 | `einops` |
| `L2_24_fwd_kv_reduce` | ShapesIncompleteEr | `_fwd_kv_reduce` | ✓ | ✗ | `linear_decode_forward_triton` | 1 | `k, q, slope, v` | `k, q, slope, v` | `—` | `—` | 5/0/0 | `einops` |
| `L2_29_fused_moe_router_cudacore` | ShapesIncompleteEr | `fused_moe_router_cudacore_kern` | ✓ | ✗ | `fused_moe_router_cudacore` | 2 | `correction_bias, moe_softcapping, router_weight, topk, x` | `BLOCK_SIZE_M, BLOCK_SIZE_N, correction_bias, moe_s` | `—` | `BLOCK_SIZE_M, BLOCK_SIZE_N` | 2/0/0 | `typing` |
| `L2_30_fused_moe_router_tensorcore` | ShapesIncompleteEr | `fused_moe_router_tensorcore_ke` | ✓ | ✗ | `fused_moe_router_tensorcore` | 2 | `correction_bias, moe_softcapping, router_weight, topk, x` | `BLOCK_SIZE_K, BLOCK_SIZE_M, BLOCK_SIZE_N, correcti` | `—` | `BLOCK_SIZE_K, BLOCK_SIZE_M, BLOCK_SIZE_N` | 2/0/0 | `typing` |
| `L2_31_moe_align_block_size_stage2` | ShapesIncompleteEr | `moe_align_block_size_stage2` | ✓ | ✗ | `moe_align_block_size_triton` | 1 | `block_size, num_experts, topk_ids` | `block_size, num_experts, topk_ids` | `—` | `—` | 4/0/0 | `itertools, sys, pytest` |
| `L2_32_experts_combine` | RuntimeError | `experts_combine_kernel` | ✓ | ✗ | `experts_combine_triton` | 6 | `mlp_hidden_states, moe_hidden_states` | `hidden_states, mlp_hidden_states, moe_hidden_state` | `—` | `hidden_states, residual, softcap_const, ` | 6/0/0 | `typing` |
| `L2_35_pre_reorder_triton_kernel_for_cu` | ShapesIncompleteEr | `pre_reorder_triton_kernel_for_` | ✓ | ✗ | `pre_reorder_for_cutlass_moe` | 6 | `input, src2dst, topk` | `down_output, input, masked_m, output, src2dst, top` | `—` | `down_output, masked_m, output, x` | 21/0/0 | `logging` |
| `L2_36_post_reorder_triton_kernel_for_c` | ShapesIncompleteEr | `post_reorder_triton_kernel_for` | ✓ | ✗ | `post_reorder_for_cutlass_moe` | 6 | `down_output, src2dst, topk, topk_weights` | `down_output, input, input_tensor, masked_m, output` | `—` | `input, input_tensor, masked_m, output, x` | 21/0/0 | `logging` |
| `L2_38_concat_and_cast_mha_k` | AssertionError | `concat_and_cast_mha_k_kernel` | ✓ | ✗ | `concat_and_cast_mha_k_triton` | 5 | `k_nope, k_rope, out_dtype` | `extend_seq_lens, input_emb, k, k_nope, k_rope, key` | `—` | `extend_seq_lens, input_emb, k, key, q, s` | 11/0/0 | `—` |
| `L3_02_kernel_paged_attention_2d` | ShapesIncompleteEr | `kernel_paged_attention_2d` | ✓ | ✗ | `chunked_prefill_paged_decode` | 1 | `alibi_slopes, block_table, key_cache, query, scale, seq_lens` | `block_table, key, key_cache, query, scale, seq_len` | `alibi_slopes, sliding_window` | `key` | 2/0/0 | `prefix_prefill` |
| `L3_03_chunk_state_varlen` | ShapesIncompleteEr | `_chunk_state_varlen_kernel` | ✓ | ✗ | `chunk_state_varlen` | 3 | `B, cu_seqlens, dA_cumsum, dt, initial_states, x` | `B, cu_seqlens, dA_cumsum, dt, x` | `initial_states` | `—` | 3/0/0 | `math, mamba_ssm` |
| `L3_05_fwd_kernel_alibi` | ShapesIncompleteEr | `_fwd_kernel_alibi` | ✓ | ✗ | `context_attention_fwd` | 1 | `alibi_slopes, k, q, scale, v` | `alibi_slopes, k, o, q, scale, v` | `—` | `o` | 2/0/0 | `typing` |
| `L3_06_fwd_grouped_kernel_stage1` | ShapesIncompleteEr | `_fwd_grouped_kernel_stage1` | ✓ | ✗ | `_decode_grouped_att_m_fwd` | 3 | `cos, k_buffer, q, rope_dim, scale, sin, v_buffer` | `att_out, k_buffer, q, scale, v_buffer` | `cos, rope_dim, sin` | `att_out` | 4/0/0 | `logging, packaging` |
| `L3_07_correct_attn_cp_out` | ShapesIncompleteEr | `_correct_attn_cp_out_kernel` | ✓ | ✗ | `pack_seq_triton` | 2 | `cp_rank, is_lse_base_on_e, lses, out` | `cp_rank, is_lse_base_on_e, lses, out, x` | `—` | `x` | 3/0/0 | `—` |
| `L3_11_expert_triton` | ShapesIncompleteEr | `expert_triton_kernel` | ✓ | ✗ | `invoke_moe_batched_triton_ke` | 1 | `a, b, expert_num_tokens` | `a, b, expert_num_tokens` | `—` | `—` | 3/0/0 | `—` |
| `L3_12_moe_mmk` | ShapesIncompleteEr | `moe_mmk` | ✓ | ✗ | `invoke_moe_batched_triton_ke` | 1 | `a, b, expert_num_tokens` | `a, b, expert_num_tokens` | `—` | `—` | 3/0/0 | `—` |
| `L3_14_fused_recurrent_gated_delta_rule` | ShapesIncompleteEr | `fused_recurrent_gated_delta_ru` | ✓ | ✗ | `fused_recurrent_gated_delta_` | 2 | `beta, g, initial_state, k, q, scale, v` | `beta, g, k, q, scale, v` | `initial_state` | `—` | 2/1/0 | `op` |
| `L3_15_fused_recurrent_gated_delta_rule` | ShapesIncompleteEr | `fused_recurrent_gated_delta_ru` | ✓ | ✗ | `fused_recurrent_gated_delta_` | 3 | `beta, g, initial_state, k, q, scale, v` | `beta, g, k, q, scale, v` | `initial_state` | `—` | 3/0/0 | `typing` |

### upstream-not-invocable-here (40 tasks)

| task_id | err | meta.name | nKn | nTr | tried[0] | nTried | oracle in | shapes in | o−s | s−o | jit/heur/at | imports |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `L1_02_compute_slot_mappings` | InterpreterError | `_compute_slot_mappings_kernel` | ✓ | ✓ | `_compute_slot_mappings_kerne` | 1 | `block_size, block_table, idx_mapping, positions` | `block_size, block_table, idx_mapping, max_num_toke` | `—` | `max_num_tokens` | 3/0/0 | `collections.abc` |
| `L1_03_convert_req_index_to_global_inde` | KeyError | `_convert_req_index_to_global_i` | ✓ | ✗ | `triton_convert_req_index_to_` | 1 | `block_size, block_table, req_id, token_indices` | `BLOCK_N, BLOCK_SIZE, NUM_TOPK_TOKENS, block_table,` | `block_size` | `BLOCK_N, BLOCK_SIZE, NUM_TOPK_TOKENS` | 1/0/0 | `—` |
| `L1_04_copy_page_indices` | ModuleNotFoundErro | `_copy_page_indices_kernel` | ✓ | ✗ | `trtllm_prefill_attn_kvfp8_de` | 1 | `block_table, page_size, seq_lens` | `block_table, page_size, seq_lens` | `—` | `—` | 2/0/0 | `dataclasses, functools, typing` |
| `L1_05_gather_block_tables` | InterpreterError | `_gather_block_tables_kernel` | ✓ | ✓ | `_gather_block_tables_kernel` | 1 | `block_table, idx_mapping` | `block_table, idx_mapping` | `—` | `—` | 3/0/0 | `collections.abc` |
| `L1_13_layer_norm_fwd` | NotImplementedErro | `layer_norm_fwd_kernel` | ✓ | ✗ | `layer_norm_fwd` | 1 | `bias, eps, weight, x` | `bias, eps, weight, x` | `—` | `—` | 1/1/0 | `einops, utils` |
| `L1_19_topk_log_softmax` | ModuleNotFoundErro | `_topk_log_softmax_kernel` | ✓ | ✗ | `compute_token_logprobs` | 2 | `logits, token_ids` | `logits, num_logprobs, token_ids` | `—` | `num_logprobs` | 3/0/0 | `numpy` |
| `L1_23_penalties_and_temperature` | ExprError | `_penalties_kernel` | ✓ | ✗ | `apply_penalties` | 2 | `expanded_idx_mapping, frequency_penalty, logits, output_toke` | `all_token_ids, expanded_idx_mapping, frequency_pen` | `—` | `all_token_ids, token_ids` | 2/0/0 | `numpy` |
| `L1_24_rejection_sample` | ModuleNotFoundErro | `rejection_random_sample_kernel` | ✓ | ✗ | `rejection_sample` | 3 | `bonus_token_ids, cu_num_draft_tokens, draft_probs, draft_tok` | `bonus_token_ids, cu_num_draft_tokens, draft_probs,` | `—` | `num_draft_tokens, x` | 4/0/0 | `__future__, collections.abc, d` |
| `L1_27_apply_grammar_bitmask` | InterpreterError | `_apply_grammar_bitmask_kernel` | ✓ | ✓ | `_apply_grammar_bitmask_kerne` | 1 | `bitmask, logits, logits_indices` | `bitmask, logits, logits_indices` | `—` | `—` | 1/0/0 | `numpy` |
| `L1_31_triton_mrope_forward` | RuntimeError | `_triton_mrope_forward` | ✓ | ✗ | `triton_mrope` | 1 | `cos, head_size, k, mrope_section, q, rotary_dim, sin` | `cos, head_size, k, mrope_interleaved, mrope_sectio` | `—` | `mrope_interleaved` | 1/0/0 | `numpy, base, yarn_scaling_rope` |
| `L1_36_compute_seg_indptr_triton` | ExprError | `compute_seg_indptr_triton_kern` | ✓ | ✗ | `moe_ep_deepgemm_preprocess` | 6 | `num_experts, topk_ids` | `down_output, hidden_states, input, masked_m, num_e` | `—` | `down_output, hidden_states, input, maske` | 21/0/0 | `logging` |
| `L1_37_compute_problem_sizes_w4a8` | ExprError | `compute_problem_sizes_w4a8_ker` | ✓ | ✗ | `compute_problem_sizes_w4a8` | 6 | `k, masked_m, n, num_experts` | `input, k, masked_m, n, num_experts, problem_sizes1` | `—` | `input, problem_sizes1, x` | 21/0/0 | `logging` |
| `L1_38_silu_mul_fp8_quant_deep_gemm` | AssertionError | `_silu_mul_fp8_quant_deep_gemm` | ✓ | ✗ | `persistent_masked_m_silu_mul` | 1 | `fp8_max, group_size, tokens_per_expert, y` | `fp8_max, group_size, tokens_per_expert, y` | `—` | `—` | 1/0/0 | `—` |
| `L2_04_chunked_lora_expand` | AttributeError | `_chunked_lora_expand_kernel` | ✓ | ✗ | `chunked_sgmv_lora_expand_for` | 1 | `base_output, lora_ranks, scalings, seg_indptr, weight_indice` | `batch_info, lora_ranks, max_slice_size, scalings, ` | `base_output` | `batch_info, max_slice_size, slice_offset` | 1/0/0 | `typing` |
| `L2_05_chunked_lora_shrink` | AttributeError | `_chunked_lora_shrink_kernel` | ✓ | ✗ | `chunked_sgmv_lora_shrink_for` | 1 | `lora_ranks, seg_indptr, weight_indices, weights, x` | `batch_info, lora_ranks, num_slices, scalings, seg_` | `—` | `batch_info, num_slices, scalings` | 1/0/0 | `—` |
| `L2_06_qkv_lora_b` | AssertionError | `_qkv_lora_b_kernel` | ✓ | ✗ | `qkv_lora_b_fwd` | 1 | `base_output, lora_ranks, scalings, seg_indptr, weight_indice` | `batch_info, lora_ranks, max_qkv_out_dim, output_of` | `base_output, weights` | `batch_info, max_qkv_out_dim, output_offs` | 1/0/0 | `—` |
| `L2_07_gate_up_lora_b` | AssertionError | `_gate_up_lora_b_kernel` | ✓ | ✗ | `gate_up_lora_b_fwd` | 1 | `base_output, lora_ranks, scalings, seg_indptr, weight_indice` | `batch_info, gate_up_lora_b, lora_ranks, output_dim` | `base_output, weights` | `batch_info, gate_up_lora_b, output_dim` | 1/0/0 | `—` |
| `L2_09_sgemm_lora_b` | upstream-not-invoc | `_sgemm_lora_b_kernel` | ✓ | ✗ | `sgemm_lora_b_fwd` | 1 | `base_output, lora_ranks, scalings, seg_indptr, weight_indice` | `batch_info, lora_ranks, scalings, seg_indptr, weig` | `base_output` | `batch_info` | 1/0/0 | `—` |
| `L2_17_layer_norm_gated_fwd` | AttributeError | `layer_norm_gated_fwd_kernel` | ✓ | ✗ | `layer_norm_gated_fwd` | 5 | `b, eps, g, residual, w, x` | `b, eps, g, k, q, residual, w, x` | `—` | `k, q` | 7/6/5 | `chunk_delta_h, cumsum, fused_r` |
| `L2_18_fused_gdn_gating` | AttributeError | `fused_gdn_gating_kernel` | ✓ | ✗ | `fused_gdn_gating` | 1 | `A_log, a, beta, dt_bias, threshold` | `A_log, a, beta, dt_bias, threshold` | `—` | `—` | 1/0/0 | `einops, transformers.activatio` |
| `L2_25_fwd_kernel_ep_gather` | ExprError | `_fwd_kernel_ep_gather` | ✓ | ✗ | `ep_gather` | 2 | `input_index, input_tensor` | `input_index, input_tensor, recv_x, recv_x_scale` | `—` | `recv_x, recv_x_scale` | 5/0/0 | `—` |
| `L2_26_fwd_kernel_ep_scatter_1` | ExprError | `_fwd_kernel_ep_scatter_1` | ✓ | ✗ | `ep_scatter` | 2 | `dst_index, out_rows, recv_x` | `dst_index, input_tensor, out_rows, recv_x, recv_x_` | `—` | `input_tensor, recv_x_scale` | 5/0/0 | `—` |
| `L2_27_deepep_permute_triton` | ExprError | `deepep_permute_triton_kernel` | ✓ | ✗ | `fp8_per_token_to_per_tensor_` | 6 | `src2dst, topk, x` | `down_output, input, masked_m, output, problem_size` | `—` | `down_output, input, masked_m, output, pr` | 21/0/0 | `logging` |
| `L2_28_deepep_post_reorder_triton` | ExprError | `deepep_post_reorder_triton_ker` | ✓ | ✗ | `post_reorder_for_cutlass_moe` | 6 | `src2dst, topk, topk_weights, x` | `down_output, input, masked_m, output, problem_size` | `—` | `down_output, input, masked_m, output, pr` | 21/0/0 | `logging` |
| `L2_33_fwd_grouped_kernel_stage1_rope` | RuntimeError | `_fwd_grouped_kernel_stage1_rop` | ✓ | ✗ | `_decode_grouped_att_m_fwd_ro` | 1 | `cos, k_buffer, q, rope_dim, scale, sin, v_buffer` | `att_out, k_buffer, q, scale, v_buffer` | `cos, rope_dim, sin` | `att_out` | 2/0/0 | `—` |
| `L2_34_silu_mul_static_tensorwise_quant` | ExprError | `silu_mul_static_tensorwise_qua` | ✓ | ✗ | `silu_and_mul_masked_post_per` | 6 | `input, scale` | `down_output, input, masked_m, output, problem_size` | `—` | `down_output, masked_m, output, problem_s` | 21/0/0 | `logging` |
| `L2_37_silu_and_mul_post_quant` | ExprError | `_silu_and_mul_post_quant_kerne` | ✓ | ✗ | `silu_and_mul_masked_post_qua` | 6 | `fp8_max, input, masked_m, quant_group_size` | `fp8_max, input, masked_m, output, problem_sizes1, ` | `—` | `output, problem_sizes1, x` | 21/0/0 | `logging` |
| `L3_01_kernel_unified_attention_3d` | ZeroDivisionError | `kernel_unified_attention` | ✓ | ✗ | `unified_attention` | 1 | `alibi_slopes, causal, cu_seqlens_q, k, q, scale, seqused_k, ` | `block_table, causal, cu_seqlens_q, k, max_seqlen_k` | `alibi_slopes, scale` | `block_table, max_seqlen_k, max_seqlen_q,` | 3/0/0 | `typing` |
| `L3_04_fwd` | — | `_fwd_kernel` | ✓ | ✗ | `—` | 0 | `b_seq_len, b_start_loc, is_causal, k, q, sm_scale, v` | `b_seq_len, b_start_loc, is_causal, k, max_input_le` | `—` | `max_input_len, o` | 1/0/0 | `—` |
| `L3_09_fused_moe` | RuntimeError | `fused_moe_kernel` | ✓ | ✗ | `invoke_fused_moe_triton_kern` | 3 | `a, apply_weight, b, block_size, expert_ids, sorted_token_ids` | `a, apply_weight, b, block_size, expert_ids, sorted` | `—` | `—` | 4/0/0 | `functools, json, os, collectio` |
| `L3_10_fused_moe_kernel_gptq_awq` | RuntimeError | `fused_moe_kernel_gptq_awq` | ✓ | ✗ | `invoke_fused_moe_wna16_trito` | 3 | `a, apply_weight, b, block_size, expert_ids, sorted_token_ids` | `a, apply_weight, b, block_size, expert_ids, sorted` | `—` | `—` | 4/0/0 | `functools, json, os, collectio` |
| `L3_13_chunk_gated_delta_rule_fwd_kerne` | AttributeError | `chunk_gated_delta_rule_fwd_ker` | ✓ | ✗ | `chunk_gated_delta_rule_fwd_h` | 1 | `g, initial_state, k, u, w` | `g, k, u, w` | `initial_state` | `—` | 1/1/1 | `index, op, utils` |
| `L3_16_chunk_fwd_kernel_o` | AttributeError | `chunk_fwd_kernel_o` | ✓ | ✗ | `chunk_fwd_o` | 1 | `g, h, k, q, scale, v` | `g, h, k, q, scale, v` | `—` | `—` | 1/1/1 | `index, op, utils` |
| `L3_17_chunk_gla_fwd_kernel_o` | AttributeError | `chunk_gla_fwd_kernel_o` | ✓ | ✗ | `chunk_gla_fwd_o_gk` | 5 | `A, g, h, q, scale, v` | `A, g, h, o, q, scale, v` | `—` | `o` | 7/6/5 | `chunk_delta_h, cumsum, fused_r` |
| `L3_18_chunk_scaled_dot_kkt_fwd` | AttributeError | `chunk_scaled_dot_kkt_fwd_kerne` | ✓ | ✗ | `chunk_scaled_dot_kkt_fwd` | 1 | `beta, g, k` | `beta, g, k` | `—` | `—` | 1/1/1 | `index, op, utils` |
| `L3_19_chunk_kda_scaled_dot_kkt_fwd_ker` | AttributeError | `chunk_kda_scaled_dot_kkt_fwd_k` | ✓ | ✗ | `chunk_kda_scaled_dot_kkt_fwd` | 5 | `beta, gk, k, q, scale` | `beta, gk, k, q, scale` | `—` | `—` | 7/6/5 | `chunk_delta_h, cumsum, fused_r` |
| `L3_20_recompute_w_u_fwd` | AttributeError | `recompute_w_u_fwd_kernel` | ✓ | ✗ | `recompute_w_u_fwd` | 1 | `A, beta, g_cumsum, k, v` | `A, beta, cu_seqlens, g_cumsum, k, v` | `—` | `cu_seqlens` | 1/1/1 | `index` |
| `L3_21_solve_tril_16x16` | AttributeError | `solve_tril_16x16_kernel` | ✓ | ✓ | `solve_tril_16x16_kernel` | 1 | `A` | `A` | `—` | `—` | 3/3/3 | `os, index, op, utils` |
| `L3_22_chunk_local_cumsum_vector` | AttributeError | `chunk_local_cumsum_vector_kern` | ✓ | ✗ | `chunk_local_cumsum_vector` | 2 | `g, reverse` | `chunk_size, g, reverse` | `—` | `chunk_size` | 2/2/2 | `index, utils` |
| `L3_23_merge_16x16_to_32x32_inverse` | AttributeError | `merge_16x16_to_32x32_inverse_k` | ✓ | ✓ | `merge_16x16_to_32x32_inverse` | 1 | `A` | `A` | `—` | `—` | 3/3/3 | `os, index, op, utils` |

### oracle-mismatch (6 tasks)

| task_id | err | meta.name | nKn | nTr | tried[0] | nTried | oracle in | shapes in | o−s | s−o | jit/heur/at | imports |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `L1_09_quantize_k_cache_fast` | AssertionError | `_quantize_k_cache_fast_kernel` | ✓ | ✗ | `_quantize_k_cache_fast` | 2 | `fp8_max, group_size, k_nope, k_rope` | `fp8_max, group_size, k_nope, k_rope` | `—` | `—` | 1/0/0 | `—` |
| `L1_14_layer_norm_fwd_1pass` | oracle-mismatch | `_layer_norm_fwd_1pass_kernel` | ✓ | ✗ | `_layer_norm_fwd` | 1 | `bias, eps, residual, weight, x` | `bias, eps, residual, weight, x` | `—` | `—` | 1/2/0 | `—` |
| `L1_22_min_p` | oracle-mismatch | `_min_p_kernel` | ✓ | ✗ | `apply_min_p` | 1 | `expanded_idx_mapping, logits, min_p` | `expanded_idx_mapping, logits, min_p` | `—` | `—` | 1/0/0 | `—` |
| `L1_25_merge_attn_states` | oracle-mismatch | `merge_attn_states_kernel` | ✓ | ✗ | `merge_attn_states` | 1 | `prefix_lse, prefix_output, suffix_lse, suffix_output` | `output, prefix_lse, prefix_output, suffix_lse, suf` | `—` | `output` | 1/0/0 | `—` |
| `L2_08_sgemm_lora_a` | oracle-mismatch | `_sgemm_lora_a_kernel` | ✓ | ✗ | `sgemm_lora_a_fwd` | 1 | `lora_ranks, seg_indptr, weight_indices, weights, x` | `batch_info, lora_ranks, scalings, seg_indptr, weig` | `—` | `batch_info, scalings` | 1/0/0 | `—` |
| `L2_16_fused_rmsnorm` | ValueError | `fused_rmsnorm_kernel` | ✓ | ✗ | `fused_rmsnorm` | 6 | `eps, residual, weight, x` | `eps, hidden_states, residual, softcap_const, weigh` | `—` | `hidden_states, softcap_const, weight1, w` | 6/0/0 | `typing` |

### smoke-fail (2 tasks)

| task_id | err | meta.name | nKn | nTr | tried[0] | nTried | oracle in | shapes in | o−s | s−o | jit/heur/at | imports |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `L2_01_bmm_chunk_fwd` | RuntimeError | `_bmm_chunk_fwd_kernel` | ✓ | ✗ | `_bmm_chunk_fwd` | 1 | `a, b, causal, chunk_size, cu_chunk_seqlens` | `a, b, causal, chunk_size, cu_chunk_seqlens` | `—` | `—` | 1/0/1 | `—` |
| `L2_10_w8a8_block_int8_matmul` | — | `_w8a8_block_int8_matmul` | ✓ | ✗ | `—` | 0 | `A, As, B, Bs, block_size, output_dtype` | `A, As, B, Bs, block_size` | `output_dtype` | `—` | 1/0/0 | `—` |

## 3. Selected Per-Task Deep Detail (5 PASS + 10 sample BLOCKED)

### `L1_01_bias` (PASS)

- **meta.name**: `_bias_kernel` (in kernel: True, in tried: False)
- **meta.oracle.gpu_triton**: `kernel.py::apply_logit_bias`
- **resolver verdict**: PASS, tried 0 candidates: ``
- **resolver error_type**: `oracle-consistent`
- **resolver msg head**: `meta already points at kernel.py::apply_logit_bias (untouched) AND re-verified under the CURRENT hardened anchor: oracle-consistent [anchored on 2/2 shape(s) ['H2', 'H4'] :: shape=`
- **kernel.py module defs (2)**: `_bias_kernel, apply_logit_bias`
- **kernel.py triton_jit/heuristics/autotune**: 1/0/0
- **kernel imports (non-stdlib non-torch non-triton non-vllm/sglang)**: ``
- **oracle_pytorch.run() inputs**: `allowed_token_ids, expanded_idx_mapping, logit_bias, logit_bias_token_ids, logits, min_lens, num_allowed_token_ids, num_logit_bias, num_stop_token_ids, pos, stop_token_ids`
- **oracle_pytorch.run() outputs**: `allowed_token_ids, expanded_idx_mapping, logit_bias, logit_bias_token_ids, logits, min_lens, num_allowed_token_ids, num_logit_bias, num_stop_token_ids, pos, stop_token_ids`
- **shapes.yaml inputs**: `allowed_token_ids, expanded_idx_mapping, logit_bias, logit_bias_token_ids, logits, min_lens, num_allowed_token_ids, num_logit_bias, num_stop_token_ids, pos, stop_token_ids`
- **oracle − shapes (in oracle, not in shapes)**: ``
- **shapes − oracle (in shapes, not in oracle)**: ``

### `L1_12_rms_norm` (PASS)

- **meta.name**: `_rms_norm_kernel` (in kernel: True, in tried: False)
- **meta.oracle.gpu_triton**: `kernel.py::rms_norm`
- **resolver verdict**: PASS, tried 0 candidates: ``
- **resolver error_type**: `oracle-consistent`
- **resolver msg head**: `meta already points at kernel.py::rms_norm (untouched) AND re-verified under the CURRENT hardened anchor: oracle-consistent [anchored on 2/2 shape(s) ['P1', 'H2'] :: shape=P1 tol=v`
- **kernel.py module defs (2)**: `_rms_norm_kernel, rms_norm`
- **kernel.py triton_jit/heuristics/autotune**: 1/0/0
- **kernel imports (non-stdlib non-torch non-triton non-vllm/sglang)**: ``
- **oracle_pytorch.run() inputs**: `eps, input, weight`
- **oracle_pytorch.run() outputs**: `output`
- **shapes.yaml inputs**: `eps, input, weight`
- **oracle − shapes (in oracle, not in shapes)**: ``
- **shapes − oracle (in shapes, not in oracle)**: ``

### `L1_16_silu_and_mul` (PASS)

- **meta.name**: `silu_and_mul_kernel` (in kernel: True, in tried: False)
- **meta.oracle.gpu_triton**: `kernel.py::silu_and_mul_triton`
- **resolver verdict**: PASS, tried 0 candidates: ``
- **resolver error_type**: `oracle-consistent`
- **resolver msg head**: `meta already points at kernel.py::silu_and_mul_triton (untouched) AND re-verified under the CURRENT hardened anchor: oracle-consistent [anchored on 2/2 shape(s) ['H2', 'P5'] :: sha`
- **kernel.py module defs (2)**: `silu_and_mul_kernel, silu_and_mul_triton`
- **kernel.py triton_jit/heuristics/autotune**: 1/0/0
- **kernel imports (non-stdlib non-torch non-triton non-vllm/sglang)**: ``
- **oracle_pytorch.run() inputs**: `hidden_states`
- **oracle_pytorch.run() outputs**: `output`
- **shapes.yaml inputs**: `hidden_states`
- **oracle − shapes (in oracle, not in shapes)**: ``
- **shapes − oracle (in shapes, not in oracle)**: ``

### `L1_26_merge_state` (PASS)

- **meta.name**: `merge_state_kernel` (in kernel: True, in tried: False)
- **meta.oracle.gpu_triton**: `kernel.py::merge_state_triton`
- **resolver verdict**: PASS, tried 0 candidates: ``
- **resolver error_type**: `oracle-consistent`
- **resolver msg head**: `meta already points at kernel.py::merge_state_triton (untouched) AND re-verified under the CURRENT hardened anchor: oracle-consistent [anchored on 2/2 shape(s) ['H2', 'P5'] :: shap`
- **kernel.py module defs (2)**: `merge_state_kernel, merge_state_triton`
- **kernel.py triton_jit/heuristics/autotune**: 1/0/0
- **kernel imports (non-stdlib non-torch non-triton non-vllm/sglang)**: `typing`
- **oracle_pytorch.run() inputs**: `prefix_lse, prefix_output, suffix_lse, suffix_output`
- **oracle_pytorch.run() outputs**: `output, output_lse`
- **shapes.yaml inputs**: `prefix_lse, prefix_output, suffix_lse, suffix_output`
- **oracle − shapes (in oracle, not in shapes)**: ``
- **shapes − oracle (in shapes, not in oracle)**: ``

### `L1_34_transform_index_page_table_decode` (PASS)

- **meta.name**: `transform_index_page_table_decode_kernel` (in kernel: True, in tried: False)
- **meta.oracle.gpu_triton**: `kernel.py::placeholder`
- **resolver verdict**: PASS, tried 1 candidates: `transform_index_page_table_decode_fast`
- **resolver error_type**: ``
- **resolver msg head**: `resolved kernel.py::transform_index_page_table_decode_fast -- oracle-consistent [anchored on 2/2 shape(s) ['H2', 'H1'] :: shape=H2 tol=attention_flash :: result: ✓ within attention`
- **kernel.py module defs (7)**: `transform_index_page_table_prefill, transform_index_page_table_decode, transform_index_page_table_decode_kernel, transform_index_page_table_decode_fast, transform_index_page_table_prefill_fast`
- **kernel.py triton_jit/heuristics/autotune**: 1/0/0
- **kernel imports (non-stdlib non-torch non-triton non-vllm/sglang)**: `typing`
- **oracle_pytorch.run() inputs**: `page_size, page_table, topk_indices`
- **oracle_pytorch.run() outputs**: `result`
- **shapes.yaml inputs**: `page_size, page_table, topk_indices`
- **oracle − shapes (in oracle, not in shapes)**: ``
- **shapes − oracle (in shapes, not in oracle)**: ``

### `L2_16_fused_rmsnorm` (oracle-mismatch)

- **meta.name**: `fused_rmsnorm_kernel` (in kernel: True, in tried: False)
- **meta.oracle.gpu_triton**: `kernel.py::placeholder`
- **resolver verdict**: oracle-mismatch, tried 6 candidates: `fused_rmsnorm, fused_dual_residual_rmsnorm, fused_softcap, gelu_and_mul_triton, silu_and_mul_triton`
- **resolver error_type**: `ValueError`
- **resolver msg head**: `BLOCKED(no-verified-upstream-entrypoint: oracle-mismatch): tried ['fused_rmsnorm', 'fused_dual_residual_rmsnorm', 'fused_softcap', 'gelu_and_mul_triton', 'silu_and_mul_triton', 'ex`
- **kernel.py module defs (12)**: `fused_softcap_kernel, fused_softcap, fused_dual_residual_rmsnorm_kernel, fused_dual_residual_rmsnorm, fused_rmsnorm_kernel`
- **kernel.py class methods**: `Softcap.__init__, Softcap.__call__, Softcap.forward, Softcap.forward_native, Softcap.forward_cuda`
- **kernel.py triton_jit/heuristics/autotune**: 6/0/0
- **kernel imports (non-stdlib non-torch non-triton non-vllm/sglang)**: `typing`
- **oracle_pytorch.run() inputs**: `eps, residual, weight, x`
- **oracle_pytorch.run() outputs**: `hidden, output`
- **shapes.yaml inputs**: `eps, hidden_states, residual, softcap_const, weight, weight1, weight2, x`
- **oracle − shapes (in oracle, not in shapes)**: ``
- **shapes − oracle (in shapes, not in oracle)**: `hidden_states, softcap_const, weight1, weight2`

### `L2_15_fused_dual_residual_rmsnorm` (shapes-incomplete)

- **meta.name**: `fused_dual_residual_rmsnorm_kernel` (in kernel: True, in tried: False)
- **meta.oracle.gpu_triton**: `kernel.py::placeholder`
- **resolver verdict**: shapes-incomplete, tried 6 candidates: `fused_dual_residual_rmsnorm, fused_rmsnorm, fused_softcap, gelu_and_mul_triton, silu_and_mul_triton`
- **resolver error_type**: `ShapesIncompleteError`
- **resolver msg head**: `BLOCKED(no-verified-upstream-entrypoint: shapes-incomplete): tried ['fused_dual_residual_rmsnorm', 'fused_rmsnorm', 'fused_softcap', 'gelu_and_mul_triton', 'silu_and_mul_triton', '`
- **kernel.py module defs (12)**: `fused_softcap_kernel, fused_softcap, fused_dual_residual_rmsnorm_kernel, fused_dual_residual_rmsnorm, fused_rmsnorm_kernel`
- **kernel.py class methods**: `Softcap.__init__, Softcap.__call__, Softcap.forward, Softcap.forward_native, Softcap.forward_cuda`
- **kernel.py triton_jit/heuristics/autotune**: 6/0/0
- **kernel imports (non-stdlib non-torch non-triton non-vllm/sglang)**: `typing`
- **oracle_pytorch.run() inputs**: `eps, residual_a, residual_b, weight, x`
- **oracle_pytorch.run() outputs**: `hidden, output`
- **shapes.yaml inputs**: `eps, hidden_states, residual, residual_a, residual_b, softcap_const, weight, weight1, weight2, x`
- **oracle − shapes (in oracle, not in shapes)**: ``
- **shapes − oracle (in shapes, not in oracle)**: `hidden_states, residual, softcap_const, weight1, weight2`

### `L1_04_copy_page_indices` (upstream-not-invocable-here)

- **meta.name**: `_copy_page_indices_kernel` (in kernel: True, in tried: False)
- **meta.oracle.gpu_triton**: `kernel.py::placeholder`
- **resolver verdict**: upstream-not-invocable-here, tried 1 candidates: `trtllm_prefill_attn_kvfp8_dequant`
- **resolver error_type**: `ModuleNotFoundError`
- **resolver msg head**: `BLOCKED(no-verified-upstream-entrypoint: upstream-not-invocable-here): tried ['trtllm_prefill_attn_kvfp8_dequant']; trtllm_prefill_attn_kvfp8_dequant: upstream-not-invocable-here [`
- **kernel.py module defs (5)**: `_get_trtllm_gen_workspace_buffer, _trtllm_prefill_attn_kvfp8_dequant, trtllm_prefill_attn_kvfp8_dequant, fast_plan_decode, _copy_page_indices_kernel`
- **kernel.py class methods**: `BatchDCPPrefillWrapper.__init__, BatchDCPPrefillWrapper.plan, BatchDCPPrefillWrapper.run, FlashInferBackend.get_supported_kernel_block_sizes, FlashInferBackend.get_name`
- **kernel.py triton_jit/heuristics/autotune**: 2/0/0
- **kernel imports (non-stdlib non-torch non-triton non-vllm/sglang)**: `dataclasses, functools, typing, numpy, flashinfer`
- **oracle_pytorch.run() inputs**: `block_table, page_size, seq_lens`
- **oracle_pytorch.run() outputs**: `kv_indices`
- **shapes.yaml inputs**: `block_table, page_size, seq_lens`
- **oracle − shapes (in oracle, not in shapes)**: ``
- **shapes − oracle (in shapes, not in oracle)**: ``

### `L3_01_kernel_unified_attention_3d` (upstream-not-invocable-here)

- **meta.name**: `kernel_unified_attention` (in kernel: True, in tried: False)
- **meta.oracle.gpu_triton**: `kernel.py::placeholder`
- **resolver verdict**: upstream-not-invocable-here, tried 1 candidates: `unified_attention`
- **resolver error_type**: `ZeroDivisionError`
- **resolver msg head**: `BLOCKED(no-verified-upstream-entrypoint: upstream-not-invocable-here): tried ['unified_attention']; unified_attention: upstream-not-invocable-here [ZeroDivisionError: integer divis`
- **kernel.py module defs (6)**: `_cast_kv_tile, kernel_unified_attention, reduce_segments, _is_gemma3_attention, _get_tile_size`
- **kernel.py triton_jit/heuristics/autotune**: 3/0/0
- **kernel imports (non-stdlib non-torch non-triton non-vllm/sglang)**: `typing`
- **oracle_pytorch.run() inputs**: `alibi_slopes, causal, cu_seqlens_q, k, q, scale, seqused_k, softcap, v`
- **oracle_pytorch.run() outputs**: `output`
- **shapes.yaml inputs**: `block_table, causal, cu_seqlens_q, k, max_seqlen_k, max_seqlen_q, out, q, seqused_k, softcap, softmax_scale, v, window_size`
- **oracle − shapes (in oracle, not in shapes)**: `alibi_slopes, scale`
- **shapes − oracle (in shapes, not in oracle)**: `block_table, max_seqlen_k, max_seqlen_q, out, softmax_scale, window_size`

### `L1_02_compute_slot_mappings` (upstream-not-invocable-here)

- **meta.name**: `_compute_slot_mappings_kernel` (in kernel: True, in tried: True)
- **meta.oracle.gpu_triton**: `kernel.py::placeholder`
- **resolver verdict**: upstream-not-invocable-here, tried 1 candidates: `_compute_slot_mappings_kernel`
- **resolver error_type**: `InterpreterError`
- **resolver msg head**: `BLOCKED(no-verified-upstream-entrypoint: upstream-not-invocable-here): tried ['_compute_slot_mappings_kernel']; _compute_slot_mappings_kernel: upstream-not-invocable-here [Interpre`
- **kernel.py module defs (3)**: `_gather_block_tables_kernel, _compute_slot_mappings_kernel, _load_ptr`
- **kernel.py class methods**: `BlockTables.__init__, BlockTables._make_ptr_tensor, BlockTables.init_block_table_layout_tensors, BlockTables.append_block_ids, BlockTables.apply_staged_writes`
- **kernel.py triton_jit/heuristics/autotune**: 3/0/0
- **kernel imports (non-stdlib non-torch non-triton non-vllm/sglang)**: `collections.abc`
- **oracle_pytorch.run() inputs**: `block_size, block_table, idx_mapping, positions`
- **oracle_pytorch.run() outputs**: `slot_mapping`
- **shapes.yaml inputs**: `block_size, block_table, idx_mapping, max_num_tokens, positions`
- **oracle − shapes (in oracle, not in shapes)**: ``
- **shapes − oracle (in shapes, not in oracle)**: `max_num_tokens`

### `L1_06_reshape_and_cache_kernel_flash` (shapes-incomplete)

- **meta.name**: `reshape_and_cache_kernel_flash` (in kernel: True, in tried: False)
- **meta.oracle.gpu_triton**: `kernel.py::placeholder`
- **resolver verdict**: shapes-incomplete, tried 3 candidates: `triton_reshape_and_cache_flash, triton_reshape_and_cache_flash_per_token_head_quant, triton_reshape_and_cache_flash_diffkv`
- **resolver error_type**: `ShapesIncompleteError`
- **resolver msg head**: `BLOCKED(no-verified-upstream-entrypoint: shapes-incomplete): tried ['triton_reshape_and_cache_flash', 'triton_reshape_and_cache_flash_per_token_head_quant', 'triton_reshape_and_cac`
- **kernel.py module defs (6)**: `reshape_and_cache_kernel_flash, _reshape_cache_per_token_head, triton_reshape_and_cache_flash_per_token_head_quant, triton_reshape_and_cache_flash, reshape_and_cache_kernel_flash_diffkv`
- **kernel.py triton_jit/heuristics/autotune**: 3/0/0
- **kernel imports (non-stdlib non-torch non-triton non-vllm/sglang)**: ``
- **oracle_pytorch.run() inputs**: `key, key_cache, slot_mapping, value, value_cache`
- **oracle_pytorch.run() outputs**: `key_cache, value_cache`
- **shapes.yaml inputs**: `key, key_cache, slot_mapping, value, value_cache`
- **oracle − shapes (in oracle, not in shapes)**: ``
- **shapes − oracle (in shapes, not in oracle)**: ``

### `L1_22_min_p` (oracle-mismatch)

- **meta.name**: `_min_p_kernel` (in kernel: True, in tried: False)
- **meta.oracle.gpu_triton**: `kernel.py::placeholder`
- **resolver verdict**: oracle-mismatch, tried 1 candidates: `apply_min_p`
- **resolver error_type**: `oracle-mismatch`
- **resolver msg head**: `BLOCKED(no-verified-upstream-entrypoint: oracle-mismatch): tried ['apply_min_p']; apply_min_p: oracle-mismatch [anchored on 1/2 shape(s) ['H2'] :: shape=H2 tol=sampling_rng :: logi`
- **kernel.py module defs (2)**: `_min_p_kernel, apply_min_p`
- **kernel.py triton_jit/heuristics/autotune**: 1/0/0
- **kernel imports (non-stdlib non-torch non-triton non-vllm/sglang)**: ``
- **oracle_pytorch.run() inputs**: `expanded_idx_mapping, logits, min_p`
- **oracle_pytorch.run() outputs**: `logits`
- **shapes.yaml inputs**: `expanded_idx_mapping, logits, min_p`
- **oracle − shapes (in oracle, not in shapes)**: ``
- **shapes − oracle (in shapes, not in oracle)**: ``

### `L2_01_bmm_chunk_fwd` (smoke-fail)

- **meta.name**: `_bmm_chunk_fwd_kernel` (in kernel: True, in tried: False)
- **meta.oracle.gpu_triton**: `kernel.py::placeholder`
- **resolver verdict**: smoke-fail, tried 1 candidates: `_bmm_chunk_fwd`
- **resolver error_type**: `RuntimeError`
- **resolver msg head**: `BLOCKED(no-verified-upstream-entrypoint: smoke-fail): tried ['_bmm_chunk_fwd']; _bmm_chunk_fwd: oracle-consistent but S2-smoke BLOCKED [S2-smoke MISMATCH (CPU/TRITON_INTERPRET pre-`
- **kernel.py module defs (2)**: `_bmm_chunk_fwd_kernel, _bmm_chunk_fwd`
- **kernel.py triton_jit/heuristics/autotune**: 1/0/1
- **kernel imports (non-stdlib non-torch non-triton non-vllm/sglang)**: ``
- **oracle_pytorch.run() inputs**: `a, b, causal, chunk_size, cu_chunk_seqlens`
- **oracle_pytorch.run() outputs**: `output`
- **shapes.yaml inputs**: `a, b, causal, chunk_size, cu_chunk_seqlens`
- **oracle − shapes (in oracle, not in shapes)**: ``
- **shapes − oracle (in shapes, not in oracle)**: ``

### `L3_14_fused_recurrent_gated_delta_rule_fwd` (shapes-incomplete)

- **meta.name**: `fused_recurrent_gated_delta_rule_fwd_kernel` (in kernel: True, in tried: False)
- **meta.oracle.gpu_triton**: `kernel.py::placeholder`
- **resolver verdict**: shapes-incomplete, tried 2 candidates: `fused_recurrent_gated_delta_rule_fwd, fused_recurrent_gated_delta_rule_packed_decode`
- **resolver error_type**: `ShapesIncompleteError`
- **resolver msg head**: `BLOCKED(no-verified-upstream-entrypoint: shapes-incomplete): tried ['fused_recurrent_gated_delta_rule_fwd', 'fused_recurrent_gated_delta_rule_packed_decode']; fused_recurrent_gated`
- **kernel.py module defs (5)**: `fused_recurrent_gated_delta_rule_fwd_kernel, fused_recurrent_gated_delta_rule_fwd, fused_recurrent_gated_delta_rule_packed_decode_kernel, fused_recurrent_gated_delta_rule_packed_decode, fused_recurrent_gated_delta_rule`
- **kernel.py class methods**: `FusedRecurrentFunction.forward`
- **kernel.py triton_jit/heuristics/autotune**: 2/1/0
- **kernel imports (non-stdlib non-torch non-triton non-vllm/sglang)**: `op`
- **oracle_pytorch.run() inputs**: `beta, g, initial_state, k, q, scale, v`
- **oracle_pytorch.run() outputs**: `final_state, output`
- **shapes.yaml inputs**: `beta, g, k, q, scale, v`
- **oracle − shapes (in oracle, not in shapes)**: `initial_state`
- **shapes − oracle (in shapes, not in oracle)**: ``

### `L1_31_triton_mrope_forward` (upstream-not-invocable-here)

- **meta.name**: `_triton_mrope_forward` (in kernel: True, in tried: False)
- **meta.oracle.gpu_triton**: `kernel.py::placeholder`
- **resolver verdict**: upstream-not-invocable-here, tried 1 candidates: `triton_mrope`
- **resolver error_type**: `RuntimeError`
- **resolver msg head**: `BLOCKED(no-verified-upstream-entrypoint: upstream-not-invocable-here): tried ['triton_mrope']; triton_mrope: upstream-not-invocable-here [RuntimeError: Tried to register an operato`
- **kernel.py module defs (3)**: `_triton_mrope_forward, triton_mrope, apply_interleaved_rope`
- **kernel.py class methods**: `MRotaryEmbedding.__init__, MRotaryEmbedding._compute_inv_freq, MRotaryEmbedding._compute_cos_sin_cache, MRotaryEmbedding.forward_native, MRotaryEmbedding.forward_cuda`
- **kernel.py triton_jit/heuristics/autotune**: 1/0/0
- **kernel imports (non-stdlib non-torch non-triton non-vllm/sglang)**: `numpy, base, yarn_scaling_rope`
- **oracle_pytorch.run() inputs**: `cos, head_size, k, mrope_section, q, rotary_dim, sin`
- **oracle_pytorch.run() outputs**: `k, q`
- **shapes.yaml inputs**: `cos, head_size, k, mrope_interleaved, mrope_section, q, rotary_dim, sin`
- **oracle − shapes (in oracle, not in shapes)**: ``
- **shapes − oracle (in shapes, not in oracle)**: `mrope_interleaved`
