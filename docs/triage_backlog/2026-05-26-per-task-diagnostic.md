---
date: 2026-05-26
status: per-task root-cause diagnostic for 100-task corpus
based-on: results/resolve_all.json + per-task source files
scope: read-only diagnosis (no code/yaml/meta changes)
---

# Per-Task Root-Cause Diagnostic

## Summary

Total tasks analysed: 100

### Root-cause categories

| category | count |
|---|---|
| `meta-resolver-mismatch` | 52 |
| `shapes-incomplete-sibling` | 13 |
| `triton-autotune-api-mismatch` | 8 |
| `needs-gpu` | 5 |
| `upstream-interpretererror` | 4 |
| `oracle-numeric-tolerance` | 4 |
| `upstream-exprerror` | 3 |
| `missing-shape-row` | 3 |
| `oracle-structure-mismatch` | 2 |
| `kernel-assertion` | 1 |
| `platform-stub-noimpl` | 1 |
| `duplicate-op-registration` | 1 |
| `smoke-fail-severed-run` | 1 |
| `oracle-dtype-mismatch` | 1 |
| `kernel-zerodivision` | 1 |

### Effort estimate distribution

| effort | count | meaning |
|---|---|---|
| S | 7 | ≤30 min per task |
| M | 83 | 30-120 min per task |
| L | 10 | ≥120 min per task |

## Per-Task Detail

### family: `attention` (n=18)

#### `L1_02_compute_slot_mappings`

- **aggregate status**: `BLOCKED(no-verified-upstream-entrypoint: upstream-not-invocable-here)`
- **intended kernel**: `_compute_slot_mappings_kernel`
- **intended substatus**: `upstream-not-invocable-here`
- **upstream exception**: `InterpreterError` (TypeError("_compute_slot_mappings_kernel() missing 14 required positional arguments: 'max_num_tokens', 'idx_mapping', 'q)
- **root cause category**: `upstream-interpretererror`
- **root cause detail**: Upstream raises InterpreterError: TypeError("_compute_slot_mappings_kernel() missing 14 required positional arguments: 'max_num_tokens', 'idx_mapping', 'query_start_loc', 'pos', 'block_table_ptrs', 'block_table_strides', 'block_sizes'
- **kernel required params** (9): `max_num_tokens, idx_mapping, query_start_loc, pos, block_table_ptrs, block_table_strides, block_sizes, slot_mappings_stride, cp_rank`
- **missing in shapes.yaml** (7): `query_start_loc, pos, block_table_ptrs, block_table_strides, block_sizes, slot_mappings_stride, cp_rank`
- **shapes.yaml declared inputs** (5): `block_size, block_table, idx_mapping, max_num_tokens, positions`
- **fix recipe**:
  - Inspect the exception's call site in upstream
  - Decide: shim, mock, or rewrite oracle_pytorch.py standalone
- **effort**: `M`

#### `L1_04_copy_page_indices`

- **aggregate status**: `BLOCKED(no-verified-upstream-entrypoint: upstream-not-invocable-here)`
- **intended kernel**: `_copy_page_indices_kernel`
- **upstream exception**: `ModuleNotFoundError` (No module named 'flashinfer' | pa_oracle_check_runner.py", line 303, in _load
    spec.loader.exec_module(mod)
    ~~~~~)
- **root cause category**: `meta-resolver-mismatch`
- **root cause detail**: Resolver candidate-search did not try the meta-declared kernel `_copy_page_indices_kernel`. Likely meta.upstream.path / meta.oracle.gpu_triton points at a stale location, OR the kernel.py vendored function is named differently than meta records. Resolver tried instead: ['trtllm_prefill_attn_kvfp8_dequant']
- **kernel required params** (4): `page_indices, block_table, block_table_stride, cu_num_blocks`
- **missing in shapes.yaml** (3): `page_indices, block_table_stride, cu_num_blocks`
- **shapes.yaml declared inputs** (3): `block_table, page_size, seq_lens`
- **fix recipe**:
  - Verify `_copy_page_indices_kernel` exists in kernel.py (grep `def _copy_page_indices_kernel`)
  - If not, find the actual entrypoint name in kernel.py and update meta.name
  - If yes, audit scripts/resolve_upstream_entrypoints.py candidate-search to understand why this kernel was not tried
  - Then add shapes.yaml inputs matching `_copy_page_indices_kernel`'s signature
- **effort**: `M`

#### `L1_06_reshape_and_cache_kernel_flash`

- **aggregate status**: `BLOCKED(no-verified-upstream-entrypoint: shapes-incomplete)`
- **intended kernel**: `reshape_and_cache_kernel_flash`
- **upstream exception**: `anchored` (on 1/2 shape(s) ['H2')
- **root cause category**: `meta-resolver-mismatch`
- **root cause detail**: Resolver candidate-search did not try the meta-declared kernel `reshape_and_cache_kernel_flash`. Likely meta.upstream.path / meta.oracle.gpu_triton points at a stale location, OR the kernel.py vendored function is named differently than meta records. Resolver tried instead: ['triton_reshape_and_cache_flash', 'triton_reshape_and_cache_flash_per_token_head_quant', 'triton_reshape_and_cache_flash_diffkv']
- **kernel required params** (13): `k_scale, v_scale, key_stride, value_stride, block_stride, head_stride, dim_stride_k, dim_stride_v, page_stride, num_heads, head_size, block_size, x`
- **missing in shapes.yaml** (13): `k_scale, v_scale, key_stride, value_stride, block_stride, head_stride, dim_stride_k, dim_stride_v, page_stride, num_heads, head_size, block_size, x`
- **shapes.yaml declared inputs** (5): `key, key_cache, slot_mapping, value, value_cache`
- **fix recipe**:
  - Verify `reshape_and_cache_kernel_flash` exists in kernel.py (grep `def reshape_and_cache_kernel_flash`)
  - If not, find the actual entrypoint name in kernel.py and update meta.name
  - If yes, audit scripts/resolve_upstream_entrypoints.py candidate-search to understand why this kernel was not tried
  - Then add shapes.yaml inputs matching `reshape_and_cache_kernel_flash`'s signature
- **effort**: `M`

#### `L1_08_dequantize_k_cache_paged`

- **aggregate status**: `BLOCKED(no-verified-upstream-entrypoint: shapes-incomplete)`
- **intended kernel**: `dequantize_k_cache_paged`
- **intended substatus**: `shapes-incomplete`
- **upstream exception**: `anchored` (on 1/2 shape(s) ['H2')
- **root cause category**: `shapes-incomplete-sibling`
- **root cause detail**: Resolver's missing-param mention (`quant_k_cache`) belongs to a sibling candidate, not the intended kernel `dequantize_k_cache_paged`. The aggregate shapes-incomplete masks the intended kernel's true status.
- **kernel required params** (4): `output_stride_0, input_nope_q_stride_0, input_nope_s_stride_0, input_rope_stride_0`
- **missing in shapes.yaml** (4): `output_stride_0, input_nope_q_stride_0, input_nope_s_stride_0, input_rope_stride_0`
- **shapes.yaml declared inputs** (5): `group_size, nope_q, nope_s, page_table, rope`
- **fix recipe**:
  - Verify intended kernel `dequantize_k_cache_paged` actually exists in kernel.py
  - If `dequantize_k_cache_paged` is correct, the resolver may be trying wrong siblings -- fix meta.upstream.path or meta.oracle.gpu_triton to point at the right symbol
- **effort**: `M`

#### `L1_25_merge_attn_states`

- **aggregate status**: `BLOCKED(no-verified-upstream-entrypoint: oracle-mismatch)`
- **intended kernel**: `merge_attn_states`
- **intended substatus**: `oracle-mismatch`
- **upstream exception**: `anchored` (on 1/2 shape(s) ['H2')
- **root cause category**: `oracle-structure-mismatch`
- **root cause detail**: Kernel output and oracle disagree on output structure (shape, dtype, or operator semantics).
- **kernel required params** (10): `output, output_lse, prefix_output, prefix_lse, suffix_output, suffix_lse, prefix_head_stride, output_head_stride, output_scale, prefill_tokens_with_context`
- **missing in shapes.yaml** (5): `output_lse, prefix_head_stride, output_head_stride, output_scale, prefill_tokens_with_context`
- **shapes.yaml declared inputs** (5): `output, prefix_lse, prefix_output, suffix_lse, suffix_output`
- **fix recipe**:
  - Read kernel.py + oracle_pytorch.py side by side
  - Identify which output dim/dtype the oracle expects vs what kernel produces
  - Fix oracle_pytorch.py to match the kernel's true output contract
- **effort**: `M`

#### `L1_31_triton_mrope_forward`

- **aggregate status**: `BLOCKED(no-verified-upstream-entrypoint: upstream-not-invocable-here)`
- **intended kernel**: `triton_mrope`
- **intended substatus**: `upstream-not-invocable-here`
- **upstream exception**: `RuntimeError` (Tried to register an operator (vllm::flashinfer_rotary_embedding(Tensor positions, Tensor(a1!) query, Tensor(a2!) key, S)
- **root cause category**: `duplicate-op-registration`
- **root cause detail**: Upstream import triggers torch.library duplicate operator registration -- another vendored copy registered the same op first.
- **kernel required params** (14): `cos, sin, num_tokens, n_qh, n_kh, hd, rd, pad_n_qh, pad_n_kh, pad_hd, mrope_section_t, mrope_section_h, mrope_section_w, is_interleaved`
- **missing in shapes.yaml** (12): `num_tokens, n_qh, n_kh, hd, rd, pad_n_qh, pad_n_kh, pad_hd, mrope_section_t, mrope_section_h, mrope_section_w, is_interleaved`
- **shapes.yaml declared inputs** (8): `cos, head_size, k, mrope_interleaved, mrope_section, q, rotary_dim, sin`
- **fix recipe**:
  - Identify the duplicate registration path (usually two vllm copies)
  - Option A: dedupe sys.path so only one vllm import wins
  - Option B: oracle_pytorch.py rewrite to not import vllm
- **effort**: `L`

#### `L1_32_create_flashinfer_kv_indices_triton`

- **aggregate status**: `BLOCKED(no-verified-upstream-entrypoint: shapes-incomplete)`
- **intended kernel**: `create_flashinfer_kv_indices_triton`
- **upstream exception**: `anchored` (on 1/2 shape(s) ['H2')
- **root cause category**: `meta-resolver-mismatch`
- **root cause detail**: Resolver candidate-search did not try the meta-declared kernel `create_flashinfer_kv_indices_triton`. Likely meta.upstream.path / meta.oracle.gpu_triton points at a stale location, OR the kernel.py vendored function is named differently than meta records. Resolver tried instead: ['fused_qk_rope_reshape_and_cache', 'seqlens_expand_triton', 'concat_and_cast_mha_k_triton', 'launch_reshape_and_cache_flash']
- **kernel required params** (3): `kv_indptr, kv_start_idx, req_to_token_ptr_stride`
- **missing in shapes.yaml** (3): `kv_indptr, kv_start_idx, req_to_token_ptr_stride`
- **shapes.yaml declared inputs** (8): `block_table, extend_seq_lens, input_emb, k, key, page_size, q, seq_lens`
- **fix recipe**:
  - Verify `create_flashinfer_kv_indices_triton` exists in kernel.py (grep `def create_flashinfer_kv_indices_triton`)
  - If not, find the actual entrypoint name in kernel.py and update meta.name
  - If yes, audit scripts/resolve_upstream_entrypoints.py candidate-search to understand why this kernel was not tried
  - Then add shapes.yaml inputs matching `create_flashinfer_kv_indices_triton`'s signature
- **effort**: `M`

#### `L1_33_create_flashmla_kv_indices_triton`

- **aggregate status**: `BLOCKED(no-verified-upstream-entrypoint: shapes-incomplete)`
- **intended kernel**: `create_flashmla_kv_indices_triton`
- **upstream exception**: `anchored` (on 1/2 shape(s) ['H2')
- **root cause category**: `meta-resolver-mismatch`
- **root cause detail**: Resolver candidate-search did not try the meta-declared kernel `create_flashmla_kv_indices_triton`. Likely meta.upstream.path / meta.oracle.gpu_triton points at a stale location, OR the kernel.py vendored function is named differently than meta records. Resolver tried instead: ['fused_qk_rope_reshape_and_cache', 'launch_reshape_and_cache_flash', 'seqlens_expand_triton', 'concat_and_cast_mha_k_triton']
- **kernel required params** (3): `kv_start_idx, req_to_token_ptr_stride, kv_indices_ptr_stride`
- **missing in shapes.yaml** (3): `kv_start_idx, req_to_token_ptr_stride, kv_indices_ptr_stride`
- **shapes.yaml declared inputs** (10): `block_table, extend_seq_lens, input_emb, k, key, page_size, q, seq_lens, v, value`
- **fix recipe**:
  - Verify `create_flashmla_kv_indices_triton` exists in kernel.py (grep `def create_flashmla_kv_indices_triton`)
  - If not, find the actual entrypoint name in kernel.py and update meta.name
  - If yes, audit scripts/resolve_upstream_entrypoints.py candidate-search to understand why this kernel was not tried
  - Then add shapes.yaml inputs matching `create_flashmla_kv_indices_triton`'s signature
- **effort**: `M`

#### `L1_34_transform_index_page_table_decode`

- **aggregate status**: `UNRUN(needs-gpu)`
- **root cause category**: `needs-gpu`
- **root cause detail**: Kernel binds successfully on dev; oracle is consistent; Phase A' verification requires GPU/NPU execution.
- **fix recipe**:
  - Run on GPU host: bash scripts/automation/runners/tab_quickstart.sh verify-all
  - Expected outcome: PASS after Phase A' completes
- **effort**: `S`

#### `L2_06_qkv_lora_b`

- **aggregate status**: `BLOCKED(no-verified-upstream-entrypoint: upstream-not-invocable-here)`
- **intended kernel**: `_qkv_lora_b_kernel`
- **upstream exception**: `AssertionError` (| ^
    )
    ^
  File "/data1/code/agentic-rl/kernel/triton-ascend-bench/scripts/_pa_oracle_check_runner.py", line 370,)
- **root cause category**: `meta-resolver-mismatch`
- **root cause detail**: Resolver candidate-search did not try the meta-declared kernel `_qkv_lora_b_kernel`. Likely meta.upstream.path / meta.oracle.gpu_triton points at a stale location, OR the kernel.py vendored function is named differently than meta records. Resolver tried instead: ['qkv_lora_b_fwd']
- **kernel required params** (18): `x, weights, output, max_qkv_out_dim, x_stride_0, x_stride_1, w_stride_0, w_stride_1, w_stride_2, output_stride_0, output_stride_1, seg_lens, seg_indptr, weight_indices, lora_ranks, n_offs, sorted_token_ids, scalings`
- **missing in shapes.yaml** (12): `weights, output, x_stride_0, x_stride_1, w_stride_0, w_stride_1, w_stride_2, output_stride_0, output_stride_1, seg_lens, n_offs, sorted_token_ids`
- **shapes.yaml declared inputs** (9): `batch_info, lora_ranks, max_qkv_out_dim, output_offset, qkv_lora_b, scalings, seg_indptr, weight_indices, x`
- **fix recipe**:
  - Verify `_qkv_lora_b_kernel` exists in kernel.py (grep `def _qkv_lora_b_kernel`)
  - If not, find the actual entrypoint name in kernel.py and update meta.name
  - If yes, audit scripts/resolve_upstream_entrypoints.py candidate-search to understand why this kernel was not tried
  - Then add shapes.yaml inputs matching `_qkv_lora_b_kernel`'s signature
- **effort**: `M`

#### `L2_22_linear_attn_decode`

- **aggregate status**: `BLOCKED(no-verified-upstream-entrypoint: shapes-incomplete)`
- **intended kernel**: `_linear_attn_decode_kernel`
- **upstream exception**: `anchored` (on 1/2 shape(s) ['H2')
- **root cause category**: `meta-resolver-mismatch`
- **root cause detail**: Resolver candidate-search did not try the meta-declared kernel `_linear_attn_decode_kernel`. Likely meta.upstream.path / meta.oracle.gpu_triton points at a stale location, OR the kernel.py vendored function is named differently than meta records. Resolver tried instead: ['linear_decode_forward_triton']
- **kernel required params** (9): `slope_rate, slot_idx, qkv_b_stride, qkv_h_stride, cache_b_stride, cache_h_stride, cache_d0_stride, cache_d1_stride, pad_slot_id`
- **missing in shapes.yaml** (8): `slot_idx, qkv_b_stride, qkv_h_stride, cache_b_stride, cache_h_stride, cache_d0_stride, cache_d1_stride, pad_slot_id`
- **shapes.yaml declared inputs** (5): `k, kv_caches, q, slope_rate, v`
- **fix recipe**:
  - Verify `_linear_attn_decode_kernel` exists in kernel.py (grep `def _linear_attn_decode_kernel`)
  - If not, find the actual entrypoint name in kernel.py and update meta.name
  - If yes, audit scripts/resolve_upstream_entrypoints.py candidate-search to understand why this kernel was not tried
  - Then add shapes.yaml inputs matching `_linear_attn_decode_kernel`'s signature
- **effort**: `M`

#### `L2_23_fwd_kv_parallel`

- **aggregate status**: `BLOCKED(no-verified-upstream-entrypoint: shapes-incomplete)`
- **intended kernel**: `_fwd_kv_parallel`
- **upstream exception**: `anchored` (on 1/2 shape(s) ['H2')
- **root cause category**: `meta-resolver-mismatch`
- **root cause detail**: Resolver candidate-search did not try the meta-declared kernel `_fwd_kv_parallel`. Likely meta.upstream.path / meta.oracle.gpu_triton points at a stale location, OR the kernel.py vendored function is named differently than meta records. Resolver tried instead: ['linear_decode_forward_triton']
- **kernel required params** (6): `K_decay, b, h, n, d, e`
- **missing in shapes.yaml** (6): `K_decay, b, h, n, d, e`
- **shapes.yaml declared inputs** (4): `k, q, slope, v`
- **fix recipe**:
  - Verify `_fwd_kv_parallel` exists in kernel.py (grep `def _fwd_kv_parallel`)
  - If not, find the actual entrypoint name in kernel.py and update meta.name
  - If yes, audit scripts/resolve_upstream_entrypoints.py candidate-search to understand why this kernel was not tried
  - Then add shapes.yaml inputs matching `_fwd_kv_parallel`'s signature
- **effort**: `M`

#### `L2_24_fwd_kv_reduce`

- **aggregate status**: `BLOCKED(no-verified-upstream-entrypoint: shapes-incomplete)`
- **intended kernel**: `_fwd_kv_reduce`
- **upstream exception**: `anchored` (on 1/2 shape(s) ['H2')
- **root cause category**: `meta-resolver-mismatch`
- **root cause detail**: Resolver candidate-search did not try the meta-declared kernel `_fwd_kv_reduce`. Likely meta.upstream.path / meta.oracle.gpu_triton points at a stale location, OR the kernel.py vendored function is named differently than meta records. Resolver tried instead: ['linear_decode_forward_triton']
- **kernel required params** (5): `b, h, n, d, e`
- **missing in shapes.yaml** (5): `b, h, n, d, e`
- **shapes.yaml declared inputs** (4): `k, q, slope, v`
- **fix recipe**:
  - Verify `_fwd_kv_reduce` exists in kernel.py (grep `def _fwd_kv_reduce`)
  - If not, find the actual entrypoint name in kernel.py and update meta.name
  - If yes, audit scripts/resolve_upstream_entrypoints.py candidate-search to understand why this kernel was not tried
  - Then add shapes.yaml inputs matching `_fwd_kv_reduce`'s signature
- **effort**: `M`

#### `L2_33_fwd_grouped_kernel_stage1_rope`

- **aggregate status**: `BLOCKED(no-verified-upstream-entrypoint: upstream-not-invocable-here)`
- **intended kernel**: `_fwd_grouped_kernel_stage1_rope`
- **upstream exception**: `RuntimeError` (0 active drivers ([)
- **root cause category**: `meta-resolver-mismatch`
- **root cause detail**: Resolver candidate-search did not try the meta-declared kernel `_fwd_grouped_kernel_stage1_rope`. Likely meta.upstream.path / meta.oracle.gpu_triton points at a stale location, OR the kernel.py vendored function is named differently than meta records. Resolver tried instead: ['_decode_grouped_att_m_fwd_rope']
- **kernel required params** (15): `K_Buffer, V_buffer, cos_sin_cache, positions, sm_scale, kv_indptr, kv_indices, Att_Out, k_pe_t_out, rotary_dim, kv_lora_rank, qk_rope_head_dim, kv_group_num, q_head_num, logit_cap`
- **missing in shapes.yaml** (15): `K_Buffer, V_buffer, cos_sin_cache, positions, sm_scale, kv_indptr, kv_indices, Att_Out, k_pe_t_out, rotary_dim, kv_lora_rank, qk_rope_head_dim, kv_group_num, q_head_num, logit_cap`
- **shapes.yaml declared inputs** (5): `att_out, k_buffer, q, scale, v_buffer`
- **fix recipe**:
  - Verify `_fwd_grouped_kernel_stage1_rope` exists in kernel.py (grep `def _fwd_grouped_kernel_stage1_rope`)
  - If not, find the actual entrypoint name in kernel.py and update meta.name
  - If yes, audit scripts/resolve_upstream_entrypoints.py candidate-search to understand why this kernel was not tried
  - Then add shapes.yaml inputs matching `_fwd_grouped_kernel_stage1_rope`'s signature
- **effort**: `M`

#### `L3_01_kernel_unified_attention_3d`

- **aggregate status**: `BLOCKED(no-verified-upstream-entrypoint: upstream-not-invocable-here)`
- **intended kernel**: `unified_attention`
- **intended substatus**: `upstream-not-invocable-here`
- **upstream exception**: `ZeroDivisionError` (integer division or modulo by zero)
- **root cause category**: `kernel-zerodivision`
- **root cause detail**: Shape sweep produces a zero-valued dim/stride the kernel does not guard.
- **kernel required params** (14): `scale, k_scale, v_scale, out_scale, softcap, num_query_heads, num_queries_per_kv, block_table_stride, query_stride_0, query_stride_1, output_stride_0, output_stride_1, qq_bias_stride_0, num_seqs`
- **missing in shapes.yaml** (13): `scale, k_scale, v_scale, out_scale, num_query_heads, num_queries_per_kv, block_table_stride, query_stride_0, query_stride_1, output_stride_0, output_stride_1, qq_bias_stride_0, num_seqs`
- **shapes.yaml declared inputs** (16): `block_table, causal, cu_seqlens_q, k, k_descale, max_seqlen_k, max_seqlen_q, out, q, q_descale, seqused_k, softcap, softmax_scale, v, v_descale, window_size`
- **fix recipe**:
  - Identify which shape row triggers the zero (usually H2/H_singleton with extreme small knobs)
  - Add a guard in shapes.yaml hidden_shapes or in oracle_pytorch.py for that row
- **effort**: `S`

#### `L3_02_kernel_paged_attention_2d`

- **aggregate status**: `BLOCKED(no-verified-upstream-entrypoint: shapes-incomplete)`
- **intended kernel**: `kernel_paged_attention_2d`
- **upstream exception**: `anchored` (on 1/2 shape(s) ['H2')
- **root cause category**: `meta-resolver-mismatch`
- **root cause detail**: Resolver candidate-search did not try the meta-declared kernel `kernel_paged_attention_2d`. Likely meta.upstream.path / meta.oracle.gpu_triton points at a stale location, OR the kernel.py vendored function is named differently than meta records. Resolver tried instead: ['chunked_prefill_paged_decode']
- **kernel required params** (14): `scale, k_scale, v_scale, out_scale_inv, num_query_heads, num_queries_per_kv, num_queries_per_kv_padded, block_table_stride, query_stride_0, query_stride_1, output_stride_0, output_stride_1, x, filter_by_query_len`
- **missing in shapes.yaml** (13): `k_scale, v_scale, out_scale_inv, num_query_heads, num_queries_per_kv, num_queries_per_kv_padded, block_table_stride, query_stride_0, query_stride_1, output_stride_0, output_stride_1, x, filter_by_query_len`
- **shapes.yaml declared inputs** (7): `block_table, key, key_cache, query, scale, seq_lens, value_cache`
- **fix recipe**:
  - Verify `kernel_paged_attention_2d` exists in kernel.py (grep `def kernel_paged_attention_2d`)
  - If not, find the actual entrypoint name in kernel.py and update meta.name
  - If yes, audit scripts/resolve_upstream_entrypoints.py candidate-search to understand why this kernel was not tried
  - Then add shapes.yaml inputs matching `kernel_paged_attention_2d`'s signature
- **effort**: `M`

#### `L3_04_fwd`

- **aggregate status**: `BLOCKED(no-verified-upstream-entrypoint: upstream-not-invocable-here)`
- **intended kernel**: `_fwd_kernel`
- **upstream exception**: `oracle` (-check subprocess produced no result (rc=-11):)
- **root cause category**: `meta-resolver-mismatch`
- **root cause detail**: Resolver candidate-search did not try the meta-declared kernel `_fwd_kernel`. Likely meta.upstream.path / meta.oracle.gpu_triton points at a stale location, OR the kernel.py vendored function is named differently than meta records. Resolver tried instead: []
- **kernel required params** (6): `sm_scale, B_Start_Loc, B_Seqlen, Out, kv_group_num, Lk`
- **missing in shapes.yaml** (5): `B_Start_Loc, B_Seqlen, Out, kv_group_num, Lk`
- **shapes.yaml declared inputs** (9): `b_seq_len, b_start_loc, is_causal, k, max_input_len, o, q, sm_scale, v`
- **fix recipe**:
  - Verify `_fwd_kernel` exists in kernel.py (grep `def _fwd_kernel`)
  - If not, find the actual entrypoint name in kernel.py and update meta.name
  - If yes, audit scripts/resolve_upstream_entrypoints.py candidate-search to understand why this kernel was not tried
  - Then add shapes.yaml inputs matching `_fwd_kernel`'s signature
- **effort**: `M`

#### `L3_07_correct_attn_cp_out`

- **aggregate status**: `BLOCKED(no-verified-upstream-entrypoint: shapes-incomplete)`
- **intended kernel**: `_correct_attn_cp_out_kernel`
- **upstream exception**: `anchored` (on 1/2 shape(s) ['H2')
- **root cause category**: `meta-resolver-mismatch`
- **root cause detail**: Resolver candidate-search did not try the meta-declared kernel `_correct_attn_cp_out_kernel`. Likely meta.upstream.path / meta.oracle.gpu_triton points at a stale location, OR the kernel.py vendored function is named differently than meta records. Resolver tried instead: ['pack_seq_triton', 'unpack_seq_triton']
- **kernel required params** (7): `outputs_stride_B, outputs_stride_H, outputs_stride_D, lses_stride_N, lses_stride_B, lses_stride_H, lse_idx`
- **missing in shapes.yaml** (7): `outputs_stride_B, outputs_stride_H, outputs_stride_D, lses_stride_N, lses_stride_B, lses_stride_H, lse_idx`
- **shapes.yaml declared inputs** (5): `cp_rank, is_lse_base_on_e, lses, out, x`
- **fix recipe**:
  - Verify `_correct_attn_cp_out_kernel` exists in kernel.py (grep `def _correct_attn_cp_out_kernel`)
  - If not, find the actual entrypoint name in kernel.py and update meta.name
  - If yes, audit scripts/resolve_upstream_entrypoints.py candidate-search to understand why this kernel was not tried
  - Then add shapes.yaml inputs matching `_correct_attn_cp_out_kernel`'s signature
- **effort**: `M`

### family: `convolution` (n=3)

#### `L1_03_convert_req_index_to_global_index`

- **aggregate status**: `BLOCKED(no-verified-upstream-entrypoint: upstream-not-invocable-here)`
- **intended kernel**: `_convert_req_index_to_global_index_kernel`
- **upstream exception**: `KeyError` ('block_size' | ts,
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    )
    ^
  File "/data1/code/agentic-rl/kernel/triton-ascen)
- **root cause category**: `meta-resolver-mismatch`
- **root cause detail**: Resolver candidate-search did not try the meta-declared kernel `_convert_req_index_to_global_index_kernel`. Likely meta.upstream.path / meta.oracle.gpu_triton points at a stale location, OR the kernel.py vendored function is named differently than meta records. Resolver tried instead: ['triton_convert_req_index_to_global_index']
- **kernel required params** (7): `max_num_blocks_per_req, bt_stride0, bt_stride1, ti_stride0, ti_stride1, out_stride0, out_stride1`
- **missing in shapes.yaml** (7): `max_num_blocks_per_req, bt_stride0, bt_stride1, ti_stride0, ti_stride1, out_stride0, out_stride1`
- **shapes.yaml declared inputs** (6): `BLOCK_N, BLOCK_SIZE, NUM_TOPK_TOKENS, block_table, req_id, token_indices`
- **fix recipe**:
  - Verify `_convert_req_index_to_global_index_kernel` exists in kernel.py (grep `def _convert_req_index_to_global_index_kernel`)
  - If not, find the actual entrypoint name in kernel.py and update meta.name
  - If yes, audit scripts/resolve_upstream_entrypoints.py candidate-search to understand why this kernel was not tried
  - Then add shapes.yaml inputs matching `_convert_req_index_to_global_index_kernel`'s signature
- **effort**: `M`

#### `L2_20_causal_conv1d_update`

- **aggregate status**: `BLOCKED(no-verified-upstream-entrypoint: shapes-incomplete)`
- **intended kernel**: `causal_conv1d_update`
- **intended substatus**: `upstream-not-invocable-here`
- **upstream exception**: `InterpreterError` (AttributeError("'NoneType' object has no attribute 'type'"))
- **root cause category**: `upstream-interpretererror`
- **root cause detail**: Upstream raises InterpreterError: AttributeError("'NoneType' object has no attribute 'type'")
- **kernel required params** (8): `block_idx_last_scheduled_token, initial_state_idx, batch, dim, seqlen, state_len, num_cache_lines, null_block_id`
- **missing in shapes.yaml** (8): `block_idx_last_scheduled_token, initial_state_idx, batch, dim, seqlen, state_len, num_cache_lines, null_block_id`
- **shapes.yaml declared inputs** (5): `activation, bias, conv_state, weight, x`
- **fix recipe**:
  - Inspect the exception's call site in upstream
  - Decide: shim, mock, or rewrite oracle_pytorch.py standalone
- **effort**: `M`

#### `L2_21_causal_conv1d_fwd`

- **aggregate status**: `BLOCKED(no-verified-upstream-entrypoint: shapes-incomplete)`
- **intended kernel**: `_causal_conv1d_fwd_kernel`
- **upstream exception**: `anchored` (on 1/2 shape(s) ['H2')
- **root cause category**: `meta-resolver-mismatch`
- **root cause detail**: Resolver candidate-search did not try the meta-declared kernel `_causal_conv1d_fwd_kernel`. Likely meta.upstream.path / meta.oracle.gpu_triton points at a stale location, OR the kernel.py vendored function is named differently than meta records. Resolver tried instead: ['causal_conv1d_fn', 'causal_conv1d_update']
- **kernel required params** (9): `block_idx_first_scheduled_token, block_idx_last_scheduled_token, initial_state_idx, num_computed_tokens, dim, seqlen, num_cache_lines, pad_slot_id, null_block_id`
- **missing in shapes.yaml** (9): `block_idx_first_scheduled_token, block_idx_last_scheduled_token, initial_state_idx, num_computed_tokens, dim, seqlen, num_cache_lines, pad_slot_id, null_block_id`
- **shapes.yaml declared inputs** (4): `activation, bias, weight, x`
- **fix recipe**:
  - Verify `_causal_conv1d_fwd_kernel` exists in kernel.py (grep `def _causal_conv1d_fwd_kernel`)
  - If not, find the actual entrypoint name in kernel.py and update meta.name
  - If yes, audit scripts/resolve_upstream_entrypoints.py candidate-search to understand why this kernel was not tried
  - Then add shapes.yaml inputs matching `_causal_conv1d_fwd_kernel`'s signature
- **effort**: `M`

### family: `elementwise` (n=7)

#### `L1_15_act_and_mul`

- **aggregate status**: `BLOCKED(no-verified-upstream-entrypoint: shapes-incomplete)`
- **intended kernel**: `act_and_mul_kernel`
- **upstream exception**: `anchored` (on 1/2 shape(s) ['H2')
- **root cause category**: `meta-resolver-mismatch`
- **root cause detail**: Resolver candidate-search did not try the meta-declared kernel `act_and_mul_kernel`. Likely meta.upstream.path / meta.oracle.gpu_triton points at a stale location, OR the kernel.py vendored function is named differently than meta records. Resolver tried instead: ['act_and_mul_triton', 'moe_sum_reduce_triton', 'fused_append_shared_experts', 'fused_append_shared_experts_with_weights']
- **kernel required params** (4): `gateup_output, down_input, hidden_size, expert_step`
- **missing in shapes.yaml** (4): `gateup_output, down_input, hidden_size, expert_step`
- **shapes.yaml declared inputs** (3): `act, input, x`
- **fix recipe**:
  - Verify `act_and_mul_kernel` exists in kernel.py (grep `def act_and_mul_kernel`)
  - If not, find the actual entrypoint name in kernel.py and update meta.name
  - If yes, audit scripts/resolve_upstream_entrypoints.py candidate-search to understand why this kernel was not tried
  - Then add shapes.yaml inputs matching `act_and_mul_kernel`'s signature
- **effort**: `M`

#### `L1_16_silu_and_mul`

- **aggregate status**: `UNRUN(needs-gpu)`
- **root cause category**: `needs-gpu`
- **root cause detail**: Kernel binds successfully on dev; oracle is consistent; Phase A' verification requires GPU/NPU execution.
- **fix recipe**:
  - Run on GPU host: bash scripts/automation/runners/tab_quickstart.sh verify-all
  - Expected outcome: PASS after Phase A' completes
- **effort**: `S`

#### `L1_17_gelu_and_mul`

- **aggregate status**: `BLOCKED(no-verified-upstream-entrypoint: shapes-incomplete)`
- **intended kernel**: `gelu_and_mul_kernel`
- **upstream exception**: `ValueError` (too many values to unpack (expected 2) | gentic-rl/kernel/triton-ascend-bench/scripts/_pa_oracle_check_runner.py", line )
- **root cause category**: `meta-resolver-mismatch`
- **root cause detail**: Resolver candidate-search did not try the meta-declared kernel `gelu_and_mul_kernel`. Likely meta.upstream.path / meta.oracle.gpu_triton points at a stale location, OR the kernel.py vendored function is named differently than meta records. Resolver tried instead: ['gelu_and_mul_triton', 'fused_softcap', 'fused_dual_residual_rmsnorm', 'silu_and_mul_triton']
- **kernel required params** (3): `quant_max, static_scale, hidden_dim`
- **missing in shapes.yaml** (3): `quant_max, static_scale, hidden_dim`
- **shapes.yaml declared inputs** (6): `approximate, hidden_states, residual, softcap_const, weight1, x`
- **fix recipe**:
  - Verify `gelu_and_mul_kernel` exists in kernel.py (grep `def gelu_and_mul_kernel`)
  - If not, find the actual entrypoint name in kernel.py and update meta.name
  - If yes, audit scripts/resolve_upstream_entrypoints.py candidate-search to understand why this kernel was not tried
  - Then add shapes.yaml inputs matching `gelu_and_mul_kernel`'s signature
- **effort**: `M`

#### `L1_18_fused_softcap`

- **aggregate status**: `BLOCKED(no-verified-upstream-entrypoint: shapes-incomplete)`
- **intended kernel**: `fused_softcap`
- **intended substatus**: `shapes-incomplete`
- **upstream exception**: `anchored` (on 1/2 shape(s) ['H2')
- **root cause category**: `shapes-incomplete-sibling`
- **root cause detail**: Resolver's missing-param mention (`full_logits`) belongs to a sibling candidate, not the intended kernel `fused_softcap`. The aggregate shapes-incomplete masks the intended kernel's true status.
- **kernel required params** (3): `softcapping_value, ncols, row_stride`
- **missing in shapes.yaml** (3): `softcapping_value, ncols, row_stride`
- **shapes.yaml declared inputs** (2): `cap, x`
- **fix recipe**:
  - Verify intended kernel `fused_softcap` actually exists in kernel.py
  - If `fused_softcap` is correct, the resolver may be trying wrong siblings -- fix meta.upstream.path or meta.oracle.gpu_triton to point at the right symbol
- **effort**: `M`

#### `L1_26_merge_state`

- **aggregate status**: `UNRUN(needs-gpu)`
- **root cause category**: `needs-gpu`
- **root cause detail**: Kernel binds successfully on dev; oracle is consistent; Phase A' verification requires GPU/NPU execution.
- **fix recipe**:
  - Run on GPU host: bash scripts/automation/runners/tab_quickstart.sh verify-all
  - Expected outcome: PASS after Phase A' completes
- **effort**: `S`

#### `L2_38_concat_and_cast_mha_k`

- **aggregate status**: `BLOCKED(no-verified-upstream-entrypoint: shapes-incomplete)`
- **intended kernel**: `concat_and_cast_mha_k_kernel`
- **upstream exception**: `AssertionError` (shape should be 3d, but got k.shape=torch.Size([2, 2, 8, 16)
- **root cause category**: `meta-resolver-mismatch`
- **root cause detail**: Resolver candidate-search did not try the meta-declared kernel `concat_and_cast_mha_k_kernel`. Likely meta.upstream.path / meta.oracle.gpu_triton points at a stale location, OR the kernel.py vendored function is named differently than meta records. Resolver tried instead: ['concat_and_cast_mha_k_triton', 'fused_qk_rope_reshape_and_cache', 'launch_reshape_and_cache_flash', 'seqlens_expand_triton']
- **kernel required params** (8): `head_cnt, k_stride0, k_stride1, nope_stride0, nope_stride1, rope_stride0, nope_dim, rope_dim`
- **missing in shapes.yaml** (8): `head_cnt, k_stride0, k_stride1, nope_stride0, nope_stride1, rope_stride0, nope_dim, rope_dim`
- **shapes.yaml declared inputs** (11): `extend_seq_lens, input_emb, k, k_nope, k_rope, key, out_dtype, q, seq_lens, v, value`
- **fix recipe**:
  - Verify `concat_and_cast_mha_k_kernel` exists in kernel.py (grep `def concat_and_cast_mha_k_kernel`)
  - If not, find the actual entrypoint name in kernel.py and update meta.name
  - If yes, audit scripts/resolve_upstream_entrypoints.py candidate-search to understand why this kernel was not tried
  - Then add shapes.yaml inputs matching `concat_and_cast_mha_k_kernel`'s signature
- **effort**: `M`

#### `L3_23_merge_16x16_to_32x32_inverse`

- **aggregate status**: `BLOCKED(no-verified-upstream-entrypoint: upstream-not-invocable-here)`
- **intended kernel**: `merge_16x16_to_32x32_inverse_kernel`
- **intended substatus**: `upstream-not-invocable-here`
- **upstream exception**: `AttributeError` ('_AutotunedKernel' object has no attribute 'arg_names')
- **root cause category**: `triton-autotune-api-mismatch`
- **root cause detail**: Kernel applies @triton.heuristics on top of a wrapper that lacks .arg_names. Current Triton API expects different ordering.
- **kernel required params** (3): `Ai, cu_seqlens, chunk_indices`
- **missing in shapes.yaml** (3): `Ai, cu_seqlens, chunk_indices`
- **shapes.yaml declared inputs** (1): `A`
- **fix recipe**:
  - Pin Triton version OR rewrite kernel.py decorator ordering
  - If oracle_pytorch.py is independent, defer kernel.py fix to Phase B
- **effort**: `L`

### family: `gemm` (n=9)

#### `L2_01_bmm_chunk_fwd`

- **aggregate status**: `BLOCKED(no-verified-upstream-entrypoint: smoke-fail)`
- **intended kernel**: `_bmm_chunk_fwd`
- **intended substatus**: `oracle-consistent`
- **upstream exception**: `S2` (-smoke MISMATCH (CPU/TRITON_INTERPRET pre-filter, phase=severed-no-run): severed side did not run (rc=2): RuntimeError: )
- **root cause category**: `smoke-fail-severed-run`
- **root cause detail**: S2-smoke side process did not run (severed or original side rc!=0). Usually CPU interpreter cannot exec the kernel within budget.
- **kernel required params** (4): `seqlen, chunk_size, ngroups, dot_dtype`
- **missing in shapes.yaml** (3): `seqlen, ngroups, dot_dtype`
- **shapes.yaml declared inputs** (5): `a, b, causal, chunk_size, cu_chunk_seqlens`
- **fix recipe**:
  - Reduce smoke shape coverage to H2 (smallest hidden shape)
  - Or skip CPU pre-filter for this task family
- **effort**: `M`

#### `L2_04_chunked_lora_expand`

- **aggregate status**: `BLOCKED(no-verified-upstream-entrypoint: upstream-not-invocable-here)`
- **intended kernel**: `_chunked_lora_expand_kernel`
- **upstream exception**: `AttributeError` ('NoneType' object has no attribute 'replace' | /data1/code/agentic-rl/kernel/infer/sglang/python/sglang/srt/lora/triton_)
- **root cause category**: `meta-resolver-mismatch`
- **root cause detail**: Resolver candidate-search did not try the meta-declared kernel `_chunked_lora_expand_kernel`. Likely meta.upstream.path / meta.oracle.gpu_triton points at a stale location, OR the kernel.py vendored function is named differently than meta records. Resolver tried instead: ['chunked_sgmv_lora_expand_forward']
- **kernel required params** (12): `x, weights, output, output_stride_0, output_stride_1, seg_indptr, weight_indices, lora_ranks, permutation, num_segs, scalings, slice_offsets`
- **missing in shapes.yaml** (5): `output, output_stride_0, output_stride_1, permutation, num_segs`
- **shapes.yaml declared inputs** (10): `base_output, batch_info, lora_ranks, max_slice_size, scalings, seg_indptr, slice_offsets, weight_indices, weights, x`
- **fix recipe**:
  - Verify `_chunked_lora_expand_kernel` exists in kernel.py (grep `def _chunked_lora_expand_kernel`)
  - If not, find the actual entrypoint name in kernel.py and update meta.name
  - If yes, audit scripts/resolve_upstream_entrypoints.py candidate-search to understand why this kernel was not tried
  - Then add shapes.yaml inputs matching `_chunked_lora_expand_kernel`'s signature
- **effort**: `M`

#### `L2_05_chunked_lora_shrink`

- **aggregate status**: `BLOCKED(no-verified-upstream-entrypoint: upstream-not-invocable-here)`
- **intended kernel**: `_chunked_lora_shrink_kernel`
- **upstream exception**: `AttributeError` ('NoneType' object has no attribute 'replace' | /data1/code/agentic-rl/kernel/infer/sglang/python/sglang/srt/lora/triton_)
- **root cause category**: `meta-resolver-mismatch`
- **root cause detail**: Resolver candidate-search did not try the meta-declared kernel `_chunked_lora_shrink_kernel`. Likely meta.upstream.path / meta.oracle.gpu_triton points at a stale location, OR the kernel.py vendored function is named differently than meta records. Resolver tried instead: ['chunked_sgmv_lora_shrink_forward']
- **kernel required params** (8): `x, weights, output, seg_indptr, weight_indices, lora_ranks, permutation, num_segs`
- **missing in shapes.yaml** (3): `output, permutation, num_segs`
- **shapes.yaml declared inputs** (8): `batch_info, lora_ranks, num_slices, scalings, seg_indptr, weight_indices, weights, x`
- **fix recipe**:
  - Verify `_chunked_lora_shrink_kernel` exists in kernel.py (grep `def _chunked_lora_shrink_kernel`)
  - If not, find the actual entrypoint name in kernel.py and update meta.name
  - If yes, audit scripts/resolve_upstream_entrypoints.py candidate-search to understand why this kernel was not tried
  - Then add shapes.yaml inputs matching `_chunked_lora_shrink_kernel`'s signature
- **effort**: `M`

#### `L2_07_gate_up_lora_b`

- **aggregate status**: `BLOCKED(no-verified-upstream-entrypoint: upstream-not-invocable-here)`
- **intended kernel**: `_gate_up_lora_b_kernel`
- **upstream exception**: `AssertionError` (| ^^^^^^^
    )
    ^
  File "/data1/code/agentic-rl/kernel/triton-ascend-bench/scripts/_pa_oracle_check_runner.py", lin)
- **root cause category**: `meta-resolver-mismatch`
- **root cause detail**: Resolver candidate-search did not try the meta-declared kernel `_gate_up_lora_b_kernel`. Likely meta.upstream.path / meta.oracle.gpu_triton points at a stale location, OR the kernel.py vendored function is named differently than meta records. Resolver tried instead: ['gate_up_lora_b_fwd']
- **kernel required params** (17): `x, weights, output, output_dim, x_stride_0, x_stride_1, w_stride_0, w_stride_1, w_stride_2, output_stride_0, output_stride_1, seg_lens, seg_indptr, weight_indices, lora_ranks, sorted_token_ids, scalings`
- **missing in shapes.yaml** (11): `weights, output, x_stride_0, x_stride_1, w_stride_0, w_stride_1, w_stride_2, output_stride_0, output_stride_1, seg_lens, sorted_token_ids`
- **shapes.yaml declared inputs** (8): `batch_info, gate_up_lora_b, lora_ranks, output_dim, scalings, seg_indptr, weight_indices, x`
- **fix recipe**:
  - Verify `_gate_up_lora_b_kernel` exists in kernel.py (grep `def _gate_up_lora_b_kernel`)
  - If not, find the actual entrypoint name in kernel.py and update meta.name
  - If yes, audit scripts/resolve_upstream_entrypoints.py candidate-search to understand why this kernel was not tried
  - Then add shapes.yaml inputs matching `_gate_up_lora_b_kernel`'s signature
- **effort**: `M`

#### `L2_08_sgemm_lora_a`

- **aggregate status**: `BLOCKED(no-verified-upstream-entrypoint: oracle-mismatch)`
- **intended kernel**: `_sgemm_lora_a_kernel`
- **upstream exception**: `anchored` (on 1/2 shape(s) ['H2')
- **root cause category**: `meta-resolver-mismatch`
- **root cause detail**: Resolver candidate-search did not try the meta-declared kernel `_sgemm_lora_a_kernel`. Likely meta.upstream.path / meta.oracle.gpu_triton points at a stale location, OR the kernel.py vendored function is named differently than meta records. Resolver tried instead: ['sgemm_lora_a_fwd']
- **kernel required params** (16): `x, weights, output, stack_num, x_stride_0, x_stride_1, w_stride_0, w_stride_1, w_stride_2, output_stride_0, output_stride_1, seg_lens, seg_indptr, weight_indices, lora_ranks, sorted_token_ids`
- **missing in shapes.yaml** (11): `output, stack_num, x_stride_0, x_stride_1, w_stride_0, w_stride_1, w_stride_2, output_stride_0, output_stride_1, seg_lens, sorted_token_ids`
- **shapes.yaml declared inputs** (7): `batch_info, lora_ranks, scalings, seg_indptr, weight_indices, weights, x`
- **fix recipe**:
  - Verify `_sgemm_lora_a_kernel` exists in kernel.py (grep `def _sgemm_lora_a_kernel`)
  - If not, find the actual entrypoint name in kernel.py and update meta.name
  - If yes, audit scripts/resolve_upstream_entrypoints.py candidate-search to understand why this kernel was not tried
  - Then add shapes.yaml inputs matching `_sgemm_lora_a_kernel`'s signature
- **effort**: `M`

#### `L2_09_sgemm_lora_b`

- **aggregate status**: `BLOCKED(no-verified-upstream-entrypoint: upstream-not-invocable-here)`
- **intended kernel**: `_sgemm_lora_b_kernel`
- **upstream exception**: `oracle` (-check subprocess produced no result (rc=-11): /Modules/main.c", line 429, in pymain_run_file
  File "/usr/local/src/con)
- **root cause category**: `meta-resolver-mismatch`
- **root cause detail**: Resolver candidate-search did not try the meta-declared kernel `_sgemm_lora_b_kernel`. Likely meta.upstream.path / meta.oracle.gpu_triton points at a stale location, OR the kernel.py vendored function is named differently than meta records. Resolver tried instead: ['sgemm_lora_b_fwd']
- **kernel required params** (16): `x, weights, output, x_stride_0, x_stride_1, w_stride_0, w_stride_1, w_stride_2, output_stride_0, output_stride_1, seg_lens, seg_indptr, weight_indices, lora_ranks, sorted_token_ids, scalings`
- **missing in shapes.yaml** (10): `output, x_stride_0, x_stride_1, w_stride_0, w_stride_1, w_stride_2, output_stride_0, output_stride_1, seg_lens, sorted_token_ids`
- **shapes.yaml declared inputs** (7): `batch_info, lora_ranks, scalings, seg_indptr, weight_indices, weights, x`
- **fix recipe**:
  - Verify `_sgemm_lora_b_kernel` exists in kernel.py (grep `def _sgemm_lora_b_kernel`)
  - If not, find the actual entrypoint name in kernel.py and update meta.name
  - If yes, audit scripts/resolve_upstream_entrypoints.py candidate-search to understand why this kernel was not tried
  - Then add shapes.yaml inputs matching `_sgemm_lora_b_kernel`'s signature
- **effort**: `M`

#### `L2_12_awq_gemm`

- **aggregate status**: `BLOCKED(no-verified-upstream-entrypoint: shapes-incomplete)`
- **intended kernel**: `awq_gemm_kernel`
- **upstream exception**: `AssertionError` (| /_pa_oracle_check_runner.py", line 370, in _check_one_shape
    cand_ret = fn(dict(kernel_inputs)) if sig is None else)
- **root cause category**: `meta-resolver-mismatch`
- **root cause detail**: Resolver candidate-search did not try the meta-declared kernel `awq_gemm_kernel`. Likely meta.upstream.path / meta.oracle.gpu_triton points at a stale location, OR the kernel.py vendored function is named differently than meta records. Resolver tried instead: ['awq_gemm_triton', 'awq_dequantize_triton']
- **kernel required params** (1): `group_size`
- **shapes.yaml declared inputs** (6): `group_size, input, qweight, qzeros, scales, split_k_iters`
- **fix recipe**:
  - Verify `awq_gemm_kernel` exists in kernel.py (grep `def awq_gemm_kernel`)
  - If not, find the actual entrypoint name in kernel.py and update meta.name
  - If yes, audit scripts/resolve_upstream_entrypoints.py candidate-search to understand why this kernel was not tried
  - Then add shapes.yaml inputs matching `awq_gemm_kernel`'s signature
- **effort**: `M`

#### `L2_13_bmm`

- **aggregate status**: `BLOCKED(no-verified-upstream-entrypoint: shapes-incomplete)`
- **intended kernel**: `bmm_kernel`
- **upstream exception**: `S2` (-smoke MISMATCH (CPU/TRITON_INTERPRET pre-filter, phase=original-no-run): original side did not run (rc=-9): /data1/code)
- **root cause category**: `meta-resolver-mismatch`
- **root cause detail**: Resolver candidate-search did not try the meta-declared kernel `bmm_kernel`. Likely meta.upstream.path / meta.oracle.gpu_triton points at a stale location, OR the kernel.py vendored function is named differently than meta records. Resolver tried instead: ['bmm_batch_invariant', 'matmul_persistent', 'log_softmax', 'mean_dim']
- **shapes.yaml declared inputs** (3): `a, b, input`
- **fix recipe**:
  - Verify `bmm_kernel` exists in kernel.py (grep `def bmm_kernel`)
  - If not, find the actual entrypoint name in kernel.py and update meta.name
  - If yes, audit scripts/resolve_upstream_entrypoints.py candidate-search to understand why this kernel was not tried
  - Then add shapes.yaml inputs matching `bmm_kernel`'s signature
- **effort**: `M`

#### `L2_14_matmul_kernel_persistent`

- **aggregate status**: `BLOCKED(no-verified-upstream-entrypoint: shapes-incomplete)`
- **intended kernel**: `matmul_kernel_persistent`
- **upstream exception**: `NotImplementedError` (num_compute_units is not implemented for the current platform. | l/kernel/infer/vllm/vllm/utils/platform_utils.py", line)
- **root cause category**: `meta-resolver-mismatch`
- **root cause detail**: Resolver candidate-search did not try the meta-declared kernel `matmul_kernel_persistent`. Likely meta.upstream.path / meta.oracle.gpu_triton points at a stale location, OR the kernel.py vendored function is named differently than meta records. Resolver tried instead: ['matmul_persistent', 'bmm_batch_invariant', 'log_softmax', 'mean_dim']
- **shapes.yaml declared inputs** (4): `a, b, bias, input`
- **fix recipe**:
  - Verify `matmul_kernel_persistent` exists in kernel.py (grep `def matmul_kernel_persistent`)
  - If not, find the actual entrypoint name in kernel.py and update meta.name
  - If yes, audit scripts/resolve_upstream_entrypoints.py candidate-search to understand why this kernel was not tried
  - Then add shapes.yaml inputs matching `matmul_kernel_persistent`'s signature
- **effort**: `M`

### family: `moe` (n=20)

#### `L1_05_gather_block_tables`

- **aggregate status**: `BLOCKED(no-verified-upstream-entrypoint: upstream-not-invocable-here)`
- **intended kernel**: `_gather_block_tables_kernel`
- **intended substatus**: `upstream-not-invocable-here`
- **upstream exception**: `InterpreterError` (TypeError("_gather_block_tables_kernel() missing 9 required positional arguments: 'batch_idx_to_req_idx', 'src_block_tab)
- **root cause category**: `upstream-interpretererror`
- **root cause detail**: Upstream raises InterpreterError: TypeError("_gather_block_tables_kernel() missing 9 required positional arguments: 'batch_idx_to_req_idx', 'src_block_table_ptrs', 'dst_block_table_ptrs', 'block_table_strides', 'num_blocks_ptr', 'num_
- **kernel required params** (7): `batch_idx_to_req_idx, src_block_table_ptrs, dst_block_table_ptrs, block_table_strides, num_blocks_stride, num_reqs, max_num_blocks`
- **missing in shapes.yaml** (7): `batch_idx_to_req_idx, src_block_table_ptrs, dst_block_table_ptrs, block_table_strides, num_blocks_stride, num_reqs, max_num_blocks`
- **shapes.yaml declared inputs** (2): `block_table, idx_mapping`
- **fix recipe**:
  - Inspect the exception's call site in upstream
  - Decide: shim, mock, or rewrite oracle_pytorch.py standalone
- **effort**: `M`

#### `L1_35_compute_src2dst_triton`

- **aggregate status**: `BLOCKED(no-verified-upstream-entrypoint: shapes-incomplete)`
- **intended kernel**: `compute_src2dst_triton_kernel`
- **upstream exception**: `anchored` (on 1/2 shape(s) ['H2')
- **root cause category**: `meta-resolver-mismatch`
- **root cause detail**: Resolver candidate-search did not try the meta-declared kernel `compute_src2dst_triton_kernel`. Likely meta.upstream.path / meta.oracle.gpu_triton points at a stale location, OR the kernel.py vendored function is named differently than meta records. Resolver tried instead: ['cutlass_w4_run_moe_ep_preproess', 'fp8_per_token_to_per_tensor_quant_triton', 'compute_problem_sizes_w4a8', 'ep_gather']
- **kernel required params** (3): `reorder_ids, src2dst, num_toks`
- **missing in shapes.yaml** (1): `src2dst`
- **shapes.yaml declared inputs** (6): `input_tensor, masked_m, num_toks, recv_x, reorder_ids, x`
- **fix recipe**:
  - Verify `compute_src2dst_triton_kernel` exists in kernel.py (grep `def compute_src2dst_triton_kernel`)
  - If not, find the actual entrypoint name in kernel.py and update meta.name
  - If yes, audit scripts/resolve_upstream_entrypoints.py candidate-search to understand why this kernel was not tried
  - Then add shapes.yaml inputs matching `compute_src2dst_triton_kernel`'s signature
- **effort**: `M`

#### `L1_36_compute_seg_indptr_triton`

- **aggregate status**: `BLOCKED(no-verified-upstream-entrypoint: upstream-not-invocable-here)`
- **intended kernel**: `compute_seg_indptr_triton_kernel`
- **upstream exception**: `ExprError` (expression 'DIM3': unknown symbol 'DIM3' (allowed: ['B', 'C', 'D', 'Dv', 'E', 'G', 'H', 'L', 'N', 'T')
- **root cause category**: `meta-resolver-mismatch`
- **root cause detail**: Resolver candidate-search did not try the meta-declared kernel `compute_seg_indptr_triton_kernel`. Likely meta.upstream.path / meta.oracle.gpu_triton points at a stale location, OR the kernel.py vendored function is named differently than meta records. Resolver tried instead: ['moe_ep_deepgemm_preprocess', 'post_reorder_for_cutlass_moe', 'deepep_run_moe_deep_preprocess', 'compute_problem_sizes_w4a8']
- **kernel required params** (3): `reorder_topk_ids, seg_indptr, num_toks`
- **missing in shapes.yaml** (3): `reorder_topk_ids, seg_indptr, num_toks`
- **shapes.yaml declared inputs** (9): `down_output, hidden_states, input, masked_m, num_experts, num_local_experts, output, problem_sizes1, topk_ids`
- **fix recipe**:
  - Verify `compute_seg_indptr_triton_kernel` exists in kernel.py (grep `def compute_seg_indptr_triton_kernel`)
  - If not, find the actual entrypoint name in kernel.py and update meta.name
  - If yes, audit scripts/resolve_upstream_entrypoints.py candidate-search to understand why this kernel was not tried
  - Then add shapes.yaml inputs matching `compute_seg_indptr_triton_kernel`'s signature
- **effort**: `M`

#### `L1_38_silu_mul_fp8_quant_deep_gemm`

- **aggregate status**: `BLOCKED(no-verified-upstream-entrypoint: upstream-not-invocable-here)`
- **intended kernel**: `_silu_mul_fp8_quant_deep_gemm`
- **upstream exception**: `AssertionError` (| ernel/triton-ascend-bench/scripts/_pa_oracle_check_runner.py", line 370, in _check_one_shape
    cand_ret = fn(dict(ke)
- **root cause category**: `meta-resolver-mismatch`
- **root cause detail**: Resolver candidate-search did not try the meta-declared kernel `_silu_mul_fp8_quant_deep_gemm`. Likely meta.upstream.path / meta.oracle.gpu_triton points at a stale location, OR the kernel.py vendored function is named differently than meta records. Resolver tried instead: ['persistent_masked_m_silu_mul_quant']
- **kernel required params** (4): `eps, fp8_min, fp8_max, ceil_ue8m0`
- **missing in shapes.yaml** (3): `eps, fp8_min, ceil_ue8m0`
- **shapes.yaml declared inputs** (4): `fp8_max, group_size, tokens_per_expert, y`
- **fix recipe**:
  - Verify `_silu_mul_fp8_quant_deep_gemm` exists in kernel.py (grep `def _silu_mul_fp8_quant_deep_gemm`)
  - If not, find the actual entrypoint name in kernel.py and update meta.name
  - If yes, audit scripts/resolve_upstream_entrypoints.py candidate-search to understand why this kernel was not tried
  - Then add shapes.yaml inputs matching `_silu_mul_fp8_quant_deep_gemm`'s signature
- **effort**: `M`

#### `L2_10_w8a8_block_int8_matmul`

- **aggregate status**: `BLOCKED(no-verified-upstream-entrypoint: smoke-fail)`
- **intended kernel**: `_w8a8_block_int8_matmul`
- **upstream exception**: `Block` (-wise W8A8 INT8 GEMM oracle anchor passes locally on the two sampled shapes (H2 + P5) under TRITON_INTERPRET=1 with the )
- **root cause category**: `meta-resolver-mismatch`
- **root cause detail**: Resolver candidate-search did not try the meta-declared kernel `_w8a8_block_int8_matmul`. Likely meta.upstream.path / meta.oracle.gpu_triton points at a stale location, OR the kernel.py vendored function is named differently than meta records. Resolver tried instead: []
- **kernel required params** (4): `As, Bs, group_n, group_k`
- **missing in shapes.yaml** (2): `group_n, group_k`
- **shapes.yaml declared inputs** (5): `A, As, B, Bs, block_size`
- **fix recipe**:
  - Verify `_w8a8_block_int8_matmul` exists in kernel.py (grep `def _w8a8_block_int8_matmul`)
  - If not, find the actual entrypoint name in kernel.py and update meta.name
  - If yes, audit scripts/resolve_upstream_entrypoints.py candidate-search to understand why this kernel was not tried
  - Then add shapes.yaml inputs matching `_w8a8_block_int8_matmul`'s signature
- **effort**: `M`

#### `L2_25_fwd_kernel_ep_gather`

- **aggregate status**: `BLOCKED(no-verified-upstream-entrypoint: upstream-not-invocable-here)`
- **intended kernel**: `ep_gather`
- **intended substatus**: `upstream-not-invocable-here`
- **upstream exception**: `ExprError` (expression 'BN': unknown symbol 'BN' (allowed: ['B', 'C', 'D', 'Dv', 'E', 'G', 'H', 'L', 'N', 'T')
- **root cause category**: `upstream-exprerror`
- **root cause detail**: Upstream raises ExprError: expression 'BN': unknown symbol 'BN' (allowed: ['B', 'C', 'D', 'Dv', 'E', 'G', 'H', 'L', 'N', 'T'
- **kernel required params** (18): `total_token_num, input_tensor, input_tensor_stride0, input_tensor_stride1, recv_topk_ids, recv_topk_ids_stride0, recv_topk_ids_stride1, recv_topk_weight, recv_topk_weight_stride0, recv_topk_weight_stride1, input_index, input_index_stride0, input_index_stride1, output_tensor, output_tensor_stride0, output_tensor_stride1, topk_num, expert_map`
- **missing in shapes.yaml** (16): `total_token_num, input_tensor_stride0, input_tensor_stride1, recv_topk_ids, recv_topk_ids_stride0, recv_topk_ids_stride1, recv_topk_weight, recv_topk_weight_stride0, recv_topk_weight_stride1, input_index_stride0, input_index_stride1, output_tensor, output_tensor_stride0, output_tensor_stride1, topk_num, expert_map`
- **shapes.yaml declared inputs** (4): `input_index, input_tensor, recv_x, recv_x_scale`
- **fix recipe**:
  - Inspect the exception's call site in upstream
  - Decide: shim, mock, or rewrite oracle_pytorch.py standalone
- **effort**: `M`

#### `L2_26_fwd_kernel_ep_scatter_1`

- **aggregate status**: `BLOCKED(no-verified-upstream-entrypoint: upstream-not-invocable-here)`
- **intended kernel**: `ep_scatter`
- **intended substatus**: `upstream-not-invocable-here`
- **upstream exception**: `ExprError` (expression 'BN': unknown symbol 'BN' (allowed: ['B', 'C', 'D', 'Dv', 'E', 'G', 'H', 'L', 'N', 'T')
- **root cause category**: `upstream-exprerror`
- **root cause detail**: Upstream raises ExprError: expression 'BN': unknown symbol 'BN' (allowed: ['B', 'C', 'D', 'Dv', 'E', 'G', 'H', 'L', 'N', 'T'
- **kernel required params** (4): `num_recv_tokens_per_expert, expert_start_loc, m_indices, num_experts`
- **missing in shapes.yaml** (4): `num_recv_tokens_per_expert, expert_start_loc, m_indices, num_experts`
- **shapes.yaml declared inputs** (5): `dst_index, input_tensor, out_rows, recv_x, recv_x_scale`
- **fix recipe**:
  - Inspect the exception's call site in upstream
  - Decide: shim, mock, or rewrite oracle_pytorch.py standalone
- **effort**: `M`

#### `L2_27_deepep_permute_triton`

- **aggregate status**: `BLOCKED(no-verified-upstream-entrypoint: upstream-not-invocable-here)`
- **intended kernel**: `deepep_permute_triton_kernel`
- **upstream exception**: `ExprError` (expression 'DIM3': unknown symbol 'DIM3' (allowed: ['B', 'C', 'D', 'Dv', 'E', 'G', 'H', 'L', 'N', 'T')
- **root cause category**: `meta-resolver-mismatch`
- **root cause detail**: Resolver candidate-search did not try the meta-declared kernel `deepep_permute_triton_kernel`. Likely meta.upstream.path / meta.oracle.gpu_triton points at a stale location, OR the kernel.py vendored function is named differently than meta records. Resolver tried instead: ['fp8_per_token_to_per_tensor_quant_triton', 'post_reorder_for_cutlass_moe', 'pre_reorder_for_cutlass_moe', 'silu_and_mul_masked_post_per_tensor_quant_fwd']
- **kernel required params** (2): `topk, hidden_size`
- **missing in shapes.yaml** (1): `hidden_size`
- **shapes.yaml declared inputs** (9): `down_output, input, masked_m, output, problem_sizes1, src2dst, topk, x, x_scale`
- **fix recipe**:
  - Verify `deepep_permute_triton_kernel` exists in kernel.py (grep `def deepep_permute_triton_kernel`)
  - If not, find the actual entrypoint name in kernel.py and update meta.name
  - If yes, audit scripts/resolve_upstream_entrypoints.py candidate-search to understand why this kernel was not tried
  - Then add shapes.yaml inputs matching `deepep_permute_triton_kernel`'s signature
- **effort**: `M`

#### `L2_28_deepep_post_reorder_triton`

- **aggregate status**: `BLOCKED(no-verified-upstream-entrypoint: upstream-not-invocable-here)`
- **intended kernel**: `deepep_post_reorder_triton_kernel`
- **upstream exception**: `ExprError` (expression 'DIM3': unknown symbol 'DIM3' (allowed: ['B', 'C', 'D', 'Dv', 'E', 'G', 'H', 'L', 'N', 'T')
- **root cause category**: `meta-resolver-mismatch`
- **root cause detail**: Resolver candidate-search did not try the meta-declared kernel `deepep_post_reorder_triton_kernel`. Likely meta.upstream.path / meta.oracle.gpu_triton points at a stale location, OR the kernel.py vendored function is named differently than meta records. Resolver tried instead: ['post_reorder_for_cutlass_moe', 'fp8_per_token_to_per_tensor_quant_triton', 'pre_reorder_for_cutlass_moe', 'silu_and_mul_masked_post_per_tensor_quant_fwd']
- **kernel required params** (2): `topk, hidden_size`
- **missing in shapes.yaml** (1): `hidden_size`
- **shapes.yaml declared inputs** (10): `down_output, input, masked_m, output, problem_sizes1, src2dst, topk, topk_weights, x, x_scale`
- **fix recipe**:
  - Verify `deepep_post_reorder_triton_kernel` exists in kernel.py (grep `def deepep_post_reorder_triton_kernel`)
  - If not, find the actual entrypoint name in kernel.py and update meta.name
  - If yes, audit scripts/resolve_upstream_entrypoints.py candidate-search to understand why this kernel was not tried
  - Then add shapes.yaml inputs matching `deepep_post_reorder_triton_kernel`'s signature
- **effort**: `M`

#### `L2_29_fused_moe_router_cudacore`

- **aggregate status**: `BLOCKED(no-verified-upstream-entrypoint: shapes-incomplete)`
- **intended kernel**: `fused_moe_router_cudacore`
- **intended substatus**: `oracle-mismatch`
- **upstream exception**: `anchored` (on 1/2 shape(s) ['H2')
- **root cause category**: `oracle-numeric-tolerance`
- **root cause detail**: Numeric divergence between kernel and oracle under the task's tolerance class. Likely CPU interpreter precision drift but may be real kernel bug.
- **kernel required params** (6): `is_correction_bias, num_experts, topk, moe_softcapping, moe_renormalize, hidden_dim`
- **missing in shapes.yaml** (4): `is_correction_bias, num_experts, moe_renormalize, hidden_dim`
- **shapes.yaml declared inputs** (7): `BLOCK_SIZE_M, BLOCK_SIZE_N, correction_bias, moe_softcapping, router_weight, topk, x`
- **fix recipe**:
  - Re-run on GPU to distinguish interpreter drift from real divergence
  - If real, audit oracle_pytorch.py against kernel.py mathematically
- **effort**: `M`

#### `L2_30_fused_moe_router_tensorcore`

- **aggregate status**: `BLOCKED(no-verified-upstream-entrypoint: shapes-incomplete)`
- **intended kernel**: `fused_moe_router_tensorcore`
- **intended substatus**: `shapes-incomplete`
- **upstream exception**: `anchored` (on 1/2 shape(s) ['H2')
- **root cause category**: `missing-shape-row`
- **root cause detail**: Resolver cache is STALE: it claims `BLOCK_SIZE_K` missing but shapes.yaml already declares it. Current real gap: kernel `fused_moe_router_tensorcore` requires params ['bs', 'num_experts', 'moe_renormalize', 'is_correction_bias', 'dp_attn_workaround_flag'] that shapes.yaml does not declare.
- **kernel required params** (7): `bs, num_experts, topk, moe_softcapping, moe_renormalize, is_correction_bias, dp_attn_workaround_flag`
- **missing in shapes.yaml** (5): `bs, num_experts, moe_renormalize, is_correction_bias, dp_attn_workaround_flag`
- **shapes.yaml declared inputs** (8): `BLOCK_SIZE_K, BLOCK_SIZE_M, BLOCK_SIZE_N, correction_bias, moe_softcapping, router_weight, topk, x`
- **fix recipe**:
  - First: refresh resolver cache (resolve --refresh-cache --task <this>)
  - Then add the still-missing params to shapes.yaml: ['bs', 'num_experts', 'moe_renormalize', 'is_correction_bias', 'dp_attn_workaround_flag']
- **effort**: `M`

#### `L2_31_moe_align_block_size_stage2`

- **aggregate status**: `BLOCKED(no-verified-upstream-entrypoint: shapes-incomplete)`
- **intended kernel**: `moe_align_block_size_stage2`
- **upstream exception**: `anchored` (on 1/2 shape(s) ['H2')
- **root cause category**: `meta-resolver-mismatch`
- **root cause detail**: Resolver candidate-search did not try the meta-declared kernel `moe_align_block_size_stage2`. Likely meta.upstream.path / meta.oracle.gpu_triton points at a stale location, OR the kernel.py vendored function is named differently than meta records. Resolver tried instead: ['moe_align_block_size_triton']
- **kernel required params** (1): `num_experts`
- **shapes.yaml declared inputs** (3): `block_size, num_experts, topk_ids`
- **fix recipe**:
  - Verify `moe_align_block_size_stage2` exists in kernel.py (grep `def moe_align_block_size_stage2`)
  - If not, find the actual entrypoint name in kernel.py and update meta.name
  - If yes, audit scripts/resolve_upstream_entrypoints.py candidate-search to understand why this kernel was not tried
  - Then add shapes.yaml inputs matching `moe_align_block_size_stage2`'s signature
- **effort**: `M`

#### `L2_32_experts_combine`

- **aggregate status**: `BLOCKED(no-verified-upstream-entrypoint: shapes-incomplete)`
- **intended kernel**: `experts_combine_kernel`
- **upstream exception**: `RuntimeError` (sglang::experts_combine_triton() is missing value for argument 'moe_hidden_states'. Declaration: sglang::experts_combine)
- **root cause category**: `meta-resolver-mismatch`
- **root cause detail**: Resolver candidate-search did not try the meta-declared kernel `experts_combine_kernel`. Likely meta.upstream.path / meta.oracle.gpu_triton points at a stale location, OR the kernel.py vendored function is named differently than meta records. Resolver tried instead: ['experts_combine_triton', 'fused_softcap', 'gelu_and_mul_triton', 'silu_and_mul_triton']
- **kernel required params** (5): `out_hidden_states, moe_hidden_states, mlp_hidden_states, combine_k, hidden_dim`
- **missing in shapes.yaml** (3): `out_hidden_states, combine_k, hidden_dim`
- **shapes.yaml declared inputs** (6): `hidden_states, mlp_hidden_states, moe_hidden_states, residual, softcap_const, x`
- **fix recipe**:
  - Verify `experts_combine_kernel` exists in kernel.py (grep `def experts_combine_kernel`)
  - If not, find the actual entrypoint name in kernel.py and update meta.name
  - If yes, audit scripts/resolve_upstream_entrypoints.py candidate-search to understand why this kernel was not tried
  - Then add shapes.yaml inputs matching `experts_combine_kernel`'s signature
- **effort**: `M`

#### `L2_34_silu_mul_static_tensorwise_quant_triton_kernel_for_cutlass_moe`

- **aggregate status**: `BLOCKED(no-verified-upstream-entrypoint: upstream-not-invocable-here)`
- **intended kernel**: `silu_mul_static_tensorwise_quant_triton_kernel_for_cutlass_moe`
- **upstream exception**: `ExprError` (expression 'DIM3': unknown symbol 'DIM3' (allowed: ['B', 'C', 'D', 'Dv', 'E', 'G', 'H', 'L', 'N', 'T')
- **root cause category**: `meta-resolver-mismatch`
- **root cause detail**: Resolver candidate-search did not try the meta-declared kernel `silu_mul_static_tensorwise_quant_triton_kernel_for_cutlass_moe`. Likely meta.upstream.path / meta.oracle.gpu_triton points at a stale location, OR the kernel.py vendored function is named differently than meta records. Resolver tried instead: ['silu_and_mul_masked_post_per_tensor_quant_fwd', 'silu_mul_static_tensorwise_quant_for_cutlass_moe', 'fp8_per_token_to_per_tensor_quant_triton', 'silu_and_mul_masked_post_quant_fwd']
- **kernel required params** (1): `intermediate_size`
- **missing in shapes.yaml** (1): `intermediate_size`
- **shapes.yaml declared inputs** (7): `down_output, input, masked_m, output, problem_sizes1, scale, x`
- **fix recipe**:
  - Verify `silu_mul_static_tensorwise_quant_triton_kernel_for_cutlass_moe` exists in kernel.py (grep `def silu_mul_static_tensorwise_quant_triton_kernel_for_cutlass_moe`)
  - If not, find the actual entrypoint name in kernel.py and update meta.name
  - If yes, audit scripts/resolve_upstream_entrypoints.py candidate-search to understand why this kernel was not tried
  - Then add shapes.yaml inputs matching `silu_mul_static_tensorwise_quant_triton_kernel_for_cutlass_moe`'s signature
- **effort**: `M`

#### `L2_35_pre_reorder_triton_kernel_for_cutlass_moe`

- **aggregate status**: `BLOCKED(no-verified-upstream-entrypoint: shapes-incomplete)`
- **intended kernel**: `pre_reorder_triton_kernel_for_cutlass_moe`
- **upstream exception**: `anchored` (on 1/2 shape(s) ['H2')
- **root cause category**: `meta-resolver-mismatch`
- **root cause detail**: Resolver candidate-search did not try the meta-declared kernel `pre_reorder_triton_kernel_for_cutlass_moe`. Likely meta.upstream.path / meta.oracle.gpu_triton points at a stale location, OR the kernel.py vendored function is named differently than meta records. Resolver tried instead: ['pre_reorder_for_cutlass_moe', 'post_reorder_for_cutlass_moe', 'fp8_per_token_to_per_tensor_quant_triton', 'silu_and_mul_masked_post_per_tensor_quant_fwd']
- **kernel required params** (4): `num_local_experts, topk, num_tokens, hidden_size`
- **missing in shapes.yaml** (3): `num_local_experts, num_tokens, hidden_size`
- **shapes.yaml declared inputs** (7): `down_output, input, masked_m, output, src2dst, topk, x`
- **fix recipe**:
  - Verify `pre_reorder_triton_kernel_for_cutlass_moe` exists in kernel.py (grep `def pre_reorder_triton_kernel_for_cutlass_moe`)
  - If not, find the actual entrypoint name in kernel.py and update meta.name
  - If yes, audit scripts/resolve_upstream_entrypoints.py candidate-search to understand why this kernel was not tried
  - Then add shapes.yaml inputs matching `pre_reorder_triton_kernel_for_cutlass_moe`'s signature
- **effort**: `M`

#### `L2_36_post_reorder_triton_kernel_for_cutlass_moe`

- **aggregate status**: `BLOCKED(no-verified-upstream-entrypoint: shapes-incomplete)`
- **intended kernel**: `post_reorder_triton_kernel_for_cutlass_moe`
- **upstream exception**: `anchored` (on 1/2 shape(s) ['H2')
- **root cause category**: `meta-resolver-mismatch`
- **root cause detail**: Resolver candidate-search did not try the meta-declared kernel `post_reorder_triton_kernel_for_cutlass_moe`. Likely meta.upstream.path / meta.oracle.gpu_triton points at a stale location, OR the kernel.py vendored function is named differently than meta records. Resolver tried instead: ['post_reorder_for_cutlass_moe', 'fp8_per_token_to_per_tensor_quant_triton', 'pre_reorder_for_cutlass_moe', 'silu_and_mul_masked_post_per_tensor_quant_fwd']
- **kernel required params** (5): `num_local_experts, topk, num_tokens, hidden_size, routed_scaling_factor`
- **missing in shapes.yaml** (4): `num_local_experts, num_tokens, hidden_size, routed_scaling_factor`
- **shapes.yaml declared inputs** (9): `down_output, input, input_tensor, masked_m, output, src2dst, topk, topk_weights, x`
- **fix recipe**:
  - Verify `post_reorder_triton_kernel_for_cutlass_moe` exists in kernel.py (grep `def post_reorder_triton_kernel_for_cutlass_moe`)
  - If not, find the actual entrypoint name in kernel.py and update meta.name
  - If yes, audit scripts/resolve_upstream_entrypoints.py candidate-search to understand why this kernel was not tried
  - Then add shapes.yaml inputs matching `post_reorder_triton_kernel_for_cutlass_moe`'s signature
- **effort**: `M`

#### `L3_09_fused_moe`

- **aggregate status**: `BLOCKED(no-verified-upstream-entrypoint: upstream-not-invocable-here)`
- **intended kernel**: `fused_moe_kernel`
- **upstream exception**: `RuntimeError` (Tried to register an operator (vllm::inplace_fused_experts(Tensor(a0!) hidden_states, Tensor w1, Tensor w2, Tensor topk_)
- **root cause category**: `meta-resolver-mismatch`
- **root cause detail**: Resolver candidate-search did not try the meta-declared kernel `fused_moe_kernel`. Likely meta.upstream.path / meta.oracle.gpu_triton points at a stale location, OR the kernel.py vendored function is named differently than meta records. Resolver tried instead: ['invoke_fused_moe_triton_kernel', 'invoke_fused_moe_wna16_triton_kernel', 'zero_experts_compute_triton']
- **kernel required params** (10): `num_valid_tokens, group_n, group_k, naive_block_assignment, top_k, compute_type, use_fp8_w8a8, use_int8_w8a8, use_int8_w8a16, per_channel_quant`
- **missing in shapes.yaml** (10): `num_valid_tokens, group_n, group_k, naive_block_assignment, top_k, compute_type, use_fp8_w8a8, use_int8_w8a8, use_int8_w8a16, per_channel_quant`
- **shapes.yaml declared inputs** (7): `a, apply_weight, b, block_size, expert_ids, sorted_token_ids, topk_weights`
- **fix recipe**:
  - Verify `fused_moe_kernel` exists in kernel.py (grep `def fused_moe_kernel`)
  - If not, find the actual entrypoint name in kernel.py and update meta.name
  - If yes, audit scripts/resolve_upstream_entrypoints.py candidate-search to understand why this kernel was not tried
  - Then add shapes.yaml inputs matching `fused_moe_kernel`'s signature
- **effort**: `M`

#### `L3_10_fused_moe_kernel_gptq_awq`

- **aggregate status**: `BLOCKED(no-verified-upstream-entrypoint: upstream-not-invocable-here)`
- **intended kernel**: `fused_moe_kernel_gptq_awq`
- **upstream exception**: `RuntimeError` (Tried to register an operator (vllm::inplace_fused_experts(Tensor(a0!) hidden_states, Tensor w1, Tensor w2, Tensor topk_)
- **root cause category**: `meta-resolver-mismatch`
- **root cause detail**: Resolver candidate-search did not try the meta-declared kernel `fused_moe_kernel_gptq_awq`. Likely meta.upstream.path / meta.oracle.gpu_triton points at a stale location, OR the kernel.py vendored function is named differently than meta records. Resolver tried instead: ['invoke_fused_moe_wna16_triton_kernel', 'invoke_fused_moe_triton_kernel', 'zero_experts_compute_triton']
- **kernel required params** (8): `num_valid_tokens, block_k_diviable, group_size, top_k, compute_type, has_zp, use_int4_w4a16, use_int8_w8a16`
- **missing in shapes.yaml** (8): `num_valid_tokens, block_k_diviable, group_size, top_k, compute_type, has_zp, use_int4_w4a16, use_int8_w8a16`
- **shapes.yaml declared inputs** (7): `a, apply_weight, b, block_size, expert_ids, sorted_token_ids, topk_weights`
- **fix recipe**:
  - Verify `fused_moe_kernel_gptq_awq` exists in kernel.py (grep `def fused_moe_kernel_gptq_awq`)
  - If not, find the actual entrypoint name in kernel.py and update meta.name
  - If yes, audit scripts/resolve_upstream_entrypoints.py candidate-search to understand why this kernel was not tried
  - Then add shapes.yaml inputs matching `fused_moe_kernel_gptq_awq`'s signature
- **effort**: `M`

#### `L3_11_expert_triton`

- **aggregate status**: `BLOCKED(no-verified-upstream-entrypoint: shapes-incomplete)`
- **intended kernel**: `expert_triton_kernel`
- **upstream exception**: `anchored` (on 1/2 shape(s) ['H2')
- **root cause category**: `meta-resolver-mismatch`
- **root cause detail**: Resolver candidate-search did not try the meta-declared kernel `expert_triton_kernel`. Likely meta.upstream.path / meta.oracle.gpu_triton points at a stale location, OR the kernel.py vendored function is named differently than meta records. Resolver tried instead: ['invoke_moe_batched_triton_kernel']
- **kernel required params** (8): `expert_id, compute_type, offs_bn, group_n, group_k, use_fp8_w8a8, use_int8_w8a16, per_act_token_quant`
- **missing in shapes.yaml** (8): `expert_id, compute_type, offs_bn, group_n, group_k, use_fp8_w8a8, use_int8_w8a16, per_act_token_quant`
- **shapes.yaml declared inputs** (3): `a, b, expert_num_tokens`
- **fix recipe**:
  - Verify `expert_triton_kernel` exists in kernel.py (grep `def expert_triton_kernel`)
  - If not, find the actual entrypoint name in kernel.py and update meta.name
  - If yes, audit scripts/resolve_upstream_entrypoints.py candidate-search to understand why this kernel was not tried
  - Then add shapes.yaml inputs matching `expert_triton_kernel`'s signature
- **effort**: `M`

#### `L3_12_moe_mmk`

- **aggregate status**: `BLOCKED(no-verified-upstream-entrypoint: shapes-incomplete)`
- **intended kernel**: `moe_mmk`
- **upstream exception**: `anchored` (on 1/2 shape(s) ['H2')
- **root cause category**: `meta-resolver-mismatch`
- **root cause detail**: Resolver candidate-search did not try the meta-declared kernel `moe_mmk`. Likely meta.upstream.path / meta.oracle.gpu_triton points at a stale location, OR the kernel.py vendored function is named differently than meta records. Resolver tried instead: ['invoke_moe_batched_triton_kernel']
- **kernel required params** (13): `a_ptrs, b_ptrs, expert_id, offs_m, offs_n, offs_bn, mask_m, group_n, group_k, compute_type, use_w8a8, use_w8a16, per_act_token_quant`
- **missing in shapes.yaml** (13): `a_ptrs, b_ptrs, expert_id, offs_m, offs_n, offs_bn, mask_m, group_n, group_k, compute_type, use_w8a8, use_w8a16, per_act_token_quant`
- **shapes.yaml declared inputs** (3): `a, b, expert_num_tokens`
- **fix recipe**:
  - Verify `moe_mmk` exists in kernel.py (grep `def moe_mmk`)
  - If not, find the actual entrypoint name in kernel.py and update meta.name
  - If yes, audit scripts/resolve_upstream_entrypoints.py candidate-search to understand why this kernel was not tried
  - Then add shapes.yaml inputs matching `moe_mmk`'s signature
- **effort**: `M`

### family: `norm` (n=1)

#### `L1_12_rms_norm`

- **aggregate status**: `UNRUN(needs-gpu)`
- **root cause category**: `needs-gpu`
- **root cause detail**: Kernel binds successfully on dev; oracle is consistent; Phase A' verification requires GPU/NPU execution.
- **fix recipe**:
  - Run on GPU host: bash scripts/automation/runners/tab_quickstart.sh verify-all
  - Expected outcome: PASS after Phase A' completes
- **effort**: `S`

### family: `normalization` (n=5)

#### `L1_13_layer_norm_fwd`

- **aggregate status**: `BLOCKED(no-verified-upstream-entrypoint: upstream-not-invocable-here)`
- **intended kernel**: `layer_norm_fwd`
- **intended substatus**: `upstream-not-invocable-here`
- **upstream exception**: `NotImplementedError` (num_compute_units is not implemented for the current platform.)
- **root cause category**: `platform-stub-noimpl`
- **root cause detail**: Upstream calls vllm.platforms.current_platform.num_compute_units() which raises NotImplementedError on the dev host's stub platform.
- **kernel required params** (3): `Mean, Rstd, eps`
- **missing in shapes.yaml** (2): `Mean, Rstd`
- **shapes.yaml declared inputs** (4): `bias, eps, weight, x`
- **fix recipe**:
  - Option A: mock num_compute_units in tests/conftest.py or shim layer
  - Option B: rewrite oracle_pytorch.py to not depend on platform helpers
- **effort**: `M`

#### `L1_14_layer_norm_fwd_1pass`

- **aggregate status**: `BLOCKED(no-verified-upstream-entrypoint: oracle-mismatch)`
- **intended kernel**: `_layer_norm_fwd`
- **intended substatus**: `oracle-mismatch`
- **upstream exception**: `anchored` (on 1/2 shape(s) ['P1')
- **root cause category**: `oracle-structure-mismatch`
- **root cause detail**: Kernel output and oracle disagree on output structure (shape, dtype, or operator semantics).
- **kernel required params** (3): `Mean, Rstd, eps`
- **missing in shapes.yaml** (2): `Mean, Rstd`
- **shapes.yaml declared inputs** (5): `bias, eps, residual, weight, x`
- **fix recipe**:
  - Read kernel.py + oracle_pytorch.py side by side
  - Identify which output dim/dtype the oracle expects vs what kernel produces
  - Fix oracle_pytorch.py to match the kernel's true output contract
- **effort**: `M`

#### `L2_15_fused_dual_residual_rmsnorm`

- **aggregate status**: `BLOCKED(no-verified-upstream-entrypoint: shapes-incomplete)`
- **intended kernel**: `fused_dual_residual_rmsnorm`
- **intended substatus**: `shapes-incomplete`
- **upstream exception**: `anchored` (on 1/2 shape(s) ['P1')
- **root cause category**: `missing-shape-row`
- **root cause detail**: Resolver cache is STALE: it claims `weight2` missing but shapes.yaml already declares it. Current real gap: kernel `fused_dual_residual_rmsnorm` requires params ['hidden_dim'] that shapes.yaml does not declare.
- **kernel required params** (2): `eps, hidden_dim`
- **missing in shapes.yaml** (1): `hidden_dim`
- **shapes.yaml declared inputs** (10): `eps, hidden_states, residual, residual_a, residual_b, softcap_const, weight, weight1, weight2, x`
- **fix recipe**:
  - First: refresh resolver cache (resolve --refresh-cache --task <this>)
  - Then add the still-missing params to shapes.yaml: ['hidden_dim']
- **effort**: `M`

#### `L2_16_fused_rmsnorm`

- **aggregate status**: `BLOCKED(no-verified-upstream-entrypoint: oracle-mismatch)`
- **intended kernel**: `fused_rmsnorm`
- **intended substatus**: `oracle-mismatch`
- **upstream exception**: `anchored` (on 1/2 shape(s) ['P1')
- **root cause category**: `oracle-dtype-mismatch`
- **root cause detail**: Output dtype divergence between kernel and oracle.
- **kernel required params** (2): `eps, hidden_dim`
- **missing in shapes.yaml** (1): `hidden_dim`
- **shapes.yaml declared inputs** (8): `eps, hidden_states, residual, softcap_const, weight, weight1, weight2, x`
- **fix recipe**:
  - Align oracle_pytorch.py output dtype with kernel.py's emitted dtype
- **effort**: `S`

#### `L2_17_layer_norm_gated_fwd`

- **aggregate status**: `BLOCKED(no-verified-upstream-entrypoint: upstream-not-invocable-here)`
- **intended kernel**: `layer_norm_gated_fwd`
- **intended substatus**: `upstream-not-invocable-here`
- **upstream exception**: `AttributeError` ('_AutotunedKernel' object has no attribute 'arg_names')
- **root cause category**: `triton-autotune-api-mismatch`
- **root cause detail**: Kernel applies @triton.heuristics on top of a wrapper that lacks .arg_names. Current Triton API expects different ordering.
- **kernel required params** (10): `x, g, y, w, b, residual, residual_out, mean, rstd, eps`
- **missing in shapes.yaml** (4): `y, residual_out, mean, rstd`
- **shapes.yaml declared inputs** (8): `b, eps, g, k, q, residual, w, x`
- **fix recipe**:
  - Pin Triton version OR rewrite kernel.py decorator ordering
  - If oracle_pytorch.py is independent, defer kernel.py fix to Phase B
- **effort**: `L`

### family: `other` (n=4)

#### `L1_28_pack_bitmatrix`

- **aggregate status**: `BLOCKED(no-verified-upstream-entrypoint: shapes-incomplete)`
- **intended kernel**: `pack_bitmatrix`
- **upstream exception**: `anchored` (on 1/2 shape(s) ['H2')
- **root cause category**: `meta-resolver-mismatch`
- **root cause detail**: Resolver candidate-search did not try the meta-declared kernel `pack_bitmatrix`. Likely meta.upstream.path / meta.oracle.gpu_triton points at a stale location, OR the kernel.py vendored function is named differently than meta records. Resolver tried instead: ['make_routing_data']
- **kernel required params** (5): `bitmatrix, topk_ids, n_rows, bm_cols, n_expts_act`
- **missing in shapes.yaml** (4): `bitmatrix, n_rows, bm_cols, n_expts_act`
- **shapes.yaml declared inputs** (2): `num_experts, topk_ids`
- **fix recipe**:
  - Verify `pack_bitmatrix` exists in kernel.py (grep `def pack_bitmatrix`)
  - If not, find the actual entrypoint name in kernel.py and update meta.name
  - If yes, audit scripts/resolve_upstream_entrypoints.py candidate-search to understand why this kernel was not tried
  - Then add shapes.yaml inputs matching `pack_bitmatrix`'s signature
- **effort**: `M`

#### `L3_05_fwd_kernel_alibi`

- **aggregate status**: `BLOCKED(no-verified-upstream-entrypoint: shapes-incomplete)`
- **intended kernel**: `_fwd_kernel_alibi`
- **upstream exception**: `anchored` (on 1/2 shape(s) ['H2')
- **root cause category**: `meta-resolver-mismatch`
- **root cause detail**: Resolver candidate-search did not try the meta-declared kernel `_fwd_kernel_alibi`. Likely meta.upstream.path / meta.oracle.gpu_triton points at a stale location, OR the kernel.py vendored function is named differently than meta records. Resolver tried instead: ['context_attention_fwd']
- **kernel required params** (13): `K_cache, V_cache, B_Loc, sm_scale, k_scale, v_scale, B_Start_Loc, B_Seqlen, Alibi_slopes, block_size, x, Out, num_queries_per_kv`
- **missing in shapes.yaml** (13): `K_cache, V_cache, B_Loc, sm_scale, k_scale, v_scale, B_Start_Loc, B_Seqlen, Alibi_slopes, block_size, x, Out, num_queries_per_kv`
- **shapes.yaml declared inputs** (6): `alibi_slopes, k, o, q, scale, v`
- **fix recipe**:
  - Verify `_fwd_kernel_alibi` exists in kernel.py (grep `def _fwd_kernel_alibi`)
  - If not, find the actual entrypoint name in kernel.py and update meta.name
  - If yes, audit scripts/resolve_upstream_entrypoints.py candidate-search to understand why this kernel was not tried
  - Then add shapes.yaml inputs matching `_fwd_kernel_alibi`'s signature
- **effort**: `M`

#### `L3_06_fwd_grouped_kernel_stage1`

- **aggregate status**: `BLOCKED(no-verified-upstream-entrypoint: shapes-incomplete)`
- **intended kernel**: `_fwd_grouped_kernel_stage1`
- **upstream exception**: `anchored` (on 1/2 shape(s) ['H2')
- **root cause category**: `meta-resolver-mismatch`
- **root cause detail**: Resolver candidate-search did not try the meta-declared kernel `_fwd_grouped_kernel_stage1`. Likely meta.upstream.path / meta.oracle.gpu_triton points at a stale location, OR the kernel.py vendored function is named differently than meta records. Resolver tried instead: ['_decode_grouped_att_m_fwd', '_decode_att_m_fwd', '_decode_softmax_reducev_fwd']
- **kernel required params** (13): `K_Buffer, V_Buffer, sm_scale, Req_to_tokens, B_Seqlen, Att_Out, k_scale, v_scale, kv_group_num, q_head_num, logit_cap, Lk, Lv`
- **missing in shapes.yaml** (13): `K_Buffer, V_Buffer, sm_scale, Req_to_tokens, B_Seqlen, Att_Out, k_scale, v_scale, kv_group_num, q_head_num, logit_cap, Lk, Lv`
- **shapes.yaml declared inputs** (5): `att_out, k_buffer, q, scale, v_buffer`
- **fix recipe**:
  - Verify `_fwd_grouped_kernel_stage1` exists in kernel.py (grep `def _fwd_grouped_kernel_stage1`)
  - If not, find the actual entrypoint name in kernel.py and update meta.name
  - If yes, audit scripts/resolve_upstream_entrypoints.py candidate-search to understand why this kernel was not tried
  - Then add shapes.yaml inputs matching `_fwd_grouped_kernel_stage1`'s signature
- **effort**: `M`

#### `L3_20_recompute_w_u_fwd`

- **aggregate status**: `BLOCKED(no-verified-upstream-entrypoint: upstream-not-invocable-here)`
- **intended kernel**: `recompute_w_u_fwd`
- **intended substatus**: `upstream-not-invocable-here`
- **upstream exception**: `AttributeError` ('_AutotunedKernel' object has no attribute 'arg_names')
- **root cause category**: `triton-autotune-api-mismatch`
- **root cause detail**: Kernel applies @triton.heuristics on top of a wrapper that lacks .arg_names. Current Triton API expects different ordering.
- **kernel required params** (9): `k, v, beta, w, u, g, cu_seqlens, chunk_indices, Hg`
- **missing in shapes.yaml** (5): `w, u, g, chunk_indices, Hg`
- **shapes.yaml declared inputs** (6): `A, beta, cu_seqlens, g_cumsum, k, v`
- **fix recipe**:
  - Pin Triton version OR rewrite kernel.py decorator ordering
  - If oracle_pytorch.py is independent, defer kernel.py fix to Phase B
- **effort**: `L`

### family: `quantization` (n=8)

#### `L1_07_dequantize_k_cache_fast`

- **aggregate status**: `BLOCKED(no-verified-upstream-entrypoint: shapes-incomplete)`
- **intended kernel**: `_dequantize_k_cache_fast`
- **intended substatus**: `shapes-incomplete`
- **upstream exception**: `anchored` (on 1/2 shape(s) ['H2')
- **root cause category**: `shapes-incomplete-sibling`
- **root cause detail**: Resolver's missing-param mention (`quant_k_cache`) belongs to a sibling candidate, not the intended kernel `_dequantize_k_cache_fast`. The aggregate shapes-incomplete masks the intended kernel's true status.
- **kernel required params** (4): `output_stride_0, input_nope_q_stride_0, input_nope_s_stride_0, input_rope_stride_0`
- **missing in shapes.yaml** (4): `output_stride_0, input_nope_q_stride_0, input_nope_s_stride_0, input_rope_stride_0`
- **shapes.yaml declared inputs** (4): `group_size, nope_q, nope_s, rope`
- **fix recipe**:
  - Verify intended kernel `_dequantize_k_cache_fast` actually exists in kernel.py
  - If `_dequantize_k_cache_fast` is correct, the resolver may be trying wrong siblings -- fix meta.upstream.path or meta.oracle.gpu_triton to point at the right symbol
- **effort**: `M`

#### `L1_09_quantize_k_cache_fast`

- **aggregate status**: `BLOCKED(no-verified-upstream-entrypoint: oracle-mismatch)`
- **intended kernel**: `_quantize_k_cache_fast`
- **intended substatus**: `upstream-not-invocable-here`
- **upstream exception**: `AssertionError` ()
- **root cause category**: `kernel-assertion`
- **root cause detail**: Kernel function `_quantize_k_cache_fast` hits an AssertionError when called with the current shapes: ``
- **kernel required params** (5): `output_nope_q_stride_0, output_nope_s_stride_0, output_rope_stride_0, k_nope_stride_0, k_rope_stride_0`
- **missing in shapes.yaml** (5): `output_nope_q_stride_0, output_nope_s_stride_0, output_rope_stride_0, k_nope_stride_0, k_rope_stride_0`
- **shapes.yaml declared inputs** (4): `fp8_max, group_size, k_nope, k_rope`
- **fix recipe**:
  - Read the asserted condition in kernel.py
  - Adjust shapes.yaml so the input tensors satisfy the assertion
  - (e.g. paired dtype, divisibility, shape compatibility)
- **effort**: `M`

#### `L1_10_per_token_group_quant_int8`

- **aggregate status**: `BLOCKED(no-verified-upstream-entrypoint: shapes-incomplete)`
- **intended kernel**: `per_token_group_quant_int8`
- **intended substatus**: `oracle-mismatch`
- **upstream exception**: `anchored` (on 1/2 shape(s) ['H2')
- **root cause category**: `oracle-numeric-tolerance`
- **root cause detail**: Numeric divergence between kernel and oracle under the task's tolerance class. Likely CPU interpreter precision drift but may be real kernel bug.
- **kernel required params** (4): `y_stride, eps, int8_min, int8_max`
- **missing in shapes.yaml** (3): `y_stride, int8_min, int8_max`
- **shapes.yaml declared inputs** (3): `eps, group_size, x`
- **fix recipe**:
  - Re-run on GPU to distinguish interpreter drift from real divergence
  - If real, audit oracle_pytorch.py against kernel.py mathematically
- **effort**: `M`

#### `L1_11_awq_dequantize`

- **aggregate status**: `BLOCKED(no-verified-upstream-entrypoint: shapes-incomplete)`
- **intended kernel**: `awq_dequantize_kernel`
- **upstream exception**: `AssertionError` (| check_runner.py", line 370, in _check_one_shape
    cand_ret = fn(dict(kernel_inputs)) if sig is None else fn(**kwargs)
- **root cause category**: `meta-resolver-mismatch`
- **root cause detail**: Resolver candidate-search did not try the meta-declared kernel `awq_dequantize_kernel`. Likely meta.upstream.path / meta.oracle.gpu_triton points at a stale location, OR the kernel.py vendored function is named differently than meta records. Resolver tried instead: ['awq_dequantize_triton', 'awq_gemm_triton']
- **kernel required params** (3): `group_size, num_cols, num_rows`
- **missing in shapes.yaml** (2): `num_cols, num_rows`
- **shapes.yaml declared inputs** (5): `group_size, input, qweight, scales, zeros`
- **fix recipe**:
  - Verify `awq_dequantize_kernel` exists in kernel.py (grep `def awq_dequantize_kernel`)
  - If not, find the actual entrypoint name in kernel.py and update meta.name
  - If yes, audit scripts/resolve_upstream_entrypoints.py candidate-search to understand why this kernel was not tried
  - Then add shapes.yaml inputs matching `awq_dequantize_kernel`'s signature
- **effort**: `M`

#### `L1_30_act_quant`

- **aggregate status**: `BLOCKED(no-verified-upstream-entrypoint: shapes-incomplete)`
- **intended kernel**: `act_quant`
- **intended substatus**: `oracle-mismatch`
- **upstream exception**: `anchored` (on 1/2 shape(s) ['H2')
- **root cause category**: `oracle-numeric-tolerance`
- **root cause detail**: Numeric divergence between kernel and oracle under the task's tolerance class. Likely CPU interpreter precision drift but may be real kernel bug.
- **kernel required params** (2): `group_size, round_scale`
- **missing in shapes.yaml** (2): `group_size, round_scale`
- **shapes.yaml declared inputs** (2): `block_size, x`
- **fix recipe**:
  - Re-run on GPU to distinguish interpreter drift from real divergence
  - If real, audit oracle_pytorch.py against kernel.py mathematically
- **effort**: `M`

#### `L1_37_compute_problem_sizes_w4a8`

- **aggregate status**: `BLOCKED(no-verified-upstream-entrypoint: upstream-not-invocable-here)`
- **intended kernel**: `compute_problem_sizes_w4a8`
- **intended substatus**: `upstream-not-invocable-here`
- **upstream exception**: `ExprError` (expression 'DIM3': unknown symbol 'DIM3' (allowed: ['B', 'C', 'D', 'Dv', 'E', 'G', 'H', 'L', 'N', 'T')
- **root cause category**: `upstream-exprerror`
- **root cause detail**: Upstream raises ExprError: expression 'DIM3': unknown symbol 'DIM3' (allowed: ['B', 'C', 'D', 'Dv', 'E', 'G', 'H', 'L', 'N', 'T'
- **kernel required params** (3): `n, k, num_experts`
- **shapes.yaml declared inputs** (7): `input, k, masked_m, n, num_experts, problem_sizes1, x`
- **fix recipe**:
  - Inspect the exception's call site in upstream
  - Decide: shim, mock, or rewrite oracle_pytorch.py standalone
- **effort**: `M`

#### `L2_11_w8a8_triton_block_scaled_mm`

- **aggregate status**: `BLOCKED(no-verified-upstream-entrypoint: shapes-incomplete)`
- **intended kernel**: `w8a8_triton_block_scaled_mm`
- **intended substatus**: `shapes-incomplete`
- **upstream exception**: `anchored` (on 1/2 shape(s) ['H2')
- **root cause category**: `shapes-incomplete-sibling`
- **root cause detail**: Resolver's missing-param mention (`block_size`) belongs to a sibling candidate, not the intended kernel `w8a8_triton_block_scaled_mm`. The aggregate shapes-incomplete masks the intended kernel's true status.
- **kernel required params** (4): `As, Bs, group_n, group_k`
- **shapes.yaml declared inputs** (8): `A, As, B, Bs, group_k, group_n, input, x`
- **fix recipe**:
  - Verify intended kernel `w8a8_triton_block_scaled_mm` actually exists in kernel.py
  - If `w8a8_triton_block_scaled_mm` is correct, the resolver may be trying wrong siblings -- fix meta.upstream.path or meta.oracle.gpu_triton to point at the right symbol
- **effort**: `M`

#### `L2_37_silu_and_mul_post_quant`

- **aggregate status**: `BLOCKED(no-verified-upstream-entrypoint: upstream-not-invocable-here)`
- **intended kernel**: `_silu_and_mul_post_quant_kernel`
- **upstream exception**: `ExprError` (expression 'DIM3': unknown symbol 'DIM3' (allowed: ['B', 'C', 'D', 'Dv', 'E', 'G', 'H', 'L', 'N', 'T')
- **root cause category**: `meta-resolver-mismatch`
- **root cause detail**: Resolver candidate-search did not try the meta-declared kernel `_silu_and_mul_post_quant_kernel`. Likely meta.upstream.path / meta.oracle.gpu_triton points at a stale location, OR the kernel.py vendored function is named differently than meta records. Resolver tried instead: ['silu_and_mul_masked_post_quant_fwd', 'fp8_per_token_to_per_tensor_quant_triton', 'silu_and_mul_masked_post_per_tensor_quant_fwd', 'compute_problem_sizes_w4a8']
- **kernel required params** (3): `size_n, fp8_max, fp8_min`
- **missing in shapes.yaml** (2): `size_n, fp8_min`
- **shapes.yaml declared inputs** (7): `fp8_max, input, masked_m, output, problem_sizes1, quant_group_size, x`
- **fix recipe**:
  - Verify `_silu_and_mul_post_quant_kernel` exists in kernel.py (grep `def _silu_and_mul_post_quant_kernel`)
  - If not, find the actual entrypoint name in kernel.py and update meta.name
  - If yes, audit scripts/resolve_upstream_entrypoints.py candidate-search to understand why this kernel was not tried
  - Then add shapes.yaml inputs matching `_silu_and_mul_post_quant_kernel`'s signature
- **effort**: `M`

### family: `sampling` (n=8)

#### `L1_01_bias`

- **aggregate status**: `UNRUN(needs-gpu)`
- **root cause category**: `needs-gpu`
- **root cause detail**: Kernel binds successfully on dev; oracle is consistent; Phase A' verification requires GPU/NPU execution.
- **fix recipe**:
  - Run on GPU host: bash scripts/automation/runners/tab_quickstart.sh verify-all
  - Expected outcome: PASS after Phase A' completes
- **effort**: `S`

#### `L1_19_topk_log_softmax`

- **aggregate status**: `BLOCKED(no-verified-upstream-entrypoint: upstream-not-invocable-here)`
- **intended kernel**: `_topk_log_softmax_kernel`
- **upstream exception**: `ModuleNotFoundError` (No module named 'uvloop' | ng_params import MAX_LOGPROB_TOKEN_IDS, SamplingParams
  File "/data1/code/agentic-rl/kernel/)
- **root cause category**: `meta-resolver-mismatch`
- **root cause detail**: Resolver candidate-search did not try the meta-declared kernel `_topk_log_softmax_kernel`. Likely meta.upstream.path / meta.oracle.gpu_triton points at a stale location, OR the kernel.py vendored function is named differently than meta records. Resolver tried instead: ['compute_token_logprobs', 'compute_topk_logprobs']
- **kernel required params** (3): `logits_stride, topk, vocab_size`
- **missing in shapes.yaml** (3): `logits_stride, topk, vocab_size`
- **shapes.yaml declared inputs** (3): `logits, num_logprobs, token_ids`
- **fix recipe**:
  - Verify `_topk_log_softmax_kernel` exists in kernel.py (grep `def _topk_log_softmax_kernel`)
  - If not, find the actual entrypoint name in kernel.py and update meta.name
  - If yes, audit scripts/resolve_upstream_entrypoints.py candidate-search to understand why this kernel was not tried
  - Then add shapes.yaml inputs matching `_topk_log_softmax_kernel`'s signature
- **effort**: `M`

#### `L1_20_log_softmax`

- **aggregate status**: `BLOCKED(no-verified-upstream-entrypoint: shapes-incomplete)`
- **intended kernel**: `log_softmax`
- **intended substatus**: `oracle-mismatch`
- **upstream exception**: `anchored` (on 2/2 shape(s) ['H2', 'H4')
- **root cause category**: `oracle-numeric-tolerance`
- **root cause detail**: Numeric divergence between kernel and oracle under the task's tolerance class. Likely CPU interpreter precision drift but may be real kernel bug.
- **kernel required params** (3): `input_row_stride, output_row_stride, n_cols`
- **missing in shapes.yaml** (3): `input_row_stride, output_row_stride, n_cols`
- **shapes.yaml declared inputs** (2): `dim, input`
- **fix recipe**:
  - Re-run on GPU to distinguish interpreter drift from real divergence
  - If real, audit oracle_pytorch.py against kernel.py mathematically
- **effort**: `M`

#### `L1_21_gumbel_sample`

- **aggregate status**: `BLOCKED(no-verified-upstream-entrypoint: shapes-incomplete)`
- **intended kernel**: `gumbel_sample`
- **intended substatus**: `shapes-incomplete`
- **upstream exception**: `anchored` (on 1/2 shape(s) ['P1')
- **root cause category**: `shapes-incomplete-sibling`
- **root cause detail**: Resolver's missing-param mention (`logits`) belongs to a sibling candidate, not the intended kernel `gumbel_sample`. The aggregate shapes-incomplete masks the intended kernel's true status.
- **kernel required params** (5): `local_argmax_stride, local_max_stride, processed_logits_stride, logits_stride, vocab_size`
- **missing in shapes.yaml** (5): `local_argmax_stride, local_max_stride, processed_logits_stride, logits_stride, vocab_size`
- **fix recipe**:
  - Verify intended kernel `gumbel_sample` actually exists in kernel.py
  - If `gumbel_sample` is correct, the resolver may be trying wrong siblings -- fix meta.upstream.path or meta.oracle.gpu_triton to point at the right symbol
- **effort**: `M`

#### `L1_22_min_p`

- **aggregate status**: `BLOCKED(no-verified-upstream-entrypoint: oracle-mismatch)`
- **intended kernel**: `_min_p_kernel`
- **upstream exception**: `anchored` (on 1/2 shape(s) ['H2')
- **root cause category**: `meta-resolver-mismatch`
- **root cause detail**: Resolver candidate-search did not try the meta-declared kernel `_min_p_kernel`. Likely meta.upstream.path / meta.oracle.gpu_triton points at a stale location, OR the kernel.py vendored function is named differently than meta records. Resolver tried instead: ['apply_min_p']
- **kernel required params** (2): `logits_stride, vocab_size`
- **missing in shapes.yaml** (2): `logits_stride, vocab_size`
- **shapes.yaml declared inputs** (3): `expanded_idx_mapping, logits, min_p`
- **fix recipe**:
  - Verify `_min_p_kernel` exists in kernel.py (grep `def _min_p_kernel`)
  - If not, find the actual entrypoint name in kernel.py and update meta.name
  - If yes, audit scripts/resolve_upstream_entrypoints.py candidate-search to understand why this kernel was not tried
  - Then add shapes.yaml inputs matching `_min_p_kernel`'s signature
- **effort**: `M`

#### `L1_23_penalties_and_temperature`

- **aggregate status**: `BLOCKED(no-verified-upstream-entrypoint: upstream-not-invocable-here)`
- **intended kernel**: `_penalties_kernel`
- **upstream exception**: `ExprError` (expression 'V': unknown symbol 'V' (allowed: ['B', 'C', 'D', 'Dv', 'E', 'G', 'H', 'L', 'N', 'T')
- **root cause category**: `meta-resolver-mismatch`
- **root cause detail**: Resolver candidate-search did not try the meta-declared kernel `_penalties_kernel`. Likely meta.upstream.path / meta.oracle.gpu_triton points at a stale location, OR the kernel.py vendored function is named differently than meta records. Resolver tried instead: ['apply_penalties', 'bincount']
- **kernel required params** (4): `logits_stride, prompt_bin_mask_stride, output_bin_counts_stride, vocab_size`
- **missing in shapes.yaml** (4): `logits_stride, prompt_bin_mask_stride, output_bin_counts_stride, vocab_size`
- **shapes.yaml declared inputs** (9): `all_token_ids, expanded_idx_mapping, frequency_penalty, logits, output_token_counts, presence_penalty, prompt_bin_mask, repetition_penalty, token_ids`
- **fix recipe**:
  - Verify `_penalties_kernel` exists in kernel.py (grep `def _penalties_kernel`)
  - If not, find the actual entrypoint name in kernel.py and update meta.name
  - If yes, audit scripts/resolve_upstream_entrypoints.py candidate-search to understand why this kernel was not tried
  - Then add shapes.yaml inputs matching `_penalties_kernel`'s signature
- **effort**: `M`

#### `L1_24_rejection_sample`

- **aggregate status**: `BLOCKED(no-verified-upstream-entrypoint: upstream-not-invocable-here)`
- **intended kernel**: `rejection_random_sample_kernel`
- **upstream exception**: `ModuleNotFoundError` (No module named 'uvloop' | e>
    from vllm.sampling_params import SamplingParams
  File "/data1/code/agentic-rl/kernel/)
- **root cause category**: `meta-resolver-mismatch`
- **root cause detail**: Resolver candidate-search did not try the meta-declared kernel `rejection_random_sample_kernel`. Likely meta.upstream.path / meta.oracle.gpu_triton points at a stale location, OR the kernel.py vendored function is named differently than meta records. Resolver tried instead: ['rejection_sample', 'sample_recovered_tokens', 'expand_batch_to_tokens']
- **kernel required params** (2): `max_spec_len, vocab_size`
- **missing in shapes.yaml** (1): `vocab_size`
- **shapes.yaml declared inputs** (11): `bonus_token_ids, cu_num_draft_tokens, draft_probs, draft_token_ids, is_greedy, max_spec_len, num_draft_tokens, recovered_token_ids, target_probs, uniform_probs, x`
- **fix recipe**:
  - Verify `rejection_random_sample_kernel` exists in kernel.py (grep `def rejection_random_sample_kernel`)
  - If not, find the actual entrypoint name in kernel.py and update meta.name
  - If yes, audit scripts/resolve_upstream_entrypoints.py candidate-search to understand why this kernel was not tried
  - Then add shapes.yaml inputs matching `rejection_random_sample_kernel`'s signature
- **effort**: `M`

#### `L1_27_apply_grammar_bitmask`

- **aggregate status**: `BLOCKED(no-verified-upstream-entrypoint: upstream-not-invocable-here)`
- **intended kernel**: `_apply_grammar_bitmask_kernel`
- **intended substatus**: `upstream-not-invocable-here`
- **upstream exception**: `InterpreterError` (TypeError("_apply_grammar_bitmask_kernel() missing 7 required positional arguments: 'logits_ptr', 'logits_stride', 'logi)
- **root cause category**: `upstream-interpretererror`
- **root cause detail**: Upstream raises InterpreterError: TypeError("_apply_grammar_bitmask_kernel() missing 7 required positional arguments: 'logits_ptr', 'logits_stride', 'logits_indices_ptr', 'bitmask_ptr', 'bitmask_stride', 'vocab_size', and 'BLOCK_SIZE'
- **kernel required params** (3): `logits_stride, bitmask_stride, vocab_size`
- **missing in shapes.yaml** (3): `logits_stride, bitmask_stride, vocab_size`
- **shapes.yaml declared inputs** (3): `bitmask, logits, logits_indices`
- **fix recipe**:
  - Inspect the exception's call site in upstream
  - Decide: shim, mock, or rewrite oracle_pytorch.py standalone
- **effort**: `M`

### family: `spec_decode` (n=1)

#### `L1_29_prepare_eagle_inputs`

- **aggregate status**: `BLOCKED(no-verified-upstream-entrypoint: shapes-incomplete)`
- **intended kernel**: `prepare_eagle_inputs`
- **intended substatus**: `shapes-incomplete`
- **upstream exception**: `anchored` (on 1/2 shape(s) ['H2')
- **root cause category**: `shapes-incomplete-sibling`
- **root cause detail**: Resolver's missing-param mention (`input_buffers`) belongs to a sibling candidate, not the intended kernel `prepare_eagle_inputs`. The aggregate shapes-incomplete masks the intended kernel's true status.
- **kernel required params** (1): `max_num_reqs`
- **missing in shapes.yaml** (1): `max_num_reqs`
- **shapes.yaml declared inputs** (4): `current_draft_step, last_token_indices, num_rejected, num_sampled`
- **fix recipe**:
  - Verify intended kernel `prepare_eagle_inputs` actually exists in kernel.py
  - If `prepare_eagle_inputs` is correct, the resolver may be trying wrong siblings -- fix meta.upstream.path or meta.oracle.gpu_triton to point at the right symbol
- **effort**: `M`

### family: `ssm` (n=16)

#### `L1_39_state_passing_fwd`

- **aggregate status**: `BLOCKED(no-verified-upstream-entrypoint: shapes-incomplete)`
- **intended kernel**: `_state_passing_fwd`
- **intended substatus**: `shapes-incomplete`
- **upstream exception**: `anchored` (on 1/2 shape(s) ['H2')
- **root cause category**: `shapes-incomplete-sibling`
- **root cause detail**: Resolver's missing-param mention (`last_chunk_indices`) belongs to a sibling candidate, not the intended kernel `_state_passing_fwd`. The aggregate shapes-incomplete masks the intended kernel's true status.
- **kernel required params** (2): `dim, chunk_size`
- **missing in shapes.yaml** (2): `dim, chunk_size`
- **shapes.yaml declared inputs** (2): `dA_cumsum, states`
- **fix recipe**:
  - Verify intended kernel `_state_passing_fwd` actually exists in kernel.py
  - If `_state_passing_fwd` is correct, the resolver may be trying wrong siblings -- fix meta.upstream.path or meta.oracle.gpu_triton to point at the right symbol
- **effort**: `M`

#### `L1_40_chunk_cumsum_fwd`

- **aggregate status**: `BLOCKED(no-verified-upstream-entrypoint: shapes-incomplete)`
- **intended kernel**: `_chunk_cumsum_fwd`
- **intended substatus**: `shapes-incomplete`
- **upstream exception**: `anchored` (on 1/2 shape(s) ['H2')
- **root cause category**: `shapes-incomplete-sibling`
- **root cause detail**: Resolver's missing-param mention (`cu_chunk_seqlens`) belongs to a sibling candidate, not the intended kernel `_chunk_cumsum_fwd`. The aggregate shapes-incomplete masks the intended kernel's true status.
- **kernel required params** (5): `seqlen, nheads, chunk_size, dt_min, dt_max`
- **missing in shapes.yaml** (4): `seqlen, nheads, dt_min, dt_max`
- **shapes.yaml declared inputs** (4): `A, chunk_size, dt, dt_softplus`
- **fix recipe**:
  - Verify intended kernel `_chunk_cumsum_fwd` actually exists in kernel.py
  - If `_chunk_cumsum_fwd` is correct, the resolver may be trying wrong siblings -- fix meta.upstream.path or meta.oracle.gpu_triton to point at the right symbol
- **effort**: `M`

#### `L2_02_chunk_scan_fwd`

- **aggregate status**: `BLOCKED(no-verified-upstream-entrypoint: shapes-incomplete)`
- **intended kernel**: `_chunk_scan_fwd`
- **intended substatus**: `shapes-incomplete`
- **upstream exception**: `anchored` (on 1/2 shape(s) ['H2')
- **root cause category**: `shapes-incomplete-sibling`
- **root cause detail**: Resolver's missing-param mention (`cu_chunk_seqlens`) belongs to a sibling candidate, not the intended kernel `_chunk_scan_fwd`. The aggregate shapes-incomplete masks the intended kernel's true status.
- **kernel required params** (5): `chunk_size, hdim, dstate, seqlen, nheads_ngroups_ratio`
- **missing in shapes.yaml** (5): `chunk_size, hdim, dstate, seqlen, nheads_ngroups_ratio`
- **shapes.yaml declared inputs** (8): `C, D, cb, dA_cumsum, dt, states, x, z`
- **fix recipe**:
  - Verify intended kernel `_chunk_scan_fwd` actually exists in kernel.py
  - If `_chunk_scan_fwd` is correct, the resolver may be trying wrong siblings -- fix meta.upstream.path or meta.oracle.gpu_triton to point at the right symbol
- **effort**: `M`

#### `L2_03_chunk_state_fwd`

- **aggregate status**: `BLOCKED(no-verified-upstream-entrypoint: shapes-incomplete)`
- **intended kernel**: `_chunk_state_fwd`
- **intended substatus**: `shapes-incomplete`
- **upstream exception**: `anchored` (on 1/2 shape(s) ['H2')
- **root cause category**: `shapes-incomplete-sibling`
- **root cause detail**: Resolver's missing-param mention (`cu_chunk_seqlens`) belongs to a sibling candidate, not the intended kernel `_chunk_state_fwd`. The aggregate shapes-incomplete masks the intended kernel's true status.
- **kernel required params** (5): `hdim, dstate, chunk_size, seqlen, nheads_ngroups_ratio`
- **missing in shapes.yaml** (5): `hdim, dstate, chunk_size, seqlen, nheads_ngroups_ratio`
- **shapes.yaml declared inputs** (4): `B, dA_cumsum, dt, x`
- **fix recipe**:
  - Verify intended kernel `_chunk_state_fwd` actually exists in kernel.py
  - If `_chunk_state_fwd` is correct, the resolver may be trying wrong siblings -- fix meta.upstream.path or meta.oracle.gpu_triton to point at the right symbol
- **effort**: `M`

#### `L2_18_fused_gdn_gating`

- **aggregate status**: `BLOCKED(no-verified-upstream-entrypoint: upstream-not-invocable-here)`
- **intended kernel**: `fused_gdn_gating`
- **intended substatus**: `upstream-not-invocable-here`
- **upstream exception**: `AttributeError` ('_AutotunedKernel' object has no attribute 'arg_names')
- **root cause category**: `triton-autotune-api-mismatch`
- **root cause detail**: Kernel applies @triton.heuristics on top of a wrapper that lacks .arg_names. Current Triton API expects different ordering.
- **kernel required params** (9): `g, beta_output, A_log, a, b, dt_bias, seq_len, beta, threshold`
- **missing in shapes.yaml** (4): `g, beta_output, b, seq_len`
- **shapes.yaml declared inputs** (5): `A_log, a, beta, dt_bias, threshold`
- **fix recipe**:
  - Pin Triton version OR rewrite kernel.py decorator ordering
  - If oracle_pytorch.py is independent, defer kernel.py fix to Phase B
- **effort**: `L`

#### `L2_19_fused_sigmoid_gating_delta_rule_update`

- **aggregate status**: `BLOCKED(no-verified-upstream-entrypoint: shapes-incomplete)`
- **intended kernel**: `fused_sigmoid_gating_delta_rule_update`
- **intended substatus**: `shapes-incomplete`
- **upstream exception**: `anchored` (on 1/2 shape(s) ['H2')
- **root cause category**: `missing-shape-row`
- **root cause detail**: Kernel function `fused_sigmoid_gating_delta_rule_update` requires 10 param(s) that shapes.yaml does not declare: ['softplus_threshold', 'b', 'o', 'h0_source', 'h0_indices', 'cu_seqlens', 'intermediate_states_buffer', 'intermediate_state_indices', 'cache_steps', 'scale']. Resolver reports `softplus_threshold` first; subsequent params surface in cascade.
- **kernel required params** (17): `A_log, a, dt_bias, softplus_beta, softplus_threshold, q, k, v, b, o, h0_source, h0_indices, cu_seqlens, intermediate_states_buffer, intermediate_state_indices, cache_steps, scale`
- **missing in shapes.yaml** (10): `softplus_threshold, b, o, h0_source, h0_indices, cu_seqlens, intermediate_states_buffer, intermediate_state_indices, cache_steps, scale`
- **shapes.yaml declared inputs** (9): `A_log, a, beta, dt_bias, k, q, softplus_beta, state, v`
- **fix recipe**:
  - Open tasks/L2_19_fused_sigmoid_gating_delta_rule_update/shapes.yaml
  - Add ALL 10 missing entries: ['softplus_threshold', 'b', 'o', 'h0_source', 'h0_indices', 'cu_seqlens', 'intermediate_states_buffer', 'intermediate_state_indices', 'cache_steps', 'scale']
  - For each, refer to kernel.py signature for dtype + dims + dist semantics
  - Re-run resolver after each batch to confirm progress
- **effort**: `L`

#### `L3_03_chunk_state_varlen`

- **aggregate status**: `BLOCKED(no-verified-upstream-entrypoint: shapes-incomplete)`
- **intended kernel**: `chunk_state_varlen`
- **intended substatus**: `shapes-incomplete`
- **upstream exception**: `anchored` (on 1/2 shape(s) ['H2')
- **root cause category**: `shapes-incomplete-sibling`
- **root cause detail**: Resolver's missing-param mention (`chunk_states`) belongs to a sibling candidate, not the intended kernel `chunk_state_varlen`. The aggregate shapes-incomplete masks the intended kernel's true status.
- **kernel required params** (5): `hdim, dstate, chunk_size, seqlen, nheads_ngroups_ratio`
- **missing in shapes.yaml** (5): `hdim, dstate, chunk_size, seqlen, nheads_ngroups_ratio`
- **shapes.yaml declared inputs** (5): `B, cu_seqlens, dA_cumsum, dt, x`
- **fix recipe**:
  - Verify intended kernel `chunk_state_varlen` actually exists in kernel.py
  - If `chunk_state_varlen` is correct, the resolver may be trying wrong siblings -- fix meta.upstream.path or meta.oracle.gpu_triton to point at the right symbol
- **effort**: `M`

#### `L3_13_chunk_gated_delta_rule_fwd_kernel_h_blockdim64`

- **aggregate status**: `BLOCKED(no-verified-upstream-entrypoint: upstream-not-invocable-here)`
- **intended kernel**: `chunk_gated_delta_rule_fwd_kernel_h_blockdim64`
- **upstream exception**: `AttributeError` ('_AutotunedKernel' object has no attribute 'arg_names' | on-ascend-bench/tasks/level3/L3_13_chunk_gated_delta_rule_fwd_k)
- **root cause category**: `meta-resolver-mismatch`
- **root cause detail**: Resolver candidate-search did not try the meta-declared kernel `chunk_gated_delta_rule_fwd_kernel_h_blockdim64`. Likely meta.upstream.path / meta.oracle.gpu_triton points at a stale location, OR the kernel.py vendored function is named differently than meta records. Resolver tried instead: ['chunk_gated_delta_rule_fwd_h']
- **kernel required params** (12): `k, v, w, v_new, g, gk, h, h0, ht, cu_seqlens, chunk_offsets, Hg`
- **missing in shapes.yaml** (9): `v, v_new, gk, h, h0, ht, cu_seqlens, chunk_offsets, Hg`
- **shapes.yaml declared inputs** (4): `g, k, u, w`
- **fix recipe**:
  - Verify `chunk_gated_delta_rule_fwd_kernel_h_blockdim64` exists in kernel.py (grep `def chunk_gated_delta_rule_fwd_kernel_h_blockdim64`)
  - If not, find the actual entrypoint name in kernel.py and update meta.name
  - If yes, audit scripts/resolve_upstream_entrypoints.py candidate-search to understand why this kernel was not tried
  - Then add shapes.yaml inputs matching `chunk_gated_delta_rule_fwd_kernel_h_blockdim64`'s signature
- **effort**: `M`

#### `L3_14_fused_recurrent_gated_delta_rule_fwd`

- **aggregate status**: `BLOCKED(no-verified-upstream-entrypoint: shapes-incomplete)`
- **intended kernel**: `fused_recurrent_gated_delta_rule_fwd`
- **intended substatus**: `shapes-incomplete`
- **upstream exception**: `anchored` (on 1/2 shape(s) ['H2')
- **root cause category**: `shapes-incomplete-sibling`
- **root cause detail**: Resolver's missing-param mention (`initial_state`) belongs to a sibling candidate, not the intended kernel `fused_recurrent_gated_delta_rule_fwd`. The aggregate shapes-incomplete masks the intended kernel's true status.
- **kernel required params** (12): `q, k, v, g, beta, o, h0, ht, cu_seqlens, ssm_state_indices, num_accepted_tokens, scale`
- **missing in shapes.yaml** (6): `o, h0, ht, cu_seqlens, ssm_state_indices, num_accepted_tokens`
- **shapes.yaml declared inputs** (6): `beta, g, k, q, scale, v`
- **fix recipe**:
  - Verify intended kernel `fused_recurrent_gated_delta_rule_fwd` actually exists in kernel.py
  - If `fused_recurrent_gated_delta_rule_fwd` is correct, the resolver may be trying wrong siblings -- fix meta.upstream.path or meta.oracle.gpu_triton to point at the right symbol
- **effort**: `M`

#### `L3_15_fused_recurrent_gated_delta_rule_update_fwd`

- **aggregate status**: `BLOCKED(no-verified-upstream-entrypoint: shapes-incomplete)`
- **intended kernel**: `fused_recurrent_gated_delta_rule_update_fwd`
- **intended substatus**: `shapes-incomplete`
- **upstream exception**: `anchored` (on 1/2 shape(s) ['H2')
- **root cause category**: `shapes-incomplete-sibling`
- **root cause detail**: Resolver's missing-param mention (`initial_state_source`) belongs to a sibling candidate, not the intended kernel `fused_recurrent_gated_delta_rule_update_fwd`. The aggregate shapes-incomplete masks the intended kernel's true status.
- **kernel required params** (13): `q, k, v, g, beta, o, h0_source, h0_indices, cu_seqlens, scale, intermediate_states_buffer, intermediate_state_indices, cache_steps`
- **missing in shapes.yaml** (7): `o, h0_source, h0_indices, cu_seqlens, intermediate_states_buffer, intermediate_state_indices, cache_steps`
- **shapes.yaml declared inputs** (6): `beta, g, k, q, scale, v`
- **fix recipe**:
  - Verify intended kernel `fused_recurrent_gated_delta_rule_update_fwd` actually exists in kernel.py
  - If `fused_recurrent_gated_delta_rule_update_fwd` is correct, the resolver may be trying wrong siblings -- fix meta.upstream.path or meta.oracle.gpu_triton to point at the right symbol
- **effort**: `M`

#### `L3_16_chunk_fwd_kernel_o`

- **aggregate status**: `BLOCKED(no-verified-upstream-entrypoint: upstream-not-invocable-here)`
- **intended kernel**: `chunk_fwd_kernel_o`
- **upstream exception**: `AttributeError` ('_AutotunedKernel' object has no attribute 'arg_names' | /code/agentic-rl/kernel/triton-ascend-bench/tasks/level3/L3_16_)
- **root cause category**: `meta-resolver-mismatch`
- **root cause detail**: Resolver candidate-search did not try the meta-declared kernel `chunk_fwd_kernel_o`. Likely meta.upstream.path / meta.oracle.gpu_triton points at a stale location, OR the kernel.py vendored function is named differently than meta records. Resolver tried instead: ['chunk_fwd_o']
- **kernel required params** (10): `q, k, v, h, g, o, cu_seqlens, chunk_indices, scale, Hg`
- **missing in shapes.yaml** (4): `o, cu_seqlens, chunk_indices, Hg`
- **shapes.yaml declared inputs** (6): `g, h, k, q, scale, v`
- **fix recipe**:
  - Verify `chunk_fwd_kernel_o` exists in kernel.py (grep `def chunk_fwd_kernel_o`)
  - If not, find the actual entrypoint name in kernel.py and update meta.name
  - If yes, audit scripts/resolve_upstream_entrypoints.py candidate-search to understand why this kernel was not tried
  - Then add shapes.yaml inputs matching `chunk_fwd_kernel_o`'s signature
- **effort**: `M`

#### `L3_17_chunk_gla_fwd_kernel_o`

- **aggregate status**: `BLOCKED(no-verified-upstream-entrypoint: upstream-not-invocable-here)`
- **intended kernel**: `chunk_gla_fwd_kernel_o`
- **upstream exception**: `AttributeError` ('_AutotunedKernel' object has no attribute 'arg_names' | data1/code/agentic-rl/kernel/infer/vllm/vllm/model_executor/lay)
- **root cause category**: `meta-resolver-mismatch`
- **root cause detail**: Resolver candidate-search did not try the meta-declared kernel `chunk_gla_fwd_kernel_o`. Likely meta.upstream.path / meta.oracle.gpu_triton points at a stale location, OR the kernel.py vendored function is named differently than meta records. Resolver tried instead: ['chunk_gla_fwd_o_gk', 'recompute_w_u_fwd', 'chunk_kda_scaled_dot_kkt_fwd', 'fused_kda_gate']
- **kernel required params** (8): `q, v, g, h, o, cu_seqlens, chunk_indices, scale`
- **missing in shapes.yaml** (2): `cu_seqlens, chunk_indices`
- **shapes.yaml declared inputs** (7): `A, g, h, o, q, scale, v`
- **fix recipe**:
  - Verify `chunk_gla_fwd_kernel_o` exists in kernel.py (grep `def chunk_gla_fwd_kernel_o`)
  - If not, find the actual entrypoint name in kernel.py and update meta.name
  - If yes, audit scripts/resolve_upstream_entrypoints.py candidate-search to understand why this kernel was not tried
  - Then add shapes.yaml inputs matching `chunk_gla_fwd_kernel_o`'s signature
- **effort**: `M`

#### `L3_18_chunk_scaled_dot_kkt_fwd`

- **aggregate status**: `BLOCKED(no-verified-upstream-entrypoint: upstream-not-invocable-here)`
- **intended kernel**: `chunk_scaled_dot_kkt_fwd`
- **intended substatus**: `upstream-not-invocable-here`
- **upstream exception**: `AttributeError` ('_AutotunedKernel' object has no attribute 'arg_names')
- **root cause category**: `triton-autotune-api-mismatch`
- **root cause detail**: Kernel applies @triton.heuristics on top of a wrapper that lacks .arg_names. Current Triton API expects different ordering.
- **kernel required params** (6): `k, beta, g, cu_seqlens, chunk_indices, Hg`
- **missing in shapes.yaml** (3): `cu_seqlens, chunk_indices, Hg`
- **shapes.yaml declared inputs** (3): `beta, g, k`
- **fix recipe**:
  - Pin Triton version OR rewrite kernel.py decorator ordering
  - If oracle_pytorch.py is independent, defer kernel.py fix to Phase B
- **effort**: `L`

#### `L3_19_chunk_kda_scaled_dot_kkt_fwd_kernel_intra_sub_inter`

- **aggregate status**: `BLOCKED(no-verified-upstream-entrypoint: upstream-not-invocable-here)`
- **intended kernel**: `chunk_kda_scaled_dot_kkt_fwd`
- **intended substatus**: `upstream-not-invocable-here`
- **upstream exception**: `AttributeError` ('_AutotunedKernel' object has no attribute 'arg_names')
- **root cause category**: `triton-autotune-api-mismatch`
- **root cause detail**: Kernel applies @triton.heuristics on top of a wrapper that lacks .arg_names. Current Triton API expects different ordering.
- **kernel required params** (8): `q, k, g, beta, Aqk, scale, cu_seqlens, chunk_indices`
- **missing in shapes.yaml** (4): `g, Aqk, cu_seqlens, chunk_indices`
- **shapes.yaml declared inputs** (5): `beta, gk, k, q, scale`
- **fix recipe**:
  - Pin Triton version OR rewrite kernel.py decorator ordering
  - If oracle_pytorch.py is independent, defer kernel.py fix to Phase B
- **effort**: `L`

#### `L3_21_solve_tril_16x16`

- **aggregate status**: `BLOCKED(no-verified-upstream-entrypoint: upstream-not-invocable-here)`
- **intended kernel**: `solve_tril_16x16_kernel`
- **intended substatus**: `upstream-not-invocable-here`
- **upstream exception**: `AttributeError` ('_AutotunedKernel' object has no attribute 'arg_names')
- **root cause category**: `triton-autotune-api-mismatch`
- **root cause detail**: Kernel applies @triton.heuristics on top of a wrapper that lacks .arg_names. Current Triton API expects different ordering.
- **kernel required params** (3): `Ai, cu_seqlens, chunk_indices`
- **missing in shapes.yaml** (3): `Ai, cu_seqlens, chunk_indices`
- **shapes.yaml declared inputs** (1): `A`
- **fix recipe**:
  - Pin Triton version OR rewrite kernel.py decorator ordering
  - If oracle_pytorch.py is independent, defer kernel.py fix to Phase B
- **effort**: `L`

#### `L3_22_chunk_local_cumsum_vector`

- **aggregate status**: `BLOCKED(no-verified-upstream-entrypoint: upstream-not-invocable-here)`
- **intended kernel**: `chunk_local_cumsum_vector`
- **intended substatus**: `upstream-not-invocable-here`
- **upstream exception**: `AttributeError` ('_AutotunedKernel' object has no attribute 'arg_names')
- **root cause category**: `triton-autotune-api-mismatch`
- **root cause detail**: Kernel applies @triton.heuristics on top of a wrapper that lacks .arg_names. Current Triton API expects different ordering.
- **kernel required params** (4): `s, o, cu_seqlens, chunk_indices`
- **missing in shapes.yaml** (4): `s, o, cu_seqlens, chunk_indices`
- **shapes.yaml declared inputs** (3): `chunk_size, g, reverse`
- **fix recipe**:
  - Pin Triton version OR rewrite kernel.py decorator ordering
  - If oracle_pytorch.py is independent, defer kernel.py fix to Phase B
- **effort**: `L`
