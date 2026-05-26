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
| `missing-shape-row` | 35 |
| `triton-autotune-api-mismatch` | 11 |
| `upstream-exprerror` | 9 |
| `kernel-assertion` | 7 |
| `needs-gpu` | 5 |
| `oracle-numeric-tolerance` | 5 |
| `upstream-interpretererror` | 4 |
| `import-error` | 3 |
| `oracle-structure-mismatch` | 3 |
| `duplicate-op-registration` | 3 |
| `platform-stub-noimpl` | 2 |
| `smoke-fail-severed-run` | 2 |
| `upstream-attributeerror` | 2 |
| `sigsegv` | 2 |
| `upstream-runtimeerror` | 2 |
| `kernel-config-keyerror` | 1 |
| `upstream-valueerror` | 1 |
| `smoke-fail-cpu-budget` | 1 |
| `oracle-dtype-mismatch` | 1 |
| `kernel-zerodivision` | 1 |

### Effort estimate distribution

| effort | count | meaning |
|---|---|---|
| S | 9 | ≤30 min per task |
| M | 75 | 30-120 min per task |
| L | 16 | ≥120 min per task |

## Per-Task Detail

### family: `attention` (n=18)

#### `L1_02_compute_slot_mappings`

- **aggregate status**: `BLOCKED(no-verified-upstream-entrypoint: upstream-not-invocable-here)`
- **intended kernel**: `_compute_slot_mappings_kernel`
- **intended substatus**: `upstream-not-invocable-here`
- **upstream exception**: `InterpreterError` (TypeError("_compute_slot_mappings_kernel() missing 14 required positional arguments: 'max_num_tokens', 'idx_mapping', 'q)
- **root cause category**: `upstream-interpretererror`
- **root cause detail**: Upstream raises InterpreterError: TypeError("_compute_slot_mappings_kernel() missing 14 required positional arguments: 'max_num_tokens', 'idx_mapping', 'query_start_loc', 'pos', 'block_table_ptrs', 'block_table_strides', 'block_sizes'
- **kernel required params** (14): `max_num_tokens, idx_mapping, query_start_loc, pos, block_table_ptrs, block_table_strides, block_sizes, slot_mappings_ptr, slot_mappings_stride, cp_rank, CP_SIZE, CP_INTERLEAVE, PAD_ID, TRITON_BLOCK_SIZE`
- **missing in shapes.yaml** (12): `query_start_loc, pos, block_table_ptrs, block_table_strides, block_sizes, slot_mappings_ptr, slot_mappings_stride, cp_rank, CP_SIZE, CP_INTERLEAVE, PAD_ID, TRITON_BLOCK_SIZE`
- **shapes.yaml declared inputs** (5): `block_size, block_table, idx_mapping, max_num_tokens, positions`
- **fix recipe**:
  - Inspect the exception's call site in upstream
  - Decide: shim, mock, or rewrite oracle_pytorch.py standalone
- **effort**: `M`

#### `L1_04_copy_page_indices`

- **aggregate status**: `BLOCKED(no-verified-upstream-entrypoint: upstream-not-invocable-here)`
- **intended kernel**: `trtllm_prefill_attn_kvfp8_dequant`
- **intended substatus**: `upstream-not-invocable-here`
- **upstream exception**: `ModuleNotFoundError` (No module named 'flashinfer')
- **root cause category**: `import-error`
- **root cause detail**: Upstream import chain fails: ModuleNotFoundError for `flashinfer`.
- **kernel required params** (5): `kv_cache, block_tables_prefill, k_scale, v_scale, dequant_dtype`
- **missing in shapes.yaml** (5): `kv_cache, block_tables_prefill, k_scale, v_scale, dequant_dtype`
- **shapes.yaml declared inputs** (3): `block_table, page_size, seq_lens`
- **fix recipe**:
  - Option A: `pip install flashinfer` (if pure-Python, low risk)
  - Option B: oracle_pytorch.py implement a standalone reference (no upstream import)
  - Option C: GPU-only -- defer to Phase B if module is CUDA-bound
- **effort**: `M`

#### `L1_06_reshape_and_cache_kernel_flash`

- **aggregate status**: `BLOCKED(no-verified-upstream-entrypoint: shapes-incomplete)`
- **intended kernel**: `triton_reshape_and_cache_flash`
- **intended substatus**: `shapes-incomplete`
- **upstream exception**: `anchored` (on 1/2 shape(s) ['H2')
- **root cause category**: `missing-shape-row`
- **root cause detail**: Kernel function `triton_reshape_and_cache_flash` requires param `kv_cache_dtype` (in its signature), but shapes.yaml `inputs:` block does not declare it. Resolver's by-name adapter stops at first missing param.
- **kernel required params** (8): `key, value, key_cache, value_cache, slot_mapping, kv_cache_dtype, k_scale, v_scale`
- **missing in shapes.yaml** (3): `kv_cache_dtype, k_scale, v_scale`
- **shapes.yaml declared inputs** (5): `key, key_cache, slot_mapping, value, value_cache`
- **fix recipe**:
  - Open tasks/L1_06_reshape_and_cache_kernel_flash/shapes.yaml
  - Add `kv_cache_dtype:` entry under `inputs:` with the kernel's expected dtype + dims + dist
  - Refer to kernel.py signature / docstring for shape constraints
  - Re-run resolver to confirm advance to next stage (often more missing params surface in cascade)
- **effort**: `M`

#### `L1_08_dequantize_k_cache_paged`

- **aggregate status**: `BLOCKED(no-verified-upstream-entrypoint: shapes-incomplete)`
- **intended kernel**: `dequantize_k_cache_paged`
- **intended substatus**: `shapes-incomplete`
- **upstream exception**: `anchored` (on 1/2 shape(s) ['H2')
- **root cause category**: `missing-shape-row`
- **root cause detail**: Kernel function `dequantize_k_cache_paged` requires param `quant_k_cache` (in its signature), but shapes.yaml `inputs:` block does not declare it. Resolver's by-name adapter stops at first missing param.
- **kernel required params** (2): `quant_k_cache, page_table_1_flattened`
- **missing in shapes.yaml** (2): `quant_k_cache, page_table_1_flattened`
- **shapes.yaml declared inputs** (5): `group_size, nope_q, nope_s, page_table, rope`
- **fix recipe**:
  - Open tasks/L1_08_dequantize_k_cache_paged/shapes.yaml
  - Add `quant_k_cache:` entry under `inputs:` with the kernel's expected dtype + dims + dist
  - Refer to kernel.py signature / docstring for shape constraints
  - Re-run resolver to confirm advance to next stage (often more missing params surface in cascade)
- **effort**: `M`

#### `L1_25_merge_attn_states`

- **aggregate status**: `BLOCKED(no-verified-upstream-entrypoint: oracle-mismatch)`
- **intended kernel**: `merge_attn_states`
- **intended substatus**: `oracle-mismatch`
- **upstream exception**: `anchored` (on 1/2 shape(s) ['H2')
- **root cause category**: `oracle-structure-mismatch`
- **root cause detail**: Kernel output and oracle disagree on output structure (shape, dtype, or operator semantics).
- **kernel required params** (5): `output, prefix_output, prefix_lse, suffix_output, suffix_lse`
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
- **kernel required params** (8): `q, k, cos, sin, mrope_section, head_size, rotary_dim, mrope_interleaved`
- **shapes.yaml declared inputs** (8): `cos, head_size, k, mrope_interleaved, mrope_section, q, rotary_dim, sin`
- **fix recipe**:
  - Identify the duplicate registration path (usually two vllm copies)
  - Option A: dedupe sys.path so only one vllm import wins
  - Option B: oracle_pytorch.py rewrite to not import vllm
- **effort**: `L`

#### `L1_32_create_flashinfer_kv_indices_triton`

- **aggregate status**: `BLOCKED(no-verified-upstream-entrypoint: shapes-incomplete)`
- **intended kernel**: `fused_qk_rope_reshape_and_cache`
- **intended substatus**: `shapes-incomplete`
- **upstream exception**: `anchored` (on 1/2 shape(s) ['H2')
- **root cause category**: `missing-shape-row`
- **root cause detail**: Kernel function `fused_qk_rope_reshape_and_cache` requires param `v` (in its signature), but shapes.yaml `inputs:` block does not declare it. Resolver's by-name adapter stops at first missing param.
- **kernel required params** (12): `q, k, v, key_cache, value_cache, slot_mapping, pos, cos_sin, k_scale, v_scale, is_neox, flash_layout`
- **missing in shapes.yaml** (10): `v, key_cache, value_cache, slot_mapping, pos, cos_sin, k_scale, v_scale, is_neox, flash_layout`
- **shapes.yaml declared inputs** (8): `block_table, extend_seq_lens, input_emb, k, key, page_size, q, seq_lens`
- **fix recipe**:
  - Open tasks/L1_32_create_flashinfer_kv_indices_triton/shapes.yaml
  - Add `v:` entry under `inputs:` with the kernel's expected dtype + dims + dist
  - Refer to kernel.py signature / docstring for shape constraints
  - Re-run resolver to confirm advance to next stage (often more missing params surface in cascade)
- **effort**: `M`

#### `L1_33_create_flashmla_kv_indices_triton`

- **aggregate status**: `BLOCKED(no-verified-upstream-entrypoint: shapes-incomplete)`
- **intended kernel**: `fused_qk_rope_reshape_and_cache`
- **intended substatus**: `shapes-incomplete`
- **upstream exception**: `anchored` (on 1/2 shape(s) ['H2')
- **root cause category**: `missing-shape-row`
- **root cause detail**: Kernel function `fused_qk_rope_reshape_and_cache` requires param `key_cache` (in its signature), but shapes.yaml `inputs:` block does not declare it. Resolver's by-name adapter stops at first missing param.
- **kernel required params** (12): `q, k, v, key_cache, value_cache, slot_mapping, pos, cos_sin, k_scale, v_scale, is_neox, flash_layout`
- **missing in shapes.yaml** (9): `key_cache, value_cache, slot_mapping, pos, cos_sin, k_scale, v_scale, is_neox, flash_layout`
- **shapes.yaml declared inputs** (10): `block_table, extend_seq_lens, input_emb, k, key, page_size, q, seq_lens, v, value`
- **fix recipe**:
  - Open tasks/L1_33_create_flashmla_kv_indices_triton/shapes.yaml
  - Add `key_cache:` entry under `inputs:` with the kernel's expected dtype + dims + dist
  - Refer to kernel.py signature / docstring for shape constraints
  - Re-run resolver to confirm advance to next stage (often more missing params surface in cascade)
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
- **intended kernel**: `qkv_lora_b_fwd`
- **intended substatus**: `upstream-not-invocable-here`
- **upstream exception**: `AssertionError` ()
- **root cause category**: `kernel-assertion`
- **root cause detail**: Kernel function `qkv_lora_b_fwd` hits an AssertionError when called with the current shapes: ``
- **kernel required params** (5): `x, qkv_lora_b, batch_info, output_offset, max_qkv_out_dim`
- **shapes.yaml declared inputs** (9): `batch_info, lora_ranks, max_qkv_out_dim, output_offset, qkv_lora_b, scalings, seg_indptr, weight_indices, x`
- **fix recipe**:
  - Read the asserted condition in kernel.py
  - Adjust shapes.yaml so the input tensors satisfy the assertion
  - (e.g. paired dtype, divisibility, shape compatibility)
- **effort**: `M`

#### `L2_22_linear_attn_decode`

- **aggregate status**: `BLOCKED(no-verified-upstream-entrypoint: shapes-incomplete)`
- **intended kernel**: `linear_decode_forward_triton`
- **intended substatus**: `shapes-incomplete`
- **upstream exception**: `anchored` (on 1/2 shape(s) ['H2')
- **root cause category**: `missing-shape-row`
- **root cause detail**: Kernel function `linear_decode_forward_triton` requires param `slot_idx` (in its signature), but shapes.yaml `inputs:` block does not declare it. Resolver's by-name adapter stops at first missing param.
- **kernel required params** (6): `q, k, v, kv_caches, slope_rate, slot_idx`
- **missing in shapes.yaml** (1): `slot_idx`
- **shapes.yaml declared inputs** (5): `k, kv_caches, q, slope_rate, v`
- **fix recipe**:
  - Open tasks/L2_22_linear_attn_decode/shapes.yaml
  - Add `slot_idx:` entry under `inputs:` with the kernel's expected dtype + dims + dist
  - Refer to kernel.py signature / docstring for shape constraints
  - Re-run resolver to confirm advance to next stage (often more missing params surface in cascade)
- **effort**: `M`

#### `L2_23_fwd_kv_parallel`

- **aggregate status**: `BLOCKED(no-verified-upstream-entrypoint: shapes-incomplete)`
- **intended kernel**: `linear_decode_forward_triton`
- **intended substatus**: `shapes-incomplete`
- **upstream exception**: `anchored` (on 1/2 shape(s) ['H2')
- **root cause category**: `missing-shape-row`
- **root cause detail**: Kernel function `linear_decode_forward_triton` requires param `kv_caches` (in its signature), but shapes.yaml `inputs:` block does not declare it. Resolver's by-name adapter stops at first missing param.
- **kernel required params** (6): `q, k, v, kv_caches, slope_rate, slot_idx`
- **missing in shapes.yaml** (3): `kv_caches, slope_rate, slot_idx`
- **shapes.yaml declared inputs** (4): `k, q, slope, v`
- **fix recipe**:
  - Open tasks/L2_23_fwd_kv_parallel/shapes.yaml
  - Add `kv_caches:` entry under `inputs:` with the kernel's expected dtype + dims + dist
  - Refer to kernel.py signature / docstring for shape constraints
  - Re-run resolver to confirm advance to next stage (often more missing params surface in cascade)
- **effort**: `M`

#### `L2_24_fwd_kv_reduce`

- **aggregate status**: `BLOCKED(no-verified-upstream-entrypoint: shapes-incomplete)`
- **intended kernel**: `linear_decode_forward_triton`
- **intended substatus**: `shapes-incomplete`
- **upstream exception**: `anchored` (on 1/2 shape(s) ['H2')
- **root cause category**: `missing-shape-row`
- **root cause detail**: Kernel function `linear_decode_forward_triton` requires param `kv_caches` (in its signature), but shapes.yaml `inputs:` block does not declare it. Resolver's by-name adapter stops at first missing param.
- **kernel required params** (6): `q, k, v, kv_caches, slope_rate, slot_idx`
- **missing in shapes.yaml** (3): `kv_caches, slope_rate, slot_idx`
- **shapes.yaml declared inputs** (4): `k, q, slope, v`
- **fix recipe**:
  - Open tasks/L2_24_fwd_kv_reduce/shapes.yaml
  - Add `kv_caches:` entry under `inputs:` with the kernel's expected dtype + dims + dist
  - Refer to kernel.py signature / docstring for shape constraints
  - Re-run resolver to confirm advance to next stage (often more missing params surface in cascade)
- **effort**: `M`

#### `L2_33_fwd_grouped_kernel_stage1_rope`

- **aggregate status**: `BLOCKED(no-verified-upstream-entrypoint: upstream-not-invocable-here)`
- **intended kernel**: `_decode_grouped_att_m_fwd_rope`
- **intended substatus**: `upstream-not-invocable-here`
- **upstream exception**: `RuntimeError` (0 active drivers ([)
- **root cause category**: `upstream-runtimeerror`
- **root cause detail**: Upstream raises RuntimeError: 0 active drivers ([
- **kernel required params** (15): `q, k_buffer, v_buffer, att_out, k_pe_tokens_out, kv_lora_rank, cos_sin_cache, positions, rotary_dim, kv_indptr, kv_indices, num_kv_splits, sm_scale, logit_cap, use_rope`
- **missing in shapes.yaml** (11): `k_pe_tokens_out, kv_lora_rank, cos_sin_cache, positions, rotary_dim, kv_indptr, kv_indices, num_kv_splits, sm_scale, logit_cap, use_rope`
- **shapes.yaml declared inputs** (5): `att_out, k_buffer, q, scale, v_buffer`
- **fix recipe**:
  - Inspect the exception's call site in upstream
  - Decide: shim, mock, or rewrite oracle_pytorch.py standalone
- **effort**: `M`

#### `L3_01_kernel_unified_attention_3d`

- **aggregate status**: `BLOCKED(no-verified-upstream-entrypoint: upstream-not-invocable-here)`
- **intended kernel**: `unified_attention`
- **intended substatus**: `upstream-not-invocable-here`
- **upstream exception**: `ZeroDivisionError` (integer division or modulo by zero)
- **root cause category**: `kernel-zerodivision`
- **root cause detail**: Shape sweep produces a zero-valued dim/stride the kernel does not guard.
- **kernel required params** (16): `q, k, v, out, cu_seqlens_q, max_seqlen_q, seqused_k, max_seqlen_k, softmax_scale, causal, window_size, block_table, softcap, q_descale, k_descale, v_descale`
- **shapes.yaml declared inputs** (16): `block_table, causal, cu_seqlens_q, k, k_descale, max_seqlen_k, max_seqlen_q, out, q, q_descale, seqused_k, softcap, softmax_scale, v, v_descale, window_size`
- **fix recipe**:
  - Identify which shape row triggers the zero (usually H2/H_singleton with extreme small knobs)
  - Add a guard in shapes.yaml hidden_shapes or in oracle_pytorch.py for that row
- **effort**: `S`

#### `L3_02_kernel_paged_attention_2d`

- **aggregate status**: `BLOCKED(no-verified-upstream-entrypoint: shapes-incomplete)`
- **intended kernel**: `chunked_prefill_paged_decode`
- **intended substatus**: `shapes-incomplete`
- **upstream exception**: `anchored` (on 1/2 shape(s) ['H2')
- **root cause category**: `missing-shape-row`
- **root cause detail**: Kernel function `chunked_prefill_paged_decode` requires param `value` (in its signature), but shapes.yaml `inputs:` block does not declare it. Resolver's by-name adapter stops at first missing param.
- **kernel required params** (14): `query, key, value, output, kv_cache_dtype, key_cache, value_cache, block_table, query_start_loc, seq_lens, max_seq_len, max_query_len, k_scale, v_scale`
- **missing in shapes.yaml** (8): `value, output, kv_cache_dtype, query_start_loc, max_seq_len, max_query_len, k_scale, v_scale`
- **shapes.yaml declared inputs** (7): `block_table, key, key_cache, query, scale, seq_lens, value_cache`
- **fix recipe**:
  - Open tasks/L3_02_kernel_paged_attention_2d/shapes.yaml
  - Add `value:` entry under `inputs:` with the kernel's expected dtype + dims + dist
  - Refer to kernel.py signature / docstring for shape constraints
  - Re-run resolver to confirm advance to next stage (often more missing params surface in cascade)
- **effort**: `M`

#### `L3_04_fwd`

- **aggregate status**: `BLOCKED(no-verified-upstream-entrypoint: upstream-not-invocable-here)`
- **upstream exception**: `oracle` (-check subprocess produced no result (rc=-11):)
- **root cause category**: `sigsegv`
- **root cause detail**: Resolver subprocess SIGSEGV under CPU interpreter. Crash is in third-party C (Triton interpreter / numpy backend); not reproducible on GPU/NPU.
- **shapes.yaml declared inputs** (9): `b_seq_len, b_start_loc, is_causal, k, max_input_len, o, q, sm_scale, v`
- **fix recipe**:
  - Phase B golden capture on GPU/NPU is the authoritative path
  - Optional: rewrite oracle_pytorch.py to avoid interpreter-only codepath
- **effort**: `L`

#### `L3_07_correct_attn_cp_out`

- **aggregate status**: `BLOCKED(no-verified-upstream-entrypoint: shapes-incomplete)`
- **intended kernel**: `pack_seq_triton`
- **intended substatus**: `shapes-incomplete`
- **upstream exception**: `anchored` (on 1/2 shape(s) ['H2')
- **root cause category**: `missing-shape-row`
- **root cause detail**: Kernel function `pack_seq_triton` requires param `lengths` (in its signature), but shapes.yaml `inputs:` block does not declare it. Resolver's by-name adapter stops at first missing param.
- **kernel required params** (2): `x, lengths`
- **missing in shapes.yaml** (1): `lengths`
- **shapes.yaml declared inputs** (5): `cp_rank, is_lse_base_on_e, lses, out, x`
- **fix recipe**:
  - Open tasks/L3_07_correct_attn_cp_out/shapes.yaml
  - Add `lengths:` entry under `inputs:` with the kernel's expected dtype + dims + dist
  - Refer to kernel.py signature / docstring for shape constraints
  - Re-run resolver to confirm advance to next stage (often more missing params surface in cascade)
- **effort**: `M`

### family: `convolution` (n=3)

#### `L1_03_convert_req_index_to_global_index`

- **aggregate status**: `BLOCKED(no-verified-upstream-entrypoint: upstream-not-invocable-here)`
- **intended kernel**: `triton_convert_req_index_to_global_index`
- **intended substatus**: `upstream-not-invocable-here`
- **upstream exception**: `KeyError` ('block_size')
- **root cause category**: `kernel-config-keyerror`
- **root cause detail**: Upstream raises KeyError(`block_size`). The missing dict key is provided by an upstream config layer / model_executor caller that does not initialise on dev.
- **kernel required params** (3): `req_id, block_table, token_indices`
- **shapes.yaml declared inputs** (6): `BLOCK_N, BLOCK_SIZE, NUM_TOPK_TOKENS, block_table, req_id, token_indices`
- **fix recipe**:
  - Read the call site to identify which dict expected `block_size`
  - Option A: shim the dict in oracle_pytorch.py with a known-good default
  - Option B: rewrite oracle_pytorch.py as a standalone reference (no upstream import)
- **effort**: `M`

#### `L2_20_causal_conv1d_update`

- **aggregate status**: `BLOCKED(no-verified-upstream-entrypoint: shapes-incomplete)`
- **intended kernel**: `causal_conv1d_update`
- **intended substatus**: `upstream-not-invocable-here`
- **upstream exception**: `InterpreterError` (AttributeError("'NoneType' object has no attribute 'type'"))
- **root cause category**: `upstream-interpretererror`
- **root cause detail**: Upstream raises InterpreterError: AttributeError("'NoneType' object has no attribute 'type'")
- **kernel required params** (3): `x, conv_state, weight`
- **shapes.yaml declared inputs** (5): `activation, bias, conv_state, weight, x`
- **fix recipe**:
  - Inspect the exception's call site in upstream
  - Decide: shim, mock, or rewrite oracle_pytorch.py standalone
- **effort**: `M`

#### `L2_21_causal_conv1d_fwd`

- **aggregate status**: `BLOCKED(no-verified-upstream-entrypoint: shapes-incomplete)`
- **intended kernel**: `causal_conv1d_fn`
- **intended substatus**: `shapes-incomplete`
- **upstream exception**: `anchored` (on 1/2 shape(s) ['H2')
- **root cause category**: `missing-shape-row`
- **root cause detail**: Kernel function `causal_conv1d_fn` requires param `conv_states` (in its signature), but shapes.yaml `inputs:` block does not declare it. Resolver's by-name adapter stops at first missing param.
- **kernel required params** (5): `x, weight, bias, conv_states, query_start_loc`
- **missing in shapes.yaml** (2): `conv_states, query_start_loc`
- **shapes.yaml declared inputs** (4): `activation, bias, weight, x`
- **fix recipe**:
  - Open tasks/L2_21_causal_conv1d_fwd/shapes.yaml
  - Add `conv_states:` entry under `inputs:` with the kernel's expected dtype + dims + dist
  - Refer to kernel.py signature / docstring for shape constraints
  - Re-run resolver to confirm advance to next stage (often more missing params surface in cascade)
- **effort**: `M`

### family: `elementwise` (n=7)

#### `L1_15_act_and_mul`

- **aggregate status**: `BLOCKED(no-verified-upstream-entrypoint: shapes-incomplete)`
- **intended kernel**: `act_and_mul_triton`
- **intended substatus**: `shapes-incomplete`
- **upstream exception**: `anchored` (on 1/2 shape(s) ['H2')
- **root cause category**: `missing-shape-row`
- **root cause detail**: Kernel function `act_and_mul_triton` requires param `gateup_output` (in its signature), but shapes.yaml `inputs:` block does not declare it. Resolver's by-name adapter stops at first missing param.
- **kernel required params** (3): `gateup_output, down_input, config`
- **missing in shapes.yaml** (3): `gateup_output, down_input, config`
- **shapes.yaml declared inputs** (3): `act, input, x`
- **fix recipe**:
  - Open tasks/L1_15_act_and_mul/shapes.yaml
  - Add `gateup_output:` entry under `inputs:` with the kernel's expected dtype + dims + dist
  - Refer to kernel.py signature / docstring for shape constraints
  - Re-run resolver to confirm advance to next stage (often more missing params surface in cascade)
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
- **intended kernel**: `gelu_and_mul_triton`
- **intended substatus**: `upstream-not-invocable-here`
- **upstream exception**: `ValueError` (too many values to unpack (expected 2))
- **root cause category**: `upstream-valueerror`
- **root cause detail**: Upstream raises ValueError: too many values to unpack (expected 2)
- **kernel required params** (1): `hidden_states`
- **shapes.yaml declared inputs** (6): `approximate, hidden_states, residual, softcap_const, weight1, x`
- **fix recipe**:
  - Inspect the exception's call site in upstream
  - Decide: shim, mock, or rewrite oracle_pytorch.py standalone
- **effort**: `M`

#### `L1_18_fused_softcap`

- **aggregate status**: `BLOCKED(no-verified-upstream-entrypoint: shapes-incomplete)`
- **intended kernel**: `fused_softcap`
- **intended substatus**: `shapes-incomplete`
- **upstream exception**: `anchored` (on 1/2 shape(s) ['H2')
- **root cause category**: `missing-shape-row`
- **root cause detail**: Kernel function `fused_softcap` requires param `full_logits` (in its signature), but shapes.yaml `inputs:` block does not declare it. Resolver's by-name adapter stops at first missing param.
- **kernel required params** (2): `full_logits, final_logit_softcapping`
- **missing in shapes.yaml** (2): `full_logits, final_logit_softcapping`
- **shapes.yaml declared inputs** (2): `cap, x`
- **fix recipe**:
  - Open tasks/L1_18_fused_softcap/shapes.yaml
  - Add `full_logits:` entry under `inputs:` with the kernel's expected dtype + dims + dist
  - Refer to kernel.py signature / docstring for shape constraints
  - Re-run resolver to confirm advance to next stage (often more missing params surface in cascade)
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
- **intended kernel**: `concat_and_cast_mha_k_triton`
- **intended substatus**: `upstream-not-invocable-here`
- **upstream exception**: `AssertionError` (shape should be 3d, but got k.shape=torch.Size([2, 2, 8, 16)
- **root cause category**: `kernel-assertion`
- **root cause detail**: Kernel function `concat_and_cast_mha_k_triton` hits an AssertionError when called with the current shapes: `shape should be 3d, but got k.shape=torch.Size([2, 2, 8, 16`
- **kernel required params** (3): `k, k_nope, k_rope`
- **shapes.yaml declared inputs** (11): `extend_seq_lens, input_emb, k, k_nope, k_rope, key, out_dtype, q, seq_lens, v, value`
- **fix recipe**:
  - Read the asserted condition in kernel.py
  - Adjust shapes.yaml so the input tensors satisfy the assertion
  - (e.g. paired dtype, divisibility, shape compatibility)
- **effort**: `M`

#### `L3_23_merge_16x16_to_32x32_inverse`

- **aggregate status**: `BLOCKED(no-verified-upstream-entrypoint: upstream-not-invocable-here)`
- **intended kernel**: `merge_16x16_to_32x32_inverse_kernel`
- **intended substatus**: `upstream-not-invocable-here`
- **upstream exception**: `AttributeError` ('_AutotunedKernel' object has no attribute 'arg_names')
- **root cause category**: `triton-autotune-api-mismatch`
- **root cause detail**: Kernel applies @triton.heuristics on top of a wrapper that lacks .arg_names. Current Triton API expects different ordering.
- **kernel required params** (10): `A, Ai, cu_seqlens, chunk_indices, T, H, BT, USE_TMA, IS_VARLEN, DOT_PRECISION`
- **missing in shapes.yaml** (9): `Ai, cu_seqlens, chunk_indices, T, H, BT, USE_TMA, IS_VARLEN, DOT_PRECISION`
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
- **kernel required params** (4): `a, b, chunk_size, cu_chunk_seqlens`
- **shapes.yaml declared inputs** (5): `a, b, causal, chunk_size, cu_chunk_seqlens`
- **fix recipe**:
  - Reduce smoke shape coverage to H2 (smallest hidden shape)
  - Or skip CPU pre-filter for this task family
- **effort**: `M`

#### `L2_04_chunked_lora_expand`

- **aggregate status**: `BLOCKED(no-verified-upstream-entrypoint: upstream-not-invocable-here)`
- **intended kernel**: `chunked_sgmv_lora_expand_forward`
- **intended substatus**: `upstream-not-invocable-here`
- **upstream exception**: `AttributeError` ('NoneType' object has no attribute 'replace')
- **root cause category**: `upstream-attributeerror`
- **root cause detail**: Upstream raises AttributeError: 'NoneType' object has no attribute 'replace'
- **kernel required params** (6): `x, weights, batch_info, slice_offsets, max_slice_size, base_output`
- **shapes.yaml declared inputs** (10): `base_output, batch_info, lora_ranks, max_slice_size, scalings, seg_indptr, slice_offsets, weight_indices, weights, x`
- **fix recipe**:
  - Inspect the exception's call site in upstream
  - Decide: shim, mock, or rewrite oracle_pytorch.py standalone
- **effort**: `M`

#### `L2_05_chunked_lora_shrink`

- **aggregate status**: `BLOCKED(no-verified-upstream-entrypoint: upstream-not-invocable-here)`
- **intended kernel**: `chunked_sgmv_lora_shrink_forward`
- **intended substatus**: `upstream-not-invocable-here`
- **upstream exception**: `AttributeError` ('NoneType' object has no attribute 'replace')
- **root cause category**: `upstream-attributeerror`
- **root cause detail**: Upstream raises AttributeError: 'NoneType' object has no attribute 'replace'
- **kernel required params** (4): `x, weights, batch_info, num_slices`
- **shapes.yaml declared inputs** (8): `batch_info, lora_ranks, num_slices, scalings, seg_indptr, weight_indices, weights, x`
- **fix recipe**:
  - Inspect the exception's call site in upstream
  - Decide: shim, mock, or rewrite oracle_pytorch.py standalone
- **effort**: `M`

#### `L2_07_gate_up_lora_b`

- **aggregate status**: `BLOCKED(no-verified-upstream-entrypoint: upstream-not-invocable-here)`
- **intended kernel**: `gate_up_lora_b_fwd`
- **intended substatus**: `upstream-not-invocable-here`
- **upstream exception**: `AssertionError` ()
- **root cause category**: `kernel-assertion`
- **root cause detail**: Kernel function `gate_up_lora_b_fwd` hits an AssertionError when called with the current shapes: ``
- **kernel required params** (4): `x, gate_up_lora_b, batch_info, output_dim`
- **shapes.yaml declared inputs** (8): `batch_info, gate_up_lora_b, lora_ranks, output_dim, scalings, seg_indptr, weight_indices, x`
- **fix recipe**:
  - Read the asserted condition in kernel.py
  - Adjust shapes.yaml so the input tensors satisfy the assertion
  - (e.g. paired dtype, divisibility, shape compatibility)
- **effort**: `M`

#### `L2_08_sgemm_lora_a`

- **aggregate status**: `BLOCKED(no-verified-upstream-entrypoint: oracle-mismatch)`
- **intended kernel**: `sgemm_lora_a_fwd`
- **intended substatus**: `oracle-mismatch`
- **upstream exception**: `anchored` (on 1/2 shape(s) ['H2')
- **root cause category**: `oracle-numeric-tolerance`
- **root cause detail**: Numeric divergence between kernel and oracle under the task's tolerance class. Likely CPU interpreter precision drift but may be real kernel bug.
- **kernel required params** (3): `x, weights, batch_info`
- **shapes.yaml declared inputs** (7): `batch_info, lora_ranks, scalings, seg_indptr, weight_indices, weights, x`
- **fix recipe**:
  - Re-run on GPU to distinguish interpreter drift from real divergence
  - If real, audit oracle_pytorch.py against kernel.py mathematically
- **effort**: `M`

#### `L2_09_sgemm_lora_b`

- **aggregate status**: `BLOCKED(no-verified-upstream-entrypoint: upstream-not-invocable-here)`
- **intended kernel**: `sgemm_lora_b_fwd`
- **intended substatus**: `upstream-not-invocable-here`
- **upstream exception**: `oracle` (-check subprocess produced no result (rc=-11): /Modules/main.c", line 429, in pymain_run_file
  File "/usr/local/src/con)
- **root cause category**: `sigsegv`
- **root cause detail**: Subprocess SIGSEGV under CPU interpreter (likely Triton interpreter bug).
- **kernel required params** (3): `x, weights, batch_info`
- **shapes.yaml declared inputs** (7): `batch_info, lora_ranks, scalings, seg_indptr, weight_indices, weights, x`
- **fix recipe**:
  - Phase B golden capture on GPU/NPU is the only authoritative path
  - On dev: try TRITON_INTERPRET=0 to skip interpreter path
- **effort**: `L`

#### `L2_12_awq_gemm`

- **aggregate status**: `BLOCKED(no-verified-upstream-entrypoint: shapes-incomplete)`
- **intended kernel**: `awq_gemm_triton`
- **intended substatus**: `upstream-not-invocable-here`
- **upstream exception**: `AssertionError` ()
- **root cause category**: `kernel-assertion`
- **root cause detail**: Kernel function `awq_gemm_triton` hits an AssertionError when called with the current shapes: ``
- **kernel required params** (5): `input, qweight, scales, qzeros, split_k_iters`
- **shapes.yaml declared inputs** (6): `group_size, input, qweight, qzeros, scales, split_k_iters`
- **fix recipe**:
  - Read the asserted condition in kernel.py
  - Adjust shapes.yaml so the input tensors satisfy the assertion
  - (e.g. paired dtype, divisibility, shape compatibility)
- **effort**: `M`

#### `L2_13_bmm`

- **aggregate status**: `BLOCKED(no-verified-upstream-entrypoint: shapes-incomplete)`
- **intended kernel**: `bmm_batch_invariant`
- **intended substatus**: `oracle-consistent`
- **upstream exception**: `S2` (-smoke MISMATCH (CPU/TRITON_INTERPRET pre-filter, phase=original-no-run): original side did not run (rc=-9): /data1/code)
- **root cause category**: `smoke-fail-severed-run`
- **root cause detail**: S2-smoke side process did not run (severed or original side rc!=0). Usually CPU interpreter cannot exec the kernel within budget.
- **kernel required params** (2): `a, b`
- **shapes.yaml declared inputs** (3): `a, b, input`
- **fix recipe**:
  - Reduce smoke shape coverage to H2 (smallest hidden shape)
  - Or skip CPU pre-filter for this task family
- **effort**: `M`

#### `L2_14_matmul_kernel_persistent`

- **aggregate status**: `BLOCKED(no-verified-upstream-entrypoint: shapes-incomplete)`
- **intended kernel**: `matmul_persistent`
- **intended substatus**: `upstream-not-invocable-here`
- **upstream exception**: `NotImplementedError` (num_compute_units is not implemented for the current platform.)
- **root cause category**: `platform-stub-noimpl`
- **root cause detail**: Upstream calls vllm.platforms.current_platform.num_compute_units() which raises NotImplementedError on the dev host's stub platform.
- **kernel required params** (2): `a, b`
- **shapes.yaml declared inputs** (4): `a, b, bias, input`
- **fix recipe**:
  - Option A: mock num_compute_units in tests/conftest.py or shim layer
  - Option B: rewrite oracle_pytorch.py to not depend on platform helpers
- **effort**: `M`

### family: `moe` (n=20)

#### `L1_05_gather_block_tables`

- **aggregate status**: `BLOCKED(no-verified-upstream-entrypoint: upstream-not-invocable-here)`
- **intended kernel**: `_gather_block_tables_kernel`
- **intended substatus**: `upstream-not-invocable-here`
- **upstream exception**: `InterpreterError` (TypeError("_gather_block_tables_kernel() missing 9 required positional arguments: 'batch_idx_to_req_idx', 'src_block_tab)
- **root cause category**: `upstream-interpretererror`
- **root cause detail**: Upstream raises InterpreterError: TypeError("_gather_block_tables_kernel() missing 9 required positional arguments: 'batch_idx_to_req_idx', 'src_block_table_ptrs', 'dst_block_table_ptrs', 'block_table_strides', 'num_blocks_ptr', 'num_
- **kernel required params** (9): `batch_idx_to_req_idx, src_block_table_ptrs, dst_block_table_ptrs, block_table_strides, num_blocks_ptr, num_blocks_stride, num_reqs, max_num_blocks, BLOCK_SIZE`
- **missing in shapes.yaml** (9): `batch_idx_to_req_idx, src_block_table_ptrs, dst_block_table_ptrs, block_table_strides, num_blocks_ptr, num_blocks_stride, num_reqs, max_num_blocks, BLOCK_SIZE`
- **shapes.yaml declared inputs** (2): `block_table, idx_mapping`
- **fix recipe**:
  - Inspect the exception's call site in upstream
  - Decide: shim, mock, or rewrite oracle_pytorch.py standalone
- **effort**: `M`

#### `L1_35_compute_src2dst_triton`

- **aggregate status**: `BLOCKED(no-verified-upstream-entrypoint: shapes-incomplete)`
- **intended kernel**: `cutlass_w4_run_moe_ep_preproess`
- **intended substatus**: `shapes-incomplete`
- **upstream exception**: `anchored` (on 1/2 shape(s) ['H2')
- **root cause category**: `missing-shape-row`
- **root cause detail**: Kernel function `cutlass_w4_run_moe_ep_preproess` requires param `topk_ids` (in its signature), but shapes.yaml `inputs:` block does not declare it. Resolver's by-name adapter stops at first missing param.
- **kernel required params** (1): `topk_ids`
- **missing in shapes.yaml** (1): `topk_ids`
- **shapes.yaml declared inputs** (6): `input_tensor, masked_m, num_toks, recv_x, reorder_ids, x`
- **fix recipe**:
  - Open tasks/L1_35_compute_src2dst_triton/shapes.yaml
  - Add `topk_ids:` entry under `inputs:` with the kernel's expected dtype + dims + dist
  - Refer to kernel.py signature / docstring for shape constraints
  - Re-run resolver to confirm advance to next stage (often more missing params surface in cascade)
- **effort**: `M`

#### `L1_36_compute_seg_indptr_triton`

- **aggregate status**: `BLOCKED(no-verified-upstream-entrypoint: upstream-not-invocable-here)`
- **intended kernel**: `moe_ep_deepgemm_preprocess`
- **intended substatus**: `upstream-not-invocable-here`
- **upstream exception**: `ExprError` (expression 'DIM3': unknown symbol 'DIM3' (allowed: ['B', 'C', 'D', 'Dv', 'E', 'G', 'H', 'L', 'N', 'T')
- **root cause category**: `upstream-exprerror`
- **root cause detail**: Upstream raises ExprError: expression 'DIM3': unknown symbol 'DIM3' (allowed: ['B', 'C', 'D', 'Dv', 'E', 'G', 'H', 'L', 'N', 'T'
- **kernel required params** (5): `topk_ids, num_local_experts, hidden_states, top_k, block_shape`
- **missing in shapes.yaml** (2): `top_k, block_shape`
- **shapes.yaml declared inputs** (9): `down_output, hidden_states, input, masked_m, num_experts, num_local_experts, output, problem_sizes1, topk_ids`
- **fix recipe**:
  - Inspect the exception's call site in upstream
  - Decide: shim, mock, or rewrite oracle_pytorch.py standalone
- **effort**: `M`

#### `L1_38_silu_mul_fp8_quant_deep_gemm`

- **aggregate status**: `BLOCKED(no-verified-upstream-entrypoint: upstream-not-invocable-here)`
- **intended kernel**: `persistent_masked_m_silu_mul_quant`
- **intended substatus**: `upstream-not-invocable-here`
- **upstream exception**: `AssertionError` ()
- **root cause category**: `kernel-assertion`
- **root cause detail**: Kernel function `persistent_masked_m_silu_mul_quant` hits an AssertionError when called with the current shapes: ``
- **kernel required params** (2): `y, tokens_per_expert`
- **shapes.yaml declared inputs** (4): `fp8_max, group_size, tokens_per_expert, y`
- **fix recipe**:
  - Read the asserted condition in kernel.py
  - Adjust shapes.yaml so the input tensors satisfy the assertion
  - (e.g. paired dtype, divisibility, shape compatibility)
- **effort**: `M`

#### `L2_10_w8a8_block_int8_matmul`

- **aggregate status**: `BLOCKED(no-verified-upstream-entrypoint: smoke-fail)`
- **upstream exception**: `Block` (-wise W8A8 INT8 GEMM oracle anchor passes locally on the two sampled shapes (H2 + P5) under TRITON_INTERPRET=1 with the )
- **root cause category**: `smoke-fail-cpu-budget`
- **root cause detail**: Resolver records the kernel as oracle-consistent but S2-smoke (CPU interpreter pre-filter) fails or exceeds budget on the dev host. The kernel runs correctly on GPU/NPU.
- **shapes.yaml declared inputs** (5): `A, As, B, Bs, block_size`
- **fix recipe**:
  - Phase B golden capture on GPU/NPU is the authoritative path
  - Optional: reduce smoke shape coverage to H2 to skip the worst row
- **effort**: `S`

#### `L2_25_fwd_kernel_ep_gather`

- **aggregate status**: `BLOCKED(no-verified-upstream-entrypoint: upstream-not-invocable-here)`
- **intended kernel**: `ep_gather`
- **intended substatus**: `upstream-not-invocable-here`
- **upstream exception**: `ExprError` (expression 'BN': unknown symbol 'BN' (allowed: ['B', 'C', 'D', 'Dv', 'E', 'G', 'H', 'L', 'N', 'T')
- **root cause category**: `upstream-exprerror`
- **root cause detail**: Upstream raises ExprError: expression 'BN': unknown symbol 'BN' (allowed: ['B', 'C', 'D', 'Dv', 'E', 'G', 'H', 'L', 'N', 'T'
- **kernel required params** (6): `input_tensor, recv_topk_ids, recv_topk_weight, input_index, expert_map, output_tensor`
- **missing in shapes.yaml** (4): `recv_topk_ids, recv_topk_weight, expert_map, output_tensor`
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
- **kernel required params** (10): `recv_x, recv_x_scale, recv_topk, num_recv_tokens_per_expert, expert_map, expert_start_loc, output_tensor, output_tensor_scale, m_indices, output_index`
- **missing in shapes.yaml** (8): `recv_topk, num_recv_tokens_per_expert, expert_map, expert_start_loc, output_tensor, output_tensor_scale, m_indices, output_index`
- **shapes.yaml declared inputs** (5): `dst_index, input_tensor, out_rows, recv_x, recv_x_scale`
- **fix recipe**:
  - Inspect the exception's call site in upstream
  - Decide: shim, mock, or rewrite oracle_pytorch.py standalone
- **effort**: `M`

#### `L2_27_deepep_permute_triton`

- **aggregate status**: `BLOCKED(no-verified-upstream-entrypoint: upstream-not-invocable-here)`
- **intended kernel**: `fp8_per_token_to_per_tensor_quant_triton`
- **intended substatus**: `upstream-not-invocable-here`
- **upstream exception**: `ExprError` (expression 'DIM3': unknown symbol 'DIM3' (allowed: ['B', 'C', 'D', 'Dv', 'E', 'G', 'H', 'L', 'N', 'T')
- **root cause category**: `upstream-exprerror`
- **root cause detail**: Upstream raises ExprError: expression 'DIM3': unknown symbol 'DIM3' (allowed: ['B', 'C', 'D', 'Dv', 'E', 'G', 'H', 'L', 'N', 'T'
- **kernel required params** (5): `x, x_scale, masked_m, output_scale, output`
- **missing in shapes.yaml** (1): `output_scale`
- **shapes.yaml declared inputs** (9): `down_output, input, masked_m, output, problem_sizes1, src2dst, topk, x, x_scale`
- **fix recipe**:
  - Inspect the exception's call site in upstream
  - Decide: shim, mock, or rewrite oracle_pytorch.py standalone
- **effort**: `M`

#### `L2_28_deepep_post_reorder_triton`

- **aggregate status**: `BLOCKED(no-verified-upstream-entrypoint: upstream-not-invocable-here)`
- **intended kernel**: `post_reorder_for_cutlass_moe`
- **intended substatus**: `upstream-not-invocable-here`
- **upstream exception**: `ExprError` (expression 'DIM3': unknown symbol 'DIM3' (allowed: ['B', 'C', 'D', 'Dv', 'E', 'G', 'H', 'L', 'N', 'T')
- **root cause category**: `upstream-exprerror`
- **root cause detail**: Upstream raises ExprError: expression 'DIM3': unknown symbol 'DIM3' (allowed: ['B', 'C', 'D', 'Dv', 'E', 'G', 'H', 'L', 'N', 'T'
- **kernel required params** (10): `down_output, output, src2dst, topk_ids, topk_weights, num_local_experts, topk, num_tokens, hidden_size, routed_scaling_factor`
- **missing in shapes.yaml** (5): `topk_ids, num_local_experts, num_tokens, hidden_size, routed_scaling_factor`
- **shapes.yaml declared inputs** (10): `down_output, input, masked_m, output, problem_sizes1, src2dst, topk, topk_weights, x, x_scale`
- **fix recipe**:
  - Inspect the exception's call site in upstream
  - Decide: shim, mock, or rewrite oracle_pytorch.py standalone
- **effort**: `M`

#### `L2_29_fused_moe_router_cudacore`

- **aggregate status**: `BLOCKED(no-verified-upstream-entrypoint: shapes-incomplete)`
- **intended kernel**: `fused_moe_router_cudacore`
- **intended substatus**: `oracle-mismatch`
- **upstream exception**: `anchored` (on 1/2 shape(s) ['H2')
- **root cause category**: `oracle-numeric-tolerance`
- **root cause detail**: Numeric divergence between kernel and oracle under the task's tolerance class. Likely CPU interpreter precision drift but may be real kernel bug.
- **kernel required params** (4): `x, router_weight, topk, moe_softcapping`
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
- **root cause detail**: Kernel function `fused_moe_router_tensorcore` requires param `BLOCK_SIZE_K` (in its signature), but shapes.yaml `inputs:` block does not declare it. Resolver's by-name adapter stops at first missing param.
- **kernel required params** (7): `x, router_weight, topk, moe_softcapping, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K`
- **shapes.yaml declared inputs** (8): `BLOCK_SIZE_K, BLOCK_SIZE_M, BLOCK_SIZE_N, correction_bias, moe_softcapping, router_weight, topk, x`
- **fix recipe**:
  - Open tasks/L2_30_fused_moe_router_tensorcore/shapes.yaml
  - Add `BLOCK_SIZE_K:` entry under `inputs:` with the kernel's expected dtype + dims + dist
  - Refer to kernel.py signature / docstring for shape constraints
  - Re-run resolver to confirm advance to next stage (often more missing params surface in cascade)
- **effort**: `S`

#### `L2_31_moe_align_block_size_stage2`

- **aggregate status**: `BLOCKED(no-verified-upstream-entrypoint: shapes-incomplete)`
- **intended kernel**: `moe_align_block_size_triton`
- **intended substatus**: `shapes-incomplete`
- **upstream exception**: `anchored` (on 1/2 shape(s) ['H2')
- **root cause category**: `missing-shape-row`
- **root cause detail**: Kernel function `moe_align_block_size_triton` requires param `sorted_token_ids` (in its signature), but shapes.yaml `inputs:` block does not declare it. Resolver's by-name adapter stops at first missing param.
- **kernel required params** (6): `topk_ids, num_experts, block_size, sorted_token_ids, expert_ids, num_tokens_post_pad`
- **missing in shapes.yaml** (3): `sorted_token_ids, expert_ids, num_tokens_post_pad`
- **shapes.yaml declared inputs** (3): `block_size, num_experts, topk_ids`
- **fix recipe**:
  - Open tasks/L2_31_moe_align_block_size_stage2/shapes.yaml
  - Add `sorted_token_ids:` entry under `inputs:` with the kernel's expected dtype + dims + dist
  - Refer to kernel.py signature / docstring for shape constraints
  - Re-run resolver to confirm advance to next stage (often more missing params surface in cascade)
- **effort**: `M`

#### `L2_32_experts_combine`

- **aggregate status**: `BLOCKED(no-verified-upstream-entrypoint: shapes-incomplete)`
- **intended kernel**: `experts_combine_triton`
- **intended substatus**: `upstream-not-invocable-here`
- **upstream exception**: `RuntimeError` (sglang::experts_combine_triton() is missing value for argument 'moe_hidden_states'. Declaration: sglang::experts_combine)
- **root cause category**: `upstream-runtimeerror`
- **root cause detail**: Upstream raises RuntimeError: sglang::experts_combine_triton() is missing value for argument 'moe_hidden_states'. Declaration: sglang::experts_combine_triton(Tensor moe_hidden_states, Tensor mlp_hidden_states, Tensor? output_buffe
- **kernel required params** (2): `moe_hidden_states, mlp_hidden_states`
- **shapes.yaml declared inputs** (6): `hidden_states, mlp_hidden_states, moe_hidden_states, residual, softcap_const, x`
- **fix recipe**:
  - Inspect the exception's call site in upstream
  - Decide: shim, mock, or rewrite oracle_pytorch.py standalone
- **effort**: `M`

#### `L2_34_silu_mul_static_tensorwise_quant_triton_kernel_for_cutlass_moe`

- **aggregate status**: `BLOCKED(no-verified-upstream-entrypoint: upstream-not-invocable-here)`
- **intended kernel**: `silu_and_mul_masked_post_per_tensor_quant_fwd`
- **intended substatus**: `upstream-not-invocable-here`
- **upstream exception**: `ExprError` (expression 'DIM3': unknown symbol 'DIM3' (allowed: ['B', 'C', 'D', 'Dv', 'E', 'G', 'H', 'L', 'N', 'T')
- **root cause category**: `upstream-exprerror`
- **root cause detail**: Upstream raises ExprError: expression 'DIM3': unknown symbol 'DIM3' (allowed: ['B', 'C', 'D', 'Dv', 'E', 'G', 'H', 'L', 'N', 'T'
- **kernel required params** (4): `input, output, masked_m, scale`
- **shapes.yaml declared inputs** (7): `down_output, input, masked_m, output, problem_sizes1, scale, x`
- **fix recipe**:
  - Inspect the exception's call site in upstream
  - Decide: shim, mock, or rewrite oracle_pytorch.py standalone
- **effort**: `M`

#### `L2_35_pre_reorder_triton_kernel_for_cutlass_moe`

- **aggregate status**: `BLOCKED(no-verified-upstream-entrypoint: shapes-incomplete)`
- **intended kernel**: `pre_reorder_for_cutlass_moe`
- **intended substatus**: `shapes-incomplete`
- **upstream exception**: `anchored` (on 1/2 shape(s) ['H2')
- **root cause category**: `missing-shape-row`
- **root cause detail**: Kernel function `pre_reorder_for_cutlass_moe` requires param `gateup_input` (in its signature), but shapes.yaml `inputs:` block does not declare it. Resolver's by-name adapter stops at first missing param.
- **kernel required params** (9): `input, gateup_input, src2dst, topk_ids, a1_scales, num_local_experts, topk, num_tokens, hidden_size`
- **missing in shapes.yaml** (6): `gateup_input, topk_ids, a1_scales, num_local_experts, num_tokens, hidden_size`
- **shapes.yaml declared inputs** (7): `down_output, input, masked_m, output, src2dst, topk, x`
- **fix recipe**:
  - Open tasks/L2_35_pre_reorder_triton_kernel_for_cutlass_moe/shapes.yaml
  - Add `gateup_input:` entry under `inputs:` with the kernel's expected dtype + dims + dist
  - Refer to kernel.py signature / docstring for shape constraints
  - Re-run resolver to confirm advance to next stage (often more missing params surface in cascade)
- **effort**: `M`

#### `L2_36_post_reorder_triton_kernel_for_cutlass_moe`

- **aggregate status**: `BLOCKED(no-verified-upstream-entrypoint: shapes-incomplete)`
- **intended kernel**: `post_reorder_for_cutlass_moe`
- **intended substatus**: `shapes-incomplete`
- **upstream exception**: `anchored` (on 1/2 shape(s) ['H2')
- **root cause category**: `missing-shape-row`
- **root cause detail**: Kernel function `post_reorder_for_cutlass_moe` requires param `topk_ids` (in its signature), but shapes.yaml `inputs:` block does not declare it. Resolver's by-name adapter stops at first missing param.
- **kernel required params** (10): `down_output, output, src2dst, topk_ids, topk_weights, num_local_experts, topk, num_tokens, hidden_size, routed_scaling_factor`
- **missing in shapes.yaml** (5): `topk_ids, num_local_experts, num_tokens, hidden_size, routed_scaling_factor`
- **shapes.yaml declared inputs** (9): `down_output, input, input_tensor, masked_m, output, src2dst, topk, topk_weights, x`
- **fix recipe**:
  - Open tasks/L2_36_post_reorder_triton_kernel_for_cutlass_moe/shapes.yaml
  - Add `topk_ids:` entry under `inputs:` with the kernel's expected dtype + dims + dist
  - Refer to kernel.py signature / docstring for shape constraints
  - Re-run resolver to confirm advance to next stage (often more missing params surface in cascade)
- **effort**: `M`

#### `L3_09_fused_moe`

- **aggregate status**: `BLOCKED(no-verified-upstream-entrypoint: upstream-not-invocable-here)`
- **intended kernel**: `invoke_fused_moe_triton_kernel`
- **intended substatus**: `upstream-not-invocable-here`
- **upstream exception**: `RuntimeError` (Tried to register an operator (vllm::inplace_fused_experts(Tensor(a0!) hidden_states, Tensor w1, Tensor w2, Tensor topk_)
- **root cause category**: `duplicate-op-registration`
- **root cause detail**: Upstream import triggers torch.library duplicate operator registration -- another vendored copy registered the same op first.
- **kernel required params** (18): `A, B, C, A_scale, B_scale, topk_weights, sorted_token_ids, expert_ids, num_tokens_post_padded, mul_routed_weight, top_k, config, compute_type, use_fp8_w8a8, use_int8_w8a8, use_int8_w8a16, use_int4_w4a16, per_channel_quant`
- **missing in shapes.yaml** (15): `A, B, C, A_scale, B_scale, num_tokens_post_padded, mul_routed_weight, top_k, config, compute_type, use_fp8_w8a8, use_int8_w8a8, use_int8_w8a16, use_int4_w4a16, per_channel_quant`
- **shapes.yaml declared inputs** (7): `a, apply_weight, b, block_size, expert_ids, sorted_token_ids, topk_weights`
- **fix recipe**:
  - Identify the duplicate registration path (usually two vllm copies)
  - Option A: dedupe sys.path so only one vllm import wins
  - Option B: oracle_pytorch.py rewrite to not import vllm
- **effort**: `L`

#### `L3_10_fused_moe_kernel_gptq_awq`

- **aggregate status**: `BLOCKED(no-verified-upstream-entrypoint: upstream-not-invocable-here)`
- **intended kernel**: `invoke_fused_moe_wna16_triton_kernel`
- **intended substatus**: `upstream-not-invocable-here`
- **upstream exception**: `RuntimeError` (Tried to register an operator (vllm::inplace_fused_experts(Tensor(a0!) hidden_states, Tensor w1, Tensor w2, Tensor topk_)
- **root cause category**: `duplicate-op-registration`
- **root cause detail**: Upstream import triggers torch.library duplicate operator registration -- another vendored copy registered the same op first.
- **kernel required params** (16): `A, B, C, B_scale, B_zp, topk_weights, sorted_token_ids, expert_ids, num_tokens_post_padded, mul_routed_weight, top_k, config, compute_type, use_int8_w8a16, use_int4_w4a16, block_shape`
- **missing in shapes.yaml** (13): `A, B, C, B_scale, B_zp, num_tokens_post_padded, mul_routed_weight, top_k, config, compute_type, use_int8_w8a16, use_int4_w4a16, block_shape`
- **shapes.yaml declared inputs** (7): `a, apply_weight, b, block_size, expert_ids, sorted_token_ids, topk_weights`
- **fix recipe**:
  - Identify the duplicate registration path (usually two vllm copies)
  - Option A: dedupe sys.path so only one vllm import wins
  - Option B: oracle_pytorch.py rewrite to not import vllm
- **effort**: `L`

#### `L3_11_expert_triton`

- **aggregate status**: `BLOCKED(no-verified-upstream-entrypoint: shapes-incomplete)`
- **intended kernel**: `invoke_moe_batched_triton_kernel`
- **intended substatus**: `shapes-incomplete`
- **upstream exception**: `anchored` (on 1/2 shape(s) ['H2')
- **root cause category**: `missing-shape-row`
- **root cause detail**: Kernel function `invoke_moe_batched_triton_kernel` requires param `A` (in its signature), but shapes.yaml `inputs:` block does not declare it. Resolver's by-name adapter stops at first missing param.
- **kernel required params** (13): `A, B, C, expert_num_tokens, compute_type, A_scale, B_scale, B_zp, use_fp8_w8a8, use_int8_w8a16, use_int4_w4a16, config, per_act_token_quant`
- **missing in shapes.yaml** (12): `A, B, C, compute_type, A_scale, B_scale, B_zp, use_fp8_w8a8, use_int8_w8a16, use_int4_w4a16, config, per_act_token_quant`
- **shapes.yaml declared inputs** (3): `a, b, expert_num_tokens`
- **fix recipe**:
  - Open tasks/L3_11_expert_triton/shapes.yaml
  - Add `A:` entry under `inputs:` with the kernel's expected dtype + dims + dist
  - Refer to kernel.py signature / docstring for shape constraints
  - Re-run resolver to confirm advance to next stage (often more missing params surface in cascade)
- **effort**: `M`

#### `L3_12_moe_mmk`

- **aggregate status**: `BLOCKED(no-verified-upstream-entrypoint: shapes-incomplete)`
- **intended kernel**: `invoke_moe_batched_triton_kernel`
- **intended substatus**: `shapes-incomplete`
- **upstream exception**: `anchored` (on 1/2 shape(s) ['H2')
- **root cause category**: `missing-shape-row`
- **root cause detail**: Kernel function `invoke_moe_batched_triton_kernel` requires param `A` (in its signature), but shapes.yaml `inputs:` block does not declare it. Resolver's by-name adapter stops at first missing param.
- **kernel required params** (13): `A, B, C, expert_num_tokens, compute_type, A_scale, B_scale, B_zp, use_fp8_w8a8, use_int8_w8a16, use_int4_w4a16, config, per_act_token_quant`
- **missing in shapes.yaml** (12): `A, B, C, compute_type, A_scale, B_scale, B_zp, use_fp8_w8a8, use_int8_w8a16, use_int4_w4a16, config, per_act_token_quant`
- **shapes.yaml declared inputs** (3): `a, b, expert_num_tokens`
- **fix recipe**:
  - Open tasks/L3_12_moe_mmk/shapes.yaml
  - Add `A:` entry under `inputs:` with the kernel's expected dtype + dims + dist
  - Refer to kernel.py signature / docstring for shape constraints
  - Re-run resolver to confirm advance to next stage (often more missing params surface in cascade)
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
- **kernel required params** (4): `x, weight, bias, eps`
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
- **kernel required params** (4): `x, weight, bias, eps`
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
- **root cause detail**: Kernel function `fused_dual_residual_rmsnorm` requires param `weight2` (in its signature), but shapes.yaml `inputs:` block does not declare it. Resolver's by-name adapter stops at first missing param.
- **kernel required params** (5): `x, residual, weight1, weight2, eps`
- **shapes.yaml declared inputs** (10): `eps, hidden_states, residual, residual_a, residual_b, softcap_const, weight, weight1, weight2, x`
- **fix recipe**:
  - Open tasks/L2_15_fused_dual_residual_rmsnorm/shapes.yaml
  - Add `weight2:` entry under `inputs:` with the kernel's expected dtype + dims + dist
  - Refer to kernel.py signature / docstring for shape constraints
  - Re-run resolver to confirm advance to next stage (often more missing params surface in cascade)
- **effort**: `M`

#### `L2_16_fused_rmsnorm`

- **aggregate status**: `BLOCKED(no-verified-upstream-entrypoint: oracle-mismatch)`
- **intended kernel**: `fused_rmsnorm`
- **intended substatus**: `oracle-mismatch`
- **upstream exception**: `anchored` (on 1/2 shape(s) ['P1')
- **root cause category**: `oracle-dtype-mismatch`
- **root cause detail**: Output dtype divergence between kernel and oracle.
- **kernel required params** (3): `x, weight, eps`
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
- **kernel required params** (4): `x, g, weight, bias`
- **missing in shapes.yaml** (2): `weight, bias`
- **shapes.yaml declared inputs** (8): `b, eps, g, k, q, residual, w, x`
- **fix recipe**:
  - Pin Triton version OR rewrite kernel.py decorator ordering
  - If oracle_pytorch.py is independent, defer kernel.py fix to Phase B
- **effort**: `L`

### family: `other` (n=4)

#### `L1_28_pack_bitmatrix`

- **aggregate status**: `BLOCKED(no-verified-upstream-entrypoint: shapes-incomplete)`
- **intended kernel**: `make_routing_data`
- **intended substatus**: `shapes-incomplete`
- **upstream exception**: `anchored` (on 1/2 shape(s) ['H2')
- **root cause category**: `missing-shape-row`
- **root cause detail**: Kernel function `make_routing_data` requires param `topk_weights` (in its signature), but shapes.yaml `inputs:` block does not declare it. Resolver's by-name adapter stops at first missing param.
- **kernel required params** (3): `topk_ids, topk_weights, num_local_experts`
- **missing in shapes.yaml** (2): `topk_weights, num_local_experts`
- **shapes.yaml declared inputs** (2): `num_experts, topk_ids`
- **fix recipe**:
  - Open tasks/L1_28_pack_bitmatrix/shapes.yaml
  - Add `topk_weights:` entry under `inputs:` with the kernel's expected dtype + dims + dist
  - Refer to kernel.py signature / docstring for shape constraints
  - Re-run resolver to confirm advance to next stage (often more missing params surface in cascade)
- **effort**: `M`

#### `L3_05_fwd_kernel_alibi`

- **aggregate status**: `BLOCKED(no-verified-upstream-entrypoint: shapes-incomplete)`
- **intended kernel**: `context_attention_fwd`
- **intended substatus**: `shapes-incomplete`
- **upstream exception**: `anchored` (on 1/2 shape(s) ['H2')
- **root cause category**: `missing-shape-row`
- **root cause detail**: Kernel function `context_attention_fwd` requires param `kv_cache_dtype` (in its signature), but shapes.yaml `inputs:` block does not declare it. Resolver's by-name adapter stops at first missing param.
- **kernel required params** (14): `q, k, v, o, kv_cache_dtype, k_cache, v_cache, b_loc, b_start_loc, b_seq_len, max_seq_len, max_input_len, k_scale, v_scale`
- **missing in shapes.yaml** (10): `kv_cache_dtype, k_cache, v_cache, b_loc, b_start_loc, b_seq_len, max_seq_len, max_input_len, k_scale, v_scale`
- **shapes.yaml declared inputs** (6): `alibi_slopes, k, o, q, scale, v`
- **fix recipe**:
  - Open tasks/L3_05_fwd_kernel_alibi/shapes.yaml
  - Add `kv_cache_dtype:` entry under `inputs:` with the kernel's expected dtype + dims + dist
  - Refer to kernel.py signature / docstring for shape constraints
  - Re-run resolver to confirm advance to next stage (often more missing params surface in cascade)
- **effort**: `M`

#### `L3_06_fwd_grouped_kernel_stage1`

- **aggregate status**: `BLOCKED(no-verified-upstream-entrypoint: shapes-incomplete)`
- **intended kernel**: `_decode_grouped_att_m_fwd`
- **intended substatus**: `shapes-incomplete`
- **upstream exception**: `anchored` (on 1/2 shape(s) ['H2')
- **root cause category**: `missing-shape-row`
- **root cause detail**: Kernel function `_decode_grouped_att_m_fwd` requires param `Req_to_tokens` (in its signature), but shapes.yaml `inputs:` block does not declare it. Resolver's by-name adapter stops at first missing param.
- **kernel required params** (12): `q, k_buffer, v_buffer, att_out, Req_to_tokens, B_Seqlen, num_kv_splits, sm_scale, page_size, logit_cap, k_scale, v_scale`
- **missing in shapes.yaml** (8): `Req_to_tokens, B_Seqlen, num_kv_splits, sm_scale, page_size, logit_cap, k_scale, v_scale`
- **shapes.yaml declared inputs** (5): `att_out, k_buffer, q, scale, v_buffer`
- **fix recipe**:
  - Open tasks/L3_06_fwd_grouped_kernel_stage1/shapes.yaml
  - Add `Req_to_tokens:` entry under `inputs:` with the kernel's expected dtype + dims + dist
  - Refer to kernel.py signature / docstring for shape constraints
  - Re-run resolver to confirm advance to next stage (often more missing params surface in cascade)
- **effort**: `M`

#### `L3_20_recompute_w_u_fwd`

- **aggregate status**: `BLOCKED(no-verified-upstream-entrypoint: upstream-not-invocable-here)`
- **intended kernel**: `recompute_w_u_fwd`
- **intended substatus**: `upstream-not-invocable-here`
- **upstream exception**: `AttributeError` ('_AutotunedKernel' object has no attribute 'arg_names')
- **root cause category**: `triton-autotune-api-mismatch`
- **root cause detail**: Kernel applies @triton.heuristics on top of a wrapper that lacks .arg_names. Current Triton API expects different ordering.
- **kernel required params** (6): `k, v, beta, g_cumsum, A, cu_seqlens`
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
- **root cause category**: `missing-shape-row`
- **root cause detail**: Kernel function `_dequantize_k_cache_fast` requires param `quant_k_cache` (in its signature), but shapes.yaml `inputs:` block does not declare it. Resolver's by-name adapter stops at first missing param.
- **kernel required params** (1): `quant_k_cache`
- **missing in shapes.yaml** (1): `quant_k_cache`
- **shapes.yaml declared inputs** (4): `group_size, nope_q, nope_s, rope`
- **fix recipe**:
  - Open tasks/L1_07_dequantize_k_cache_fast/shapes.yaml
  - Add `quant_k_cache:` entry under `inputs:` with the kernel's expected dtype + dims + dist
  - Refer to kernel.py signature / docstring for shape constraints
  - Re-run resolver to confirm advance to next stage (often more missing params surface in cascade)
- **effort**: `M`

#### `L1_09_quantize_k_cache_fast`

- **aggregate status**: `BLOCKED(no-verified-upstream-entrypoint: oracle-mismatch)`
- **intended kernel**: `_quantize_k_cache_fast`
- **intended substatus**: `upstream-not-invocable-here`
- **upstream exception**: `AssertionError` ()
- **root cause category**: `kernel-assertion`
- **root cause detail**: Kernel function `_quantize_k_cache_fast` hits an AssertionError when called with the current shapes: ``
- **kernel required params** (2): `k_nope, k_rope`
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
- **kernel required params** (2): `x, group_size`
- **shapes.yaml declared inputs** (3): `eps, group_size, x`
- **fix recipe**:
  - Re-run on GPU to distinguish interpreter drift from real divergence
  - If real, audit oracle_pytorch.py against kernel.py mathematically
- **effort**: `M`

#### `L1_11_awq_dequantize`

- **aggregate status**: `BLOCKED(no-verified-upstream-entrypoint: shapes-incomplete)`
- **intended kernel**: `awq_dequantize_triton`
- **intended substatus**: `upstream-not-invocable-here`
- **upstream exception**: `AssertionError` ()
- **root cause category**: `kernel-assertion`
- **root cause detail**: Kernel function `awq_dequantize_triton` hits an AssertionError when called with the current shapes: ``
- **kernel required params** (3): `qweight, scales, zeros`
- **shapes.yaml declared inputs** (5): `group_size, input, qweight, scales, zeros`
- **fix recipe**:
  - Read the asserted condition in kernel.py
  - Adjust shapes.yaml so the input tensors satisfy the assertion
  - (e.g. paired dtype, divisibility, shape compatibility)
- **effort**: `M`

#### `L1_30_act_quant`

- **aggregate status**: `BLOCKED(no-verified-upstream-entrypoint: shapes-incomplete)`
- **intended kernel**: `act_quant`
- **intended substatus**: `oracle-mismatch`
- **upstream exception**: `anchored` (on 1/2 shape(s) ['H2')
- **root cause category**: `oracle-numeric-tolerance`
- **root cause detail**: Numeric divergence between kernel and oracle under the task's tolerance class. Likely CPU interpreter precision drift but may be real kernel bug.
- **kernel required params** (1): `x`
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
- **kernel required params** (6): `masked_m, problem_sizes1, problem_sizes2, n, k, num_experts`
- **missing in shapes.yaml** (1): `problem_sizes2`
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
- **root cause category**: `missing-shape-row`
- **root cause detail**: Kernel function `w8a8_triton_block_scaled_mm` requires param `block_size` (in its signature), but shapes.yaml `inputs:` block does not declare it. Resolver's by-name adapter stops at first missing param.
- **kernel required params** (5): `A, B, As, Bs, block_size`
- **missing in shapes.yaml** (1): `block_size`
- **shapes.yaml declared inputs** (8): `A, As, B, Bs, group_k, group_n, input, x`
- **fix recipe**:
  - Open tasks/L2_11_w8a8_triton_block_scaled_mm/shapes.yaml
  - Add `block_size:` entry under `inputs:` with the kernel's expected dtype + dims + dist
  - Refer to kernel.py signature / docstring for shape constraints
  - Re-run resolver to confirm advance to next stage (often more missing params surface in cascade)
- **effort**: `M`

#### `L2_37_silu_and_mul_post_quant`

- **aggregate status**: `BLOCKED(no-verified-upstream-entrypoint: upstream-not-invocable-here)`
- **intended kernel**: `silu_and_mul_masked_post_quant_fwd`
- **intended substatus**: `upstream-not-invocable-here`
- **upstream exception**: `ExprError` (expression 'DIM3': unknown symbol 'DIM3' (allowed: ['B', 'C', 'D', 'Dv', 'E', 'G', 'H', 'L', 'N', 'T')
- **root cause category**: `upstream-exprerror`
- **root cause detail**: Upstream raises ExprError: expression 'DIM3': unknown symbol 'DIM3' (allowed: ['B', 'C', 'D', 'Dv', 'E', 'G', 'H', 'L', 'N', 'T'
- **kernel required params** (5): `input, output, output_scale, quant_group_size, masked_m`
- **missing in shapes.yaml** (1): `output_scale`
- **shapes.yaml declared inputs** (7): `fp8_max, input, masked_m, output, problem_sizes1, quant_group_size, x`
- **fix recipe**:
  - Inspect the exception's call site in upstream
  - Decide: shim, mock, or rewrite oracle_pytorch.py standalone
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
- **intended kernel**: `compute_token_logprobs`
- **intended substatus**: `upstream-not-invocable-here`
- **upstream exception**: `ModuleNotFoundError` (No module named 'uvloop')
- **root cause category**: `import-error`
- **root cause detail**: Upstream import chain fails: ModuleNotFoundError for `uvloop`.
- **kernel required params** (2): `logits, token_ids`
- **shapes.yaml declared inputs** (3): `logits, num_logprobs, token_ids`
- **fix recipe**:
  - Option A: `pip install uvloop` (if pure-Python, low risk)
  - Option B: oracle_pytorch.py implement a standalone reference (no upstream import)
  - Option C: GPU-only -- defer to Phase B if module is CUDA-bound
- **effort**: `M`

#### `L1_20_log_softmax`

- **aggregate status**: `BLOCKED(no-verified-upstream-entrypoint: shapes-incomplete)`
- **intended kernel**: `log_softmax`
- **intended substatus**: `oracle-mismatch`
- **upstream exception**: `anchored` (on 2/2 shape(s) ['H2', 'H4')
- **root cause category**: `oracle-numeric-tolerance`
- **root cause detail**: Numeric divergence between kernel and oracle under the task's tolerance class. Likely CPU interpreter precision drift but may be real kernel bug.
- **kernel required params** (1): `input`
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
- **root cause category**: `missing-shape-row`
- **root cause detail**: Kernel function `gumbel_sample` requires param `logits` (in its signature), but shapes.yaml `inputs:` block does not declare it. Resolver's by-name adapter stops at first missing param.
- **kernel required params** (6): `logits, expanded_idx_mapping, temperature, seed, pos, apply_temperature`
- **missing in shapes.yaml** (6): `logits, expanded_idx_mapping, temperature, seed, pos, apply_temperature`
- **fix recipe**:
  - Open tasks/L1_21_gumbel_sample/shapes.yaml
  - Add `logits:` entry under `inputs:` with the kernel's expected dtype + dims + dist
  - Refer to kernel.py signature / docstring for shape constraints
  - Re-run resolver to confirm advance to next stage (often more missing params surface in cascade)
- **effort**: `M`

#### `L1_22_min_p`

- **aggregate status**: `BLOCKED(no-verified-upstream-entrypoint: oracle-mismatch)`
- **intended kernel**: `apply_min_p`
- **intended substatus**: `oracle-mismatch`
- **upstream exception**: `anchored` (on 1/2 shape(s) ['H2')
- **root cause category**: `oracle-structure-mismatch`
- **root cause detail**: Kernel output and oracle disagree on output structure (shape, dtype, or operator semantics).
- **kernel required params** (3): `logits, expanded_idx_mapping, min_p`
- **shapes.yaml declared inputs** (3): `expanded_idx_mapping, logits, min_p`
- **fix recipe**:
  - Read kernel.py + oracle_pytorch.py side by side
  - Identify which output dim/dtype the oracle expects vs what kernel produces
  - Fix oracle_pytorch.py to match the kernel's true output contract
- **effort**: `M`

#### `L1_23_penalties_and_temperature`

- **aggregate status**: `BLOCKED(no-verified-upstream-entrypoint: upstream-not-invocable-here)`
- **intended kernel**: `apply_penalties`
- **intended substatus**: `upstream-not-invocable-here`
- **upstream exception**: `ExprError` (expression 'V': unknown symbol 'V' (allowed: ['B', 'C', 'D', 'Dv', 'E', 'G', 'H', 'L', 'N', 'T')
- **root cause category**: `upstream-exprerror`
- **root cause detail**: Upstream raises ExprError: expression 'V': unknown symbol 'V' (allowed: ['B', 'C', 'D', 'Dv', 'E', 'G', 'H', 'L', 'N', 'T'
- **kernel required params** (9): `logits, expanded_idx_mapping, token_ids, expanded_local_pos, repetition_penalty, frequency_penalty, presence_penalty, prompt_bin_mask, output_bin_counts`
- **missing in shapes.yaml** (2): `expanded_local_pos, output_bin_counts`
- **shapes.yaml declared inputs** (9): `all_token_ids, expanded_idx_mapping, frequency_penalty, logits, output_token_counts, presence_penalty, prompt_bin_mask, repetition_penalty, token_ids`
- **fix recipe**:
  - Inspect the exception's call site in upstream
  - Decide: shim, mock, or rewrite oracle_pytorch.py standalone
- **effort**: `M`

#### `L1_24_rejection_sample`

- **aggregate status**: `BLOCKED(no-verified-upstream-entrypoint: upstream-not-invocable-here)`
- **intended kernel**: `rejection_sample`
- **intended substatus**: `upstream-not-invocable-here`
- **upstream exception**: `ModuleNotFoundError` (No module named 'uvloop')
- **root cause category**: `import-error`
- **root cause detail**: Upstream import chain fails: ModuleNotFoundError for `uvloop`.
- **kernel required params** (8): `draft_token_ids, num_draft_tokens, max_spec_len, cu_num_draft_tokens, draft_probs, target_logits, bonus_token_ids, sampling_metadata`
- **missing in shapes.yaml** (2): `target_logits, sampling_metadata`
- **shapes.yaml declared inputs** (11): `bonus_token_ids, cu_num_draft_tokens, draft_probs, draft_token_ids, is_greedy, max_spec_len, num_draft_tokens, recovered_token_ids, target_probs, uniform_probs, x`
- **fix recipe**:
  - Option A: `pip install uvloop` (if pure-Python, low risk)
  - Option B: oracle_pytorch.py implement a standalone reference (no upstream import)
  - Option C: GPU-only -- defer to Phase B if module is CUDA-bound
- **effort**: `M`

#### `L1_27_apply_grammar_bitmask`

- **aggregate status**: `BLOCKED(no-verified-upstream-entrypoint: upstream-not-invocable-here)`
- **intended kernel**: `_apply_grammar_bitmask_kernel`
- **intended substatus**: `upstream-not-invocable-here`
- **upstream exception**: `InterpreterError` (TypeError("_apply_grammar_bitmask_kernel() missing 7 required positional arguments: 'logits_ptr', 'logits_stride', 'logi)
- **root cause category**: `upstream-interpretererror`
- **root cause detail**: Upstream raises InterpreterError: TypeError("_apply_grammar_bitmask_kernel() missing 7 required positional arguments: 'logits_ptr', 'logits_stride', 'logits_indices_ptr', 'bitmask_ptr', 'bitmask_stride', 'vocab_size', and 'BLOCK_SIZE'
- **kernel required params** (7): `logits_ptr, logits_stride, logits_indices_ptr, bitmask_ptr, bitmask_stride, vocab_size, BLOCK_SIZE`
- **missing in shapes.yaml** (7): `logits_ptr, logits_stride, logits_indices_ptr, bitmask_ptr, bitmask_stride, vocab_size, BLOCK_SIZE`
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
- **root cause category**: `missing-shape-row`
- **root cause detail**: Kernel function `prepare_eagle_inputs` requires param `input_buffers` (in its signature), but shapes.yaml `inputs:` block does not declare it. Resolver's by-name adapter stops at first missing param.
- **kernel required params** (9): `last_token_indices, current_draft_step, input_buffers, input_batch, num_sampled, num_rejected, last_sampled, next_prefill_tokens, max_num_reqs`
- **missing in shapes.yaml** (5): `input_buffers, input_batch, last_sampled, next_prefill_tokens, max_num_reqs`
- **shapes.yaml declared inputs** (4): `current_draft_step, last_token_indices, num_rejected, num_sampled`
- **fix recipe**:
  - Open tasks/L1_29_prepare_eagle_inputs/shapes.yaml
  - Add `input_buffers:` entry under `inputs:` with the kernel's expected dtype + dims + dist
  - Refer to kernel.py signature / docstring for shape constraints
  - Re-run resolver to confirm advance to next stage (often more missing params surface in cascade)
- **effort**: `M`

### family: `ssm` (n=16)

#### `L1_39_state_passing_fwd`

- **aggregate status**: `BLOCKED(no-verified-upstream-entrypoint: shapes-incomplete)`
- **intended kernel**: `_state_passing_fwd`
- **intended substatus**: `shapes-incomplete`
- **upstream exception**: `anchored` (on 1/2 shape(s) ['H2')
- **root cause category**: `missing-shape-row`
- **root cause detail**: Kernel function `_state_passing_fwd` requires param `last_chunk_indices` (in its signature), but shapes.yaml `inputs:` block does not declare it. Resolver's by-name adapter stops at first missing param.
- **kernel required params** (3): `states, dA_cumsum, last_chunk_indices`
- **missing in shapes.yaml** (1): `last_chunk_indices`
- **shapes.yaml declared inputs** (2): `dA_cumsum, states`
- **fix recipe**:
  - Open tasks/L1_39_state_passing_fwd/shapes.yaml
  - Add `last_chunk_indices:` entry under `inputs:` with the kernel's expected dtype + dims + dist
  - Refer to kernel.py signature / docstring for shape constraints
  - Re-run resolver to confirm advance to next stage (often more missing params surface in cascade)
- **effort**: `M`

#### `L1_40_chunk_cumsum_fwd`

- **aggregate status**: `BLOCKED(no-verified-upstream-entrypoint: shapes-incomplete)`
- **intended kernel**: `_chunk_cumsum_fwd`
- **intended substatus**: `shapes-incomplete`
- **upstream exception**: `anchored` (on 1/2 shape(s) ['H2')
- **root cause category**: `missing-shape-row`
- **root cause detail**: Kernel function `_chunk_cumsum_fwd` requires param `cu_chunk_seqlens` (in its signature), but shapes.yaml `inputs:` block does not declare it. Resolver's by-name adapter stops at first missing param.
- **kernel required params** (4): `dt, A, chunk_size, cu_chunk_seqlens`
- **missing in shapes.yaml** (1): `cu_chunk_seqlens`
- **shapes.yaml declared inputs** (4): `A, chunk_size, dt, dt_softplus`
- **fix recipe**:
  - Open tasks/L1_40_chunk_cumsum_fwd/shapes.yaml
  - Add `cu_chunk_seqlens:` entry under `inputs:` with the kernel's expected dtype + dims + dist
  - Refer to kernel.py signature / docstring for shape constraints
  - Re-run resolver to confirm advance to next stage (often more missing params surface in cascade)
- **effort**: `M`

#### `L2_02_chunk_scan_fwd`

- **aggregate status**: `BLOCKED(no-verified-upstream-entrypoint: shapes-incomplete)`
- **intended kernel**: `_chunk_scan_fwd`
- **intended substatus**: `shapes-incomplete`
- **upstream exception**: `anchored` (on 1/2 shape(s) ['H2')
- **root cause category**: `missing-shape-row`
- **root cause detail**: Kernel function `_chunk_scan_fwd` requires param `cu_chunk_seqlens` (in its signature), but shapes.yaml `inputs:` block does not declare it. Resolver's by-name adapter stops at first missing param.
- **kernel required params** (9): `cb, x, dt, dA_cumsum, C, states, cu_chunk_seqlens, out, seq_idx`
- **missing in shapes.yaml** (3): `cu_chunk_seqlens, out, seq_idx`
- **shapes.yaml declared inputs** (8): `C, D, cb, dA_cumsum, dt, states, x, z`
- **fix recipe**:
  - Open tasks/L2_02_chunk_scan_fwd/shapes.yaml
  - Add `cu_chunk_seqlens:` entry under `inputs:` with the kernel's expected dtype + dims + dist
  - Refer to kernel.py signature / docstring for shape constraints
  - Re-run resolver to confirm advance to next stage (often more missing params surface in cascade)
- **effort**: `M`

#### `L2_03_chunk_state_fwd`

- **aggregate status**: `BLOCKED(no-verified-upstream-entrypoint: shapes-incomplete)`
- **intended kernel**: `_chunk_state_fwd`
- **intended substatus**: `shapes-incomplete`
- **upstream exception**: `anchored` (on 1/2 shape(s) ['H2')
- **root cause category**: `missing-shape-row`
- **root cause detail**: Kernel function `_chunk_state_fwd` requires param `cu_chunk_seqlens` (in its signature), but shapes.yaml `inputs:` block does not declare it. Resolver's by-name adapter stops at first missing param.
- **kernel required params** (5): `B, x, dt, dA_cumsum, cu_chunk_seqlens`
- **missing in shapes.yaml** (1): `cu_chunk_seqlens`
- **shapes.yaml declared inputs** (4): `B, dA_cumsum, dt, x`
- **fix recipe**:
  - Open tasks/L2_03_chunk_state_fwd/shapes.yaml
  - Add `cu_chunk_seqlens:` entry under `inputs:` with the kernel's expected dtype + dims + dist
  - Refer to kernel.py signature / docstring for shape constraints
  - Re-run resolver to confirm advance to next stage (often more missing params surface in cascade)
- **effort**: `M`

#### `L2_18_fused_gdn_gating`

- **aggregate status**: `BLOCKED(no-verified-upstream-entrypoint: upstream-not-invocable-here)`
- **intended kernel**: `fused_gdn_gating`
- **intended substatus**: `upstream-not-invocable-here`
- **upstream exception**: `AttributeError` ('_AutotunedKernel' object has no attribute 'arg_names')
- **root cause category**: `triton-autotune-api-mismatch`
- **root cause detail**: Kernel applies @triton.heuristics on top of a wrapper that lacks .arg_names. Current Triton API expects different ordering.
- **kernel required params** (4): `A_log, a, b, dt_bias`
- **missing in shapes.yaml** (1): `b`
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
- **root cause detail**: Kernel function `fused_sigmoid_gating_delta_rule_update` requires param `softplus_threshold` (in its signature), but shapes.yaml `inputs:` block does not declare it. Resolver's by-name adapter stops at first missing param.
- **kernel required params** (11): `A_log, a, dt_bias, softplus_beta, softplus_threshold, q, k, v, b, initial_state_source, initial_state_indices`
- **missing in shapes.yaml** (4): `softplus_threshold, b, initial_state_source, initial_state_indices`
- **shapes.yaml declared inputs** (9): `A_log, a, beta, dt_bias, k, q, softplus_beta, state, v`
- **fix recipe**:
  - Open tasks/L2_19_fused_sigmoid_gating_delta_rule_update/shapes.yaml
  - Add `softplus_threshold:` entry under `inputs:` with the kernel's expected dtype + dims + dist
  - Refer to kernel.py signature / docstring for shape constraints
  - Re-run resolver to confirm advance to next stage (often more missing params surface in cascade)
- **effort**: `M`

#### `L3_03_chunk_state_varlen`

- **aggregate status**: `BLOCKED(no-verified-upstream-entrypoint: shapes-incomplete)`
- **intended kernel**: `chunk_state_varlen`
- **intended substatus**: `shapes-incomplete`
- **upstream exception**: `anchored` (on 1/2 shape(s) ['H2')
- **root cause category**: `missing-shape-row`
- **root cause detail**: Kernel function `chunk_state_varlen` requires param `chunk_states` (in its signature), but shapes.yaml `inputs:` block does not declare it. Resolver's by-name adapter stops at first missing param.
- **kernel required params** (6): `B, x, dt, dA_cumsum, cu_seqlens, chunk_states`
- **missing in shapes.yaml** (1): `chunk_states`
- **shapes.yaml declared inputs** (5): `B, cu_seqlens, dA_cumsum, dt, x`
- **fix recipe**:
  - Open tasks/L3_03_chunk_state_varlen/shapes.yaml
  - Add `chunk_states:` entry under `inputs:` with the kernel's expected dtype + dims + dist
  - Refer to kernel.py signature / docstring for shape constraints
  - Re-run resolver to confirm advance to next stage (often more missing params surface in cascade)
- **effort**: `M`

#### `L3_13_chunk_gated_delta_rule_fwd_kernel_h_blockdim64`

- **aggregate status**: `BLOCKED(no-verified-upstream-entrypoint: upstream-not-invocable-here)`
- **intended kernel**: `chunk_gated_delta_rule_fwd_h`
- **intended substatus**: `upstream-not-invocable-here`
- **upstream exception**: `AttributeError` ('_AutotunedKernel' object has no attribute 'arg_names')
- **root cause category**: `triton-autotune-api-mismatch`
- **root cause detail**: Kernel applies @triton.heuristics on top of a wrapper that lacks .arg_names. Current Triton API expects different ordering.
- **kernel required params** (3): `k, w, u`
- **shapes.yaml declared inputs** (4): `g, k, u, w`
- **fix recipe**:
  - Pin Triton version OR rewrite kernel.py decorator ordering
  - If oracle_pytorch.py is independent, defer kernel.py fix to Phase B
- **effort**: `L`

#### `L3_14_fused_recurrent_gated_delta_rule_fwd`

- **aggregate status**: `BLOCKED(no-verified-upstream-entrypoint: shapes-incomplete)`
- **intended kernel**: `fused_recurrent_gated_delta_rule_fwd`
- **intended substatus**: `shapes-incomplete`
- **upstream exception**: `anchored` (on 1/2 shape(s) ['H2')
- **root cause category**: `missing-shape-row`
- **root cause detail**: Kernel function `fused_recurrent_gated_delta_rule_fwd` requires param `initial_state` (in its signature), but shapes.yaml `inputs:` block does not declare it. Resolver's by-name adapter stops at first missing param.
- **kernel required params** (7): `q, k, v, g, beta, scale, initial_state`
- **missing in shapes.yaml** (1): `initial_state`
- **shapes.yaml declared inputs** (6): `beta, g, k, q, scale, v`
- **fix recipe**:
  - Open tasks/L3_14_fused_recurrent_gated_delta_rule_fwd/shapes.yaml
  - Add `initial_state:` entry under `inputs:` with the kernel's expected dtype + dims + dist
  - Refer to kernel.py signature / docstring for shape constraints
  - Re-run resolver to confirm advance to next stage (often more missing params surface in cascade)
- **effort**: `M`

#### `L3_15_fused_recurrent_gated_delta_rule_update_fwd`

- **aggregate status**: `BLOCKED(no-verified-upstream-entrypoint: shapes-incomplete)`
- **intended kernel**: `fused_recurrent_gated_delta_rule_update_fwd`
- **intended substatus**: `shapes-incomplete`
- **upstream exception**: `anchored` (on 1/2 shape(s) ['H2')
- **root cause category**: `missing-shape-row`
- **root cause detail**: Kernel function `fused_recurrent_gated_delta_rule_update_fwd` requires param `initial_state_source` (in its signature), but shapes.yaml `inputs:` block does not declare it. Resolver's by-name adapter stops at first missing param.
- **kernel required params** (8): `q, k, v, g, beta, scale, initial_state_source, initial_state_indices`
- **missing in shapes.yaml** (2): `initial_state_source, initial_state_indices`
- **shapes.yaml declared inputs** (6): `beta, g, k, q, scale, v`
- **fix recipe**:
  - Open tasks/L3_15_fused_recurrent_gated_delta_rule_update_fwd/shapes.yaml
  - Add `initial_state_source:` entry under `inputs:` with the kernel's expected dtype + dims + dist
  - Refer to kernel.py signature / docstring for shape constraints
  - Re-run resolver to confirm advance to next stage (often more missing params surface in cascade)
- **effort**: `M`

#### `L3_16_chunk_fwd_kernel_o`

- **aggregate status**: `BLOCKED(no-verified-upstream-entrypoint: upstream-not-invocable-here)`
- **intended kernel**: `chunk_fwd_o`
- **intended substatus**: `upstream-not-invocable-here`
- **upstream exception**: `AttributeError` ('_AutotunedKernel' object has no attribute 'arg_names')
- **root cause category**: `triton-autotune-api-mismatch`
- **root cause detail**: Kernel applies @triton.heuristics on top of a wrapper that lacks .arg_names. Current Triton API expects different ordering.
- **kernel required params** (4): `q, k, v, h`
- **shapes.yaml declared inputs** (6): `g, h, k, q, scale, v`
- **fix recipe**:
  - Pin Triton version OR rewrite kernel.py decorator ordering
  - If oracle_pytorch.py is independent, defer kernel.py fix to Phase B
- **effort**: `L`

#### `L3_17_chunk_gla_fwd_kernel_o`

- **aggregate status**: `BLOCKED(no-verified-upstream-entrypoint: upstream-not-invocable-here)`
- **intended kernel**: `chunk_gla_fwd_o_gk`
- **intended substatus**: `upstream-not-invocable-here`
- **upstream exception**: `AttributeError` ('_AutotunedKernel' object has no attribute 'arg_names')
- **root cause category**: `triton-autotune-api-mismatch`
- **root cause detail**: Kernel applies @triton.heuristics on top of a wrapper that lacks .arg_names. Current Triton API expects different ordering.
- **kernel required params** (7): `q, v, g, A, h, o, scale`
- **shapes.yaml declared inputs** (7): `A, g, h, o, q, scale, v`
- **fix recipe**:
  - Pin Triton version OR rewrite kernel.py decorator ordering
  - If oracle_pytorch.py is independent, defer kernel.py fix to Phase B
- **effort**: `L`

#### `L3_18_chunk_scaled_dot_kkt_fwd`

- **aggregate status**: `BLOCKED(no-verified-upstream-entrypoint: upstream-not-invocable-here)`
- **intended kernel**: `chunk_scaled_dot_kkt_fwd`
- **intended substatus**: `upstream-not-invocable-here`
- **upstream exception**: `AttributeError` ('_AutotunedKernel' object has no attribute 'arg_names')
- **root cause category**: `triton-autotune-api-mismatch`
- **root cause detail**: Kernel applies @triton.heuristics on top of a wrapper that lacks .arg_names. Current Triton API expects different ordering.
- **kernel required params** (1): `k`
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
- **kernel required params** (2): `q, k`
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
- **kernel required params** (10): `A, Ai, cu_seqlens, chunk_indices, T, H, BT, USE_TMA, IS_VARLEN, DOT_PRECISION`
- **missing in shapes.yaml** (9): `Ai, cu_seqlens, chunk_indices, T, H, BT, USE_TMA, IS_VARLEN, DOT_PRECISION`
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
- **kernel required params** (2): `g, chunk_size`
- **shapes.yaml declared inputs** (3): `chunk_size, g, reverse`
- **fix recipe**:
  - Pin Triton version OR rewrite kernel.py decorator ordering
  - If oracle_pytorch.py is independent, defer kernel.py fix to Phase B
- **effort**: `L`
