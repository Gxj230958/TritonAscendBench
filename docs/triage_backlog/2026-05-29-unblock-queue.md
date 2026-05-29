# Unblock queue 2026-05-29

Note: repo-backed late-session tally is `13 PASS / 87 remaining`; `L2_12_awq_gemm` is now included in PASS after the AWQ wrapper-contract alignment landed.

## Verified PASS (13 repo-backed)
- `L1_01_bias` -- S2 GPU PASS; bit-exact baseline.
- `L1_03_convert_req_index_to_global_index` -- S2 GPU PASS after entrypoint/meta alignment.
- `L1_11_awq_dequantize` -- S2 GPU PASS after AWQ wrapper contract alignment.
- `L1_12_rms_norm` -- S2 GPU PASS; bit-exact baseline.
- `L2_12_awq_gemm` -- S2 GPU PASS after reusing the `L1_11` AWQ pattern (`int32`, `N//32`, `split_k_iters`, H2 bump).
- `L1_16_silu_and_mul` -- S2 GPU PASS; bit-exact baseline.
- `L1_22_min_p` -- S2 GPU PASS after oracle-equivalence fix.
- `L1_26_merge_state` -- S2 GPU PASS; baseline merge-state contract held.
- `L1_30_act_quant` -- S2 GPU PASS after restoring missing `fp8_max` contract.
- `L1_34_transform_index_page_table_decode` -- S2 GPU PASS after real entrypoint + `page_size == 1` contract alignment.
- `L2_01_bmm_chunk_fwd` -- S2 GPU PASS; GPU-authoritative rerun cleared the earlier smoke false-negative.
- `L2_10_w8a8_block_int8_matmul` -- S2 GPU PASS after `block_size=[8,32]` / ceil-div K-group fix.
- `L3_04_fwd` -- S2 GPU PASS after shapes/golden alignment.

## Blocker class breakdown (87 remaining, repo-backed; buckets below sum to 85 plus 2 architectural skips)

| blocker_class | count | owner | next_unblocker |
|---|---:|---|---|
| `venv-vllm-C` | 41 | user | build `scripts/loop/setup_vllm_venv.sh`, then restart supervisor with `--phase-python` |
| `sglang-class-A` | 29 | bench-team | Class-A pool / import-surface architecture decision, then rerun `resolve`/`verify` |
| `fp8-packed-dist` | 5 | bench-team | add packed-fp8 / int-packed generator support in `build_inputs.py` |
| `structural-oracle-rewrite` | 5 | per-task careful work | side-by-side oracle/kernel rewrite; budget `30-60 min` per task |
| `misc` | 5 | codex/operator | case-by-case follow-up after env/Class-A lanes are cleared |

## Post-venv batch plan (executable)

Once `vllm_venv` is ready and the daemon supervisor is restarted with `--phase-python`:

1. Run the deep batch:

```bash
python3 scripts/loop/orchestrator.py run --tasks \
L1_02_compute_slot_mappings \
L1_04_copy_page_indices \
L1_05_gather_block_tables \
L1_06_reshape_and_cache_kernel_flash \
L1_10_per_token_group_quant_int8 \
L1_19_topk_log_softmax \
L1_20_log_softmax \
L1_23_penalties_and_temperature \
L1_24_rejection_sample \
L1_25_merge_attn_states \
L1_27_apply_grammar_bitmask \
L1_28_pack_bitmatrix \
L1_29_prepare_eagle_inputs \
L1_38_silu_mul_fp8_quant_deep_gemm \
L1_39_state_passing_fwd \
L1_40_chunk_cumsum_fwd \
L2_02_chunk_scan_fwd \
L2_03_chunk_state_fwd \
L2_11_w8a8_triton_block_scaled_mm \
L2_13_bmm \
L2_14_matmul_kernel_persistent \
L2_17_layer_norm_gated_fwd \
L2_18_fused_gdn_gating \
L2_20_causal_conv1d_update \
L2_21_causal_conv1d_fwd \
L2_22_linear_attn_decode \
L2_23_fwd_kv_parallel \
L2_24_fwd_kv_reduce \
L2_25_fwd_kernel_ep_gather \
L2_26_fwd_kernel_ep_scatter_1 \
L3_01_kernel_unified_attention_3d \
L3_02_kernel_paged_attention_2d \
L3_05_fwd_kernel_alibi \
L3_06_fwd_grouped_kernel_stage1 \
L3_07_correct_attn_cp_out \
L3_09_fused_moe \
L3_10_fused_moe_kernel_gptq_awq \
L3_11_expert_triton \
L3_12_moe_mmk \
L3_17_chunk_gla_fwd_kernel_o \
L3_19_chunk_kda_scaled_dot_kkt_fwd_kernel_intra_sub_inter \
--timeout 1800
```

2. Expected first rerun outcome:
- each task should either `PASS` or surface its real local blocker (`shapes`, `oracle`, `meta`, runtime assert);
- no task in this 41-task batch should remain blocked on `ModuleNotFoundError: vllm._C` / `original-no-run`;
- rerun `resolve` immediately on any task that drops from env-blocked to contract-blocked, then queue only the narrowed remainder for manual edits.

3. Copy-paste task list only:

```text
L1_02_compute_slot_mappings L1_04_copy_page_indices L1_05_gather_block_tables L1_06_reshape_and_cache_kernel_flash L1_10_per_token_group_quant_int8 L1_19_topk_log_softmax L1_20_log_softmax L1_23_penalties_and_temperature L1_24_rejection_sample L1_25_merge_attn_states L1_27_apply_grammar_bitmask L1_28_pack_bitmatrix L1_29_prepare_eagle_inputs L1_38_silu_mul_fp8_quant_deep_gemm L1_39_state_passing_fwd L1_40_chunk_cumsum_fwd L2_02_chunk_scan_fwd L2_03_chunk_state_fwd L2_11_w8a8_triton_block_scaled_mm L2_13_bmm L2_14_matmul_kernel_persistent L2_17_layer_norm_gated_fwd L2_18_fused_gdn_gating L2_20_causal_conv1d_update L2_21_causal_conv1d_fwd L2_22_linear_attn_decode L2_23_fwd_kv_parallel L2_24_fwd_kv_reduce L2_25_fwd_kernel_ep_gather L2_26_fwd_kernel_ep_scatter_1 L3_01_kernel_unified_attention_3d L3_02_kernel_paged_attention_2d L3_05_fwd_kernel_alibi L3_06_fwd_grouped_kernel_stage1 L3_07_correct_attn_cp_out L3_09_fused_moe L3_10_fused_moe_kernel_gptq_awq L3_11_expert_triton L3_12_moe_mmk L3_17_chunk_gla_fwd_kernel_o L3_19_chunk_kda_scaled_dot_kkt_fwd_kernel_intra_sub_inter
```

## Tasks to manually prepare during venv build

- `L2_36_post_reorder_triton_kernel_for_cutlass_moe` -- same as `L2_35`, plus the extra `input`/`input_tensor` surface tensors.

## Known dead-ends (don't retry)

- `vllm._C` empty stub -- today `FAIL`: Class-A unvendorable; defer-to-golden / real built extension is the canonical path.
- `sglang` shim extension -- today `FAIL`: hardcoded CUDA-only behavior violates U3 inert rule; symbols are Class-A.
- `L2_19` oracle rewrite -- today: structural drift goes beyond rename; `q/k/v` shapes need `(B,T,H,K)`, not `(B,H,D)`.

## Late-session update 22:00-23:30 PST

- `L2_12_awq_gemm` is now PASS via the `L1_11_awq_dequantize` reuse pattern (`int32`, `N//32`, `split_k_iters`, H2 bump); true tally is `13 PASS / 87 remaining`.
- `L1_25_merge_attn_states` is architecturally blocked: `kernel.py:28` returns `None`, the Phase A' GPU runner rejects `None`, validator/resolver require `kernel.py::...`, and bench rules forbid editing the byte-frozen wrapper.
- `L1_10_per_token_group_quant_int8` is architecturally blocked in the current bench flow: wrapper gates on `current_platform.is_cuda()`, the shim lacks `is_cuda`, and the CUDA branch still requires built `vllm._C`.
- Rebucketed from this pass: `L1_06_reshape_and_cache_kernel_flash` moves to `venv-vllm-C`; `L1_36`, `L1_37`, `L2_31`, `L2_33`, and `L2_35` move to `sglang-class-A`.
