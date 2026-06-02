# Per-task dtype alignment audit — 103 tasks vs vLLM/SGLang (FINAL, codex-arbitrated)

**Status: review only — NO task modified. Codex (gpt-5.5:xhigh) arbitrated the 35-agent
trace; 5 rows corrected. Date 2026-06-01.**

## Headline counts (codex-corrected)

| bucket | n | gate-safe / faithful? |
|---|---:|---|
| ALIGNED (exact match) | 29 | yes — already production dtype |
| FORCED_FP32_TOLERANCE | 3 | yes — production IS fp32 (state-passing/solve-tril/merge) |
| FORCED_FP32_INPLACE | 7 | yes — in-place norm, tight tol forces fp32 both ends |
| REAL_BUG (raw-dtype tl.dot) | 33 | fp32 fixture runs the kernel's fp32 dot path, production runs bf16 dot |
| INPUT_BF16_GAP | 31 | kernel upcasts loads; bf16 input would be gate-safe (cosmetic) |

- **Exactly dtype-aligned: 29/103.**
- **Aligned OR intentionally/forced fp32 (gate-safe & faithful): 39/103.**
- Not exactly aligned: 64/103 (33 different-dot-precision-path + 31 cheap input relabels).

## What this DOES and does NOT mean (the key reconciliation)

The bench's GPU verification has two gates, and BOTH are fully satisfied at fp32:
1. **bit-exact severance** — original frozen kernel vs severed kernel, SAME byte-exact code,
   so they agree bit-for-bit at ANY dtype. Extraction fidelity is dtype-INDEPENDENT.
2. **oracle-consistency at fp32** — kernel vs independent PyTorch reference; passing proves the
   kernel LOGIC is correct (the kernel code is dtype-generic).

So the 33 `REAL_BUG` rows are NOT correctness bugs. They are tasks where a `tl.dot` consumes
the raw input tensors, so the fp32 fixture exercises the kernel's fp32-precision dot while
production uses bf16 — SAME algorithm/code, lower input mantissa. Two facts make this immaterial
for the GPU stage:
- the bench daemon is **CUDA, not 910B**; bf16 numerics on NPU are a hardware property that CUDA
  cannot reproduce, so the production-precision path is the **910B stage's** job regardless of the
  GPU fixture dtype;
- a faithful bf16 test would also need a **bf16-aware oracle** (replicating bf16 tensor-core dot),
  not just relabeled inputs — so fp32 + fp32-oracle is the correct way to verify these kernels'
  LOGIC on GPU.

**Bottom line: no correctness problem. fp32 fully verifies what the GPU stage CAN verify
(extraction + logic). Dtype-vs-production is a 910B-stage coverage note, not a bug.**

## Codex's 5 corrections to the agent pass
- L1_07 / L1_08 dequantize_k_cache: UNKNOWN -> ALIGNED (production cache IS fp8_e4m3 + bf16 out; bench matches).
- L2_23 / L2_24 lightning-attn: INPUT_BF16_GAP -> REAL_BUG (KV-parallel stage raw-loads k_trans/v into tl.dot).
- L2_30 fused_moe_router_tensorcore: REAL_BUG -> INPUT_BF16_GAP (over-flagged; promotes both operands to fp32 before dot).


## ALIGNED (29)

| task | conf | gaps (bench->prod) |
|---|---|---|
| L1_02_compute_slot_mappings | confirmed |  |
| L1_03_convert_req_index_to_global_index | confirmed |  |
| L1_04_copy_page_indices | confirmed |  |
| L1_05_gather_block_tables | likely |  |
| L1_07_dequantize_k_cache_fast *(codex)* | speculative | k_cache:various→fp8_e4m3 |
| L1_08_dequantize_k_cache_paged *(codex)* | speculative | k_cache:various→fp8_e4m3 |
| L1_09_quantize_k_cache_fast | confirmed |  |
| L1_18_fused_softcap | likely |  |
| L1_19_topk_log_softmax | confirmed |  |
| L1_20_log_softmax | likely |  |
| L1_21_gumbel_sample | likely |  |
| L1_22_min_p | confirmed |  |
| L1_23_penalties_and_temperature | confirmed |  |
| L1_24_rejection_sample | likely |  |
| L1_27_apply_grammar_bitmask | likely |  |
| L1_28_pack_bitmatrix | likely |  |
| L1_29_prepare_eagle_inputs | likely |  |
| L1_32_create_flashinfer_kv_indices_triton | confirmed |  |
| L1_33_create_flashmla_kv_indices_triton | confirmed |  |
| L1_34_transform_index_page_table_decode | confirmed |  |
| L1_35_compute_src2dst_triton | confirmed |  |
| L1_36_compute_seg_indptr_triton | confirmed |  |
| L1_37_compute_problem_sizes_w4a8 | likely |  |
| L2_10_w8a8_block_int8_matmul | confirmed |  |
| L2_11_w8a8_triton_block_scaled_mm | confirmed |  |
| L2_26_fwd_kernel_ep_scatter_1 | likely |  |
| L2_31_moe_align_block_size_stage2 | likely |  |
| L2_39_scaled_mm | confirmed |  |
| L2_41_selective_state_update | confirmed |  |

## FORCED_FP32_TOLERANCE (3)

| task | conf | gaps (bench->prod) |
|---|---|---|
| L1_39_state_passing_fwd | confirmed |  |
| L3_21_solve_tril_16x16 | confirmed |  |
| L3_23_merge_16x16_to_32x32_inverse | confirmed |  |

## FORCED_FP32_INPLACE (7)

| task | conf | gaps (bench->prod) |
|---|---|---|
| L1_01_bias | confirmed |  |
| L2_15_fused_dual_residual_rmsnorm | likely | x:float32→bfloat16, residual:float32→bfloat16, weight1:float32→bfloat16, weight2:float32→bfloat16 |
| L2_16_fused_rmsnorm | likely | x:float32→bfloat16, weight:float32→bfloat16 |
| L2_17_layer_norm_gated_fwd | likely | x:float32→bfloat16, g:float32→bfloat16, w:float32→bfloat16, residual:float32→bfloat16 |
| L2_20_causal_conv1d_update | likely | x:float32→bfloat16, conv_state:float32→bfloat16, weight:float32→bfloat16, bias:float32→bfloat16 |
| L2_40_gemma_qkv_rmsnorm | confirmed | q:float32→bfloat16, k:float32→bfloat16, v:float32→bfloat16, q_weight:float32→bfloat16 |
| L3_07_correct_attn_cp_out | likely | out:float32→bfloat16 |

## REAL_BUG (33)

| task | conf | gaps (bench->prod) |
|---|---|---|
| L1_06_reshape_and_cache_kernel_flash | confirmed | key:float32→bfloat16, value:float32→bfloat16, key_cache:float32→bfloat16, value_cache:float32→bfloat16 |
| L1_38_silu_mul_fp8_quant_deep_gemm | likely | y:float32→bfloat16 |
| L2_01_bmm_chunk_fwd | confirmed | a:float32→bfloat16, b:float32→bfloat16 |
| L2_02_chunk_scan_fwd | confirmed | x:float32→bfloat16, C:float32→bfloat16 |
| L2_03_chunk_state_fwd | confirmed | B:float32→bfloat16, x:float32→bfloat16, dt:float32→bfloat16 |
| L2_04_chunked_lora_expand | likely | x:float32→bfloat16, weights:float32→bfloat16 |
| L2_05_chunked_lora_shrink | likely | x:float32→bfloat16, weights:float32→bfloat16 |
| L2_06_qkv_lora_b | likely | x:float32→bfloat16, qkv_lora_b:float32→bfloat16 |
| L2_07_gate_up_lora_b | likely | x:float32→bfloat16, gate_up_lora_b:float32→bfloat16 |
| L2_08_sgemm_lora_a | likely | x:float32→bfloat16, weights:float32→bfloat16 |
| L2_09_sgemm_lora_b | likely | x:float32→bfloat16, weights:float32→bfloat16 |
| L2_12_awq_gemm | likely | input:float32→float16, scales:float32→float16 |
| L2_13_bmm | likely | a:float32→bfloat16, b:float32→bfloat16 |
| L2_14_matmul_kernel_persistent | likely | a:float32→bfloat16, b:float32→bfloat16, bias:float32→bfloat16 |
| L2_23_fwd_kv_parallel *(codex)* | confirmed | q:float32→bfloat16, k:float32→bfloat16, v:float32→bfloat16 |
| L2_24_fwd_kv_reduce *(codex)* | confirmed | q:float32→bfloat16, k:float32→bfloat16, v:float32→bfloat16 |
| L2_33_fwd_grouped_kernel_stage1_rope | likely | q:float32→bfloat16, k_buffer:float32→bfloat16, v_buffer:float32→bfloat16 |
| L3_01_kernel_unified_attention_3d | confirmed | q:float32→bfloat16, k:float32→bfloat16, v:float32→bfloat16 |
| L3_02_kernel_paged_attention_2d | confirmed | query:float32→bfloat16, key_cache:float32→bfloat16, value_cache:float32→bfloat16 |
| L3_03_chunk_state_varlen | confirmed | B:float32→bfloat16, x:float32→bfloat16, dt:float32→bfloat16 |
| L3_04_fwd | confirmed | q:float32→bfloat16, k:float32→bfloat16, v:float32→bfloat16 |
| L3_05_fwd_kernel_alibi | likely | q:float32→bfloat16, k:float32→bfloat16, v:float32→bfloat16 |
| L3_06_fwd_grouped_kernel_stage1 | likely | q:float32→bfloat16, k_buffer:float32→bfloat16, v_buffer:float32→bfloat16 |
| L3_09_fused_moe | confirmed | a:float32→bfloat16, b:float32→bfloat16 |
| L3_10_fused_moe_kernel_gptq_awq | confirmed | a:float32→bfloat16, b_scale:float32→float16 |
| L3_11_expert_triton | confirmed | a:float32→bfloat16, b:float32→bfloat16 |
| L3_12_moe_mmk | confirmed | a:float32→bfloat16, b:float32→bfloat16 |
| L3_13_chunk_gated_delta_rule_fwd_kernel_h_blockdim64 | confirmed | k:float32→bfloat16, w:float32→bfloat16, u:float32→bfloat16 |
| L3_16_chunk_fwd_kernel_o | confirmed | q:float32→bfloat16, k:float32→bfloat16, v:float32→bfloat16 |
| L3_17_chunk_gla_fwd_kernel_o | likely | q:float32→bfloat16, v:float32→bfloat16 |
| L3_18_chunk_scaled_dot_kkt_fwd | likely | k:float32→bfloat16 |
| L3_19_chunk_kda_scaled_dot_kkt_fwd_kernel_intra_sub_inter | likely | q:float32→bfloat16, k:float32→bfloat16 |
| L3_20_recompute_w_u_fwd | likely | k:float32→bfloat16, v:float32→bfloat16 |

## INPUT_BF16_GAP (31)

| task | conf | gaps (bench->prod) |
|---|---|---|
| L1_10_per_token_group_quant_int8 | likely | x:float32→bfloat16 |
| L1_11_awq_dequantize | likely | scales:float32→float16 |
| L1_12_rms_norm | likely | input:float32→bfloat16, weight:float32→bfloat16 |
| L1_13_layer_norm_fwd | likely | x:float32→bfloat16, weight:float32→bfloat16, bias:float32→bfloat16 |
| L1_14_layer_norm_fwd_1pass | likely | x:float32→bfloat16, weight:float32→bfloat16, bias:float32→bfloat16 |
| L1_15_act_and_mul | likely | gateup_output:float32→bfloat16 |
| L1_16_silu_and_mul | confirmed | hidden_states:float32→bfloat16 |
| L1_17_gelu_and_mul | likely | hidden_states:float32→bfloat16 |
| L1_25_merge_attn_states | likely | prefix_output:float32→bfloat16, suffix_output:float32→bfloat16, output:float32→bfloat16 |
| L1_26_merge_state | likely | prefix_output:float32→bfloat16, suffix_output:float32→bfloat16 |
| L1_30_act_quant | likely | x:float32→bfloat16 |
| L1_31_triton_mrope_forward | likely | q:float32→bfloat16, k:float32→bfloat16 |
| L1_40_chunk_cumsum_fwd | likely | dt:float32→bfloat16 |
| L2_18_fused_gdn_gating | likely | a:float32→bfloat16, b:float32→bfloat16 |
| L2_19_fused_sigmoid_gating_delta_rule_update | confirmed | a:float32→bfloat16, q:float32→bfloat16, k:float32→bfloat16, v:float32→bfloat16 |
| L2_21_causal_conv1d_fwd | likely | x:float32→bfloat16, weight:float32→bfloat16, bias:float32→bfloat16 |
| L2_22_linear_attn_decode | confirmed | q:float32→bfloat16, k:float32→bfloat16, v:float32→bfloat16 |
| L2_25_fwd_kernel_ep_gather | likely | input_tensor:float32→bfloat16 |
| L2_27_deepep_permute_triton | likely | x:float32→bfloat16 |
| L2_28_deepep_post_reorder_triton | likely | x:float32→bfloat16 |
| L2_29_fused_moe_router_cudacore | speculative | x:float32→bfloat16, router_weight:float32→bfloat16 |
| L2_30_fused_moe_router_tensorcore *(codex)* | speculative | x:float32→bfloat16, router_weight:float32→bfloat16 |
| L2_32_experts_combine | likely | moe_hidden_states:float32→bfloat16, mlp_hidden_states:float32→bfloat16 |
| L2_34_silu_mul_static_tensorwise_quant_triton_kernel_for_cutlass_moe | likely | input:float32→bfloat16 |
| L2_35_pre_reorder_triton_kernel_for_cutlass_moe | likely | input:float32→bfloat16 |
| L2_36_post_reorder_triton_kernel_for_cutlass_moe | likely | down_output:float32→bfloat16 |
| L2_37_silu_and_mul_post_quant | likely | input:float32→bfloat16 |
| L2_38_concat_and_cast_mha_k | likely | k_nope:float32→bfloat16, k_rope:float32→bfloat16 |
| L3_14_fused_recurrent_gated_delta_rule_fwd | confirmed | q:float32→bfloat16, k:float32→bfloat16, v:float32→bfloat16 |
| L3_15_fused_recurrent_gated_delta_rule_update_fwd | likely | q:float32→bfloat16, k:float32→bfloat16, v:float32→bfloat16 |
| L3_22_chunk_local_cumsum_vector | confirmed | g:float32→bfloat16 |