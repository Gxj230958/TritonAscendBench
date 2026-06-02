---
date: 2026-05-27
tool: scripts/apply_oracle_equivalence_verdict.py
purpose: V9 M2.1c — apply R10 GPU verdict to meta.yaml provenance.golden
verdict-dir: /data1/code/exp_data/tritonbench/r10_verdict
---

# Oracle Equivalence Verdict Apply Report

Tasks processed: **32**  
Promoted to truth source (`oracle_pytorch_verified`): **4**

## Overall verdict tally

| overall | count |
|---|---|
| `LAUNCH_BLOCKED` | 15 |
| `FAILED` | 12 |
| `VERIFIED` | 4 |
| `MIXED` | 1 |

## VERIFIED (4)

### `L1_12_rms_norm`
- **oracle_pytorch.py 升 truth source**
- per-shape numeric diff:
  - `P1`: max_abs=0.000e+00, max_rel=0.000e+00
  - `P2`: max_abs=4.768e-07, max_rel=2.409e-07

### `L1_16_silu_and_mul`
- **oracle_pytorch.py 升 truth source**
- per-shape numeric diff:
  - `P1`: max_abs=1.192e-07, max_rel=2.356e-07
  - `P2`: max_abs=9.537e-07, max_rel=3.519e-07

### `L1_26_merge_state`
- **oracle_pytorch.py 升 truth source**
- per-shape numeric diff:
  - `P1`: max_abs=4.768e-07, max_rel=4.512e-05
  - `P2`: max_abs=4.768e-07, max_rel=1.490e-03

### `L1_34_transform_index_page_table_decode`
- **oracle_pytorch.py 升 truth source**
- per-shape numeric diff:
  - `P1`: max_abs=0.000e+00, max_rel=0.000e+00
  - `P2`: max_abs=0.000e+00, max_rel=0.000e+00

## FAILED (12)

### `L1_01_bias`
- per-shape numeric diff:
  - `P1`: 
  - `P2`: 

### `L1_06_reshape_and_cache_kernel_flash`
- per-shape numeric diff:
  - `P1`: 
  - `P2`: 

### `L1_10_per_token_group_quant_int8`
- per-shape numeric diff:
  - `P1`: max_abs=inf, max_rel=inf
  - `P2`: max_abs=inf, max_rel=inf

### `L1_20_log_softmax`
- per-shape numeric diff:
  - `P1`: max_abs=7.308e+00, max_rel=1.155e+00
  - `P2`: max_abs=1.010e+01, max_rel=1.047e+00

### `L1_22_min_p`
- per-shape numeric diff:
  - `P1`: 
  - `P2`: 

### `L1_25_merge_attn_states`
- per-shape numeric diff:
  - `P1`: 
  - `P2`: 

### `L1_31_triton_mrope_forward`
- per-shape numeric diff:
  - `P1`: max_abs=inf, max_rel=inf
  - `P2`: max_abs=inf, max_rel=inf

### `L1_36_compute_seg_indptr_triton`
- per-shape numeric diff:
  - `P1`: max_abs=inf, max_rel=inf
  - `P2`: max_abs=inf, max_rel=inf

### `L2_01_bmm_chunk_fwd`
- per-shape numeric diff:
  - `P1`: max_abs=1.727e-02, max_rel=3.521e-01
  - `P2`: max_abs=4.243e+01, max_rel=4.243e+13

### `L2_29_fused_moe_router_cudacore`
- per-shape numeric diff:
  - `P1`: max_abs=6.004e-02, max_rel=8.947e-02
  - `P2`: max_abs=1.378e-01, max_rel=1.918e-01

### `L2_30_fused_moe_router_tensorcore`
- per-shape numeric diff:
  - `P1`: max_abs=6.004e-02, max_rel=8.947e-02
  - `P2`: max_abs=1.378e-01, max_rel=1.918e-01

### `L3_04_fwd`
- per-shape numeric diff:
  - `P1`: 
  - `P2`: 

## MIXED (1)

### `L1_19_topk_log_softmax`
- per-shape numeric diff:
  - `P1`: max_abs=0.000e+00, max_rel=0.000e+00
  - `P2`: max_abs=4.768e-07, max_rel=8.697e-08

## LAUNCH_BLOCKED (15)

### `L1_13_layer_norm_fwd`
- per-shape numeric diff:
  - `P1`: 
  - `P2`: 

### `L2_10_w8a8_block_int8_matmul`
- per-shape numeric diff:
  - `P1`: 
  - `P2`: 

### `L2_13_bmm`
- per-shape numeric diff:
  - `P1`: 
  - `P2`: 

### `L2_14_matmul_kernel_persistent`
- per-shape numeric diff:
  - `P1`: 
  - `P2`: 

### `L2_20_causal_conv1d_update`
- per-shape numeric diff:
  - `P1`: 
  - `P2`: 

### `L2_32_experts_combine`
- per-shape numeric diff:
  - `P1`: 
  - `P2`: 

### `L2_34_silu_mul_static_tensorwise_quant_triton_kernel_for_cutlass_moe`
- per-shape numeric diff:
  - `P1`: 
  - `P2`: 

### `L3_13_chunk_gated_delta_rule_fwd_kernel_h_blockdim64`
- per-shape numeric diff:
  - `P1`: 
  - `P2`: 

### `L3_16_chunk_fwd_kernel_o`
- per-shape numeric diff:
  - `P1`: 
  - `P2`: 

### `L3_17_chunk_gla_fwd_kernel_o`
- per-shape numeric diff:
  - `P1`: 
  - `P2`: 

### `L3_18_chunk_scaled_dot_kkt_fwd`
- per-shape numeric diff:
  - `P1`: 
  - `P2`: 

### `L3_19_chunk_kda_scaled_dot_kkt_fwd_kernel_intra_sub_inter`
- per-shape numeric diff:
  - `*`: 

### `L3_20_recompute_w_u_fwd`
- per-shape numeric diff:
  - `P1`: 
  - `P2`: 

### `L3_22_chunk_local_cumsum_vector`
- per-shape numeric diff:
  - `P1`: 
  - `P2`: 

### `L3_23_merge_16x16_to_32x32_inverse`
- per-shape numeric diff:
  - `P1`: 
  - `P2`: 
