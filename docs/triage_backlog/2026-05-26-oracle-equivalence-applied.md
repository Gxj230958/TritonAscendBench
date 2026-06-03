---
date: 2026-06-03
tool: scripts/apply_oracle_equivalence_verdict.py
purpose: V9 M2.1c — apply R10 GPU verdict to meta.yaml provenance.golden
verdict-dir: /tmp/conv_verdict
---

# Oracle Equivalence Verdict Apply Report

Tasks processed: **8**  
Promoted to truth source (`oracle_pytorch_verified`): **8**

## Overall verdict tally

| overall | count |
|---|---|
| `VERIFIED` | 8 |

## VERIFIED (8)

### `L1_19_topk_log_softmax`
- **oracle_pytorch.py promoted to truth source**
- per-shape numeric diff:
  - `P1`: max_abs=4.768e-07, max_rel=1.172e-07
  - `P2`: max_abs=9.537e-07, max_rel=1.190e-07

### `L2_02_chunk_scan_fwd`
- **oracle_pytorch.py promoted to truth source**
- per-shape numeric diff:
  - `P1`: max_abs=0.000e+00, max_rel=0.000e+00
  - `P2`: max_abs=0.000e+00, max_rel=0.000e+00

### `L2_37_silu_and_mul_post_quant`
- **oracle_pytorch.py promoted to truth source**
- per-shape numeric diff:
  - `P1`: max_abs=3.200e+01, max_rel=1.250e-01
  - `P2`: max_abs=3.200e+01, max_rel=1.250e-01

### `L2_40_gemma_qkv_rmsnorm`
- **oracle_pytorch.py promoted to truth source**
- per-shape numeric diff:
  - `P1`: max_abs=0.000e+00, max_rel=0.000e+00
  - `P2`: max_abs=0.000e+00, max_rel=0.000e+00

### `L3_03_chunk_state_varlen`
- **oracle_pytorch.py promoted to truth source**
- per-shape numeric diff:
  - `P1`: max_abs=1.907e-06, max_rel=1.222e-04
  - `P2`: max_abs=5.722e-06, max_rel=6.601e-03

### `L3_10_fused_moe_kernel_gptq_awq`
- **oracle_pytorch.py promoted to truth source**
- per-shape numeric diff:
  - `P1`: max_abs=4.061e-01, max_rel=3.787e-03
  - `P2`: max_abs=4.956e-01, max_rel=3.839e-03

### `L3_12_moe_mmk`
- **oracle_pytorch.py promoted to truth source**
- per-shape numeric diff:
  - `P1`: max_abs=3.815e-06, max_rel=3.503e-05
  - `P2`: max_abs=3.433e-05, max_rel=5.155e-03

### `L3_16_chunk_fwd_kernel_o`
- **oracle_pytorch.py promoted to truth source**
- per-shape numeric diff:
  - `P1`: max_abs=3.815e-06, max_rel=7.042e-03
  - `P2`: max_abs=3.125e-02, max_rel=3.659e-02
