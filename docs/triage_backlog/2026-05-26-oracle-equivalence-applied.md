---
date: 2026-06-02
tool: scripts/apply_oracle_equivalence_verdict.py
purpose: V9 M2.1c — apply R10 GPU verdict to meta.yaml provenance.golden
verdict-dir: ../loop_mailbox/results/b-slice-c2-120705/r10
---

# Oracle Equivalence Verdict Apply Report

Tasks processed: **9**  
Promoted to truth source (`oracle_pytorch_verified`): **0**

## Overall verdict tally

| overall | count |
|---|---|
| `VERIFIED` | 9 |

## VERIFIED (9)

### `L1_35_compute_src2dst_triton`
- per-shape numeric diff:
  - `P1`: max_abs=0.000e+00, max_rel=0.000e+00
  - `P2`: max_abs=0.000e+00, max_rel=0.000e+00

### `L1_36_compute_seg_indptr_triton`
- per-shape numeric diff:
  - `P1`: max_abs=0.000e+00, max_rel=0.000e+00
  - `P2`: max_abs=0.000e+00, max_rel=0.000e+00

### `L1_37_compute_problem_sizes_w4a8`
- per-shape numeric diff:
  - `P1`: max_abs=0.000e+00, max_rel=0.000e+00
  - `P2`: max_abs=0.000e+00, max_rel=0.000e+00

### `L2_27_deepep_permute_triton`
- per-shape numeric diff:
  - `P1`: max_abs=0.000e+00, max_rel=0.000e+00
  - `P2`: max_abs=0.000e+00, max_rel=0.000e+00

### `L2_28_deepep_post_reorder_triton`
- per-shape numeric diff:
  - `P1`: max_abs=2.384e-07, max_rel=1.837e-06
  - `P2`: max_abs=2.384e-07, max_rel=2.090e-04

### `L2_34_silu_mul_static_tensorwise_quant_triton_kernel_for_cutlass_moe`
- per-shape numeric diff:
  - `P1`: max_abs=1.863e-09, max_rel=2.796e-07
  - `P2`: max_abs=3.200e+01, max_rel=1.000e-01

### `L2_35_pre_reorder_triton_kernel_for_cutlass_moe`
- per-shape numeric diff:
  - `P1`: max_abs=0.000e+00, max_rel=0.000e+00
  - `P2`: max_abs=0.000e+00, max_rel=0.000e+00

### `L2_36_post_reorder_triton_kernel_for_cutlass_moe`
- per-shape numeric diff:
  - `P1`: max_abs=0.000e+00, max_rel=0.000e+00
  - `P2`: max_abs=0.000e+00, max_rel=0.000e+00

### `L2_37_silu_and_mul_post_quant`
- per-shape numeric diff:
  - `P1`: max_abs=1.863e-09, max_rel=2.796e-07
  - `P2`: max_abs=3.200e+01, max_rel=1.000e-01
