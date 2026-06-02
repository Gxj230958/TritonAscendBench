---
date: 2026-06-02
tool: scripts/apply_oracle_equivalence_verdict.py
purpose: V9 M2.1c — apply R10 GPU verdict to meta.yaml provenance.golden
verdict-dir: ../loop_mailbox/results/b-l211c2-124921/r10
---

# Oracle Equivalence Verdict Apply Report

Tasks processed: **1**  
Promoted to truth source (`oracle_pytorch_verified`): **1**

## Overall verdict tally

| overall | count |
|---|---|
| `VERIFIED` | 1 |

## VERIFIED (1)

### `L2_11_w8a8_triton_block_scaled_mm`
- **oracle_pytorch.py promoted to truth source**
- per-shape numeric diff:
  - `P1`: max_abs=0.000e+00, max_rel=0.000e+00
  - `P2`: max_abs=5.000e-01, max_rel=7.027e-04
