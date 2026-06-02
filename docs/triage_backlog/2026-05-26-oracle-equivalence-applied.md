---
date: 2026-06-02
tool: scripts/apply_oracle_equivalence_verdict.py
purpose: V9 M2.1c — apply R10 GPU verdict to meta.yaml provenance.golden
verdict-dir: results/b-l233-c2i-145755/r10
---

# Oracle Equivalence Verdict Apply Report

Tasks processed: **1**  
Promoted to truth source (`oracle_pytorch_verified`): **1**

## Overall verdict tally

| overall | count |
|---|---|
| `VERIFIED` | 1 |

## VERIFIED (1)

### `L2_33_fwd_grouped_kernel_stage1_rope`
- **oracle_pytorch.py promoted to truth source**
- per-shape numeric diff:
  - `P1`: max_abs=3.470e-03, max_rel=3.236e-02
  - `P2`: max_abs=2.885e-03, max_rel=1.104e+00
