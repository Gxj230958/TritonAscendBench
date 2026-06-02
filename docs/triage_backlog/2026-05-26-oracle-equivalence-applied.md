---
date: 2026-06-02
tool: scripts/apply_oracle_equivalence_verdict.py
purpose: V9 M2.1c — apply R10 GPU verdict to meta.yaml provenance.golden
verdict-dir: ../loop_mailbox/results/b-l13norm2-062445/r10
---

# Oracle Equivalence Verdict Apply Report

Tasks processed: **1**  
Promoted to truth source (`oracle_pytorch_verified`): **1**

## Overall verdict tally

| overall | count |
|---|---|
| `VERIFIED` | 1 |

## VERIFIED (1)

### `L3_13_chunk_gated_delta_rule_fwd_kernel_h_blockdim64`
- **oracle_pytorch.py promoted to truth source**
- per-shape numeric diff:
  - `P1`: max_abs=8.128e-03, max_rel=9.389e-02
  - `P2`: max_abs=9.095e-03, max_rel=1.125e+01
