---
date: 2026-06-02
tool: scripts/apply_oracle_equivalence_verdict.py
purpose: V9 M2.1c — apply R10 GPU verdict to meta.yaml provenance.golden
verdict-dir: ../loop_mailbox/results/b-l213fix-055905/r10
---

# Oracle Equivalence Verdict Apply Report

Tasks processed: **1**  
Promoted to truth source (`oracle_pytorch_verified`): **1**

## Overall verdict tally

| overall | count |
|---|---|
| `VERIFIED` | 1 |

## VERIFIED (1)

### `L2_13_bmm`
- **oracle_pytorch.py promoted to truth source**
- per-shape numeric diff:
  - `P1`: max_abs=9.537e-06, max_rel=3.289e-04
  - `P2`: max_abs=4.196e-05, max_rel=1.628e-01
