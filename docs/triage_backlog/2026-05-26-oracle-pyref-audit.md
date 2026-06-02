---
date: 2026-05-26
tool: scripts/oracle_pyref_precompute.py
purpose: V9 M2.1a (dev) — R10 stage-1 oracle PyTorch reference precompute
---

# Oracle PyRef Precompute Audit

Tasks audited: **1**  

## Status distribution

| status | count |
|---|---|
| `SIGNATURE_MISMATCH` | 1 |

## SIGNATURE_MISMATCH (1)

### `L1_05_gather_block_tables` (level1)

- kernel required (2): `idx_mapping, num_reqs_padded`
- oracle inputs[..] (2): `block_table, idx_mapping`
- oracle outputs (1): `output`
- shapes inputs (2): `block_table, idx_mapping`
- **missing in shapes**: `['num_reqs_padded']`
- extra-in-oracle (oracle reads non-kernel param): `['block_table']`
- **suggested fix**: shapes.yaml.inputs is missing ['num_reqs_padded'] that kernel `gather_block_tables` requires; add or rename keys.
