---
tool: scripts/auto_patch_shapes_from_wrapper.py
purpose: V9 P6 — auto-patch shapes.yaml dims from kernel host wrapper unpacks
---

# Auto-patch shapes from wrapper contract

Tasks scanned: **1**

## Confidence tally

| confidence | count |
|---|---|
| `LOW` | 1 |

## LOW (1)

### `L2_13_bmm` (bmm_batch_invariant)
- `a`: OK current=['B', 'L', 'D'] proposed=['B', 'M', 'K'] -- current dims already match wrapper rank
- `b`: OK current=['B', 'D', 'N'] proposed=['_', '_', 'N'] -- current dims already match wrapper rank
