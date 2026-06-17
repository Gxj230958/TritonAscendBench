# Shape knob glossary (common conventions — NOT authority)

This is a **convenience legend** for the single-letter size knobs used across `shapes.yaml`. It is
**not** authoritative: a knob's true meaning for a given operator is whatever its `inputs[].dims`
(and, after the explicit-shape migration, the per-row `explicit_view`/`tensors[].axes`) say. The same
letter can mean different things in different operator families — that ambiguity is exactly why the
explicit per-tensor view exists. Use this table to orient, then read the task's own inputs/view to be
sure.

## Core knobs (each used in 80+ tasks)

| Knob | Common meaning | Notes / family-dependent uses |
|------|----------------|-------------------------------|
| `B`  | batch size | tokens-batch in some kernels |
| `L`  | sequence length (tokens) | sometimes a generic row count |
| `D`  | feature / hidden dim | often the head dim in attention (`q:[…, H, D]`) |
| `Dv` | value head dim | the `v`-side dim when it differs from `D` |
| `H`  | number of (attention) heads | hidden width in some norm/MLP kernels |
| `N`  | a count | max blocks / KV tokens / routed slots — **highly family-dependent** |
| `T`  | tile / chunk / time dim | chunk size in SSM/linear-attn; tokens in others |
| `E`  | experts (MoE) | num-sequences in some attention layouts |
| `C`  | channels / cache dim | conv channels; KV-cache page dim |
| `G`  | groups | grouped-norm / grouped-MoE; **often a non-tensor `params:` symbol** |

## Task-specific knobs

Knobs beyond the core ten (`K`, `V`, `CAUSAL`, `S`, `P`, `BT`, `APPW`, `KW`, `SOFTCAP`, `Hg`, `R`,
`M`, `CS`, `GS`, `BS`, `HD`, `DS`, `NSEG`, `QH`, `KVH`, `DTSP`, `Dn`, `Dr`, `REV`, `LSEB`) appear in
only 1–6 tasks each and are operator-local. Several (`CAUSAL`, `SOFTCAP`, `CS`, `APPW`, `LSEB`,
`DTSP`) are flags / block sizes / group counts with **no tensor-axis home** — they are row-level
parameters (the `params:` map in the explicit format), not tensor dimensions.

> **Authority:** for any operator the per-task `shapes.yaml` `inputs:` (knob form) is the source of
> truth today. The generated `explicit_view` shipped in Phase 1 is an **advisory, non-authoritative**
> readable mirror (it reports the *declared* dtype; integer index dists are materialised as int64 at
> build time). It becomes authoritative only at the Phase-2 cutover, when the explicit
> `tensors[].axes` + `params` form replaces the knob grid. This glossary is orientation only.
