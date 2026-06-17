# Symbol-provenance classification (Phase-2 U4)

Corpus-wide classification of every declared `size_knobs` symbol by its authoring role, produced by
`scripts/corpus/classify_symbols.py`. This grounds the Phase-2 explicit schema (U5 — which symbols
become tensor axes vs named `params:`) and the `cost_model` re-base (U7 — the formula universe is NOT
tensor-axes-only). Regenerate with `python3 scripts/corpus/classify_symbols.py`.

## Corpus totals (101 tasks)

| Class | Count | Meaning → explicit-format home |
|-------|-------|--------------------------------|
| `dead` | 549 | declared but referenced nowhere → **droppable** (the 60% template boilerplate) |
| `tensor-axis` | 335 | appears in some input's `dims` → a named tensor **axis** |
| `scalar-input` | 15 | a `scalar_knob`/`derive`/`scalar` input → a `params:` entry backed by a scalar |
| `dist-param` | 6 | appears ONLY in a `dist` expression → a `params:` entry backed by the fill rule |
| `row-only-param` | 6 | used by cost/constraints with **no input home** → a standalone named `params:` entry |

## The 6 `row-only-param` symbols (the load-bearing finding for U7)

These cost/constraint symbols are **not** any tensor dimension and have **no input expression home**,
so a "cost_model over named tensor axes" basis is impossible — they must become named `params:`:

| Task | Symbol | What it is |
|------|--------|-----------|
| `L1_09_quantize_k_cache_fast` | `D` | a width used only in the cost formula; tensor widths are literals |
| `L3_13_kernel_unified_attention_3d` | `CAUSAL` | a flag (the input scalar is literal) gating cost terms |
| `L2_29_silu_mul_static_tensorwise_quant…` | `G` | a group count |
| `L3_15_chunk_state_varlen` | `G` | a group count |
| `L3_18_fwd_grouped_kernel_stage1` | `G` | a group count |
| `L3_09_causal_conv1d_fwd` | `B` | batch used in cost; conv inputs are `[dim, seqlen]` |

(`scalar-input` symbols — e.g. `CS`, `DTSP`, `APPW`, `SOFTCAP`, `LSEB`, sequence-derived ones — are
non-axis too but already have a scalar/dist input role; they need no new home.)

## Input-form inventory (what the explicit schema must host)

Six forms, confirmed corpus-wide: `dims`-tensor (+`dist`, all 101), `scalar` (51), `scalar_knob` (22),
`derive` (13), `scalar_seq` (4), optional/`null` (2 tasks). **No `struct`/`ref` forms in the corpus.**

## Implications

- **U5 schema:** rows carry `tensors[].axes` (the 335 tensor-axis symbols) + a `params:` map (the 15
  scalar-input + 6 dist-param + 6 row-only-param symbols, all named). The 549 dead symbols are dropped.
- **U7 cost-model re-base:** the formula universe = tensor axes **+** declared `params:` **+**
  scalar/dist input roles. Every symbol resolves to one of these three; **none free**. Validate this.
