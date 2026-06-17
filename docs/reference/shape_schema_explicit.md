# Explicit shape schema + per-op spec (Phase-2 target format)

> Directional design for the Phase-2 cutover. The authoritative format today is still the knob grid;
> this document defines what replaces it. The typed IR is `ascendbench/input_spec/row_ir.py` (U5).

## 1. Authoritative row format (`RowSpec`)

Each shape row is explicit and self-documenting — no decoder ring:

```yaml
schema_version: 1
public_shapes:
  - id: P3
    role: public
    provenance: model-sourced        # synthetic | npukb-borrowed | model-sourced | edge | baseline
    tensors:
      q: {shape: [2048, 16, 128], dtype: bfloat16, axes: [num_seqs*seqlen, num_heads, head_dim], dist: {kind: randn}}
      k: {shape: [2048, 16, 128], dtype: bfloat16, axes: [num_seqs*seqlen, num_heads, head_dim], dist: {kind: randn}}
      block_table: {shape: [4, 64], dtype: int32, axes: [num_seqs, max_blocks], dist: {kind: randint, args: {lo: 0, hi: num_seqs}}}
    params:                          # the named ROW ENVIRONMENT (non-axis symbols, U4-classified)
      causal: {kind: scalar, value: false}
      softmax_scale: {kind: derive, expr: "1/sqrt(head_dim)", value: 0.088}
      groups: {kind: row-only, value: 8}     # block/group/flag with no tensor home
hidden_shapes: [...]                 # same shape; sealed from the candidate sandbox
```

Maps directly onto the IR: `TensorSpec(shape, dtype, axes, dist)` + `ParamSpec(kind, value, expr)`
per `RowSpec(id, role, provenance, tensors, params, structs)`.

**Why each field:** `axes` names every dimension (the glossary maps the short symbols);
`dist` is the explicit fill rule (required — no implicit `randn` default); `params` holds the six
non-tensor symbol classes from `docs/reviews/2026-06-17-symbol-provenance.md` (scalar / scalar_knob /
derive / scalar_seq / optional / row-only); `provenance` is a tracked, improvable coverage metric.

## 2. Per-op spec (the human-edited generator source)

Authors edit a small spec; the generator (U6) expands it into the explicit rows above (so a 50-shape
sweep is generated, not hand-written), and the linter enforces axis links:

```yaml
op: L3_13_kernel_unified_attention_3d
tensors:
  q: {axes: [E*L, H, D], dtype: bfloat16, dist: randn}
  k: {axes: [E*L, H, D], dtype: bfloat16, dist: randn}   # shares H, D with q (linked)
axis_links: [q.head_dim == k.head_dim, q.num_heads == k.num_heads]
params: {causal: {kind: scalar}, softmax_scale: {kind: derive, expr: "1/sqrt(D)"}}
sweep: {axes: {seqlen: {ladder: [...]}, ...}, edges: [all_min, all_max, aspect]}
```

## 3. The IR is the extension boundary (`row_ir.py`)

`dtype`, `dist` kind, and input kind are **registered handlers**, not hard-coded tables in every
consumer. Adding an NPU dtype or a new fill rule = `register_dtype(...)` / `register_dist(...)` +
bump `SCHEMA_VERSION`; the manifest / build_inputs / IPC / sandbox / cost / constraints (wired in U8)
all read the IR, so none of them changes. The v1 registries are bootstrapped from the existing
`inputs.py` tables so the known set is exactly what `build_inputs` supports today.

## 4. Invariants (enforced by `RowSpec.validate` + the linter)

- every `dtype` and `dist.kind` is registered (unknown → fail-loud);
- `len(axes) == len(shape)` per tensor;
- no dead public knobs (every declared symbol is a tensor axis, a `params` entry, or a scalar/dist
  input role — `none free`, per U4);
- hidden rows are sealed from the candidate sandbox public view;
- the generated rows reproduce the *full* `build_inputs` output (U6 equivalence gate).
