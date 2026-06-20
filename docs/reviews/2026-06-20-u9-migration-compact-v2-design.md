# U9 migration M1 — the compact v2 explicit format + expander

codex design consult (2026-06-20) ruled MIGRATION-TARGET **B**: the committed migrated shapes.yaml is a COMPACT
readable v2 form, expanded into the existing v1 structured IR the adapter already reads. The verbose
generate_explicit_shapes output (8123 lines vs knob 941 for L1_01) is NOT the committed form. M1 builds the
v2 format + expander + writer + gates; M2 (separate) flips the 101 files + GPU re-grades.

## v2 schema (the committed shapes.yaml)

```yaml
schema_version: 2
public:
  - id: P1
    role: public
    env: {B: 4, N: 128, E: 4}        # resolved row-environment symbol -> value (kept; NOT derivable
                                      # from axes+shape -- composite axes E*L underdetermined, row-only
                                      # params CAUSAL/G/flags have no tensor home, dist-only cost symbols)
    inputs:                          # ORDERED list (coupled dists ref an earlier tensor) -- not a map
      - {name: logits, kind: tensor, shape: [4, 128], dtype: float32, axes: [B, N], dist: randn}
      - {name: idx,    kind: tensor, shape: [4], dtype: int64, axes: [B], dist: "randint:0:E"}
      - {name: scale,  kind: scalar, value: 8}
      - {name: q,      kind: optional, value: null}
hidden: [...]
```

Each tensor input is ONE inline line: `dist` is the SYMBOLIC string (like knob), NOT the structured
{kind,args}; `dist_symbolic` and the per-row RESOLVED dist are DERIVED, never committed (removes the v1
redundancy). Non-tensor inputs keep their `kind` + `value`/`expr` (derive/scalar_knob carry `expr`).

## The expander (the missing piece) — `ascendbench/input_spec/compact_shapes.py`

`expand_compact(v2_doc) -> v1_explicit_doc`: per tensor input, derive `dist_symbolic = parse_dist_spec(dist)`
and the resolved `dist = resolve_dist_args(dist_symbolic, env)`; pass shape/dtype/axes through. Non-tensor
inputs pass through. Output is byte-identical to what generate_explicit_shapes(knob) emits. parse_dist_spec /
resolve_dist_args pull torch, so the expander is TORCH-OK and only invoked on the materialization path.

## shape_format v2-dispatch (most readers read v2 NATIVELY -- only materialization expands)

- `is_explicit_shapes_doc`: accept schema_version 1 (existing v1) AND 2 (validate v2 structure: rows with id +
  env-map + ordered inputs, each tensor input carrying shape/dtype/axes + a non-empty string `dist`). TORCH-FREE
  (no parse_dist_spec) -- shape_manifest stays torch-free.
- `public_ids`/`hidden_ids`/`iter_rows`: read v2 row ids/rows. torch-free.
- `effective_env(v2, sid)`: return the row's `env` directly (it is already resolved in v2). torch-free.
- `representative_dtype_size`/`input_specs`: read the first/all v2 tensor inputs' dtype/structure. torch-free.
- `explicit_canonical_projection` / `canonical_manifest`: project the COMMITTED v2 doc (torch-free; a v2 edit
  moves the hash). NOTE the v2 manifest != the knob manifest (a real format change -> re-grade, expected).
- `build_inputs_for(v2, sid, seed)`: `build_from_explicit(expand_compact(v2_row), ...)`. TORCH-OK.
- The cost classifier (classify_symbols._classify_explicit): for v2 it reads the dist STRING via `_syms(...)`
  (exactly like the knob path), so dist-param symbols are recovered WITHOUT parse_dist_spec. dist_symbolic on
  v1 stays the source for v1 docs.

## knob -> v2 writer — `scripts/corpus/gen_compact_shapes.py`

`generate_compact_shapes(knob_doc) -> v2_doc`: like generate_explicit_shapes but emits the compact rows (dist
as the symbolic string from the knob input's `dist`, env = effective_knobs(row), inputs in authored order).

## M1 gates (all over 101 knob tasks, dev-side)

1. `is_explicit_shapes_doc(generate_compact_shapes(knob))` is True (v2 validates).
2. `explicit_canonical_projection(expand_compact(v2)) == explicit_canonical_projection(generate_explicit_shapes(knob))`
   (the expanded v2 == the knob-generated v1, modulo a documented normalization).
3. build_inputs_for(v2, sid, seed) == build_inputs(knob, sid, seed) BYTE-equal, seed 0 + sampled timing seeds.
4. classify(v2) == classify(knob) and check_cost_symbols(v2) == check_cost_symbols(knob) (0 mismatch).
5. check_param_constraints(raw, v2).ok == check_param_constraints(raw, knob).ok for every task.

Only after all 5 gates are green over 101 does M2 (flip the files + staged GPU re-grade) run.

## codex M1-design fixes (round 1, all adopted)

1. **v2-aware dist accessors**: `explicit_row_dist` (resolved) + `explicit_row_dist_symbolic` (symbolic) MUST
   synthesize for v2 (parse_dist_spec(string) + resolve_dist_args(.,env)), else param_constraints._validate_dists_explicit
   silently weakens (None symbolic / string-vs-args resolved compare). These accessors become torch-OK for v2.
2. **dtype + cost-VALUE gates** (check_cost_symbols only proves the symbol UNIVERSE): add gate
   `representative_dtype_size(v2) == representative_dtype_size(knob)` (it feeds roofline t_ref) AND a cost-value
   parity `snapshot_cost_outcomes(v2) == snapshot_cost_outcomes(knob)` (FormulaEnv output identity) over 101.
3. **explicit v1/v2 BRANCHES, do NOT bump SCHEMA_VERSION**: keep SCHEMA_VERSION=1; is_explicit accepts
   `schema_version in {1,2}` with a separate v2 validator (v1 keeps requiring dist_symbolic; v2 requires a string dist).
4. **drop `role` from v2 rows**: the public/hidden SECTION is authoritative (manifest derives role from section);
   committing a per-row role invites a hash-invisible edit. The writer emits no `role`; is_explicit ignores it.
5. **v2 classify scans dist ARGS only**, not the whole string: a small TORCH-FREE dist-token extractor
   (`split "randint:0:E" -> args [0, E]`) so a knob named like a DSL keyword (`to`/`offsets`/`randint`) is not
   falsely marked dist-param. Matches v1's dist_symbolic.args scan exactly.

Plus (codex, for FUTURE hand-authored v2, not gating M1 now): a v2 consistency lint that tensor `axes` evaluate
to `shape` under `env`, and derive/scalar_knob `value` matches `expr`. The knob->v2 writer is correct by
construction (env=effective_knobs(row)), so M1's generated v2 needs no such lint; note it for the authoring cutover.

## M2 (separate, GPU-gated)
Flip each shapes.yaml to its v2 (preserve nothing of the knob grid; the explicit_view annotation is subsumed),
batch ~20 tasks, S2/C2/golden/roofline re-grade on the daemon, confirm VERIFIED == knob, release gate PROCEED.
Rollback = git revert. Commit daemon status: only from the box. Authoring tools (extend_task_shapes etc.) move
to v2 at this cutover.
