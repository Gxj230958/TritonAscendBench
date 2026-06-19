# U9 c4c — route shapes.yaml readers through the shape_format adapter

Goal: make every runtime shapes.yaml reader handle BOTH knob and explicit formats by routing structural
access through the `shape_format` adapter, staying **byte-identical on knob-format** (no manifest/grade drift).
Authoring/writer tools (extend_task_shapes, annotate_explicit_view, quarantine, report_dead_knobs,
fill_missing, patch_shapes_from_wrapper, lint_shapes_against_kernel, diagnose_blocked_tasks) WRITE knob-format
and change at the U9 generation cutover — NOT c4c.

## The 4 access patterns and their adapter replacement

| pattern | replacement | risk |
|---|---|---|
| id enumeration: `doc[public_shapes]+doc[hidden_shapes]` row `id`s | `public_ids(doc)` + `hidden_ids(doc)` (byte-identical knob branch, public-then-hidden, same dict/truthy-id filter) | LOW (mechanical) |
| input build: `build_inputs(doc, sid, seed)` | `build_inputs_for(doc, sid, seed)` (knob dispatches verbatim) | NONE |
| knob env: `dict(size_knobs)` + row `knobs` | `effective_env(doc, sid)` (== effective_knobs) | LOW value, but see wrappers |
| dtype width: local `_representative_dtype_size` / first-dtype | `representative_dtype_size(doc)` | MEDIUM (selection differs) |

## Preserved invariants (the knob-neutrality wrappers — do NOT drop)

1. **numeric-filter + None-on-missing** around `effective_env`: eval `_shape_knobs` / cross_check `_shape_knobs`
   return ONLY numeric (int/float non-bool) knobs and `None` on a missing id; `effective_env` returns the FULL
   map and RAISES KeyError on a missing id. Keep the filter + `try/except KeyError -> None` wrapper around it.
2. **ShapeFormatError -> conservative**: `is_explicit_shapes_doc` RAISES on a versioned-but-malformed explicit
   doc. Readers whose contract is "return None/[] on bad doc" (eligibility.shapes_eval_ids,
   validate_consistency._shape_ids) must catch ShapeFormatError and map to None/[] (fail-closed, no crash).
3. **anchor `_vol` raw per-row knobs**: oracle_consistency_anchor_runner `_vol` multiplies ONLY raw per-row
   `knobs` (no size_knobs overlay). `effective_env` overlays size_knobs and would reorder the volume sort ->
   reselect anchors. Keep the raw per-row read for the knob branch (explicit branch uses env).
4. **dtype_size selection**: local helpers pick "first input with a dtype key" (audit) or "first tensor with
   dims" (eval); the adapter picks "first dims-bearing input, default float32". Verify-then-adopt: regenerate
   `cost_baseline.json` (snapshot) + re-run the auditor over all 101 and require byte-identical dtype_size
   before swapping; else keep local.

## DEFINITIVE site checklist (exhaustive grep of every build_inputs() call + every public_shapes/hidden_shapes read)

**build_inputs(doc, sid, seed) -> build_inputs_for(doc, sid, seed)** — 16 sites:
eval 270/544/1138/1143/1146; precompute 430/512; run_oracle_equivalence_gpu 493/**645**; audit 112;
cross_check worker-template 427/433; capture_golden_baselines 209; anchor_runner 394; severance_runner 833;
lint_shapes_against_kernel 106.

**id enumeration `public_shapes+hidden_shapes -> id` -> public_ids(doc)+hidden_ids(doc)** — 14 sites:
dataset 130/299; eval 628-629 (inside _shape_knobs); eligibility 292-293; validate_corpus_consistency 55;
**validate_tasks 722**; **summary_metrics 197**; classify_correctness_sweep 60; shape_feasibility_report 150;
oracle_equiv_one **107**; precompute 77/99; **run_oracle_equivalence_gpu 1185**; audit 94; cross_check 254-255;
snapshot 62; anchor_runner 91; standalone_freeze_verifier 2163/2166. (knob branch byte-identical; catch
ShapeFormatError->None/[] in conservative readers eligibility/validate_consistency.)

**knob env `size_knobs`+row knobs -> effective_env(doc, sid)** — eval 631 (_shape_knobs, KEEP numeric+None
wrapper); cross_check 259 (same wrapper); snapshot 63-64; audit 110; anchor_runner _vol 72-80 (KEEP raw per-row
for knob branch); **validate_tasks 250 `_cost_model_symbol_universe` (size_knobs+row knobs keys -> env keys)**.

**dtype width local helper -> representative_dtype_size(doc)** — eval 594-621; cross_check 217-235; snapshot 35-37;
audit 87-89. GATE: direct per-task parity (adapter == each local) over all 101 knob docs BEFORE deleting the local.

**doc["inputs"] -> input_specs(doc)** — precompute 198 (the C2 SIGNATURE gate at 386 — explicit must not
false-MISMATCH); resolve_upstream_entrypoints 1120; audit 85/90-93 (declared_widths).

**`inputs.shape_ids(doc)` CALLERS -> public_ids(doc)+hidden_ids(doc)** (codex r3): eval L1074
`_inputs.shape_ids(shapes_doc)`; capture_golden_baselines (its shape_ids() call). `inputs.shape_ids` itself is
knob-only; replace the call sites with the adapter (or make shape_ids dispatch). gen_explicit_shapes/annotate
callers are generator/authoring -> leave.

**validate_tasks 250 `_cost_model_symbol_universe`** (codex r3): for explicit, the size_knobs+row-knobs key
union -> the union of `effective_env(doc, sid).keys()` over rows (== env keys, the direct equivalent). This is
a MORE-permissive "any declared symbol is legal in a formula" check, distinct from c4b's stricter
named_cost_universe (homed+declared+derived); route it to env-keys (shared env source-of-truth), do NOT merge
the two checks (different strictness by design).

**DEAD / leave**: precompute 671/674 (dead `_load_task_for_inputs`); cross_check 809 `task.public_shapes`
(TaskDef, inherits dataset's C1 fix); eval 1061 PHASE-C-MANUAL raw-text gate (format-agnostic). lint_shapes is
a dev tool (NOT daemon-gated) — route it fully (ids+input_specs+build_inputs_for) or it half-breaks; keep it in C3.

## Clusters (each its own commit, codex-reviewed, knob-neutrality tested)

- **C1 — id-enumeration (LOW, dev-verifiable)**: dataset.load_task/load_canary_task (row selection ->
  iter_rows/public_ids, keep id=sid, name=row.get(name) or sid, params packing unchanged), eligibility
  shapes_eval_ids, validate_corpus_consistency `_shape_ids`, precompute `_declared_shape_ids`/`_sweep_shape_ids`,
  standalone_freeze_verifier `_shape_ids_for`, eval `shape_ids`. Test: re-enumerate all 101 knob docs, id sets
  byte-identical; ShapeFormatError->conservative.
- **C2 — central eval helpers + build_inputs (mix, the high-leverage move)**: route eval `_representative_dtype_size`
  -> adapter.representative_dtype_size (delete local dup + _DTYPE_BYTES), eval `_shape_knobs` -> effective_env
  WRAPPED in the numeric-filter/None wrapper. gpu_rank_one imports these two -> inherits the COST helpers (but see
  the S2/C2 build paths below -- it is NOT fully explicit-aware from that alone). Swap build_inputs ->
  build_inputs_for in **every materialization site** (codex): eval L270 (sampled-output recheck), L544
  (metamorphic), L1138 (correctness/timing); precompute L430/L512; anchor_runner L394; the cross_check NCU
  worker template L427/L433; **severance_subprocess_runner L833 (S2 -- the freeze-verifier JSON-passes the full
  doc to it; without this swap explicit S2 BREAKS)**; **run_oracle_equivalence_gpu L493 (C2/gpu-rank -- shape
  ids come from the ref-file glob, only build_inputs needs routing)**; **capture_golden_baselines L209 (golden
  -- + its shape_ids via public_ids/hidden_ids in C1)**. Test: eval helpers byte-identical over 101 knob docs;
  build_inputs_for == build_inputs for knob (proven by the U6 equivalence gate).
- **C3 — cost auditors (MEDIUM, verify-then-adopt)**: cross_check `_representative_dtype_size`/`_shape_knobs`
  (delete dups, route to adapter+wrapper), snapshot_cost_outcomes (iter_rows+effective_env for the merge),
  audit_cost_model_formulas (public_ids+effective_env+build_inputs_for+input_specs). dtype_size gate (codex):
  a cost_baseline diff can MISS a dtype change when no formula references `dtype_size`, so the gate is a DIRECT
  per-task parity assertion -- `representative_dtype_size(doc)` == each local helper's value over all 101 knob
  docs -- BEFORE deleting the local helper; if any task diverges, keep the local selection (or add a param).
- **C4 — anchor _vol (MEDIUM, keep raw per-row for knob)**: oracle_consistency_anchor_runner iter_rows + _vol
  keeps raw per-row knobs (knob branch) / env (explicit), build_inputs_for. Test: anchor id selection byte-identical
  over the knob corpus.

## GPU-gated (end-to-end validation needs the daemon, dev does plumbing + knob-neutrality only)
eval.py, cross_check_cost_model_gpu.py, oracle_consistency_anchor_runner.py, standalone_freeze_verifier.py,
gpu_rank_one.py. The dev box verifies knob-neutral plumbing (id sets / dtype_size / knob env / build_inputs
byte-identical); the real grade runs on the daemon after the migration cutover.

## Readers the first survey MISSED (codex-found + exhaustive grep, now IN scope — live grading paths)
- **severance_subprocess_runner.py L833** — S2 input materialization (freeze-verifier JSON-passes the doc). C2.
- **run_oracle_equivalence_gpu.py L493** — C2 + gpu-rank input rebuild (shape ids from the ref-file glob). C2.
- **capture_golden_baselines.py L209 + shape_ids L~37** — verified-mode golden artifacts for explicit UOR tasks. C1+C2.
- **resolve_upstream_entrypoints.py L1120** — `set(doc["inputs"].keys())` -> `set(input_specs(doc))`; the S2
  resolver (standalone_freeze_verifier delegates `_resolver_verdict_for` to it). C1.
- **lint_shapes_against_kernel.py L35/L63** — dev lint tool (NOT daemon-gated) build_inputs -> build_inputs_for;
  cheap mechanical swap so a migrated task lints. C2 (low priority).

Exhaustive grep confirms the remaining build_inputs callers are AUTHORING/writer tools (extend_task_shapes,
generate_shapes_and_contract, annotate_explicit_view) -> change at the generation cutover, not c4c.

## Out of c4c scope
- The explicit ShapeSpec.name/params semantics — migration-cutover design choice (knob keeps current packing).
- summary_metrics.py — meta-only, no shapes structure.
- Authoring/writer tools — change at the generation cutover, not c4c.
