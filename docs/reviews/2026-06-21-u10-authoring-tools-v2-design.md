# U10 design — make the authoring tools produce/edit the compact v2 format

## Problem

The corpus was migrated to compact v2 (schema_version 2; `public`/`hidden` rows with `env` + materialized
`inputs`). Three authoring tools still read/write the retired KNOB format (`size_knobs` + `public_shapes`/
`hidden_shapes` + symbolic `inputs`):

- **generate_shapes_and_contract** (`_emit_shapes_yaml`): scaffolds a NEW task's shapes.yaml as a knob doc.
- **quarantine_shapes**: removes kernel-rejected shape rows.
- **extend_task_shapes**: appends new shape rows (candidate gen + CPU validate + byte-splice).

On the v2 corpus they break or corrupt: generate writes a knob file (mixed corpus), quarantine reads
`public_shapes` (absent in v2 → wrong counts / vacuous survivor check), extend reads `size_knobs`/`inputs`
(absent → no candidates). Any knob file they emit also fails `test_committed_v2_corpus_matches_regenerated`.

## Model: v2 is canonical, so authoring = parse/reconstruct → modify → `to_yaml`

The migrated v2 files are MACHINE-canonical (no hand comments; `migrate_to_v2.to_yaml` output, byte-pinned by
`test_committed_v2_corpus_matches_regenerated`). So authoring no longer needs byte-preserving text splices to
protect hand-authored knob comments — it can parse, modify in memory, and re-serialize through the one
canonical writer `migrate_to_v2.to_yaml`. Deterministic serialization means unchanged rows stay byte-identical.

### Key decision — relax `test_committed_v2_corpus_matches_regenerated` to a SELF-CONSISTENT semantic invariant

That test pins every committed v2 to `to_yaml(generate_compact_shapes(knob_at_e563ac2))`. It is a MIGRATION-time
check: it only holds while the corpus is the untouched e563ac2 migration. The moment quarantine/extend changes a
task's rows, that task's v2 legitimately diverges from the e563ac2-knob regeneration, so the test would fail on
a correct edit. (codex round-1 F3: plain canonicality `to_yaml(parse)==text` is too weak — it would NOT catch a
canonical-but-semantically-wrong row, e.g. `shape` not matching `resolve(axes, env)`, since `expand_compact`
reads `shape` and `env` independently.) Replace the e563ac2-baseline equality with a **baseline-independent
SELF-CONSISTENCY round-trip**, which holds for an authored corpus AND catches semantic corruption:

- `migrate_to_v2.to_yaml(generate_compact_shapes(v2_to_knob_spec(yaml.safe_load(text)))) == text` for every
  committed file — a TEXT-STRICT self-consistency round-trip (codex round-2 F1: parsed-DICT equality has the
  `4 == 4.0` hole, so `shape: [4.0]` would pass; comparing the serialized TEXT is type-strict). Because
  `v2_to_knob_spec` is lossless (proven 101/101, see below), the committed text round-trips to ITSELF for a
  correct file, but a corrupted row (wrong `shape` vs `axes`+`env`, wrong scalar value, drifted dist, a float
  dim, a non-canonical hand-edit) re-materializes to a DIFFERENT text and is caught. No e563ac2 dependency.
- plus `is_explicit_shapes_doc(doc) is True` (structural validity; already in `test_all_101_corpus_docs_are_explicit`).

The knob→v2 *machinery* equivalence (round-trip, parity) stays tested against the e563ac2 knob via `knob_doc` —
that validates the EXPANDER/WRITER, not the committed corpus content.

**Empirical proof (already run):** `generate_compact_shapes(v2_to_knob_spec(v2)) == v2` is **101/101** across the
whole corpus, including every non-tensor kind (scalar 3300 / scalar_knob 1145 / scalar_seq 172 / derive 587 /
optional 120; zero struct rows). So the reconstruction is lossless and this invariant holds today.

## Shared helper — `scripts/corpus/v2_authoring.py`

- `load_v2(path) -> dict`: read + `is_explicit_shapes_doc` assert (fail-closed on a non-v2 file).
- `write_v2(doc, path)`: `path.write_text(migrate_to_v2.to_yaml(doc))` — the single canonical writer.
- `v2_to_knob_spec(v2_doc) -> dict`: reconstruct a knob-equivalent doc so the existing knob-shaped logic in
  extend works unchanged. A v2 row's per-input fields ARE the knob `inputs` symbolic template; its `env` IS the
  row's resolved knobs. Reconstruct from ROW 0 (input order preserved, no set), reversing materialize_row:
  - tensor -> `{dims: axes, dtype: dtype, dist: dist}` (the dist STRING verbatim).
  - scalar -> `{scalar: value}`; scalar_seq -> `{scalar_seq: value}`; optional -> the bare value (`null`).
  - derive -> `{derive: expr}`; scalar_knob -> `{scalar_knob: expr}` (rebuild from `expr`, NOT `value` -- codex
    round-1 F2; `value` is re-derived by materialize against the row knobs).
  - struct -> FAIL-LOUD (raise): v2 keeps only `struct`, not the full `fields`, so it is not provably lossless;
    the current 101 have zero struct rows, so a struct row means a new/unsupported case, not a silent guess.
  - UNIFORM-INPUT guard: reconstruct from row 0 only; FAIL-LOUD if any other row's (name, kind) input list
    differs from row 0's. The knob model has ONE `inputs` template for all rows, so a per-row-varying input set
    cannot round-trip to knob (current corpus: 0 such tasks, verified). The F3 round-trip would also catch it,
    but failing in `v2_to_knob_spec` keeps the authoring tools from silently using a partial template.
  - `size_knobs = dict(row0.env)` -- the FULL row-0 env in env order (codex round-1 F1: NOT axes-only; symbols
    like L1_17's `N` appear only in dist args, and `_active_dims` only accepts tokens present in size_knobs).
  - `public_shapes/hidden_shapes = [{id, knobs: row.env} ...]` (per-row full env as knobs; effective_knobs =
    size_knobs (+) row.knobs = row.env, so build_inputs resolves exactly).
  - INVARIANT (tested, proven 101/101): `generate_compact_shapes(v2_to_knob_spec(v2)) == v2`.

## Per-tool changes

1. **generate** (🟢): `_emit_shapes_yaml(inputs_spec)` builds the knob scaffold doc as today, then returns
   `migrate_to_v2.to_yaml(generate_compact_shapes(knob_doc))`. The "# Auto-generated / TODO" comment header is
   DROPPED from shapes.yaml (canonical v2 carries no comments); the guidance moves to a stderr/log line and the
   contract.md the same tool already writes. A scaffold is born as canonical v2.

2. **quarantine** (🟢): switch from byte-surgery to **parse -> drop rows by id -> `write_v2`** (codex round-1 F4).
   The byte-surgical `remove_rows` mostly works on v2, but removing EVERY row in a section leaves a dangling
   `public:`/`hidden:` that parses to `null`; parse/drop/re-emit avoids that (`to_yaml` writes `[]` for an empty
   section) and is guaranteed canonical. Drop the rows whose `id` is in the target set from `doc["public"]`/
   `doc["hidden"]`; assert the targets are gone + report not-found; recount `pub`/`hid` from the surviving v2
   rows (`len(doc["public"])`/`len(doc["hidden"])`, NOT `public_shapes`). FAIL-LOUD if the removal would leave
   ZERO total rows (codex round-2 F2: an empty task has no row-0 template for `v2_to_knob_spec`; task-removal is
   a separate explicit flow, not a quarantine). Meta `shape_coverage` rewrite: make the `total:` line OPTIONAL
   in the regex (codex round-2 F3: two live metas carry only `public`/`hidden`, no `total`) -- update `total`
   if present, add it after `hidden` if absent. A removed shape is still only one the real frozen kernel
   rejects -- never tuned to a number.

3. **extend** (🔴): wrap read/write, leave the candidate-gen + CPU-validate core UNCHANGED:
   - read: `knob = v2_to_knob_spec(load_v2(path))`.
   - run `_active_dims` / `_gen_self_candidates` / `_cpu_validate` on `knob` exactly as today (build_inputs on
     the reconstructed knob trial is byte-identical to the v2 — same env, same template).
   - append valid candidate rows to `knob` (knob `{id, knobs}` rows).
   - write: `write_v2(generate_compact_shapes(knob), path)` — whole-doc canonical rewrite (existing rows
     deterministic → byte-identical; the new rows appended). The byte-splice `_yaml_row_text` is retired.
   - `gpu_shape_probe` (the GPU candidate validation) is format-orthogonal — UNTOUCHED.

## Tests / guards

- `tests/unit_tests/test_v2_authoring.py`: `v2_to_knob_spec` round-trip (generate_compact_shapes(reconstruct(v2))
  == v2) over all 101; struct fail-loud; load_v2 fail-closed; write_v2 canonical.
- Per-tool: generate emits a valid v2 scaffold (is_explicit + self-consistent round-trip); quarantine on a v2
  task removes the id + survivors BYTE-UNCHANGED + counts correct + empty-section handled; extend on a v2 task
  appends a row + the PRE-EXISTING rows are byte-identical (codex round-1 F5) + the new row's shape == the
  candidate env resolved against axes.
- `test_committed_v2_corpus_matches_regenerated` -> `test_committed_v2_corpus_is_self_consistent`:
  `generate_compact_shapes(v2_to_knob_spec(committed)) == committed` for all 101 (baseline-independent semantic
  invariant) + keep `is_explicit`.
- Authoring never tunes-to-pass: extend still requires CPU-valid + (separately) GPU-probe; quarantine only
  removes kernel-rejected ids.

## Sequencing

helper → generate → quarantine → extend → test relaxation. Each: codex review → 0 FAIL → commit. Dev-side only
(no GPU); extend's gpu_shape_probe path is untouched.
