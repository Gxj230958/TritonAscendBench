# U9 explicit-shape migration — completion report

**Status: the corpus is migrated + GPU-validated; a dev-side test-migration is the only open follow-up.**

The `shapes.yaml` corpus has been flipped from the cryptic single-letter **knob grid** to a compact, readable
**explicit v2** per-tensor format, and the v2 corpus grades **bit-identical to the knob corpus** on the GPU
daemon. The knob grid is retired.

## What shipped (each unit: design -> codex -> impl -> codex to 0 FAIL, then commit to main)

| Unit | commit | what |
|---|---|---|
| approach A | d4a68a8 | explicit tensor rows carry a row-invariant `dist_symbolic` |
| c4b | b7025e7 | cost-symbol universe reads `dist_symbolic` (recovers L1_21:N, L2_26:T) |
| c4c C1 | bacd24b | every public/hidden id enumeration routes through `public_ids`/`hidden_ids` |
| c4c C2 | cf151b7 | every `build_inputs(...)` -> `build_inputs_for(...)` (16 sites) |
| c4c C3+C4 | 6ee52b0 | cost dtype/knob-env/inputs reads + the anchor `_vol` route through the adapter |
| M1 | 83c1b4d | the compact **v2** format + `expand_compact(v2)->v1` + adapter v2-awareness |
| M2 writer | e563ac2 | `migrate_to_v2.py` — flow-style one-line-per-tensor serializer (~343 lines/task) |
| daemon-path fix | a234184 | put `REPO_ROOT` on the S2 verifier's `sys.path` (a c4c regression the pilot caught) |
| **the flip** | 947e09a | **all 101 `shapes.yaml` -> v2** (net **-37,812 lines**) |
| **re-grade metas** | ec33420 | the daemon's v2 re-grade verdicts + new v2 manifests, committed from the box |

## The v2 format

```yaml
schema_version: 2
public:
  - id: P1
    env: {B: 4, N: 128, E: 4}
    inputs:
      - {name: logits, kind: tensor, shape: [4, 128], dtype: float32, axes: [B, N], dist: randn}
      - {name: idx,    kind: tensor, shape: [4], dtype: int64, axes: [B], dist: 'randint:0:E'}
```

The symbolic `dist` is stored **once** as a string; `dist_symbolic` and the per-row resolved `dist` are
**derived** on `expand_compact()`. `env` (the resolved row symbol->value map) is kept — it is not derivable
from axes+shape (composite axes `E*L` are underdetermined; row-only params like `CAUSAL`/`G`/flags have no
tensor home). `expand_compact(v2)` is **byte-identical** to `generate_explicit_shapes(knob)`, so every reader
and `build_from_explicit` work unchanged. The torch-free adapter readers (is_explicit / public_ids / manifest)
read v2 natively; only materialization + the dist accessors expand it.

## GPU re-grade on the v2 corpus (the authoritative validation)

| gate | v2 result | vs knob |
|---|---|---|
| S2 verify | **101/101 PASS** (fresh, bit-exact) | identical |
| C2 oracle-equiv | **83 VERIFIED + 18 CEILING_ACCEPTED = 101/101** | identical |
| corpus consistency | **GREEN** (101 aligned) | identical |
| release planning gate | **PROCEED** (101 eligible, rare-class 1.00) | identical |
| roofline (gpu-rank) | **97 RANKED + 4 FLOOR** | cleaner (the einops env was fixed, so cost-xcheck no longer blocks) |

## Discipline / what the pilot caught

- A naive flip to `generate_explicit_shapes`'s verbose form was **8,123 lines/task** (8.6x). A codex design
  consult ruled MIGRATION-TARGET B (compact v2 + expander), avoiding 101 unreadable files.
- A **5-task pilot** (before flipping all 101) surfaced a real **c4c daemon-path regression**:
  `standalone_freeze_verifier` put only `scripts/verify/` on `sys.path`, so the c4c `from ascendbench ...`
  import in `_shape_ids_for` raised `ModuleNotFoundError` on a FRESH S2 verify (earlier S2 was all cached).
  Diagnosed **not a v2 bug** by proving the committed-v2 P9 inputs == the pre-flip knob P9 inputs byte-equal;
  fixed by adding `REPO_ROOT` to the verifier's path. After the fix the pilot PASSed bit-exact.
- Also fixed 3 pre-existing failures earlier work left (U7 cost fixtures, eval_w2_1 stale shape ids) and a
  ruff `target-version` / `UP017` config trap that was reverting the `datetime.timezone.utc` 3.10 shims.

## Dev-side test-migration (DONE)

Flipping the corpus to v2 broke 81 unit tests whose PREMISE was "the corpus is knob" — the full suite went
from all-green to **81 failed / 2364 passed**. None were code regressions (the GPU re-grade is 101/101); each
test read `tasks/*/shapes.yaml` assuming knob. Fixed across 14 files; the suite is now **2446 passed, 0 failed,
0 skipped**.

- NEW `tests/unit_tests/_knob_corpus.py`: `knob_doc(task)` reads the pre-flip KNOB shapes.yaml from git at the
  all-knob ref `e563ac2` (the M2-writer commit). The knob->v2 / knob->v1 machinery tests (round-trip, parity,
  the materializer, the linter, generate_explicit_shapes) source the knob form from there, decoupling them
  from the working-tree corpus format.
- Format-assertion tests inverted: `test_all_101_knob_docs_are_not_explicit` ->
  `test_all_101_corpus_docs_are_explicit` (+ a `test_knob_docs_are_not_explicit` on git knob); the manifest
  all-101 test now asserts the explicit projection.
- **Vacuous-pass tests caught**: several corpus loops gated on `doc.get("size_knobs")` — on v2 that gate skips
  EVERY row, so the test passes having asserted nothing (a silent coverage loss, worse than a failure). These
  now read the knob via `knob_doc` so the loop runs, plus a `len(tasks) >= 100` guard against a future all-skip.
- `load_task`'s v2 ShapeSpec packing (`name` falls back to id, `params` carry env+inputs) is the documented
  migration choice (dataset.py) — the unchanged-vs-knob test now asserts the v2 packing.
- The v2-only `tasks/_canary` fixture (no knob baseline at the all-knob ref) is excluded from knob-parity loops.

## Open follow-up (non-blocking)

**Authoring tools** (extend_task_shapes / generate_shapes_and_contract / annotate / quarantine / fill_missing /
patch_shapes) still emit knob-format — a follow-up moves them to v2 authoring. Until then `migrate_to_v2.py`
regenerates a task's v2 from its git-history knob.
