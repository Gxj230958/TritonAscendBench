# `scripts/` — bench tooling

Scripts are grouped by role. The subdirs below are the supported pipeline +
provenance tooling and ship in the release tarball.

### `verify/` — severance + equivalence verification
- `standalone_freeze_verifier.py` — turn each frozen `kernel.py` into a standalone,
  version-pinned snapshot and verify original-vs-severed bit-exact equivalence
  (the correctness proof).
- `freeze_upstream_sources.py` — batch-vendor the upstream GPU Triton source as each task's
  `kernel.py` with provenance + sha pins.
- `resolve_upstream_entrypoints.py` — resolve each task's real upstream entrypoint.
- `severance_subprocess_runner.py`, `oracle_consistency_anchor_runner.py` — isolated
  subprocess workers used by the verifier.

### `oracle/` — oracle-equivalence pipeline
- `precompute_oracle_reference.py` (stage 1, CPU) → `run_oracle_equivalence_gpu.py` (stage 2,
  GPU) → `apply_oracle_equivalence_verdict.py` (stage 3, write verdict to meta).

### `provenance/` — upstream provenance + release gating
- `census_source_provenance.py`, `verify_upstream_commit_chain.py`,
  `compute_release_planning_gate.py`, `write_golden_manifest.py`,
  `capture_golden_baselines.py`, `verify_version_matrix.py`.

### `corpus/` — task authoring + validation
- `validate_tasks.py`, `validate_corpus_consistency.py`, `generate_task_scaffolds.py`,
  `generate_oracle_from_kernel.py`, `generate_shapes_and_contract.py`.

### `perf/` — cost-model + timing
- `cross_check_cost_model_gpu.py`, `audit_cost_model_formulas.py`.

### `gpu_pipeline/` — operator shell wrappers around the pipeline scripts

A few scripts are exposed as `tab` subcommands; when `tab` is installed from a pip
wheel (no `scripts/` tree) those degrade with a clear "source-checkout-only" message.

Maintainer-only helper directories are excluded from the release tarball; the
boundary is enforced by a CI tarball smoke check plus a single-direction import grep.
