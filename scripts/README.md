# `scripts/` — bench tooling

Scripts are grouped by role. Public (shipped) subdirs hold the supported pipeline +
provenance tooling; `internal/` holds dev-only helpers that are export-ignored from the
release tarball. `loop/` and `automation/` are also internal (closed-loop orchestration).

## Public subdirs (ship in the release tarball)

### `verify/` — severance + S2-equivalence verification
- `standalone_freeze_verifier.py` — turn each frozen `kernel.py` into a standalone,
  version-pinned snapshot and verify S2 original-vs-severed equivalence (the correctness proof).
- `freeze_upstream_sources.py` — batch-vendor the upstream GPU Triton source as each task's
  `kernel.py` with provenance + sha pins.
- `resolve_upstream_entrypoints.py` — resolve each task's real upstream entrypoint.
- `severance_subprocess_runner.py`, `oracle_consistency_anchor_runner.py` — isolated
  subprocess workers used by the verifier.

### `oracle/` — oracle-equivalence pipeline (R10)
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

## Internal (export-ignored — NOT in the release tarball)

- `internal/` — dev-only helpers: `run_full_regression.py`, `extract_upstream_kernel.py`,
  `freeze_upstream_snapshot.py`, `slice_single_operator_kernel.py`,
  `list_multi_launch_tasks.py`, `classify_correctness_sweep.py`, `diagnose_blocked_tasks.py`,
  `fill_missing_shapes.py`, `patch_shapes_from_wrapper.py`, `lint_shapes_against_kernel.py`,
  `build_oracle_kernel_digest.py`, `probe_triton_env.py`, `test_ceiling_acceptance.py`.
- `loop/` — closed-loop autonomous verification orchestration (dev ↔ tunnel ↔ gpfs ↔ GPU).
- `automation/` — internal orchestrator + its libraries.
- `gpu_pipeline/` — operator shell wrappers around the public pipeline scripts.

A few public scripts are exposed as `tab` subcommands; when `tab` is installed from a pip
wheel (no `scripts/` tree) those degrade with a clear "source-checkout-only" message.

The release boundary is enforced in CI: a tarball smoke check rejects any internal path
(`scripts/internal/`, `scripts/loop/`, `scripts/automation/`) that leaks, and a
single-direction grep rejects any shipped module importing `scripts.automation`.
