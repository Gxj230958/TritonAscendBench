# `scripts/` — bench tooling

Two kinds of scripts live here, split by release boundary.

## Public (ships in the release tarball)

Pipeline and provenance tooling invoked by the `tab` CLI or run directly
from a source checkout. These are part of the supported surface:

- `phase_a_prime.py`, `phase_b_oracle.py` — per-task correctness verification.
- `oracle_pyref_precompute.py`, `oracle_equivalence_gpu_runner.py`,
  `apply_oracle_equivalence_verdict.py` — oracle-equivalence pipeline.
- `resolve_upstream_entrypoints.py`, `verify_upstream_commit_chain.py`,
  `source_provenance_census.py`, `provenance_release_planning_gate.py` —
  upstream provenance + release gating.
- `validate_tasks.py`, `validate_corpus_consistency.py`,
  `generate_task_scaffolds.py`, `phase_c_docs.py` — corpus authoring/validation.
- `golden_manifest.py`, `audit_cost_model_formulas.py`, the C2 trio,
  `_pa_*` runners — graders and helpers used by the above.

A few of these are exposed as `tab` subcommands. When `tab` is installed from
a pip wheel (no `scripts/` tree), those subcommands degrade with a clear
"source-checkout-only" message instead of an import error.

## Internal (export-ignored — NOT in the release tarball)

Dev-only tooling, fenced via `.gitattributes` (`export-ignore`) and
`.dockerignore`. Not supported as a public interface and may change freely:

- `loop/` — closed-loop autonomous verification orchestration
  (dev ↔ SSH tunnel ↔ gpfs ↔ GPU host).
- `automation/` — internal orchestrator + its libraries.
- `full_regression.py` — full-pipeline regression driver.
- corpus-authoring / diagnostic helpers: `auto_shapes_filler.py`,
  `auto_patch_shapes_from_wrapper.py`, `lint_shapes_against_kernel.py`,
  `per_task_diagnostic.py`, `_c2_sweep_analyze.py`, `probe_triton_env.py`,
  `build_oracle_digest.py`, `multi_launched_manifest.py`, `operator_slice.py`,
  `freeze_source.py`, `extract_kernel.py`, `_test_ceiling_acceptance.py`.

The boundary is enforced in CI: a release-tarball smoke check rejects any
internal path that leaks, and a single-direction import grep rejects any
shipped module that imports `scripts.automation`.
