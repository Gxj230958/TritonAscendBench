# scripts/ + infra/ release-quality cleanup — plan

Status: PROPOSAL (no code changed). 5-group audit (workflow) + cross-verified. For codex
review then user scope + open-question decisions.

## 1. Diagnosis (release-quality problems — all cross-verified)

| # | Problem | Evidence |
|---|---|---|
| P0a | **WHEEL/SOURCE incoherence (scripts)** | `cli.py:202-244` does `from scripts.{source_provenance_census,oracle_pyref_precompute,apply_oracle_equivalence_verdict,verify_upstream_commit_chain,provenance_release_planning_gate} import main`, but pyproject `exclude=[...,"scripts*"]` → `tab census/...` ImportError on a pip wheel |
| P0b | **WHEEL/SOURCE incoherence (infra)** | `perf/peak_table.py` + `integrity/static_checker.py` load `REPO_ROOT/infra/*.toml`, but pyproject `exclude=[...,"infra*"]` → cost-model/version-matrix break on pip install |
| P0c | **Split-blocker: PUBLIC scripts import INTERNAL automation** | `apply_oracle_equivalence_verdict.py:36` + `golden_manifest.py:49` do `import scripts.automation.lib.atomic` while `automation/` is export-ignored → the public/internal split is INCOHERENT until `atomic` (and any shared automation.lib helper) moves to a PUBLIC home (`ascendbench/...` or `scripts/common/`). **Must fix first.** |
| P1a | **Process-noise in shipping code** (zero-tolerance rule) | `codex/V9/R10/M2.1/FIX N/BLOCKER` across ~10 shipping scripts (_pa_*, apply_oracle_equivalence_verdict, golden_manifest, ...). **Comments-only scrub** — verified: `r10_pyref/` is a live DATA PATH (101 dirs / 317 refs, do NOT rename here), but `oracle_pyref_status` is **docstring-only** (0 meta.yaml fields, no code reads it), so the C2-trio process labels scrub now with NO data-path migration |
| P1b | **Chinese in engine-consumed config** (English-only rule) | `infra/peak_table_910b.toml` 38 Chinese lines + `h200.toml` 9; parsed by perf/peak_table.py + cost_model.py + loop/gpu_rank_one.py |
| P1c | **Personal infra leaks into public tree** | hardcoded `/inspire/.../liuliming-...` + reverse tunnel `127.0.0.1:2222` in loop/{README,_wf_diagnose,daemon_a,orchestrator} **AND `probe_triton_env.py` (`/inspire/` text, in scripts/ not loop/)**; RFC_0001_scope.md (FTE/headcount) + upstream_contacts.md are internal planning docs. ⚠️ `.gitattributes` fences only `automation/` (NOT loop/ or helpers) and does not hide a public GitHub mirror — paths MUST be edited, not just export-ignored |
| P2 | **Dead / one-off cruft ships** | renumber/*.json (0 refs, identity maps), _apply_dtype_specs.py (0 refs), automation/spike/* (every probe `raise NotImplementedError`), freeze_source.py (0 refs) + extract_kernel.py (superseded by phase_a_freeze) |
| P3 | CI import-direction guard incomplete; _test_ceiling_acceptance.py misplaced in scripts/; phase_c_docs clobbers shapes.yaml; full_regression.py (bench-team daemon driver) sits at scripts/ root | audit details |

## 2. Release split (public / internal / remove)

**PUBLIC (27)** — the eval framework a releaser/user needs:
- extraction: phase_a_freeze, phase_a_prime, resolve_upstream_entrypoints, _pa_subproc_runner, _pa_oracle_check_runner, oracle_pyref_precompute, oracle_equivalence_gpu_runner, apply_oracle_equivalence_verdict, phase_b_oracle, phase_b_golden_capture, phase_c_docs
- grading/release-gates: validate_tasks, golden_manifest, phase_a_cost_xcheck, audit_cost_model_formulas, validate_corpus_consistency, provenance_release_planning_gate, verify_upstream_commit_chain, verify_version_matrix
- authoring: generate_task_scaffolds, source_provenance_census (CLI-wired `tab census`)
- gpu_pipeline/* (00→40, run_all, lib_common, oracle_equivalence_verify, README)
- infra: Dockerfile.gpu/npu, peak_table_910b/h200.toml (scrub Chinese), version_matrix.toml

**INTERNAL (17)** — bench-team-only, export-ignore:
- automation/ (already fenced), loop/ (daemon orchestration), full_regression.py
- authoring/diagnostic helpers: auto_shapes_filler, auto_patch_shapes_from_wrapper, lint_shapes_against_kernel, per_task_diagnostic, _c2_sweep_analyze, probe_triton_env, build_oracle_digest, multi_launched_manifest, operator_slice
- superseded: freeze_source, extract_kernel
- internal docs: RFC_0001_scope.md, upstream_contacts.md
- `_test_ceiling_acceptance.py` → relocate to `tests/unit_tests/` (NOT internal)

**REMOVE (6)** — dead/one-off:
- renumber/{renumber_map,renumber_changed}.json, _apply_dtype_specs.py, automation/spike/*

## 3. Target layout

```
scripts/
  extraction/   phase_a_freeze phase_a_prime resolve_upstream_entrypoints _pa_subproc_runner
                _pa_oracle_check_runner oracle_pyref_precompute oracle_equivalence_gpu_runner
                apply_oracle_equivalence_verdict phase_b_oracle phase_b_golden_capture
  authoring/    phase_c_docs generate_task_scaffolds source_provenance_census
  grading/      validate_tasks golden_manifest phase_a_cost_xcheck audit_cost_model_formulas
                validate_corpus_consistency provenance_release_planning_gate
                verify_upstream_commit_chain verify_version_matrix
  gpu_pipeline/ (keep as-is)
  internal/     (export-ignore whole dir) full_regression loop/ automation/ + helpers + freeze_source extract_kernel
infra/
  docker/       Dockerfile.gpu Dockerfile.npu
  internal/     (export-ignore) RFC_0001_scope.md upstream_contacts.md
  peak_table_910b.toml peak_table_h200.toml version_matrix.toml   (or → ascendbench/data/, see Q)
```

## 4. Cross-cutting fixes (regardless of layout)
- **Fix wheel/source coherence** (P0): EITHER package the engine's hard deps (move the 5 cli-imported scripts' logic, peak_table/version_matrix into ascendbench/) OR declare the release vehicle = source tarball + make `tab census` degrade gracefully. (decision → Q2/Q6)
- **Scrub process-noise** (P1a) from PUBLIC scripts — **comments-only**. `r10_pyref/` is a live data path (do NOT rename); `oracle_pyref_status` is docstring-only, so the C2-trio labels scrub cleanly with no migration.
- **Scrub Chinese** (P1b) from peak_table TOMLs → English.
- **De-personalize paths** (P1c): `/inspire/.../liuliming` + tunnel (loop/) + `probe_triton_env.py` `/inspire/` → env vars / config. `.gitattributes` fences only `automation/` (NOT loop/ or helpers) and does not hide a GitHub mirror, so these paths MUST be edited — not left to export-ignore.
- **Complete the CI guard**: extend `test_release_boundary._BENCH_CORE_PATHS` to all PUBLIC scripts.

## 5. Blast radius — machine-enumerated with a PRECISE counting rule
(Earlier drafts said "67 callers"; that was a `grep -l` FILE count that conflated real imports with
string/comment mentions — a counting error codex caught. Corrected below by reference KIND.)

**Will break on a MOVE** (executable refs):
- **package → scripts imports: 2 files** — `engine/cli.py:202-244` (5 `from scripts.X import main`),
  `release_gate/eligibility.py:235/264` (`from scripts.golden_manifest import ...`). (Other ascendbench
  files only NAME scripts in comments/docstrings, e.g. dataset.py:72, peak_table.py:12 — NOT imports.)
- **scripts/ internal AST imports: ~7** incl. SIBLING chains — phase_a_prime→phase_a_freeze,
  resolve→_pa_oracle_check_runner, etc. (subprocess `scripts/X.py` launchers add more — see next bullet).
- **shell / CI / subprocess path-calls: ~10 files** — `.github/workflows/{ci.yml,verified_runner.yml}`,
  `gpu_pipeline/*.sh`, `automation/runners/{tab_quickstart,setup_machine_a_gpu}.sh` call `scripts/X.py` by path.
- **tests AST imports: ~8** (≈35 test files mention a moved script by name, but only ~8 import one).
≈ **~17 imports + ~10 path-calls = ~27 executable callers** that a MOVE breaks.

**Will NOT break on a move** (~305): tasks/*/contract.md + kernel.py provenance headers + frozen/deps
strings NAME scripts — a MOVE is fine; only a RENAME would touch them. `r10_pyref/` is a data path that
STAYS PUT (comment labels scrubbed, no migration).

**Implication (unchanged)**: across 4 rounds codex kept finding executable callers a hand list had missed
(2 package + ~7 sibling-chain + ~10 shell/CI + ~8 tests, scattered). A hand-lockstep move is unsafe; Stage 2,
if done at all, needs an AUTOMATED caller-rewrite (grep-enumerate → rewrite → 0 residual + full 101-task GPU
regression), like the package-reorg rewriter — NOT a manual edit.

## 6. Open questions (need YOUR decision before execution)
1. **DROP vs KEEP `automation/`** — already fully fenced from tarball/wheel/docker, but still in git (a public GitHub mirror exposes the LLM-loop source). Delete from the public branch, or keep fenced?
2. **Release vehicle** — is the public artifact the git-archive **source tarball** (ships scripts/+infra/, needs export-ignore fences) or the **pip wheel** (ascendbench* only, needs engine deps packaged)? This decides P0's fix shape.
3. **peak_table/version_matrix** → move into `ascendbench/data/` (fixes wheel, changes daemon gpu_rank_one resolution) vs keep in infra/ + fix MANIFEST/packaging?
4. **freeze_source + extract_kernel** — `git rm` outright (0 live importers) vs keep as internal historical reference?
5. **phase_c_docs clobber** — add `--no-clobber` additive-scaffold guard for external contributors, or keep destructive-overwrite default?
6. **source_provenance_census** — keep as public `tab census` (de-noise its V9/M0a vocab) vs hide in a dev-only subcommand?
7. **audit_cost_model_formulas** (cert-stage, advisory/non-gating) — ship PUBLIC with a non-gating note vs prune?

## 7. Recommended scope (revised after the §5 caller enumeration)
**Stage 1 — DO (in-place, NO subdir move, small blast radius). This alone reaches release quality:**
- P0c circular-dep fix (move `automation.lib.atomic` → a PUBLIC home) · P0 packaging fix (importlib.resources)
- P1a process-noise comment-scrub · P1b Chinese scrub · P1c path de-personalize · P2 remove dead cruft
- CI-guard extension · mark INTERNAL via export-ignore (`.gitattributes` rule + a README "internal" note — **NO physical move**)

The public/internal boundary is achieved by export-ignore fences + README, **not** by physically relocating
27 scripts. Every Stage-1 item edits files in place; blast radius is tiny and there is no daemon-path move.

**Caveat (codex):** "reaches release quality" is conditional on the §6 decisions, not unconditional. The
release vehicle (Q2: wheel vs source) shapes HOW P0/P0c packaging is fixed (importlib.resources package-data
vs MANIFEST/source-fence), and the public-mirror policy (Q1: drop vs fence automation/) decides whether
`.gitattributes`-only fencing is enough or the LLM-loop source must leave the public branch. These change the
SHAPE of Stage 1, not WHETHER Stage 1 (without the subdir move) suffices. Current state: `pyproject.toml:159`
excludes `scripts*`+`infra*`; `.gitattributes:15` fences only `automation/`.

**Stage 2 — NOT recommended (cosmetic; blast radius ≫ benefit):** physically relocating scripts into
extraction/grading/authoring subdirs touches **~27 executable callers + sibling chains** (§5), requires an automated
caller-rewrite (a hand-lockstep is unsafe — codex kept finding misses across 3 rounds) **and** a full
101-task GPU regression, and buys only a flat→subdir cosmetic grouping. Flat `scripts/` + a clear internal
fence already reads as release-quality. **Defer** unless you specifically want the subdir taxonomy and accept
the automated-rewrite + regression cost.

## 8. codex review corrections (NEEDS-FIX → all adopted)
- **P0c (NEW, split-blocker)**: PUBLIC scripts import INTERNAL automation —
  `apply_oracle_equivalence_verdict.py:36` + `golden_manifest.py:49` do `import scripts.automation.lib.atomic`
  while automation/ is export-ignored. The public/internal split is INCOHERENT until `atomic` (and any
  other shared automation.lib helper) moves to a PUBLIC home (`ascendbench/...` or `scripts/common/`).
  Must fix before any split ships.
- **R10 is TWO concerns (de-risks P1a)**: `r10_pyref/` IS a live data path (101 dirs / 317 refs, read by
  precompute + GPU runner + golden_store + loop sync + shell) → renaming it is a migration. BUT
  `oracle_pyref_status` is docstring-only (not a live schema key). ⇒ **process-noise comments CAN be
  scrubbed now without touching r10_pyref**; only a path rename is a coordinated migration. (§1 P1a and
  §7 corrected — it is NOT inherently a data-path migration.)
- **Blast radius additions (§5)**: source_provenance_census→verify_upstream_commit_chain;
  oracle_pyref_precompute→oracle_equivalence_gpu_runner; loop/gpu_rank_one→oracle_equivalence_gpu_runner;
  _test_ceiling_acceptance.py; `.github/workflows/ci.yml` path lists; oracle_equiv_one auto-apply/cert subprocs.
- **P0 fix sharper**: package engine deps via `ascendbench` modules + `importlib.resources` package-data
  (NOT `REPO_ROOT/infra`). Third option: keep `scripts/*.py` as thin source wrappers over packaged
  `ascendbench.tools.*`. Source-tarball-only is sound ONLY if broken-wheel CLI commands degrade/are removed.
- **peak_table move is lockstep**: `loop/gpu_rank_one.py:190` defaults to `infra/peak_table_h200.toml` and
  `orchestrator.py:137` sync explicitly includes `infra/`; moving to `ascendbench/data/` must update both.
- **Classification fix**: source_provenance_census is CLI-wired + unit-tested but does NOT feed the hard
  eligibility gate (overstated in §5); still public. full_regression internal confirmed. phase_c_docs needs
  the `--no-clobber` guard if shipped public.
- **P1c additions**: probe_triton_env.py also has `/inspire/` text (in scripts/, not loop/); export-ignore
  fences automation/ only (not loop/ or internal helpers) AND does not hide a public GitHub mirror —
  so de-personalization must actually edit the paths, not rely on export-ignore alone.
