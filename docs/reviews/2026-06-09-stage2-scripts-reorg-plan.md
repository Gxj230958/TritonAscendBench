# Stage 2 — scripts/ reorg + full rename + Frozen-by removal (PLAN)

Status: DRAFT PLAN (no code changed). Must pass codex deep-review to 0-FAIL before ANY
implementation, then implement as ONE atomic scripted change, then a FULL 101-task GPU
regression must be GREEN before commit/push.

User-confirmed scope (do NOT re-ask):
1. Move all 34 top-level `scripts/*.py` into semantic subdirs.
2. Rename every cryptic script to a self-documenting name; rename the cryptic `tab` CLI
   subcommands too. Change size is fine — go for what is reasonable.
3. REMOVE the `Frozen by: scripts/X.py` provenance lines entirely (existing 109 byte-locked +
   stop emitting them going forward). The user considers these lines a design smell and wants
   them gone, not left stale.

## A. Rename + reorg map (34 scripts)

Subdir rule: `internal/` = the dev-only EXPORT-IGNORE fence set (defined by fence, not topic —
lets the 13 individual export-ignore lines collapse to one `scripts/internal/`). The other 5
subdirs = semantic role for shipped/core scripts. Tightly path-coupled clusters are CO-LOCATED
so their intra-cluster sibling-imports / `_SCRIPTS_DIR/"X.py"` spec-loads need ZERO edits.

**verify/** (severance + S2-equivalence verify chain — co-located):
- phase_a_prime.py            -> verify/standalone_freeze_verifier.py
- phase_a_freeze.py           -> verify/freeze_upstream_sources.py   (phase_a_prime does `import phase_a_freeze`)
- _pa_subproc_runner.py       -> verify/severance_subprocess_runner.py
- _pa_oracle_check_runner.py  -> verify/oracle_consistency_anchor_runner.py
- resolve_upstream_entrypoints.py -> verify/resolve_upstream_entrypoints.py (clear; move only)

**oracle/** (R10 oracle-equivalence pipeline — co-located):
- oracle_pyref_precompute.py        -> oracle/precompute_oracle_reference.py
- oracle_equivalence_gpu_runner.py  -> oracle/run_oracle_equivalence_gpu.py
- apply_oracle_equivalence_verdict.py -> oracle/apply_oracle_equivalence_verdict.py (clear; move only)

**provenance/** (upstream provenance + release gating):
- source_provenance_census.py        -> provenance/census_source_provenance.py
- verify_upstream_commit_chain.py     -> provenance/verify_upstream_commit_chain.py (clear; move only)
- provenance_release_planning_gate.py -> provenance/compute_release_planning_gate.py
- golden_manifest.py                  -> provenance/write_golden_manifest.py
- phase_b_golden_capture.py           -> provenance/capture_golden_baselines.py
- verify_version_matrix.py            -> provenance/verify_version_matrix.py (clear; move only; CI-referenced)

**corpus/** (task authoring + validation):
- validate_tasks.py             -> corpus/validate_tasks.py (clear; move only)
- validate_corpus_consistency.py-> corpus/validate_corpus_consistency.py (clear; move only)
- generate_task_scaffolds.py    -> corpus/generate_task_scaffolds.py (clear; move only)
- phase_b_oracle.py             -> corpus/generate_oracle_from_kernel.py
- phase_c_docs.py               -> corpus/generate_shapes_and_contract.py

**perf/** (cost-model + timing):
- phase_a_cost_xcheck.py        -> perf/cross_check_cost_model_gpu.py
- audit_cost_model_formulas.py  -> perf/audit_cost_model_formulas.py (clear; move only)

**internal/** (the 13 export-ignored dev-only helpers — fence collapses to one dir line):
- full_regression.py                 -> internal/run_full_regression.py
- auto_shapes_filler.py              -> internal/fill_missing_shapes.py
- auto_patch_shapes_from_wrapper.py  -> internal/patch_shapes_from_wrapper.py
- lint_shapes_against_kernel.py      -> internal/lint_shapes_against_kernel.py
- per_task_diagnostic.py             -> internal/diagnose_blocked_tasks.py
- _c2_sweep_analyze.py               -> internal/classify_correctness_sweep.py
- probe_triton_env.py                -> internal/probe_triton_env.py
- build_oracle_digest.py             -> internal/build_oracle_kernel_digest.py
- multi_launched_manifest.py         -> internal/list_multi_launch_tasks.py
- operator_slice.py                  -> internal/slice_single_operator_kernel.py (imports multi_launched_manifest -> co-located)
- freeze_source.py                   -> internal/freeze_upstream_snapshot.py
- extract_kernel.py                  -> internal/extract_upstream_kernel.py
- _test_ceiling_acceptance.py        -> internal/test_ceiling_acceptance.py

**CLI subcommands** (cli.py — rename cryptic ones; file moves also force the dispatch-import update):
- census                  -> source-provenance-census
- oracle-pyref-precompute -> precompute-pytorch-oracle-reference
- apply-equivalence-verdict -> apply-oracle-equivalence-verdict
- verify-commit-chain     -> verify-upstream-commit-chain
- r13c-gate               -> release-planning-gate
(already-clear subcommands unchanged: init-solution, run, run-suite, package, submit,
validate-submission, validate-result, static-check, analyze)

Stem-collision check: the map has NO duplicate stems across subdirs (required — conftest +
~30 spec-loaders register modules by short stem in sys.modules; same stem would collide).

## B. Frozen-by REMOVAL (the user's decision; the highest-scrutiny section)

`Frozen by:` appears in **86 byte-locked `tasks/**/kernel.py`** (`Frozen by: scripts/phase_a_freeze.py`)
+ **23 `ascendbench/frozen/deps/**` files** (`Frozen by: scripts/phase_a_prime.py (U2 ... vendoring)`)
= 109. (NOT `frozen/snapshots/` — codex R1 corrected; verified 0 in snapshots.) Emission sites to
stop future emission: phase_a_freeze.py:230, phase_a_prime.py:918 + :1479 — delete the f-string line.

No signing blocker (codex-confirmed): 0 `provenance.sig`, trust_roots has no active key, golden
signing refuses without `$TAB_GPFS_ROOT/keys/<key_id>.priv`. So NO re-sign needed.

The two buckets hash DIFFERENTLY — handle separately:

**(a) 86 task `kernel.py`** — DUAL-HASH (verified: validate_tasks.py:1288 phase-aware step-8,
validate_corpus_consistency.py:86, phase_a_prime.py:1821 docstring, eligibility.py:513). The
on-disk kernel.py is compared to **`meta.phase_a_prime.standalone_sha256`** (== `kernel_sha256`),
NOT `upstream.file_sha256` (the latter is the ORIGINAL upstream provenance hash, deliberately
PRESERVED unchanged). So per kernel: strip the line, recompute sha256(kernel.py), write the new
value to BOTH `phase_a_prime.standalone_sha256` and `phase_a_prime.kernel_sha256`, and DO NOT touch
`upstream.file_sha256`. (Phase-aware fallback: if a task is legacy/unprocessed — no
`standalone_sha256` — validate compares `upstream.file_sha256`; the strip script must detect which
field is the on-disk-compared one per task and update THAT one. Enumerate the split at impl time.)

**(b) 23 `frozen/deps/**`** — content-addressed by UPSTREAM BODY, not whole-file (verified: dir
`h_119bf34e8937` == header `Content SHA-256: 119bf34e8937…`, while whole-file sha = `9a46…` differs;
`meta.frozen_deps` references `ascendbench.frozen.deps.h_<bodysha>.…` by module path). The `Frozen by:`
line lives in the docstring HEADER, NOT the body, so stripping it does NOT change the body content-SHA
→ dir name `h_<sha>` unchanged → `frozen_deps` references unchanged. Deps need NO sha update — just
strip. MUST still confirm at impl time that no WHOLE-FILE dep hash is recorded anywhere.

Other anchors over kernel.py to re-confirm at impl time (codex R1 open item): the resolver code-fp /
`_PROVENANCE_CODE_FILES` (resolve_upstream_entrypoints.py) — regenerate if it hashes kernel.py; the
golden envelope `kernel_py_sha256` (rebuilt on next golden run, no sig); the kernel.py header's own
`File SHA-256 (this vendored copy)` self-line is the SNAPSHOT sha (unaffected by editing kernel.py).

Approach: deterministic `strip_frozen_by.py` — per file removes EXACTLY the `Frozen by:` line (assert
diff touches only that line), and for bucket (a) recomputes + writes the correct phase-aware field.
NO re-freeze of upstream code (zero drift). Then `validate_tasks` + `validate_corpus_consistency`
must pass repo-wide.

## C. Reference updates (~95 functional python + ~40 path/config; PATH-SCOPED, not token-scoped)

CRITICAL footgun: `phase_a_prime` is ALSO a meta.yaml field key (`meta["phase_a_prime"]["status"]`,
`phase_a_prime.standalone_sha256`) and the human "Phase A'" status name. A blind
`s/phase_a_prime/standalone_freeze_verifier/g` corrupts 232 test assertions + 103 meta reads.
The rewrite MUST be path/import-scoped (rename the FILE, the module import `import phase_a_prime`,
and `scripts/phase_a_prime.py` PATH literals) and NEVER the meta field key or status string.

1. Repo-root depth anchors (~20 sites) — they use MULTIPLE idioms, NOT only `.parent.parent`
   (codex R2-1): also `_SCRIPTS_DIR.parent` (phase_a_cost_xcheck:102), `.parents[1]`
   (multi_launched_manifest:27, operator_slice:25), `THIS.parent.parent` (auto_shapes_filler:56),
   plain `.parent.parent` (most). Nesting deepens by 1, so the impl driver enumerates EACH moved
   script's ACTUAL idiom (do NOT assume one form) and adds exactly one level to it, then verifies
   every downstream `REPO_ROOT/"tasks"`, `.parent/"infer"`, `REPO_ROOT/"ascendbench"/...` still
   resolves. Gate: `import-smoke + --help` every moved script (a wrong depth fails fast on path use).
2. Sibling-import / `_SCRIPTS_DIR/"X.py"` spec-loads: intra-cluster (co-located) stay valid —
   only update the literal filename stem. ~6 genuine cross-subdir edges qualified explicitly:
   run_oracle_equivalence_gpu->verify/oracle_consistency_anchor_runner; cross_check_cost_model_gpu
   ->verify/severance_subprocess_runner; test_ceiling_acceptance->oracle/run_oracle_equivalence_gpu;
   slice_single_operator_kernel->internal sys.path; precompute_oracle_reference->oracle/ (co-located);
   audit_cost_model_formulas->comment-only.
3. cli.py 5 dispatch imports `from scripts.X import main` -> `from scripts.<sub>.<new> import main`.
4. ascendbench/release_gate/eligibility.py:235,264 `from scripts.golden_manifest` ->
   `from scripts.provenance.write_golden_manifest`.
5. loop/ + automation/ runtime importers + subprocess path strings (oracle_equiv_one, gpu_rank_one,
   gpu_grade_one, bench_evaluator_daemon:419, tier1_reverify:72) -> new subdir+stem.
   PLUS (codex R5): `full_regression.py:240` runs `scripts/validate_tasks.py` by path — after
   validate_tasks -> corpus/ AND full_regression -> internal/, update that subprocess path. (Re-grep
   ALL of scripts/ for intra-scripts `scripts/<oldstem>.py` subprocess literals — do not rely on this
   list being exhaustive; the impl driver greps to 0.)
6. ~30 tests (spec_from_file_location path literals + `from scripts.X import`) + conftest
   `_GUARDED_KEYS`.
7. Path/config: gpu_pipeline/*.sh, automation/runners/*.sh, ci.yml (139/153/143 + line-48 dev-skip
   grep add `scripts/internal`), verified_runner.yml, .gitattributes/.dockerignore (collapse 13
   lines -> one `scripts/internal/`), test_release_boundary `_BENCH_CORE_PATHS` (21 entries ->
   new subpaths) + regex/allowlist add `scripts/internal/`, the READMEs.
8. Editable prose: update copy-pasteable `python scripts/X.py` runbook commands in docs/specs;
   LEAVE historical/conceptual "phase_a_prime" phase-name mentions (only the FILE renames).

## D. Verification gate (cheap -> expensive; ALL must pass)
1. grep-0: no functional ref to any old stem in `*.py *.sh *.yml *.toml` EXCEPT byte-locked
   tasks/ + ascendbench/frozen/. AND `Frozen by:` = 0 — but SCOPED to exclude docs (codex R6: this
   plan + older review docs literally contain "Frozen by:", so the grep MUST be
   `git grep 'Frozen by:' -- ':!docs'` else it self-trips). After step B both greps are 0 over
   code+tasks+frozen.
2. Import smoke: the 5 cli-dispatched new modules + one per subdir import; `import ascendbench.engine.cli`;
   `tab source-provenance-census --help` etc.
3. Unit suite: 2277 passed, 0 failed, 0 skipped.
4. Tarball boundary: scripts/internal/ + automation/ + loop/ export-ignored; 21 core scripts
   present at new subpaths; no shipped .py imports a fenced module.
5. validate_tasks + validate_corpus_consistency green (proves the Frozen-by strip + sha recompute
   is consistent).
6. FULL 101-task GPU regression (run LAST, via the 4-card daemon): unit + s2 + c2 + gpu-rank +
   release-gate all GREEN — the only gate proving the moved spec-load paths + gpu_pipeline shell
   rewrites + the new run_full_regression.py path resolve at runtime, and that the Frozen-by
   strip did not break any task's verify/oracle/cost/golden stage.

### Bonus change (codex R7 + the L3_30 flake co-review): add a bounded S2 retry
`full_regression.py::stage_s2` (line ~223) currently calls non-PASS s2 a hard FAIL with NO retry,
so a single GPU-nondeterminism flake (L3_30, AGREE-FLAKY) false-reds the whole regression. Add a
bounded retry: when an s2 task verdict is BLOCKED specifically on `S2-gpu NOT equivalent
(phase=compare) ... NOT bit-exact`, re-submit ONLY that task up to N=2 times before calling FAIL;
preserve the diff/artifacts on persistent failure. Do NOT retry import-leak / no-run / shapes /
resolver / audit failures (those are real). This makes the Stage-2 GPU regression robust to the
known flake. Implemented in the same atomic change (full_regression.py is being moved to
internal/run_full_regression.py anyway).

## E. Sequencing — ONE atomic scripted change
Single logical commit. NOT split move-now/fix-later (any intermediate state breaks all importers
+ the boundary test). Driver:
  1. `rename_scripts.py`: literal old->new map; `git mv` each (preserve blame).
  2. `strip_frozen_by.py`: remove the 109 lines + recompute/update shas (section B).
  3. path-scoped ref-rewrite (section C) as explicit per-file edits (NOT blind sed).
  4. change the 3 emission sites.
  5. run gate steps 1-5 locally; iterate to green.
  6. codex review of the driver + cross-subdir edits + the Frozen-by strip + sha recompute.
  7. push + orchestrator sync-code (force-touch meta + rsync --delete) + gate step 6 (GPU regression).

## F. Residual risks for codex to stress-test
- Frozen-by strip (RESOLVED per codex R1/R2/R3): task kernels update
  `phase_a_prime.standalone_sha256`+`kernel_sha256` phase-aware (NOT `upstream.file_sha256`);
  deps strip needs NO sha update (body-content-addressed); 0 sigs confirmed so no re-sign. Remaining
  impl-time confirm: the resolver code-fp / `_PROVENANCE_CODE_FILES` regen, and that no whole-file
  dep hash exists. Decision: REMOVAL (user's call) is now correctly specified; exact-line strip is
  safer than re-freeze (codex agreed); accept-stale+NOTICE is the fallback ONLY if impl uncovers a
  signed/unrecoverable anchor (none found).
- path-scoped vs token-scoped rewrite for phase_a_prime (field-key collision) — codex R4 confirmed
  valid; also check validate_tasks/golden_manifest/resolve stems for the same key-vs-filename clash.
- implicit-namespace-package: `from scripts.<sub>.X` works without new `__init__.py` (repo-root
  on sys.path) — confirm no direct `python scripts/<sub>/X.py` exec path needs repo-root first.
- stem-collision freedom across subdirs (sys.modules short-name registration).
- whether removing Frozen-by is genuinely better than accept-stale+NOTICE (the safer alt the
  strategy analysis preferred) — codex should give an explicit recommendation; if it judges the
  byte-locked-file edit too risky vs the user's stated preference, surface that tension.
