# Plan — move tasks/_frozen_* into triton_ascend_bench/frozen/, clean tasks/ to level1/2/3 only

Date 2026-06-08. Built by a 5-dimension Workflow inventory + synthesis; pending codex deep review before any execution. Branch: `frozen-into-package` off `renumber-public-release`. Lands as ONE revertible commit; nothing pushed without user approval.

## Target layout
```
triton_ascend_bench/frozen/
  shim/         (was tasks/_frozen_shim)      — importable inert symbol stubs
  deps/         (was tasks/_frozen_deps_pool) — importable vendored dep closures (h_<sha>/...)
  snapshots/    (was tasks/_vendored)         — upstream snapshots, DATA (attribution), not imported
tasks/  level1/ level2/ level3/               — corpus ONLY
```
Import rewrite: `tasks._frozen_shim` → `triton_ascend_bench.frozen.shim`; `tasks._frozen_deps_pool` → `triton_ascend_bench.frozen.deps`. Deletions: `tasks/_corrected`, `tasks/_deprecated` (SCORE.md keeps a one-line off-by-one note + docs/verification proof). Seal: `tasks/_canary` (git rm --cached + gitignore, maintainer copy). Delete `tasks/__init__.py`.

## Why a full GPU S2 rerun is MANDATORY (not a dev-side SHA edit)
Unlike the renumber precedent (header-comment only; `golden bytes unchanged`), this rewrites **executable import statements** in 69 severed kernels. Only a real S2 run faithfully establishes: (1) the severed subprocess actually executes the NEW import path and dispatches run() — S2 verifies the rewritten bytes; (2) the severed import-audit resolution changes (pip env resolves `triton_ascend_bench.frozen.*` from site-packages, forbidden until the allowlist edit); (3) `h_af43637ae818` (a shim-bearing pool module) changes content → its content-address must be re-derived. Bumping `standalone_sha256`/`kernel_sha256` to sha256(new kernel.py) WITHOUT rerunning S2 = validate passes + gate green while claiming GPU verification that never ran = tune-to-pass, forbidden. The SHA recompute is the **by-product** of the daemon S2 rerun (Step 14), after all edits are final.

## Ordered steps
1. Branch `frozen-into-package`; record pre-change SHA for rollback.
2. `git mv` the 3 dirs into `triton_ascend_bench/frozen/{shim,deps,snapshots}` (bytes preserved); add `frozen/__init__.py`.
3. **Packaging foundation**: add `__init__.py` to all 8 `frozen/deps/h_<sha>/` dirs (+ confirm nested fla_ops/sglang_* have theirs) — they were PEP-420 namespace pkgs resolvable only via tasks/ on path; `find_packages` won't ship them otherwise (this is THE pip-install fix). Ship `frozen/snapshots/*.py` as package-data/MANIFEST (data, not import).
4. Rewrite generator `scripts/phase_a_prime.py`: `FROZEN_SHIM`/`FROZEN_DEPS_POOL` consts (L69-70) → package anchor; ~62 emit-string sites + `_POOL_IMPORT_RE` (L6691) → `triton_ascend_bench.frozen.{shim,deps}`.
5. Rewrite severed runner `scripts/_pa_subproc_runner.py`: drop obsolete `tasks/_frozen_*` sys.path block (keep REPO_ROOT insert); **add `triton_ascend_bench` to `_SEVERED_ALLOWED_TOP` allowlist** (else pip-env audit false-blocks every task as a leak).
6. Rewrite other consumers: `phase_a_freeze.py`/`freeze_source.py`/`phase_b_golden_capture.py` (snapshots path), automation prompts (_vendored), `validate_tasks.py`/`dataset.py`/`adversarial.py`/`cli.py`/`eval.py`/`param_constraints.py` (CANARY_ROOT + vendored consts).
7. Bulk-rewrite on-disk: sed `tasks._frozen_shim`→`triton_ascend_bench.frozen.shim` and `tasks._frozen_deps_pool`→`triton_ascend_bench.frozen.deps` over 69 kernel.py + 2 _pa_launcher comments + the in-pool fla_ops modules; rewrite 23 meta `frozen_deps` + 96 meta `vendored_snapshot` paths. (Makes both SHAs stale by design — recomputed at Step 14.)
8. Update tests: `test_phase_a_prime_rewrite.py` + `test_phase_a_prime_vendor.py` asserted strings → new prefixes; canary-seal robustness.
9. Delete `_corrected` + `_deprecated`; collapse SCORE.md path refs to one off-by-one line (keep docs/verification proof + upstream_reports).
10. Seal `_canary`: `git rm --cached -r tasks/_canary` + `.gitignore` + `.gitattributes export-ignore` (working copy kept).
11. Delete `tasks/__init__.py` LAST (safe only after every `from tasks._frozen_*` is rewritten).
12. Dev sanity (no GPU): `pip install -e .`, import `triton_ascend_bench.frozen.{shim,deps,deps.h_<sha>.<mod>}`, load one rewritten kernel by path, `git grep -nP 'tasks\._frozen'` == 0.
13. Unit suite (no GPU): `pytest tests/unit_tests -q` → 0 failed / 0 skipped (bar the intended sealed-canary skip on a clean clone).
14. **FULL GPU S2 rerun** (daemon mailbox) for the 69 affected tasks — executes the rewritten bytes; THIS produces the fresh `standalone_sha256`/`kernel_sha256` (re-vendor `h_af43637ae818` to a fresh content-address in lockstep).
15. Re-apply C2 oracle-equivalence verdict (`apply_oracle_equivalence_verdict.py`) so golden cross_check re-promotes to S2_FRESH.
16. **FULL 101-task GPU regression** (validate + S2 + C2 + cost_model_xcheck + golden + release_gate + gpu-rank).
17. Two-env verify: repo checkout AND fresh `pip install .` both import `triton_ascend_bench.frozen.*`; build wheel, confirm shim/deps shipped + importable.
18. Squash to ONE clean commit → mandatory codex review → (await user) push + daemon sync.

## Biggest risks (from the Workflow synthesis)
1. **SHA/S2 false-positive** (headline): hand-recomputing SHAs to silence validate without the Step-14 daemon S2 rerun → claims verification that never happened; next C2 flips cross_check → CROSS_INCONSISTENT, gate RED.
2. **Silent canary death**: `git rm --cached` + gitignore → fresh clone/CI has no `tasks/_canary`; `dataset.py:319`/`validate_tasks.py:1129` return `[]` with NO warning → 0 canaries scored silently. Mitigate: test skip guard + `TAB_CANARY_ROOT` env override + never `git clean -fdx` the maintainer copy.
3. **Packaging trap**: the 8 `h_<sha>/` dirs have NO `__init__.py` (PEP-420 via tasks/ path) → `find_packages` won't ship them → same ModuleNotFoundError persists post-move. Must add all 8 + nested.
4. **Severed import-audit false-block**: `_SEVERED_ALLOWED_TOP` lacks `triton_ascend_bench` → pip-env audit flags the shim/pool as a leak → every reverified task BLOCKED. Add to allowlist BEFORE the S2 rerun.
5. **Missed generator/meta**: `_POOL_IMPORT_RE` is a separate regex (stale → future promotes record `frozen_deps=[]` silently); updating only one of the two SHAs half-satisfies the gate; un-rewritten emit-strings re-emit old paths on future regen.
6. **`h_af43637ae818` content-address lie**: in-place sed of the shim-bearing pool module keeps the dir name while its content changed → dir-name ≠ content hash. Faithful fix = re-vendor to a fresh `h_<sha>` during the S2 rerun.

## Rollback
All on branch `frozen-into-package`, ONE commit, unpushed. Abort: `git reset --hard <pre-change SHA>` or delete the branch — git mv is tracked, restores all dirs verbatim. `_canary` working copy untouched by reset (only untracked). If already pushed: revert the commit + `orchestrator sync-code --delete` to drop the stale daemon tree.

## Scope
Medium refactor, ~1 day. Dev edits ~2-4h (3 git mv + ~10 __init__.py + pyproject + sed 69 kernel/2 launcher/4 pool/~100 meta + phase_a_prime ~62 sites + _POOL_IMPORT_RE + 6 consumers + 6 tests + SCORE/gitignore + deletes). Then ONE mandatory full GPU regression (NOT samplable): 69-task S2 rerun (sole faithful SHA source) + C2 re-apply + all-101 pipeline. Risk is in S2/SHA provenance discipline + packaging completeness, not line count. ~75 code + ~100 meta + ~12 config/test/doc files.

---

## v2 — post-codex-deep-review resolutions (the 6 must-fixes)

Two codex passes reviewed the v1 plan. v1 was ruled **incomplete + unsafe as written**; the S2 linchpin was then nailed separately. Resolutions:

1. **S2 rerun — confirmed mandatory AND confirmed semantically SOUND.** codex: a consistent module-path rename of identical (git-mv'd) shim/deps bytes on BOTH the original (td/kernel.py) and severed (staged) sides cannot change the severed-vs-original verdict; the live-vllm entrypoint path is untouched. The rerun **revalidates the rewritten kernel's severance** (both sides new-path), it does NOT compare against a pre-rewrite snapshot. So no S2-semantic breakage — but the rename must be complete.
2. **Corpus rewrite is MANDATORY, generator-only is insufficient (codex C correction).** `build_staged_candidate` → `rewrite_imports_to_standalone` COPIES already-severed `from tasks._frozen_*` lines through the fallthrough path (phase_a_prime.py:6225). So updating only the generator/emit-strings leaves staged copying the OLD literal → staged imports a moved dir → fail. The 69 kernel.py + generator emit-strings + `_POOL_IMPORT_RE` must ALL be rewritten in lockstep.
3. **Reference inventory — v1 MISSED these, add to worklist:** `NOTICE:16/36` (vendored attribution — also a release-NOTICE concern), `README.md:66`, `specs/task_api_contract.md:22`, `scripts/phase_c_docs.py:1003`, `scripts/automation/summary_metrics.py:116`, `scripts/resolve_upstream_entrypoints.py:277`, `scripts/loop/_wf_authoritative_route.js:56` + `_wf_diagnose_remaining.js:37`, `triton_ascend_bench/upstream_shims/vllm_runtime_hygiene.py:500`, and delete `tests/unit_tests/test_corrected_variant_nonscored.py`.
4. **h_a9f952fcc731 (new blocker).** Besides `h_af43637ae818` (used by L2_23, re-vendor to a fresh content-address during the S2 rerun since its bytes change), the ORPHAN `h_a9f952fcc731` also carries an old-shim import. It is unused by any task → **DELETE it** (the audit already flagged it as cruft) rather than re-vendor.
5. **CANARY_ROOT override must be IMPLEMENTED FIRST (codex point 5 — the plan assumed a `TAB_CANARY_ROOT` that does NOT exist).** `CANARY_ROOT` is hardcoded to `tasks/_canary` (dataset.py:28/33, validate_tasks.py); a missing root silently returns `[]` (dataset.py:310, validate_tasks.py:1127) → canary scoring dies with no error. Before sealing: (a) add a `TAB_CANARY_ROOT` env override to both consumers, (b) add a test, (c) THEN `git rm --cached` + gitignore. **GENUINE USER DECISION:** seal only matters if the public repo is submitter-visible. If the public release is maintainer-only, the canary can stay tracked as-is and this whole item drops.
6. **Packaging — make discovery explicit.** Add `__init__.py` to all 7 (after deleting the orphan) `frozen/deps/h_<sha>/` dirs + nested fla_ops/sglang_*; ship `frozen/snapshots/*.py` via `[tool.setuptools.package-data]` (data, not importable); confirm `find_packages` resolves `triton_ascend_bench.frozen.deps.h_<sha>.<mod>` from a built wheel; keep `exclude=['tasks*']`.

**Net:** the refactor is sound and worth doing (fixes the real pip-install ModuleNotFoundError + cleans tasks/ to level1/2/3). The only genuine USER decision is item 5's release model (submitter-visible → seal canary; maintainer-only → skip sealing). Everything else is mechanical engineering gated by the full GPU S2 rerun.
