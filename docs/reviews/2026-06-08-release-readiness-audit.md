# Release-Readiness Audit — triton-ascend-bench

Date: 2026-06-08. Method: 4-dimension parallel audit (`_xx` dirs / release artifacts / corpus+reproducibility / docs) + synthesis. **Verdict: NOT ready to ship publicly as-is** — 5 blockers, all fixable.

## Good news (the #1 worry is clear)

Eval is a **real standalone path**: the shipped `triton_ascend_bench/` package + `eval.py`/`cli.py` have **0** daemon/mailbox/ssh/internal-path refs. `tab run-suite --backend gpu` resolves all 101 tasks on an external user's own GPU — the bench does NOT hard-depend on the author's private daemon infra. Corpus is complete (all 101 have kernel/meta/shapes/contract/oracle).

## Release blockers (must fix before public)

1. **Internal infra + process docs are git-tracked → public on GitHub.** 67 internal docs (`docs/reviews`=25 run-reports with absolute `/inspire/...` paths + PASS/FAIL internals, `docs/triage_backlog`=19, `docs/plans`=18, `docs/release`=4, `docs/brainstorms`=1) + `scripts/loop/` (11 files hard-coding `/inspire/hdd/.../liuliming-liuliming/` + `127.0.0.1:2222` jumphost) are browsable on a public repo. `export-ignore` only strips the tarball, not GitHub. → `git rm -r --cached` the internal tree, move to a private branch; add `scripts/loop/` to `.gitattributes export-ignore`; scrub `/inspire/` in `probe_triton_env.py` + `full_regression.py`.
2. **LICENSE non-compliant + NOTICE incomplete.** LICENSE is a 19-line stub (URL pointer, not the Apache-2.0 body — §4(a) requires a *copy*). NOTICE omits a confirmed BSD-3 vendored source (`L3_08 jagged_dense_flash_attention`, FBGEMM/Meta). Redistributing 99 vendored kernels needs full Apache-2.0 text + the BSD-3 attribution. → full LICENSE text + real copyright; add BSD-3 source to NOTICE.
3. **pyproject `<TBD>` URLs + placeholder authors; README `pip install` fails.** `[project.urls]` has 3 literal `<TBD>`, authors = "TritonAscendBench Authors". README says `pip install triton-ascend-bench` (not on PyPI). → fill real org/repo + authors; change install to `pip install git+<repo>`.
4. **README headline facts stale + banned process-noise.** Task table claims L1=40/L2=38/L3=23 (disk is **36/35/30**); ships "19 codex review rounds" / "v4.13 FROZEN"; Selection link points to stale `benchmark_select.md` (40/38/23) not `kernel-select.md` (36/35/30). → fix counts, delete process-noise, repoint link.
5. **`_canary` anti-cheat trap is git-tracked.** All 5 files of `tasks/_canary/C1_identity_trap/` (identity oracle = the "answer", meta exposing `source_task_id=L1_01`, `_trap_intent.md` which itself says "MAINTAINER ONLY / NOT shipped"). On a submitter-visible repo the trap is defeated. → `git rm --cached -r tasks/_canary/` + gitignore; keep a maintainer-only copy.

## `_xx` directory decisions

| dir | decision | reason |
|---|---|---|
| `_frozen_shim/` | **KEEP** | eval-runtime load-bearing: 57 shipped kernels import inert symbols via sys.path injection (S2 severance); dropping breaks C2+S2 |
| `_frozen_deps_pool/` | **SLIM** | keep 7 hash dirs referenced by 23 kernels; delete orphan `h_a9f952fcc731/` (only test string-fixture refs, 8 on disk vs 7 used) |
| `_vendored/` | **KEEP** | attribution backbone for NOTICE/LICENSE Apache+BSD claims + freeze reproducibility (2.5M, 0 eval role but needed for compliant redistribution) |
| `_corrected/` | **KEEP** | non-scored transparency proof (116K, score_eligible:false) that the upstream off-by-one fix bit-exact PASSes |
| `_deprecated/` | **KEEP** | only 13 small bug-record files (~80K) ship; the 21M r10_pyref safetensors are already gitignored + daemon-regenerated |
| `_canary/` | **DROP** (from repo) | anti-cheat trap leak (see blocker 5); git rm --cached + gitignore, keep maintainer copy |
| `__pycache__/` | already clean | gitignored + untracked, 0 tracked .pyc anywhere |

## Should-fix (important, not blocking)
- Two contradictory selection docs ship (`kernel-select.md` 36/35/30 vs `benchmark_select.md` 40/38/23 + 9 process-noise hits) → make kernel-select.md canonical, fold/remove the other.
- `benchmark_impl_plan.md` 122KB internal plan with ~306 process-noise hits, README links it as canonical → slim or demote.
- `.gitignore` lacks blanket `results/` + the internal docs dirs → add them, verify 0 internal docs stage-eligible.
- Shipped dev scripts leak internal vocab (`probe_triton_env.py` 2× `/inspire/`, `full_regression.py` default `../loop_mailbox`).

## Nice-to-have
- `CONTRIBUTING.md` (submission/manifest format), `CITATION.cff` + BibTeX (defer until real authorship), README "reproduce on one GPU" block, English abstract atop Chinese-only canonical docs, delete untracked `docs/*-V{0,1,2,3}.md` drafts, `rm -rf tasks/_deprecated/*/r10_pyref` local hygiene.

## Shortest path to ship
1. Scrub internal infra/docs (git rm --cached scripts/loop + 5 internal docs dirs + DESIGN-opt/presentation-guide; export-ignore scripts/loop; scrub /inspire/).
2. Seal anti-cheat (git rm --cached tasks/_canary + gitignore); delete orphan `_frozen_deps_pool/h_a9f952fcc731`.
3. Legal: full Apache-2.0 LICENSE + real copyright; add BSD-3 source to NOTICE.
4. Metadata: fill pyproject `<TBD>` + real authors; README install → `pip install git+<repo>`.
5. README correctness: counts → 36/35/30, delete process-noise, repoint Selection link, fold benchmark_select.md.
6. Tighten `.gitignore` (results/, internal docs dirs).
7. Verify: `git archive HEAD | tar t | grep -cE 'scripts/loop/|inspire|docs/reviews'` == 0; `git grep -nIE '(codex|round[- ]?[0-9]|v4\.13|/inspire/|127\.0\.0\.1:2222)' -- '*.py' '*.md'` clean; `tab run-suite --backend gpu` still resolves 101.
