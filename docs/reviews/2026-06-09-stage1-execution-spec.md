# Stage 1 — execution spec (release-quality cleanup, in-place)

Status: SPEC (no code changed). From an 8-item spec workflow (cross-verified). Decisions:
release vehicle = **source tarball** (git archive ships scripts/+infra/, internal export-ignored;
engine loads config from repo tree — no importlib.resources repackaging); automation/ = keep
fenced; peak_table = stays in infra/. **No daemon-path move → NO full GPU regression** (verify =
unit + lint + import/packaging smoke + tarball-boundary check).

## Ordered steps (apply in this sequence)
1. **P0c decouple** (everything builds on the new import paths; CI-guard is red until this lands)
2. **P2 cruft delete** (shrink surface before scrub/fence)
3. **P1b Chinese** (peak_table TOMLs)
4. **P1c paths** (de-personalize in place)
5. **P1a process-noise scrub** (13 public scripts)
6. **P0 cli.py graceful-degrade**
7. **fence-internal** (.gitattributes + .dockerignore + scripts/README — owns the loop/ fence, added ONCE)
8. **CI-guard** (extend `_BENCH_CORE_PATHS` + tarball-import grep — green only after step 1)

Total: **39 paths** — 3 NEW (`ascendbench/_util/{__init__,atomic}.py`, `scripts/README.md`),
30 MODIFIED, 6 DELETED. r10_pyref/, tasks/, gpu_pipeline/ untouched.

## Per-item changes

### P0c — move shared util out of fenced automation (4 real import sites, cross-verified)
- NEW `ascendbench/_util/__init__.py` (docstring) + `ascendbench/_util/atomic.py` = byte-for-byte
  `git mv` of `scripts/automation/lib/atomic.py` (321 lines, stdlib + PyYAML, zero intra-lib deps).
- `scripts/automation/lib/atomic.py` → compat re-export shim (stays fenced; keeps ~9 internal importers unchanged).
  **Use EXPLICIT named re-export of all 6 helpers** — `atomic.py.__all__` (:314) OMITS `atomic_write_bytes`, so a
  `from ... import *` shim would silently drop it (codex). List them: atomic_write_{bytes,json,text,yaml}, snapshot_task_baseline, rollback_task.
- Rewrite 4 public sites → `from ascendbench._util.atomic import ...`:
  `apply_oracle_equivalence_verdict.py:36`, `auto_patch_shapes_from_wrapper.py:42`,
  `golden_manifest.py:49-52`, `tests/unit_tests/test_golden_manifest.py:37` (shipped test).
- **+ test_r19_boundary.py:22** (spec-found extra leak): imports `scripts.automation.bench_evaluator_daemon`
  — this shipped test must EITHER be export-ignored (fence step 7) OR its import guarded. **Decision: fence it**
  (it's an internal-daemon test, belongs out of the public tarball). Add to step-7 fence list.

### P2 — delete dead cruft (git rm, 0 live refs each, verified)
- `scripts/renumber/{renumber_changed,renumber_map}.json`, `scripts/_apply_dtype_specs.py`,
  `scripts/automation/spike/{__init__,gpfs_posix_spike,llm_format_spike}.py`.

### P1b — Chinese → English (8 shipped .py, NOT 4 — codex systematic git-archive scan; 3 handling classes)
shipped `.py` with Han chars (`git archive --worktree-attributes`): eval.py, peak_table.py,
generate_task_scaffolds.py, phase_a_freeze.py, test_{peak_table, phase_c_generators_wa5, schema_doc, scoring_w1_4}.py.
**Handle by CLASS — do NOT blanket-translate:**
- **(a) TRANSLATE** comments/docstrings/logs only: `eval.py:1466`, `peak_table.py`,
  `test_peak_table.py:108` (细化), `test_scoring_w1_4.py:6` (约束), `phase_a_freeze.py:17 + :183` (doc/comment).
- **(b) PRESERVE — translating breaks semantics/data/tests** (codex R2+R3):
  - `phase_a_freeze.py:62` `AMBIGUOUS_PATH_MARKERS=("通用","同上")` — code MARKER matched at :132.
  - `test_schema_doc.py:206` `re.findall(r"[一-龥]")` — the ENGLISH-ONLY ENFORCER regex (MUST contain CJK).
  - `generate_task_scaffolds.py` — **ALL CJK is data-matching normalization, NOT just :29-40** (codex R4):
    `:29-40` DIFFICULTY_MAP + repo dict keys (`"低/中/高"→low/medium/high`, `"字节中台编译"→Bytedance_internal`),
    `:159` cv_vv value (`未标`), `:186` `("中高","中低","高","中","低")` difficulty tuple matched against `difficulty_raw`.
    Translating ANY (esp. :186) silently breaks parsing (→ default `"medium"`). Preserve/Unicode-escape the WHOLE file's
    CJK; do NOT translate. Implementer MUST re-grep the full file and confirm each CJK is data, not prose.
  - `test_phase_c_generators_wa5.py:334` — `difficulty_raw="中"` is the **fixture exercising that CJK-input path**.
  Do NOT translate any (b) item. If CJK-free source bytes are required, use Unicode escapes (`"低"`), not English.
- **(c) note**: phase_a_freeze's 通用/同上 marker now matches **0 meta.yaml paths** (data layer already CJK-free) —
  a dormant detector; keep as-is (optionally Unicode-escape `"通用"` for a CJK-free source byte).
- **peak_table TOMLs** (38+9 comment lines) as before — translate comments, keep keys/values byte-stable.
- **OPEN — decide scope**: 13 `tasks/*/contract.md` contain Chinese (CLAUDE.md permits Chinese in `*.md`, but a
  public bench wants English task contracts) — sizeable separate effort; include in Stage 1 or defer.

### P1c — de-personalize PERSONAL paths (PRECISE pattern + method + verify; codex R5/R6 refined)
**The leak is the personal gpfs path / username / reverse-tunnel SETUP — NOT generic localhost** (verified):
- **CLEAN**: `/inspire/...` + `liuliming` + `qianghuaxuexi` (personal gpfs path + username); `root@127.0.0.1`
  + `ssh -p 2222` (the personal reverse-tunnel setup).
- **EXEMPT — do NOT touch (generic localhost)**: bare `127.0.0.1` as an ssh-tunnel host param
  (`ssh_tunnel.py:34,132` — parameterized) or a test-server localhost (`test_ssh_tunnel.py:51`, `test_runners.py:544-555`).
  My earlier draft wrongly treated `127.0.0.1` as a leak; it's legitimate localhost.
**Files (8 current, machine-enumerated — but TRUST the method+verify, not this list)**: scripts/loop/{orchestrator,
daemon_a, README, _wf_authoritative_route.js, _wf_diagnose_remaining.js}, scripts/probe_triton_env.py,
docs/plans/2026-05-23-001-*, docs/release/internal-acceptance-quickstart.md.
**METHOD — do NOT hand-list lines** (they keep getting missed): `git grep -nE '/inspire/|liuliming|qianghuaxuexi|root@127\.0\.0\.1|\b2222\b'`
whole-repo; replace each personal path/username/tunnel-setup with an env-var (`TAB_REMOTE_BENCH`, `LOOP_SSH_HOST`,
`LOOP_SSH_PORT`, `LOOP_MAILBOX`; orchestrator `REMOTE_BENCH`/`REMOTE_LOOP` DEFAULTS → `os.environ.get(..., <neutral>)`).
loop/+probe+internal-docs are ALSO fenced (step 7), but de-personalize content regardless (GitHub mirror shows source).
**VERIFY**: `git grep -nE '/inspire/|liuliming|qianghuaxuexi|root@127\.0\.0\.1|\b2222\b' -- ':!docs/reviews'` = **0** WHOLE repo (generic `127.0.0.1` stays).

### P1a — process-noise scrub (13 public scripts; 8 others already clean)
- Heavy: `resolve_upstream_entrypoints.py` (84) + `phase_a_prime.py` (71) = ~67% of the work.
- Others: oracle_equivalence_gpu_runner (8), oracle_pyref_precompute (8), apply_oracle_equivalence_verdict (5),
  verify_upstream_commit_chain (8), source_provenance_census (12), provenance_release_planning_gate (4),
  golden_manifest (3), validate_tasks (9), _pa_oracle_check_runner (8), _pa_subproc_runner (9), phase_b_oracle (1).
- **Drop**: `codex` / `V9` / `V4` / `M<milestone>` / `FIX <n>` / `MUST-FIX <n>` / `BLOCKER <n>` / `codex finding` / `DESIGN-V2`.
- **GUARDRAIL — KEEP byte-identical** (NOT process-noise):
  - the `r10_pyref` / `r10_verdict` / `R10.a/b` / `R10 stage-N` DATA PATH;
  - R-**requirement IDs** cited from shipped specs (`R5`, `R12`, `R17`, `R18`, `R13c`, `R1-4`, `R9-3`, `R4-3`, `R7-2`, `R14-P*`...);
  - **non-R DATA VALUES that merely contain a milestone token**, e.g. `V9_DEPRECATED_M1_1`
    (`source_provenance_census.py:189` field default, **asserted by `test_source_provenance_census.py:145`**) —
    here `V9`/`M1_1` are part of an ENUM VALUE, not a label. Deleting it breaks the test + the data contract.
  - i.e. do NOT blanket-delete `R\d+` or `V9`/`M*` as a regex; scrub only the prose milestone/round LABELS.
    Method: enumerate candidate lines, then for each decide label-vs-identifier/value; re-grep 0 prose labels after,
    and run `pytest test_source_provenance_census.py test_validate_*` to confirm no asserted value was touched.

### P0 — cli.py graceful-degrade (source-tarball vehicle)
- `cli.py:198-246` — the 5 `from scripts.X import main` lazy-imports wrap in try/except → print a clear
  "source-checkout-only subcommand" message instead of a raw ModuleNotFoundError on a pip wheel.
- `peak_table.py` + `static_checker.py`: **confirm-only, NO change** (REPO_ROOT/infra loads fine from repo tree, ships in tarball).

### fence-internal (step 7, sole owner of the export-ignore set)
- `.gitattributes` export-ignore + `.dockerignore` mirror for: `scripts/loop/`, `scripts/full_regression.py`,
  the internal helpers (auto_shapes_filler, auto_patch_shapes_from_wrapper, lint_shapes_against_kernel,
  per_task_diagnostic, _c2_sweep_analyze, probe_triton_env, build_oracle_digest, multi_launched_manifest,
  operator_slice, freeze_source, extract_kernel), `infra/{RFC_0001_scope,upstream_contacts}.md`,
  `_test_ceiling_acceptance.py`, **+ test_r19_boundary.py + test_resolve_loader.py** (two shipped-test→automation
  leaks codex found — both import a fenced internal module). NEW `scripts/README.md` public/internal note.

### CI-guard (step 8)
- `tests/unit_tests/test_release_boundary.py` — extend `_BENCH_CORE_PATHS` to all public scripts
  (phase_c_docs, C2 trio, audit_cost_model_formulas, golden_manifest, validate_corpus_consistency,
  provenance_release_planning_gate, verify_upstream_commit_chain, generate_task_scaffolds, source_provenance_census, _pa_*).
- `.github/workflows/ci.yml` — add a tarball-extract + grep step rejecting any shipped `.py` that imports
  `scripts.automation|from automation|import automation`.

## Open risks (from the spec) + resolution
- **ci.yml edited by 3 items / 3 regions** → edit ci.yml LAST among its touchers, re-anchor by content not line#.
- **scripts/loop/ fence requested twice** → fence-internal (step 7) is sole owner; P0 only does cli.py.
- **cli.py local `import sys as _sys` / `Path as _P`** in the 5 funcs → the try/except wrap must match that exact form.
- **CI-guard red until P0c lands** → ordering enforces it (step 8 last).
- **test_r19_boundary leak** → resolved by fencing it (step-7 list above).
- **mechanical fatigue on resolve + phase_a_prime** → scrub them with an explicit grep-enumerated line list, re-grep 0 noise after.

## Verification gate (no GPU regression needed)
1. `ruff check ascendbench/ scripts/ tests/` + format check on edited files — clean.
2. `python3 -c "import ascendbench._util.atomic"` + import-smoke the 4 P0c public scripts with automation fenced.
3. shim regression: `pytest tests/unit_tests/test_lib_atomic.py test_golden_manifest.py -q`.
4. full unit suite green (2277, 0 skipped).
5. tarball boundary: `git archive HEAD | tar t` shows NO internal path; extract + grep finds NO shipped `.py`
   importing `scripts.automation`/`from automation`. **P1c (WHOLE repo, not just tarball — GitHub mirror;
   generic `127.0.0.1` EXEMPT):** `git grep -nE '/inspire/|liuliming|qianghuaxuexi|root@127\.0\.0\.1|\b2222\b' -- ':!docs/reviews'` = **0**.
   NO Han chars in the (a)-TRANSLATED set (peak_table TOMLs + the comment-class `.py`);
   the (b)-PRESERVED literals (`phase_a_freeze` marker, `test_schema_doc` regex, `generate_task_scaffolds`
   whole-file normalization CJK incl :186, `test_phase_c_generators_wa5:334` fixture) are DOCUMENTED EXEMPTIONS
   (Unicode-escape if CJK-free bytes required); the 13 `contract.md` are the OPEN scope item — NOT gated here.
6. `tab census --help` etc. degrade gracefully (no raw ImportError) when run from a pip-installed wheel.
