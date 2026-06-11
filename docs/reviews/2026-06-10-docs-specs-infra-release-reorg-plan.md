# docs/ specs/ infra/ — Public-Release Reorganization Plan

Goal: bring `docs/`, `specs/`, `infra/` to public-release quality — one clean
public docs tree an external user can navigate, with all internal dev-process
artifacts, generated dumps, sensitive paths, and process-noise removed from the
public surface. Scope is these three directories only; the broader release
blockers (LICENSE body, pyproject metadata, README facts, `_canary` seal) are
tracked in `2026-06-08-release-readiness-audit.md` and listed here as adjacent
out-of-scope items.

Scope: `docs/`, `specs/`, `infra/` **plus `README.md`** (its links into these
dirs break on the reorg, so README link/fact fixes ship in the same PR — see
§6a). The remaining release blockers (LICENSE body, NOTICE, pyproject metadata,
`_canary` seal) stay in `2026-06-08-release-readiness-audit.md` as an adjacent
follow-up (§8).

This plan **aligns with the existing release boundary** (`.gitattributes
export-ignore` + `.dockerignore` + CI grep) and the maintainer audit, not a new
mechanism.

---

## 1. The core constraint (drives every decision)

`export-ignore` strips a path from the `git archive` tarball **only**. It does
**not** hide the path from a public GitHub repo — anyone can browse it. So the
boundary today (which export-ignores `scripts/loop/`, `docs/triage_backlog/`,
`docs/release/internal-*`, …) is sufficient for the *tarball/wheel* release but
**not** for a *public GitHub* release.

Therefore the public-release target requires internal docs to actually **leave
the tracked tree of the public branch** (`git rm --cached`), not merely be
export-ignored.

Two things leak on a public GitHub repo, and both must be addressed:
- the **working tree** of the public branch (fixed by `git rm --cached`), and
- the **git history** — every old commit is browsable on GitHub, so internal
  docs sitting in history are still a leak even after `git rm`. A true public
  release therefore needs the history purged too (`git filter-repo`) **or** a
  fresh public repo seeded from a scrubbed snapshot with no internal history.

**DECISION-1 (needs your call): public-release strategy + where internal docs live.**

| Option | How | Public tree clean | Public *history* clean | Team keeps internal docs |
|---|---|---|---|---|
| **A (recommended)** | fresh **public repo** seeded from a scrubbed snapshot (squash or `git filter-repo`); current repo stays the **private** dev repo | ✅ | ✅ | ✅ — they live in the private repo |
| B | `git filter-repo` the existing repo to purge internal paths from all history, then it becomes the public repo; internal tree preserved on a **private remote** first | ✅ | ✅ | ✅ — private remote/mirror |
| C | `git rm --cached` internal tree on main + push it to an `internal` branch on the **same** remote | ✅ | ❌ history still leaks | ✅ but branch is public ⇒ still leaked |
| D | export-ignore only (status quo) | ❌ browsable | ❌ | n/a |

C and D are rejected for a public GitHub repo (C's `internal` branch and the
history are both browsable; D leaves the working tree browsable). **Recommend A as
a single SQUASHED snapshot**: the existing `sii-ai-infra/TritonAscendBench` remains
the **private** dev repo (internal docs keep living here), and we publish a separate
**public** repo seeded from one scrubbed, internal-free commit with **no inherited
history**. This (i) avoids history-rewriting the live dev repo, (ii) gives the
cleanest public face, and (iii) **eliminates the history-content leak by
construction** (no old commits to carry stale `/inspire/`/`v4.13`/`kernelpilot`
tokens — see §7 gate-3). B (in-place `filter-repo`, history preserved) is the
alternative if a single repo must flip private→public in place, but then the §7
gate-3 per-commit content grep is mandatory. The `git rm --cached`/structure work
in this plan is identical under A or B; the only difference is *which repo becomes
public and how history is handled* — the call needed before P0.

---

## 2. Current-state diagnosis (from the 6-area survey + audit)

121 tracked files across the three dirs (`git ls-files docs specs infra`;
subcounts: docs/reviews=50, docs/plans=18, docs/triage_backlog=19). Four mess
categories:

1. **Versioned duplicates** — DESIGN-V1/V2/V3, ARCHITECTURE-DATAFLOW-V0/V1 are
   already **untracked** (`.gitignore` handles `docs/DESIGN-V*.md` +
   `docs/ARCHITECTURE-DATAFLOW-V*.md`). Not a public problem; local clutter only.
   `DESIGN-opt.md` + `DESIGN-presentation-guide.md` ARE tracked (not V-numbered).
2. **Internal-process dirs (tracked, public on GitHub)** — `docs/plans/` (18),
   `docs/reviews/` (50), `docs/triage_backlog/` (19), `docs/brainstorms/` (1),
   `docs/architecture/` (1), `docs/release/` (4). This is audit blocker #1.
3. **Generated dumps (tracked)** — `docs/plans/phase_a_prime_classification.md`
   (7.8M, machine-generated), `phase_a_prime_status.md` (217K, stale no-GPU
   state), ~16 `docs/reviews/*-full-regression.md` CI logs, several large
   triage dumps (`per-task-diagnostic` 108K, `truth-inventory`, …). All
   regenerable; zero public value.
4. **Process-noise + sensitive content** in otherwise-keepable files (next section).

---

## 3. Sensitive content to scrub (hard blockers — confirmed by survey)

| File | Issue | Fix |
|---|---|---|
| `docs/benchmark_select.md` | absolute `/data1/code/agentic-rl/...` paths | drop file (stale, superseded by kernel-select.md) |
| `docs/reviews/*` (4 files), `docs/triage_backlog/2026-05-25-bench-shim-layer-analysis.md` | `/inspire/` cluster paths | leave-with-internal (removed from public) |
| `docs/release/internal-acceptance-quickstart.md` | real SSH target `exp2@ai-infra` | genericize to `<user@host>` (kept internal regardless) |
| `docs/release/internal-release-process.md` | split-token internal name `kernel''pilot`/`Kernel''Pilot`/`kp''_` | kept internal; do not publish |
| `specs/canary_task_design.md` | reveals canary trap **detection rules** (aids evasion) | move-to-internal |
| `specs/submission_throttling.md` | names `@huawei/@bytedance/@stanford` as examples | genericize before publish (or move-to-internal) |
| `specs/task_api_contract.md` §6a (L558-738) | bench-team-internal automation state machine | split §6a → internal, publish §1-6+§7 |
| all 11 `specs/*.md` headers | `> v0 — per v4.13 §5b.X + R11/R12` process-noise tags | strip header + inline `R##-P#` tags |
| `infra/Dockerfile.npu`, `peak_table_910b.toml`, `version_matrix.toml` | comments cite internal `v4.13 §`, `DESIGN-V2 §`, `plan U6` | repoint to public DESIGN.md / strip |
| `docs/automated_triton_operator_extraction.md` | `codex` ×13, daemon `loop_mailbox/results` paths, interim `53/100` | move-to-internal (or strip-then-publish) |

**Two-track handling (important):** the table's "scrub" only applies to files
that stay **published**. Sensitive content inside files that move to the internal
set (every `docs/triage_backlog/*`, `docs/plans/*` status dump, `docs/release/
internal-*`, `benchmark_select.md`, …) is handled by their **removal** from the
public tree — they are not individually scrubbed, they just leave. So the section
is not an exhaustive scrub list of every `/inspire//data1/` hit (codex confirmed
more exist in the internal targets); it is the scrub list for the *surviving
public* files only.

Verification runs against the **final public tree/snapshot**, not the current
working dir: `git grep -nIE '(/inspire/|/data1/|127\.0\.0\.1:2222|kernel.?pilot|exp2@|<personal-host-tokens>|v4\.13|codex|round[- ]?[0-9])'` over the published
`docs/ specs-folded infra/` set == 0 (third-party `external/`, `tasks/_vendored/`
excluded). Personal-host tokens per the audit's list.

---

## 4. Target public structure (what GitHub shows)

Directory roles (each answers ONE question for an external user):
- **`docs/`** — *what is this benchmark and why* (design, architecture, selection).
- **`docs/reference/`** — *the exact contracts a submitter codes against*
  (meta/shapes schema, scoring math, tolerance classes, timing protocol).
- **`docs/findings/`** — *proof the bench surfaces real things* (upstream bugs
  found during corpus construction, each with a runnable proof). Growth point —
  one entry today, gets a README defining the bar for adding more.
- **`docker/`** (renamed from `infra/`, DECISION-8) — runner images. With only
  the two Dockerfiles left public, `infra` is a misnomer; `docker/` is the
  convention. Bonus: after the move, the dev repo's `infra/` holds ONLY internal
  files (peak_table_h200, RFC_0001_scope, upstream_contacts) ⇒ the whole dir
  becomes internal (one `infra/` export-ignore line replaces per-file rules).
  **Correction (codex round-8, [confirmed] by wheel build)**: the peak tables +
  version_matrix CANNOT stay only in `infra/` — they are **runtime-consumed**:
  `ascendbench/perf/peak_table.py:28` hardcodes `REPO_ROOT/"infra"/peak_table_910b.toml`
  and `ascendbench/integrity/static_checker.py:29` reads `infra/version_matrix.toml`
  for the import ALLOWLIST (silently falling back to built-in defaults when
  missing — a security-relevant silent degrade). The wheel excludes `infra/*`
  (`pyproject.toml:157-162`), so `pip install git+` + `tab run-suite` breaks /
  degrades. Fix (P1.6): move the three runtime tomls INTO package data
  (`ascendbench/policies/` alongside tolerance_policy.yaml), loaders read via
  `importlib.resources` with an explicit `--peak-table`/path override; `infra/`
  keeps the Dockerfiles (pure environment layer).

```
README.md            (links/counts fixed — §6a)
docs/
  DESIGN.md                 ** REWRITTEN ** — English, code-aligned (see below)
  ARCHITECTURE.md           ** REWRITTEN ** — English, code-aligned (see below)
  kernel-select.md          canonical task selection (current, stays)
  reference/                submitter-facing contracts (was specs/, TRIAGED):
    task_api_contract.md      §1-6 + §7 (the real shipped contract; §6a → internal)
    scoring_spec.md           real shipped scoring math (C2+S2 gate era)
    tolerance_policy.md       mirrors ascendbench/policies/tolerance_policy.yaml
    timing_arbitration.md     shipped timing protocol
  findings/
    README.md                 (defines what qualifies as a finding)
    vllm_lightning_attn_crossblock_decay_offbyone.md
    L2_24_lightning_attn_offbyone_proof.py
docker/                                    (renamed from infra/ — DECISION-8)
  Dockerfile.gpu  Dockerfile.npu
ascendbench/policies/                      (runtime-consumed data, ships in wheel)
  tolerance_policy.yaml  ceiling_acceptance.yaml
  peak_table_910b.toml  version_matrix.toml   ← moved (P1.6)
```
(`infra/` no longer exists publicly: Dockerfiles → `docker/`, runtime tomls →
package data, the rest is internal. `peak_table_h200.toml` does NOT ship —
provider-side GPU self-check artifact, stays in the dev repo's `infra/`;
DECISION-7.)

**REWRITE, not launder (user decision)**: the current `DESIGN.md` /
`ARCHITECTURE-DATAFLOW.md` are Chinese, carry internal codenames (S2/S3/S6),
grew through V1→V3 iterations, and describe *design intent* that has since
drifted from shipped code (e.g. verified-runner posture, cost_model's role).
Rather than fold/translate/scrub them, write **two fresh English documents from
the shipped code state**:
- `docs/DESIGN.md` — what is measured and why: migration-benchmark rationale,
  task tiers (36/35/30), correctness model (C2 oracle-equivalence + S2 bit-exact
  severance), scoring (Roofline-normalized perf + coverage gate; cost_model =
  evidence-only, NOT in the release gate), anti-gaming posture (high-level only;
  no canary detection rules).
- `docs/ARCHITECTURE.md` — how it works: corpus construction pipeline (freeze →
  sever → oracle → golden), eval flow (resolve → run → grade), repo layout map
  (tasks/ anatomy: kernel/meta/shapes/contract/oracle), what runs where
  (submitter GPU vs provider verification).
The old documents move to the internal set as source material. The two
distill-worthy internal reviews (`bench-faithfulness-architecture`,
`device-timing-methodology`) feed the rewrite. Acceptance bar: every claim in
the rewritten docs must be checkable against shipped code/corpus — no
described-but-unbuilt mechanisms except in a clearly marked "Roadmap" section.

**reference/ triage — shipped contracts vs design-stage specs (user decision)**:
publishing a detailed spec of an unbuilt security architecture over-promises and
aids attackers planning against it. Split:
- **Publish (4)** — describe SHIPPED behavior a submitter relies on:
  `task_api_contract` (§1-6+7), `scoring_spec`, `tolerance_policy`,
  `timing_arbitration`.
- **Internal (design-stage / sensitive, 7)**: `process_isolation`,
  `artifact_provenance`, `ast_denylist` (verified-runner security stack — parts
  unbuilt; publish a 1-paragraph summary inside DESIGN.md's anti-gaming section
  instead), `reward_hack_defense` (internal cross-refs + canary specifics),
  `submission_throttling` (unbuilt leaderboard ops), `canary_task_design`
  (detection rules ⇒ evasion aid), `extraction_failure_taxonomy` (corpus dev
  process).

`specs/` as a top-level dir disappears (folded into `docs/reference/`). The two
upstream-finding files move from `docs/upstream_reports/` + `docs/verification/`
into `docs/findings/`.

---

## 5. Per-area disposition

### docs/ root .md
| File | Action |
|---|---|
| DESIGN.md (current zh) | **internal as source material**; replaced by a fresh English code-aligned `docs/DESIGN.md` (§4 rewrite) |
| ARCHITECTURE-DATAFLOW.md (current zh) | **internal as source material**; replaced by fresh `docs/ARCHITECTURE.md` (§4 rewrite; mine V0's online-eval/runtime-sequence content) |
| kernel-select.md | **publish** (canonical; current + clean) |
| DESIGN-opt.md | **internal** (its plain-language framing feeds the rewrite, then retires) |
| DESIGN-presentation-guide.md | **internal** (outreach aid for superseded V2) |
| benchmark_select.md | **delete** (stale + `/data1/` paths; superseded) |
| benchmark_impl_plan.md | **internal** (122K codex-history dump; README must stop linking it) |
| automated_triton_operator_extraction.md | **internal** (process noise) — or strip-then-publish later |
| DESIGN-V1/V2/V3, ARCH-V0/V1 (untracked) | local `rm` (already gitignored) |

### docs/plans/ (18) → **all internal**
- **delete** `phase_a_prime_classification.md` (7.8M), `phase_a_prime_status.md` (217K) — generated dumps.
- **internal** the 16 human plans. Collapse dtype cluster: delete `dtype-alignment-plan.md`, merge `dtype-alignment-OVERALL.md` → `consolidated-issues-and-fix-plan.md`.

### docs/reviews/ (50) → **all internal**
- **delete** the ~16 `*-full-regression.md` CI dumps + generated result dumps (full-oracle-equiv-gpu-run, full-shape-c2, pipeline-status, build-progress, final-ranking, precision-diff). Regenerable.
- collapse renumber trio → keep `level-renumber-migration-plan` + `renumber-map`; package trio → keep `package-reorg-final`; drop design-v1/v2 review series (tied to deleted drafts).
- **keep internal**: `release-readiness-audit` (the playbook), the two distill-worthy ones (`bench-faithfulness-architecture`, `device-timing-methodology`) — flagged as future public-methodology-page source.

### docs/triage_backlog/ (19) → **all internal** (already export-ignored)
- **delete** generated dumps (per-task-diagnostic 108K, truth-inventory, 100-task-real-attribution, r13c-gate.json, the 4 tool-output reports, v1-deprecated/v2 files).
- internal the remaining design/roadmap notes.

### docs/brainstorms/ (1) → **internal**

### docs/architecture/ (1) → **internal**
- `automation-llm-repair-architecture.md` specs an export-ignored subsystem; its self-declared "public" status is a CONFLICT to resolve (DECISION-2). Recommend internal (documents code not in the release).

### docs/release/ (4) → **internal** (already export-ignored)
- **delete** `internal-m1-acceptance-report.md` (unfilled placeholder).
- keep `internal-acceptance-runbook` + `internal-acceptance-quickstart` (scrub SSH host) + `internal-release-process` internal.

### docs/upstream_reports/ + docs/verification/ → **publish** as `docs/findings/`
- the vLLM off-by-one report + its standalone proof script — strongest public showcase, no sensitivity issues.

### specs/ (11) → fold into `docs/reference/` (TRIAGED — §4)
- **publish (after header strip), shipped contracts only (4)**: task_api_contract
  (§1-6+7), scoring_spec, tolerance_policy, timing_arbitration.
- **internal — design-stage verified-runner security stack (3)**:
  process_isolation, artifact_provenance, ast_denylist. Parts describe unbuilt
  Docker/broker/audit-hook infra; publishing a detailed unbuilt security spec
  over-promises and aids evasion planning. The shipped subset (two-process
  isolation that exists today) gets a 1-paragraph summary in the rewritten
  DESIGN.md anti-gaming section. Note: `ascendbench/integrity/process_isolation.py`
  docstrings reference this spec — genericize those pointers (P1.5 already
  covers them).
- **publish task_api_contract §1-6 + §7**, but §6a split-out has a follow-through.
  Actual field presence in public `tasks/*/meta.yaml` (verified): `phase_a_prime`
  **101/101** (universal), `phase_b_capture` **18/101** (optional),
  `phase_b_capture_required` **0/101**, `release_scope` **0/101** (absent from
  public metas). So the public contract documents `phase_a_prime` (always-present)
  + `phase_b_capture` (optional) as **read-only provenance** fields; the
  `release_scope` / automation-write **state-machine narrative** (which references
  fields not even present in public metas) moves fully internal. (Stripping these
  provenance fields from task metas is a separate, larger `tasks/`-touching change
  — out of scope; see the Sub-flag.) `tests/unit_tests/test_schema_doc.py` (NOT
  export-ignored ⇒ runs in the public suite) hardcodes `specs/task_api_contract.md`
  + asserts §6a headers → must be updated for both the path move and the split.
- **reward_hack_defense.md → rewrite-then-publish OR internal**: not just a header
  strip — it cross-links the internal `benchmark_impl_plan.md` v4.13 §5b (which
  stays internal ⇒ dangling refs), has a Chinese fragment, and embeds concrete
  canary examples. Inline/resolve the §5b refs + translate + drop the canary
  specifics, else hold internal.
- **submission_throttling.md → default internal**: even after genericizing the
  org examples it publishes anti-sybil thresholds + hidden-result-log policy for a
  verified-runner leaderboard that **does not exist yet**. Hold internal until that
  ops infra is built (resolves DECISION-3).
- **internal**: canary_task_design (reveals trap detection ⇒ evasion risk),
  extraction_failure_taxonomy (corpus-construction dev process), task_api_contract
  §6a narrative, submission_throttling, reward_hack_defense (unless rewritten).

### infra/ (7) → dissolves: `docker/` + package data + internal (DECISION-7/8)
Docs explain, scripts operate, the package computes, **docker/ stands up the
environment**. The `infra/` grab-bag dissolves three ways:
- **publish as `docker/`** (DECISION-8, `git mv`): Dockerfile.gpu, Dockerfile.npu
  (after ref scrub). Rename consumers (verified): `.github/workflows/ci.yml:188` +
  `verified_runner.yml:39` (`docker build -f infra/Dockerfile.*` → `docker/`),
  Dockerfile header comments citing `infra/version_matrix.toml` → point to
  `ascendbench/policies/` (P1.6).
- **move into package data (P1.6)**: peak_table_910b.toml, version_matrix.toml
  (+ ref scrub of their internal doc-section comments). Public spec consumers of
  the old location repoint too: `scoring_spec.md:20,80` ("`infra/` peak table" →
  package data path).
- **internal — NOT shipped (DECISION-7)**: peak_table_h200.toml — provider-side
  GPU self-check reference; no default code path loads it; stays at the dev
  repo's `infra/` so `scripts/loop/gpu_rank_one.py`'s default keeps working.
  Public users wanting local GPU roofline numbers pass `--peak-table`.
- **internal**: RFC_0001_scope.md, upstream_contacts.md. After the Dockerfile
  move the dev repo's `infra/` is 100% internal ⇒ replace the two per-file
  `.gitattributes`/`.dockerignore` rules with a single `infra/` dir rule. Before
  archiving `upstream_contacts.md`, lift its repo+license attribution table into
  the root `NOTICE`/CREDITS so public users keep upstream attribution.

---

## 6. Execution phases (after plan approval + codex sign-off)

- **P0 — strategy + internal preservation**: execute DECISION-1 (A: stand up the
  public repo from a scrubbed snapshot, current repo stays private; or B: mirror
  internal tree to a private remote before an in-place `filter-repo`). No public
  push until P4 verify is green.
- **P1 — public docs assembly + REWRITE**:
  - **Rewrite** `docs/DESIGN.md` + `docs/ARCHITECTURE.md` fresh in English from
    the shipped code state (§4 scope; sources: old zh DESIGN/ARCH + DESIGN-opt
    framing + ARCH-V0 diagrams + the 2 distill-worthy internal reviews). Every
    claim checkable against code/corpus; unbuilt mechanisms only in a marked
    Roadmap section. Each rewritten doc gets its own codex review pass for
    code-alignment before it ships (same bar as this plan).
  - Create `docs/reference/` (move + header-strip the 4 shipped-contract specs),
    `docs/findings/` (move the upstream-finding pair + write findings/README.md);
    split task_api_contract §6a narrative out (keep field docs public).
- **P1.6 — packaging/loader fix (codex round-8 [confirmed]; scope per
  DECISION-7)**: move `peak_table_910b.toml` + `version_matrix.toml` →
  `ascendbench/policies/`; update `pyproject.toml` package-data (`*.toml`).
  **Necessity tiers** (verified consumers): `peak_table_910b` = HARD (default
  scoring path `eval.py:687` fails closed without it); `version_matrix` = SOFT
  (runtime reads only its `import_allowlist.modules` section, today with a
  silent built-in fallback — packaging it makes the allowlist explicit);
  `peak_table_h200` = OPTIONAL with **zero** default-code-path consumers →
  per DECISION-7 it is CLEANED from the public release entirely (stays in the
  dev repo's `infra/` for the internal GPU rank pipeline). Public-side
  consequences: `cli.py:53,76` help text mentioning it → rewrite to "pass your
  own GPU peak table"; `test_eval_w1_4.py:210-225` path-threading assertion →
  rewrite against a tmp fixture table (public test must not depend on a
  non-public file); `scripts/loop/gpu_rank_one.py` default → UNCHANGED (file
  keeps existing at dev-repo `infra/`).
  rewrite `ascendbench/perf/peak_table.py:28` + `ascendbench/integrity/
  static_checker.py:29` to load via `importlib.resources` (explicit path
  override retained; static_checker's silent-fallback-when-missing becomes
  unnecessary — fail closed on a malformed table, default only when the
  resource genuinely absent in dev checkouts). Add a unit test that the
  installed package resolves both resources. Plus spec-text cross-refs that
  would dangle (A-item): `task_api_contract.md:57,554` (ast_denylist /
  process_isolation links → genericize), `timing_arbitration.md:68` (same),
  and **align the public static-check summary in task_api_contract with the
  real `static_checker.py` behavior** (it denies more than the doc lists —
  e.g. `safetensors`/`torch.load`, the no-`@`-decorator AST rule) so
  internalizing ast_denylist.md is not submitter-hostile: the contract doc
  must tell submitters exactly what gets rejected.
  **Consumer updates the toml move requires (codex round-9, all [confirmed])**:
  - `scripts/provenance/verify_version_matrix.py` default path → package resource
    (this also fixes its callers `.github/workflows/ci.yml` +
    `verified_runner.yml` + `scripts/gpu_pipeline/40_release_gate.sh`, which
    call it without a path);
  - `scripts/gpu_pipeline/00_setup_env.sh` passes `${REPO_ROOT}/infra/version_matrix.toml`
    literally → update;
  - tests: `test_peak_table.py`, `test_version_matrix_gpu_capture.py` (read old
    `infra/` paths), `test_eval_w1_4.py` (hardcodes `infra/peak_table_h200.toml`
    in a path-threading assertion) → repoint;
  - internal-side: `scripts/loop/gpu_rank_one.py` `--peak-table` default →
    UNCHANGED (h200 stays at dev-repo `infra/` per DECISION-7);
  - `ascendbench/engine/cli.py` stale help text mentioning `infra/peak_table_h200.toml` → rewrite ("pass your own GPU peak table");
  - **new grep gate**: `git grep -nE 'infra/(peak_table_910b|peak_table_h200|version_matrix)\.toml|peak_table_h200'`
    == 0 in the public set after the move (h200 must not be referenced publicly at all).
- **P2 — scrub**: run the sensitive-content scrub (section 3) on the surviving
  public set; scrub infra/ internal cross-refs (`v4.13 §`, `DESIGN-V2 §`,
  `plan U6`); lift `upstream_contacts.md` attribution table → root `NOTICE`.
- **P1.5 — repo-wide reference sweep**. The completeness criterion is a SINGLE
  grep (gate-7) over **every tracked file except the export-ignored internal set
  and third-party** = 0 hits. The sweep keeps widening each review round because
  the refs are everywhere; so the rule is "0 repo-wide", not a hand-list. Verified
  hit categories (all [confirmed]) to repoint/scrub until the grep is clean:
  - **docs (public)**: the old zh `docs/DESIGN.md` specs refs (`:127,268,471,
    497,589,715`) retire with it (it moves internal; the REWRITTEN DESIGN/
    ARCHITECTURE are born clean — gate-7 still verifies them); the moved
    `docs/findings/` pair's self-refs to old `docs/upstream_reports/`+
    `docs/verification/` paths → update.
  - **root**: `SCORE.md:36-37` → `docs/findings/`; `NOTICE:33` links
    `specs/reward_hack_defense.md` (internal) → repoint or drop; `README` :5/:62/
    :92/:112/:125/:129/:130 + the count line **:140** (`L1=40,L2=38,L3=23` →
    36/35/30) — see §6a.
  - **tasks (leaf, from generators)**: `contract.md` 90×`canary` + 2×`ast_denylist`
    + 1×`tolerance` ; `shapes.yaml` 78×`docs/plans/2026-05-21…` comment ;
    `meta.yaml` 41×`docs/reviews/2026-06-05…` (`cost_model.status_reason` strings)
    + 3×`docs/npu/benchmark_select.md` + L2_08 `contract.md` (`provenance.errata_note`).
  - **public scripts (NOT export-ignored ⇒ they SHIP)**: these are the *generators
    + consumers* of the moved paths and need real fixes, not just comment scrubs —
    - `scripts/verify/standalone_freeze_verifier.py:71-73` **WRITES** the deleted
      dumps to `docs/plans/phase_a_prime_{classification,status}.md` → **repoint its
      output out of `docs/`** (e.g. a gitignored `build/`/local artifact dir) so the
      deleted dumps are not regenerated into the public tree.
    - `scripts/corpus/generate_shapes_and_contract.py:32,897` is the SOURCE of the
      78 shapes comments + 90 contract canary refs → fix here, then regenerate.
    - `scripts/corpus/generate_task_scaffolds.py:26` `SELECT_MD = …/docs/npu/
      benchmark_select.md` (a path that does not exist) → repoint to the canonical
      selection source / `docs/kernel-select.md`.
    - **default-output paths that REGENERATE internal dirs via split-path joins**
      (literal grep misses these — codex round-5): `precompute_oracle_reference.py:655`,
      `provenance/census_source_provenance.py:363,370`,
      `provenance/compute_release_planning_gate.py:240` (→ the `r13c-gate.json`),
      `provenance/verify_upstream_commit_chain.py:315`,
      `oracle/apply_oracle_equivalence_verdict.py:324`, and
      `verify/standalone_freeze_verifier.py:71` (`PLANS_DIR = REPO_ROOT/"docs"/"plans"`)
      → repoint every default out of `docs/` to a gitignored artifact dir.
    - `scripts/corpus/validate_tasks.py` ~12 `specs/task_api_contract.md §…` +
      `specs/canary_task_design.md` docstring refs → repoint to `docs/reference/`
      (keep the validator logic; only the doc pointer; §6a-section refs → genericize
      to "the internal task-state contract").
  - **package (ascendbench/, SHIPS in the wheel)**: `__init__.py:19` ("See specs/"),
    `integrity/adversarial.py:5` (canary), `integrity/process_isolation.py:3,22,161,1110`
    (process_isolation/timing specs), `input_spec/expr.py:27,309` +
    `input_spec/inputs.py:58,80,442` (task_api §5.2), `input_spec/param_constraints.py:108,562`
    (docs/plans), `policies/tolerance_policy.yaml:3` (tolerance spec),
    `policies/ceiling_acceptance.yaml:6` (docs/reviews) → repoint to `docs/reference/`
    or genericize.
  - **public tests** (NOT export-ignored): `test_schema_doc.py` (`specs/task_api_contract.md`
    → `docs/reference/` + split) and **`test_pa_fix1_honest_categories_and_fix3_anchor.py:329`
    which READS the deleted `docs/plans/phase_a_prime_status.md`** (split-path) → it
    breaks the public suite when the dump is deleted; repoint to the new artifact
    location or relocate/skip the assertion.
  - **leaf special-case**: of the 78 stale `shapes.yaml` comments, **5 are
    scaffold/PoC tasks that `generate_shapes_and_contract.py` skips** (`skip-poc:5`)
    ⇒ regen does NOT clear them → scrub those 5 by hand.
  - **non-doc internal CODE/infra path refs in public files** (codex round-6):
    `scripts/README.md:49` lists `scripts/{internal,loop,automation}/` → rewrite to
    describe only the public script dirs (corpus/verify/oracle/provenance/perf);
    `scripts/corpus/validate_tasks.py:180` comment names
    `scripts/automation/lib/file_blocks.py` → genericize. (The `.github/ci.yml:48,74`
    boundary greps legitimately name the internal trees → rewritten in P4, not scrubbed.)
  - `.github/workflows/ci.yml:92` comment names the old internal set → update.
  - **Approach**: fix at GENERATORS first (regenerates ~73 leaf tasks files), then
    mechanical repoint of the remaining leaves (incl. the 5 skip-poc + the 3
    hand-authored ast_denylist/tolerance contract refs) + the script default-output
    constants + package docstrings. Never touch `kernel.py`, scoring numbers,
    oracle, or shapes/meta *values* — only comments, note-strings, links,
    docstrings, and output-path constants. No GPU regression.
- **P3 — construct the internal-free public artifact**: under DECISION-1A this is
  the *snapshot* (exclude the internal set; the live dev repo is left intact, so
  its `.gitattributes`/internal tests/`docs/release/` stay coherent). Under 1B it
  is an in-place `git rm --cached -r docs/{plans,reviews,triage_backlog,brainstorms,architecture,release}`
  + now-internal specs/infra files, with the internal tree already mirrored to a
  private remote. Either way **delete** the generated dumps from the public
  artifact (phase_a_prime_classification.md, phase_a_prime_status.md, the
  *-full-regression.md set, the large triage dumps) and `rm` the untracked V-drafts.
- **P4 — boundary + gitignore + verify** (codex under-spec fix):
  1. **Public artifact** needs NO internal export-ignore rules (nothing internal
     is present) — so its `.gitattributes` keeps only generic rules and its
     `.dockerignore` mirrors. Under 1B only: prune the now-dead export-ignore
     lines for purged paths — **except** keep any line an export-ignored *internal
     test* still asserts (see point 4).
  2. `.github/workflows/ci.yml:47-49` — the release grep is **hardcoded to the old
     internal set**; rewrite it to assert the public tree has zero internal dirs +
     update the bench-core single-direction grep to the new layout.
  3. `tests/unit_tests/test_schema_doc.py` (PUBLIC, ships) — repoint
     `specs/task_api_contract.md` → `docs/reference/task_api_contract.md` + adjust
     §6a assertions to the split. `tests/unit_tests/test_release_boundary.py`
     (PUBLIC) — update `_BENCH_CORE_PATHS` + internal-pattern lists to the new
     layout.
  4. **Internal export-ignored tests travel WITH their internal docs** —
     `test_runners.py` (`.gitattributes:55`, export-ignored) asserts the
     `docs/release/` runbook exists AND that `docs/release/internal-*.md` stays in
     `.gitattributes` (`test_runners.py:114,133`). It is NOT in the public suite,
     so it cannot break the public build; in the dev repo it stays coherent
     because `docs/release/` + the `.gitattributes` line are retained there. Do
     not naively strip that line while the internal test+doc still coexist.
  5. extend `.gitignore` (blanket `results/`, the generated-dump filenames).
- **P5 — README**: see section 6a; README link/fact fixes ship in THIS PR because
  the reorg invalidates its links (codex E: README is coupled, not deferrable).

Each phase ends with the relevant unit tests + the release-boundary CI grep.
The P1.5 edits are **references only** — comments, docstrings, note-strings,
links, and two non-graded code touches: a generator's emitted-comment template and
`standalone_freeze_verifier.py`'s dump **output-path constant** (moves the dump out
of `docs/`). No `kernel.py`, scoring numbers, oracle, shapes/meta *values*, eval
runtime, or daemon logic changes ⇒ a full GPU regression is **not** required. Gate
= CPU unit suite (incl. updated test_schema_doc + test_release_boundary) +
release-boundary grep + `tab run-suite` resolves-101 smoke + the §7 repo-wide
reference sweep. (One reasonable belt-and-braces add: re-run the standalone-freeze
+ contract-regen on a couple of tasks to confirm the generator edits produce
byte-identical kernels and clean contracts.)

## 6a. README updates (in-scope, ship with this PR)

The reorg breaks README links and some headline facts are already stale (audit
#4). Required edits to `README.md`:

Exact README link inventory (verified) + fix:

| Line | Current | Fix |
|---|---|---|
| `:5` | Plan badge → `docs/benchmark_impl_plan.md`, text `Plan-v4.13_FROZEN` | **drop badge** (impl_plan internal; `v4.13 FROZEN` is banned process-noise) |
| `:62` | comment "scaffold from `docs/benchmark_select.md`" | repoint → `docs/kernel-select.md` |
| `:92` | Selection rationale → `docs/benchmark_select.md` | repoint → `docs/kernel-select.md` |
| `:112` | `specs/scoring_spec.md` | repoint → `docs/reference/scoring_spec.md` |
| `:125` | Full defense spec → `specs/reward_hack_defense.md` | **drop** (held internal) unless rewritten-published; if rewritten, repoint `docs/reference/` |
| `:129` | "Selection rationale (100 tasks) … v3.1" → `benchmark_select.md` | repoint → `kernel-select.md`; fix "100 tasks/v3.1" → 101 |
| `:130` | "Implementation plan v4.13 FROZEN — 19 codex review rounds" → `benchmark_impl_plan.md` | **delete line** (internal doc + banned process-noise) |
| `:131` | Task API contract → `specs/task_api_contract.md` | repoint → `docs/reference/task_api_contract.md` |
| `:132` | Tolerance policy → `specs/tolerance_policy.md` | repoint → `docs/reference/tolerance_policy.md` |
| `:138` | status row `\| Plan \| ✅ FROZEN v4.13 \|` | drop the row (banned process-noise) |
| `:140` | `\| 101 task directories (L1=40, L2=38, L3=23) \| ✅ Done \|` | correct counts → **36/35/30** |
| `pip install triton-ascend-bench` | not on PyPI | `pip install git+<public-repo-url>` (URL pending DECISION-1) |
| (add) | — | add a `docs/findings/` showcase link (the strongest public artifact) |

Gate: `git grep -nE 'specs/|benchmark_select|benchmark_impl_plan|v4\.13|FROZEN|codex|round[- ]?[0-9]'` over README == 0; every remaining link target exists in the public tree.

This is README link-integrity + fact-correctness only. The deeper README rewrite
(English abstract, reproduce-on-one-GPU block) is nice-to-have, not required here.
LICENSE/NOTICE/pyproject metadata stay out (section 8).

## 7. Verification gates (must all pass before commit / public push)

1. **Tarball boundary**: `git archive HEAD | tar t | grep -cE 'docs/(plans|reviews|triage_backlog|brainstorms|architecture|release)/|specs/|infra/'` == 0 (specs/ folded into docs/reference; infra/ dissolved into docker/ + package data + internal — DECISION-8).
2. **Public-tree leak grep** on the final public set (excl. `external/`, `tasks/_vendored/`): `git grep -nIE '(/inspire/|/data1/|127\.0\.0\.1:2222|kernel.?pilot|exp2@|<personal-host-tokens>|v4\.13|FROZEN|DESIGN-V[0-9]|codex|round[- ]?[0-9]|R[0-9]+-P[0-9])'` == 0.
3. **History — path AND content** (codex round-7). Two leaks live in history, not
   just the working tree:
   - **path**: no internal path in any commit — `git log --all --name-only --pretty=format: | sort -u | grep -cE '^(docs/(plans|reviews|triage_backlog|brainstorms|architecture|release)/|scripts/(loop|automation|internal)/|tab_automation/|infra/(RFC_0001_scope|upstream_contacts)\.md|specs/(canary_task_design|extraction_failure_taxonomy|submission_throttling|process_isolation|artifact_provenance|ast_denylist|reward_hack_defense)\.md|docs/(benchmark_select|benchmark_impl_plan|DESIGN-opt|DESIGN-presentation-guide|automated_triton_operator_extraction)\.md)'` == 0.
   - **content**: a file that is clean in the working tree may still carry old
     `/inspire/`, `v4.13`, `codex`, `kernelpilot`, or stale-ref tokens in an EARLIER
     commit — `git log -p` / `git grep <gate-2 + gate-7 patterns> $(all public commits)` == 0.
   - **The clean way to satisfy both at once (and the reason DECISION-1A is
     recommended as a SQUASH): publish the public repo as a single scrubbed-snapshot
     commit with NO inherited history.** Then there is no history to grep. Only
     DECISION-1B (in-place `filter-repo` preserving history) needs the per-commit
     content grep above.
4. `pytest tests/unit_tests/` green — explicitly including the **updated** `test_schema_doc.py` (new task_api_contract path + §6a split) and `test_release_boundary.py` (new layout). No skips.
5. **Resolve + installed-wheel smoke** (codex round-8: `run-suite` has no `--dry-run` and requires `--solutions-root`/`--submission`): (a) `python3 scripts/corpus/validate_tasks.py --release-gate` still passes with all 101 release-ready (resolves every meta — proves no eval path referenced a moved doc); (b) after P1.6, in a clean venv `pip install git+<repo>` then `python3 -c "from ascendbench.perf.peak_table import load_peak_table ..."` + a static_checker allowlist load — both resolve from package data, no `infra/` checkout needed.
6. **README**: every link resolves to an existing public path; `git grep` of README for process-noise (`v4.13|FROZEN|codex|round-[0-9]`) == 0; task counts == 36/35/30.
7. **Repo-wide reference sweep — THREE grep forms + behavioral backstop** (codex
   round-5: a literal-path grep alone misses split-path code + bare basenames).
   Over every tracked file except the export-ignored internal set
   (`scripts/{loop,automation,internal}/`, `tab_automation/`, internal `docs/*`
   dirs, export-ignored `tests/`) and third-party (`external/`, `tasks/_vendored/`),
   ALL THREE return 0 (allow only intended `docs/reference/` + `docs/findings/`):
   - **(a) literal paths**: `docs/(plans|reviews|triage_backlog|brainstorms|architecture|release|verification|upstream_reports|npu)/` and `benchmark_select|benchmark_impl_plan|DESIGN-(opt|V[0-9])|automated_triton_operator_extraction` and `infra/` (dir dissolved — DECISION-8; any surviving public ref is stale).
   - **(b) split-path / code construction**: `"docs"[, ]+["/ ]*"(plans|reviews|triage_backlog|brainstorms|architecture|release|npu)"` and `/ +"(plans|triage_backlog|reviews)"` and bare `"specs"`.
   - **(c) bare `specs/` + internal-doc basenames** (no `docs/` prefix): `specs/|canary_task_design|reward_hack_defense|submission_throttling|extraction_failure_taxonomy|process_isolation\.md|artifact_provenance|ast_denylist|phase_a_prime_(classification|status)` (the 3 design-stage security specs are internal now too).
   - **(d) non-doc internal CODE/infra paths** (codex round-6 — escapes a-c):
     `scripts/(automation|loop|internal)/|tab_automation|infra/(RFC_0001_scope|upstream_contacts)` and split forms (`scripts.automation`, `import automation`). These live in public files like `scripts/README.md:49` and `scripts/corpus/validate_tasks.py:180`; the only *legitimate* occurrences are the boundary checks in `.github/workflows/ci.yml:48,74`, which P4 rewrites.
   Covers README, SCORE.md, NOTICE, docs/, all `tasks/*/{contract,shapes,meta}`,
   all public `scripts/`, **`ascendbench/` (ships)**, public `tests/`, `.github/`,
   `*.yaml/*.toml`.
   - **Behavioral backstop** (static grep can't catch a path assembled from
     variables across lines): public `pytest tests/unit_tests/` green (catches
     `test_pa_fix1*` + `test_schema_doc` doc-path breaks), `tab run-suite` resolves
     101, and each touched generator/default-output script writes **outside**
     `docs/` (no internal dir recreated).

## 8. Out of scope (adjacent release blockers; NOT in this plan)

LICENSE Apache-2.0 body, NOTICE BSD-3 (`L3_08 jagged_dense_flash_attention`)
addition, pyproject `<TBD>` URLs + real authors, `tasks/_canary/` seal,
`_frozen_deps_pool` orphan slim, scrubbing `/inspire/` inside shipped
`scripts/*.py` (probe/full_regression). These are real blockers from the audit
but live outside `docs/specs/infra`. **README is now IN scope** (section 6a) since
the reorg directly breaks its links. Recommend a follow-up release-prep PR for the
remaining audit blockers; flag if you want them folded into this one.

## Resolved decisions (codex-reviewed) + remaining open call

- **DECISION-1 (OPEN — needs your call, blocks P0)**: public-release strategy —
  **A** = new public repo from scrubbed snapshot, current repo stays private
  (recommended) / **B** = in-place `filter-repo` on this repo / C, D rejected.
  This is the one genuine fork; everything else below is resolved.
- **DECISION-2 → INTERNAL** (resolved): `automation-llm-repair-architecture.md`
  specs the export-ignored `scripts/automation/` subsystem (`.gitattributes:14`),
  so it documents code not in the public release → internal.
- **DECISION-3 → INTERNAL** (resolved): `submission_throttling.md` (+ the
  verified-runner/leaderboard ops specs) describe unbuilt future infra and publish
  anti-sybil thresholds → hold internal until that infra exists.
- **DECISION-4 → README IN, rest follow-up** (resolved): README link/fact fixes
  ship here (coupled); LICENSE/NOTICE/pyproject/_canary go to a follow-up PR.
- **DECISION-5 → REWRITE core docs** (user, 2026-06-10): public DESIGN/ARCHITECTURE
  are rewritten fresh in English from shipped code, not folded/translated from the
  zh V3-era documents (which go internal as source material). §4 scope.
- **DECISION-6 → reference/ = shipped contracts only** (user, 2026-06-10): publish
  the 4 specs describing shipped behavior; the 3 design-stage verified-runner
  security specs (process_isolation, artifact_provenance, ast_denylist) go
  internal, summarized in one paragraph inside the rewritten DESIGN.md.
- **DECISION-7 → peak_table_h200 cleaned from public** (user, 2026-06-10): zero
  default-code-path consumers ⇒ not shipped publicly; stays at dev-repo `infra/`
  for the internal GPU rank pipeline. P1.6 ships only peak_table_910b (HARD) +
  version_matrix (SOFT/allowlist) as package data.
- **DECISION-8 → `infra/` renamed `docker/`** (user, 2026-06-10): with only the
  two Dockerfiles left public the name is a misnomer; `docker/` is the
  convention. Dev repo's `infra/` becomes 100% internal (single dir boundary
  rule). Consumers: 2 CI `docker build -f` paths + Dockerfile header comments.
- **Sub-flag (your call, larger)**: whether to also strip `phase_a_prime.*` /
  `phase_b_capture` write-state fields from the 101 public `tasks/*/meta.yaml`
  (they are documented as read-only provenance in this plan; stripping is a
  separate `tasks/`-touching change requiring a full GPU regression).
