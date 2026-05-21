# Corpus Consistency Cross-Check — DESIGN.md ↔ specs/

**Date:** 2026-05-19
**Scope:** Expert-facing design corpus = `docs/DESIGN.md` + the 11 `specs/*.md` it references in §12.1, after the two-objective Roofline alignment.
**Method:** Deterministic, evidence-backed cross-check (grep/read with `file:line` citations + code-authority verification). Serves as the auditable stand-in for the codex round-5 confirmation pass, which was environmentally blocked (codex binary healthy — `codex exec` smoke `PONG` rc=0 — but the large high-reasoning corpus-review call hung once at 33 min/0 output and returned empty exit-1 on two retries; this is codex-API/long-run flakiness, not a corpus defect). Codex corpus rounds 1–4 (5→2→4→4 P1, every finding code-verified and fixed) and DESIGN.md rounds 1–3 (→ READY TO PRESENT) had already completed.

## Verdict

**CORPUS CONSISTENT** — zero P0/P1. One P2 precision item found and fixed during this audit (item 5 below). All checks below are reproducible from the repo.

## Cross-Check Matrix

| # | Consistency claim | Evidence | Result |
|---|---|---|---|
| 1 | `t_ref` multi-resource formula uniform; scalar only as explicit degeneration | `DESIGN.md:22` (`max(max_c W_c/π_c, max_r M_r/β_r)` + "单类单资源即 max(W/π,M/β)"), `DESIGN.md:90,447,615,652`, `scoring_spec.md` all multi-resource | PASS |
| 2 | Correctness gate predicate `correctness==ok ∧ ¬fallback`; fallback normalized to correctness fail pre-scoring | `DESIGN.md:414,447,581,584` ↔ `scoring_spec.md:9,11,49` | PASS |
| 3 | S6 composite `release_status ⟺ phase_a_prime.status==PASS ∧ cost_model.status==PASS` | `DESIGN.md:20,115,303` ↔ `scoring_spec.md:56,59,62,128` | PASS |
| 4 | `P_CAP = 5.0`, `ε = 0.01` identical across corpus | `DESIGN.md:23,590` ↔ `scoring_spec.md:26,28` | PASS |
| 5 | `eligibility.py` is the sole score-eligibility authority (non-stub oracle ∧ non-`PHASE-C-MANUAL`); no competing rule in specs | `eligibility.py:36,50` authoritative; sole prior conflict at `extraction_failure_taxonomy.md:13` (conflated "excluded at extraction" with "BLOCKED / not score-eligible") | **P2 — FIXED** (reworded: cpp_stub_dep has no `@triton.jit` upstream → excluded at extraction, explicitly distinct from `BLOCKED` which stays in `|S|`) |
| 6 | CLI subcommand names match `cli.py` authority | `cli.py` = init-solution/run/run-suite/package/submit/validate-submission/validate-result/static-check/analyze; corpus refs `DESIGN.md:475,477,600`, `ast_denylist.md:111` all valid; stale `tab validate-task/validate-solution/extract-kernel` eliminated | PASS |
| 7 | Oracle is **not** an independent truth source (aligned to frozen upstream via dual-anchor) — DESIGN §1.3 ↔ specs | `DESIGN.md:12,339` ↔ `task_api_contract.md:34` ↔ `extraction_failure_taxonomy.md:13` | PASS |
| 8 | Oracle / kernel entrypoint names match code & real tasks | `oracle_pytorch.py::run` (tasks `meta.yaml`, `eval.py:135`, `oracle_pytorch.py:26`), kernel `gpu_triton: placeholder` (resolver-filled); `task_api_contract.md:101,102,141`, `DESIGN.md:295,459` | PASS |
| 9 | §12.1 appendix spec list == actual `specs/*.md` (11) | `DESIGN.md:660` lists exactly the 11 files present in `specs/` | PASS |
| 10 | `reward_hack_defense.md` layer count consistent with DESIGN §8 (L1–L7 + addendum, no "L8") | `reward_hack_defense.md:9,19,20` (7-Layer + "Addendum — R18") ↔ `DESIGN.md:549,550` (L1→L7) | PASS |
| 11 | `s_t` per-task geomean over shapes + `exp((1/|S|)Σ log s_t)` identical | `scoring_spec.md:28` ↔ `DESIGN.md:447,593` | PASS |
| 12 | No old-model residue (AscendC / `S_native` / `in_s_native` / "performance deferred" / ERRATA banners / `FastP_native`) anywhere | 0 hits across all 11 specs + DESIGN (excluding legitimate "no AscendC baseline" declarations and forbidden-ops lists) | PASS |
| 13 | Mermaid diagrams valid | 10/10 render via `@mermaid-js/mermaid-cli` | PASS |
| 14 | No stale cross-references / broken file paths (`tools/*.py`, `leaderboard_validator.py`, `specs/meta_yaml_schema.yaml`, "keys above/below", `oracle.run`) | 0 hits across corpus | PASS |

## Code-Authority Anchors (spec claims verified against code)

| Claim | Code authority | Status |
|---|---|---|
| score-eligible = non-stub oracle ∧ non-`PHASE-C-MANUAL` | `triton_ascend_bench/eligibility.py:36–55` | confirmed |
| `native_coverage` counts `correctness_status=="ok"` (spec adds the designed `∧ ¬fallback` normalization as TARGET) | `triton_ascend_bench/scoring.py:231–242` | confirmed (code-vs-target gap is the known, authorized later phase — out of scope here) |
| `P_CAP = 5.0`, `EPSILON = 0.01` | `triton_ascend_bench/scoring.py:30,31` | confirmed |
| CLI subcommands | `triton_ascend_bench/cli.py` `add_parser(...)` | confirmed |
| oracle entrypoint `oracle_pytorch.py::run` | task `meta.yaml`, `eval.py`, `oracle_pytorch.py` | confirmed |
| release-gate predicate (code currently `phase_a_prime.status != PASS`; composite `∧ cost_model.status` is TARGET) | `scripts/validate_tasks.py:206,255` | confirmed (composite is DECIDED TARGET design) |

## Scope Note

`docs/DESIGN.md` describes the **target** design in present tense and deliberately omits implementation progress (a standing, explicit decision). Known code-vs-target gaps — `scoring.py` no-baseline schema-shim, `scoring.py`/`eval.py` "4 statuses" comments, no `cost_model` field in code, `validate_tasks` not yet checking `cost_model.status`, no `infra/` peak-table file — are **expected** and are a separately-authorized later implementation phase, not corpus-consistency defects. See `docs/.../memory project_tab_two_objective_roofline`.

## Outstanding

- Codex round-5 confirmation pass to be re-run opportunistically when the codex API stabilises; this document is its deterministic equivalent and should converge to the same verdict (zero P0/P1) given codex rounds 1–4 already resolved all substantive contradictions and rounds 3–4 were pre-existing long-tail hygiene (all fixed).
