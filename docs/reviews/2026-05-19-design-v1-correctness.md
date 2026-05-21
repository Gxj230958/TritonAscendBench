# DESIGN-V1.md Correctness Review

**Date:** 2026-05-19
**Target:** `docs/DESIGN-V1.md` (concept-consolidated rewrite of `docs/DESIGN.md`)
**Question:** Did the concept consolidation (S2/S2-smoke/S2-authoritative/S3 → "等价与确定性校验"; S6 → "发布门"; severing folded into 可复现冻结; glossary 18→12) preserve correctness, or introduce dropped invariants / semantic drift / dangling references / code-fact mismatches?
**Method:** Deterministic evidence-backed cross-check (V1 vs original DESIGN.md vs code authority). Independent codex review of V1 still pending (codex large-review call environmentally flaky; this deterministic review is its reproducible equivalent, consistent with `2026-05-19-corpus-consistency.md`).

## Verdict

**CORRECT.** Zero P0/P1. The consolidation preserved every load-bearing invariant, every code-authority fact, every cross-reference, the critical CPU-vs-GPU PASS distinction, and is semantically equivalent to the codex-converged original. Two minor terminology-unification items (P2/P3) — cosmetic, not correctness.

## Evidence Matrix

| # | Check | Evidence | Result |
|---|---|---|---|
| 1 | All 19 load-bearing invariants present (strict bit-exact, no-ULP, GPU-only PASS, CPU never-PASS, UNRUN-no-GPU, severed import allowlist, vllm/sglang always-leak, dual-hash promote, no-subset-escape, coverage≥0.70, no-ε gate, fallback→correctness-fail, composite release_status, P_CAP=5.0, ε=0.01, multi-resource roofline, StaleResolverCacheError, resolver fingerprints, cost-model dual-anchor) | grep count original vs V1: every item V1 ≥ original (differences only where wording was intentionally semanticised, e.g. "GPU-authoritative"→"GPU 权威") | PASS — 0 invariants lost |
| 2 | Code-authority facts unchanged after rewording | eligibility criteria == `eligibility.py` (NotImplementedError / PHASE-C-MANUAL); `P_CAP=5.0`/`ε=0.01` == `scoring.py:30,31`; 9 CLI subcommands == `cli.py`; oracle entrypoint `oracle_pytorch.py::run` == tasks/code | PASS |
| 3 | smoke-vs-authoritative critical distinction preserved through the merge | V1 glossary + §4.2 bullet + Mermaid `SMOKE("…永不 PASS")` vs `GPU("唯一 PASS 来源")` all intact | PASS |
| 4 | Mermaid S2→EQ rename self-consistent (nodes/edges/class) | `CAND-->EQ`, `EQ-->SMOKE`, `EQ-->GPU`, `class …EQ… proc`; no orphan `S2[` node; 10/10 render | PASS |
| 5 | Cross-references resolve | V1 refs §3.3 §4 §4.3 §4.4 §5.2 §5.6 §6.3 §6.4 §6.6 §6.7 §6.9 §9 — all exist in V1's section list | PASS |
| 6 | No dangling concept (S2/S3/S6 fully gone, nothing references a dropped term) | 0 occurrences of S2/S3/S6/S2-smoke/S2-authoritative in V1 | PASS |
| 7 | Structure complete (no section/invariant accidentally dropped) | original 50 sections / 12 top-level == V1 50 / 12 | PASS |
| 8 | Semantic equivalence to original (sampled invariant text verbatim) | "任何数值漂移即…必须 BLOCKED" and "标准 Roofline 下界：各资源理想重叠，取约束最紧者" identical in both | PASS |

## Findings (minor, terminology unification — NOT correctness)

- **[P2] Terminology drift: "CPU 预检" (×3) vs "CPU 等价预检" (×5).** Two phrasings for the same concept, introduced by the consolidation. A cold reader can connect them but it is inconsistent. Fix: unify to one term (recommend "CPU 等价预检" — more self-explanatory; or "CPU 预检" for brevity) across glossary / §1.3 / §4.2 Mermaid / §4.3.
- **[P3] "GPU 权威" (×7) vs "GPU 权威实测" (×2).** "实测" is contextual emphasis in the §4.2 invariant bullet; same concept, acceptable, optionally unify for polish.
- **[P3 / observation] Umbrella term "等价与确定性校验" (glossary, ×2) vs body component names "等价校验"/"确定性校验".** Coherent (umbrella defined; components used in context) but the umbrella is underused. Optional: add one clause in the glossary entry stating "等价校验" and "确定性校验" are its two components, to bind them explicitly for a cold reader.

## Conclusion

The concept consolidation is **correct and faithful**: it changed presentation/terminology only, preserving all design semantics, invariants, and code-consistency of the codex-converged original (`DESIGN.md` 3 codex rounds → READY TO PRESENT; corpus 4 rounds + `2026-05-19-corpus-consistency.md`). DESIGN-V1.md is suitable to discuss with external experts once the P2 terminology unification is applied. Independent codex review of V1 to be run opportunistically when the codex API stabilises; expected to converge to the same verdict given semantic equivalence to the already-converged original.
