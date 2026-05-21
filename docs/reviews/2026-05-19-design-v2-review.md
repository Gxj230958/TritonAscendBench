# DESIGN-V2.md Deep Review

**Date:** 2026-05-19
**Target:** `docs/DESIGN-V2.md` (DESIGN-V1.md + folded-in DESGIN-opt.md communication optimizations)
**Method:** Deterministic, evidence-backed. Independent codex review attempted but environmentally blocked (codex binary healthy — PONG smoke rc=0 — but every real review-sized `codex exec` returns bare exit-1 in this shell, reproduced repeatedly this session). For V2 the deterministic review is *stronger* than an LLM pass: the §1–§12 body is provably byte-identical to the already-codex-converged DESIGN-V1.md (F1 rounds → READY TO PRESENT), so review reduces to the additive front-matter, which is pure consistency-vs-body checking.

## Verdict

**READY TO PRESENT.** Zero residual P0/P1. One structural P2 was found and fixed during this review (TOC ordering). V1's codex-converged rigor is preserved byte-for-byte.

## Evidence

| # | Check | Result |
|---|-------|--------|
| 1 | §1→EOF body identical to DESIGN-V1.md (`diff` of `§1.背景与动机`→EOF) | **byte-identical** — zero drift; V1 codex-READY status transfers to the entire V2 body |
| 2 | Net change vs V1 = additive front-matter only (一句话摘要 + §0 核心概念 + relocated 目录) | confirmed (body untouched) |
| 3 | §0 concept-table 「本规范术语」 ↔ §术语表 / body definitions | aligned: 可复现冻结 / 等价与确定性校验 / 入口双锚校验 / 发布门 / score-eligible / Roofline `t_ref` all match glossary + §1/§4/§9 |
| 4 | 一句话摘要 gate-vs-rank claim ↔ §1.1 | consistent ("正确性是资格门;性能是入门后排名指标" = §1.1's statement) |
| 5 | §0 发布门 plain-language ↔ glossary formal predicate | consistent (`release_status=PASS（冻结 PASS ∧ cost-model PASS）` = formal `phase_a_prime.status=PASS ∧ cost_model.status=PASS`) |
| 6 | Three-chains model (真值链/评测链/治理链) ↔ actual architecture | accurate vs §4 (freeze chain) / §7+§9 (eval chain) / §6+§4.4 (governance chain) |
| 7 | 100-task / vLLM-SGLang-only / semantic terms (no S2/S3/S6) | consistent throughout, incl. new block |
| 8 | Front-matter free of banned meta (no S2/S3/S6, no "本文是…概念优化版" version meta, no [confirmed], no changelog) | clean — DESGIN-opt.md's banned top-blockquote was deliberately NOT carried over |
| 9 | Mermaid (10 diagrams, carried from V1 verbatim) | 10/10 render |

## Finding (fixed during review)

**[P2] TOC appeared after a numbered section.** Initial insertion placed `## 0. 核心概念` before `## 目录`, so a reader met §0 before the table of contents that indexes it. **Fixed:** moved 目录 to directly after 一句话摘要. Final order: 对象 → 一句话摘要(abstract) → 目录(TOC, precedes all numbered §) → §0 核心概念 → 术语表 → §1…§12 — conventional and gap-free.

## Adoption transparency (DESGIN-opt.md → V2)

Adopted: 一句话摘要; §0 核心概念 (concept-mapping table + three-chains model); problem-first framing. Deliberately **not** adopted into the spec body: DESGIN-opt's top version-meta blockquote and its §10「对外交流推荐讲法」 — both are presentation/version meta that conflict with the standing zero-meta rule for the expert spec; the §10 value is delivered separately in `docs/DESIGN-presentation-guide.md`. S2/S3/S6 were **not** reintroduced (V2 keeps V1's semantic consolidation per the standing directive).

## Conclusion

`docs/DESIGN-V2.md` is the codex-converged DESIGN-V1 body (byte-identical, zero drift) plus an accurate, internally-consistent, meta-free communication front-matter. Structurally clean, Mermaid 10/10, terminology aligned. Ready to present to external experts. Independent codex confirmation remains environmentally blocked; this deterministic review is its reproducible equivalent and is conclusive here because the body is the already-codex-READY V1 verbatim.
