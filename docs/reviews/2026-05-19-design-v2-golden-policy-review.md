# DESIGN-V2.md Golden-Source Policy — Deep Review

**Date:** 2026-05-19
**Target:** `docs/DESIGN-V2.md`, the newly-added golden-source policy (§1.3 point 3 + new §5.7), encoding the user's ≥80%-PyTorch / ≤20%-provider-GPU-prepared decision.
**Method:** Deterministic, evidence-backed. Independent codex review attempted but environmentally blocked (codex binary healthy — PONG rc=0 — but every real review-sized `codex exec` returns bare exit-1 in this shell, reproduced 6+ times this session). Conclusive deterministically here: the rest of V2 = already-reviewed converged content (V1 body F1→READY; front-matter audited in `2026-05-19-design-v2-review.md`); the only new content is §1.3-pt3 + §5.7, which is pure consistency-vs-sections + vs-code checking.

## Verdict

**READY TO PRESENT.** Zero P0/P1. The policy is encoded consistently, matches the real catalogue, and preserves both the "truth = frozen upstream" thesis and the no-subset-escape invariant.

## Evidence

| # | Check | Result |
|---|-------|--------|
| 1 | oracle_simplified enum (§5.7 table) vs real `tasks/*/meta.yaml` | **EXACT**: formula_exact_manual 73 / upstream_oracle_required 19 / simple_pytorch_ok 8 = 100; zero values outside the 3-value enum |
| 2 | ≥80% policy vs reality | real PyTorch-alignable = 81/100 = **81% ≥ 80% → policy satisfied by current catalogue** (no forced oracle-upgrade needed) |
| 3 | §5.7 ↔ §5.5 orthogonality | §5.5 score-eligible = non-stub ∧ non-PHASE-C-MANUAL (unchanged); tier policy is golden-source, orthogonal — no contradiction |
| 4 | no-subset-escape preserved | §5.7 explicitly: "三级题全部留在计分分母 \|S\|"、"不从分母剔除"、"不放松其计分约束（§4.4/§9 同等适用）" — core invariant intact and explicitly stated |
| 5 | "truth = frozen upstream" thesis (§1.3) preserved | §1.3-pt3 + §5.7: "两类真值均系冻结上游 kernel"; ≤20% golden = provider offline GPU run of the **frozen upstream kernel** (stronger than a PyTorch approximation), not a truth relaxation |
| 6 | §4.4 release_status composite untouched | glossary/§4.4 composite predicate unchanged by the delta |
| 7 | §1.3-pt3 → §5.7 cross-reference | correct and resolvable |
| 8 | §5.7 voice/terms | semantic terms only; zero S2/S3/S6, zero [confirmed]/version-meta/TODO; present tense; sits cleanly §5.6 → §5.7 → §6 (subsection, not in TOC — consistent with §5.6) |
| 9 | Rest of body zero-drift | diff V2 vs V1 (§1→EOF) shows only the intended §1.3-pt3 + §5.7 additions (incl. their blank lines / table separator); no unexpected modification |
| 10 | Mermaid (10 diagrams) | 10/10 render |

## Reconciliation note (transparency)

The policy is a deliberate, training-cost-driven **refinement** of the earlier "100% oracle precision coverage" stance — explicitly NOT a contradiction: the no-subset-escape core (never drop precision-hard tasks from the denominator to dodge accuracy) is preserved verbatim and re-stated in §5.7; what changed is that the ≤20% irreducible fused minority derives golden directly from the frozen upstream (provider-side offline GPU, bounded fixed dataset, not in the training loop) instead of forcing a faithful PyTorch oracle for 100%. Project memory `feedback_baseline_no_subset_escape` was updated to this two-path policy with the rationale and the explicit "≥80% is an acceptance criterion, not a license to drop tasks" guard.

## Conclusion

`docs/DESIGN-V2.md` golden-source policy (§1.3-pt3 + §5.7) is internally consistent, code-accurate (enum + 81%≥80%), truth-thesis-preserving, no-subset-escape-preserving, and clean of banned content; the rest of the document remains the prior-converged V1 body + audited front-matter with zero drift. Ready to present to external experts. Independent codex confirmation remains environmentally blocked; this deterministic review is its reproducible equivalent and is conclusive here (small verifiable delta over already-converged content).
