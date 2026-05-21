# DESIGN-V2.md — Codex Review Loop (converged)

**Date:** 2026-05-19 → 2026-05-20
**Target:** `docs/DESIGN-V2.md` (triton-ascend-bench final TARGET design spec, for external-expert alignment)
**Method:** `/codex-loop` — real Codex review via the `humanize:ask-codex` skill (codex-cli 0.131.0, gpt-5.4 high), Claude critically adjudicates each finding (project rules + settled decisions override Codex), batch-fixes genuine must-fix, re-reviews. Converged at **Round 11 = ALL PASS**.

## Round record

| Round | Codex verdict | Adjudication |
|------:|---------------|--------------|
| R1 | 3 MUST-FIX | all genuine — fixed (oracle/golden vs §5.7; cost_model schema scalar↔multi-resource; dual-anchor name-set overclaim) |
| R2 | 4 reported | 1 genuine (glossary cost_model row) fixed; 3 **rejected** — Codex relitigated the SETTLED target-vs-code convention (composite release_status / scoring Roofline / fallback→correctness all decided-target, present-tense by design, no impl caveats) |
| R3 | 3 MUST-FIX + 1 NICE | all genuine §5.7-ripple — fixed (pipeline scaffold-oracle ordering; offline golden must cover hidden shapes; §6.3/§7.1 two-branch golden; oracle_tier alias) |
| R4 | 1 MUST-FIX | genuine — fixed (§4.3 anchor-2 = wrong-operator filter at family tolerance, not precision cert; reconciles §4.3↔§5.7) |
| R5 | 3 MUST-FIX + 1 NICE | all genuine — fixed (BLOCKED status-code drift; §7.2 allowlist label; §7.1 missing timing step; §1.3 anchor-2 note) |
| R6 | 1 MUST-FIX | genuine — fixed (§7.2 "零 site" → "零 vllm/sglang + 非允许集 site-packages") |
| R7 | 2 MUST-FIX + 1 NICE | all genuine — fixed (sanitized-mount input model §3.3/§6.6; §4.2 import-audit-before-compare ordering; §6.4 Layer E title) |
| R8 | 1 MUST-FIX + 1 NICE | genuine — fixed (candidate mount must be sanitized, not raw task-dir; runner-secret RNG scope) |
| R9 | 1 MUST-FIX + 1 NICE | genuine — fixed (§3.1 panorama oracle-finalize ordering; P_CAP↔P_{cap} symbol unify) |
| R10 | 1 MUST-FIX + 1 NICE | genuine — fixed (cost-model cross-check placement unified §3.1/§6.10/§7.2 = inside phase_a_prime; §6.1 oracle_pytorch.py path handling) |
| **R11** | **ALL PASS** | **converged** |

Total: 11 rounds; ~17 genuine MUST-FIX + ~7 NICE-TO-HAVE applied; 3 findings rejected (transparently — Codex relitigating an explicitly-settled decision). Mermaid validated 10/10 after every round.

## Adjudication transparency

Per CLAUDE.md (external review = input, not command; settled decisions override): R2's 3 rejected findings all asked to add "this is target-state, not current code" caveats. The spec is, by the author's explicit and repeated decision, a present-tense TARGET design that deliberately omits implementation-status — code-vs-target lag is tracked separately (memory `project_tab_two_objective_roofline`, `docs/reviews/`). Adding such caveats would violate that decision; rejected with documented reason. Codex's #1 R1 fix (split glossary "oracle/golden" into two rows) was adopted in intent but adapted to a single precise row, respecting the settled glossary-consolidation decision. All other findings were genuine internal-consistency defects (largely ripple effects of the §5.7 golden-source policy and accumulated cross-section/diagram drift) and were fixed directly.

## Outcome

`docs/DESIGN-V2.md` is internally consistent (Codex ALL PASS at R11), Mermaid 10/10, terminology/symbols unified, all §5.7 golden-policy ripples reconciled across §1.3/§3.x/§4.x/§5.x/§6.x/§7.x/§9. Ready to present to external experts.
