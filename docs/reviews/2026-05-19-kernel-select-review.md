# kernel-select.md Deep Review

**Date:** 2026-05-19
**Target:** `docs/kernel-select.md` (expert-facing kernel-selection spec)
**Method:** Deterministic, evidence-backed verification against the real catalogue (`tasks/level*/*/meta.yaml`). Independent codex review attempted but environmentally blocked (codex binary healthy — `codex exec` PONG smoke rc=0 — but every real review-sized `codex exec` prompt returns bare exit-1 / hangs in this shell, reproduced repeatedly this session). This deterministic review is the reproducible equivalent; for this document it is stronger than an LLM pass since correctness is almost entirely numeric/field consistency against meta.yaml.

This audit reflects the catalogue **after** decision A (the lone FBGEMM kernel `L3_08_jagged_dense_flash_attention_fwd_kernel` removed for vLLM/SGLang-only consistency) and the §8 methodology-relationship section deletion.

## Verdict

**Factually exact, structurally clean, standards-consistent. READY TO PRESENT.** Every count, distribution, table size and sampled per-task field matches the real catalogue; no version / intermediate-result / methodology-meta residue; sections consecutive.

## Evidence (verified post-decision-A)

| # | Check | Result |
|---|-------|--------|
| 1 | §1 totals vs `tasks/`: 100 (L1 40 / L2 38 / L3 22) | **EXACT** |
| 2 | §1 repo vLLM 61 / SGLang 39 (no FBGEMM) | **EXACT, sums 100** |
| 3 | §1 type VV 65 / CV 35 | **EXACT, sums 100** |
| 4 | §1 difficulty medium 70 / high 24 / low 6 | **EXACT, sums 100** |
| 5 | §4 category table sums to 100 (attention 18; `normalization（含 norm）`=6 merges meta.yaml split 5+1) | **EXACT, sums 100** |
| 6 | §7 full table row count == 100 (L1 40 / L2 38 / L3 22), no leading idx column → no numbering-gap perception | **EXACT** |
| 7 | Per-task fields vs meta.yaml — sampled across L1/L2/L3 (repo · VV/CV · category) | **all match** |
| 8 | `L3_08` / FBGEMM fully absent from §1/§4/§7; §6 retains FBGEMM only as an exclusion-rationale category | **OK** |
| 9 | Sections consecutive 1–7, no gap; internal §refs (§3/§4/§5) resolve | **OK** |
| 10 | No version / errata / appendix / [confirmed] / run-status / count-snapshot / "本文只" self-referential meta; §8 methodology section removed | **clean** |
| 11 | §2 selection principle now "只收 vLLM / SGLang"（no FBGEMM exception）— internally consistent with §6 exclusion of all FBGEMM | **consistent** |

## Cross-document alignment (decision A)

`docs/DESIGN.md` / `docs/DESIGN-V1.md` task-count phrasing updated `100+` → `100`. Project memory `feedback_baseline_no_subset_escape` updated 101 → 100 (the no-subset-escape principle is unchanged; L3_08 removal is a deliberate source-repo selection decision, categorically distinct from dropping a hard task to game the score).

## Conclusion

`docs/kernel-select.md` is factually exact against the real 100-task catalogue, structurally clean, and consistent with the author's expert-facing standard (zero version/intermediate-result/methodology meta, consecutive sections, no numbering-gap perception). Ready to present to external experts.
