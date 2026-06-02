# OVERALL plan — align bench fixture dtypes to vLLM/SGLang production

**Status: PLAN ONLY. No task modified, no GPU run. Supersedes the fragments in
`2026-06-01-dtype-alignment-plan.md`; builds on the per-task audit in
`docs/reviews/2026-06-01-dtype-per-task-audit.md`. Date 2026-06-01.**

## 0. Goal and the precision REFERENCE (corrected understanding)

Goal: make the bench's precision reference match what vLLM/SGLang actually compute
in production — i.e. at the production dtype (bf16/fp16), not fp32.

The authoritative precision reference in this bench is the **GOLDEN** — the output of
running the **frozen ORIGINAL upstream kernel** (`eligibility.py:86`,
`phase_a_prime.py:2389`: "authoritative truth source ... offline phase B golden
capture"). The golden is captured **at the fixture dtype**. Therefore:

> **Aligning dtype = set the fixture to the production dtype, so the golden captured
> from the original kernel IS vLLM/SGLang's real production-dtype output.** The
> independent PyTorch oracle is a SECONDARY dev cross-check, not the reference.

This corrects the earlier framing that treated the oracle as the precision reference.

## 1. Current state (codex-arbitrated audit)

103 tasks. **32 already aligned/faithful** (29 exact + 3 production-fp32) → no change.
**71 need alignment** (fixture fp32 but production bf16/fp16).

## 2. Efficiency: the 71 collapse into ~6 dtype PATTERNS

We do NOT investigate 71 tasks independently. They share a small number of patterns;
verify each pattern once, then bulk-apply.

| # | pattern | production dtype | example tasks |
|---|---|---|---|
| P1 | activation-fed elementwise/reduce/attention/gemm | inputs bf16, output bf16 | silu/gelu/norm/lora/gemm/attention (majority) |
| P2 | SSM mixed | x/B/C/dt/z bf16; A/D/dt_bias/state fp32; out bf16/fp32 | chunk_scan/state, gdn, recurrent |
| P3 | quantize/dequantize | activation bf16 in, int8/fp8 out (or reverse) + fp32 scale | act_quant, per_token_group_quant |
| P4 | fp16 model path | inputs fp16 | awq_gemm / awq_dequantize (verify) |
| P5 | output-dtype-tied (7) | output dtype = input dtype (empty_like / out-as-both) | L1_01,L2_15,L2_16,L2_17,L2_20,L2_40,L3_07 |
| P6 | index/scalar | already int/scalar (no change) | (mostly already ALIGNED) |

## 3. How to FIND the production dtype + ENSURE it is correct (3-tier, conservative)

Evidence strength: ① kernel/wrapper explicit `assert`/`.to(dtype)` > ② upstream unit
test pinning dtype > ③ call-site tensor-origin trace > ④ role inference (weakest).

- **Tier 1 — static trace (cheap, fallible):** per tensor, cite the strongest evidence
  available (`file:line`) and a confidence (confirmed/likely/spec). The audit did this;
  `likely`/`spec` rows must be hardened, not trusted.
- **Tier 2 — dynamic interception (gold standard, for AMBIGUOUS only):** instantiate the
  upstream layer/wrapper with a representative (bf16) config, monkeypatch the kernel to
  LOG each arg's real dtype at call time, run one forward. This OBSERVES the production
  dtype instead of inferring it. Run it on **~1-2 representatives per pattern** (≈8 runs),
  not all 71; the pattern then transfers to its group.
- **Tier 3 — codex cross-check:** codex independently re-verifies the per-pattern dtype
  decisions and every edge case.
- **Conservative rule (hard):** only change a task's dtype where production is CONFIRMED
  (Tier-1 ① evidence or Tier-2 observation). Anything still inferential → leave fp32
  (no false alignment). Activation dtype = model compute dtype: take **bf16** as the
  canonical production dtype, note fp16 as an alternative deployment.

## 4. The two gates after alignment (hybrid; keep the fine logic gate)

- **PRIMARY — GOLDEN (the "aligned with vLLM/SGLang" gate):** with the fixture at
  production dtype, the golden = original kernel's bf16 output = vLLM/SGLang's real
  output. On GPU the extracted kernel == original (bit-exact) → matches. The real
  production-precision test is **NPU vs golden at the 910B stage**.
- **SECONDARY — oracle cross-check (dev, keeps bug sensitivity):** keep the existing
  fp32 fixture + 0.001 tolerance as the SCORED logic gate (catches sub-bf16-ULP logic
  bugs). Add a production-track comparison: oracle computes fp32 then **casts its output
  to the production dtype**; compare kernel-bf16 vs oracle-bf16 with a **1-ULP** policy
  (`max_ulps=1`, or allclose `rtol=0.0078125`/`atol≈1e-7` for bf16; fp16 `rtol=0.000977`).
  This is tight, not loose — it catches every bf16-visible bug; sub-ULP bugs are covered
  by the fp32 track. (No loose 0.01-0.02 tolerance classes — dropped per codex.)
- **bf16 INPUTS are gate-safe even for `tl.dot` kernels** (codex confirmed: Triton bf16
  dot uses fp32 accumulation; products bf16×bf16 are fp32-exact), so inputs can switch
  without breaking the existing tight gate.

## 5. Execution sequence (efficient, grouped; RUN LAST)

1. **Group** the 71 into the 6 patterns (from the audit data).
2. **Verify each pattern** (Tier 1 cite + Tier 2 dynamic on ~8 representatives + Tier 3
   codex). Output: a confirmed production-dtype spec per pattern + flagged edge cases.
3. **Implement the 1-ULP comparison policy** in `tolerance.py` (`max_ulps` mode) + unit
   tests (correct→pass, 2-ULP→fail, expected-0 floor).
4. **Bulk-apply** the dtype changes by group via a script (meta.dtypes + shapes inputs +
   output dtype) + make oracles return `fp32_result.to(out_dtype)`. Edge cases hand-done.
5. **Local validate** all (schema + oracle-smoke + severance). NO GPU.
6. **codex review** the comparison-policy impl + a sample of aligned fixtures per pattern.
7. **ONE GPU batch** (the aligned set) at production dtype on the daemon; confirm PASS.
8. **Audit**: a task that FAILS the production track but PASSES the fp32 track = a real
   bf16-precision-path finding (investigate), not a fixture bug.

## 6. Effort and scope options

| scope | what | effort |
|---|---|---|
| **Full (recommended)** | all 71 aligned to production dtype | ~1 day dev + 1 GPU batch |
| Subset | only the clear P1 bf16-activation group, skip ambiguous | ~half day + 1 GPU batch |
| Defer | keep fp32 (already correctness-complete); align when 910B stage needs it | 0 now |

Pattern-verify (~8 dynamic runs) + scripted bulk edits make this a grouped effort, not
71 independent investigations.

## 7. Honest framing

- This is a **fidelity upgrade** (golden = production output), **not a correctness fix**:
  the fp32 fixtures already verify the kernel logic on GPU. Value is realized at the
  910B NPU migration stage (golden must be the real bf16 output to test NPU against).
- Conservative: unconfirmed dtype → no change. The fp32 logic gate stays as the scored
  bug-catcher; the production track is additive.
- Not all 71 are bf16 — P3/P4 are quant/fp16; resolve each task's real output dtype.

**No code changed; no run started. Next: codex DEEP review of this overall plan, then
(on your go-ahead) start §5 step 1 grouping — still local-first, GPU last.**
