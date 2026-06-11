# TAB score — dual-metric reporting

## HEADLINE (the only scored figure)

**97 / 100 faithful** — of 100 real, **byte-exact frozen upstream** GPU Triton kernels,
97 extract + verify (GPU bit-exact ∧ independent faithfulness audit), staleness-free,
reproducible across two full GPU runs (zero flakiness).

This number may change **only** by re-running the unmodified frozen tasks. It must
**never** move because we judged a kernel buggy and patched it. Every scored
`kernel.py` is byte-identical to its cited upstream commit (`meta.upstream.file_sha256`
/ `phase_a_prime.standalone_sha256`); that is what makes the number independently
auditable.

### 3 BLOCKED (all proven UPSTREAM issues — the bench's highest-value output)

| Task | Upstream | Nature |
|------|----------|--------|
| **L2_24** fwd_kv_reduce | vLLM `lightning_attn.py:444` | cross-block decay **off-by-one** (`array = arange + 1`); numerically proven (remove +1 → matches textbook ref to 0.00) |
| **L2_23** fwd_kv_parallel | vLLM `lightning_attn.py:444` | **same** off-by-one (was a D=256 scaffold-OOM masking it; de-scaffolded to D≤128, now a tolerance-absorbed false positive) |
| **L2_33** rope stage1 | sglang `rocm_mla_decode_rope.py:187` | non-scalar-`if` **debug-assert** — Triton compile-compat, **NOT a numerical bug** |

BLOCKED here means "fix UPSTREAM before NPU migration is meaningful." We refuse to
silently fix them — that refusal is what makes the discovered bugs credible.

> Note: `L2_23` / `L2_24` are EXCLUDED from the `iter_all_tasks()` metrics
> (cost_model / C2 denominator) because their vendored upstream kernel carries a
> real off-by-one bug (vLLM `lightning_attn.py` cross-block decay
> `array = arange + 1`). A correct oracle therefore necessarily diverges
> (~80 dtype-ULP, far above tolerance), so grading them is not meaningful. This
> is **exclusion-with-evidence, not a silent fix or hide** — the bug is proven in
> the backing artifacts below. (`L2_33` stays scored: its debug-assert is
> compile-compat, not numerical.)

Backing artifacts: the reproduction proof script
(`L2_24_lightning_attn_offbyone_proof.py`, 0.28 → 0.00) and the filed-bug
writeup are retained in the repository git history.

## Rigour bar — what qualifies a task for the SECONDARY number

ALL FOUR required (L2_24/L2_23 satisfy all; L2_33 satisfies none):
1. an **independent-oracle** proof showing **structured** (not fp-noise) divergence that
   vanishes on the patch;
2. a **minimal mechanical** patch with an **external canonical witness** (e.g. OpenNLPLab
   `arange(0,BLOCK)` with no `+1`);
3. the oracle is canonical/self-consistent and **not kernel-mirrored** (here it also
   matches the kernel's OWN diagonal stage);
4. a **filed upstream** bug report.

Taste/refactor/loose-tolerance/compile-shim **never** qualify. A daemon GPU PASS on a
corrected variant is **confirmatory only** (the op-family tolerance absorbs the off-by-one
both ways); the **discriminating** evidence is always the independent-oracle proof.

## Reporting rules (mandatory)

- A bare "99" with no qualifier is a **reporting violation**. The two numbers never appear apart.
- **97 is the default** everywhere a single number is needed (release gate, top-line).
- Surface the BLOCKED-with-proof story **prominently** — it is a primary deliverable.
- One-line frame: *"97/100 faithful (real upstream, as-is); 3 BLOCKED — 2 are one proven
  vLLM off-by-one (1-line fix filed upstream + GPU-bit-exact corrected variants → 99
  migratable-after-fix), 1 is an sglang compile-compat issue."*

## If upstream merges the fix

Re-freeze the scored task to the new fixed commit via the normal Phase-A freeze flow
(new `file_sha256`) — **never** by hand-editing the pinned kernel. The task then
legitimately flips to PASS at the new hash.
