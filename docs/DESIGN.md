# TritonAscendBench — Design Rationale & Scoring Model

> What TAB measures, and why. For the machine-facing scoring detail see
> [`docs/reference/scoring_spec.md`](reference/scoring_spec.md); this document
> is the design rationale that spec implements.

TritonAscendBench (TAB) is a **migration benchmark**. Each task gives you an
existing production GPU Triton kernel taken from a real vLLM, SGLang, or
flash-linear-attention inference path, and asks one question: *can this kernel be
ported to the Huawei Ascend 910B NPU via `triton-ascend` without losing
correctness or regressing performance?* The corpus is 101 tasks across three
difficulty tiers (L1=36, L2=35, L3=30).

---

## 1. What TAB is — the migration thesis

Most kernel benchmarks test **generation**: given a PyTorch reference, can a
model *write* a fast kernel from scratch?

- [KernelBench](https://github.com/ScalingIntelligence/KernelBench) — PyTorch → CUDA generation.
- [TritonBench](https://github.com/thunlp/TritonBench) — PyTorch → Triton generation.

TAB tests **migration** instead. The starting point is not a blank file — it is
*real production GPU Triton code that already ships and already works*. vLLM and
SGLang run today on H100/A100 with mature, tuned Triton kernels; the industrial
question is whether those exact kernels can move to a different accelerator
without behaving differently. So the thing being scored is **fidelity, not
creativity**: a submission is good when its 910B port reproduces the upstream
kernel's numerics and saturates the NPU's Roofline — not when it invents a
clever new algorithm.

This framing drives every other design decision:

- The reference is the **upstream kernel's observable behaviour**, captured as a
  PyTorch ground-truth oracle, not a hand-written "ideal" implementation.
- There is **no AscendC hand-written baseline**. The comparison is "port vs the
  GPU-Triton original," not "triton-ascend vs native assembly." (The `_native`
  suffix on scoring symbols such as `capped_geomean_native` is retained only for
  output-schema stability and denotes the *NPU-triton track*, not AscendC.)
- Performance is normalized against the **910B Roofline**, because the source GPU
  and the 910B differ in peak capability and raw latency is not cross-hardware
  comparable (§4).

---

## 2. Task selection and tiers

All 101 tasks are real `@triton.jit` kernels lifted from production inference
paths — never synthetic kernels constructed for the benchmark. Full selection
rationale, the per-task source table, and the 910B hardware-fit criteria live in
[`docs/kernel-select.md`](kernel-select.md); the essentials:

**Selection principles.** Production-path only (training-backward `_bwd_*` /
`*_backward` kernels are excluded — the target scenario is inference / agentic-RL
rollout); paired coverage of related operators (GELU↔SiLU, ALiBi↔RoPE,
quantize↔dequant, vLLM↔SGLang dual-source state merges); and de-duplication of
near-identical fused-op stage variants.

**910B hardware fit.** Selection is constrained by what `triton-ascend` can
actually express and what the 910B can actually *measure*. FP16 / BF16 / INT8
are native on both cube and vector units, so they stay. FP8 has no cube/vector
arithmetic on 910B (storage + software cast only), so FP8-cube-arithmetic
operators are excluded — their performance cannot be honestly measured on the
target hardware — and only a single fused FP8 software-cast operator is kept as a
correctness/stress representative. Other constraints (192 KB Unified Buffer,
`tl.dot` contraction-dim alignment, no BF16 `tl.atomic_add`, no async-copy /
pipeline / TMA) shape which fused and attention kernels are tractable.

**Tiers** track migration complexity:

| Tier | Tasks | Shape | 910B units |
|------|------:|-------|-----------|
| **L1** | 36 | Single-op: bias / norm / activation / sampling / KV-index / quant | Vector |
| **L2** | 35 | Fused / cube matmul: LoRA matmul, W8A8 INT8 GEMM, MoE routing+permute, Mamba conv1d, gated-delta gating | Cube + Vector |
| **L3** | 30 | Full subsystem: Flash / Paged / Unified Attention, Fused MoE, Mamba / GDN / KDA chunk pipelines | Full pipeline |

Difficulty rises with tier: L1 is mostly `medium`/`low`, L3 is almost entirely
`high` — chunk-recurrent + online-softmax fused kernels are the genuine hard edge
of `triton-ascend` migration.

---

## 3. The correctness model

A task is *correct* only when it clears **two independent proofs**, each
answering a different question. Both are real, code-enforced gates — not
heuristics.

### 3.1 Oracle-equivalence — "does the kernel compute the right math?"

Every task ships a PyTorch ground-truth oracle (`oracle_pytorch.py`). A
candidate's output is compared against the oracle per shape by
`ascendbench.correctness.tolerance.compare`, using the task's declared tolerance
class. The tolerance policy (full list in
[`docs/reference/tolerance_policy.md`](reference/tolerance_policy.md)) is
dtype-aware rather than one-size-fits-all:

- Floating-point classes cast to fp32 and run `torch.allclose(atol, rtol)`, with
  a sampled fast-fail followed by an **authoritative full-tensor check** so NaN /
  Inf or drift outside the sample window cannot sneak through
  (`tolerance.compare`).
- Integer / bool / RNG-sampling outputs default to **bit-exact** `torch.equal`
  (`_bit_exact_compare`) — they must never be fp32-cast, which would silently
  round e.g. `int64(2^24+1)` to `2^24`.
- Quantization outputs that legitimately round either way at a bucket boundary
  use explicit ULP tolerance: `int_atol` for int8/int32 rounding
  (`_int_atol_compare`), and ordered-code ±1 ULP for fp8 e4m3
  (`_fp8_ulp_compare`, which also caps the *fraction* of off-by-one elements so a
  systematic rounding-mode bias is still caught).

This gate is recorded on each task as
`provenance.golden.oracle_kernel_equivalence`, and is honoured by the predicate
`is_provenance_eligible` (`ascendbench/release_gate/eligibility.py`). A task is
oracle-equivalence-eligible when that status is `VERIFIED` — or `CEILING_ACCEPTED`,
an allowlisted large-shape dtype-ULP ceiling for a correctly-aligned task whose
exact tolerance is structurally unreachable at large sizes, or a task whose
provider-side `gpu_kernel_capture_status` is `DONE` — **and** its
`tolerance_source` is not the deferred `migration_drift_budget` path.

The oracle anchor is honest about its own strength: it is validated against the
upstream kernel's wrapper over a sampled set of shapes, which the source code
itself documents as *a strong empirical anchor, not a formal impossibility
proof*. Input discriminating power (shape distributions that actually separate a
correct port from a plausible-but-wrong one) is therefore part of the corpus
design, not an afterthought.

### 3.2 Bit-exact severance verification — "is the extracted kernel faithful to upstream?"

Each task's `kernel.py` is a byte-frozen snapshot of the upstream source. To make
it runnable standalone (away from the full vLLM/SGLang runtime), the build
pipeline rewrites its imports and slices its dependency closure — but it must
**never** touch the numeric logic. The severance verification proves exactly
that: it runs the *original* upstream-wired kernel and the *severed* standalone
kernel side by side on GPU and asserts they are **bit-identical** —

- same output key set, and per output identical `dtype` / `shape` / `stride` /
  `storage_offset`;
- `torch.equal` bit-exact equality (no fp32-cast escape, no ULP tolerance);
- identical aliasing (which outputs share a storage buffer) and identical
  in-place input mutation;
- determinism: the severed side is re-run several times and must reproduce its
  own output bit-for-bit.

The verdict is recorded as `phase_a_prime.status`. A pass requires a **real GPU
run** — a CPU / interpreter pre-screen can never produce a pass, and a host
without CUDA returns an honest "needs GPU" rather than a fake success. If the
severed kernel introduces even a single bit of deviation, or leaks an upstream
runtime import, the verification blocks.

What this proves and does not prove is explicit by design: it is a strong
guarantee that *severance introduced no observable change* on the enumerated
shapes, aliasing, and mutation surface — the hardest gate against a port silently
drifting from upstream. It is **not** an independent proof that the upstream
kernel itself is correct; that is the job of the oracle-equivalence gate above.
The two are separate stages with separate criteria, and a task must satisfy both.

### 3.3 What "PASS" means

A task is **release-ready** iff it clears *both* correctness proofs over the
oracle-aligned eligible set. The composite predicate is
`release_ready_with_provenance` =
`release_status_pass` (severance / `phase_a_prime.status == PASS`) **AND**
`is_provenance_eligible` (oracle-equivalence). Crucially, **`cost_model.status`
is not part of this gate** (§4.3) — release readiness is a *correctness*
statement.

### 3.4 The standalone, non-GPU correctness path

The candidate correctness grader runs **without any GPU or NPU**.
`ascendbench.integrity.process_isolation` executes the submitted
`npu_triton_impl(inputs)` in an isolated subprocess — safetensors-only IPC, the
child never sees the oracle, the oracle output, `golden/`, or the
`hidden_shapes` section — and `tolerance.compare` grades each shape against the
in-process oracle. This makes the correctness signal reproducible on a plain dev
box, independent of hardware availability.

Hardware authority is layered on top of that path: the **provider-side
verification of a submitted port runs on the 910B NPU** (the actual migration
target, where both correctness and performance are confirmed), while the GPU is
used only **offline**, to establish the oracle-equivalence baseline and the
severance cross-check that anchor the corpus. A dev-box or GPU verdict is never
substituted for an NPU verdict.

---

## 4. The scoring model

Scoring has two objectives over a **fixed** score-eligible set `S` (single
authority: `ascendbench/release_gate/eligibility.py`). `|S|` never shrinks to
flatter a submission — failures stay in the denominator (no subset escape).

### 4.1 Objective 1 — the correctness coverage gate

```
native_coverage = #{ t ∈ S : correctness_status_t == ok ∧ ¬fallback_t } / |S|
```

Ranking is gated on `native_coverage ≥ 0.70`
(`MIN_COVERAGE_FOR_RANKING = 0.70`, `scoring.is_rank_eligible`). A detected
PyTorch fallback is normalized to a correctness failure *before* scoring, so a
"correct but it secretly fell back to eager PyTorch" solution cannot clear the
gate (`native_coverage` keeps the `¬fallback` conjunct as a second line of
defence). There is no ε on coverage: failed or missing tasks contribute 0 to the
numerator but are **not** removed from `|S|`.

### 4.2 Objective 2 — Roofline-normalized performance (the ranked metric)

Because the source GPU and the 910B differ in peak capability, absolute latency
is not cross-hardware comparable. Performance is instead an **efficiency against
the 910B Roofline**. For each shape, the per-task cost model
(`ascendbench/perf/cost_model.py`) gives per-compute-class FLOPs `W_c` and
per-resource bytes `M_r`; the packaged peak table
(`ascendbench/policies/peak_table_910b.toml`) gives peak compute `π_c` and peak
bandwidth `β_r`. The theoretical-latency floor is the standard Roofline lower
bound (`scoring.compute_t_ref`):

```
t_ref = max( max_c W_c/π_c ,  max_r M_r/β_r )
```

The per-shape efficiency is `e = min(P_CAP, t_ref / t_npu)`
(`scoring.shape_efficiency`). Theory bounds `e` to `(0, 1]` because `t_ref` is a
floor; an `e > 1` means the measured time fell below the theoretical floor (a
cost-model or timing artifact) and is clamped at `P_CAP = 5.0`. The task-level
score is the geomean of its per-shape
efficiencies, with ε-fill on any failure:

```
s_t = (∏_s e(t,s))^(1/n_t)   if all statuses ok and no fallback
s_t = ε = 0.01               otherwise
```

and the corpus headline is the capped geomean over the fixed `|S|`:

```
capped_geomean_native = exp( (1/|S|) · Σ_{t∈S} log s_t )
```

`compute_t_ref` is **fail-closed**: a formula referencing a compute class or
resource the peak table does not cover, or a non-positive / non-finite peak,
raises rather than scoring on bad data. Cost-model formulas are size-knob
expressions evaluated through a shared AST-whitelisted evaluator — no `eval`, no
attribute access, no arbitrary calls (`cost_model.evaluate_formula`).

`s_t ≈ 1` means the 910B Roofline is saturated. `RooflineP@τ` (the fraction of
`S` with `e ≥ τ`) is a diagnostic, not a ranking key. The illustrative 910B peaks
(HBM ≈ 1.6 TB/s, FP16 cube ≈ 256 TFLOPS, INT8 ≈ 512 TOPS) live authoritatively in
the packaged peak table.

**Device-side timing.** For `e` to be a well-posed Roofline efficiency, `t_npu`
must reflect genuine device work. The authoritative timer is host-side
`time.clock_gettime(CLOCK_MONOTONIC_RAW)`, with a backend stream `synchronize()`
bracketing each measured call (`ascendbench/perf/timing.py`):
`torch.npu.Event.elapsed_time` sees only its own stream and misses cross-stream
gaps, so the NPU event timer is kept as a **secondary cross-check** and arbitrated
against the host time (`arbitrate_timing`), never used as the primary. Each of the
`n_repeat` measured trials runs on a **fresh** per-trial input (warmup trials
excluded), so a candidate cannot cache an answer across repeats, and warmup
absorbs JIT-compile cost. A large host-vs-event drift together with a
non-default-stream signal marks the trial invalid. Rationale in
[`docs/reference/timing_arbitration.md`](reference/timing_arbitration.md).

### 4.3 Cost model is evidence-only, not a gate

The cost model exists **only** to build `t_ref`. Its per-shape agreement with an
instrumented GPU-Triton reference (`cost_model.status`) is ncu-evidenced
performance/access metadata — *not* a release blocker. A portable analytic bytes
formula is structurally unable to hit a tight per-shape tolerance on every shape:
conditional access (e.g. `HAS_Z` branches), L2 reuse, tile geometry, and the HBM
burst floor all move real traffic away from the closed-form count. Forcing
`cost_model.status` into the release gate would either pin release red on a
performance-model technicality or push toward dishonest shape-specific fitting.
So its status is *recorded as metadata and disclosed*, and the correctness gate
(§3) stands on its own.

### 4.4 Release eligibility — score-eligible ∩ oracle-aligned

The release-ready set is computed by **one** authority,
`compute_release_ready_eligible` =
`compute_score_eligible(tasks) ∩ { t : is_provenance_eligible(t.meta) }`. Both
`cli analyze` and the corpus release gate call it, so the eligible set cannot
drift between call sites.

- `compute_score_eligible` drops tasks that cannot be fairly scored — a stub
  oracle (`raise NotImplementedError`), a `PHASE-C-MANUAL` shapes file, or an
  operator-marked `release_scope.excluded` task — so ungradeable tasks do not
  depress valid submissions. The corpus still contains them; only the *scored
  denominator* subtracts them.
- The intersection with `is_provenance_eligible` adds the oracle-equivalence
  requirement.

Within that set, a task blocks release if its composite `release_status != PASS`,
and a blocked task **cannot** evade the gate by being dropped from `|S|` — it
stays in the denominator and scores ε. Two mode-aware denominator rules keep this
honest (`scoring.py`):

- An `upstream_oracle_required` task with no provider golden in self-reported mode
  is `unsupported`: out of the scored set entirely (dropped from numerator *and*
  denominator), because no fair comparison exists.
- A verified-mode task whose offline golden is missing or corrupt is `blocked`:
  it **stays** in the fixed `|S|` and scores as a failure (ε). Blocked is the
  release gate's target and must never escape the denominator.

---

## 5. Anti-gaming posture

Because the score is the thing a submitter optimizes, the harness assumes an
adversarial submitter and defends in depth. At a high level:

- **Process isolation.** Candidates run in a separate subprocess that never
  shares interpreter state or module cache with the trusted oracle, communicates
  only via safetensors, and never receives the oracle, golden outputs, the task
  directory, or the hidden-shape section. Correctness and timing are graded in
  two *independent* subprocess passes, with the first pass's process group killed
  and `/tmp` + Triton compile cache scrubbed between them, so no precomputation
  channel survives (`ascendbench.integrity.process_isolation`).
- **Static import / AST checks.** Submitted `solution.py` passes a layered static
  defense — import denylist, a positive AST node-type allowlist, call/attribute
  dotted-name lists, a runtime audit hook installed inside the isolated child, and
  a source-string denylist (`ascendbench.integrity.static_checker`) — so a
  solution cannot reach for the oracle, the filesystem, or the network.
- **Canaries exist.** The corpus includes canary tasks designed to catch a
  submission that is gaming the runner rather than computing the operator; a
  triggered canary escalates the whole submission. The detection rules are
  intentionally **not** documented here.

The fixed-denominator, no-subset-escape discipline in §4 is itself an
anti-gaming property: a submitter cannot improve a headline number by making
hard tasks disappear.

---

## 6. Roadmap (not yet implemented)

The following are **designed but not yet built**, and are listed here so they are
not mistaken for shipped behaviour:

- **Hardened verified-runner sandbox.** The shipped process-isolation model
  enforces the host-independent, security-critical properties on a dev box. The
  full hardened model — container digest pinning, seccomp/cgroup confinement, and
  NPU-device isolation on the 910B verification machine — is future work.
- **Public leaderboard and submission broker.** The two-tier
  (self-reported / verified) public leaderboard, the submission endpoint, and the
  signed-artifact broker that would arbitrate official 910B re-execution are not
  yet operational.
- **Stronger correctness anchors.** Multi-seed and property-based input sweeps,
  and a per-task disclosed strength label for the oracle anchor (shapes used,
  tolerance class, bit-exactness), are planned to raise input discriminating
  power beyond the current single-seed materialization.

---

## See also

- [`README.md`](../README.md) — quick start, project status.
- [`docs/kernel-select.md`](kernel-select.md) — task selection and the full corpus table.
- [`docs/reference/scoring_spec.md`](reference/scoring_spec.md) — machine-facing scoring detail.
- [`docs/reference/tolerance_policy.md`](reference/tolerance_policy.md) — tolerance classes.
- [`docs/reference/task_api_contract.md`](reference/task_api_contract.md) — task bundle schema.
- [`docs/reference/timing_arbitration.md`](reference/timing_arbitration.md) — device-timing methodology.
- [`docs/findings/`](findings/) — upstream bugs surfaced while building the corpus.
