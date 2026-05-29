# Resolve-stage global diagnosis (authoritative)

> **Outcome (2026-05-29): 18 → 20 PASS.** The `primary_blocker` reducer fix
> surfaced 3 `oracle-consistent-smoke-blocked` tasks (right wrapper found,
> blocked only by the severed-import audit). For L2_13_bmm / L2_14, the leak
> was 4 import-time/setup-only vllm helpers (`vllm.envs.VLLM_BATCH_INVARIANT`,
> `mem_utils.get_max_shared_memory_bytes`, `torch_utils.is_torch_equal_or_newer`,
> `platform_utils.num_compute_units`) — none on the verified wrapper's compute
> path. A narrowly-allowlisted inert shim (verified verbatim defaults +
> loud-fail on Class-S semantic knobs) severed the leak; both then RESOLVED
> and VERIFIED PASS on GPU (bit-exact). Codex design+impl reviewed (2 REFINE
> rounds, closed). L1_19 (3rd oracle-consistent) leaks heavier class imports
> (sampling_params/v1.outputs/buffer_utils) — separate, deferred.

Date: 2026-05-29. Source: full `resolve` sweep of all 82 non-PASS tasks on
the real GPU daemon (batch `b20260529T082912-60b77e`). 18/100 PASS.

## Key correction to the prior import-wall model

The earlier global snapshot bucketed tasks by their **import wall**
(vllm/sglang). That model was wrong about the ceiling: **every** non-PASS
task blocks at the **resolve stage** (verified-upstream-entrypoint
discovery), which runs *before* the import surface even matters for many.
Import-clean ≠ ready-to-PASS.

## Authoritative resolve-stage blocker distribution (82 tasks)

| bucket | count | meaning |
|--------|-------|---------|
| `shapes-incomplete` | ~40 | entrypoint resolves but `shapes.yaml` omits a required no-default param |
| `upstream-not-invocable-here` | ~33 | candidate found, cannot invoke in env (missing-dep / relative-import packaging / triton-API drift / TypeError) |
| `oracle-mismatch` | 8 | a resolvable candidate's output != `oracle_pytorch` (wrong operator picked) |
| `smoke-fail` | 1 | L1_20_log_softmax |

## `upstream-not-invocable-here` (33) joined with import wall

- **sglang.srt (7)**: L1_18, L2_04/05/06/07/08/09 (lora + softcap). No
  sound batch shim (re-exports side-effectful Class-A) → bench-team arch
  decision. Park.
- **vllm.model_executor.layers (5)**: L1_06, L2_25, L2_26, L3_11, L3_12.
- **vllm bare (4)**: L2_11, L2_18, L3_09, L3_10.
- **FLA relative-import (8)**: L3_13/16/18/20 (`.index`), L2_17/L3_17/L3_19
  (`.chunk_delta_h` etc.), L1_28 (`..utils`). Beyond the `{index,op,utils}`
  closure currently vendored (commit 31f1c67). NOTE: L3_13/16/18/20 also
  hit a downstream shape-rank error in verify, so vendoring alone won't
  PASS them.
- **vllm.v1.worker (3)**: L1_02, L1_05, L1_27.
- **singletons (6)**: L1_23 (sampling_params), L1_29 (config), L1_31
  (`.base`), L1_38 (forward_context), L3_01 (envs).

## `shapes-incomplete` is NOT a cheap deterministic batch-unlock

`scripts/auto_shapes_filler.py` (pre-built infra that maps the resolver's
"missing param 'X'" reason to conventional `(dtype, dims, dist)` rows)
fills **0 rows** on this corpus. Every missing param is domain-specific
(`quant_k_cache`, `cu_chunk_seqlens`, `initial_state`,
`softplus_threshold`, `chunk_states`, `gateup_output`, …) and routes to
the LLM/manual path, not the ~30-idiom conventions table. So the
shapes-incomplete bucket is genuine per-task (or LLM-assisted) shapes
authoring, not infra. (auto_shapes_filler also had a `NameError`
crash — undefined `skipped` at the `not missing` branch — fixed.)

## Infra fixes landed this session

- `frozen_deps` now records the shipped pool artifacts (regex-scan of the
  promoted standalone source) instead of the noisy CLASS_A classifier
  closure (L2_22 had 600+ vllm.* symbols). Pool files carry their own
  upstream provenance headers.
- `orchestrator.py status` markers single-quoted (remote zsh EQUALS
  expansion broke bare `=== jobs ===`).
- `auto_shapes_filler.py` `NameError` fixed.

## Topology note (operational)

Daemon runs on **GPU A** (unreachable directly). dev → ssh GPU B jumphost
(gpfs + file movement only) → shared gpfs mailbox → A. Never start a
daemon on B. Liveness is probed by submitting a mailbox job. Heartbeats
are per-batch (idle = stale, not dead).

## Codex co-review conclusions (resolved)

1. **Ranking is NOT the problem.** The resolver already prefers invocable
   host wrappers (`rank_candidates`, score-sorted host-first) and only
   tries a bare jit when no wrapper exists. A ranking tweak would not
   batch-unlock passes.
2. **The real bug was the reducer** (`resolve_upstream_entrypoints.py`
   ~L1511): the headline bucket was worst-news-first across ALL tried
   candidates, so one sibling's `shapes-incomplete` masked the best-ranked
   candidate's real status. Of 47 shapes-incomplete in the cache, ~35 are
   genuine (the intended candidate fails strict bind) and ~12 are
   contamination. L2_13_bmm is contamination: `bmm_batch_invariant(a,b)`
   matched the oracle, tried first, blocked only by a dev-host vllm SIGKILL
   (would run on GPU — same `oracle-consistent-smoke-blocked` class as the
   already-PASSED L2_01). **Fixed**: added `Resolution.primary_blocker`
   surfacing the best-ranked candidate's own status (a heuristic, not a
   guarantee) + surfaced it on the resolve progress line so the loop daemon
   captures it. The headline bucket is unchanged (back-compat).
3. **FLA tasks are Triton-API drift, NOT vendoring closure.** L2_17, L3_13
   etc. fail on `_AutotunedKernel.arg_names` (triton-version drift), so
   extending the FLA pool would have been wasted effort. L1_28_pack_bitmatrix
   is a plain shapes gap, not packaging. (Lever #2 dropped.)
4. **Realistic solo ceiling this session ≈ high-20s/low-30s** (from 18),
   NOT 100. The honest highest-ROI lever is targeted per-task `shapes.yaml`
   authoring for the ~35 tasks whose top candidate is the intended wrapper
   and fails only strict bind. This is the [[project-tab-pertask-tail]]
   long tail — slow, ~1 task per design+author+verify cycle.

## DEFINITIVE CEILING (codex-confirmed, empirical) — the bare-jit launcher gap

A task PASSes Phase A' verify **iff its oracle operator has a host-wrapper
in kernel.py callable directly with tensors.** All 20 PASS resolved via
host-wrappers; **zero** via a bare `@triton.jit`. The stuck tasks are
mostly **bare jits with no dedicated host-wrapper**:
- The oracle-check runner (`_pa_oracle_check_runner.py:370`) binds by name
  and calls `fn(**kwargs)`. Under `TRITON_INTERPRET=1` a bare jit is an
  `InterpretedFunction(*args, **kwargs)` → name-binding finds nothing →
  cannot synthesize grid/output/constexprs. Codex ran L1_35's bare
  `compute_src2dst_triton_kernel`: `InterpreterError: missing 4 required
  positional arguments`.
- `JIT_LAUNCHER_TEMPLATE` (`phase_a_prime.py:1269`) is a
  `raise NotImplementedError` placeholder (anti-fake) — there is **no
  generic bare-jit auto-launcher**.

**Therefore the "include name-matched bare jits in the resolver `ordered`"
lever is a DEAD END** (codex-rejected): it converts "never tried" →
`upstream-not-invocable-here`, not PASS. Only **5** truly bare-only tasks
exist anyway; mixed modules already try their host-wrappers.

**Implication for the path to 100:** the infra levers (stub / shim /
reducer / frozen_deps) that unlock *host-wrapper-backed* operators are
largely **exhausted near ~20-30**. Beyond that, every task is a manual
tail: (a) host-wrapper-with-missing-shapes → per-task `shapes.yaml`
(with fp8 / variable-length / abstraction-mismatch deep cases),
(b) bare-jit operator → **manual per-task launcher authoring** (grid +
constexpr + output alloc — a "separate launcher-authoring project," heavier
than shapes), (c) some need bench-team golden/tolerance policy (fp8
dequant-compare, variable-vs-fixed output-length comparison). **Full
automation to 100 is not reachable via infra** — it is bounded by the
corpus's host-wrapper coverage plus a large manual per-task tail.

## Per-task shapes lane — characterization (2026-05-29, probed 8 tasks)

The `shapes-incomplete` bucket is **heavily contaminated** and the lane is
deeper than the count implies. Probed 8 (GPU-authoritative missing-param +
chosen entrypoint):

**Abstraction-mismatch (authoring shapes would encode the WRONG operator —
the resolver picked a bigger same-family wrapper, not the oracle's simple
op). Do NOT author shapes; needs candidate-selection work or is genuinely
out of scope:**
- L1_15_act_and_mul → picked `act_and_mul_triton` (MoE-fused), oracle is
  simple act_and_mul.
- L1_32 / L1_33 (create_*_kv_indices) → picked `fused_qk_rope_reshape_and_cache`.
- L1_35_compute_src2dst → picked `cutlass_w4_run_moe_ep_preproess`.
- L1_36_compute_seg_indptr → picked `moe_ep_deepgemm_preprocess`.

**Name-matching (right operator) but deep per-task blocker — authorable
but each is a real investigation:**
- L2_37_silu_and_mul_post_quant: shapes fix is KNOWN-CORRECT (output
  [E,T,D] fp8 + output_scale [E,T,D//quant_group_size] fp32, `dist: zeros`;
  confirmed `BLOCK_N = quant_group_size` so D//8, not the stale `//128`
  docstring). It binds the kernel and advances shapes-incomplete →
  oracle-mismatch, but the fp8 quantized `output` mismatches (max_abs=64,
  max_rel=0.5): a reference-vs-kernel fp8 **scale-rounding fidelity** wall.
  Needs a tolerance/golden-policy decision (fp8 dequant-compare?), NOT a
  shapes edit. Reverted the shapes change pending that policy call.
- L2_31_moe_align_block_size_stage2: 3 candidate_inplace int32 output
  buffers; `sorted_token_ids` would init to sentinel `full:B*D`, size
  `B*D + num_experts*(block_size-1)`. BUT the oracle returns
  VARIABLE-length outputs (`torch.cat` of tight per-expert runs = `total`
  elements), while the kernel writes the FIXED `max_num_tokens_padded`
  buffer. Fixed-vs-variable output-length convention mismatch → needs a
  length-aware comparison policy (compare kernel[:num_tokens_post_pad]),
  not just shapes. NOT a clean one-shot.
- L1_40_chunk_cumsum_fwd: `_chunk_cumsum_fwd` matches oracle but needs a
  varlen `cu_chunk_seqlens` index (oracle doesn't read it — kernel-only
  varlen param); needs the chunk-boundary convention.

**Implication:** the genuinely shapes-authorable+PASSable subset is well
under the 32 `shapes-incomplete` count — many are candidate-selection
mismatches or have fp8/varlen/output-buffer subtleties. Each clean one is
~1 multi-step investigation (size formulas, sentinel inits, conventions) +
GPU verify. This sharpens codex's ~high-20s/low-30s ceiling DOWNWARD for
the pure-shapes lane.

## Note on dev-vs-GPU drift (resolve runs against RAW upstream)

The resolve stage imports the raw kernel.py (real vllm/sglang), so dev-side
resolve hits `num_compute_units` NotImplementedError / vllm cold-import
SIGKILL that DO NOT occur on the GPU. Several already-PASSED tasks (L2_01,
L3_21/22/23, L1_03/11, L2_12) were `upstream-not-invocable` /
`oracle-consistent-smoke-blocked` / `oracle-mismatch` in the dev cache yet
PASSED on GPU. => Only the GPU daemon resolve is authoritative; the dev
cache is a stale, drift-contaminated preview. A GPU re-resolve with the
`primary_blocker` fix is the authoritative triage input.
