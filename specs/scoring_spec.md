# Scoring Specification

> Machine-facing detail of the scoring model. The authoritative model is `docs/DESIGN.md §9`; this spec must not diverge from it.

This is a GPU-Triton → 910B NPU-Triton **migration** benchmark with a **two-objective** score: correctness coverage is the **ranking eligibility gate**; 910B Roofline-normalized performance efficiency is the **ranked metric**. Both use the fixed score-eligible set `S` (single source: `ascendbench/release_gate/eligibility.py`); `|S|` is fixed and never reduced (no subset escape).

## Objective 1 — Correctness Gate

Per task `t ∈ S`, each shape is compared via `tolerance.compare(candidate, golden, class)`; results aggregate into the status group (`correctness_status` / `runtime_status` / `performance_status` + `fallback_detected` + `performance_flags`). `fallback_detected` is normalized to a correctness failure before scoring, so the gate predicate is explicitly `correctness_status == ok ∧ ¬fallback_detected`.

$$\text{native\_coverage} = \frac{\big|\{t \in S:\ \text{correctness\_status}_t = \textsf{ok}\ \wedge\ \neg\,\text{fallback}_t\}\big|}{|S|},\qquad \text{ranking eligibility}:\ \text{native\_coverage} \ge 0.70$$

No ε on coverage; missing/failed tasks contribute 0 to the numerator but are **not** removed from `|S|`. Below 0.70 → `self-reported` board only, not `verified` ranking.

## Objective 2 — Roofline-Normalized Performance (ranked metric)

910B and the source GPU differ in peak capability, so absolute latency is not cross-hardware comparable; performance is normalized against the 910B Roofline. For task `t`, shape `s`:

- `W_c(t,s)` = FLOPs of compute class `c`, `M_r(t,s)` = bytes of data resource `r` (HBM / inter-device comm) — from the per-task cost model, cross-validated against an instrumented GPU-Triton reference run (fail-closed; `task_api_contract.md §cost_model`);
- `π_c` / `β_r` = 910B peak compute / bandwidth for that class / resource (`infra/` peak table);
- **Roofline ideal latency** (standard Roofline lower bound; assumes ideal resource overlap → tightest theoretical floor; fully serialized non-overlap would instead be the sum):

$$t_{ref}(t,s) = \max\!\Big(\max_c \tfrac{W_c}{\pi_c},\ \max_r \tfrac{M_r}{\beta_r}\Big)$$

- measured NPU latency `t_npu(t,s)` (host-authoritative timing; `timing_arbitration.md`);
- per-shape efficiency `e(t,s) = min(P_CAP, t_ref / t_npu)`, theoretically `∈ (0,1]` (`t_ref` is a lower bound); `> 1` arises only from cost-model underestimate / timing noise and is clamped by `P_CAP = 5.0` and flagged for review.

$$s_t = \begin{cases}\big(\textstyle\prod_s e(t,s)\big)^{1/n_t} & \text{all three statuses ok and no fallback} \\ \epsilon = 0.01 & \text{otherwise}\end{cases}\qquad \text{capped\_geomean\_native} = \exp\!\Big(\tfrac{1}{|S|}\textstyle\sum_{t \in S}\log s_t\Big)$$

The three statuses: `correctness_status == ok`, `runtime_status == ok`, `performance_status == ok`; plus `fallback_detected == false`. `baseline_ns = t_ref`, `measured_ns = t_npu`. ε-fill and the fixed `|S|` denominator apply to the performance metric exactly as to the gate — non-ok / fallback tasks score ε and are not removed from `|S|` (no subset escape).

### Headline indicators

- `native_coverage` — the ranking eligibility gate (correctness).
- `capped_geomean_native` — the ranked metric (Roofline-efficiency geomean); `s_t ≈ 1` means the 910B Roofline is saturated.
- `RooflineP@τ` (diagnostic only) = fraction of `S` with `e ≥ τ` (e.g. τ ∈ {0.5, 0.8}); not a ranking key.

Correctness-only (`p = ∞`) is **not** a headline indicator — it is the gate, not the rank.

## Status Fields

```
correctness_status ∈ {ok, fail, error}
runtime_status     ∈ {ok, timeout, oom, compile_fail, invalid_timing}
performance_status ∈ {ok, suspicious, quarantined}
performance_flags  ∈ list of {low_performance, roofline_exceeded, invalid_stream,
                              measured_outputs_drift, suspected_buffer_reuse,
                              thermal_throttle, high_variance, ...}
fallback_detected  ∈ bool   # normalized to a correctness failure before scoring
```

`performance_flags` is a list, not part of the enum.

## Release Gate (S6)

The release-eligible set is the C2-aligned eligible set (`compute_release_ready_eligible` = `compute_score_eligible` ∩ `is_provenance_eligible`, i.e. `oracle_kernel_equivalence ∈ {VERIFIED, CEILING_ACCEPTED}`). Over that set, a task is release-eligible only if `release_status == PASS`, where

```
release_status == PASS  ⟺  phase_a_prime.status == PASS
```

(S2 reproducible severance freeze). `cost_model.status` is ncu-evidenced performance/access metadata (conditional access, L2 reuse, tile geometry, HBM burst floor make a portable analytic bytes formula structurally unable to hit per-shape tol everywhere) — it is **not** a release blocker. Any C2-aligned task with `release_status != PASS` blocks benchmark release (`validate_tasks.py --release-gate`). BLOCKED is hard and cannot be evaded by removing the task from `|S|`. Tasks outside the C2-aligned set (e.g. `oracle_kernel_equivalence == BLOCKED_UPSTREAM_BUG`) are excluded from the released corpus.

## Sanity Bounds

| Condition | Action |
|-----------|--------|
| `runtime_status = invalid_timing` (t_npu ≤ 0 / NaN / Inf) | hard fail |
| Profiler captures 0 NPU kernel events | hard fail (likely CPU fallback / empty kernel) |
| `hardware_utilization > 100%` | physically impossible → same handling as `e > 1`: clamp + flag `roofline_exceeded` for mandatory review (cost-model / peak-table / measurement artifact), not a silent hard fail |
| `e > 1` (t_npu < t_ref) | cost-model underestimate / noise → clamp at `P_CAP`, flag `roofline_exceeded` for review |
| `e < 0.1` (far below Roofline) | **not** a hard fail — flag `low_performance` |
| host vs event timing drift over threshold | per `timing_arbitration.md`: `runtime_status=invalid_timing` + flag `invalid_stream` |
| Measured sampled output ≠ golden_meas_outputs | `performance_status=quarantined` + flag `measured_outputs_drift` |

## Hardware Utilization (auxiliary diagnostic)

$$\text{hardware\_utilization} = \max\!\Big(\frac{\text{achieved}_{HBM}}{\beta_{HBM}} \times 100,\ \frac{\text{achieved}_{FLOPS}}{\pi_{class}} \times 100\Big)\,\%$$

910B peaks (`π_class`, `β`) are sourced from the `infra/` peak table (illustratively HBM ≈ 1.6 TB/s, FP16 cube ≈ 256 TFLOPS, INT8 ≈ 512 TOPS — authoritative values live in `infra/`). This is the empirical counterpart of the analytic `t_ref`; diagnostic, not a ranking key.

## Result JSON Schema (per task per run)

```json
{
  "task_id": "L1_12_rms_norm",
  "task_bundle_sha": "<frozen sha>",
  "solution_sha": "<submitter sha>",
  "runner_hash": "tab-0.1.0+<git sha>",
  "container_digest": "sha256:...",
  "hardware_id": "910B-A2-card-0",
  "software_hash": {"cann": "8.5.0", "torch_npu": "2.8.0", "triton_ascend": "3.2.0"},
  "version_matrix_sha": "<from infra/version_matrix.toml>",
  "commit_sha_repo": "<bench repo>",
  "commit_sha_upstream": {"vllm": "...", "sglang": "..."},

  "correctness_status": "ok",
  "runtime_status": "ok",
  "performance_status": "ok",
  "performance_flags": [],
  "fallback_detected": false,

  "per_shape": {
    "small":  {"median_ms": 0.042, "p99_ms": 0.051, "cv": 0.03, "trials": 100,
               "t_ref_ns": 38000,   "roofline_efficiency": 0.90},
    "medium": {"median_ms": 0.318, "p99_ms": 0.389, "cv": 0.04, "trials": 100,
               "t_ref_ns": 290000,  "roofline_efficiency": 0.91},
    "large":  {"median_ms": 2.140, "p99_ms": 2.580, "cv": 0.05, "trials": 100,
               "t_ref_ns": 1950000, "roofline_efficiency": 0.91}
  },
  "runs": [
    {"run_id": 1, "summary": {}, "temp_c": 52, "freq_mhz": 1850},
    {"run_id": 2, "summary": {}, "temp_c": 53, "freq_mhz": 1850},
    {"run_id": 3, "summary": {}, "temp_c": 54, "freq_mhz": 1850}
  ],

  "adversarial": {
    "extreme_small":    {"correctness": true, "ms": 0.012},
    "long_seq":         {"correctness": true, "ms": 12.5},
    "non_align_hidden": {"correctness": true, "ms": 0.345}
  },

  "denominator_source": "roofline_ideal",
  "baseline_ns": 290000,
  "measured_ns": 318000,
  "s_t_value": 0.91,
  "cost_model_status": "PASS",
  "release_status": "PASS",
  "hardware_utilization_pct": 91.0,

  "reward_hack_flags": []
}
```
