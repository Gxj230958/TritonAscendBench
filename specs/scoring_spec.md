# Scoring Specification

> **v3.2 ERRATA — AscendC removed.** This is a GPU-triton → NPU-triton *migration* benchmark; there is no AscendC native baseline, no `ascendc_baseline.yaml`, no `in_s_native`/`oracle.ascendc`. Correctness is the scored axis; performance is raw NPU latency (deferred to 910B). AscendC-specific text below is **superseded** and kept only as historical context.


> v0 — per v4.13 §5 + R12-P1-1/5 + R13-P1-1 + R16-P1 + R18-S5

## Primary Metric: `score_native_only(p)`

For each task `t` in `S_native` (the set of tasks with AscendC native baseline):

$$s_t = \begin{cases}
\max\left(\epsilon,\ \min\left(5,\ \dfrac{t_{ascendc}(t)}{t_{npu\_triton}(t)}\right)\right) & \text{if all 4 statuses = ok and no fallback} \\
\epsilon = 0.01 & \text{otherwise}
\end{cases}$$

The 4 statuses required:
- `correctness_status == ok`
- `runtime_status == ok`
- `performance_status == ok`
- `no fallback detected` (from §5b.4 artifact provenance check)

### Headline indicators

- `FastP_native@1.0` = fraction of `S_native` tasks where candidate beats AscendC native (speedup ≥ 1.0)
- `FastP_native@1.5` = fraction within 1.5× of AscendC native
- `capped_geomean_native` = $\exp\left(\dfrac{1}{|S_{native}|}\sum_t \log(s_t)\right)$

`p = ∞` (correctness-only) is **NOT a headline indicator** — too easy to game.

## Secondary Metrics (diagnostic)

- `score_all` — denominator `|S_scored|` (all 100 scored tasks); $t_{denom}$ is AscendC if available, else torch eager.
- `score_fallback_only` — for `S_fallback = S_scored \ S_native`, denominator is torch eager.

**Warning**: `score_all` mixes denominators (AscendC vs torch eager) and is biased toward tasks without AscendC (where torch eager is slow). Use only for diagnostics, not ranking.

## Status Fields (per v4.13)

```
correctness_status ∈ {ok, fail, error}
runtime_status     ∈ {ok, timeout, oom, compile_fail, invalid_timing}
performance_status ∈ {ok, suspicious, quarantined}
performance_flags  ∈ list of {low_performance, invalid_stream, measured_outputs_drift,
                              suspected_buffer_reuse, thermal_throttle, high_variance, ...}
```

Note: `performance_flags` is a list, NOT in the enum, per R12-P1-1 / R13-P1-1.

## Ranking Eligibility

Per R12-P1-5, a submission qualifies for the native track ranking only if:

$$\text{native\_coverage} = \frac{|\{t \in S_{native}: \text{correctness\_status}_t = ok\}|}{|S_{native}|} \ge 0.70$$

Otherwise → submission lands on `self-reported` board only, not `verified` ranking.

Similarly for all-track ranking: `all_track_coverage = correct_in_S_scored / |S_scored| >= 0.70`.

## Sanity Bounds (per §5b.6 + R11-P1-6 / R16-P1)

| Condition | Action |
|-----------|--------|
| `runtime_status = invalid_timing` (t_npu ≤ 0 / NaN / Inf) | hard fail |
| Profiler captures 0 NPU kernel events | hard fail (likely CPU fallback / B5 / F4) |
| `hardware_utilization > 100%` | hard fail (physically impossible) |
| speedup > 10× vs AscendC native | quarantine + require profiler trace + memory lower bound proof + output dependency proof |
| speedup < 0.1× vs AscendC native | **not a hard fail** — flag `performance_flags.append("low_performance")` (per R11-P1-6) |
| host_ms vs event_ms diff > 10% | check stream list; if non-runner stream → `runtime_status=invalid_timing` + flag `invalid_stream` (R12-P1-2) |
| Measured sampled output ≠ golden_meas_outputs | `performance_status=quarantined` + flag `measured_outputs_drift` (R16-P1) |

## Hardware Utilization (auxiliary diagnostic)

$$\text{hardware\_utilization} = \max\left(\frac{\text{achieved}_{HBM}}{1600\,\text{GB/s}} \times 100,\ \frac{\text{achieved}_{TFLOPS}}{256} \times 100\right)\,\%$$

(910B specs: HBM 1.6 TB/s, FP16 peak 256 TFLOPS, INT8 peak 512 TOPS)

## Result JSON Schema (per task per run)

```json
{
  "task_id": "L1_12_rms_norm_kernel",
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

  "per_shape": {
    "small":  {"median_ms": 0.042, "p99_ms": 0.051, "cv": 0.03, "trials": 100},
    "medium": {"median_ms": 0.318, "p99_ms": 0.389, "cv": 0.04, "trials": 100},
    "large":  {"median_ms": 2.140, "p99_ms": 2.580, "cv": 0.05, "trials": 100}
  },
  "runs": [
    {"run_id": 1, "summary": {...}, "temp_c": 52, "freq_mhz": 1850},
    {"run_id": 2, "summary": {...}, "temp_c": 53, "freq_mhz": 1850},
    {"run_id": 3, "summary": {...}, "temp_c": 54, "freq_mhz": 1850}
  ],

  "adversarial": {
    "extreme_small":    {"correctness": true, "ms": 0.012},
    "long_seq":         {"correctness": true, "ms": 12.5},
    "non_align_hidden": {"correctness": true, "ms": 0.345}
  },

  "denominator_source": "ascendc_native",
  "speedup_vs_ascendc": 0.82,
  "s_t_value": 0.82,
  "score_native_only_p1.0_pass": false,
  "score_native_only_p1.5_pass": true,
  "hardware_utilization_pct": 67.3,

  "fallback_detected": false,
  "reward_hack_flags": []
}
```
