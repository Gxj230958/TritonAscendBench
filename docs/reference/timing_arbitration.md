# Timing Arbitration Spec


## Authoritative Timing Source

**Primary**: `CLOCK_MONOTONIC_RAW` host time (-2).

NOT `torch.npu.Event.elapsed_time` alone — event only times the recording stream, missing non-default stream work.

## Timing Protocol (in trusted shim, candidate cannot see)

```python
torch.npu.synchronize()  # full sync
host_t0 = time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW)
event_start = torch.npu.Event(enable_timing=True)
event_start.record()  # current stream

# Phase a: warm-up with warm_inputs (NOT timed)
warm_inputs = gen_inputs(shape_config, seed=warm_seed)
for _ in range(10):
    solution.npu_triton_impl(warm_inputs)
torch.npu.synchronize()

# Phase b: measured timing with fresh meas_inputs per trial
torch.npu.synchronize()
host_t0 = time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW)
event_start = torch.npu.Event(enable_timing=True)
event_start.record()

measured_outputs = []
sampled_indices = runner_secret_rng.choice(100, size=10, replace=False)  # secret RNG
for t in range(100):
    meas_inputs_t = gen_inputs(shape_config, seed=meas_seed + t)  # fresh per trial
    out = solution.npu_triton_impl(meas_inputs_t)
    if t in sampled_indices:
        measured_outputs.append((t, out))  # for correctness verification

event_end = torch.npu.Event(enable_timing=True)
event_end.record()
torch.npu.synchronize()  # device-wide sync — capture all streams

host_t1 = time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW)
host_ms = (host_t1 - host_t0) / 1e6
event_ms = event_start.elapsed_time(event_end)  # start → end (receiver = earlier event)

# Authoritative: host_ms (covers all streams)
# Cross-check: event_ms
```

## Arbitration Rules

| Condition | Verdict |
|-----------|---------|
| `\|host_ms - event_ms\| / host_ms ≤ 10%` | use `host_ms` as authoritative; event matches |
| `\|host_ms - event_ms\| / host_ms > 10%` | check profiler stream list; if non-runner stream detected → `runtime_status=invalid_timing` + `performance_flags.append("invalid_stream")` |
| `host_ms ≤ 0` or `host_ms = NaN` or `host_ms = ∞` | `runtime_status=invalid_timing` |
| `host_ms < 1 μs` | likely no real execution → `performance_status=quarantined` |

## Sandbox Forbidden Imports

Candidate sandbox must NOT import:
- `torch.npu.Event`, `torch.cuda.Event`, `torch_npu.npu.Event`
- `time.perf_counter`, `time.monotonic`, `time.process_time`, `time.clock_gettime_ns`
- `torch.profiler`, `torch_npu.profiler`
- `LD_PRELOAD` setup

Enforced by the import hook + AST static check.
