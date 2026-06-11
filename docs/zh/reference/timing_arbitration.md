# 计时仲裁规范

> English: [../../reference/timing_arbitration.md](../../reference/timing_arbitration.md)

## 权威计时来源

**主要来源**：`CLOCK_MONOTONIC_RAW` 主机时间（-2）。

不能单独使用 `torch.npu.Event.elapsed_time` —— event 仅对记录流（recording stream）计时，会遗漏非默认流上的工作。

## 计时协议（位于受信任的 shim 中，候选实现无法看到）

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

## 仲裁规则

| 条件 | 裁决 |
|-----------|---------|
| `\|host_ms - event_ms\| / host_ms ≤ 10%` | 以 `host_ms` 为权威；event 与之吻合 |
| `\|host_ms - event_ms\| / host_ms > 10%` | 检查 profiler 的流列表；若检测到非 runner 流 → `runtime_status=invalid_timing` + `performance_flags.append("invalid_stream")` |
| `host_ms ≤ 0` 或 `host_ms = NaN` 或 `host_ms = ∞` | `runtime_status=invalid_timing` |
| `host_ms < 1 μs` | 很可能没有真实执行 → `performance_status=quarantined` |

## 沙箱禁止导入项

候选实现沙箱中必须禁止导入：
- `torch.npu.Event`、`torch.cuda.Event`、`torch_npu.npu.Event`
- `time.perf_counter`、`time.monotonic`、`time.process_time`、`time.clock_gettime_ns`
- `torch.profiler`、`torch_npu.profiler`
- `LD_PRELOAD` 设置

由 import hook + AST 静态检查强制执行。
