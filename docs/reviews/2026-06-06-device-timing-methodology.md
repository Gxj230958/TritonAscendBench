# Device-only kernel timing: methodology study (2026-06-06)

## Problem

TAB ranks each task by Roofline efficiency `e = t_ref / t_npu`. `t_ref` is the
device-side theoretical floor (`max(W_c/pi_c, M_r/beta_r)`). For `e` to be a
well-posed efficiency, `t_npu` must be a device-side time too.

The ranking originally timed `t_npu` as whole-operator **host wall time**
(`perf_counter_ns` bracketing the call with `torch.cuda.synchronize` on both
sides). For small/launch-bound kernels this is dominated by host dispatch
overhead (~15-50us floor), so `e` collapses to ~1e-5..1e-7 with no physical
meaning.

## What was tried and why it failed

Replacing host timing with CUDA events **but synchronizing every iteration**:

```python
for _ in range(100):
    start.record(); call(); end.record(); end.synchronize()
    durs.append(start.elapsed_time(end))
```

Measured on H200: CUDA-event `t_npu == host-wall t_npu` for small kernels, even
for pure `@triton.jit` tasks (L1_18 14835->18304ns, L1_16 17011->16800ns,
L1_26 21676->21632ns; L1_28 ~32000->~54000ns).

Root cause (codex-confirmed): per-iteration `end.synchronize()` drains the
stream each iteration, so the next `start.record()` marker executes before the
CPU has finished dispatching the next kernel. The `start->end` interval then
captures the GPU-idle-waiting-for-dispatch gap, not just kernel execution. CUDA
events re-measured the same dispatch-bound quantity from the GPU clock side.

## Industry standard (three converging sources)

Triton `do_bench` source, KernelBench, and jan.ai all use **CUDA events done
correctly**, NOT a profiler as the primary measurement:

1. **Back-to-back launches, no per-iteration sync.** `do_bench` records into N
   event pairs in a tight loop and synchronizes ONCE at the end, so the GPU
   pipelines and dispatch overhead is amortized; the idle falls between event
   pairs, not inside `start[i]->end[i]`.
2. **L2 cache flush before each iteration** (`clear_cache` / `cache.zero_()`).
   jan.ai showed un-flushed timing yields SOL% > 100% (impossible); flushing
   makes it < 100%. (Same failure mode TAB's `P_CAP=5` cap guards against.)
3. **Short-lived-kernel guard.** jan.ai measured a kernel at 103.9us in Python
   vs 71.36us in Nsight Compute because the kernel finished before the CPU
   could enqueue the end event. Fix: a dummy FP32 matmul before the timed call
   occupies the pipeline so the end event is enqueued in time (+ replicate over
   data copies to widen the window). This is exactly the starve case codex
   flagged; the industry fix is the dummy matmul, not a profiler.
4. **Profiler/ncu is a REFERENCE check**, not the primary timer ("always good
   to have a point of reference"; "use them for sanity-checking").
5. **Metric is SOL%** (Speed-of-Light percentage = percent-of-peak) = TAB's
   `e`.

Triton ships `triton.testing.do_bench` (continuous launch + L2 flush +
quantiles) and `do_bench_cudagraph` (CUDA-graph capture+replay, fully removes
host dispatch). KernelBench offers cuda_event / do_bench / host_time E2E.

## KernelBench vs TAB — a key distinction

KernelBench reports **wall-clock speedup** (ref_time / kernel_time); the timing
口径 cancels in the ratio, so absolute device time is not required. TAB's `e =
t_ref / t_npu` is a **Roofline** ratio against a device-side floor, so `t_npu`
must be genuine device time. The measurement METHOD is still the same do_bench
style; TAB just cares more about the absolute number.

## Decision

Do NOT switch to a profiler. Adopt the industry-standard do_bench-style timer
for `t_npu`: continuous launches without per-iteration sync, L2 flush, a
short-lived-kernel guard, median over the run. Validate a few tasks against
`ncu` as a reference. Sub-microsecond kernels that remain measurement-floor
dominated stay `FLOOR_DOMINATED` (not given a comparable `e`).

Next: probe on the H200 daemon comparing per-sync (current) vs do_bench vs
do_bench_cudagraph on representative tasks (L1_18 pure-small, L2_13_bmm large,
L1_28 launcher-floor), confirm small-kernel `t_npu` drops to near isolated
kernel time, then change the ranking timer (codex-reviewed) and re-run all 101.

## H200 result (measured, 5-task smoke)

do_bench vs host-wall `t_npu`: small pure kernels drop 2.5-4.9x (L1_18 15-25us
-> 5-7us, L1_16 17-37us -> 5-19us), large/big-shape kernels unchanged (L1_26 P4
424753->416544ns, L2_13_bmm P4 7.14ms->7.10ms). Big shapes reach 30-80% of
Roofline peak (L1_26 P4 e=0.806, L2_13_bmm P4/P3 e=0.535/0.495, L1_18 P4
e=0.318). per-sync CUDA-event == host-wall (confirms it was a no-op). Confirms
the dispatch-overhead diagnosis and the do_bench fix.

## Adopted implementation (codex-reviewed: PASS after fallback)

- Primary timer: `triton.testing.do_bench(fn, warmup=25, rep=100,
  quantiles=[0.5,0.0,1.0])` -> (median, min, max) ms, x1e6 -> ns. median is the
  ranking statistic; min/max are diagnostic (`t_npu_min_ns`/`t_npu_max_ns`).
- L2 flush is metric-consistent: `cost_model.bytes` is per-resource HBM traffic
  (ncu_dram_bytes validated) and `compute_t_ref` divides by HBM bandwidth, so a
  cold-cache device time matches the cold-HBM `t_ref` numerator.
- Fallback (required for the all-101-must-rank bottom line): if `do_bench`
  raises, a bounded CUDA-event estimate keeps the task ranked, flagged via
  `timing_method="event_fallback"` (a degraded host-dispatch-inclusive number,
  never silently treated as do_bench). `timing_method` surfaces in the rank
  verdict so fallback tasks are visible.
- NOT `do_bench_cudagraph` for ranking: its replay has no per-run L2 flush, so
  it measures graph-warm behavior, inconsistent with the cold-HBM `t_ref`.

## Known limitation (disclosed, not fixed)

`do_bench` calls the timed closure hundreds-to-thousands of times on a single
input clone, so an in-place / state kernel accumulates mutations and a
value-dependent control-flow kernel could have a drifting per-iteration runtime.
Most kernels here are shape/control-dominated (runtime independent of input
values), so the risk is small; `t_npu_min_ns`/`t_npu_max_ns` expose any drift.
A per-iteration `reset_fn` (forking do_bench) is deferred as not worth the cost
for the affected minority.

## Sources

- Triton `do_bench` / `do_bench_cudagraph` (local `triton/testing.py`)
- KernelBench EVAL.md — https://github.com/ScalingIntelligence/KernelBench/blob/main/EVAL.md
- jan.ai, How we benchmark kernels — https://www.jan.ai/post/how-we-benchmark-kernels
- KernelBench, Stanford — https://scalingintelligence.stanford.edu/blogs/kernelbench/
