"""timing.py — host-side authoritative timing using CLOCK_MONOTONIC_RAW.

Per `specs/timing_arbitration.md` (R12-P0-2):
- `torch.npu.Event.elapsed_time` measures only the current stream and
  misses cross-stream gaps. We use `time.clock_gettime(CLOCK_MONOTONIC_RAW)`
  on the host as the authoritative timer.
- An NPU event timer is still recorded as a secondary signal; if it
  diverges from host time by > 5%, the result is flagged
  `timing_status: drift_detected` and excluded from ranking.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import torch

DRIFT_THRESHOLD = 0.05


@dataclass
class TimingSample:
    host_ns: int
    event_ns: int | None
    drift_ratio: float | None
    drift_status: str          # "ok" | "drift_detected" | "single_source"


def _host_ns() -> int:
    return time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW)


def time_one_run(
    fn,
    *args,
    sync: callable | None = None,
    use_event: bool = False,
    **kwargs,
) -> TimingSample:
    """Measure `fn(*args, **kwargs)` once.

    `sync` is the per-backend stream synchronizer
    (e.g. `torch.cuda.synchronize` or `torch.npu.synchronize`).
    """
    if use_event:
        # Optional cross-check; left as a hook for backends/npu.py to fill.
        raise NotImplementedError("event timing wiring is per-backend")

    if sync:
        sync()
    t0 = _host_ns()
    fn(*args, **kwargs)
    if sync:
        sync()
    t1 = _host_ns()
    return TimingSample(
        host_ns=t1 - t0,
        event_ns=None,
        drift_ratio=None,
        drift_status="single_source",
    )


def measure(
    fn,
    *args,
    n_warmup: int = 5,
    n_repeat: int = 30,
    sync: callable | None = None,
    **kwargs,
) -> list[int]:
    """Run `fn` `n_repeat` times after warmup; return list of host_ns durations."""
    for _ in range(n_warmup):
        if sync:
            sync()
        fn(*args, **kwargs)
    if sync:
        sync()

    durations: list[int] = []
    for _ in range(n_repeat):
        sample = time_one_run(fn, *args, sync=sync, **kwargs)
        durations.append(sample.host_ns)
    return durations
