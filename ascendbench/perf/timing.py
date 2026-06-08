"""timing.py — host-side authoritative timing using CLOCK_MONOTONIC_RAW.

Per `specs/timing_arbitration.md` (R12-P0-2 + R16-P1):

- `torch.npu.Event.elapsed_time` measures only the current stream and
  misses cross-stream gaps. We use `time.clock_gettime(CLOCK_MONOTONIC_RAW)`
  on the host as the authoritative timer (R12-P0-2).
- An NPU event timer is recorded as a secondary cross-check signal; the
  host/event drift arbitration lives in `arbitrate_timing` below and is
  consumed by the trusted parent (`process_isolation`).
- W1.3: measured trials each run a **fresh** inputs dict produced by a
  per-trial input factory (R16-P1) — a fixed-args repeat would let a
  candidate cache the answer across trials. The factory closes over the
  runner's `meas_seed + t` so warmup and measured inputs always differ.

R6-1: the per-backend cross-check seam is `backends/npu.NpuBackend.
event_timer()` — an `EventTimer` *context manager* whose NPU
`Event.record()` calls bracket the SAME single measured call host time
sees. `time_one_run` accepts an `event_timer_factory` that yields one
fresh `EventTimer` per call; it NEVER re-runs `fn` to take a separate
event sample (that would execute each measured input twice). On the dev
machine the factory yields `None` and timing degrades to host-only.
"""
from __future__ import annotations

import contextlib
import math
import time
from collections.abc import Callable
from dataclasses import dataclass, field

DRIFT_THRESHOLD = 0.10  # per specs/timing_arbitration.md (single source of truth)
QUARANTINE_NS = 1_000   # host_ns < 1 µs ⇒ likely no real execution


@dataclass
class TimingSample:
    """One measured trial: host time + optional NPU-event cross-check."""

    host_ns: int
    event_ns: int | None = None
    drift_ratio: float | None = None
    drift_status: str = "single_source"   # "ok" | "drift_detected" | "single_source"


@dataclass
class TimingArbitration:
    """Verdict of the timing-arbitration rules (specs/timing_arbitration.md).

    Carries the status bits the trusted parent stamps onto the eval result;
    W1.3 maps them onto the *existing* result fields — there is no new
    ``timing_status`` field:

      * ``runtime_status``      -> "invalid_timing" on a hard fault
      * ``performance_status``  -> "quarantined" on a sub-µs run
      * ``performance_flags``   -> "invalid_stream" / drift detail items
    """

    runtime_status: str | None = None        # "invalid_timing" when set
    performance_status: str | None = None     # "quarantined" when set
    performance_flags: list[str] = field(default_factory=list)
    drift_ratio: float | None = None


def _host_ns() -> int:
    return time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW)


def time_one_run(
    fn,
    *args,
    sync: Callable | None = None,
    event_timer_factory: Callable | None = None,
    **kwargs,
) -> TimingSample:
    """Measure `fn(*args, **kwargs)` once.

    `sync` is the per-backend stream synchronizer
    (e.g. `torch.cuda.synchronize` or `torch.npu.synchronize`).

    R6-1: `event_timer_factory`, when supplied by the backend, is called
    as ``event_timer_factory() -> EventTimer | None`` and yields ONE fresh
    NPU event timer. The timer is a context manager whose `Event.record()`
    calls bracket the SAME single `fn` invocation host time measures —
    `fn` is NEVER called a second time to take a separate event sample
    (that executed each measured input twice and reopened the stateful-
    candidate hole). On the dev machine the factory yields `None`; the
    sample is host-only (`drift_status="single_source"`).
    """
    timer = event_timer_factory() if event_timer_factory is not None else None
    bracket = timer if timer is not None else contextlib.nullcontext()
    if sync:
        sync()
    t0 = _host_ns()
    # The post sync runs in `finally` so an exception from `fn` still drains the
    # device; the event timer records only and is read after that sync.
    try:
        with bracket:
            fn(*args, **kwargs)
    finally:
        if sync:
            sync()
    t1 = _host_ns()
    host_ns = t1 - t0

    event_ns: int | None = timer.elapsed_ns() if timer is not None else None
    return _classify_sample(host_ns, event_ns)


def _classify_sample(host_ns: int, event_ns: int | None) -> TimingSample:
    """Build a `TimingSample`, computing the host/event drift status."""
    if event_ns is None:
        return TimingSample(host_ns=host_ns, event_ns=None,
                            drift_ratio=None, drift_status="single_source")
    if host_ns <= 0:
        return TimingSample(host_ns=host_ns, event_ns=event_ns,
                            drift_ratio=None, drift_status="drift_detected")
    drift = abs(host_ns - event_ns) / host_ns
    status = "ok" if drift <= DRIFT_THRESHOLD else "drift_detected"
    return TimingSample(host_ns=host_ns, event_ns=event_ns,
                        drift_ratio=drift, drift_status=status)


def measure(
    input_factory: Callable[[int], object],
    run: Callable[[object], object],
    *,
    n_warmup: int = 5,
    n_repeat: int = 30,
    sync: Callable | None = None,
    event_timer_factory: Callable | None = None,
) -> list[TimingSample]:
    """Time `run` over `n_repeat` trials, each on a *fresh* input.

    W1.3 reshape (R16-P1): the old fixed-`*args` repeat is replaced by a
    per-trial input factory. `input_factory(t)` returns the inputs object
    for trial ``t``; warmup uses ``t`` in ``[-n_warmup, -1]`` and measured
    trials use ``t`` in ``[0, n_repeat)`` so warmup and measured inputs
    never coincide. The caller (the trusted shim / `process_isolation`)
    owns the seed schedule — here we only consume the factory.

    Returns one `TimingSample` per measured trial (host-authoritative,
    with the optional NPU-event cross-check folded in).
    """
    # Phase a: warm-up — not timed. Distinct trial indices so warmup
    # inputs differ from every measured input.
    for w in range(n_warmup):
        if sync:
            sync()
        run(input_factory(-(w + 1)))
    if sync:
        sync()

    # Phase b: measured — fresh input per trial.
    samples: list[TimingSample] = []
    for t in range(n_repeat):
        inp = input_factory(t)
        samples.append(
            time_one_run(run, inp, sync=sync,
                        event_timer_factory=event_timer_factory)
        )
    return samples


def arbitrate_timing(
    durations_ns: list[float],
    *,
    event_ns: list[float] | None = None,
    stream_nondefault: bool | None = None,
) -> TimingArbitration:
    """Apply the `specs/timing_arbitration.md` rules to a trial sequence.

    Consumed by the trusted parent (`process_isolation` / `eval`) over the
    `durations` the timing-pass child sent up. `durations_ns` is the
    host-`CLOCK_MONOTONIC_RAW` series (authoritative). `event_ns`, when the
    NPU backend supplied it, is the per-trial event-timer cross-check;
    `stream_nondefault` is the profiler's non-default-stream verdict.

    Rules (verbatim from the spec table):

      * any ``host_ns`` non-finite / ≤ 0                  -> invalid_timing
      * host/event drift > 10% AND a non-default stream   -> invalid_timing
        was detected by the profiler                         + "invalid_stream"
      * median ``host_ns`` < 1 µs                         -> quarantined
    """
    out = TimingArbitration()

    if not durations_ns:
        out.runtime_status = "invalid_timing"
        out.performance_flags.append("no_timing_samples")
        return out

    # Rule: any non-finite / non-positive host duration is a hard fault.
    for d in durations_ns:
        if d is None or not math.isfinite(d) or d <= 0:
            out.runtime_status = "invalid_timing"
            out.performance_flags.append("nonfinite_host_ns")
            return out

    # Rule: host/event drift cross-check. >10% drift alone is a soft
    # signal; combined with a profiler-detected non-default stream it
    # invalidates the timing (the candidate hid work on another stream).
    if event_ns:
        n = min(len(durations_ns), len(event_ns))
        drifts = []
        for i in range(n):
            ev = event_ns[i]
            if ev is None or not math.isfinite(ev) or durations_ns[i] <= 0:
                continue
            drifts.append(abs(durations_ns[i] - ev) / durations_ns[i])
        if drifts:
            out.drift_ratio = sorted(drifts)[len(drifts) // 2]
            if out.drift_ratio > DRIFT_THRESHOLD:
                if stream_nondefault:
                    out.runtime_status = "invalid_timing"
                    out.performance_flags.append("invalid_stream")
                else:
                    out.performance_flags.append("host_event_drift")

    # Rule: a sub-µs median means there was likely no real execution.
    ordered = sorted(durations_ns)
    median = ordered[len(ordered) // 2]
    if median < QUARANTINE_NS:
        out.performance_status = "quarantined"

    return out
