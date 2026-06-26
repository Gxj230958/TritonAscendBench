"""NPU backend — Atlas 910B via torch_npu + triton-ascend."""
from __future__ import annotations

# Stdlib imports used by the post-audit-hook graded path (`_profile_npu`) are
# imported HERE at module load so they are cached in sys.modules BEFORE the
# Layer E audit hook installs: a fresh pure-Python stdlib import inside the
# graded path would emit an `exec` audit event the hook blocks. (torch_npu's
# profiler submodule is pre-warmed separately in `ensure_initialized`, since it
# cannot be imported on the dev box.)
import os as _os
import tempfile
from collections.abc import Callable

import torch

try:
    import torch_npu  # noqa: F401 — feature-detect torch_npu availability.
    _NPU_AVAILABLE = True
except ImportError:
    _NPU_AVAILABLE = False


class EventTimer:
    """: an NPU event timer that BRACKETS one externally-made call.

    `docs/reference/timing_arbitration.md` records the NPU `Event` start/end around
    the *same* measured invocation that host time and the profiler see —
    it never re-runs the candidate to take a separate event sample. This
    context manager makes that contract explicit:

        timer = backend.event_timer()          # None on the dev machine
        with timer:                            # records the start event
            out = fn(inp)                      # the ONE measured call
        event_ns = timer.elapsed_ns()          # reads start.elapsed_time(end)

    `__enter__`/`__exit__` only RECORD the start/end events on the current
    stream — they do NOT synchronize. The trusted-shim caller already drains
    the device before h0 and again before reading, so an in-bracket sync would
    only add a redundant synchronize to host_ns and bias small-kernel latency.
    The candidate `fn` is invoked exactly once — by the caller, inside the
    `with` block — so a stateful candidate cannot be handed a second, untimed
    call ( / re-closes the  stateful-candidate hole).

    On the dev machine there is no NPU; `NpuBackend.event_timer()` returns
    `None` and the caller degrades to host-only timing.
    """

    def __init__(self) -> None:
        self._ev_start = torch.npu.Event(enable_timing=True)
        self._ev_end = torch.npu.Event(enable_timing=True)
        self._ended = False
        self._elapsed_ns: int | None = None

    def __enter__(self) -> EventTimer:
        self._ev_start.record()  # current stream
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        # Record the end event only on a clean exit; an exception leaves the
        # bracket incomplete and elapsed_ns() stays None.
        if exc_type is None:
            self._ev_end.record()
            self._ended = True
        return False  # never suppress an exception from the measured call

    def elapsed_ns(self) -> int | None:
        """Event-measured nanoseconds for the bracketed call, or ``None`` when
        the bracket did not complete cleanly.

        CONTRACT: the caller must device-sync between ``__exit__`` and this call
        (the trusted shim's post-bracket synchronize) so both events are
        complete. PyTorch / torch_npu API: ``start.elapsed_time(end)`` returns
        the elapsed time FROM the start event TO the end event — the receiver
        must be the earlier (start) event, the argument the later (end) event;
        the reversed form yields a wrong / negative duration on a real 910B.
        Result is milliseconds; convert to ns.
        """
        if self._ended and self._elapsed_ns is None:
            self._elapsed_ns = int(
                self._ev_start.elapsed_time(self._ev_end) * 1e6)
        return self._elapsed_ns


class NpuBackend:
    name = "npu"

    @property
    def device(self) -> torch.device:
        if not _NPU_AVAILABLE:
            raise RuntimeError("torch_npu not installed; cannot use NPU backend")
        return torch.device("npu:0")

    def synchronize(self) -> None:
        if _NPU_AVAILABLE:
            torch.npu.synchronize()

    def needs_compile_warmup(self) -> bool:
        """True when a real NPU is present: a triton-ascend kernel JIT-compiles
        + loads on first call, which must happen before the Layer E hook. False
        on the dev box (no device, no JIT) — preserving the exactly-one-graded-
        call R5/R6 invariant there."""
        return _NPU_AVAILABLE

    def is_available(self) -> bool:
        """True iff torch_npu is importable on this host (G4 backend_active).
        The child ships this so the trusted parent can tell a real-runner unknown
        signal (profiler failure -> fail closed) from a dev-machine stub unknown
        (no torch_npu here -> not measured, no flip).

        This deliberately returns the IMMUTABLE import-time ``_NPU_AVAILABLE``,
        NOT a live ``torch.npu.is_available()`` device probe. The probe is unsafe
        here on two counts: (1) it is computed in the child AFTER the candidate
        module loads, and ``torch.npu.is_available`` is a mutable attribute a
        static-clean candidate could monkeypatch to force backend_active False
        and suppress the anti-cheat fail-close; (2) the first device probe does
        ``ctypes.CDLL`` which the Layer E audit hook blocks, so it would fail and
        fall back to the import signal anyway. So the import bool is the only
        safe, tamper-proof signal in the sandbox. Consequence (documented): a
        host with torch_npu installed but no usable device reports True; that
        rare case surfaces as a grade FAIL with a device error, not as a stub."""
        return _NPU_AVAILABLE

    def ensure_initialized(self) -> None:
        """Pre-load the torch_npu profiler NOW, in the trusted-shim phase, so any
        lazy ``dlopen`` it does happens BEFORE the Layer E audit hook blocks
        dlopen. ``import torch_npu`` inits the device eagerly, but the
        ``torch_npu.profiler`` SUBMODULE the G1 path (``_profile_npu``) imports on
        the graded call can load its native lib lazily on first import; if that
        first import landed post-hook the ``ctypes``/``dlopen`` would be rejected.
        Warming it here (mirroring GpuBackend's CUDA/cuBLAS pre-init) makes the
        later ``import torch_npu.profiler`` a cached no-op. Best-effort: a profiler
        that cannot import is left for ``_profile_npu`` to surface, never fatal
        here. No-op on the dev machine (no torch_npu)."""
        if not _NPU_AVAILABLE:
            return
        try:
            import torch_npu.profiler  # noqa: F401 — side-effect: dlopen pre-hook.
        except Exception:
            pass

    def dtype_supported(self, dtype: torch.dtype) -> bool:
        # 910B does not have FP8 cube arithmetic; software emulation only.
        if dtype in {getattr(torch, "float8_e4m3fn", None), getattr(torch, "float8_e5m2", None)}:
            return False
        return True

    def device_name(self) -> str:
        if not _NPU_AVAILABLE:
            return "<no torch_npu>"
        return torch.npu.get_device_name(0)

    def event_timer(self) -> EventTimer | None:
        """ /  /  timing cross-check seam
        (docs/reference/timing_arbitration.md).

        The host `CLOCK_MONOTONIC_RAW` timer is authoritative; the NPU
        event timer is a secondary signal used only to detect work hidden
        on a non-default stream.

        : this returns an :class:`EventTimer` *context manager*, not a
        callable that runs `fn` itself. The event `record()` calls bracket
        the SAME single measured `fn(inp)` invocation that host time and
        the profiler observe — the timing-pass child does::

            timer = backend.event_timer()
            with (timer or contextlib.nullcontext()):
                out = fn(inp)            # the one measured call
            event_ns = timer.elapsed_ns() if timer else None

        The timer records only; the caller owns the surrounding device syncs.

        The candidate is never invoked a second time merely to take an
        event sample (that re-opened the  stateful-candidate hole —
        each measured input was executed twice). One measured trial is now
        exactly one candidate call, host-timed AND event-bracketed.

        : the resulting `event_ns` series travels up the status.json
        and the trusted parent feeds it to
        `timing.arbitrate_timing(event_ns=...)`, so the "host/event drift
        > 10% + non-default stream ⇒ invalid_timing" rule is reachable on
        the 910B.

        On the dev machine there is no NPU, so this returns `None`; the
        child's `event_ns` series stays all-`None` and timing degrades to
        host-only (`drift_status="single_source"`). The real
        `torch.npu.Event` implementation lands on the 910B machine — the
        dev machine simply never constructs an :class:`EventTimer`.
        """
        if not _NPU_AVAILABLE:
            return None
        return EventTimer()

    def profile_call(self, fn: Callable, inputs: object) -> dict:
        """ runtime-defense profiler seam.

        Runs ``fn(inputs)`` **exactly once** under an NPU profiler and
        reports — alongside that single call's return value — three
        signals the runner uses to detect reward hacking (DESIGN §8):

        * ``output`` — the value ``fn(inputs)`` returned on the profiled
          call. The caller MUST consume this rather than re-invoking
          ``fn``: a re-invocation would let a stateful candidate behave
          honestly on the profiled call and cheat on the call that is
          actually graded/timed .
        * ``kernel_event_present`` — did the call dispatch at least one
          real NPU/triton kernel? An empty-kernel cheat (B5) lands here.
        * ``stream_nondefault`` — did any work run on a non-default
          stream? Work hidden off the default stream (B2) escapes host
          timing and is flagged.
        * ``fallback_signal`` — derived: a candidate that produced **no**
          NPU/triton kernel event has silently fallen back to the generic
          torch (CPU/eager) path (C7) rather than running a real Ascend
          kernel; ``fallback_signal = not kernel_event_present``.

        On the dev machine there is no NPU and no profiler. The hook
        returns a documented stub: it still *runs* ``fn`` once (and
        returns that call's output) so the child's plumbing is exercised,
        but reports ``kernel_event_present=None`` /
        ``stream_nondefault=None`` and — because absence of a kernel event
        cannot be confirmed without a profiler — ``fallback_signal=None``.
        ``None`` means "not measured on this host"; the real ``torch_npu``
        profiler implementation lands on the 910B machine. The
        fallback-from-kernel-event derivation below is host-independent
        and is the production code path; only the kernel-event *source*
        is stubbed here.

        Any exception from ``fn`` propagates — the caller records the run
        error; profiling never swallows it.
        """
        if not _NPU_AVAILABLE:
            # dev-machine stub: run fn once for plumbing; signals are
            # unknowable without a profiler. The single call's output is
            # returned so the caller never has to re-invoke `fn`.
            output = fn(inputs)
            return {
                "output": output,
                "kernel_event_present": None,
                "stream_nondefault": None,
                "fallback_signal": None,
                "npu_prof_raw_dir": None,
            }
        # NPU machine: `_profile_npu` RECORDS the candidate under the profiler
        # (one call total) and returns the raw trace dir; the kernel-event /
        # stream signals are DEFERRED (None) here and filled by the trusted
        # parent's offline parse (the in-child CSV parse is blocked by the
        # graded sandbox). fallback is likewise deferred until the parent fills
        # kernel_event_present.
        output, kernel_event_present, stream_nondefault, raw_dir = (
            self._profile_npu(fn, inputs))
        return {
            "output": output,
            "kernel_event_present": kernel_event_present,
            "stream_nondefault": stream_nondefault,
            "fallback_signal": _derive_fallback(kernel_event_present),
            "npu_prof_raw_dir": raw_dir,
        }

    def _profile_npu(self, fn: Callable, inputs: object) -> tuple:
        """Record ``fn(inputs)`` once under ``torch_npu.profiler`` and hand the
        RAW trace dir to the trusted parent to parse — the kernel-event signal is
        DEFERRED, not extracted here.

        WHY deferred: ``kernel_details.csv`` (which lists the NPU kernels that
        ran) is generated by the profiler spawning CANN parser SUBPROCESSES
        (``exec``). This runs in the graded Layer E sandbox, which blocks
        ``exec`` — so the CSV is never produced in-child (verified on 910B:
        "Profiling data parsing failed, error: Layer E: blocked runtime operation
        'exec'"). The RECORDING works in-sandbox (the raw ``*_ascend_pt`` dir IS
        written); only the parse is blocked. So we record here and let the
        trusted PARENT (no hook) run ``torch_npu.profiler.profiler.analyse`` on
        the raw dir post-child and fill ``kernel_event_present`` /
        ``fallback_signal`` (see ``process_isolation`` parent-side parse).

        Returns ``(output, None, None, raw_dir)``: kernel_event_present and
        stream_nondefault are DEFERRED (None); ``raw_dir`` is the parent-owned
        trace dir to parse. The raw dir lives under ``AB_NPU_PROF_ROOT`` (a
        parent-created root OUTSIDE the candidate cwd, so the trust boundary is
        the parent's, not the candidate's). If that root is absent (a
        misconfiguration), raw_dir is None and the signal stays deferred None ->
        the parent leaves kernel_event None -> G4 fails closed (real device +
        unverifiable primary signal). The single call's output is returned so the
        caller never re-invokes ``fn`` (stateful-candidate invariant).
        """
        # torch_npu.profiler is pre-warmed in ensure_initialized() (pre-hook), so
        # this is a cached sys.modules lookup here. It stays in-function because
        # it cannot be imported on the dev box (no torch_npu).
        import torch_npu.profiler as _prof

        root = _os.environ.get("AB_NPU_PROF_ROOT")
        if not root or not _os.path.isdir(root):
            # No parent-owned trace root -> parse-in-parent impossible. Run the
            # one call for plumbing; leave the signal deferred (None) so the
            # parent fails closed rather than inventing a verdict.
            return fn(inputs), None, None, None

        # Record into a per-call subdir of the PARENT-owned root (outside the
        # candidate cwd). NOT cleaned here — the parent parses then removes it.
        prof_dir = tempfile.mkdtemp(prefix="t_", dir=root)
        with _prof.profile(
            activities=[_prof.ProfilerActivity.CPU, _prof.ProfilerActivity.NPU],
            on_trace_ready=_prof.tensorboard_trace_handler(prof_dir),
        ):
            output = fn(inputs)
        torch.npu.synchronize()
        # NO in-child parse: the CSV export spawns parser subprocesses (exec) the
        # graded hook blocks. The raw `*_ascend_pt` dir is written regardless; the
        # trusted parent parses it. kernel_event_present / stream_nondefault are
        # deferred to None (the parent fills the real value).
        return output, None, None, prof_dir


def _derive_fallback(kernel_event_present: bool | None) -> bool | None:
    """Derive the fallback signal from kernel-event presence (C7).

    A candidate that emitted no NPU/triton kernel event has silently
    fallen back to the generic torch path instead of running a real
    Ascend kernel — ``fallback_signal = not kernel_event_present``. When
    the kernel-event source is unknown (``None``, dev-machine stub) the
    fallback signal is likewise unknown. This derivation is
    host-independent: it is the production code path even though the
    dev-machine stub feeds it ``None``.
    """
    if kernel_event_present is None:
        return None
    return not kernel_event_present
