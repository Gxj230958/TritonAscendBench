"""GPU backend — CUDA via standard triton (oracle / H200 self-check device)."""
from __future__ import annotations

from collections.abc import Callable

import torch

_CUDA_AVAILABLE = torch.cuda.is_available()


class CudaEventTimer:
    """CUDA event timer that brackets one externally-made call, mirroring the
    NPU EventTimer. __enter__/__exit__ only RECORD the start/end events around
    the SAME measured `fn(inp)` — they do NOT synchronize. The trusted-shim
    caller already drains the device once before h0 and once before reading, so
    an in-bracket sync would only add a redundant synchronize to host_ns and
    bias the latency of small kernels. The candidate is never re-run for a
    separate sample (R6-1 / R5-2)."""

    def __init__(self) -> None:
        self._ev_start = torch.cuda.Event(enable_timing=True)
        self._ev_end = torch.cuda.Event(enable_timing=True)
        self._ended = False
        self._elapsed_ns: int | None = None

    def __enter__(self) -> CudaEventTimer:
        self._ev_start.record()
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        # Record the end event only on a clean exit; an exception leaves the
        # bracket incomplete and elapsed_ns() stays None.
        if exc_type is None:
            self._ev_end.record()
            self._ended = True
        return False  # never suppress an exception from the measured call

    def elapsed_ns(self) -> int | None:
        """Event-measured ns for the bracketed call, or None if it did not
        complete cleanly. CONTRACT: the caller must device-sync between __exit__
        and this call (the trusted shim's post-bracket synchronize) so both
        events are complete. start.elapsed_time(end) is ms; convert to ns."""
        if self._ended and self._elapsed_ns is None:
            self._elapsed_ns = int(
                self._ev_start.elapsed_time(self._ev_end) * 1e6)
        return self._elapsed_ns


class GpuBackend:
    name = "gpu"

    @property
    def device(self) -> torch.device:
        if not _CUDA_AVAILABLE:
            raise RuntimeError("CUDA not available; cannot use GPU backend")
        return torch.device("cuda")

    def synchronize(self) -> None:
        if _CUDA_AVAILABLE:
            torch.cuda.synchronize()

    def needs_compile_warmup(self) -> bool:
        """True when a real CUDA device is present: a triton kernel JIT-compiles
        + loads (dlopen / compiler subprocess) on first call, which must happen
        before the Layer E hook. False on the dev box (no device, no JIT, so no
        warmup call — preserving the exactly-one-graded-call R5/R6 invariant)."""
        return _CUDA_AVAILABLE

    def ensure_initialized(self) -> None:
        """Force CUDA's lazy init now, in the trusted-shim phase, so the
        driver / runtime / cuBLAS ``ctypes.dlopen`` happens BEFORE the Layer E
        audit hook blocks dlopen. Without this the candidate's first CUDA op
        triggers a deferred init the hook rejects (DeferredCudaCallError). A
        tiny elementwise + matmul pre-loads the core runtime and cuBLAS."""
        if not _CUDA_AVAILABLE:
            return
        torch.cuda.init()
        a = torch.zeros((8, 8), device="cuda")
        _ = (a + 1.0) @ a
        torch.cuda.synchronize()

    def dtype_supported(self, dtype: torch.dtype) -> bool:
        return True

    def device_name(self) -> str:
        if not _CUDA_AVAILABLE:
            return "<no cuda>"
        return torch.cuda.get_device_name(0)

    def event_timer(self) -> CudaEventTimer | None:
        """CUDA event timer for ONE measured call, or None without CUDA (dev
        box) — the caller then degrades to host-only timing. The timer records
        only; the caller owns the surrounding device syncs."""
        if not _CUDA_AVAILABLE:
            return None
        return CudaEventTimer()

    def profile_call(self, fn: Callable, inputs: object) -> dict:
        """Run `fn(inputs)` exactly once and report reward-hacking signals.
        The caller MUST consume the returned output, never re-invoke `fn`
        (a stateful candidate could cheat the un-profiled call)."""
        if not _CUDA_AVAILABLE:
            # dev stub: run once for plumbing; signals unknowable without CUDA.
            return {
                "output": fn(inputs),
                "kernel_event_present": None,
                "stream_nondefault": None,
                "fallback_signal": None,
            }
        return self._profile_cuda(fn, inputs)

    def _profile_cuda(self, fn: Callable, inputs: object) -> dict:
        """Real CUDA profiler trace inspection — H200-verified seam. The
        kernel-event / non-default-stream classification must be validated on
        a real GPU before its signals are trusted; until then it runs `fn`
        once and reports unknown diagnostics (no blind profiler parser)."""
        return {
            "output": fn(inputs),
            "kernel_event_present": None,
            "stream_nondefault": None,
            "fallback_signal": None,
        }
