"""trusted_shim — C-level host timer compiled at install time.

Per `specs/process_isolation.md` + `specs/timing_arbitration.md`:

- The trusted runner process spawns the candidate as a subprocess.
- The shim's `host_clock_ns()` is the only timing primitive the runner
  trusts. Implementing it in C avoids Python-level interpolation noise.
- Built from `trusted_shim/timer.c` at `pip install -e .` time via
  pyproject.toml's `build-system` extension (Phase 2.B will add this).

At v0.1 this is a Python fallback wrapping `time.clock_gettime_ns`;
v1.0 (verified mode) replaces it with a compiled extension.
"""
from __future__ import annotations

import time


def host_clock_ns() -> int:
    """CLOCK_MONOTONIC_RAW host timer in nanoseconds."""
    return time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW)
