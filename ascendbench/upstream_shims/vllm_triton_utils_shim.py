"""Forward-shim module body for ``vllm.triton_utils``.

This module's *public surface* mirrors upstream's
``infer/vllm/vllm/triton_utils/__init__.py.__all__`` exactly --
``HAS_TRITON``, ``triton``, ``tl``, ``tldevice``, ``LOG2E``, ``LOGE2``
-- but every name is *forwarded to the real Triton* installed on the
host instead of upstream's ``TritonPlaceholder`` stub. ``HAS_TRITON``
is hard-set to ``True`` (the shim is only ever registered when we want
the real Triton surface), so the conditional at upstream's line 11 is
unconditionally taken.

``triton`` here is a *module clone* of the real ``triton`` (not the
real module itself), so the autotune wrapper described below can be
swapped in without ever mutating the global ``triton.autotune``
attribute -- a vendored kernel that says ``from vllm.triton_utils
import triton`` gets the wrapped clone, while any other code path that
says ``import triton`` continues to see the unmodified upstream Triton.
Every other attribute (``triton.jit``, ``triton.Config``,
``triton.heuristics``, ``triton.language``, ``triton.runtime``, the
``JITFunction`` class, etc.) is re-exported verbatim from the real
Triton via the clone's ``__getattr__``-style copy of ``__dict__``.

Autotune fallback (the key fix vs. upstream's placeholder)
----------------------------------------------------------
On a CPU/``TRITON_INTERPRET=1`` host with no active Triton driver, the real
``triton.autotune`` decorator aborts with ``RuntimeError: 0 active drivers``
the moment it tries to benchmark a ``Config`` (the L2_01-style failure mode
the resolver hits). The shim's ``autotune`` instead skips benchmarking under
``TRITON_INTERPRET=1`` or no active driver and binds the first ``Config``
(deterministic, not "best") -- see ``triton_autotune_interp`` for the
soundness argument. With a live driver it delegates to the real
``triton.autotune`` so a GPU run still benchmarks normally.
"""

from __future__ import annotations

import math
import os
import types
from typing import Any

import triton as _real_triton
import triton.language as _real_tl

try:
    import triton.language.extra.libdevice as _real_libdevice
except ImportError:
    # Some Triton builds may not ship libdevice; upstream falls back to
    # a placeholder in that case, but on a CPython box with real Triton
    # installed it is overwhelmingly present. If it really is missing we
    # surface the real Triton namespace so a kernel that does not touch
    # libdevice still works.
    _real_libdevice = _real_tl


HAS_TRITON: bool = True


# The first-config interpreter fallback CLASS (``_AutotunedKernel``) is the
# SINGLE shared implementation in ``triton_autotune_interp`` -- the Phase A'
# severed side binds it via the SAME module, so the two sides cannot drift in
# how they pick a config under interpret. The mode gate + ``autotune`` /
# clone wrappers stay LOCAL here so this module's ``_has_active_triton_driver``
# remains the patch point the resolver-side tests mock.
from ascendbench.upstream_shims.triton_autotune_interp import (  # noqa: E402
    _AutotunedKernel,
)
from ascendbench.upstream_shims.triton_autotune_interp import (  # noqa: E402
    _has_active_triton_driver as _shared_has_active_triton_driver,
)


def _has_active_triton_driver() -> bool:
    """True iff at least one Triton backend driver is active (delegates to the
    shared probe). Kept as a local module attribute so callers/tests can patch
    ``vllm_triton_utils_shim._has_active_triton_driver`` and have it govern the
    delegation below."""
    return _shared_has_active_triton_driver()


def _interpreter_mode_active() -> bool:
    """``True`` when ``TRITON_INTERPRET`` is truthy OR no backend driver is
    active. Re-resolves the LOCAL ``_has_active_triton_driver`` each call so a
    test patch on this module takes effect."""
    if os.environ.get("TRITON_INTERPRET", "").strip() not in ("", "0"):
        return True
    return not _has_active_triton_driver()


def autotune(
    configs: list[Any],
    key: list[str] | None = None,
    **kwargs: Any,
):
    """Drop-in ``triton.autotune``: real autotuner on a GPU box, shared
    first-config ``_AutotunedKernel`` fallback under interpreter/no-driver
    mode. Mirrors ``triton_autotune_interp.autotune`` but routes the mode
    check through the LOCAL ``_interpreter_mode_active`` so the resolver-side
    tests that patch this module's driver probe keep governing delegation."""
    if not _interpreter_mode_active():
        return _real_triton.autotune(configs=configs, key=key, **kwargs)
    if not configs:
        raise ValueError(
            "vllm.triton_utils shim: autotune() invoked with no configs under "
            "interpreter/no-driver mode -- cannot pick a fallback."
        )

    def _decorator(jit_fn: Any) -> _AutotunedKernel:
        return _AutotunedKernel(jit_fn, list(configs))

    return _decorator


def _make_triton_clone() -> types.ModuleType:
    """Clone of real ``triton`` with ``autotune`` swapped for the local
    wrapper above (so the clone honours this module's patchable mode gate).
    Everything else (``jit`` / ``Config`` / submodules) is the real module by
    reference."""
    clone = types.ModuleType("triton")
    clone.__dict__.update(_real_triton.__dict__)
    clone.autotune = autotune
    if hasattr(_real_triton, "__path__"):
        clone.__path__ = list(_real_triton.__path__)
    clone.__file__ = getattr(_real_triton, "__file__", clone.__file__)
    clone.__doc__ = _real_triton.__doc__
    return clone


triton: types.ModuleType = _make_triton_clone()
tl: types.ModuleType = _real_tl
tldevice: types.ModuleType = _real_libdevice

LOG2E: float = math.log2(math.e)
LOGE2: float = math.log(2.0)


__all__ = [
    "HAS_TRITON", "LOG2E", "LOGE2", "tl", "tldevice", "triton",
    "autotune", "_AutotunedKernel", "_interpreter_mode_active",
    "_has_active_triton_driver",
]
