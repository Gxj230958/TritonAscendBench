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
On a CPU/``TRITON_INTERPRET=1`` host with no active Triton driver, the
real ``triton.autotune`` decorator still produces an ``Autotuner``
object, but the moment its ``__call__`` runs it tries to benchmark
each ``Config`` on the GPU and aborts with ``RuntimeError: 0 active
drivers``. That is the L2_01-style failure mode the resolver hits.

The shim's ``autotune`` replaces the decorator with one that:

  * when ``TRITON_INTERPRET=1`` *or* no Triton backend driver is
    active, **skips benchmarking entirely**. It returns a wrapper
    whose ``__getitem__(grid)`` builds a callable that merges the first
    ``Config``'s ``kwargs`` into the user kwargs, then dispatches into
    the underlying ``@triton.jit`` function's ``[grid](...)`` directly.
    The first config is a documented, deterministic choice (NOT a
    "best" config -- we explicitly do not pretend to autotune without
    a GPU; this is a soundness-preserving fallback used solely to let
    the resolver verify *operator identity* under ``TRITON_INTERPRET``);
  * when a real backend driver is active, delegates to the real
    ``triton.autotune`` so a GPU run still benchmarks normally.

This is exactly the behaviour an interpreter-mode resolver needs: it
never silently picks a wrong operator (the operator is the same JIT
function under any config); it only avoids the benchmark step that
requires a GPU. The Config kwargs (``BLOCK_SIZE`` etc.) and
``num_warps`` / ``num_stages`` propagate to the JIT call exactly as
they would after autotune picks them.
"""

from __future__ import annotations

import math
import os
import types
from collections.abc import Callable
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


def _has_active_triton_driver() -> bool:
    """Return ``True`` iff at least one Triton backend driver reports
    itself active in this process.

    The check is defensive: a missing ``triton.backends``, a driver
    whose ``is_active`` raises, or any other introspection failure is
    treated as *not* active -- the fallback then fires. This mirrors
    the spirit of ``infer/vllm/vllm/triton_utils/importing.py``'s own
    catch-broadly-and-disable logic.
    """
    try:
        from triton.backends import backends

        for entry in backends.values():
            drv = getattr(entry, "driver", None)
            if drv is None:
                continue
            try:
                if drv.is_active():
                    return True
            except Exception:
                continue
        return False
    except Exception:
        return False


def _interpreter_mode_active() -> bool:
    """``True`` when ``TRITON_INTERPRET`` is set to a truthy value OR no
    backend driver is active. The autotune fallback fires in either
    case (the resolver always sets ``TRITON_INTERPRET=1``; a CPU dev
    host typically also has no active driver). A real GPU run with
    ``TRITON_INTERPRET`` unset and at least one active driver goes
    through the real autotuner."""
    if os.environ.get("TRITON_INTERPRET", "").strip() not in ("", "0"):
        return True
    return not _has_active_triton_driver()


class _AutotunedKernel:
    """The object returned by the shim's ``autotune`` decorator under
    interpreter / no-driver mode. It mimics the small slice of
    ``triton.runtime.autotuner.Autotuner`` the call sites use --
    namely ``kernel[grid](*args, **kwargs)`` -- by binding the first
    ``Config``'s kwargs and forwarding into the underlying JIT
    function's normal ``[grid](...)`` launch path.

    The wrapper deliberately mirrors only the ``__getitem__`` surface
    (and a few light passthroughs) because the resolver only invokes
    the kernel via ``kernel[grid](...)``; broader Autotuner API
    surface (``warmup``, ``rep``, ``do_bench``, etc.) is *intentionally
    not* emulated -- if a caller depends on it under interpreter mode
    that is a benchmark-correctness signal we want to surface, not
    paper over."""

    __slots__ = ("_configs", "_first_kwargs", "_jit_fn")

    def __init__(self, jit_fn: Any, configs: list[Any]) -> None:
        self._jit_fn = jit_fn
        self._configs = configs
        first = configs[0]
        # Real Triton's Autotuner launches via ``config.all_kwargs()``
        # (see ``triton.runtime.autotuner.Autotuner.run``), which merges
        # ``Config.kwargs`` with the launch params (``num_warps``,
        # ``num_stages``, ``num_ctas``, ``maxnreg``, ``ir_override``).
        # Using only ``Config.kwargs`` here would silently drop
        # ``num_warps`` / ``num_stages`` for kernels whose first config
        # specifies them (e.g. L2_01_bmm_chunk_fwd lists ``num_warps=8,
        # num_stages=3`` on its first config). ``all_kwargs`` is the
        # public method real Triton exposes for exactly this purpose, so
        # the fallback now matches the launch contract.
        all_kwargs_fn = getattr(first, "all_kwargs", None)
        if callable(all_kwargs_fn):
            self._first_kwargs = dict(all_kwargs_fn())
        else:
            # Defensive fallback for a Config stub that does not expose
            # ``all_kwargs`` (older Triton, test double): merge the
            # public attribute slots manually.
            merged_kw: dict[str, Any] = dict(getattr(first, "kwargs", {}) or {})
            # Mirrors the launch-param subset that ``Config.all_kwargs()``
            # surfaces; ``ir_override`` is included because real Triton
            # emits it when non-None (see triton.runtime.autotuner.Config).
            for attr in ("num_warps", "num_stages", "num_ctas", "maxnreg",
                         "ir_override"):
                v = getattr(first, attr, None)
                if v is not None:
                    merged_kw[attr] = v
            self._first_kwargs = merged_kw

    def __getitem__(self, grid: Any) -> Callable[..., Any]:
        bound = self._jit_fn[grid]
        first_kwargs = self._first_kwargs

        def _launch(*args: Any, **kwargs: Any) -> Any:
            merged = dict(first_kwargs)
            merged.update(kwargs)  # caller kwargs win over Config defaults
            return bound(*args, **merged)

        return _launch

    # A few attribute passthroughs commonly used by callers that
    # introspect the wrapped JIT function. We do NOT pretend to be an
    # Autotuner -- callers that need autotuner internals should run on
    # a real GPU.
    @property
    def fn(self) -> Any:
        return self._jit_fn

    @property
    def configs(self) -> list[Any]:
        return list(self._configs)

    @property
    def arg_names(self) -> list[str]:
        """The wrapped kernel's parameter names, in declaration order
        (constexprs INCLUDED). ``triton.heuristics`` -- which the FLA /
        gated-delta kernels stack ON TOP of ``@triton.autotune`` -- reads
        ``fn.arg_names`` and builds ``dict(zip(arg_names, args))`` to bind
        its heuristic lambdas (triton.runtime.autotuner.Heuristics.run).
        Real ``Autotuner`` copies this from the JITFunction; the
        interpreter-mode wrapper omitted it, so a stacked
        ``@heuristics -> @autotune -> @jit`` kernel failed with
        ``AttributeError: '_AutotunedKernel' object has no attribute
        'arg_names'``. Delegate to the wrapped jit fn (real JITFunction
        sets ``arg_names = [p.name for p in self.params]``); fall back to
        the underlying Python signature."""
        names = getattr(self._jit_fn, "arg_names", None)
        if names is not None:
            return list(names)
        import inspect

        fn = getattr(self._jit_fn, "fn", None) or self._jit_fn
        return [p.name for p in inspect.signature(fn).parameters.values()]

    def run(self, *args: Any, **kwargs: Any) -> Any:
        """``triton.heuristics``'s launch path calls ``self.fn.run(...)``
        (KernelInterface contract), NOT ``self.fn[grid](...)``. Mirror it
        by routing through the SAME ``jit_fn[grid]`` launch that
        ``__getitem__`` uses, so the autotune-config kwargs + the injected
        heuristic kwargs (e.g. ``IS_VARLEN``) are bound identically.
        ``grid``/``warmup`` arrive in kwargs from
        ``KernelInterface.__getitem__``; ``[grid]`` re-supplies
        ``warmup=False`` itself."""
        merged = dict(self._first_kwargs)
        merged.update(kwargs)  # caller/heuristic kwargs win over Config
        grid = merged.pop("grid", None)
        merged.pop("warmup", None)
        return self._jit_fn[grid](*args, **merged)

    def __repr__(self) -> str:
        return (
            f"<_AutotunedKernel (interpreter fallback) fn={self._jit_fn!r} "
            f"configs={len(self._configs)}>"
        )


def autotune(
    configs: list[Any],
    key: list[str] | None = None,
    **kwargs: Any,
) -> Callable[[Any], Any]:
    """Drop-in replacement for ``triton.autotune`` that delegates to the
    real autotuner on a GPU box and falls back to an
    interpreter-friendly wrapper when no driver is active.

    The signature mirrors ``triton.autotune(configs, key, ...)`` --
    every extra keyword argument (``prune_configs_by``,
    ``reset_to_zero``, ``restore_value``, ``warmup``, etc.) is
    forwarded verbatim to the real autotuner so the GPU path stays
    bit-identical to upstream."""
    if not _interpreter_mode_active():
        return _real_triton.autotune(configs=configs, key=key, **kwargs)

    if not configs:
        # An empty config list under interpreter mode is a benchmark
        # configuration error, not something we silently paper over.
        raise ValueError(
            "vllm.triton_utils shim: autotune() invoked with no configs "
            "under interpreter/no-driver mode -- cannot pick a fallback."
        )

    def _decorator(jit_fn: Any) -> _AutotunedKernel:
        return _AutotunedKernel(jit_fn, list(configs))

    return _decorator


def _make_triton_clone() -> types.ModuleType:
    """Return a *clone* of the real ``triton`` module with
    ``autotune`` swapped for the shim wrapper above.

    A clone (a fresh ``types.ModuleType`` whose ``__dict__`` is copied
    from the real triton module) lets us patch ``autotune`` without
    mutating the global ``triton.autotune`` -- a vendored kernel that
    imports ``triton`` *via* ``vllm.triton_utils.triton`` gets the
    interpreter-aware wrapper, while any code path that goes through
    the plain ``import triton`` (e.g. inside Triton itself, or
    ``triton.heuristics`` reaching at runtime for ``triton.runtime``)
    keeps the unmodified real module. The clone shares the same
    ``__dict__`` *entries* (functions / classes / submodules) by
    reference, so things like ``triton.Config`` and ``triton.jit``
    behave identically to the real module."""
    clone = types.ModuleType("triton")
    clone.__dict__.update(_real_triton.__dict__)
    clone.autotune = autotune
    # ``triton.__path__`` is the real package path; preserve it so
    # downstream submodule lookups against the clone still work.
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


__all__ = ["HAS_TRITON", "LOG2E", "LOGE2", "tl", "tldevice", "triton"]
