"""Interpreter-mode ``triton.autotune`` fallback -- the SINGLE shared
implementation used by BOTH the resolver/original side (the
``vllm.triton_utils`` shim re-exports from here) AND the Phase A' severed
standalone side (the rewrite binds a Triton clone built from here for
autotuned kernels).

Why this exists: real ``triton.autotune`` benchmarks each ``Config`` on a
device to pick the best, which aborts with
``RuntimeError: 0 active drivers`` on a CPU / ``TRITON_INTERPRET=1`` box.
The fallback below selects the FIRST config deterministically (NOT "best" --
we do not pretend to autotune without a device) and forwards into the jit's
normal ``[grid](...)`` launch. The first config is a documented, fixed
choice; both the original and severed sides use it, so their launches pick
the SAME specialization -> bit-exact S2 parity. On a real GPU with a driver
active and ``TRITON_INTERPRET`` unset, the real autotuner runs unchanged.
"""
from __future__ import annotations

import os
import types
from collections.abc import Callable
from typing import Any

import triton as _real_triton


def _has_active_triton_driver() -> bool:
    """Return ``True`` iff at least one Triton backend driver reports
    itself active in this process.

    The check is defensive: a missing ``triton.backends``, a driver whose
    ``is_active`` raises, or any other introspection failure is treated as
    *not* active -- the fallback then fires.
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
    backend driver is active. The autotune fallback fires in either case
    (the resolver always sets ``TRITON_INTERPRET=1``; a CPU dev host
    typically also has no active driver). A real GPU run with
    ``TRITON_INTERPRET`` unset and at least one active driver goes through
    the real autotuner."""
    if os.environ.get("TRITON_INTERPRET", "").strip() not in ("", "0"):
        return True
    return not _has_active_triton_driver()


def _pin_first_config_active() -> bool:
    """``True`` when ``PA_PIN_AUTOTUNE_FIRST`` is set -- forces the
    first-config fallback even on a live GPU with an active driver. Used ONLY
    by the Phase A' S3 determinism re-runs: the real autotuner picks a config
    by min-timing, which varies run-to-run and is in-tolerance but
    bit-different (a different ``BK`` changes the ``tl.dot`` accumulation
    order), so an autotuned kernel fails S3's strict bit-exact check
    non-deterministically. Pinning configs[0] makes every S3 run pick the
    SAME specialization. The env is set only around the S3 subprocesses;
    c2 / S2-gpu / gpu-rank keep the real autotuner."""
    return os.environ.get("PA_PIN_AUTOTUNE_FIRST", "").strip() not in ("", "0")


class _AutotunedKernel:
    """The object returned by the shim's ``autotune`` decorator under
    interpreter / no-driver mode. It mimics the small slice of
    ``triton.runtime.autotuner.Autotuner`` the call sites use --
    ``kernel[grid](*args, **kwargs)`` -- by binding the first ``Config``'s
    kwargs and forwarding into the underlying JIT function's normal
    ``[grid](...)`` launch path.

    The wrapper deliberately mirrors only the ``__getitem__`` surface (and a
    few light passthroughs) because the resolver only invokes the kernel via
    ``kernel[grid](...)``; broader Autotuner API surface (``warmup``, ``rep``,
    ``do_bench``, etc.) is *intentionally not* emulated."""

    __slots__ = ("_configs", "_first_kwargs", "_jit_fn")

    def __init__(self, jit_fn: Any, configs: list[Any]) -> None:
        self._jit_fn = jit_fn
        self._configs = configs
        first = configs[0]
        # Real Triton's Autotuner launches via ``config.all_kwargs()``, which
        # merges ``Config.kwargs`` with the launch params (``num_warps``,
        # ``num_stages``, ``num_ctas``, ``maxnreg``, ``ir_override``). Using
        # only ``Config.kwargs`` here would silently drop ``num_warps`` /
        # ``num_stages`` for kernels whose first config specifies them.
        all_kwargs_fn = getattr(first, "all_kwargs", None)
        if callable(all_kwargs_fn):
            self._first_kwargs = dict(all_kwargs_fn())
        else:
            # Defensive fallback for a Config stub that does not expose
            # ``all_kwargs`` (older Triton, test double): merge the public
            # attribute slots manually.
            merged_kw: dict[str, Any] = dict(getattr(first, "kwargs", {}) or {})
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

    @property
    def fn(self) -> Any:
        return self._jit_fn

    @property
    def configs(self) -> list[Any]:
        return list(self._configs)

    @property
    def arg_names(self) -> list[str]:
        """The wrapped kernel's parameter names, in declaration order
        (constexprs INCLUDED). ``triton.heuristics`` -- which FLA /
        gated-delta kernels stack ON TOP of ``@triton.autotune`` -- reads
        ``fn.arg_names`` and builds ``dict(zip(arg_names, args))`` to bind its
        heuristic lambdas. Delegate to the wrapped jit fn; fall back to the
        underlying Python signature."""
        names = getattr(self._jit_fn, "arg_names", None)
        if names is not None:
            return list(names)
        import inspect

        fn = getattr(self._jit_fn, "fn", None) or self._jit_fn
        return [p.name for p in inspect.signature(fn).parameters.values()]

    def run(self, *args: Any, **kwargs: Any) -> Any:
        """``triton.heuristics``'s launch path calls ``self.fn.run(...)``
        (KernelInterface contract), NOT ``self.fn[grid](...)``. Mirror it by
        routing through the SAME ``jit_fn[grid]`` launch that ``__getitem__``
        uses, so the autotune-config kwargs + the injected heuristic kwargs
        (e.g. ``IS_VARLEN``) are bound identically."""
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
    """Drop-in replacement for ``triton.autotune`` that delegates to the real
    autotuner on a GPU box and falls back to the interpreter-friendly
    first-config wrapper when no driver is active.

    The signature mirrors ``triton.autotune(configs, key, ...)`` -- every
    extra keyword argument (``prune_configs_by``, ``reset_to_zero``,
    ``restore_value``, ``warmup``, etc.) is forwarded verbatim to the real
    autotuner so the GPU path stays bit-identical to upstream."""
    if not _interpreter_mode_active() and not _pin_first_config_active():
        return _real_triton.autotune(configs=configs, key=key, **kwargs)

    if not configs:
        raise ValueError(
            "triton_autotune_interp: autotune() invoked with no configs under "
            "interpreter/no-driver mode -- cannot pick a fallback."
        )

    def _decorator(jit_fn: Any) -> _AutotunedKernel:
        return _AutotunedKernel(jit_fn, list(configs))

    return _decorator


def make_triton_clone() -> types.ModuleType:
    """Return a *clone* of the real ``triton`` module with ``autotune``
    swapped for the interpreter-aware wrapper above.

    A clone (a fresh ``types.ModuleType`` whose ``__dict__`` is copied from
    the real triton module) lets a kernel pick up the wrapper without mutating
    the global ``triton.autotune`` -- any other code path that goes through
    plain ``import triton`` keeps the unmodified real module. The clone shares
    the same ``__dict__`` entries (``triton.Config`` / ``triton.jit`` /
    ``triton.heuristics`` / submodules) by reference, so they behave
    identically to the real module."""
    clone = types.ModuleType("triton")
    clone.__dict__.update(_real_triton.__dict__)
    clone.autotune = autotune
    if hasattr(_real_triton, "__path__"):
        clone.__path__ = list(_real_triton.__path__)
    clone.__file__ = getattr(_real_triton, "__file__", clone.__file__)
    clone.__doc__ = _real_triton.__doc__
    return clone
