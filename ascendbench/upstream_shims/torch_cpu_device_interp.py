"""Interpret-mode shim: a no-op ``torch.cpu.device`` context manager.

Upstream Mamba2 / SSD host wrappers (and many other GPU kernels) guard their
Triton launch with the device-placement context manager::

    with torch.get_device_module(x.device).device(x.device.index):
        kernel[grid](...)

``torch.get_device_module(dev)`` returns the device's runtime module
(``torch.cuda`` for a CUDA tensor, ``torch.cpu`` for a CPU tensor). On CUDA
``torch.cuda.device(index)`` is a real placement guard; on CPU,
``torch.cpu`` has NO ``device`` attribute, so the wrapper raises
``AttributeError`` the moment it is invoked.

The benchmark's oracle-consistency check runs under ``TRITON_INTERPRET=1`` with
all inputs on CPU (``_pa_oracle_check_runner.py``: ``device="cpu" if interpret
else "cuda"``). Every CPU-interpret invocation of such a host wrapper therefore
crashes with that ``AttributeError`` and the resolver mis-buckets the task as
``upstream-not-invocable-here`` -- a systemic environmental gap, not a kernel or
launcher defect.

This shim installs a no-op ``torch.cpu.device(index)`` context manager ONLY when
it is absent (i.e. only on CPU; ``torch.cuda.device`` is left native). Under
interpret everything already executes on CPU, so a host-side device-placement
guard has zero numeric effect: the bit-exact severed-vs-original (S2) and the
oracle-consistency comparisons are unaffected -- a wrong kernel still fails. The
shim is reversible via :func:`uninstall`.
"""

from __future__ import annotations

import contextlib

_installed = False


def install() -> bool:
    """Install a no-op ``torch.cpu.device`` context manager if missing.

    Returns:
        True if the shim was installed (it was absent), False if a native
        ``torch.cpu.device`` already exists (nothing patched).
    """
    global _installed
    try:
        import torch
    except ImportError:
        return False
    cpu_mod = getattr(torch, "cpu", None)
    if cpu_mod is None:
        return False
    if hasattr(cpu_mod, "device"):
        # Native attribute present (future torch, or already shimmed) -- leave
        # it untouched so we never shadow real behaviour.
        return False

    @contextlib.contextmanager
    def _noop_device(index=None):
        # CPU interpret has no device placement to enter; the guard is purely a
        # host-side context with no numeric effect.
        yield

    cpu_mod.device = _noop_device  # type: ignore[attr-defined]
    _installed = True
    return True


def uninstall() -> None:
    """Remove a shim previously installed by :func:`install` (idempotent)."""
    global _installed
    if not _installed:
        return
    try:
        import torch

        cpu_mod = getattr(torch, "cpu", None)
        if cpu_mod is not None and hasattr(cpu_mod, "device"):
            delattr(cpu_mod, "device")
    except Exception:
        pass
    _installed = False
