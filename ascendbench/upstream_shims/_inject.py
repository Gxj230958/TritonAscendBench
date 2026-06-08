"""Register / unregister the ``vllm.triton_utils`` forward shim into
``sys.modules`` so a freshly imported vendored kernel sees the real
Triton surface instead of upstream's ``TritonPlaceholder``.

The injection is intentionally a ``sys.modules`` swap rather than a
``sys.path`` shim directory. Reasons:

  * Adding a ``vllm/`` directory ahead of ``infer/vllm`` on ``sys.path``
    would shadow the *entire* ``vllm.*`` namespace, which the resolver
    needs (every other ``vllm.utils`` / ``vllm.model_executor``
    submodule must keep resolving to the real upstream source so the
    operator identity check sees the real kernel bytes). Selectively
    overriding *only* ``vllm.triton_utils`` is the smallest possible
    surface.
  * ``sys.modules`` injection is reversible (``unregister()`` restores
    pre-state) and ordering-independent: it does not matter whether
    the real ``vllm`` package is imported before or after registration
    because the lookup ``vllm.triton_utils`` consults ``sys.modules``
    before invoking the import machinery.

The runner subprocesses call ``register(force=True)`` *before* loading
the vendored kernel; the ``force`` flag lets us override an
already-imported upstream ``vllm.triton_utils`` (which a stray
``import vllm`` higher in the resolver stack may have populated with
the no-driver placeholder before our hook ran).
"""

from __future__ import annotations

import logging
import sys
import types

logger = logging.getLogger(__name__)

_PARENT_MODULE = "vllm.triton_utils"
_CHILD_MODULE = "vllm.triton_utils.importing"

_saved: dict[str, types.ModuleType | None] = {}


def _build_shim_modules() -> tuple[types.ModuleType, types.ModuleType]:
    """Return ``(parent_shim, child_shim)`` -- imported lazily so a
    user that never calls ``register()`` does not pay the import-time
    cost (the shim itself imports real Triton, which can be slow on
    cold disk)."""
    from . import vllm_triton_utils_importing_shim as child
    from . import vllm_triton_utils_shim as parent

    return parent, child


def register(force: bool = False) -> bool:
    """Inject the shim into ``sys.modules``.

    Args:
        force: When ``True`` (the runner default), always overwrite an
            existing ``vllm.triton_utils`` entry, recording its prior
            value so ``unregister()`` can restore it. When ``False``,
            leave an existing entry alone (so a real upstream module
            that genuinely loaded successfully is not clobbered).

    Returns:
        ``True`` if the shim was newly installed (or overwritten with
        ``force=True``); ``False`` if a pre-existing entry was kept.

    The function is idempotent under ``force=True`` -- repeated calls
    keep replacing the entry with the same shim object, so a runner
    that calls ``register(force=True)`` multiple times never accretes
    stale state. The saved-state slot is only written on the FIRST
    overwrite so a later ``unregister()`` restores the *original*
    upstream module, not an intermediate shim copy.
    """
    parent_shim, child_shim = _build_shim_modules()

    existing_parent = sys.modules.get(_PARENT_MODULE)
    if existing_parent is parent_shim:
        # Already installed; refresh child anyway in case a different
        # path popped it.
        sys.modules[_CHILD_MODULE] = child_shim
        return False
    if existing_parent is not None and not force:
        logger.debug(
            "vllm.triton_utils already in sys.modules and force=False; "
            "leaving upstream entry in place"
        )
        return False

    # First overwrite: remember the prior value so unregister() can
    # restore it. A subsequent register(force=True) call MUST NOT
    # overwrite the saved slot (else the restored module would be a
    # shim copy, not the original upstream module).
    if _PARENT_MODULE not in _saved:
        _saved[_PARENT_MODULE] = existing_parent
    if _CHILD_MODULE not in _saved:
        _saved[_CHILD_MODULE] = sys.modules.get(_CHILD_MODULE)

    sys.modules[_PARENT_MODULE] = parent_shim
    sys.modules[_CHILD_MODULE] = child_shim
    logger.debug("Installed vllm.triton_utils shim (force=%s)", force)
    return True


def unregister() -> None:
    """Restore the pre-``register()`` ``sys.modules`` entries.

    Removes the shim entries entirely when the pre-state was "not
    present" (``None``); restores the real module reference otherwise.
    The saved-state dict is cleared so a subsequent ``register()``
    treats itself as a fresh first-overwrite.
    """
    for key in (_PARENT_MODULE, _CHILD_MODULE):
        prior = _saved.get(key, ...)
        if prior is ...:
            # No saved slot -> register() was never called; nothing to
            # undo for this key.
            continue
        if prior is None:
            sys.modules.pop(key, None)
        else:
            sys.modules[key] = prior
    _saved.clear()


def is_registered() -> bool:
    """``True`` iff the parent shim module is currently the active
    ``sys.modules['vllm.triton_utils']`` entry."""
    from . import vllm_triton_utils_shim as parent

    return sys.modules.get(_PARENT_MODULE) is parent
