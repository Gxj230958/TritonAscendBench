"""Forward-shim module body for ``vllm.triton_utils.importing``.

Some vendored kernels (and a few upstream call sites) reach for
``from vllm.triton_utils.importing import HAS_TRITON`` directly --
bypassing the package-level ``vllm.triton_utils`` namespace. This
submodule shim mirrors that symbol set so the deeper import path also
sees the real Triton surface when ``_inject.register()`` is active.

The symbols mirror the upstream import surface we care about:
``HAS_TRITON`` (the value), ``TritonPlaceholder`` /
``TritonLanguagePlaceholder`` (the classes -- re-exported as real
``triton`` / ``triton.language`` so an ``isinstance`` check would
fail loud rather than silently match a stub), plus ``triton`` / ``tl``
forwards for the convenience case.
"""

from __future__ import annotations

from .vllm_triton_utils_shim import HAS_TRITON, tl, triton

# Re-export the real triton / tl as the names upstream's importing
# module uses, so a deeper ``from vllm.triton_utils.importing import
# TritonPlaceholder`` no longer hands out the no-op placeholder. We do
# NOT export the placeholder *classes* themselves -- a kernel that
# happened to ``isinstance(triton, TritonPlaceholder)`` should fail
# loudly (we want the real triton, not a stub).

__all__ = ["HAS_TRITON", "tl", "triton"]
