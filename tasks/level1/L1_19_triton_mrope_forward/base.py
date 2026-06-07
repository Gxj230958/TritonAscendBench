"""Fail-closed task-local stub for the OFF-GRADED-PATH upstream sibling
``vllm/model_executor/layers/rotary_embedding/base.py``.

The vendored ``kernel.py`` does ``from .base import RotaryEmbeddingBase``
at module scope. That symbol is the base class of ``MRotaryEmbedding`` --
a class that is NEVER instantiated on the graded path. The graded op is
the bare ``@triton.jit`` ``_triton_mrope_forward`` kernel (driven by the
``triton_mrope`` host wrapper / ``_pa_launcher.py``), which references
NEITHER ``MRotaryEmbedding`` NOR ``RotaryEmbeddingBase``.

Importing the REAL upstream ``base.py`` transitively executes
``rotary_embedding/common.py``, which calls ``direct_register_custom_op(
"flashinfer_rotary_embedding", ...)`` at module-import time against the
process-global ``vllm`` torch.library. The resolver execs ``kernel.py``
once per shape / per stage, so the second exec re-registers the op and
crashes with a duplicate-registration ``RuntimeError`` -- a pure import
side effect that has nothing to do with the graded rotary math.

The bench loader resolves ``from .base`` against ``[task_dir,
upstream_sibling]`` (task_dir FIRST), so this stub wins and severs the
off-path import. It supplies ONLY the import-time class symbol so the
``MRotaryEmbedding`` class body can be defined; any real use raises (the
graded path never reaches it -- fail-closed, no silent fake).
"""
from __future__ import annotations

import torch.nn as nn


class RotaryEmbeddingBase(nn.Module):
    """Definition-only stub base for ``MRotaryEmbedding``.

    Subclassable so ``class MRotaryEmbedding(RotaryEmbeddingBase)`` is
    importable. ``MRotaryEmbedding`` is never instantiated on the graded
    path; if it ever were, ``__init__`` raises rather than fabricating a
    rotary cache (fail-closed).
    """

    def __init__(self, *args, **kwargs):  # noqa: D401
        raise NotImplementedError(
            "RotaryEmbeddingBase is an off-graded-path stub in this "
            "benchmark task: only the bare _triton_mrope_forward kernel is "
            "graded, and it never instantiates MRotaryEmbedding. Reaching "
            "this constructor means the off-path class was exercised -- "
            "fail-closed."
        )

    def _compute_inv_freq(self, base: float):
        raise NotImplementedError(
            "RotaryEmbeddingBase._compute_inv_freq is off the graded path "
            "(fail-closed stub)."
        )

    def _compute_cos_sin_cache(self):
        raise NotImplementedError(
            "RotaryEmbeddingBase._compute_cos_sin_cache is off the graded "
            "path (fail-closed stub)."
        )
