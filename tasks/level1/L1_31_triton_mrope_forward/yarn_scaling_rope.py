"""Fail-closed task-local stub for the OFF-GRADED-PATH upstream sibling
``vllm/model_executor/layers/rotary_embedding/yarn_scaling_rope.py``.

The vendored ``kernel.py`` does
``from .yarn_scaling_rope import YaRNScalingRotaryEmbedding, yarn_get_mscale``
at module scope. Both symbols are used ONLY inside ``MRotaryEmbedding``
(its ``__init__`` mscale computation and the YaRN ``_compute_*`` overrides)
-- a class that is NEVER instantiated on the graded path. The graded op is
the bare ``@triton.jit`` ``_triton_mrope_forward`` kernel, which uses
neither symbol.

Importing the real upstream ``yarn_scaling_rope.py`` re-drags the full
``rotary_embedding`` package (base/common), re-triggering the
``flashinfer_rotary_embedding`` custom-op registration that crashes a
second ``kernel.py`` exec with a duplicate-registration error. This stub
severs that off-path import (task_dir wins the relative-import resolution)
and provides import-time placeholders that raise on real use (fail-closed
-- the graded path never calls them).
"""
from __future__ import annotations

import torch.nn as nn


def yarn_get_mscale(scale: float = 1.0, mscale: float = 1.0) -> float:
    """Off-graded-path stub. ``MRotaryEmbedding`` calls this only when a
    YaRN ``scaling_factor`` is given; the graded kernel never does."""
    raise NotImplementedError(
        "yarn_get_mscale is off the graded path in this benchmark task "
        "(the graded op is the bare _triton_mrope_forward kernel, which "
        "never computes a YaRN mscale) -- fail-closed stub."
    )


class YaRNScalingRotaryEmbedding(nn.Module):
    """Off-graded-path stub. Referenced only by ``MRotaryEmbedding``'s
    YaRN ``_compute_inv_freq`` / ``_compute_cos_sin_cache`` overrides,
    which the graded path never reaches."""

    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "YaRNScalingRotaryEmbedding is off the graded path (fail-closed "
            "stub); the graded _triton_mrope_forward kernel never uses YaRN."
        )

    def _compute_inv_freq(self, base: float):
        raise NotImplementedError(
            "YaRNScalingRotaryEmbedding._compute_inv_freq is off the graded "
            "path (fail-closed stub)."
        )

    def _compute_cos_sin_cache(self):
        raise NotImplementedError(
            "YaRNScalingRotaryEmbedding._compute_cos_sin_cache is off the "
            "graded path (fail-closed stub)."
        )
