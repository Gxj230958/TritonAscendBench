"""Phase A' standalone shim for
``vllm.model_executor.layers.fused_moe.modular_kernel``.

When the rewriter severs a frozen kernel.py's
``import vllm.model_executor.layers.fused_moe.modular_kernel as mk``, the
only reference into this module on the corpus is the def-time annotation
``mk.ExpertTokensMetadata | None`` (deep_gemm_utils.py:45,367). Python
evaluates that annotation at function-definition time (the frozen kernel
has no ``from __future__ import annotations``), so ``mk.ExpertTokensMetadata``
must resolve to a real class for the module body to import standalone.

The graded entrypoint ``_fwd_kernel_ep_scatter_1`` (driven by
_pa_launcher.py) is annotated with NONE of these types and never
instantiates ExpertTokensMetadata -- it is reached only via the
off-graded-path host wrappers ``compute_aligned_M`` /
``deepgemm_moe_permute``, neither of which the launcher calls. This shim
therefore provides the dataclass shape verbatim from upstream
(modular_kernel.py:97) so the annotation + any off-path field access
resolves, without pulling vllm's modular_kernel closure into the severed
standalone. It carries no behaviour beyond the dataclass fields.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class ExpertTokensMetadata:
    """Verbatim field shape of
    ``vllm.model_executor.layers.fused_moe.modular_kernel.ExpertTokensMetadata``
    (upstream commit 18f6bf5, modular_kernel.py:97). Used only as a
    def-time type annotation by the frozen kernel's off-graded-path host
    wrappers; the graded scatter kernel never constructs or reads it."""

    expert_num_tokens: torch.Tensor
    expert_num_tokens_cpu: torch.Tensor | None

    @staticmethod
    def make_from_list(
        expert_num_tokens_list: list[int], device: str
    ) -> "ExpertTokensMetadata":
        expert_num_tokens_cpu = torch.tensor(
            expert_num_tokens_list, device="cpu", dtype=torch.int32
        )
        return ExpertTokensMetadata(
            expert_num_tokens=expert_num_tokens_cpu.to(device, non_blocking=True),
            expert_num_tokens_cpu=expert_num_tokens_cpu,
        )


# ───────────────────────────────────────────────────────────────────────────
# Off-graded-path modular_kernel symbols for L1_28_pack_bitmatrix. The frozen
# kernel.py is upstream's whole gpt_oss_triton_kernels_moe.py; the graded op is
# the tiny @triton.jit ``pack_bitmatrix`` driven by _pa_launcher.py::run. The
# MoE expert classes below it (BaseOAITritonExperts(mk.FusedMoEExpertsModular),
# OAITritonMxfp4ExpertsMonolithic(mk.FusedMoEExpertsMonolithic)) are imported
# at module-load time as CLASS BASES, and ``mk.TopKWeightAndReduce`` /
# ``mk.FusedMoEActivationFormat`` are read as DEF-TIME TYPE ANNOTATIONS (the
# frozen file has no ``from __future__ import annotations``). None of these
# enter the graded pack_bitmatrix path. The shim provides them as plain inert
# classes so the module body imports standalone, without pulling vllm's
# modular_kernel closure (the severed audit forbids any vllm/sglang module).
# Construction/use is a no-op base; nothing here runs on the graded path.
# ───────────────────────────────────────────────────────────────────────────
class FusedMoEExpertsModular:
    """Inert stand-in for
    ``vllm...modular_kernel.FusedMoEExpertsModular`` -- the base class of the
    frozen file's off-path ``BaseOAITritonExperts``. A plain ``type`` (no
    metaclass, inherits only ``object``) so the subclass statement executes at
    import time; the subclass is never instantiated on the graded path."""


class FusedMoEExpertsMonolithic:
    """Inert stand-in for
    ``vllm...modular_kernel.FusedMoEExpertsMonolithic`` -- the base class of the
    frozen file's off-path ``OAITritonMxfp4ExpertsMonolithic``. Same rationale
    as ``FusedMoEExpertsModular``."""


class TopKWeightAndReduce:
    """Inert stand-in for ``vllm...modular_kernel.TopKWeightAndReduce`` -- an
    abstract base used by the frozen file ONLY as a def-time return-type
    annotation on the off-path ``finalize_weight_and_reduce_impl``. Resolving
    the name to a class satisfies the annotation; never reached on the graded
    path."""


class FusedMoEActivationFormat:
    """Inert stand-in for ``vllm...modular_kernel.FusedMoEActivationFormat`` --
    an enum used by the frozen file ONLY as a def-time return-type annotation on
    the off-path ``activation_format`` properties. A bare class satisfies the
    annotation read; never reached on the graded path."""
