"""Shared prompt prefixes for the LLM-driven fix orchestrator.

Every per-decision prompt (``shapes_incomplete`` here; ``oracle_mismatch``
and ``defer_to_*`` in M2) builds on top of this shared system message.

The output protocol is the single source of truth for what the parser
in ``automation.lib.file_blocks`` expects.
"""
from __future__ import annotations

SYSTEM_PREFIX = (
    "You are a GPU Triton kernel test infrastructure engineer. Your job "
    "is to repair benchmark task metadata so an isolated PyTorch oracle "
    "can be invoked to verify a Triton kernel under test.\n\n"
)


OUTPUT_PROTOCOL = """\
OUTPUT FORMAT (STRICT)
======================

Your reply MUST contain exactly one META control block and one or more
FILE blocks. META and FILE blocks may appear in any order. Free-form
prose OUTSIDE the blocks is allowed and will be ignored (use it to
explain your decision if you wish). Free-form prose INSIDE a FILE
block is FORBIDDEN -- file blocks contain a single complete file body
and nothing else (no markdown fences, no diff hunks).

META block (exactly one). It MUST contain ALL FOUR required keys
(``decision``, ``missing_params``, ``rationale``, ``evidence``) -- a
META that drops any of these will be rejected by the patch validator
and the attempt will be retried, burning your turn budget.

Required META keys:
  * ``decision``       -- one of the allowed decision strings for this
                          turn (e.g. ``"patch-shapes"``).
  * ``missing_params`` -- JSON array of parameter names that the
                          repaired ``shapes.yaml`` now supplies. For
                          the ``patch-shapes`` decision this is the
                          subset of ``shapes.yaml`` keys that the
                          original task was missing -- e.g.
                          ``["BLOCK_SIZE_M", "BLOCK_SIZE_N",
                          "GROUP_SIZE_M"]``. Use ``[]`` only if the
                          decision is one for which no params were
                          missing (this is unusual for patch-shapes).
  * ``rationale``      -- short free-text justification (1-3 sentences).
  * ``evidence``       -- JSON object citing the call sites / wrapper
                          signatures / inferred dtypes you relied on
                          (e.g. ``{"call_sites": ["infer/vllm/model.py:73"],
                          "wrapper_sig": "def silu(x: Tensor) -> Tensor"}``).

    ### META
    {"decision": "patch-shapes",
     "missing_params": ["BLOCK_SIZE_M", "BLOCK_SIZE_N", "GROUP_SIZE_M"],
     "rationale": "Wrapper signature + caller dtypes pin the missing tile params.",
     "evidence": {"call_sites": ["infer/vllm/model.py:73"],
                  "wrapper_sig": "def silu(x: Tensor) -> Tensor"}}
    ### END

FILE blocks (one or more; the `path` MUST be a whitelisted target,
e.g. tasks/<level>/<task_id>/{shapes.yaml,meta.yaml,oracle_pytorch.py}):

    ### FILE: tasks/<level>/<task_id>/shapes.yaml
    <complete file contents>
    ### END

Do NOT emit unified diffs. Do NOT emit zero META blocks. Do NOT emit
two or more META blocks. Do NOT emit FILE paths outside the task
directory.
"""


HIGH_LEVEL_OP_WARNING = """\
HIGH-LEVEL OP BLACKLIST (when modifying oracle_pytorch.py)
=========================================================

The oracle body MUST NOT call any of the following high-level ops --
they are forbidden because the bench wants the oracle to express the
operator in terms of low-level math primitives (so the Triton kernel
under test can be validated against it).

Forbidden function names (lowercase, anywhere on the call chain):
    silu, gelu, softmax, log_softmax, sigmoid, relu, leaky_relu, elu,
    tanh, mish, hardswish, hardsigmoid, layer_norm, group_norm,
    batch_norm, rms_norm, scaled_dot_product_attention, linear,
    embedding, cross_entropy, softplus, glu

Forbidden torch.nn class instantiations (capitalised):
    SiLU, GELU, Softmax, LogSoftmax, Sigmoid, ReLU, LeakyReLU, ELU,
    Tanh, Mish, Hardswish, Hardsigmoid, LayerNorm, GroupNorm,
    BatchNorm1d/2d/3d, RMSNorm, MultiheadAttention, Linear, Embedding,
    CrossEntropyLoss, Softplus, GLU
"""


__all__ = ["HIGH_LEVEL_OP_WARNING", "OUTPUT_PROTOCOL", "SYSTEM_PREFIX"]
