"""Shared prompt prefixes for the LLM-driven fix orchestrator.

Every per-decision prompt (``shapes_incomplete``, ``oracle_mismatch``,
``defer_to_golden``) builds on top of these shared building blocks.

The output protocol is split into two parts so individual prompts can
embed only the keys list relevant to their decision space:

  * ``BLOCK_GRAMMAR``         -- the META + FILE block syntax (decision
                                 agnostic).
  * ``META_KEYS_BY_DECISION`` -- per-decision required META key table,
                                 the single source of truth that mirrors
                                 ``fix_orchestrator._DECISION_TO_REQUIRED_KEYS``.

The legacy ``OUTPUT_PROTOCOL`` constant is preserved as
``BLOCK_GRAMMAR + a full keys table`` for backwards compatibility with
any consumer that still wants the all-decisions superset; new prompts
should compose ``BLOCK_GRAMMAR`` with only the keys list relevant to
their decision.
"""
from __future__ import annotations

SYSTEM_PREFIX = (
    "You are a GPU Triton kernel test infrastructure engineer. Your job "
    "is to repair benchmark task metadata so an isolated PyTorch oracle "
    "can be invoked to verify a Triton kernel under test.\n\n"
)


# ---------------------------------------------------------------------------
# META keys table -- single source of truth, mirrors
# ``automation.fix_orchestrator._DECISION_TO_REQUIRED_KEYS``. A
# regression test guards the two from drifting apart.
# ---------------------------------------------------------------------------
META_KEYS_BY_DECISION: dict[str, tuple[str, ...]] = {
    "patch-shapes": (
        "decision", "missing_params", "rationale", "evidence",
    ),
    "tolerance-bump": (
        "decision", "rationale", "evidence", "tolerance_delta",
    ),
    "rewrite-primitives": (
        "decision", "rationale", "evidence", "primitives_changed",
    ),
    "defer-to-upstream-oracle": (
        "decision", "rationale", "evidence", "upstream_callable",
    ),
    "defer-to-golden": (
        "decision", "rationale", "evidence",
        "upstream_callable", "blocker_reason",
    ),
}


def render_meta_keys_table(decisions: tuple[str, ...] | None = None) -> str:
    """Render a markdown table listing the required META keys for the
    given subset of decisions (or all five if ``decisions`` is None).

    Each per-decision prompt embeds only the rows relevant to its
    decision space -- this gives the LLM a concise, decision-specific
    keys reference inside the system message.
    """
    if decisions is None:
        decisions = tuple(META_KEYS_BY_DECISION.keys())
    lines: list[str] = [
        "Required META keys per decision",
        "-------------------------------",
        "",
        "| decision | required META keys |",
        "|---|---|",
    ]
    for dec in decisions:
        keys = META_KEYS_BY_DECISION[dec]
        lines.append(f"| {dec} | {', '.join(keys)} |")
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Block grammar -- META + FILE syntax, decision agnostic.
# ---------------------------------------------------------------------------
BLOCK_GRAMMAR = """\
OUTPUT FORMAT (STRICT)
======================

Your reply MUST contain exactly one META control block and one or more
FILE blocks. META and FILE blocks may appear in any order. Free-form
prose OUTSIDE the blocks is allowed and will be ignored (use it to
explain your decision if you wish). Free-form prose INSIDE a FILE
block is FORBIDDEN -- file blocks contain a single complete file body
and nothing else (no markdown fences, no diff hunks).

META block (exactly one). It MUST carry every required key for the
decision you choose -- see the per-decision keys table below. A META
that drops a required key will be rejected by the patch validator and
the attempt will be retried, burning your turn budget.

META block syntax:

    ### META
    {"decision": "<one of the allowed decision strings>", ...}
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


# ---------------------------------------------------------------------------
# Backwards-compat: the legacy combined string. New prompts should
# compose ``BLOCK_GRAMMAR`` + ``render_meta_keys_table(<subset>)``
# directly instead of relying on this all-decisions superset.
# ---------------------------------------------------------------------------
OUTPUT_PROTOCOL = (
    BLOCK_GRAMMAR
    + "\n"
    + render_meta_keys_table()
)


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


__all__ = [
    "BLOCK_GRAMMAR",
    "HIGH_LEVEL_OP_WARNING",
    "META_KEYS_BY_DECISION",
    "OUTPUT_PROTOCOL",
    "SYSTEM_PREFIX",
    "render_meta_keys_table",
]
