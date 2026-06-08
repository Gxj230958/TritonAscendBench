"""Prompt template for the oracle-mismatch fix (Unit 2.6).

The resolver reports verdict_class=oracle-mismatch when the bench task
ran the isolated PyTorch oracle but it disagreed numerically with the
upstream kernel (or with the captured solution output).

Three remediation paths are allowed -- they match the canonical decision
set understood by ``automation.lib.file_blocks``:

  * ``tolerance-bump``           -- numerical gap is within reduction-class
                                    tolerance; only ``meta.yaml`` changes
                                    (e.g. tolerance_class elementwise -> reduction).
  * ``rewrite-primitives``       -- oracle is structurally wrong; rewrite
                                    ``oracle_pytorch.py`` to mirror upstream
                                    using LOW-LEVEL math primitives only
                                    (no F.gelu / F.softmax / nn.SiLU / ...).
  * ``defer-to-upstream-oracle`` -- the PyTorch oracle CANNOT faithfully
                                    reproduce the upstream kernel from a
                                    clean Python entry point (proprietary
                                    backend, C++ binding, etc.); the LLM
                                    must populate the 5-field defer schema
                                    in meta.yaml and the orchestrator will
                                    park the patch behind a human-review
                                    flag before applying.

This builder returns the chat messages list that the orchestrator hands
to ``LLMClient.chat(...)``.
"""
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

import yaml

from automation.prompts.shared import (
    BLOCK_GRAMMAR,
    HIGH_LEVEL_OP_WARNING,
    OUTPUT_PROTOCOL,  # noqa: F401  -- kept for backwards-compat consumers
    SYSTEM_PREFIX,
    render_meta_keys_table,
)

logger = logging.getLogger("automation.prompts.oracle_mismatch")


# Maximum number of mismatch evidence lines we will splice into the user
# prompt. Anything beyond is truncated with an explicit marker so the LLM
# does not get a confusing tail with no context.
MAX_MISMATCH_EVIDENCE_LINES = 8

# Cap on the oracle log excerpt. A long pa_oracle.log can blow the prompt
# budget; we keep the most recent slice because the tail is where the
# numerical failure surfaces.
MAX_ORACLE_LOG_CHARS = 500


SYSTEM_TASK = (
    "Task: the resolver reported that the bench task is BLOCKED with "
    "verdict_class=oracle-mismatch -- the isolated PyTorch oracle "
    "disagreed numerically with the upstream kernel.\n\n"
    "Pick exactly ONE of the three remediation paths below and emit a "
    "patch whose META `decision` is the corresponding string. The "
    "validator will reject any other decision string and burn your turn "
    "budget.\n\n"
    "DECISION TREE\n"
    "=============\n"
    "  Path 1 -- `tolerance-bump`\n"
    "      Trigger: max_rel_diff is small (<= 1e-3) AND dtype is "
    "fp16/bf16; the oracle is structurally right but the tolerance "
    "class is too tight (elementwise vs reduction).\n"
    "      Patch: meta.yaml ONLY (bump `tolerance_class`). Do NOT "
    "include oracle_pytorch.py in this case.\n\n"
    "  Path 2 -- `rewrite-primitives`\n"
    "      Trigger: the gap exceeds tolerance and the oracle is "
    "structurally wrong; the formula must be rewritten to mirror the "
    "upstream kernel.\n"
    "      Patch: oracle_pytorch.py REQUIRED. Express the operator in "
    "LOW-LEVEL math primitives only (see the HIGH-LEVEL OP BLACKLIST "
    "below). Do not call F.silu / F.gelu / F.softmax / nn.SiLU / etc.\n\n"
    "  Path 3 -- `defer-to-upstream-oracle`\n"
    "      Trigger: the PyTorch oracle cannot faithfully reproduce the "
    "upstream kernel from a clean Python entry point (proprietary "
    "backend, C++ binding, vendor-specific runtime, or oracle is a "
    "deliberately simplified model).\n"
    "      Patch: meta.yaml MUST set the full 5-field defer schema:\n"
    "        * `oracle_simplified: upstream_oracle_required`\n"
    "        * `phase_a_prime.skip_invocation: true`\n"
    "        * `phase_a_prime.status: SKIPPED_BY_DESIGN`\n"
    "        * `phase_a_prime.reason: <non-empty string>`\n"
    "        * `phase_b_capture_required: true`\n"
    "        * `phase_b_capture.status: PENDING`  (NOT 'CAPTURED')\n"
    "        * `phase_b_capture.upstream_callable.module: "
    "<canonical dotted path>`\n"
    "        * `phase_b_capture.upstream_callable.function: "
    "<valid Python identifier>`\n"
    "      The orchestrator will park this patch behind a "
    "needs-human-review.flag; it is NOT applied automatically.\n\n"
    "If you choose Path 3 with `oracle_action: rewrite-primitives` "
    "(downgrading the oracle while deferring the truth source), the "
    "candidate oracle_pytorch.py is ALSO scanned for high-level ops -- "
    "no leftover F.gelu / F.silu / nn.SiLU / etc.\n\n"
    "FORBIDDEN OUTPUTS\n"
    "=================\n"
    "  * `oracle_action: placeholder`                (Layer 1 schema reject)\n"
    "  * function body just `pass` / `...`           (Layer 3 stub reject)\n"
    "  * function body just `raise NotImplementedError` (Layer 3 stub reject)\n"
    "  * `torch.empty_like(x)` as the entire body    (Layer 3 stub reject)\n"
    "  * any high-level op leak in a rewrite-primitives oracle (Layer 5)\n"
    "  * missing any of the 5 defer fields for `defer-to-upstream-oracle` (Layer 5)\n"
    "  * `phase_b_capture.status` other than PENDING|DONE|FAILED (Layer 5)\n"
)


def _read_meta(task_dir: Path) -> dict[str, Any]:
    meta_path = task_dir / "meta.yaml"
    if not meta_path.is_file():
        return {}
    try:
        return yaml.safe_load(meta_path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError:
        logger.warning("meta.yaml for %s is not valid YAML", task_dir)
        return {}


def _read_oracle(task_dir: Path) -> str:
    op = task_dir / "oracle_pytorch.py"
    if not op.is_file():
        raise FileNotFoundError(
            f"oracle_pytorch.py not found under {task_dir} -- cannot build "
            "oracle-mismatch prompt without the current oracle body"
        )
    return op.read_text(encoding="utf-8")


def _find_task_dir(repo_root: Path, task_id: str) -> Path | None:
    """Locate ``tasks/<level>/<task_id>/`` under ``repo_root``.

    The bench keeps tasks under ``tasks/level1`` / ``tasks/level2`` / etc.
    so a glob over the ``tasks/`` parent is the canonical lookup.
    """
    tasks_root = repo_root / "tasks"
    if not tasks_root.is_dir():
        return None
    for level_dir in tasks_root.glob("*"):
        if not level_dir.is_dir():
            continue
        cand = level_dir / task_id
        if cand.is_dir():
            return cand
    return None


def _find_wrapper_signature(repo_root: Path, task_id: str) -> str:
    """Extract the first ``def`` line from a matching vendored file.

    Returns an explicit ``(no ...)`` marker (not an exception) when no
    candidate is found -- callers may choose to ignore it because the
    oracle-mismatch prompt can still proceed without the upstream
    signature (the oracle body itself is the primary evidence).
    """
    vendored = repo_root / "tasks" / "_vendored"
    if not vendored.is_dir():
        return "(no vendored sources under ascendbench/frozen/snapshots)"
    candidates = list(vendored.glob(f"{task_id}_*.py"))
    if not candidates:
        return f"(no vendored file matching {task_id}_*.py)"
    text = candidates[0].read_text(encoding="utf-8", errors="replace")
    m = re.search(r"^def\s+\w+\([^)]*\)[^:]*:", text, re.MULTILINE)
    if m:
        return f"{candidates[0].name}:\n{m.group(0)}"
    return f"(could not locate def in {candidates[0].name})"


def _truncate_evidence(lines: list[str]) -> str:
    """Truncate mismatch tensor description to MAX_MISMATCH_EVIDENCE_LINES.

    Appends an explicit `(truncated; N more lines)` marker when the
    original list was longer so the LLM does not silently lose context.
    """
    if len(lines) <= MAX_MISMATCH_EVIDENCE_LINES:
        return "\n".join(lines)
    kept = lines[:MAX_MISMATCH_EVIDENCE_LINES]
    extra = len(lines) - MAX_MISMATCH_EVIDENCE_LINES
    return "\n".join(kept) + f"\n(truncated; {extra} more line(s))"


def _truncate_oracle_log(text: str) -> str:
    if len(text) <= MAX_ORACLE_LOG_CHARS:
        return text
    # Keep the tail -- numerical failures surface in the last few lines.
    keep = text[-MAX_ORACLE_LOG_CHARS:]
    return f"(truncated; showing last {MAX_ORACLE_LOG_CHARS} chars)\n{keep}"


def build_prompt(
    task_id: str,
    repo_root: Path,
    *,
    oracle_log: str | None = None,
    mismatch_evidence: list[str] | None = None,
) -> list[dict[str, str]]:
    """Build a ``[system, user]`` list of chat messages.

    Args:
      task_id: e.g. ``L1_01_bias``.
      repo_root: bench repo root (parent of ``tasks/``).
      oracle_log: optional ``pa_oracle.log`` excerpt; capped to
        ``MAX_ORACLE_LOG_CHARS`` characters from the tail.
      mismatch_evidence: optional list of mismatch tensor description
        lines (one tensor or call site per entry); capped to
        ``MAX_MISMATCH_EVIDENCE_LINES`` with an explicit truncation marker.

    Raises:
      FileNotFoundError: if the task directory or ``oracle_pytorch.py``
        is missing -- the prompt builder refuses to ask the LLM to fix
        an oracle whose current state it cannot show.
    """
    repo_root = Path(repo_root)
    task_dir = _find_task_dir(repo_root, task_id)
    if task_dir is None:
        raise FileNotFoundError(
            f"task directory tasks/<level>/{task_id}/ not found under {repo_root}"
        )

    meta = _read_meta(task_dir)
    task_name = str(meta.get("name", task_id))
    oracle_src = _read_oracle(task_dir)
    wrapper_sig = _find_wrapper_signature(repo_root, task_id)

    rel_task = task_dir.relative_to(repo_root)

    # Compose system message: shared prefix + decision tree + block
    # grammar + the THREE oracle-mismatch decision rows of the keys
    # table (tolerance-bump, rewrite-primitives, defer-to-upstream-oracle)
    # + high-level op blacklist (used by rewrite-primitives and the
    # optional rewrite under Path 3).
    system = (
        SYSTEM_PREFIX
        + SYSTEM_TASK
        + "\n"
        + BLOCK_GRAMMAR
        + "\n"
        + render_meta_keys_table((
            "tolerance-bump",
            "rewrite-primitives",
            "defer-to-upstream-oracle",
        ))
        + "\n"
        + HIGH_LEVEL_OP_WARNING
    )

    evidence_block = _truncate_evidence(list(mismatch_evidence or []))
    if not evidence_block:
        evidence_block = "(no mismatch evidence captured)"
    log_block = _truncate_oracle_log(oracle_log) if oracle_log else "(no oracle log)"

    user_parts = [
        f"task_id: {task_id}",
        f"task_name: {task_name}",
        f"task_dir (FILE path prefix to use): {rel_task}/",
        "",
        "## wrapper signature (from ascendbench/frozen/snapshots/, may be empty)",
        wrapper_sig,
        "",
        "## current oracle_pytorch.py",
        oracle_src,
        "",
        "## mismatch tensor evidence",
        evidence_block,
        "",
        "## pa_oracle.log excerpt",
        log_block,
        "",
        "## reminders",
        "  * Pick exactly one decision: tolerance-bump | "
        "rewrite-primitives | defer-to-upstream-oracle.",
        "  * `tolerance-bump`           -> FILE meta.yaml ONLY.",
        "  * `rewrite-primitives`       -> FILE oracle_pytorch.py "
        "REQUIRED (low-level math only).",
        "  * `defer-to-upstream-oracle` -> FILE meta.yaml with the full "
        "5-field defer schema; `phase_b_capture.status: PENDING`.",
        "  * If you set `oracle_action: rewrite-primitives` under "
        "Path 3, the candidate oracle_pytorch.py is ALSO scanned for "
        "high-level op leaks.",
    ]
    user = "\n".join(user_parts)
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


__all__ = [
    "MAX_MISMATCH_EVIDENCE_LINES",
    "MAX_ORACLE_LOG_CHARS",
    "build_prompt",
]
