"""Prompt template for the upstream-not-invocable-here defer-to-golden
fix (Unit 2.7).

The resolver reports verdict_class=upstream-not-invocable-here when the
upstream wrapper cannot be imported / invoked from the isolated bench
sandbox at Phase A' (proprietary backend, C++ binding, vendor-specific
runtime that cannot be loaded without a real device, ...).

The ONLY remediation path is ``defer-to-golden`` -- the
``isolated_invocation_shim.py`` path was retired in design doc Sec. 4.4.
The LLM must patch ``meta.yaml`` so the orchestrator can mark the task
as ``SKIPPED_BY_DESIGN`` at Phase A' and route the truth source to the
Phase B golden capture (which runs on a real GPU machine outside the
bench dev box).

The 5 mandatory ``meta.yaml`` fields (Layer 5 ``validate_meta_content``
gates each):

  * ``oracle_simplified: upstream_oracle_required``
  * ``phase_a_prime.skip_invocation: true``
  * ``phase_a_prime.status: SKIPPED_BY_DESIGN``
  * ``phase_a_prime.reason: <non-empty string>``
  * ``phase_b_capture_required: true``
  * ``phase_b_capture.status: PENDING``   (enum PENDING|DONE|FAILED;
                                            NOT 'CAPTURED')
  * ``phase_b_capture.upstream_callable.module: <canonical dotted path>``
  * ``phase_b_capture.upstream_callable.function: <valid Python identifier>``

If the current ``oracle_pytorch.py`` already uses only low-level math
primitives, the patch sets ``oracle_action: unchanged`` and only meta
changes. If the oracle still contains high-level torch ops (F.gelu,
F.silu, nn.SiLU, ...), the LLM MUST also rewrite the oracle in the same
patch (``oracle_action: rewrite-primitives``) so the bench's enforced
no-high-level-op invariant is restored.
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

logger = logging.getLogger("automation.prompts.defer_to_golden")


SYSTEM_TASK = (
    "Task: the resolver reported that the bench task is BLOCKED with "
    "verdict_class=upstream-not-invocable-here -- the upstream wrapper "
    "cannot be invoked from a clean Python entry point inside the bench "
    "sandbox (proprietary backend, C++ binding, vendor runtime, ...).\n\n"
    "The ONLY valid remediation is `defer-to-golden`. The "
    "`isolated_invocation_shim.py` path was retired in design doc Sec. "
    "4.4 -- do NOT propose a shim.\n\n"
    "Five-step procedure\n"
    "===================\n"
    "  1. Read the resolver evidence and the vendored snapshot to "
    "infer `upstream_callable.function` (the bare function name, a "
    "valid Python identifier).\n"
    "  2. Infer `upstream_callable.module` as a canonical dotted Python "
    "path (e.g. `vllm.model_executor.layers.layernorm`, "
    "`sglang.srt.layers.elementwise`). Slashes, leading dots, empty "
    "segments, and digit-prefixed segments are ALL rejected by the "
    "validator.\n"
    "  3. Scan the current `oracle_pytorch.py` body for high-level "
    "torch ops (see the HIGH-LEVEL OP BLACKLIST below).\n"
    "       * If absent -> `oracle_action: unchanged`; FILE contains "
    "meta.yaml ONLY.\n"
    "       * If present -> `oracle_action: rewrite-primitives`; FILE "
    "contains BOTH meta.yaml AND a rewritten oracle_pytorch.py that "
    "expresses the operator with only LOW-LEVEL math primitives.\n"
    "  4. Emit the meta.yaml patch with ALL 5 mandatory fields populated\n"
    "     (see schema list below).\n"
    "  5. The orchestrator will revalidate Layer 1..5 before applying. "
    "If you cannot derive a clean module/function, prefer to omit the "
    "patch (the orchestrator will retry / route to human review) over "
    "emitting an invalid one -- bad dotted paths burn the turn budget.\n"
    "\n"
    "Mandatory meta.yaml fields\n"
    "==========================\n"
    "  * oracle_simplified:                   upstream_oracle_required\n"
    "  * phase_a_prime.skip_invocation:       true\n"
    "  * phase_a_prime.status:                SKIPPED_BY_DESIGN\n"
    "  * phase_a_prime.reason:                <non-empty string>\n"
    "  * phase_b_capture_required:            true\n"
    "  * phase_b_capture.status:              PENDING   "
    "(enum PENDING|DONE|FAILED; NOT 'CAPTURED')\n"
    "  * phase_b_capture.upstream_callable.module:   <canonical dotted path>\n"
    "  * phase_b_capture.upstream_callable.function: <valid Python identifier>\n"
    "\n"
    "Allowed decision values\n"
    "=======================\n"
    "  ONE valid value only: `defer-to-golden`. Any other value, "
    "including the retired `defer-to-upstream-oracle`, is rejected.\n"
    "\n"
    "FORBIDDEN OUTPUTS\n"
    "=================\n"
    "  * `oracle_action: placeholder`                (Layer 1 schema reject)\n"
    "  * function body just `raise NotImplementedError` (Layer 3 stub reject)\n"
    "  * `phase_b_capture.status` other than PENDING|DONE|FAILED (Layer 5)\n"
    "  * dotted path with `..`, `/`, leading `.` or digit-prefixed segment (Layer 5)\n"
    "  * function name containing `-`, `.` or other non-identifier chars (Layer 5)\n"
    "  * missing any of the 5 mandatory fields above (Layer 5)\n"
)


def _read_meta_raw(task_dir: Path) -> str:
    meta_path = task_dir / "meta.yaml"
    if not meta_path.is_file():
        raise FileNotFoundError(
            f"meta.yaml not found under {task_dir} -- cannot build "
            "defer-to-golden prompt without the current meta to patch"
        )
    return meta_path.read_text(encoding="utf-8")


def _read_meta_dict(meta_text: str) -> dict[str, Any]:
    try:
        data = yaml.safe_load(meta_text) or {}
    except yaml.YAMLError:
        logger.warning("meta.yaml is not valid YAML")
        return {}
    if not isinstance(data, dict):
        return {}
    return data


def _find_task_dir(repo_root: Path, task_id: str) -> Path | None:
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


def _format_upstream_hint(hint: dict[str, Any] | None) -> str:
    if not hint:
        return "(no resolver hint -- derive from meta.upstream.path + vendored header)"
    module = hint.get("module", "")
    function = hint.get("function", "")
    return (
        f"resolver hint:\n"
        f"  module:   {module!r}\n"
        f"  function: {function!r}\n"
        "(validate canonical dotted path / Python identifier before emitting)"
    )


def build_prompt(
    task_id: str,
    repo_root: Path,
    *,
    upstream_callable_hint: dict[str, Any] | None = None,
    blocker_reason: str | None = None,
) -> list[dict[str, str]]:
    """Build a ``[system, user]`` list of chat messages.

    Args:
      task_id: e.g. ``L1_01_bias``.
      repo_root: bench repo root (parent of ``tasks/``).
      upstream_callable_hint: optional ``{"module": str, "function": str}``
        candidate from the resolver. The LLM should validate and copy
        it verbatim if canonical; otherwise correct it.
      blocker_reason: optional short string describing WHY the wrapper
        is not invocable here (e.g. ``"C++ binding requires real GPU
        runtime"``); becomes the ``phase_a_prime.reason`` candidate.

    Raises:
      FileNotFoundError: if the task directory or ``meta.yaml`` is
        missing -- the prompt builder refuses to ask the LLM to patch a
        meta whose current state it cannot show.
    """
    repo_root = Path(repo_root)
    task_dir = _find_task_dir(repo_root, task_id)
    if task_dir is None:
        raise FileNotFoundError(
            f"task directory tasks/<level>/{task_id}/ not found under {repo_root}"
        )

    meta_text = _read_meta_raw(task_dir)
    meta = _read_meta_dict(meta_text)
    task_name = str(meta.get("name", task_id))
    upstream_path = str(((meta.get("upstream") or {}).get("path")) or "(absent)")
    wrapper_sig = _find_wrapper_signature(repo_root, task_id)
    rel_task = task_dir.relative_to(repo_root)

    system = (
        SYSTEM_PREFIX
        + SYSTEM_TASK
        + "\n"
        + BLOCK_GRAMMAR
        + "\n"
        + render_meta_keys_table(("defer-to-golden",))
        + "\n"
        + HIGH_LEVEL_OP_WARNING
    )

    hint_block = _format_upstream_hint(upstream_callable_hint)
    reason_block = (
        blocker_reason
        if (blocker_reason and blocker_reason.strip())
        else "(no blocker reason captured; populate phase_a_prime.reason from "
             "resolver evidence)"
    )

    user_parts = [
        f"task_id: {task_id}",
        f"task_name: {task_name}",
        f"task_dir (FILE path prefix to use): {rel_task}/",
        "",
        "## wrapper signature (from ascendbench/frozen/snapshots/, may be empty)",
        wrapper_sig,
        "",
        "## meta.upstream.path",
        upstream_path,
        "",
        "## current meta.yaml (full)",
        meta_text,
        "",
        "## upstream_callable hint",
        hint_block,
        "",
        "## blocker reason",
        reason_block,
        "",
        "## reminders",
        "  * Exactly one decision is legal: `defer-to-golden`.",
        "  * `oracle_action` is ONE OF: `unchanged` | `rewrite-primitives`.",
        "  * FILE block path MUST be `tasks/<level>/<task_id>/meta.yaml` "
        "(and optionally `oracle_pytorch.py` when "
        "`oracle_action: rewrite-primitives`).",
        "  * `phase_b_capture.status` MUST be one of PENDING|DONE|FAILED "
        "(use PENDING for new patches; DONE / FAILED are written by "
        "the Phase B capture pipeline, not by you).",
    ]
    user = "\n".join(user_parts)
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


__all__ = ["build_prompt"]
