"""Prompt template for the shapes-incomplete fix (Unit 1.10).

Builds an OpenAI / Anthropic compatible List[ChatMessage] tuple --
each item is ``{"role": "system|user", "content": "..."}``. The
orchestrator passes this directly to ``LLMClient.chat(...)``.
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

logger = logging.getLogger("automation.prompts.shapes_incomplete")


SYSTEM_TASK = (
    "Task: the resolver reported that the bench task is BLOCKED with "
    "verdict_class=shapes-incomplete. Your job is to write a complete "
    "shapes.yaml that lets the resolver invoke the upstream wrapper "
    "from a clean Python entrypoint.\n\n"
    "Output decision MUST be `patch-shapes`. You may also touch "
    "oracle_pytorch.py if (and only if) the wrapper signature requires "
    "an additional oracle input -- in that case obey the high-level op "
    "blacklist below.\n"
)


INPUTS_SCHEMA = """\
shapes.yaml SCHEMA (authoritative)
==================================

  size_knobs:        dict[str, int]              # e.g. {M: 16, K: 64}
  public_shapes:                                  # >= 3 entries (small/medium/large)
    - {id: s1, name: small,  knobs: {M: 16, K: 64}}
    - {id: s2, name: medium, knobs: {M: 64, K: 256}}
    - {id: s3, name: large,  knobs: {M: 256, K: 1024}}
  hidden_shapes: []                               # optional
  inputs:
    <input_name>:
      dims: ["M", "K"]                           # references size_knobs
      dtype: torch.float16                       # MUST match wrapper signature
      dist: randn                                # one of:
                                                 #   randn, rand, ones, zeros,
                                                 #   randint, choice
      # OR (alternative shapes for non-tensor inputs):
      scalar: 0.0
      scalar_seq: [1, 2, 3]
      derive: "<expression referencing other inputs>"
"""


def _read_meta(task_dir: Path) -> dict[str, Any]:
    meta_path = task_dir / "meta.yaml"
    if not meta_path.is_file():
        return {}
    try:
        return yaml.safe_load(meta_path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError:
        logger.warning("meta.yaml for %s is not valid YAML", task_dir)
        return {}


def _read_shapes(task_dir: Path) -> str:
    sp = task_dir / "shapes.yaml"
    if not sp.is_file():
        return "(shapes.yaml missing -- scaffold from scratch)"
    return sp.read_text(encoding="utf-8")


def _find_task_dir(repo_root: Path, task_id: str) -> Path | None:
    for level_dir in (repo_root / "tasks").glob("*"):
        if not level_dir.is_dir():
            continue
        cand = level_dir / task_id
        if cand.is_dir():
            return cand
    return None


def _find_wrapper_signature(repo_root: Path, task_id: str) -> str:
    """Look up ``tasks/_vendored/<task>_*.py`` for the wrapper entry
    point and pull its signature. If not present, return an explicit
    fail-loud marker -- the orchestrator should raise."""
    vendored = repo_root / "tasks" / "_vendored"
    if not vendored.is_dir():
        return "(no vendored sources under tasks/_vendored)"
    candidates = list(vendored.glob(f"{task_id}_*.py"))
    if not candidates:
        return f"(no vendored file matching {task_id}_*.py)"
    text = candidates[0].read_text(encoding="utf-8", errors="replace")
    # Try to extract the first def line + dtype annotations.
    m = re.search(r"^def\s+\w+\([^)]*\)[^:]*:", text, re.MULTILINE)
    if m:
        return f"{candidates[0].name}:\n{m.group(0)}"
    return f"(could not locate def in {candidates[0].name})"


def _grep_caller_evidence(repo_root: Path, task_id: str) -> str:
    """Look under infer/{vllm,sglang}/ for caller hits."""
    infer = repo_root / "infer"
    if not infer.is_dir():
        return "(no infer/ directory; caller evidence unavailable)"
    hits: list[str] = []
    needle = task_id.split("_", 2)[-1].lower()  # last segment, e.g. swiglu
    for backend in ("vllm", "sglang"):
        b = infer / backend
        if not b.is_dir():
            continue
        for f in b.rglob("*.py"):
            try:
                text = f.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            for ln, line in enumerate(text.splitlines(), start=1):
                if needle in line.lower():
                    hits.append(f"{f.relative_to(repo_root)}:{ln}: {line.strip()[:120]}")
            if len(hits) >= 12:
                break
        if len(hits) >= 12:
            break
    return "\n".join(hits) if hits else "(no caller hits in infer/{vllm,sglang})"


class WrapperSourceMissingError(RuntimeError):
    """Raised when no vendored wrapper source can be found -- the
    prompt builder refuses to ask the LLM to guess in that case."""


def build_prompt(
    task_id: str,
    *,
    repo_root: Path,
    missing_params_evidence: str = "",
) -> list[dict[str, str]]:
    """Build a ``[system, user]`` list of chat messages."""
    repo_root = Path(repo_root)
    task_dir = _find_task_dir(repo_root, task_id)
    if task_dir is None:
        raise FileNotFoundError(
            f"task directory tasks/<level>/{task_id}/ not found"
        )

    meta = _read_meta(task_dir)
    current_shapes = _read_shapes(task_dir)
    wrapper_sig = _find_wrapper_signature(repo_root, task_id)
    if wrapper_sig.startswith("(no "):
        raise WrapperSourceMissingError(
            f"refusing to build prompt for {task_id}: {wrapper_sig}"
        )
    caller_ev = _grep_caller_evidence(repo_root, task_id)

    rel_task = task_dir.relative_to(repo_root)

    system = (
        SYSTEM_PREFIX
        + SYSTEM_TASK
        + "\n"
        + BLOCK_GRAMMAR
        + "\n"
        + render_meta_keys_table(("patch-shapes",))
        + "\n"
        + HIGH_LEVEL_OP_WARNING
    )
    user_parts = [
        f"task_id: {task_id}",
        f"task_dir (FILE path prefix to use): {rel_task}/",
        "",
        "## meta.yaml (excerpt)",
        yaml.safe_dump({k: meta.get(k) for k in ("schema_version", "name", "kind", "phase_a_prime")
                        if k in meta}, sort_keys=False) or "(empty)",
        "",
        "## current shapes.yaml",
        current_shapes,
        "",
        "## wrapper signature (from tasks/_vendored/)",
        wrapper_sig,
        "",
        "## caller evidence (greps from infer/{vllm,sglang})",
        caller_ev,
        "",
        "## resolver missing-param evidence",
        missing_params_evidence or "(no specific evidence captured)",
        "",
        "## inputs schema (use exactly these keys)",
        INPUTS_SCHEMA,
        "",
        "Constraints:",
        "  * Emit at least 3 entries in public_shapes (small/medium/large).",
        "  * Default to the smallest tensor sizes that still exercise the kernel.",
        "  * Tensor dtype MUST match what the wrapper signature requires; "
        "    do not default to fp32 if the caller passes fp16/bf16.",
        "  * Decision MUST be patch-shapes.",
    ]
    user = "\n".join(user_parts)
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


__all__ = ["WrapperSourceMissingError", "build_prompt"]
