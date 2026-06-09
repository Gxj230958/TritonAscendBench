"""Unified loader for resolver artefacts.

Supports three on-disk formats produced by ``scripts/verify/resolve_upstream_entrypoints.py``
historically and currently:

  1. Aggregate JSON: {"_provenance": {...}, "verdicts": [{task_id, status, reason, ...}, ...]}
  2. JSONL: one JSON object per line, each shaped like a verdicts[i]
  3. Legacy plain-text: lines like '[<task_id>] BLOCKED - :: BLOCKED(no-verified-upstream-entrypoint: <verdict_class>)'

All three return a list[dict] with at minimum:
  - task_id: str
  - verdict_class: str ("PASS" | "shapes-incomplete" | "oracle-mismatch" |
                       "upstream-not-invocable-here" | "unexpected-blocked" |
                       "smoke-fail" | other future)
  - status: str (raw line/JSON status field for evidence extraction)
  - reason: str (optional, '' if absent)
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

logger = logging.getLogger("automation.lib.resolve_loader")

_BLOCKED_STATUS_RE = re.compile(r"^BLOCKED\s*\(\s*no-verified-upstream-entrypoint:\s*([\w-]+)\s*\)")
_LEGACY_LINE_RE = re.compile(r"^\[(?P<task_id>[^\]]+)\]\s+(?P<rest>.+)$")

# Canonical verdict_class enum recognised by the loader. Mirrors
# ``automation.state_machine.CANONICAL_VERDICT_CLASSES`` but includes
# ``PASS`` (which is not a "blocked" class so it does not appear in the
# state machine's blocked-class enum). Kept here as a local literal so
# the loader does not depend on the state machine module (and to avoid
# a circular import when the state machine imports the loader in tests).
CANONICAL_VERDICTS = {
    "PASS",
    "shapes-incomplete",
    "oracle-mismatch",
    "upstream-not-invocable-here",
    "unexpected-blocked",
    "smoke-fail",
}


def _normalize_verdict_class(status: str) -> str:
    """Extract canonical verdict_class from a status string.

    Returns 'PASS' for PASS-style, the inner verdict_class for
    'BLOCKED(no-verified-upstream-entrypoint: ...)' style, or
    'unexpected-blocked' fallback for unknown statuses.
    """
    status = status.strip()
    if status == "PASS" or status.startswith("PASS"):
        return "PASS"
    m = _BLOCKED_STATUS_RE.match(status)
    if m:
        return m.group(1)
    # Fallback heuristic: split on first ':' to recover legacy free-form
    if ":" in status:
        candidate = status.split(":", 1)[1].strip().rstrip(")").strip()
        if candidate in CANONICAL_VERDICTS:
            return candidate
    return "unexpected-blocked"


def _normalize_record(rec: dict) -> dict:
    """Normalize one record (from JSON aggregate or JSONL) into the standard shape."""
    status = str(rec.get("status", ""))
    verdict_class = _normalize_verdict_class(status)
    return {
        "task_id": str(rec["task_id"]),
        "verdict_class": verdict_class,
        "status": status,
        "reason": str(rec.get("reason", "")),
    }


def load_resolve_artifact(path: Path | str) -> list[dict]:
    """Load resolver output regardless of format.

    Detection order:
      1. Try parsing whole file as aggregate JSON (must have 'verdicts' key)
      2. Fall back to JSONL (one JSON object per non-empty line)
      3. Fall back to legacy plain-text line parsing
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"resolve artefact not found: {p}")
    text = p.read_text(encoding="utf-8")
    if not text.strip():
        raise ValueError(f"resolve artefact empty: {p}")

    # 1. Aggregate JSON
    try:
        data = json.loads(text)
        if isinstance(data, dict) and "verdicts" in data:
            verdicts = data["verdicts"]
            if not isinstance(verdicts, list):
                raise ValueError(f"verdicts key is not a list in {p}")
            return [_normalize_record(rec) for rec in verdicts if isinstance(rec, dict) and "task_id" in rec]
        # If it parsed as JSON but had no 'verdicts' key, fall through to
        # JSONL/text fallbacks (which will likely produce 0 records).
    except json.JSONDecodeError:
        pass

    # 2. JSONL
    records: list[dict] = []
    jsonl_ok = True
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            jsonl_ok = False
            continue
        if isinstance(rec, dict) and "task_id" in rec:
            records.append(_normalize_record(rec))
    if records and jsonl_ok:
        return records
    if records:
        # Mixed JSONL + garbage: keep the valid records, drop bad lines silently.
        return records

    # 3. Legacy plain-text
    records = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        m = _LEGACY_LINE_RE.match(line)
        if not m:
            continue
        task_id = m.group("task_id")
        rest = m.group("rest")
        # rest format: 'RESOLVED <ep> :: PASS' or 'BLOCKED - :: BLOCKED(...)'
        # Pull status portion after the last '::'.
        status = rest.split("::", 1)[-1].strip() if "::" in rest else rest
        verdict_class = _normalize_verdict_class(status)
        records.append(
            {
                "task_id": task_id,
                "verdict_class": verdict_class,
                "status": status,
                "reason": "",
            }
        )
    return records


__all__ = [
    "CANONICAL_VERDICTS",
    "load_resolve_artifact",
]
