"""Stratified task-ID selector for the acceptance gate Phase 2 sampler.

Reads a resolver artefact (any format supported by
:mod:`automation.lib.resolve_loader`), groups by ``verdict_class``,
applies a per-class cap, and emits a CSV of task IDs on stdout.

Default policy matches the legacy 5+5+(up-to-5) heuristic:
  5 shapes-incomplete + 5 oracle-mismatch + up to 5 upstream-not-invocable-here.

Use ``--classes`` and ``--cap`` to override.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Bootstrap: scripts/ on sys.path so ``automation.lib.resolve_loader``
# resolves when this file is run as ``python3 scripts/automation/select_tasks.py``.
_SCRIPTS_DIR = Path(__file__).resolve().parent.parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from automation.lib.resolve_loader import load_resolve_artifact  # noqa: E402


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Stratified task-id selector (CSV stdout).",
    )
    p.add_argument(
        "resolve_log",
        type=Path,
        help="Resolver artefact (aggregate JSON / JSONL / legacy text).",
    )
    p.add_argument(
        "--classes",
        default="shapes-incomplete,oracle-mismatch,upstream-not-invocable-here",
        help="Comma-separated verdict classes to sample, in order.",
    )
    p.add_argument(
        "--cap",
        type=int,
        default=5,
        help="Per-class cap (default: 5).",
    )
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    try:
        records = load_resolve_artifact(args.resolve_log)
    except (FileNotFoundError, ValueError) as exc:
        print(f"resolve_log unreadable: {exc}", file=sys.stderr)
        return 2
    wanted_classes = [c.strip() for c in args.classes.split(",") if c.strip()]
    buckets: dict[str, list[str]] = {c: [] for c in wanted_classes}
    seen: set[str] = set()
    for rec in records:
        vc = rec["verdict_class"]
        tid = rec["task_id"]
        if vc not in buckets:
            continue
        if tid in seen:
            continue
        buckets[vc].append(tid)
        seen.add(tid)
    selected: list[str] = []
    for c in wanted_classes:
        selected.extend(buckets[c][: args.cap])
    print(",".join(selected))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
