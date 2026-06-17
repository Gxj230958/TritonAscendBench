#!/usr/bin/env python3
"""Phase-1 visibility: report per-task DEAD size knobs (declared but referenced by no input
`dims`/`dist`/`derive`/`scalar_knob`). This is a REPORT, not a hard gate -- in the knob-grid format
~60% of declared knobs are dead boilerplate today, so failing on them would RED the whole corpus; the
*hard* "no dead public knobs" enforcement lands in Phase 2 on the explicit format. Until then this
surfaces the cleanup target.

Reuses the (dist-aware, U1-fixed) active-knob detector so a knob used only inside a `dist` string is
NOT mis-counted as dead.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from scripts.corpus.extend_task_shapes import _active_knobs_in_order  # noqa: E402


def dead_knobs(shapes_doc: dict) -> list[str]:
    declared = list((shapes_doc.get("size_knobs") or {}).keys())
    active = set(_active_knobs_in_order(shapes_doc))
    return [k for k in declared if k not in active]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tasks-root", default="tasks", help="corpus root")
    ap.add_argument("--json-out", help="write the full per-task report JSON here")
    args = ap.parse_args()

    report, tot_decl, tot_dead = {}, 0, 0
    for sp in sorted(Path(args.tasks_root).rglob("shapes.yaml")):
        if sp.parent.name.startswith("C1_"):
            continue
        try:
            doc = yaml.safe_load(sp.read_text())
        except Exception:
            continue
        if not isinstance(doc, dict) or not doc.get("size_knobs"):
            continue
        dead = dead_knobs(doc)
        decl = len(doc["size_knobs"])
        tot_decl += decl
        tot_dead += len(dead)
        report[sp.parent.name] = {"declared": decl, "dead": dead}

    lean = sum(1 for r in report.values() if not r["dead"])
    print(f"tasks={len(report)} declared_knobs={tot_decl} dead_knobs={tot_dead} "
          f"({tot_dead / max(tot_decl, 1):.0%}) lean_tasks={lean}")
    for name, r in sorted(report.items(), key=lambda kv: -len(kv[1]["dead"]))[:15]:
        if r["dead"]:
            print(f"  {name}: {len(r['dead'])}/{r['declared']} dead -> {r['dead']}")
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
