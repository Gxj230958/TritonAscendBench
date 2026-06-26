#!/usr/bin/env python3
"""G7 — pre-run offline-golden completeness report for the NPU verified suite.

A full verified-mode NPU run needs the offline upstream-kernel golden for every
``upstream_oracle_required`` task; eval ALREADY fails-closed per task on a
missing/incomplete golden (``verified_golden_incomplete -> blocked``). This
report is the pre-run convenience: it tells you UP FRONT which of those tasks
have a complete golden set and which would block, so a missing golden is not
discovered one-task-at-a-time mid-suite. It is read-only and changes nothing.

Completeness uses the SAME predicate as the eval loader
(``GoldenStore.is_complete`` -> ``eligibility.golden_artifact_usable``), so the
report can never disagree with what the run will actually do.

Usage:
    python3 scripts/provenance/golden_completeness_report.py [--json]
"""
from __future__ import annotations

import argparse
import json
import sys

UPSTREAM_ORACLE_TIER = "upstream_oracle_required"


def report_completeness(tasks, is_complete_fn) -> dict[str, list[str]]:
    """Partition the given tasks into complete / missing golden sets.

    ``tasks`` is any iterable of objects exposing ``.task_id`` and
    ``.oracle_tier``; only ``upstream_oracle_required`` tasks are considered
    (others have an in-process oracle and need no offline golden).
    ``is_complete_fn(task) -> bool`` is the completeness predicate (production
    passes ``GoldenStore.is_complete``)."""
    complete: list[str] = []
    missing: list[str] = []
    for t in tasks:
        if getattr(t, "oracle_tier", None) != UPSTREAM_ORACLE_TIER:
            continue
        (complete if is_complete_fn(t) else missing).append(t.task_id)
    return {"complete": sorted(complete), "missing": sorted(missing)}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", action="store_true",
                    help="emit the report as JSON instead of a table")
    args = ap.parse_args(argv)

    from ascendbench.corpus.dataset import iter_all_tasks
    from ascendbench.correctness.golden_store import GoldenStore

    rep = report_completeness(iter_all_tasks(), GoldenStore.is_complete)
    total = len(rep["complete"]) + len(rep["missing"])

    if args.json:
        print(json.dumps(rep, indent=2))
    else:
        print(f"upstream_oracle_required golden completeness: "
              f"{len(rep['complete'])}/{total} complete")
        for tid in rep["missing"]:
            print(f"  MISSING: {tid}")
        if not rep["missing"]:
            print("  all goldens present — verified suite will not block on golden")
    # exit 1 if any golden is missing, so CI / a pre-run gate can react.
    return 1 if rep["missing"] else 0


if __name__ == "__main__":
    sys.exit(main())
