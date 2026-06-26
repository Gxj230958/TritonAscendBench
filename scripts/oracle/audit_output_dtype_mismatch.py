#!/usr/bin/env python3
"""G6(iii) — report-only output-dtype-mismatch audit over the GPU C2 verdicts.

The GPU oracle-equivalence grader (``run_oracle_equivalence_gpu.py``) already
RECORDS, per shape, every reference-vs-kernel output-dtype mismatch in
``ShapeVerdict.bad_dtype_keys`` (line ~994). The ceiling-acceptance path
fail-closes on those keys (``_ceiling_*`` line ~91), but a normal ``all_pass``
shape can still land ``status == "VERIFIED"`` (line ~1062) DESPITE a non-empty
``bad_dtype_keys`` — the value allclose runs after both sides are fp32-cast, so a
fp32-vs-bf16 output passes on values even though the dtype differs.

This script is the REPORT-ONLY half of the migration output-dtype gate (G6): it
scans the persisted verdict JSONs and surfaces exactly the VERIFIED shapes whose
output dtype does not match the reference. That set is what WOULD flip to FAIL if
the NPU/migration output-dtype gate (``tolerance.compare(...,
require_output_dtype_match=True)``) were ever promoted onto the shared GPU C2
path. It changes NOTHING — no verdict is mutated, no task is dropped from the
denominator (no subset escape). It exists so the fp32-fixture vs bf16-contract
decision is made on concrete evidence, not guesswork.

The verdict JSONs are GPU-produced (the daemon writes one ``<task_id>.json`` per
task via ``serialize_verdict``); this reader is dev-side and offline.

Usage:
    python3 scripts/oracle/audit_output_dtype_mismatch.py <verdict_dir> [--json]
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class FlaggedShape:
    task_id: str
    shape_id: str
    status: str
    bad_dtype_keys: list[str] = field(default_factory=list)


def audit_verdicts(verdicts: list[dict]) -> list[FlaggedShape]:
    """Return the VERIFIED shapes that carry a non-empty ``bad_dtype_keys`` —
    i.e. shapes that pass on values but whose output dtype differs from the
    reference, and so would flip under the migration output-dtype gate.

    Only ``status == "VERIFIED"`` shapes are flagged: a FAILED / *_BLOCKED /
    CEILING_ACCEPTED shape is already non-VERIFIED (the ceiling path already
    fail-closes on bad dtype), so it cannot "flip" — it is not a hidden pass.
    """
    flagged: list[FlaggedShape] = []
    for v in verdicts:
        task_id = v.get("task_id", "<unknown>")
        for sh in v.get("shapes", []) or []:
            bad = sh.get("bad_dtype_keys") or []
            if sh.get("status") == "VERIFIED" and bad:
                flagged.append(FlaggedShape(
                    task_id=task_id,
                    shape_id=sh.get("shape_id", "<unknown>"),
                    status=sh.get("status", ""),
                    bad_dtype_keys=list(bad),
                ))
    return flagged


def _load_verdict_dir(verdict_dir: Path) -> list[dict]:
    out: list[dict] = []
    for p in sorted(verdict_dir.glob("*.json")):
        try:
            out.append(json.loads(p.read_text()))
        except (OSError, json.JSONDecodeError) as e:
            print(f"WARN: skipping unreadable verdict {p}: {e}", file=sys.stderr)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("verdict_dir", type=Path,
                    help="directory of <task_id>.json oracle-equiv verdicts")
    ap.add_argument("--json", action="store_true",
                    help="emit the flagged set as JSON instead of a table")
    args = ap.parse_args(argv)

    if not args.verdict_dir.is_dir():
        print(f"ERROR: not a directory: {args.verdict_dir}", file=sys.stderr)
        return 2

    verdicts = _load_verdict_dir(args.verdict_dir)
    flagged = audit_verdicts(verdicts)

    if args.json:
        print(json.dumps([f.__dict__ for f in flagged], indent=2))
    else:
        if not flagged:
            print(f"No VERIFIED-with-dtype-mismatch shapes in "
                  f"{len(verdicts)} verdict(s). Output-dtype gate would flip "
                  f"0 tasks.")
        else:
            tasks = {f.task_id for f in flagged}
            print(f"REPORT-ONLY: {len(flagged)} VERIFIED shape(s) across "
                  f"{len(tasks)} task(s) would FLIP under the migration "
                  f"output-dtype gate (values pass, output dtype != reference):")
            for f in flagged:
                print(f"  {f.task_id} / {f.shape_id}: "
                      f"bad_dtype_keys={f.bad_dtype_keys}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
