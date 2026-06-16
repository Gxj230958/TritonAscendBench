#!/usr/bin/env python3
"""Surgically remove kernel-rejected shape rows from a task's shapes.yaml.

The shape-quarantine writer that pairs with ``gpu_shape_probe.py``: the probe (or the
daemon S2/C2 verdict) names the shape ids the frozen kernel rejects; this deletes exactly
those ``- id: <sid>`` list items so the re-graded corpus has no declared-but-ungraded shape
(which would force ``SHAPE_COVERAGE_GAP``). The deletion is byte-surgical -- only the removed
rows change, every surviving shape stays byte-identical -- and the result is parsed back to
prove the survivors are intact and the target ids are gone. The meta ``shape_coverage`` counts
are rewritten from the surviving shape set; ``equivalence_per_shape`` is left for the S2/C2
re-grade to repopulate.

We NEVER tune to pass: a removed shape is one the real frozen kernel cannot compile/run, not
one trimmed to hit a number.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import yaml

_ID_LINE = re.compile(r"- id:\s*(\S+)\s*$")


def remove_rows(text: str, sids: set[str]) -> tuple[str, list[str]]:
    """Delete the ``- id: <sid>`` items for ``sids``. An item spans from its column-0
    ``- id:`` line to the line before the next column-0 ``- id:`` or dedented top-level key."""
    lines = text.splitlines(keepends=True)
    starts = [
        (i, m.group(1))
        for i, ln in enumerate(lines)
        if (m := _ID_LINE.match(ln.rstrip("\n")))
    ]
    drop = [False] * len(lines)
    removed: list[str] = []
    for start, sid in starts:
        if sid not in sids:
            continue
        end = len(lines)
        for j in range(start + 1, len(lines)):
            ln = lines[j]
            is_next_item = _ID_LINE.match(ln.rstrip("\n")) is not None
            is_toplevel = bool(re.match(r"\S", ln)) and not ln.startswith("- ")
            if is_next_item or is_toplevel:
                end = j
                break
        for k in range(start, end):
            drop[k] = True
        removed.append(sid)
    new = "".join(ln for i, ln in enumerate(lines) if not drop[i])
    return new, removed


def _rewrite_coverage(meta_text: str, pub: int, hid: int) -> tuple[str, bool]:
    """Surgically rewrite the ``shape_coverage: {public,hidden,total}`` block in meta text."""
    pat = re.compile(
        r"(shape_coverage:\n"
        r"(?P<ip>\s+)public:\s*)\d+"
        r"(\n\s+hidden:\s*)\d+"
        r"(\n\s+total:\s*)\d+"
    )

    def _sub(m: re.Match) -> str:
        return f"{m.group(1)}{pub}{m.group(3)}{hid}{m.group(4)}{pub + hid}"

    new, n = pat.subn(_sub, meta_text)
    return new, n == 1


def quarantine(task_dir: Path, sids: set[str], dry_run: bool) -> dict:
    sp = task_dir / "shapes.yaml"
    text = sp.read_text()
    new_text, removed = remove_rows(text, sids)

    doc = yaml.safe_load(new_text)  # parse-back: must stay valid YAML
    surviving = {
        s["id"]
        for role in ("public_shapes", "hidden_shapes")
        for s in (doc.get(role) or [])
        if isinstance(s, dict) and s.get("id")
    }
    leftover = sids & surviving
    if leftover:
        raise RuntimeError(f"removal incomplete -- still present: {sorted(leftover)}")
    not_found = sids - set(removed)
    pub = len(doc.get("public_shapes") or [])
    hid = len(doc.get("hidden_shapes") or [])

    mp = task_dir / "meta.yaml"
    meta_text = mp.read_text()
    new_meta, ok = _rewrite_coverage(meta_text, pub, hid)

    report = {
        "task": task_dir.name,
        "requested": sorted(sids),
        "removed": sorted(removed),
        "not_found": sorted(not_found),
        "public_after": pub,
        "hidden_after": hid,
        "total_after": pub + hid,
        "coverage_rewritten": ok,
    }
    if not dry_run:
        sp.write_text(new_text)
        if ok:
            mp.write_text(new_meta)
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--task-dir", required=True)
    ap.add_argument("--sids", required=True, help="comma-separated shape ids to remove")
    ap.add_argument("--apply", action="store_true", help="write changes (default: dry-run)")
    args = ap.parse_args()

    sids = {s for s in args.sids.split(",") if s}
    rep = quarantine(Path(args.task_dir).resolve(), sids, dry_run=not args.apply)
    import json

    print(json.dumps(rep, indent=1))
    if rep["not_found"]:
        print(f"WARNING: ids not found in shapes.yaml: {rep['not_found']}", file=sys.stderr)
    if not rep["coverage_rewritten"]:
        print("WARNING: shape_coverage block not rewritten (pattern miss)", file=sys.stderr)


if __name__ == "__main__":
    main()
