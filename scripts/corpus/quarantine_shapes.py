#!/usr/bin/env python3
"""Surgically remove kernel-rejected shape rows from a task's compact-v2 shapes.yaml.

The shape-quarantine writer that pairs with ``gpu_shape_probe.py``: the probe (or the daemon S2/C2 verdict)
names the shape ids the frozen kernel rejects; this drops exactly those ``- id: <sid>`` rows from the v2
``public`` / ``hidden`` sections so the re-graded corpus has no declared-but-ungraded shape (which would force
``SHAPE_COVERAGE_GAP``). The corpus is compact v2 (canonical, comment-free), so the edit is parse -> drop ->
re-emit through the single canonical writer (``v2_authoring.write_v2``): the surviving rows are deterministic,
the result is canonical, and an emptied section becomes ``[]`` rather than a dangling ``public:``. The meta
``shape_coverage`` counts are rewritten from the surviving shape set; ``equivalence_per_shape`` is left for the
S2/C2 re-grade to repopulate.

We NEVER tune to pass: a removed shape is one the real frozen kernel cannot compile/run, not one trimmed to hit
a number. A quarantine that would leave ZERO total rows fails loud -- emptying a task is a separate explicit
flow, not a quarantine.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.corpus.v2_authoring import load_v2, write_v2  # noqa: E402


def _find_shape_coverage(d: object) -> dict | None:
    """The first ``shape_coverage`` mapping anywhere in a parsed meta (it is nested under a parent)."""
    if isinstance(d, dict):
        if isinstance(d.get("shape_coverage"), dict):
            return d["shape_coverage"]
        for v in d.values():
            r = _find_shape_coverage(v)
            if r is not None:
                return r
    return None


def _rewrite_coverage(meta_text: str, pub: int, hid: int) -> tuple[str, bool]:
    """Surgically rewrite the ``shape_coverage: {public, hidden[, total]}`` block in meta text.

    ``total`` is OPTIONAL -- two live metas carry only ``public`` / ``hidden``. Update ``public`` / ``hidden``
    always; update ``total`` ONLY when it directly follows ``hidden`` (the canonical layout). A total absent or
    separated from ``hidden`` is left untouched here -- the caller post-verifies the parsed result and fails
    loud on any inconsistency, so a stale/duplicate total can never be written silently.
    """
    pat = re.compile(
        r"(shape_coverage:\n(?P<ip>\s+)public:\s*)\d+"
        r"(\n(?P=ip)hidden:\s*)\d+"
        r"((?P<tl>\n(?P=ip)total:\s*)\d+)?"
    )

    def _sub(m: re.Match) -> str:
        out = f"{m.group(1)}{pub}{m.group(3)}{hid}"
        if m.group("tl") is not None:
            out += f"{m.group('tl')}{pub + hid}"          # update total only when it directly follows hidden
        return out

    new, n = pat.subn(_sub, meta_text)
    return new, n == 1


def quarantine(task_dir: Path, sids: set[str], dry_run: bool) -> dict:
    sp = task_dir / "shapes.yaml"
    doc = load_v2(sp)
    present = {r["id"] for r in (doc.get("public") or []) + (doc.get("hidden") or [])}
    removed = sorted(sids & present)
    not_found = sorted(sids - present)

    doc["public"] = [r for r in (doc.get("public") or []) if r["id"] not in sids]
    doc["hidden"] = [r for r in (doc.get("hidden") or []) if r["id"] not in sids]
    pub, hid = len(doc["public"]), len(doc["hidden"])
    if pub + hid == 0:
        raise RuntimeError(
            f"{task_dir.name}: removing {removed} would leave ZERO shapes -- emptying a task is a separate "
            f"explicit flow, not a quarantine")

    mp = task_dir / "meta.yaml"
    meta_text = mp.read_text()
    new_meta, ok = _rewrite_coverage(meta_text, pub, hid)
    # post-verify: the parsed rewritten coverage MUST reflect the survivors. Catches a regex that left a
    # stale / comment-separated total, missed on CRLF, or otherwise produced an inconsistent block -- refuse a
    # half-update rather than write shapes.yaml against a wrong meta.
    if "shape_coverage:" in meta_text:
        cov = _find_shape_coverage(yaml.safe_load(new_meta)) if ok else None
        if (cov is None or cov.get("public") != pub or cov.get("hidden") != hid
                or ("total" in cov and cov.get("total") != pub + hid)):
            raise RuntimeError(
                f"{task_dir.name}: meta shape_coverage could not be consistently rewritten (unexpected "
                f"layout) -- refusing a stale/duplicate-coverage quarantine")

    report = {
        "task": task_dir.name,
        "requested": sorted(sids),
        "removed": removed,
        "not_found": not_found,
        "public_after": pub,
        "hidden_after": hid,
        "total_after": pub + hid,
        "coverage_rewritten": ok,
    }
    if not dry_run:
        write_v2(doc, sp)        # canonical v2 (deterministic; surviving rows byte-identical)
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
