#!/usr/bin/env python3
"""Per-shape S2-original GPU probe -- shape-quarantine triage.

For ONE task, run each declared shape through the EXACT S2 original-side machinery
(``_spawn_side("original", ...)`` from the freeze verifier) in ISOLATION, so a shape that
makes the frozen kernel reject at compile/launch surfaces as ``rc != 0`` with its own
``sid`` -- the daemon's batched S2 aborts the whole side at the first bad shape and never
names it. This recovers the offending shape ids so they can be removed from shapes.yaml.

Triage only: box-side results are advisory (GPU arch / Triton-CUDA version / numeric
nondeterminism can diverge). The authoritative truth stays the daemon S2/C2 re-grade.
A shape that COMPILES + RUNS but is only non-bit-exact (a phase=compare failure) is NOT
caught here -- those are removed by their daemon-named sid separately.
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "verify"))

import yaml  # noqa: E402
from standalone_freeze_verifier import (  # noqa: E402
    CLASS_C,
    _entrypoint_from_meta,
    _shape_ids_for,
    _spawn_side,
    classify_task,
)


def probe_task(td: Path, sids: list[str] | None) -> dict:
    """Probe ``sids`` (or every declared shape) of task dir ``td``; return per-sid rc."""
    meta = yaml.safe_load((td / "meta.yaml").read_text())
    shapes_doc = yaml.safe_load((td / "shapes.yaml").read_text())
    kernel_path = td / "kernel.py"
    cls_tc = classify_task(
        td.name, kernel_path.read_text(encoding="utf-8", errors="replace")
    )
    accel_disabled = bool(cls_tc.by_class(CLASS_C))
    pinned_config = meta.get("pinned_semantic_config")
    entrypoint = _entrypoint_from_meta(td)

    all_ids = _shape_ids_for(shapes_doc)
    probe_ids = sids or all_ids
    out: dict = {}
    for sid in probe_ids:
        if sid not in all_ids:
            out[sid] = {"rc": -1, "error": "sid not declared in shapes.yaml"}
            continue
        work = Path(tempfile.mkdtemp(prefix=f"probe_{td.name}_{sid}_"))
        try:
            rc, st = _spawn_side(
                "original", kernel_path, shapes_doc, [sid], 0, work,
                accel_disabled, pinned_config, entrypoint=entrypoint,
            )
        except Exception as e:  # spawn-machinery error -- record, do not crash the sweep
            rc, st = 98, {"error": f"{type(e).__name__}: {e}"}
        out[sid] = {"rc": rc, "error": (st or {}).get("error")}
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--task-dir", required=True, help="path to tasks/<lvl>/<task>")
    ap.add_argument(
        "--sids",
        help="comma-separated shape ids to probe (default: every declared shape)",
    )
    ap.add_argument("--json-out", help="also write the report JSON here")
    args = ap.parse_args()

    td = Path(args.task_dir).resolve()
    sids = [s for s in args.sids.split(",") if s] if args.sids else None
    res = probe_task(td, sids)
    fails = {k: v for k, v in res.items() if v["rc"] != 0}
    report = {
        "task": td.name,
        "n_probed": len(res),
        "n_fail": len(fails),
        "fail_sids": sorted(fails),
        "fail_detail": fails,
    }
    print(json.dumps(report, indent=1))
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
