#!/usr/bin/env python3
"""Phase-2 U7: characterization snapshot of every cost_model flops/bytes OUTCOME.

The U7 re-base (cost formula symbol universe: raw knobs -> named row environment) must be value-neutral
-- it changes which symbols are LEGAL, not what a formula EVALUATES to. This snapshot is the blocking
guard: it records, per (task, sid, flops-class / bytes-resource), the numeric value OR the error string
(a divide-by-zero / undefined symbol is a first-class outcome, e.g. L1_21:P16 has L=1 and a `L-1`
denominator), over EVERY public + hidden row using scoring's real ``dtype_size`` convention. The U7 test
asserts the post-change eval reproduces this baseline exactly; a mismatch blocks the change (never tune).

Usage: ``snapshot_cost_outcomes.py [--out tests/unit_tests/data/cost_baseline.json]`` (regenerate baseline).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from ascendbench.perf.cost_model import CostModelError, FormulaEnv, parse_cost_model  # noqa: E402


def _dtype_size(shapes_doc: dict) -> float:
    """Representative dtype_size = first dims-bearing input's width, via the shape_format adapter (so an
    explicit doc resolves too); keep the 2.0 fallback for a scalar-only block. Byte-identical to the old
    local helper over all 101 knob docs."""
    from ascendbench.input_spec.shape_format import representative_dtype_size
    return representative_dtype_size(shapes_doc) or 2.0


def _outcome(formula: str, knobs: dict, dtype_size: float):
    """Evaluate one formula -> a float value, or an ``{"error": ...}`` marker (failures are first-class)."""
    try:
        return FormulaEnv(dict(knobs), derived={"dtype_size": dtype_size}).evaluate(formula)
    except (CostModelError, ZeroDivisionError, ValueError, KeyError) as e:
        return {"error": f"{type(e).__name__}: {e}"}


def snapshot_task(task_dir: Path) -> dict:
    """Return ``{sid: {flops: {class: outcome}, bytes: {resource: outcome}}}`` for every public+hidden sid."""
    shapes_doc = yaml.safe_load((task_dir / "shapes.yaml").read_text())
    meta = yaml.safe_load((task_dir / "meta.yaml").read_text()) or {}
    cm_raw = meta.get("cost_model")
    if not cm_raw:
        return {}
    model = parse_cost_model(cm_raw)
    from ascendbench.input_spec.shape_format import effective_env, iter_rows
    dtype_size = _dtype_size(shapes_doc)
    out: dict = {}
    for _role, sid, _row in iter_rows(shapes_doc):
        knobs = effective_env(shapes_doc, sid)   # size_knobs+row knobs (knob) / row env (explicit)
        out[sid] = {
            "flops": {c: _outcome(f, knobs, dtype_size) for c, f in model.flops.items()},
            "bytes": {r: _outcome(f, knobs, dtype_size) for r, f in model.bytes.items()},
        }
    return out


def snapshot_corpus() -> dict:
    """Snapshot every task with a cost_model, keyed by task id."""
    out: dict = {}
    for shp in sorted(REPO.glob("tasks/level*/*/shapes.yaml")):
        tdir = shp.parent
        snap = snapshot_task(tdir)
        if snap:
            out[tdir.name] = snap
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Snapshot cost_model flops/bytes outcomes (characterization).")
    ap.add_argument("--out", default="tests/unit_tests/data/cost_baseline.json")
    args = ap.parse_args(argv)
    snap = snapshot_corpus()
    out_path = REPO / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(snap, indent=1, sort_keys=True) + "\n")
    n_sid = sum(len(v) for v in snap.values())
    print(f"snapshot: {len(snap)} tasks, {n_sid} shape rows -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
