#!/usr/bin/env python3
"""Phase-2 U4: classify every size-knob symbol a task uses, corpus-wide.

For each declared `size_knobs` symbol, decide which authoring role it plays so the Phase-2 explicit
schema (U5) knows whether it becomes a tensor axis or a named `params:` entry, and U7 (cost_model
re-base) knows the formula universe is NOT tensor-axes-only:

  - tensor-axis      : appears in some input's `dims`            -> becomes a named tensor axis
  - scalar-input     : a `scalar_knob`/`derive`/`scalar` input   -> a params entry backed by a scalar
  - dist-param       : appears ONLY inside a `dist` expression    -> a params entry backed by the fill rule
  - row-only-param   : used by cost/constraints but has NO input home -> a standalone named `params:` entry
  - dead             : declared but referenced nowhere            -> droppable

`row-only-param` is the codex finding that makes a pure tensor-axis cost basis impossible (e.g.
`L1_09:D`, `L3_13:CAUSAL`, `G`@{L2_29,L3_15,L3_18}, `L3_09:B`). This is a REPORT (dev analysis), not a
gate.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

_TOK = re.compile(r"[A-Za-z_]\w*")
# expression helpers that are NOT knobs
_NONKNOB = {"and", "or", "not", "min", "max", "sqrt", "ceil", "floor", "int", "abs", "sum",
            "len", "gcd", "is_pow2", "if", "else", "True", "False", "dtype_size"}


def _syms(expr) -> set[str]:
    if expr is None:
        return set()
    return set(_TOK.findall(str(expr))) - _NONKNOB


def _syms_deep(obj) -> set[str]:
    """Knob symbols in every STRING VALUE of a nested structure (dict VALUES + list items, not keys).
    Scanning values-only keeps input/field NAMES (dict keys, e.g. an `input:` value's tensor name) out
    of false positives while reaching expression strings nested in args.value / dists / invariants."""
    if isinstance(obj, str):
        return _syms(obj)
    if isinstance(obj, dict):
        return set().union(*(_syms_deep(v) for v in obj.values())) if obj else set()
    if isinstance(obj, list):
        return set().union(*(_syms_deep(v) for v in obj)) if obj else set()
    return set()


def classify(shapes_doc: dict, meta: dict) -> dict[str, str]:
    knobs = set((shapes_doc.get("size_knobs") or {}).keys())
    in_dims: set[str] = set()
    in_scalar: set[str] = set()   # scalar_knob / derive / scalar
    in_dist: set[str] = set()
    for spec in (shapes_doc.get("inputs") or {}).values():
        if not isinstance(spec, dict):
            continue
        for d in spec.get("dims") or []:
            in_dims |= _syms(d) & knobs
        for fld in ("scalar_knob", "derive", "scalar"):
            if fld in spec:
                in_scalar |= _syms(spec[fld]) & knobs
        if spec.get("dist") is not None:
            in_dist |= _syms(spec["dist"]) & knobs

    # scan the WHOLE cost_model + param_constraints blocks (values, deeply): cost references live in
    # flops/bytes; constraint references live in dims AND dists/scalars/tensor_invariants[*].args.value
    # (codex: 149 refs the dims-only scan missed -- none currently dead, but the scan must be complete).
    in_cost = _syms_deep(meta.get("cost_model") or {}) & knobs
    in_constraint = _syms_deep(meta.get("param_constraints") or {}) & knobs

    out: dict[str, str] = {}
    for k in knobs:
        if k in in_dims:
            out[k] = "tensor-axis"
        elif k in in_scalar:
            out[k] = "scalar-input"
        elif k in in_dist:
            out[k] = "dist-param"
        elif k in in_cost or k in in_constraint:
            out[k] = "row-only-param"      # homeless: cost/constraint use but no input home
        else:
            out[k] = "dead"
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tasks-root", default="tasks")
    ap.add_argument("--json-out")
    args = ap.parse_args()

    cls_tot = Counter()
    row_only: dict[str, list[str]] = {}
    report: dict[str, dict] = {}
    for sp in sorted(Path(args.tasks_root).rglob("shapes.yaml")):
        if sp.parent.name.startswith("C1_"):
            continue
        doc = yaml.safe_load(sp.read_text())
        if not isinstance(doc, dict) or not doc.get("size_knobs"):
            continue
        meta = yaml.safe_load((sp.parent / "meta.yaml").read_text()) or {}
        c = classify(doc, meta)
        report[sp.parent.name] = c
        cls_tot.update(c.values())
        ro = sorted(k for k, v in c.items() if v == "row-only-param")
        if ro:
            row_only[sp.parent.name] = ro

    print("=== symbol classes (corpus-wide) ===")
    for cls, n in cls_tot.most_common():
        print(f"  {cls:16s} {n}")
    print(f"\n=== row-only-param (no tensor-axis/input home -> must be named params): "
          f"{sum(len(v) for v in row_only.values())} pairs in {len(row_only)} tasks ===")
    for t, syms in sorted(row_only.items()):
        print(f"  {t}: {syms}")
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
