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


def _classify_from_homes(knobs: set[str], in_dims: set[str], in_scalar: set[str], in_dist: set[str],
                         meta: dict) -> dict[str, str]:
    """Assign each knob a class from its discovered input homes + the (format-independent) meta references."""
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


def _classify_explicit(shapes_doc: dict, meta: dict) -> dict[str, str]:
    """Classify on an EXPLICIT-format doc (U9 c4b). The symbol universe is the row ``env`` keys -- the named
    row environment that replaces ``size_knobs`` (the explicit format has no top-level ``size_knobs``). For
    every task env keys == size_knobs keys; the corpus-wide parity test enforces that equality (it cannot be
    re-derived from the explicit doc alone, since size_knobs is gone). Input homes come from the explicit
    rows, mirroring the knob path's (scalar_knob/derive/scalar) field scan so the verdict is format-invariant:
      * tensor-axis  -- each tensor's ``axes`` (the dim EXPRESSIONS, so a ``2*D`` dim still homes ``D``);
      * scalar-input -- a derive/scalar_knob input's source ``expr``, or a plain ``scalar`` input's raw
        ``value`` (which may itself be a symbol string -- knob scans the scalar field, so we must too);
      * dist-param   -- each tensor's row-invariant ``dist_symbolic`` args (approach A -- the resolved rows
        alone lost these symbols). NOTE the ``randn_inf_excl`` ref arg is a tensor NAME; both classifiers
        regex-scan dist args, so a ref that collides with a knob name would be mis-homed (0 in the corpus).
    """
    rows = (shapes_doc.get("public") or []) + (shapes_doc.get("hidden") or [])
    knobs: set[str] = set()
    for row in rows:
        knobs |= set((row.get("env") or {}).keys())
    in_dims: set[str] = set()
    in_scalar: set[str] = set()
    in_dist: set[str] = set()
    for row in rows:
        for inp in row.get("inputs") or []:
            if inp.get("kind") == "tensor":
                for ax in inp.get("axes") or []:
                    in_dims |= _syms(ax) & knobs
                for v in ((inp.get("dist_symbolic") or {}).get("args") or {}).values():
                    in_dist |= _syms(v) & knobs
            elif inp.get("expr") is not None:        # derive / scalar_knob source formula
                in_scalar |= _syms(inp["expr"]) & knobs
            elif inp.get("kind") == "scalar":        # plain scalar: the raw value may be a symbol string
                in_scalar |= _syms(inp.get("value")) & knobs
    return _classify_from_homes(knobs, in_dims, in_scalar, in_dist, meta)


def classify(shapes_doc: dict, meta: dict) -> dict[str, str]:
    from ascendbench.input_spec.shape_format import is_explicit_shapes_doc
    if is_explicit_shapes_doc(shapes_doc):
        return _classify_explicit(shapes_doc, meta)
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
    return _classify_from_homes(knobs, in_dims, in_scalar, in_dist, meta)


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
