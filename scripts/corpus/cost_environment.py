#!/usr/bin/env python3
"""Phase-2 U7: the NAMED cost-model symbol environment (shared by validate / audit / xcheck).

The single source of truth (codex Decision D) for "which symbols may a cost_model formula reference".
Before U7 each consumer derived this independently from raw ``size_knobs`` keys, which drifts and admits
a typo'd knob. U7 re-bases the universe onto the U4 NAMED row environment:

    universe = {symbols with an input home: tensor-axis / scalar-input / dist-param}
             + {explicitly DECLARED row-only params (meta.cost_model.row_only_params)}
             + {reserved derived symbols: dtype_size, ...}

Key correctness points (codex-vetted):
  * named symbols are the SYMBOLIC row-env factors (``D`` / ``L`` / ``E``), NOT concrete tensor extents --
    a cost term ``D`` is legal even when the tensor dim is ``2*D`` (23 such composite-axis cases corpus
    wide), because the factor ``D`` is the row-env symbol the dim expression is built from.
  * a homeless cost symbol is NOT auto-promoted to a row-only param -- it must be DECLARED, else it fails
    as a likely typo. So a NEW cost-only symbol does not silently sail through.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from ascendbench.input_spec.expr import free_symbols  # noqa: E402
from ascendbench.perf.cost_model import DERIVED_SYMBOL_DEFAULTS, parse_cost_model  # noqa: E402
from scripts.corpus import classify_symbols as _cs  # noqa: E402

#: symbol classes that carry an INPUT home (a tensor axis factor, a scalar input, or a dist param).
_HOMED_CLASSES = {"tensor-axis", "scalar-input", "dist-param"}


def cost_formula_symbols(meta: dict) -> set[str]:
    """Every free variable symbol referenced by the task's cost_model flops/bytes formulas.

    Uses the shared AST extractor (:func:`expr.free_symbols`), so a helper name in call position
    (``ceil`` / ``min`` / ...) is excluded while a variable that merely shares a helper's spelling but is
    used as a VALUE is still reported -- a regex-minus-funcs scan would mis-drop the latter.
    """
    cm = (meta or {}).get("cost_model") or {}
    syms: set[str] = set()
    for fld in ("flops", "bytes"):
        v = cm.get(fld)
        if isinstance(v, str):
            syms |= free_symbols(v)
        elif isinstance(v, dict):
            for f in v.values():
                syms |= free_symbols(str(f))
    return syms


def declared_row_only(meta: dict) -> tuple[str, ...]:
    """The task's declared cost row-only params (validated for type via parse_cost_model)."""
    cm = (meta or {}).get("cost_model")
    if not cm:
        return ()
    return parse_cost_model(cm).row_only_params


def named_cost_universe(shapes_doc: dict, meta: dict) -> set[str]:
    """The set of symbols a cost formula may legally reference (the named row environment)."""
    classes = _cs.classify(shapes_doc, meta or {})
    homed = {s for s, c in classes.items() if c in _HOMED_CLASSES}
    return homed | set(declared_row_only(meta)) | set(DERIVED_SYMBOL_DEFAULTS)


def check_cost_symbols(shapes_doc: dict, meta: dict) -> list[str]:
    """Return cost-symbol universe violations for a task (empty = clean).

    Flags: (a) a cost symbol that is neither homed, declared row-only, nor derived (a homeless/typo
    symbol); (b) a declared row_only_param not actually used in any cost formula (dead declaration); (c) a
    declared row_only_param that is NOT classified row-only (it has an input home -- don't declare it).
    """
    cm = (meta or {}).get("cost_model")
    if not cm:
        return []
    out: list[str] = []
    classes = _cs.classify(shapes_doc, meta or {})
    universe = named_cost_universe(shapes_doc, meta)
    cost_syms = cost_formula_symbols(meta)
    declared = set(declared_row_only(meta))

    for s in sorted(cost_syms):
        if s not in universe:
            out.append(f"cost symbol {s!r} is not a tensor-axis/scalar/dist input, a declared "
                       f"row_only_param, or a derived symbol (homeless -- declare it or fix the typo)")
    for s in sorted(declared):
        cls = classes.get(s)
        if s not in cost_syms:
            out.append(f"row_only_param {s!r} is declared but used in no cost formula (dead declaration)")
        elif cls is None:
            out.append(f"row_only_param {s!r} is not a declared size_knob, so it is not a row-environment "
                       f"symbol (declare it in size_knobs or fix the name)")
        elif cls != "row-only-param":
            out.append(f"row_only_param {s!r} is classified {cls!r}, not row-only -- it has an input home, "
                       f"so do not declare it row-only")
    return out
