#!/usr/bin/env python3
"""Compact-v2 (explicit) authoring helpers -- parse / reconstruct / write the canonical v2 form.

The corpus is migrated to compact v2 (schema_version 2). Authoring tools (generate / quarantine / extend)
modify it by: ``load_v2`` -> mutate in memory -> ``write_v2`` (the single canonical writer,
``migrate_to_v2.to_yaml``). ``v2_to_knob_spec`` reconstructs a knob-equivalent doc from a v2 doc so the
knob-shaped candidate-generation / ``build_inputs`` logic in extend runs unchanged; the reconstruction is
lossless -- ``generate_compact_shapes(v2_to_knob_spec(v2)) == v2`` (the U10 invariant, proven over the corpus).
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ascendbench.input_spec.compact_shapes import is_compact_shapes_doc  # noqa: E402
from ascendbench.input_spec.shape_format import is_explicit_shapes_doc  # noqa: E402
from scripts.corpus.migrate_to_v2 import to_yaml  # noqa: E402

#: every input kind the v2 form admits (anything else is a malformed / unsupported row).
_KNOWN_KINDS = frozenset({"tensor", "scalar", "scalar_seq", "scalar_knob", "derive", "optional", "struct"})


def _is_json_scalar(v: Any) -> bool:
    """A JSON scalar -- the only thing a materialised scalar/derive/scalar_knob value may be."""
    return isinstance(v, (bool, int, float, str, type(None)))


def _is_json_safe(v: Any) -> bool:
    """A JSON scalar, or a (recursively) list of JSON-safe values -- never a dict anywhere."""
    if _is_json_scalar(v):
        return True
    return isinstance(v, list) and all(_is_json_safe(x) for x in v)


def _validate_nontensor_inputs(doc: dict) -> None:
    """Per-kind structural validation of NON-tensor inputs (fully fail-closed on JSON-unsafe content).

    is_explicit_shapes_doc deep-checks tensor inputs only (name+kind for the rest). Everything a non-tensor
    input feeds to build_from_explicit must be JSON-safe (a scalar, or a list of scalars) -- a dict hiding in a
    scalar/seq/optional value, a missing derive/scalar_knob resolved value (-> None at build), an UNKNOWN kind,
    or a non-string expr would otherwise slip through load_v2 and KeyError later or reconstruct with CHANGED
    MEANING. Fail closed here so the authoring path never loads/mutates a bad doc.
    """
    for r in (doc.get("public") or []) + (doc.get("hidden") or []):
        rid = r.get("id")
        for i in r.get("inputs") or []:
            name, kind = i.get("name"), i.get("kind")
            if kind not in _KNOWN_KINDS:
                raise ValueError(f"row {rid!r} input {name!r} has unknown kind {kind!r}")
            if kind in ("derive", "scalar_knob"):
                if type(i.get("expr")) is not str or not i.get("expr"):
                    raise ValueError(f"row {rid!r} {kind} input {name!r} needs a non-empty string 'expr'")
                if not _is_json_scalar(i.get("value")):
                    raise ValueError(f"row {rid!r} {kind} input {name!r} needs a resolved JSON-scalar 'value'")
            elif kind == "scalar_seq":
                if not (isinstance(i.get("value"), list) and all(_is_json_scalar(x) for x in i["value"])):
                    raise ValueError(f"row {rid!r} scalar_seq input {name!r} needs a list of JSON-scalar values")
            elif kind == "scalar":
                if not _is_json_scalar(i.get("value")):
                    raise ValueError(f"row {rid!r} scalar input {name!r} needs a JSON-scalar 'value'")
            elif kind == "optional":
                if "value" not in i or not _is_json_safe(i.get("value")):
                    raise ValueError(f"row {rid!r} optional input {name!r} needs a JSON-safe 'value' (e.g. null)")
            elif kind == "struct":
                if "struct" not in i:
                    raise ValueError(f"row {rid!r} struct input {name!r} needs a 'struct'")


def load_v2(path: str | Path) -> dict:
    """Parse a committed v2 shapes.yaml. Fail-closed: raise if it is not a valid compact-v2 doc."""
    doc = yaml.safe_load(Path(path).read_text())
    if not is_compact_shapes_doc(doc):
        raise ValueError(f"{path}: not a compact v2 (schema_version 2) shapes.yaml")
    is_explicit_shapes_doc(doc)   # raises ShapeFormatError on a malformed v2 tensor row
    _validate_nontensor_inputs(doc)   # is_explicit only deep-checks tensors; close the non-tensor gap
    return doc


def write_v2(doc: dict, path: str | Path) -> None:
    """Serialise a v2 doc through the single canonical writer and write it (no comments, deterministic)."""
    Path(path).write_text(to_yaml(doc))


def _knob_input_entry(inp: dict) -> Any:
    """Reverse materialize_row for one v2 input -> its knob ``inputs`` entry (the symbolic template)."""
    kind = inp["kind"]
    if kind == "tensor":
        return {"dims": inp["axes"], "dtype": inp["dtype"], "dist": inp["dist"]}
    if kind == "optional":
        return inp.get("value")               # a bare value (e.g. null) -- not a dict spec
    if kind == "scalar":
        return {"scalar": inp["value"]}
    if kind == "scalar_seq":
        return {"scalar_seq": inp["value"]}
    if kind == "derive":
        return {"derive": inp["expr"]}        # rebuild from expr; value is re-derived by materialize
    if kind == "scalar_knob":
        return {"scalar_knob": inp["expr"]}
    if kind == "struct":
        raise NotImplementedError(
            "struct input reconstruction is not lossless (v2 keeps only `struct`, not the full `fields`); "
            "the corpus has no struct rows -- a struct row needs a dedicated authoring path")
    raise ValueError(f"unknown v2 input kind {kind!r}")


def _row_template(inputs: list[dict]) -> list:
    """The ROW-INVARIANT part of each input -- everything the knob template fixes across rows (only the
    per-row resolved ``shape`` / scalar ``value`` vary). tensor: name/kind/axes/dtype/dist; derive &
    scalar_knob: name/kind/expr (the value re-derives per row); scalar/scalar_seq/optional: name/kind/value
    (fixed); struct: name/kind/struct.
    """
    out: list = []
    for i in inputs:
        k = i["kind"]
        if k == "tensor":
            out.append((i["name"], k, list(i.get("axes") or []), i.get("dtype"), i.get("dist")))
        elif k in ("derive", "scalar_knob"):
            out.append((i["name"], k, i.get("expr")))
        elif k == "struct":
            out.append((i["name"], k, i.get("struct")))
        else:   # scalar / scalar_seq / optional -- the value is row-invariant
            out.append((i["name"], k, i.get("value")))
    return out


def v2_to_knob_spec(v2: dict) -> dict:
    """Reconstruct a knob-equivalent doc from a v2 doc.

    Lossless: ``generate_compact_shapes(v2_to_knob_spec(v2)) == v2``. The symbolic ``inputs`` template + the
    base ``size_knobs`` come from ROW 0 (input order preserved); each row's ``env`` becomes its per-row
    ``knobs`` (so ``effective_knobs == env`` and ``build_inputs`` resolves exactly). Fail-loud if any later
    row's FULL per-input template (name/kind + axes/dtype/dist or value/expr -- not just name+kind) differs
    from row 0's: the knob model has a single template for all rows, so a row-varying template cannot round-trip
    and must not be silently collapsed to row 0.
    """
    pub = v2.get("public") or []
    hid = v2.get("hidden") or []
    rows = pub + hid
    if not rows:
        raise ValueError("v2 doc has no rows -- nothing to reconstruct")
    row0 = rows[0]
    tpl0 = _row_template(row0["inputs"])
    for r in rows[1:]:
        if _row_template(r["inputs"]) != tpl0:
            raise ValueError(
                f"row {r['id']!r} input template differs from row 0 -- not a uniform-template (knob) task")
    inputs = {i["name"]: _knob_input_entry(i) for i in row0["inputs"]}
    return {
        "size_knobs": dict(row0["env"]),       # FULL row-0 env, in env order (NOT axes-only)
        "inputs": inputs,
        "public_shapes": [{"id": r["id"], "knobs": dict(r["env"])} for r in pub],
        "hidden_shapes": [{"id": r["id"], "knobs": dict(r["env"])} for r in hid],
    }
