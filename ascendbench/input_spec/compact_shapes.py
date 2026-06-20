#!/usr/bin/env python3
"""Phase-2 U9 migration M1: the COMPACT explicit shapes format (schema_version 2) + its expander.

codex design consult (MIGRATION-TARGET B): the committed migrated shapes.yaml is a COMPACT, readable v2 doc
-- NOT generate_explicit_shapes' verbose structured v1 output (8.6x larger; it spells out a resolved + a
symbolic dist per input per row). v2 stores the SYMBOLIC ``dist`` string ONCE per tensor and DERIVES the
row-invariant ``dist_symbolic`` (= parse_dist_spec(dist)) and the per-row resolved ``dist`` (=
resolve_dist_args(., env)) on expansion. ``env`` (the resolved row symbol->value map) is kept -- it is NOT
derivable from axes+shape (composite axes like ``E*L`` are underdetermined, row-only params like ``CAUSAL`` /
``G`` / flags have no tensor home).

v2 row schema (committed):
  ``{id, env:{sym->val}, inputs:[ORDERED]}`` -- each tensor input
  ``{name, kind: tensor, shape:[...], dtype, axes:[...], dist: "<symbolic string>"}`` (no dist_symbolic, no
  resolved dist, no per-row ``role`` -- the public/hidden SECTION is authoritative).

``expand_compact(v2) -> v1`` is byte-identical to ``generate_explicit_shapes(knob)`` (the M1 gate), so every
v1-aware reader works unchanged on the expanded form, and ``build_from_explicit`` stays byte-equal to
``build_inputs``. parse_dist_spec / resolve_dist_args pull torch, so the expander + the writer are TORCH-OK and
run on the materialisation / migration path; the torch-free adapter readers (is_explicit / public_ids /
manifest) read v2 NATIVELY (see shape_format).
"""
from __future__ import annotations

import re
from typing import Any

#: the v1 (verbose, structured) schema version expand_compact() targets -- kept in lockstep with row_ir.
from ascendbench.input_spec.row_ir import SCHEMA_VERSION as V1_SCHEMA_VERSION

COMPACT_SCHEMA_VERSION = 2

_TOK = re.compile(r"[A-Za-z_]\w*")


def dist_arg_symbols(dist: str) -> set[str]:
    """Identifiers appearing in a dist DSL string's ARGS only (not its kind keyword).

    Mirrors the v1 classifier's ``dist_symbolic.args`` value scan WITHOUT pulling torch (parse_dist_spec
    lives in the torch-coupled inputs module): ``"randint:0:E" -> {E}``, ``"offsets:to=B" -> {B}``,
    ``"full:B*N" -> {B, N}``, ``"randn" -> set()``. A ``k=v`` arg contributes only ``v``. The caller
    intersects with the known symbol set, so a ref tensor name that is not a symbol drops out.
    """
    if not isinstance(dist, str) or ":" not in dist:
        return set()
    _kind, args = dist.split(":", 1)
    syms: set[str] = set()
    for part in args.split(":"):
        value = part.split("=", 1)[1] if "=" in part else part
        syms |= set(_TOK.findall(value))
    return syms


def is_compact_shapes_doc(doc: Any) -> bool:
    """True iff ``doc`` is a schema-2 compact doc (cheap, torch-free check on the version marker)."""
    return isinstance(doc, dict) and type(doc.get("schema_version")) is int \
        and doc.get("schema_version") == COMPACT_SCHEMA_VERSION


def _expand_row(row: dict, *, role: str) -> dict:
    from ascendbench.input_spec.explicit_build import resolve_dist_args
    from ascendbench.input_spec.inputs import parse_dist_spec
    env = dict(row.get("env") or {})
    out_inputs: list[dict] = []
    for inp in row.get("inputs") or []:
        if inp.get("kind") == "tensor":
            ds = parse_dist_spec(inp["dist"])           # row-invariant symbolic {kind, args}
            out_inputs.append({
                "name": inp["name"], "kind": "tensor",
                "shape": inp["shape"], "dtype": inp["dtype"], "axes": inp["axes"],
                "dist": resolve_dist_args(ds, env),     # per-row resolved literal args
                "dist_symbolic": ds,
            })
        else:   # struct / scalar / derive / scalar_seq / scalar_knob / optional -- pass through verbatim
            out_inputs.append(dict(inp))
    return {"id": row["id"], "inputs": out_inputs, "env": env, "role": role}


def expand_compact(doc: dict) -> dict:
    """Expand a compact v2 doc into the verbose structured v1 doc the v1-aware readers consume.

    Byte-identical to ``generate_explicit_shapes(knob)`` for a v2 produced by ``generate_compact_shapes(knob)``
    (the M1 equivalence gate). The ``role`` for each row comes from its section (public/hidden), never a
    committed per-row field."""
    return {
        "schema_version": V1_SCHEMA_VERSION,
        "public": [_expand_row(r, role="public") for r in (doc.get("public") or [])],
        "hidden": [_expand_row(r, role="hidden") for r in (doc.get("hidden") or [])],
    }


def _compact_row(shapes_doc: dict, sid: str) -> dict:
    from ascendbench.input_spec.materialize import effective_knobs, materialize_row
    knobs = effective_knobs(shapes_doc, sid)
    desc = materialize_row(shapes_doc, sid)
    raw_inputs = shapes_doc.get("inputs") or {}
    inputs: list[dict] = []
    for name, d in desc.items():   # authored inputs order
        if d["kind"] == "tensor":
            # the SYMBOLIC dist string straight from the knob input (materialize parsed THIS same string, so
            # parse_dist_spec(raw) == desc dist_symbolic -> expand round-trips exactly). Default to "randn".
            # A knob may declare a STRUCTURED {kind,args} dist; normalise it to the canonical string so v2
            # always commits a string (parse_dist_spec(unparse(x)) == parse_dist_spec(x)).
            raw_dist = (raw_inputs.get(name) or {}).get("dist", "randn")
            if not isinstance(raw_dist, str):
                from ascendbench.input_spec.inputs import parse_dist_spec, unparse_dist_spec
                raw_dist = unparse_dist_spec(parse_dist_spec(raw_dist))
            inputs.append({
                "name": name, "kind": "tensor",
                "shape": d["shape"], "dtype": d["dtype"], "axes": d["axes"], "dist": raw_dist,
            })
        elif d["kind"] == "struct":
            inputs.append({"name": name, "kind": "struct", "struct": d.get("struct")})
        else:
            entry: dict = {"name": name, "kind": d["kind"], "value": d.get("value")}
            if d.get("expr") is not None:
                entry["expr"] = d["expr"]
            inputs.append(entry)
    return {"id": sid, "env": dict(knobs), "inputs": inputs}


def generate_compact_shapes(shapes_doc: dict) -> dict:
    """Emit the COMPACT v2 doc for a knob-format task: ``{schema_version: 2, public:[...], hidden:[...]}``.

    The knob->v2 writer the U9 migration commits. Uses the same materialize path as the v1 generator but keeps
    the SYMBOLIC dist string + the resolved row ``env`` instead of the verbose resolved+symbolic dist pair."""
    from ascendbench.input_spec.inputs import shape_ids
    pub_ids, hid_ids = shape_ids(shapes_doc)
    return {
        "schema_version": COMPACT_SCHEMA_VERSION,
        "public": [_compact_row(shapes_doc, sid) for sid in pub_ids],
        "hidden": [_compact_row(shapes_doc, sid) for sid in hid_ids],
    }
