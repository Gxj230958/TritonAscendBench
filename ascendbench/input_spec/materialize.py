"""Explicit-shape materializer.

Turn a ``(shapes_doc, shape_id)`` into an EXPLICIT per-input description -- literal `shape` + `dtype` +
named `axes` + `dist` for tensors, and resolved values for scalar/derive/scalar_seq/scalar_knob/optional
params -- that is faithful to :func:`ascendbench.input_spec.inputs.build_inputs` by reusing its dim
resolution (`_resolve_dims`), dist normalisation (`parse_dist_spec`), and dtype default. It does NOT
allocate tensors, so it is cheap to run corpus-wide.

This is the basis for the Phase-1 human-readable explicit *view* (non-authoritative annotation) and,
in Phase 2, for the explicit authoritative format. Dispatch mirrors ``build_inputs`` exactly so the
materialised `shape` equals the built tensor's shape for every input.
"""
from __future__ import annotations

from typing import Any

from ascendbench.input_spec.expr import eval_numeric
from ascendbench.input_spec.inputs import _resolve_dims, _resolve_dtype, parse_dist_spec

_DEFAULT_DTYPE = "float32"   # build_inputs: spec.get("dtype", "float32")
_DEFAULT_DIST = "randn"      # build_inputs: spec.get("dist", "randn")


def effective_knobs(shapes_doc: dict[str, Any], shape_id: str) -> dict[str, Any]:
    """size_knobs defaults overlaid by the row's per-shape knobs (same as build_inputs)."""
    rows = (shapes_doc.get("public_shapes") or []) + (shapes_doc.get("hidden_shapes") or [])
    row = next((r for r in rows if r.get("id") == shape_id), None)
    if row is None:
        raise KeyError(f"shape id {shape_id!r} not in shapes.yaml")
    knobs = dict(shapes_doc.get("size_knobs") or {})
    knobs.update(row.get("knobs") or {})
    return knobs


def materialize_row(shapes_doc: dict[str, Any], shape_id: str) -> dict[str, dict]:
    """Return ``{input_name: descriptor}`` for one shape row.

    Tensor descriptor: ``{kind: "tensor", shape: [...], dtype, axes: [...], dist}``. Param descriptors:
    ``scalar`` / ``derive`` (with ``expr`` + resolved ``value``) / ``scalar_seq`` / ``scalar_knob`` /
    ``optional`` (a non-dict spec such as ``null``) / ``struct``. Dispatch order matches build_inputs.
    """
    knobs = effective_knobs(shapes_doc, shape_id)
    out: dict[str, dict] = {}
    for name, spec in (shapes_doc.get("inputs") or {}).items():
        if not isinstance(spec, dict):
            out[name] = {"kind": "optional", "value": spec}   # e.g. `null` optional input
            continue
        if "struct" in spec:
            out[name] = {"kind": "struct", "struct": spec.get("struct")}
        elif "scalar" in spec:
            out[name] = {"kind": "scalar", "value": spec["scalar"]}
        elif "derive" in spec:
            out[name] = {"kind": "derive", "expr": spec["derive"],
                         "value": eval_numeric(str(spec["derive"]), knobs)}
        elif "scalar_seq" in spec:
            out[name] = {"kind": "scalar_seq", "value": list(spec["scalar_seq"])}
        elif "scalar_knob" in spec:
            out[name] = {"kind": "scalar_knob", "expr": spec["scalar_knob"],
                         "value": int(eval_numeric(str(spec["scalar_knob"]), knobs))}
        else:   # tensor with dims
            dims = spec.get("dims") or []
            declared_dtype = str(spec.get("dtype", _DEFAULT_DTYPE))
            _resolve_dtype(declared_dtype)   # fail-loud on an unknown dtype, exactly like build_inputs
            out[name] = {
                "kind": "tensor",
                "shape": _resolve_dims(dims, knobs),   # exact: reuses build_inputs' _resolve_dims
                # DECLARED (authoring) dtype. NOTE: build_inputs overrides this to int64 for integer
                # index dists (offsets / arange / full / randperm / *_ids), so the *produced* dtype
                # may differ; exact produced-dtype fidelity is the Phase-2 / U6 concern (it builds the
                # real tensors and compares). The view shows what the spec authors + the dist kind.
                "dtype": declared_dtype,
                # named axes = the dim expressions themselves (the knob glossary maps the symbols to
                # human names); a literal-int dim has no symbolic axis.
                "axes": [str(d) for d in dims],
                # render the default explicit so the view carries no implicit defaults.
                "dist": parse_dist_spec(spec.get("dist", _DEFAULT_DIST)),
            }
    return out
