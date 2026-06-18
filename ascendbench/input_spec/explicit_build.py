"""Phase-2 U6: build full inputs from EXPLICIT resolved-literal rows + a byte-exact comparator.

This is the safety gate U9's 101-task migration rides on: if the generated explicit rows reproduce the
current ``build_inputs`` output byte-for-byte, then migrating a task to the explicit format cannot change
its graded inputs -- any divergence is a representation bug caught BEFORE grading, never a shape to
quarantine.

The equivalence STRATEGY (codex-vetted) is to reuse ``_make_tensor`` verbatim rather than re-implement
tensor construction: a single seeded ``torch.Generator`` threaded through every input in the SAME authored
order, with the partial ``out`` dict carrying earlier tensors for coupled dists (``randn_inf_excl``). So
``build_from_explicit`` is byte-identical to ``build_inputs`` BY CONSTRUCTION; the gate confirms it
empirically across the actual seed set.

Two correctness traps the resolver/builder must honour (both code-grounded):
  * dist args are resolved to literal ints PER ROW, but ``randn_inf_excl.ref`` is a TENSOR NAME, never a
    number -- it is preserved verbatim or the coupled-LSE reference breaks (e.g. L1_15_merge_state).
  * the DECLARED dtype string is fed to ``_make_tensor`` unchanged; integer index dists override it to
    int64 internally (e.g. L3_11 kv_indptr declares int32 but ``offsets`` returns int64), so rebuilding a
    tensor from row metadata directly would diverge -- only re-running ``_make_tensor`` reproduces it.
"""
from __future__ import annotations

from typing import Any

import torch

from ascendbench.input_spec.expr import eval_numeric
from ascendbench.input_spec.inputs import _make_tensor
from ascendbench.input_spec.materialize import effective_knobs, materialize_row

# dist arg keys that are NUMERIC expressions -> resolved to a literal int per row. Allowlisted (codex):
# a future dist kind with a new numeric arg must be added here deliberately, not resolved by accident.
_NUMERIC_DIST_ARG_KEYS = frozenset({"lo", "hi", "step", "to", "value", "E", "B", "R", "W"})
# dist arg keys that name a SIBLING TENSOR -> preserved verbatim (resolving a name is meaningless).
_REF_DIST_ARG_KEYS = frozenset({"ref"})


def resolve_dist_args(dist: dict[str, Any], knobs: dict[str, Any]) -> dict[str, Any]:
    """Resolve a parsed ``{kind, args}`` dist's numeric args to literal ints for one row.

    Numeric args are consumed only via ``eval_numeric`` inside ``_make_tensor``, so substituting the
    literal value for the knob expression reproduces the identical draw. ``ref`` (a tensor name) is kept
    as-is. An unknown arg key fails loud rather than passing an unresolved expression downstream.
    """
    kind = dist["kind"]
    args = dist.get("args") or {}
    out_args: dict[str, Any] = {}
    for k, v in args.items():
        if k in _REF_DIST_ARG_KEYS:
            out_args[k] = v
        elif k in _NUMERIC_DIST_ARG_KEYS:
            out_args[k] = int(eval_numeric(str(v), knobs))
        else:
            raise ValueError(
                f"dist {kind!r}: unknown arg key {k!r} -- add it to the resolver allowlist "
                f"(_NUMERIC_DIST_ARG_KEYS or _REF_DIST_ARG_KEYS) deliberately")
    return {"kind": kind, "args": out_args}


def explicit_row(shapes_doc: dict[str, Any], shape_id: str) -> dict[str, Any]:
    """Build one self-contained explicit row from ``materialize_row`` + per-row dist-arg resolution.

    ``inputs`` is an ORDERED list (authored ``shapes.yaml`` inputs order) -- load-bearing because a
    coupled dist references a tensor declared earlier. Returns
    ``{id, inputs: [{name, kind, ...}, ...]}``: tensor entries carry ``shape``/``dtype``/``axes``/``dist``
    (dist args resolved), non-tensor entries carry the resolved ``value``, struct entries the raw spec.
    """
    knobs = effective_knobs(shapes_doc, shape_id)
    desc = materialize_row(shapes_doc, shape_id)
    inputs: list[dict[str, Any]] = []
    for name, d in desc.items():   # dict insertion order == authored inputs order
        if d["kind"] == "tensor":
            inputs.append({
                "name": name, "kind": "tensor", "shape": d["shape"], "dtype": d["dtype"],
                "axes": d["axes"], "dist": resolve_dist_args(d["dist"], knobs),
            })
        elif d["kind"] == "struct":
            inputs.append({"name": name, "kind": "struct", "struct": d.get("struct")})
        else:   # scalar / derive / scalar_seq / scalar_knob / optional -- resolved value
            inputs.append({"name": name, "kind": d["kind"], "value": d.get("value")})
    return {"id": shape_id, "inputs": inputs}


def build_from_explicit(row: dict[str, Any], seed: int = 0) -> dict[str, Any]:
    """Reconstruct the full inputs dict from an explicit row, reusing ``_make_tensor`` verbatim.

    Byte-identical to ``build_inputs(shapes_doc, row['id'], seed)`` by construction: one generator seeded
    ``seed``, inputs in authored order, ``out`` threaded for refs, declared dtype fed unchanged. ``knobs``
    is empty because shapes and numeric dist args are already literals.
    """
    gen = torch.Generator()
    gen.manual_seed(seed)
    out: dict[str, Any] = {}
    for inp in row["inputs"]:
        kind = inp["kind"]
        if kind == "tensor":
            spec = {"dims": inp["shape"], "dtype": inp["dtype"], "dist": inp["dist"]}
            out[inp["name"]] = _make_tensor(spec, {}, gen, out)
        elif kind == "struct":
            raise NotImplementedError(
                f"input {inp['name']!r}: struct inputs are not modelled by the explicit format yet "
                f"(0 in the current corpus); U8/U9 must extend the format before a struct task migrates")
        else:   # scalar / derive / scalar_seq / scalar_knob / optional -> the resolved value verbatim
            out[inp["name"]] = inp.get("value")
    return out


def _tensor_values_equal(a: torch.Tensor, b: torch.Tensor) -> bool:
    """Exact value equality treating NaN==NaN as equal (``torch.equal`` reports matching NaNs as unequal).

    +inf==+inf is already True under ``==`` so randn_inf sentinels compare correctly; NaN only arises if a
    future dist produces it, and the gate must not silently pass two NaN-bearing tensors as different.
    """
    if a.is_floating_point():
        return bool(((a == b) | (a.isnan() & b.isnan())).all())
    return torch.equal(a, b)


def assert_inputs_equal(expected: dict[str, Any], got: dict[str, Any]) -> None:
    """Byte-exact input-dict equality: same keys; per tensor same dtype+shape+values; scalars ==.

    ``torch.equal`` alone is not enough (it is True across dtype for equal values, False for matching
    NaNs), so dtype and shape are checked explicitly and values via :func:`_tensor_values_equal`.
    """
    if set(expected) != set(got):
        raise AssertionError(f"input keys differ: {sorted(expected)} vs {sorted(got)}")
    for k in expected:
        a, b = expected[k], got[k]
        a_t, b_t = torch.is_tensor(a), torch.is_tensor(b)
        if a_t or b_t:
            if not (a_t and b_t):
                raise AssertionError(f"input {k!r}: tensor vs non-tensor mismatch")
            if a.dtype != b.dtype:
                raise AssertionError(f"input {k!r}: dtype {a.dtype} != {b.dtype}")
            if tuple(a.shape) != tuple(b.shape):
                raise AssertionError(f"input {k!r}: shape {tuple(a.shape)} != {tuple(b.shape)}")
            if not _tensor_values_equal(a, b):
                raise AssertionError(f"input {k!r}: tensor values differ")
        else:
            if type(a) is not type(b) or a != b:
                raise AssertionError(f"input {k!r}: scalar {a!r} ({type(a).__name__}) "
                                     f"!= {b!r} ({type(b).__name__})")
