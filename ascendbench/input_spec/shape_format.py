"""Phase-2 U8: the central dual-format shape adapter.

Every consumer that reads a ``shapes.yaml`` -- manifest hashing, build_inputs/IPC, the sandbox public
view, param_constraints, the cost universe, scoring/eval/release/perf/oracle scripts -- routes through THIS
module instead of reaching into ``size_knobs`` / ``public_shapes`` / ``hidden_shapes`` directly. That makes
the knob-grid -> explicit-format cutover a ONE-PLACE change: a doc is dispatched by :func:`is_explicit_shapes_doc`
and every reader gets the same answer for "the public ids", "this row's environment", "build this row's
inputs", "the manifest projection", "the candidate-safe view" regardless of format.

The corpus is still 100% knob-format; U8 only makes the readers ALSO understand the explicit format (U9
migrates tasks + re-grades on the daemon). So for a knob-format doc every function MUST behave exactly as
the pre-U8 code did -- the manifest hash in particular stays byte-identical (proven by re-hashing all 101).

Discriminator is STRICT + fail-closed (codex): a doc is explicit iff ``schema_version`` is exactly the
supported int version AND it carries top-level ``public`` (+ optional ``hidden``) row lists with per-row
``inputs`` lists and unique ids; a versioned-but-malformed doc, or top-level ``public``/``hidden`` WITHOUT a
``schema_version``, RAISES rather than silently falling back to the knob path. The Phase-1 ``explicit_view``
is a nested advisory map (no top-level ``public`` list, no ``schema_version``) and is never mistaken for it.
"""
from __future__ import annotations

from ascendbench.input_spec.explicit_build import build_from_explicit
from ascendbench.input_spec.inputs import build_inputs
from ascendbench.input_spec.materialize import effective_knobs
from ascendbench.input_spec.row_ir import SCHEMA_VERSION

# declared-dtype -> element byte width (the representative dtype_size convention shared by the cost
# audit / xcheck). Kept here so the adapter answers dtype_size for BOTH formats from one table.
_DTYPE_BYTES = {
    "float64": 8, "int64": 8, "float32": 4, "fp32": 4, "int32": 4, "uint32": 4,
    "bfloat16": 2, "bf16": 2, "float16": 2, "fp16": 2, "int16": 2,
    "float8_e4m3fn": 1, "float8_e5m2": 1, "int8": 1, "uint8": 1, "bool": 1,
}


class ShapeFormatError(ValueError):
    """A versioned shapes doc is malformed (fail-closed -- never silently treated as knob-format)."""


def is_explicit_shapes_doc(doc: dict) -> bool:
    """True iff ``doc`` is a valid explicit-format shapes doc; False for knob-format; RAISE if malformed.

    Strict: ``schema_version`` must be exactly ``int`` (not ``bool``) and equal to the supported
    ``SCHEMA_VERSION``; top-level ``public`` must be a list, ``hidden`` (if present) a list; every row must
    carry an ``inputs`` list; ids unique across public+hidden. Top-level ``public``/``hidden`` without a
    ``schema_version`` is malformed-explicit (fail closed), not knob-format.
    """
    v = doc.get("schema_version")
    # Detect the explicit marker by KEY PRESENCE (any type), not list-typed -- a malformed explicit doc
    # whose `public` is not a list must still fail closed, not silently route to the knob path. A knob doc
    # uses `public_shapes`/`hidden_shapes`, so it never carries a top-level `public`/`hidden` key.
    has_explicit_keys = "public" in doc or "hidden" in doc
    if v is None:
        if has_explicit_keys:
            raise ShapeFormatError(
                "shapes doc has top-level 'public'/'hidden' but no schema_version -- malformed explicit doc")
        return False
    if type(v) is not int:   # excludes bool (YAML true) -- bool is an int subclass
        raise ShapeFormatError(f"schema_version must be an int, got {type(v).__name__}")
    if v != SCHEMA_VERSION:
        raise ShapeFormatError(f"unsupported schema_version {v} (this reader supports {SCHEMA_VERSION})")
    if not isinstance(doc.get("public"), list):
        raise ShapeFormatError("explicit shapes doc missing a top-level 'public' list")
    if "hidden" in doc and not isinstance(doc["hidden"], list):
        raise ShapeFormatError("explicit shapes doc 'hidden' must be a list")
    rows = (doc.get("public") or []) + (doc.get("hidden") or [])
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("inputs"), list):
            raise ShapeFormatError(f"explicit row {row.get('id') if isinstance(row, dict) else row!r} "
                                   f"missing an 'inputs' list")
    ids = [r.get("id") for r in rows]
    if len(ids) != len(set(ids)):
        raise ShapeFormatError(f"explicit shapes doc has duplicate row ids: {ids}")
    return True


def public_ids(doc: dict) -> list[str]:
    if is_explicit_shapes_doc(doc):
        return [r["id"] for r in (doc.get("public") or [])]
    return [r["id"] for r in (doc.get("public_shapes") or [])]


def hidden_ids(doc: dict) -> list[str]:
    if is_explicit_shapes_doc(doc):
        return [r["id"] for r in (doc.get("hidden") or [])]
    return [r["id"] for r in (doc.get("hidden_shapes") or [])]


def iter_rows(doc: dict) -> list[tuple[str, str, dict]]:
    """``[(role, id, row)]`` over public then hidden, for either format."""
    if is_explicit_shapes_doc(doc):
        pub, hid = doc.get("public") or [], doc.get("hidden") or []
    else:
        pub, hid = doc.get("public_shapes") or [], doc.get("hidden_shapes") or []
    return ([("public", r["id"], r) for r in pub] + [("hidden", r["id"], r) for r in hid])


def _explicit_row(doc: dict, sid: str) -> dict:
    for r in (doc.get("public") or []) + (doc.get("hidden") or []):
        if r.get("id") == sid:
            return r
    raise KeyError(f"shape id {sid!r} not in explicit shapes doc")


def effective_env(doc: dict, sid: str) -> dict:
    """The named row-environment symbol->value map for a row (knobs for knob-format; row ``env`` for explicit)."""
    if is_explicit_shapes_doc(doc):
        return dict(_explicit_row(doc, sid).get("env") or {})
    return effective_knobs(doc, sid)


def build_inputs_for(doc: dict, sid: str, seed: int = 0) -> dict:
    """Materialise a row's full inputs dict, dispatching to build_inputs (knob) or build_from_explicit."""
    if is_explicit_shapes_doc(doc):
        return build_from_explicit(_explicit_row(doc, sid), seed)
    return build_inputs(doc, sid, seed)


def representative_dtype_size(doc: dict) -> float | None:
    """Byte width of the first declared TENSOR input's dtype -- the cost ``dtype_size`` convention.

    Exact parity with ``eval._representative_dtype_size`` (task_api_contract §5.1) for knob-format: the
    first ``inputs`` entry carrying ``dims`` (in declaration order), dtype DEFAULTING to ``float32``, via
    ``_DTYPE_BYTES.get`` (so an exotic dtype yields ``None``, not a skip); ``None`` if no tensor input.
    Explicit-format: the first ``kind=="tensor"`` input of the first row (dtypes are row-invariant).
    """
    if is_explicit_shapes_doc(doc):
        rows = iter_rows(doc)
        if rows:
            for inp in rows[0][2].get("inputs") or []:
                if inp.get("kind") == "tensor":
                    return _DTYPE_BYTES.get(str(inp.get("dtype", "float32")))
        return None
    for spec in (doc.get("inputs") or {}).values():
        if isinstance(spec, dict) and "dims" in spec:
            return _DTYPE_BYTES.get(str(spec.get("dtype", "float32")))
    return None


def explicit_canonical_projection(doc: dict) -> dict:
    """The hash-input projection for an EXPLICIT doc: the FULL row (codex) so any semantic edit moves the hash.

    Rows sorted by (role, id); inputs kept in authored ORDER (a reorder is a real change); each input keeps
    every field (tensor: name/kind/shape/dtype/axes/dist; non-tensor: name/kind/value/expr); plus the row
    ``env``. (Knob-format docs keep ``shape_manifest.canonical_manifest`` unchanged -- not this function.)
    """
    rows = []
    for role in ("public", "hidden"):
        for row in doc.get(role) or []:
            rows.append({"role": role, "id": row.get("id"),
                         "env": row.get("env") or {}, "inputs": row.get("inputs") or []})
    rows.sort(key=lambda r: (r["role"], str(r["id"])))
    return {"schema_version": doc.get("schema_version"), "rows": rows}


def sanitised_view(doc: dict) -> dict:
    """The candidate-safe public projection: hidden rows sealed for either format.

    Knob-format: drop ``hidden_shapes`` and project the advisory ``explicit_view`` to public ids. Explicit:
    drop the ``hidden`` row list (the explicit format carries no advisory ``explicit_view``).
    """
    if is_explicit_shapes_doc(doc):
        return {k: v for k, v in doc.items() if k != "hidden"}
    pub = {r.get("id") for r in (doc.get("public_shapes") or [])}
    view = {k: v for k, v in doc.items() if k != "hidden_shapes"}
    ev = view.get("explicit_view")
    if isinstance(ev, dict):
        view["explicit_view"] = {k: val for k, val in ev.items() if k in pub}
    return view
