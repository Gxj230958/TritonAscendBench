"""inputs.py — build oracle/solution input dicts from shapes.yaml.

The Phase C shapes.yaml schema (consumed here):

    size_knobs: {B: 2, L: 8, D: 16}          # default knob values
    public_shapes:
      - {id: P1, name: tiny, knobs: {B: 1, L: 4, D: 8}}
      ...
    hidden_shapes:
      - {id: H1, name: h_odd, knobs: {B: 3, L: 13, D: 40}}
    inputs:
      x:      {dims: [B, L, D], dtype: float32, dist: randn}
      weight: {dims: [D],       dtype: float32, dist: ones}
      eps:    {scalar: 1.0e-5}
      scale:  {derive: "1/sqrt(D)"}           # WA.3 derived scalar
      block:  {scalar_seq: [8, 8]}            # WA.3 list literal
      idx:    {dims: [B], dtype: int64, dist: "randint:0:B"}

`build_inputs(shapes_doc, shape_id, seed)` returns the `dict` passed to
`oracle_pytorch.run(inputs)` and to the candidate solution. Tensor specs
are size-knob formulas so one inputs block covers every shape row.

This is the bridge eval.py was missing (was a Phase 2.B TODO): with it,
shapes.yaml is actually executable and the oracle can be run per shape.

WA.3 adds three non-tensor refinements to the materialisation layer:

* **derived scalars** — ``{derive: "1/sqrt(D)"}`` replaces a hardcoded
  ``{scalar: 0.125}``; the value is computed per shape row through the
  shared :func:`expr.eval_numeric` so it tracks the head-dim knob.
* **literal sequences** — a list literal (``L2_10`` ``block_size``,
  ``L1_31`` ``mrope_section``) moves to the dedicated ``scalar_seq`` key.
  ``scalar`` / ``derive`` values are constrained to JSON scalars so they
  match the W1.0 sidecar IPC type boundary; ``scalar_seq`` is a new IPC
  type carrying a list of JSON scalars.
* **dist normalisation** — :func:`parse_dist_spec` is the single shared
  bridge between the ``shapes.yaml`` ``dist`` string DSL and the
  structured ``{kind, args}`` form ``param_constraints.dists`` uses.
"""
from __future__ import annotations

from typing import Any

import torch

from triton_ascend_bench.expr import eval_numeric

_DTYPES = {
    "float32": torch.float32, "float16": torch.float16,
    "bfloat16": torch.bfloat16, "bf16": torch.bfloat16,
    "fp16": torch.float16, "fp32": torch.float32,
    "int8": torch.int8, "int32": torch.int32, "int64": torch.int64,
    "long": torch.int64, "bool": torch.bool,
    "float8_e4m3fn": torch.float8_e4m3fn,
}

# Closed enum of legal dtype alias strings — the **single authority** the
# `param_constraints.dtypes` subsection (WA.1, `specs/task_api_contract.md
# §5.2`) is bound to. `scripts/validate_tasks.py` imports this so an unknown
# dtype string (a typo) in `param_constraints` is statically rejected against
# the very same table `build_inputs` materialises with — no second list to
# drift. Kept as a `frozenset` so callers cannot mutate it.
DTYPE_ALIASES = frozenset(_DTYPES.keys())


# --------------------------------------------------------------------------
# WA.3 ③ — the shared `dist` representation bridge.
# --------------------------------------------------------------------------
#
# `shapes.yaml.inputs.*.dist` is authored as a compact string DSL (`randn`,
# `randint:0:E`, `offsets:to=B*L`, `full:expr`); `param_constraints.dists`
# (WA.1) declares the same intent structurally as `{kind, args}`. Both forms
# must agree, so a single normaliser — `parse_dist_spec` — maps the string DSL
# to a canonical `{kind, args}` dict (and `unparse_dist_spec` back). `inputs.py`
# materialisation, `validate_tasks` dist checks (WA.4) and the generators all
# route through it, so there is exactly one DSL grammar and no drift. The
# string DSL stays the authoritative `shapes.yaml` authoring form.

#: Closed dist-kind enum — bound to the `build_inputs` distribution
#: vocabulary and to the `specs/task_api_contract.md §5.2` dist-kind table.
DIST_KINDS = frozenset({
    "randn", "rand", "ones", "zeros", "eye", "tril", "randint",
    "softmax_rows", "offsets", "cumcount", "arange", "full",
})

#: Dist kinds that carry no `args` (the bare-name DSL forms).
_NULLARY_DIST_KINDS = frozenset({
    "randn", "rand", "ones", "zeros", "eye", "tril", "softmax_rows", "arange",
})


class DistSpecError(ValueError):
    """Raised on a malformed ``dist`` string DSL or ``{kind, args}`` spec."""


def parse_dist_spec(dist: Any) -> dict[str, Any]:
    """Normalise a ``dist`` declaration to a canonical ``{kind, args}`` dict.

    This is the single shared bridge between the two ``dist`` representations
    (WA.1 ``dists`` subsection / WA.3 ③): the ``shapes.yaml`` string DSL and
    the structured ``param_constraints.dists`` form. Both ``inputs.py``
    materialisation and the ``validate_tasks`` dist check route through it.

    Accepted inputs:

    * a **string DSL** form — ``"randn"``, ``"randint:<lo>:<hi>"``,
      ``"offsets:<step>"``, ``"offsets:to=<expr>"``, ``"full:<expr>"``;
    * an already-structured ``{kind, args}`` mapping (passed through after
      validation).

    Args:
        dist: the ``dist`` declaration (string DSL or ``{kind, args}`` map).

    Returns:
        A canonical ``{"kind": <kind>, "args": <dict>}`` mapping. ``args`` is
        always a dict (empty for the nullary kinds). The knob expressions in
        ``args`` (``lo`` / ``hi`` / ``step`` / ``to`` / ``value``) are kept as
        strings — :func:`eval_numeric` resolves them at materialisation time.

    Raises:
        DistSpecError: on an unknown kind, a malformed DSL form, or an
            ill-typed structured spec.
    """
    if isinstance(dist, dict):
        return _normalise_structured_dist(dist)
    if not isinstance(dist, str):
        raise DistSpecError(
            f"dist must be a string DSL or a {{kind, args}} mapping, "
            f"got {type(dist).__name__}"
        )
    text = dist.strip()
    if not text:
        raise DistSpecError("dist string is empty")
    head, _, rest = text.partition(":")
    if head in _NULLARY_DIST_KINDS:
        if rest:
            raise DistSpecError(
                f"dist {dist!r}: kind {head!r} takes no arguments"
            )
        return {"kind": head, "args": {}}
    if head == "randint":
        parts = rest.split(":")
        if len(parts) != 2 or not all(p.strip() for p in parts):
            raise DistSpecError(
                f"dist {dist!r}: 'randint' needs 'randint:<lo>:<hi>'"
            )
        return {"kind": "randint",
                "args": {"lo": parts[0].strip(), "hi": parts[1].strip()}}
    if head == "offsets":
        # `offsets:to=<expr>` (final value pinned) or `offsets:<step>`
        # (random increments ≤ step); a bare `offsets` defaults step to 3.
        if not rest:
            return {"kind": "offsets", "args": {"step": "3"}}
        if rest.startswith("to="):
            total = rest[3:].strip()
            if not total:
                raise DistSpecError(
                    f"dist {dist!r}: 'offsets:to=' needs an expression"
                )
            return {"kind": "offsets", "args": {"to": total}}
        return {"kind": "offsets", "args": {"step": rest.strip()}}
    if head == "cumcount":
        # `cumcount:to=<expr>` — a non-leading-zero cumulative count vector
        # of length `n`: element `r` is the running count through segment
        # `r`, and the final element equals `to`. Distinct from `offsets`,
        # whose leading-zero indptr puts the total in `[-1]` with a `0` in
        # `[0]`; `cumcount` matches the speculative-decode rejection-sampler
        # convention `starts = cat([zeros(1), cu[:-1]])`, `end = cu[r]`.
        if not rest.startswith("to="):
            raise DistSpecError(
                f"dist {dist!r}: 'cumcount' needs 'cumcount:to=<expr>'"
            )
        total = rest[3:].strip()
        if not total:
            raise DistSpecError(
                f"dist {dist!r}: 'cumcount:to=' needs an expression"
            )
        return {"kind": "cumcount", "args": {"to": total}}
    if head == "full":
        if not rest.strip():
            raise DistSpecError(
                f"dist {dist!r}: 'full' needs 'full:<expr>'"
            )
        return {"kind": "full", "args": {"value": rest.strip()}}
    raise DistSpecError(
        f"dist {dist!r}: unknown kind {head!r} (allowed: {sorted(DIST_KINDS)})"
    )


def _normalise_structured_dist(spec: dict[str, Any]) -> dict[str, Any]:
    """Validate and canonicalise an already-structured ``{kind, args}`` dist."""
    extra = set(spec.keys()) - {"kind", "args"}
    if extra:
        raise DistSpecError(
            f"dist spec has unknown keys {sorted(extra)} "
            f"(allowed: 'kind', 'args')"
        )
    kind = spec.get("kind")
    if not isinstance(kind, str) or kind not in DIST_KINDS:
        raise DistSpecError(
            f"dist spec kind {kind!r} not in {sorted(DIST_KINDS)}"
        )
    args = spec.get("args") or {}
    if not isinstance(args, dict):
        raise DistSpecError(
            f"dist spec args must be a mapping, got {type(args).__name__}"
        )
    if kind in _NULLARY_DIST_KINDS:
        if args:
            raise DistSpecError(f"dist kind {kind!r} takes no args")
        return {"kind": kind, "args": {}}
    if kind == "randint":
        if set(args) != {"lo", "hi"}:
            raise DistSpecError(
                f"dist 'randint' args must be exactly {{lo, hi}}, "
                f"got {sorted(args)}"
            )
    elif kind == "offsets":
        if set(args) not in ({"to"}, {"step"}):
            raise DistSpecError(
                f"dist 'offsets' args must be {{to}} or {{step}}, "
                f"got {sorted(args)}"
            )
    elif kind == "cumcount":
        if set(args) != {"to"}:
            raise DistSpecError(
                f"dist 'cumcount' args must be exactly {{to}}, "
                f"got {sorted(args)}"
            )
    elif kind == "full":
        if set(args) != {"value"}:
            raise DistSpecError(
                f"dist 'full' args must be exactly {{value}}, "
                f"got {sorted(args)}"
            )
    return {"kind": kind, "args": {k: args[k] for k in sorted(args)}}


def unparse_dist_spec(spec: dict[str, Any]) -> str:
    """Render a canonical ``{kind, args}`` dist back to the string DSL.

    Inverse of :func:`parse_dist_spec` — the round-trip
    ``parse_dist_spec(unparse_dist_spec(spec))`` reproduces ``spec``. Used
    where the ``shapes.yaml`` authoring form must be re-emitted (generators).

    Args:
        spec: a ``{kind, args}`` mapping (or string DSL — passed through
            after canonicalisation).

    Returns:
        The compact string DSL form.

    Raises:
        DistSpecError: on a malformed spec.
    """
    canon = parse_dist_spec(spec)
    kind, args = canon["kind"], canon["args"]
    if kind in _NULLARY_DIST_KINDS:
        return kind
    if kind == "randint":
        return f"randint:{args['lo']}:{args['hi']}"
    if kind == "offsets":
        if "to" in args:
            return f"offsets:to={args['to']}"
        return f"offsets:{args['step']}"
    if kind == "cumcount":
        return f"cumcount:to={args['to']}"
    if kind == "full":
        return f"full:{args['value']}"
    raise DistSpecError(f"cannot unparse dist kind {kind!r}")


def _resolve_dims(dims: list[Any], knobs: dict[str, int]) -> list[int]:
    """Resolve a tensor's ``dims`` list to concrete integer extents.

    Integer dims pass through; string dims are size-knob arithmetic
    expressions (e.g. ``"2*D"`` / ``"L+1"``) evaluated by the shared
    AST-whitelist numeric evaluator (WA.2 :func:`expr.eval_numeric`) — never
    a bare :func:`eval`.

    Args:
        dims: the ``shapes.yaml`` dims list (ints and/or knob expressions).
        knobs: the resolved size-knob values for this shape row.

    Returns:
        The concrete integer extents.

    Raises:
        ValueError: if a dim is neither an int nor a string.
        ExprError: if a string dim is not a valid numeric expression.
    """
    out: list[int] = []
    for d in dims:
        if isinstance(d, int):
            out.append(d)
        elif isinstance(d, str):
            out.append(int(eval_numeric(d, knobs)))
        else:
            raise ValueError(f"bad dim {d!r}")
    return out


def _resolve_dtype(name: str) -> "torch.dtype":
    """Resolve a ``shapes.yaml`` dtype alias to a ``torch.dtype`` — fail-loud.

    The dtype alias must be a key of the shared :data:`DTYPE_ALIASES` table
    (`specs/task_api_contract.md §5.2`). WA.4 ③: an unknown alias (a typo) is
    a hard :class:`DistSpecError`-style failure — there is **no** silent
    fallback to ``float32``, which would have materialised oracle and
    candidate at a dtype neither the task author nor the validator intended.

    Args:
        name: the dtype alias string from a ``shapes.yaml`` tensor spec.

    Returns:
        The corresponding ``torch.dtype``.

    Raises:
        TypeError: if ``name`` is not a known dtype alias.
    """
    dtype = _DTYPES.get(name)
    if dtype is None:
        raise TypeError(
            f"unknown dtype {name!r}; allowed dtype aliases are "
            f"{sorted(DTYPE_ALIASES)} (no silent float32 fallback — WA.4)")
    return dtype


def _make_tensor(spec: dict[str, Any], knobs: dict[str, int],
                 gen: torch.Generator) -> torch.Tensor:
    """Materialise one tensor input from a ``shapes.yaml`` tensor spec.

    The spec's ``dist`` declaration is normalised through the shared
    :func:`parse_dist_spec` bridge, so the string DSL and the structured
    ``{kind, args}`` form materialise identically. An unknown dist kind is a
    hard :class:`DistSpecError` and an unknown dtype is a hard
    :class:`TypeError` — there is no silent fallback for either (WA.3 / WA.4).
    """
    dims = _resolve_dims(spec.get("dims", []), knobs)
    dtype = _resolve_dtype(str(spec.get("dtype", "float32")))
    dist = parse_dist_spec(spec.get("dist", "randn"))
    kind, args = dist["kind"], dist["args"]
    shape = tuple(dims) if dims else ()

    if kind == "randn":
        t = torch.randn(shape, generator=gen, dtype=torch.float32)
    elif kind == "rand":
        t = torch.rand(shape, generator=gen, dtype=torch.float32)
    elif kind == "ones":
        t = torch.ones(shape, dtype=torch.float32)
    elif kind == "zeros":
        t = torch.zeros(shape, dtype=torch.float32)
    elif kind == "eye":
        n = shape[-1] if shape else 1
        t = torch.eye(n, dtype=torch.float32)
        if len(shape) > 2:
            t = t.expand(shape).contiguous()
    elif kind == "tril":
        n = shape[-1] if shape else 1
        base = torch.tril(torch.randn(shape, generator=gen, dtype=torch.float32))
        t = base + torch.eye(n, dtype=torch.float32) * (n + 1)  # well-conditioned
    elif kind == "randint":
        lo_i = int(eval_numeric(str(args["lo"]), knobs))
        hi_i = max(int(eval_numeric(str(args["hi"]), knobs)), lo_i + 1)
        t = torch.randint(lo_i, hi_i, shape, generator=gen, dtype=torch.int64)
    elif kind == "softmax_rows":
        t = torch.softmax(torch.randn(shape, generator=gen, dtype=torch.float32), dim=-1)
    elif kind == "offsets":
        # Monotonic [0, c1, ..., total] int64 of length = shape[-1].
        # `offsets:<step>`     random increments ≤ step.
        # `offsets:to=<expr>`  final value == eval_numeric(expr, knobs) —
        #   used when the offset total must equal another tensor's length
        #   (cu_seqlens[-1] == n_tokens, etc.).
        n = (shape[-1] if shape else 2)
        segs = max(n - 1, 1)
        t = torch.zeros(n, dtype=torch.int64)
        if "to" in args:
            total = int(eval_numeric(str(args["to"]), knobs))
            # Distribute exactly `total` over `segs` non-negative
            # increments (empty segments are valid — oracles skip s==e).
            inc = torch.zeros(segs, dtype=torch.int64)
            if total > 0:
                base = total // segs
                inc += base
                inc[: total - base * segs] += 1
            t[1:] = torch.cumsum(inc, 0)
        else:
            step = int(eval_numeric(str(args["step"]), knobs))
            inc = torch.randint(1, max(step, 2) + 1, (segs,), generator=gen)
            t[1:] = torch.cumsum(inc, 0)
        return t
    elif kind == "cumcount":
        # Non-leading-zero cumulative count vector of length `n`: element `r`
        # is the running token count through segment `r`, the final element
        # equals `to`. Segment `r` then spans `[c[r-1], c[r])` with `c[-1]`
        # taken as 0 — the speculative-decode rejection-sampler convention
        # (`starts = cat([zeros(1), cu[:-1]])`, `end = cu[r]`). Contrast
        # `offsets`, whose leading-zero indptr keeps the total in `[-1]`.
        n = shape[-1] if shape else 1
        total = int(eval_numeric(str(args["to"]), knobs))
        inc = torch.zeros(n, dtype=torch.int64)
        if total > 0:
            base = total // n
            inc += base
            inc[: total - base * n] += 1
        return torch.cumsum(inc, 0)
    elif kind == "arange":
        n = shape[-1] if shape else 1
        return torch.arange(n, dtype=torch.int64).reshape(shape)
    elif kind == "full":
        # Constant-filled int64 tensor; value = eval_numeric(expr, knobs).
        # Used when a length vector must match an offsets cumsum (e.g.
        # b_seq_len == per-segment length of b_start_loc).
        val = int(eval_numeric(str(args["value"]), knobs))
        return torch.full(shape, val, dtype=torch.int64)
    else:  # pragma: no cover - parse_dist_spec already gates the kind set.
        raise DistSpecError(f"unhandled dist kind {kind!r}")

    if dtype in (torch.int8, torch.int32, torch.int64, torch.bool):
        # A `randn` / `rand` source is float-valued — scale it into a small
        # non-negative integer range before the cast. (`dist` is the parsed
        # `{kind, args}` dict, so the discriminant is `kind`, not `dist`.)
        if kind in ("randn", "rand"):
            t = (t.abs() * 8).to(dtype)
        else:
            t = t.to(dtype)
    elif dtype == torch.float8_e4m3fn:
        t = t.clamp(-448, 448).to(dtype)
    else:
        t = t.to(dtype)
    return t


# JSON-representable non-tensor scalar types accepted in an inputs dict.
# `scalar` / `derive` values must be one of these — the W1.0 sidecar IPC
# carries no pickle (`inputs.py` `split_inputs`, `process_isolation.py`
# `_merge`), so a non-JSON scalar (e.g. a list under `scalar`) cannot cross
# the channel. WA.3 ② makes that boundary a hard error at materialisation.
_JSON_SCALAR_TYPES = (bool, int, float, str, type(None))


def _materialise_scalar(name: str, spec: dict[str, Any],
                        knobs: dict[str, int]) -> Any:
    """Materialise one non-tensor input (``scalar`` / ``derive`` / ``scalar_seq``).

    WA.3 ① / ②: the value of a ``scalar`` (a literal) or a ``derive`` rule
    (a knob expression evaluated through :func:`expr.eval_numeric`) must be a
    JSON scalar — the W1.0 sidecar IPC type boundary. A ``scalar_seq`` is a
    list of JSON scalars (the dedicated key for list literals such as
    ``L2_10`` ``block_size`` and ``L1_31`` ``mrope_section``).

    Args:
        name: the input name (for error messages).
        spec: the ``shapes.yaml`` input spec mapping.
        knobs: the resolved size-knob values for the active shape row.

    Returns:
        The materialised non-tensor value (a JSON scalar or a list of them).

    Raises:
        TypeError: if a ``scalar`` / ``derive`` value is not a JSON scalar, or
            a ``scalar_seq`` is not a list of JSON scalars.
        ExprError: if a ``derive`` expression is not a valid numeric formula.
    """
    if "scalar" in spec:
        value = spec["scalar"]
        if not isinstance(value, _JSON_SCALAR_TYPES):
            raise TypeError(
                f"input {name!r}: 'scalar' value is {type(value).__name__}; "
                f"only JSON scalars (bool/int/float/str/None) are allowed — "
                f"use 'scalar_seq' for a list literal"
            )
        return value
    if "derive" in spec:
        # WA.3 ①: a derived scalar — `{derive: "1/sqrt(D)"}` — computed per
        # shape row so it tracks the knob universe (replaces a hardcoded
        # `{scalar: 0.125}`). `eval_numeric` returns a float.
        value = eval_numeric(str(spec["derive"]), knobs)
        if not isinstance(value, _JSON_SCALAR_TYPES):
            raise TypeError(
                f"input {name!r}: 'derive' value is {type(value).__name__}; "
                f"a derive rule must yield a JSON scalar"
            )
        return value
    # `scalar_seq` — a list literal of JSON scalars.
    seq = spec["scalar_seq"]
    if not isinstance(seq, list):
        raise TypeError(
            f"input {name!r}: 'scalar_seq' must be a list, "
            f"got {type(seq).__name__}"
        )
    for i, item in enumerate(seq):
        if not isinstance(item, _JSON_SCALAR_TYPES):
            raise TypeError(
                f"input {name!r}: 'scalar_seq'[{i}] is "
                f"{type(item).__name__}; every element must be a JSON scalar"
            )
    return list(seq)


def build_inputs(
    shapes_doc: dict[str, Any], shape_id: str, seed: int = 0,
) -> dict[str, Any]:
    """Construct the input dict for a given shape row.

    Non-tensor inputs are materialised per :func:`_materialise_scalar`:
    a ``scalar`` literal is passed through verbatim, a ``derive`` rule is
    evaluated against the shape row's knobs, and a ``scalar_seq`` carries a
    list literal. ``scalar_knob`` is the legacy derived-integer form. Tensor
    specs are materialised from size-knob formulas with a fixed seed so
    oracle and candidate see identical inputs.
    """
    rows = (shapes_doc.get("public_shapes", [])
            + shapes_doc.get("hidden_shapes", []))
    row = next((r for r in rows if r.get("id") == shape_id), None)
    if row is None:
        raise KeyError(f"shape id {shape_id!r} not in shapes.yaml")

    knobs = dict(shapes_doc.get("size_knobs", {}))
    knobs.update(row.get("knobs", {}))

    gen = torch.Generator()
    gen.manual_seed(seed)

    out: dict[str, Any] = {}
    for name, spec in (shapes_doc.get("inputs", {}) or {}).items():
        if not isinstance(spec, dict):
            out[name] = spec
            continue
        if "scalar" in spec or "derive" in spec or "scalar_seq" in spec:
            out[name] = _materialise_scalar(name, spec, knobs)
        elif "scalar_knob" in spec:
            out[name] = int(eval_numeric(str(spec["scalar_knob"]), knobs))
        else:
            out[name] = _make_tensor(spec, knobs, gen)
    return out


def shape_ids(shapes_doc: dict[str, Any]) -> tuple[list[str], list[str]]:
    """(public_ids, hidden_ids)."""
    pub = [r["id"] for r in shapes_doc.get("public_shapes", [])]
    hid = [r["id"] for r in shapes_doc.get("hidden_shapes", [])]
    return pub, hid


# --- W1.0: inputs materialisation for the process-isolation IPC ----------
#
# The runner (trusted parent) materialises a shape's inputs ONCE, then ships
# them to the candidate child over the narrow safetensors+sidecar channel so
# the child never gets `repo_root` / `shapes.yaml` / `build_inputs`. A built
# inputs dict mixes tensors and non-tensor scalars; the IPC splits the two:
#
#   * tensor values     -> safetensors payload  (data-only, no pickle)
#   * non-tensor values -> JSON sidecar manifest (scalar / derive /
#     scalar_knob results, and scalar_seq lists)
#
# `split_inputs` produces the (tensors, sidecar) pair the runner writes;
# `merge_inputs` is the inverse the child uses to rebuild the full dict.


def _json_sidecar_value(name: str, value: Any) -> Any:
    """Coerce a non-tensor input value to its JSON-sidecar representation.

    The IPC sidecar carries no pickle, so a non-tensor value must be a JSON
    scalar (``bool`` / ``int`` / ``float`` / ``str`` / ``None``) or a
    ``scalar_seq`` list of JSON scalars (WA.3 ②). Anything else fails loudly.

    Args:
        name: the input name (for error messages).
        value: the non-tensor input value.

    Returns:
        The value, ready for ``json.dumps`` (lists are copied).

    Raises:
        TypeError: if the value is not a JSON scalar or a list of them.
    """
    if isinstance(value, _JSON_SCALAR_TYPES):
        return value
    if isinstance(value, list):
        for i, item in enumerate(value):
            if not isinstance(item, _JSON_SCALAR_TYPES):
                raise TypeError(
                    f"input {name!r}: scalar_seq element [{i}] is "
                    f"{type(item).__name__}; the IPC carries only lists of "
                    f"JSON scalars (bool/int/float/str/None)")
        return list(value)
    raise TypeError(
        f"input {name!r} is {type(value).__name__}; only torch.Tensor, "
        f"JSON scalars (bool/int/float/str/None), or a scalar_seq list of "
        f"them cross the IPC")


def split_inputs(
    built: dict[str, Any],
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """Split a built inputs dict into (tensors, sidecar).

    ``tensors`` holds every ``torch.Tensor`` value (safetensors-serialisable);
    ``sidecar`` is a JSON-serialisable manifest of the non-tensor values
    (``scalar`` / ``derive`` / ``scalar_knob`` results, and ``scalar_seq``
    lists) keyed by input name.

    Raises ``TypeError`` if a non-tensor value is not JSON-representable —
    the IPC carries no pickle, so unexpected object types must fail loudly.
    """
    tensors: dict[str, torch.Tensor] = {}
    sidecar: dict[str, Any] = {}
    for name, value in built.items():
        if torch.is_tensor(value):
            tensors[name] = value.detach().contiguous().cpu()
        else:
            sidecar[name] = _json_sidecar_value(name, value)
    return tensors, sidecar


def merge_inputs(
    tensors: dict[str, torch.Tensor], sidecar: dict[str, Any],
) -> dict[str, Any]:
    """Rebuild a full mixed inputs dict from a safetensors payload + sidecar.

    Inverse of :func:`split_inputs`; used by the candidate child to
    reconstruct the dict ``build_inputs`` would have returned, without ever
    touching ``shapes.yaml`` or ``build_inputs`` itself.
    """
    merged: dict[str, Any] = dict(tensors)
    for name, value in sidecar.items():
        merged[name] = value
    return merged
