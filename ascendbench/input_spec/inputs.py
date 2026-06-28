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
  match the  sidecar IPC type boundary; ``scalar_seq`` is a new IPC
  type carrying a list of JSON scalars.
* **dist normalisation** — :func:`parse_dist_spec` is the single shared
  bridge between the ``shapes.yaml`` ``dist`` string DSL and the
  structured ``{kind, args}`` form ``param_constraints.dists`` uses.
"""
from __future__ import annotations

from typing import Any

import torch

from ascendbench.input_spec.expr import eval_numeric

_DTYPES = {
    "float32": torch.float32, "float16": torch.float16,
    "bfloat16": torch.bfloat16, "bf16": torch.bfloat16,
    "fp16": torch.float16, "fp32": torch.float32,
    "int8": torch.int8, "int32": torch.int32, "int64": torch.int64,
    "long": torch.int64, "bool": torch.bool,
    "float8_e4m3fn": torch.float8_e4m3fn,
}

# Closed enum of legal dtype alias strings — the **single authority** the
# `param_constraints.dtypes` subsection (WA.1, `docs/reference/task_api_contract.md
# §5.2`) is bound to. `scripts/corpus/validate_tasks.py` imports this so an unknown
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
#: vocabulary and to the `docs/reference/task_api_contract.md §5.2` dist-kind table.
DIST_KINDS = frozenset({
    "randn", "randn_inf", "randn_inf_excl",
    "rand", "ones", "zeros", "eye", "tril", "randint",
    "tril_chunk", "softmax_rows", "l2norm_rows", "logsigmoid", "offsets",
    "cumcount", "arange", "randperm", "cyclic_perm", "full", "routing_ids",
    "window_pos", "window_req", "gumbel_temps", "gumbel_routing",
    "moe_sorted_ids", "rope_cos_sin",
})

#: Dist kinds that carry no `args` (the bare-name DSL forms).
_NULLARY_DIST_KINDS = frozenset({
    "randn", "randn_inf", "rand", "ones", "zeros", "eye", "tril",
    "tril_chunk", "softmax_rows", "l2norm_rows", "logsigmoid", "arange",
    "randperm", "cyclic_perm", "gumbel_temps", "rope_cos_sin",
})

#: Fraction of `randn_inf` entries replaced with +inf. Models a
#: flash-decoding empty-split LSE sentinel (FA2 returns +inf for a
#: zero-length split); merge kernels collapse +inf -> -inf to drop the
#: empty side. A single fixed fraction keeps a healthy mix of empty /
#: non-empty splits per shape row so the sentinel branch always fires
#: while the finite merge path stays dominant.
_RANDN_INF_FRACTION = 0.3


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
    if head == "randn_inf_excl":
        # `randn_inf_excl:<ref>` -- randn with +inf injected at the same
        # fraction as randn_inf, but ONLY where the referenced (already
        # materialised) tensor is FINITE, so the two never carry +inf at the
        # same position. Used to pair the two flash-decoding split LSEs so a
        # position is empty on at most ONE side (both-empty is the undefined /
        # non-physical NaN domain; a merge always has >=1 non-empty split).
        if not rest.strip():
            raise DistSpecError(
                f"dist {dist!r}: 'randn_inf_excl' needs 'randn_inf_excl:<ref>'"
            )
        return {"kind": "randn_inf_excl", "args": {"ref": rest.strip()}}
    if head == "routing_ids":
        # `routing_ids:<num_experts_expr>` -- a per-token MoE expert-assignment
        # vector in [0, E) with a deterministic drop fraction set to -1 (the
        # upstream "token dropped / no expert" sentinel act_and_mul_kernel skips
        # via `if expert_id == -1: return`). Guarantees >=1 dropped and >=1 valid
        # entry whenever n >= 2 so the routing mask is always discriminating.
        if not rest.strip():
            raise DistSpecError(
                f"dist {dist!r}: 'routing_ids' needs 'routing_ids:<num_experts>'"
            )
        return {"kind": "routing_ids", "args": {"E": rest.strip()}}
    if head == "moe_sorted_ids":
        # `moe_sorted_ids:<B_expr>` -- a faithful moe_align_block_size
        # sorted_token_ids buffer: distinct route-slot ids in [0, B) padded
        # with the sentinel == B (num_valid_tokens). Unique ids are required
        # because the fused_moe_kernel scatters C[sorted_token_ids[r]] (a
        # duplicate id would race-write the same row).
        if not rest.strip():
            raise DistSpecError(
                f"dist {dist!r}: 'moe_sorted_ids' needs 'moe_sorted_ids:<B>'"
            )
        return {"kind": "moe_sorted_ids", "args": {"B": rest.strip()}}
    if head == "gumbel_routing":
        # `gumbel_routing:<R>` -- per-token request index in [0, R) for the
        # gumbel-max sampler, pinned so token 0 -> request 0 and token 1 ->
        # request 1 (paired with `gumbel_temps`, which pins request 0 greedy and
        # request 1 noisy). This GUARANTEES by construction that every shape
        # routes >=1 token through the greedy branch and >=1 through the gumbel
        # branch -- random routing alone would not. Remaining tokens are random.
        if not rest.strip():
            raise DistSpecError(
                f"dist {dist!r}: 'gumbel_routing' needs 'gumbel_routing:<R>'"
            )
        return {"kind": "gumbel_routing", "args": {"R": rest.strip()}}
    if head == "window_pos":
        # `window_pos:<W>` -- per-token local position within fixed-size draft
        # windows of length W tiling [0, n): pos[t] = t % W. Models the
        # speculative-decode draft-window local index the penalties kernel reads
        # as `expanded_local_pos` (kernel.py draft scan: start = t - pos). pos<=t
        # by construction, so the kernel's unclamped `token_ids[t-pos+1 .. t]`
        # reads are always in-bounds. Paired with `window_req:<W>:<E>` so request
        # ids are constant within each window (the upstream layout precondition).
        if not rest.strip():
            raise DistSpecError(
                f"dist {dist!r}: 'window_pos' needs 'window_pos:<W>'"
            )
        return {"kind": "window_pos", "args": {"W": rest.strip()}}
    if head == "window_req":
        # `window_req:<W>:<E>` -- per-token request id, CONSTANT within each
        # length-W draft window: req[t] = (t // W) % E. The window-constant
        # routing the penalties kernel's `expanded_idx_mapping` carries (each
        # speculative-decode window belongs to one request). Pairs with
        # `window_pos:<W>` on the SAME W so (req, local_pos) are mutually
        # consistent.
        parts = rest.split(":")
        if len(parts) != 2 or not all(p.strip() for p in parts):
            raise DistSpecError(
                f"dist {dist!r}: 'window_req' needs 'window_req:<W>:<E>'"
            )
        return {"kind": "window_req",
                "args": {"W": parts[0].strip(), "E": parts[1].strip()}}
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
    elif kind == "randn_inf_excl":
        if set(args) != {"ref"}:
            raise DistSpecError(
                f"dist 'randn_inf_excl' args must be exactly {{ref}}, "
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
    elif kind == "routing_ids":
        if set(args) != {"E"}:
            raise DistSpecError(
                f"dist 'routing_ids' args must be exactly {{E}}, "
                f"got {sorted(args)}"
            )
    elif kind == "moe_sorted_ids":
        if set(args) != {"B"}:
            raise DistSpecError(
                f"dist 'moe_sorted_ids' args must be exactly {{B}}, "
                f"got {sorted(args)}"
            )
    elif kind == "gumbel_routing":
        if set(args) != {"R"}:
            raise DistSpecError(
                f"dist 'gumbel_routing' args must be exactly {{R}}, "
                f"got {sorted(args)}"
            )
    elif kind == "window_pos":
        if set(args) != {"W"}:
            raise DistSpecError(
                f"dist 'window_pos' args must be exactly {{W}}, "
                f"got {sorted(args)}"
            )
    elif kind == "window_req":
        if set(args) != {"W", "E"}:
            raise DistSpecError(
                f"dist 'window_req' args must be exactly {{W, E}}, "
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
    if kind == "randn_inf_excl":
        return f"randn_inf_excl:{args['ref']}"
    if kind == "offsets":
        if "to" in args:
            return f"offsets:to={args['to']}"
        return f"offsets:{args['step']}"
    if kind == "cumcount":
        return f"cumcount:to={args['to']}"
    if kind == "full":
        return f"full:{args['value']}"
    if kind == "routing_ids":
        return f"routing_ids:{args['E']}"
    if kind == "moe_sorted_ids":
        return f"moe_sorted_ids:{args['B']}"
    if kind == "gumbel_routing":
        return f"gumbel_routing:{args['R']}"
    if kind == "window_pos":
        return f"window_pos:{args['W']}"
    if kind == "window_req":
        return f"window_req:{args['W']}:{args['E']}"
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


def _resolve_dtype(name: str) -> torch.dtype:
    """Resolve a ``shapes.yaml`` dtype alias to a ``torch.dtype`` — fail-loud.

    The dtype alias must be a key of the shared :data:`DTYPE_ALIASES` table
    (`docs/reference/task_api_contract.md §5.2`). WA.4 ③: an unknown alias (a typo) is
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
                 gen: torch.Generator,
                 out: dict[str, Any] | None = None) -> torch.Tensor:
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
    elif kind == "randn_inf":
        # randn with a fixed fraction of entries set to +inf -- a
        # flash-decoding empty-split LSE sentinel. The two LSE tensors are
        # materialised independently (no cross-tensor coordination), so some
        # (token, head) positions get +inf on one side, some on both. The
        # merge kernel maps +inf -> -inf, so a one-sided +inf selects the
        # other split and a both-sided +inf yields NaN at that position --
        # the oracle must mirror this exactly (matching NaN positions pass
        # the inf/nan-position comparator).
        t = torch.randn(shape, generator=gen, dtype=torch.float32)
        inf_mask = torch.rand(shape, generator=gen) < _RANDN_INF_FRACTION
        t = torch.where(inf_mask, torch.full_like(t, float("inf")), t)
    elif kind == "randn_inf_excl":
        # randn with +inf injected at _RANDN_INF_FRACTION, but ONLY where the
        # referenced (already materialised) tensor is FINITE -- so the two
        # never carry +inf at the same position (no both-empty). both-empty is
        # the undefined / non-physical NaN domain (a flash-decoding merge
        # always has >=1 non-empty split); excluding it keeps the load-bearing
        # one-sided +inf -> -inf DROP branch fully exercised on BOTH sides.
        # Fail-closed: the ref tensor must be materialised earlier (declared
        # before this input in shapes.yaml) and share this tensor's shape.
        ref_name = str(args["ref"])
        if out is None or ref_name not in out:
            raise DistSpecError(
                f"randn_inf_excl:{ref_name} -- reference tensor {ref_name!r} "
                f"not materialised before this input; declare it earlier in "
                f"shapes.yaml inputs:"
            )
        ref = out[ref_name]
        if not torch.is_tensor(ref) or tuple(ref.shape) != shape:
            raise DistSpecError(
                f"randn_inf_excl:{ref_name} -- reference must be a tensor with "
                f"the same shape {shape}, got "
                f"{tuple(ref.shape) if torch.is_tensor(ref) else type(ref).__name__}"
            )
        t = torch.randn(shape, generator=gen, dtype=torch.float32)
        inf_mask = (torch.rand(shape, generator=gen) < _RANDN_INF_FRACTION) & (
            ref != float("inf")
        )
        t = torch.where(inf_mask, torch.full_like(t, float("inf")), t)
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
        # tril of randn; for square last-two dims we add a diagonal bias
        # so the matrix is well-conditioned for triangular solves. Non-
        # square last-two (e.g. wrapper-required [B, T, H, BT]) skip the
        # bias because broadcasting eye(n) against the non-square plane
        # is undefined; oracles in that case mask the upper triangle
        # internally per-chunk so the bias would be a no-op anyway.
        base = torch.tril(torch.randn(shape, generator=gen, dtype=torch.float32))
        if len(shape) >= 2 and shape[-1] == shape[-2]:
            n = shape[-1]
            t = base + torch.eye(n, dtype=torch.float32) * (n + 1)  # well-conditioned
        else:
            t = base
    elif kind == "tril_chunk":
        # Strict lower-triangular per chunk in the (T, BT) plane. For
        # shape [B, T, H, BT], infer the chunk size from the last dim
        # and apply the same [BT, BT] strict-lower mask to each T-slice
        # chunk; the kernel adds identity itself, so the diagonal stays 0.
        if len(shape) != 4:
            raise DistSpecError(
                f"dist 'tril_chunk' needs shape [B, T, H, BT], got {shape}"
            )
        b_dim, t_dim, h_dim, cs = shape
        if cs <= 0:
            raise DistSpecError(
                f"dist 'tril_chunk' needs a positive last dim, got shape {shape}"
            )
        if t_dim % cs != 0:
            raise DistSpecError(
                f"dist 'tril_chunk' requires T % chunk == 0, got shape {shape}"
            )
        base = torch.randn(shape, generator=gen, dtype=torch.float32)
        mask = torch.tril(torch.ones((cs, cs), dtype=torch.bool), diagonal=-1)
        t = base.reshape(b_dim, t_dim // cs, cs, h_dim, cs)
        t = t * mask.view(1, 1, cs, 1, cs)
        t = t.reshape(shape)
    elif kind == "randint":
        lo_i = int(eval_numeric(str(args["lo"]), knobs))
        hi_i = max(int(eval_numeric(str(args["hi"]), knobs)), lo_i + 1)
        t = torch.randint(lo_i, hi_i, shape, generator=gen, dtype=torch.int64)
    elif kind == "softmax_rows":
        t = torch.softmax(torch.randn(shape, generator=gen, dtype=torch.float32), dim=-1)
    elif kind == "l2norm_rows":
        # randn rows L2-normalised over the last dim (unit-norm keys). The
        # delta-rule / gated-delta recurrence is only numerically stable when
        # the keys are unit-norm -- this is the upstream input recipe
        # (fla gated_delta_rule docstring: k = F.normalize(randn, p=2, dim=-1)).
        base = torch.randn(shape, generator=gen, dtype=torch.float32)
        t = torch.nn.functional.normalize(base, p=2.0, dim=-1)
    elif kind == "logsigmoid":
        # logsigmoid(rand) -- a strictly negative gate so exp(g) in (0, 1) is a
        # genuine decay (upstream uses g = F.logsigmoid(rand) for the gated
        # delta-rule forget gate; a positive gate would make exp(g) > 1 grow
        # the recurrent state without bound).
        base = torch.rand(shape, generator=gen, dtype=torch.float32)
        t = torch.nn.functional.logsigmoid(base)
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
        # Honour the declared dtype: some upstream cumulative-count tensors are
        # int32 on device (e.g. vLLM `cu_num_draft_tokens`, built from an int32
        # numpy cumsum), and a kernel that mixes the loaded value with an int32
        # literal in a dynamic ternary will not compile if forced to int64.
        return torch.cumsum(inc, 0).to(dtype)
    elif kind == "arange":
        n = shape[-1] if shape else 1
        return torch.arange(n, dtype=torch.int64).reshape(shape)
    elif kind == "rope_cos_sin":
        # Real RoPE cos/sin cache, shape [max_pos, rotary_dim]: row p is
        # [cos(p*inv_freq), sin(p*inv_freq)] with inv_freq = base^(-arange(0,
        # rotary_dim, 2)/rotary_dim) (length rotary_dim/2), base = 10000. The
        # MLA-decode kernel reads cos at columns [0, rotary_dim/2) and sin at
        # [rotary_dim/2, rotary_dim) -- so cos and sin each occupy half the row.
        # Deterministic (no RNG); the faithful rotation table (cos^2+sin^2==1),
        # not a random twin.
        if len(shape) != 2:
            raise DistSpecError(
                f"dist 'rope_cos_sin' needs shape [max_pos, rotary_dim], got {shape}"
            )
        max_pos, rotary_dim = shape
        if rotary_dim % 2 != 0:
            raise DistSpecError(
                f"dist 'rope_cos_sin' needs an even rotary_dim, got {rotary_dim}"
            )
        inv_freq = 1.0 / (
            10000.0 ** (torch.arange(0, rotary_dim, 2, dtype=torch.float32) / rotary_dim)
        )
        ang = torch.outer(torch.arange(max_pos, dtype=torch.float32), inv_freq)
        return torch.cat([torch.cos(ang), torch.sin(ang)], dim=-1).to(dtype)
    elif kind == "randperm":
        # Random permutation of [0, n) -- a non-trivial bijection. MoE token
        # reordering (argsort of expert ids) IS such a permutation; a bare
        # arange (identity) would let an inverse-permutation scatter kernel
        # pass even when it computes the wrong direction.
        n = shape[-1] if shape else 1
        return torch.randperm(n, generator=gen).to(torch.int64).reshape(shape)
    elif kind == "cyclic_perm":
        # Single-cycle permutation `perm[i] = (i + 1) % n` -- a GUARANTEED
        # non-identity bijection of [0, n) (every position is remapped for
        # n >= 2). Unlike `randperm`, which can collapse to the identity at
        # small n (e.g. n == 2 hits [0, 1] half the time), this never lets an
        # index-ignoring kernel coincide with the oracle: the kernel that
        # reads/writes cache line `i` instead of `perm[i]` mismatches on every
        # row. Used for `conv_state_indices` (causal_conv1d_update), where the
        # index selects which cache line each batch element reads its state
        # from and writes the rolled state back to.
        n = shape[-1] if shape else 1
        idx_dtype = dtype if dtype in (torch.int32, torch.int64) else torch.int64
        return (
            torch.arange(1, n + 1, dtype=idx_dtype).remainder(n).reshape(shape)
        )
    elif kind == "routing_ids":
        # MoE per-token expert assignment: a length-n int64 vector of expert
        # ids in [0, E) with a deterministic drop fraction set to -1, the
        # upstream "token dropped / no expert" sentinel that act_and_mul_kernel
        # honours via `if expert_id == -1: return` (the masked row is left
        # untouched). Random valid ids over [0, E) come from the seeded gen; a
        # fixed ~1/3 of the positions are then forced to -1. To make the routing
        # mask ALWAYS discriminating (a kernel ignoring the ids would write the
        # dropped rows), position 0 is pinned valid and position 1 is pinned -1
        # whenever n >= 2 -- so every shape has >=1 written and >=1 skipped row.
        n = shape[-1] if shape else 1
        E = max(int(eval_numeric(str(args["E"]), knobs)), 1)
        ids = torch.randint(0, E, (n,), generator=gen, dtype=torch.int64)
        if n >= 1:
            drop = torch.rand(n, generator=gen) < 0.34
            ids[drop] = -1
            ids[0] = int(torch.randint(0, E, (1,), generator=gen).item())  # pin valid
            if n >= 2:
                ids[1] = -1  # pin dropped -> guaranteed discrimination
        return ids.reshape(shape)
    elif kind == "moe_sorted_ids":
        # `moe_align_block_size` output: the sorted_token_ids buffer the
        # fused_moe_kernel scatters its per-block GEMM into. Each of its `n`
        # slots holds a flattened route-slot id in [0, B); the kernel stores
        # block row r to C[sorted_token_ids[r], :], so the ids MUST be unique
        # (injective) -- a duplicate id makes two blocks race-write the same C
        # row (non-atomic tl.store, undefined winner), which no deterministic
        # oracle can match. Unused slots carry the sentinel == B (the kernel's
        # num_valid_tokens), masked out via `offs_token < num_valid_tokens`.
        # A bare `randint:0:B` (the scaffold default) collides heavily and is
        # un-oracalable; this draws min(n, B) DISTINCT ids, scatters them to
        # random slot positions, and pads the rest with the sentinel -- the
        # real alignment invariant, not a degenerate narrowing.
        # When the buffer is shorter than B (n = E*4 < B, e.g. shape H1) only
        # min(n, B) tokens are routed; the rest are legitimately unrouted (a
        # decode-like regime with fewer aligned slots than tokens) and their
        # output rows stay zero -- this partial-routing case is in scope, not a
        # bug. The injectivity (no duplicate valid id) is what matters for the
        # kernel's C[token_id] scatter to be deterministic.
        n = shape[-1] if shape else 1
        B = max(int(eval_numeric(str(args["B"]), knobs)), 1)
        n_valid = min(n, B)
        # distinct token ids (a size-n_valid prefix of a permutation of [0, B))
        token_ids = torch.randperm(B, generator=gen)[:n_valid].to(torch.int64)
        # scatter them to random distinct slot positions; rest = sentinel B
        out = torch.full((n,), B, dtype=torch.int64)
        slots = torch.randperm(n, generator=gen)[:n_valid]
        out[slots] = token_ids
        return out.reshape(shape)
    elif kind == "gumbel_routing":
        # Per-token request index in [0, R) for the gumbel-max sampler. Random
        # ids from the seeded gen, then token 0 is pinned to request 0 and token
        # 1 to request 1 (when present). Paired with `gumbel_temps` (request 0
        # greedy, request 1 noisy), this guarantees by construction that every
        # shape routes >=1 token through the greedy branch and >=1 through the
        # gumbel-noise branch -- so both kernel paths are always exercised and
        # any kernel ignoring `expanded_idx_mapping` is discriminated.
        n = shape[-1] if shape else 1
        R = max(int(eval_numeric(str(args["R"]), knobs)), 1)
        ids = torch.randint(0, R, (n,), generator=gen, dtype=torch.int64)
        if n >= 1:
            ids[0] = 0  # token 0 -> request 0 (greedy)
        if n >= 2 and R >= 2:
            ids[1] = 1  # token 1 -> request 1 (noisy)
        return ids.reshape(shape)
    elif kind == "window_pos":
        # Per-token local position within fixed-size draft windows tiling
        # [0, n): pos[t] = t % W. Deterministic (no RNG) so it stays consistent
        # with the paired `window_req` over the same W. pos<=t by construction
        # keeps the penalties kernel's unclamped draft scan in-bounds.
        n = shape[-1] if shape else 1
        W = max(int(eval_numeric(str(args["W"]), knobs)), 1)
        t = torch.arange(n, dtype=torch.int64)
        return (t % W).reshape(shape)
    elif kind == "window_req":
        # Per-token request id, CONSTANT within each length-W draft window:
        # req[t] = (t // W) % E. Deterministic, so it is mutually consistent
        # with `window_pos:<W>` on the same W (each window maps to one request).
        n = shape[-1] if shape else 1
        W = max(int(eval_numeric(str(args["W"]), knobs)), 1)
        E = max(int(eval_numeric(str(args["E"]), knobs)), 1)
        t = torch.arange(n, dtype=torch.int64)
        return ((t // W) % E).reshape(shape)
    elif kind == "gumbel_temps":
        # Per-request sampling temperature for the gumbel-max sampler: a float
        # vector mixing greedy (temp == 0.0, the argmax-only branch) and noisy
        # (temp > 0, the gumbel-noise + temperature-scaling branch) requests so
        # BOTH kernel paths are exercised. ~1/3 of entries are forced to 0.0;
        # the rest get positive temps in [0.3, 1.7) from the seeded gen. To make
        # the temperature ALWAYS discriminating (a kernel ignoring temp would
        # take the wrong branch / skip the scale), entry 0 is pinned greedy
        # (0.0) and entry 1 is pinned noisy whenever n >= 2 -- so every shape
        # has >=1 greedy and >=1 noisy request.
        n = shape[-1] if shape else 1
        base = torch.rand(n, generator=gen, dtype=torch.float32)
        t = 0.3 + 1.4 * base  # positive temps in [0.3, 1.7)
        greedy = torch.rand(n, generator=gen) < 0.34
        t[greedy] = 0.0
        if n >= 1:
            t[0] = 0.0  # pin greedy -> guaranteed argmax-branch coverage
        if n >= 2:
            t[1] = 0.3 + 1.4 * float(torch.rand(1, generator=gen).item())  # pin noisy
        return t.reshape(shape)
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
# `scalar` / `derive` values must be one of these — the  sidecar IPC
# carries no pickle (`inputs.py` `split_inputs`, `process_isolation.py`
# `_merge`), so a non-JSON scalar (e.g. a list under `scalar`) cannot cross
# the channel. WA.3 ② makes that boundary a hard error at materialisation.
_JSON_SCALAR_TYPES = (bool, int, float, str, type(None))


def _materialise_scalar(name: str, spec: dict[str, Any],
                        knobs: dict[str, int]) -> Any:
    """Materialise one non-tensor input (``scalar`` / ``derive`` / ``scalar_seq``).

    WA.3 ① / ②: the value of a ``scalar`` (a literal) or a ``derive`` rule
    (a knob expression evaluated through :func:`expr.eval_numeric`) must be a
    JSON scalar — the  sidecar IPC type boundary. A ``scalar_seq`` is a
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


def _materialise_struct(
    name: str,
    spec: dict[str, Any],
    knobs: dict[str, int],
    gen: torch.Generator,
    siblings: dict[str, Any] | None = None,
) -> Any:
    """Materialise a dataclass-shaped input from
    ``{struct: '<dotted.path.Class>', fields: {<fname>: <input_spec>}}``.

    Each ``fields[fname]`` entry uses the same spec grammar as a top-level
    inputs entry: ``scalar`` / ``derive`` / ``scalar_seq`` / ``scalar_knob``
    for non-tensor fields, a ``{dims, dtype, dist}`` mapping for tensor
    fields, a nested ``{struct, fields}`` for a sub-dataclass, or a
    ``{ref: '<sibling_input_name>'}`` to share an already-materialised
    tensor with another top-level input (so the oracle / candidate /
    other consumers see THE SAME tensor object, not a re-randomised
    twin). ``null`` binds the field to ``None`` (Optional fields).

    The struct class is imported via ``importlib.import_module`` -- the
    caller subprocess MUST have the upstream package on ``sys.path``
    (the oracle / S2 runners already do this). Instantiated as
    ``cls(**materialised_fields)``, so the dataclass must accept the
    declared field names as keyword arguments.

    Designed for upstream callables whose host wrapper takes a
    structured info object (e.g. ``LoRABatchInfo``,
    ``ForwardBatchInfo``) -- the bench's tensor-or-scalar input model
    cannot express those directly, and refusing to invoke them is the
    only honest path otherwise.
    """
    import importlib
    import os
    import sys

    klass_path = spec.get("struct")
    if not isinstance(klass_path, str) or "." not in klass_path:
        raise TypeError(
            f"input {name!r}: 'struct' must be a dotted import path "
            f"(e.g. 'sglang.srt.lora.utils.LoRABatchInfo')"
        )
    mod_path, _, cls_name = klass_path.rpartition(".")
    # Runtime-purity guard (R7): on the runtime path (`tab run`) a struct input
    # may import only ascendbench / stdlib / torch. ANY other top-level package
    # (vllm/sglang, or a third-party dep that imports upstream indirectly) is a
    # release-boundary violation -- an allowlist, so an arbitrary YAML-driven
    # import cannot smuggle upstream in. The provenance/oracle path opts in with
    # AB_ALLOW_UPSTREAM_STRUCT=1. Fail closed by default so a future task's
    # shapes.yaml cannot silently import upstream at run time.
    top_pkg = mod_path.split(".", 1)[0]
    runtime_allowed = {"ascendbench", "torch"} | sys.stdlib_module_names
    if top_pkg not in runtime_allowed and os.environ.get("AB_ALLOW_UPSTREAM_STRUCT") != "1":
        raise ImportError(
            f"input {name!r}: struct class {klass_path!r} imports non-runtime "
            f"package {top_pkg!r} -- a release-boundary violation on the runtime "
            f"path. Set AB_ALLOW_UPSTREAM_STRUCT=1 on the provenance/oracle path "
            f"to allow it."
        )
    try:
        mod = importlib.import_module(mod_path)
        cls = getattr(mod, cls_name)
    except (ImportError, AttributeError) as exc:
        raise ImportError(
            f"input {name!r}: cannot import struct class {klass_path!r} "
            f"-- is the upstream tree on sys.path? ({exc})"
        ) from exc

    fields_spec = spec.get("fields", {})
    if not isinstance(fields_spec, dict):
        raise TypeError(
            f"input {name!r}: 'fields' must be a mapping, got "
            f"{type(fields_spec).__name__}"
        )

    siblings = siblings or {}
    kwargs: dict[str, Any] = {}
    for field_name, field_spec in fields_spec.items():
        if field_spec is None:
            kwargs[field_name] = None
            continue
        if not isinstance(field_spec, dict):
            raise TypeError(
                f"input {name!r}: field {field_name!r} spec must be a "
                f"dict or null, got {type(field_spec).__name__}"
            )
        if "ref" in field_spec:
            target = field_spec["ref"]
            if target not in siblings:
                raise ValueError(
                    f"input {name!r}: field {field_name!r} refs sibling "
                    f"{target!r} which has not been materialised yet -- "
                    f"declare it BEFORE this struct in shapes.yaml inputs"
                )
            kwargs[field_name] = siblings[target]
        elif "struct" in field_spec:
            kwargs[field_name] = _materialise_struct(
                f"{name}.{field_name}", field_spec, knobs, gen, siblings,
            )
        elif "scalar" in field_spec or "derive" in field_spec or "scalar_seq" in field_spec:
            kwargs[field_name] = _materialise_scalar(field_name, field_spec, knobs)
        elif "scalar_knob" in field_spec:
            kwargs[field_name] = int(eval_numeric(str(field_spec["scalar_knob"]), knobs))
        else:
            kwargs[field_name] = _make_tensor(field_spec, knobs, gen)

    return cls(**kwargs)


def build_inputs(
    shapes_doc: dict[str, Any], shape_id: str, seed: int = 0,
) -> dict[str, Any]:
    """Construct the input dict for a given shape row.

    Non-tensor inputs are materialised per :func:`_materialise_scalar`:
    a ``scalar`` literal is passed through verbatim, a ``derive`` rule is
    evaluated against the shape row's knobs, and a ``scalar_seq`` carries a
    list literal. ``scalar_knob`` is the legacy derived-integer form.
    A ``struct`` spec materialises a dataclass-shaped input via
    :func:`_materialise_struct` (required for upstream callables that
    take a structured info object such as ``LoRABatchInfo``). Tensor
    specs are materialised from size-knob formulas with a fixed seed
    so oracle and candidate see identical inputs.
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
        if "struct" in spec:
            # Pass already-materialised siblings so struct fields can
            # `{ref: <name>}` to share tensors with flat inputs the
            # oracle / candidate consume directly. Order in
            # shapes.yaml inputs: matters -- the referenced sibling
            # must appear BEFORE the struct.
            out[name] = _materialise_struct(name, spec, knobs, gen, out)
        elif "scalar" in spec or "derive" in spec or "scalar_seq" in spec:
            out[name] = _materialise_scalar(name, spec, knobs)
        elif "scalar_knob" in spec:
            out[name] = int(eval_numeric(str(spec["scalar_knob"]), knobs))
        else:
            # Pass the partial `out` so a coupled dist (randn_inf_excl) can
            # reference an EARLIER-declared sibling tensor. Ordering in
            # shapes.yaml inputs: matters -- the ref must appear first.
            out[name] = _make_tensor(spec, knobs, gen, out)
    return out


def shape_ids(shapes_doc: dict[str, Any]) -> tuple[list[str], list[str]]:
    """(public_ids, hidden_ids), for either the knob or explicit format.

    Explicit docs route through the adapter; KNOB docs keep the exact old ``r["id"]`` enumeration, which is
    FAIL-LOUD on an id-less row (a malformed task raises rather than silently grading a reduced shape set --
    the adapter's defensive ``.get("id")`` skip would hide that, so it is reserved for the explicit path).
    Lazy-imported to avoid an import cycle (shape_format pulls input_spec siblings).
    """
    from ascendbench.input_spec.shape_format import (
        hidden_ids,
        is_explicit_shapes_doc,
        public_ids,
    )
    if is_explicit_shapes_doc(shapes_doc):
        return list(public_ids(shapes_doc)), list(hidden_ids(shapes_doc))
    pub = [r["id"] for r in shapes_doc.get("public_shapes", [])]
    hid = [r["id"] for r in shapes_doc.get("hidden_shapes", [])]
    return pub, hid


# --- : inputs materialisation for the process-isolation IPC ----------
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
