"""param_constraints.py — the shared ``param_constraints`` validation helper.

WA.4 ②(a): ``scripts/corpus/validate_tasks.py``, ``ascendbench/engine/eval.py`` and
the ``cli.py`` repo-authoritative status resolver all consult **one** helper
so the pass/fail verdict for an operator's input contract
(`docs/reference/task_api_contract.md §5.2`) cannot drift between the three.

:func:`check_param_constraints` is the single entrypoint. It returns a
:class:`ParamConstraintsResult` — a clean ``ok`` / not-``ok`` verdict plus a
reason list. Its decision order is fixed:

1. **scaffold short-circuit** — when ``scaffold == true`` the block is an
   un-reconciled draft. The helper returns a single ``scaffold`` failure
   *immediately* and does **not** run any schema or semantic check: a
   scaffold task's ``shapes.yaml`` is a placeholder (no ``size_knobs`` /
   ``inputs``), so materialising it would raise unrelated noise and pollute
   the "this is just a scaffold" signal (WA.5 non-regression).
2. **schema-shape check** — the six subsections exist with the right
   container types (the WA.1 layer, shared verbatim).
3. **five-subsection semantic check** — ``dims`` predicates evaluate against
   every knob row, ``dtypes`` / ``dists`` agree with ``shapes.yaml.inputs``,
   ``scalars`` value-domain predicates hold, and ``tensor_invariants`` hold on
   the materialised sample tensors.

A missing block is also a failure (a real task must carry the contract); the
``is_canary`` exemption is applied by the *callers*, not here — the helper has
no task identity, only the raw block + the ``shapes.yaml`` document.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from ascendbench.input_spec.expr import (
    ExprError,
    check_expr_syntax,
    eval_numeric,
    eval_predicate,
    exprs_equivalent,
)
from ascendbench.input_spec.inputs import (
    DIST_KINDS,
    DTYPE_ALIASES,
    DistSpecError,
    parse_dist_spec,
)

logger = logging.getLogger(__name__)


# The five semantic `param_constraints` subsections that carry the operator
# input contract (`docs/reference/task_api_contract.md §5.2`).
_SEMANTIC_SUBSECTIONS = ("dims", "dtypes", "dists", "scalars",
                         "tensor_invariants")

# The six `param_constraints` subsections (`docs/reference/task_api_contract.md §5.2`).
# `scaffold` is the sentinel; the other five carry the semantic contract.
_SUBSECTIONS = (*_SEMANTIC_SUBSECTIONS, "scaffold")

# Closed `tensor_invariants[*].type` enum.
_TENSOR_INVARIANT_TYPES = frozenset({
    "monotonic_nondecreasing", "first_eq", "last_eq", "length_eq",
    "bounded_above_by", "index_in_range",
})


@dataclass
class ParamConstraintsResult:
    """The verdict of :func:`check_param_constraints`.

    Attributes:
        ok: ``True`` iff the block is present, not a scaffold, schema-valid,
            and every semantic check passed.
        scaffold: ``True`` iff the block carried ``scaffold: true`` — the
            helper short-circuited and ran no schema / semantic check. A
            scaffold result is always ``ok == False``.
        missing: ``True`` iff the block was absent (``None``).
        reasons: human-readable failure strings; empty when ``ok``.
    """

    ok: bool
    scaffold: bool = False
    missing: bool = False
    reasons: list[str] = field(default_factory=list)


# Task status verdicts (mirror `scoring.TASK_STATUS_*` — kept as plain
# strings here so this module imports nothing from `scoring`).
TASK_STATUS_EVALUATED = "evaluated"
TASK_STATUS_UNSUPPORTED = "unsupported"
TASK_STATUS_BLOCKED = "blocked"


def resolve_task_status(
    *,
    is_canary: bool,
    self_reported_unsupported: bool,
    verified_golden_incomplete: bool,
    param_constraints_failed: bool,
) -> str:
    """The repo-authoritative ``task_status`` for one task — WA.4 ②(b).

    This is the SINGLE fixed-priority arbiter shared by ``cli.py``'s
    ``_authoritative_task_status`` and ``eval.evaluate``. Both must feed it
    the same trusted-repo-derived booleans so the verdict cannot drift. The
    priority is fixed (high → low, first match wins):

    1. ``is_canary`` ⇒ ``evaluated``. A canary is exempt from the
       ``param_constraints`` gate (it reuses its source task's solution and
       is never score-eligible).
    2. ``self_reported_unsupported`` ⇒ ``unsupported``. A self-reported-mode
       ``upstream_oracle_required`` task has no provider golden — it is
       deliberately out of that mode's denominator . The
       ``param_constraints`` gate must **not** pull it back in as ``blocked``.
    3. ``verified_golden_incomplete`` ⇒ ``blocked``. A verified-mode task
       whose offline ``golden/`` set is missing / incomplete keeps the
       existing verified-blocked verdict (priority over the
       ``param_constraints`` gate).
    4. ``param_constraints_failed`` ⇒ ``blocked``. A task whose
       ``param_constraints`` is missing / ``scaffold: true`` / schema-illegal
       / semantically violated is ``blocked`` — it stays in the fixed
       score-eligible denominator and is scored as ε (NOT dropped from the
       set — dropping it would be a subset-escape).
    5. otherwise ⇒ ``evaluated``.

    Args:
        is_canary: the task's ``meta.is_canary`` is ``True``.
        self_reported_unsupported: self-reported mode AND the task is
            ``upstream_oracle_required`` (no provider golden).
        verified_golden_incomplete: verified mode AND the task's offline
            ``golden/`` artifact set is missing / incomplete.
        param_constraints_failed: the shared ``param_constraints`` helper
            returned not-``ok`` for this task.

    Returns:
        One of ``"evaluated"`` / ``"unsupported"`` / ``"blocked"``.
    """
    if is_canary:
        return TASK_STATUS_EVALUATED
    if self_reported_unsupported:
        return TASK_STATUS_UNSUPPORTED
    if verified_golden_incomplete:
        return TASK_STATUS_BLOCKED
    if param_constraints_failed:
        return TASK_STATUS_BLOCKED
    return TASK_STATUS_EVALUATED


def check_param_constraints(
    raw: dict[str, Any] | None,
    shapes_doc: dict[str, Any] | None = None,
    *,
    tid: str = "task",
) -> ParamConstraintsResult:
    """Validate a task's ``param_constraints`` block — the shared WA.4 helper.

    The single pass/fail entrypoint shared by ``validate_tasks`` /
    ``eval.evaluate`` / the ``cli.py`` status resolver. The decision order is
    fixed (scaffold short-circuit → schema shape → five-subsection
    semantics); see the module docstring.

    Args:
        raw: the raw ``meta.yaml.param_constraints`` mapping (``TaskDef.
            param_constraints``). ``None`` ⇒ the block is absent.
        shapes_doc: the parsed ``shapes.yaml`` document. Required for the
            semantic checks (``dims`` / ``dtypes`` / ``dists`` / ``scalars`` /
            ``tensor_invariants``). For a ``scaffold: true`` block ``None``
            is fine — the scaffold short-circuit returns before any
            ``shapes_doc`` is needed. For a **non-scaffold** block ``None``
            (a missing / unreadable / unparseable ``shapes.yaml``) is
            **fail-closed**: the verdict is ``ok == False`` so the task
            resolves to ``blocked`` rather than slipping through as
            ``evaluated``.
        tid: task id, used only to prefix failure messages.

    Returns:
        A :class:`ParamConstraintsResult`. ``ok`` is ``True`` only when the
        block is present, non-scaffold, schema-valid and semantically clean.
    """
    if raw is None:
        return ParamConstraintsResult(
            ok=False, missing=True,
            reasons=[f"{tid}: param_constraints block is missing"])

    if not isinstance(raw, dict):
        return ParamConstraintsResult(
            ok=False,
            reasons=[f"{tid}: param_constraints must be a mapping, "
                     f"got {type(raw).__name__}"])

    # Decision order step 1 — scaffold short-circuit. An un-reconciled draft
    # is a single clean `scaffold` failure; no schema / semantic check runs.
    if raw.get("scaffold") is True:
        return ParamConstraintsResult(
            ok=False, scaffold=True,
            reasons=[f"{tid}: param_constraints.scaffold is true "
                     f"(un-reconciled draft — WB.1 must reconcile it)"])

    # Decision order step 2 — schema-shape check (the WA.1 layer).
    schema_errs = validate_schema_shape(raw, tid=tid)
    if schema_errs:
        return ParamConstraintsResult(ok=False, reasons=schema_errs)

    # Decision order step 3 — five-subsection semantic check. A malformed EXPLICIT shapes doc makes the
    # dual-format adapter raise ShapeFormatError; this entrypoint must return a clean fail-closed verdict
    # (callers expect a result, never an exception), so the task resolves to `blocked`, not a crash.
    from ascendbench.input_spec.shape_format import ShapeFormatError
    try:
        semantic_errs = _validate_semantics(raw, shapes_doc, tid=tid)
    except ShapeFormatError as e:
        return ParamConstraintsResult(
            ok=False, reasons=[f"{tid}: shapes.yaml is a malformed explicit-format doc: {e}"])
    if semantic_errs:
        return ParamConstraintsResult(ok=False, reasons=semantic_errs)

    return ParamConstraintsResult(ok=True)


# --------------------------------------------------------------------------
# Schema-shape layer (WA.1) — the six subsections + container types.
# --------------------------------------------------------------------------

def validate_schema_shape(raw: dict[str, Any], *, tid: str = "task") -> list[str]:
    """Validate only the **shape** of a ``param_constraints`` block (WA.1).

    Confirms the block is a mapping, every key is one of the six known
    subsections, and each present subsection has the right container type and
    coarse per-element structure. No expression is evaluated and no tensor is
    materialised — that is :func:`_validate_semantics`.

    The ``scaffold`` key is **mandatory** on every block — scaffold and
    non-scaffold alike . It is one of the six required subsections and
    is the single sentinel the generator / validator / eval gate (WA.4 /
    WA.5) all key off; a block with no ``scaffold`` key would otherwise be
    silently treated as non-scaffold and never flagged. A missing ``scaffold``
    key or a non-boolean value is therefore a schema-shape error, exactly like
    any other malformed subsection.

    For a **non-scaffold** block (``scaffold`` is ``false``) the corpus must
    additionally be *schema-closed*: all five semantic subsections (``dims`` /
    ``dtypes`` / ``dists`` / ``scalars`` / ``tensor_invariants``) must be
    present . An empty container (``[]`` / ``{}``) is allowed — it is an
    *explicit* declaration that the operator carries no constraint of that
    kind — but the key itself may not be silently absent, otherwise a
    half-reconciled block ``slips`` through validation. A ``scaffold: true``
    block never reaches here via :func:`check_param_constraints` (it
    short-circuits first); the standalone ``validate_tasks`` caller still runs
    this function on scaffold blocks, so the five-semantic-subsection rule is
    gated on ``scaffold`` not being ``true`` while the ``scaffold``-key
    requirement is unconditional.

    Args:
        raw: the ``param_constraints`` mapping (already confirmed a ``dict``).
        tid: task id for message prefixing.

    Returns:
        A list of schema-shape error strings; empty when shape-valid.
    """
    errs: list[str] = []

    unknown = set(raw.keys()) - set(_SUBSECTIONS)
    if unknown:
        errs.append(f"{tid}: param_constraints has unknown subsections "
                    f"{sorted(unknown)} (allowed: {sorted(_SUBSECTIONS)})")

    #  — a non-scaffold block must declare all five semantic subsections
    # (an empty container is fine; a missing key is not — it would let a
    # half-reconciled block pass undetected).
    if raw.get("scaffold") is not True:
        missing = [s for s in _SEMANTIC_SUBSECTIONS if s not in raw]
        if missing:
            errs.append(f"{tid}: param_constraints (scaffold: false) is "
                        f"missing required subsection(s) {sorted(missing)} — "
                        f"a reconciled block must declare all five of "
                        f"{list(_SEMANTIC_SUBSECTIONS)} (use an empty "
                        f"container to declare 'no constraint of this kind')")

    dims = raw.get("dims")
    if dims is not None:
        if not isinstance(dims, list):
            errs.append(f"{tid}: param_constraints.dims must be a list, "
                        f"got {type(dims).__name__}")
        elif not all(isinstance(d, str) for d in dims):
            errs.append(f"{tid}: param_constraints.dims must be a list of "
                        f"predicate expression strings")

    dtypes = raw.get("dtypes")
    if dtypes is not None:
        if not isinstance(dtypes, dict):
            errs.append(f"{tid}: param_constraints.dtypes must be a mapping, "
                        f"got {type(dtypes).__name__}")
        else:
            for name, dt in dtypes.items():
                if not isinstance(dt, str):
                    errs.append(f"{tid}: param_constraints.dtypes[{name!r}] "
                                f"must be a dtype string")
                elif dt not in DTYPE_ALIASES:
                    errs.append(f"{tid}: param_constraints.dtypes[{name!r}] "
                                f"dtype {dt!r} not in {sorted(DTYPE_ALIASES)}")

    dists = raw.get("dists")
    if dists is not None:
        if not isinstance(dists, dict):
            errs.append(f"{tid}: param_constraints.dists must be a mapping, "
                        f"got {type(dists).__name__}")
        else:
            for name, spec in dists.items():
                if not isinstance(spec, dict):
                    errs.append(f"{tid}: param_constraints.dists[{name!r}] "
                                f"must be a {{kind, args}} mapping")
                    continue
                kind = spec.get("kind")
                if not isinstance(kind, str):
                    errs.append(f"{tid}: param_constraints.dists[{name!r}] "
                                f"missing string 'kind'")
                elif kind not in DIST_KINDS:
                    errs.append(f"{tid}: param_constraints.dists[{name!r}] "
                                f"kind {kind!r} not in {sorted(DIST_KINDS)}")
                args = spec.get("args")
                if args is not None and not isinstance(args, dict):
                    errs.append(f"{tid}: param_constraints.dists[{name!r}]."
                                f"args must be a mapping when present")
                extra = set(spec.keys()) - {"kind", "args"}
                if extra:
                    errs.append(f"{tid}: param_constraints.dists[{name!r}] "
                                f"has unknown keys {sorted(extra)}")

    scalars = raw.get("scalars")
    if scalars is not None:
        if not isinstance(scalars, dict):
            errs.append(f"{tid}: param_constraints.scalars must be a mapping, "
                        f"got {type(scalars).__name__}")
        else:
            for name, rule in scalars.items():
                if isinstance(rule, str):
                    continue
                if isinstance(rule, dict):
                    if "derive" not in rule:
                        errs.append(f"{tid}: param_constraints.scalars["
                                    f"{name!r}] mapping form must carry a "
                                    f"'derive' key")
                    continue
                errs.append(f"{tid}: param_constraints.scalars[{name!r}] "
                            f"must be a predicate string or a derive mapping")

    invariants = raw.get("tensor_invariants")
    if invariants is not None:
        if not isinstance(invariants, list):
            errs.append(f"{tid}: param_constraints.tensor_invariants must be "
                        f"a list, got {type(invariants).__name__}")
        else:
            for i, inv in enumerate(invariants):
                if not isinstance(inv, dict):
                    errs.append(f"{tid}: param_constraints.tensor_invariants"
                                f"[{i}] must be a {{type, args}} mapping")
                    continue
                inv_type = inv.get("type")
                if not isinstance(inv_type, str):
                    errs.append(f"{tid}: param_constraints.tensor_invariants"
                                f"[{i}] missing string 'type'")
                elif inv_type not in _TENSOR_INVARIANT_TYPES:
                    errs.append(f"{tid}: param_constraints.tensor_invariants"
                                f"[{i}] type {inv_type!r} not in "
                                f"{sorted(_TENSOR_INVARIANT_TYPES)}")
                inv_args = inv.get("args")
                if inv_args is not None and not isinstance(inv_args, dict):
                    errs.append(f"{tid}: param_constraints.tensor_invariants"
                                f"[{i}].args must be a mapping when present")
                extra = set(inv.keys()) - {"type", "args"}
                if extra:
                    errs.append(f"{tid}: param_constraints.tensor_invariants"
                                f"[{i}] has unknown keys {sorted(extra)}")

    #  — `scaffold` is one of the six required subsections and is the
    # single un-reconciled-draft sentinel. It must be present and boolean on
    # every block; a missing key would otherwise be silently treated as
    # non-scaffold and never reported.
    if "scaffold" not in raw:
        errs.append(f"{tid}: param_constraints is missing the required "
                    f"'scaffold' key (boolean — true marks an un-reconciled "
                    f"draft)")
    else:
        scaffold = raw["scaffold"]
        if not isinstance(scaffold, bool):
            errs.append(f"{tid}: param_constraints.scaffold must be a "
                        f"boolean, got {type(scaffold).__name__}")

    return errs


# --------------------------------------------------------------------------
# Semantic layer (WA.4 ①) — the five contract subsections.
# --------------------------------------------------------------------------

def _knob_rows(shapes_doc: dict[str, Any]) -> list[tuple[str, dict[str, int]]]:
    """Enumerate every (label, row-environment) pair a ``shapes.yaml`` declares (U8: dual-format).

    Knob-format: a row's effective env is ``size_knobs`` overlaid by the per-row ``knobs``; the
    ``size_knobs`` defaults alone are ALSO returned as a row so a ``dims`` predicate must hold at the
    document's default point too. Explicit-format: each row carries its own named ``env`` (the atomic
    factors + row-only params) and there is no defaults point, so only the authored rows are returned.
    """
    from ascendbench.input_spec.shape_format import (
        effective_env,
        is_explicit_shapes_doc,
        iter_rows,
    )
    if is_explicit_shapes_doc(shapes_doc):
        return [(sid, effective_env(shapes_doc, sid)) for _role, sid, _row in iter_rows(shapes_doc)]
    base = dict(shapes_doc.get("size_knobs", {}) or {})
    rows: list[tuple[str, dict[str, int]]] = [("size_knobs", dict(base))]
    for section in ("public_shapes", "hidden_shapes"):
        for row in shapes_doc.get(section, []) or []:
            if not isinstance(row, dict):
                continue
            knobs = dict(base)
            knobs.update(row.get("knobs", {}) or {})
            rows.append((str(row.get("id", "?")), knobs))
    return rows


def _validate_dims(
    dims: list[Any], shapes_doc: dict[str, Any], tid: str,
) -> list[str]:
    """Evaluate each ``dims`` predicate against every knob row (WA.4 ①)."""
    errs: list[str] = []
    rows = _knob_rows(shapes_doc)
    for expr in dims:
        for label, knobs in rows:
            try:
                holds = eval_predicate(str(expr), knobs)
            except ExprError as e:
                errs.append(f"{tid}: param_constraints.dims predicate "
                            f"{expr!r} is not evaluable: {e}")
                break  # one bad predicate ⇒ report once, skip its rows
            if not holds:
                errs.append(f"{tid}: param_constraints.dims predicate "
                            f"{expr!r} is false for shape row {label} "
                            f"(knobs={knobs})")
    return errs


def _input_specs(shapes_doc: dict[str, Any]) -> dict[str, Any]:
    """Return ``{name: knob-style input spec}`` for either format (U8).

    Knob-format: the top-level ``inputs`` mapping. Explicit-format: the adapter reconstructs the structural
    form per input (no resolved dist args -- the dist check revalidates those per row). So the dtype /
    dist-kind / derive-formula structural checks below are format-agnostic.
    """
    from ascendbench.input_spec.shape_format import input_specs
    return input_specs(shapes_doc)


# The keys that mark a `shapes.yaml.inputs.*` entry as a **non-tensor** input.
# A spec carrying any of these is a literal (`scalar` / `scalar_seq`), a
# per-row derived scalar (`derive`), or a derived integer (`scalar_knob`) —
# `build_inputs` (`ascendbench/input_spec/inputs.py`) dispatches on exactly these
# keys. A tensor input carries none of them (it has `dims` and friends). The
# `dtypes` / `dists` subsections only constrain real tensor inputs, so a
# non-tensor input declared there is a schema misuse .
_NON_TENSOR_INPUT_KEYS = ("scalar", "derive", "scalar_seq", "scalar_knob")


def _is_tensor_input(spec: dict[str, Any]) -> bool:
    """True iff a ``shapes.yaml.inputs.*`` spec materialises as a tensor.

    Mirrors :func:`ascendbench.input_spec.inputs.build_inputs` dispatch: a spec
    is a non-tensor input iff it carries one of ``scalar`` / ``derive`` /
    ``scalar_seq`` / ``scalar_knob``; otherwise it is a tensor input.

    Args:
        spec: a ``shapes.yaml.inputs[name]`` mapping.

    Returns:
        ``True`` for a tensor input, ``False`` for a non-tensor (scalar) one.
    """
    return not any(k in spec for k in _NON_TENSOR_INPUT_KEYS)


def _validate_dtypes(
    dtypes: dict[str, Any], shapes_doc: dict[str, Any], tid: str,
) -> list[str]:
    """Confirm each declared ``dtypes`` entry matches ``shapes.yaml.inputs``.

    The ``dtypes`` subsection only constrains **tensor** inputs — a non-tensor
    input (a ``scalar`` / ``scalar_seq`` / ``derive`` / ``scalar_knob``) has no
    materialised dtype, so declaring one under ``dtypes`` is a schema misuse
    and is rejected rather than silently accepted via a default-dtype
    fallback .
    """
    errs: list[str] = []
    inputs = _input_specs(shapes_doc)
    for name, declared in dtypes.items():
        spec = inputs.get(name)
        if not isinstance(spec, dict):
            errs.append(f"{tid}: param_constraints.dtypes declares {name!r} "
                        f"but shapes.yaml.inputs has no such tensor input")
            continue
        if not _is_tensor_input(spec):
            errs.append(f"{tid}: param_constraints.dtypes declares {name!r} "
                        f"but shapes.yaml.inputs[{name!r}] is a non-tensor "
                        f"(scalar) input — only tensor inputs have a "
                        f"materialised dtype (constrain it under 'scalars')")
            continue
        actual = str(spec.get("dtype", "float32"))
        # Compare through the shared alias table: `bf16` and `bfloat16`
        # name the same torch dtype, so a constraint may use either spelling.
        if not _dtype_alias_eq(actual, str(declared)):
            errs.append(f"{tid}: param_constraints.dtypes[{name!r}] declares "
                        f"{declared!r} but shapes.yaml materialises "
                        f"dtype {actual!r}")
    return errs


def _dtype_alias_eq(a: str, b: str) -> bool:
    """True iff two dtype alias strings name the same torch dtype."""
    from ascendbench.input_spec.inputs import _DTYPES

    return _DTYPES.get(a) is not None and _DTYPES.get(a) == _DTYPES.get(b)


def _validate_dists(
    dists: dict[str, Any], shapes_doc: dict[str, Any], tid: str,
) -> list[str]:
    """Confirm each declared ``dists`` entry matches ``shapes.yaml.inputs``.

    Both the declared ``{kind, args}`` form and the ``shapes.yaml`` ``dist``
    string DSL are normalised through the shared :func:`parse_dist_spec`, so
    a wrong-but-supported distribution (``randn`` where ``softmax_rows`` is
    required) is statically rejected.

    Like ``dtypes`` (:func:`_validate_dtypes`), the ``dists`` subsection only
    constrains **tensor** inputs — a non-tensor input has no materialised
    distribution, so declaring one under ``dists`` is a schema misuse and is
    rejected rather than silently passing through a missing-``dist``
    fallback to ``randn`` .
    """
    from ascendbench.input_spec.shape_format import is_explicit_shapes_doc
    if is_explicit_shapes_doc(shapes_doc):
        return _validate_dists_explicit(dists, shapes_doc, tid)
    errs: list[str] = []
    inputs = _input_specs(shapes_doc)
    for name, declared in dists.items():
        spec = inputs.get(name)
        if not isinstance(spec, dict):
            errs.append(f"{tid}: param_constraints.dists declares {name!r} "
                        f"but shapes.yaml.inputs has no such tensor input")
            continue
        if not _is_tensor_input(spec):
            errs.append(f"{tid}: param_constraints.dists declares {name!r} "
                        f"but shapes.yaml.inputs[{name!r}] is a non-tensor "
                        f"(scalar) input — only tensor inputs have a "
                        f"materialised distribution (constrain it under "
                        f"'scalars')")
            continue
        try:
            want = parse_dist_spec(declared)
        except DistSpecError as e:
            errs.append(f"{tid}: param_constraints.dists[{name!r}] is not a "
                        f"valid dist spec: {e}")
            continue
        try:
            got = parse_dist_spec(spec.get("dist", "randn"))
        except DistSpecError as e:
            errs.append(f"{tid}: shapes.yaml.inputs[{name!r}].dist is not a "
                        f"valid dist spec: {e}")
            continue
        if want != got:
            errs.append(f"{tid}: param_constraints.dists[{name!r}] requires "
                        f"{want} but shapes.yaml materialises {got}")
    return errs


def _validate_dists_explicit(
    dists: dict[str, Any], shapes_doc: dict[str, Any], tid: str,
) -> list[str]:
    """Explicit-format dists check: the constraint's SYMBOLIC dist, resolved per row's env, must equal that
    row's RESOLVED dist (kind + args). The explicit row stores literal dist args, so a symbolic-vs-symbolic
    compare is impossible -- instead the constraint's args are resolved to each row's env (reusing the same
    resolver, so ``randn_inf_excl.ref`` stays a tensor name) and matched against the row's literal args."""
    from ascendbench.input_spec.explicit_build import resolve_dist_args
    from ascendbench.input_spec.shape_format import (
        effective_env,
        explicit_row_dist,
        hidden_ids,
        public_ids,
    )
    errs: list[str] = []
    specs = _input_specs(shapes_doc)
    sids = public_ids(shapes_doc) + hidden_ids(shapes_doc)
    for name, declared in dists.items():
        spec = specs.get(name)
        if not isinstance(spec, dict) or not _is_tensor_input(spec):
            errs.append(f"{tid}: param_constraints.dists declares {name!r} but it is not a tensor input "
                        f"in the explicit rows (only tensor inputs have a materialised distribution)")
            continue
        try:
            want = parse_dist_spec(declared)
        except DistSpecError as e:
            errs.append(f"{tid}: param_constraints.dists[{name!r}] is not a valid dist spec: {e}")
            continue
        for sid in sids:
            got = explicit_row_dist(shapes_doc, sid, name)
            try:
                want_r = resolve_dist_args(want, effective_env(shapes_doc, sid))
            except (ValueError, ExprError) as e:
                errs.append(f"{tid}: param_constraints.dists[{name!r}] not resolvable at row {sid}: {e}")
                break
            if want_r != got:
                errs.append(f"{tid}: param_constraints.dists[{name!r}] requires {want_r} but row {sid} "
                            f"materialises {got}")
                break   # one mismatch per name is enough
    return errs


def _validate_scalars(
    scalars: dict[str, Any], shapes_doc: dict[str, Any], tid: str,
) -> list[str]:
    """Check each ``scalars`` value-domain predicate (WA.4 ①).

    A predicate-string rule is evaluated against a symbol table that merges
    the shape row's knobs with the materialised ``scalar`` / ``scalar_seq``
    value, so a constraint may reference knobs alongside the scalar itself —
    e.g. ``L1_31``'s ``"sum(mrope_section) == D // 2"`` . This mirrors the
    knob-keyed ``dims`` predicate evaluation. A ``{derive: ...}`` rule is
    reconciled against ``shapes.yaml.inputs[name]`` by
    :func:`_validate_derive_rule` — the formula must be a legal numeric
    expression, the ``shapes.yaml`` input must itself be a ``derive`` form,
    and the two formulas must be structurally equivalent .
    """
    errs: list[str] = []
    inputs = _input_specs(shapes_doc)
    rows = _knob_rows(shapes_doc)
    for name, rule in scalars.items():
        spec = inputs.get(name)
        if not isinstance(spec, dict):
            errs.append(f"{tid}: param_constraints.scalars declares {name!r} "
                        f"but shapes.yaml.inputs has no such input")
            continue
        if isinstance(rule, dict):
            errs.extend(_validate_derive_rule(name, rule, spec, tid))
            continue
        # predicate-string rule — materialise the value for every knob row
        # and require the predicate to hold each time.
        for label, knobs in rows:
            try:
                value = _materialise_scalar_value(name, spec, knobs)
            except Exception as e:  # report, do not crash
                errs.append(f"{tid}: param_constraints.scalars[{name!r}] "
                            f"input could not be materialised for row "
                            f"{label}: {e}")
                break
            # Merge the row's knobs with the scalar's own name so a
            # constraint can reference knobs (e.g. `sum(mrope_section) ==
            # D // 2`). The scalar name wins on a knob/scalar name clash.
            symbols = {**knobs, name: value}
            try:
                holds = eval_predicate(str(rule), symbols)
            except ExprError as e:
                errs.append(f"{tid}: param_constraints.scalars[{name!r}] "
                            f"predicate {rule!r} is not evaluable: {e}")
                break
            if not holds:
                errs.append(f"{tid}: param_constraints.scalars[{name!r}] "
                            f"predicate {rule!r} is false for row {label} "
                            f"(value={value!r})")
    return errs


def _validate_derive_rule(
    name: str, rule: dict[str, Any], spec: dict[str, Any], tid: str,
) -> list[str]:
    """Reconcile a ``scalars.{name}.derive`` rule with ``shapes.yaml`` .

    A ``{derive: ...}`` rule in ``param_constraints.scalars`` declares that
    the scalar input is *derived* from the knobs. The validator must confirm
    ``shapes.yaml.inputs[name]`` agrees:

    1. the derive formula is a legal numeric expression;
    2. the ``shapes.yaml`` input is itself a ``derive`` form — a task may not
       declare ``scale: {derive: "1/sqrt(D)"}`` while ``shapes.yaml``
       materialises ``scale`` as a fixed ``{scalar: 0.125}``;
    3. the two derive formulas are structurally equivalent (`exprs_equivalent`
       — whitespace-insensitive, AST-level).

    Args:
        name: the scalar input name.
        rule: the ``param_constraints.scalars[name]`` mapping (carries
            ``derive``).
        spec: the ``shapes.yaml.inputs[name]`` mapping.
        tid: task id for message prefixing.

    Returns:
        A list of reconciliation error strings; empty when consistent.
    """
    errs: list[str] = []
    pc_formula = rule.get("derive")
    if pc_formula is None:
        errs.append(f"{tid}: param_constraints.scalars[{name!r}] derive "
                    f"rule has no 'derive' formula")
        return errs

    try:
        check_expr_syntax(str(pc_formula), "numeric")
    except ExprError as e:
        errs.append(f"{tid}: param_constraints.scalars[{name!r}] derive "
                    f"expression is invalid: {e}")
        return errs

    if "derive" not in spec:
        materialised = next(
            (k for k in ("scalar", "scalar_seq", "scalar_knob")
             if k in spec),
            "a tensor",
        )
        errs.append(f"{tid}: param_constraints.scalars[{name!r}] declares a "
                    f"derive rule but shapes.yaml.inputs[{name!r}] "
                    f"materialises {name!r} as {materialised!r}, not a "
                    f"'derive' form — the derived contract is not honoured")
        return errs

    shapes_formula = spec.get("derive")
    if not exprs_equivalent(str(pc_formula), str(shapes_formula)):
        errs.append(f"{tid}: param_constraints.scalars[{name!r}] derive "
                    f"formula {pc_formula!r} differs from "
                    f"shapes.yaml.inputs[{name!r}].derive "
                    f"{shapes_formula!r}")
    return errs


def _materialise_scalar_value(
    name: str, spec: dict[str, Any], knobs: dict[str, int],
) -> Any:
    """Materialise one non-tensor input's value for a knob row.

    Handles the three non-tensor forms (``scalar`` / ``derive`` /
    ``scalar_seq``) plus the legacy ``scalar_knob``.
    """
    from ascendbench.input_spec.inputs import _materialise_scalar

    if "scalar" in spec or "derive" in spec or "scalar_seq" in spec:
        return _materialise_scalar(name, spec, knobs)
    if "scalar_knob" in spec:
        return int(eval_numeric(str(spec["scalar_knob"]), knobs))
    raise ValueError(f"input {name!r} is not a non-tensor (scalar) input")


# Materialised-tensor invariant dispatch (WA.4 ①). Each checker receives the
# 1-D-or-N-D tensor and a `value` (already an int from the knob expression,
# or `None` when the invariant takes no value) and returns an error string or
# `None`.

def _inv_monotonic_nondecreasing(t, value, name):
    flat = t.reshape(-1)
    if flat.numel() >= 2 and bool((flat[1:] < flat[:-1]).any()):
        return f"{name!r} is not monotonic non-decreasing"
    return None


def _inv_first_eq(t, value, name):
    flat = t.reshape(-1)
    if flat.numel() == 0:
        return f"{name!r} is empty — first_eq cannot hold"
    if int(flat[0]) != int(value):
        return f"{name!r} first element {int(flat[0])} != {int(value)}"
    return None


def _inv_last_eq(t, value, name):
    flat = t.reshape(-1)
    if flat.numel() == 0:
        return f"{name!r} is empty — last_eq cannot hold"
    if int(flat[-1]) != int(value):
        return f"{name!r} last element {int(flat[-1])} != {int(value)}"
    return None


def _inv_length_eq(t, value, name):
    n = t.shape[-1] if t.dim() else t.numel()
    if int(n) != int(value):
        return f"len({name!r}) == {int(n)} != {int(value)}"
    return None


def _inv_bounded_above_by(t, value, name):
    if t.numel() and bool((t > int(value)).any()):
        return f"{name!r} has an element above the bound {int(value)}"
    return None


def _inv_index_in_range(t, value, name):
    if t.numel() and (bool((t < 0).any()) or bool((t >= int(value)).any())):
        return f"{name!r} has an element outside the index range "\
               f"[0, {int(value)})"
    return None


_INVARIANT_CHECKERS = {
    "monotonic_nondecreasing": (_inv_monotonic_nondecreasing, False),
    "first_eq": (_inv_first_eq, True),
    "last_eq": (_inv_last_eq, True),
    "length_eq": (_inv_length_eq, True),
    "bounded_above_by": (_inv_bounded_above_by, True),
    "index_in_range": (_inv_index_in_range, True),
}


def _validate_tensor_invariants(
    invariants: list[Any], shapes_doc: dict[str, Any], tid: str,
) -> list[str]:
    """Check every ``tensor_invariants`` entry on materialised sample tensors.

    The tensors are built once per representative shape row through
    :func:`ascendbench.input_spec.inputs.build_inputs`; each enumerated invariant
    type dispatches to its checker (`docs/reference/task_api_contract.md §5.2`).
    """
    from ascendbench.input_spec.shape_format import (
        build_inputs_for,
        effective_env,
        hidden_ids,
        public_ids,
    )
    errs: list[str] = []
    # Materialise once per real shape-id row (dual-format via the adapter -- build_inputs_for dispatches
    # knob vs explicit; the env for invariant value expressions comes from effective_env, not a knob merge).
    real_ids = list(public_ids(shapes_doc)) + list(hidden_ids(shapes_doc))
    if not real_ids:
        errs.append(f"{tid}: param_constraints.tensor_invariants need at "
                    f"least one public/hidden shape row to materialise")
        return errs

    for sid in real_ids:
        try:
            built = build_inputs_for(shapes_doc, sid, seed=0)
        except Exception as e:  # report, do not crash
            errs.append(f"{tid}: param_constraints.tensor_invariants — "
                        f"shape row {sid} did not materialise: {e}")
            continue
        knobs = effective_env(shapes_doc, sid)
        for i, inv in enumerate(invariants):
            inv_type = inv.get("type")
            args = inv.get("args", {}) or {}
            checker, needs_value = _INVARIANT_CHECKERS[inv_type]
            input_name = args.get("input")
            tensor = built.get(input_name)
            if not _is_checkable_tensor(tensor):
                errs.append(f"{tid}: tensor_invariants[{i}] ({inv_type}) "
                            f"input {input_name!r} is not a materialised "
                            f"tensor in shape row {sid}")
                continue
            value: int | None = None
            if needs_value:
                try:
                    value = int(eval_numeric(str(args.get("value")), knobs))
                except ExprError as e:
                    errs.append(f"{tid}: tensor_invariants[{i}] ({inv_type}) "
                                f"value expression invalid: {e}")
                    continue
            problem = checker(tensor, value, input_name)
            if problem is not None:
                errs.append(f"{tid}: tensor_invariants[{i}] ({inv_type}) "
                            f"violated in shape row {sid}: {problem}")
    return errs


def _is_checkable_tensor(t: Any) -> bool:
    """True iff ``t`` is a torch tensor the invariant checkers can read."""
    import torch

    return torch.is_tensor(t)


def _validate_semantics(
    raw: dict[str, Any], shapes_doc: dict[str, Any] | None, *, tid: str,
) -> list[str]:
    """Run the five-subsection semantic checks (WA.4 ①).

    Called only after the scaffold short-circuit and the schema-shape check
    have both passed — so ``raw`` here is always a **non-scaffold** block
    that carries the operator input contract and therefore *requires* the
    semantic checks to actually run.

    ``shapes_doc`` being unavailable (``None`` because the task's
    ``shapes.yaml`` is missing / unreadable / unparseable, or not a mapping)
    is **fail-closed**: the contract cannot be verified, so the block fails
    rather than silently passing. Skipping the checks would let a task with a
    corrupt ``shapes.yaml`` resolve to ``evaluated`` instead of ``blocked``,
    defeating the WA.4 fail-closed gate. The ``scaffold: true`` short-circuit
    in :func:`check_param_constraints` runs *before* this function, so a
    genuine scaffold draft (whose placeholder ``shapes.yaml`` is expected to
    be unmaterialisable) never reaches here.
    """
    if shapes_doc is None:
        return [f"{tid}: param_constraints (scaffold: false) cannot be "
                f"verified — shapes.yaml is missing, unreadable or "
                f"unparseable; failing closed (treated as blocked)"]
    if not isinstance(shapes_doc, dict):
        return [f"{tid}: shapes.yaml is not a mapping — cannot run "
                f"param_constraints semantic checks; failing closed "
                f"(treated as blocked)"]

    errs: list[str] = []
    dims = raw.get("dims")
    if dims:
        errs.extend(_validate_dims(dims, shapes_doc, tid))
    dtypes = raw.get("dtypes")
    if dtypes:
        errs.extend(_validate_dtypes(dtypes, shapes_doc, tid))
    dists = raw.get("dists")
    if dists:
        errs.extend(_validate_dists(dists, shapes_doc, tid))
    scalars = raw.get("scalars")
    if scalars:
        errs.extend(_validate_scalars(scalars, shapes_doc, tid))
    invariants = raw.get("tensor_invariants")
    if invariants:
        errs.extend(_validate_tensor_invariants(invariants, shapes_doc, tid))
    return errs
