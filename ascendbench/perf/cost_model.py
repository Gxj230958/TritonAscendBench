"""cost_model.py — `meta.yaml.cost_model` schema + shared formula evaluator.

The design spec §5.6 / `docs/reference/task_api_contract.md` §5: every task may declare a
performance cost model so scoring can build the Roofline theoretical-latency
baseline ``t_ref = max(max_c W_c/π_c, max_r M_r/β_r)`` (§9).

``meta.yaml.cost_model`` has two formula fields:

* ``flops`` — per-compute-class FLOPs ``{c: W_c(knobs)}``
* ``bytes`` — per-resource HBM/interconnect bytes ``{r: M_r(knobs)}``

Each of the two fields accepts **two shapes**:

* **scalar** — a single formula string, the single-class / single-resource
  degenerate form, e.g. ``flops: "2*B*L*D"``. A scalar ``flops`` **must** be
  accompanied by a sibling ``flops_class`` naming the compute class the
  formula belongs to (one of the ``peak_table_910b.toml`` compute classes);
  a scalar ``bytes`` **must** be accompanied by a sibling ``bytes_resource``
  naming the resource (``hbm`` / ``interconnect``).
* **map** — ``{class: formula}`` / ``{resource: formula}`` when the operator
  mixes compute classes or data resources, e.g.
  ``flops: {fp16_cube: "2*M*N*K", vector: "M*N"}``.

The validator and the dataset loader accept either shape and **normalise to a
map internally** — the scalar form becomes a one-entry map keyed by its
companion ``flops_class`` / ``bytes_resource`` (a *real* peak-table class /
resource, never a synthetic key) so the normalised map is directly scorable
against the peak table.

Formulas are size-knob expressions over the same parameterisation as
``shapes.yaml`` (``B`` / ``L`` / ``D`` / …) plus a fixed set of *derived
symbols* (currently ``dtype_size``; see ``task_api_contract.md`` §5). A single
shared :class:`FormulaEnv` defines exactly which symbols are legal and supplies
helper builtins; ``dataset.py`` / ``validate_tasks.py`` /  all reuse the
same evaluator so the symbol set can never drift between call sites.

**Safe evaluation** — formulas are *not* run through :func:`eval`. They are
parsed with :mod:`ast` and walked through an arithmetic whitelist (``+ - * /
** //``, unary ``+`` / ``-``, numeric literals, whitelisted names, whitelisted
calls). Anything else (attribute access, subscripts, comprehensions, unknown
names, …) raises :class:`CostModelError`.

The AST-whitelist safety core was extracted to :mod:`ascendbench.input_spec.expr`
by WA.2 so the numeric path and the ``param_constraints`` predicate path share
one evaluator. ``cost_model`` keeps the public surface below
(:class:`FormulaEnv` / :func:`evaluate_formula` / :func:`parse_cost_model` /
:class:`CostModelError`) unchanged and is layered on :func:`expr.eval_numeric`
— the  numeric contract (values, signatures, exception behaviour) is
preserved exactly.
"""
from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from ascendbench.input_spec.expr import ExprError, check_expr_syntax, eval_numeric

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------
# Derived symbols + helpers — the SHARED evaluation environment.
# --------------------------------------------------------------------------
#
# A cost-model formula is evaluated against `knobs` (size knobs from
# shapes.yaml: B/L/D/…) PLUS this fixed set of derived symbols. Keeping the
# derived-symbol set in one place is the whole point of : dataset.py,
# validate_tasks.py and the  GPU cross-check all import it from here, so a
# formula that validates is guaranteed to be evaluable everywhere.
#
# `dtype_size` is the element size (bytes) of the task's primary dtype; it lets
# a `bytes` formula stay dtype-agnostic, e.g. "(B*L*D + B*D) * dtype_size".

#: Derived (non-knob) symbols a formula may reference, with the default value
#: the evaluator supplies when a caller does not override it.
DERIVED_SYMBOL_DEFAULTS: dict[str, float] = {
    # fp16/bf16 element size; the validator only needs *a* finite value to
    # confirm the formula evaluates — real per-task dtype_size is supplied by
    # the  scoring path.
    "dtype_size": 2.0,
}

#: Compute classes a scalar-form ``flops_class`` may name — the
#: ``peak_table_910b.toml`` compute classes (kept here as the schema-level
#: allowed set so ``cost_model`` validation does not need the TOML file).
VALID_FLOPS_CLASSES: frozenset[str] = frozenset(
    {"fp16_cube", "bf16_cube", "int8", "vector"}
)
#: Resources a scalar-form ``bytes_resource`` may name.
VALID_BYTES_RESOURCES: frozenset[str] = frozenset({"hbm", "interconnect"})


class CostModelError(ExprError):
    """Raised on a malformed ``cost_model`` block or an unparseable /
    unevaluable cost-model formula.

    Subclasses :class:`ascendbench.input_spec.expr.ExprError` (WA.2): the
    AST-whitelist safety core raises ``ExprError``, and the cost-model layer
    re-raises it as ``CostModelError`` so the  exception contract — every
    cost-model failure is a ``CostModelError`` — is preserved unchanged.
    """


@dataclass(frozen=True)
class CostModel:
    """Normalised, parsed ``meta.yaml.cost_model``.

    Both ``flops`` and ``bytes`` are always maps here regardless of whether the
    task authored the scalar or the map form — the scalar form is normalised to
    a one-entry map keyed by its companion ``flops_class`` / ``bytes_resource``,
    which is a real :data:`VALID_FLOPS_CLASSES` / :data:`VALID_BYTES_RESOURCES`
    peak-table key. There is no synthetic key: the normalised map is always
    directly scorable against the peak table.
    """

    #: ``{compute_class: formula_string}`` — per-class FLOPs W_c.
    flops: Mapping[str, str]
    #: ``{resource: formula_string}`` — per-resource bytes M_r.
    bytes: Mapping[str, str]
    #: Analytic-vs-instrumented GPU-Triton mismatch tolerance (GPU cross-check).
    gpu_xcheck_tol: float = 0.05
    #: True when ``flops`` was authored in the scalar (degenerate) form.
    flops_scalar: bool = False
    #: True when ``bytes`` was authored in the scalar (degenerate) form.
    bytes_scalar: bool = False


# --------------------------------------------------------------------------
# Safe formula evaluation.
# --------------------------------------------------------------------------

class FormulaEnv:
    """Shared evaluation environment for cost-model formulas.

    An instance bundles the concrete symbol table (size ``knobs`` + derived
    symbols) a formula is evaluated against. The same class is what the
    validator uses (with derived-symbol defaults and dummy knob values) to
    confirm a formula merely *parses and evaluates*, and what scoring uses
    (with real knobs + real ``dtype_size``) to obtain a number.
    """

    def __init__(
        self,
        knobs: Mapping[str, Any] | None = None,
        *,
        derived: Mapping[str, Any] | None = None,
    ) -> None:
        """Build an environment.

        Args:
            knobs: size-knob values (``B`` / ``L`` / ``D`` / …). Non-numeric
                values are rejected.
            derived: overrides for the derived symbols (e.g. a real
                ``dtype_size``); unspecified derived symbols fall back to
                :data:`DERIVED_SYMBOL_DEFAULTS`.
        """
        env: dict[str, float] = dict(DERIVED_SYMBOL_DEFAULTS)
        if derived:
            for name, value in derived.items():
                if name not in DERIVED_SYMBOL_DEFAULTS:
                    raise CostModelError(
                        f"unknown derived symbol {name!r}; the shared "
                        f"environment defines only "
                        f"{sorted(DERIVED_SYMBOL_DEFAULTS)}"
                    )
                env[name] = _as_number(name, value)
        for name, value in (knobs or {}).items():
            # A knob must not shadow a derived symbol — that would make a
            # formula's meaning depend on which call site evaluated it.
            if name in DERIVED_SYMBOL_DEFAULTS:
                raise CostModelError(
                    f"size knob {name!r} collides with a reserved derived "
                    f"symbol"
                )
            env[name] = _as_number(name, value)
        self._env = env

    @property
    def symbols(self) -> dict[str, float]:
        """The concrete symbol table (knobs + derived) — read-only copy."""
        return dict(self._env)

    def evaluate(self, formula: str) -> float:
        """Evaluate one formula string, returning a float.

        Raises:
            CostModelError: on a syntax error, an unknown name, or any AST
                node outside the arithmetic whitelist.
        """
        return evaluate_formula(formula, self._env)


def _as_number(name: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CostModelError(
            f"symbol {name!r} must be a number, got {type(value).__name__}"
        )
    return float(value)


def evaluate_formula(formula: str, symbols: Mapping[str, Any]) -> float:
    """Safely evaluate a cost-model formula against ``symbols``.

    Thin wrapper over :func:`ascendbench.input_spec.expr.eval_numeric` — the
    shared AST-whitelist numeric evaluator (WA.2). No :func:`eval`, no
    attribute access, no subscripts, no arbitrary builtins. The wrapper exists
    only to re-raise the core's :class:`ExprError` as :class:`CostModelError`,
    keeping the  exception contract intact for every cost-model call site.

    Args:
        formula: the size-knob expression string.
        symbols: the symbol table (knob values + derived symbols).

    Returns:
        The formula value as a float.

    Raises:
        CostModelError: on a non-string formula, a syntax error, an unknown
            name, an unknown call, or any disallowed AST node.
    """
    try:
        return eval_numeric(formula, symbols)
    except CostModelError:
        raise
    except ExprError as e:
        raise CostModelError(str(e)) from e


# --------------------------------------------------------------------------
# `meta.yaml.cost_model` parsing + normalisation.
# --------------------------------------------------------------------------

def _normalise_field(
    value: Any, field_name: str, companion: Any,
) -> tuple[dict[str, str], bool]:
    """Normalise one ``flops`` / ``bytes`` field to ``(map, was_scalar)``.

    Accepts the scalar form (a formula string) or the map form
    (``{key: formula}``) and always returns a map.

    The scalar form **requires** its companion class/resource — ``flops`` →
    ``flops_class`` (a :data:`VALID_FLOPS_CLASSES` value), ``bytes`` →
    ``bytes_resource`` (a :data:`VALID_BYTES_RESOURCES` value). The scalar
    formula is keyed under that real peak-table class/resource, so the
    normalised map is directly scorable — there is no synthetic ``"default"``
    key. ``companion`` is the raw value of that sibling field (or ``None`` when
    the task did not author it).

    : the **map** form keys the class/resource itself, so every map key
    is held to the SAME legal set the scalar companion is — a ``flops`` map
    key must be a :data:`VALID_FLOPS_CLASSES` value, a ``bytes`` map key a
    :data:`VALID_BYTES_RESOURCES` value. A typo'd key (e.g. ``"vecotr"``) is
    rejected statically here, not silently carried until ``compute_t_ref``
    fails closed at scoring time.
    """
    if field_name == "flops":
        companion_key, allowed = "flops_class", VALID_FLOPS_CLASSES
    else:
        companion_key, allowed = "bytes_resource", VALID_BYTES_RESOURCES
    if isinstance(value, str):
        if not value.strip():
            raise CostModelError(f"cost_model.{field_name} formula is empty")
        if companion is None:
            raise CostModelError(
                f"cost_model.{field_name} is scalar form but the required "
                f"companion cost_model.{companion_key} is missing — a scalar "
                f"{field_name} formula must name its "
                f"{'compute class' if field_name == 'flops' else 'resource'}"
            )
        if not isinstance(companion, str) or companion not in allowed:
            raise CostModelError(
                f"cost_model.{companion_key} must be one of "
                f"{sorted(allowed)}, got {companion!r}"
            )
        return {companion: value}, True
    if isinstance(value, Mapping):
        if not value:
            raise CostModelError(
                f"cost_model.{field_name} map is empty; declare at least one "
                f"{'compute class' if field_name == 'flops' else 'resource'}"
            )
        out: dict[str, str] = {}
        for key, formula in value.items():
            if not isinstance(key, str) or not key:
                raise CostModelError(
                    f"cost_model.{field_name}: map key {key!r} must be a "
                    f"non-empty string"
                )
            # : a map key names a peak-table compute class / resource,
            # so it must be in the same legal set the scalar companion is.
            if key not in allowed:
                raise CostModelError(
                    f"cost_model.{field_name}: map key {key!r} is not a "
                    f"valid {'compute class' if field_name == 'flops' else 'resource'}; "
                    f"must be one of {sorted(allowed)}"
                )
            if not isinstance(formula, str) or not formula.strip():
                raise CostModelError(
                    f"cost_model.{field_name}.{key}: formula must be a "
                    f"non-empty string"
                )
            out[key] = formula
        return out, False
    raise CostModelError(
        f"cost_model.{field_name} must be a formula string (scalar form) or "
        f"a {{key: formula}} map, got {type(value).__name__}"
    )


def parse_cost_model(raw: Any) -> CostModel:
    """Parse + validate a raw ``meta.yaml.cost_model`` block.

    Accepts either formula shape (scalar / map) for ``flops`` and ``bytes``
    independently, normalises both to maps, and confirms every formula parses
    and evaluates under :class:`FormulaEnv` (with placeholder knob values).

    Args:
        raw: the value of ``meta.yaml['cost_model']``.

    Returns:
        A normalised :class:`CostModel`.

    Raises:
        CostModelError: if the block is malformed or any formula is invalid.
    """
    if not isinstance(raw, Mapping):
        raise CostModelError(
            f"cost_model must be a mapping, got {type(raw).__name__}"
        )
    missing = {"flops", "bytes"} - set(raw.keys())
    if missing:
        raise CostModelError(
            f"cost_model missing required key(s): {sorted(missing)}"
        )

    flops, flops_scalar = _normalise_field(
        raw["flops"], "flops", raw.get("flops_class"))
    bytes_, bytes_scalar = _normalise_field(
        raw["bytes"], "bytes", raw.get("bytes_resource"))

    # A companion class/resource is meaningful only for the scalar form; a
    # stray one on a map form would silently mislead, so reject it.
    if not flops_scalar and "flops_class" in raw:
        raise CostModelError(
            "cost_model.flops_class is only valid with the scalar flops "
            "form; the map form keys the class itself"
        )
    if not bytes_scalar and "bytes_resource" in raw:
        raise CostModelError(
            "cost_model.bytes_resource is only valid with the scalar bytes "
            "form; the map form keys the resource itself"
        )

    tol_raw = raw.get("gpu_xcheck_tol", 0.05)
    if isinstance(tol_raw, bool) or not isinstance(tol_raw, (int, float)):
        raise CostModelError(
            f"cost_model.gpu_xcheck_tol must be a number, got "
            f"{type(tol_raw).__name__}"
        )
    if not (0.0 < float(tol_raw) < 1.0):
        raise CostModelError(
            f"cost_model.gpu_xcheck_tol must be in (0, 1), got {tol_raw}"
        )

    model = CostModel(
        flops=flops,
        bytes=bytes_,
        gpu_xcheck_tol=float(tol_raw),
        flops_scalar=flops_scalar,
        bytes_scalar=bytes_scalar,
    )
    # Confirm every formula is parseable + evaluable so a malformed formula is
    # caught at load/validate time, not deep inside scoring. The knob set a
    # formula references is unknown here, so unknown *names* are not an error
    # at parse time — they are caught by `validate_formulas` against a known
    # knob universe. Here we only reject syntax errors / disallowed nodes.
    for field_name, fmap in (("flops", flops), ("bytes", bytes_)):
        for key, formula in fmap.items():
            _check_formula_syntax(formula, f"cost_model.{field_name}.{key}")
    return model


def _check_formula_syntax(formula: str, where: str) -> None:
    """Confirm a formula parses and uses only whitelisted AST nodes.

    Thin wrapper over :func:`ascendbench.input_spec.expr.check_expr_syntax` in
    ``numeric`` mode (WA.2). Unknown *names* are tolerated here (the knob set
    is task-specific); only syntax errors and disallowed node types are
    rejected. The wrapper prefixes the error with ``where`` and re-raises the
    core's :class:`ExprError` as :class:`CostModelError`.
    """
    if not isinstance(formula, str) or not formula.strip():
        raise CostModelError(f"{where}: formula must be a non-empty string")
    try:
        check_expr_syntax(formula, "numeric")
    except ExprError as e:
        raise CostModelError(f"{where}: {e}") from e


def validate_formulas(model: CostModel, knob_names: set[str]) -> None:
    """Confirm every formula evaluates given a known knob universe.

    Used by ``validate_tasks.py``: it builds a placeholder symbol table from
    ``knob_names`` (all set to a positive dummy value) plus the derived-symbol
    defaults, then evaluates each formula — catching unknown symbols a task
    typo'd that ``parse_cost_model`` could not catch without knowing the knobs.

    Args:
        model: the parsed cost model.
        knob_names: the size-knob names declared in the task's ``shapes.yaml``
            (``size_knobs`` keys plus every per-row ``knobs`` key).

    Raises:
        CostModelError: if a formula references an unknown symbol or fails to
            evaluate.
    """
    # Dummy positive knob values — 2.0 keeps `a/b`, `a**b`, `a//b` finite and
    # non-zero so a structurally valid formula always evaluates.
    dummy_knobs = dict.fromkeys(knob_names, 2.0)
    env = FormulaEnv(dummy_knobs)
    for field_name, fmap in (
        ("flops", model.flops),
        ("bytes", model.bytes),
    ):
        for key, formula in fmap.items():
            try:
                env.evaluate(formula)
            except CostModelError as e:
                raise CostModelError(
                    f"cost_model.{field_name}.{key}: {e}"
                ) from e
