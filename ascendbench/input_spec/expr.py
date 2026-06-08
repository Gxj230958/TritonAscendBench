"""expr.py — the shared AST-whitelist expression evaluator (WA.2).

The benchmark evaluates two distinct classes of size-knob expressions and,
before WA.2, did so through two unrelated code paths:

* **numeric expressions** — ``cost_model`` FLOPs/bytes formulas (W1.2's AST
  whitelist evaluator in ``cost_model.py``) and the ``shapes.yaml`` input
  specs (``dims`` / ``randint`` bounds / ``offsets:to=`` / ``full:`` /
  ``scalar_knob``), which ``inputs.py`` materialised through bare
  :func:`eval` calls.
* **predicate expressions** — the ``param_constraints`` knob constraints
  (WA.1 schema), boolean expressions over the knob universe.

WA.2 unifies both on **one AST-whitelist safety core** with two independent
entrypoints whose *semantic boundary is explicit*:

* :func:`eval_numeric` — returns a number. Arithmetic only (``+ - * / **
  // %``, unary ``+``/``-``), numeric literals, whitelisted names, and the
  numeric helpers ``min`` / ``max`` / ``ceil`` / ``floor`` / ``sqrt``.
* :func:`eval_predicate` — returns a ``bool``. Adds comparisons
  (``== != < <= > >=`` including Python chained comparisons), boolean
  operators (``and`` / ``or`` / ``not``), the membership operator ``in`` /
  ``not in``, ``list`` / ``tuple`` literals, the predicate helpers
  ``is_pow2`` / ``sqrt`` / ``gcd`` / ``len`` / ``sum``, and **constant
  integer indexing** of a ``list`` / ``tuple`` value (``block_size[0] ==
  8``) — a narrow ``ast.Subscript`` path that backs the ``scalar_seq``
  index constraints (`specs/task_api_contract.md §5.2`).

Both entrypoints reject arbitrary code **statically** — imports, attribute
access, comprehensions, lambdas, dunder names, and any call to a
non-whitelisted name raise :class:`ExprError` while walking the AST, never at
run time and never via :func:`eval`. Subscripting is rejected outright in
numeric mode and, in predicate mode, restricted to a constant integer index
into a sequence value (slices and non-constant indices are still rejected).

``cost_model.py`` keeps its own public surface (``FormulaEnv`` /
``evaluate_formula`` / ``parse_cost_model`` / ``CostModelError``) unchanged
and is layered on top of :func:`eval_numeric` — the W1.2 numeric contract
(values, call signatures, exception behaviour) is preserved exactly.
"""
from __future__ import annotations

import ast
import logging
import math
import operator
from collections.abc import Callable, Mapping
from typing import Any

logger = logging.getLogger(__name__)


class ExprError(ValueError):
    """Raised on an unparseable expression or any AST node / name / call
    outside the whitelist for the requested evaluation mode."""


# --------------------------------------------------------------------------
# Whitelisted operators.
# --------------------------------------------------------------------------

#: Binary arithmetic operators legal in both numeric and predicate modes.
_BIN_OPS: dict[type, Callable[[Any, Any], Any]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
}

#: Unary arithmetic operators legal in both modes.
_UNARY_OPS: dict[type, Callable[[Any], Any]] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}

#: Comparison operators — predicate mode only.
_CMP_OPS: dict[type, Callable[[Any, Any], Any]] = {
    ast.Eq: operator.eq,
    ast.NotEq: operator.ne,
    ast.Lt: operator.lt,
    ast.LtE: operator.le,
    ast.Gt: operator.gt,
    ast.GtE: operator.ge,
    ast.In: lambda a, b: a in b,
    ast.NotIn: lambda a, b: a not in b,
}


# --------------------------------------------------------------------------
# Whitelisted helper callables.
# --------------------------------------------------------------------------

def _is_pow2(x: Any) -> bool:
    """Return ``True`` iff ``x`` is a positive integer power of two."""
    if isinstance(x, bool) or not isinstance(x, int):
        n = int(x)
        if n != x:
            return False
    else:
        n = x
    return n > 0 and (n & (n - 1)) == 0


#: Numeric-mode helpers. ``min`` / ``max`` / ``ceil`` / ``floor`` are the
#: W1.2 cost-model set; ``sqrt`` is added by WA.2 to support derived scalars
#: such as ``1/sqrt(D)`` (WA.3) — it does not affect formulas that never call
#: it, so the cost-model numeric contract is unchanged.
_NUMERIC_HELPERS: dict[str, Callable[..., Any]] = {
    "min": min,
    "max": max,
    "ceil": lambda x: float(math.ceil(x)),
    "floor": lambda x: float(math.floor(x)),
    "sqrt": lambda x: math.sqrt(x),
}

#: Predicate-mode helpers. ``is_pow2`` / ``gcd`` express divisibility and
#: power-of-two knob constraints; ``len`` / ``sum`` support ``scalar_seq``
#: constraints (e.g. ``len(block_size) == 2``); ``sqrt`` is shared with the
#: numeric set so a predicate may reference it inside a comparison.
_PREDICATE_HELPERS: dict[str, Callable[..., Any]] = {
    "is_pow2": _is_pow2,
    "sqrt": lambda x: math.sqrt(x),
    "gcd": math.gcd,
    "len": len,
    "sum": sum,
}


# --------------------------------------------------------------------------
# The shared AST-whitelist safety core.
# --------------------------------------------------------------------------

class _Evaluator:
    """One AST walk over a whitelist.

    A single instance carries the evaluation *mode* (``"numeric"`` /
    ``"predicate"``), the concrete symbol table, the helper set legal for the
    mode, and the original expression text (for error messages). The mode
    decides which node types are accepted: predicate mode is a strict
    superset of numeric mode (it additionally allows comparisons, boolean
    operators, and list/tuple literals).
    """

    def __init__(
        self,
        mode: str,
        symbols: Mapping[str, Any],
        helpers: Mapping[str, Callable[..., Any]],
        expr: str,
    ) -> None:
        self._mode = mode
        self._symbols = symbols
        self._helpers = helpers
        self._expr = expr

    def eval(self, node: ast.AST) -> Any:
        """Recursively evaluate one AST node within the whitelist."""
        if isinstance(node, ast.Constant):
            return self._eval_constant(node)
        if isinstance(node, ast.Name):
            return self._eval_name(node)
        if isinstance(node, ast.BinOp):
            return self._eval_binop(node)
        if isinstance(node, ast.UnaryOp):
            return self._eval_unaryop(node)
        if isinstance(node, ast.Call):
            return self._eval_call(node)
        if self._mode == "predicate":
            if isinstance(node, ast.Compare):
                return self._eval_compare(node)
            if isinstance(node, ast.BoolOp):
                return self._eval_boolop(node)
            if isinstance(node, (ast.List, ast.Tuple)):
                return self._eval_sequence(node)
            if isinstance(node, ast.Subscript):
                return self._eval_subscript(node)
        raise ExprError(
            f"expression {self._expr!r}: {type(node).__name__} node not "
            f"allowed in {self._mode} mode"
        )

    def _eval_constant(self, node: ast.Constant) -> Any:
        value = node.value
        if self._mode == "numeric":
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ExprError(
                    f"expression {self._expr!r}: only numeric literals are "
                    f"allowed in numeric mode"
                )
            return value
        # Predicate mode also accepts bool / str / None literals (used by
        # `in` membership tests and list-literal constraints).
        if isinstance(value, (bool, int, float, str)) or value is None:
            return value
        raise ExprError(
            f"expression {self._expr!r}: literal of type "
            f"{type(value).__name__} not allowed"
        )

    def _eval_name(self, node: ast.Name) -> Any:
        name = node.id
        # A dunder name can never be a legal knob — reject explicitly so a
        # `__import__`-style escape fails with a clear message.
        if name.startswith("__") and name.endswith("__"):
            raise ExprError(
                f"expression {self._expr!r}: dunder name {name!r} not allowed"
            )
        if name not in self._symbols:
            raise ExprError(
                f"expression {self._expr!r}: unknown symbol {name!r} "
                f"(allowed: {sorted(self._symbols)})"
            )
        return self._symbols[name]

    def _eval_binop(self, node: ast.BinOp) -> Any:
        op = _BIN_OPS.get(type(node.op))
        if op is None:
            raise ExprError(
                f"expression {self._expr!r}: operator "
                f"{type(node.op).__name__} not allowed"
            )
        left = self.eval(node.left)
        right = self.eval(node.right)
        try:
            return op(left, right)
        except ZeroDivisionError as e:
            raise ExprError(
                f"expression {self._expr!r}: division by zero"
            ) from e

    def _eval_unaryop(self, node: ast.UnaryOp) -> Any:
        # `not` is a unary operator but belongs to predicate mode only.
        if isinstance(node.op, ast.Not):
            if self._mode != "predicate":
                raise ExprError(
                    f"expression {self._expr!r}: `not` not allowed in "
                    f"numeric mode"
                )
            return not self.eval(node.operand)
        op = _UNARY_OPS.get(type(node.op))
        if op is None:
            raise ExprError(
                f"expression {self._expr!r}: unary "
                f"{type(node.op).__name__} not allowed"
            )
        return op(self.eval(node.operand))

    def _eval_call(self, node: ast.Call) -> Any:
        if not isinstance(node.func, ast.Name) or (
            node.func.id not in self._helpers
        ):
            raise ExprError(
                f"expression {self._expr!r}: only {sorted(self._helpers)} "
                f"calls are allowed in {self._mode} mode"
            )
        if node.keywords:
            raise ExprError(
                f"expression {self._expr!r}: keyword arguments not allowed"
            )
        if any(isinstance(a, ast.Starred) for a in node.args):
            raise ExprError(
                f"expression {self._expr!r}: starred arguments not allowed"
            )
        args = [self.eval(a) for a in node.args]
        return self._helpers[node.func.id](*args)

    def _eval_compare(self, node: ast.Compare) -> bool:
        # Python chained comparison: `0 < eps <= 1` -> all adjacent pairs.
        left = self.eval(node.left)
        result = True
        for op_node, comparator_node in zip(
            node.ops, node.comparators, strict=True,
        ):
            op = _CMP_OPS.get(type(op_node))
            if op is None:
                raise ExprError(
                    f"expression {self._expr!r}: comparison "
                    f"{type(op_node).__name__} not allowed"
                )
            right = self.eval(comparator_node)
            if not op(left, right):
                result = False
                break
            left = right
        return result

    def _eval_boolop(self, node: ast.BoolOp) -> bool:
        if isinstance(node.op, ast.And):
            return all(bool(self.eval(v)) for v in node.values)
        if isinstance(node.op, ast.Or):
            return any(bool(self.eval(v)) for v in node.values)
        raise ExprError(
            f"expression {self._expr!r}: boolean operator "
            f"{type(node.op).__name__} not allowed"
        )

    def _eval_sequence(self, node: ast.List | ast.Tuple) -> list[Any]:
        # list / tuple literals back `D in [16, 32, 64]`; normalise both to a
        # list — membership semantics are identical.
        return [self.eval(elt) for elt in node.elts]

    def _eval_subscript(self, node: ast.Subscript) -> Any:
        """Evaluate a constant-integer index into a sequence value.

        Predicate mode only. Backs the ``scalar_seq`` index constraints
        (`specs/task_api_contract.md §5.2`), e.g. ``block_size[0] == 8``.
        The narrow contract:

        * the index must be a constant integer literal (``ast.Constant``),
          optionally negated (``seq[-1]``); a non-constant index — a name,
          an arithmetic expression — is rejected;
        * a slice (``seq[1:2]``) is rejected;
        * the subscripted value must be a ``list`` / ``tuple``; indexing a
          scalar is rejected;
        * an out-of-range index raises :class:`ExprError`, never a bare
          ``IndexError``.
        """
        index = self._const_int_index(node.slice)
        value = self.eval(node.value)
        if not isinstance(value, (list, tuple)):
            raise ExprError(
                f"expression {self._expr!r}: cannot index a "
                f"{type(value).__name__} value — only list / tuple"
            )
        try:
            return value[index]
        except IndexError as e:
            raise ExprError(
                f"expression {self._expr!r}: index {index} is out of range "
                f"for a sequence of length {len(value)}"
            ) from e

    def _const_int_index(self, slice_node: ast.AST) -> int:
        """Extract a constant integer index from a subscript ``slice`` node.

        Accepts a bare integer ``ast.Constant`` or an ``ast.UnaryOp`` that
        negates one (``seq[-1]``). Rejects slices and any non-constant index.
        """
        if isinstance(slice_node, ast.Slice):
            raise ExprError(
                f"expression {self._expr!r}: slice indexing not allowed — "
                f"only a constant integer index"
            )
        if isinstance(slice_node, ast.UnaryOp) and isinstance(
            slice_node.op, ast.USub
        ) and isinstance(slice_node.operand, ast.Constant):
            inner = slice_node.operand.value
            if isinstance(inner, bool) or not isinstance(inner, int):
                raise ExprError(
                    f"expression {self._expr!r}: subscript index must be a "
                    f"constant integer"
                )
            return -inner
        if isinstance(slice_node, ast.Constant) and not isinstance(
            slice_node.value, bool
        ) and isinstance(slice_node.value, int):
            return slice_node.value
        raise ExprError(
            f"expression {self._expr!r}: subscript index must be a constant "
            f"integer literal (got {type(slice_node).__name__})"
        )


def _parse(expr: str) -> ast.AST:
    """Parse one expression string, raising :class:`ExprError` on bad input."""
    if not isinstance(expr, str):
        raise ExprError(
            f"expression must be a string, got {type(expr).__name__}"
        )
    text = expr.strip()
    if not text:
        raise ExprError("expression is empty")
    try:
        return ast.parse(text, mode="eval")
    except SyntaxError as e:
        raise ExprError(f"cannot parse expression {expr!r}: {e.msg}") from e


# --------------------------------------------------------------------------
# Public entrypoints.
# --------------------------------------------------------------------------

def eval_numeric(expr: str, symbols: Mapping[str, Any] | None = None) -> float:
    """Safely evaluate a numeric size-knob expression.

    This is the numeric entrypoint of the shared evaluator. It backs both the
    ``cost_model`` FLOPs/bytes formulas and every numeric ``shapes.yaml``
    input expression (``dims`` / ``randint`` bounds / ``offsets:to=`` /
    ``full:`` / ``scalar_knob``).

    Args:
        expr: the size-knob expression string.
        symbols: the symbol table (knob values + any derived symbols). All
            values must be numbers; ``None`` is treated as an empty table.

    Returns:
        The expression value as a ``float``.

    Raises:
        ExprError: on a non-string / empty expression, a syntax error, an
            unknown name, an unknown call, a division by zero, or any AST
            node outside the numeric whitelist.
    """
    tree = _parse(expr)
    evaluator = _Evaluator("numeric", symbols or {}, _NUMERIC_HELPERS, expr)
    value = evaluator.eval(tree.body)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ExprError(
            f"expression {expr!r} did not evaluate to a number"
        )
    return float(value)


def eval_predicate(
    expr: str, symbols: Mapping[str, Any] | None = None,
) -> bool:
    """Safely evaluate a predicate (constraint) expression.

    This is the predicate entrypoint of the shared evaluator. It backs the
    ``param_constraints`` knob constraints (``dims`` / ``scalars`` value
    domains). It is a strict superset of :func:`eval_numeric`: in addition to
    arithmetic it allows comparisons (including Python chained comparisons),
    boolean operators, the membership operator ``in`` / ``not in``,
    ``list`` / ``tuple`` literals, and the predicate helpers ``is_pow2`` /
    ``sqrt`` / ``gcd`` / ``len`` / ``sum``.

    Args:
        expr: the constraint expression string.
        symbols: the symbol table (knob values, scalar values, ``scalar_seq``
            lists, …); ``None`` is treated as an empty table.

    Returns:
        The truth value of the expression as a ``bool``.

    Raises:
        ExprError: on a non-string / empty expression, a syntax error, an
            unknown name, an unknown call, or any AST node outside the
            predicate whitelist; also if the expression does not evaluate to
            a boolean-coercible value.
    """
    tree = _parse(expr)
    evaluator = _Evaluator("predicate", symbols or {}, _PREDICATE_HELPERS, expr)
    value = evaluator.eval(tree.body)
    # A constraint must read as a yes/no answer. Numbers / strings would make
    # a typo'd predicate (`D` instead of `is_pow2(D)`) silently truthy, so
    # only genuine booleans are accepted.
    if not isinstance(value, bool):
        raise ExprError(
            f"expression {expr!r} did not evaluate to a boolean "
            f"(got {type(value).__name__})"
        )
    return value


def exprs_equivalent(a: str, b: str) -> bool:
    """Return ``True`` iff two expression strings parse to the same AST.

    A whitespace- and formatting-insensitive structural comparison: the two
    strings are parsed and their ASTs dumped to a canonical form. ``"1/sqrt(D)
    "`` and ``"1 / sqrt( D )"`` are equivalent; ``"1/sqrt(D)"`` and
    ``"sqrt(D)*1"`` are not (the operand order differs). Used by the WA.4
    ``scalars.derive`` reconciliation to confirm a ``param_constraints``
    derive formula matches the ``shapes.yaml`` materialisation formula.

    Args:
        a: the first expression string.
        b: the second expression string.

    Returns:
        ``True`` iff both strings parse and yield structurally identical
        ASTs; ``False`` if either fails to parse or they differ.
    """
    try:
        tree_a = ast.parse(str(a).strip(), mode="eval")
        tree_b = ast.parse(str(b).strip(), mode="eval")
    except SyntaxError:
        return False
    return ast.dump(tree_a) == ast.dump(tree_b)


def _check_const_int_subscript(node: ast.Subscript, expr: str) -> None:
    """Statically reject a subscript that is not a constant integer index.

    Mirrors :meth:`_Evaluator._const_int_index` for the symbol-table-free
    syntax check: a slice or a non-constant index is a hard error.

    Args:
        node: the ``ast.Subscript`` node.
        expr: the original expression text, for error messages.

    Raises:
        ExprError: on a slice or a non-constant-integer index.
    """
    sl = node.slice
    if isinstance(sl, ast.Slice):
        raise ExprError(
            f"expression {expr!r}: slice indexing not allowed — only a "
            f"constant integer index"
        )
    if isinstance(sl, ast.UnaryOp) and isinstance(sl.op, ast.USub):
        operand = sl.operand
        if isinstance(operand, ast.Constant) and not isinstance(
            operand.value, bool
        ) and isinstance(operand.value, int):
            return
        raise ExprError(
            f"expression {expr!r}: subscript index must be a constant "
            f"integer literal"
        )
    if isinstance(sl, ast.Constant) and not isinstance(
        sl.value, bool
    ) and isinstance(sl.value, int):
        return
    raise ExprError(
        f"expression {expr!r}: subscript index must be a constant integer "
        f"literal (got {type(sl).__name__})"
    )


def check_expr_syntax(expr: str, mode: str) -> None:
    """Confirm an expression parses and uses only whitelisted AST nodes.

    Static check without a symbol table — unknown *names* are tolerated (the
    knob set is task-specific and unknown at schema-validation time); only
    syntax errors, disallowed node types, and non-whitelisted calls are
    rejected. Used by ``cost_model.parse_cost_model`` (and available to WA.4)
    to fail a malformed expression at load/validate time.

    Args:
        expr: the expression string.
        mode: ``"numeric"`` or ``"predicate"``.

    Raises:
        ExprError: on a syntax error or a disallowed node / call / literal;
            also if ``mode`` is not a recognised value.
    """
    if mode not in ("numeric", "predicate"):
        raise ExprError(f"unknown evaluation mode {mode!r}")
    tree = _parse(expr)
    helpers = _NUMERIC_HELPERS if mode == "numeric" else _PREDICATE_HELPERS
    allowed_nodes: tuple[type, ...] = (
        ast.Constant, ast.Name, ast.BinOp, ast.UnaryOp, ast.Call, ast.Load,
    )
    if mode == "predicate":
        allowed_nodes += (
            ast.Compare, ast.BoolOp, ast.List, ast.Tuple, ast.Subscript,
            ast.And, ast.Or, ast.Not,
            ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE,
            ast.In, ast.NotIn,
        )
    for node in ast.walk(tree):
        if isinstance(node, ast.expr) and not isinstance(node, allowed_nodes):
            raise ExprError(
                f"expression {expr!r}: {type(node).__name__} node not "
                f"allowed in {mode} mode"
            )
        # A `Subscript` is allowed in predicate mode only as a constant
        # integer index; reject slices and non-constant indices statically.
        if isinstance(node, ast.Subscript):
            _check_const_int_subscript(node, expr)
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name) or (
                node.func.id not in helpers
            ):
                raise ExprError(
                    f"expression {expr!r}: only {sorted(helpers)} calls "
                    f"allowed in {mode} mode"
                )
        if isinstance(node, ast.Name) and (
            node.id.startswith("__") and node.id.endswith("__")
        ):
            raise ExprError(
                f"expression {expr!r}: dunder name {node.id!r} not allowed"
            )
        if isinstance(node, ast.BinOp) and type(node.op) not in _BIN_OPS:
            raise ExprError(
                f"expression {expr!r}: operator "
                f"{type(node.op).__name__} not allowed"
            )
        if isinstance(node, ast.UnaryOp) and not isinstance(node.op, ast.Not):
            if type(node.op) not in _UNARY_OPS:
                raise ExprError(
                    f"expression {expr!r}: unary "
                    f"{type(node.op).__name__} not allowed"
                )
        if mode == "numeric" and isinstance(node, ast.Constant) and (
            isinstance(node.value, bool)
            or not isinstance(node.value, (int, float))
        ):
            raise ExprError(
                f"expression {expr!r}: only numeric literals allowed in "
                f"numeric mode"
            )
