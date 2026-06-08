"""WA.2: shared AST-whitelist expression evaluator tests.

Covers the two independent entrypoints of ``ascendbench.input_spec.expr``:

* :func:`eval_numeric` — numeric size-knob expressions: normal arithmetic,
  the numeric helpers (``min`` / ``max`` / ``ceil`` / ``floor`` / ``sqrt``),
  and static rejection of every non-arithmetic / unsafe construct.
* :func:`eval_predicate` — constraint expressions: comparisons (including
  Python chained comparisons), boolean operators, ``in`` membership,
  list / tuple literals, the predicate helpers (``is_pow2`` / ``sqrt`` /
  ``gcd`` / ``len`` / ``sum``), and static rejection of unsafe code.

Both share one AST safety core, so the unsafe-code tests assert that imports,
attribute access, subscripts, comprehensions, lambdas, dunder names, and
non-whitelisted calls raise :class:`ExprError` *statically* (at AST-walk time,
never via :func:`eval`).
"""
from __future__ import annotations

import math

import pytest

from ascendbench.input_spec.expr import (
    ExprError,
    check_expr_syntax,
    eval_numeric,
    eval_predicate,
    exprs_equivalent,
)

# A representative set of arbitrary-code / unsafe expressions. Both
# entrypoints must reject every one of these statically.
#
# A constant-integer `Subscript` (`block_size[0]`) is *not* listed here: it
# is a legal shape in predicate mode (the `scalar_seq` index constraints) and
# is exercised separately. It is unsafe only in numeric mode, where it appears
# in `_NUMERIC_ONLY_UNSAFE_EXPRS`.
_UNSAFE_EXPRS = [
    "__import__('os').system('echo hi')",
    "open('/etc/passwd')",
    "B.real",                       # attribute access
    "B.__class__",                  # attribute access to a dunder
    "[x for x in range(B)]",        # comprehension
    "{x: x for x in range(B)}",     # dict comprehension
    "(x for x in range(B))",        # generator expression
    "lambda: B",                    # lambda
    "B if B else B",                # conditional expression (IfExp)
    "f'{B}'",                       # f-string / JoinedStr
    "__builtins__",                 # dunder name
    "B := 3",                       # walrus / named expression
]

# Expressions that are unsafe in numeric mode but legal (as a shape) in
# predicate mode — subscripting is a predicate-only construct.
_NUMERIC_ONLY_UNSAFE_EXPRS = [
    "B[0]",                         # subscript — numeric mode rejects it
]


# ==========================================================================
# eval_numeric — normal evaluation
# ==========================================================================

def test_numeric_basic_arithmetic():
    assert eval_numeric("2*B*L*D", {"B": 2, "L": 3, "D": 4}) == 48.0
    assert eval_numeric("(B + L) * D", {"B": 1, "L": 2, "D": 5}) == 15.0
    assert eval_numeric("B**2", {"B": 4}) == 16.0
    assert eval_numeric("D // 3", {"D": 10}) == 3.0
    assert eval_numeric("D % 3", {"D": 10}) == 1.0
    assert eval_numeric("-B + 10", {"B": 3}) == 7.0
    assert eval_numeric("+B", {"B": 5}) == 5.0


def test_numeric_literal_only():
    assert eval_numeric("42") == 42.0
    assert eval_numeric("2 * 3 + 1") == 7.0


def test_numeric_returns_float():
    # Even an integer-valued expression comes back as a float.
    result = eval_numeric("B", {"B": 7})
    assert isinstance(result, float)
    assert result == 7.0


def test_numeric_none_symbols_is_empty_table():
    assert eval_numeric("2 + 2") == 4.0
    assert eval_numeric("2 + 2", None) == 4.0


# ==========================================================================
# eval_numeric — helper functions (min / max / ceil / floor / sqrt)
# ==========================================================================

def test_numeric_helper_min_max():
    assert eval_numeric("max(B, L)", {"B": 2, "L": 9}) == 9.0
    assert eval_numeric("min(B, L)", {"B": 2, "L": 9}) == 2.0
    assert eval_numeric("max(B, L, D)", {"B": 2, "L": 9, "D": 5}) == 9.0


def test_numeric_helper_ceil_floor():
    assert eval_numeric("ceil(B / 4)", {"B": 9}) == 3.0
    assert eval_numeric("floor(B / 4)", {"B": 9}) == 2.0


def test_numeric_helper_sqrt():
    # WA.2 adds `sqrt` to support derived scalars like `1/sqrt(D)` (WA.3).
    assert eval_numeric("sqrt(D)", {"D": 64}) == 8.0
    assert eval_numeric("1 / sqrt(D)", {"D": 64}) == pytest.approx(0.125)
    assert eval_numeric("1 / sqrt(D)", {"D": 128}) == pytest.approx(
        1.0 / math.sqrt(128))


def test_numeric_predicate_helpers_not_available():
    # `is_pow2` / `gcd` / `len` / `sum` are predicate-mode helpers; calling
    # them in numeric mode is a static rejection, not a silent success.
    for expr in ("is_pow2(D)", "gcd(B, L)", "len(B)", "sum(B)"):
        with pytest.raises(ExprError):
            eval_numeric(expr, {"B": 4, "L": 6, "D": 8})


# ==========================================================================
# eval_numeric — error paths
# ==========================================================================

def test_numeric_unknown_symbol_rejected():
    with pytest.raises(ExprError, match="unknown symbol 'X'"):
        eval_numeric("X * 2", {"B": 2})


def test_numeric_division_by_zero():
    with pytest.raises(ExprError, match="division by zero"):
        eval_numeric("B / 0", {"B": 4})


def test_numeric_empty_expression():
    with pytest.raises(ExprError, match="empty"):
        eval_numeric("   ")


def test_numeric_non_string_expression():
    with pytest.raises(ExprError, match="must be a string"):
        eval_numeric(123)  # type: ignore[arg-type]


def test_numeric_syntax_error():
    with pytest.raises(ExprError, match="cannot parse"):
        eval_numeric("2 * * B", {"B": 2})


def test_numeric_rejects_string_literal():
    with pytest.raises(ExprError, match="numeric literals"):
        eval_numeric("'str'")


def test_numeric_rejects_bool_literal():
    # A bool literal in numeric mode is rejected (bool is not a number here).
    with pytest.raises(ExprError, match="numeric literals"):
        eval_numeric("True")


def test_numeric_rejects_comparison():
    # Comparisons belong to predicate mode; numeric mode must not accept them
    # (they would otherwise return a bool, not a number).
    with pytest.raises(ExprError, match="numeric mode"):
        eval_numeric("B > 3", {"B": 5})


def test_numeric_rejects_boolean_operator():
    with pytest.raises(ExprError, match="numeric mode"):
        eval_numeric("B and L", {"B": 1, "L": 2})


def test_numeric_rejects_not_operator():
    with pytest.raises(ExprError, match="numeric mode"):
        eval_numeric("not B", {"B": 1})


def test_numeric_rejects_list_literal():
    with pytest.raises(ExprError, match="numeric mode"):
        eval_numeric("[1, 2, 3]")


@pytest.mark.parametrize(
    "expr", _UNSAFE_EXPRS + _NUMERIC_ONLY_UNSAFE_EXPRS,
)
def test_numeric_rejects_arbitrary_code(expr):
    # The whole point of the AST whitelist: no attribute access, no
    # subscripts, no comprehensions, no `eval`-style escape — caught
    # statically while walking the AST. Subscripting is rejected in numeric
    # mode (it is a predicate-only construct).
    with pytest.raises(ExprError):
        eval_numeric(expr, {"B": 4})


def test_numeric_rejects_keyword_arguments():
    with pytest.raises(ExprError, match="keyword arguments"):
        eval_numeric("min(B, key=1)", {"B": 4})


def test_numeric_rejects_starred_arguments():
    with pytest.raises(ExprError, match="starred"):
        eval_numeric("max(*B)", {"B": 4})


def test_numeric_rejects_dunder_name():
    with pytest.raises(ExprError, match="dunder"):
        eval_numeric("__import__", {"__import__": 1})


# ==========================================================================
# eval_predicate — normal evaluation
# ==========================================================================

def test_predicate_comparisons():
    assert eval_predicate("D == 64", {"D": 64}) is True
    assert eval_predicate("D != 64", {"D": 32}) is True
    assert eval_predicate("L >= 16", {"L": 16}) is True
    assert eval_predicate("L >= 16", {"L": 8}) is False
    assert eval_predicate("B < L", {"B": 2, "L": 9}) is True
    assert eval_predicate("B <= B", {"B": 3}) is True
    assert eval_predicate("H % G == 0", {"H": 12, "G": 4}) is True
    assert eval_predicate("H % G == 0", {"H": 13, "G": 4}) is False


def test_predicate_chained_comparison():
    # Python chained comparison: `0 < eps <= 1e-2` — both adjacent pairs.
    assert eval_predicate("0 < eps <= 1e-2", {"eps": 1e-5}) is True
    assert eval_predicate("0 < eps <= 1e-2", {"eps": 0.0}) is False
    assert eval_predicate("0 < eps <= 1e-2", {"eps": 1.0}) is False
    assert eval_predicate("1 <= D <= 128", {"D": 64}) is True
    assert eval_predicate("1 <= D <= 128", {"D": 256}) is False


def test_predicate_boolean_operators():
    assert eval_predicate("D == 64 and L >= 16", {"D": 64, "L": 32}) is True
    assert eval_predicate("D == 64 and L >= 16", {"D": 64, "L": 8}) is False
    assert eval_predicate("D == 64 or D == 128", {"D": 128}) is True
    assert eval_predicate("not (D == 64)", {"D": 32}) is True
    assert eval_predicate("not (D == 64)", {"D": 64}) is False


def test_predicate_in_membership():
    # The `D in [16, 32, 64, 128]` closed-knob-set form from the spec.
    assert eval_predicate("D in [16, 32, 64, 128]", {"D": 64}) is True
    assert eval_predicate("D in [16, 32, 64, 128]", {"D": 48}) is False
    assert eval_predicate("D not in [16, 32]", {"D": 64}) is True
    # tuple literal works identically.
    assert eval_predicate("D in (16, 32, 64)", {"D": 32}) is True


def test_predicate_list_literal_with_expressions():
    # List-literal elements may themselves be knob expressions.
    assert eval_predicate("X in [B, 2*B, 3*B]", {"X": 6, "B": 3}) is True
    assert eval_predicate("X in [B, 2*B, 3*B]", {"X": 7, "B": 3}) is False


def test_predicate_helper_is_pow2():
    assert eval_predicate("is_pow2(D)", {"D": 64}) is True
    assert eval_predicate("is_pow2(D)", {"D": 48}) is False
    assert eval_predicate("is_pow2(D)", {"D": 1}) is True
    assert eval_predicate("is_pow2(D)", {"D": 0}) is False


def test_predicate_helper_gcd():
    assert eval_predicate("gcd(H, G) == G", {"H": 12, "G": 4}) is True
    assert eval_predicate("gcd(H, G) == G", {"H": 13, "G": 4}) is False


def test_predicate_helper_sqrt_in_comparison():
    assert eval_predicate("sqrt(D) == 8", {"D": 64}) is True


def test_predicate_helper_len_sum_scalar_seq():
    # `scalar_seq` constraints: `len(block_size) == 2`,
    # `sum(mrope_section) == D // 2` — symbols may bind to lists.
    assert eval_predicate(
        "len(block_size) == 2", {"block_size": [8, 8]}) is True
    assert eval_predicate(
        "len(block_size) == 2", {"block_size": [8, 8, 8]}) is False
    assert eval_predicate(
        "sum(mrope_section) == D // 2",
        {"mrope_section": [16, 24, 24], "D": 128},
    ) is True
    assert eval_predicate(
        "sum(mrope_section) == D // 2",
        {"mrope_section": [16, 24, 20], "D": 128},
    ) is False


def test_predicate_none_symbols_is_empty_table():
    assert eval_predicate("1 < 2") is True
    assert eval_predicate("1 < 2", None) is True


# ==========================================================================
# eval_predicate — error paths
# ==========================================================================

def test_predicate_unknown_symbol_rejected():
    with pytest.raises(ExprError, match="unknown symbol 'X'"):
        eval_predicate("X == 1", {"D": 64})


def test_predicate_empty_expression():
    with pytest.raises(ExprError, match="empty"):
        eval_predicate("  ")


def test_predicate_non_string_expression():
    with pytest.raises(ExprError, match="must be a string"):
        eval_predicate(123)  # type: ignore[arg-type]


def test_predicate_syntax_error():
    with pytest.raises(ExprError, match="cannot parse"):
        eval_predicate("D == == 64", {"D": 64})


def test_predicate_must_return_boolean():
    # A bare arithmetic expression is not a yes/no answer — a typo'd
    # predicate (`D` instead of `is_pow2(D)`) must fail, not be truthy.
    with pytest.raises(ExprError, match="did not evaluate to a boolean"):
        eval_predicate("D", {"D": 64})
    with pytest.raises(ExprError, match="did not evaluate to a boolean"):
        eval_predicate("D * 2", {"D": 64})


def test_predicate_unknown_call_rejected():
    with pytest.raises(ExprError, match="calls are allowed"):
        eval_predicate("abs(D) == 1", {"D": -1})


@pytest.mark.parametrize("expr", _UNSAFE_EXPRS)
def test_predicate_rejects_arbitrary_code(expr):
    # The shared AST safety core rejects unsafe code in predicate mode too.
    with pytest.raises(ExprError):
        eval_predicate(expr, {"B": 4})


def test_predicate_rejects_dunder_name():
    with pytest.raises(ExprError, match="dunder"):
        eval_predicate("__class__ == 1", {"__class__": 1})


def test_predicate_rejects_keyword_arguments():
    with pytest.raises(ExprError, match="keyword arguments"):
        eval_predicate("is_pow2(D, x=1)", {"D": 4})


# ==========================================================================
# eval_predicate — constant-integer subscript (scalar_seq index constraints)
# ==========================================================================

def test_predicate_subscript_constant_index():
    # `scalar_seq` index constraints: `block_size[0] == 8`.
    syms = {"block_size": [8, 8]}
    assert eval_predicate("block_size[0] == 8", syms) is True
    assert eval_predicate("block_size[1] == 8", syms) is True
    assert eval_predicate("block_size[0] == block_size[1]", syms) is True


def test_predicate_subscript_negative_index():
    # A negative constant index addresses from the tail.
    syms = {"mrope_section": [3, 3, 2]}
    assert eval_predicate("mrope_section[-1] == 2", syms) is True
    assert eval_predicate("mrope_section[-3] == 3", syms) is True


def test_predicate_subscript_on_tuple():
    assert eval_predicate("xs[1] == 5", {"xs": (4, 5, 6)}) is True


def test_predicate_subscript_out_of_range_raises_exprerror():
    # An out-of-range index surfaces as ExprError, never a bare IndexError.
    with pytest.raises(ExprError, match="out of range"):
        eval_predicate("block_size[5] == 8", {"block_size": [8, 8]})
    with pytest.raises(ExprError, match="out of range"):
        eval_predicate("block_size[-9] == 8", {"block_size": [8, 8]})


def test_predicate_subscript_slice_rejected():
    with pytest.raises(ExprError, match="slice"):
        eval_predicate("block_size[0:1] == [8]", {"block_size": [8, 8]})


def test_predicate_subscript_non_constant_index_rejected():
    # A name / arithmetic index is rejected — only a constant integer.
    with pytest.raises(ExprError, match="constant integer"):
        eval_predicate("block_size[i] == 8", {"block_size": [8, 8], "i": 0})
    with pytest.raises(ExprError, match="constant integer"):
        eval_predicate("block_size[0 + 0] == 8", {"block_size": [8, 8]})


def test_predicate_subscript_on_scalar_rejected():
    # Indexing a non-sequence value is rejected.
    with pytest.raises(ExprError, match="cannot index"):
        eval_predicate("D[0] == 8", {"D": 16})


def test_numeric_subscript_rejected():
    # Subscripting is a predicate-only construct; numeric mode rejects it.
    with pytest.raises(ExprError):
        eval_numeric("xs[0]", {"xs": [1, 2]})


def test_check_expr_syntax_predicate_subscript_ok():
    # A constant-integer subscript passes the static predicate check.
    check_expr_syntax("block_size[0] == 8", "predicate")
    check_expr_syntax("mrope_section[-1] == 2", "predicate")


def test_check_expr_syntax_predicate_rejects_slice():
    with pytest.raises(ExprError, match="slice"):
        check_expr_syntax("block_size[0:1] == [8]", "predicate")


def test_check_expr_syntax_predicate_rejects_non_constant_index():
    with pytest.raises(ExprError, match="constant integer"):
        check_expr_syntax("block_size[i] == 8", "predicate")
    with pytest.raises(ExprError, match="constant integer"):
        check_expr_syntax("block_size[0 + 0] == 8", "predicate")


# ==========================================================================
# check_expr_syntax — static syntax check (no symbol table)
# ==========================================================================

def test_check_expr_syntax_numeric_ok():
    # Unknown names are tolerated — the knob set is unknown at this stage.
    check_expr_syntax("2 * B * unknown_knob", "numeric")
    check_expr_syntax("ceil(B / 4)", "numeric")


def test_check_expr_syntax_predicate_ok():
    check_expr_syntax("D in [16, 32, 64]", "predicate")
    check_expr_syntax("is_pow2(D) and L >= 16", "predicate")


def test_check_expr_syntax_rejects_bad_mode():
    with pytest.raises(ExprError, match="unknown evaluation mode"):
        check_expr_syntax("B", "boolean")


def test_check_expr_syntax_numeric_rejects_comparison():
    with pytest.raises(ExprError, match="numeric mode"):
        check_expr_syntax("B > 3", "numeric")


def test_check_expr_syntax_rejects_syntax_error():
    with pytest.raises(ExprError, match="cannot parse"):
        check_expr_syntax("2 * * B", "numeric")


@pytest.mark.parametrize("expr", _UNSAFE_EXPRS)
def test_check_expr_syntax_rejects_arbitrary_code(expr):
    # Static syntax check catches unsafe code without ever evaluating it.
    with pytest.raises(ExprError):
        check_expr_syntax(expr, "numeric")
    with pytest.raises(ExprError):
        check_expr_syntax(expr, "predicate")


@pytest.mark.parametrize("expr", _NUMERIC_ONLY_UNSAFE_EXPRS)
def test_check_expr_syntax_numeric_rejects_subscript(expr):
    # Subscripting is rejected statically in numeric mode — it is a
    # predicate-only construct (the `scalar_seq` index constraints).
    with pytest.raises(ExprError):
        check_expr_syntax(expr, "numeric")


def test_check_expr_syntax_numeric_rejects_predicate_helper():
    with pytest.raises(ExprError, match="numeric mode"):
        check_expr_syntax("is_pow2(D)", "numeric")


# ==========================================================================
# exprs_equivalent — structural AST comparison (WA.4 derive reconciliation)
# ==========================================================================

def test_exprs_equivalent_whitespace_insensitive():
    assert exprs_equivalent("1/sqrt(D)", "1 / sqrt( D )") is True
    assert exprs_equivalent("1/sqrt(D)", "  1/sqrt(D)  ") is True


def test_exprs_equivalent_structural_difference():
    # A different operand order is not equivalent — structural comparison.
    assert exprs_equivalent("1/sqrt(D)", "sqrt(D)*1") is False
    assert exprs_equivalent("2/sqrt(D)", "1/sqrt(D)") is False


def test_exprs_equivalent_unparseable_is_false():
    assert exprs_equivalent("1 +", "1 + 1") is False
    assert exprs_equivalent("1 + 1", "* *") is False
