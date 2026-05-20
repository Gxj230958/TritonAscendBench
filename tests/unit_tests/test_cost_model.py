"""W1.2: cost_model schema + shared formula evaluator tests.

Covers: the scalar and the map form for `flops` / `bytes` both parse and
normalise to a map; the safe AST formula evaluator (arithmetic whitelist,
`dtype_size` derived symbol, helper calls, rejection of `eval`-style code);
schema-error paths (missing keys, bad gpu_xcheck_tol, empty formula);
`validate_formulas` against a known knob universe.
"""
from __future__ import annotations

import pytest

from triton_ascend_bench.cost_model import (
    DERIVED_SYMBOL_DEFAULTS,
    VALID_BYTES_RESOURCES,
    VALID_FLOPS_CLASSES,
    CostModelError,
    FormulaEnv,
    evaluate_formula,
    parse_cost_model,
    validate_formulas,
)

# --------------------------------------------------------------------------
# formula evaluator — arithmetic whitelist
# --------------------------------------------------------------------------

def test_evaluate_basic_arithmetic():
    assert evaluate_formula("2*B*L*D", {"B": 2, "L": 3, "D": 4}) == 48.0
    assert evaluate_formula("(B + L) * D", {"B": 1, "L": 2, "D": 5}) == 15.0
    assert evaluate_formula("B**2", {"B": 4}) == 16.0
    assert evaluate_formula("D // 3", {"D": 10}) == 3.0
    assert evaluate_formula("-B + 10", {"B": 3}) == 7.0


def test_evaluate_helper_calls():
    assert evaluate_formula("max(B, L)", {"B": 2, "L": 9}) == 9.0
    assert evaluate_formula("min(B, L)", {"B": 2, "L": 9}) == 2.0
    assert evaluate_formula("ceil(B / 4)", {"B": 9}) == 3.0
    assert evaluate_formula("floor(B / 4)", {"B": 9}) == 2.0


def test_evaluate_unknown_symbol_rejected():
    with pytest.raises(CostModelError, match="unknown symbol 'X'"):
        evaluate_formula("X * 2", {"B": 2})


def test_evaluate_rejects_arbitrary_code():
    # The whole point of the AST whitelist: no attribute access, no
    # subscripts, no calls outside the helper set, no `eval`-style escape.
    for malicious in (
        "__import__('os').system('echo hi')",
        "B.real",
        "B[0]",
        "[x for x in range(B)]",
        "lambda: B",
        "open('/etc/passwd')",
    ):
        with pytest.raises(CostModelError):
            evaluate_formula(malicious, {"B": 2})


def test_evaluate_rejects_non_numeric_literal():
    with pytest.raises(CostModelError):
        evaluate_formula("'str'", {})


def test_evaluate_division_by_zero():
    with pytest.raises(CostModelError, match="division by zero"):
        evaluate_formula("B / 0", {"B": 4})


def test_evaluate_empty_formula():
    with pytest.raises(CostModelError, match="empty"):
        evaluate_formula("   ", {})


def test_evaluate_non_string_formula():
    with pytest.raises(CostModelError, match="must be a string"):
        evaluate_formula(123, {})  # type: ignore[arg-type]


# --------------------------------------------------------------------------
# FormulaEnv — derived symbols (dtype_size)
# --------------------------------------------------------------------------

def test_formula_env_dtype_size_default():
    # `dtype_size` is a derived symbol available without the caller
    # supplying it; the validator-facing default is finite.
    env = FormulaEnv({"B": 2, "L": 3, "D": 4})
    expected = (2 * 3 * 4) * DERIVED_SYMBOL_DEFAULTS["dtype_size"]
    assert env.evaluate("B*L*D * dtype_size") == expected


def test_formula_env_dtype_size_override():
    env = FormulaEnv({"B": 2}, derived={"dtype_size": 4.0})
    assert env.evaluate("B * dtype_size") == 8.0


def test_formula_env_unknown_derived_rejected():
    with pytest.raises(CostModelError, match="unknown derived symbol"):
        FormulaEnv({"B": 2}, derived={"bogus": 1.0})


def test_formula_env_knob_cannot_shadow_derived():
    with pytest.raises(CostModelError, match="collides"):
        FormulaEnv({"dtype_size": 99})


def test_formula_env_non_numeric_knob_rejected():
    with pytest.raises(CostModelError, match="must be a number"):
        FormulaEnv({"B": "huge"})


# --------------------------------------------------------------------------
# parse_cost_model — scalar form
# --------------------------------------------------------------------------

def test_parse_scalar_form():
    raw = {
        "flops": "2*B*L*D",
        "flops_class": "fp16_cube",
        "bytes": "(B*L*D + B*D) * dtype_size",
        "bytes_resource": "hbm",
        "gpu_xcheck_tol": 0.05,
    }
    model = parse_cost_model(raw)
    # scalar normalises to a one-entry map keyed by the companion class /
    # resource — a real peak-table key, not a synthetic "default".
    assert model.flops == {"fp16_cube": "2*B*L*D"}
    assert model.bytes == {"hbm": "(B*L*D + B*D) * dtype_size"}
    assert model.flops_scalar is True
    assert model.bytes_scalar is True
    assert model.gpu_xcheck_tol == 0.05


def test_parse_scalar_form_default_tol():
    model = parse_cost_model({
        "flops": "B", "flops_class": "vector",
        "bytes": "B*dtype_size", "bytes_resource": "hbm",
    })
    assert model.gpu_xcheck_tol == 0.05


def test_parse_scalar_form_missing_companion_class():
    # A scalar `flops` with no `flops_class` companion is rejected.
    with pytest.raises(CostModelError, match="flops_class is missing"):
        parse_cost_model({"flops": "B*N", "bytes": {"hbm": "B*N"}})


def test_parse_scalar_form_missing_companion_resource():
    with pytest.raises(CostModelError, match="bytes_resource is missing"):
        parse_cost_model(
            {"flops": {"vector": "B*N"}, "bytes": "B*N"}
        )


def test_parse_scalar_form_bad_companion_class():
    # An unknown compute class for the companion field is rejected.
    with pytest.raises(CostModelError, match="flops_class must be one of"):
        parse_cost_model({
            "flops": "B*N", "flops_class": "tensorcore",
            "bytes": {"hbm": "B*N"},
        })


def test_parse_map_form_rejects_stray_companion():
    # A `flops_class` on a map-form `flops` is meaningless and rejected.
    with pytest.raises(CostModelError, match="flops_class is only valid"):
        parse_cost_model({
            "flops": {"vector": "B*N"}, "flops_class": "vector",
            "bytes": {"hbm": "B*N"},
        })


def test_valid_companion_sets_are_peak_table_keys():
    # The schema-level allowed sets mirror the peak-table classes/resources.
    assert VALID_FLOPS_CLASSES == frozenset(
        {"fp16_cube", "bf16_cube", "int8", "vector"})
    assert VALID_BYTES_RESOURCES == frozenset({"hbm", "interconnect"})


# --------------------------------------------------------------------------
# parse_cost_model — map form
# --------------------------------------------------------------------------

def test_parse_map_form():
    raw = {
        "flops": {"fp16_cube": "2*M*N*K", "vector": "M*N"},
        "bytes": {
            "hbm": "(M*K + K*N + M*N) * dtype_size",
            "interconnect": "M*N * dtype_size",
        },
        "gpu_xcheck_tol": 0.1,
    }
    model = parse_cost_model(raw)
    assert model.flops == {"fp16_cube": "2*M*N*K", "vector": "M*N"}
    assert set(model.bytes) == {"hbm", "interconnect"}
    assert model.flops_scalar is False
    assert model.bytes_scalar is False
    assert model.gpu_xcheck_tol == 0.1


def test_parse_mixed_scalar_and_map():
    # `flops` scalar + `bytes` map is a legal independent combination; the
    # scalar `flops` still carries its companion class.
    model = parse_cost_model(
        {"flops": "B*N", "flops_class": "vector",
         "bytes": {"hbm": "2*B*N*dtype_size"}}
    )
    assert model.flops_scalar is True
    assert model.bytes_scalar is False
    assert model.flops == {"vector": "B*N"}
    assert model.bytes == {"hbm": "2*B*N*dtype_size"}


# --------------------------------------------------------------------------
# parse_cost_model — error paths
# --------------------------------------------------------------------------

def test_parse_missing_key():
    with pytest.raises(CostModelError, match="missing required key"):
        parse_cost_model({"flops": "B"})


def test_parse_not_a_mapping():
    with pytest.raises(CostModelError, match="must be a mapping"):
        parse_cost_model("2*B*L*D")


def test_parse_empty_scalar_formula():
    with pytest.raises(CostModelError, match="empty"):
        parse_cost_model({"flops": "  ", "bytes": "B"})


def test_parse_empty_map():
    with pytest.raises(CostModelError, match="map is empty"):
        parse_cost_model({"flops": {}, "bytes": "B"})


def test_parse_bad_tol_range():
    with pytest.raises(CostModelError, match=r"\(0, 1\)"):
        parse_cost_model({
            "flops": "B", "flops_class": "vector",
            "bytes": "B", "bytes_resource": "hbm", "gpu_xcheck_tol": 1.5,
        })


def test_parse_bad_tol_type():
    with pytest.raises(CostModelError, match="must be a number"):
        parse_cost_model({
            "flops": "B", "flops_class": "vector",
            "bytes": "B", "bytes_resource": "hbm", "gpu_xcheck_tol": "loose",
        })


def test_parse_rejects_unsafe_formula():
    with pytest.raises(CostModelError):
        parse_cost_model({"flops": "__import__('os')", "bytes": "B"})


def test_parse_rejects_syntax_error():
    with pytest.raises(CostModelError, match="cannot parse"):
        parse_cost_model({"flops": "2 * * B", "flops_class": "vector",
                          "bytes": "B", "bytes_resource": "hbm"})


# --------------------------------------------------------------------------
# validate_formulas — known knob universe
# --------------------------------------------------------------------------

def test_validate_formulas_ok():
    model = parse_cost_model(
        {"flops": "2*B*L*D", "flops_class": "fp16_cube",
         "bytes": "B*L*D*dtype_size", "bytes_resource": "hbm"}
    )
    # No raise: every symbol is a knob or the derived `dtype_size`.
    validate_formulas(model, {"B", "L", "D"})


def test_validate_formulas_unknown_knob():
    # `parse_cost_model` cannot catch an unknown *name* (knob set unknown);
    # `validate_formulas` does, against the real shapes.yaml knob universe.
    model = parse_cost_model({"flops": "2*B*Z", "flops_class": "vector",
                              "bytes": "B*dtype_size", "bytes_resource": "hbm"})
    with pytest.raises(CostModelError, match="unknown symbol 'Z'"):
        validate_formulas(model, {"B"})


def test_validate_formulas_dtype_size_is_known():
    # `dtype_size` is a derived symbol — never needs to be a declared knob.
    model = parse_cost_model({"flops": "B", "flops_class": "vector",
                              "bytes": "B*dtype_size", "bytes_resource": "hbm"})
    validate_formulas(model, {"B"})
