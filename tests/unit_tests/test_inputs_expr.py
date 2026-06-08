"""WA.2: `inputs.py` numeric-expression sites route through `eval_numeric`.

`inputs.py` previously materialised its numeric size-knob expressions
(``dims`` / ``randint`` bounds / ``offsets:to=`` / ``full:`` / ``scalar_knob``)
through bare :func:`eval` calls. WA.2 replaces every one with the shared
AST-whitelist :func:`ascendbench.input_spec.expr.eval_numeric`. These tests
confirm the materialisation behaviour is unchanged (same knobs → same values)
and that an unsafe expression at an input site is now statically rejected.
"""
from __future__ import annotations

import pytest
import torch

from ascendbench.input_spec.expr import ExprError
from ascendbench.input_spec.inputs import _resolve_dims, build_inputs


def _shapes_doc(inputs: dict) -> dict:
    """A minimal one-row shapes.yaml doc wrapping the given inputs block."""
    return {
        "size_knobs": {"B": 2, "L": 8, "D": 16},
        "public_shapes": [{"id": "P1", "name": "tiny", "knobs": {}}],
        "hidden_shapes": [],
        "inputs": inputs,
    }


# --------------------------------------------------------------------------
# dims — knob arithmetic expressions
# --------------------------------------------------------------------------

def test_resolve_dims_passes_through_ints():
    assert _resolve_dims([4, 8, 16], {}) == [4, 8, 16]


def test_resolve_dims_evaluates_knob_expressions():
    knobs = {"B": 2, "L": 8, "D": 16}
    assert _resolve_dims(["B", "L", "D"], knobs) == [2, 8, 16]
    assert _resolve_dims(["2*D", "L+1"], knobs) == [32, 9]
    assert _resolve_dims(["D // 4"], knobs) == [4]


def test_resolve_dims_rejects_bad_dim_type():
    with pytest.raises(ValueError, match="bad dim"):
        _resolve_dims([1.5], {})  # type: ignore[list-item]


def test_resolve_dims_rejects_unsafe_expression():
    # A dim string is now an `eval_numeric` expression — arbitrary code is
    # statically rejected, not silently run as it would be via bare `eval`.
    with pytest.raises(ExprError):
        _resolve_dims(["__import__('os')"], {"B": 2})


def test_build_inputs_dims_materialise():
    doc = _shapes_doc({"x": {"dims": ["B", "L", "D"], "dtype": "float32",
                             "dist": "zeros"}})
    out = build_inputs(doc, "P1", seed=0)
    assert out["x"].shape == (2, 8, 16)


# --------------------------------------------------------------------------
# randint bounds — lo / hi knob expressions
# --------------------------------------------------------------------------

def test_build_inputs_randint_bounds_from_knobs():
    doc = _shapes_doc({"idx": {"dims": ["B"], "dtype": "int64",
                               "dist": "randint:0:L"}})
    out = build_inputs(doc, "P1", seed=0)
    # hi bound is `L` (= 8); every sampled index is in [0, 8).
    assert out["idx"].min().item() >= 0
    assert out["idx"].max().item() < 8


# --------------------------------------------------------------------------
# offsets:to= — total knob expression
# --------------------------------------------------------------------------

def test_build_inputs_offsets_to_total():
    doc = _shapes_doc({"cu": {"dims": ["B"], "dtype": "int64",
                              "dist": "offsets:to=L*D"}})
    out = build_inputs(doc, "P1", seed=0)
    # Final cumulative value equals eval_numeric("L*D") = 8 * 16 = 128.
    assert out["cu"][0].item() == 0
    assert out["cu"][-1].item() == 128
    assert torch.all(out["cu"][1:] >= out["cu"][:-1])  # monotonic


# --------------------------------------------------------------------------
# full: — constant value knob expression
# --------------------------------------------------------------------------

def test_build_inputs_full_constant_from_knob():
    doc = _shapes_doc({"lens": {"dims": ["B"], "dtype": "int64",
                                "dist": "full:D // 4"}})
    out = build_inputs(doc, "P1", seed=0)
    # value is eval_numeric("D // 4") = 16 // 4 = 4.
    assert torch.all(out["lens"] == 4)


# --------------------------------------------------------------------------
# scalar_knob — derived integer from knobs
# --------------------------------------------------------------------------

def test_build_inputs_scalar_knob_from_knobs():
    doc = _shapes_doc({"n": {"scalar_knob": "B * L"}})
    out = build_inputs(doc, "P1", seed=0)
    assert out["n"] == 16
    assert isinstance(out["n"], int)


def test_build_inputs_scalar_passthrough_unchanged():
    # A plain `scalar` value is still passed through verbatim (not evaluated).
    doc = _shapes_doc({"eps": {"scalar": 1e-5}})
    out = build_inputs(doc, "P1", seed=0)
    assert out["eps"] == 1e-5
