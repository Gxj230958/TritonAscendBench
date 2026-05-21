"""WA.3: derived scalars, JSON-scalar literal schema, dist normalisation.

WA.3 closes three gaps in the non-tensor side of the ``shapes.yaml``
materialisation layer:

* **① derived scalars** — ``{derive: "1/sqrt(D)"}`` replaces a hardcoded
  ``{scalar: 0.125}``; ``build_inputs`` computes the value per shape row
  through the shared :func:`expr.eval_numeric`, so it tracks the head-dim
  knob (``D=64 ⇒ 0.125``, ``D=128 ⇒ 1/√128``).
* **② JSON-scalar literal schema** — ``scalar`` / ``derive`` values are
  constrained to JSON scalars (the W1.0 sidecar IPC boundary); a list
  literal moves to the dedicated ``scalar_seq`` key, which the IPC carries
  as a JSON array.
* **③ dist normalisation** — :func:`parse_dist_spec` is the single shared
  bridge between the ``shapes.yaml`` string DSL and the structured
  ``{kind, args}`` form.
"""
from __future__ import annotations

import math

import pytest
import torch

from triton_ascend_bench.expr import ExprError
from triton_ascend_bench.inputs import (
    DistSpecError,
    build_inputs,
    merge_inputs,
    parse_dist_spec,
    split_inputs,
    unparse_dist_spec,
)


def _shapes_doc(inputs: dict, knobs: dict | None = None) -> dict:
    """A minimal one-row shapes.yaml doc wrapping the given inputs block."""
    return {
        "size_knobs": {"B": 2, "L": 8, "D": 16},
        "public_shapes": [{"id": "P1", "name": "tiny", "knobs": knobs or {}}],
        "hidden_shapes": [],
        "inputs": inputs,
    }


# ==========================================================================
# WA.3 ① — derived scalars
# ==========================================================================

def test_derive_scalar_d64_gives_eighth():
    """`1/sqrt(D)` with `D=64` is exactly 0.125 (the old hardcoded value)."""
    doc = _shapes_doc({"scale": {"derive": "1/sqrt(D)"}}, knobs={"D": 64})
    out = build_inputs(doc, "P1", seed=0)
    assert out["scale"] == pytest.approx(0.125)


def test_derive_scalar_d128_tracks_knob():
    """`1/sqrt(D)` with `D=128` is `1/√128` — the derive form tracks the knob."""
    doc = _shapes_doc({"scale": {"derive": "1/sqrt(D)"}}, knobs={"D": 128})
    out = build_inputs(doc, "P1", seed=0)
    assert out["scale"] == pytest.approx(1.0 / math.sqrt(128))


def test_derive_scalar_returns_float():
    doc = _shapes_doc({"scale": {"derive": "1/sqrt(D)"}}, knobs={"D": 16})
    out = build_inputs(doc, "P1", seed=0)
    assert isinstance(out["scale"], float)
    assert out["scale"] == pytest.approx(0.25)  # 1/√16


def test_derive_scalar_helpers_available():
    """Derive expressions may use the numeric helpers (`max` / `ceil` / …)."""
    doc = _shapes_doc({"v": {"derive": "max(D, 32) / 2"}}, knobs={"D": 16})
    out = build_inputs(doc, "P1", seed=0)
    assert out["v"] == pytest.approx(16.0)


def test_derive_scalar_rejects_unsafe_expression():
    """A derive rule is an `eval_numeric` expression — arbitrary code fails."""
    doc = _shapes_doc({"x": {"derive": "__import__('os')"}})
    with pytest.raises(ExprError):
        build_inputs(doc, "P1", seed=0)


# ==========================================================================
# WA.3 ② — `scalar` / `derive` must be JSON scalars
# ==========================================================================

def test_scalar_list_value_rejected():
    """A list under `scalar` is rejected — it must use `scalar_seq`."""
    doc = _shapes_doc({"block_size": {"scalar": [8, 8]}})
    with pytest.raises(TypeError, match="scalar_seq"):
        build_inputs(doc, "P1", seed=0)


def test_scalar_dict_value_rejected():
    doc = _shapes_doc({"bad": {"scalar": {"nested": 1}}})
    with pytest.raises(TypeError, match="JSON scalar"):
        build_inputs(doc, "P1", seed=0)


@pytest.mark.parametrize("value", [True, 3, 1.5, "fp16", None])
def test_scalar_json_scalars_pass_through(value):
    doc = _shapes_doc({"x": {"scalar": value}})
    out = build_inputs(doc, "P1", seed=0)
    assert out["x"] == value or (value is None and out["x"] is None)


def test_derive_non_numeric_result_rejected():
    """A derive rule must yield a JSON scalar — a string expr is not numeric."""
    # `eval_numeric` itself rejects a non-numeric result; surfaced as ExprError.
    doc = _shapes_doc({"x": {"derive": "[1, 2]"}})
    with pytest.raises(ExprError):
        build_inputs(doc, "P1", seed=0)


# ==========================================================================
# WA.3 ② — `scalar_seq` materialisation
# ==========================================================================

def test_scalar_seq_materialises_as_list():
    doc = _shapes_doc({"mrope_section": {"scalar_seq": [3, 3, 2]}})
    out = build_inputs(doc, "P1", seed=0)
    assert out["mrope_section"] == [3, 3, 2]
    assert isinstance(out["mrope_section"], list)


def test_scalar_seq_must_be_a_list():
    doc = _shapes_doc({"bad": {"scalar_seq": 8}})
    with pytest.raises(TypeError, match="must be a list"):
        build_inputs(doc, "P1", seed=0)


def test_scalar_seq_elements_must_be_json_scalars():
    doc = _shapes_doc({"bad": {"scalar_seq": [8, {"x": 1}]}})
    with pytest.raises(TypeError, match="JSON scalar"):
        build_inputs(doc, "P1", seed=0)


def test_scalar_seq_ipc_round_trip():
    """A `scalar_seq` list survives the W1.0 sidecar split/merge round trip."""
    built = build_inputs(
        _shapes_doc({
            "block_size": {"scalar_seq": [8, 8]},
            "x": {"dims": ["D"], "dtype": "float32", "dist": "zeros"},
        }),
        "P1",
        seed=0,
    )
    tensors, sidecar = split_inputs(built)
    # The list went to the JSON sidecar, the tensor to the safetensors side.
    assert "block_size" in sidecar
    assert "x" in tensors
    assert sidecar["block_size"] == [8, 8]
    merged = merge_inputs(tensors, sidecar)
    assert merged["block_size"] == [8, 8]
    assert torch.is_tensor(merged["x"])


def test_split_inputs_rejects_non_json_list_element():
    """A list with a non-JSON element cannot cross the IPC."""
    with pytest.raises(TypeError, match="JSON scalar"):
        split_inputs({"bad": [1, object()]})


def test_split_inputs_rejects_unknown_object():
    with pytest.raises(TypeError, match="cross the IPC"):
        split_inputs({"bad": object()})


def test_derived_scalar_ipc_round_trip():
    """A derived float survives the sidecar round trip as a JSON scalar."""
    built = build_inputs(
        _shapes_doc({"scale": {"derive": "1/sqrt(D)"}}, knobs={"D": 64}),
        "P1",
        seed=0,
    )
    tensors, sidecar = split_inputs(built)
    assert sidecar["scale"] == pytest.approx(0.125)
    merged = merge_inputs(tensors, sidecar)
    assert merged["scale"] == pytest.approx(0.125)


# ==========================================================================
# WA.3 ③ — `parse_dist_spec` normaliser
# ==========================================================================

@pytest.mark.parametrize("kind", [
    "randn", "rand", "ones", "zeros", "eye", "tril", "softmax_rows", "arange",
])
def test_parse_dist_nullary_kinds(kind):
    assert parse_dist_spec(kind) == {"kind": kind, "args": {}}


def test_parse_dist_nullary_rejects_args():
    with pytest.raises(DistSpecError, match="takes no arguments"):
        parse_dist_spec("randn:3")


def test_parse_dist_randint():
    assert parse_dist_spec("randint:0:E") == {
        "kind": "randint", "args": {"lo": "0", "hi": "E"}}


def test_parse_dist_randint_malformed():
    with pytest.raises(DistSpecError, match="randint"):
        parse_dist_spec("randint:0")


def test_parse_dist_offsets_to():
    assert parse_dist_spec("offsets:to=B*L") == {
        "kind": "offsets", "args": {"to": "B*L"}}


def test_parse_dist_offsets_step():
    assert parse_dist_spec("offsets:2") == {
        "kind": "offsets", "args": {"step": "2"}}


def test_parse_dist_offsets_bare_defaults_step():
    assert parse_dist_spec("offsets") == {
        "kind": "offsets", "args": {"step": "3"}}


def test_parse_dist_offsets_to_empty_rejected():
    with pytest.raises(DistSpecError, match="needs an expression"):
        parse_dist_spec("offsets:to=")


def test_parse_dist_cumcount_to():
    assert parse_dist_spec("cumcount:to=E*2") == {
        "kind": "cumcount", "args": {"to": "E*2"}}


def test_parse_dist_cumcount_needs_to():
    with pytest.raises(DistSpecError, match="cumcount:to="):
        parse_dist_spec("cumcount:4")


def test_parse_dist_cumcount_to_empty_rejected():
    with pytest.raises(DistSpecError, match="needs an expression"):
        parse_dist_spec("cumcount:to=")


def test_parse_dist_structured_cumcount_bad_args_rejected():
    with pytest.raises(DistSpecError, match="cumcount"):
        parse_dist_spec({"kind": "cumcount", "args": {"step": "2"}})


def test_parse_dist_full():
    assert parse_dist_spec("full:L") == {
        "kind": "full", "args": {"value": "L"}}


def test_parse_dist_full_empty_rejected():
    with pytest.raises(DistSpecError, match="needs 'full:"):
        parse_dist_spec("full:")


def test_parse_dist_unknown_kind_rejected():
    with pytest.raises(DistSpecError, match="unknown kind"):
        parse_dist_spec("gaussian")


def test_parse_dist_empty_string_rejected():
    with pytest.raises(DistSpecError, match="empty"):
        parse_dist_spec("")


def test_parse_dist_non_string_rejected():
    with pytest.raises(DistSpecError, match="string DSL"):
        parse_dist_spec(42)


def test_parse_dist_structured_passthrough():
    """An already-structured {kind, args} spec is validated and canonicalised."""
    spec = {"kind": "randint", "args": {"hi": "L", "lo": "0"}}
    assert parse_dist_spec(spec) == {
        "kind": "randint", "args": {"lo": "0", "hi": "L"}}


def test_parse_dist_structured_unknown_kind_rejected():
    with pytest.raises(DistSpecError, match="not in"):
        parse_dist_spec({"kind": "gaussian"})


def test_parse_dist_structured_unknown_keys_rejected():
    with pytest.raises(DistSpecError, match="unknown keys"):
        parse_dist_spec({"kind": "randn", "extra": 1})


def test_parse_dist_structured_nullary_with_args_rejected():
    with pytest.raises(DistSpecError, match="takes no args"):
        parse_dist_spec({"kind": "randn", "args": {"x": 1}})


def test_parse_dist_structured_randint_bad_args_rejected():
    with pytest.raises(DistSpecError, match="randint"):
        parse_dist_spec({"kind": "randint", "args": {"lo": "0"}})


# ==========================================================================
# WA.3 ③ — round-trip: DSL ⇄ {kind, args}
# ==========================================================================

@pytest.mark.parametrize("dsl", [
    "randn", "rand", "ones", "zeros", "eye", "tril", "softmax_rows", "arange",
    "randint:0:E", "randint:1:L", "offsets:to=B*L", "offsets:2", "full:L",
    "full:D // 4", "cumcount:to=E*2",
])
def test_dist_round_trip(dsl):
    """`parse_dist_spec(unparse_dist_spec(parse_dist_spec(x)))` is stable."""
    canon = parse_dist_spec(dsl)
    assert unparse_dist_spec(canon) == dsl
    assert parse_dist_spec(unparse_dist_spec(canon)) == canon


def test_unparse_offsets_bare_step():
    # A bare `offsets` normalises to step 3, so unparse yields `offsets:3`.
    assert unparse_dist_spec(parse_dist_spec("offsets")) == "offsets:3"


# ==========================================================================
# WA.3 ③ — `build_inputs` materialises via the normalised dist
# ==========================================================================

def test_build_inputs_dist_kinds_materialise():
    """Each dist kind still materialises after the parse_dist_spec rework."""
    doc = _shapes_doc({
        "a": {"dims": ["D"], "dtype": "float32", "dist": "randn"},
        "b": {"dims": ["D"], "dtype": "float32", "dist": "ones"},
        "c": {"dims": ["D"], "dtype": "int64", "dist": "randint:0:L"},
        "d": {"dims": ["B"], "dtype": "int64", "dist": "offsets:to=L*D"},
        "e": {"dims": ["B"], "dtype": "int64", "dist": "full:D // 4"},
    })
    out = build_inputs(doc, "P1", seed=0)
    assert out["a"].shape == (16,)
    assert torch.all(out["b"] == 1)
    assert out["c"].max().item() < 8
    assert out["d"][0].item() == 0 and out["d"][-1].item() == 128
    assert torch.all(out["e"] == 4)


def test_build_inputs_cumcount_non_leading_zero():
    """`cumcount:to=` yields a non-leading-zero cumulative count vector.

    Distinct from `offsets`: element `[-1]` equals the `to` total, element
    `[0]` is the first segment's count (not a leading zero), and the vector
    is monotonic non-decreasing — the speculative-decode rejection-sampler
    convention `starts = cat([zeros(1), cu[:-1]])`, `end = cu[r]`.
    """
    doc = _shapes_doc({
        "cu": {"dims": ["B"], "dtype": "int64", "dist": "cumcount:to=L*D"},
    }, knobs={"B": 4, "L": 8, "D": 16})
    out = build_inputs(doc, "P1", seed=0)
    cu = out["cu"]
    assert cu.shape == (4,)
    assert cu[-1].item() == 128
    assert cu[0].item() > 0  # not a leading zero — contrast `offsets`
    assert torch.all(cu[1:] >= cu[:-1])  # monotonic non-decreasing


def test_build_inputs_structured_dist_materialises():
    """A structured {kind, args} dist materialises identically to the DSL."""
    doc = _shapes_doc({
        "idx": {"dims": ["B"], "dtype": "int64",
                "dist": {"kind": "randint", "args": {"lo": "0", "hi": "L"}}},
    })
    out = build_inputs(doc, "P1", seed=0)
    assert out["idx"].min().item() >= 0
    assert out["idx"].max().item() < 8


def test_build_inputs_unknown_dist_fails_loud():
    """An unknown dist kind raises — no silent fallback to `randn` (WA.3 ③)."""
    doc = _shapes_doc({"x": {"dims": ["D"], "dtype": "float32",
                             "dist": "gaussian"}})
    with pytest.raises(DistSpecError, match="unknown kind"):
        build_inputs(doc, "P1", seed=0)
