"""Phase 2 U5: the typed RowSpec IR + handler registries (the extension boundary).

The IR must (a) host all six input forms via the U2 materializer bridge, (b) populate its v1 dtype/dist
registries from the existing inputs.py tables so the known set matches build_inputs, (c) reject an
unknown dtype/dist loudly, and (d) admit a new dtype/dist by REGISTERING a handler (extension without
editing consumers).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))

from ascendbench.input_spec import row_ir  # noqa: E402
from ascendbench.input_spec.materialize import materialize_row  # noqa: E402


def test_registries_bootstrap_from_inputs():
    row_ir._bootstrap_registries()
    # v1 sets EXACTLY match the inputs.py authorities (no drift, no hand-copied list)
    from ascendbench.input_spec.inputs import _DTYPES, DIST_KINDS
    assert set(row_ir.DTYPE_REGISTRY) == set(_DTYPES)
    assert set(row_ir.DIST_REGISTRY) == set(DIST_KINDS)


def test_validate_rejects_bad_role_and_provenance_and_param():
    good_t = {"q": row_ir.TensorSpec([2], "float32", ["X"], {"kind": "randn"})}
    import pytest as _pt
    with _pt.raises(ValueError, match="bad role"):
        row_ir.RowSpec("P1", "nope", "baseline", tensors=dict(good_t)).validate()
    with _pt.raises(ValueError, match="bad provenance"):
        row_ir.RowSpec("P1", "public", "made-up", tensors=dict(good_t)).validate()
    with _pt.raises(ValueError, match="bad kind"):
        row_ir.RowSpec("P1", "public", "baseline", tensors=dict(good_t),
                       params={"s": row_ir.ParamSpec(kind="weird")}).validate()


def test_row_from_materialized_hosts_all_forms():
    doc = yaml.safe_load((REPO / "tasks/level3/L3_13_kernel_unified_attention_3d/shapes.yaml").read_text())
    desc = materialize_row(doc, doc["public_shapes"][0]["id"])
    row = row_ir.row_from_materialized("P1", "public", "baseline", desc)
    row.validate()                                  # all tensors have known dtype + dist, rank-consistent
    assert row.tensors["q"].axes == ["E*L", "H", "D"]
    # optional (null) q_descale lands as a ParamSpec(kind=optional), not a tensor
    if "q_descale" in row.params:
        assert row.params["q_descale"].kind == "optional" and row.params["q_descale"].value is None


def test_validate_rejects_unknown_dtype_and_dist():
    row_ir._bootstrap_registries()
    bad_dt = row_ir.TensorSpec(shape=[2], dtype="float9_e9", axes=["X"], dist={"kind": "randn"})
    with pytest.raises(ValueError, match="unknown dtype"):
        bad_dt.validate()
    bad_dist = row_ir.TensorSpec(shape=[2], dtype="float32", axes=["X"], dist={"kind": "no_such_dist"})
    with pytest.raises(ValueError, match="unknown dist"):
        bad_dist.validate()


def test_axes_shape_rank_mismatch_rejected():
    t = row_ir.TensorSpec(shape=[2, 3], dtype="float32", axes=["X"], dist={"kind": "randn"})
    with pytest.raises(ValueError, match="rank mismatch"):
        t.validate()


def test_register_new_dtype_and_dist_is_additive():
    row_ir.register_dtype("float4_e2m1", object(), 1)     # a hypothetical NPU dtype
    row_ir.register_dist("my_new_fill", None)
    t = row_ir.TensorSpec(shape=[2], dtype="float4_e2m1", axes=["X"], dist={"kind": "my_new_fill"})
    t.validate()                                          # now accepted -- registered, no consumer edit


def test_register_input_kind_param_wires_validation():
    # a new non-tensor input form must be admitted by RowSpec.validate too, not just INPUT_KINDS --
    # is_param=True keeps INPUT_KINDS and VALID_PARAM_KINDS in sync (no manual second edit).
    row_ir.register_input_kind("nd_block", is_param=True)
    assert "nd_block" in row_ir.INPUT_KINDS and "nd_block" in row_ir.VALID_PARAM_KINDS
    good_t = {"q": row_ir.TensorSpec([2], "float32", ["X"], {"kind": "randn"})}
    row_ir.RowSpec("P1", "public", "baseline", tensors=good_t,
                   params={"blk": row_ir.ParamSpec(kind="nd_block")}).validate()  # accepted
