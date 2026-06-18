"""Phase-2 U8 commit 1: the central dual-format shape adapter."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))

from ascendbench.input_spec import shape_format as sf  # noqa: E402
from ascendbench.input_spec.explicit_build import assert_inputs_equal  # noqa: E402
from ascendbench.input_spec.inputs import build_inputs  # noqa: E402
from ascendbench.input_spec.row_ir import SCHEMA_VERSION  # noqa: E402
from scripts.corpus.gen_explicit_shapes import generate_explicit_shapes  # noqa: E402

_KNOB_TASK = "tasks/level3/L3_13_kernel_unified_attention_3d"


def _knob_doc():
    return yaml.safe_load((REPO / _KNOB_TASK / "shapes.yaml").read_text())


# --- discriminator (strict + fail-closed) ----------------------------------------------------------

def test_all_101_knob_docs_are_not_explicit():
    # every current task is knob-format -- the discriminator must NOT false-positive (the Phase-1
    # explicit_view + provenance are nested/advisory, not a top-level explicit doc).
    for shp in sorted(REPO.glob("tasks/level*/*/shapes.yaml")):
        assert sf.is_explicit_shapes_doc(yaml.safe_load(shp.read_text())) is False, shp


def test_generated_explicit_doc_is_explicit():
    doc = generate_explicit_shapes(_knob_doc())
    assert sf.is_explicit_shapes_doc(doc) is True


def test_discriminator_fail_closed_cases():
    with pytest.raises(sf.ShapeFormatError, match=r"no schema_version"):
        sf.is_explicit_shapes_doc({"public": [{"id": "P1", "inputs": []}]})        # public, no version
    with pytest.raises(sf.ShapeFormatError, match="must be an int"):
        sf.is_explicit_shapes_doc({"schema_version": True, "public": []})           # bool version
    with pytest.raises(sf.ShapeFormatError, match="unsupported schema_version"):
        sf.is_explicit_shapes_doc({"schema_version": SCHEMA_VERSION + 99, "public": []})
    with pytest.raises(sf.ShapeFormatError, match="missing an 'inputs' list"):
        sf.is_explicit_shapes_doc({"schema_version": SCHEMA_VERSION, "public": [{"id": "P1"}]})
    with pytest.raises(sf.ShapeFormatError, match="duplicate row ids"):
        sf.is_explicit_shapes_doc({"schema_version": SCHEMA_VERSION,
                                   "public": [{"id": "P1", "inputs": []}, {"id": "P1", "inputs": []}]})
    # a malformed explicit doc whose `public` is NOT a list (and no schema_version) must fail closed,
    # not silently route to the knob path (detect by key presence, not list-typed).
    with pytest.raises(sf.ShapeFormatError, match="no schema_version"):
        sf.is_explicit_shapes_doc({"public": {"oops": "not a list"}})


# --- uniform accessors ------------------------------------------------------------------------------

def test_ids_and_env_both_formats():
    knob = _knob_doc()
    exp = generate_explicit_shapes(knob)
    assert sf.public_ids(knob) == sf.public_ids(exp)
    assert sf.hidden_ids(knob) == sf.hidden_ids(exp)
    sid = sf.public_ids(knob)[0]
    # knob env == size_knobs⊕row.knobs; explicit env == the carried row env; they must agree numerically.
    assert sf.effective_env(knob, sid) == sf.effective_env(exp, sid)


def test_representative_dtype_size_both_formats():
    knob = _knob_doc()
    exp = generate_explicit_shapes(knob)
    assert sf.representative_dtype_size(knob) == sf.representative_dtype_size(exp)
    assert sf.representative_dtype_size(knob) is not None


def test_representative_dtype_size_knob_parity_all_101():
    # the adapter's knob path must EQUAL the existing eval._representative_dtype_size for every task
    # (commits 2-4 replace the duplicated helpers with the adapter -- parity guarantees no drift).
    from ascendbench.engine.eval import _representative_dtype_size as eval_rds
    for shp in sorted(REPO.glob("tasks/level*/*/shapes.yaml")):
        doc = yaml.safe_load(shp.read_text())
        assert sf.representative_dtype_size(doc) == eval_rds(doc), shp


def test_build_inputs_for_dispatch_is_byte_faithful():
    # the adapter's explicit dispatch reproduces build_inputs byte-for-byte (the U6 gate, via the adapter)
    knob = _knob_doc()
    exp = generate_explicit_shapes(knob)
    for sid in sf.public_ids(knob)[:3]:
        assert_inputs_equal(build_inputs(knob, sid, 0), sf.build_inputs_for(exp, sid, 0))
        # and the knob path is just build_inputs
        assert_inputs_equal(build_inputs(knob, sid, 0), sf.build_inputs_for(knob, sid, 0))


# --- manifest projection (full row) -----------------------------------------------------------------

def test_explicit_projection_is_sensitive_to_semantic_edits():
    exp = generate_explicit_shapes(_knob_doc())
    base = sf.explicit_canonical_projection(exp)
    # mutating a tensor axis name in a row must change the projection (a regex-blind hash would miss it)
    import copy
    m = copy.deepcopy(exp)
    t = next(i for i in m["public"][0]["inputs"] if i["kind"] == "tensor")
    t["axes"] = [a + "_X" for a in t["axes"]]
    assert sf.explicit_canonical_projection(m) != base


# --- sandbox sealing --------------------------------------------------------------------------------

def test_sanitised_view_seals_hidden_both_formats():
    knob = {"size_knobs": {"B": 2}, "public_shapes": [{"id": "P1", "knobs": {}}],
            "hidden_shapes": [{"id": "H1", "knobs": {}}],
            "inputs": {"x": {"dims": ["B"], "dtype": "float32"}},
            "explicit_view": {"P1": "pub", "H1": "hid"}}
    v = sf.sanitised_view(knob)
    assert "hidden_shapes" not in v and v["explicit_view"] == {"P1": "pub"}
    exp = generate_explicit_shapes(_knob_doc())
    ve = sf.sanitised_view(exp)
    assert "hidden" not in ve and "public" in ve
