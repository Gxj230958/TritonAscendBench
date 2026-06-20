"""Phase-2 U9 migration M1: the compact v2 explicit format + expander + adapter v2-awareness.

The committed migrated shapes.yaml is the COMPACT v2 form; expand_compact(v2) is byte-identical to
generate_explicit_shapes(knob) (the round-trip gate), so every v1-aware reader and build_from_explicit work
unchanged. The adapter reads v2 NATIVELY for the torch-free readers (is_explicit/public_ids/effective_env/
representative_dtype_size/manifest) and expands it only for materialisation + the dist accessors.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
import yaml

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))

from ascendbench.input_spec import shape_format as sf  # noqa: E402
from ascendbench.input_spec.compact_shapes import (  # noqa: E402
    COMPACT_SCHEMA_VERSION,
    dist_arg_symbols,
    expand_compact,
    generate_compact_shapes,
)
from ascendbench.input_spec.inputs import build_inputs, shape_ids  # noqa: E402
from scripts.corpus.classify_symbols import classify  # noqa: E402
from scripts.corpus.gen_explicit_shapes import generate_explicit_shapes  # noqa: E402

_TASKS = ["tasks/level1/L1_01_bias", "tasks/level1/L1_21_create_flashmla_kv_indices_triton",
          "tasks/level2/L2_01_layer_norm_fwd", "tasks/level3/L3_13_kernel_unified_attention_3d"]


def _knob(task: str) -> dict:
    p = REPO / task / "shapes.yaml"
    if not p.exists():
        pytest.skip(f"{task} absent")
    return yaml.safe_load(p.read_text())


def test_dist_arg_symbols_extracts_args_only():
    assert dist_arg_symbols("randint:0:E") == {"E"}
    assert dist_arg_symbols("offsets:to=B") == {"B"}      # k=v -> v only
    assert dist_arg_symbols("full:B*N") == {"B", "N"}
    assert dist_arg_symbols("randn") == set()             # nullary, no args
    assert dist_arg_symbols("randint") == set()


@pytest.mark.parametrize("task", _TASKS)
def test_round_trip_expand_equals_v1(task):
    knob = _knob(task)
    assert expand_compact(generate_compact_shapes(knob)) == generate_explicit_shapes(knob)


@pytest.mark.parametrize("task", _TASKS)
def test_adapter_reads_v2_natively(task):
    knob = _knob(task)
    v2 = generate_compact_shapes(knob)
    assert v2["schema_version"] == COMPACT_SCHEMA_VERSION
    assert sf.is_explicit_shapes_doc(v2) is True
    assert list(sf.public_ids(v2)) == list(sf.public_ids(knob))
    assert list(sf.hidden_ids(v2)) == list(sf.hidden_ids(knob))
    assert sf.representative_dtype_size(v2) == sf.representative_dtype_size(knob)
    pub, hid = shape_ids(knob)
    for sid in pub + hid:
        assert sf.effective_env(v2, sid) == sf.effective_env(knob, sid)
    assert classify(v2, {}) == classify(knob, {})


def test_build_inputs_for_v2_byte_equal():
    knob = _knob("tasks/level1/L1_01_bias")
    v2 = generate_compact_shapes(knob)
    pub, _ = shape_ids(knob)
    for sid in pub[:3]:
        a, b = build_inputs(knob, sid, seed=0), sf.build_inputs_for(v2, sid, seed=0)
        assert a.keys() == b.keys()
        for k in a:
            assert torch.equal(a[k], b[k]) if torch.is_tensor(a[k]) else a[k] == b[k]


def test_v2_dist_accessors_synthesize():
    # the v2 row stores only the string dist; the dist accessors must SYNTHESIZE the symbolic + resolved forms.
    knob = _knob("tasks/level1/L1_21_create_flashmla_kv_indices_triton")
    v2 = generate_compact_shapes(knob)
    sid = sf.public_ids(v2)[0]
    name = next(i["name"] for i in v2["public"][0]["inputs"] if i["kind"] == "tensor")
    assert sf.explicit_row_dist(v2, sid, name) == sf.explicit_row_dist(generate_explicit_shapes(knob), sid, name)
    assert sf.explicit_row_dist_symbolic(v2, sid, name) is not None   # synthesized, not None


def test_v2_malformed_tensor_missing_string_dist_fails_closed():
    # shape+dtype present so the dist check (not the earlier shape/dtype guards) is what fires.
    bad = {"schema_version": COMPACT_SCHEMA_VERSION, "public": [
        {"id": "P1", "inputs": [{"name": "x", "kind": "tensor", "shape": [4], "dtype": "float32", "axes": ["N"]}]}]}
    with pytest.raises(sf.ShapeFormatError, match="string 'dist'"):
        sf.is_explicit_shapes_doc(bad)


def _v2_tensor(**over):
    t = {"name": "x", "kind": "tensor", "shape": [4], "dtype": "float32", "axes": ["N"], "dist": "randn"}
    t.update(over)
    return {"schema_version": COMPACT_SCHEMA_VERSION, "public": [{"id": "P1", "env": {"N": 4}, "inputs": [t]}]}


def test_v2_rejects_mixed_knob_keys():
    # a versioned doc that ALSO carries knob keys (a botched migration leaving hidden_shapes) must fail closed,
    # else the leftover hidden rows escape the sandbox sealer + the manifest.
    bad = dict(_v2_tensor())
    bad["size_knobs"] = {"N": 4}
    with pytest.raises(sf.ShapeFormatError, match="knob-format keys"):
        sf.is_explicit_shapes_doc(bad)


def test_v2_rejects_committed_dist_symbolic():
    with pytest.raises(sf.ShapeFormatError, match="must NOT commit 'dist_symbolic'"):
        sf.is_explicit_shapes_doc(_v2_tensor(dist_symbolic={"kind": "randn", "args": {}}))


def test_v2_requires_shape_and_dtype():
    bad = _v2_tensor()
    del bad["public"][0]["inputs"][0]["shape"]
    with pytest.raises(sf.ShapeFormatError, match="missing a 'shape' list"):
        sf.is_explicit_shapes_doc(bad)


def test_corpus_wide_round_trip():
    mism = []
    for sp in sorted((REPO / "tasks").rglob("shapes.yaml")):
        knob = yaml.safe_load(sp.read_text())
        if not isinstance(knob, dict) or not knob.get("size_knobs"):
            continue
        if expand_compact(generate_compact_shapes(knob)) != generate_explicit_shapes(knob):
            mism.append(sp.parent.name)
    assert not mism, mism
