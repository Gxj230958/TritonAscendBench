"""Phase 1 U2: the explicit-shape materializer must agree with build_inputs.

The materializer renders literal shapes WITHOUT allocating tensors; this test proves it matches the
tensor shapes/dtypes that build_inputs actually produces (the faithfulness contract the explicit view
and the Phase-2 cutover both depend on), across a diverse task sample including the optional/`null`
input case.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
import yaml

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))

from ascendbench.input_spec.inputs import _DTYPES, build_inputs, shape_ids  # noqa: E402
from ascendbench.input_spec.materialize import materialize_row  # noqa: E402

# diverse families: norm, attention (+null optional), moe, elementwise, lora, the 2 null-optional tasks
SAMPLE = [
    "level1/L1_01_bias",
    "level2/L2_01_layer_norm_fwd",
    "level3/L3_13_kernel_unified_attention_3d",   # q/k/v_descale: null
    "level3/L3_03_chunked_lora_expand",           # base_output: null
    "level2/L2_03_act_and_mul",
    "level3/L3_19_fused_moe",
]


def _doc(rel):
    return yaml.safe_load((REPO / "tasks" / rel / "shapes.yaml").read_text())


@pytest.mark.parametrize("rel", SAMPLE)
def test_materializer_matches_build_inputs(rel):
    doc = _doc(rel)
    pub, hid = shape_ids(doc)
    assert pub or hid, f"{rel} has no shapes"
    for sid in pub + hid:
        built = build_inputs(doc, sid, seed=0)
        desc = materialize_row(doc, sid)
        assert set(desc) == set(built), f"{rel}/{sid}: input-name set differs"
        for name, d in desc.items():
            val = built[name]
            if d["kind"] == "tensor":
                assert torch.is_tensor(val), f"{rel}/{sid}/{name}: built is not a tensor"
                # SHAPE is the hard faithfulness contract (reuses _resolve_dims).
                assert list(val.shape) == d["shape"], f"{rel}/{sid}/{name}: shape mismatch"
                # dtype matches the DECLARED dtype, OR build_inputs overrode it to an integer (the
                # documented index-dist behaviour: offsets/arange/full/randperm/*_ids -> int64). Any
                # OTHER divergence is a real bug.
                declared = _DTYPES[d["dtype"]]
                if val.dtype != declared:
                    assert val.dtype in (torch.int32, torch.int64), (
                        f"{rel}/{sid}/{name}: built dtype {val.dtype} != declared {declared} "
                        f"and is not an integer index-dist override"
                    )
            elif d["kind"] == "optional":
                assert val == d["value"]          # e.g. None
            elif d["kind"] in ("scalar", "derive", "scalar_seq", "scalar_knob"):
                assert not torch.is_tensor(val), f"{rel}/{sid}/{name}: expected non-tensor"


def test_optional_null_input_is_materialized_as_optional():
    doc = _doc("level3/L3_13_kernel_unified_attention_3d")
    pub, _ = shape_ids(doc)
    desc = materialize_row(doc, pub[0])
    for nm in ("q_descale", "k_descale", "v_descale"):
        if nm in desc:
            assert desc[nm]["kind"] == "optional" and desc[nm]["value"] is None


def test_dist_default_rendered_explicit():
    # a tensor input that omits dist must materialise dist explicit (default randn), no implicit default
    doc = {"size_knobs": {"D": 4}, "public_shapes": [{"id": "P1", "knobs": {}}],
           "inputs": {"x": {"dims": ["D"], "dtype": "float32"}}}
    desc = materialize_row(doc, "P1")
    assert desc["x"]["dist"]["kind"] == "randn"
    assert desc["x"]["shape"] == [4]
