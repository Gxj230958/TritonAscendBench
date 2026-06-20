"""Phase-2 U6: explicit-row generator + the FULL-input equivalence gate.

The load-bearing test is the equivalence gate: for diverse tasks, the generated explicit rows must
reproduce ``build_inputs`` BYTE-FOR-BYTE (tensor dtype+shape+values incl +inf sentinels, scalars, authored
order) across the actual seed set. This is the invariant U9's migration rides on -- any divergence is a
representation bug, not a shape to quarantine.
"""
from __future__ import annotations

import gc
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))

from ascendbench.input_spec.explicit_build import (  # noqa: E402
    assert_inputs_equal,
    build_from_explicit,
    explicit_row,
    resolve_dist_args,
)
from ascendbench.input_spec.inputs import build_inputs, shape_ids  # noqa: E402
from scripts.corpus.gen_explicit_shapes import generate_explicit_shapes  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _knob_corpus import knob_doc  # noqa: E402

# Diverse tasks: +inf-sentinel coupled dist (L1_15), int32 offsets + arange + rope + randint (L3_11),
# optional-null input (L3_13), norm / gated / cumsum / moe / elementwise / chunked-offsets.
_GATE_TASKS = [
    "tasks/level1/L1_15_merge_state",                       # randn_inf_excl (+inf), ref ordering
    "tasks/level3/L3_11_fwd_grouped_kernel_stage1_rope",    # int32 offsets, int64 arange, rope_cos_sin
    "tasks/level3/L3_13_kernel_unified_attention_3d",       # optional null q_descale
    "tasks/level2/L2_01_layer_norm_fwd",                    # norm
    "tasks/level2/L2_23_layer_norm_gated_fwd",              # gated
    "tasks/level1/L1_27_chunk_cumsum_fwd",                  # cumsum
    "tasks/level1/L1_01_bias",                              # elementwise
    "tasks/level2/L2_28_moe_align_block_size_stage2",       # moe
    "tasks/level2/L2_14_bmm_chunk_fwd",                     # offsets + chunk
    "tasks/level2/L2_29_silu_mul_static_tensorwise_quant_triton_kernel_for_cutlass_moe",  # moe quant
]
_SEEDS = [0, 1000]   # correctness seed 0 + a sampled measured-timing seed; values, not shapes, vary by seed


def _load(task: str) -> dict:
    # the corpus is migrated to v2; generate_explicit_shapes' input is the KNOB form (from git history).
    return knob_doc(task)


@pytest.mark.parametrize("task", _GATE_TASKS)
def test_explicit_rows_reproduce_build_inputs_byte_for_byte(task):
    doc = _load(task)
    pub, hid = shape_ids(doc)
    assert pub or hid, f"{task}: no shapes"
    for sid in pub + hid:
        row = explicit_row(doc, sid)
        for seed in _SEEDS:
            expected = build_inputs(doc, sid, seed)
            got = build_from_explicit(row, seed)
            assert_inputs_equal(expected, got)   # raises with the offending input named on mismatch
            del expected, got
        gc.collect()


def test_generate_explicit_shapes_structure():
    doc = _load("tasks/level3/L3_13_kernel_unified_attention_3d")
    out = generate_explicit_shapes(doc)
    assert out["schema_version"] >= 1
    assert out["public"] and all(r["role"] == "public" for r in out["public"])
    assert all(r["role"] == "hidden" for r in out["hidden"])
    # rows carry an ordered inputs list; a tensor entry has literal shape + declared dtype + axes + dist
    r0 = out["public"][0]
    assert isinstance(r0["inputs"], list)
    t = next(i for i in r0["inputs"] if i["kind"] == "tensor")
    assert all(isinstance(d, int) for d in t["shape"]) and len(t["axes"]) == len(t["shape"])


def test_resolve_dist_args_literals_and_ref():
    # numeric args resolve to literal ints against the row knobs; a ref (tensor name) is preserved verbatim
    knobs = {"E": 8, "L": 4}
    assert resolve_dist_args({"kind": "offsets", "args": {"to": "E*L"}}, knobs) == \
        {"kind": "offsets", "args": {"to": 32}}
    assert resolve_dist_args({"kind": "randint", "args": {"lo": "0", "hi": "E"}}, knobs) == \
        {"kind": "randint", "args": {"lo": 0, "hi": 8}}
    assert resolve_dist_args({"kind": "randn_inf_excl", "args": {"ref": "prefix_lse"}}, knobs) == \
        {"kind": "randn_inf_excl", "args": {"ref": "prefix_lse"}}
    assert resolve_dist_args({"kind": "randn", "args": {}}, knobs) == {"kind": "randn", "args": {}}


def test_resolve_dist_args_unknown_key_fails_loud():
    with pytest.raises(ValueError, match="unknown arg key"):
        resolve_dist_args({"kind": "weird", "args": {"bogus": "E"}}, {"E": 2})
