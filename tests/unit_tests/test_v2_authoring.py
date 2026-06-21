"""U10: the compact-v2 authoring helpers (load_v2 / write_v2 / v2_to_knob_spec).

The load-bearing invariant is LOSSLESS reconstruction: generate_compact_shapes(v2_to_knob_spec(v2)) == v2 for
every committed task. It lets the knob-shaped candidate-gen / build_inputs logic in extend run unchanged, and
it is the basis of the corpus self-consistency check (test_compact_shapes).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))

from ascendbench.input_spec.compact_shapes import generate_compact_shapes  # noqa: E402
from ascendbench.input_spec.shape_format import ShapeFormatError  # noqa: E402
from scripts.corpus.migrate_to_v2 import to_yaml  # noqa: E402
from scripts.corpus.v2_authoring import load_v2, v2_to_knob_spec, write_v2  # noqa: E402

_TASKS = sorted((REPO / "tasks").glob("level*/*/shapes.yaml"))


def test_v2_to_knob_spec_lossless_all_101():
    # the reconstruction reproduces every committed v2 EXACTLY (so re-materialising is a no-op on a clean file).
    mism, processed = [], 0
    for sp in _TASKS:
        v2 = load_v2(sp)
        if generate_compact_shapes(v2_to_knob_spec(v2)) != v2:
            mism.append(sp.parent.name)
        processed += 1
    assert mism == [], mism
    assert processed >= 100, processed


def test_struct_input_reconstruction_fails_loud():
    bad = {"schema_version": 2, "public": [
        {"id": "P1", "env": {"B": 4}, "inputs": [
            {"name": "s", "kind": "struct", "struct": {"x": 1}}]}], "hidden": []}
    with pytest.raises(NotImplementedError, match="struct"):
        v2_to_knob_spec(bad)


def test_nonuniform_input_set_fails_loud():
    # row P2 has an extra input not in P1 -> not a single-template (knob) task -> reconstruction must refuse.
    t = {"name": "x", "kind": "tensor", "shape": [4], "dtype": "float32", "axes": ["B"], "dist": "randn"}
    bad = {"schema_version": 2, "public": [
        {"id": "P1", "env": {"B": 4}, "inputs": [dict(t)]},
        {"id": "P2", "env": {"B": 8}, "inputs": [dict(t), {**t, "name": "y"}]}], "hidden": []}
    with pytest.raises(ValueError, match="differs from row 0"):
        v2_to_knob_spec(bad)


def test_load_v2_rejects_knob_and_malformed(tmp_path):
    knob = tmp_path / "knob.yaml"
    knob.write_text(yaml.safe_dump({"size_knobs": {"B": 2}, "public_shapes": [{"id": "P1", "knobs": {}}]}))
    with pytest.raises(ValueError, match="not a compact v2"):
        load_v2(knob)
    bad_v2 = tmp_path / "bad.yaml"   # schema 2 but a tensor missing shape -> ShapeFormatError
    bad_v2.write_text(yaml.safe_dump({"schema_version": 2, "public": [
        {"id": "P1", "env": {"B": 4}, "inputs": [{"name": "x", "kind": "tensor", "dtype": "float32",
                                                  "axes": ["B"], "dist": "randn"}]}]}))
    with pytest.raises(ShapeFormatError):
        load_v2(bad_v2)


@pytest.mark.parametrize("bad_input", [
    {"name": "d", "kind": "derive", "value": 4},                       # derive missing expr
    {"name": "s", "kind": "scalar_seq", "value": 4},                   # scalar_seq non-list value
    {"name": "o", "kind": "optional", "value": {"oops": 1}},           # optional with a dict value
    {"name": "k", "kind": "scalar_knob", "value": 4},                  # scalar_knob missing expr
    {"name": "u", "kind": "bogus_kind", "value": 1},                   # unknown kind
    {"name": "c", "kind": "scalar", "value": [1, 2]},                  # scalar with a list value
    {"name": "q", "kind": "scalar_seq", "value": [{"oops": 1}]},       # scalar_seq with a non-scalar element
    {"name": "o2", "kind": "optional", "value": [{"oops": 1}]},        # optional list-of-dict (JSON-unsafe)
    {"name": "d2", "kind": "derive", "expr": "B", "value": [1]},       # derive with a non-scalar value
])
def test_load_v2_rejects_malformed_nontensor(tmp_path, bad_input):
    p = tmp_path / "shapes.yaml"
    p.write_text(yaml.safe_dump({"schema_version": 2, "public": [
        {"id": "P1", "env": {"B": 4}, "inputs": [
            {"name": "x", "kind": "tensor", "shape": [4], "dtype": "float32", "axes": ["B"], "dist": "randn"},
            bad_input]}]}))
    with pytest.raises(ValueError):
        load_v2(p)


def test_write_v2_is_canonical(tmp_path):
    # write_v2 emits the canonical writer text; a committed file written through it is byte-identical.
    v2 = load_v2(_TASKS[0])
    out = tmp_path / "shapes.yaml"
    write_v2(v2, out)
    assert out.read_text() == to_yaml(v2)
    assert load_v2(out) == v2          # re-parses to the same doc
