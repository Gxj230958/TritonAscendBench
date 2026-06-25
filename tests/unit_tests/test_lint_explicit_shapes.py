"""Phase-2 U6: the explicit-row linter (strict axis-link semantics + per-tensor validity)."""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))

from scripts.corpus.gen_explicit_shapes import generate_explicit_shapes  # noqa: E402
from scripts.corpus.lint_explicit_shapes import lint_row, lint_task  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _knob_corpus import knob_doc  # noqa: E402


def test_real_task_lints_clean():
    # a real generated task obeys the invariant by construction (axis name -> single extent per row).
    # corpus is v2 now; generate the explicit view from the KNOB (reconstructed from the committed v2).
    doc = generate_explicit_shapes(knob_doc("tasks/level3/L3_13_kernel_unified_attention_3d"))
    assert lint_task(doc) == []


def test_shared_axis_name_conflicting_extent_flagged():
    # two tensors share axis name "D" but with different extents in the same row -> rule-1 violation
    row = {"id": "P1", "inputs": [
        {"name": "q", "kind": "tensor", "shape": [4, 64], "dtype": "float32",
         "axes": ["L", "D"], "dist": {"kind": "randn", "args": {}}},
        {"name": "k", "kind": "tensor", "shape": [4, 128], "dtype": "float32",
         "axes": ["L", "D"], "dist": {"kind": "randn", "args": {}}},
    ]}
    v = lint_row(row)
    assert any("conflicting extents" in s and "'D'" in s for s in v)


def test_declared_axis_link_violation_flagged():
    # a declared cross-name link q.head_dim == k.kd must hold; here 64 != 128 -> violation
    row = {"id": "P1", "inputs": [
        {"name": "q", "kind": "tensor", "shape": [4, 64], "dtype": "float32",
         "axes": ["L", "head_dim"], "dist": {"kind": "randn", "args": {}}},
        {"name": "k", "kind": "tensor", "shape": [4, 128], "dtype": "float32",
         "axes": ["L", "kd"], "dist": {"kind": "randn", "args": {}}},
    ]}
    assert lint_row(row, axis_links=["q.head_dim == k.kd"]) == \
        ["row P1: axis_link 'q.head_dim == k.kd' violated (64 != 128)"]
    # the same link holds when extents match
    row["inputs"][1]["shape"] = [4, 64]
    assert lint_row(row, axis_links=["q.head_dim == k.kd"]) == []


def test_rank_mismatch_flagged():
    row = {"id": "P1", "inputs": [
        {"name": "q", "kind": "tensor", "shape": [4, 64], "dtype": "float32",
         "axes": ["L"], "dist": {"kind": "randn", "args": {}}},
    ]}
    assert any("rank mismatch" in s for s in lint_row(row))
