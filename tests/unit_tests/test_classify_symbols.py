"""Phase 2 U4: symbol-provenance classification.

Each declared size-knob is classified into tensor-axis / scalar-input / dist-param / row-only-param /
dead so the explicit schema (U5) and the cost_model re-base (U7) know whether a symbol is a tensor
axis or a named param. `row-only-param` is the class that makes a pure tensor-axis cost basis
impossible (codex finding).
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))

cs = importlib.import_module("scripts.corpus.classify_symbols")


def test_all_five_classes():
    doc = {
        "size_knobs": {"B": 1, "L": 1, "D": 1, "E": 1, "G": 1, "S": 1, "Z": 1},
        "inputs": {
            "x": {"dims": ["B", "L"]},            # B, L -> tensor-axis
            "scale": {"derive": "1/sqrt(D)"},     # D -> scalar-input
            "idx": {"dims": ["E"], "dist": "randint:0:S"},  # E tensor-axis; S only in dist -> dist-param
        },
        "public_shapes": [{"id": "P1", "knobs": {}}],
    }
    meta = {
        "cost_model": {"flops": "2*B*L*G", "bytes": "B*L"},   # G used only in cost -> row-only-param
        "param_constraints": {"dims": ["B >= 1"]},
    }
    c = cs.classify(doc, meta)
    assert c["B"] == "tensor-axis"
    assert c["L"] == "tensor-axis"
    assert c["E"] == "tensor-axis"
    assert c["D"] == "scalar-input"
    assert c["S"] == "dist-param"        # only inside a dist expression
    assert c["G"] == "row-only-param"    # cost-only, no input home
    assert c["Z"] == "dead"              # referenced nowhere


def test_dims_precedence_over_dist():
    # a symbol in BOTH dims and dist is a tensor-axis (dims wins)
    doc = {"size_knobs": {"E": 1}, "inputs": {"x": {"dims": ["E"], "dist": "randint:0:E"}}}
    assert cs.classify(doc, {})["E"] == "tensor-axis"


def test_deep_constraint_scan_catches_invariant_and_dist_refs():
    # a knob used ONLY in param_constraints.tensor_invariants[*].args.value (or .dists) must NOT be
    # mislabeled dead -- the deep scan reaches nested value strings, not just dims.
    doc = {"size_knobs": {"B": 1, "C": 1, "T": 1}, "inputs": {"x": {"dims": ["B"]}}}
    meta = {"param_constraints": {
        "dists": {"slot_mapping": {"kind": "offsets", "args": {"to": "C*T-1"}}},   # C, T here
        "tensor_invariants": [{"type": "last_eq", "args": {"input": "slot_mapping", "value": "C*T-1"}}],
    }}
    c = cs.classify(doc, meta)
    assert c["B"] == "tensor-axis"
    assert c["C"] == "row-only-param"     # only in constraint dists/invariants, no input/cost home
    assert c["T"] == "row-only-param"


def test_field_names_not_false_knobs():
    # an `input:` value (a tensor NAME) that is NOT a knob must not be picked up; only declared knobs.
    doc = {"size_knobs": {"B": 1}, "inputs": {"x": {"dims": ["B"]}}}
    meta = {"param_constraints": {"tensor_invariants": [{"args": {"input": "slot_mapping"}}]}}
    assert cs.classify(doc, meta) == {"B": "tensor-axis"}   # slot_mapping ignored (not a knob)
