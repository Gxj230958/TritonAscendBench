"""Phase 1 U3: explicit-view annotation must be manifest-neutral + structured; dead-knob report.

The annotation adds a top-level `explicit_view` (not a comment, so it survives a YAML re-dump) and
MUST NOT change `shape_manifest_sha256` (no re-grade). The dead-knob report reuses the dist-aware
active-knob detector.
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))

from ascendbench.corpus.shape_manifest import shape_manifest_sha256  # noqa: E402

ann = importlib.import_module("scripts.corpus.annotate_explicit_view")
deadrep = importlib.import_module("scripts.corpus.report_dead_knobs")


def _doc():
    return {
        "size_knobs": {"E": 2, "L": 4, "H": 2, "D": 8, "Z": 1},   # Z is dead
        "public_shapes": [{"id": "P1", "name": "tiny", "knobs": {}},
                          {"id": "P2", "name": "edge_all_max", "knobs": {"L": 8}}],
        "hidden_shapes": [{"id": "H1", "name": "h", "knobs": {}}],
        "inputs": {
            "q": {"dims": ["E*L", "H", "D"], "dtype": "bfloat16", "dist": "randn"},
            "scale": {"derive": "1/sqrt(D)"},
            "causal": {"scalar": False},
        },
    }


def test_annotation_is_manifest_neutral():
    doc = _doc()
    before = shape_manifest_sha256(doc)
    pub = ["P1", "P2"]
    rows = {r["id"]: r for r in doc["public_shapes"] + doc["hidden_shapes"]}
    view = {sid: ann._view_entry(doc, sid, rows[sid]) for sid in [*pub, "H1"]}
    new_doc = dict(doc)
    new_doc["explicit_view"] = view
    assert shape_manifest_sha256(new_doc) == before     # adding the view does not change the hash


def test_annotation_survives_yaml_redump():
    # a structured field (not a comment) round-trips through the YAML rewrite path
    doc = _doc()
    new_doc = dict(doc)
    new_doc["explicit_view"] = {"P1": ann._view_entry(doc, "P1", doc["public_shapes"][0])}
    reloaded = yaml.safe_load(yaml.safe_dump(new_doc, sort_keys=False))
    assert "explicit_view" in reloaded and "P1" in reloaded["explicit_view"]
    assert reloaded["explicit_view"]["P1"]["tensors"]["q"]["shape"] == [8, 2, 8]   # E*L=2*4


def test_view_entry_structure_and_provenance():
    doc = _doc()
    e = ann._view_entry(doc, "P1", doc["public_shapes"][0])
    assert e["provenance"] == "baseline"                       # name "tiny"
    assert e["tensors"]["q"] == {"shape": [8, 2, 8], "dtype": "bfloat16",
                                 "axes": ["E*L", "H", "D"], "dist": "randn"}
    assert e["params"]["causal"] is False
    assert abs(e["params"]["scale"] - (1 / (8 ** 0.5))) < 1e-9  # derive resolved
    e2 = ann._view_entry(doc, "P2", doc["public_shapes"][1])
    assert e2["provenance"] == "edge"                          # name "edge_all_max"


def test_dead_knobs_excludes_dist_only_and_active():
    doc = _doc()
    # active: E,L (E*L), H, D ; dead: Z
    assert deadrep.dead_knobs(doc) == ["Z"]
