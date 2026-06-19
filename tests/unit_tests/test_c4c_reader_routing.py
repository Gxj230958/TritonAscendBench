"""Phase-2 U9 c4c (C1 id-enumeration): shapes.yaml readers route id enumeration through the shape_format
adapter so an explicit-format doc enumerates too, staying byte-identical on the knob format.

The shared guarantee tested here: `public_ids(doc)+hidden_ids(doc)` (which every C1 reader now uses) equals
the old inline `[s["id"] for grp in (public_shapes,hidden_shapes) for s in doc[grp] if s.get("id")]` for all
101 knob docs, and load_task's ShapeSpec list is unchanged.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))

from ascendbench.corpus.dataset import load_task  # noqa: E402
from ascendbench.input_spec.inputs import shape_ids  # noqa: E402
from ascendbench.input_spec.shape_format import hidden_ids, public_ids  # noqa: E402
from scripts.corpus.gen_explicit_shapes import generate_explicit_shapes  # noqa: E402


def _old_inline_ids(doc: dict) -> tuple[list[str], list[str]]:
    pub = [s["id"] for s in doc.get("public_shapes", []) if isinstance(s, dict) and s.get("id")]
    hid = [s["id"] for s in doc.get("hidden_shapes", []) if isinstance(s, dict) and s.get("id")]
    return pub, hid


def _knob_docs():
    for sp in sorted((REPO / "tasks").rglob("shapes.yaml")):
        doc = yaml.safe_load(sp.read_text())
        if isinstance(doc, dict) and doc.get("size_knobs"):
            yield sp.parent.name, doc


def test_shape_ids_byte_identical_to_old_inline_all_101():
    mism = []
    for name, doc in _knob_docs():
        old_pub, old_hid = _old_inline_ids(doc)
        if list(public_ids(doc)) != old_pub or list(hidden_ids(doc)) != old_hid:
            mism.append(name)
        if shape_ids(doc) != (old_pub, old_hid):
            mism.append(name + ":shape_ids")
    assert not mism, mism


def test_load_task_shapespec_unchanged():
    # load_task now selects rows via iter_rows; for knob that is byte-identical, so the ShapeSpec id/name/params
    # must match the old `for s in doc["public_shapes"]` packing.
    for task in ("L1_01_bias", "L1_09_quantize_k_cache_fast", "L2_01_layer_norm_fwd"):
        td = None
        for lvl in ("level1", "level2", "level3"):
            if (REPO / "tasks" / lvl / task).is_dir():
                td = REPO / "tasks" / lvl / task
                break
        if td is None:
            continue
        doc = yaml.safe_load((td / "shapes.yaml").read_text())
        t = load_task(task)
        old = [(s["id"], s["name"], {k: v for k, v in s.items() if k not in {"id", "name"}})
               for s in doc["public_shapes"]]
        got = [(sp.id, sp.name, sp.params) for sp in t.public_shapes]
        assert got == old, task


def test_shape_ids_stays_fail_loud_on_idless_knob_row():
    # the KNOB path keeps the old `r["id"]` (fail-loud): a malformed row with no id must RAISE, not silently
    # drop the shape (the adapter's defensive `.get("id")` skip is reserved for the explicit path, so a
    # malformed real task fails the grade loudly instead of grading a reduced set).
    bad = {"size_knobs": {}, "public_shapes": [{"name": "x"}], "hidden_shapes": []}
    with pytest.raises(KeyError):
        shape_ids(bad)


def test_explicit_doc_enumerates_without_crashing():
    # the whole point: an explicit doc routes through the same adapter and yields the same ids.
    td = REPO / "tasks" / "level1" / "L1_01_bias"
    knob = yaml.safe_load((td / "shapes.yaml").read_text())
    exp = generate_explicit_shapes(knob)
    assert shape_ids(exp) == shape_ids(knob)
    assert list(public_ids(exp)) == list(public_ids(knob))
