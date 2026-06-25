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

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _knob_corpus import knob_doc  # noqa: E402


def _old_inline_ids(doc: dict) -> tuple[list[str], list[str]]:
    pub = [s["id"] for s in doc.get("public_shapes", []) if isinstance(s, dict) and s.get("id")]
    hid = [s["id"] for s in doc.get("hidden_shapes", []) if isinstance(s, dict) and s.get("id")]
    return pub, hid


def _knob_docs():
    # the 101 real corpus tasks (excludes the v2-only tasks/_canary fixture, which has no knob baseline).
    for sp in sorted((REPO / "tasks").glob("level*/*/shapes.yaml")):
        doc = knob_doc(str(sp.parent))   # corpus is v2; this routing test compares against the KNOB ids (reconstructed from the committed v2)
        if isinstance(doc, dict) and doc.get("size_knobs"):
            yield sp.parent.name, doc


def test_shape_ids_byte_identical_to_old_inline_all_101():
    mism = []
    processed = 0
    for name, doc in _knob_docs():
        old_pub, old_hid = _old_inline_ids(doc)
        if list(public_ids(doc)) != old_pub or list(hidden_ids(doc)) != old_hid:
            mism.append(name)
        if shape_ids(doc) != (old_pub, old_hid):
            mism.append(name + ":shape_ids")
        processed += 1
    assert not mism, mism
    assert processed >= 100, processed   # rows actually checked (guards a vacuous all-skip)


def test_load_task_v2_shapespec_packing():
    # the corpus is v2 now; load_task selects rows via iter_rows, packing each as
    # ShapeSpec(id=sid, name=row.name or sid, params=row minus id/name). For a v2 row (no 'name') the name
    # falls back to the id and params carry env+inputs -- the documented migration packing (dataset.py).
    for task in ("L1_01_bias", "L1_09_quantize_k_cache_fast", "L2_01_layer_norm_fwd"):
        td = None
        for lvl in ("level1", "level2", "level3"):
            if (REPO / "tasks" / lvl / task).is_dir():
                td = REPO / "tasks" / lvl / task
                break
        if td is None:
            continue
        doc = yaml.safe_load((td / "shapes.yaml").read_text())   # the committed v2 doc
        t = load_task(task)
        exp = [(r["id"], r.get("name") or r["id"], {k: v for k, v in r.items() if k not in {"id", "name"}})
               for r in doc["public"]]
        got = [(sp.id, sp.name, sp.params) for sp in t.public_shapes]
        assert got == exp, task


def test_shape_ids_stays_fail_loud_on_idless_knob_row():
    # the KNOB path keeps the old `r["id"]` (fail-loud): a malformed row with no id must RAISE, not silently
    # drop the shape (the adapter's defensive `.get("id")` skip is reserved for the explicit path, so a
    # malformed real task fails the grade loudly instead of grading a reduced set).
    bad = {"size_knobs": {}, "public_shapes": [{"name": "x"}], "hidden_shapes": []}
    with pytest.raises(KeyError):
        shape_ids(bad)


def test_adapter_dtype_table_covers_every_inputs_alias():
    # the cost dtype_size now comes from the adapter's single _DTYPE_BYTES table; it MUST map every legal
    # inputs._DTYPES alias (e.g. `long` -> int64 -> 8) or a task declaring that dtype silently drifts to the
    # cost fallback. Pins the superset invariant so a future inputs alias can't regress dtype_size.
    from ascendbench.input_spec.inputs import _DTYPES
    from ascendbench.input_spec.shape_format import _DTYPE_BYTES, representative_dtype_size
    missing = [k for k in _DTYPES if k not in _DTYPE_BYTES]
    assert not missing, f"adapter _DTYPE_BYTES misses legal inputs dtypes: {missing}"
    doc = {"size_knobs": {"B": 4}, "public_shapes": [{"id": "P1", "knobs": {"B": 4}}],
           "inputs": {"x": {"dims": ["B"], "dtype": "long"}}}
    assert representative_dtype_size(doc) == 8


def test_explicit_doc_enumerates_without_crashing():
    # the whole point: an explicit doc routes through the same adapter and yields the same ids.
    td = REPO / "tasks" / "level1" / "L1_01_bias"
    knob = knob_doc(str(td))   # corpus is v2; build the explicit view from the KNOB (reconstructed from the committed v2)
    exp = generate_explicit_shapes(knob)
    assert shape_ids(exp) == shape_ids(knob)
    assert list(public_ids(exp)) == list(public_ids(knob))
