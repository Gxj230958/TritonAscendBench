"""Phase-2 U8 commit 2: shape_manifest dual-format dispatch (manifest-neutral for knob-format)."""
from __future__ import annotations

import copy
import hashlib
import json
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))

from ascendbench.corpus.shape_manifest import canonical_manifest, shape_manifest_sha256  # noqa: E402
from ascendbench.input_spec import shape_format as sf  # noqa: E402
from scripts.corpus.gen_explicit_shapes import generate_explicit_shapes  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _knob_corpus import knob_doc  # noqa: E402

_KNOB_TASK = "tasks/level3/L3_13_kernel_unified_attention_3d"


def _knob_doc():
    # the corpus is migrated to v2; the KNOB form for these projection tests comes from git history.
    return knob_doc(_KNOB_TASK)


def _sha(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()


def test_knob_doc_uses_knob_projection_unchanged():
    # a knob doc is NOT explicit, so canonical_manifest produces the knob projection (no explicit keys)
    knob = _knob_doc()
    assert sf.is_explicit_shapes_doc(knob) is False
    m = canonical_manifest(knob)
    assert set(m) == {"inputs", "size_knobs", "shapes"}     # the knob projection shape, unchanged
    assert shape_manifest_sha256(knob) == shape_manifest_sha256(knob)   # deterministic


def test_explicit_doc_uses_full_row_projection():
    exp = generate_explicit_shapes(_knob_doc())
    m = canonical_manifest(exp)
    assert m == sf.explicit_canonical_projection(exp)       # dispatch took the explicit branch
    assert "schema_version" in m and "rows" in m
    # the hash is the sha of that full-row projection
    assert shape_manifest_sha256(exp) == _sha(sf.explicit_canonical_projection(exp))


def test_explicit_manifest_is_edit_sensitive():
    exp = generate_explicit_shapes(_knob_doc())
    base = shape_manifest_sha256(exp)
    m = copy.deepcopy(exp)
    # editing a row env value (a named symbol) must move the hash
    first_env = m["public"][0]["env"]
    k = next(iter(first_env))
    first_env[k] = first_env[k] + 1
    assert shape_manifest_sha256(m) != base


def test_manifest_all_101_corpus_explicit():
    # the corpus is migrated to v2 -> the manifest dispatch takes the EXPLICIT branch (full-row projection)
    # for every task, so each task's manifest is the explicit projection carrying schema_version + rows.
    for shp in sorted(REPO.glob("tasks/level*/*/shapes.yaml")):
        doc = yaml.safe_load(shp.read_text())
        assert sf.is_explicit_shapes_doc(doc) is True, shp
        m = canonical_manifest(doc)
        assert m == sf.explicit_canonical_projection(doc), shp
        assert "schema_version" in m and "rows" in m, shp
