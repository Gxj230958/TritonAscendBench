"""U10: quarantine_shapes drops kernel-rejected rows from a compact-v2 shapes.yaml.

parse -> drop by id -> canonical re-emit: surviving rows stay byte-identical, an emptied section becomes ``[]``,
removing every row fails loud, and the meta shape_coverage counts follow the survivors (with `total` optional).
"""
from __future__ import annotations

import re
import shutil
import sys
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts" / "corpus"))

import quarantine_shapes as q  # noqa: E402

from scripts.corpus.migrate_to_v2 import to_yaml  # noqa: E402
from scripts.corpus.v2_authoring import load_v2  # noqa: E402

_TASK = "level1/L1_01_bias"


def _copy_task(tmp_path: Path) -> Path:
    td = tmp_path / Path(_TASK).name
    shutil.copytree(REPO / "tasks" / _TASK, td)
    return td


def test_quarantine_removes_rows_survivors_byte_identical(tmp_path):
    td = _copy_task(tmp_path)
    before = load_v2(td / "shapes.yaml")
    expected = {**before,
                "public": [r for r in before["public"] if r["id"] not in {"P2", "P3"}],
                "hidden": before["hidden"]}
    rep = q.quarantine(td, {"P2", "P3"}, dry_run=False)
    assert rep["removed"] == ["P2", "P3"] and rep["not_found"] == []
    assert rep["public_after"] == len(before["public"]) - 2
    # the surviving rows re-emit byte-for-byte (deterministic writer) + the file is canonical.
    assert (td / "shapes.yaml").read_text() == to_yaml(expected)
    assert load_v2(td / "shapes.yaml") == expected


def test_quarantine_reports_not_found(tmp_path):
    td = _copy_task(tmp_path)
    rep = q.quarantine(td, {"P2", "ZZ_absent"}, dry_run=True)
    assert "P2" in rep["removed"] and rep["not_found"] == ["ZZ_absent"]


def test_quarantine_empty_section_becomes_list(tmp_path):
    td = _copy_task(tmp_path)
    hids = {r["id"] for r in load_v2(td / "shapes.yaml")["hidden"]}
    assert hids
    q.quarantine(td, hids, dry_run=False)
    after = load_v2(td / "shapes.yaml")
    assert after["hidden"] == [] and len(after["public"]) > 0   # emptied section is [], not a dangling key


def test_quarantine_zero_total_fails_loud(tmp_path):
    td = _copy_task(tmp_path)
    doc = load_v2(td / "shapes.yaml")
    all_ids = {r["id"] for r in doc["public"] + doc["hidden"]}
    with pytest.raises(RuntimeError, match="ZERO shapes"):
        q.quarantine(td, all_ids, dry_run=True)


def test_quarantine_meta_coverage_follows_survivors(tmp_path):
    td = _copy_task(tmp_path)
    q.quarantine(td, {"P2", "P3"}, dry_run=False)
    meta = yaml.safe_load((td / "meta.yaml").read_text())
    after = load_v2(td / "shapes.yaml")
    # shape_coverage may be nested; find it wherever it lives and check it matches the survivors.
    def _find(d):
        if isinstance(d, dict):
            if "shape_coverage" in d:
                return d["shape_coverage"]
            for v in d.values():
                r = _find(v)
                if r is not None:
                    return r
        return None
    cov = _find(meta)
    assert cov["public"] == len(after["public"]) and cov["hidden"] == len(after["hidden"])
    assert cov["total"] == len(after["public"]) + len(after["hidden"])


def test_rewrite_coverage_total_optional():
    with_total = "root:\n  shape_coverage:\n    public: 5\n    hidden: 3\n    total: 8\n  other: 1\n"
    new, ok = q._rewrite_coverage(with_total, 4, 2)
    assert ok and "public: 4" in new and "hidden: 2" in new and "total: 6" in new
    no_total = "root:\n  shape_coverage:\n    public: 5\n    hidden: 3\n  other: 1\n"
    new2, ok2 = q._rewrite_coverage(no_total, 4, 2)
    # a missing total is NOT synthesised (avoids a duplicate when an existing total is comment-separated);
    # public/hidden are updated, total stays absent (it is optional), and the rest of the meta is intact.
    assert ok2 and "public: 4" in new2 and "hidden: 2" in new2 and "total" not in new2
    assert yaml.safe_load(new2)["root"]["other"] == 1


def test_quarantine_fails_loud_on_inconsistent_coverage(tmp_path):
    # a total separated from hidden by a comment is NOT updated by the regex -> the post-verify must catch the
    # now-stale total and refuse the quarantine (rather than write shapes.yaml against a wrong meta).
    td = _copy_task(tmp_path)
    meta_text = (td / "meta.yaml").read_text()
    stale = re.sub(r"(hidden:\s*\d+\n)(\s+)(total:)", r"\1\2# a stray comment\n\2\3", meta_text, count=1)
    assert stale != meta_text
    (td / "meta.yaml").write_text(stale)
    with pytest.raises(RuntimeError, match="shape_coverage could not be consistently rewritten"):
        q.quarantine(td, {"P2"}, dry_run=False)
