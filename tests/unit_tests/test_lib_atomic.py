"""Unit tests for ``automation.lib.atomic`` (Unit 1.1).

Covers:
  * atomic_write_yaml/json round-trip (happy)
  * snapshot + rollback round-trip equality
  * --delete cleans stray .orig / .rej / .bak after rollback
  * Restore of deleted-from-task-dir file via rollback
  * Atomic semantics under interrupted replace (tmp file invisible)
  * Corrupt tarball -> tarfile.ReadError, task_dir untouched
  * file_list mismatch after rollback -> raise + tmp_baseline preserved
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPTS = _REPO_ROOT / "scripts"


@pytest.fixture(autouse=True)
def _ensure_scripts_on_path():
    inserted = False
    if str(_SCRIPTS) not in sys.path:
        sys.path.insert(0, str(_SCRIPTS))
        inserted = True
    yield
    if inserted:
        try:
            sys.path.remove(str(_SCRIPTS))
        except ValueError:
            pass


def _make_task_dir(parent: Path, name: str = "L1_99_demo") -> Path:
    d = parent / name
    d.mkdir()
    (d / "meta.yaml").write_text("schema_version: 1.0\nkind: triton\n", encoding="utf-8")
    (d / "shapes.yaml").write_text("public_shapes: []\n", encoding="utf-8")
    (d / "oracle_pytorch.py").write_text("def oracle(x):\n    return x\n", encoding="utf-8")
    sub = d / "sub"
    sub.mkdir()
    (sub / "note.txt").write_text("hello\n", encoding="utf-8")
    return d


def test_atomic_write_yaml_round_trip(tmp_path):
    from automation.lib import atomic

    p = tmp_path / "out.yaml"
    payload = {"a": 1, "b": [1, 2, 3], "c": {"x": "y"}}
    atomic.atomic_write_yaml(p, payload)
    import yaml as _y
    assert _y.safe_load(p.read_text(encoding="utf-8")) == payload


def test_atomic_write_yaml_does_not_sort_keys(tmp_path):
    from automation.lib import atomic

    p = tmp_path / "ordered.yaml"
    # Insertion order is preserved by Python dict (3.7+); writer must
    # not sort.
    payload = {"z_first": 1, "a_last": 2}
    atomic.atomic_write_yaml(p, payload)
    text = p.read_text(encoding="utf-8")
    assert text.index("z_first") < text.index("a_last")


def test_atomic_write_json_sorted(tmp_path):
    from automation.lib import atomic

    p = tmp_path / "out.json"
    payload = {"b": 1, "a": 2}
    atomic.atomic_write_json(p, payload)
    text = p.read_text(encoding="utf-8")
    # sort_keys=True for JSON (stable diffs)
    assert text.index('"a"') < text.index('"b"')
    assert json.loads(text) == payload


def test_atomic_write_text_round_trip(tmp_path):
    from automation.lib import atomic

    p = tmp_path / "oracle.py"
    body = "def f(x):\n    return x * 2\n"
    atomic.atomic_write_text(p, body)
    assert p.read_text(encoding="utf-8") == body


def test_snapshot_and_rollback_byte_equal(tmp_path):
    from automation.lib import atomic

    task = _make_task_dir(tmp_path)
    tar = tmp_path / "snap" / "baseline.tar.gz"
    manifest = tmp_path / "snap" / "manifest.json"
    info = atomic.snapshot_task_baseline(task, tar, manifest)
    assert tar.is_file() and manifest.is_file()
    expected_files = sorted(info["file_list"])

    # Mutate task dir: change a file, add stray files
    (task / "meta.yaml").write_text("MUTATED\n", encoding="utf-8")
    (task / "extra.tmp").write_text("garbage\n", encoding="utf-8")
    (task / "shapes.yaml.orig").write_text("orig\n", encoding="utf-8")
    (task / "oracle_pytorch.py.rej").write_text("rej\n", encoding="utf-8")
    (task / "sub" / "note.txt.bak").write_text("bak\n", encoding="utf-8")

    atomic.rollback_task(task, tar, manifest)

    # All stray files gone, original files restored byte-equal.
    actual = []
    for root, _dirs, files in os.walk(task):
        for f in files:
            actual.append(str(Path(root, f).relative_to(task)))
    actual.sort()
    assert actual == expected_files
    assert (task / "meta.yaml").read_text(encoding="utf-8") == "schema_version: 1.0\nkind: triton\n"


def test_rollback_restores_deleted_file(tmp_path):
    from automation.lib import atomic

    task = _make_task_dir(tmp_path)
    tar = tmp_path / "baseline.tar.gz"
    manifest = tmp_path / "manifest.json"
    atomic.snapshot_task_baseline(task, tar, manifest)
    # Delete a baseline file
    (task / "shapes.yaml").unlink()
    atomic.rollback_task(task, tar, manifest)
    assert (task / "shapes.yaml").is_file()
    assert (task / "shapes.yaml").read_text(encoding="utf-8") == "public_shapes: []\n"


def test_atomic_write_yaml_interrupted_replace_leaves_old(tmp_path, monkeypatch):
    """If ``os.replace`` raises mid-write, the destination file keeps its
    old content (POSIX atomic) and the tmp file is cleaned up."""
    from automation.lib import atomic

    p = tmp_path / "x.yaml"
    atomic.atomic_write_yaml(p, {"v": 1})

    original_replace = os.replace

    def boom(src, dst):
        raise OSError("simulated interrupt before replace")

    monkeypatch.setattr(os, "replace", boom)
    with pytest.raises(OSError, match="simulated interrupt"):
        atomic.atomic_write_yaml(p, {"v": 2})
    monkeypatch.setattr(os, "replace", original_replace)
    # Old content survives
    import yaml as _y
    assert _y.safe_load(p.read_text(encoding="utf-8")) == {"v": 1}
    # No leftover tmp files
    leftovers = list(tmp_path.glob("x.yaml.tmp.*"))
    assert leftovers == []


def test_rollback_corrupt_tar_raises_readerror(tmp_path):
    import tarfile as _tar

    from automation.lib import atomic

    task = _make_task_dir(tmp_path)
    tar = tmp_path / "bad.tar.gz"
    tar.write_bytes(b"this is not a valid gzip tarball")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"file_list": []}), encoding="utf-8")
    before = sorted(p.name for p in task.iterdir())
    with pytest.raises((_tar.ReadError, OSError)):
        atomic.rollback_task(task, tar, manifest)
    # task dir untouched
    after = sorted(p.name for p in task.iterdir())
    assert before == after


def test_rollback_file_list_mismatch_preserves_tmp(tmp_path):
    """If the file_list verification fails after rsync, the helper must
    raise + leave tmp_baseline on disk for forensics."""
    from automation.lib import atomic

    task = _make_task_dir(tmp_path)
    tar = tmp_path / "baseline.tar.gz"
    manifest_path = tmp_path / "manifest.json"
    atomic.snapshot_task_baseline(task, tar, manifest_path)
    # Corrupt the manifest to claim an extra file that doesn't exist in tar
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["file_list"].append("nonexistent_extra_file.txt")
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(RuntimeError, match="file_list mismatch"):
        atomic.rollback_task(task, tar, manifest_path)
    # tmp_baseline should be preserved (we cannot easily find its
    # location here; just assert /tmp has at least one of our dirs).
    import tempfile as _tf
    leftovers = list(Path(_tf.gettempdir()).glob("tab-rollback-*"))
    assert leftovers, "tmp_baseline should be preserved for forensics"
    # Cleanup forensics manually for test hygiene
    for d in leftovers:
        import shutil as _sh
        _sh.rmtree(d, ignore_errors=True)
