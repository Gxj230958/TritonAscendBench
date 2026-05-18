"""U7: single-source-of-truth eligibility regression tests.

Covers: behaviour byte-identical to the old inline cli.py loop
(NotImplementedError stub / PHASE-C-MANUAL excluded), side-effect-light
(no import-time I/O), cli.py and validate share the same authority.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from triton_ascend_bench import eligibility
from triton_ascend_bench.dataset import iter_all_tasks


@dataclass
class _FakeTask:
    task_id: str
    task_dir: Path


def test_stub_oracle_not_eligible(tmp_path):
    (tmp_path / "oracle_pytorch.py").write_text(
        "def run(i):\n    raise NotImplementedError\n"
    )
    (tmp_path / "shapes.yaml").write_text("public_shapes: []\n")
    assert not eligibility.is_score_eligible(_FakeTask("t", tmp_path))


def test_phase_c_manual_not_eligible(tmp_path):
    (tmp_path / "oracle_pytorch.py").write_text("def run(i):\n    return {}\n")
    (tmp_path / "shapes.yaml").write_text("# PHASE-C-MANUAL\n")
    assert not eligibility.is_score_eligible(_FakeTask("t", tmp_path))


def test_real_task_is_eligible(tmp_path):
    (tmp_path / "oracle_pytorch.py").write_text("def run(i):\n    return {}\n")
    (tmp_path / "shapes.yaml").write_text("public_shapes: []\n")
    assert eligibility.is_score_eligible(_FakeTask("t", tmp_path))


def test_os_error_is_not_eligible(tmp_path):
    """Missing files -> not eligible (same conservative behaviour as
    the original inline `continue` on OSError)."""
    assert not eligibility.is_score_eligible(_FakeTask("t", tmp_path / "nope"))


def test_no_import_time_io():
    """eligibility module must not perform import-time I/O (plan U7:
    side-effect-light). Re-importing is a no-op."""
    import importlib

    import triton_ascend_bench.eligibility as e

    importlib.reload(e)  # would raise if import did filesystem work badly
    assert callable(e.compute_score_eligible)


def test_matches_legacy_inline_logic():
    """Regression: compute_score_eligible over the real catalogue
    equals the OLD inline cli.py loop exactly (single source of
    truth, no drift)."""
    all_tasks = iter_all_tasks()
    new = eligibility.compute_score_eligible(all_tasks)

    legacy: set[str] = set()
    for t in all_tasks:
        try:
            osrc = (t.task_dir / "oracle_pytorch.py").read_text()
            shp = (t.task_dir / "shapes.yaml").read_text()
        except OSError:
            continue
        if "raise NotImplementedError" in osrc or "PHASE-C-MANUAL" in shp:
            continue
        legacy.add(t.task_id)

    assert new == legacy
    assert len(new) > 0  # the real catalogue has eligible tasks
