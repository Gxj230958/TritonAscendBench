"""Unit tests for V9 plan M0a.1 -- source_provenance_census.

Covers:

* per-category R17 risk lookup (HIGH for elementwise/norm/gemm/sampling)
* tolerance_source classification (V9 provenance.golden vs corpus
  fallback vs absent)
* GPU/platform skip pattern detection in kernel text
* commit_sha verifiable routing (delegates to verify_upstream_commit_chain)
* family taxonomy: norm and normalization are disjoint
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from scripts.provenance.census_source_provenance import (
    _CATEGORY_RISK,
    _classify_tolerance_source,
    _gpu_skip_in_text,
    _risk_for_category,
    census_task,
)


# ─────────────────────────────────────────────────────────────────────
# R17 reconstructability risk
# ─────────────────────────────────────────────────────────────────────


def test_high_risk_categories_are_correctly_set():
    """V9 plan: elementwise/norm/gemm/sampling -> HIGH (math closed-form)."""
    for cat in ("elementwise", "norm", "gemm", "sampling"):
        assert _CATEGORY_RISK[cat] == "high"


def test_normalization_is_distinct_from_norm():
    """V9 plan decision: norm and normalization are DISTINCT categories.

    norm = HIGH (closed-form math), normalization = MEDIUM (layout-
    dependent). This is critical for M0a.2 R13c rare-class gate;
    merging them would mis-count rare-class coverage."""
    assert _CATEGORY_RISK["norm"] == "high"
    assert _CATEGORY_RISK["normalization"] == "medium"
    assert _CATEGORY_RISK["norm"] != _CATEGORY_RISK["normalization"]


def test_unknown_category_returns_unknown_risk():
    assert _risk_for_category("not_a_real_category") == "unknown"
    assert _risk_for_category(None) == "unknown"
    assert _risk_for_category("") == "unknown"


# ─────────────────────────────────────────────────────────────────────
# Tolerance source classification
# ─────────────────────────────────────────────────────────────────────


def test_tolerance_source_from_v9_provenance_golden():
    """V9 nested provenance.golden.tolerance_source takes precedence."""
    meta = {
        "tolerance_class": "elementwise",
        "provenance": {
            "golden": {"tolerance_source": "upstream_test_explicit"}
        },
    }
    assert _classify_tolerance_source(meta) == "upstream_test_explicit"


def test_tolerance_source_corpus_only_when_no_provenance():
    """Legacy meta with only tolerance_class -> corpus_only."""
    meta = {"tolerance_class": "elementwise"}
    assert _classify_tolerance_source(meta) == "corpus_only"


def test_tolerance_source_absent_when_nothing_declared():
    meta: dict = {}
    assert _classify_tolerance_source(meta) == "absent"


# ─────────────────────────────────────────────────────────────────────
# GPU / platform skip patterns
# ─────────────────────────────────────────────────────────────────────


def test_gpu_skip_detects_pytest_gpu_mark():
    src = "@pytest.mark.gpu\ndef test_x(): ..."
    hits = _gpu_skip_in_text(src)
    assert hits, "should detect @pytest.mark.gpu"


def test_gpu_skip_detects_torch_distributed_import():
    """Pattern strings escape the dot; assert the pattern's substring."""
    src = "from torch.distributed import barrier\n"
    hits = _gpu_skip_in_text(src)
    # hit strings are the regex *patterns*, which escape the dot:
    assert any(r"torch\.distributed" in h for h in hits)


def test_gpu_skip_detects_skipif_cuda_available():
    src = '@pytest.mark.skipif(not torch.cuda.is_available(), reason="gpu only")\n'
    hits = _gpu_skip_in_text(src)
    assert hits


def test_gpu_skip_returns_empty_on_clean_kernel():
    src = "def kernel(*args): return args\n"
    assert _gpu_skip_in_text(src) == []


# ─────────────────────────────────────────────────────────────────────
# census_task end-to-end
# ─────────────────────────────────────────────────────────────────────


def _write_meta(tmp_path: Path, meta: dict) -> Path:
    td = tmp_path / "L_synth"
    td.mkdir()
    (td / "meta.yaml").write_text(yaml.safe_dump(meta))
    (td / "kernel.py").write_text("def kernel(x): return x\n")
    return td


def test_census_task_basic_fields(tmp_path):
    meta = {
        "task_id": "L_synth",
        "category": "elementwise",
        "repo": "vLLM",
        "tolerance_class": "elementwise",
        "upstream": {"commit_sha": "a" * 40},
    }
    td = _write_meta(tmp_path, meta)
    with patch(
        "scripts.provenance.verify_upstream_commit_chain._git_object_reachable",
        return_value=True,
    ):
        rec = census_task(td)
    assert rec is not None
    assert rec.task_id == "L_synth"
    assert rec.category == "elementwise"
    assert rec.reconstructability_risk == "high"
    assert rec.source_1_static_extractable == "V9_DEPRECATED_M1_1"
    assert rec.tolerance_source == "corpus_only"
    assert rec.commit_sha == "a" * 40
    assert rec.commit_sha_verifiable is True


def test_census_task_snapshot_no_git_is_not_verifiable(tmp_path):
    meta = {
        "category": "elementwise",
        "repo": "FBGEMM",
        "upstream": {"commit_sha": "snapshot-no-git"},
        "tolerance_class": "elementwise",
    }
    td = _write_meta(tmp_path, meta)
    rec = census_task(td)
    assert rec is not None
    assert rec.commit_sha_verifiable is False
    assert rec.commit_sha == "snapshot-no-git"


def test_census_task_missing_meta_returns_none(tmp_path):
    td = tmp_path / "L_no_meta"
    td.mkdir()
    assert census_task(td) is None


def test_census_task_kernel_with_gpu_skip_pattern_reports_it(tmp_path):
    """A kernel.py that imports torch.distributed surfaces in the GPU
    skip dimension."""
    meta = {
        "category": "attention",
        "repo": "vLLM",
        "upstream": {"commit_sha": "snapshot-no-git"},
        "tolerance_class": "elementwise",
    }
    td = tmp_path / "L_synth"
    td.mkdir()
    (td / "meta.yaml").write_text(yaml.safe_dump(meta))
    (td / "kernel.py").write_text(
        "from torch.distributed import barrier\n"
        "def kernel(x): return x\n"
    )
    rec = census_task(td)
    assert rec is not None
    assert rec.gpu_skip_patterns, "should detect torch.distributed import"


# ─────────────────────────────────────────────────────────────────────
# Corpus integration smoke test
# ─────────────────────────────────────────────────────────────────────


def test_census_on_real_corpus_task_l1_12_rms_norm():
    """L1_12_rms_norm is category=norm -> HIGH risk."""
    from scripts.provenance.census_source_provenance import census_task as ct, TASKS_ROOT

    td = TASKS_ROOT / "level1" / "L1_12_rms_norm"
    if not td.is_dir():
        pytest.skip("L1_12_rms_norm not available in this checkout")
    rec = ct(td)
    assert rec is not None
    assert rec.task_id == "L1_12_rms_norm"
    assert rec.category == "norm"
    assert rec.reconstructability_risk == "high"
