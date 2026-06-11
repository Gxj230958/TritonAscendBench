"""Unit tests for V9 plan M0a.2 -- R13c go/no-go release planning gate.

Covers all 4 decision paths (PROCEED / PROVISIONAL /
PROVISIONAL_MISSING_RARE / STOP) on synthetic corpora.
"""
from __future__ import annotations

import pytest

from scripts.provenance.compute_release_planning_gate import (
    RARE_CLASS_SET,
    STATE_PROCEED,
    STATE_PROVISIONAL,
    STATE_PROVISIONAL_MISSING_RARE,
    STATE_STOP,
    THRESH_PROCEED_CATEGORIES,
    THRESH_PROCEED_SIZE,
    THRESH_PROVISIONAL_CATEGORIES,
    THRESH_PROVISIONAL_SIZE,
    THRESH_RARE_CLASS_COVERAGE_RATIO,
    _MiniTask,
    compute_gate,
)


def _task(
    task_id: str,
    *,
    category: str,
    verified: bool = True,
    risk: str = "low",
    tolerance_source: str = "upstream_test_default",
) -> _MiniTask:
    """Build a synthetic task object with projected-eligible state."""
    golden = {}
    if verified:
        golden["oracle_kernel_equivalence"] = "VERIFIED"
    if risk:
        golden["reconstructability_risk"] = risk
    if tolerance_source:
        golden["tolerance_source"] = tolerance_source
    meta = {
        "category": category,
        "tolerance_class": "elementwise",
        "provenance": {"golden": golden},
    }
    return _MiniTask(task_id=task_id, meta=meta)


# ─────────────────────────────────────────────────────────────────────
# STOP states
# ─────────────────────────────────────────────────────────────────────


def test_empty_corpus_returns_stop():
    out = compute_gate([])
    assert out.state == STATE_STOP
    assert out.projected_size == 0


def test_small_corpus_returns_stop():
    """Below the PROVISIONAL size threshold -> STOP."""
    tasks = [
        _task(f"T{i:02d}", category=f"cat{i % 4}")
        for i in range(THRESH_PROVISIONAL_SIZE - 1)
    ]
    out = compute_gate(tasks)
    assert out.state == STATE_STOP
    assert any("STOP threshold" in r for r in out.reasons)


def test_too_few_categories_returns_stop():
    """Big enough corpus but only 2 categories -> STOP."""
    tasks = [
        _task(f"T{i:02d}", category="cat_a" if i % 2 == 0 else "cat_b")
        for i in range(THRESH_PROCEED_SIZE + 5)
    ]
    out = compute_gate(tasks)
    assert out.state == STATE_STOP


# ─────────────────────────────────────────────────────────────────────
# PROCEED state
# ─────────────────────────────────────────────────────────────────────


def test_full_proceed_with_rare_class_coverage():
    """size >= 60, categories >= 8, rare-class coverage OK -> PROCEED."""
    tasks: list[_MiniTask] = []
    # 8 non-rare categories x 7 tasks each = 56
    for cat in ("attention", "moe", "gemm_mid", "elementwise_mid",
                "softmax", "ssm", "paged-attention", "scan"):
        for i in range(7):
            tasks.append(_task(f"{cat}_{i}", category=cat))
    # Add rare classes -- all 9 rare tasks covered
    for cat, n in (("convolution", 3), ("spec_decode", 1),
                   ("normalization", 5)):
        for i in range(n):
            tasks.append(_task(f"{cat}_{i}", category=cat))

    out = compute_gate(tasks)
    assert out.state == STATE_PROCEED
    assert out.projected_size >= THRESH_PROCEED_SIZE
    assert out.distinct_categories >= THRESH_PROCEED_CATEGORIES
    assert out.rare_class_ratio >= THRESH_RARE_CLASS_COVERAGE_RATIO


# ─────────────────────────────────────────────────────────────────────
# PROVISIONAL state
# ─────────────────────────────────────────────────────────────────────


def test_provisional_when_size_between_30_and_60():
    """PROVISIONAL size threshold met but below PROCEED size."""
    tasks: list[_MiniTask] = []
    # 6 non-rare categories x 6 tasks each = 36
    for cat in ("attention", "moe", "ssm",
                "paged-attention", "softmax", "scan"):
        for i in range(6):
            tasks.append(_task(f"{cat}_{i}", category=cat))
    # All rare classes covered fully
    for cat, n in (("convolution", 3), ("spec_decode", 1),
                   ("normalization", 5)):
        for i in range(n):
            tasks.append(_task(f"{cat}_{i}", category=cat))

    out = compute_gate(tasks)
    assert out.state == STATE_PROVISIONAL
    assert THRESH_PROVISIONAL_SIZE <= out.projected_size < THRESH_PROCEED_SIZE
    assert out.distinct_categories >= THRESH_PROVISIONAL_CATEGORIES


# ─────────────────────────────────────────────────────────────────────
# PROVISIONAL_MISSING_RARE state
# ─────────────────────────────────────────────────────────────────────


def test_provisional_missing_rare_when_rare_classes_uncovered():
    """size + categories OK but rare-class coverage < 50% -> MISSING_RARE."""
    tasks: list[_MiniTask] = []
    # Fill the projected_eligible set with NON-rare categories only,
    # keeping it within the PROVISIONAL window so a separate PROCEED
    # outcome doesn't trigger.
    for cat in ("attention", "moe", "ssm",
                "paged-attention", "softmax", "scan"):
        for i in range(6):  # 6*6 = 36 tasks -> PROVISIONAL band
            tasks.append(_task(f"{cat}_{i}", category=cat))
    # Add rare classes to the CORPUS but NOT verified -> they will not
    # enter projected_eligible.
    for cat, n in (("convolution", 3), ("spec_decode", 1),
                   ("normalization", 5)):
        for i in range(n):
            tasks.append(
                _task(f"{cat}_{i}", category=cat, verified=False)
            )

    out = compute_gate(tasks)
    # The verified non-rare set is in PROVISIONAL band; rare coverage
    # = 0/9 = 0.0% which is < 50%, so PROVISIONAL_MISSING_RARE wins.
    assert out.state == STATE_PROVISIONAL_MISSING_RARE
    assert out.rare_class_ratio == 0.0
    assert any("rare-class coverage" in r for r in out.reasons)


# ─────────────────────────────────────────────────────────────────────
# projected_eligible filter behaviour
# ─────────────────────────────────────────────────────────────────────


def test_r17_high_risk_excluded_from_projected_eligible():
    """elementwise tasks (R17 high risk) never enter projected_eligible,
    even when they're individually VERIFIED."""
    tasks = [
        _task(f"elem_{i}", category="elementwise", risk="high")
        for i in range(50)
    ]
    out = compute_gate(tasks)
    # All filtered out -> STOP
    assert out.state == STATE_STOP
    assert out.projected_size == 0


def test_migration_drift_budget_excluded_from_projected_eligible():
    """R11 v2 deferred path excludes migration_drift_budget tasks."""
    tasks: list[_MiniTask] = []
    # 50 attention tasks, half migration_drift_budget (excluded)
    for i in range(50):
        tasks.append(
            _task(
                f"att_{i}",
                category="attention",
                tolerance_source=(
                    "migration_drift_budget" if i % 2 == 0 else
                    "upstream_test_default"
                ),
            )
        )
    out = compute_gate(tasks)
    # Only ~25 tasks enter projected_eligible -> STOP (below 30)
    assert out.projected_size == 25


def test_rare_class_set_excludes_r17_high_risk():
    """V9 V4 final: rare class set is DISJOINT from R17 high-risk.

    `norm` is in R17 high-risk (excluded from projected_eligible) so it
    is NOT in the rare set. `normalization` is in the rare set.
    """
    assert "norm" not in RARE_CLASS_SET
    assert "normalization" in RARE_CLASS_SET
    assert "convolution" in RARE_CLASS_SET
    assert "spec_decode" in RARE_CLASS_SET
