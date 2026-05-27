"""Unit tests for V9 plan M0.0 -- ProvenanceEligibility helpers in
``triton_ascend_bench.eligibility``."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from triton_ascend_bench.eligibility import (
    compute_release_ready_eligible,
    is_provenance_eligible,
    projected_eligible,
    release_ready_with_provenance,
)


# ─────────────────────────────────────────────────────────────────────
# is_provenance_eligible
# ─────────────────────────────────────────────────────────────────────


def test_eligible_none_meta_returns_false():
    assert is_provenance_eligible(None) is False


def test_eligible_no_provenance_block_returns_false():
    assert is_provenance_eligible({}) is False


def test_eligible_top_level_provenance_only_returns_false():
    """Legacy provenance (added_in/authors) alone does NOT establish
    the V9 baseline trust chain."""
    assert is_provenance_eligible(
        {"provenance": {"added_in": "v0.1", "authors": ["alice"]}}
    ) is False


def test_eligible_oracle_kernel_verified():
    assert is_provenance_eligible(
        {
            "provenance": {
                "golden": {"oracle_kernel_equivalence": "VERIFIED"}
            }
        }
    ) is True


def test_eligible_gpu_capture_done():
    assert is_provenance_eligible(
        {
            "provenance": {
                "golden": {"gpu_kernel_capture_status": "DONE"}
            }
        }
    ) is True


def test_eligible_either_path_is_enough():
    """VERIFIED OR DONE -- either path qualifies."""
    assert is_provenance_eligible(
        {
            "provenance": {
                "golden": {
                    "oracle_kernel_equivalence": "FAILED",
                    "gpu_kernel_capture_status": "DONE",
                }
            }
        }
    ) is True


def test_eligible_migration_drift_excluded_even_when_verified():
    """v1 deferred path: migration_drift_budget always excludes the
    task even if R10 verified."""
    assert is_provenance_eligible(
        {
            "provenance": {
                "golden": {
                    "oracle_kernel_equivalence": "VERIFIED",
                    "tolerance_source": "migration_drift_budget",
                }
            }
        }
    ) is False


def test_eligible_failed_status_returns_false():
    """FAILED / SKIPPED / NEEDS_MANUAL / LAUNCH_BLOCKED are not eligible."""
    for status in (
        "FAILED", "MIXED", "SKIPPED", "NEEDS_MANUAL", "LAUNCH_BLOCKED"
    ):
        assert is_provenance_eligible(
            {
                "provenance": {
                    "golden": {"oracle_kernel_equivalence": status}
                }
            }
        ) is False, f"{status} should not be release-ready"


def test_eligible_pending_gpu_capture_returns_false():
    """PENDING is in-progress, not done -- excluded from release."""
    assert is_provenance_eligible(
        {"provenance": {"golden": {"gpu_kernel_capture_status": "PENDING"}}}
    ) is False


# ─────────────────────────────────────────────────────────────────────
# release_ready_with_provenance -- composite predicate
# ─────────────────────────────────────────────────────────────────────


def _verified_provenance() -> dict:
    return {
        "provenance": {
            "golden": {
                "oracle_kernel_equivalence": "VERIFIED",
                "tolerance_source": "upstream_test_explicit",
            }
        }
    }


def _release_status_pass_meta() -> dict:
    """A meta that would pass release_status_pass on its own (PHASE A'
    PASS + cost_model PASS)."""
    return {
        "phase_a_prime": {"status": "PASS"},
        "cost_model": {"status": "PASS"},
    }


def test_release_ready_requires_both_status_and_provenance():
    """release_status_pass alone is not enough; provenance must also pass."""
    only_status = _release_status_pass_meta()
    assert release_ready_with_provenance(only_status) is False
    only_provenance = _verified_provenance()
    assert release_ready_with_provenance(only_provenance) is False

    both = {**only_status, **only_provenance}
    assert release_ready_with_provenance(both) is True


def test_release_ready_handles_none_meta():
    assert release_ready_with_provenance(None) is False


# ─────────────────────────────────────────────────────────────────────
# compute_release_ready_eligible -- set algebra
# ─────────────────────────────────────────────────────────────────────


def _make_task(task_id: str, meta: dict, *, tmp_path: Path) -> SimpleNamespace:
    """Build a task that compute_score_eligible() will consider eligible."""
    td = tmp_path / task_id
    td.mkdir(exist_ok=True)
    # oracle_pytorch.py must NOT contain `raise NotImplementedError`
    (td / "oracle_pytorch.py").write_text("def run(inputs): return inputs\n")
    # shapes.yaml must NOT be PHASE-C-MANUAL
    (td / "shapes.yaml").write_text("public_shapes:\n  - id: P1\n    name: x\n")
    return SimpleNamespace(task_id=task_id, task_dir=td, meta=meta)


def test_compute_release_ready_intersects_with_score_eligible(tmp_path):
    """Only score-eligible AND provenance-eligible tasks count."""
    t_ok = _make_task("T_OK", _verified_provenance(), tmp_path=tmp_path)
    t_no_prov = _make_task("T_NO_PROV", {}, tmp_path=tmp_path)
    # Stub oracle -> not score-eligible
    stub_td = tmp_path / "T_STUB"
    stub_td.mkdir()
    (stub_td / "oracle_pytorch.py").write_text(
        "def run(_): raise NotImplementedError\n"
    )
    (stub_td / "shapes.yaml").write_text("public_shapes:\n  - id: P1\n    name: x\n")
    t_stub = SimpleNamespace(
        task_id="T_STUB", task_dir=stub_td, meta=_verified_provenance()
    )

    out = compute_release_ready_eligible([t_ok, t_no_prov, t_stub])
    assert out == {"T_OK"}


# ─────────────────────────────────────────────────────────────────────
# projected_eligible -- R13c gate input
# ─────────────────────────────────────────────────────────────────────


def test_projected_eligible_default_excludes_r11_pending_and_r17_high(tmp_path):
    """Default flags exclude both migration_drift_budget AND
    reconstructability_risk=high."""
    t1 = _make_task(
        "T_VERIFIED",
        {"provenance": {"golden": {"oracle_kernel_equivalence": "VERIFIED"}}},
        tmp_path=tmp_path,
    )
    t2 = _make_task(
        "T_R11_PENDING",
        {
            "provenance": {
                "golden": {
                    "oracle_kernel_equivalence": "VERIFIED",
                    "tolerance_source": "migration_drift_budget",
                }
            }
        },
        tmp_path=tmp_path,
    )
    t3 = _make_task(
        "T_R17_HIGH",
        {
            "provenance": {
                "golden": {
                    "oracle_kernel_equivalence": "VERIFIED",
                    "reconstructability_risk": "high",
                }
            }
        },
        tmp_path=tmp_path,
    )
    t4 = _make_task(
        "T_PENDING_GPU",
        {
            "provenance": {
                "golden": {"gpu_kernel_capture_status": "PENDING"}
            }
        },
        tmp_path=tmp_path,
    )

    out = projected_eligible([t1, t2, t3, t4])
    assert out == {"T_VERIFIED", "T_PENDING_GPU"}


def test_projected_eligible_excludes_terminal_failures(tmp_path):
    """FAILED / NEEDS_MANUAL / LAUNCH_BLOCKED never become release-ready."""
    for bad_status in ("FAILED", "MIXED", "NEEDS_MANUAL", "LAUNCH_BLOCKED"):
        t = _make_task(
            f"T_{bad_status}",
            {
                "provenance": {
                    "golden": {"oracle_kernel_equivalence": bad_status}
                }
            },
            tmp_path=tmp_path,
        )
        assert projected_eligible([t]) == set(), (
            f"{bad_status} should not enter projected_eligible"
        )


def test_projected_eligible_excludes_pre_v9_tasks(tmp_path):
    """Tasks without nested provenance.golden have no trust chain claim
    and are excluded from the R13c gate input."""
    t = _make_task(
        "T_LEGACY",
        {"provenance": {"added_in": "v0.1"}},
        tmp_path=tmp_path,
    )
    assert projected_eligible([t]) == set()


def test_projected_eligible_can_disable_r11_exclusion(tmp_path):
    """exclude_r11_pending=False lets migration_drift_budget tasks
    through (for what-if analysis once R11 v2 is ready)."""
    t = _make_task(
        "T_R11",
        {
            "provenance": {
                "golden": {
                    "oracle_kernel_equivalence": "VERIFIED",
                    "tolerance_source": "migration_drift_budget",
                }
            }
        },
        tmp_path=tmp_path,
    )
    assert projected_eligible([t]) == set()
    assert projected_eligible([t], exclude_r11_pending=False) == {"T_R11"}


def test_projected_eligible_can_disable_r17_exclusion(tmp_path):
    """exclude_r17_high=False lets high-risk tasks through (for
    diagnostic only -- they are leaderboard-excluded otherwise)."""
    t = _make_task(
        "T_HIGH",
        {
            "provenance": {
                "golden": {
                    "oracle_kernel_equivalence": "VERIFIED",
                    "reconstructability_risk": "high",
                }
            }
        },
        tmp_path=tmp_path,
    )
    assert projected_eligible([t]) == set()
    assert projected_eligible([t], exclude_r17_high=False) == {"T_HIGH"}
