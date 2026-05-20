"""R10 codex-review regression tests.

Pins the R10 MUST-FIX so a regression re-surfaces as a test failure:

  * R10-1 — ``scoring.aggregate_perf`` must NOT treat every status-ok
    task with no perf signal as a Roofline-data gap. A task is excluded
    from the perf geomean as a *benchmark-data* gap ONLY when the repo
    metadata authoritatively says its Roofline data is incomplete
    (``roofline_data_complete is False`` — a missing cost model or a
    referenced peak still ``[unverified]``). A status-ok task that the
    repo authoritatively marks ``roofline_data_complete=True`` but whose
    result JSON carries no ``s_t`` (e.g. a submitter stripped it) stays
    in the scored set and is ε-collapsed by ``s_t_value`` — it must NOT
    escape the fixed ``|S|`` denominator (R9-1 no-subset-escape).

Dev machine has no NPU; these verify the scoring plumbing with synthetic
measurements and (for the ``analyze`` path) a monkeypatched peak table.
"""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from triton_ascend_bench import peak_table
from triton_ascend_bench.cli import _cmd_analyze
from triton_ascend_bench.scoring import (
    EPSILON,
    TaskMeasurement,
    aggregate_perf,
)


# ──────────────────────────────────────────────────────────────────────────
# R10-1 — aggregate_perf: a stripped s_t cannot escape the ε denominator.
# ──────────────────────────────────────────────────────────────────────────
class TestR10AggregatePerf:
    def test_data_complete_task_with_no_s_t_stays_in_denominator(self):
        """A status-ok ``evaluated`` task that the repo authoritatively
        marks ``roofline_data_complete=True`` but whose result carries no
        ``s_t`` is NOT a Roofline-data gap — it stays in the scored set
        and ``s_t_value`` ε-collapses it. With one scorable task
        (``s_t=1.0``) and one such no-signal task over ``|S|=2`` the
        geomean is ``sqrt(1.0 * ε)``, NOT ``1.0`` (which would mean the
        no-signal task escaped numerator + denominator)."""
        ms = [
            TaskMeasurement(task_id="a", s_t=1.0),
            TaskMeasurement(task_id="b", s_t=None,
                            roofline_data_complete=True),
        ]
        out = aggregate_perf(ms, eligible_size=2)
        assert out["perf_status"] == "ok"
        # task b is an ε term in |S|=2 — it did NOT escape the denominator.
        assert out["capped_geomean_native"] == pytest.approx(EPSILON ** 0.5)
        # it is NOT a Roofline-data gap (repo says data complete).
        assert out["roofline_data_incomplete_count"] == 0

    def test_all_data_complete_no_signal_is_ok_not_incomplete(self):
        """When every in-scored-set task is status-ok with no perf signal
        but the repo authoritatively marks them ``roofline_data_complete
        =True``, the status is ``ok`` with an honest all-ε geomean — NOT
        ``roofline_data_incomplete``. A stripped ``s_t`` is a candidate
        failure to disclose a perf number, not a benchmark-data gap."""
        ms = [
            TaskMeasurement(task_id="a", s_t=None,
                            roofline_data_complete=True),
            TaskMeasurement(task_id="b", s_t=None,
                            roofline_data_complete=True),
        ]
        out = aggregate_perf(ms, eligible_size=2)
        assert out["perf_status"] == "ok"
        assert out["capped_geomean_native"] == pytest.approx(EPSILON)
        assert out["fastp_native_at_1_0"] == pytest.approx(0.0)
        assert out["roofline_data_incomplete_count"] == 0

    def test_data_incomplete_task_still_excluded(self):
        """The genuine data-gap exemption is preserved: a status-ok task
        the repo marks ``roofline_data_complete=False`` is still dropped
        from the perf geomean (not the candidate's fault)."""
        ms = [
            TaskMeasurement(task_id="a", s_t=2.0),
            TaskMeasurement(task_id="b", s_t=None,
                            roofline_data_complete=False),
        ]
        out = aggregate_perf(ms, eligible_size=2)
        assert out["perf_status"] == "ok"
        # denom = |S|(2) - data_gap(1) = 1 ; geomean({2.0}) = 2.0.
        assert out["capped_geomean_native"] == pytest.approx(2.0)
        assert out["roofline_data_incomplete_count"] == 1

    def test_mixed_data_complete_no_signal_and_genuine_gap(self):
        """1 scorable + 1 data-complete-but-stripped-s_t + 1 genuine
        data gap over ``|S|=3``: only the genuine gap leaves the
        denominator (3-1=2); the stripped-s_t task stays as ε."""
        ms = [
            TaskMeasurement(task_id="a", s_t=2.0),
            TaskMeasurement(task_id="b", s_t=None,
                            roofline_data_complete=True),
            TaskMeasurement(task_id="c", s_t=None,
                            roofline_data_complete=False),
        ]
        out = aggregate_perf(ms, eligible_size=3)
        assert out["perf_status"] == "ok"
        # geomean(2.0, ε) over denom 2 = sqrt(2.0 * ε).
        assert out["capped_geomean_native"] == pytest.approx(
            (2.0 * EPSILON) ** 0.5)
        assert out["roofline_data_incomplete_count"] == 1


# ──────────────────────────────────────────────────────────────────────────
# R10-1 — analyze: a hand-edited result with s_t stripped cannot escape.
# ──────────────────────────────────────────────────────────────────────────
# A FULLY verified peak table — `vector` / `interconnect` are positive
# (unlike the real `infra/peak_table_910b.toml`, where they are still
# `[unverified]` placeholders). With this table the eligible
# ('hbm', 'vector') tasks are authoritatively `roofline_data_complete=True`.
def _all_verified_peaks(_path=None, *, require_verified=False):
    compute = {"fp16_cube": 256e12, "bf16_cube": 200e12, "int8": 512e12,
               "vector": 100e12}
    resource = {"hbm": 1.6e12, "interconnect": 800e9}
    return compute, resource


def test_analyze_stripped_s_t_cannot_escape_epsilon_denominator(
    tmp_path, monkeypatch,
):
    """`analyze` reads a result whose `s_t` was hand-stripped to `null`
    while the repo metadata authoritatively marks the task
    `roofline_data_complete=True` (cost model present, all referenced
    π/β verified). The task must still be ε-scored inside the fixed
    `|S|` denominator — it must NOT be silently dropped from the perf
    geomean (the R10-1 subset-escape)."""
    monkeypatch.setattr(peak_table, "load_peak_table", _all_verified_peaks)

    # `L1_01_bias_kernel` is a score-eligible, non-upstream task whose
    # cost model references ('hbm', 'vector'); with the all-verified
    # table above it is authoritatively `roofline_data_complete=True`.
    verified = {
        "mode": "verified",
        "results": [
            # A real scorable task to keep `perf_status == "ok"`.
            {"task_id": "L1_02_compute_slot_mappings_kernel",
             "static_checks_passed": True, "correctness_status": "ok",
             "runtime_status": "ok", "performance_status": "ok",
             "performance_flags": [], "fallback_detected": False,
             "aggregate_score": 1.0, "s_t": 1.0, "task_status": "evaluated"},
            # Hand-edited: status all-ok, `task_status="evaluated"`, but
            # `s_t` stripped to null. Repo says data is complete -> this
            # is NOT a data gap; it must be ε-scored, not dropped.
            {"task_id": "L1_01_bias_kernel",
             "static_checks_passed": True, "correctness_status": "ok",
             "runtime_status": "ok", "performance_status": "ok",
             "performance_flags": [], "fallback_detected": False,
             "aggregate_score": 0.0, "s_t": None,
             "baseline_ns": None, "measured_ns": None,
             "task_status": "evaluated"},
        ],
    }
    v_path = tmp_path / "v.json"
    v_path.write_text(json.dumps(verified))
    out_path = tmp_path / "score.json"
    rc = _cmd_analyze(SimpleNamespace(
        verified_results=str(v_path),
        hidden_results=None,
        out=str(out_path),
        submitter=None,
    ))
    assert rc == 0
    doc = json.loads(out_path.read_text())

    # The stripped-s_t task is NOT a Roofline-data gap — the repo
    # authority says its data is complete.
    assert doc["roofline_data_incomplete_count"] == 0
    assert doc["perf_status"] == "ok"
    # Both results are measured; the stripped-s_t task stays in the
    # scored set rather than escaping it.
    assert doc["n_native_measurements"] == 2
    # The geomean must reflect the stripped-s_t task as an ε term inside
    # the fixed denominator: it cannot equal what it would be if that
    # task had escaped (i.e. it must be strictly below the geomean of
    # just the scorable task).
    assert doc["capped_geomean_native"] is not None
    assert doc["capped_geomean_native"] < 1.0
