"""W1.4 — eval.py Roofline aggregation tests.

Validates that `eval._apply_roofline`:
  * fills each shape's t_ref / W_c / M_r / efficiency / median / p99 / cv,
  * aggregates the task-level s_t as the geomean of the per-shape e
    (DESIGN §9), and
  * fails closed when the peak table is unverified.

The dev machine has no NPU; per-shape `t_npu` is the median of a
host-CPU-timed `durations` series carried by W1.3. These tests inject
synthetic durations + a synthetic verified peak table so the scoring
*logic* is exercised without an NPU.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from triton_ascend_bench import eval as _eval
from triton_ascend_bench import peak_table
from triton_ascend_bench.cost_model import CostModel
from triton_ascend_bench.eval import ShapeRunResult, TaskEvalResult

# A two-shape document: knobs feed the cost-model formulas.
_SHAPES_DOC = {
    "size_knobs": {"B": 2, "N": 4},
    "public_shapes": [
        {"id": "P1", "knobs": {"B": 2, "N": 4}},
        {"id": "P2", "knobs": {"B": 4, "N": 8}},
    ],
}


def _verified_peaks(_path=None, *, require_verified=False):
    """Synthetic *verified* peak table — all positive, so require_verified
    passes. π_vector picked so t_ref is a round number per shape."""
    return ({"vector": 1.0}, {"hbm": 1e30})


def _make_result(durations_by_shape: dict[str, list[float]]) -> TaskEvalResult:
    r = TaskEvalResult(task_id="T", static_checks_passed=True)
    for sid, durs in durations_by_shape.items():
        from triton_ascend_bench.scoring import duration_stats
        st = duration_stats(durs)
        r.shape_results.append(ShapeRunResult(
            shape_id=sid, correctness_passed=True, correctness_reason=None,
            measured_ns=None, baseline_ns=None,
            durations=list(durs),
            t_npu=st.median, median=st.median, p99=st.p99, cv=st.cv,
            trials=st.trials))
    return r


# A cost model where W = B*N FLOPs (vector class), bytes negligible.
_COST_MODEL = CostModel(flops={"vector": "B*N"}, bytes={"hbm": "1"})
_TASK = SimpleNamespace(cost_model=_COST_MODEL)


def test_apply_roofline_fills_per_shape_fields(monkeypatch):
    monkeypatch.setattr(peak_table, "load_peak_table", _verified_peaks)
    # P1: W = 2*4 = 8, π = 1.0  -> t_ref = 8.0
    # P2: W = 4*8 = 32, π = 1.0 -> t_ref = 32.0
    result = _make_result({"P1": [16.0], "P2": [16.0]})
    _eval._apply_roofline(result, _TASK, _SHAPES_DOC)

    p1, p2 = result.shape_results
    assert p1.t_ref == pytest.approx(8.0)
    assert p2.t_ref == pytest.approx(32.0)
    assert p1.w_per_class == {"vector": pytest.approx(8.0)}
    assert p2.w_per_class == {"vector": pytest.approx(32.0)}
    # DESIGN §9 provenance: baseline_ns = t_ref, measured_ns = t_npu.
    assert p1.baseline_ns == pytest.approx(8.0)
    assert p1.measured_ns == pytest.approx(16.0)
    # disclosure stats present.
    assert p1.median == pytest.approx(16.0) and p1.trials == 1


def test_s_t_is_geomean_of_per_shape_e(monkeypatch):
    """The headline W1.4 invariant: a multi-shape task's s_t equals the
    geometric mean of its per-shape efficiencies e."""
    monkeypatch.setattr(peak_table, "load_peak_table", _verified_peaks)
    # P1: t_ref = 8.0,  t_npu = 16.0  -> e = 8/16  = 0.5
    # P2: t_ref = 32.0, t_npu = 16.0  -> e = 32/16 = 2.0
    result = _make_result({"P1": [16.0], "P2": [16.0]})
    _eval._apply_roofline(result, _TASK, _SHAPES_DOC)

    p1, p2 = result.shape_results
    assert p1.efficiency == pytest.approx(0.5)
    assert p2.efficiency == pytest.approx(2.0)
    # s_t = geomean(0.5, 2.0) = 1.0
    assert result.s_t == pytest.approx(1.0)


def test_s_t_single_shape_equals_its_efficiency(monkeypatch):
    monkeypatch.setattr(peak_table, "load_peak_table", _verified_peaks)
    # P1 only: t_ref = 8.0, t_npu = 32.0 -> e = 0.25.
    result = _make_result({"P1": [32.0]})
    _eval._apply_roofline(result, _TASK, _SHAPES_DOC)
    assert result.s_t == pytest.approx(0.25)


def test_no_cost_model_leaves_s_t_none(monkeypatch):
    monkeypatch.setattr(peak_table, "load_peak_table", _verified_peaks)
    result = _make_result({"P1": [16.0]})
    _eval._apply_roofline(result, SimpleNamespace(cost_model=None), _SHAPES_DOC)
    assert result.s_t is None
    assert result.shape_results[0].efficiency is None


def test_unverified_peaks_fail_closed(monkeypatch):
    """An unverified π/β must fail closed: no s_t, a roofline_error on
    every shape — never scored with bad data."""
    def _raises(_path=None, *, require_verified=False):
        raise peak_table.PeakTableError("vector peak is an [unverified] placeholder")

    monkeypatch.setattr(peak_table, "load_peak_table", _raises)
    result = _make_result({"P1": [16.0], "P2": [16.0]})
    _eval._apply_roofline(result, _TASK, _SHAPES_DOC)

    assert result.s_t is None
    for r in result.shape_results:
        assert r.roofline_error is not None
        assert "unverified" in r.roofline_error
        assert r.efficiency is None


def test_partial_roofline_not_scored(monkeypatch):
    """A shape with no timing (empty durations) yields no e; the task s_t
    stays None rather than scoring a partial Roofline."""
    monkeypatch.setattr(peak_table, "load_peak_table", _verified_peaks)
    result = _make_result({"P1": [16.0], "P2": []})  # P2 has no durations
    _eval._apply_roofline(result, _TASK, _SHAPES_DOC)
    # P1 got an e, P2 did not -> not all graded shapes scored -> s_t None.
    assert result.shape_results[0].efficiency is not None
    assert result.shape_results[1].efficiency is None
    assert result.s_t is None


def test_task_measurement_carries_s_t_and_status(monkeypatch):
    monkeypatch.setattr(peak_table, "load_peak_table", _verified_peaks)
    result = _make_result({"P1": [16.0], "P2": [16.0]})
    result.correctness_status = "ok"
    result.runtime_status = "ok"
    result.performance_status = "ok"
    _eval._apply_roofline(result, _TASK, _SHAPES_DOC)
    tm = result.task_measurement()
    assert tm.s_t == pytest.approx(result.s_t)
    assert tm.task_status == "evaluated"
    # provenance pair carried through.
    assert tm.baseline_ns is not None and tm.measured_ns is not None
