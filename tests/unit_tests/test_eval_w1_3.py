"""W1.3 — eval.py orchestration tests: two-pass driving, durations
uplink, timing arbitration mapping, sampled-output cheat-path closure.

The full `evaluate()` happy path is covered by test_eval_grading_smoke.py;
here we (1) confirm `evaluate()` carries the timing-pass durations and
(2) unit-test the W1.3 logic that maps timing arbitration / sampled-output
drift onto the result status fields without a slow full grading run.
"""
from __future__ import annotations

from pathlib import Path

from triton_ascend_bench import eval as _eval
from triton_ascend_bench import inputs as _inputs
from triton_ascend_bench import timing
from triton_ascend_bench.eval import ShapeRunResult, evaluate

REPO = Path(__file__).resolve().parents[2]
_TASK = "L1_01_bias_kernel"
_SOLUTION = REPO / "solutions" / _TASK / "solution.py"


# ---------------------------------------------------------------------------
# evaluate() drives the two-pass pipeline and records timing durations.
# ---------------------------------------------------------------------------

def test_evaluate_records_timing_durations():
    """A graded task carries the timing pass's per-trial host_ns series
    on its shape results — proof eval orchestrated the timing pass."""
    assert _SOLUTION.exists(), f"fixture missing: {_SOLUTION}"
    result = evaluate(_TASK, _SOLUTION)

    assert result.error_reason is None, result.error_reason
    assert len(result.shape_results) > 0
    # At least one shape that ran in the timing pass has durations.
    timed = [r for r in result.shape_results if r.durations]
    # Either the candidate ran (timed) or every shape recorded a reason.
    assert timed or all(r.correctness_reason for r in result.shape_results)
    if timed:
        for r in timed:
            assert all(d > 0 for d in r.durations)


def test_evaluate_no_parent_side_timing():
    """eval.py must not time candidate code in the parent — measured_ns
    and baseline_ns stay None (W1.4 fills them from durations later)."""
    result = evaluate(_TASK, _SOLUTION)
    for r in result.shape_results:
        # W1.3: parent does no wall-clock timing of its own.
        assert r.measured_ns is None
        assert r.baseline_ns is None


# ---------------------------------------------------------------------------
# Timing arbitration -> result-field mapping (the W1.3 status landing).
# ---------------------------------------------------------------------------

def test_timing_arbitration_invalid_timing_maps_runtime_status():
    """A non-finite host duration ⇒ runtime_status='invalid_timing'."""
    arb = timing.arbitrate_timing([5_000_000, float("inf"), 5_000_000])
    assert arb.runtime_status == "invalid_timing"
    # eval applies arb.runtime_status onto TaskEvalResult.runtime_status.


def test_timing_arbitration_quarantined_maps_performance_status():
    """A sub-µs run ⇒ performance_status='quarantined'."""
    arb = timing.arbitrate_timing([200, 150, 180])
    assert arb.performance_status == "quarantined"


def test_timing_arbitration_invalid_stream_flag():
    """Drift + non-default stream ⇒ performance_flags has 'invalid_stream'
    and runtime_status='invalid_timing'."""
    arb = timing.arbitrate_timing(
        [10_000_000] * 3, event_ns=[1_000_000] * 3, stream_nondefault=True)
    assert "invalid_stream" in arb.performance_flags
    assert arb.runtime_status == "invalid_timing"


# ---------------------------------------------------------------------------
# Sampled-output cheat-path closure (_verify_sampled_outputs).
# ---------------------------------------------------------------------------

_SHAPES_DOC = {
    "size_knobs": {"D": 8},
    "public_shapes": [{"id": "P1", "name": "tiny", "knobs": {"D": 4}}],
    "inputs": {
        "x": {"dims": ["D"], "dtype": "float32", "dist": "randn"},
    },
}


class _Oracle:
    """Stand-in oracle module exposing run(inputs) -> dict."""

    @staticmethod
    def run(inputs):
        return {"y": inputs["x"] * 2.0}


def _tol_cls():
    from triton_ascend_bench import tolerance
    pol = tolerance.load_policy()
    # any elementwise-ish class works; pick a permissive one.
    return next(iter(pol.values()))


def test_verify_sampled_outputs_matching_passes():
    """Sampled outputs that match golden ⇒ no mismatch."""
    meas_seed = 1000
    sampled = {}
    for t in (0, 2, 5):
        inp = _inputs.build_inputs(_SHAPES_DOC, "P1", seed=meas_seed + t)
        golden = _Oracle.run(inp)
        for k, v in golden.items():
            sampled[f"{t}::{k}"] = v

    drift = _eval._verify_sampled_outputs(
        sampled, _SHAPES_DOC, "P1", meas_seed, _Oracle, _tol_cls())
    assert drift["checked"] == 3
    assert drift["mismatch"] is False


def test_verify_sampled_outputs_drift_detected():
    """A sampled output that does NOT match golden ⇒ mismatch=True.

    Simulates a candidate that returned a correct answer in the
    correctness pass but a wrong one in the timing pass.
    """
    meas_seed = 1000
    sampled = {}
    for t in (0, 1):
        inp = _inputs.build_inputs(_SHAPES_DOC, "P1", seed=meas_seed + t)
        golden = _Oracle.run(inp)
        for k, v in golden.items():
            # Corrupt trial t=1's output — the timing-pass cheat.
            sampled[f"{t}::{k}"] = v + (100.0 if t == 1 else 0.0)

    drift = _eval._verify_sampled_outputs(
        sampled, _SHAPES_DOC, "P1", meas_seed, _Oracle, _tol_cls())
    assert drift["checked"] == 2
    assert drift["mismatch"] is True


def test_shape_run_result_carries_w1_3_fields():
    """ShapeRunResult exposes the W1.3 carrier fields with safe defaults."""
    r = ShapeRunResult("P1", True, None, None, None)
    assert r.durations == []
    assert r.sampled_outputs_checked == 0
    r2 = ShapeRunResult("P2", True, None, None, None,
                        durations=[1.0, 2.0], sampled_outputs_checked=3)
    assert r2.durations == [1.0, 2.0]
    assert r2.sampled_outputs_checked == 3
