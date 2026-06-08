"""W2.2 — fallback → correctness-fail normalisation tests.

A torch-fallback (the candidate produced no NPU/triton kernel event and
silently fell back to the generic torch path) must be normalised to a
correctness fail *before* scoring (DESIGN §6.3 / §6.7): a correct-but-
fallback solution must not clear the correctness gate.

W2.2 has two halves:
  * eval — an explicit child-side ``fallback_signal=True`` per shape sets
    ``TaskEvalResult.fallback_detected`` (W2.4) and W2.2 then forces
    ``correctness_status="fail"``;
  * scoring — ``native_coverage``'s predicate is
    ``correctness == "ok" ∧ ¬fallback_detected`` (a second line of
    defence even if a result reaches scoring un-normalised).

Critically: the dev machine has no NPU, so the profiler ships
``fallback_signal=None`` ("not measured on this host"). ``None`` must
NOT be mistaken for a fallback — it never sets ``fallback_detected`` and
never triggers the normalisation. ``False`` (kernel event present, no
fallback) likewise must not.
"""
from __future__ import annotations

from ascendbench.engine import eval as _eval
from ascendbench.engine.eval import (
    ShapeRunResult,
    TaskEvalResult,
)
from ascendbench.integrity.process_isolation import (
    CandidateRunResult,
    ShapeCandidateResult,
    TwoPassResult,
)
from ascendbench.perf.scoring import TaskMeasurement, native_coverage


def _two_pass(diag: dict) -> TwoPassResult:
    """A TwoPassResult whose timing pass carries one shape's W2.4
    diagnostic signals (kernel_event_present / stream_nondefault /
    fallback_signal)."""
    tscr = ShapeCandidateResult(shape_id="P1", ran=True, **diag)
    cscr = ShapeCandidateResult(shape_id="P1", ran=True)
    corr = CandidateRunResult(True, False, None, per_shape={"P1": cscr})
    tim = CandidateRunResult(True, False, None, per_shape={"P1": tscr},
                             pass_kind="timing")
    return TwoPassResult(correctness=corr, timing=tim)


# ─────────────────────────────────────────────────────────────────────────────
# _normalise_fallback_to_correctness_fail (the W2.2 hook in isolation)
# ─────────────────────────────────────────────────────────────────────────────
def test_normalise_forces_correctness_fail_when_fallback_detected():
    """fallback_detected=True ⇒ correctness_status normalised to fail."""
    result = TaskEvalResult(task_id="T", static_checks_passed=True)
    result.fallback_detected = True
    result.correctness_status = "ok"

    _eval._normalise_fallback_to_correctness_fail(result)

    assert result.correctness_status == "fail"


def test_normalise_noop_when_no_fallback():
    """No fallback ⇒ correctness_status is left exactly as graded."""
    result = TaskEvalResult(task_id="T", static_checks_passed=True)
    result.fallback_detected = False
    result.correctness_status = "ok"

    _eval._normalise_fallback_to_correctness_fail(result)

    assert result.correctness_status == "ok"


def test_normalise_leaves_error_status_intact():
    """A harness 'error' verdict is a strictly stronger signal than a
    graded fail — the normalisation must not weaken it to 'fail'."""
    result = TaskEvalResult(task_id="T", static_checks_passed=True)
    result.fallback_detected = True
    result.correctness_status = "error"

    _eval._normalise_fallback_to_correctness_fail(result)

    assert result.correctness_status == "error"


def test_normalise_is_idempotent():
    """Calling the hook twice is the same as calling it once."""
    result = TaskEvalResult(task_id="T", static_checks_passed=True)
    result.fallback_detected = True
    result.correctness_status = "ok"

    _eval._normalise_fallback_to_correctness_fail(result)
    _eval._normalise_fallback_to_correctness_fail(result)

    assert result.correctness_status == "fail"


# ─────────────────────────────────────────────────────────────────────────────
# diagnostics → normalisation, the three fallback_signal states
# ─────────────────────────────────────────────────────────────────────────────
def test_fallback_signal_true_normalises_to_correctness_fail():
    """fallback_signal=True ⇒ fallback_detected=True ∧ correctness=fail."""
    result = TaskEvalResult(task_id="T", static_checks_passed=True)
    result.correctness_status = "ok"
    result.shape_results.append(ShapeRunResult("P1", True, None, None, None))
    two = _two_pass({"kernel_event_present": False,
                     "stream_nondefault": False,
                     "fallback_signal": True})

    _eval._apply_runtime_diagnostics(result, two)
    _eval._normalise_fallback_to_correctness_fail(result)

    assert result.fallback_detected is True
    assert result.correctness_status == "fail"
    assert "torch_fallback" in result.performance_flags


def test_fallback_signal_none_does_not_normalise():
    """The dev machine ships fallback_signal=None (profiler not measured).
    None ≠ True — it must NOT set fallback_detected nor flip correctness."""
    result = TaskEvalResult(task_id="T", static_checks_passed=True)
    result.correctness_status = "ok"
    result.shape_results.append(ShapeRunResult("P1", True, None, None, None))
    two = _two_pass({"kernel_event_present": None,
                     "stream_nondefault": None,
                     "fallback_signal": None})

    _eval._apply_runtime_diagnostics(result, two)
    _eval._normalise_fallback_to_correctness_fail(result)

    assert result.fallback_detected is False
    assert result.correctness_status == "ok"
    assert "torch_fallback" not in result.performance_flags


def test_fallback_signal_false_does_not_normalise():
    """fallback_signal=False (kernel event present, no fallback) must not
    set fallback_detected nor flip correctness."""
    result = TaskEvalResult(task_id="T", static_checks_passed=True)
    result.correctness_status = "ok"
    result.shape_results.append(ShapeRunResult("P1", True, None, None, None))
    two = _two_pass({"kernel_event_present": True,
                     "stream_nondefault": False,
                     "fallback_signal": False})

    _eval._apply_runtime_diagnostics(result, two)
    _eval._normalise_fallback_to_correctness_fail(result)

    assert result.fallback_detected is False
    assert result.correctness_status == "ok"
    assert "torch_fallback" not in result.performance_flags


# ─────────────────────────────────────────────────────────────────────────────
# scoring.native_coverage — fallback excluded from the numerator
# ─────────────────────────────────────────────────────────────────────────────
def test_native_coverage_excludes_fallback_task():
    """A fallback task with correctness_status still 'ok' (un-normalised
    result reaching scoring) is excluded from the coverage numerator —
    the ¬fallback conjunct is the second line of defence."""
    m = [
        TaskMeasurement(task_id="t1", correctness_status="ok"),
        TaskMeasurement(task_id="t2", correctness_status="ok",
                        fallback_detected=True),
    ]
    # Only t1 counts: 1 / 2.
    assert native_coverage(m, s_native_size=2) == 0.5


def test_native_coverage_normalised_fallback_also_excluded():
    """The normal path: eval already set correctness_status='fail' for the
    fallback task. It must still be excluded — 1 / 2."""
    m = [
        TaskMeasurement(task_id="t1", correctness_status="ok"),
        TaskMeasurement(task_id="t2", correctness_status="fail",
                        fallback_detected=True),
    ]
    assert native_coverage(m, s_native_size=2) == 0.5


def test_native_coverage_no_fallback_full_coverage():
    """No fallback anywhere ⇒ the predicate degrades to the plain
    correctness == 'ok' count — full coverage."""
    m = [
        TaskMeasurement(task_id="t1", correctness_status="ok"),
        TaskMeasurement(task_id="t2", correctness_status="ok"),
    ]
    assert native_coverage(m, s_native_size=2) == 1.0
