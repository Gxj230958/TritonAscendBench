"""Smoke test for eval.py phase-2b grading -- E1 regression guard.

E1 bug (fixed in W0.1): eval.py called ``torch.is_tensor`` without
``import torch``; the resulting NameError was swallowed by a broad
``except`` so phase-2b grading silently aborted and ``correctness_status``
stayed at its default ``"error"`` -- looking like "not graded" rather than
a real verdict.

This test pins the grading chain: it must run to completion (non-empty
``shape_results``, ``correctness_status`` set to ok/fail) with no harness
``error_reason`` recorded.
"""
from ascendbench.engine.eval import evaluate


def test_phase2b_grading_runs_to_completion(tiny_eval_fixture):
    """Grading must reach the per-shape loop and set a real verdict."""
    task, solution = tiny_eval_fixture
    result = evaluate(task.task_id, solution, task_def=task)

    # static check must pass so grading actually enters phase-2b.
    assert result.static_checks_passed, result.static_findings
    # E1 regression guard: a harness error (e.g. a missing ``import torch``)
    # is captured in error_reason instead of being silently swallowed.
    assert result.error_reason is None, (
        f"phase-2b grading hit a harness error (E1 regression?): "
        f"{result.error_reason}"
    )
    # grading must reach the per-shape loop and produce shape results.
    assert len(result.shape_results) > 0, (
        "phase-2b grading produced no shape results"
    )
    # correctness_status must be set by grading, not stuck at default "error".
    assert result.correctness_status in ("ok", "fail"), (
        f"correctness_status not set by grading: {result.correctness_status!r}"
    )
