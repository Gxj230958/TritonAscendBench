"""W2.4 — eval.py runtime-defense wiring tests.

Covers:
  * `_apply_runtime_diagnostics` — child-side fallback / non-default-stream
    signals folding into fallback_detected / performance_status / flags;
  * `_run_task_metamorphic` — the task's declared metamorphic checks run
    and land on each shape's ShapeRunResult.metamorphic_results;
  * a full `evaluate()` carries metamorphic_results through.
"""
from __future__ import annotations

from pathlib import Path

from triton_ascend_bench import eval as _eval
from triton_ascend_bench.eval import (
    ShapeRunResult,
    TaskEvalResult,
    evaluate,
)
from triton_ascend_bench.process_isolation import (
    CandidateRunResult,
    ShapeCandidateResult,
    TwoPassResult,
)

REPO = Path(__file__).resolve().parents[2]
_TASK = "L1_01_bias_kernel"
_SOLUTION = REPO / "solutions" / _TASK / "solution.py"


def _two_pass(diag: dict) -> TwoPassResult:
    """Build a TwoPassResult whose timing pass carries one shape's W2.4
    diagnostic signals (`diag` keys: kernel_event_present /
    stream_nondefault / fallback_signal)."""
    tscr = ShapeCandidateResult(shape_id="P1", ran=True, **diag)
    cscr = ShapeCandidateResult(shape_id="P1", ran=True)
    corr = CandidateRunResult(True, False, None, per_shape={"P1": cscr})
    tim = CandidateRunResult(True, False, None, per_shape={"P1": tscr},
                             pass_kind="timing")
    return TwoPassResult(correctness=corr, timing=tim)


# ─────────────────────────────────────────────────────────────────────────────
# _apply_runtime_diagnostics
# ─────────────────────────────────────────────────────────────────────────────
def test_fallback_signal_sets_fallback_detected_and_flag():
    """A fallback_signal=True shape ⇒ fallback_detected + torch_fallback."""
    result = TaskEvalResult(task_id="T", static_checks_passed=True)
    result.shape_results.append(
        ShapeRunResult("P1", True, None, None, None))
    two = _two_pass({"kernel_event_present": False,
                     "stream_nondefault": False,
                     "fallback_signal": True})

    _eval._apply_runtime_diagnostics(result, two)

    assert result.fallback_detected is True
    assert "torch_fallback" in result.performance_flags


def test_nondefault_stream_sets_invalid_stream_and_suspicious():
    """stream_nondefault=True ⇒ invalid_stream flag + suspicious status.

    R6-3: `_apply_runtime_diagnostics` escalates `performance_status` via
    the `ok < suspicious < quarantined` precedence ladder; in the real
    `evaluate()` flow it runs *after* the R1-1/R2-1 promotion has lifted
    the verdict off the "error" default to "ok". This test sets that
    realistic precondition (`"ok"`) — escalating a not-graded "error"
    verdict would be wrong, as "error" is the orthogonal "perf never
    graded" state, not a milder graded verdict.
    """
    result = TaskEvalResult(task_id="T", static_checks_passed=True)
    result.performance_status = "ok"
    result.shape_results.append(
        ShapeRunResult("P1", True, None, None, None))
    two = _two_pass({"kernel_event_present": True,
                     "stream_nondefault": True,
                     "fallback_signal": False})

    _eval._apply_runtime_diagnostics(result, two)

    assert "invalid_stream" in result.performance_flags
    assert result.performance_status == "suspicious"


def test_none_signals_do_not_flip_status():
    """The dev-machine stub ships None — no status is changed."""
    result = TaskEvalResult(task_id="T", static_checks_passed=True)
    result.performance_status = "ok"
    result.shape_results.append(
        ShapeRunResult("P1", True, None, None, None))
    two = _two_pass({"kernel_event_present": None,
                     "stream_nondefault": None,
                     "fallback_signal": None})

    _eval._apply_runtime_diagnostics(result, two)

    assert result.fallback_detected is False
    assert result.performance_flags == []
    assert result.performance_status == "ok"


def test_quarantined_not_downgraded_by_nondefault_stream():
    """A non-default stream must not overwrite a quarantined verdict."""
    result = TaskEvalResult(task_id="T", static_checks_passed=True)
    result.performance_status = "quarantined"
    result.shape_results.append(
        ShapeRunResult("P1", True, None, None, None))
    two = _two_pass({"kernel_event_present": True,
                     "stream_nondefault": True,
                     "fallback_signal": False})

    _eval._apply_runtime_diagnostics(result, two)

    assert result.performance_status == "quarantined"
    assert "invalid_stream" in result.performance_flags


# ─────────────────────────────────────────────────────────────────────────────
# _run_task_metamorphic
# ─────────────────────────────────────────────────────────────────────────────
class _FakeTask:
    def __init__(self, meta):
        self.meta = meta


def test_run_task_metamorphic_no_checks_is_empty():
    task = _FakeTask({"runner_checks": {"metamorphic_checks": []}})
    assert _eval._run_task_metamorphic(task) == []


def test_run_task_metamorphic_declared_check_surfaces():
    """Called with no solution/shape context, a declared check still
    surfaces (as a non-pass MetamorphicResult), never silently dropped."""
    task = _FakeTask({"runner_checks": {
        "metamorphic_checks": [{"name": "scale_homogeneity_a"}]}})
    out = _eval._run_task_metamorphic(task)
    assert len(out) == 1
    assert out[0].name == "scale_homogeneity_a"
    assert not out[0].passed


# ─────────────────────────────────────────────────────────────────────────────
# R7-3 — real metamorphic transformations wired through _run_task_metamorphic
# ─────────────────────────────────────────────────────────────────────────────
def test_run_task_metamorphic_real_registry_executes_transform(monkeypatch):
    """With a solution + shape, _run_task_metamorphic builds a real
    transformation registry and the declared check is actually executed
    (not the `no transformation registered` placeholder)."""
    import torch

    from triton_ascend_bench import adversarial

    task = _FakeTask({"runner_checks": {
        "metamorphic_checks": [{"name": "zero_x1_yields_zero"}]}})
    shapes_doc = {
        "inputs": {"x1": {"dims": ["B"], "dtype": "float32", "dist": "randn"}},
        "size_knobs": {"B": 8},
        "public_shapes": [{"id": "P1"}],
    }

    # Stub the context build: honest kernel — silu(0)*1 == 0.
    def _fake_ctx(solution_path, sd, sid, tol):
        base = {"x1": torch.randn(8)}
        return adversarial.MetamorphicContext(
            base_inputs=base,
            run_candidate=lambda inp: {
                "out": torch.nn.functional.silu(inp["x1"])},
            tensors_close=lambda a, b: torch.allclose(a, b, atol=1e-5),
        )

    monkeypatch.setattr(_eval, "_build_metamorphic_context", _fake_ctx)

    out = _eval._run_task_metamorphic(
        task, solution_path=Path("/dev/null"), shapes_doc=shapes_doc,
        rep_shape_id="P1", tol_cls=object())
    assert len(out) == 1
    assert out[0].name == "zero_x1_yields_zero"
    # The real transform ran — detail is the transform's verdict, NOT the
    # `no transformation registered` placeholder.
    assert out[0].passed
    assert "no transformation registered" not in (out[0].detail or "")


def test_run_task_metamorphic_real_registry_catches_cheat(monkeypatch):
    """The real registry FAILS a candidate that violates the invariant."""
    import torch

    from triton_ascend_bench import adversarial

    task = _FakeTask({"runner_checks": {
        "metamorphic_checks": [{"name": "zero_x1_yields_zero"}]}})

    def _fake_ctx(solution_path, sd, sid, tol):
        base = {"x1": torch.randn(8)}
        # cheat: constant output, ignores x1.
        return adversarial.MetamorphicContext(
            base_inputs=base,
            run_candidate=lambda inp: {"out": torch.ones(8)},
            tensors_close=lambda a, b: torch.allclose(a, b, atol=1e-5),
        )

    monkeypatch.setattr(_eval, "_build_metamorphic_context", _fake_ctx)

    out = _eval._run_task_metamorphic(
        task, solution_path=Path("/dev/null"), shapes_doc={},
        rep_shape_id="P1", tol_cls=object())
    assert not out[0].passed
    assert "all-zero" in (out[0].detail or "")


# ─────────────────────────────────────────────────────────────────────────────
# full evaluate() carries metamorphic_results
# ─────────────────────────────────────────────────────────────────────────────
def test_evaluate_attaches_metamorphic_results():
    """A graded task's shape results carry a metamorphic_results list
    (empty when the task declares no checks — the field still exists)."""
    assert _SOLUTION.exists(), f"fixture missing: {_SOLUTION}"
    result = evaluate(_TASK, _SOLUTION)
    assert result.error_reason is None, result.error_reason
    for r in result.shape_results:
        assert hasattr(r, "metamorphic_results")
        assert isinstance(r.metamorphic_results, list)
