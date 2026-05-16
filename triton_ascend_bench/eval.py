"""eval.py — orchestrate one (task, solution, backend) evaluation.

Pipeline:
    1. Load task (dataset.load_task)
    2. Static checks on solution.py (static_checker)
    3. Spawn isolated runner process (process_isolation.py — Phase 2.B)
    4. Per shape:
         a. Generate inputs (rand seed + canary plant)
         b. Run oracle in trusted process; serialize tensors via safetensors
         c. Run candidate solution; collect (output, host_ns)
         d. Compare with tolerance.compare
         e. Record measurement
    5. Score via scoring.capped_geomean_native
    6. Emit round_result.json
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

from . import scoring, static_checker, tolerance
from .dataset import TaskDef, load_task

logger = logging.getLogger(__name__)


@dataclass
class ShapeRunResult:
    shape_id: str
    correctness_passed: bool
    correctness_reason: str | None
    measured_ns: float | None
    baseline_ns: float | None
    metamorphic_results: list = field(default_factory=list)


@dataclass
class TaskEvalResult:
    task_id: str
    static_checks_passed: bool
    static_findings: list = field(default_factory=list)
    shape_results: list[ShapeRunResult] = field(default_factory=list)
    aggregate_score: float = 0.0
    coverage: float = 0.0
    rank_eligible: bool = False
    # codex round-3 N-P0-1: explicit per-run status fields. Downstream
    # analyze() must NOT infer correctness from static_checks_passed alone.
    correctness_status: str = "error"  # "ok" | "fail" | "error"
    runtime_status: str = "error"      # "ok" | "timeout" | "oom" | "compile_fail" | "invalid_timing" | "error"
    performance_status: str = "error"  # "ok" | "suspicious" | "quarantined" | "error"
    performance_flags: list[str] = field(default_factory=list)
    fallback_detected: bool = False

    def shape_measurements(self) -> list[scoring.ShapeMeasurement]:
        return [
            scoring.ShapeMeasurement(
                task_id=self.task_id,
                shape_id=r.shape_id,
                passed=r.correctness_passed,
                measured_ns=r.measured_ns,
                baseline_ns=r.baseline_ns,
                failure_reason=r.correctness_reason,
            )
            for r in self.shape_results
        ]


def evaluate(
    task_id: str,
    solution_path: Path,
    backend: str = "npu",
) -> TaskEvalResult:
    """Top-level entrypoint. Phase 2.A is skeleton — runner orchestration
    completes in Phase 2.B."""
    task = load_task(task_id)
    solution_source = Path(solution_path).read_text()

    static_report = static_checker.check_solution(
        solution_source,
        required_checks=task.meta.get("runner_checks", {}).get("static_checks", []),
    )

    result = TaskEvalResult(
        task_id=task_id,
        static_checks_passed=not static_report.blocked,
        static_findings=static_report.findings,
    )

    if static_report.blocked:
        result.correctness_status = "fail"
        result.runtime_status = "ok"               # static-only path didn't run anything
        result.performance_status = "ok"
        return result

    # Phase 2.B: grade correctness. Oracle (trusted) runs in-process;
    # candidate runs in an isolated subprocess (process_isolation). Per
    # public shape, every output tensor is compared with tolerance.compare.
    # Timing/performance needs NPU hardware (GPU-machine layer) so
    # performance_status stays "error" here — never falsely "ok".
    try:
        import importlib.util

        import yaml

        from . import inputs as _inputs
        from . import process_isolation, tolerance

        shapes_path = task.task_dir / "shapes.yaml"
        oracle_path = task.task_dir / "oracle_pytorch.py"
        oracle_src = oracle_path.read_text() if oracle_path.exists() else ""
        shapes_txt = shapes_path.read_text() if shapes_path.exists() else ""

        gradeable = (
            shapes_path.exists() and oracle_path.exists()
            and "raise NotImplementedError" not in oracle_src
            and "PHASE-C-MANUAL" not in shapes_txt
        )
        if gradeable:
            shapes_doc = yaml.safe_load(shapes_txt)
            pub, _ = _inputs.shape_ids(shapes_doc)
            spec = importlib.util.spec_from_file_location(
                f"_oracle_{task_id}", oracle_path)
            omod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(omod)

            tol_cls = tolerance.load_policy().get(task.tolerance_class)
            cand = process_isolation.run_candidate_isolated(
                task.task_dir, solution_path, pub, seed=0)

            for sid in pub:
                reason: str | None = None
                passed = False
                try:
                    expected = omod.run(
                        _inputs.build_inputs(shapes_doc, sid, seed=0))
                except Exception as e:  # noqa: BLE001
                    reason = f"oracle error: {e}"
                    result.shape_results.append(
                        ShapeRunResult(sid, False, reason, None, None))
                    continue
                scr = cand.per_shape.get(sid)
                if scr is None or not scr.ran:
                    reason = (scr.error if scr and scr.error
                              else cand.load_error or "candidate did not run")
                elif tol_cls is None:
                    reason = f"unknown tolerance_class {task.tolerance_class!r}"
                else:
                    okk = True
                    for key, exp in expected.items():
                        if not torch.is_tensor(exp):
                            continue
                        act = scr.outputs.get(key)
                        if act is None:
                            okk = False
                            reason = f"missing output {key!r}"
                            break
                        cmp = tolerance.compare(act, exp, tol_cls)
                        if not cmp.passed:
                            okk = False
                            reason = f"{key}: {cmp.reason}"
                            break
                    passed = okk
                result.shape_results.append(
                    ShapeRunResult(sid, passed, reason, None, None))

            ran = [r for r in result.shape_results]
            if ran:
                result.runtime_status = (
                    "timeout" if cand.timed_out and not cand.ran_any
                    else "ok" if cand.ran_any else "error")
                result.correctness_status = (
                    "ok" if all(r.correctness_passed for r in ran)
                    else "fail")
    except Exception as e:  # noqa: BLE001
        logger.debug("phase-2b grading skipped %s: %s", task_id, e)

    shape_meas = result.shape_measurements()
    if shape_meas:
        result.aggregate_score = scoring.capped_geomean_native(
            shape_meas, s_native_size=max(len(shape_meas), 1)
        )
        result.coverage = scoring.coverage_for_ranking(shape_meas)
        result.rank_eligible = result.coverage >= scoring.MIN_COVERAGE_FOR_RANKING
    # performance_status intentionally left non-"ok": no NPU timing here.
    return result
