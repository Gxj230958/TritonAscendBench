"""adversarial.py — canary detection + metamorphic-check execution.

Implements:
- Metamorphic check runner (per task's `meta.runner_checks.metamorphic_checks`)
- Canary trap detection (per `specs/canary_task_design.md` C1–C10)
- Hidden adversarial shape rotation (R18-S5; sealed shape access protocol)

Canary traps (live inside `tasks/_canary/`) are tasks that look like real
ones but have a known wrong oracle or impossible shape. A submitter that
passes them is flagged as gaming the runner.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class MetamorphicResult:
    name: str
    passed: bool
    detail: str | None = None


@dataclass
class CanaryFinding:
    canary_id: str
    triggered: bool
    severity: str          # "soft" | "hard"
    message: str


def run_metamorphic_checks(
    checks: list[dict[str, Any]],
    transformations: dict[str, Callable[..., MetamorphicResult]],
) -> list[MetamorphicResult]:
    """Run each declared check.

    `transformations` maps a check's `name` to the actual transformation +
    re-run + comparison callable. The runner provides these per task.
    """
    out: list[MetamorphicResult] = []
    for spec in checks:
        name = spec["name"]
        fn = transformations.get(name)
        if fn is None:
            out.append(MetamorphicResult(name, False, "no transformation registered"))
            continue
        try:
            out.append(fn())
        except Exception as exc:                       # noqa: BLE001
            out.append(MetamorphicResult(name, False, f"raised: {type(exc).__name__}: {exc}"))
    return out


def check_canaries(
    submitted_results: dict[str, bool], canary_truth: dict[str, bool]
) -> list[CanaryFinding]:
    """Compare submitter outputs on canary tasks against ground truth.

    Args:
        submitted_results: canary_id → did the submission claim PASS on it?
        canary_truth: canary_id → should this canary be PASSing if the
            submission is honest? (Some canaries should fail; some are
            intentionally trivial and should pass.)
    """
    out: list[CanaryFinding] = []
    for cid, expected in canary_truth.items():
        got = submitted_results.get(cid)
        if got is None:
            continue
        if got != expected:
            severity = "hard" if cid.startswith(("C1", "C2", "C3")) else "soft"
            out.append(
                CanaryFinding(
                    canary_id=cid,
                    triggered=True,
                    severity=severity,
                    message=f"canary {cid}: expected pass={expected}, got pass={got}",
                )
            )
        else:
            out.append(
                CanaryFinding(cid, False, "soft", f"canary {cid}: ok")
            )
    return out
