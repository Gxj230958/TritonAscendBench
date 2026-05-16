"""scoring.py — primary metric per specs/scoring_spec.md.

Per v4.13 §5 + R12-P1-1/5 + R13-P1-1 + R16-P1 + R18-S5:

    s_t = ε                         if any of 4 statuses != ok OR fallback detected
    s_t = max(ε, min(P_CAP, baseline_ns / measured_ns))   otherwise

    capped_geomean_native = exp( (1/|S_native|) Σ_t log(s_t) )
    FastP_native@1.0 = |{t : s_t ≥ 1.0}| / |S_native|
    FastP_native@1.5 = |{t : s_t ≥ 1.5}| / |S_native|

Note `|S_native|` is the **fixed** denominator (all native-baseline tasks),
not just attempted ones. Failures contribute ε to the geomean, NOT zero.

Ranking eligibility: native_coverage ≥ 0.70 over `S_native`.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable

# ─────────────────────────────────────────────────────────────────────────────
# Constants from specs/scoring_spec.md
# ─────────────────────────────────────────────────────────────────────────────
P_CAP = 5.0                      # speedup cap per spec §sanity_bounds
EPSILON = 0.01                   # epsilon contribution for failed shapes/tasks
MIN_COVERAGE_FOR_RANKING = 0.70

VALID_CORRECTNESS = {"ok", "fail", "error"}
# "error" is added beyond the scoring_spec enum so a Phase 2.A skeleton
# `evaluate()` (which has not yet wired its runner subprocess) can emit
# a structured "not-yet-run" status without violating the schema.
VALID_RUNTIME = {"ok", "timeout", "oom", "compile_fail", "invalid_timing", "error"}
VALID_PERFORMANCE = {"ok", "suspicious", "quarantined", "error"}


@dataclass
class TaskMeasurement:
    """One task's aggregated result across its shapes.

    `measured_ns` and `baseline_ns` are typically per-task aggregates
    (e.g. geomean across that task's shapes) computed upstream of scoring.
    """
    task_id: str
    correctness_status: str = "ok"
    runtime_status: str = "ok"
    performance_status: str = "ok"
    performance_flags: list[str] = field(default_factory=list)
    fallback_detected: bool = False
    measured_ns: float | None = None
    baseline_ns: float | None = None

    @property
    def all_statuses_ok(self) -> bool:
        return (
            self.correctness_status == "ok"
            and self.runtime_status == "ok"
            and self.performance_status == "ok"
            and not self.fallback_detected
        )


# Back-compat alias used by older eval.py paths.
@dataclass
class ShapeMeasurement:
    """Per-shape correctness + timing measurement (sub-aggregate of TaskMeasurement)."""
    task_id: str
    shape_id: str
    passed: bool
    measured_ns: float | None
    baseline_ns: float | None
    failure_reason: str | None = None


def s_t_value(m: TaskMeasurement, p_cap: float = P_CAP, eps: float = EPSILON) -> float:
    """Per-task speedup score with epsilon-fill on failure.

    codex round-6 #5: NaN / Inf must collapse to ε (or be rejected upstream);
    `min(p_cap, nan)` returns `p_cap` on most platforms — gaming vector.
    """
    if not m.all_statuses_ok:
        return eps
    if m.measured_ns is None or m.baseline_ns is None:
        return eps
    if not (math.isfinite(m.measured_ns) and math.isfinite(m.baseline_ns)):
        return eps
    if m.measured_ns <= 0 or m.baseline_ns <= 0:
        return eps
    ratio = m.baseline_ns / m.measured_ns
    if not math.isfinite(ratio):
        return eps
    return max(eps, min(p_cap, ratio))


# Legacy single-shape primitive kept for back-compat with tests.
def fastp_native(
    measured_ns: float, baseline_ns: float, p_cap: float = P_CAP
) -> float:
    """Clamped speedup ratio with cap = P_CAP. Lower bound is 0 (not ε)."""
    if measured_ns <= 0:
        raise ValueError("measured_ns must be positive")
    if baseline_ns <= 0:
        raise ValueError("baseline_ns must be positive")
    return min(max(baseline_ns / measured_ns, 0.0), p_cap)


def _dedupe_by_task_id(
    measurements: Iterable[TaskMeasurement | ShapeMeasurement],
) -> list[TaskMeasurement | ShapeMeasurement]:
    """Reject duplicate identity (codex round-2 P1, codex round-3 P1).

    For `TaskMeasurement`: duplicate `task_id` would inflate coverage
    above 1.0, so we raise.

    For `ShapeMeasurement`: many shapes per task is legitimate, so the
    dedupe key is `(task_id, shape_id)` and only exact (task_id, shape_id)
    duplicates raise.
    """
    seen_tasks: set[str] = set()
    seen_shapes: set[tuple[str, str]] = set()
    out: list[TaskMeasurement | ShapeMeasurement] = []
    for m in measurements:
        if isinstance(m, ShapeMeasurement):
            key = (m.task_id, m.shape_id)
            if key in seen_shapes:
                raise ValueError(
                    f"duplicate (task_id, shape_id) in shape measurements: "
                    f"{key!r}"
                )
            seen_shapes.add(key)
            out.append(m)
            continue
        tid = getattr(m, "task_id", "")
        if not tid:
            out.append(m)
            continue
        if tid in seen_tasks:
            raise ValueError(
                f"duplicate task_id in TaskMeasurement list: {tid!r} — "
                "scoring requires one TaskMeasurement per task"
            )
        seen_tasks.add(tid)
        out.append(m)
    return out


def capped_geomean_native(
    task_measurements: Iterable[TaskMeasurement | ShapeMeasurement],
    s_native_size: int | None = None,
    p_cap: float = P_CAP,
    eps: float = EPSILON,
) -> float:
    """Geomean over **|S_native|** tasks. Missing or failed tasks contribute ε.

    `s_native_size` is the cardinality of S_native (the fixed denominator).
    If omitted, falls back to the number of items provided — but that
    disables the "missing task = ε" rule and is only safe when caller has
    already padded the list to |S_native|.

    Raises ValueError on duplicate task_id (codex round-2 P1).
    """
    deduped = _dedupe_by_task_id(task_measurements)

    s_t_values: list[float] = []
    for m in deduped:
        if isinstance(m, ShapeMeasurement):
            tm = TaskMeasurement(
                task_id=m.task_id,
                correctness_status="ok" if m.passed else "fail",
                measured_ns=m.measured_ns,
                baseline_ns=m.baseline_ns,
            )
            s = s_t_value(tm, p_cap=p_cap, eps=eps)
        else:
            s = s_t_value(m, p_cap=p_cap, eps=eps)
        s_t_values.append(s)

    if s_native_size is not None and s_native_size > len(s_t_values):
        missing = s_native_size - len(s_t_values)
        s_t_values.extend([eps] * missing)

    if not s_t_values:
        return 0.0
    return math.exp(sum(math.log(s) for s in s_t_values) / len(s_t_values))


def fastp_at_threshold(
    task_measurements: Iterable[TaskMeasurement],
    threshold: float,
    s_native_size: int | None = None,
    p_cap: float = P_CAP,
    eps: float = EPSILON,
) -> float:
    """Fraction of S_native tasks with s_t ≥ threshold.

    Missing tasks (s_native_size > len(task_measurements)) count as
    threshold-not-met. Duplicate task_id raises ValueError.
    """
    deduped = _dedupe_by_task_id(task_measurements)
    values = [s_t_value(m, p_cap=p_cap, eps=eps) for m in deduped]
    n_meet = sum(1 for v in values if v >= threshold)
    denom = s_native_size if s_native_size is not None else max(len(values), 1)
    if denom == 0:
        return 0.0
    return min(n_meet / denom, 1.0)


def native_coverage(
    task_measurements: Iterable[TaskMeasurement],
    s_native_size: int,
) -> float:
    """Per scoring_spec.md §Ranking Eligibility.

    coverage = #{t ∈ S_native : correctness_status_t == "ok"} / |S_native|
    Duplicate task_id raises ValueError.
    """
    deduped = _dedupe_by_task_id(task_measurements)
    n_correct = sum(
        1 for m in deduped if m.correctness_status == "ok"
    )
    return min(n_correct / max(s_native_size, 1), 1.0)


def is_rank_eligible(
    task_measurements: Iterable[TaskMeasurement],
    s_native_size: int,
) -> bool:
    return native_coverage(task_measurements, s_native_size) >= MIN_COVERAGE_FOR_RANKING


# ─────────────────────────────────────────────────────────────────────────────
# Legacy helpers (kept for the ShapeMeasurement-based older code path)
# ─────────────────────────────────────────────────────────────────────────────
def coverage_for_ranking(measurements: Iterable[ShapeMeasurement]) -> float:
    measurements = list(measurements)
    if not measurements:
        return 0.0
    n_pass = sum(1 for m in measurements if m.passed)
    return n_pass / len(measurements)
