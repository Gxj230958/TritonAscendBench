"""scoring.py — primary metric per specs/scoring_spec.md + DESIGN-V2 §9.

This is a GPU-triton → NPU-triton **migration** benchmark. Performance is
scored as a **910B Roofline normalised efficiency** (DESIGN-V2 §5.6 / §9):
per shape the cost model gives per-class FLOPs ``W_c`` and per-resource
bytes ``M_r``; the 910B peak table gives peak compute ``π_c`` and peak
bandwidth ``β_r``; the theoretical-latency floor is

    t_ref = max( max_c W_c / π_c , max_r M_r / β_r )

and the per-shape efficiency is ``e = min(P_CAP, t_ref / t_npu)``.
``*_native`` symbol/key names are retained for output-schema stability and
denote the **NPU-triton track**, not AscendC.

    s_t = ε                         if any of 4 statuses != ok OR fallback detected
    s_t = (∏_s e(s))^(1/n)          otherwise   (geomean of per-shape e)

The per-shape ``e`` is computed here; the per-shape ``e`` → task-level
``s_t`` geomean is aggregated in ``eval.py`` and lands on
``TaskMeasurement.s_t``. ``s_t_value`` consumes that pre-aggregated
``s_t``; if absent it falls back to the legacy ``baseline_ns/measured_ns``
ratio so older call paths keep working.

    capped_geomean_native = exp( (1/|catalogue|) Σ_t log(s_t) )
    FastP_native@1.0 = |{t : s_t ≥ 1.0}| / |catalogue|
    FastP_native@1.5 = |{t : s_t ≥ 1.5}| / |catalogue|

`|catalogue|` (passed as `s_native_size`) is the **fixed** denominator —
the full scored task set, not just attempted ones. Failures contribute
ε to the geomean, NOT zero. Ranking eligibility: coverage ≥ 0.70.

``task_status`` (``evaluated`` / ``unsupported`` / ``blocked``, DESIGN §5.7
/ W4.3) is orthogonal to the four ok/fail statuses: it says whether a task
*could* be evaluated in this mode, not how it scored. The two non-evaluated
states are NOT interchangeable for the denominator (W4.3 ④ / R8-1):

* ``unsupported`` — a self-reported-mode ``upstream_oracle_required`` task
  has no provider golden, so it is **out of the scored set** entirely:
  dropped from numerator AND denominator. (`analyze` derives the
  self-reported denominator as ``|S| − upstream_oracle_required`` and never
  even builds a ``TaskMeasurement`` for these; the drop here is a
  defensive twin for direct ``scoring`` callers.)
* ``blocked`` — a verified-mode task whose golden is missing/corrupt. The
  verified-mode denominator is the **fixed ``|S|``** (DESIGN §9); a
  ``blocked`` task MUST stay in it and is scored as a **failure** (``ε`` in
  the geomean, threshold-not-met for FastP, not an ``ok`` for coverage).
  BLOCKED is the publish-gate's target — it must not escape the
  denominator (no-subset-escape).

Scored-set membership is therefore mode-aware and decided upstream by
``analyze`` (it passes the correct fixed denominator); ``scoring`` only
distinguishes ``unsupported`` (leaves the set) from ``blocked`` /
``evaluated`` (stay, ``blocked`` as ε).
"""
from __future__ import annotations

import logging
import math
import statistics
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field

from .cost_model import CostModel, CostModelError, FormulaEnv

logger = logging.getLogger(__name__)

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

# W1.4 / W4.3: `task_status` is an INDEPENDENT field, orthogonal to the four
# ok/fail statuses above. It records whether the task was evaluable *at all*
# in the run's mode (DESIGN §5.7):
#   * "evaluated"   — graded normally; numeric perf fields are meaningful.
#   * "unsupported" — no provider golden for this mode (self-reported mode,
#                     the upstream_oracle_required tasks). W4.3 semantics.
#   * "blocked"     — golden missing where it should exist (verified mode,
#                     fail-closed). W4.3 semantics.
# W1.4 only introduces the field (default "evaluated"); the unsupported /
# blocked producers land in W4.3. Scoring already skips non-"evaluated"
# tasks so the field is wired end-to-end now.
VALID_TASK_STATUS = {"evaluated", "unsupported", "blocked"}
TASK_STATUS_EVALUATED = "evaluated"
TASK_STATUS_UNSUPPORTED = "unsupported"
TASK_STATUS_BLOCKED = "blocked"


@dataclass
class TaskMeasurement:
    """One task's aggregated result across its shapes.

    W1.4: `s_t` is the task-level Roofline score — the geomean of the
    per-shape efficiencies ``e``, aggregated in ``eval.py`` (DESIGN §9).
    When `s_t` is present, scoring consumes it directly. `measured_ns` /
    `baseline_ns` are kept for back-compat and provenance: per DESIGN §9
    provenance they carry ``t_npu`` / ``t_ref`` respectively; when `s_t`
    is absent `s_t_value` falls back to the legacy ``baseline_ns/measured_ns``
    ratio so older call paths still produce a number.

    `task_status` (DESIGN §5.7 / W4.3) records whether the task was
    evaluable at all in the run's mode; a non-"evaluated" task carries no
    valid perf numbers and is skipped by every aggregate (not ε-collapsed).
    """
    task_id: str
    correctness_status: str = "ok"
    runtime_status: str = "ok"
    performance_status: str = "ok"
    performance_flags: list[str] = field(default_factory=list)
    fallback_detected: bool = False
    measured_ns: float | None = None
    baseline_ns: float | None = None
    # W1.4: task-level Roofline score = geomean of per-shape e (DESIGN §9).
    s_t: float | None = None
    # W1.4 / W4.3: orthogonal evaluability status. Default "evaluated".
    task_status: str = TASK_STATUS_EVALUATED
    # R9-1: True iff the Roofline path had the data it needs to score this
    # task — a cost model AND verified π/β for every compute-class/resource
    # the cost model references. False is a *benchmark-data* gap (the
    # cost_model is missing, or a referenced peak is still an
    # ``[unverified]`` placeholder — Phase 1-2 backfill not yet complete),
    # NOT a candidate failure: such a task must NOT be ε-collapsed against
    # the candidate's score. ``aggregate_perf`` keeps these out of the
    # geomean and counts them in a dedicated "Roofline data incomplete"
    # column. A genuine candidate failure (blocked / non-ok / fallback)
    # leaves this True and is still scored as ε — its data was fine, the
    # *candidate* failed.
    roofline_data_complete: bool = True
    # True iff every referenced peak is vendor-official / on-device-measured.
    # Gates absolute "% of Roofline" claims; ranking only needs
    # roofline_data_complete (an academic-spec peak ranks, not absolute).
    absolute_verified: bool = False

    @property
    def all_statuses_ok(self) -> bool:
        return (
            self.correctness_status == "ok"
            and self.runtime_status == "ok"
            and self.performance_status == "ok"
            and not self.fallback_detected
        )

    @property
    def is_evaluated(self) -> bool:
        """True iff this task was graded in the run's mode — i.e. its
        numeric perf fields are meaningful and may enter an aggregate as a
        real (non-ε) term."""
        return self.task_status == TASK_STATUS_EVALUATED

    @property
    def in_scored_set(self) -> bool:
        """True iff this task is part of this mode's scored denominator
        (W4.3 ④ / R8-1).

        ``evaluated`` and ``blocked`` are both in the set: ``evaluated``
        scores on its real perf signal, ``blocked`` scores as a failure
        (``ε``) — it must NOT escape the fixed ``|S|`` denominator (it is
        the publish-gate's target). Only ``unsupported`` is out of the set
        (no provider golden for this mode); it is dropped from numerator
        AND denominator.
        """
        return self.task_status != TASK_STATUS_UNSUPPORTED


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


# ─────────────────────────────────────────────────────────────────────────────
# W1.4 — Roofline theoretical-latency baseline + per-shape efficiency.
# ─────────────────────────────────────────────────────────────────────────────
class RooflineError(ValueError):
    """Raised when a Roofline ``t_ref`` cannot be computed from a sound
    cost model + verified peak table (fail-closed; see ``compute_t_ref``)."""


@dataclass
class DurationStats:
    """``durations`` (per-trial host_ns) reduced to the disclosure stats
    DESIGN §6.6 requires alongside every shape: median / p99 / cv / trials."""
    trials: int
    median: float | None
    p99: float | None
    cv: float | None


def duration_stats(durations: Iterable[float]) -> DurationStats:
    """Reduce a per-trial ``host_ns`` series to median / p99 / cv / trials.

    * ``median`` — robust central tendency, used as ``t_npu`` for scoring.
    * ``p99``    — tail latency (DESIGN §6.6 disclosure).
    * ``cv``     — coefficient of variation (stddev / mean); a noise gauge.

    Non-finite / non-positive samples are dropped before reduction so a
    stray ``0`` / ``inf`` cannot poison the stats. An empty (post-filter)
    series yields all-``None`` stats with ``trials == 0``.
    """
    clean = [float(d) for d in durations
             if d is not None and math.isfinite(d) and d > 0]
    n = len(clean)
    if n == 0:
        return DurationStats(trials=0, median=None, p99=None, cv=None)
    clean.sort()
    median = statistics.median(clean)
    # p99: nearest-rank on the sorted series (index clamped into range).
    idx = min(n - 1, max(0, math.ceil(0.99 * n) - 1))
    p99 = clean[idx]
    mean = sum(clean) / n
    if n >= 2 and mean > 0:
        cv = statistics.stdev(clean) / mean
    else:
        cv = 0.0 if n == 1 else None
    return DurationStats(trials=n, median=median, p99=p99, cv=cv)


def compute_t_ref(
    cost_model: CostModel,
    knobs: Mapping[str, float],
    compute_peaks: Mapping[str, float],
    resource_peaks: Mapping[str, float],
    *,
    dtype_size: float | None = None,
) -> float:
    """Roofline theoretical-latency floor for one shape (DESIGN §5.6 / §9).

        t_ref = max( max_c W_c / π_c , max_r M_r / β_r )

    ``W_c`` / ``M_r`` come from evaluating the task's cost-model formulas
    against the shape's size ``knobs`` (+ derived symbols, e.g. a real
    ``dtype_size``); ``π_c`` / ``β_r`` are the 910B peaks (scoring passes
    ``require_verified=True`` peak tables, so an ``[unverified]`` placeholder
    never reaches here as a usable number).

    Fail-closed: a formula referencing a class/resource the peak table does
    not cover, or a non-positive / non-finite peak, raises
    :class:`RooflineError` — scoring must surface that, never silently use
    bad data.

    Returns ``t_ref`` in seconds (FLOP / (FLOP/s) and byte / (byte/s)).
    """
    derived = {"dtype_size": dtype_size} if dtype_size is not None else None
    env = FormulaEnv(dict(knobs), derived=derived)

    terms: list[float] = []
    for compute_class, formula in cost_model.flops.items():
        if compute_class not in compute_peaks:
            raise RooflineError(
                f"cost_model.flops class {compute_class!r} has no peak in the "
                f"compute peak table (have {sorted(compute_peaks)})"
            )
        pi = compute_peaks[compute_class]
        if not math.isfinite(pi) or pi <= 0:
            raise RooflineError(
                f"compute peak for {compute_class!r} is non-positive/"
                f"non-finite ({pi!r}) — unverified placeholder?"
            )
        try:
            w_c = env.evaluate(formula)
        except CostModelError as e:
            raise RooflineError(
                f"cost_model.flops.{compute_class}: {e}"
            ) from e
        if not math.isfinite(w_c) or w_c < 0:
            raise RooflineError(
                f"cost_model.flops.{compute_class} evaluated to {w_c!r}"
            )
        terms.append(w_c / pi)

    for resource, formula in cost_model.bytes.items():
        if resource not in resource_peaks:
            raise RooflineError(
                f"cost_model.bytes resource {resource!r} has no peak in the "
                f"resource peak table (have {sorted(resource_peaks)})"
            )
        beta = resource_peaks[resource]
        if not math.isfinite(beta) or beta <= 0:
            raise RooflineError(
                f"resource peak for {resource!r} is non-positive/"
                f"non-finite ({beta!r}) — unverified placeholder?"
            )
        try:
            m_r = env.evaluate(formula)
        except CostModelError as e:
            raise RooflineError(
                f"cost_model.bytes.{resource}: {e}"
            ) from e
        if not math.isfinite(m_r) or m_r < 0:
            raise RooflineError(
                f"cost_model.bytes.{resource} evaluated to {m_r!r}"
            )
        terms.append(m_r / beta)

    if not terms:
        raise RooflineError(
            "cost_model has neither flops nor bytes terms — t_ref undefined"
        )
    return max(terms)


def shape_efficiency(
    t_ref: float, t_npu: float, p_cap: float = P_CAP
) -> float:
    """Per-shape Roofline efficiency ``e = min(P_CAP, t_ref / t_npu)``.

    Theory bounds ``e`` to ``(0, 1]`` (``t_ref`` is the floor); a value
    ``> 1`` is a cost-model over-estimate (t_ref above the measured latency)
    or measurement noise, and is clamped by ``P_CAP``. Non-finite / non-positive inputs raise ``ValueError`` so a
    caller cannot smuggle a NaN past the cap (``min(p_cap, nan) == p_cap``
    on most platforms — a gaming vector).
    """
    if not math.isfinite(t_ref) or t_ref <= 0:
        raise ValueError(f"t_ref must be a positive finite number, got {t_ref!r}")
    if not math.isfinite(t_npu) or t_npu <= 0:
        raise ValueError(f"t_npu must be a positive finite number, got {t_npu!r}")
    ratio = t_ref / t_npu
    if not math.isfinite(ratio):
        raise ValueError(f"t_ref/t_npu is non-finite ({ratio!r})")
    return min(p_cap, ratio)


def geomean(values: Iterable[float]) -> float:
    """Geometric mean of a non-empty positive series — the per-shape ``e``
    → task-level ``s_t`` reducer (DESIGN §9: ``s_t = (∏_s e)^(1/n)``).

    Raises ``ValueError`` on an empty series or a non-positive element
    (``log`` of a non-positive value is undefined).
    """
    vals = list(values)
    if not vals:
        raise ValueError("geomean of an empty series is undefined")
    for v in vals:
        if not math.isfinite(v) or v <= 0:
            raise ValueError(f"geomean element must be positive finite, got {v!r}")
    return math.exp(sum(math.log(v) for v in vals) / len(vals))


def perf_measurable(m: TaskMeasurement) -> bool:
    """True iff this task can yield a real Roofline efficiency: all
    statuses ok, the task was ``evaluated``, AND a usable perf signal is
    present (the W1.4 task-level ``s_t``, or the legacy positive-finite
    ``baseline_ns``/``measured_ns`` pair).

    Post-AscendC (codex P0-1): a task with ok statuses but NO perf signal
    is perf-**unavailable**, NOT a failure. Callers must exclude it from
    the perf metric rather than ε-collapse it (which would make a correct
    fast solution indistinguishable from total failure).
    """
    if not m.is_evaluated:
        return False
    if not m.all_statuses_ok:
        return False
    # W1.4: a pre-aggregated Roofline s_t is itself a usable perf signal.
    if m.s_t is not None:
        return math.isfinite(m.s_t) and m.s_t > 0
    if m.measured_ns is None or m.baseline_ns is None:
        return False
    if not (math.isfinite(m.measured_ns) and math.isfinite(m.baseline_ns)):
        return False
    return m.measured_ns > 0 and m.baseline_ns > 0


def s_t_value(m: TaskMeasurement, p_cap: float = P_CAP, eps: float = EPSILON) -> float:
    """Per-task Roofline score with epsilon-fill on failure (DESIGN §9).

    W1.4: when `m.s_t` is present it is the geomean of the per-shape
    Roofline efficiencies (aggregated in ``eval.py``); scoring consumes it
    directly, clamped to ``[eps, p_cap]``. When `m.s_t` is absent the
    legacy ``baseline_ns/measured_ns`` ratio path is used so older call
    sites keep producing a number.

    NaN / Inf collapse to ε (`min(p_cap, nan) == p_cap` on most platforms —
    a gaming vector).

    W4.3 ④ / R8-1: a ``blocked`` task has no usable golden — it is scored
    as a **failure** (``ε``), since it stays in the fixed ``|S|``
    denominator and must not escape it. An ``unsupported`` task is out of
    the scored set entirely; ``capped_geomean_native`` drops it before
    calling here, but defensively it also collapses to ``ε``.
    """
    if not m.is_evaluated:
        return eps
    if not m.all_statuses_ok:
        return eps
    # W1.4: prefer the pre-aggregated task-level Roofline score.
    if m.s_t is not None:
        if not math.isfinite(m.s_t) or m.s_t <= 0:
            return eps
        return max(eps, min(p_cap, m.s_t))
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

    W4.3 ④ / R8-1: the two non-evaluated ``task_status`` values are NOT
    treated alike:

    * ``unsupported`` — out of this mode's scored set (no provider golden):
      **dropped from numerator AND the implied item count**. The
      ``s_native_size`` denominator passed in is already
      ``|S| − unsupported`` for this mode, so the drop here keeps the two
      consistent for direct callers.
    * ``blocked`` — stays in the fixed ``|S|`` denominator and is scored as
      a **failure** (``ε``). It must NOT escape the denominator — BLOCKED is
      the publish-gate's target (no-subset-escape). ``s_t_value`` collapses
      a ``blocked`` task to ``ε``.

    Raises ValueError on duplicate task_id (codex round-2 P1).
    """
    deduped = _dedupe_by_task_id(task_measurements)

    s_t_values: list[float] = []
    n_unsupported = 0
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
            # W4.3 ④ / R8-1: only `unsupported` leaves the scored set; a
            # `blocked` task stays in the denominator and `s_t_value`
            # collapses it to ε (a failure, not a denominator escape).
            if not m.in_scored_set:
                n_unsupported += 1
                continue
            s = s_t_value(m, p_cap=p_cap, eps=eps)
        s_t_values.append(s)

    if s_native_size is not None:
        # The denominator is the fixed |S| for this mode minus only the
        # `unsupported` tasks (out of the scored set); `blocked` tasks
        # already contributed an ε term above. Remaining missing tasks
        # (never produced a result) contribute ε.
        target = s_native_size - n_unsupported
        if target > len(s_t_values):
            s_t_values.extend([eps] * (target - len(s_t_values)))

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
    threshold-not-met. W4.3 ④ / R8-1: ``unsupported`` tasks are out of the
    scored set — excluded from both numerator and denominator. ``blocked``
    tasks stay in the fixed ``|S|`` denominator and count as
    threshold-not-met (``s_t_value`` collapses them to ``ε < threshold``);
    they must not escape the denominator. Duplicate task_id raises
    ValueError.
    """
    deduped = _dedupe_by_task_id(task_measurements)
    scored = [m for m in deduped if m.in_scored_set]
    n_unsupported = len(deduped) - len(scored)
    values = [s_t_value(m, p_cap=p_cap, eps=eps) for m in scored]
    n_meet = sum(1 for v in values if v >= threshold)
    if s_native_size is not None:
        denom = s_native_size - n_unsupported
    else:
        denom = max(len(values), 1)
    if denom <= 0:
        return 0.0
    return min(n_meet / denom, 1.0)


def native_coverage(
    task_measurements: Iterable[TaskMeasurement],
    s_native_size: int,
) -> float:
    """Per scoring_spec.md §Ranking Eligibility / DESIGN §9.

    coverage = #{t ∈ S_native : correctness_t == "ok" ∧ ¬fallback_t} / |S_native|

    W2.2: the gate predicate explicitly excludes ``fallback_detected``
    (DESIGN §6.3 / §6.7) — a correct-but-fallback solution must not clear
    the correctness gate. ``eval`` already normalises ``fallback_detected``
    to ``correctness_status="fail"`` before emit, so on a well-formed
    result the ``correctness == "ok"`` test alone would suffice; the
    ``¬fallback_detected`` conjunct here is a deliberate second line of
    defence in case a result reaches scoring without that normalisation.

    W4.3 ④ / R8-1: ``unsupported`` tasks (no provider golden for this
    mode) are out of the scored set — excluded from both the numerator and
    the ``|S_native|`` denominator. ``blocked`` tasks stay in the fixed
    ``|S_native|`` denominator and are counted as **not correct** (a
    blocked task has no usable golden, so it cannot clear the correctness
    gate — it must not escape the denominator). Duplicate task_id raises
    ValueError.
    """
    deduped = _dedupe_by_task_id(task_measurements)
    n_unsupported = sum(1 for m in deduped if not m.in_scored_set)
    n_correct = sum(
        1 for m in deduped
        if m.is_evaluated
        and m.correctness_status == "ok"
        and not m.fallback_detected
    )
    denom = max(s_native_size - n_unsupported, 1)
    return min(n_correct / denom, 1.0)


def is_rank_eligible(
    task_measurements: Iterable[TaskMeasurement],
    s_native_size: int,
) -> bool:
    return native_coverage(task_measurements, s_native_size) >= MIN_COVERAGE_FOR_RANKING


def aggregate_perf(
    task_measurements: list[TaskMeasurement],
    eligible_size: int,
) -> dict:
    """Honest perf aggregate (codex P0-1/P0-2, R9-1).

    DESIGN §9 / P6: the perf metric runs over the **fixed** scored
    denominator. A task that genuinely *failed* (``blocked`` / a non-ok
    correctness / runtime / performance status / a detected torch
    fallback) is a real failure — it stays in the fixed ``|S|``
    denominator and is scored as ``ε`` (R8-1: no-subset-escape; failures
    are NOT silently dropped). The earlier "no ``perf_measurable`` task ⇒
    short-circuit to ``unavailable`` with null metrics" rule wrongly
    discarded that ε-fill: a corpus of all-failed candidates would have
    reported ``perf_status="unavailable"`` instead of an honest ε-floored
    geomean.

    The one case that must NOT be ε-collapsed is a **Roofline-data gap**:
    a task whose ``cost_model`` is missing, or whose cost model references
    a π/β still held as an ``[unverified]`` placeholder (the Phase 1-2
    peak-table backfill is not yet complete). That is a *benchmark-data*
    incompleteness, not the candidate's fault — pressing the candidate's
    score with ε for it would be dishonest. Such a task is signalled by
    ``roofline_data_complete is False``; ``aggregate_perf`` keeps it OUT
    of the geomean and counts it in ``roofline_data_incomplete_count``.

    Status resolution:

    * Some task is **scorable** (a real perf signal) OR a **real
      failure** ⇒ ``perf_status="ok"`` — the geomean / FastP run over the
      fixed denominator with ε for the failures, the Roofline-data-gap
      tasks honestly excluded from both numerator and denominator.
    * EVERY in-scored-set task is only uncomputable because of a
      Roofline-data gap ⇒ ``perf_status="roofline_data_incomplete"`` with
      null metrics — the benchmark cannot yet produce a perf number, and
      that is reported as a data-completeness issue, never as a candidate
      score.

    ``unsupported`` tasks (no provider golden for this mode) are out of
    the scored set entirely — dropped before any of the above, exactly as
    ``capped_geomean_native`` / ``fastp_at_threshold`` do.

    `eligible_size` must be the count of *score-eligible* tasks (real
    gradeable oracle), NOT the full catalogue — using the full catalogue
    would systematically depress valid submissions with ε-padding for
    ungradeable tasks.
    """
    deduped = _dedupe_by_task_id(list(task_measurements))
    # `unsupported` tasks leave the scored set entirely (no provider
    # golden for this mode) — drop them before partitioning.
    in_set = [m for m in deduped
              if isinstance(m, TaskMeasurement) and m.in_scored_set]
    n_unsupported = len(deduped) - len(in_set)

    # A task is excluded from the perf geomean ONLY when it cannot be
    # scored AND that is a genuine Roofline-data gap (not a candidate
    # failure). A real failure (blocked / non-ok / fallback) is
    # `all_statuses_ok == False` (or not `is_evaluated`) and is kept —
    # `s_t_value` ε-collapses it.
    #
    # R10-1: a status-ok task with no perf signal is a Roofline-data gap
    # *only* when `roofline_data_complete is False` — i.e. the repo
    # metadata (the authority) says the cost model is missing or a
    # referenced peak is still an `[unverified]` placeholder. A status-ok
    # task that the repo metadata authoritatively marks
    # `roofline_data_complete is True` but whose result JSON nevertheless
    # carries no `s_t` (e.g. a submitter stripped it) is NOT a data gap:
    # it stays in the scored set and `s_t_value` ε-collapses it, so a
    # missing perf signal can never escape the fixed `|S|` denominator.
    def _is_real_failure(m: TaskMeasurement) -> bool:
        return not m.is_evaluated or not m.all_statuses_ok

    def _is_roofline_data_gap(m: TaskMeasurement) -> bool:
        if _is_real_failure(m):
            return False
        # status-ok, evaluated: a genuine data gap iff it carries no real
        # perf signal AND the repo metadata authoritatively says the
        # Roofline data is incomplete. Otherwise it is kept and ε-scored.
        return not perf_measurable(m) and not m.roofline_data_complete

    data_gap = [m for m in in_set if _is_roofline_data_gap(m)]
    scored = [m for m in in_set if not _is_roofline_data_gap(m)]

    if not scored:
        # Every in-scored-set task is uncomputable purely because the
        # Roofline benchmark data is not complete — there is no real
        # failure and nothing scorable. Report the data-completeness
        # state honestly; do NOT fabricate an ε-collapsed number.
        return {
            "perf_status": "roofline_data_incomplete",
            "perf_status_reason": (
                "no task has a computable Roofline score — every "
                "evaluated task is missing a cost model or references a "
                "peak still held as an [unverified] placeholder "
                "(benchmark peak-table / cost-model backfill incomplete)"
            ),
            "capped_geomean_native": None,
            "fastp_native_at_1_0": None,
            "fastp_native_at_1_5": None,
            "roofline_data_incomplete_count": len(data_gap),
        }

    # The fixed denominator is `|S|` for this mode minus the `unsupported`
    # tasks (out of the scored set) minus the Roofline-data-gap tasks
    # (honestly excluded — not the candidate's fault). Real failures stay
    # in and are ε-scored by `capped_geomean_native` / `fastp_at_threshold`.
    denom = max(eligible_size - n_unsupported - len(data_gap), len(scored))
    return {
        "perf_status": "ok",
        "perf_status_reason": None,
        "capped_geomean_native": capped_geomean_native(
            scored, s_native_size=denom),
        "fastp_native_at_1_0": fastp_at_threshold(
            scored, threshold=1.0, s_native_size=denom),
        "fastp_native_at_1_5": fastp_at_threshold(
            scored, threshold=1.5, s_native_size=denom),
        "roofline_data_incomplete_count": len(data_gap),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Legacy helpers (kept for the ShapeMeasurement-based older code path)
# ─────────────────────────────────────────────────────────────────────────────
def coverage_for_ranking(measurements: Iterable[ShapeMeasurement]) -> float:
    measurements = list(measurements)
    if not measurements:
        return 0.0
    n_pass = sum(1 for m in measurements if m.passed)
    return n_pass / len(measurements)
