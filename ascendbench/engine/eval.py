"""eval.py — orchestrate one (task, solution, backend) evaluation.

Pipeline:
    1. Load task (dataset.load_task)
    2. Static checks on solution.py (static_checker)
    3. Spawn isolated candidate passes (process_isolation.py):
         a. Correctness pass — one subprocess, one inputs dict per shape.
         b. SIGKILL + scrub (compile cache / /tmp / device memory).
         c. Timing pass — a SEPARATE subprocess, per-trial fresh inputs.
    4. Per shape:
         a. Runner materialises inputs (correctness: single; timing:
            per-trial) and injects them down the IPC.
         b. Run oracle in this trusted process for the golden output.
         c. Compare the candidate's correctness-pass output with
            tolerance.compare.
         d. Re-verify the timing pass's runner-sampled outputs against
            golden (sampled-output cheat-path closure).
         e. Arbitrate the timing-pass durations per
            docs/reference/timing_arbitration.md.
    5. Score via scoring.capped_geomean_native
    6. Emit round_result.json

: this module only *orchestrates* the two passes and records the
returned ``durations`` / ``sampled_outputs`` — it never times candidate
code in the parent process (timing is host-`CLOCK_MONOTONIC_RAW` inside
the timing-pass child's trusted-shim segment).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import torch

from ascendbench.corpus.dataset import TaskDef, load_task
from ascendbench.correctness import tolerance
from ascendbench.integrity import static_checker
from ascendbench.perf import scoring

logger = logging.getLogger(__name__)


# : explicit precedence for `performance_status`. The graded verdicts
# form a strict severity ordering — a more severe verdict must never be
# overwritten by a milder one. `error` (a harness/run failure) is handled
# separately by the callers (it is a "did not grade" verdict, orthogonal
# to this graded-perf scale) and is therefore NOT in this ladder.
#
# docs/reference/timing_arbitration.md grades `host_ns < 1 µs` as `quarantined`; a
# run that is both drift-flagged (`suspicious`) AND sub-µs must end up
# `quarantined`, not stuck at `suspicious`. `_raise_performance_status`
# below takes the MORE severe of the current and proposed verdicts so
# arbitration / diagnostics can only escalate, never roll back.
_PERFORMANCE_STATUS_PRECEDENCE: dict[str, int] = {
    "ok": 0,
    "suspicious": 1,
    "quarantined": 2,
}


def _raise_performance_status(result: TaskEvalResult, proposed: str) -> None:
    """Escalate ``result.performance_status`` to ``proposed`` iff ``proposed``
    is strictly more severe .

    Precedence: ``ok < suspicious < quarantined``. ``error`` is outside this
    ladder — it means "perf was never graded" and is set only on the
    not-run / harness-failure paths; arbitration and runtime diagnostics
    never feed ``error`` here. If the current status is ``error`` (still the
    default, perf has not been promoted off it) this is a no-op: only the
    explicit / promotion lifts ``error`` to ``ok``, and that gate is
    intentionally separate from severity escalation.

    This is the single chokepoint for *raising* the verdict — timing
    arbitration, sampled-output drift, and the  non-default-stream
    diagnostic all route through it, so none of them can downgrade a
    ``quarantined`` run back to ``suspicious`` (or ``suspicious`` to ``ok``).
    """
    cur = result.performance_status
    cur_rank = _PERFORMANCE_STATUS_PRECEDENCE.get(cur)
    new_rank = _PERFORMANCE_STATUS_PRECEDENCE.get(proposed)
    if new_rank is None:
        # Not a graded-severity verdict — ignore (callers handle `error`).
        return
    if cur_rank is None:
        # Current verdict is outside the ladder (`error`): perf has not
        # been promoted off its default. Do not escalate — the /
        # promotion owns the error→ok transition.
        return
    if new_rank > cur_rank:
        result.performance_status = proposed


@dataclass
class ShapeRunResult:
    shape_id: str
    correctness_passed: bool
    correctness_reason: str | None
    measured_ns: float | None
    baseline_ns: float | None
    metamorphic_results: list = field(default_factory=list)
    # : per-trial host_ns from the timing pass (the timing-pass child
    # measured these with CLOCK_MONOTONIC_RAW). The  producer schema
    # derives median/p99/cv from this series;  only carries it.
    durations: list[float] = field(default_factory=list)
    # : per-trial NPU event-timer ns (timing pass). `None` on the dev
    # machine (no NPU); aligned with `durations` so the timing-arbitration
    # host/event-drift cross-check can pool the two together.
    event_ns: list[float] | None = None
    # : count of runner-sampled timing-pass outputs re-verified
    # against golden (cheat-path closure receipt).
    sampled_outputs_checked: int = 0
    # ──  producer schema (DESIGN §6.6 / §9) ─────────────────────────
    # Roofline theoretical-latency floor t_ref and measured latency t_npu
    # (the durations' median). `e` is the per-shape efficiency
    # min(P_CAP, t_ref/t_npu). All `None` until the Roofline path runs
    # (cost model + verified peaks + a non-empty durations series).
    t_ref: float | None = None
    t_npu: float | None = None
    efficiency: float | None = None
    # Per-class W_c (FLOPs) and per-resource M_r (bytes) from the cost
    # model evaluated at this shape's knobs — disclosed for provenance.
    w_per_class: dict[str, float] = field(default_factory=dict)
    m_per_resource: dict[str, float] = field(default_factory=dict)
    # `durations`-derived disclosure stats (DESIGN §6.6).
    median: float | None = None
    p99: float | None = None
    cv: float | None = None
    trials: int = 0
    # Set when the Roofline path was attempted but fail-closed (unverified
    # peaks / cost-model gap); the per-shape e then stays None.
    roofline_error: str | None = None


@dataclass
class TaskEvalResult:
    task_id: str
    static_checks_passed: bool
    static_findings: list = field(default_factory=list)
    shape_results: list[ShapeRunResult] = field(default_factory=list)
    aggregate_score: float = 0.0
    coverage: float = 0.0
    rank_eligible: bool = False
    # Explicit per-run status fields. Downstream
    # analyze() must NOT infer correctness from static_checks_passed alone.
    correctness_status: str = "error"  # "ok" | "fail" | "error"
    runtime_status: str = "error"      # "ok" | "timeout" | "oom" | "compile_fail" | "invalid_timing" | "error"
    performance_status: str = "error"  # "ok" | "suspicious" | "quarantined" | "error"
    performance_flags: list[str] = field(default_factory=list)
    fallback_detected: bool = False
    # : task-level Roofline score s_t = geomean of the per-shape
    # efficiencies e (DESIGN §9). `None` until the Roofline path produced
    # an `e` for every graded shape.
    s_t: float | None = None
    # : True iff the Roofline path had complete benchmark data for
    # this task — a cost model AND verified π/β for every compute-class /
    # resource the cost model references. Set False by `_apply_roofline`
    # when the `cost_model` is absent, or a referenced peak is still an
    # `[unverified]` placeholder. A False value flags a *benchmark-data*
    # gap (Phase 1-2 backfill incomplete), NOT a candidate failure;
    # `scoring.aggregate_perf` keeps such a task out of the perf geomean
    # rather than ε-collapsing the candidate's score for it.
    roofline_data_complete: bool = True
    # True iff every referenced peak is vendor-official / on-device-measured.
    # Gates absolute efficiency claims (not ranking). Set by _apply_roofline.
    absolute_verified: bool = False
    #  / : orthogonal evaluability status.  only introduces it
    # (always "evaluated" here); the unsupported / blocked producers land
    # in .
    task_status: str = scoring.TASK_STATUS_EVALUATED
    # E1: harness/orchestration error reason -- populated only when phase-2b
    # grading hits a non-candidate (harness) failure; never silently dropped.
    error_reason: str | None = None

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

    def task_measurement(self) -> scoring.TaskMeasurement:
        """: collapse this task to the single ``TaskMeasurement``
        scoring consumes for the cross-task geomean. The task-level
        Roofline ``s_t`` (per-shape ``e`` geomean) and ``task_status`` are
        carried through so ``capped_geomean_native`` scores or skips the
        task per DESIGN §9."""
        # DESIGN §9 provenance: baseline_ns = t_ref, measured_ns = t_npu.
        # : the pair MUST come from the SAME representative shape — the
        # first shape that has BOTH a t_ref and a t_npu. Taking the first
        # non-null t_ref and the first non-null t_npu independently could
        # splice values from two different shapes into a fake provenance
        # pair. When no single shape has both, both are left None.
        # Scoring uses s_t, not this ratio; the pair is provenance only.
        rep = next((r for r in self.shape_results
                    if r.t_ref is not None and r.t_npu is not None), None)
        # baseline_ns is ns — convert rep.t_ref (seconds) here.
        t_ref_ns = rep.t_ref * 1e9 if rep is not None else None
        t_npu = rep.t_npu if rep is not None else None
        return scoring.TaskMeasurement(
            task_id=self.task_id,
            correctness_status=self.correctness_status,
            runtime_status=self.runtime_status,
            performance_status=self.performance_status,
            performance_flags=list(self.performance_flags),
            fallback_detected=self.fallback_detected,
            measured_ns=t_npu,
            baseline_ns=t_ref_ns,
            s_t=self.s_t,
            task_status=self.task_status,
            roofline_data_complete=self.roofline_data_complete,
            absolute_verified=self.absolute_verified,
        )


def _verify_sampled_outputs(
    sampled: dict,
    shapes_doc: dict,
    shape_id: str,
    meas_seed: int,
    omod,
    tol_cls,
) -> dict:
    """Re-verify the timing pass's runner-sampled outputs against golden.

    The timing-pass child shipped up outputs for a runner-secret subset of
    measured trials, keyed ``"{trial}::{tensor_key}"``. For each sampled
    trial ``t`` the oracle is re-run on its ``meas_seed + t`` input and the
    candidate's sampled tensor is compared with ``tolerance.compare``.

    This closes the correctness-pass → timing-pass cheat path: a candidate
    that returns a cached/correct answer only in the correctness pass and
    a wrong-but-fast answer in the timing pass is caught here (per
    docs/reference/timing_arbitration.md "for correctness verification").

     — fail-closed: re-running the trusted oracle on a sampled trial's
    timing input must not be allowed to disable the anti-cheat by simply
    erroring. Any oracle re-run exception is treated as a **verification
    failure** (``mismatch=True``), exactly as a real output drift would be:
    a candidate must not get a clean ``performance_status`` on a trial the
    runner could not reverify. The exception's index is also reported in
    ``errors`` for diagnostics.

    Returns ``{"checked": int, "mismatch": bool, "errors": int}``.
    """
    from ascendbench.input_spec.shape_format import build_inputs_for

    # Group the flat "{t}::{key}" payload by trial index.
    by_trial: dict[int, dict[str, torch.Tensor]] = {}
    for flat_key, tensor in sampled.items():
        t_str, _, tensor_key = flat_key.partition("::")
        try:
            t = int(t_str)
        except ValueError:
            continue
        by_trial.setdefault(t, {})[tensor_key] = tensor

    checked = 0
    mismatch = False
    errors = 0
    for t, cand_out in sorted(by_trial.items()):
        try:
            golden = omod.run(
                build_inputs_for(shapes_doc, shape_id,
                                     seed=meas_seed + t))
        except Exception as e:
            # : fail-closed — an unverifiable sampled trial invalidates
            # the performance result rather than being silently skipped.
            logger.warning("sampled-output golden re-run failed "
                            "(%s t=%d) — fail-closed: %s", shape_id, t, e)
            errors += 1
            mismatch = True
            continue
        checked += 1
        for key, exp in golden.items():
            if not torch.is_tensor(exp):
                continue
            act = cand_out.get(key)
            if act is None or not tolerance.compare(act, exp, tol_cls).passed:
                mismatch = True
                break
    return {"checked": checked, "mismatch": mismatch, "errors": errors}


def _verify_sampled_outputs_offline(
    sampled: dict,
    golden_dict: dict,
    tol_cls,
) -> dict:
    """: re-verify the timing pass's runner-sampled outputs against the
    OFFLINE golden, for an ``upstream_oracle_required`` task in verified
    mode.

    Such a task has no per-trial PyTorch oracle, so the runner forced
    every runner-secret *sampled* measured trial onto the shape's fixed
    seed-0 input — exactly the input the offline ``golden/<shape_id>.
    safetensors`` artifact was captured on (see
    ``process_isolation._apply_golden_fixed_inputs``). Every sampled
    trial's output must therefore match ``golden_dict``.

    The candidate cannot tell a sampled trial apart from a non-sampled one
    (the indices are runner-secret, ), so it cannot return a fast-but-
    wrong answer only on the timed-but-non-sampled trials: any such cheat
    has an unbounded chance of landing on a sampled trial and being caught
    here. ``mismatch=True`` then invalidates the performance result.

    Returns ``{"checked": int, "mismatch": bool, "errors": int}``.
    ``errors`` is always 0 here — this path compares against an already
    loaded static ``golden_dict`` and never re-runs an oracle, so the
     fail-closed oracle-error case cannot arise; the key is present
    only so both ``_verify_sampled_outputs*`` share one return schema.
    """
    by_trial: dict[int, dict[str, torch.Tensor]] = {}
    for flat_key, tensor in sampled.items():
        t_str, _, tensor_key = flat_key.partition("::")
        try:
            t = int(t_str)
        except ValueError:
            continue
        by_trial.setdefault(t, {})[tensor_key] = tensor

    checked = 0
    mismatch = False
    for _t, cand_out in sorted(by_trial.items()):
        checked += 1
        for key, exp in golden_dict.items():
            if not torch.is_tensor(exp):
                continue
            act = cand_out.get(key)
            if act is None or not tolerance.compare(act, exp, tol_cls).passed:
                mismatch = True
                break
    return {"checked": checked, "mismatch": mismatch, "errors": 0}


def _apply_runtime_diagnostics(
    result: TaskEvalResult, two, backend: str = "npu") -> None:
    """: fold the child-side profiler diagnostics into the run status.

    The candidate child profiled one call per shape and shipped up three
    signals on the  carrier fields:

    * ``fallback_signal`` — the candidate produced no NPU/triton kernel
      event and silently fell back to the generic torch path (C7). Any
      shape with ``fallback_signal=True`` sets ``fallback_detected=True``
      and adds the ``torch_fallback`` performance flag. ( then
      normalises ``fallback_detected`` to a correctness fail before
      scoring — that lives in eval's  hook, not here.)
    * ``stream_nondefault`` — work ran on a non-default stream (B2),
      escaping host timing. Any such shape adds the ``invalid_stream``
      flag and downgrades ``performance_status`` to ``suspicious``.
    * ``kernel_event_present`` — purely informational here; the empty-
      kernel cheat (B5) surfaces through ``fallback_signal``.

    Signals are read from the timing pass first (its profiling reflects
    the warmed-up steady state); the correctness pass is the fallback
    source when a shape has no timing record. A ``None`` signal means
    "not measured on this host" (dev-machine stub) and is ignored — it
    never flips a status. On the GPU backend, where the profiler parser is
    not yet implemented, an all-``None`` run adds the non-status
    ``gpu_antihack_unverified`` provenance flag so the result is marked
    "anti-hack gate not active" instead of passing as if checked.
    """
    fallback = False
    nondefault = False
    # GPU anti-hack gate provenance: the GPU profiler trace parser is not yet
    # implemented (its diagnostics are all-None until the H200 parser lands),
    # so a GPU run's anti-hack gate is inactive. A shape is "verified" only
    # when the profiler emitted ALL THREE signals; a partial parser leaving any
    # signal None still counts as unverified (future-proof).
    gpu_all_verified = True
    gpu_saw_shape = False
    for r in result.shape_results:
        sid = r.shape_id
        tscr = two.timing.per_shape.get(sid)
        cscr = two.correctness.per_shape.get(sid)
        # Prefer the timing pass's diagnostics; fall back to correctness.
        diag = tscr if (tscr is not None
                        and tscr.kernel_event_present is not None
                        ) else None
        if diag is None and tscr is not None and tscr.fallback_signal is not None:
            diag = tscr
        if diag is None:
            diag = cscr if (cscr is not None
                            and (cscr.fallback_signal is not None
                                 or cscr.stream_nondefault is not None)) else tscr
        if backend == "gpu":
            gpu_saw_shape = True
            src = tscr if tscr is not None else cscr
            if (src is None or src.kernel_event_present is None
                    or src.stream_nondefault is None
                    or src.fallback_signal is None):
                gpu_all_verified = False
        if diag is None:
            continue
        if diag.fallback_signal is True:
            fallback = True
        if diag.stream_nondefault is True:
            nondefault = True

    if fallback:
        result.fallback_detected = True
        if "torch_fallback" not in result.performance_flags:
            result.performance_flags.append("torch_fallback")
    if backend == "gpu" and (not gpu_saw_shape or not gpu_all_verified):
        if "gpu_antihack_unverified" not in result.performance_flags:
            result.performance_flags.append("gpu_antihack_unverified")
    if nondefault:
        if "invalid_stream" not in result.performance_flags:
            result.performance_flags.append("invalid_stream")
        # : escalate to `suspicious` via the precedence helper — it is
        # a no-op when the verdict is already as severe or more severe
        # (`quarantined`), and never rolls back.
        _raise_performance_status(result, "suspicious")


def _normalise_fallback_to_correctness_fail(result: TaskEvalResult) -> None:
    """ hook: normalise a detected torch-fallback to a correctness fail.

    DESIGN §6.3 / §6.7: a candidate that produced no NPU/triton kernel
    event silently fell back to the generic torch path. Such a solution
    may still be numerically *correct*, but a correct-yet-fallback
    implementation must not clear the correctness gate — the gate predicate
    is ``correctness == ok ∧ ¬fallback``. So once ``fallback_detected`` is
    set (by :func:`_apply_runtime_diagnostics` from the child-side
    ``fallback_signal``), the run's ``correctness_status`` is forced to
    ``"fail"`` *before* scoring.

    This consumes the  producer signal: only an explicit
    ``fallback_signal is True`` per shape sets ``fallback_detected``.
    A ``None`` signal (the dev machine has no NPU, so the profiler never
    measured) is "not measured on this host" — it never sets
    ``fallback_detected`` and therefore never triggers this normalisation.
    A ``False`` signal (kernel event present, no fallback) likewise does
    not.

    Idempotent and order-independent: when ``fallback_detected`` is false
    this is a no-op, so it is safe to call after the per-shape grading and
    after :func:`_apply_runtime_diagnostics`. It only ever *downgrades*
    ``correctness_status`` to ``"fail"`` — an ``"error"`` verdict (harness
    failure) is left intact, since that is a strictly stronger signal than
    a graded fail.
    """
    if not result.fallback_detected:
        return
    if result.correctness_status != "error":
        result.correctness_status = "fail"


def _normalise_metamorphic_checks(task) -> list[dict]:
    """Normalise ``meta.runner_checks.metamorphic_checks`` to ``[{name:..}]``.

    Each declared item is either a ``{name: ..., description: ...}`` mapping
    or a bare name string; both collapse to a ``{"name": ...}`` dict.
    """
    checks = (task.meta.get("runner_checks", {})
              .get("metamorphic_checks") or [])
    norm: list[dict] = []
    for c in checks:
        if isinstance(c, dict) and "name" in c:
            norm.append(c)
        elif isinstance(c, str):
            norm.append({"name": c})
    return norm


def _run_task_metamorphic(
    task,
    solution_path: Path | None = None,
    shapes_doc: dict | None = None,
    rep_shape_id: str | None = None,
    tol_cls=None,
    backend: str = "npu",
) -> list:
    """ / : run the task's declared metamorphic checks for real.

    ``meta.runner_checks.metamorphic_checks`` names the metamorphic
    transformations for this *real* task. When the runner can supply a
    candidate solution and a representative shape (the normal grading
    path), a real transformation registry is built
    (:func:`adversarial.build_metamorphic_transformations`): each declared
    check transforms the candidate's inputs, re-runs the candidate in an
    isolated subprocess, and verifies a structural invariant
    (annihilation / homogeneity / compensating-pair / row-permute /
    differential). A genuinely operator-specific check with no generic
    structural form is surfaced with an explicit
    ``not generically implementable`` non-pass.

    With no solution / shape available (a bare ``_run_task_metamorphic
    (task)`` call) the registry is empty and every declared check is
    surfaced as ``no transformation registered`` — still reported, never
    silent. The result list lands on every shape's
    ``ShapeRunResult.metamorphic_results``.

    This is the metamorphic path — distinct from the canary path
    (`adversarial.check_canaries`), whose findings have their own schema
    and never mix into ``metamorphic_results``.
    """
    from ascendbench.integrity import adversarial

    norm = _normalise_metamorphic_checks(task)
    if not norm:
        return []

    transformations: dict = {}
    if (solution_path is not None and shapes_doc is not None
            and rep_shape_id is not None and tol_cls is not None):
        ctx = _build_metamorphic_context(
            solution_path, shapes_doc, rep_shape_id, tol_cls, backend=backend)
        if ctx is not None:
            transformations = adversarial.build_metamorphic_transformations(
                norm, ctx)

    return adversarial.run_metamorphic_checks(
        norm, transformations=transformations)


def _build_metamorphic_context(
    solution_path: Path,
    shapes_doc: dict,
    rep_shape_id: str,
    tol_cls,
    backend: str = "npu",
):
    """Build the :class:`adversarial.MetamorphicContext` for a task .

    The context bundles a representative shape's built inputs, a
    ``run_candidate`` callable that runs the candidate once in an isolated
    subprocess (the SAME isolation as the correctness pass —
    ``process_isolation.run_candidate_isolated``), and a tolerance-aware
    tensor compare. Returns ``None`` if the representative shape's inputs
    cannot be built.
    """
    from ascendbench.input_spec.shape_format import build_inputs_for
    from ascendbench.integrity import adversarial, process_isolation

    try:
        base_inputs = build_inputs_for(shapes_doc, rep_shape_id, seed=0)
    except Exception as e:
        logger.warning("metamorphic: cannot build inputs for %s: %s",
                        rep_shape_id, e)
        return None

    def run_candidate(inp: dict) -> dict | None:
        """Run the candidate once on `inp` in an isolated subprocess."""
        try:
            res = process_isolation.run_candidate_isolated(
                solution_path, {rep_shape_id: inp},
                pass_kind="correctness", backend=backend)
        except Exception as e:
            logger.warning("metamorphic: candidate run failed: %s", e)
            return None
        scr = res.per_shape.get(rep_shape_id)
        if scr is None or not scr.ran:
            return None
        return dict(scr.outputs)

    def tensors_close(actual, expected) -> bool:
        if not (torch.is_tensor(actual) and torch.is_tensor(expected)):
            return False
        if actual.shape != expected.shape:
            return False
        return tolerance.compare(actual, expected, tol_cls).passed

    return adversarial.MetamorphicContext(
        base_inputs=base_inputs,
        run_candidate=run_candidate,
        tensors_close=tensors_close,
    )


# : byte width of every dtype `inputs.py` can materialise. Used to
# derive the representative cost-model `dtype_size` from a task's inputs
# block so a non-fp16 task (e.g. float32) scores against the right
# element size instead of the fp16 default (2.0).
def _representative_dtype_size(shapes_doc: dict) -> float | None:
    """The cost model's representative ``dtype_size`` (first dims-bearing input's element bytes).

    Delegates to the shape_format adapter so an explicit-format doc resolves too; the adapter's knob branch
    is byte-identical to the old local helper (verified equal over all 101 knob docs). Returns ``None`` for a
    scalar-only inputs block (the caller falls back to the default dtype_size)."""
    from ascendbench.input_spec.shape_format import representative_dtype_size
    return representative_dtype_size(shapes_doc)


def _shape_knobs(shapes_doc: dict, shape_id: str) -> dict[str, float] | None:
    """Resolve a shape's row environment (``size_knobs`` defaults + per-row ``knobs``, or the explicit row
    ``env``), filtered to the NUMERIC values a cost formula needs. Returns ``None`` if the shape id is absent.

    effective_env supplies the same merge for knob docs (and the named row env for explicit); the numeric
    filter + None-on-missing wrapper is kept here because effective_env returns the FULL map and raises
    KeyError on a missing id."""
    from ascendbench.input_spec.shape_format import effective_env
    try:
        env = effective_env(shapes_doc, shape_id)
    except KeyError:
        return None
    return {k: float(v) for k, v in env.items()
            if isinstance(v, (int, float)) and not isinstance(v, bool)}


def _apply_roofline(
    result: TaskEvalResult,
    task,
    shapes_doc: dict,
    peak_table_path: Path | str | None = None,
) -> None:
    """: fill each shape's Roofline ``t_ref`` / ``W_c`` / ``M_r`` /
    ``efficiency`` and aggregate the task-level ``s_t``.

    Per DESIGN §9 the per-shape efficiency ``e = min(P_CAP, t_ref/t_npu)``
    is computed in ``scoring``; the per-shape ``e`` → task-level ``s_t``
    geomean is aggregated *here*.  ``s_t`` lands only when every graded
    shape produced an ``e``; a missing cost model, unverified peaks, or a
    shape with no timing leaves ``s_t = None``.

     — fail-closed, but *per-task scoped*. The peak table is loaded
    WITHOUT ``require_verified``; the verification check is then applied
    only to the π/β keys **this task's normalised ``cost_model`` actually
    references**. The table-wide ``require_verified=True`` load used to
    raise on ANY ``[unverified]`` entry — so a single placeholder peak
    (``vector`` / ``bf16_cube`` / ``interconnect`` are 0.0 placeholders
    today) disabled the Roofline path for *every* task, including ones
    that only ever touch ``fp16_cube`` / ``hbm`` and could score fine. A
    task is now ``roofline_error`` only when IT references an unverified
    peak — an unrelated placeholder no longer drags it down.

    ``roofline_data_complete`` is set False when the Roofline path is
    skipped because of a benchmark-data gap (no cost model, or this
    task's referenced peak is an unverified placeholder). That flags a
    Phase 1-2 data-completeness issue, not a candidate failure — scoring
    keeps such a task out of the perf geomean rather than ε-collapsing it.
    """
    cost_model = task.cost_model
    if cost_model is None:
        # No cost model (pre- task) — no Roofline score. This is a
        # benchmark-data gap (cost-model backfill not done), not a
        # candidate failure: flag it so scoring excludes the task from
        # the perf geomean instead of ε-collapsing the candidate.
        result.roofline_data_complete = False
        return

    from ascendbench.perf import peak_table

    try:
        # : load WITHOUT require_verified — the whole-table check
        # would reject any task the moment an unrelated peak is an
        # [unverified] placeholder. Per-task verification is applied
        # below against only the keys this cost model references.
        compute_peaks, resource_peaks = peak_table.load_peak_table(peak_table_path)
    except peak_table.PeakTableError as e:
        # The table itself is missing / unparseable / omits a required
        # key — a real benchmark-data fault, fail-closed for every shape.
        msg = f"peak table unusable: {e}"
        logger.warning("Roofline disabled for %s: %s", result.task_id, msg)
        for r in result.shape_results:
            r.roofline_error = msg
        result.roofline_data_complete = False
        return

    # : verify only the compute-class / resource keys THIS task's
    # normalised cost model references. A peak is usable iff it is
    # present and positive-finite; a 0.0 / non-finite value is an
    # `[unverified]` placeholder. A task that references such a key is a
    # genuine Roofline-data gap (e.g. a `vector`-class task while the
    # vector π is still pending a 910B measurement — the expected
    # data limitation); a task that only references verified keys
    # (`fp16_cube` / `int8` / `hbm`) is unaffected.
    import math as _math

    unverified_refs: list[str] = []
    for compute_class in cost_model.flops:
        pi = compute_peaks.get(compute_class)
        if pi is None or not _math.isfinite(pi) or pi <= 0:
            unverified_refs.append(f"compute_class.{compute_class}")
    for resource in cost_model.bytes:
        beta = resource_peaks.get(resource)
        if beta is None or not _math.isfinite(beta) or beta <= 0:
            unverified_refs.append(f"resource.{resource}")
    if unverified_refs:
        msg = (
            f"cost_model references peak(s) with no positive value: "
            f"{sorted(unverified_refs)} — Roofline data not yet complete "
            f"for this task (peak still 0.0; fill it to rank)"
        )
        logger.warning("Roofline disabled for %s: %s", result.task_id, msg)
        for r in result.shape_results:
            r.roofline_error = msg
        result.roofline_data_complete = False
        return

    # Peaks rank above; absolute efficiency claims also need every referenced
    # peak absolute_verified (official / measured).
    compute_prov, resource_prov = peak_table.load_peak_provenance(peak_table_path)
    result.absolute_verified = (
        all(peak_table.is_absolute_verified(
                compute_peaks[c], compute_prov.get(c, "unspecified"))
            for c in cost_model.flops)
        and all(peak_table.is_absolute_verified(
                    resource_peaks[r], resource_prov.get(r, "unspecified"))
                for r in cost_model.bytes)
    )

    # : derive the representative element size from the task's inputs
    # dtype so a non-2-byte task (float32 / int64 / …) feeds the right
    # `dtype_size` into the cost-model formulas. `None` (scalar-only inputs
    # block) means "use the fp16 default" — passing `dtype_size=None` to
    # `compute_t_ref` / `derived` does exactly that.
    dtype_size = _representative_dtype_size(shapes_doc)
    derived = ({"dtype_size": dtype_size}
               if dtype_size is not None else None)

    efficiencies: list[float] = []
    for r in result.shape_results:
        knobs = _shape_knobs(shapes_doc, r.shape_id)
        if knobs is None:
            r.roofline_error = "shape not found in shapes.yaml"
            continue
        try:
            r.t_ref = scoring.compute_t_ref(
                cost_model, knobs, compute_peaks, resource_peaks,
                dtype_size=dtype_size)
            # Disclose per-class W_c / per-resource M_r for provenance —
            # evaluated with the SAME representative dtype_size as t_ref.
            env = scoring.FormulaEnv(knobs, derived=derived)
            r.w_per_class = {c: env.evaluate(f)
                             for c, f in cost_model.flops.items()}
            r.m_per_resource = {res: env.evaluate(f)
                                for res, f in cost_model.bytes.items()}
        except (scoring.RooflineError, scoring.CostModelError) as e:
            r.roofline_error = str(e)
            continue
        # compute_t_ref() returns seconds; baseline_ns and
        # shape_efficiency expect ns. Dividing s by ns was a 1e9 bug -> e≈1e-9.
        t_ref_ns = r.t_ref * 1e9
        if r.t_npu is None or r.t_npu <= 0:
            # No timing: leave the baseline_ns/measured_ns provenance pair
            # None (DESIGN §9 — both set together or not at all).
            continue
        r.baseline_ns = t_ref_ns
        r.measured_ns = r.t_npu
        try:
            r.efficiency = scoring.shape_efficiency(t_ref_ns, r.t_npu)
        except ValueError as e:
            r.roofline_error = f"efficiency: {e}"
            continue
        efficiencies.append(r.efficiency)

    # Task-level s_t = geomean of the per-shape e (DESIGN §9). Only when
    # every graded shape produced an e — a partial Roofline is not scored.
    graded = [r for r in result.shape_results if r.roofline_error is None]
    if graded and len(efficiencies) == len(graded):
        try:
            result.s_t = scoring.geomean(efficiencies)
        except ValueError as e:
            logger.warning("s_t geomean failed for %s: %s", result.task_id, e)


# : the `oracle_simplified` tier whose golden output can only be
# obtained by running the frozen upstream kernel offline (no PyTorch
# oracle / closed-form formula). Such a task has no provider golden in
# self-reported mode; in verified mode the golden is the offline
# `tasks/.../golden/` artifact set (collected on the GPU machine).
_UPSTREAM_ORACLE_TIER = "upstream_oracle_required"


class GoldenIntegrityError(RuntimeError):
    """Raised when the golden chain fails R15/R18 verify.

    Distinct from "golden missing" (a complete absence of artifacts,
    handled via the ``blocked`` task_status). This exception is raised
    only when a manifest exists but its SHA chain or signature does
    not verify -- a tamper / corruption signal that must surface to
    the submitter as "evaluation failed: golden integrity" instead of
    silently downgrading to ``blocked``.
    """


def _resolve_golden(
    task,
    mode: str,
    eval_shape_ids: list[str],
) -> tuple[str, str, dict[str, dict] | None]:
    """ : route the golden source by ``oracle_simplified``
    x ``mode`` and derive the task-level ``task_status``.

    Returns ``(task_status, golden_kind, offline_golden)``:

    * ``("evaluated", "oracle", None)`` — non-upstream task: the
      in-process PyTorch oracle is the golden.
    * ``("evaluated", "offline", {shape_id: golden_dict})`` — upstream
      task in verified mode with a complete offline golden artifact set.
    * ``("unsupported", "none", None)`` — upstream task in self-reported
      mode: no provider golden exists for this mode.
    * ``("blocked", "none", None)`` — upstream task in verified mode whose
      ``tasks/.../golden/`` directory is missing or incomplete (does not
      cover every graded shape). Fail-closed.

    The offline golden artifacts (``golden/<shape_id>.safetensors``) are
    collected on the GPU machine in  Phase 4; on the dev machine the
    directory does not exist, so every verified-mode upstream task routes
    to ``blocked`` — the expected pre-Phase-4 state.
    """
    try:
        tier = task.oracle_tier
    except (KeyError, TypeError, AttributeError):
        # A task whose meta omits `oracle_simplified` is treated as a
        # plain PyTorch-oracle task (back-compat with pre- metas).
        tier = ""

    if tier != _UPSTREAM_ORACLE_TIER:
        # simple_pytorch_ok / formula_exact_manual — the PyTorch oracle is
        # the golden; evaluate normally in both modes.
        return (scoring.TASK_STATUS_EVALUATED, "oracle", None)

    if mode != "verified":
        # upstream_oracle_required + self-reported: no provider golden.
        return ("unsupported", "none", None)

    # upstream_oracle_required + verified: load the offline golden.
    # Run the R15 SHA + R18 signature integrity chain
    # via `eligibility.golden_integrity_chain_ok`. Grace-period
    # semantics: a pre-M3.2 manifest-less capture still loads; a
    # present-but-bad manifest fails closed with GoldenIntegrityError
    # (NOT "blocked" -- the submitter must see the integrity claim).
    golden_dir = task.task_dir / "golden"
    if not golden_dir.is_dir():
        # Pre-Phase-4 dev machine: no golden/ directory — fail-closed.
        return ("blocked", "none", None)

    from safetensors.torch import load_file

    from ascendbench.release_gate import eligibility

    if not eligibility.golden_integrity_chain_ok(
        task.task_dir, eval_shape_ids
    ):
        raise GoldenIntegrityError(
            f"task {task.task_id}: golden manifest SHA/signature chain "
            "did not verify; evaluation refused"
        )

    offline: dict[str, dict] = {}
    for sid in eval_shape_ids:
        artifact = golden_dir / f"{sid}.safetensors"
        if not eligibility.golden_artifact_usable(artifact):
            # Missing OR corrupt / unreadable artifact — fail-closed
            # (same predicate as eligibility.golden_complete).
            logger.warning("offline golden missing/unusable for %s/%s",
                            task.task_id, sid)
            return ("blocked", "none", None)
        offline[sid] = dict(load_file(str(artifact)))
    return (scoring.TASK_STATUS_EVALUATED, "offline", offline)


def _resolve_eval_task_status(task: TaskDef, mode: str) -> str:
    """The repo-authoritative ``task_status`` for one task — WA.4 ②(b).

    ``eval.evaluate`` calls this once, immediately after the task is loaded
    and **before any early return**, so every early-return branch
    (static-check ``blocked``, non-gradeable stub / manual, oracle-import /
    input-build short-circuit) carries the same verdict ``cli analyze``'s
    ``_authoritative_task_status`` derives. Both route through the shared
    :func:`param_constraints.resolve_task_status` arbiter with the same
    trusted-repo-derived booleans, so the two layers cannot drift.

    The fixed priority (high → low): ``is_canary`` ⇒ ``evaluated``;
    self-reported ``upstream_oracle_required`` ⇒ ``unsupported``;
    verified-mode incomplete offline golden ⇒ ``blocked``;
    ``param_constraints`` fails the shared helper ⇒ ``blocked``;
    otherwise ⇒ ``evaluated``.

    Args:
        task: the loaded :class:`TaskDef`.
        mode: ``"self-reported"`` or ``"verified"``.

    Returns:
        One of ``"evaluated"`` / ``"unsupported"`` / ``"blocked"``.
    """
    import yaml

    from ascendbench.input_spec import param_constraints
    from ascendbench.release_gate import eligibility

    meta = getattr(task, "meta", None) or {}
    is_canary = meta.get("is_canary") is True

    try:
        upstream = task.oracle_tier == _UPSTREAM_ORACLE_TIER
    except (KeyError, TypeError, AttributeError):
        upstream = False

    self_reported_unsupported = mode == "self-reported" and upstream
    verified_golden_incomplete = (
        mode == "verified" and upstream
        and not eligibility.golden_complete(task))

    # The `param_constraints` gate: load the task's shapes.yaml and run the
    # shared helper (scaffold short-circuit → schema → semantics).
    shapes_doc: dict | None = None
    shapes_path = task.task_dir / "shapes.yaml"
    try:
        if shapes_path.is_file():
            loaded = yaml.safe_load(shapes_path.read_text())
            if isinstance(loaded, dict):
                shapes_doc = loaded
    except (OSError, yaml.YAMLError) as e:
        logger.warning("param_constraints gate: shapes.yaml unreadable for "
                        "%s: %s", task.task_id, e)
    pc_result = param_constraints.check_param_constraints(
        getattr(task, "param_constraints", None), shapes_doc,
        tid=task.task_id)

    return param_constraints.resolve_task_status(
        is_canary=is_canary,
        self_reported_unsupported=self_reported_unsupported,
        verified_golden_incomplete=verified_golden_incomplete,
        param_constraints_failed=not pc_result.ok,
    )


def evaluate(
    task_id: str,
    solution_path: Path,
    backend: str = "npu",
    task_def: TaskDef | None = None,
    mode: str = "self-reported",
    peak_table_path: Path | str | None = None,
) -> TaskEvalResult:
    """Top-level entrypoint. Phase 2.A is skeleton — runner orchestration
    completes in Phase 2.B.

    `task_def` lets a caller inject an already-loaded :class:`TaskDef`
    instead of resolving `task_id` through `load_task` (which only scans
    `tasks/level{1,2,3}/`). The  canary runner uses this to grade a
    `tasks/_canary/` task — discovered by its dedicated enumerator — with
    the submission's solution for the canary's bound `source_task_id`.
    When omitted the behaviour is unchanged: `load_task(task_id)`.

    : `mode` selects which eval shapes phase-2b grades.

    * ``"self-reported"`` — only the public shapes are graded; the task's
      ``hidden_shapes`` are never materialised or injected.
    * ``"verified"`` — public **and** hidden shapes are graded. The runner
      (this trusted parent) materialises every hidden shape's inputs from
      the task's own ``shapes.yaml.hidden_shapes`` section and injects them
      down the  IPC. The candidate child still receives only the
      sanitised minimal set ( / DESIGN §3.3) — materialised inputs and
      ``solution.py``, never the shapes document — so injecting a hidden
      shape's inputs does not leak the hidden knobs to the candidate.

     : the golden source and ``task_status`` are routed
    by ``oracle_simplified`` x ``mode`` (see :func:`_resolve_golden`):

    * non-``upstream_oracle_required`` task (``simple_pytorch_ok`` /
      ``formula_exact_manual``) — the in-process PyTorch oracle is the
      golden; ``task_status = "evaluated"``.
    * ``upstream_oracle_required`` + ``self-reported`` — there is no
      provider golden for this mode; ``task_status = "unsupported"`` and
      every perf field (``s_t`` / ``baseline_ns`` / ``measured_ns`` /
      per-shape timing) is left ``None``.
    * ``upstream_oracle_required`` + ``verified`` — the runner loads the
      offline golden from ``tasks/.../golden/<shape_id>.safetensors``
      ( Phase-4 GPU artifacts). Complete ⇒ grade against the offline
      golden, ``task_status = "evaluated"``; missing / incomplete ⇒
      fail-closed ``task_status = "blocked"`` with null perf. On the dev
      machine no ``golden/`` directory exists, so verified-mode upstream
      tasks are all ``blocked`` — that is the expected pre-Phase-4 state.
    """
    if mode not in ("self-reported", "verified"):
        raise ValueError(
            f"mode must be 'self-reported' or 'verified', got {mode!r}")
    task = task_def if task_def is not None else load_task(task_id)
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

    # WA.4 ②(b): resolve the repo-authoritative `task_status` ONCE, before
    # any early return. Every early-return branch below (static-check
    # `blocked`, non-gradeable stub / manual, `_resolve_golden`
    # short-circuit) then carries this verdict — none falls through to the
    # default `evaluated`. The verdict is the SAME one `cli analyze`'s
    # `_authoritative_task_status` derives (shared `resolve_task_status`).
    result.task_status = _resolve_eval_task_status(task, mode)

    if static_report.blocked:
        result.correctness_status = "fail"
        result.runtime_status = "ok"               # static-only path didn't run anything
        result.performance_status = "ok"
        return result

    # Phase 2.B: grade correctness + timing. Oracle (trusted) runs
    # in-process; the candidate runs in TWO independent isolated
    # subprocesses — a correctness pass then a timing pass, with a SIGKILL
    # + cache/device scrub between them (process_isolation.run_two_pass).
    # Correctness is graded against the correctness pass; the timing
    # pass's runner-sampled outputs are re-verified against golden, and
    # its host-CLOCK_MONOTONIC_RAW durations are arbitrated per
    # docs/reference/timing_arbitration.md. The parent never times candidate code.
    try:
        import importlib.util

        import yaml

        from ascendbench.correctness import tolerance
        from ascendbench.input_spec import inputs as _inputs
        from ascendbench.input_spec.shape_format import build_inputs_for
        from ascendbench.integrity import process_isolation
        from ascendbench.perf import timing

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
            # : the graded eval-shape set depends on `mode`. The runner
            # always holds the complete shapes.yaml (public + hidden); the
            # candidate child never gets the document either way ( —
            # only materialised inputs cross the IPC). In verified mode the
            # hidden shapes are graded too; in self-reported mode they are
            # neither materialised nor injected.
            pub, hid = _inputs.shape_ids(shapes_doc)
            eval_shape_ids = list(pub)
            if mode == "verified":
                eval_shape_ids += list(hid)

            #  : route the golden source and task_status
            # by `oracle_simplified` x `mode`. An `upstream_oracle_required`
            # task has no PyTorch oracle that can serve as golden:
            #   * self-reported ⇒ task_status="unsupported" (no provider
            #     golden) — skip grading, leave all perf fields null.
            #   * verified ⇒ load the offline `golden/` artifacts; complete
            #     ⇒ grade against them; missing/incomplete ⇒
            #     task_status="blocked" (fail-closed), perf fields null.
            # Everything else ⇒ the in-process PyTorch oracle is golden.
            golden_status, golden_kind, offline_golden = _resolve_golden(
                task, mode, eval_shape_ids)
            if golden_status != scoring.TASK_STATUS_EVALUATED:
                # No evaluable golden in this mode — return with null perf.
                # `result.task_status` was already set by the WA.4 ②(b)
                # resolver (`_resolve_task_status_for_eval`) before any early
                # return; `golden_status` here is the same "unsupported" /
                # "blocked" verdict (the resolver's tiers 2-3 mirror
                # `_resolve_golden`), so it is consistent. correctness /
                # runtime / performance statuses are left at their "not
                # graded" defaults; the orthogonal task_status field carries
                # the verdict (DESIGN §5.7).
                result.task_status = golden_status
                result.correctness_status = "error"
                result.runtime_status = "error"
                result.performance_status = "error"
                if golden_status == "blocked":
                    result.error_reason = (
                        "verified-mode offline golden missing or "
                        "incomplete (tasks/.../golden/)")
                return result

            # : load/execute `oracle_pytorch.py` ONLY on the
            # `golden_kind == "oracle"` path. An `upstream_oracle_required`
            # task graded in verified mode has already resolved its golden
            # to the offline `golden/<shape_id>.safetensors` artifact set
            # (`golden_kind == "offline"`) — its `oracle_pytorch.py` is a
            # low-level non-runnable stand-in and must NOT be imported. A
            # broken / un-importable oracle on that path used to fail the
            # whole grading as a harness error even with a complete
            # `golden/` set; the offline path now never touches it.
            omod = None
            if golden_kind == "oracle":
                spec = importlib.util.spec_from_file_location(
                    f"_oracle_{task_id}", oracle_path)
                omod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(omod)

            tol_cls = tolerance.load_policy().get(task.tolerance_class)

            # //: the runner materialises every graded shape's
            # inputs (public always; hidden too in verified mode) and
            # injects them down the IPC. Correctness pass gets one dict
            # per shape (seed=0); the timing pass gets a per-trial
            # sequence — n_warmup warm trials (warm_seed) then n_repeat
            # measured trials, each a FRESH input at meas_seed + t.
            n_warmup = process_isolation.DEFAULT_N_WARMUP
            n_repeat = process_isolation.DEFAULT_N_TRIALS
            warm_seed, meas_seed = 1, 1000
            correctness_inputs = {
                sid: build_inputs_for(shapes_doc, sid, seed=0)
                for sid in eval_shape_ids
            }
            timing_inputs = {
                sid: (
                    [build_inputs_for(shapes_doc, sid,
                                          seed=warm_seed + w)
                     for w in range(n_warmup)]
                    + [build_inputs_for(shapes_doc, sid,
                                            seed=meas_seed + t)
                       for t in range(n_repeat)]
                )
                for sid in eval_shape_ids
            }
            # : `run_two_pass` no longer takes a `seed` — the
            # runner-secret sampling seed is generated fresh per call from
            # a CSPRNG inside `run_two_pass`, strictly distinct from the
            # deterministic input-generation `meas_seed`.
            #
            # : for an `upstream_oracle_required` task graded in
            # verified mode the offline golden is captured on the seed-0
            # input — i.e. the same input as `correctness_inputs[sid]`.
            # Hand `run_two_pass` that fixed input per shape: it then
            # forces every runner-secret sampled timing trial onto it so
            # the timing-pass output can be reverified against the offline
            # golden (the offline-golden cheat-path closure). The
            # non-sampled trials keep fresh per-trial inputs (anti-cache).
            golden_fixed_inputs = (
                dict(correctness_inputs) if golden_kind == "offline"
                else None)
            two = process_isolation.run_two_pass(
                solution_path, correctness_inputs, timing_inputs,
                n_warmup=n_warmup,
                golden_fixed_inputs=golden_fixed_inputs, backend=backend)
            cand = two.correctness

            #  / : metamorphic checks for this real task. The
            # runner builds a real transformation registry — each declared
            # check transforms a representative shape's inputs, re-runs the
            # candidate in isolation, and verifies a structural invariant.
            # The same result list is attached to every shape's
            # ShapeRunResult.metamorphic_results — these are the
            # metamorphic path, distinct from canary findings.
            metamorphic = _run_task_metamorphic(
                task,
                solution_path=solution_path,
                shapes_doc=shapes_doc,
                rep_shape_id=eval_shape_ids[0] if eval_shape_ids else None,
                tol_cls=tol_cls, backend=backend)

            # : a sampled-output mismatch on ANY shape escalates the
            # task-level `performance_status` to `suspicious` — but only
            # after the / promotion lifts the "error" default to
            # "ok". Recorded across the per-shape loop, applied below.
            sampled_outputs_drift = False

            for sid in eval_shape_ids:
                reason: str | None = None
                passed = False
                #  : the golden output for this shape. For a
                # non-upstream task it is the in-process PyTorch oracle on
                # the correctness-pass input; for an `upstream_oracle_
                # required` task graded in verified mode it is the offline
                # `golden/<shape_id>.safetensors` artifact loaded by
                # `_resolve_golden`. Used for correctness grading AND
                #  for re-verifying the timing pass's runner-sampled
                # outputs — the offline-golden path's sampled trials were
                # forced onto the seed-0 input the artifact covers, so
                # both golden kinds reverify (no unverified cheat path).
                if golden_kind == "offline":
                    expected = offline_golden[sid]
                else:
                    try:
                        expected = omod.run(correctness_inputs[sid])
                    except Exception as e:
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

                # --- : carry the timing pass's durations -----------
                # : also carry the per-trial NPU event-timer series
                # (None on the dev machine — host-only timing).
                tscr = two.timing.per_shape.get(sid)
                durations = list(tscr.durations) if tscr else []
                event_ns = (list(tscr.event_ns)
                            if tscr and tscr.event_ns else None)

                # ---  / : sampled-output cheat-path closure ----
                # The timing pass shipped up outputs for a runner-secret
                # subset of measured trials. Re-verify every one against
                # golden; any mismatch invalidates the performance result.
                #
                # Two golden sources, both enforced — there is NO
                # unverified path:
                #   * oracle  — re-run the in-process PyTorch oracle on the
                #     matching `meas_seed + t` fresh input.
                #   * offline — the `upstream_oracle_required` verified
                #     task has no per-trial oracle, so the runner forced
                #     every runner-secret sampled trial onto the shape's
                #     fixed seed-0 input (`_apply_golden_fixed_inputs`).
                #     Each sampled output is then compared against the
                #     offline `golden/<shape_id>.safetensors` artifact.
                # A candidate cannot tell a sampled trial apart from a
                # non-sampled one (runner-secret indices, ), so it
                # cannot return a fast-but-wrong answer only on the timed
                # trials without risking the sampled check .
                #
                #  — fail-closed: the sampled-output anti-cheat must
                # never be disabled by a trial simply being unverifiable.
                # `_verify_sampled_outputs` already raises `mismatch=True`
                # on any oracle re-run exception; here we additionally
                # treat "sampled outputs were shipped up but NONE got
                # verified" (`checked == 0`) as a verification failure —
                # otherwise a candidate whose every sampled trial happens
                # to be unreverifiable would keep `performance_status=ok`.
                sampled_checked = 0
                if (tscr is not None and tscr.sampled_outputs
                        and tol_cls is not None):
                    if golden_kind == "offline":
                        drift = _verify_sampled_outputs_offline(
                            tscr.sampled_outputs, expected, tol_cls)
                    else:
                        drift = _verify_sampled_outputs(
                            tscr.sampled_outputs, shapes_doc, sid,
                            meas_seed, omod, tol_cls)
                    sampled_checked = drift["checked"]
                    # : sampled outputs present but 0 verified — the
                    # anti-cheat could not run; fail-closed, never `ok`.
                    unverified = sampled_checked == 0
                    if drift["mismatch"] or unverified:
                        # : only record the flag here; the
                        # `performance_status` escalation is applied AFTER
                        # the / promotion (which lifts the "error"
                        # default to "ok") via `_raise_performance_status`,
                        # so drift + a sub-µs run both escalate by
                        # precedence and the more severe verdict wins.
                        sampled_outputs_drift = True
                        if ("measured_outputs_drift"
                                not in result.performance_flags):
                            result.performance_flags.append(
                                "measured_outputs_drift")

                # --- : durations-derived disclosure stats ----------
                # median / p99 / cv / trials (DESIGN §6.6). The median is
                # the t_npu the Roofline path scores against below.
                dstats = scoring.duration_stats(durations)

                result.shape_results.append(
                    ShapeRunResult(sid, passed, reason, None, None,
                                   metamorphic_results=list(metamorphic),
                                   durations=durations,
                                   event_ns=event_ns,
                                   sampled_outputs_checked=sampled_checked,
                                   t_npu=dstats.median,
                                   median=dstats.median,
                                   p99=dstats.p99,
                                   cv=dstats.cv,
                                   trials=dstats.trials))

            ran = list(result.shape_results)
            if ran:
                result.runtime_status = (
                    "timeout" if cand.timed_out and not cand.ran_any
                    else "ok" if cand.ran_any else "error")
                result.correctness_status = (
                    "ok" if all(r.correctness_passed for r in ran)
                    else "fail")

                # : the timing pass is a SEPARATE subprocess from the
                # correctness pass — it can time out, crash, or fail to
                # load entirely while the correctness pass succeeded. A
                # `performance_status` promotion off the "error" default
                # must therefore be gated on the *timing* pass really
                # succeeding, not on `cand` (the correctness pass). The
                # timing pass is "ok" only when the child neither timed
                # out nor hit a load error AND it actually produced a
                # per-shape host-timing series for at least one graded
                # shape. A failed timing pass leaves `performance_status`
                # at "error" and reflects the failure on `runtime_status`.
                timing_pass = two.timing
                timing_has_durations = any(
                    bool(r.durations) for r in ran
                )
                timing_ok = (
                    not timing_pass.timed_out
                    and timing_pass.load_error is None
                    and timing_pass.ran_any
                    and timing_has_durations
                )
                if not timing_ok:
                    # Surface the timing-pass failure on runtime_status
                    # (only when the correctness pass had not already set
                    # a non-"ok" verdict — a real correctness-pass
                    # timeout/error is the stronger signal and stays).
                    if result.runtime_status == "ok":
                        if timing_pass.timed_out:
                            result.runtime_status = "timeout"
                        else:
                            result.runtime_status = "error"

                # : a clean two-pass run — the candidate ran and the
                # timing pass produced a clean per-shape host-timing
                # series — promotes the performance verdict off its
                # "error" default to "ok". `performance_status` defaults
                # to "error" (no run); the arbitration / diagnostics /
                # Roofline steps below are escalate-only (via
                # `_raise_performance_status`: ok → suspicious →
                # quarantined, never a roll-back).
                # : the promotion is gated on `timing_ok` — a
                # timing pass that timed out / crashed / shipped no
                # per-shape timing data must NOT yield an ok/ok/ok
                # verdict with an empty timing series.
                if result.performance_status == "error" and timing_ok:
                    result.performance_status = "ok"

                # : now that the verdict has been promoted off the
                # "error" default, apply the sampled-output drift verdict
                # via the precedence helper. It escalates "ok" to
                # "suspicious"; a later sub-µs arbitration can still
                # escalate further to "quarantined" because every set-site
                # routes through `_raise_performance_status`.
                if sampled_outputs_drift:
                    _raise_performance_status(result, "suspicious")

                # ---  / : timing arbitration ------------------
                # Pool every shape's host_ns series and apply the
                # docs/reference/timing_arbitration.md rules. The verdict maps onto
                # the EXISTING result fields (runtime/performance_status,
                # performance_flags) — no new timing_status field.
                #
                # : the host/event-drift rule is now end-to-end
                # connected. The NPU event-timer series travels up from
                # the timing-pass child; it is pooled aligned with the
                # host durations (a shape with no event timer pads `None`,
                # which `arbitrate_timing` skips). `stream_nondefault` is
                # the  profiler verdict — drift > 10% AND a non-default
                # stream ⇒ invalid_timing. On the dev machine there is no
                # NPU, so `event_ns` is all-None and arbitration degrades
                # to host-only; on the 910B the channel is live.
                all_durations: list[float] = []
                all_event_ns: list[float | None] = []
                any_event = False
                for r in ran:
                    ev = r.event_ns
                    for i, d in enumerate(r.durations):
                        all_durations.append(d)
                        if ev is not None and i < len(ev):
                            all_event_ns.append(ev[i])
                            any_event = True
                        else:
                            all_event_ns.append(None)
                stream_nondefault = any(
                    (two.timing.per_shape.get(r.shape_id) is not None
                     and two.timing.per_shape[r.shape_id].stream_nondefault
                     is True)
                    for r in ran
                )
                if all_durations:
                    arb = timing.arbitrate_timing(
                        all_durations,
                        event_ns=all_event_ns if any_event else None,
                        stream_nondefault=stream_nondefault,
                    )
                    if arb.runtime_status:
                        result.runtime_status = arb.runtime_status
                    # : route the arbitration verdict through the
                    # precedence helper. `arbitrate_timing` only ever
                    # emits "quarantined" — strictly more severe than
                    # "suspicious" — so a run that is both drift-flagged
                    # and sub-µs is correctly escalated to "quarantined"
                    # (the old `!= "suspicious"` guard wrongly froze it at
                    # "suspicious"). The helper is a no-op when the
                    # current verdict is already at least as severe.
                    if arb.performance_status:
                        _raise_performance_status(
                            result, arb.performance_status)
                    for flag in arb.performance_flags:
                        if flag not in result.performance_flags:
                            result.performance_flags.append(flag)

                # --- : child-side runtime defense diagnostics -----
                # Fold the candidate child's profiler signals (fallback /
                # non-default stream / kernel-event) into the run status:
                # a fallback sets fallback_detected + the torch_fallback
                # flag; a non-default stream sets invalid_stream +
                # suspicious. Done before Roofline so a fallback/
                # quarantined task is already marked.
                _apply_runtime_diagnostics(result, two, backend=backend)

                # --- : fallback → correctness-fail normalisation ---
                # A detected torch-fallback (fallback_detected, set just
                # above only on an explicit child-side fallback_signal=True)
                # forces correctness_status="fail" before scoring: a
                # correct-but-fallback solution must not clear the
                # correctness gate (DESIGN §6.3 / §6.7). A None signal
                # (dev machine, profiler not measured) never reaches here.
                _normalise_fallback_to_correctness_fail(result)

                # --- : Roofline scoring -------------------------------
                # Per-shape t_ref / W_c / M_r / e and the task-level s_t
                # (geomean of the per-shape e, DESIGN §9). The per-shape
                # efficiency math lives in `scoring`; the e → s_t geomean
                # is aggregated here. Fail-closed on unverified peaks.
                _apply_roofline(result, task, shapes_doc, peak_table_path)
    except Exception as e:
        # E1 fix: orchestration/harness-level errors in phase-2b grading (missing import, yaml
        # parse failure, oracle module load failure, etc.) must never be silently swallowed by `logger.debug` --
        # otherwise the task stalls at the default `correctness_status="error"`, appearing as "not evaluated".
        # Explicitly mark the error status and log the reason at ERROR level. The candidate's own runtime exceptions
        # have already been downgraded to per-shape fail inside the per-shape try block above, and never reach here.
        reason = f"{type(e).__name__}: {e}"
        logger.error("phase-2b grading harness error for %s: %s", task_id, reason)
        result.correctness_status = "error"
        result.runtime_status = "error"
        result.error_reason = reason

    shape_meas = result.shape_measurements()
    if shape_meas:
        # : `aggregate_score` is this task's Roofline score s_t when
        # the Roofline path produced one (DESIGN §9); otherwise it falls
        # back to the legacy per-shape capped geomean so a task without a
        # cost model still reports a diagnostic number.
        if result.s_t is not None:
            result.aggregate_score = result.s_t
        else:
            result.aggregate_score = scoring.capped_geomean_native(
                shape_meas, s_native_size=max(len(shape_meas), 1)
            )
        result.coverage = scoring.coverage_for_ranking(shape_meas)
        result.rank_eligible = result.coverage >= scoring.MIN_COVERAGE_FOR_RANKING
    return result
