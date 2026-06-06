"""adversarial.py — canary detection + metamorphic-check execution.

Implements:
- Metamorphic check runner (per task's `meta.runner_checks.metamorphic_checks`)
- Canary trap detection (per `specs/canary_task_design.md` C1–C10)
- Hidden adversarial shape rotation (R18-S5; sealed shape access protocol)
- The authoritative metamorphic-check registry (`KNOWN_METAMORPHIC_CHECKS`)

Canary traps (live inside `tasks/_canary/`) are tasks that look like real
ones but have a known wrong oracle or impossible shape. A submitter that
passes them is flagged as gaming the runner.

W2.4 — two **separate** detection paths, with distinct result schemas:

* :func:`run_metamorphic_checks` runs declared transformations against a
  *real* task and yields :class:`MetamorphicResult` records the runner
  attaches to that task's ``ShapeRunResult.metamorphic_results``.
* :func:`check_canaries` judges a submission's behaviour on the *canary
  trap* tasks and yields :class:`CanaryFinding` records — an independent
  schema that never mixes into ``metamorphic_results``. Any triggered
  canary escalates the whole submission to ``suspicious``.

The metamorphic-check registry (:data:`KNOWN_METAMORPHIC_CHECKS`) is the
single source of truth for which ``meta.runner_checks.metamorphic_checks``
names are legal — ``scripts/validate_tasks.py`` imports it so an unknown
check name is rejected at validation time, not silently skipped at run
time.
"""
from __future__ import annotations

import functools
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Metamorphic-check registry (W2.4 — authoritative source of truth)
# ─────────────────────────────────────────────────────────────────────────────
# Every legal `meta.runner_checks.metamorphic_checks` *name*. A task may
# only declare a check listed here; `scripts/validate_tasks.py` imports
# this set and rejects an unknown name at validation time (W2.3 used to
# carry a bootstrap copy — it now imports this registry instead).
#
# Membership ≠ a registered transformation callable. The runner supplies
# the actual transform+rerun+compare callable per task and passes it to
# `run_metamorphic_checks`; this set governs which *names* are admissible.
# `differential_perturbation_general` is the design-plan baseline generic
# check (W2.4); the rest are the task-specific checks the corpus declares.
KNOWN_METAMORPHIC_CHECKS: frozenset[str] = frozenset({
    "differential_perturbation_general",
    "empty_allowed_is_noop_stage1",
    "linear_x3_scaling",
    "pos_ge_min_len_keeps_stop_tokens",
    "qk_scale_inverse",
    "row_independence",
    "scale_homogeneity_a",
    "single_token_per_seq_equals_v",
    "weight_one_identity_eps_zero",
    "zero_a_yields_zero",
    "zero_bias_is_noop_stage2",
    "zero_v_yields_zero",
    "zero_x1_yields_zero",
})


def is_known_metamorphic_check(name: str) -> bool:
    """True if ``name`` is a registered metamorphic-check name.

    The validator (`scripts/validate_tasks.py`) uses this — and the
    :data:`KNOWN_METAMORPHIC_CHECKS` set directly — as the single
    authority for which check names a task may declare.
    """
    return name in KNOWN_METAMORPHIC_CHECKS


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
    """Run each declared metamorphic check against a real task.

    `checks` is the task's ``meta.runner_checks.metamorphic_checks`` list;
    `transformations` maps a check's ``name`` to the actual transform +
    re-run + compare callable the runner supplies per task. Results feed
    the task's ``ShapeRunResult.metamorphic_results`` — they are NOT
    canary findings (see :func:`check_canaries`).

    A check name absent from :data:`KNOWN_METAMORPHIC_CHECKS` is rejected
    here as a runtime backstop; the primary gate is the validator
    (`scripts/validate_tasks.py`), which refuses an unknown name at
    validation time.
    """
    out: list[MetamorphicResult] = []
    for spec in checks:
        name = spec["name"]
        if not is_known_metamorphic_check(name):
            out.append(MetamorphicResult(
                name, False,
                f"unknown metamorphic check {name!r} (not in registry)"))
            continue
        fn = transformations.get(name)
        if fn is None:
            out.append(MetamorphicResult(name, False, "no transformation registered"))
            continue
        try:
            out.append(fn())
        except Exception as exc:                       # noqa: BLE001
            out.append(MetamorphicResult(name, False, f"raised: {type(exc).__name__}: {exc}"))
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Metamorphic transformation registry (W2.4 / R7-3 — real transformations)
# ─────────────────────────────────────────────────────────────────────────────
# A metamorphic check is: transform the candidate's inputs, re-run the
# candidate, and verify the output satisfies a structural invariant. The
# 13 corpus check names fall into a handful of GENERIC structural families
# — implemented once here as operator-agnostic transformations, never
# per-operator:
#
#   * annihilation   — zero a named input ⇒ every output is all-zero.
#       (zero_x1_yields_zero / zero_a_yields_zero / zero_v_yields_zero)
#   * homogeneity    — scale a named input by k ⇒ output scales by k.
#       (linear_x3_scaling / scale_homogeneity_a)
#   * compensating   — scale input A by α and input B by 1/α ⇒ output
#       unchanged. (qk_scale_inverse)
#   * row-permute    — permute leading-dim rows of every rank≥2 input ⇒
#       output rows permute identically. (row_independence)
#   * differential   — perturb an input ⇒ output MUST change (a candidate
#       that ignores inputs / returns a constant is caught).
#       (differential_perturbation_general)
#
# A few corpus checks (`weight_one_identity_eps_zero`,
# `single_token_per_seq_equals_v`, `empty_allowed_is_noop_stage1`,
# `zero_bias_is_noop_stage2`, `pos_ge_min_len_keeps_stop_tokens`) encode
# an operator-specific invariant — which input is "the weight", what
# "stage 1" is, which row slice is a no-op — that cannot be expressed as a
# generic structural transform. They are NOT generically implementable;
# the dispatcher reports them with an explicit
# `not generically implementable` detail (a passive non-pass), distinct
# from the old misleading `no transformation registered`.

# Each generic family maps to one of the dispatch kinds below; the
# per-check parameters (which input to zero / scale, etc.) are inferred
# from the input set at dispatch time.
_ANNIHILATION_CHECKS: frozenset[str] = frozenset({
    "zero_x1_yields_zero", "zero_a_yields_zero", "zero_v_yields_zero",
})
_HOMOGENEITY_CHECKS: frozenset[str] = frozenset({
    "linear_x3_scaling", "scale_homogeneity_a",
})
_COMPENSATING_CHECKS: frozenset[str] = frozenset({"qk_scale_inverse"})
_ROW_PERMUTE_CHECKS: frozenset[str] = frozenset({"row_independence"})
_DIFFERENTIAL_CHECKS: frozenset[str] = frozenset({
    "differential_perturbation_general",
})

# Operator-specific invariants with no generic structural form (see the
# block comment above). Reported as a non-pass with an explicit reason.
_OPERATOR_SPECIFIC_CHECKS: frozenset[str] = frozenset({
    "weight_one_identity_eps_zero",
    "single_token_per_seq_equals_v",
    "empty_allowed_is_noop_stage1",
    "zero_bias_is_noop_stage2",
    "pos_ge_min_len_keeps_stop_tokens",
})

# The hint each check encodes about WHICH input it targets. The generic
# transform picks the named input when present; otherwise it falls back to
# the first eligible float tensor (annihilation/homogeneity) so the check
# still exercises the candidate.
_CHECK_INPUT_HINT: dict[str, str] = {
    "zero_x1_yields_zero": "x1",
    "zero_a_yields_zero": "A",
    "zero_v_yields_zero": "v",
    "linear_x3_scaling": "x3",
    "scale_homogeneity_a": "As",
}


@dataclass
class MetamorphicContext:
    """Everything a metamorphic transformation needs to actually run.

    The runner (``eval._run_task_metamorphic``) builds this once per task
    and hands it to :func:`build_metamorphic_transformations`, which turns
    it into the ``{check_name: callable}`` registry
    :func:`run_metamorphic_checks` consumes. Keeping the heavy bits
    (candidate-run callable, tensor compare) behind this struct lets
    ``adversarial`` stay import-light — no torch / process_isolation
    import at module load.

    Attributes
    ----------
    base_inputs:
        A representative built inputs dict (tensors + scalars) for one
        shape — the metamorphic transforms are applied to a copy of it.
    run_candidate:
        ``callable(inputs_dict) -> dict | None``. Runs the candidate
        once on the given inputs and returns its output tensor dict, or
        ``None`` if the candidate did not run (load error / crash).
    tensors_close:
        ``callable(actual, expected) -> bool``. Tolerance-aware tensor
        comparison (the task's tolerance class).
    """

    base_inputs: dict
    run_candidate: "Callable[[dict], dict | None]"
    tensors_close: "Callable[[Any, Any], bool]"


def _float_tensor_names(inputs: dict) -> list[str]:
    """Names of the floating-point tensor inputs, declaration order."""
    import torch

    return [
        name for name, val in inputs.items()
        if torch.is_tensor(val) and torch.is_floating_point(val)
    ]


def _pick_target(ctx: "MetamorphicContext", check_name: str) -> str | None:
    """Resolve which input tensor a check targets.

    Prefers the check's declared hint (``x1`` / ``A`` / ``v`` / …); falls
    back to the first floating-point tensor input. ``None`` when the task
    has no usable float tensor input at all.
    """
    floats = _float_tensor_names(ctx.base_inputs)
    if not floats:
        return None
    hint = _CHECK_INPUT_HINT.get(check_name)
    if hint and hint in floats:
        return hint
    return floats[0]


def _annihilation_transform(
    ctx: "MetamorphicContext", check_name: str,
) -> MetamorphicResult:
    """Zero a named input ⇒ every candidate output must be all-zero.

    Catches a candidate that ignores the zeroed operand (e.g. a hard-coded
    or cached non-zero answer).
    """
    import torch

    target = _pick_target(ctx, check_name)
    if target is None:
        return MetamorphicResult(
            check_name, False,
            "no floating-point tensor input to zero")
    perturbed = dict(ctx.base_inputs)
    perturbed[target] = torch.zeros_like(ctx.base_inputs[target])
    out = ctx.run_candidate(perturbed)
    if not out:
        return MetamorphicResult(
            check_name, False, "candidate did not run on zeroed input")
    zero_tensors = [v for v in out.values()
                    if torch.is_tensor(v) and torch.is_floating_point(v)]
    if not zero_tensors:
        return MetamorphicResult(
            check_name, False, "candidate produced no float output")
    for v in zero_tensors:
        if not ctx.tensors_close(v, torch.zeros_like(v)):
            return MetamorphicResult(
                check_name, False,
                f"zeroing {target!r} did not yield an all-zero output "
                f"(max |out| = {float(v.abs().max()):.3e})")
    return MetamorphicResult(
        check_name, True, f"zeroing {target!r} yields all-zero output")


def _homogeneity_transform(
    ctx: "MetamorphicContext", check_name: str,
) -> MetamorphicResult:
    """Scale a named input by k ⇒ every output scales by exactly k.

    A candidate that clamps / saturates / ignores the operand, or applies
    a non-linear stage where the task is linear in this input, fails here.
    """
    import torch

    k = 2.0
    target = _pick_target(ctx, check_name)
    if target is None:
        return MetamorphicResult(
            check_name, False,
            "no floating-point tensor input to scale")
    base_out = ctx.run_candidate(ctx.base_inputs)
    if not base_out:
        return MetamorphicResult(
            check_name, False, "candidate did not run on the base input")
    scaled_inputs = dict(ctx.base_inputs)
    scaled_inputs[target] = ctx.base_inputs[target] * k
    scaled_out = ctx.run_candidate(scaled_inputs)
    if not scaled_out:
        return MetamorphicResult(
            check_name, False, "candidate did not run on the scaled input")
    compared = 0
    for key, base_v in base_out.items():
        if not (torch.is_tensor(base_v)
                and torch.is_floating_point(base_v)):
            continue
        scaled_v = scaled_out.get(key)
        if not torch.is_tensor(scaled_v):
            return MetamorphicResult(
                check_name, False, f"output {key!r} missing on scaled run")
        compared += 1
        if not ctx.tensors_close(scaled_v, base_v * k):
            return MetamorphicResult(
                check_name, False,
                f"scaling {target!r} by {k} did not scale output {key!r} "
                f"by {k}")
    if compared == 0:
        return MetamorphicResult(
            check_name, False, "candidate produced no float output")
    return MetamorphicResult(
        check_name, True,
        f"scaling {target!r} by {k} scales every output by {k}")


def _compensating_transform(
    ctx: "MetamorphicContext", check_name: str,
) -> MetamorphicResult:
    """Scale one input by α and another by 1/α ⇒ output unchanged.

    The ``qk_scale_inverse`` invariant: (Q, K) → (αQ, K/α) preserves the
    q·k product. Generic form: scale the first two float tensor inputs by
    α and 1/α; the output must be unchanged within tolerance.
    """
    import torch

    alpha = 4.0
    floats = _float_tensor_names(ctx.base_inputs)
    if len(floats) < 2:
        return MetamorphicResult(
            check_name, False,
            "needs >=2 float tensor inputs for a compensating pair")
    a_name, b_name = floats[0], floats[1]
    base_out = ctx.run_candidate(ctx.base_inputs)
    if not base_out:
        return MetamorphicResult(
            check_name, False, "candidate did not run on the base input")
    perturbed = dict(ctx.base_inputs)
    perturbed[a_name] = ctx.base_inputs[a_name] * alpha
    perturbed[b_name] = ctx.base_inputs[b_name] / alpha
    comp_out = ctx.run_candidate(perturbed)
    if not comp_out:
        return MetamorphicResult(
            check_name, False,
            "candidate did not run on the compensated input")
    compared = 0
    for key, base_v in base_out.items():
        if not (torch.is_tensor(base_v)
                and torch.is_floating_point(base_v)):
            continue
        comp_v = comp_out.get(key)
        if not torch.is_tensor(comp_v):
            return MetamorphicResult(
                check_name, False,
                f"output {key!r} missing on compensated run")
        compared += 1
        if not ctx.tensors_close(comp_v, base_v):
            return MetamorphicResult(
                check_name, False,
                f"compensating ({a_name}*{alpha}, {b_name}/{alpha}) changed "
                f"output {key!r} — invariant violated")
    if compared == 0:
        return MetamorphicResult(
            check_name, False, "candidate produced no float output")
    return MetamorphicResult(
        check_name, True,
        f"compensating ({a_name}*{alpha}, {b_name}/{alpha}) leaves the "
        f"output unchanged")


def _row_permute_transform(
    ctx: "MetamorphicContext", check_name: str,
) -> MetamorphicResult:
    """Permute leading-dim rows of every rank>=2 input ⇒ output rows
    permute identically.

    Detects cross-row contamination (a candidate where row r's output
    leaks into row r'). The permutation is applied to every rank>=2 input
    AND inverted on every rank>=2 output before the compare.
    """
    import torch

    base_out = ctx.run_candidate(ctx.base_inputs)
    if not base_out:
        return MetamorphicResult(
            check_name, False, "candidate did not run on the base input")
    # Row count = leading dim shared by the rank>=2 tensor inputs.
    rank2 = [v for v in ctx.base_inputs.values()
             if torch.is_tensor(v) and v.dim() >= 2]
    if not rank2:
        return MetamorphicResult(
            check_name, False, "no rank>=2 tensor input to permute rows of")
    n_rows = rank2[0].shape[0]
    if n_rows < 2:
        return MetamorphicResult(
            check_name, False, "leading dim < 2 — no row permutation")
    perm = torch.randperm(n_rows, generator=torch.Generator().manual_seed(0))
    permuted = {}
    for name, val in ctx.base_inputs.items():
        if (torch.is_tensor(val) and val.dim() >= 2
                and val.shape[0] == n_rows):
            permuted[name] = val[perm].contiguous()
        else:
            permuted[name] = val
    perm_out = ctx.run_candidate(permuted)
    if not perm_out:
        return MetamorphicResult(
            check_name, False, "candidate did not run on the permuted input")
    inv = torch.argsort(perm)
    compared = 0
    for key, base_v in base_out.items():
        if not (torch.is_tensor(base_v) and base_v.dim() >= 2
                and base_v.shape[0] == n_rows):
            continue
        perm_v = perm_out.get(key)
        if not torch.is_tensor(perm_v) or perm_v.shape[0] != n_rows:
            return MetamorphicResult(
                check_name, False,
                f"output {key!r} not row-shaped on the permuted run")
        compared += 1
        # Un-permuting the permuted output must reproduce the base output.
        if not ctx.tensors_close(perm_v[inv].contiguous(), base_v):
            return MetamorphicResult(
                check_name, False,
                f"permuting input rows did not permute output {key!r} "
                f"identically — cross-row contamination")
    if compared == 0:
        return MetamorphicResult(
            check_name, False,
            "candidate produced no row-shaped float output")
    return MetamorphicResult(
        check_name, True, "input row permutation permutes the output rows")


def _differential_transform(
    ctx: "MetamorphicContext", check_name: str,
) -> MetamorphicResult:
    """Perturb a float input ⇒ the output MUST change.

    The design-plan generic baseline check: a candidate that returns a
    constant / cached / input-independent answer produces the same output
    for a perturbed input and is caught here.
    """
    import torch

    target = _pick_target(ctx, check_name)
    if target is None:
        return MetamorphicResult(
            check_name, False,
            "no floating-point tensor input to perturb")
    base_out = ctx.run_candidate(ctx.base_inputs)
    if not base_out:
        return MetamorphicResult(
            check_name, False, "candidate did not run on the base input")
    perturbed = dict(ctx.base_inputs)
    noise = torch.randn(
        ctx.base_inputs[target].shape,
        generator=torch.Generator().manual_seed(0),
        dtype=torch.float32,
    ).to(ctx.base_inputs[target].dtype)
    perturbed[target] = ctx.base_inputs[target] + noise
    pert_out = ctx.run_candidate(perturbed)
    if not pert_out:
        return MetamorphicResult(
            check_name, False, "candidate did not run on the perturbed input")
    changed = False
    compared = 0
    for key, base_v in base_out.items():
        if not (torch.is_tensor(base_v)
                and torch.is_floating_point(base_v)):
            continue
        pert_v = pert_out.get(key)
        if not torch.is_tensor(pert_v):
            return MetamorphicResult(
                check_name, False,
                f"output {key!r} missing on the perturbed run")
        compared += 1
        if not ctx.tensors_close(pert_v, base_v):
            changed = True
    if compared == 0:
        return MetamorphicResult(
            check_name, False, "candidate produced no float output")
    if not changed:
        return MetamorphicResult(
            check_name, False,
            f"perturbing {target!r} left every output unchanged — the "
            f"candidate ignores its input (constant / cached answer)")
    return MetamorphicResult(
        check_name, True,
        f"perturbing {target!r} changes the output (input-dependent)")


def build_metamorphic_transformations(
    checks: list[dict[str, Any]],
    ctx: "MetamorphicContext",
) -> dict[str, "Callable[[], MetamorphicResult]"]:
    """Build the ``{check_name: callable}`` transformation registry for a
    task's declared metamorphic checks (R7-3).

    Each entry is a zero-arg closure :func:`run_metamorphic_checks` calls.
    A check whose name maps to a generic structural family
    (annihilation / homogeneity / compensating / row-permute /
    differential) gets a real transformation; an operator-specific check
    with no generic form gets a closure that returns an explicit
    ``not generically implementable`` non-pass — never the misleading
    ``no transformation registered``.

    An unknown check name is left ABSENT from the registry on purpose:
    :func:`run_metamorphic_checks` rejects it as
    ``unknown metamorphic check`` before the transform would run.
    """
    registry: dict[str, "Callable[[], MetamorphicResult]"] = {}
    for spec in checks:
        name = spec.get("name") if isinstance(spec, dict) else spec
        if not isinstance(name, str) or not is_known_metamorphic_check(name):
            # Unknown name — leave it out so run_metamorphic_checks emits
            # its own `unknown metamorphic check` verdict.
            continue
        if name in _ANNIHILATION_CHECKS:
            registry[name] = (
                lambda n=name: _annihilation_transform(ctx, n))
        elif name in _HOMOGENEITY_CHECKS:
            registry[name] = (
                lambda n=name: _homogeneity_transform(ctx, n))
        elif name in _COMPENSATING_CHECKS:
            registry[name] = (
                lambda n=name: _compensating_transform(ctx, n))
        elif name in _ROW_PERMUTE_CHECKS:
            registry[name] = (
                lambda n=name: _row_permute_transform(ctx, n))
        elif name in _DIFFERENTIAL_CHECKS:
            registry[name] = (
                lambda n=name: _differential_transform(ctx, n))
        elif name in _OPERATOR_SPECIFIC_CHECKS:
            registry[name] = (
                lambda n=name: MetamorphicResult(
                    n, False,
                    "not generically implementable — operator-specific "
                    "invariant (no structural transform); runner long-tail"))
    return registry


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


def canary_finding_json(f: CanaryFinding) -> dict[str, Any]:
    """Serialise a :class:`CanaryFinding` for a round_result.json.

    Canary findings carry their OWN schema — keyed by ``canary_id`` and
    ``severity`` — and are emitted in a dedicated ``canary_findings``
    array, kept distinct from per-task ``metamorphic_results`` (W2.4 §3).
    """
    return {
        "canary_id": f.canary_id,
        "triggered": f.triggered,
        "severity": f.severity,
        "message": f.message,
    }


@dataclass
class CanaryAggregate:
    """Submission-level canary verdict (W2.4 escalation chain).

    `canary_caught` is the count of *triggered* canary findings across the
    whole submission. ``canary_caught > 0`` escalates the submission to
    ``suspicious`` (`suspicious is True`), gating it for manual review
    (DESIGN §6.8 / §8). Canaries never count toward the main score — they
    are a behaviour probe, not a score-eligible task.
    """

    canary_caught: int
    suspicious: bool
    findings: list[CanaryFinding] = field(default_factory=list)


def aggregate_canary_findings(
    findings: list[CanaryFinding],
) -> CanaryAggregate:
    """Roll per-canary findings up to a submission-level verdict.

    Counts the triggered findings; any triggered canary (``canary_caught
    > 0``) flips ``suspicious`` so the submission is held for manual
    review. This is the W2.4 canary escalation chain `cli run-suite` /
    `analyze` invoke for submission-level aggregation.
    """
    caught = sum(1 for f in findings if f.triggered)
    return CanaryAggregate(
        canary_caught=caught,
        suspicious=caught > 0,
        findings=list(findings),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Canary suite runner (W2.5 — discovery → bound-solution run → check_canaries)
# ─────────────────────────────────────────────────────────────────────────────
# `iter_all_tasks()` skips `_`-prefixed dirs, so canaries never enter the
# score-eligible set. The runner below is the independent path: it discovers
# canaries via `dataset.iter_canary_tasks()`, runs each one with the
# submission's solution for its bound `source_task_id`, and feeds the verdict
# to `check_canaries`. Canaries are a behaviour probe — never a scored task.


def _evaluate_canary_pass(
    canary: Any, solution_path: Path, backend: str = "npu") -> bool:
    """Default canary grader: True iff `solution_path` PASSES `canary`'s trap.

    Grades the submission's bound solution against the canary's (trap) oracle
    via `eval.evaluate(task_def=...)` — "passing the canary" means clearing
    its correctness gate. A harness error grading the canary counts as "did
    not pass" so an honest canary is never spuriously flagged. `backend`
    routes the trap run to the same device as the scored tasks.

    Imported lazily: `eval` pulls in heavy modules (torch, process_isolation)
    that the lightweight canary discovery / judgement path does not need.
    """
    from .eval import evaluate

    try:
        result = evaluate(
            canary.canary_id, solution_path, task_def=canary.as_task_def(),
            backend=backend)
        return result.correctness_status == "ok"
    except Exception as exc:
        logger.warning(
            "canary %s: grading error: %s: %s",
            canary.canary_id, type(exc).__name__, exc)
        return False


def run_canary_suite(
    solutions_root: Path,
    grader: Callable[[Any, Path], bool] | None = None,
    backend: str = "npu",
) -> list[CanaryFinding]:
    """Run every discovered canary against a submission and judge it.

    For each canary task under `tasks/_canary/`:

    1. Discover it via the dedicated enumerator
       :func:`dataset.iter_canary_tasks` (NOT `iter_all_tasks`, which skips
       `_`-prefixed dirs — so canaries are never score-eligible).
    2. Resolve the canary's bound `source_task_id` to the submission's
       `solution.py` for that real task. The submitter writes no canary
       solution; the canary reuses the bound task's solution to observe how
       the submitter's kernel behaves on the trap.
    3. Grade that solution against the canary's (trap) oracle — by default
       :func:`_evaluate_canary_pass`, which runs `eval.evaluate(task_def=...)`.
       `grader` is injectable so the framework can be exercised without a
       full NPU/triton run (dev-machine wiring test).
    4. Compare each observation against the canary's `expected_honest_pass`
       through :func:`check_canaries`; a `got != expected` mismatch is a
       triggered finding that escalates the submission to `suspicious`.

    A canary whose bound solution is missing is skipped (no finding) — an
    absent solution is graded by the real task, not the canary.
    """
    from .dataset import iter_canary_tasks

    grade = grader if grader is not None else functools.partial(
        _evaluate_canary_pass, backend=backend)
    canaries = iter_canary_tasks()
    submitted: dict[str, bool] = {}
    truth: dict[str, bool] = {}

    for canary in canaries:
        sol_py = Path(solutions_root) / canary.source_task_id / "solution.py"
        if not sol_py.is_file():
            logger.info(
                "canary %s: no solution for bound source_task_id %s — skipping",
                canary.canary_id, canary.source_task_id)
            continue
        truth[canary.canary_id] = canary.expected_honest_pass
        submitted[canary.canary_id] = bool(grade(canary, sol_py))

    return check_canaries(submitted, truth)
