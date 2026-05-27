"""eligibility.py -- single source of truth for the score-eligible set.

Extracted from the inline logic previously in ``cli.py`` (~L597). A
task is *score-eligible* iff it has a real working oracle (not a
``raise NotImplementedError`` stub) and its ``shapes.yaml`` is not
``PHASE-C-MANUAL``. Counting stub/manual tasks as failures would
systematically depress valid submissions, so they are excluded from
the denominator (NOT counted as failures).

This module is **side-effect-light** by design (plan U7):

  * pure functions only;
  * file reads happen at **call time**, never at import time;
  * ZERO import-time I/O.

Both ``cli.py`` (analyze / release_blocked) and
``scripts/validate_tasks.py`` (release gate) import this module so the
score-eligible computation has a single source of truth and cannot
drift between the two call sites (plan: "API surface parity").
"""
from __future__ import annotations

from collections.abc import Iterable
from typing import Protocol


class _TaskLike(Protocol):
    task_id: str

    # `task_dir` is a pathlib.Path in practice; kept structural so this
    # module never imports dataset (avoids an import cycle / I/O).
    @property
    def task_dir(self): ...

    # `meta` is the task's parsed meta.yaml dict; structural so this
    # module never imports dataset.
    @property
    def meta(self) -> dict: ...


def is_score_eligible(task: _TaskLike) -> bool:
    """True iff `task` counts toward the scored denominator.

    A task is NOT eligible when:
      * ``meta.yaml.release_scope.excluded`` is ``True`` — the
        orchestrator has explicitly marked the task unfixable by the
        LLM patch flow and removed it from release scoring
        (``specs/task_api_contract.md`` §6a.5). The task still exists
        in the corpus (no subset escape — the denominator is NOT
        reduced by silently dropping the task here; the eligibility
        gate alone subtracts it from the scored set).
      * its oracle is a stub (``raise NotImplementedError``), or
      * its ``shapes.yaml`` is ``PHASE-C-MANUAL``.

    Any OS error reading the task is treated as NOT eligible (the
    same conservative behaviour as the original inline ``cli.py``
    code, which ``continue``-d on ``OSError``).
    """
    meta = getattr(task, "meta", None) or {}
    release_scope = meta.get("release_scope") or {}
    if release_scope.get("excluded") is True:
        return False
    try:
        osrc = (task.task_dir / "oracle_pytorch.py").read_text()
        shp = (task.task_dir / "shapes.yaml").read_text()
    except OSError:
        return False
    if "raise NotImplementedError" in osrc or "PHASE-C-MANUAL" in shp:
        return False
    return True


def compute_score_eligible(tasks: Iterable[_TaskLike]) -> set[str]:
    """Return the set of score-eligible ``task_id``s.

    This is the single authority used by both ``cli.py analyze`` and
    ``scripts/validate_tasks.py --release-gate``. Behaviour is
    byte-identical to the previous inline ``cli.py`` loop.
    """
    return {t.task_id for t in tasks if is_score_eligible(t)}


# ── W4.3 ④: self-reported analyze-denominator authority ──────────────
#
# `meta.yaml.oracle_simplified == "upstream_oracle_required"` marks a
# task whose golden output can only come from running the frozen
# upstream kernel offline (`specs/task_api_contract.md` §oracle_simplified
# enum). In `self-reported` mode the verified runner has no provider
# golden for such a task, so it is `task_status = unsupported` and is
# excluded from that mode's scored denominator.
#
# This set is derived ONLY from trusted repo `meta` — it never depends
# on a submission's reported `task_status`. `cli.py analyze` subtracts it
# from the score-eligible set to fix the self-reported denominator, so a
# malicious / corrupt result cannot move a failing task out of the
# denominator by claiming `unsupported`.


def is_upstream_oracle_required(task: _TaskLike) -> bool:
    """True iff `task`'s ``meta.yaml.oracle_simplified`` is
    ``"upstream_oracle_required"`` — i.e. it has no provider golden in
    ``self-reported`` mode."""
    meta = getattr(task, "meta", None) or {}
    return meta.get("oracle_simplified") == "upstream_oracle_required"


def compute_upstream_oracle_required(tasks: Iterable[_TaskLike]) -> set[str]:
    """Return the set of ``task_id``s whose golden requires the upstream
    oracle (no provider golden in ``self-reported`` mode).

    Derived purely from trusted repo ``meta`` — this is the fixed set
    ``cli.py analyze`` subtracts from the score-eligible set to obtain the
    ``self-reported`` denominator (W4.3 ④). It is intentionally NOT
    influenced by any submission-reported ``task_status``.
    """
    return {t.task_id for t in tasks if is_upstream_oracle_required(t)}


# ── W4.3 (R3-2 / R4-2): verified-mode offline-golden completeness ─────
#
# In `verified` mode an `upstream_oracle_required` task is graded against
# the offline `golden/<shape_id>.safetensors` artifacts (collected on the
# GPU machine in W4.3 Phase 4). When that artifact set is missing or does
# not cover every eval shape (public AND hidden), the task cannot be
# graded — it is fail-closed `task_status = "blocked"`.
#
# `analyze` derives this status from TRUSTED REPO STATE (the presence of
# the golden files), never from a submission-reported `task_status`, so a
# hand-edited result cannot fake an "evaluated"/"blocked" verdict.
#
# R4-2: the "golden usable" predicate below is the SINGLE shared check
# both `eligibility` (`golden_complete` / `compute_verified_blocked`) and
# `eval._resolve_golden` consume verbatim. Both must agree on whether a
# task's golden is loadable — a path-exists-only check in one and a
# load-succeeds check in the other let `evaluate()` emit
# `task_status="blocked"` while `compute_verified_blocked()` calls the
# same task `evaluated`, which `analyze` then mis-scores. "Usable" means
# the file exists AND `safetensors.load_file` reads it without raising
# (an unreadable / corrupt artifact is NOT usable, fail-closed).


def golden_artifact_usable(path) -> bool:
    """True iff ``path`` is a usable offline-golden artifact (R4-2).

    The single shared "golden usable" predicate: the file must exist AND
    ``safetensors.torch.load_file`` must read it without raising. A
    missing path or an unreadable / corrupt ``safetensors`` file returns
    ``False`` — fail-closed. Both :func:`golden_complete` (the eligibility
    authority) and ``eval._resolve_golden`` (the runtime loader) call
    this, so their "golden available" judgement cannot drift.
    """
    from pathlib import Path

    p = Path(path)
    if not p.is_file():
        return False
    try:
        from safetensors.torch import load_file

        load_file(str(p))
    except Exception:
        # Corrupt / unreadable / truncated artifact — not usable.
        return False
    return True


def shapes_eval_ids(task: _TaskLike) -> list[str] | None:
    """Enumerate a task's public + hidden eval-shape ids from shapes.yaml.

    Returns the ordered id list, or ``None`` on an OS error / malformed
    document / empty shape set (conservative — the caller treats ``None``
    as "cannot determine the golden set").
    """
    import yaml

    try:
        shapes_doc = yaml.safe_load(
            (task.task_dir / "shapes.yaml").read_text())
    except OSError:
        return None
    if not isinstance(shapes_doc, dict):
        return None
    shape_ids = [
        s.get("id")
        for s in (list(shapes_doc.get("public_shapes") or [])
                  + list(shapes_doc.get("hidden_shapes") or []))
        if isinstance(s, dict) and s.get("id")
    ]
    return shape_ids or None


def golden_complete(task: _TaskLike) -> bool:
    """True iff `task` has a USABLE offline ``golden/`` artifact for every
    eval shape (public AND hidden) it declares.

    Reads ``shapes.yaml`` to enumerate the public + hidden shape ids and
    checks ``tasks/.../golden/<shape_id>.safetensors`` exists AND is
    loadable for each (R4-2: the same usability check ``eval`` applies, so
    the two cannot disagree). An empty / missing shape set, a missing
    ``golden/`` directory, or any missing / corrupt artifact returns
    ``False``. Any OS error is treated as incomplete (conservative,
    matching :func:`is_score_eligible`).
    """
    shape_ids = shapes_eval_ids(task)
    if shape_ids is None:
        return False
    golden_dir = task.task_dir / "golden"
    if not golden_dir.is_dir():
        return False
    return all(
        golden_artifact_usable(golden_dir / f"{sid}.safetensors")
        for sid in shape_ids
    )


def compute_verified_blocked(tasks: Iterable[_TaskLike]) -> set[str]:
    """Return the set of ``task_id``s that are ``blocked`` in verified
    mode — ``upstream_oracle_required`` tasks whose offline ``golden/``
    artifact set is missing or incomplete (W4.3 / R3-2).

    Derived purely from trusted repo state (the presence of the golden
    files). On the dev machine no ``golden/`` directory exists, so every
    ``upstream_oracle_required`` task is in this set — the expected
    pre-Phase-4 state. ``cli.py analyze`` uses it to assign
    ``task_status = "blocked"`` for verified-mode runs.
    """
    return {
        t.task_id
        for t in tasks
        if is_upstream_oracle_required(t) and not golden_complete(t)
    }


# ── W1.5: composite release gate predicate ──────────────────────────
#
# DESIGN-V2 §4.4: a task's release_status is PASS *iff* its provider
# freeze (`phase_a_prime.status`) AND its cost-model GPU cross-check
# (`cost_model.status`) are BOTH PASS:
#
#     release_status == PASS
#         <=> phase_a_prime.status == PASS  AND  cost_model.status == PASS
#
# `cost_model.status` lives at `meta.cost_model.status` and is written
# by the W4.2 GPU cross-check step (PASS / BLOCKED). Most task metas do
# not carry it yet (W4.2 unrun) -- a MISSING `cost_model.status` is
# treated as NOT PASS, so the release gate stays red until Phase 4.
#
# All three release-status authorities -- `validate_tasks.py
# --release-gate`, `cli.py` analyze's `release_blocking`, and
# `phase_a_prime.py`'s S6 verdict report -- call the SAME two functions
# below, so the blocking set is identical across all three by
# construction (no predicate drift).


def release_status_pass(meta: dict | None) -> bool:
    """True iff a task's composite ``release_status`` is PASS.

    Base predicate (DESIGN-V2 §4.4)::

        release_status == PASS
            <=> phase_a_prime.status PASSes  AND  cost_model.status == PASS

    A missing ``phase_a_prime`` block, a missing ``cost_model`` block,
    or a missing ``status`` inside either is treated as NOT PASS.

    ``specs/task_api_contract.md`` §6a refinements:

    * ``phase_a_prime.status == "SKIPPED_BY_DESIGN"`` is also accepted
      as the phase A' clause — the operator cannot be invoked
      standalone on the GPU verify host and the offline phase B golden
      capture (§6a.3) is the authoritative truth source. The status
      is treated as PASS *only when* the corresponding
      ``phase_b_capture.status == "DONE"`` (cross-field invariant
      §6a.6 #5 — without a captured golden the
      ``SKIPPED_BY_DESIGN`` status alone is not a release-ready
      signal).
    * ``phase_b_capture_required == True`` promotes the offline golden
      into the conjunctive predicate: when set, ``release_status``
      requires ``phase_b_capture.status == "DONE"`` regardless of
      whether phase A' is ``PASS`` or ``SKIPPED_BY_DESIGN``.
    * ``release_scope.excluded == True`` is a hard release-blocker: an
      operator-marked excluded task is by definition NOT release-ready
      (the conjunctive gate includes ``NOT release_scope.excluded``).
      This keeps the release-gate authorities aligned with the
      eligibility-layer exclusion -- a task removed from the scored
      set must not simultaneously satisfy ``release_status_pass``.
    """
    m = meta or {}
    pap_status = (m.get("phase_a_prime") or {}).get("status")
    cm_status = (m.get("cost_model") or {}).get("status")
    pbc_status = (m.get("phase_b_capture") or {}).get("status")
    pbc_required = m.get("phase_b_capture_required") is True
    excluded = (m.get("release_scope") or {}).get("excluded") is True

    if excluded:
        return False

    if cm_status != "PASS":
        return False

    if pap_status == "PASS":
        pap_ok = True
    elif pap_status == "SKIPPED_BY_DESIGN":
        # Invariant §6a.6 #5: SKIPPED_BY_DESIGN delegates the truth
        # source to the offline golden — only PASS once the capture
        # is DONE. Invariant §6a.6 #2: SKIPPED_BY_DESIGN must carry a
        # non-empty reason — fail-closed defence in case a malformed
        # block ever reached disk past the patch validator.
        pap_reason = (m.get("phase_a_prime") or {}).get("reason")
        reason_ok = isinstance(pap_reason, str) and bool(pap_reason.strip())
        pap_ok = pbc_status == "DONE" and reason_ok
    else:
        pap_ok = False

    if not pap_ok:
        return False

    if pbc_required and pbc_status != "DONE":
        return False

    return True


# ── V9 plan M0.0: ProvenanceEligibility authority ───────────────────
#
# The release gate previously asked only "is the task PHASE A' + cost-
# model PASS?" (``release_status_pass``). V9 adds a second gate: the
# baseline must have a verified trust chain (oracle equivalence VERIFIED
# OR GPU capture DONE) and the tolerance source must not be the v1-
# deferred ``migration_drift_budget`` fallback.
#
# Same loader-lenient / validator-strict split as the rest of the
# eligibility predicates: these read raw ``meta`` dicts (NO load_task /
# safetensors I/O) so they can be called from any pure-Python context.


def is_provenance_eligible(meta: dict | None) -> bool:
    """V9 plan M0.0 predicate: True iff a task's baseline trust chain
    is release-ready.

    Definition (V9 plan, v1 scope):

    * ``provenance.golden.oracle_kernel_equivalence == VERIFIED`` (the
      corpus oracle_pytorch.py has been GPU-equivalence-verified, R10
      stage 3), OR
    * ``provenance.golden.gpu_kernel_capture_status == DONE`` (the
      provider GPU rerun has been captured + manifest-signed, R5/R20)

    AND

    * ``provenance.golden.tolerance_source != migration_drift_budget``
      (v1 does not implement the migration-drift budget path; tasks
      with this source are not release-ready until v2 lands)

    Grace period: an absent ``provenance.golden:`` block returns
    ``False`` (the task has not yet had its baseline trust chain
    established). This is the expected pre-R10 state for legacy
    corpus tasks.

    Loader-lenient: caller pre-pulls the raw ``meta`` dict; no Python
    object construction, no validation, no file I/O.
    """
    m = meta or {}
    prov = m.get("provenance")
    if not isinstance(prov, dict):
        return False
    golden = prov.get("golden")
    if not isinstance(golden, dict):
        return False

    tol_src = golden.get("tolerance_source")
    if tol_src == "migration_drift_budget":
        # v1 deferred -- task is NOT release-ready until R11 v2 lands.
        return False

    oke = golden.get("oracle_kernel_equivalence")
    gkcs = golden.get("gpu_kernel_capture_status")
    return oke == "VERIFIED" or gkcs == "DONE"


def release_ready_with_provenance(meta: dict | None) -> bool:
    """V9 plan M0.0 composite predicate: PHASE A' + cost_model + R10/R5
    trust chain all green.

    Tightening of :func:`release_status_pass` with the V9 provenance
    gate. Used by ``cli analyze`` and ``validate_tasks --release-gate``
    once the V9 schema has been backfilled. Until then, callers can
    keep using ``release_status_pass`` directly (backward-compatible).
    """
    return release_status_pass(meta) and is_provenance_eligible(meta)


def compute_release_ready_eligible(
    tasks: Iterable[_TaskLike],
) -> set[str]:
    """Return the V9 release-ready set: score-eligible AND provenance-eligible.

    Defined as ``compute_score_eligible(tasks) ∩ {t : is_provenance_eligible(t.meta)}``.

    ``compute_score_eligible`` (the historical denominator authority) is
    UNCHANGED -- this is a new conjunctive helper layered on top.
    Callers that still need the pre-V9 denominator keep their existing
    call.
    """
    materialised = list(tasks)
    score_eligible = compute_score_eligible(materialised)
    return {
        t.task_id
        for t in materialised
        if t.task_id in score_eligible
        and is_provenance_eligible(getattr(t, "meta", None))
    }


def projected_eligible(
    tasks: Iterable[_TaskLike],
    *,
    exclude_r11_pending: bool = True,
    exclude_r17_high: bool = True,
) -> set[str]:
    """V9 R13c gate input: the set of tasks projected to be release-ready
    after R10 / R5 finish.

    ``exclude_r11_pending=True`` (default): drop tasks whose
    ``tolerance_source == migration_drift_budget`` (R11 v2 path not
    landed; these tasks are not release-ready in v1 even after R10/R5
    pass).

    ``exclude_r17_high=True`` (default): drop tasks with
    ``provenance.golden.reconstructability_risk == high`` (math
    closed-form ops -- elementwise/norm/gemm/sampling; leaderboard-
    excluded so a determined adversary cannot replay the math
    perfectly).

    Returns ``set[str]`` of task ids. Pre-R10 / pre-schema-backfill
    tasks (no nested ``golden:`` block) are excluded -- they have no
    trust chain claim yet.
    """
    out: set[str] = set()
    for t in tasks:
        m = getattr(t, "meta", None) or {}
        prov = m.get("provenance")
        if not isinstance(prov, dict):
            continue
        golden = prov.get("golden")
        if not isinstance(golden, dict):
            continue

        tol_src = golden.get("tolerance_source")
        if exclude_r11_pending and tol_src == "migration_drift_budget":
            continue

        risk = golden.get("reconstructability_risk")
        if exclude_r17_high and risk == "high":
            continue

        oke = golden.get("oracle_kernel_equivalence")
        gkcs = golden.get("gpu_kernel_capture_status")
        # Projected: VERIFIED OR PENDING (R10 might still be running) /
        # DONE captures count too. STOP states (FAILED / SKIPPED /
        # NEEDS_MANUAL / LAUNCH_BLOCKED / BLOCKED_NON_DETERMINISTIC)
        # are excluded -- they will NOT become release-ready without
        # human triage.
        if oke in {"VERIFIED"} or gkcs in {"DONE", "PENDING"}:
            out.add(t.task_id)
    return out


def compute_release_blocking(
    tasks: Iterable[_TaskLike], eligible_ids: set[str]
) -> list[str]:
    """Return the sorted list of score-eligible ``task_id``s that block
    release (i.e. whose composite ``release_status != PASS``).

    This is the single shared blocking-set computation behind all three
    release-status authorities. ``tasks`` must be a materialised
    iterable (callers that hold a one-shot generator must ``list()`` it
    first); each task is expected to expose ``task_id`` and ``meta``.
    """
    blocking = [
        t.task_id
        for t in tasks
        if t.task_id in eligible_ids
        and not release_status_pass(getattr(t, "meta", None))
    ]
    blocking.sort()
    return blocking
