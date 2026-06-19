"""Triage state machine -- canonical state enum + transition validator.

Imported by ``fix_orchestrator`` (Unit 1.9 main loop) and by maintenance
CLIs (M2 ``--mark-unfixable`` / ``--accept-deferral``). Pure stdlib --
no GPFS / lock / file IO here; the loader/writer lives in
``fix_orchestrator``.

Canonical state enum (17 states; ANY alias is REJECTED by validators)
=====================================================================

  NEW                       initial
  QUEUED                    active
  PROMPTING                 active
  LLM_TRANSIENT_ERROR       active
  PATCH_VALIDATING          active
  BASELINE_SNAPSHOTTED      active
  PATCH_APPLIED             transient (recorded only in history)
  BENCH_RUNNING             active
  ATTEMPTING_NEXT_ROUND     active
  HUMAN_REVIEW_PENDING      active
  HUMAN_REVIEW_EXPIRED      active
  DEFERRAL_ACCEPTED         transient
  BACKLOG_PENDING_GOLDEN    active
  BACKLOG_UNEXPECTED        semi-terminal
  UNFIXABLE_BY_LLM          semi-terminal
  MANUAL_BACKLOG            terminal
  DONE                      terminal
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger("automation.state_machine")

# ---------------------------------------------------------------------------
# Canonical sets
# ---------------------------------------------------------------------------

CANONICAL_STATES: frozenset[str] = frozenset({
    "NEW", "QUEUED", "PROMPTING", "LLM_TRANSIENT_ERROR",
    "PATCH_VALIDATING", "BASELINE_SNAPSHOTTED", "PATCH_APPLIED",
    "BENCH_RUNNING", "ATTEMPTING_NEXT_ROUND",
    "HUMAN_REVIEW_PENDING", "HUMAN_REVIEW_EXPIRED", "DEFERRAL_ACCEPTED",
    "BACKLOG_PENDING_GOLDEN", "BACKLOG_UNEXPECTED",
    "UNFIXABLE_BY_LLM", "MANUAL_BACKLOG", "DONE",
})

TERMINAL_STATES: frozenset[str] = frozenset({"DONE", "MANUAL_BACKLOG"})
SEMI_TERMINAL_STATES: frozenset[str] = frozenset({
    "BACKLOG_UNEXPECTED", "UNFIXABLE_BY_LLM",
})

CANONICAL_VERDICT_CLASSES: frozenset[str] = frozenset({
    "shapes-incomplete", "oracle-mismatch", "upstream-not-invocable-here",
    "unexpected-blocked",
    "smoke-fail",  # manual-only: routed to BACKLOG_UNEXPECTED, no LLM repair
})

# Subset of canonical that does NOT get LLM repair attempts. These are
# parked in BACKLOG_UNEXPECTED for manual triage. Routing rule mirrors
# ``unexpected-blocked`` but the raw verdict_class is preserved so the
# manual reviewer can see whether the blocker was a smoke crash, an
# unknown subclass, or some other manual-only verdict.
MANUAL_ONLY_VERDICT_CLASSES: frozenset[str] = frozenset({
    "smoke-fail",
})

CANONICAL_HISTORY_KINDS: frozenset[str] = frozenset({
    "transition", "result_consumed", "capture_done_promoted",
    "daemon_timeout", "llm_call", "patch_validation_failed",
    "rollback", "steal_evicted",
})

# Required keys per history-record ``kind`` (the ``at`` ISO timestamp is
# always required; this table lists the extra kind-specific fields).
REQUIRED_HISTORY_FIELDS: dict[str, frozenset[str]] = {
    "transition": frozenset({"state", "to"}),
    "result_consumed": frozenset({"run_id", "cmd", "verdict"}),
    "capture_done_promoted": frozenset({
        "run_id", "capture_status_before", "capture_status_after",
    }),
    "daemon_timeout": frozenset({"run_id"}),
    "llm_call": frozenset({"round"}),
    "patch_validation_failed": frozenset({"round", "errors"}),
    "rollback": frozenset({"round", "reason"}),
    "steal_evicted": frozenset({"new_owner_uuid"}),
}

ALLOWED_TRANSITIONS: dict[str, frozenset[str]] = {
    # ``NEW -> BACKLOG_UNEXPECTED`` is the direct routing path used by the
    # orchestrator for manual-only verdict classes (smoke-fail) and the
    # resolver's ``unexpected-blocked`` fallback. Those classes have no
    # LLM remediation prompt, so the patch loop bypasses QUEUED entirely
    # and parks the task for manual triage on first sight.
    "NEW": frozenset({"QUEUED", "BACKLOG_UNEXPECTED"}),
    "QUEUED": frozenset({"PROMPTING"}),
    "PROMPTING": frozenset({"PATCH_VALIDATING", "LLM_TRANSIENT_ERROR"}),
    "LLM_TRANSIENT_ERROR": frozenset({"PROMPTING", "UNFIXABLE_BY_LLM"}),
    "PATCH_VALIDATING": frozenset({"BASELINE_SNAPSHOTTED", "PROMPTING"}),
    "BASELINE_SNAPSHOTTED": frozenset({
        "PATCH_APPLIED", "BENCH_RUNNING",
        "ATTEMPTING_NEXT_ROUND", "UNFIXABLE_BY_LLM",
    }),
    "PATCH_APPLIED": frozenset({"BENCH_RUNNING"}),
    "BENCH_RUNNING": frozenset({
        "DONE", "ATTEMPTING_NEXT_ROUND",
        "BACKLOG_PENDING_GOLDEN", "BACKLOG_UNEXPECTED",
        "HUMAN_REVIEW_PENDING", "NEW",  # class-switch re-enter
    }),
    "ATTEMPTING_NEXT_ROUND": frozenset({"PROMPTING", "UNFIXABLE_BY_LLM"}),
    "HUMAN_REVIEW_PENDING": frozenset({
        "HUMAN_REVIEW_EXPIRED", "DEFERRAL_ACCEPTED", "MANUAL_BACKLOG",
    }),
    "HUMAN_REVIEW_EXPIRED": frozenset({
        "DEFERRAL_ACCEPTED", "MANUAL_BACKLOG",
    }),
    "DEFERRAL_ACCEPTED": frozenset({"PATCH_VALIDATING"}),
    "BACKLOG_PENDING_GOLDEN": frozenset({"DONE", "BACKLOG_UNEXPECTED"}),
    "BACKLOG_UNEXPECTED": frozenset({"NEW"}),  # manual triage path
    "UNFIXABLE_BY_LLM": frozenset({"MANUAL_BACKLOG"}),
    # Terminal states have no out-transitions:
    "MANUAL_BACKLOG": frozenset(),
    "DONE": frozenset(),
}

# Required state_data fields per state (common fields excluded).
REQUIRED_STATE_DATA: dict[str, frozenset[str]] = {
    "NEW": frozenset(),
    "QUEUED": frozenset(),
    "PROMPTING": frozenset({"last_prompt_round_id"}),
    "LLM_TRANSIENT_ERROR": frozenset({"last_error_class", "retry_count"}),
    "PATCH_VALIDATING": frozenset({"last_validation_errors"}),
    "BASELINE_SNAPSHOTTED": frozenset({"baseline_tar_path", "manifest_path"}),
    "PATCH_APPLIED": frozenset(),
    "BENCH_RUNNING": frozenset({
        "pending_run_id", "pending_cmd", "pending_result_dir",
    }),
    "ATTEMPTING_NEXT_ROUND": frozenset(),
    "HUMAN_REVIEW_PENDING": frozenset({"needs_human_review_flag_path"}),
    "HUMAN_REVIEW_EXPIRED": frozenset({"expired_at", "retention_days"}),
    "DEFERRAL_ACCEPTED": frozenset({"accepted_at", "accepted_by"}),
    "BACKLOG_PENDING_GOLDEN": frozenset({"phase_b_capture_status_snapshot"}),
    "BACKLOG_UNEXPECTED": frozenset({"raw_verdict_class"}),
    "UNFIXABLE_BY_LLM": frozenset({"attempts_used"}),
    "MANUAL_BACKLOG": frozenset({"meta_release_scope_excluded_at"}),
    "DONE": frozenset({"final_verdict"}),
}

# Common (top-level) fields required on every task entry.
COMMON_FIELDS: frozenset[str] = frozenset({
    "state", "verdict_class", "attempt", "last_updated_at", "history",
    "last_consumed_verdict", "last_consumed_at", "last_consumed_run_id",
})


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class StateError(RuntimeError):
    """Base class for state-machine errors."""


class StateTransitionError(StateError):
    """Raised when a transition is rejected (bad source, bad target,
    missing required state_data)."""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def iso_utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def validate_state_name(state: str) -> None:
    if state not in CANONICAL_STATES:
        raise StateTransitionError(
            f"unknown state {state!r}; not in canonical 17-state enum"
        )


def validate_transition(
    *, src: str, dst: str, state_data: dict[str, Any],
) -> None:
    """Validate a transition. Raises ``StateTransitionError`` on
    violation; returns None on success."""
    validate_state_name(src)
    validate_state_name(dst)
    allowed = ALLOWED_TRANSITIONS.get(src, frozenset())
    if dst not in allowed:
        raise StateTransitionError(
            f"transition {src} -> {dst} not allowed; "
            f"allowed_from_{src}={sorted(allowed)}"
        )
    required = REQUIRED_STATE_DATA.get(dst, frozenset())
    missing = required - set(state_data.keys())
    if missing:
        raise StateTransitionError(
            f"transition {src} -> {dst} missing required state_data fields: "
            f"{sorted(missing)}"
        )
    # Validate field shapes for the strict cases.
    if dst == "MANUAL_BACKLOG":
        ts = state_data.get("meta_release_scope_excluded_at")
        if not isinstance(ts, str) or not ts.strip():
            raise StateTransitionError(
                "MANUAL_BACKLOG requires non-empty meta_release_scope_excluded_at "
                "(internal audit trail)"
            )
    if dst == "DONE":
        fv = state_data.get("final_verdict")
        if fv not in ("PASS", "SKIPPED_BY_DESIGN"):
            raise StateTransitionError(
                f"DONE requires final_verdict in {{PASS, SKIPPED_BY_DESIGN}}, "
                f"got {fv!r}"
            )


def validate_history_record(record: dict[str, Any]) -> None:
    """Tagged-union shape check.

    Every record must have a ``kind`` from ``CANONICAL_HISTORY_KINDS``
    plus an ``at`` ISO-8601 string; per-kind required fields are
    enumerated in ``REQUIRED_HISTORY_FIELDS``.
    """
    if not isinstance(record, dict):
        raise StateTransitionError("history record must be a dict")
    kind = record.get("kind")
    if kind not in CANONICAL_HISTORY_KINDS:
        raise StateTransitionError(
            f"history record has unknown kind {kind!r}; allowed: "
            f"{sorted(CANONICAL_HISTORY_KINDS)}"
        )
    if not isinstance(record.get("at"), str) or not record["at"].strip():
        raise StateTransitionError(
            f"history record kind={kind!r} missing non-empty 'at' timestamp"
        )
    required = REQUIRED_HISTORY_FIELDS.get(kind, frozenset())
    missing = required - set(record.keys())
    if missing:
        raise StateTransitionError(
            f"history record kind={kind!r} missing fields: {sorted(missing)}"
        )


def validate_task_entry(entry: dict[str, Any]) -> None:
    """Strict on-load and on-write validator.

    Enforces:
      * all common fields present (top-level shape);
      * ``state`` in the canonical 17-state enum;
      * ``verdict_class`` in the canonical class set;
      * ``history`` is a list whose entries pass
        ``validate_history_record`` (tagged-union shape);
      * ``state_data`` carries every key required by the current
        ``state`` per ``REQUIRED_STATE_DATA`` -- this catches the
        ``BENCH_RUNNING`` entry that is missing ``pending_cmd`` (a
        corrupt-state scenario the previous validator silently
        accepted).
    """
    missing = COMMON_FIELDS - set(entry.keys())
    if missing:
        raise StateTransitionError(
            f"task entry missing common fields: {sorted(missing)}"
        )
    validate_state_name(entry["state"])
    if entry["verdict_class"] not in CANONICAL_VERDICT_CLASSES:
        raise StateTransitionError(
            f"verdict_class {entry['verdict_class']!r} not in "
            f"{sorted(CANONICAL_VERDICT_CLASSES)}"
        )
    if not isinstance(entry["history"], list):
        raise StateTransitionError("history must be a list")
    for rec in entry["history"]:
        validate_history_record(rec)
    # Strict per-state state_data shape check: this is the on-load
    # safety net for state files written by older code / hand-edits /
    # crash recoveries that left state_data missing fields the FSM
    # expects.
    state_data = entry.get("state_data")
    if state_data is None:
        state_data = {}
    if not isinstance(state_data, dict):
        raise StateTransitionError(
            f"state_data must be a dict, got {type(state_data).__name__}"
        )
    required = REQUIRED_STATE_DATA.get(entry["state"], frozenset())
    missing_sd = required - set(state_data.keys())
    if missing_sd:
        raise StateTransitionError(
            f"task entry state={entry['state']!r} missing required "
            f"state_data fields: {sorted(missing_sd)}"
        )


# ---------------------------------------------------------------------------
# Public helpers used by orchestrator
# ---------------------------------------------------------------------------

def new_task_entry(verdict_class: str) -> dict[str, Any]:
    """Construct a fresh ``NEW`` task entry with all common fields."""
    if verdict_class not in CANONICAL_VERDICT_CLASSES:
        raise StateTransitionError(
            f"verdict_class must be in {sorted(CANONICAL_VERDICT_CLASSES)}"
        )
    return {
        "state": "NEW",
        "verdict_class": verdict_class,
        "attempt": 0,
        "last_updated_at": iso_utc_now(),
        "history": [],
        "last_consumed_verdict": None,
        "last_consumed_at": None,
        "last_consumed_run_id": None,
        "state_data": {},
    }


def transition(
    entry: dict[str, Any],
    *,
    to: str,
    state_data: dict[str, Any] | None = None,
    reason: str = "",
    extra_history: list[dict[str, Any]] | None = None,
    new_verdict_class: str | None = None,
) -> dict[str, Any]:
    """Apply a transition in-place on ``entry`` and append a 'transition'
    history record. ``state_data`` REPLACES the existing state_data (the
    canonical enum table is strict about which fields are valid in each
    state)."""
    src = entry["state"]
    new_state_data = dict(state_data or {})
    validate_transition(src=src, dst=to, state_data=new_state_data)

    if new_verdict_class is not None:
        if new_verdict_class not in CANONICAL_VERDICT_CLASSES:
            raise StateTransitionError(
                f"new_verdict_class invalid: {new_verdict_class!r}"
            )
        entry["verdict_class"] = new_verdict_class

    now = iso_utc_now()
    entry["state"] = to
    entry["state_data"] = new_state_data
    entry["last_updated_at"] = now
    entry["history"].append({
        "kind": "transition",
        "state": src,
        "to": to,
        "at": now,
        "reason": reason,
    })
    if extra_history:
        for rec in extra_history:
            validate_history_record(rec)
            entry["history"].append(rec)
    return entry


def append_history(
    entry: dict[str, Any], record: dict[str, Any],
) -> None:
    validate_history_record(record)
    entry["history"].append(record)


def consume_result(
    entry: dict[str, Any],
    *,
    run_id: str,
    cmd: str,
    verdict: str,
    verdict_class: str | None,
) -> None:
    """Update common audit fields + append a 'result_consumed' event."""
    now = iso_utc_now()
    entry["last_consumed_verdict"] = verdict
    entry["last_consumed_at"] = now
    entry["last_consumed_run_id"] = run_id
    append_history(entry, {
        "kind": "result_consumed",
        "at": now,
        "run_id": run_id,
        "cmd": cmd,
        "verdict": verdict,
        "verdict_class": verdict_class,
    })


__all__ = [
    "ALLOWED_TRANSITIONS",
    "CANONICAL_HISTORY_KINDS",
    "CANONICAL_STATES",
    "CANONICAL_VERDICT_CLASSES",
    "COMMON_FIELDS",
    "MANUAL_ONLY_VERDICT_CLASSES",
    "REQUIRED_STATE_DATA",
    "SEMI_TERMINAL_STATES",
    "TERMINAL_STATES",
    "StateError",
    "StateTransitionError",
    "append_history",
    "consume_result",
    "iso_utc_now",
    "new_task_entry",
    "transition",
    "validate_history_record",
    "validate_state_name",
    "validate_task_entry",
    "validate_transition",
]
