"""LLM-driven fix orchestrator (Units 1.9 + 2.8 + 4.2).

CLI entry-point. Drives the three canonical verdict classes
(``shapes-incomplete``, ``oracle-mismatch``, ``upstream-not-invocable-here``)
through the 17-state machine, plus the ``unexpected-blocked`` resolver
fallback that bypasses the LLM patch loop. Also sweeps
``BACKLOG_PENDING_GOLDEN`` tasks to ``DONE`` once Phase B capture
promotes them.

Two dispatch tiers are supported:

* **Tier 1** (default, ``--tier 1``): orchestrator runs on the CPU dev
  box; reverify is left as a user-runnable wrapper command emitted to
  stdout (``tier1_reverify.py``) and the operator copies it to the GPU
  host.
* **Tier 2** (``--tier 2 --queue-root <gpfs>/queue --reverify-results
  <gpfs>/results``): orchestrator enqueues a job onto the GPFS
  directory-rename queue and ``bench_evaluator_daemon`` (Unit 4.1) on
  the GPU host claims + executes + writes the result back to
  ``<gpfs>/results/<run_id>/``. The orchestrator's next sweep consumes
  the result via the same DONE barrier as Tier 1. Daemon-side stalls
  are tracked via ``history`` ``daemon_timeout`` events; after
  ``daemon_timeout_max_attempts`` (default 3) consecutive timeouts on
  the same ``pending_run_id`` the task transitions to
  ``ATTEMPTING_NEXT_ROUND`` and re-prompts.

Public helpers (also exported for callers / tests):
  * ``apply_release_scope_exclusion(task_id, reason, excluded_by, repo_root)``
  * ``load_existing_triage_state(state_path)``         (strict)
  * ``load_or_init_empty_state(state_path, *, allow_empty=False)``
  * ``sweep_backlog_pending_golden(ctx, state)``       (Unit 2.8)
  * ``submit_reverify_job(ctx, task_id, run_id, cmd)`` (Unit 4.2)
  * ``StateFileNotFoundError`` / ``StateFileCorruptError``
"""
from __future__ import annotations

# ---- KTD-11 sys.path bootstrap (CLI entry-point) -------------------------
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "scripts"))
# --------------------------------------------------------------------------

import argparse  # noqa: E402
import json  # noqa: E402
import logging  # noqa: E402
import os  # noqa: E402
import re  # noqa: E402
import uuid  # noqa: E402
from dataclasses import dataclass  # noqa: E402
from datetime import UTC, datetime  # noqa: E402
from typing import Any  # noqa: E402

import yaml  # noqa: E402

from automation import state_machine as sm  # noqa: E402
from automation.lib import atomic, file_blocks, lock  # noqa: E402
from automation.lib import queue as queue_lib  # noqa: E402

logger = logging.getLogger("automation.fix_orchestrator")

SCHEMA_VERSION = "1.0"

# ---------------------------------------------------------------------------
# Verdict / decision dispatch tables (Unit 2.8)
# ---------------------------------------------------------------------------
#
# Three canonical verdict classes feed the LLM patch loop. The fallback
# ``unexpected-blocked`` class is handled directly in the result-consumer
# (no LLM signal to act on) -- so it is intentionally absent from the
# dispatch table here.
#
# ``_VERDICT_TO_PROMPT`` and ``_VERDICT_TO_DECISIONS`` are populated at
# import time after the per-decision prompt modules are imported. They
# are MutableMapping by design so tests / future plugins can inject
# additional verdict classes without forking the orchestrator.

# Import per-decision prompt builders eagerly so the dispatch tables are
# fully populated on module load. The builders are lightweight (pure
# string templates over yaml.safe_load of task metadata) so we accept the
# small load-time cost in exchange for a single source of truth.
from automation.prompts import defer_to_golden as _p_defer_to_golden  # noqa: E402
from automation.prompts import oracle_mismatch as _p_oracle_mismatch  # noqa: E402
from automation.prompts import shapes_incomplete as _p_shapes_incomplete  # noqa: E402

_VERDICT_TO_PROMPT = {
    "shapes-incomplete": _p_shapes_incomplete.build_prompt,
    "oracle-mismatch": _p_oracle_mismatch.build_prompt,
    "upstream-not-invocable-here": _p_defer_to_golden.build_prompt,
}

# Decision sets accepted per verdict class. ``validate_patch`` uses these
# to gate the META ``decision`` field.
_VERDICT_TO_DECISIONS = {
    "shapes-incomplete": frozenset({"patch-shapes"}),
    "oracle-mismatch": frozenset({
        "tolerance-bump", "rewrite-primitives", "defer-to-upstream-oracle",
    }),
    "upstream-not-invocable-here": frozenset({"defer-to-golden"}),
}

# Per-decision required META keys -- the audit trail differs by decision
# because each remediation path needs different post-mortem context:
#   * ``patch-shapes``               -> missing_params, rationale, evidence
#   * ``tolerance-bump``             -> tolerance_delta, rationale, evidence
#   * ``rewrite-primitives``         -> primitives_changed, rationale, evidence
#   * ``defer-to-upstream-oracle``   -> upstream_callable, rationale, evidence
#   * ``defer-to-golden``            -> upstream_callable, blocker_reason,
#                                       rationale, evidence
_DECISION_TO_REQUIRED_KEYS = {
    "patch-shapes": frozenset({
        "decision", "missing_params", "rationale", "evidence",
    }),
    "tolerance-bump": frozenset({
        "decision", "rationale", "evidence", "tolerance_delta",
    }),
    "rewrite-primitives": frozenset({
        "decision", "rationale", "evidence", "primitives_changed",
    }),
    "defer-to-upstream-oracle": frozenset({
        "decision", "rationale", "evidence", "upstream_callable",
    }),
    "defer-to-golden": frozenset({
        "decision", "rationale", "evidence",
        "upstream_callable", "blocker_reason",
    }),
}

# The fallback union -- used by the first validate_patch pass (before the
# decision is known). Layer 4 will reject any decision not in the
# verdict's allowed set; once we know the decision, we re-run validate
# with the tighter per-decision key set so missing audit-trail fields
# burn a turn even though the union would have accepted them.
_VERDICT_TO_REQUIRED_KEYS_UNION = {
    vc: frozenset().union(*[
        _DECISION_TO_REQUIRED_KEYS[d] for d in decs
    ])
    for vc, decs in _VERDICT_TO_DECISIONS.items()
}

# Decision -> reverify cmd dispatch. ``patch-shapes`` exercises the
# resolver; the other paths exercise Phase A' (verify after meta edits
# / oracle rewrites / defer flags).
_DECISION_TO_REVERIFY_CMD = {
    "patch-shapes": "resolve",
    "tolerance-bump": "phase_a_verify",
    "rewrite-primitives": "phase_a_verify",
    "defer-to-upstream-oracle": "phase_a_verify",
    "defer-to-golden": "phase_a_verify",
}


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class StateFileNotFoundError(FileNotFoundError):
    """``triage_state.json`` does not exist and caller did not pass
    ``allow_empty=True``."""


class StateFileCorruptError(RuntimeError):
    """``triage_state.json`` exists but is not parseable / mis-shaped."""


# ---------------------------------------------------------------------------
# State loaders
# ---------------------------------------------------------------------------

def _validate_state_object(obj: Any) -> dict[str, Any]:
    if not isinstance(obj, dict):
        raise StateFileCorruptError("state file is not a JSON object")
    if obj.get("schema_version") != SCHEMA_VERSION:
        raise StateFileCorruptError(
            f"schema_version != {SCHEMA_VERSION!r}: "
            f"got {obj.get('schema_version')!r}"
        )
    tasks = obj.get("tasks")
    if not isinstance(tasks, dict):
        raise StateFileCorruptError("tasks is not a JSON object")
    for tid, entry in tasks.items():
        if not isinstance(entry, dict):
            raise StateFileCorruptError(
                f"tasks[{tid!r}] is not a JSON object"
            )
        try:
            sm.validate_task_entry(entry)
        except sm.StateTransitionError as exc:
            raise StateFileCorruptError(
                f"tasks[{tid!r}] invalid: {exc}"
            ) from exc
    return obj


def _quarantine_corrupt(state_path: Path) -> Path:
    ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    target = state_path.parent / f".corrupt.{ts}.bak"
    os.rename(state_path, target)
    return target


def load_existing_triage_state(state_path: os.PathLike[str] | str) -> dict[str, Any]:
    """Strict reader. Raises ``StateFileNotFoundError`` if missing,
    ``StateFileCorruptError`` (+ quarantine on disk) if mis-shaped."""
    state_path = Path(state_path)
    if not state_path.is_file():
        raise StateFileNotFoundError(f"state file not found: {state_path}")
    try:
        obj = json.loads(state_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        _quarantine_corrupt(state_path)
        raise StateFileCorruptError(
            f"state file json decode failed: {exc}; quarantined"
        ) from exc
    try:
        return _validate_state_object(obj)
    except StateFileCorruptError:
        _quarantine_corrupt(state_path)
        raise


def _validate_state_object_relaxed(obj: Any) -> dict[str, Any]:
    """Top-level invariants only -- enforces ``schema_version``, the
    ``tasks`` JSON object shape, COMMON_FIELDS presence, verdict_class
    membership, and history-record tagged-union shape. Deliberately
    does NOT enforce ``REQUIRED_STATE_DATA``: a transitional entry
    (e.g. ``state=BENCH_RUNNING`` mid-flush, with a partially written
    ``pending_run_id``) must not crash a defensive read-only consumer
    like ``backlog_report``."""
    if not isinstance(obj, dict):
        raise StateFileCorruptError("state file is not a JSON object")
    if obj.get("schema_version") != SCHEMA_VERSION:
        raise StateFileCorruptError(
            f"schema_version != {SCHEMA_VERSION!r}: "
            f"got {obj.get('schema_version')!r}"
        )
    tasks = obj.get("tasks")
    if not isinstance(tasks, dict):
        raise StateFileCorruptError("tasks is not a JSON object")
    for tid, entry in tasks.items():
        if not isinstance(entry, dict):
            raise StateFileCorruptError(
                f"tasks[{tid!r}] is not a JSON object"
            )
        missing = sm.COMMON_FIELDS - set(entry.keys())
        if missing:
            raise StateFileCorruptError(
                f"tasks[{tid!r}] missing common fields: {sorted(missing)}"
            )
        try:
            sm.validate_state_name(entry["state"])
        except sm.StateTransitionError as exc:
            raise StateFileCorruptError(
                f"tasks[{tid!r}] invalid state: {exc}"
            ) from exc
        if entry["verdict_class"] not in sm.CANONICAL_VERDICT_CLASSES:
            raise StateFileCorruptError(
                f"tasks[{tid!r}] verdict_class "
                f"{entry['verdict_class']!r} not in "
                f"{sorted(sm.CANONICAL_VERDICT_CLASSES)}"
            )
        history = entry.get("history")
        if not isinstance(history, list):
            raise StateFileCorruptError(
                f"tasks[{tid!r}] history must be a list"
            )
        for rec in history:
            try:
                sm.validate_history_record(rec)
            except sm.StateTransitionError as exc:
                raise StateFileCorruptError(
                    f"tasks[{tid!r}] history invalid: {exc}"
                ) from exc
        # state_data shape check (dict OR None) without per-state field
        # enforcement -- the whole point of the relaxed loader.
        sd = entry.get("state_data")
        if sd is not None and not isinstance(sd, dict):
            raise StateFileCorruptError(
                f"tasks[{tid!r}] state_data must be a dict or null"
            )
    return obj


def load_existing_triage_state_relaxed(
    state_path: os.PathLike[str] | str,
) -> dict[str, Any]:
    """Read-only consumer loader -- preserves top-level invariants but
    tolerates a missing-``state_data``-field entry (e.g. a transitional
    BENCH_RUNNING where ``pending_run_id`` was not yet flushed to disk).

    Used by ``backlog_report.py`` so a partial-write window does not
    crash an entire report run; the report renders ``n/a`` placeholders
    in the affected cells. Does NOT quarantine on top-level failures --
    the relaxed reader is a defensive observer, not a state authority.
    """
    state_path = Path(state_path)
    if not state_path.is_file():
        raise StateFileNotFoundError(f"state file not found: {state_path}")
    try:
        obj = json.loads(state_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise StateFileCorruptError(
            f"state file json decode failed: {exc}"
        ) from exc
    return _validate_state_object_relaxed(obj)


def load_or_init_empty_state(
    state_path: os.PathLike[str] | str,
    *,
    allow_empty: bool = False,
) -> dict[str, Any]:
    """Orchestrator-startup loader. ``allow_empty=False`` (default) is
    strict; ``allow_empty=True`` permits creating a fresh empty state."""
    state_path = Path(state_path)
    if not state_path.is_file():
        if not allow_empty:
            raise StateFileNotFoundError(
                f"state file not found at {state_path}; "
                "pass --init-empty-state to start fresh"
            )
        empty = {"schema_version": SCHEMA_VERSION, "tasks": {}}
        return empty
    return load_existing_triage_state(state_path)


def write_state(state: dict[str, Any], state_path: os.PathLike[str] | str) -> None:
    state_path = Path(state_path)
    _validate_state_object(state)
    atomic.atomic_write_json(state_path, state)


# ---------------------------------------------------------------------------
# pa_resolve.log parser (orchestrator side)
# ---------------------------------------------------------------------------

_RESOLVE_BLOCKED_RE = re.compile(
    r"^\[(?P<tid>[^\]]+)\]\s+BLOCKED\s+.*?::\s+BLOCKED\((?P<reason>[^)]+)\)",
)
# Same anchor as above, but capture the post-')' evidence blob too.
# Resolver emits:
#   [T] BLOCKED ... :: BLOCKED(no-verified-upstream-entrypoint: shapes-incomplete): tried <list>; <reject_notes>
# The evidence we want for the prompt is everything after the closing
# paren -- the reject_notes carry per-candidate reasons (often listing
# missing param names verbatim).
_RESOLVE_BLOCKED_FULL_RE = re.compile(
    r"^\[(?P<tid>[^\]]+)\]\s+BLOCKED\s+.*?::\s+"
    r"BLOCKED\((?P<reason>[^)]+)\)\s*:?\s*(?P<evidence>.*)$",
)
_RESOLVE_PASS_RE = re.compile(
    r"^\[(?P<tid>[^\]]+)\]\s+RESOLVED\s+\S+\s+::\s+PASS",
)


def _extract_resolver_evidence(
    log_path: Path, task_id: str, subclass_filter: str,
    max_lines: int = 8,
) -> str:
    """Generic resolver evidence extractor (subclass-scoped).

    Returns a small newline-joined excerpt suitable for embedding in
    the LLM prompt -- limited to the most recent ``max_lines`` matching
    lines for the task, so a prompt cannot blow up on a noisy log.
    """
    if not log_path.is_file():
        return ""
    hits: list[str] = []
    try:
        text = log_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""
    needle = f"[{task_id}]"
    for line in text.splitlines():
        if not line.startswith(needle):
            continue
        m = _RESOLVE_BLOCKED_FULL_RE.match(line)
        if m is None:
            continue
        raw = m.group("reason").strip()
        subclass = raw.split(":", 1)[1].strip() if ":" in raw else raw
        if subclass != subclass_filter:
            continue
        evidence = m.group("evidence").strip()
        if evidence:
            hits.append(f"{raw} :: {evidence}")
    # Keep the freshest ``max_lines`` (resolver appends; tail wins).
    return "\n".join(hits[-max_lines:])


def extract_missing_params_evidence(
    log_path: Path, task_id: str, max_lines: int = 8,
) -> str:
    """``shapes-incomplete`` resolver evidence (M1 helper retained for
    callers / tests)."""
    return _extract_resolver_evidence(
        log_path, task_id, "shapes-incomplete", max_lines=max_lines,
    )


def extract_oracle_mismatch_evidence(
    log_path: Path, task_id: str, max_lines: int = 8,
) -> list[str]:
    """``oracle-mismatch`` resolver evidence as a list of strings (one
    per resolver line). The list form matches the
    ``oracle_mismatch.build_prompt`` ``mismatch_evidence`` kwarg.
    """
    blob = _extract_resolver_evidence(
        log_path, task_id, "oracle-mismatch", max_lines=max_lines,
    )
    return [line for line in blob.splitlines() if line.strip()]


# ``upstream-not-invocable-here`` evidence frequently embeds a candidate
# upstream callable in the reject_notes (e.g. ``module='vllm.x' / fn=y``).
# We surface both the canonical (module, function) hint and the raw
# blocker reason so the defer-to-golden prompt can show them verbatim.

_UPSTREAM_MODULE_RE = re.compile(r"module\s*=\s*['\"]?([\w.]+)['\"]?")
_UPSTREAM_FUNC_RE = re.compile(r"(?:function|fn)\s*=\s*['\"]?(\w+)['\"]?")


def extract_upstream_callable_hint(
    log_path: Path, task_id: str,
) -> dict[str, str] | None:
    """Scan ``pa_resolve.log`` for ``module=...`` / ``function=...``
    hints from the resolver's reject notes. Returns ``None`` when no
    canonical pair is found -- the prompt builder degrades gracefully
    in that case.
    """
    if not log_path.is_file():
        return None
    try:
        text = log_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    needle = f"[{task_id}]"
    module: str | None = None
    function: str | None = None
    for line in text.splitlines():
        if not line.startswith(needle):
            continue
        m = _UPSTREAM_MODULE_RE.search(line)
        if m:
            module = m.group(1)
        f = _UPSTREAM_FUNC_RE.search(line)
        if f:
            function = f.group(1)
    if module and function:
        return {"module": module, "function": function}
    return None


def extract_blocker_reason(
    log_path: Path, task_id: str, max_lines: int = 4,
) -> str:
    """``upstream-not-invocable-here`` blocker reason -- joins the last
    few resolver evidence lines into a single short string suitable for
    the ``phase_a_prime.reason`` candidate.
    """
    blob = _extract_resolver_evidence(
        log_path, task_id, "upstream-not-invocable-here", max_lines=max_lines,
    )
    if not blob:
        return ""
    # collapse to the trailing reject_notes which usually pin the
    # binding-not-importable / runtime-required nature of the blocker.
    lines = [line for line in blob.splitlines() if line.strip()]
    return " | ".join(lines[-max_lines:])


def extract_oracle_log_excerpt(
    repo_root: Path, task_id: str, max_chars: int = 800,
) -> str:
    """Pull a tail excerpt of the per-task pa_oracle.log, if present.

    Resolver-side oracle logs live under ``results/.../pa_oracle.log`` --
    we accept ``repo_root/results/gpu_pipeline/pa_oracle.log`` as the
    canonical location (matches ``OrchestratorContext.resolve_log``'s
    sibling layout). Missing log returns ``""``; truncated excerpts get
    an explicit marker so the LLM does not see a mystery tail.
    """
    candidates = [
        repo_root / "results" / "gpu_pipeline" / "pa_oracle.log",
        repo_root / "results" / "pa_oracle.log",
    ]
    for path in candidates:
        if not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        # filter to lines that mention this task_id
        relevant = [line for line in text.splitlines() if task_id in line]
        if not relevant:
            return ""
        joined = "\n".join(relevant)
        if len(joined) <= max_chars:
            return joined
        return (f"(truncated; showing last {max_chars} chars)\n"
                f"{joined[-max_chars:]}")
    return ""


def parse_resolve_log(log_path: Path) -> list[tuple[str, str]]:
    """Return list of (task_id, verdict_class) for all BLOCKED lines.

    ``shapes-incomplete`` is normalised; everything unrecognised becomes
    ``unexpected-blocked``.
    """
    out: list[tuple[str, str]] = []
    if not log_path.is_file():
        return out
    text = log_path.read_text(encoding="utf-8", errors="replace")
    for line in text.splitlines():
        m = _RESOLVE_BLOCKED_RE.match(line)
        if not m:
            continue
        tid = m.group("tid")
        raw = m.group("reason").strip()
        subclass = raw.split(":", 1)[1].strip() if ":" in raw else raw
        if subclass in {"shapes-incomplete", "oracle-mismatch",
                        "upstream-not-invocable-here"}:
            vclass = subclass
        else:
            vclass = "unexpected-blocked"
        out.append((tid, vclass))
    return out


# ---------------------------------------------------------------------------
# Patch persistence helpers
# ---------------------------------------------------------------------------

def _patches_round_dir(repo_root: Path, task_id: str, attempt: int) -> Path:
    return repo_root / "patches" / task_id / f"round-{attempt:02d}"


def persist_prompt_and_response(
    repo_root: Path, task_id: str, attempt: int,
    *,
    prompt_messages: list[dict[str, Any]],
    response_text: str,
    meta_payload: dict[str, Any] | None = None,
    file_block_records: list[dict[str, Any]] | None = None,
) -> Path:
    """Write prompt / response / meta_block / per-file blocks under
    ``patches/<task>/round-<N>/``."""
    rd = _patches_round_dir(repo_root, task_id, attempt)
    rd.mkdir(parents=True, exist_ok=True)
    (rd / "prompt.txt").write_text(
        json.dumps(prompt_messages, indent=2), encoding="utf-8",
    )
    atomic.atomic_write_json(
        rd / "llm_response.json", {"raw": response_text},
    )
    if meta_payload is not None:
        atomic.atomic_write_json(rd / "meta_block.json", meta_payload)
    if file_block_records is not None:
        fbd = rd / "file_blocks"
        fbd.mkdir(exist_ok=True)
        for i, rec in enumerate(file_block_records):
            (fbd / f"{i:03d}_{rec['name']}").write_text(
                rec["content"], encoding="utf-8",
            )
    return rd


# ---------------------------------------------------------------------------
# apply_release_scope_exclusion (M2 helper, lands in M1 per plan)
# ---------------------------------------------------------------------------

def apply_release_scope_exclusion(
    task_id: str,
    reason: str,
    excluded_by: str,
    repo_root: os.PathLike[str] | str,
    *,
    state_root: os.PathLike[str] | str | None = None,
) -> None:
    """Mark a task as release-excluded by writing
    ``meta.yaml.release_scope`` + snapshotting baseline.

    Does NOT touch ``triage_state.json``; caller handles state
    transition separately (per plan Unit 1.9 split-of-responsibility).
    """
    repo_root = Path(repo_root)
    task_dir = _find_task_dir(repo_root, task_id)
    if task_dir is None:
        raise FileNotFoundError(
            f"cannot find tasks/<level>/{task_id}/ under {repo_root}"
        )

    state_root_path = Path(state_root or (repo_root / "state"))
    state_root_path.mkdir(parents=True, exist_ok=True)
    per_task_lock = state_root_path / f".lock.{task_id}"

    # snapshot path is intentionally distinct from round-<N>/
    snap_dir = repo_root / "patches" / task_id / "release_scope_baseline"
    snap_dir.mkdir(parents=True, exist_ok=True)
    snap_tar = snap_dir / "baseline.tar.gz"
    snap_manifest = snap_dir / "manifest.json"

    with lock.acquire_lock(per_task_lock, start_heartbeat=False):
        atomic.snapshot_task_baseline(task_dir, snap_tar, snap_manifest)
        meta_path = task_dir / "meta.yaml"
        try:
            meta = yaml.safe_load(meta_path.read_text(encoding="utf-8")) or {}
            if not isinstance(meta, dict):
                raise ValueError("meta.yaml is not a YAML mapping")
            meta["release_scope"] = {
                "excluded": True,
                "reason": reason,
                "excluded_at": sm.iso_utc_now(),
                "excluded_by": excluded_by,
            }
            atomic.atomic_write_yaml(meta_path, meta)
        except Exception:
            try:
                atomic.rollback_task(task_dir, snap_tar, snap_manifest)
            except Exception:  # pragma: no cover
                logger.exception("rollback_task failed inside helper")
            raise


def _find_task_dir(repo_root: Path, task_id: str) -> Path | None:
    tasks_root = repo_root / "tasks"
    if not tasks_root.is_dir():
        return None
    for level_dir in tasks_root.iterdir():
        if not level_dir.is_dir():
            continue
        cand = level_dir / task_id
        if cand.is_dir():
            return cand
    return None


def _read_phase_b_capture(
    repo_root: Path, task_id: str,
) -> dict[str, Any] | None:
    """Read ``meta.yaml.phase_b_capture`` for ``task_id`` -- returns the
    mapping (``status`` / ``captured_at`` / ``upstream_callable``) or
    ``None`` if the task / meta / block is missing or mis-shaped.
    """
    task_dir = _find_task_dir(repo_root, task_id)
    if task_dir is None:
        return None
    meta_path = task_dir / "meta.yaml"
    if not meta_path.is_file():
        return None
    try:
        meta = yaml.safe_load(meta_path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError:
        return None
    if not isinstance(meta, dict):
        return None
    block = meta.get("phase_b_capture")
    if not isinstance(block, dict):
        return None
    return block


def _read_phase_b_capture_status(
    repo_root: Path, task_id: str,
) -> str:
    """Return ``phase_b_capture.status`` as a canonical enum string;
    defaults to ``PENDING`` if absent / not yet captured."""
    block = _read_phase_b_capture(repo_root, task_id)
    if block is None:
        return "PENDING"
    status = block.get("status")
    if status in ("PENDING", "DONE", "FAILED"):
        return str(status)
    return "PENDING"


def sweep_backlog_pending_golden(
    ctx: OrchestratorContext, state: dict[str, Any],
) -> dict[str, Any]:
    """Promote ``BACKLOG_PENDING_GOLDEN`` entries to ``DONE`` once Phase B
    capture has succeeded.

    Reads ``meta.yaml.phase_b_capture.status`` for each parked task:
      * ``DONE``    -> transition to DONE + append ``capture_done_promoted``
                       history event; ``final_verdict=SKIPPED_BY_DESIGN``.
      * ``PENDING`` -> leave parked (sweeper is idempotent).
      * ``FAILED``  -> leave parked (operator will re-run capture).

    Returns the (mutated in-place) state dict.
    """
    for task_id, entry in state["tasks"].items():
        if entry["state"] != "BACKLOG_PENDING_GOLDEN":
            continue
        # Snapshot the capture status *before* re-reading meta.yaml: this
        # is the parked-state truth (typically ``PENDING``) and is the
        # value persisted into the ``capture_done_promoted`` history
        # event as ``capture_status_before`` -- enabling forensic
        # reconstruction of the promotion delta later.
        capture_status_before = (
            (entry.get("state_data") or {}).get(
                "phase_b_capture_status_snapshot"
            )
            or "PENDING"
        )
        block = _read_phase_b_capture(ctx.repo_root, task_id)
        if block is None:
            logger.info(
                "task %s: no phase_b_capture block on disk; stays parked",
                task_id,
            )
            continue
        status = block.get("status")
        if status == "DONE":
            captured_at = block.get("captured_at") or sm.iso_utc_now()
            run_id = block.get("run_id") or f"phase-b-{task_id}"
            sm.transition(
                entry, to="DONE",
                state_data={"final_verdict": "SKIPPED_BY_DESIGN"},
                reason="phase B capture DONE -> promoted",
            )
            sm.append_history(entry, {
                "kind": "capture_done_promoted",
                "at": sm.iso_utc_now(),
                "run_id": str(run_id),
                "captured_at": str(captured_at),
                "capture_status_before": str(capture_status_before),
                "capture_status_after": "DONE",
            })
            logger.info(
                "task %s: promoted to DONE (phase B capture captured_at=%s)",
                task_id, captured_at,
            )
        elif status == "FAILED":
            logger.info(
                "task %s: phase_b_capture.status=FAILED; stays parked",
                task_id,
            )
        else:
            # PENDING or unknown -> stay parked.
            logger.debug(
                "task %s: phase_b_capture.status=%r; stays parked",
                task_id, status,
            )
    return state


# ---------------------------------------------------------------------------
# OrchestratorContext: a tiny holder for the test surface
# ---------------------------------------------------------------------------

@dataclass
class OrchestratorContext:
    repo_root: Path
    state_root: Path
    reverify_results: Path
    resolve_log: Path
    max_tasks: int = 10
    max_attempts: int = 3
    # ``dry_run`` and ``apply`` are linked: the orchestrator only walks
    # the ``snapshot -> apply_patch -> BENCH_RUNNING`` path when
    # ``apply`` is True. The CLI makes ``--dry-run`` and ``--apply``
    # mutually exclusive. The dataclass defaults preserve the historic
    # in-process test behaviour (``apply=True`` ⇒ real apply path);
    # the CLI overrides ``apply`` to False unless ``--apply`` is given.
    dry_run: bool = False
    apply: bool = True
    # Filter is ``None`` to drive all three canonical classes; tests
    # pin a single class for behaviour-specific scenarios. ``"all"`` is
    # the sentinel for "no filter".
    verdict_class_filter: str | None = None
    # ---- Tier 2 dispatch (Unit 4.2) ------------------------------------
    # ``tier`` selects the reverify dispatch path:
    #   * 1 -> local Tier 1 (operator runs ``tier1_reverify.py`` on the
    #          GPU host; orchestrator emits the wrapper command).
    #   * 2 -> remote Tier 2 (orchestrator enqueues a job onto the GPFS
    #          queue; ``bench_evaluator_daemon`` on the GPU host claims
    #          + runs + writes results back).
    # Tier 2 requires ``queue_root`` to be set. ``reverify_results`` is
    # repurposed as the daemon's GPFS results root when tier==2.
    tier: int = 1
    queue_root: Path | None = None
    # Daemon timeout (minutes). When tier==2 and no result.json+DONE has
    # appeared within this budget for the current ``pending_run_id``, the
    # orchestrator appends a ``daemon_timeout`` history event but keeps
    # the state at BENCH_RUNNING (no new canonical state needed). After
    # ``daemon_timeout_max_attempts`` consecutive timeout events on the
    # same ``pending_run_id`` the task transitions to
    # ATTEMPTING_NEXT_ROUND and the LLM is re-prompted on the next sweep.
    daemon_timeout_minutes: int = 60
    daemon_timeout_max_attempts: int = 3


def _mint_run_id(tier: int = 1) -> str:
    """Mint a fresh ``run_id`` for the given tier.

    The ``tier{1,2}`` prefix is informational only -- queue + result
    layouts are identical across tiers (the daemon also writes the
    canonical 9-field ``result.json`` + DONE barrier).
    """
    ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"tier{tier}-{ts}-{uuid.uuid4().hex[:8]}"


# ---------------------------------------------------------------------------
# Tier 2 reverify-job dispatcher (Unit 4.2)
# ---------------------------------------------------------------------------

def submit_reverify_job(
    ctx: OrchestratorContext,
    task_id: str,
    run_id: str,
    cmd: str,
) -> Path:
    """Dispatch a reverify job for ``run_id`` under the configured tier.

    Returns the **result directory** the orchestrator will poll for the
    DONE barrier. The directory is created if it does not already
    exist; the caller persists the path on the task entry's
    ``state_data.pending_result_dir`` field so the consume sweep can
    find it.

    * Tier 1 -- no queue dispatch; the operator manually runs the
      printed ``tier1_reverify.py`` wrapper.  The result directory is
      ``ctx.reverify_results / run_id`` (same as M1).
    * Tier 2 -- writes ``<queue_root>/<run_id>/{task.json, READY}``
      atomically via ``automation.lib.queue.enqueue``. The result
      directory is ``ctx.reverify_results / run_id`` (the daemon's
      ``results_root_rel`` is configured to match by the operator).
    """
    if cmd not in ("resolve", "phase_a_verify"):
        raise ValueError(
            f"submit_reverify_job: unsupported cmd={cmd!r}; "
            "must be one of ('resolve', 'phase_a_verify')"
        )

    result_dir = ctx.reverify_results / run_id
    result_dir.mkdir(parents=True, exist_ok=True)

    if ctx.tier == 1:
        # No queue dispatch -- the wrapper command is emitted by the
        # caller alongside the BENCH_RUNNING transition.
        return result_dir

    if ctx.tier == 2:
        if ctx.queue_root is None:
            raise ValueError(
                "Tier 2 dispatch requires ctx.queue_root; "
                "got None (CLI must pass --queue-root)"
            )
        ctx.queue_root.mkdir(parents=True, exist_ok=True)
        task_json = {
            "task_id": task_id,
            "run_id": run_id,
            "cmd": cmd,
            "requested_at": sm.iso_utc_now(),
        }
        queue_lib.enqueue(ctx.queue_root, run_id, task_json)
        logger.info(
            "tier 2 enqueued task=%s run_id=%s cmd=%s queue=%s",
            task_id, run_id, cmd, ctx.queue_root,
        )
        return result_dir

    raise ValueError(f"unknown tier {ctx.tier!r}; expected 1 or 2")


# ---------------------------------------------------------------------------
# Daemon-timeout helpers (Tier 2 -- Unit 4.2)
# ---------------------------------------------------------------------------

def _count_daemon_timeouts_for_run(
    entry: dict[str, Any], run_id: str,
) -> int:
    """Return the number of ``daemon_timeout`` history events on
    ``entry`` whose ``run_id`` field matches the current pending run.

    Only events tagged with the current ``pending_run_id`` count toward
    the retry budget -- stale events from a previous (now retired) run
    must not pre-emptively exhaust the new run's budget.
    """
    count = 0
    for rec in entry.get("history", []):
        if rec.get("kind") == "daemon_timeout" and rec.get("run_id") == run_id:
            count += 1
    return count


def _iso_to_epoch(ts: str) -> float | None:
    """Parse the orchestrator's canonical UTC ISO format
    (``YYYY-MM-DDTHH:MM:SSZ``) into a unix epoch float; returns None on
    parse failure so callers can fall back to ``now``.
    """
    if not isinstance(ts, str) or not ts:
        return None
    try:
        # ``Z`` suffix is not understood by fromisoformat<3.11 -- swap it.
        return datetime.fromisoformat(ts.replace("Z", "+00:00")).timestamp()
    except (ValueError, TypeError):
        return None


# ---------------------------------------------------------------------------
# Per-task driver (test-friendly entry; one task per call)
# ---------------------------------------------------------------------------

def drive_task_once(
    ctx: OrchestratorContext,
    state: dict[str, Any],
    task_id: str,
    *,
    llm_chat,
    prompt_builder,
) -> dict[str, Any]:
    """Advance a single task by one "step" (the orchestrator main loop
    calls this once per task per run).

    Returns updated state (mutates in place too).

    The caller is responsible for global + per-task locks; this helper
    operates on the in-memory state.
    """
    entry = state["tasks"].setdefault(task_id, sm.new_task_entry("shapes-incomplete"))

    # Filter: ``None`` drives every canonical class; a pinned class
    # filters everything else out (used by tests + the M1 CLI subset).
    if (ctx.verdict_class_filter is not None
            and entry["verdict_class"] != ctx.verdict_class_filter):
        logger.info(
            "task %s class=%s skipped (filter=%s)",
            task_id, entry["verdict_class"], ctx.verdict_class_filter,
        )
        return state

    # ``unexpected-blocked`` is the resolver's fallback bucket. It must
    # never enter the LLM patch loop -- there is no remediation signal
    # to act on. The canonical FSM cannot reach BACKLOG_UNEXPECTED from
    # NEW directly (only via BENCH_RUNNING per ALLOWED_TRANSITIONS), so
    # we leave NEW-state ``unexpected-blocked`` entries untouched here.
    # They reach BACKLOG_UNEXPECTED via the consume path (handled by
    # ``_try_consume_pending`` -- the same routing M1 fix #5 established
    # for class-switch results).
    if entry["verdict_class"] == "unexpected-blocked":
        logger.info(
            "task %s class=unexpected-blocked: LLM patch loop skipped; "
            "expected to reach BACKLOG_UNEXPECTED via consume path",
            task_id,
        )
        return state

    cur = entry["state"]

    # 1) If pending: try to consume result
    if cur == "BENCH_RUNNING":
        return _try_consume_pending(ctx, state, task_id, entry)

    # 2) Terminal / semi-terminal -- no-op
    if cur in sm.TERMINAL_STATES or cur in sm.SEMI_TERMINAL_STATES:
        return state

    # 3) NEW -> QUEUED
    if cur == "NEW":
        sm.transition(entry, to="QUEUED", state_data={}, reason="initial enqueue")
        cur = entry["state"]

    # 4) ATTEMPTING_NEXT_ROUND -> PROMPTING (driver advances)
    if cur == "ATTEMPTING_NEXT_ROUND":
        sm.transition(entry, to="PROMPTING", state_data={
            "last_prompt_round_id": f"round-{entry['attempt']:02d}",
        }, reason="retry after previous round")
        cur = entry["state"]

    if cur == "QUEUED":
        sm.transition(entry, to="PROMPTING", state_data={
            "last_prompt_round_id": f"round-{entry['attempt']:02d}",
        }, reason="dispatched")
        cur = entry["state"]

    # 5) PROMPTING -> LLM -> validate / apply / pending
    if cur == "PROMPTING":
        return _do_prompt_validate_apply(
            ctx, state, task_id, entry,
            llm_chat=llm_chat, prompt_builder=prompt_builder,
        )
    return state


def _maybe_handle_tier2_daemon_timeout(
    ctx: OrchestratorContext, entry: dict[str, Any], task_id: str,
) -> bool:
    """Detect & record a Tier 2 daemon timeout for the current pending
    run. Returns ``True`` if the function transitioned the entry (so
    the caller should short-circuit); ``False`` means the function
    decided not to act (still within the per-poll grace window).

    The grace window is anchored to the *last* tracked timestamp for
    this ``pending_run_id`` -- either the most recent
    ``daemon_timeout`` history event for the same run_id, or the
    transition into BENCH_RUNNING. This way each timeout fires at
    most once per ``ctx.daemon_timeout_minutes`` interval, regardless
    of how often the orchestrator polls.
    """
    sd = entry["state_data"]
    pending_run = sd.get("pending_run_id")
    if not isinstance(pending_run, str) or not pending_run:
        return False

    timeout_window_s = max(1, ctx.daemon_timeout_minutes * 60)

    # Find the most recent anchor: either the last daemon_timeout for
    # this run, or the most recent transition INTO BENCH_RUNNING.
    anchor_ts: float | None = None
    for rec in reversed(entry.get("history", [])):
        if (rec.get("kind") == "daemon_timeout"
                and rec.get("run_id") == pending_run):
            anchor_ts = _iso_to_epoch(rec.get("at", ""))
            break
        if (rec.get("kind") == "transition"
                and rec.get("to") == "BENCH_RUNNING"):
            anchor_ts = _iso_to_epoch(rec.get("at", ""))
            break
    if anchor_ts is None:
        anchor_ts = _iso_to_epoch(entry.get("last_updated_at", "")) or 0.0

    now_s = datetime.now(UTC).timestamp()
    if now_s - anchor_ts < timeout_window_s:
        # Still within grace window; do not record yet.
        return False

    # Record the timeout event (tagged union; required fields: kind, at,
    # run_id). ``timeout_minutes`` is informational metadata.
    sm.append_history(entry, {
        "kind": "daemon_timeout",
        "at": sm.iso_utc_now(),
        "run_id": pending_run,
        "timeout_minutes": ctx.daemon_timeout_minutes,
    })
    timeouts = _count_daemon_timeouts_for_run(entry, pending_run)
    logger.warning(
        "task %s tier2 daemon_timeout #%d for run_id=%s (max=%d)",
        task_id, timeouts, pending_run, ctx.daemon_timeout_max_attempts,
    )
    if timeouts >= ctx.daemon_timeout_max_attempts:
        # Default: re-enter the LLM patch loop. Per the design doc the
        # rationale is that a stalled daemon is more often a symptom of
        # a bad patch than a daemon-side bug; if the operator suspects
        # the latter they can switch the task to BACKLOG_UNEXPECTED
        # manually. We bump ``attempt`` so the LLM gets a fresh round.
        entry["attempt"] = entry["attempt"] + 1
        if entry["attempt"] >= ctx.max_attempts:
            sm.transition(
                entry, to="ATTEMPTING_NEXT_ROUND", state_data={},
                reason=(
                    f"tier2 daemon timeout x{timeouts} run_id={pending_run}; "
                    "attempt budget exhausted"
                ),
            )
            sm.transition(
                entry, to="UNFIXABLE_BY_LLM",
                state_data={"attempts_used": entry["attempt"]},
                reason="tier2 daemon timeout: retry budget exhausted",
            )
            return True
        sm.transition(
            entry, to="ATTEMPTING_NEXT_ROUND", state_data={},
            reason=(
                f"tier2 daemon timeout x{timeouts} run_id={pending_run}; "
                "re-prompting LLM"
            ),
        )
        return True
    return False


def _try_consume_pending(
    ctx: OrchestratorContext, state: dict[str, Any],
    task_id: str, entry: dict[str, Any],
) -> dict[str, Any]:
    sd = entry["state_data"]
    pending_run = sd.get("pending_run_id")
    pending_cmd = sd.get("pending_cmd")
    pending_dir = Path(sd.get("pending_result_dir", ""))
    # The DONE marker is the writer-side read-barrier. Per
    # extract_verdict.write_verdict, the order is:
    #   _provenance.json -> result.json -> touch DONE
    # so we must NOT read result.json (or treat it as authoritative)
    # until DONE exists -- otherwise we race a partially-written
    # result.json on slow / NFS-cached filesystems.
    done_marker = pending_dir / "DONE"
    if not done_marker.is_file():
        # Tier 2 daemon-timeout detection: if the bench job has been
        # pending for longer than ``ctx.daemon_timeout_minutes`` without
        # a DONE marker, the daemon is either hung or crashed. We
        # record the stall as a ``daemon_timeout`` history event but
        # KEEP the state at BENCH_RUNNING (no new canonical state).
        # After ``daemon_timeout_max_attempts`` consecutive timeouts on
        # the same ``pending_run_id`` the task transitions to
        # ATTEMPTING_NEXT_ROUND so the LLM patch loop can re-prompt.
        if ctx.tier == 2:
            timeout_outcome = _maybe_handle_tier2_daemon_timeout(
                ctx, entry, task_id,
            )
            if timeout_outcome:
                return state
        logger.info(
            "task %s pending=%s DONE barrier not yet present",
            task_id, pending_run,
        )
        return state
    result_json = pending_dir / "result.json"
    provenance_json = pending_dir / "_provenance.json"
    if not result_json.is_file() or not provenance_json.is_file():
        # DONE present but a sidecar is missing -> writer crashed
        # between barrier creation and rename; do not consume. The
        # next sweep can re-evaluate.
        logger.warning(
            "task %s pending=%s DONE present but %s missing",
            task_id, pending_run,
            "result.json" if not result_json.is_file() else "_provenance.json",
        )
        return state
    result = json.loads(result_json.read_text(encoding="utf-8"))
    if result.get("run_id") != pending_run:
        logger.warning("task %s stale result: run_id mismatch (%s != %s)",
                       task_id, result.get("run_id"), pending_run)
        return state
    if result.get("cmd") != pending_cmd:
        logger.warning("task %s stale result: cmd mismatch (%s != %s)",
                       task_id, result.get("cmd"), pending_cmd)
        return state
    # Producer / tier consistency: each tier has exactly one legitimate
    # writer of result.json (Tier 1 -> ``tier1_reverify``, Tier 2 ->
    # ``daemon``). A producer mismatch means the result.json belongs to
    # a different execution path (e.g. a stale daemon write that landed
    # in a Tier 1 run-dir, or a tier1_reverify rerun overwriting a
    # daemon's claim). Quarantine the result directory by renaming it
    # with a ``.foreign`` suffix and skip consumption -- the task stays
    # at BENCH_RUNNING for safe re-eval / operator triage.
    expected_producer = "daemon" if ctx.tier == 2 else "tier1_reverify"
    actual_producer = result.get("producer")
    if actual_producer != expected_producer:
        logger.warning(
            "task %s producer mismatch: expected=%s actual=%s; quarantining %s",
            task_id, expected_producer, actual_producer, pending_dir,
        )
        try:
            foreign_dir = pending_dir.with_name(pending_dir.name + ".foreign")
            if not foreign_dir.exists():
                pending_dir.rename(foreign_dir)
        except OSError as exc:
            logger.warning(
                "task %s failed to quarantine %s: %s",
                task_id, pending_dir, exc,
            )
        return state

    verdict = result.get("verdict")
    vclass = result.get("verdict_class")
    sm.consume_result(
        entry, run_id=result["run_id"], cmd=result["cmd"],
        verdict=verdict, verdict_class=vclass,
    )
    if verdict == "PASS":
        sm.transition(entry, to="DONE",
                      state_data={"final_verdict": "PASS"},
                      reason="reverify PASS")
        return state
    if verdict == "SKIPPED_BY_DESIGN":
        # defer-to-golden / defer-to-upstream-oracle: the LLM has
        # patched meta.yaml to declare the upstream callable and
        # phase B capture status. We read the snapshot off disk so the
        # sweeper can later detect ``DONE`` and promote.
        snapshot = _read_phase_b_capture_status(ctx.repo_root, task_id)
        sm.transition(entry, to="BACKLOG_PENDING_GOLDEN",
                      state_data={
                          "phase_b_capture_status_snapshot": snapshot,
                      },
                      reason="phase A' SKIPPED_BY_DESIGN; awaiting golden capture")
        return state
    # BLOCKED
    # ``unexpected-blocked`` is the resolver's fallback bucket; it must
    # NOT feed back into the LLM patch loop (no signal to act on). Park
    # it in BACKLOG_UNEXPECTED for manual triage *before* the canonical
    # class-switch path -- otherwise the ``vclass in CANONICAL_VERDICT_CLASSES``
    # branch would silently re-enter NEW with the same useless class.
    if vclass == "unexpected-blocked":
        sm.transition(entry, to="BACKLOG_UNEXPECTED",
                      state_data={"raw_verdict_class": "unexpected-blocked"},
                      reason="resolver fallback class; manual triage")
        return state
    if vclass and vclass != entry["verdict_class"]:
        # class-switch: re-enter NEW with new (real) class.
        if vclass in sm.CANONICAL_VERDICT_CLASSES:
            sm.transition(
                entry, to="NEW", state_data={},
                reason=f"class-switch to {vclass}",
                new_verdict_class=vclass,
            )
            return state
        # unknown class -> backlog
        sm.transition(entry, to="BACKLOG_UNEXPECTED",
                      state_data={"raw_verdict_class": vclass or ""},
                      reason="unexpected verdict_class")
        return state
    # BLOCKED same class -> attempt next round
    entry["attempt"] = entry["attempt"] + 1
    if entry["attempt"] >= ctx.max_attempts:
        sm.transition(
            entry, to="ATTEMPTING_NEXT_ROUND", state_data={},
            reason="BLOCKED same class; attempt budget exhausted",
        )
        sm.transition(
            entry, to="UNFIXABLE_BY_LLM",
            state_data={"attempts_used": entry["attempt"]},
            reason="retry budget exhausted",
        )
        return state
    sm.transition(
        entry, to="ATTEMPTING_NEXT_ROUND", state_data={},
        reason="BLOCKED same class; retry",
    )
    return state


def _dispatch_prompt_builder(
    ctx: OrchestratorContext, task_id: str, verdict_class: str,
    *,
    override_builder=None,
) -> list[dict[str, str]]:
    """Build the per-class prompt.

    The caller may pass an ``override_builder`` (M1 compat / tests) --
    in that case we honour the explicit choice and just route the
    extracted resolver evidence into the builder's preferred kwarg.
    """
    if override_builder is not None:
        # Preserve the historic M1 signature ``(task_id, repo_root,
        # missing_params_evidence=...)`` -- callers that pass a 2-arg
        # builder also work via the TypeError fallback below.
        evidence = extract_missing_params_evidence(ctx.resolve_log, task_id)
        try:
            return override_builder(
                task_id, ctx.repo_root,
                missing_params_evidence=evidence,
            )
        except TypeError:
            return override_builder(task_id, ctx.repo_root)

    builder = _VERDICT_TO_PROMPT.get(verdict_class)
    if builder is None:
        raise ValueError(
            f"no prompt builder registered for verdict_class={verdict_class!r}"
        )

    if verdict_class == "shapes-incomplete":
        evidence = extract_missing_params_evidence(ctx.resolve_log, task_id)
        return builder(
            task_id, repo_root=ctx.repo_root,
            missing_params_evidence=evidence,
        )
    if verdict_class == "oracle-mismatch":
        mismatch = extract_oracle_mismatch_evidence(ctx.resolve_log, task_id)
        oracle_log = extract_oracle_log_excerpt(ctx.repo_root, task_id)
        return builder(
            task_id, ctx.repo_root,
            oracle_log=oracle_log or None,
            mismatch_evidence=mismatch or None,
        )
    if verdict_class == "upstream-not-invocable-here":
        hint = extract_upstream_callable_hint(ctx.resolve_log, task_id)
        blocker = extract_blocker_reason(ctx.resolve_log, task_id)
        return builder(
            task_id, ctx.repo_root,
            upstream_callable_hint=hint,
            blocker_reason=blocker or None,
        )
    raise ValueError(
        f"unhandled verdict_class for prompt dispatch: {verdict_class!r}"
    )


def _do_prompt_validate_apply(
    ctx: OrchestratorContext,
    state: dict[str, Any],
    task_id: str,
    entry: dict[str, Any],
    *,
    llm_chat,
    prompt_builder,
) -> dict[str, Any]:
    verdict_class = entry["verdict_class"]
    expected_decision_set = _VERDICT_TO_DECISIONS.get(verdict_class)
    if expected_decision_set is None:
        raise ValueError(
            f"no decision set for verdict_class={verdict_class!r}"
        )

    try:
        messages = _dispatch_prompt_builder(
            ctx, task_id, verdict_class,
            override_builder=prompt_builder,
        )
    except FileNotFoundError as exc:
        # Prompt builder refused to ask the LLM because the task tree is
        # incomplete (missing meta.yaml / oracle_pytorch.py). This is a
        # configuration bug -- park the task in UNFIXABLE rather than
        # silently retry.
        logger.warning("task %s prompt builder refused: %s", task_id, exc)
        sm.transition(entry, to="LLM_TRANSIENT_ERROR", state_data={
            "last_error_class": "FileNotFoundError",
            "retry_count": entry["state_data"].get("retry_count", 0) + 1,
        }, reason="prompt builder refused")
        sm.transition(entry, to="UNFIXABLE_BY_LLM", state_data={
            "attempts_used": entry["attempt"],
        }, reason="prompt builder refused -- task tree incomplete")
        return state

    try:
        chat_resp = llm_chat(messages)
        response_text = chat_resp.content if hasattr(chat_resp, "content") else str(chat_resp)
    except Exception as exc:
        logger.warning("task %s LLM error: %s", task_id, exc)
        sm.transition(entry, to="LLM_TRANSIENT_ERROR", state_data={
            "last_error_class": type(exc).__name__,
            "retry_count": entry["state_data"].get("retry_count", 0) + 1,
        }, reason="llm raised")
        return state

    # Parse meta + blocks. Decision-aware validation: first try the
    # tight per-decision required-keys set (so missing audit-trail
    # fields burn a turn); fall back to the union when the META is
    # missing entirely / mis-typed.
    meta = file_blocks.parse_meta(response_text)
    fbs = file_blocks.parse_blocks(response_text)
    if meta is not None and meta.decision in _DECISION_TO_REQUIRED_KEYS:
        expected_keys = _DECISION_TO_REQUIRED_KEYS[meta.decision]
    else:
        expected_keys = _VERDICT_TO_REQUIRED_KEYS_UNION[verdict_class]
    errors = file_blocks.validate_patch(
        meta, fbs, task_id, ctx.repo_root,
        expected_decision_set=set(expected_decision_set),
        expected_keys=set(expected_keys),
    )

    persist_prompt_and_response(
        ctx.repo_root, task_id, entry["attempt"],
        prompt_messages=messages,
        response_text=response_text,
        meta_payload=(meta.payload if meta else None),
        file_block_records=[{"name": b.path.name, "content": b.content} for b in fbs],
    )

    # Move into PATCH_VALIDATING for the audit trail
    sm.transition(entry, to="PATCH_VALIDATING",
                  state_data={"last_validation_errors": [str(e) for e in errors]},
                  reason="validate_patch")
    if errors:
        sm.append_history(entry, {
            "kind": "patch_validation_failed",
            "at": sm.iso_utc_now(),
            "round": entry["attempt"],
            "errors": [str(e) for e in errors],
        })
        entry["attempt"] = entry["attempt"] + 1
        if entry["attempt"] >= ctx.max_attempts:
            sm.transition(entry, to="PROMPTING", state_data={
                "last_prompt_round_id": f"round-{entry['attempt']:02d}",
            }, reason="validate fail; one more round")
            # immediately exhaust to UNFIXABLE -- mimic prompt-fail loop
            sm.transition(entry, to="LLM_TRANSIENT_ERROR", state_data={
                "last_error_class": "validation_fail",
                "retry_count": entry["attempt"],
            }, reason="validate_patch retry budget exhausted")
            sm.transition(entry, to="UNFIXABLE_BY_LLM", state_data={
                "attempts_used": entry["attempt"],
            }, reason="retry budget exhausted")
            return state
        sm.transition(entry, to="PROMPTING", state_data={
            "last_prompt_round_id": f"round-{entry['attempt']:02d}",
        }, reason="validate fail; retry")
        return state

    task_dir = _find_task_dir(ctx.repo_root, task_id)
    if task_dir is None:
        raise FileNotFoundError(f"task dir not found: {task_id}")
    rd = _patches_round_dir(ctx.repo_root, task_id, entry["attempt"])
    rd.mkdir(parents=True, exist_ok=True)

    # ---- preview-only short-circuit (dry-run / --apply not given) -------
    # Dry-run MUST short-circuit *before* snapshotting or moving past
    # PATCH_VALIDATING. Leaving state at BASELINE_SNAPSHOTTED would
    # block the next run (its only valid out-transitions assume an apply
    # is about to happen). Instead: persist a preview tree and move back
    # to PROMPTING so the next driver loop can re-prompt cleanly. The
    # CLI flips ``apply`` to False whenever the operator did not pass
    # ``--apply``; the in-process test helper relies on ``dry_run``.
    if ctx.dry_run or not ctx.apply:
        preview = rd / "preview"
        preview.mkdir(exist_ok=True)
        for b in fbs:
            (preview / b.path.name).write_text(b.content, encoding="utf-8")
        sm.transition(entry, to="PROMPTING", state_data={
            "last_prompt_round_id": f"round-{entry['attempt']:02d}",
        }, reason="dry-run: preview written; not advancing past validate")
        return state

    # Snapshot baseline (only when --apply gated this path).
    snap_tar = rd / "baseline.tar.gz"
    snap_manifest = rd / "manifest.json"
    atomic.snapshot_task_baseline(task_dir, snap_tar, snap_manifest)
    sm.transition(entry, to="BASELINE_SNAPSHOTTED", state_data={
        "baseline_tar_path": str(snap_tar),
        "manifest_path": str(snap_manifest),
    }, reason="baseline captured")

    # apply_patch -- on failure, rollback + state transition per plan
    try:
        file_blocks.apply_patch(meta, fbs, task_id, ctx.repo_root)
    except OSError as exc:
        logger.warning("task %s apply_patch failed: %s", task_id, exc)
        try:
            atomic.rollback_task(task_dir, snap_tar, snap_manifest)
        except Exception:  # pragma: no cover
            logger.exception("rollback_task failed after apply_patch error")
        sm.append_history(entry, {
            "kind": "rollback", "at": sm.iso_utc_now(),
            "round": entry["attempt"],
            "reason": "apply_patch raised OSError",
        })
        next_attempt = entry["attempt"] + 1
        if next_attempt < ctx.max_attempts:
            entry["attempt"] = next_attempt
            sm.transition(entry, to="ATTEMPTING_NEXT_ROUND",
                          state_data={}, reason="apply failure -- retry")
            return state
        sm.transition(entry, to="UNFIXABLE_BY_LLM",
                      state_data={"attempts_used": ctx.max_attempts},
                      reason="apply failure budget exhausted")
        return state

    # Apply succeeded -- advance straight through transient PATCH_APPLIED
    # and into BENCH_RUNNING with pending_* fields set.
    sm.append_history(entry, {
        "kind": "transition",
        "state": "BASELINE_SNAPSHOTTED",
        "to": "PATCH_APPLIED",
        "at": sm.iso_utc_now(),
        "reason": "apply_patch ok (transient)",
    })
    run_id = _mint_run_id(tier=ctx.tier)
    # Decision-aware reverify cmd: patch-shapes re-runs the resolver;
    # the other paths (tolerance / oracle rewrite / defer) re-run
    # Phase A' so meta-level edits + oracle rewrites + defer flags get
    # exercised end-to-end.
    decision = meta.decision if meta is not None else "patch-shapes"
    pending_cmd = _DECISION_TO_REVERIFY_CMD.get(decision, "resolve")
    # Dispatch the job under the configured tier. Tier 1 just creates
    # the result dir locally; Tier 2 also enqueues onto the GPFS
    # directory-rename queue so the daemon picks it up.
    pending_dir = submit_reverify_job(ctx, task_id, run_id, pending_cmd)
    sm.transition(entry, to="BENCH_RUNNING", state_data={
        "pending_run_id": run_id,
        "pending_cmd": pending_cmd,
        "pending_result_dir": str(pending_dir),
    }, reason="patched + pending reverify")
    if ctx.tier == 1:
        # User-visible reverify command (orchestrator main loop prints
        # this for the operator to copy onto the GPU host).
        entry.setdefault("_pending_cli", []).append(
            f"python3 scripts/automation/tier1_reverify.py "
            f"--task {task_id} --cmd {pending_cmd} "
            f"--out-dir {ctx.reverify_results} --run-id {run_id}"
        )
    return state


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--resolve-log", required=True, type=Path)
    p.add_argument(
        "--verdict-class", default="all",
        choices=[
            "all",
            "shapes-incomplete",
            "oracle-mismatch",
            "upstream-not-invocable-here",
        ],
        help="Filter to one canonical class; ``all`` (default) drives "
             "every supported class through its own prompt builder.",
    )
    p.add_argument("--max-tasks", type=int, default=30)
    p.add_argument("--max-attempts", type=int, default=3)
    p.add_argument("--llm-endpoint", required=False, default=None)
    p.add_argument("--reverify-results", required=True, type=Path)
    p.add_argument("--repo-root", type=Path, default=_REPO_ROOT)
    p.add_argument("--state-root", type=Path, default=None)
    # --dry-run / --apply are mutually exclusive: the orchestrator
    # refuses to "accidentally" write task files unless --apply is
    # explicit. Default (neither flag) is dry-run-equivalent and never
    # advances past validate/preview.
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true",
                      help="Preview only; never snapshot+apply or advance "
                           "state past PATCH_VALIDATING.")
    mode.add_argument("--apply", action="store_true",
                      help="Required to actually snapshot+apply patches "
                           "and advance to BENCH_RUNNING.")
    p.add_argument("--init-empty-state", action="store_true")
    p.add_argument("--sweep-only", action="store_true",
                   help="M2: sweep BACKLOG_PENDING_GOLDEN only; no main loop.")
    # ---- Tier 2 dispatch (Unit 4.2) -----------------------------------
    p.add_argument(
        "--tier", type=int, default=1, choices=(1, 2),
        help="Reverify dispatch tier: 1 (default; operator-runnable "
             "wrapper command) or 2 (enqueue onto GPFS queue for "
             "bench_evaluator_daemon).",
    )
    p.add_argument(
        "--queue-root", type=Path, default=None,
        help="GPFS queue directory (required when --tier 2). The daemon "
             "polls <queue_root>/<run_id>/READY markers atomically.",
    )
    p.add_argument(
        "--daemon-timeout-minutes", type=int, default=60,
        help="Tier 2 only: minutes to wait for daemon DONE before "
             "recording a daemon_timeout history event (default: 60).",
    )
    p.add_argument(
        "--daemon-timeout-max-attempts", type=int, default=3,
        help="Tier 2 only: consecutive daemon_timeout events on the "
             "same pending_run_id before transitioning to "
             "ATTEMPTING_NEXT_ROUND (default: 3).",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s :: %(message)s")
    state_root = args.state_root or (args.repo_root / "state")
    state_root.mkdir(parents=True, exist_ok=True)
    state_path = state_root / "triage_state.json"
    global_lock = state_root / ".lock"

    with lock.acquire_lock(global_lock, start_heartbeat=False):
        try:
            state = load_or_init_empty_state(state_path, allow_empty=args.init_empty_state)
        except StateFileNotFoundError as exc:
            print(f"FATAL: {exc}", file=sys.stderr)
            return 2
        except StateFileCorruptError as exc:
            print(f"FATAL: state corrupt; {exc}; quarantined; "
                  f"manual intervention required (--init-empty-state to restart)",
                  file=sys.stderr)
            return 2

        # ``--verdict-class all`` (default) drives every canonical class
        # through its own prompt builder; otherwise the orchestrator
        # filters to the explicit class for backward-compat with the M1
        # CLI (``--verdict-class shapes-incomplete``).
        if args.verdict_class == "all":
            class_filter: str | None = None
            allowed_enqueue = _VERDICT_TO_DECISIONS.keys()
        else:
            class_filter = args.verdict_class
            allowed_enqueue = {args.verdict_class}

        # CLI gating: argparse already makes --dry-run / --apply
        # mutually exclusive. Default (neither flag) is preview-only.
        cli_apply = bool(args.apply)
        # Tier 2 requires --queue-root; fail fast so the operator does
        # not discover the misconfig mid-loop.
        if args.tier == 2 and args.queue_root is None:
            print(
                "FATAL: --tier 2 requires --queue-root <path>",
                file=sys.stderr,
            )
            return 2
        ctx = OrchestratorContext(
            repo_root=args.repo_root,
            state_root=state_root,
            reverify_results=args.reverify_results,
            resolve_log=args.resolve_log,
            max_tasks=args.max_tasks,
            max_attempts=args.max_attempts,
            dry_run=(not cli_apply),
            apply=cli_apply,
            verdict_class_filter=class_filter,
            tier=args.tier,
            queue_root=args.queue_root,
            daemon_timeout_minutes=args.daemon_timeout_minutes,
            daemon_timeout_max_attempts=args.daemon_timeout_max_attempts,
        )

        if args.sweep_only:
            # Sweep BACKLOG_PENDING_GOLDEN only -- promotes to DONE when
            # the phase B capture daemon has written ``status: DONE`` to
            # meta.yaml.phase_b_capture.
            sweep_backlog_pending_golden(ctx, state)
            write_state(state, state_path)
            return 0

        # Parse resolve log + enqueue (filter by allowed class set)
        for tid, vclass in parse_resolve_log(args.resolve_log)[: args.max_tasks]:
            if vclass not in allowed_enqueue:
                continue
            state["tasks"].setdefault(tid, sm.new_task_entry(vclass))

        # Runtime requires an llm endpoint; tests pass None + bypass via
        # drive_task_once + their own mock.
        if not args.llm_endpoint:
            print("FATAL: --llm-endpoint is required for runtime use",
                  file=sys.stderr)
            return 2

        from automation.llm_client import LLMClient

        client = LLMClient(args.llm_endpoint, api_style="openai")

        def _chat(messages):
            return client.chat(messages)

        # Always sweep BACKLOG_PENDING_GOLDEN at the top of every run --
        # phase B capture may have completed between sweeps and tasks
        # need to be promoted before the LLM patch loop runs.
        sweep_backlog_pending_golden(ctx, state)

        pending_cli: list[str] = []
        for tid in list(state["tasks"].keys()):
            # Per-task lock nested inside the global lock. ``LockHandle``
            # is a context manager whose ``__exit__`` releases the lock
            # (LIFO: the per-task lock unwinds before the global lock).
            # This prevents two co-resident orchestrator processes from
            # racing on the same task_dir / patches/<task> directory
            # while still letting them progress on different tasks.
            per_task_lock = state_root / f".lock.{tid}"
            try:
                with lock.acquire_lock(per_task_lock, start_heartbeat=False):
                    try:
                        # ``prompt_builder=None`` lets the orchestrator
                        # pick the per-class builder via the dispatch
                        # table (verdict_class -> _VERDICT_TO_PROMPT).
                        drive_task_once(
                            ctx, state, tid,
                            llm_chat=_chat, prompt_builder=None,
                        )
                    except sm.StateTransitionError as exc:
                        logger.error("task %s transition error: %s", tid, exc)
            except lock.LockBusyError as exc:
                logger.warning(
                    "task %s per-task lock busy; skipped this sweep: %s",
                    tid, exc,
                )
                continue
            cli_lines = state["tasks"][tid].pop("_pending_cli", None)
            if cli_lines:
                pending_cli.extend(cli_lines)

        write_state(state, state_path)

    # Print user-actionable wrapper commands to stdout (Tier 1 only;
    # Tier 2 jobs are dispatched onto the GPFS queue automatically).
    if ctx.tier == 1:
        for cli in pending_cli:
            print(f"# user run on GPU host: {cli}")
    else:
        if pending_cli:
            logger.info(
                "tier 2: %d job(s) enqueued; daemon will execute "
                "(no operator action needed)",
                len(pending_cli),
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "OrchestratorContext",
    "StateFileCorruptError",
    "StateFileNotFoundError",
    "apply_release_scope_exclusion",
    "drive_task_once",
    "extract_blocker_reason",
    "extract_missing_params_evidence",
    "extract_oracle_log_excerpt",
    "extract_oracle_mismatch_evidence",
    "extract_upstream_callable_hint",
    "load_existing_triage_state",
    "load_or_init_empty_state",
    "parse_resolve_log",
    "persist_prompt_and_response",
    "submit_reverify_job",
    "sweep_backlog_pending_golden",
    "write_state",
]
