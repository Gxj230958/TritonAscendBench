"""LLM-driven fix orchestrator (Unit 1.9) -- M1 subset.

CLI entry-point. M1 supports the ``shapes-incomplete`` verdict_class
pipeline end-to-end through the canonical 17-state machine. M2 unlocks
the remaining verdict classes (oracle-mismatch / upstream-not-invocable-here)
and the defer-to-golden sweep.

Public helpers (also exported for callers / tests):
  * ``apply_release_scope_exclusion(task_id, reason, excluded_by, repo_root)``
  * ``load_existing_triage_state(state_path)``         (strict)
  * ``load_or_init_empty_state(state_path, *, allow_empty=False)``
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

logger = logging.getLogger("automation.fix_orchestrator")

SCHEMA_VERSION = "1.0"


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


def extract_missing_params_evidence(
    log_path: Path, task_id: str, max_lines: int = 8,
) -> str:
    """Pull resolver evidence lines for ``task_id`` from ``pa_resolve.log``.

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
        # Only surface lines whose subclass is shapes-incomplete (the
        # M1 LLM-driven loop only triages that subclass).
        raw = m.group("reason").strip()
        subclass = raw.split(":", 1)[1].strip() if ":" in raw else raw
        if subclass != "shapes-incomplete":
            continue
        evidence = m.group("evidence").strip()
        if evidence:
            hits.append(f"{raw} :: {evidence}")
    # Keep the freshest ``max_lines`` (resolver appends; tail wins).
    return "\n".join(hits[-max_lines:])


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
    verdict_class_filter: str = "shapes-incomplete"


def _mint_run_id() -> str:
    return f"tier1-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}-{uuid.uuid4().hex[:8]}"


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

    # M1: only handle shapes-incomplete
    if entry["verdict_class"] != ctx.verdict_class_filter:
        logger.info("task %s class=%s skipped (M1 only shapes-incomplete)",
                    task_id, entry["verdict_class"])
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
        # M1: not handled (defer-to-golden lives in M2). Park as
        # BACKLOG_PENDING_GOLDEN with placeholder snapshot.
        sm.transition(entry, to="BACKLOG_PENDING_GOLDEN",
                      state_data={"phase_b_capture_status_snapshot": "PENDING"},
                      reason="deferred to M2 defer-to-golden flow")
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


def _do_prompt_validate_apply(
    ctx: OrchestratorContext,
    state: dict[str, Any],
    task_id: str,
    entry: dict[str, Any],
    *,
    llm_chat,
    prompt_builder,
) -> dict[str, Any]:
    # Build prompt + call LLM. The prompt builder accepts an optional
    # ``missing_params_evidence`` kwarg -- we extract the latest
    # resolver evidence lines for this task and pass them through so
    # the LLM sees the verbatim reject_notes from
    # ``resolve_upstream_entrypoints.py`` (which usually name the
    # missing parameter directly).
    evidence = extract_missing_params_evidence(ctx.resolve_log, task_id)
    try:
        messages = prompt_builder(
            task_id, ctx.repo_root,
            missing_params_evidence=evidence,
        )
    except TypeError:
        # Tests / older callers may pass a 2-arg builder; fall back so
        # we never break them.
        messages = prompt_builder(task_id, ctx.repo_root)
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

    # Parse meta + blocks
    meta = file_blocks.parse_meta(response_text)
    fbs = file_blocks.parse_blocks(response_text)
    # M1 shapes-incomplete pipeline: the META block must carry the four
    # required keys -- ``decision`` + the audit-trail trio
    # (``missing_params``, ``rationale``, ``evidence``). The original
    # ``{decision}``-only gate let the LLM ship an empty rationale /
    # missing-params object, defeating downstream verification.
    errors = file_blocks.validate_patch(
        meta, fbs, task_id, ctx.repo_root,
        expected_decision_set={"patch-shapes"},
        expected_keys={"decision", "missing_params", "rationale", "evidence"},
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
    run_id = _mint_run_id()
    pending_dir = ctx.reverify_results / run_id
    pending_dir.mkdir(parents=True, exist_ok=True)
    sm.transition(entry, to="BENCH_RUNNING", state_data={
        "pending_run_id": run_id,
        "pending_cmd": "resolve",
        "pending_result_dir": str(pending_dir),
    }, reason="patched + pending reverify")
    # User-visible reverify command (orchestrator main loop prints this).
    entry.setdefault("_pending_cli", []).append(
        f"python3 scripts/automation/tier1_reverify.py "
        f"--task {task_id} --cmd resolve "
        f"--out-dir {ctx.reverify_results} --run-id {run_id}"
    )
    return state


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--resolve-log", required=True, type=Path)
    p.add_argument("--verdict-class", default="shapes-incomplete",
                   choices=["shapes-incomplete"])  # M1 limit
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

        if args.sweep_only:
            # M1: no-op; M2 will implement the sweep here.
            print("sweep-only is a no-op in M1", flush=True)
            return 0

        # Parse resolve log + enqueue
        for tid, vclass in parse_resolve_log(args.resolve_log)[: args.max_tasks]:
            if vclass != args.verdict_class:
                continue
            state["tasks"].setdefault(tid, sm.new_task_entry(vclass))

        # M1 requires an llm endpoint at runtime; tests pass None +
        # bypass via drive_task_once + their own mock.
        if not args.llm_endpoint:
            print("FATAL: --llm-endpoint is required for runtime use",
                  file=sys.stderr)
            return 2

        from automation.llm_client import LLMClient

        # CLI gating: argparse already makes --dry-run / --apply
        # mutually exclusive. Default (neither flag) is preview-only.
        cli_apply = bool(args.apply)
        ctx = OrchestratorContext(
            repo_root=args.repo_root,
            state_root=state_root,
            reverify_results=args.reverify_results,
            resolve_log=args.resolve_log,
            max_tasks=args.max_tasks,
            max_attempts=args.max_attempts,
            dry_run=(not cli_apply),
            apply=cli_apply,
            verdict_class_filter=args.verdict_class,
        )
        client = LLMClient(args.llm_endpoint, api_style="openai")

        def _chat(messages):
            return client.chat(messages)

        from automation.prompts.shapes_incomplete import build_prompt

        def _build_prompt(task_id, repo_root, *, missing_params_evidence=""):
            return build_prompt(
                task_id=task_id, repo_root=repo_root,
                missing_params_evidence=missing_params_evidence,
            )

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
                        drive_task_once(
                            ctx, state, tid,
                            llm_chat=_chat, prompt_builder=_build_prompt,
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

    # Print user-actionable wrapper commands to stdout
    for cli in pending_cli:
        print(f"# user run on GPU host: {cli}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "OrchestratorContext",
    "StateFileCorruptError",
    "StateFileNotFoundError",
    "apply_release_scope_exclusion",
    "drive_task_once",
    "extract_missing_params_evidence",
    "load_existing_triage_state",
    "load_or_init_empty_state",
    "parse_resolve_log",
    "persist_prompt_and_response",
    "write_state",
]
