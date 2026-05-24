"""Tests for ``automation.fix_orchestrator`` Tier 2 dispatch (Unit 4.2).

Covers the GPFS-queue dispatch path: ``submit_reverify_job`` enqueues
onto the directory-rename queue, the orchestrator polls
``<reverify_results>/<run_id>/DONE`` for the daemon's result, and the
``daemon_timeout`` history-event book-keeping that re-prompts the LLM
after consecutive stalls.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPTS = _REPO_ROOT / "scripts"


@pytest.fixture(autouse=True)
def _ensure_scripts_on_path():
    inserted = False
    if str(_SCRIPTS) not in sys.path:
        sys.path.insert(0, str(_SCRIPTS))
        inserted = True
    yield
    if inserted:
        try:
            sys.path.remove(str(_SCRIPTS))
        except ValueError:
            pass


# ---------------------------------------------------------------------------
# Shared scaffolding (mirrors test_fix_orchestrator.py minimal repo layout)
# ---------------------------------------------------------------------------

LEVEL = "L1"
TASK_ID = "L1_06_swiglu"


def _scaffold_repo(tmp_path: Path, task_id: str = TASK_ID) -> tuple[Path, Path, Path, Path]:
    repo = tmp_path / "repo"
    repo.mkdir()
    td = repo / "tasks" / LEVEL / task_id
    td.mkdir(parents=True)
    (td / "meta.yaml").write_text(
        "schema_version: 1.0\nname: x\n", encoding="utf-8",
    )
    (td / "shapes.yaml").write_text(
        "public_shapes: []\n", encoding="utf-8",
    )
    (td / "oracle_pytorch.py").write_text(
        "def oracle(x):\n    return x\n", encoding="utf-8",
    )
    state_root = repo / "state"
    state_root.mkdir()
    reverify = tmp_path / "results"
    reverify.mkdir()
    queue_root = tmp_path / "queue"
    queue_root.mkdir()
    return repo, state_root, reverify, queue_root


def _good_llm_response(task_id: str = TASK_ID) -> str:
    new_shapes = (
        "public_shapes:\n  - {id: s1, name: small, knobs: {M: 16}}\n"
    )
    meta_payload = (
        '{"decision": "patch-shapes", '
        '"missing_params": ["x"], '
        '"rationale": "fill missing param", '
        '"evidence": {"missing": ["x"]}}'
    )
    return (
        f"### META\n{meta_payload}\n### END\n\n"
        f"### FILE: tasks/{LEVEL}/{task_id}/shapes.yaml\n"
        f"{new_shapes}\n### END\n"
    )


def _fake_prompt(*a, **kw):
    return [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "u"},
    ]


def _mk_ctx(
    repo: Path, state_root: Path, reverify: Path, queue_root: Path | None,
    **overrides,
):
    from automation.fix_orchestrator import OrchestratorContext

    defaults = {
        "repo_root": repo,
        "state_root": state_root,
        "reverify_results": reverify,
        "resolve_log": repo / "results" / "gpu_pipeline" / "pa_resolve.log",
        "max_tasks": 3,
        "max_attempts": 3,
        "dry_run": False,
        "verdict_class_filter": "shapes-incomplete",
        "tier": 2,
        "queue_root": queue_root,
        "daemon_timeout_minutes": 60,
        "daemon_timeout_max_attempts": 3,
    }
    defaults.update(overrides)
    return OrchestratorContext(**defaults)


# ---------------------------------------------------------------------------
# submit_reverify_job: tier dispatch
# ---------------------------------------------------------------------------

def test_submit_reverify_job_tier1_creates_result_dir_only(tmp_path):
    from automation.fix_orchestrator import submit_reverify_job

    repo, state_root, reverify, queue_root = _scaffold_repo(tmp_path)
    # Tier 1 -- queue_root is irrelevant (left unset).
    ctx = _mk_ctx(repo, state_root, reverify, None, tier=1)
    run_id = "tier1-20260524T000000Z-abcdef01"
    result_dir = submit_reverify_job(ctx, TASK_ID, run_id, "resolve")
    assert result_dir == reverify / run_id
    assert result_dir.is_dir()
    # No queue mutation expected (queue_root None for tier 1).
    assert list(queue_root.iterdir()) == []


def test_submit_reverify_job_tier2_writes_queue_atomic(tmp_path):
    from automation.fix_orchestrator import submit_reverify_job

    repo, state_root, reverify, queue_root = _scaffold_repo(tmp_path)
    ctx = _mk_ctx(repo, state_root, reverify, queue_root, tier=2)
    run_id = "tier2-20260524T000000Z-cafebabe"
    result_dir = submit_reverify_job(ctx, TASK_ID, run_id, "resolve")

    # Atomic queue layout: <queue_root>/<run_id>/{task.json, READY}
    run_dir = queue_root / run_id
    assert run_dir.is_dir()
    task_json_path = run_dir / "task.json"
    ready_path = run_dir / "READY"
    assert task_json_path.is_file()
    assert ready_path.is_file()

    # task.json schema must carry the daemon-expected fields.
    payload = json.loads(task_json_path.read_text(encoding="utf-8"))
    assert payload["task_id"] == TASK_ID
    assert payload["run_id"] == run_id
    assert payload["cmd"] == "resolve"
    assert "requested_at" in payload

    # Result dir created locally so the orchestrator can poll it.
    assert result_dir == reverify / run_id
    assert result_dir.is_dir()


def test_submit_reverify_job_tier2_missing_queue_root_raises(tmp_path):
    from automation.fix_orchestrator import submit_reverify_job

    repo, state_root, reverify, _ = _scaffold_repo(tmp_path)
    # ctx with tier=2 but queue_root=None -> ValueError.
    ctx = _mk_ctx(repo, state_root, reverify, None, tier=2)
    with pytest.raises(ValueError, match="queue_root"):
        submit_reverify_job(ctx, TASK_ID, "run-x", "resolve")


def test_submit_reverify_job_rejects_unknown_cmd(tmp_path):
    from automation.fix_orchestrator import submit_reverify_job

    repo, state_root, reverify, queue_root = _scaffold_repo(tmp_path)
    ctx = _mk_ctx(repo, state_root, reverify, queue_root, tier=2)
    with pytest.raises(ValueError, match="cmd"):
        submit_reverify_job(ctx, TASK_ID, "run-x", "bogus_cmd")


def test_submit_reverify_job_unknown_tier_raises(tmp_path):
    from automation.fix_orchestrator import submit_reverify_job

    repo, state_root, reverify, queue_root = _scaffold_repo(tmp_path)
    ctx = _mk_ctx(repo, state_root, reverify, queue_root, tier=99)
    with pytest.raises(ValueError, match="tier"):
        submit_reverify_job(ctx, TASK_ID, "run-x", "resolve")


# ---------------------------------------------------------------------------
# drive_task_once: tier 2 full happy + DONE barrier
# ---------------------------------------------------------------------------

def test_drive_task_once_tier2_enqueues_and_marks_bench_running(tmp_path):
    from automation.fix_orchestrator import drive_task_once

    repo, state_root, reverify, queue_root = _scaffold_repo(tmp_path)
    ctx = _mk_ctx(repo, state_root, reverify, queue_root, tier=2)
    state = {"schema_version": "1.0", "tasks": {}}

    chat = MagicMock(return_value=SimpleNamespace(content=_good_llm_response()))
    drive_task_once(
        ctx, state, TASK_ID, llm_chat=chat, prompt_builder=_fake_prompt,
    )

    entry = state["tasks"][TASK_ID]
    assert entry["state"] == "BENCH_RUNNING"
    sd = entry["state_data"]
    run_id = sd["pending_run_id"]
    assert run_id.startswith("tier2-")
    assert sd["pending_cmd"] == "resolve"
    pending_dir = Path(sd["pending_result_dir"])
    assert pending_dir == reverify / run_id

    # Queue side: <queue_root>/<run_id>/{task.json, READY} present.
    run_dir = queue_root / run_id
    assert (run_dir / "task.json").is_file()
    assert (run_dir / "READY").is_file()
    # No _pending_cli message in tier 2 (operator does nothing).
    assert "_pending_cli" not in entry


def test_drive_task_once_tier2_consume_done_when_daemon_writes_result(tmp_path):
    from automation import extract_verdict as ev
    from automation.fix_orchestrator import drive_task_once

    repo, state_root, reverify, queue_root = _scaffold_repo(tmp_path)
    ctx = _mk_ctx(repo, state_root, reverify, queue_root, tier=2)
    state = {"schema_version": "1.0", "tasks": {}}

    chat = MagicMock(return_value=SimpleNamespace(content=_good_llm_response()))
    drive_task_once(
        ctx, state, TASK_ID, llm_chat=chat, prompt_builder=_fake_prompt,
    )
    entry = state["tasks"][TASK_ID]
    run_id = entry["state_data"]["pending_run_id"]

    # Daemon-side: write the canonical 9-field result.json + DONE.
    ev.write_verdict(
        task_id=TASK_ID, cmd="resolve", run_id=run_id,
        producer="daemon", out_dir=reverify,
        verdict="PASS", verdict_class=None, reason="daemon PASS",
    )

    drive_task_once(
        ctx, state, TASK_ID, llm_chat=chat, prompt_builder=_fake_prompt,
    )
    entry = state["tasks"][TASK_ID]
    assert entry["state"] == "DONE"
    assert entry["state_data"]["final_verdict"] == "PASS"
    assert entry["last_consumed_run_id"] == run_id


def test_drive_task_once_tier2_partial_write_not_consumed(tmp_path):
    from automation.fix_orchestrator import drive_task_once

    repo, state_root, reverify, queue_root = _scaffold_repo(tmp_path)
    ctx = _mk_ctx(repo, state_root, reverify, queue_root, tier=2)
    state = {"schema_version": "1.0", "tasks": {}}
    chat = MagicMock(return_value=SimpleNamespace(content=_good_llm_response()))
    drive_task_once(
        ctx, state, TASK_ID, llm_chat=chat, prompt_builder=_fake_prompt,
    )
    entry = state["tasks"][TASK_ID]
    run_id = entry["state_data"]["pending_run_id"]
    pending_dir = reverify / run_id

    # Simulate partial write: result.json + _provenance.json but NO DONE.
    (pending_dir / "_provenance.json").write_text(
        json.dumps({"producer": "daemon"}), encoding="utf-8",
    )
    (pending_dir / "result.json").write_text(
        json.dumps({
            "schema_version": "1.0",
            "task_id": TASK_ID, "cmd": "resolve", "run_id": run_id,
            "producer": "daemon", "verdict": "PASS",
            "verdict_class": None, "reason": "ok",
            "wrote_at": "2026-05-24T00:00:00Z",
        }),
        encoding="utf-8",
    )

    # Without DONE the orchestrator MUST NOT consume.
    drive_task_once(
        ctx, state, TASK_ID, llm_chat=chat, prompt_builder=_fake_prompt,
    )
    entry = state["tasks"][TASK_ID]
    assert entry["state"] == "BENCH_RUNNING"
    assert entry["state_data"]["pending_run_id"] == run_id


# ---------------------------------------------------------------------------
# Daemon timeout accumulation / re-prompt
# ---------------------------------------------------------------------------

def test_tier2_daemon_timeout_records_history_event(tmp_path):
    from automation.fix_orchestrator import drive_task_once

    repo, state_root, reverify, queue_root = _scaffold_repo(tmp_path)
    ctx = _mk_ctx(
        repo, state_root, reverify, queue_root, tier=2,
        daemon_timeout_minutes=60,
    )
    state = {"schema_version": "1.0", "tasks": {}}
    chat = MagicMock(return_value=SimpleNamespace(content=_good_llm_response()))
    drive_task_once(
        ctx, state, TASK_ID, llm_chat=chat, prompt_builder=_fake_prompt,
    )
    entry = state["tasks"][TASK_ID]
    run_id = entry["state_data"]["pending_run_id"]

    # Simulate "long time has passed without a DONE" by patching the
    # epoch clock used inside _maybe_handle_tier2_daemon_timeout. The
    # window is 60 min = 3600s; we jump 1 hour ahead.
    import datetime as dtmod
    real_now = dtmod.datetime.now(dtmod.UTC).timestamp()
    fake_now = real_now + 3601  # > 60 min

    with patch("automation.fix_orchestrator.datetime") as dt_mock:
        dt_mock.now.return_value.timestamp.return_value = fake_now
        # ``datetime`` is also used elsewhere via UTC; make UTC pass-thru
        dt_mock.UTC = dtmod.UTC
        # iso timestamps still go through sm.iso_utc_now (different module)
        dt_mock.fromisoformat = dtmod.datetime.fromisoformat
        drive_task_once(
            ctx, state, TASK_ID, llm_chat=chat, prompt_builder=_fake_prompt,
        )

    entry = state["tasks"][TASK_ID]
    # State remains BENCH_RUNNING (no canonical timeout state)
    assert entry["state"] == "BENCH_RUNNING"
    assert entry["state_data"]["pending_run_id"] == run_id
    # history must carry exactly one daemon_timeout event for this run
    dt_events = [
        rec for rec in entry["history"]
        if rec.get("kind") == "daemon_timeout"
           and rec.get("run_id") == run_id
    ]
    assert len(dt_events) == 1
    assert dt_events[0]["timeout_minutes"] == 60


def test_tier2_daemon_timeout_accumulates_to_attempting_next_round(tmp_path):
    from automation.fix_orchestrator import drive_task_once

    repo, state_root, reverify, queue_root = _scaffold_repo(tmp_path)
    ctx = _mk_ctx(
        repo, state_root, reverify, queue_root, tier=2,
        daemon_timeout_minutes=60,
        daemon_timeout_max_attempts=3,
        max_attempts=5,
    )
    state = {"schema_version": "1.0", "tasks": {}}
    chat = MagicMock(return_value=SimpleNamespace(content=_good_llm_response()))
    drive_task_once(
        ctx, state, TASK_ID, llm_chat=chat, prompt_builder=_fake_prompt,
    )
    entry = state["tasks"][TASK_ID]
    run_id = entry["state_data"]["pending_run_id"]

    import datetime as dtmod
    real_now = dtmod.datetime.now(dtmod.UTC).timestamp()

    # Fire 3 consecutive timeouts by stepping the fake clock forward
    # 60 min at a time.
    for i in range(1, 4):
        fake_now = real_now + 3601 * i
        with patch("automation.fix_orchestrator.datetime") as dt_mock:
            dt_mock.now.return_value.timestamp.return_value = fake_now
            dt_mock.UTC = dtmod.UTC
            dt_mock.fromisoformat = dtmod.datetime.fromisoformat
            drive_task_once(
                ctx, state, TASK_ID,
                llm_chat=chat, prompt_builder=_fake_prompt,
            )

    entry = state["tasks"][TASK_ID]
    # 3 timeouts -> transition to ATTEMPTING_NEXT_ROUND
    assert entry["state"] == "ATTEMPTING_NEXT_ROUND"
    # pending_* removed by the transition (state_data REPLACED).
    assert "pending_run_id" not in entry["state_data"]
    # daemon_timeout history is preserved (3 entries for this run_id)
    dt_events = [
        rec for rec in entry["history"]
        if rec.get("kind") == "daemon_timeout"
           and rec.get("run_id") == run_id
    ]
    assert len(dt_events) == 3
    # attempt counter incremented for the retry
    assert entry["attempt"] == 1


def test_tier2_daemon_timeout_budget_exhausted_unfixable(tmp_path):
    from automation.fix_orchestrator import drive_task_once

    repo, state_root, reverify, queue_root = _scaffold_repo(tmp_path)
    # max_attempts=1: the first daemon_timeout retry exceeds the
    # per-task attempt budget and should land in UNFIXABLE.
    ctx = _mk_ctx(
        repo, state_root, reverify, queue_root, tier=2,
        daemon_timeout_minutes=60,
        daemon_timeout_max_attempts=3,
        max_attempts=1,
    )
    state = {"schema_version": "1.0", "tasks": {}}
    chat = MagicMock(return_value=SimpleNamespace(content=_good_llm_response()))
    drive_task_once(
        ctx, state, TASK_ID, llm_chat=chat, prompt_builder=_fake_prompt,
    )
    entry = state["tasks"][TASK_ID]

    import datetime as dtmod
    real_now = dtmod.datetime.now(dtmod.UTC).timestamp()
    for i in range(1, 4):
        fake_now = real_now + 3601 * i
        with patch("automation.fix_orchestrator.datetime") as dt_mock:
            dt_mock.now.return_value.timestamp.return_value = fake_now
            dt_mock.UTC = dtmod.UTC
            dt_mock.fromisoformat = dtmod.datetime.fromisoformat
            drive_task_once(
                ctx, state, TASK_ID,
                llm_chat=chat, prompt_builder=_fake_prompt,
            )

    entry = state["tasks"][TASK_ID]
    # max_attempts=1 + 1 increment -> attempt budget exhausted ->
    # UNFIXABLE_BY_LLM (semi-terminal).
    assert entry["state"] == "UNFIXABLE_BY_LLM"


def test_tier2_daemon_timeout_does_not_fire_within_grace_window(tmp_path):
    from automation.fix_orchestrator import drive_task_once

    repo, state_root, reverify, queue_root = _scaffold_repo(tmp_path)
    ctx = _mk_ctx(
        repo, state_root, reverify, queue_root, tier=2,
        daemon_timeout_minutes=60,
    )
    state = {"schema_version": "1.0", "tasks": {}}
    chat = MagicMock(return_value=SimpleNamespace(content=_good_llm_response()))
    drive_task_once(
        ctx, state, TASK_ID, llm_chat=chat, prompt_builder=_fake_prompt,
    )
    entry = state["tasks"][TASK_ID]
    run_id = entry["state_data"]["pending_run_id"]

    # Re-poll immediately -- still within grace window -> no event.
    drive_task_once(
        ctx, state, TASK_ID, llm_chat=chat, prompt_builder=_fake_prompt,
    )
    entry = state["tasks"][TASK_ID]
    dt_events = [
        rec for rec in entry["history"]
        if rec.get("kind") == "daemon_timeout"
           and rec.get("run_id") == run_id
    ]
    assert len(dt_events) == 0
    assert entry["state"] == "BENCH_RUNNING"


def test_tier2_daemon_timeout_only_counts_current_run_id(tmp_path):
    """Stale ``daemon_timeout`` events from a previous (retired) run_id
    must NOT count toward the current run_id's retry budget.
    """
    from automation.fix_orchestrator import drive_task_once

    repo, state_root, reverify, queue_root = _scaffold_repo(tmp_path)
    ctx = _mk_ctx(
        repo, state_root, reverify, queue_root, tier=2,
        daemon_timeout_minutes=60, daemon_timeout_max_attempts=3,
        max_attempts=5,
    )
    state = {"schema_version": "1.0", "tasks": {}}
    chat = MagicMock(return_value=SimpleNamespace(content=_good_llm_response()))
    drive_task_once(
        ctx, state, TASK_ID, llm_chat=chat, prompt_builder=_fake_prompt,
    )
    entry = state["tasks"][TASK_ID]
    run_id_a = entry["state_data"]["pending_run_id"]

    # Inject 5 stale daemon_timeout events for a DIFFERENT run_id.
    from automation import state_machine as sm
    for _ in range(5):
        sm.append_history(entry, {
            "kind": "daemon_timeout",
            "at": "2026-01-01T00:00:00Z",
            "run_id": "tier2-stale-deadbeef",
        })

    # Now fire one real timeout for the current run -- count should be 1,
    # which is < daemon_timeout_max_attempts=3, so state stays
    # BENCH_RUNNING.
    import datetime as dtmod
    real_now = dtmod.datetime.now(dtmod.UTC).timestamp()
    with patch("automation.fix_orchestrator.datetime") as dt_mock:
        dt_mock.now.return_value.timestamp.return_value = real_now + 3601
        dt_mock.UTC = dtmod.UTC
        dt_mock.fromisoformat = dtmod.datetime.fromisoformat
        drive_task_once(
            ctx, state, TASK_ID, llm_chat=chat, prompt_builder=_fake_prompt,
        )

    entry = state["tasks"][TASK_ID]
    assert entry["state"] == "BENCH_RUNNING"
    assert entry["state_data"]["pending_run_id"] == run_id_a
    # exactly one daemon_timeout for the current run
    cur_events = [
        rec for rec in entry["history"]
        if rec.get("kind") == "daemon_timeout"
           and rec.get("run_id") == run_id_a
    ]
    assert len(cur_events) == 1


# ---------------------------------------------------------------------------
# CLI flag wiring
# ---------------------------------------------------------------------------

def test_cli_default_tier_is_1(tmp_path):
    from automation.fix_orchestrator import _build_arg_parser

    parser = _build_arg_parser()
    args = parser.parse_args([
        "--resolve-log", str(tmp_path / "x.log"),
        "--reverify-results", str(tmp_path / "rv"),
    ])
    assert args.tier == 1
    assert args.queue_root is None


def test_cli_tier2_accepts_queue_root(tmp_path):
    from automation.fix_orchestrator import _build_arg_parser

    parser = _build_arg_parser()
    args = parser.parse_args([
        "--resolve-log", str(tmp_path / "x.log"),
        "--reverify-results", str(tmp_path / "rv"),
        "--tier", "2",
        "--queue-root", str(tmp_path / "q"),
        "--daemon-timeout-minutes", "30",
        "--daemon-timeout-max-attempts", "5",
    ])
    assert args.tier == 2
    assert args.queue_root == tmp_path / "q"
    assert args.daemon_timeout_minutes == 30
    assert args.daemon_timeout_max_attempts == 5


def test_cli_tier2_requires_queue_root(tmp_path):
    """``main()`` returns 2 (FATAL) when --tier 2 lacks --queue-root."""
    from automation.fix_orchestrator import main

    rc = main([
        "--resolve-log", str(tmp_path / "x.log"),
        "--reverify-results", str(tmp_path / "rv"),
        "--tier", "2",
        "--init-empty-state",
        "--state-root", str(tmp_path / "state"),
        "--repo-root", str(tmp_path / "repo"),
    ])
    assert rc == 2


# ---------------------------------------------------------------------------
# Backward-compat: tier=1 by default in OrchestratorContext
# ---------------------------------------------------------------------------

def test_orchestrator_context_default_is_tier1(tmp_path):
    from automation.fix_orchestrator import OrchestratorContext

    ctx = OrchestratorContext(
        repo_root=tmp_path / "repo",
        state_root=tmp_path / "state",
        reverify_results=tmp_path / "rv",
        resolve_log=tmp_path / "log",
    )
    assert ctx.tier == 1
    assert ctx.queue_root is None
    assert ctx.daemon_timeout_minutes == 60
    assert ctx.daemon_timeout_max_attempts == 3


def test_tier1_happy_path_unchanged_by_tier_field(tmp_path):
    """M1 regression: tier=1 default behaviour preserves the user-runnable
    ``tier1_reverify.py`` wrapper command."""
    from automation.fix_orchestrator import drive_task_once

    repo, state_root, reverify, _ = _scaffold_repo(tmp_path)
    from automation.fix_orchestrator import OrchestratorContext

    ctx = OrchestratorContext(
        repo_root=repo, state_root=state_root,
        reverify_results=reverify,
        resolve_log=repo / "results" / "gpu_pipeline" / "pa_resolve.log",
        verdict_class_filter="shapes-incomplete",
        # No explicit ``tier`` -> default 1.
    )
    state = {"schema_version": "1.0", "tasks": {}}
    chat = MagicMock(return_value=SimpleNamespace(content=_good_llm_response()))
    drive_task_once(
        ctx, state, TASK_ID, llm_chat=chat, prompt_builder=_fake_prompt,
    )
    entry = state["tasks"][TASK_ID]
    assert entry["state"] == "BENCH_RUNNING"
    run_id = entry["state_data"]["pending_run_id"]
    # Tier 1 mint prefix
    assert run_id.startswith("tier1-")
    # The wrapper command was queued for the operator (tier 1 marker).
    assert "_pending_cli" in entry
    assert any(
        "tier1_reverify.py" in line for line in entry["_pending_cli"]
    )


# ---------------------------------------------------------------------------
# Producer / tier consistency: result.json must declare the expected
# producer for the orchestrator's current tier. A mismatch (e.g. a
# tier1_reverify result landing in a Tier 2 run-dir, or a daemon write
# observed under Tier 1) is quarantined and never consumed.
# ---------------------------------------------------------------------------

def test_tier2_rejects_non_daemon_producer(tmp_path):
    """Under Tier 2, a result.json with ``producer != "daemon"`` MUST NOT
    be consumed -- the orchestrator quarantines the run-dir and the
    task stays at BENCH_RUNNING."""
    from automation import extract_verdict as ev
    from automation.fix_orchestrator import drive_task_once

    repo, state_root, reverify, queue_root = _scaffold_repo(tmp_path)
    ctx = _mk_ctx(repo, state_root, reverify, queue_root, tier=2)
    state = {"schema_version": "1.0", "tasks": {}}

    chat = MagicMock(return_value=SimpleNamespace(content=_good_llm_response()))
    drive_task_once(
        ctx, state, TASK_ID, llm_chat=chat, prompt_builder=_fake_prompt,
    )
    entry = state["tasks"][TASK_ID]
    run_id = entry["state_data"]["pending_run_id"]

    # Foreign writer: tier1_reverify writes into a Tier 2 run-dir.
    ev.write_verdict(
        task_id=TASK_ID, cmd="resolve", run_id=run_id,
        producer="tier1_reverify", out_dir=reverify,
        verdict="PASS", verdict_class=None, reason="foreign tier1 write",
    )

    drive_task_once(
        ctx, state, TASK_ID, llm_chat=chat, prompt_builder=_fake_prompt,
    )
    entry = state["tasks"][TASK_ID]
    # Not consumed: state stays BENCH_RUNNING; run-dir quarantined.
    assert entry["state"] == "BENCH_RUNNING"
    assert entry["state_data"]["pending_run_id"] == run_id
    assert not (reverify / run_id).exists()
    assert (reverify / (run_id + ".foreign")).is_dir()


def test_tier1_rejects_daemon_producer(tmp_path):
    """Symmetric guard: under Tier 1, a ``producer == "daemon"``
    result.json is foreign and MUST NOT be consumed."""
    from automation import extract_verdict as ev
    from automation.fix_orchestrator import OrchestratorContext, drive_task_once

    repo, state_root, reverify, _ = _scaffold_repo(tmp_path)
    ctx = OrchestratorContext(
        repo_root=repo, state_root=state_root,
        reverify_results=reverify,
        resolve_log=repo / "results" / "gpu_pipeline" / "pa_resolve.log",
        verdict_class_filter="shapes-incomplete",
        # Default tier=1.
    )
    state = {"schema_version": "1.0", "tasks": {}}

    chat = MagicMock(return_value=SimpleNamespace(content=_good_llm_response()))
    drive_task_once(
        ctx, state, TASK_ID, llm_chat=chat, prompt_builder=_fake_prompt,
    )
    entry = state["tasks"][TASK_ID]
    run_id = entry["state_data"]["pending_run_id"]

    # Foreign writer: a daemon writes into a Tier 1 run-dir.
    ev.write_verdict(
        task_id=TASK_ID, cmd="resolve", run_id=run_id,
        producer="daemon", out_dir=reverify,
        verdict="PASS", verdict_class=None, reason="foreign daemon write",
    )

    drive_task_once(
        ctx, state, TASK_ID, llm_chat=chat, prompt_builder=_fake_prompt,
    )
    entry = state["tasks"][TASK_ID]
    assert entry["state"] == "BENCH_RUNNING"
    assert entry["state_data"]["pending_run_id"] == run_id
    assert not (reverify / run_id).exists()
    assert (reverify / (run_id + ".foreign")).is_dir()
