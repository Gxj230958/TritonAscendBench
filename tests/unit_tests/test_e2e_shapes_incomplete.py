"""End-to-end M1 shapes-incomplete mock test (Unit 1.11).

Mocks LLM + tier1_reverify (by writing the result.json directly) and
verifies a single task transitions NEW -> DONE through the full
canonical state machine.
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import yaml

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


LEVEL = "L1"
TASK_ID = "L1_06_swiglu"


def _scaffold(tmp_path: Path) -> tuple[Path, Path, Path]:
    repo = tmp_path / "repo"
    repo.mkdir()
    td = repo / "tasks" / LEVEL / TASK_ID
    td.mkdir(parents=True)
    (td / "meta.yaml").write_text(
        yaml.safe_dump({"schema_version": "1.0", "name": "swiglu"}),
        encoding="utf-8",
    )
    (td / "shapes.yaml").write_text("public_shapes: []\n", encoding="utf-8")
    (td / "oracle_pytorch.py").write_text("def oracle(x):\n    return x\n", encoding="utf-8")
    state_root = repo / "state"
    state_root.mkdir()
    reverify = tmp_path / "reverify"
    reverify.mkdir()
    return repo, state_root, reverify


_META_GOOD = (
    '{"decision": "patch-shapes", '
    '"missing_params": ["M"], '
    '"rationale": "fill knobs", '
    '"evidence": {}}'
)
_META_BAD = (
    '{"decision": "patch-shapes", '
    '"missing_params": [], '
    '"rationale": "x", '
    '"evidence": {}}'
)


def _llm_good():
    body = (
        f"### META\n{_META_GOOD}\n### END\n\n"
        f"### FILE: tasks/{LEVEL}/{TASK_ID}/shapes.yaml\n"
        "public_shapes:\n  - {id: s1, name: small, knobs: {M: 16}}\n### END\n"
    )
    return SimpleNamespace(content=body)


def _llm_bad():
    body = (
        f"### META\n{_META_BAD}\n### END\n\n"
        f"### FILE: tasks/{LEVEL}/{TASK_ID}/meta.yaml\nschema_version: 1.0\n### END\n"
    )
    return SimpleNamespace(content=body)


def _fake_prompt(*a, **kw):
    return [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}]


def _mock_reverify(reverify_root: Path, run_id: str, *,
                   verdict: str, verdict_class=None, task_id: str = TASK_ID,
                   cmd: str = "resolve") -> None:
    from automation import extract_verdict as ev

    ev.write_verdict(
        task_id=task_id, cmd=cmd, run_id=run_id,
        producer="tier1_reverify", out_dir=reverify_root,
        verdict=verdict, verdict_class=verdict_class,
        reason="mocked tier1 reverify",
    )


def test_e2e_happy_new_to_done(tmp_path):
    from automation.fix_orchestrator import OrchestratorContext, drive_task_once

    repo, state_root, reverify = _scaffold(tmp_path)
    ctx = OrchestratorContext(
        repo_root=repo, state_root=state_root, reverify_results=reverify,
        resolve_log=repo / "x.log",
    )
    state = {"schema_version": "1.0", "tasks": {}}

    # Round 1: NEW -> ... -> BENCH_RUNNING
    chat = MagicMock(return_value=_llm_good())
    drive_task_once(ctx, state, TASK_ID, llm_chat=chat, prompt_builder=_fake_prompt)
    entry = state["tasks"][TASK_ID]
    assert entry["state"] == "BENCH_RUNNING"
    pending = entry["state_data"]["pending_run_id"]

    # User mock-reverify on GPU host
    _mock_reverify(reverify, pending, verdict="PASS")

    # Round 2: consume + DONE
    drive_task_once(ctx, state, TASK_ID, llm_chat=chat, prompt_builder=_fake_prompt)
    entry = state["tasks"][TASK_ID]
    assert entry["state"] == "DONE"
    assert entry["state_data"]["final_verdict"] == "PASS"

    # State machine touched all required milestones in history
    kinds = [r.get("kind") for r in entry["history"]]
    states_seen = [r.get("to") for r in entry["history"] if r.get("kind") == "transition"]
    assert "result_consumed" in kinds
    assert "QUEUED" in states_seen
    assert "PROMPTING" in states_seen
    assert "BENCH_RUNNING" in states_seen
    assert "DONE" in states_seen


def test_e2e_first_invalid_then_pass(tmp_path):
    """First LLM response fails validation; second succeeds; final DONE."""
    from automation.fix_orchestrator import OrchestratorContext, drive_task_once

    repo, state_root, reverify = _scaffold(tmp_path)
    ctx = OrchestratorContext(
        repo_root=repo, state_root=state_root, reverify_results=reverify,
        resolve_log=repo / "x.log",
        max_attempts=3,
    )
    state = {"schema_version": "1.0", "tasks": {}}

    # Round 1: bad response -> retry advance
    chat = MagicMock(return_value=_llm_bad())
    drive_task_once(ctx, state, TASK_ID, llm_chat=chat, prompt_builder=_fake_prompt)
    entry = state["tasks"][TASK_ID]
    assert entry["state"] != "DONE"
    assert entry["state"] != "BENCH_RUNNING"

    # Round 2: good response -> BENCH_RUNNING
    chat2 = MagicMock(return_value=_llm_good())
    drive_task_once(ctx, state, TASK_ID, llm_chat=chat2, prompt_builder=_fake_prompt)
    entry = state["tasks"][TASK_ID]
    # may go PROMPTING -> ... -> BENCH_RUNNING in this single step depending on
    # where Round-1 landed. If it's still in PROMPTING / ATTEMPTING_NEXT_ROUND,
    # we drive once more.
    if entry["state"] != "BENCH_RUNNING":
        drive_task_once(ctx, state, TASK_ID, llm_chat=chat2, prompt_builder=_fake_prompt)
        entry = state["tasks"][TASK_ID]
    assert entry["state"] == "BENCH_RUNNING"
    pending = entry["state_data"]["pending_run_id"]

    _mock_reverify(reverify, pending, verdict="PASS")
    drive_task_once(ctx, state, TASK_ID, llm_chat=chat2, prompt_builder=_fake_prompt)
    entry = state["tasks"][TASK_ID]
    assert entry["state"] == "DONE"


def test_e2e_blocked_same_class_retries_then_unfixable(tmp_path):
    from automation.fix_orchestrator import OrchestratorContext, drive_task_once

    repo, state_root, reverify = _scaffold(tmp_path)
    ctx = OrchestratorContext(
        repo_root=repo, state_root=state_root, reverify_results=reverify,
        resolve_log=repo / "x.log", max_attempts=2,
    )
    state = {"schema_version": "1.0", "tasks": {}}
    chat = MagicMock(return_value=_llm_good())

    # Loop until UNFIXABLE_BY_LLM or DONE.
    for _round_idx in range(6):
        drive_task_once(ctx, state, TASK_ID, llm_chat=chat, prompt_builder=_fake_prompt)
        entry = state["tasks"][TASK_ID]
        if entry["state"] == "BENCH_RUNNING":
            pending = entry["state_data"]["pending_run_id"]
            _mock_reverify(
                reverify, pending,
                verdict="BLOCKED", verdict_class="shapes-incomplete",
            )
        if entry["state"] == "UNFIXABLE_BY_LLM":
            break
    assert entry["state"] == "UNFIXABLE_BY_LLM"


def test_e2e_class_switch_blocks_for_m2(tmp_path):
    from automation.fix_orchestrator import OrchestratorContext, drive_task_once

    repo, state_root, reverify = _scaffold(tmp_path)
    ctx = OrchestratorContext(
        repo_root=repo, state_root=state_root, reverify_results=reverify,
        resolve_log=repo / "x.log",
    )
    state = {"schema_version": "1.0", "tasks": {}}
    chat = MagicMock(return_value=_llm_good())
    drive_task_once(ctx, state, TASK_ID, llm_chat=chat, prompt_builder=_fake_prompt)
    pending = state["tasks"][TASK_ID]["state_data"]["pending_run_id"]
    _mock_reverify(reverify, pending,
                   verdict="BLOCKED", verdict_class="oracle-mismatch")
    drive_task_once(ctx, state, TASK_ID, llm_chat=chat, prompt_builder=_fake_prompt)
    entry = state["tasks"][TASK_ID]
    # State machine moves to NEW with new class; M2 will pick up.
    assert entry["state"] == "NEW"
    assert entry["verdict_class"] == "oracle-mismatch"


def test_e2e_state_file_persists_round_trip(tmp_path):
    """After Round-1 we persist state file; Round-2 reads it back and
    drives the same task."""
    from automation.fix_orchestrator import (
        OrchestratorContext,
        drive_task_once,
        load_existing_triage_state,
        write_state,
    )

    repo, state_root, reverify = _scaffold(tmp_path)
    ctx = OrchestratorContext(
        repo_root=repo, state_root=state_root, reverify_results=reverify,
        resolve_log=repo / "x.log",
    )
    state_path = state_root / "triage_state.json"
    state = {"schema_version": "1.0", "tasks": {}}
    chat = MagicMock(return_value=_llm_good())
    drive_task_once(ctx, state, TASK_ID, llm_chat=chat, prompt_builder=_fake_prompt)
    write_state(state, state_path)
    # Round-2 from disk
    state2 = load_existing_triage_state(state_path)
    pending = state2["tasks"][TASK_ID]["state_data"]["pending_run_id"]
    _mock_reverify(reverify, pending, verdict="PASS")
    drive_task_once(ctx, state2, TASK_ID, llm_chat=chat, prompt_builder=_fake_prompt)
    assert state2["tasks"][TASK_ID]["state"] == "DONE"
