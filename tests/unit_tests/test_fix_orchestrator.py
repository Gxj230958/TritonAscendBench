"""Tests for ``automation.fix_orchestrator`` (Unit 1.9, M1 subset).

Mocks LLM + reverify result.json round-trip; verifies state machine
behaviour across the canonical 17-state enum.
"""
from __future__ import annotations

import json
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


# ---------------------------------------------------------------------------
# Helpers to scaffold a fake repo + state root
# ---------------------------------------------------------------------------

LEVEL = "L1"
TASK_ID = "L1_06_swiglu"


def _scaffold_repo(tmp_path: Path, task_id: str = TASK_ID) -> tuple[Path, Path, Path]:
    repo = tmp_path / "repo"
    repo.mkdir()
    td = repo / "tasks" / LEVEL / task_id
    td.mkdir(parents=True)
    (td / "meta.yaml").write_text("schema_version: 1.0\nname: x\n", encoding="utf-8")
    (td / "shapes.yaml").write_text("public_shapes: []\n", encoding="utf-8")
    (td / "oracle_pytorch.py").write_text("def oracle(x):\n    return x\n", encoding="utf-8")
    state_root = repo / "state"
    state_root.mkdir()
    reverify = tmp_path / "reverify_results"
    reverify.mkdir()
    return repo, state_root, reverify


def _good_llm_response(task_id: str = TASK_ID) -> str:
    new_shapes = "public_shapes:\n  - {id: s1, name: small, knobs: {M: 16}}\n"
    meta_payload = (
        '{"decision": "patch-shapes", '
        '"missing_params": ["x"], '
        '"rationale": "fill missing param", '
        '"evidence": {"missing": ["x"]}}'
    )
    return (
        f"### META\n{meta_payload}\n### END\n\n"
        f"### FILE: tasks/{LEVEL}/{task_id}/shapes.yaml\n{new_shapes}\n### END\n"
    )


def _bad_llm_response(task_id: str = TASK_ID) -> str:
    """Returns META + FILE that fails Layer 4 (decision=patch-shapes but
    only meta.yaml, no shapes.yaml)."""
    meta = (
        '{"decision": "patch-shapes", '
        '"missing_params": [], '
        '"rationale": "x", '
        '"evidence": {}}'
    )
    return (
        f"### META\n{meta}\n### END\n\n"
        f"### FILE: tasks/{LEVEL}/{task_id}/meta.yaml\nschema_version: 1.0\n### END\n"
    )


def _mk_ctx(repo, state_root, reverify, **overrides):
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
    }
    defaults.update(overrides)
    return OrchestratorContext(**defaults)


def _fake_prompt(*a, **kw):
    return [{"role": "system", "content": "sys"}, {"role": "user", "content": "u"}]


# ---------------------------------------------------------------------------
# State loaders
# ---------------------------------------------------------------------------

def test_load_or_init_empty_missing_default_raises(tmp_path):
    from automation.fix_orchestrator import (
        StateFileNotFoundError,
        load_or_init_empty_state,
    )

    with pytest.raises(StateFileNotFoundError):
        load_or_init_empty_state(tmp_path / "triage_state.json")


def test_load_or_init_empty_allow_creates(tmp_path):
    from automation.fix_orchestrator import load_or_init_empty_state

    state = load_or_init_empty_state(tmp_path / "triage_state.json", allow_empty=True)
    assert state == {"schema_version": "1.0", "tasks": {}}


def test_load_existing_strict_missing_raises(tmp_path):
    from automation.fix_orchestrator import (
        StateFileNotFoundError,
        load_existing_triage_state,
    )

    with pytest.raises(StateFileNotFoundError):
        load_existing_triage_state(tmp_path / "triage_state.json")


def test_load_existing_corrupt_quarantines(tmp_path):
    from automation.fix_orchestrator import (
        StateFileCorruptError,
        load_existing_triage_state,
    )

    sp = tmp_path / "triage_state.json"
    sp.write_text("not json at all", encoding="utf-8")
    with pytest.raises(StateFileCorruptError):
        load_existing_triage_state(sp)
    # original is renamed away
    assert not sp.is_file()
    backups = list(tmp_path.glob(".corrupt.*.bak"))
    assert len(backups) == 1


def test_load_existing_bad_schema_version_quarantines(tmp_path):
    from automation.fix_orchestrator import (
        StateFileCorruptError,
        load_existing_triage_state,
    )

    sp = tmp_path / "triage_state.json"
    sp.write_text(json.dumps({"schema_version": "9.9", "tasks": {}}), encoding="utf-8")
    with pytest.raises(StateFileCorruptError, match="schema_version"):
        load_existing_triage_state(sp)


def test_load_existing_missing_common_fields_quarantines(tmp_path):
    from automation.fix_orchestrator import (
        StateFileCorruptError,
        load_existing_triage_state,
    )

    sp = tmp_path / "triage_state.json"
    sp.write_text(json.dumps({
        "schema_version": "1.0",
        "tasks": {"t1": {"state": "NEW"}},  # missing many common fields
    }), encoding="utf-8")
    with pytest.raises(StateFileCorruptError):
        load_existing_triage_state(sp)


# ---------------------------------------------------------------------------
# parse_resolve_log
# ---------------------------------------------------------------------------

def test_parse_resolve_log_filters_classes(tmp_path):
    from automation.fix_orchestrator import parse_resolve_log

    log = tmp_path / "pa_resolve.log"
    log.write_text(
        "[T1] BLOCKED - :: BLOCKED(no-verified-upstream-entrypoint: shapes-incomplete)\n"
        "[T2] BLOCKED - :: BLOCKED(no-verified-upstream-entrypoint: oracle-mismatch)\n"
        "[T3] BLOCKED - :: BLOCKED(no-verified-upstream-entrypoint: brand-new-class)\n"
        "[T4] RESOLVED ok :: PASS\n",
        encoding="utf-8",
    )
    result = parse_resolve_log(log)
    assert ("T1", "shapes-incomplete") in result
    assert ("T2", "oracle-mismatch") in result
    assert ("T3", "unexpected-blocked") in result
    assert all(tid != "T4" for tid, _ in result)


def test_extract_missing_params_evidence_filters_to_task_and_class(tmp_path):
    from automation.fix_orchestrator import extract_missing_params_evidence

    log = tmp_path / "pa_resolve.log"
    log.write_text(
        "[T1] BLOCKED - :: BLOCKED(no-verified-upstream-entrypoint: "
        "shapes-incomplete): tried [a]; a: missing param 'M'\n"
        "[T2] BLOCKED - :: BLOCKED(no-verified-upstream-entrypoint: "
        "shapes-incomplete): tried [b]; b: missing param 'K'\n"
        "[T1] BLOCKED - :: BLOCKED(no-verified-upstream-entrypoint: "
        "oracle-mismatch): tried [c]; c: tolerance breach\n"
        "[T1] RESOLVED ok :: PASS\n",
        encoding="utf-8",
    )
    out = extract_missing_params_evidence(log, "T1")
    assert "missing param 'M'" in out
    # other task / non-shapes-incomplete lines are filtered out
    assert "missing param 'K'" not in out
    assert "tolerance breach" not in out


def test_extract_missing_params_evidence_handles_missing_log(tmp_path):
    from automation.fix_orchestrator import extract_missing_params_evidence

    out = extract_missing_params_evidence(tmp_path / "nope.log", "T1")
    assert out == ""


# ---------------------------------------------------------------------------
# State transition validator
# ---------------------------------------------------------------------------

def test_state_transition_missing_state_data_rejected():
    from automation import state_machine as sm

    entry = sm.new_task_entry("shapes-incomplete")
    sm.transition(entry, to="QUEUED", state_data={}, reason="")
    sm.transition(entry, to="PROMPTING",
                  state_data={"last_prompt_round_id": "r1"}, reason="")
    # BENCH_RUNNING requires pending_*; missing pending_cmd should reject
    with pytest.raises(sm.StateTransitionError):
        sm.transition(entry, to="PATCH_VALIDATING",
                      state_data={"last_validation_errors": []}, reason="")
        sm.transition(entry, to="BASELINE_SNAPSHOTTED",
                      state_data={"baseline_tar_path": "x", "manifest_path": "y"},
                      reason="")
        sm.transition(entry, to="BENCH_RUNNING",
                      state_data={"pending_run_id": "r"}, reason="")


def test_state_transition_unknown_state_rejected():
    from automation import state_machine as sm

    entry = sm.new_task_entry("shapes-incomplete")
    with pytest.raises(sm.StateTransitionError):
        sm.transition(entry, to="BOGUS", state_data={}, reason="")


def test_done_requires_valid_final_verdict():
    from automation import state_machine as sm

    entry = sm.new_task_entry("shapes-incomplete")
    # Set up at BENCH_RUNNING manually
    entry["state"] = "BENCH_RUNNING"
    entry["state_data"] = {
        "pending_run_id": "r1", "pending_cmd": "resolve",
        "pending_result_dir": "/tmp",
    }
    with pytest.raises(sm.StateTransitionError, match="final_verdict"):
        sm.transition(entry, to="DONE",
                      state_data={"final_verdict": "UNKNOWN"}, reason="")


# ---------------------------------------------------------------------------
# drive_task_once: full pipeline
# ---------------------------------------------------------------------------

def test_drive_task_happy_validate_apply_pending(tmp_path):
    from automation.fix_orchestrator import drive_task_once

    repo, state_root, reverify = _scaffold_repo(tmp_path)
    ctx = _mk_ctx(repo, state_root, reverify)
    state = {"schema_version": "1.0", "tasks": {}}

    chat = MagicMock(return_value=SimpleNamespace(content=_good_llm_response()))
    drive_task_once(ctx, state, TASK_ID,
                    llm_chat=chat, prompt_builder=_fake_prompt)

    entry = state["tasks"][TASK_ID]
    assert entry["state"] == "BENCH_RUNNING"
    sd = entry["state_data"]
    assert "pending_run_id" in sd and "pending_cmd" in sd and "pending_result_dir" in sd
    assert sd["pending_cmd"] == "resolve"
    # apply succeeded: shapes.yaml content was rewritten
    new_shapes = (repo / "tasks" / LEVEL / TASK_ID / "shapes.yaml").read_text()
    assert "small" in new_shapes


def test_drive_task_consume_pass_done(tmp_path):
    from automation import extract_verdict as ev
    from automation.fix_orchestrator import drive_task_once

    repo, state_root, reverify = _scaffold_repo(tmp_path)
    ctx = _mk_ctx(repo, state_root, reverify)
    state = {"schema_version": "1.0", "tasks": {}}

    chat = MagicMock(return_value=SimpleNamespace(content=_good_llm_response()))
    drive_task_once(ctx, state, TASK_ID, llm_chat=chat, prompt_builder=_fake_prompt)
    entry = state["tasks"][TASK_ID]
    pending_run = entry["state_data"]["pending_run_id"]

    # Mock reverify result.json
    ev.write_verdict(
        task_id=TASK_ID, cmd="resolve", run_id=pending_run,
        producer="tier1_reverify", out_dir=reverify,
        verdict="PASS", verdict_class=None,
        reason="resolver PASS",
    )

    drive_task_once(ctx, state, TASK_ID, llm_chat=chat, prompt_builder=_fake_prompt)
    entry = state["tasks"][TASK_ID]
    assert entry["state"] == "DONE"
    assert entry["state_data"]["final_verdict"] == "PASS"
    # Common audit fields populated
    assert entry["last_consumed_verdict"] == "PASS"
    assert entry["last_consumed_run_id"] == pending_run


def test_drive_task_stale_run_id_ignored(tmp_path):
    from automation import extract_verdict as ev
    from automation.fix_orchestrator import drive_task_once

    repo, state_root, reverify = _scaffold_repo(tmp_path)
    ctx = _mk_ctx(repo, state_root, reverify)
    state = {"schema_version": "1.0", "tasks": {}}
    chat = MagicMock(return_value=SimpleNamespace(content=_good_llm_response()))
    drive_task_once(ctx, state, TASK_ID, llm_chat=chat, prompt_builder=_fake_prompt)
    entry = state["tasks"][TASK_ID]
    pending_run = entry["state_data"]["pending_run_id"]
    # Write a result.json with a different run_id
    ev.write_verdict(
        task_id=TASK_ID, cmd="resolve", run_id=pending_run,
        producer="tier1_reverify", out_dir=reverify,
        verdict="PASS", verdict_class=None, reason="ok",
    )
    # Mutate the on-disk file to mismatch -- simulate stale
    rd = reverify / pending_run
    result = json.loads((rd / "result.json").read_text())
    result["run_id"] = "DIFFERENT_RUN_ID"
    (rd / "result.json").write_text(json.dumps(result), encoding="utf-8")

    drive_task_once(ctx, state, TASK_ID, llm_chat=chat, prompt_builder=_fake_prompt)
    # Still BENCH_RUNNING, pending preserved
    entry = state["tasks"][TASK_ID]
    assert entry["state"] == "BENCH_RUNNING"
    assert entry["state_data"]["pending_run_id"] == pending_run


def test_drive_task_validate_fail_retries(tmp_path):
    from automation.fix_orchestrator import drive_task_once

    repo, state_root, reverify = _scaffold_repo(tmp_path)
    ctx = _mk_ctx(repo, state_root, reverify, max_attempts=3)
    state = {"schema_version": "1.0", "tasks": {}}
    chat = MagicMock(return_value=SimpleNamespace(content=_bad_llm_response()))
    drive_task_once(ctx, state, TASK_ID, llm_chat=chat, prompt_builder=_fake_prompt)
    entry = state["tasks"][TASK_ID]
    # After one validate fail attempt+1, we should be back at PROMPTING
    # (or onwards, depending on exact path). Not DONE.
    assert entry["state"] in ("PROMPTING", "ATTEMPTING_NEXT_ROUND", "PATCH_VALIDATING")
    assert entry["attempt"] >= 1
    # task_dir unchanged
    shapes = (repo / "tasks" / LEVEL / TASK_ID / "shapes.yaml").read_text()
    assert shapes == "public_shapes: []\n"


def test_drive_task_llm_transient_error(tmp_path):
    from automation.fix_orchestrator import drive_task_once

    repo, state_root, reverify = _scaffold_repo(tmp_path)
    ctx = _mk_ctx(repo, state_root, reverify)
    state = {"schema_version": "1.0", "tasks": {}}

    def explode(messages):
        raise RuntimeError("llm 503 simulated")

    drive_task_once(ctx, state, TASK_ID, llm_chat=explode, prompt_builder=_fake_prompt)
    entry = state["tasks"][TASK_ID]
    assert entry["state"] == "LLM_TRANSIENT_ERROR"
    assert entry["state_data"]["last_error_class"] == "RuntimeError"


def test_drive_task_class_switch_re_enters_new(tmp_path):
    from automation import extract_verdict as ev
    from automation.fix_orchestrator import drive_task_once

    repo, state_root, reverify = _scaffold_repo(tmp_path)
    ctx = _mk_ctx(repo, state_root, reverify)
    state = {"schema_version": "1.0", "tasks": {}}
    chat = MagicMock(return_value=SimpleNamespace(content=_good_llm_response()))
    drive_task_once(ctx, state, TASK_ID, llm_chat=chat, prompt_builder=_fake_prompt)
    entry = state["tasks"][TASK_ID]
    pending_run = entry["state_data"]["pending_run_id"]

    # Reverify returns BLOCKED with a DIFFERENT class
    ev.write_verdict(
        task_id=TASK_ID, cmd="resolve", run_id=pending_run,
        producer="tier1_reverify", out_dir=reverify,
        verdict="BLOCKED", verdict_class="oracle-mismatch",
        reason="now oracle-mismatch",
    )
    drive_task_once(ctx, state, TASK_ID, llm_chat=chat, prompt_builder=_fake_prompt)
    entry = state["tasks"][TASK_ID]
    assert entry["state"] == "NEW"
    assert entry["verdict_class"] == "oracle-mismatch"


def test_drive_task_apply_fails_rollback_attempt_advances(tmp_path, monkeypatch):
    from automation.fix_orchestrator import drive_task_once
    from automation.lib import file_blocks

    repo, state_root, reverify = _scaffold_repo(tmp_path)
    ctx = _mk_ctx(repo, state_root, reverify, max_attempts=3)
    state = {"schema_version": "1.0", "tasks": {}}
    chat = MagicMock(return_value=SimpleNamespace(content=_good_llm_response()))

    def boom_apply(*a, **kw):
        raise OSError("simulated disk full")

    monkeypatch.setattr(file_blocks, "apply_patch", boom_apply)
    drive_task_once(ctx, state, TASK_ID, llm_chat=chat, prompt_builder=_fake_prompt)
    entry = state["tasks"][TASK_ID]
    # apply_patch failed, attempt advanced, state moved to next-round
    assert entry["state"] == "ATTEMPTING_NEXT_ROUND"
    assert entry["attempt"] == 1
    # rollback event recorded
    kinds = [r.get("kind") for r in entry["history"]]
    assert "rollback" in kinds


def test_drive_task_apply_fails_budget_exhausted_unfixable(tmp_path, monkeypatch):
    from automation.fix_orchestrator import drive_task_once
    from automation.lib import file_blocks

    repo, state_root, reverify = _scaffold_repo(tmp_path)
    ctx = _mk_ctx(repo, state_root, reverify, max_attempts=1)
    state = {"schema_version": "1.0", "tasks": {}}
    chat = MagicMock(return_value=SimpleNamespace(content=_good_llm_response()))

    def boom_apply(*a, **kw):
        raise OSError("disk full")

    monkeypatch.setattr(file_blocks, "apply_patch", boom_apply)
    drive_task_once(ctx, state, TASK_ID, llm_chat=chat, prompt_builder=_fake_prompt)
    entry = state["tasks"][TASK_ID]
    assert entry["state"] == "UNFIXABLE_BY_LLM"
    assert entry["state_data"]["attempts_used"] == 1


def test_apply_release_scope_exclusion_writes_meta(tmp_path):
    from automation.fix_orchestrator import apply_release_scope_exclusion

    repo, state_root, _ = _scaffold_repo(tmp_path)
    apply_release_scope_exclusion(
        TASK_ID, reason="unfixable by LLM", excluded_by="operator-x",
        repo_root=repo, state_root=state_root,
    )
    meta = yaml.safe_load(
        (repo / "tasks" / LEVEL / TASK_ID / "meta.yaml").read_text(encoding="utf-8")
    )
    assert meta["release_scope"]["excluded"] is True
    assert meta["release_scope"]["reason"] == "unfixable by LLM"
    assert meta["release_scope"]["excluded_by"] == "operator-x"
    # snapshot baseline exists
    assert (repo / "patches" / TASK_ID / "release_scope_baseline" /
            "baseline.tar.gz").is_file()


def test_apply_release_scope_exclusion_rollback_on_failure(tmp_path, monkeypatch):
    from automation.fix_orchestrator import apply_release_scope_exclusion
    from automation.lib import atomic

    repo, state_root, _ = _scaffold_repo(tmp_path)
    original_meta = (repo / "tasks" / LEVEL / TASK_ID / "meta.yaml").read_text()

    def boom(*a, **kw):
        raise OSError("disk full")

    monkeypatch.setattr(atomic, "atomic_write_yaml", boom)
    with pytest.raises(OSError):
        apply_release_scope_exclusion(
            TASK_ID, reason="x", excluded_by="op", repo_root=repo,
            state_root=state_root,
        )
    # Rollback restored original meta
    assert (repo / "tasks" / LEVEL / TASK_ID / "meta.yaml").read_text() == original_meta


def test_dry_run_does_not_advance_to_bench_running(tmp_path):
    from automation.fix_orchestrator import drive_task_once

    repo, state_root, reverify = _scaffold_repo(tmp_path)
    ctx = _mk_ctx(repo, state_root, reverify, dry_run=True)
    state = {"schema_version": "1.0", "tasks": {}}
    chat = MagicMock(return_value=SimpleNamespace(content=_good_llm_response()))
    drive_task_once(ctx, state, TASK_ID, llm_chat=chat, prompt_builder=_fake_prompt)
    entry = state["tasks"][TASK_ID]
    assert entry["state"] != "BENCH_RUNNING"
    # task_dir not modified
    shapes = (repo / "tasks" / LEVEL / TASK_ID / "shapes.yaml").read_text()
    assert shapes == "public_shapes: []\n"


def test_terminal_states_are_no_op(tmp_path):
    from automation import state_machine as sm
    from automation.fix_orchestrator import drive_task_once

    repo, state_root, reverify = _scaffold_repo(tmp_path)
    ctx = _mk_ctx(repo, state_root, reverify)
    entry = sm.new_task_entry("shapes-incomplete")
    entry["state"] = "DONE"
    entry["state_data"] = {"final_verdict": "PASS"}
    state = {"schema_version": "1.0", "tasks": {TASK_ID: entry}}

    drive_task_once(ctx, state, TASK_ID,
                    llm_chat=MagicMock(), prompt_builder=_fake_prompt)
    # Still DONE
    assert state["tasks"][TASK_ID]["state"] == "DONE"


def test_write_state_round_trip(tmp_path):
    from automation import state_machine as sm
    from automation.fix_orchestrator import (
        load_existing_triage_state,
        write_state,
    )

    entry = sm.new_task_entry("shapes-incomplete")
    sm.transition(entry, to="QUEUED", state_data={}, reason="x")
    state = {"schema_version": "1.0", "tasks": {TASK_ID: entry}}
    sp = tmp_path / "triage_state.json"
    write_state(state, sp)
    loaded = load_existing_triage_state(sp)
    assert loaded["tasks"][TASK_ID]["state"] == "QUEUED"


def test_canonical_state_set_is_17():
    from automation import state_machine as sm
    assert len(sm.CANONICAL_STATES) == 17


def test_cli_apply_and_dry_run_are_mutually_exclusive():
    """--apply and --dry-run cannot both appear on the command line."""
    from automation.fix_orchestrator import _build_arg_parser

    parser = _build_arg_parser()
    base = [
        "--resolve-log", "x.log",
        "--reverify-results", "/tmp/r",
    ]
    with pytest.raises(SystemExit):
        parser.parse_args([*base, "--apply", "--dry-run"])


def test_preview_path_without_apply_does_not_modify_task_dir(tmp_path):
    """When ctx.apply is False the orchestrator must NOT snapshot the
    task_dir, MUST NOT advance past PATCH_VALIDATING, and MUST leave
    the on-disk task files untouched. State is kept in a re-entrant
    PROMPTING state so the next sweep can re-prompt cleanly."""
    from automation.fix_orchestrator import drive_task_once

    repo, state_root, reverify = _scaffold_repo(tmp_path)
    ctx = _mk_ctx(repo, state_root, reverify, apply=False)
    state = {"schema_version": "1.0", "tasks": {}}
    chat = MagicMock(return_value=SimpleNamespace(content=_good_llm_response()))
    drive_task_once(ctx, state, TASK_ID,
                    llm_chat=chat, prompt_builder=_fake_prompt)
    entry = state["tasks"][TASK_ID]
    assert entry["state"] != "BENCH_RUNNING"
    assert entry["state"] != "BASELINE_SNAPSHOTTED"
    # task_dir untouched
    shapes = (repo / "tasks" / LEVEL / TASK_ID / "shapes.yaml").read_text()
    assert shapes == "public_shapes: []\n"
    # preview was written for operator review
    preview = repo / "patches" / TASK_ID / "round-00" / "preview"
    assert preview.is_dir()
    assert (preview / "shapes.yaml").is_file()


def test_consume_pending_requires_done_barrier(tmp_path):
    """A result.json without a DONE marker MUST NOT be consumed --
    the writer guarantees DONE is the last marker written so its
    absence means a partial write may be in flight.
    """
    from automation.fix_orchestrator import drive_task_once

    repo, state_root, reverify = _scaffold_repo(tmp_path)
    ctx = _mk_ctx(repo, state_root, reverify)
    state = {"schema_version": "1.0", "tasks": {}}
    chat = MagicMock(return_value=SimpleNamespace(content=_good_llm_response()))
    drive_task_once(ctx, state, TASK_ID, llm_chat=chat, prompt_builder=_fake_prompt)
    entry = state["tasks"][TASK_ID]
    pending_run = entry["state_data"]["pending_run_id"]

    # Write a result.json + _provenance.json but NO DONE marker.
    rd = reverify / pending_run
    rd.mkdir(parents=True, exist_ok=True)
    (rd / "_provenance.json").write_text("{}", encoding="utf-8")
    (rd / "result.json").write_text(
        json.dumps({
            "schema_version": "1.0", "run_id": pending_run,
            "task_id": TASK_ID, "cmd": "resolve", "verdict": "PASS",
            "verdict_class": None, "reason": "x",
            "produced_at": "2026-01-01T00:00:00Z",
            "producer": "tier1_reverify",
        }),
        encoding="utf-8",
    )
    drive_task_once(ctx, state, TASK_ID, llm_chat=chat, prompt_builder=_fake_prompt)
    # Without DONE we MUST stay in BENCH_RUNNING (do not consume).
    assert state["tasks"][TASK_ID]["state"] == "BENCH_RUNNING"

    # Touch DONE -> now consume succeeds.
    (rd / "DONE").touch()
    drive_task_once(ctx, state, TASK_ID, llm_chat=chat, prompt_builder=_fake_prompt)
    assert state["tasks"][TASK_ID]["state"] == "DONE"


def test_validate_task_entry_requires_state_data(tmp_path):
    """A persisted BENCH_RUNNING entry that lacks pending_cmd must be
    rejected as corrupt -- not silently re-loaded."""
    from automation import state_machine as sm
    from automation.fix_orchestrator import (
        StateFileCorruptError,
        load_existing_triage_state,
    )

    bad_entry = sm.new_task_entry("shapes-incomplete")
    bad_entry["state"] = "BENCH_RUNNING"
    # Drop pending_cmd / pending_result_dir; only pending_run_id present.
    bad_entry["state_data"] = {"pending_run_id": "r1"}
    sp = tmp_path / "triage_state.json"
    sp.write_text(json.dumps({
        "schema_version": "1.0", "tasks": {TASK_ID: bad_entry},
    }), encoding="utf-8")
    with pytest.raises(StateFileCorruptError, match="pending_cmd"):
        load_existing_triage_state(sp)


def test_unexpected_blocked_goes_to_backlog_not_new(tmp_path):
    """``unexpected-blocked`` is the resolver's fallback bucket. It
    must route to BACKLOG_UNEXPECTED, NOT re-enter NEW via the
    canonical-class switch path."""
    from automation import extract_verdict as ev
    from automation.fix_orchestrator import drive_task_once

    repo, state_root, reverify = _scaffold_repo(tmp_path)
    ctx = _mk_ctx(repo, state_root, reverify)
    state = {"schema_version": "1.0", "tasks": {}}
    chat = MagicMock(return_value=SimpleNamespace(content=_good_llm_response()))
    drive_task_once(ctx, state, TASK_ID, llm_chat=chat, prompt_builder=_fake_prompt)
    entry = state["tasks"][TASK_ID]
    pending_run = entry["state_data"]["pending_run_id"]
    ev.write_verdict(
        task_id=TASK_ID, cmd="resolve", run_id=pending_run,
        producer="tier1_reverify", out_dir=reverify,
        verdict="BLOCKED", verdict_class="unexpected-blocked",
        reason="resolver fallback",
    )
    drive_task_once(ctx, state, TASK_ID, llm_chat=chat, prompt_builder=_fake_prompt)
    entry = state["tasks"][TASK_ID]
    assert entry["state"] == "BACKLOG_UNEXPECTED"
    assert entry["state_data"]["raw_verdict_class"] == "unexpected-blocked"
