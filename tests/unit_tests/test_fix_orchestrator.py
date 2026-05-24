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


def test_task_ids_filter_overrides_max_tasks():
    """``--task-ids`` is an explicit allowlist: ``--max-tasks`` is
    ignored, every listed id is enqueued regardless of slice size."""
    from automation.fix_orchestrator import _enqueue_from_resolve_log

    parsed = [
        ("t1", "shapes-incomplete"),
        ("t2", "shapes-incomplete"),
        ("t3", "oracle-mismatch"),
        ("t4", "shapes-incomplete"),
        ("t5", "oracle-mismatch"),
    ]
    state = {"schema_version": "1.0", "tasks": {}}
    _enqueue_from_resolve_log(
        state, parsed,
        task_ids_csv="t1,t3,t5",
        max_tasks=2,                       # would have sliced to t1, t2
        allowed_enqueue={"shapes-incomplete"},  # would have dropped t3/t5
    )
    # All three explicit ids enqueued; max_tasks=2 ignored; class filter
    # bypassed (per the task_ids "explicit white-list" semantics).
    assert set(state["tasks"]) == {"t1", "t3", "t5"}
    # And each entry inherits the verdict_class from the resolve log.
    assert state["tasks"]["t3"]["verdict_class"] == "oracle-mismatch"
    assert state["tasks"]["t5"]["verdict_class"] == "oracle-mismatch"
    assert state["tasks"]["t1"]["verdict_class"] == "shapes-incomplete"


def test_task_ids_filter_skips_ids_not_in_resolve_log():
    """ids absent from the resolve log must be silently skipped, not
    crash the orchestrator (the next sweep will surface them)."""
    from automation.fix_orchestrator import _enqueue_from_resolve_log

    parsed = [("t1", "shapes-incomplete")]
    state = {"schema_version": "1.0", "tasks": {}}
    _enqueue_from_resolve_log(
        state, parsed,
        task_ids_csv="t1,t99,",   # t99 missing; trailing empty token ignored
        max_tasks=30,
        allowed_enqueue={"shapes-incomplete"},
    )
    assert set(state["tasks"]) == {"t1"}


def test_no_task_ids_falls_back_to_max_tasks_and_class_filter():
    """Without ``--task-ids`` the historic behaviour (max_tasks slice
    plus verdict-class enqueue filter) must still apply."""
    from automation.fix_orchestrator import _enqueue_from_resolve_log

    parsed = [
        ("t1", "shapes-incomplete"),
        ("t2", "oracle-mismatch"),
        ("t3", "shapes-incomplete"),
        ("t4", "unexpected-blocked"),
    ]
    state = {"schema_version": "1.0", "tasks": {}}
    _enqueue_from_resolve_log(
        state, parsed,
        task_ids_csv=None,
        max_tasks=3,
        allowed_enqueue={"shapes-incomplete"},
    )
    # Slice is parsed[:3] -> (t1, t2, t3); class filter keeps only
    # shapes-incomplete -> {t1, t3}.
    assert set(state["tasks"]) == {"t1", "t3"}


def test_task_ids_cli_flag_round_trip():
    """The CLI must accept ``--task-ids`` and surface it as
    ``args.task_ids`` for the enqueue helper."""
    from automation.fix_orchestrator import _build_arg_parser

    parser = _build_arg_parser()
    args = parser.parse_args([
        "--resolve-log", "x.log",
        "--reverify-results", "/tmp/r",
        "--task-ids", "alpha,beta,gamma",
    ])
    assert args.task_ids == "alpha,beta,gamma"

    # Default (omitted) is None so the legacy slice path stays in
    # effect.
    default_args = parser.parse_args([
        "--resolve-log", "x.log",
        "--reverify-results", "/tmp/r",
    ])
    assert default_args.task_ids is None


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


# ---------------------------------------------------------------------------
# LLM transport CLI flags (--llm-transport / --llm-ssh-host / --llm-cli-cmd)
# ---------------------------------------------------------------------------

def test_cli_llm_transport_default_http():
    """Omitting --llm-transport defaults to 'http' for back-compat."""
    from automation.fix_orchestrator import _build_arg_parser

    parser = _build_arg_parser()
    args = parser.parse_args([
        "--resolve-log", "x.log",
        "--reverify-results", "/tmp/r",
    ])
    assert args.llm_transport == "http"
    # ssh-cli specific flags default to None.
    assert args.llm_ssh_host is None
    assert args.llm_cli_cmd is None
    # Per-call timeout has a sensible default.
    assert args.llm_timeout_seconds == 600


def test_cli_llm_transport_accepts_ssh_cli():
    """--llm-transport ssh-cli + --llm-ssh-host + --llm-cli-cmd round-trip."""
    from automation.fix_orchestrator import _build_arg_parser

    parser = _build_arg_parser()
    args = parser.parse_args([
        "--resolve-log", "x.log",
        "--reverify-results", "/tmp/r",
        "--llm-transport", "ssh-cli",
        "--llm-ssh-host", "user@c-host",
        "--llm-cli-cmd", "some-cli exec {prompt}",
    ])
    assert args.llm_transport == "ssh-cli"
    assert args.llm_ssh_host == "user@c-host"
    assert args.llm_cli_cmd == "some-cli exec {prompt}"


def test_cli_llm_transport_rejects_unknown_value():
    """argparse choices=('http','ssh-cli') rejects anything else."""
    from automation.fix_orchestrator import _build_arg_parser

    parser = _build_arg_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([
            "--resolve-log", "x.log",
            "--reverify-results", "/tmp/r",
            "--llm-transport", "bogus",
        ])


def test_cli_llm_timeout_seconds_round_trip():
    from automation.fix_orchestrator import _build_arg_parser

    parser = _build_arg_parser()
    args = parser.parse_args([
        "--resolve-log", "x.log",
        "--reverify-results", "/tmp/r",
        "--llm-timeout-seconds", "1200",
    ])
    assert args.llm_timeout_seconds == 1200


def test_main_ssh_cli_missing_host_fails_loud(tmp_path, capsys):
    """transport=ssh-cli + missing --llm-ssh-host -> rc=2 + FATAL line."""
    from automation.fix_orchestrator import main

    state_root = tmp_path / "state"
    resolve_log = tmp_path / "resolve.log"
    resolve_log.write_text("[T1] BLOCKED - :: shapes-incomplete\n",
                           encoding="utf-8")
    rc = main([
        "--resolve-log", str(resolve_log),
        "--reverify-results", str(tmp_path / "results"),
        "--state-root", str(state_root),
        "--init-empty-state",
        "--llm-transport", "ssh-cli",
        # deliberately omit --llm-ssh-host
        "--llm-cli-cmd", "some-cli exec {prompt}",
    ])
    captured = capsys.readouterr()
    assert rc == 2
    assert "FATAL" in captured.err
    assert "llm-ssh-host" in captured.err


def test_main_ssh_cli_missing_cli_cmd_fails_loud(tmp_path, capsys):
    """transport=ssh-cli + missing --llm-cli-cmd -> rc=2 + FATAL line."""
    from automation.fix_orchestrator import main

    state_root = tmp_path / "state"
    resolve_log = tmp_path / "resolve.log"
    resolve_log.write_text("[T1] BLOCKED - :: shapes-incomplete\n",
                           encoding="utf-8")
    rc = main([
        "--resolve-log", str(resolve_log),
        "--reverify-results", str(tmp_path / "results"),
        "--state-root", str(state_root),
        "--init-empty-state",
        "--llm-transport", "ssh-cli",
        "--llm-ssh-host", "user@c-host",
        # deliberately omit --llm-cli-cmd
    ])
    captured = capsys.readouterr()
    assert rc == 2
    assert "FATAL" in captured.err
    assert "llm-cli-cmd" in captured.err


def test_main_http_missing_endpoint_still_fails_loud(tmp_path, capsys):
    """Back-compat: transport=http + missing --llm-endpoint -> rc=2."""
    from automation.fix_orchestrator import main

    state_root = tmp_path / "state"
    resolve_log = tmp_path / "resolve.log"
    resolve_log.write_text("[T1] BLOCKED - :: shapes-incomplete\n",
                           encoding="utf-8")
    rc = main([
        "--resolve-log", str(resolve_log),
        "--reverify-results", str(tmp_path / "results"),
        "--state-root", str(state_root),
        "--init-empty-state",
        # Default transport=http; no --llm-endpoint supplied.
    ])
    captured = capsys.readouterr()
    assert rc == 2
    assert "FATAL" in captured.err
    assert "llm-endpoint" in captured.err


def test_main_catches_value_error_from_factory(tmp_path):
    """Bad ssh-cli template -> rc=2 + FATAL line, NO Python traceback.

    Runs ``fix_orchestrator`` as a subprocess so we can observe the
    full stderr that an operator would see. A missing try/except
    around ``create_llm_client(...)`` would let
    ``SSHCLIClient.__init__`` ValueError propagate out of main(), and
    Python's default ``sys.excepthook`` would print
    ``Traceback (most recent call last):`` followed by the stack. The
    fix wraps the factory call so the same misconfiguration surfaces
    as ``FATAL: invalid LLM client configuration: ...`` on stderr
    with rc=2, matching the existing required-arg validation style.
    """
    import subprocess

    state_root = tmp_path / "state"
    resolve_log = tmp_path / "resolve.log"
    resolve_log.write_text("[T1] BLOCKED - :: shapes-incomplete\n",
                           encoding="utf-8")
    # Outer double quotes around {prompt} -- rejected by the
    # quote-depth check in SSHCLIClient.__init__.
    bad_template = 'cli exec "prefix {prompt} suffix"'
    proc = subprocess.run(
        [
            sys.executable, "-m", "automation.fix_orchestrator",
            "--resolve-log", str(resolve_log),
            "--reverify-results", str(tmp_path / "results"),
            "--state-root", str(state_root),
            "--init-empty-state",
            "--llm-transport", "ssh-cli",
            "--llm-ssh-host", "user@c-host",
            "--llm-cli-cmd", bad_template,
        ],
        env={
            "PYTHONPATH": str(_SCRIPTS),
            "PATH": __import__("os").environ.get("PATH", ""),
        },
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert proc.returncode == 2, (
        f"expected rc=2, got {proc.returncode}; "
        f"stdout={proc.stdout!r} stderr={proc.stderr!r}"
    )
    assert "FATAL" in proc.stderr, proc.stderr
    assert (
        "invalid LLM client configuration" in proc.stderr
        or "do NOT wrap" in proc.stderr
    ), proc.stderr
    # The whole point of the fix: no raw Python traceback leaks to
    # the operator. ``main()`` must catch ValueError and translate
    # it into the FATAL line.
    assert "Traceback (most recent call last)" not in proc.stderr, proc.stderr


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


# ---------------------------------------------------------------------------
# Unit 2.8: three-class verdict dispatch + state machine closure
# ---------------------------------------------------------------------------

def _scaffold_oracle_mismatch_repo(tmp_path: Path):
    """Scaffold a task tree + a vendored wrapper signature stub so the
    oracle_mismatch prompt builder can succeed (it requires a tasks/
    tree but tolerates absent vendored sources)."""
    repo, state_root, reverify = _scaffold_repo(tmp_path)
    # Make a fake pa_resolve.log line so the evidence extractor returns
    # something realistic.
    log_dir = repo / "results" / "gpu_pipeline"
    log_dir.mkdir(parents=True, exist_ok=True)
    log = log_dir / "pa_resolve.log"
    log.write_text(
        f"[{TASK_ID}] BLOCKED - :: BLOCKED(no-verified-upstream-entrypoint: "
        "oracle-mismatch): tried [a]; a: max_rel_diff=2.3e-1 exceeds tol\n",
        encoding="utf-8",
    )
    return repo, state_root, reverify, log


def _scaffold_defer_repo(tmp_path: Path):
    """Scaffold a tree for upstream-not-invocable-here -- includes the
    resolver hint line so ``extract_upstream_callable_hint`` returns a
    canonical (module, function) pair."""
    repo, state_root, reverify = _scaffold_repo(tmp_path)
    log_dir = repo / "results" / "gpu_pipeline"
    log_dir.mkdir(parents=True, exist_ok=True)
    log = log_dir / "pa_resolve.log"
    log.write_text(
        f"[{TASK_ID}] BLOCKED - :: BLOCKED(no-verified-upstream-entrypoint: "
        "upstream-not-invocable-here): tried [a]; a: module='vllm.layers' "
        "function=silu requires real CUDA runtime\n",
        encoding="utf-8",
    )
    return repo, state_root, reverify, log


def test_verdict_dispatch_shapes_incomplete(tmp_path):
    """When entry.verdict_class == 'shapes-incomplete' the orchestrator
    calls the shapes_incomplete builder (override + dispatcher both)."""
    from automation.fix_orchestrator import _VERDICT_TO_PROMPT
    from automation.prompts import shapes_incomplete

    assert _VERDICT_TO_PROMPT["shapes-incomplete"] is shapes_incomplete.build_prompt


def test_verdict_dispatch_oracle_mismatch(tmp_path):
    """oracle-mismatch is wired to the oracle_mismatch builder."""
    from automation.fix_orchestrator import _VERDICT_TO_PROMPT
    from automation.prompts import oracle_mismatch

    assert _VERDICT_TO_PROMPT["oracle-mismatch"] is oracle_mismatch.build_prompt


def test_verdict_dispatch_upstream_not_invocable(tmp_path):
    """upstream-not-invocable-here is wired to the defer_to_golden builder."""
    from automation.fix_orchestrator import _VERDICT_TO_PROMPT
    from automation.prompts import defer_to_golden

    assert _VERDICT_TO_PROMPT["upstream-not-invocable-here"] is defer_to_golden.build_prompt


def test_verdict_unexpected_blocked_to_backlog(tmp_path):
    """A NEW-state ``unexpected-blocked`` entry must NOT call the LLM
    patch loop (regression for M1 fix #5). The driver short-circuits
    before any prompt is built; the entry stays at NEW until the
    consume path routes it to BACKLOG_UNEXPECTED."""
    from automation import state_machine as sm
    from automation.fix_orchestrator import drive_task_once

    repo, state_root, reverify = _scaffold_repo(tmp_path)
    ctx = _mk_ctx(repo, state_root, reverify,
                  verdict_class_filter=None)
    entry = sm.new_task_entry("unexpected-blocked")
    state = {"schema_version": "1.0", "tasks": {TASK_ID: entry}}
    chat = MagicMock(side_effect=AssertionError("LLM must NOT be called"))
    drive_task_once(ctx, state, TASK_ID, llm_chat=chat, prompt_builder=None)
    # Entry untouched -- still NEW, no LLM call recorded.
    assert state["tasks"][TASK_ID]["state"] == "NEW"
    chat.assert_not_called()


def _meta_payload(decision: str, **extra) -> str:
    base = {
        "decision": decision,
        "rationale": "auto-generated",
        "evidence": {"src": "test"},
    }
    base.update(extra)
    return json.dumps(base)


_DEFER_META_YAML = """\
schema_version: 1.0
name: swiglu
oracle_simplified: upstream_oracle_required
phase_a_prime:
  skip_invocation: true
  status: SKIPPED_BY_DESIGN
  reason: requires real CUDA runtime
phase_b_capture_required: true
phase_b_capture:
  status: PENDING
  upstream_callable:
    module: vllm.layers
    function: silu
"""


def _good_tolerance_bump_response(extra_meta: str = "") -> str:
    body_meta = _meta_payload(
        "tolerance-bump",
        tolerance_delta={"old": "elementwise", "new": "reduction"},
    )
    return (
        f"### META\n{body_meta}\n### END\n\n"
        f"### FILE: tasks/{LEVEL}/{TASK_ID}/meta.yaml\n"
        "schema_version: 1.0\nname: x\ntolerance_class: reduction\n### END\n"
    )


def _good_rewrite_primitives_response() -> str:
    body_meta = _meta_payload(
        "rewrite-primitives",
        primitives_changed=["sigmoid->mul", "exp"],
    )
    return (
        f"### META\n{body_meta}\n### END\n\n"
        f"### FILE: tasks/{LEVEL}/{TASK_ID}/oracle_pytorch.py\n"
        "def oracle(x):\n    return x * (1.0 / (1.0 + (-x).exp()))\n### END\n"
    )


def _good_defer_upstream_oracle_response() -> str:
    body_meta = _meta_payload(
        "defer-to-upstream-oracle",
        oracle_action="unchanged",
        upstream_callable={"module": "vllm.layers", "function": "silu"},
    )
    return (
        f"### META\n{body_meta}\n### END\n\n"
        f"### FILE: tasks/{LEVEL}/{TASK_ID}/meta.yaml\n{_DEFER_META_YAML}### END\n"
    )


def _good_defer_to_golden_response() -> str:
    body_meta = _meta_payload(
        "defer-to-golden",
        oracle_action="unchanged",
        upstream_callable={"module": "vllm.layers", "function": "silu"},
        blocker_reason="C++ binding needs real GPU runtime",
    )
    return (
        f"### META\n{body_meta}\n### END\n\n"
        f"### FILE: tasks/{LEVEL}/{TASK_ID}/meta.yaml\n{_DEFER_META_YAML}### END\n"
    )


def test_decision_tolerance_bump_required_keys(tmp_path):
    """META missing ``tolerance_delta`` for decision=tolerance-bump must
    fail validation; entry retries via PATCH_VALIDATING -> PROMPTING."""
    from automation import state_machine as sm
    from automation.fix_orchestrator import drive_task_once

    repo, state_root, reverify, _ = _scaffold_oracle_mismatch_repo(tmp_path)
    ctx = _mk_ctx(
        repo, state_root, reverify,
        verdict_class_filter="oracle-mismatch",
        resolve_log=repo / "results" / "gpu_pipeline" / "pa_resolve.log",
    )
    # META without ``tolerance_delta``
    bad_meta = json.dumps({
        "decision": "tolerance-bump",
        "rationale": "x",
        "evidence": {},
    })
    bad = (
        f"### META\n{bad_meta}\n### END\n\n"
        f"### FILE: tasks/{LEVEL}/{TASK_ID}/meta.yaml\n"
        "schema_version: 1.0\nname: x\n### END\n"
    )
    chat = MagicMock(return_value=SimpleNamespace(content=bad))
    entry = sm.new_task_entry("oracle-mismatch")
    state = {"schema_version": "1.0", "tasks": {TASK_ID: entry}}
    drive_task_once(ctx, state, TASK_ID, llm_chat=chat, prompt_builder=None)
    # validation failed -> not BENCH_RUNNING
    final = state["tasks"][TASK_ID]
    assert final["state"] != "BENCH_RUNNING"
    last_errs = final["state_data"].get("last_validation_errors") or []
    err_text = " ".join(str(e) for e in last_errs) + " ".join(
        " ".join(str(v) for v in r.get("errors", []))
        for r in final["history"] if r.get("kind") == "patch_validation_failed"
    )
    assert "tolerance_delta" in err_text


def test_decision_rewrite_primitives_required_keys(tmp_path):
    """META missing ``primitives_changed`` for decision=rewrite-primitives
    must fail validation."""
    from automation import state_machine as sm
    from automation.fix_orchestrator import drive_task_once

    repo, state_root, reverify, _ = _scaffold_oracle_mismatch_repo(tmp_path)
    ctx = _mk_ctx(
        repo, state_root, reverify,
        verdict_class_filter="oracle-mismatch",
        resolve_log=repo / "results" / "gpu_pipeline" / "pa_resolve.log",
    )
    bad_meta = json.dumps({
        "decision": "rewrite-primitives",
        "rationale": "x",
        "evidence": {},
    })
    bad = (
        f"### META\n{bad_meta}\n### END\n\n"
        f"### FILE: tasks/{LEVEL}/{TASK_ID}/oracle_pytorch.py\n"
        "def oracle(x):\n    return x\n### END\n"
    )
    chat = MagicMock(return_value=SimpleNamespace(content=bad))
    entry = sm.new_task_entry("oracle-mismatch")
    state = {"schema_version": "1.0", "tasks": {TASK_ID: entry}}
    drive_task_once(ctx, state, TASK_ID, llm_chat=chat, prompt_builder=None)
    final = state["tasks"][TASK_ID]
    assert final["state"] != "BENCH_RUNNING"
    err_text = " ".join(
        " ".join(str(v) for v in r.get("errors", []))
        for r in final["history"] if r.get("kind") == "patch_validation_failed"
    )
    assert "primitives_changed" in err_text


def test_decision_defer_to_upstream_oracle_keys(tmp_path):
    """META missing ``upstream_callable`` for decision=defer-to-upstream-oracle
    must fail validation."""
    from automation import state_machine as sm
    from automation.fix_orchestrator import drive_task_once

    repo, state_root, reverify, _ = _scaffold_oracle_mismatch_repo(tmp_path)
    ctx = _mk_ctx(
        repo, state_root, reverify,
        verdict_class_filter="oracle-mismatch",
        resolve_log=repo / "results" / "gpu_pipeline" / "pa_resolve.log",
    )
    bad_meta = json.dumps({
        "decision": "defer-to-upstream-oracle",
        "rationale": "x",
        "evidence": {},
        "oracle_action": "unchanged",
    })
    bad = (
        f"### META\n{bad_meta}\n### END\n\n"
        f"### FILE: tasks/{LEVEL}/{TASK_ID}/meta.yaml\n{_DEFER_META_YAML}### END\n"
    )
    chat = MagicMock(return_value=SimpleNamespace(content=bad))
    entry = sm.new_task_entry("oracle-mismatch")
    state = {"schema_version": "1.0", "tasks": {TASK_ID: entry}}
    drive_task_once(ctx, state, TASK_ID, llm_chat=chat, prompt_builder=None)
    final = state["tasks"][TASK_ID]
    assert final["state"] != "BENCH_RUNNING"
    err_text = " ".join(
        " ".join(str(v) for v in r.get("errors", []))
        for r in final["history"] if r.get("kind") == "patch_validation_failed"
    )
    assert "upstream_callable" in err_text


def test_decision_defer_to_golden_keys(tmp_path):
    """META missing ``blocker_reason`` for decision=defer-to-golden must
    fail validation."""
    from automation import state_machine as sm
    from automation.fix_orchestrator import drive_task_once

    repo, state_root, reverify, _ = _scaffold_defer_repo(tmp_path)
    ctx = _mk_ctx(
        repo, state_root, reverify,
        verdict_class_filter="upstream-not-invocable-here",
        resolve_log=repo / "results" / "gpu_pipeline" / "pa_resolve.log",
    )
    bad_meta = json.dumps({
        "decision": "defer-to-golden",
        "rationale": "x",
        "evidence": {},
        "oracle_action": "unchanged",
        "upstream_callable": {"module": "vllm.layers", "function": "silu"},
    })
    bad = (
        f"### META\n{bad_meta}\n### END\n\n"
        f"### FILE: tasks/{LEVEL}/{TASK_ID}/meta.yaml\n{_DEFER_META_YAML}### END\n"
    )
    chat = MagicMock(return_value=SimpleNamespace(content=bad))
    entry = sm.new_task_entry("upstream-not-invocable-here")
    state = {"schema_version": "1.0", "tasks": {TASK_ID: entry}}
    drive_task_once(ctx, state, TASK_ID, llm_chat=chat, prompt_builder=None)
    final = state["tasks"][TASK_ID]
    assert final["state"] != "BENCH_RUNNING"
    err_text = " ".join(
        " ".join(str(v) for v in r.get("errors", []))
        for r in final["history"] if r.get("kind") == "patch_validation_failed"
    )
    assert "blocker_reason" in err_text


def test_tier1_reverify_cmd_resolve_for_patch_shapes(tmp_path):
    """After patch-shapes the pending_cmd MUST be 'resolve' so the
    reverify wrapper re-runs the resolver."""
    from automation.fix_orchestrator import drive_task_once

    repo, state_root, reverify = _scaffold_repo(tmp_path)
    ctx = _mk_ctx(repo, state_root, reverify,
                  verdict_class_filter="shapes-incomplete")
    state = {"schema_version": "1.0", "tasks": {}}
    chat = MagicMock(return_value=SimpleNamespace(content=_good_llm_response()))
    drive_task_once(ctx, state, TASK_ID, llm_chat=chat, prompt_builder=_fake_prompt)
    entry = state["tasks"][TASK_ID]
    assert entry["state"] == "BENCH_RUNNING"
    assert entry["state_data"]["pending_cmd"] == "resolve"


def test_tier1_reverify_cmd_phase_a_verify_for_tolerance_bump(tmp_path):
    """After tolerance-bump the pending_cmd MUST be 'phase_a_verify' so
    the reverify wrapper re-runs Phase A' (which re-evaluates oracle
    against captured solution under the new tolerance)."""
    from automation import state_machine as sm
    from automation.fix_orchestrator import drive_task_once

    repo, state_root, reverify, _ = _scaffold_oracle_mismatch_repo(tmp_path)
    ctx = _mk_ctx(
        repo, state_root, reverify,
        verdict_class_filter="oracle-mismatch",
        resolve_log=repo / "results" / "gpu_pipeline" / "pa_resolve.log",
    )
    chat = MagicMock(return_value=SimpleNamespace(
        content=_good_tolerance_bump_response(),
    ))
    entry = sm.new_task_entry("oracle-mismatch")
    state = {"schema_version": "1.0", "tasks": {TASK_ID: entry}}
    drive_task_once(ctx, state, TASK_ID, llm_chat=chat, prompt_builder=None)
    final = state["tasks"][TASK_ID]
    assert final["state"] == "BENCH_RUNNING", final["state"]
    assert final["state_data"]["pending_cmd"] == "phase_a_verify"


def test_tier1_reverify_cmd_phase_a_verify_for_defer_to_golden(tmp_path):
    """After defer-to-golden the pending_cmd MUST be 'phase_a_verify' so
    the reverify wrapper re-runs Phase A' (which now yields
    SKIPPED_BY_DESIGN)."""
    from automation import state_machine as sm
    from automation.fix_orchestrator import drive_task_once

    repo, state_root, reverify, _ = _scaffold_defer_repo(tmp_path)
    ctx = _mk_ctx(
        repo, state_root, reverify,
        verdict_class_filter="upstream-not-invocable-here",
        resolve_log=repo / "results" / "gpu_pipeline" / "pa_resolve.log",
    )
    chat = MagicMock(return_value=SimpleNamespace(
        content=_good_defer_to_golden_response(),
    ))
    entry = sm.new_task_entry("upstream-not-invocable-here")
    state = {"schema_version": "1.0", "tasks": {TASK_ID: entry}}
    drive_task_once(ctx, state, TASK_ID, llm_chat=chat, prompt_builder=None)
    final = state["tasks"][TASK_ID]
    assert final["state"] == "BENCH_RUNNING", final["state"]
    assert final["state_data"]["pending_cmd"] == "phase_a_verify"


def test_capture_done_promoted_transitions_to_done(tmp_path):
    """A BACKLOG_PENDING_GOLDEN entry whose meta.yaml.phase_b_capture.status
    == 'DONE' must be promoted to DONE by the sweeper, with a
    ``capture_done_promoted`` history record."""
    from automation import state_machine as sm
    from automation.fix_orchestrator import sweep_backlog_pending_golden

    repo, state_root, reverify = _scaffold_repo(tmp_path)
    ctx = _mk_ctx(repo, state_root, reverify, verdict_class_filter=None)
    # Patch meta.yaml with a DONE phase_b_capture block.
    meta_path = repo / "tasks" / LEVEL / TASK_ID / "meta.yaml"
    meta_path.write_text(yaml.safe_dump({
        "schema_version": "1.0",
        "name": "x",
        "phase_b_capture": {
            "status": "DONE",
            "captured_at": "2026-05-23T10:00:00Z",
            "run_id": "phase-b-run-1",
            "upstream_callable": {
                "module": "vllm.layers", "function": "silu",
            },
        },
    }), encoding="utf-8")
    entry = sm.new_task_entry("upstream-not-invocable-here")
    entry["state"] = "BACKLOG_PENDING_GOLDEN"
    entry["state_data"] = {"phase_b_capture_status_snapshot": "PENDING"}
    state = {"schema_version": "1.0", "tasks": {TASK_ID: entry}}
    sweep_backlog_pending_golden(ctx, state)
    final = state["tasks"][TASK_ID]
    assert final["state"] == "DONE"
    assert final["state_data"]["final_verdict"] == "SKIPPED_BY_DESIGN"
    kinds = [r.get("kind") for r in final["history"]]
    assert "capture_done_promoted" in kinds
    promoted = next(
        r for r in final["history"]
        if r.get("kind") == "capture_done_promoted"
    )
    assert promoted["captured_at"] == "2026-05-23T10:00:00Z"
    # Schema invariant: capture_done_promoted carries the full 3-field
    # delta (at + capture_status_before + capture_status_after) so the
    # promotion event is replayable / auditable.
    assert "at" in promoted
    assert promoted["capture_status_before"] == "PENDING"
    assert promoted["capture_status_after"] == "DONE"


def test_capture_pending_stays_in_backlog(tmp_path):
    """A BACKLOG_PENDING_GOLDEN entry whose meta.yaml.phase_b_capture.status
    == 'PENDING' MUST stay parked (no transition)."""
    from automation import state_machine as sm
    from automation.fix_orchestrator import sweep_backlog_pending_golden

    repo, state_root, reverify = _scaffold_repo(tmp_path)
    ctx = _mk_ctx(repo, state_root, reverify, verdict_class_filter=None)
    meta_path = repo / "tasks" / LEVEL / TASK_ID / "meta.yaml"
    meta_path.write_text(yaml.safe_dump({
        "schema_version": "1.0",
        "name": "x",
        "phase_b_capture": {
            "status": "PENDING",
            "upstream_callable": {
                "module": "vllm.layers", "function": "silu",
            },
        },
    }), encoding="utf-8")
    entry = sm.new_task_entry("upstream-not-invocable-here")
    entry["state"] = "BACKLOG_PENDING_GOLDEN"
    entry["state_data"] = {"phase_b_capture_status_snapshot": "PENDING"}
    state = {"schema_version": "1.0", "tasks": {TASK_ID: entry}}
    sweep_backlog_pending_golden(ctx, state)
    final = state["tasks"][TASK_ID]
    assert final["state"] == "BACKLOG_PENDING_GOLDEN"
    kinds = [r.get("kind") for r in final["history"]]
    assert "capture_done_promoted" not in kinds


def test_capture_failed_stays_in_backlog(tmp_path):
    """phase_b_capture.status='FAILED' MUST leave the entry parked --
    the operator re-runs capture; no auto-promotion."""
    from automation import state_machine as sm
    from automation.fix_orchestrator import sweep_backlog_pending_golden

    repo, state_root, reverify = _scaffold_repo(tmp_path)
    ctx = _mk_ctx(repo, state_root, reverify, verdict_class_filter=None)
    meta_path = repo / "tasks" / LEVEL / TASK_ID / "meta.yaml"
    meta_path.write_text(yaml.safe_dump({
        "schema_version": "1.0",
        "name": "x",
        "phase_b_capture": {
            "status": "FAILED",
            "upstream_callable": {
                "module": "vllm.layers", "function": "silu",
            },
        },
    }), encoding="utf-8")
    entry = sm.new_task_entry("upstream-not-invocable-here")
    entry["state"] = "BACKLOG_PENDING_GOLDEN"
    entry["state_data"] = {"phase_b_capture_status_snapshot": "PENDING"}
    state = {"schema_version": "1.0", "tasks": {TASK_ID: entry}}
    sweep_backlog_pending_golden(ctx, state)
    final = state["tasks"][TASK_ID]
    assert final["state"] == "BACKLOG_PENDING_GOLDEN"


def test_m1_shapes_incomplete_regression(tmp_path):
    """M1 happy-path: shapes-incomplete drives NEW -> ... -> BENCH_RUNNING
    -> consume PASS -> DONE through the M2-extended orchestrator."""
    from automation import extract_verdict as ev
    from automation.fix_orchestrator import drive_task_once

    repo, state_root, reverify = _scaffold_repo(tmp_path)
    ctx = _mk_ctx(repo, state_root, reverify,
                  verdict_class_filter="shapes-incomplete")
    state = {"schema_version": "1.0", "tasks": {}}
    chat = MagicMock(return_value=SimpleNamespace(content=_good_llm_response()))
    # First call: NEW -> BENCH_RUNNING (M1 builder path)
    drive_task_once(ctx, state, TASK_ID, llm_chat=chat,
                    prompt_builder=_fake_prompt)
    entry = state["tasks"][TASK_ID]
    assert entry["state"] == "BENCH_RUNNING"
    pending_run = entry["state_data"]["pending_run_id"]
    assert entry["state_data"]["pending_cmd"] == "resolve"
    # Second call: consume PASS -> DONE
    ev.write_verdict(
        task_id=TASK_ID, cmd="resolve", run_id=pending_run,
        producer="tier1_reverify", out_dir=reverify,
        verdict="PASS", verdict_class=None,
        reason="resolver PASS",
    )
    drive_task_once(ctx, state, TASK_ID, llm_chat=chat,
                    prompt_builder=_fake_prompt)
    final = state["tasks"][TASK_ID]
    assert final["state"] == "DONE"
    assert final["state_data"]["final_verdict"] == "PASS"


def test_skipped_by_design_routes_to_backlog_pending_golden(tmp_path):
    """When tier1_reverify returns verdict=SKIPPED_BY_DESIGN (defer
    flow), the entry must be parked in BACKLOG_PENDING_GOLDEN and the
    snapshot field must reflect the current phase_b_capture.status."""
    from automation import extract_verdict as ev
    from automation import state_machine as sm
    from automation.fix_orchestrator import drive_task_once

    repo, state_root, reverify, _ = _scaffold_defer_repo(tmp_path)
    ctx = _mk_ctx(
        repo, state_root, reverify,
        verdict_class_filter="upstream-not-invocable-here",
        resolve_log=repo / "results" / "gpu_pipeline" / "pa_resolve.log",
    )
    chat = MagicMock(return_value=SimpleNamespace(
        content=_good_defer_to_golden_response(),
    ))
    entry = sm.new_task_entry("upstream-not-invocable-here")
    state = {"schema_version": "1.0", "tasks": {TASK_ID: entry}}
    drive_task_once(ctx, state, TASK_ID, llm_chat=chat, prompt_builder=None)
    final = state["tasks"][TASK_ID]
    assert final["state"] == "BENCH_RUNNING"
    pending_run = final["state_data"]["pending_run_id"]
    ev.write_verdict(
        task_id=TASK_ID, cmd="phase_a_verify", run_id=pending_run,
        producer="tier1_reverify", out_dir=reverify,
        verdict="SKIPPED_BY_DESIGN", verdict_class=None,
        reason="phase A' skipped by design",
    )
    drive_task_once(ctx, state, TASK_ID, llm_chat=chat, prompt_builder=None)
    final = state["tasks"][TASK_ID]
    assert final["state"] == "BACKLOG_PENDING_GOLDEN"
    # On-disk meta.yaml currently carries the defer block; snapshot
    # field reflects that.
    assert final["state_data"]["phase_b_capture_status_snapshot"] == "PENDING"


def test_extract_upstream_callable_hint_from_log(tmp_path):
    """Parser extracts (module, function) hints from resolver
    reject_notes lines for upstream-not-invocable-here."""
    from automation.fix_orchestrator import extract_upstream_callable_hint

    log = tmp_path / "pa_resolve.log"
    log.write_text(
        f"[{TASK_ID}] BLOCKED - :: BLOCKED(no-verified-upstream-entrypoint: "
        "upstream-not-invocable-here): tried [a]; a: module='vllm.layers' "
        "function=silu requires real CUDA runtime\n",
        encoding="utf-8",
    )
    hint = extract_upstream_callable_hint(log, TASK_ID)
    assert hint == {"module": "vllm.layers", "function": "silu"}


def test_extract_oracle_mismatch_evidence_returns_list(tmp_path):
    """oracle-mismatch evidence extractor returns a list (the prompt
    builder kwarg is a list)."""
    from automation.fix_orchestrator import extract_oracle_mismatch_evidence

    log = tmp_path / "pa_resolve.log"
    log.write_text(
        f"[{TASK_ID}] BLOCKED - :: BLOCKED(no-verified-upstream-entrypoint: "
        "oracle-mismatch): tried [a]; a: max_rel_diff=2.3e-1\n",
        encoding="utf-8",
    )
    out = extract_oracle_mismatch_evidence(log, TASK_ID)
    assert isinstance(out, list)
    assert any("max_rel_diff" in line for line in out)
