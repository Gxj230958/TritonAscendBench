"""Tests for ``automation.backlog_report`` (Unit 3.1)."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

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
# Builders for mock triage_state.json content
# ---------------------------------------------------------------------------

def _mk_entry(
    *,
    state: str,
    verdict_class: str = "shapes-incomplete",
    attempt: int = 0,
    state_data: dict[str, Any] | None = None,
    history: list[dict[str, Any]] | None = None,
    last_consumed_verdict: str | None = None,
    last_consumed_at: str | None = None,
    last_consumed_run_id: str | None = None,
) -> dict[str, Any]:
    return {
        "state": state,
        "verdict_class": verdict_class,
        "attempt": attempt,
        "last_updated_at": "2026-05-23T00:00:00Z",
        "history": history or [],
        "last_consumed_verdict": last_consumed_verdict,
        "last_consumed_at": last_consumed_at,
        "last_consumed_run_id": last_consumed_run_id,
        "state_data": state_data or {},
    }


def _write_state(path: Path, tasks: dict[str, dict[str, Any]]) -> None:
    state = {"schema_version": "1.0", "tasks": tasks}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2), encoding="utf-8")


def _write_result_json(
    results_root: Path,
    run_id: str,
    *,
    verdict: str = "BLOCKED",
    verdict_class: str = "oracle-mismatch",
    reason: str = "mock reason",
    cmd: str = "resolve",
) -> Path:
    run_dir = results_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "1.0",
        "run_id": run_id,
        "task_id": "mock_task",
        "cmd": cmd,
        "verdict": verdict,
        "verdict_class": verdict_class,
        "reason": reason,
        "produced_at": "2026-05-23T12:34:56Z",
        "producer": "tier1_reverify",
    }
    out = run_dir / "result.json"
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out


# ---------------------------------------------------------------------------
# Test scenarios (Unit 3.1)
# ---------------------------------------------------------------------------

def test_empty_state_no_backlog(tmp_path):
    """Empty state file -> report has 0 backlog rows in every bucket."""
    from automation import backlog_report as br

    state_path = tmp_path / "triage_state.json"
    _write_state(state_path, tasks={})

    state = br.load_state(state_path)
    partitioned = br.partition_tasks(state)
    for st in br.BACKLOG_STATES:
        assert partitioned[st] == []

    md = br.render_markdown(
        partitioned,
        date_iso="2026-05-23",
        state_path=state_path,
        state=state,
        include_done=False,
    )
    assert "Total tasks: 0 (0 backlog + 0 done)" in md
    for st in br.BACKLOG_STATES:
        assert f"## {st} (0 tasks)" in md


def test_partition_by_backlog_state(tmp_path):
    """5 tasks across 5 distinct backlog states -> 5 non-empty buckets."""
    from automation import backlog_report as br

    state_path = tmp_path / "triage_state.json"
    tasks = {
        "t1": _mk_entry(
            state="BACKLOG_PENDING_GOLDEN",
            state_data={"phase_b_capture_status_snapshot": "PENDING"},
        ),
        "t2": _mk_entry(
            state="BACKLOG_UNEXPECTED",
            verdict_class="unexpected-blocked",
            state_data={"raw_verdict_class": "unexpected-blocked"},
        ),
        "t3": _mk_entry(
            state="UNFIXABLE_BY_LLM",
            state_data={"attempts_used": 3},
        ),
        "t4": _mk_entry(
            state="HUMAN_REVIEW_PENDING",
            verdict_class="oracle-mismatch",
            state_data={
                "needs_human_review_flag_path":
                    "/tmp/needs_human_review.flag",
            },
        ),
        "t5": _mk_entry(
            state="MANUAL_BACKLOG",
            state_data={
                "meta_release_scope_excluded_at": "2026-05-23T00:00:00Z",
            },
        ),
        "t6_done": _mk_entry(
            state="DONE",
            state_data={"final_verdict": "PASS"},
        ),
    }
    _write_state(state_path, tasks)

    state = br.load_state(state_path)
    partitioned = br.partition_tasks(state)
    assert len(partitioned["BACKLOG_PENDING_GOLDEN"]) == 1
    assert len(partitioned["BACKLOG_UNEXPECTED"]) == 1
    assert len(partitioned["UNFIXABLE_BY_LLM"]) == 1
    assert len(partitioned["HUMAN_REVIEW_PENDING"]) == 1
    assert len(partitioned["MANUAL_BACKLOG"]) == 1
    # DONE excluded by default.
    assert "DONE" not in partitioned


def test_enrich_with_result_json(tmp_path):
    """Task with last_consumed_run_id -> result.json fields merged."""
    from automation import backlog_report as br

    results_root = tmp_path / "results"
    _write_result_json(
        results_root, "run-001",
        verdict="BLOCKED", verdict_class="oracle-mismatch",
        reason="upstream mismatch on lp_norm",
    )
    entry = _mk_entry(
        state="BACKLOG_UNEXPECTED",
        verdict_class="unexpected-blocked",
        last_consumed_verdict="BLOCKED",
        last_consumed_at="2026-05-23T12:34:56Z",
        last_consumed_run_id="run-001",
        state_data={"raw_verdict_class": "unexpected-blocked"},
    )
    enriched = br.enrich_with_result(entry, results_root)
    assert enriched["result_verdict"] == "BLOCKED"
    assert enriched["result_verdict_class"] == "oracle-mismatch"
    assert enriched["result_reason"] == "upstream mismatch on lp_norm"
    assert enriched["result_cmd"] == "resolve"
    assert enriched["result_producer"] == "tier1_reverify"


def test_enrich_without_run_id(tmp_path):
    """Task with no last_consumed_run_id -> result_* fields None, no raise."""
    from automation import backlog_report as br

    results_root = tmp_path / "results"
    entry = _mk_entry(
        state="HUMAN_REVIEW_PENDING",
        state_data={
            "needs_human_review_flag_path": "/tmp/x.flag",
        },
    )
    enriched = br.enrich_with_result(entry, results_root)
    assert enriched["result_verdict"] is None
    assert enriched["result_verdict_class"] is None
    assert enriched["result_reason"] is None
    # Original entry fields preserved on the shallow copy.
    assert enriched["state"] == "HUMAN_REVIEW_PENDING"

    # Also test: run_id present but result.json missing -> no raise.
    entry2 = _mk_entry(
        state="HUMAN_REVIEW_PENDING",
        last_consumed_run_id="nonexistent-run",
        state_data={"needs_human_review_flag_path": "/tmp/y.flag"},
    )
    enriched2 = br.enrich_with_result(entry2, results_root)
    assert enriched2["result_verdict"] is None


def test_render_markdown_includes_all_states(tmp_path):
    """Markdown output contains an H2 section for every canonical backlog state."""
    from automation import backlog_report as br

    state_path = tmp_path / "triage_state.json"
    tasks = {
        "t1": _mk_entry(
            state="BACKLOG_PENDING_GOLDEN",
            state_data={"phase_b_capture_status_snapshot": "PENDING"},
        ),
        "t2": _mk_entry(
            state="UNFIXABLE_BY_LLM",
            state_data={"attempts_used": 3},
        ),
    }
    _write_state(state_path, tasks)

    state = br.load_state(state_path)
    partitioned = br.partition_tasks(state)
    md = br.render_markdown(
        partitioned,
        date_iso="2026-05-23",
        state_path=state_path,
        state=state,
        include_done=False,
    )

    for st in br.BACKLOG_STATES:
        assert f"## {st}" in md, f"missing H2 header for state {st}"
    assert "## Backlog Summary" in md
    assert "Triage Backlog Report -- 2026-05-23" in md
    # Verdict class mix line present for occupied state.
    assert "shapes-incomplete:1" in md


def test_render_json_schema_complete(tmp_path):
    """JSON sidecar contains generated_at / totals / backlog with correct shapes."""
    from automation import backlog_report as br

    state_path = tmp_path / "triage_state.json"
    tasks = {
        "t1": _mk_entry(
            state="BACKLOG_PENDING_GOLDEN",
            state_data={"phase_b_capture_status_snapshot": "PENDING"},
            history=[
                {
                    "kind": "result_consumed",
                    "at": "2026-05-23T12:00:00Z",
                    "run_id": "r1",
                    "cmd": "resolve",
                    "verdict": "BLOCKED",
                },
            ],
            last_consumed_verdict="BLOCKED",
            last_consumed_at="2026-05-23T12:00:00Z",
            last_consumed_run_id="r1",
        ),
    }
    _write_state(state_path, tasks)

    state = br.load_state(state_path)
    partitioned = br.partition_tasks(state)
    js = br.render_json(
        partitioned,
        date_iso="2026-05-23",
        state_path=state_path,
        state=state,
        include_done=False,
    )
    assert "generated_at" in js
    assert "totals" in js
    assert "backlog" in js
    assert js["schema_version"] == "1.0"
    assert js["date_iso"] == "2026-05-23"
    assert js["totals"]["total"] == 1
    assert js["totals"]["backlog"] == 1
    assert js["totals"]["done"] == 0
    row = js["backlog"]["BACKLOG_PENDING_GOLDEN"][0]
    assert row["task_id"] == "t1"
    assert row["attempt"] == 0
    assert row["last_consumed_run_id"] == "r1"
    assert row["result_consumed_count"] == 1
    assert row["patch_validation_failed_count"] == 0
    # state_data preserved verbatim.
    assert row["state_data"]["phase_b_capture_status_snapshot"] == "PENDING"
    # history_tail kept (truncated to last 3).
    assert len(row["history_tail"]) == 1
    assert row["history_tail"][0]["kind"] == "result_consumed"


def test_done_excluded_by_default(tmp_path):
    """``--include-done`` False -> DONE bucket not in partitioned dict."""
    from automation import backlog_report as br

    state_path = tmp_path / "triage_state.json"
    tasks = {
        "done1": _mk_entry(
            state="DONE",
            state_data={"final_verdict": "PASS"},
        ),
        "backlog1": _mk_entry(
            state="UNFIXABLE_BY_LLM",
            state_data={"attempts_used": 3},
        ),
    }
    _write_state(state_path, tasks)

    state = br.load_state(state_path)
    # Default: DONE excluded.
    partitioned_default = br.partition_tasks(state, include_done=False)
    assert "DONE" not in partitioned_default
    assert len(partitioned_default["UNFIXABLE_BY_LLM"]) == 1

    # With include_done=True: DONE included.
    partitioned_inc = br.partition_tasks(state, include_done=True)
    assert "DONE" in partitioned_inc
    assert len(partitioned_inc["DONE"]) == 1
    # Backlog bucket unchanged.
    assert len(partitioned_inc["UNFIXABLE_BY_LLM"]) == 1

    md = br.render_markdown(
        partitioned_default,
        date_iso="2026-05-23",
        state_path=state_path,
        state=state,
        include_done=False,
    )
    # DONE section not rendered in default.
    assert "## DONE (" not in md
    # But the done count appears in summary header.
    assert "Total tasks: 2 (1 backlog + 1 done)" in md


def test_atomic_write_both_formats(tmp_path, monkeypatch, capsys):
    """``--format both`` writes .md + .json with consistent content."""
    from automation import backlog_report as br

    state_path = tmp_path / "triage_state.json"
    tasks = {
        "t1": _mk_entry(
            state="BACKLOG_UNEXPECTED",
            verdict_class="unexpected-blocked",
            state_data={"raw_verdict_class": "unexpected-blocked"},
            last_consumed_verdict="BLOCKED",
            last_consumed_at="2026-05-23T12:00:00Z",
            last_consumed_run_id="run-001",
        ),
    }
    _write_state(state_path, tasks)
    results_root = tmp_path / "results"
    _write_result_json(
        results_root, "run-001",
        verdict="BLOCKED", verdict_class="unexpected-blocked",
        reason="mock unexpected reason",
    )

    out_dir = tmp_path / "out"
    rc = br.main([
        "--state-path", str(state_path),
        "--results-root", str(results_root),
        "--out-dir", str(out_dir),
        "--format", "both",
        "--date-iso", "2026-05-23",
    ])
    assert rc == 0

    md_path = out_dir / "backlog-report-2026-05-23.md"
    js_path = out_dir / "backlog-report-2026-05-23.json"
    assert md_path.is_file()
    assert js_path.is_file()

    # JSON parses + has expected row.
    js = json.loads(js_path.read_text(encoding="utf-8"))
    assert js["totals"]["total"] == 1
    assert js["totals"]["backlog"] == 1
    row = js["backlog"]["BACKLOG_UNEXPECTED"][0]
    assert row["task_id"] == "t1"
    assert row["result_verdict"] == "BLOCKED"
    assert row["result_reason"] == "mock unexpected reason"

    # MD contains the same task + result reason.
    md = md_path.read_text(encoding="utf-8")
    assert "### t1" in md
    assert "mock unexpected reason" in md
    assert "## BACKLOG_UNEXPECTED (1 tasks)" in md


# ---------------------------------------------------------------------------
# Additional edge cases beyond the 8 required scenarios -- light coverage of
# Unit 3.1 strict-loader integration (corrupt / missing state file paths).
# ---------------------------------------------------------------------------

def test_capture_done_promoted_counted_in_summary(tmp_path):
    """defer-to-golden: sweep-promote DONE tasks counted distinctly from
    rule-a DONE tasks. (Unit 3.1 Happy scenario for capture_done_promoted.)"""
    from automation import backlog_report as br

    state_path = tmp_path / "triage_state.json"
    tasks: dict[str, dict[str, Any]] = {}
    # 3 tasks with capture_done_promoted history (rule b).
    for i in range(3):
        tasks[f"sweep_{i}"] = _mk_entry(
            state="DONE",
            state_data={"final_verdict": "SKIPPED_BY_DESIGN"},
            history=[
                {
                    "kind": "capture_done_promoted",
                    "at": "2026-05-23T12:00:00Z",
                    "run_id": f"r{i}",
                    "capture_status_before": "PENDING",
                    "capture_status_after": "DONE",
                },
            ],
        )
    # 2 tasks without that history (rule a, immediate DONE).
    for i in range(2):
        tasks[f"immediate_{i}"] = _mk_entry(
            state="DONE",
            state_data={"final_verdict": "SKIPPED_BY_DESIGN"},
        )
    _write_state(state_path, tasks)

    state = br.load_state(state_path)
    partitioned = br.partition_tasks(state)
    md = br.render_markdown(
        partitioned,
        date_iso="2026-05-23",
        state_path=state_path,
        state=state,
        include_done=False,
    )
    # All 5 are DONE/SKIPPED_BY_DESIGN; 3 were swept.
    assert (
        "defer-to-golden: 3 of 5 SKIPPED_BY_DESIGN tasks reached DONE "
        "via sweep promote"
    ) in md
    # None of them appear in BACKLOG_PENDING_GOLDEN section.
    assert "## BACKLOG_PENDING_GOLDEN (0 tasks)" in md


def test_cli_returns_nonzero_when_state_missing(tmp_path, capsys):
    """No state file at path -> CLI exits non-zero + refuses to generate."""
    from automation import backlog_report as br

    out_dir = tmp_path / "out"
    rc = br.main([
        "--state-path", str(tmp_path / "does-not-exist.json"),
        "--out-dir", str(out_dir),
        "--format", "both",
    ])
    assert rc != 0
    captured = capsys.readouterr()
    assert "state file not found" in captured.err
    # Output dir may be created but no report should be written.
    assert not (out_dir / "backlog-report-2026-05-23.md").exists()


# ---------------------------------------------------------------------------
# Relaxed-loader fail-soft semantics: ``backlog_report`` must tolerate a
# transitional ``state_data`` (e.g. BENCH_RUNNING mid-flush whose
# ``pending_run_id`` was not yet persisted) -- the report renders
# ``n/a``-style placeholders instead of crashing the whole pass. The
# top-level invariants (schema_version, history kind enum) stay strict.
# ---------------------------------------------------------------------------

def test_backlog_report_tolerates_missing_state_data(tmp_path):
    """A BENCH_RUNNING entry that is missing every state_data field
    must NOT crash the report -- the renderer ships an n/a placeholder
    and the rest of the report continues."""
    from automation import backlog_report as br

    state_path = tmp_path / "triage_state.json"
    # BENCH_RUNNING normally requires pending_run_id / pending_cmd /
    # pending_result_dir. Here we leave state_data as ``{}`` to mimic
    # an orchestrator that transitioned + crashed before flushing the
    # state_data block.
    tasks = {
        "t_transitional": _mk_entry(
            state="BENCH_RUNNING",
            verdict_class="shapes-incomplete",
            state_data={},
        ),
        # A clean parked entry so partition_tasks has at least one
        # backlog bucket member.
        "t_parked": _mk_entry(
            state="BACKLOG_PENDING_GOLDEN",
            state_data={"phase_b_capture_status_snapshot": "PENDING"},
        ),
    }
    _write_state(state_path, tasks)

    # Relaxed loader must accept the file rather than raise.
    state = br.load_state(state_path)
    partitioned = br.partition_tasks(state)
    # Transitional entry is not in a backlog state -> not bucketed.
    assert "t_transitional" not in {
        r["task_id"] for bucket in partitioned.values() for r in bucket
    }
    assert len(partitioned["BACKLOG_PENDING_GOLDEN"]) == 1

    # Render the markdown end-to-end; the transitional entry is fine
    # because it does not surface in a backlog bucket. As a stronger
    # test, force it into a backlog state with empty state_data so the
    # renderer must produce an n/a-flavoured output.
    tasks2 = {
        "t_partial": _mk_entry(
            state="BACKLOG_UNEXPECTED",
            verdict_class="unexpected-blocked",
            state_data={},  # raw_verdict_class missing
        ),
    }
    state_path2 = tmp_path / "triage_state_2.json"
    _write_state(state_path2, tasks2)
    state2 = br.load_state(state_path2)
    partitioned2 = br.partition_tasks(state2)
    md = br.render_markdown(
        partitioned2,
        date_iso="2026-05-24",
        state_path=state_path2,
        state=state2,
        include_done=False,
    )
    # The renderer surfaces the entry without raising; the absent
    # state_data fields collapse to default markers (no key=value
    # rows for them) but the entry itself appears.
    assert "t_partial" in md
    assert "## BACKLOG_UNEXPECTED" in md


def test_backlog_report_still_fails_on_unknown_history_kind(tmp_path):
    """The relaxed loader keeps the history tagged-union enum strict --
    an unknown ``kind`` is a top-level schema invariant violation and
    MUST raise."""
    from automation import backlog_report as br
    from automation.fix_orchestrator import StateFileCorruptError

    state_path = tmp_path / "triage_state.json"
    entry = _mk_entry(
        state="BACKLOG_UNEXPECTED",
        verdict_class="unexpected-blocked",
        state_data={"raw_verdict_class": "unexpected-blocked"},
        history=[{
            "kind": "totally_not_a_kind",  # not in CANONICAL_HISTORY_KINDS
            "at": "2026-05-23T00:00:00Z",
        }],
    )
    _write_state(state_path, {"t1": entry})

    with pytest.raises(StateFileCorruptError):
        br.load_state(state_path)


def test_backlog_report_still_fails_on_missing_schema_version(tmp_path):
    """``schema_version`` is a top-level invariant -- a state file
    missing it (or carrying a wrong value) is corrupt and MUST raise
    even under the relaxed loader."""
    from automation import backlog_report as br
    from automation.fix_orchestrator import StateFileCorruptError

    state_path = tmp_path / "triage_state.json"
    # Deliberately omit ``schema_version``.
    state_path.write_text(json.dumps({"tasks": {}}), encoding="utf-8")

    with pytest.raises(StateFileCorruptError):
        br.load_state(state_path)
