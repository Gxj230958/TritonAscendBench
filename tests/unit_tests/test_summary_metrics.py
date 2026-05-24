"""Tests for ``automation.summary_metrics`` (Unit 3.2).

The 4 backlog dials are independent: each reads its own source of truth
(common-audit `last_consumed_verdict`, `meta.phase_a_prime.status`,
`meta.phase_b_capture.status` OR legacy `golden/`, and
`meta.release_scope.excluded`). A single task may legally count toward
all four; the sum can exceed total.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

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
# Entry constructors -- minimal valid state entries built via the
# canonical factory so the strict triage_state loader accepts them.
# ---------------------------------------------------------------------------

def _entry_with_consumed(
    *,
    verdict: str | None,
    state: str = "DONE",
    verdict_class: str = "shapes-incomplete",
    state_data: dict | None = None,
) -> dict:
    """Build a state entry with ``last_consumed_verdict=verdict`` and the
    given ``state`` enum. ``state_data`` defaults to the canonical block
    for the target state so the strict validator passes."""
    from automation import state_machine as sm
    entry = sm.new_task_entry(verdict_class)
    entry["state"] = state
    entry["last_consumed_verdict"] = verdict
    default_state_data: dict[str, object]
    if state == "DONE":
        default_state_data = {"final_verdict": verdict or "PASS"}
    elif state == "BACKLOG_PENDING_GOLDEN":
        default_state_data = {"phase_b_capture_status_snapshot": "PENDING"}
    elif state == "BENCH_RUNNING":
        default_state_data = {
            "pending_run_id": "r1",
            "pending_cmd": "resolve",
            "pending_result_dir": "/tmp/r1",
        }
    elif state == "MANUAL_BACKLOG":
        default_state_data = {
            "meta_release_scope_excluded_at": "2026-05-24T00:00:00Z",
        }
    elif state == "UNFIXABLE_BY_LLM":
        default_state_data = {"attempts_used": 3}
    else:
        default_state_data = {}
    entry["state_data"] = state_data if state_data is not None else default_state_data
    return entry


def _write_state(state_path: Path, tasks: dict[str, dict]) -> None:
    payload = {"schema_version": "1.0", "tasks": tasks}
    state_path.write_text(json.dumps(payload), encoding="utf-8")


def _write_meta(
    tasks_root: Path,
    task_id: str,
    *,
    phase_a_status: str | None = None,
    phase_b_status: str | None = None,
    release_scope_excluded: bool | None = None,
    extra: dict | None = None,
    group: str = "level1",
) -> Path:
    """Write ``tasks_root/<group>/<task_id>/meta.yaml`` with the named
    blocks (omitted when ``None``)."""
    td = tasks_root / group / task_id
    td.mkdir(parents=True, exist_ok=True)
    meta: dict = {"schema_version": "1.0", "task_id": task_id}
    if phase_a_status is not None:
        meta["phase_a_prime"] = {"status": phase_a_status}
    if phase_b_status is not None:
        meta["phase_b_capture"] = {"status": phase_b_status}
    if release_scope_excluded is not None:
        meta["release_scope"] = {
            "excluded": release_scope_excluded,
            "reason": "test",
            "excluded_at": "2026-05-24T00:00:00Z",
            "excluded_by": "op",
        }
    if extra:
        meta.update(extra)
    meta_path = td / "meta.yaml"
    meta_path.write_text(yaml.safe_dump(meta), encoding="utf-8")
    return meta_path


def _write_legacy_golden(
    tasks_root: Path,
    task_id: str,
    shape_ids: list[str],
    *,
    group: str = "level1",
) -> None:
    """Create a legacy ``golden/`` dir with one USABLE safetensors per
    shape id. ``shapes.yaml`` is also written so the legacy predicate
    can enumerate shapes the same way ``validate_tasks.py`` does.
    """
    import torch
    from safetensors.torch import save_file

    td = tasks_root / group / task_id
    td.mkdir(parents=True, exist_ok=True)
    shapes_doc = {
        "public_shapes": [{"id": sid, "name": sid} for sid in shape_ids],
    }
    (td / "shapes.yaml").write_text(yaml.safe_dump(shapes_doc), encoding="utf-8")
    gd = td / "golden"
    gd.mkdir(exist_ok=True)
    for sid in shape_ids:
        save_file({"y": torch.zeros(1)}, str(gd / f"{sid}.safetensors"))


# ---------------------------------------------------------------------------
# 1. Empty state -> zero metrics.
# ---------------------------------------------------------------------------

def test_empty_state_zero_metrics(tmp_path):
    from automation import summary_metrics as sm

    state_path = tmp_path / "triage_state.json"
    _write_state(state_path, tasks={})
    tasks_root = tmp_path / "tasks"
    tasks_root.mkdir()

    state = sm.load_state(state_path)
    metrics = sm.compute_metrics(state, tasks_root)

    assert metrics == {
        "reverify_nonblocked": 0,
        "phase_a_promoted": 0,
        "golden_ready": 0,
        "manual_backlog": 0,
    }


# ---------------------------------------------------------------------------
# 2. reverify_nonblocked reads ``last_consumed_verdict`` from the common-
#    audit fields. It is independent of ``state == DONE``.
# ---------------------------------------------------------------------------

def test_reverify_nonblocked_reads_common_audit(tmp_path):
    """``last_consumed_verdict`` lives in the common-audit set and does
    NOT require ``state == DONE`` to count -- a SKIPPED_BY_DESIGN task
    parked at BACKLOG_PENDING_GOLDEN still counts as reverify-non-blocked.
    """
    from automation import summary_metrics as sm

    tasks = {
        # PASS counted, even when state moved past DONE.
        "T_pass_done": _entry_with_consumed(verdict="PASS", state="DONE"),
        # SKIPPED_BY_DESIGN counted, parked at BACKLOG_PENDING_GOLDEN.
        "T_skip_parked": _entry_with_consumed(
            verdict="SKIPPED_BY_DESIGN",
            state="BACKLOG_PENDING_GOLDEN",
        ),
        # BLOCKED not counted.
        "T_blocked": _entry_with_consumed(
            verdict="BLOCKED", state="UNFIXABLE_BY_LLM",
        ),
        # No consumption yet -> not counted.
        "T_no_consume": _entry_with_consumed(verdict=None, state="DONE"),
    }
    state_path = tmp_path / "triage_state.json"
    _write_state(state_path, tasks)
    tasks_root = tmp_path / "tasks"
    tasks_root.mkdir()

    state = sm.load_state(state_path)
    metrics = sm.compute_metrics(state, tasks_root)
    flags = sm.compute_per_task_flags(state, tasks_root)

    assert flags["T_pass_done"]["reverify_nonblocked"] is True
    assert flags["T_skip_parked"]["reverify_nonblocked"] is True
    assert flags["T_blocked"]["reverify_nonblocked"] is False
    assert flags["T_no_consume"]["reverify_nonblocked"] is False
    assert metrics["reverify_nonblocked"] == 2


# ---------------------------------------------------------------------------
# 3. phase_a_promoted is decided by meta.phase_a_prime.status == PASS;
#    state enum is irrelevant.
# ---------------------------------------------------------------------------

def test_phase_a_promoted_independent_of_state(tmp_path):
    """phase_a_promoted counts on meta status only -- the entry's
    state enum can be DONE, BACKLOG_PENDING_GOLDEN, BENCH_RUNNING, etc."""
    from automation import summary_metrics as sm

    tasks = {
        "T_done": _entry_with_consumed(verdict="PASS", state="DONE"),
        "T_parked": _entry_with_consumed(
            verdict="SKIPPED_BY_DESIGN",
            state="BACKLOG_PENDING_GOLDEN",
        ),
        "T_running": _entry_with_consumed(verdict=None, state="BENCH_RUNNING"),
        "T_skip_meta": _entry_with_consumed(verdict="PASS", state="DONE"),
    }
    state_path = tmp_path / "triage_state.json"
    _write_state(state_path, tasks)
    tasks_root = tmp_path / "tasks"
    tasks_root.mkdir()
    # All three first tasks have phase_a_prime.status=PASS.
    _write_meta(tasks_root, "T_done", phase_a_status="PASS")
    _write_meta(tasks_root, "T_parked", phase_a_status="PASS")
    _write_meta(tasks_root, "T_running", phase_a_status="PASS")
    # SKIPPED_BY_DESIGN does NOT count toward phase_a_promoted.
    _write_meta(tasks_root, "T_skip_meta", phase_a_status="SKIPPED_BY_DESIGN")

    state = sm.load_state(state_path)
    metrics = sm.compute_metrics(state, tasks_root)
    flags = sm.compute_per_task_flags(state, tasks_root)

    assert flags["T_done"]["phase_a_promoted"] is True
    assert flags["T_parked"]["phase_a_promoted"] is True
    assert flags["T_running"]["phase_a_promoted"] is True
    assert flags["T_skip_meta"]["phase_a_promoted"] is False
    assert metrics["phase_a_promoted"] == 3


# ---------------------------------------------------------------------------
# 4. golden_ready: legacy golden/ alone is sufficient even without a
#    phase_b_capture block in meta.
# ---------------------------------------------------------------------------

def test_golden_ready_includes_legacy_golden_dir(tmp_path):
    """A task with no phase_b_capture block but a complete legacy
    ``golden/`` directory still counts toward golden_ready (the static
    `validate_tasks.py` golden gate considers the same artifacts
    sufficient)."""
    from automation import summary_metrics as sm

    tasks = {"T_legacy": _entry_with_consumed(verdict="PASS", state="DONE")}
    state_path = tmp_path / "triage_state.json"
    _write_state(state_path, tasks)
    tasks_root = tmp_path / "tasks"
    tasks_root.mkdir()
    # No phase_b_capture, but a complete legacy golden dir.
    _write_meta(tasks_root, "T_legacy")  # bare meta.yaml
    _write_legacy_golden(tasks_root, "T_legacy", ["s1", "s2"])

    state = sm.load_state(state_path)
    metrics = sm.compute_metrics(state, tasks_root)
    flags = sm.compute_per_task_flags(state, tasks_root)

    assert flags["T_legacy"]["golden_ready"] is True
    assert metrics["golden_ready"] == 1


def test_golden_ready_phase_b_done_or_legacy(tmp_path):
    """Either source (meta.phase_b_capture.status==DONE OR complete
    legacy golden/ dir) credits golden_ready; missing both leaves the
    flag False."""
    from automation import summary_metrics as sm

    tasks = {
        "T_phase_b": _entry_with_consumed(verdict="PASS", state="DONE"),
        "T_legacy":  _entry_with_consumed(verdict="PASS", state="DONE"),
        "T_neither": _entry_with_consumed(verdict="PASS", state="DONE"),
    }
    state_path = tmp_path / "triage_state.json"
    _write_state(state_path, tasks)
    tasks_root = tmp_path / "tasks"
    tasks_root.mkdir()
    _write_meta(tasks_root, "T_phase_b", phase_b_status="DONE")
    _write_meta(tasks_root, "T_legacy")
    _write_legacy_golden(tasks_root, "T_legacy", ["s1"])
    _write_meta(tasks_root, "T_neither")

    state = sm.load_state(state_path)
    metrics = sm.compute_metrics(state, tasks_root)
    flags = sm.compute_per_task_flags(state, tasks_root)

    assert flags["T_phase_b"]["golden_ready"] is True
    assert flags["T_legacy"]["golden_ready"] is True
    assert flags["T_neither"]["golden_ready"] is False
    assert metrics["golden_ready"] == 2


# ---------------------------------------------------------------------------
# 5. manual_backlog: keyed on meta.release_scope.excluded only.
# ---------------------------------------------------------------------------

def test_manual_backlog_keys_off_meta_excluded(tmp_path):
    """A DONE-state task can also be manual_backlog when the operator
    has marked ``release_scope.excluded=True`` in meta; the metric is
    independent of the state enum."""
    from automation import summary_metrics as sm

    tasks = {
        "T_done_excluded": _entry_with_consumed(verdict="PASS", state="DONE"),
        "T_done_normal":   _entry_with_consumed(verdict="PASS", state="DONE"),
        "T_unfixable":     _entry_with_consumed(
            verdict="BLOCKED", state="UNFIXABLE_BY_LLM",
        ),
    }
    state_path = tmp_path / "triage_state.json"
    _write_state(state_path, tasks)
    tasks_root = tmp_path / "tasks"
    tasks_root.mkdir()
    _write_meta(tasks_root, "T_done_excluded", release_scope_excluded=True)
    _write_meta(tasks_root, "T_done_normal",   release_scope_excluded=False)
    # T_unfixable: no meta exclusion -> NOT counted toward manual_backlog
    # (manual_backlog keys ONLY on meta.release_scope.excluded, not on
    # the state enum -- the orchestrator's UNFIXABLE_BY_LLM bucket is
    # a separate signal surfaced by per_state_count).
    _write_meta(tasks_root, "T_unfixable")

    state = sm.load_state(state_path)
    metrics = sm.compute_metrics(state, tasks_root)
    flags = sm.compute_per_task_flags(state, tasks_root)

    assert flags["T_done_excluded"]["manual_backlog"] is True
    assert flags["T_done_normal"]["manual_backlog"] is False
    assert flags["T_unfixable"]["manual_backlog"] is False
    assert metrics["manual_backlog"] == 1


# ---------------------------------------------------------------------------
# 6. Overlap: phase_a_promoted + manual_backlog can coexist on the same
#    task (release_scope.excluded does NOT remove the phase_a_promoted
#    credit -- the metrics are independent).
# ---------------------------------------------------------------------------

def test_phase_a_promoted_overlaps_manual_backlog(tmp_path):
    from automation import summary_metrics as sm

    tasks = {"T_dual": _entry_with_consumed(verdict="PASS", state="DONE")}
    state_path = tmp_path / "triage_state.json"
    _write_state(state_path, tasks)
    tasks_root = tmp_path / "tasks"
    tasks_root.mkdir()
    _write_meta(
        tasks_root, "T_dual",
        phase_a_status="PASS",
        release_scope_excluded=True,
    )

    state = sm.load_state(state_path)
    metrics = sm.compute_metrics(state, tasks_root)
    flags = sm.compute_per_task_flags(state, tasks_root)

    assert flags["T_dual"]["phase_a_promoted"] is True
    assert flags["T_dual"]["manual_backlog"] is True
    assert metrics["phase_a_promoted"] == 1
    assert metrics["manual_backlog"] == 1


# ---------------------------------------------------------------------------
# 7. Overlap: golden_ready + manual_backlog likewise coexist.
# ---------------------------------------------------------------------------

def test_golden_ready_overlaps_manual_backlog(tmp_path):
    from automation import summary_metrics as sm

    tasks = {"T_dual_g": _entry_with_consumed(verdict="PASS", state="DONE")}
    state_path = tmp_path / "triage_state.json"
    _write_state(state_path, tasks)
    tasks_root = tmp_path / "tasks"
    tasks_root.mkdir()
    _write_meta(
        tasks_root, "T_dual_g",
        phase_b_status="DONE",
        release_scope_excluded=True,
    )

    state = sm.load_state(state_path)
    metrics = sm.compute_metrics(state, tasks_root)
    flags = sm.compute_per_task_flags(state, tasks_root)

    assert flags["T_dual_g"]["golden_ready"] is True
    assert flags["T_dual_g"]["manual_backlog"] is True
    assert metrics["golden_ready"] == 1
    assert metrics["manual_backlog"] == 1


# ---------------------------------------------------------------------------
# 8. sum(metrics) can legally exceed total (independent metrics).
# ---------------------------------------------------------------------------

def test_metrics_sum_can_exceed_total(tmp_path):
    """101 tasks, every one of them PASS + phase_a PASS + phase_b DONE
    + release_scope.excluded -> 4 * 101 = 404 > total 101. The 4
    metrics are independent by design (R4)."""
    from automation import summary_metrics as sm

    n = 101
    tasks = {
        f"T{i}": _entry_with_consumed(verdict="PASS", state="DONE")
        for i in range(n)
    }
    state_path = tmp_path / "triage_state.json"
    _write_state(state_path, tasks)
    tasks_root = tmp_path / "tasks"
    tasks_root.mkdir()
    for i in range(n):
        _write_meta(
            tasks_root, f"T{i}",
            phase_a_status="PASS",
            phase_b_status="DONE",
            release_scope_excluded=True,
        )

    state = sm.load_state(state_path)
    metrics = sm.compute_metrics(state, tasks_root)

    assert metrics["reverify_nonblocked"] == n
    assert metrics["phase_a_promoted"] == n
    assert metrics["golden_ready"] == n
    assert metrics["manual_backlog"] == n
    metric_sum = sum(metrics.values())
    assert metric_sum == 4 * n
    assert metric_sum > n


# ---------------------------------------------------------------------------
# 9. per_task flags carry all four bool keys for every task.
# ---------------------------------------------------------------------------

def test_per_task_flags_complete(tmp_path):
    from automation import summary_metrics as sm

    tasks = {
        "T_done": _entry_with_consumed(verdict="PASS", state="DONE"),
        "T_manual": _entry_with_consumed(verdict=None, state="MANUAL_BACKLOG"),
        "T_unfix": _entry_with_consumed(
            verdict="BLOCKED", state="UNFIXABLE_BY_LLM",
        ),
    }
    state_path = tmp_path / "triage_state.json"
    _write_state(state_path, tasks)
    tasks_root = tmp_path / "tasks"
    tasks_root.mkdir()
    _write_meta(tasks_root, "T_manual", release_scope_excluded=True)

    state = sm.load_state(state_path)
    flags = sm.compute_per_task_flags(state, tasks_root)

    assert set(flags.keys()) == {"T_done", "T_manual", "T_unfix"}
    required_keys = {
        "reverify_nonblocked", "phase_a_promoted",
        "golden_ready", "manual_backlog",
    }
    for tid, per_task in flags.items():
        assert set(per_task.keys()) == required_keys, tid
        for k, v in per_task.items():
            assert isinstance(v, bool), f"{tid}.{k} not bool: {v!r}"


# ---------------------------------------------------------------------------
# 10. Missing meta.yaml: meta-driven metrics fall back to False; the
#     entry-level reverify_nonblocked metric is unaffected.
# ---------------------------------------------------------------------------

def test_missing_meta_yaml_does_not_raise(tmp_path, caplog):
    from automation import summary_metrics as sm

    tasks = {"T_no_meta": _entry_with_consumed(verdict="PASS", state="DONE")}
    state_path = tmp_path / "triage_state.json"
    _write_state(state_path, tasks)
    tasks_root = tmp_path / "tasks"
    tasks_root.mkdir()
    # Deliberately no meta.yaml on disk.

    state = sm.load_state(state_path)
    with caplog.at_level("WARNING", logger="automation.summary_metrics"):
        flags = sm.compute_per_task_flags(state, tasks_root)
        metrics = sm.compute_metrics(state, tasks_root)

    # last_consumed_verdict still credits reverify_nonblocked.
    assert flags["T_no_meta"]["reverify_nonblocked"] is True
    # meta-dependent metrics fall back to False.
    assert flags["T_no_meta"]["phase_a_promoted"] is False
    assert flags["T_no_meta"]["golden_ready"] is False
    assert flags["T_no_meta"]["manual_backlog"] is False
    assert metrics["phase_a_promoted"] == 0
    assert metrics["golden_ready"] == 0
    assert metrics["manual_backlog"] == 0
    assert any(
        "T_no_meta" in rec.getMessage() for rec in caplog.records
    ), "expected a warning that mentions the missing-meta task"


# ---------------------------------------------------------------------------
# 11. --format both: writes .md + .json with consistent counts and the
#     markdown explicitly mentions overlap is allowed.
# ---------------------------------------------------------------------------

def test_atomic_write_both_formats(tmp_path):
    from automation import summary_metrics as sm

    tasks = {
        "T_a": _entry_with_consumed(verdict="PASS", state="DONE"),
        "T_b": _entry_with_consumed(verdict="PASS", state="DONE"),
        "T_excluded": _entry_with_consumed(verdict=None, state="MANUAL_BACKLOG"),
    }
    state_path = tmp_path / "triage_state.json"
    _write_state(state_path, tasks)
    tasks_root = tmp_path / "tasks"
    tasks_root.mkdir()
    _write_meta(tasks_root, "T_a", phase_a_status="PASS")
    _write_meta(tasks_root, "T_b", phase_b_status="DONE")
    _write_meta(tasks_root, "T_excluded", release_scope_excluded=True)

    out_dir = tmp_path / "out"
    date_iso = "2026-05-24"

    rc = sm.main([
        "--state-path", str(state_path),
        "--tasks-root", str(tasks_root),
        "--out-dir", str(out_dir),
        "--format", "both",
        "--date-iso", date_iso,
    ])
    assert rc == 0

    md_path = out_dir / f"summary-metrics-{date_iso}.md"
    json_path = out_dir / f"summary-metrics-{date_iso}.json"
    assert md_path.is_file(), "markdown artifact missing"
    assert json_path.is_file(), "json artifact missing"

    leftovers = list(out_dir.glob("*.tmp.*"))
    assert leftovers == [], f"atomic tmp file leaked: {leftovers}"

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    metrics = payload["metrics"]
    # T_a, T_b -> reverify_nonblocked (last_consumed_verdict=PASS).
    # T_excluded -> no consumed verdict -> not counted.
    assert metrics["reverify_nonblocked"] == 2
    assert metrics["phase_a_promoted"] == 1   # T_a
    assert metrics["golden_ready"] == 1       # T_b
    assert metrics["manual_backlog"] == 1     # T_excluded
    assert payload["total"] == 3
    assert set(payload["per_task_flags"].keys()) == {
        "T_a", "T_b", "T_excluded",
    }

    md_text = md_path.read_text(encoding="utf-8")
    assert "| reverify_nonblocked | 2 |" in md_text
    assert "| phase_a_promoted | 1 |" in md_text
    assert "| golden_ready | 1 |" in md_text
    assert "| manual_backlog | 1 |" in md_text
    assert "Total tasks: 3" in md_text
    assert "INDEPENDENT" in md_text
    # The markdown definitions are the canonical 4-formula version.
    assert "last_consumed_verdict" in md_text
    assert "meta.phase_a_prime.status == PASS" in md_text
    assert "meta.phase_b_capture.status == DONE" in md_text
    assert "meta.release_scope.excluded" in md_text
