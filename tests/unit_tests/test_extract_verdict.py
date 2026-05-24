"""Tests for ``automation.extract_verdict`` (Unit 1.7)."""
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


def test_resolve_log_pass(tmp_path):
    from automation import extract_verdict as ev

    log = tmp_path / "pa_resolve.log"
    log.write_text(
        "[L1_99_other] RESOLVED other_fn :: PASS\n"
        "[L1_12_rms_norm] RESOLVED rms_norm :: PASS\n",
        encoding="utf-8",
    )
    v, vc, _reason, _raw = ev.parse_resolve_log(log, "L1_12_rms_norm")
    assert v == "PASS"
    assert vc is None


def test_resolve_log_blocked_shapes_incomplete(tmp_path):
    from automation import extract_verdict as ev

    log = tmp_path / "pa_resolve.log"
    log.write_text(
        "[L1_06_swiglu] BLOCKED - :: BLOCKED(no-verified-upstream-entrypoint: "
        "shapes-incomplete)\n",
        encoding="utf-8",
    )
    v, vc, reason, _raw = ev.parse_resolve_log(log, "L1_06_swiglu")
    assert v == "BLOCKED"
    assert vc == "shapes-incomplete"
    assert "shapes-incomplete" in reason


def test_resolve_log_unknown_subclass_falls_to_unexpected(tmp_path):
    from automation import extract_verdict as ev

    log = tmp_path / "pa_resolve.log"
    log.write_text(
        "[L1_xx_demo] BLOCKED - :: BLOCKED(no-verified-upstream-entrypoint: "
        "something-totally-new)\n",
        encoding="utf-8",
    )
    v, vc, _reason, raw = ev.parse_resolve_log(log, "L1_xx_demo")
    assert v == "BLOCKED"
    assert vc == "unexpected-blocked"
    assert raw == "something-totally-new"


def test_phase_a_verify_pass_from_meta(tmp_path):
    from automation import extract_verdict as ev

    repo = tmp_path
    task_id = "L2_99_demo"
    td = repo / "tasks" / "L2" / task_id
    td.mkdir(parents=True)
    (td / "meta.yaml").write_text(
        yaml.safe_dump({"phase_a_prime": {"status": "PASS", "reason": "ok"}}),
        encoding="utf-8",
    )
    v, vc, _, _ = ev.parse_phase_a_verify(repo, task_id)
    assert v == "PASS"
    assert vc is None


def test_phase_a_verify_skipped_by_design(tmp_path):
    from automation import extract_verdict as ev

    repo = tmp_path
    task_id = "L3_42_x"
    td = repo / "tasks" / "L3" / task_id
    td.mkdir(parents=True)
    (td / "meta.yaml").write_text(
        yaml.safe_dump({
            "phase_a_prime": {
                "status": "SKIPPED_BY_DESIGN",
                "reason": "upstream gating; skip_invocation=true",
                "skip_invocation": True,
            }
        }),
        encoding="utf-8",
    )
    v, vc, reason, _ = ev.parse_phase_a_verify(repo, task_id)
    assert v == "SKIPPED_BY_DESIGN"
    assert vc is None
    assert "SKIPPED_BY_DESIGN" in reason


def test_phase_a_verify_no_meta_no_stdout_unexpected(tmp_path):
    from automation import extract_verdict as ev

    v, vc, reason, _ = ev.parse_phase_a_verify(tmp_path, "L1_404_missing")
    assert v == "BLOCKED"
    assert vc == "unexpected-blocked"
    assert "no verdict source" in reason


def test_phase_a_verify_stdout_fallback(tmp_path):
    from automation import extract_verdict as ev

    repo = tmp_path
    stdout = tmp_path / "log.txt"
    stdout.write_text(
        "...running phase_a_prime...\n"
        "S2Result(status=BLOCKED, reason='shapes-incomplete: missing M')\n",
        encoding="utf-8",
    )
    v, vc, _, _ = ev.parse_phase_a_verify(repo, "L1_06_swiglu", stdout_log=stdout)
    assert v == "BLOCKED"
    assert vc == "shapes-incomplete"


def test_write_verdict_writes_three_files(tmp_path):
    from automation import extract_verdict as ev

    out = tmp_path / "results"
    ev.write_verdict(
        task_id="L1_06_swiglu", cmd="resolve", run_id="r1",
        producer="tier1_reverify", out_dir=out,
        verdict="PASS", verdict_class=None,
        reason="resolver PASS",
    )
    run_dir = out / "r1"
    assert (run_dir / "result.json").is_file()
    assert (run_dir / "_provenance.json").is_file()
    assert (run_dir / "DONE").is_file()
    result = json.loads((run_dir / "result.json").read_text())
    assert set(result.keys()) == {
        "schema_version", "run_id", "task_id", "cmd", "verdict",
        "verdict_class", "reason", "produced_at", "producer",
    }


def test_write_verdict_strict_schema_no_provenance_key(tmp_path):
    from automation import extract_verdict as ev

    out = tmp_path / "results"
    ev.write_verdict(
        task_id="L1_06_swiglu", cmd="resolve", run_id="r1",
        producer="tier1_reverify", out_dir=out,
        verdict="BLOCKED", verdict_class="shapes-incomplete",
        reason="missing param",
        raw_subclass="shapes-incomplete",
    )
    result = json.loads((out / "r1" / "result.json").read_text())
    assert "_provenance" not in result
    # sidecar exists
    prov = json.loads((out / "r1" / "_provenance.json").read_text())
    assert "code_fingerprint" in prov
    assert prov["raw_verdict_class"] == "shapes-incomplete"


def test_write_verdict_provenance_records_raw_subclass(tmp_path):
    from automation import extract_verdict as ev

    out = tmp_path / "results"
    ev.write_verdict(
        task_id="L1_06_swiglu", cmd="resolve", run_id="r1",
        producer="tier1_reverify", out_dir=out,
        verdict="BLOCKED", verdict_class="unexpected-blocked",
        reason="weird new reason",
        raw_subclass="something-totally-new",
    )
    prov = json.loads((out / "r1" / "_provenance.json").read_text())
    assert prov["raw_verdict_class"] == "something-totally-new"


def test_write_verdict_invalid_producer_raises(tmp_path):
    from automation import extract_verdict as ev

    out = tmp_path / "results"
    with pytest.raises(ValueError, match="producer"):
        ev.write_verdict(
            task_id="x", cmd="resolve", run_id="r1",
            producer="bogus_producer", out_dir=out,
            verdict="PASS", verdict_class=None,
            reason="x",
        )


def test_out_dir_auto_created(tmp_path):
    from automation import extract_verdict as ev

    nested = tmp_path / "a" / "b" / "c" / "results"
    ev.write_verdict(
        task_id="x", cmd="resolve", run_id="r1",
        producer="tier1_reverify", out_dir=nested,
        verdict="PASS", verdict_class=None,
        reason="ok",
    )
    assert (nested / "r1" / "DONE").exists()


def test_two_run_ids_dont_interfere(tmp_path):
    from automation import extract_verdict as ev

    out = tmp_path / "results"
    ev.write_verdict(
        task_id="L1_x", cmd="resolve", run_id="r1",
        producer="tier1_reverify", out_dir=out,
        verdict="PASS", verdict_class=None, reason="r1",
    )
    ev.write_verdict(
        task_id="L1_x", cmd="resolve", run_id="r2",
        producer="tier1_reverify", out_dir=out,
        verdict="BLOCKED", verdict_class="shapes-incomplete", reason="r2",
    )
    r1 = json.loads((out / "r1" / "result.json").read_text())
    r2 = json.loads((out / "r2" / "result.json").read_text())
    assert r1["verdict"] == "PASS"
    assert r2["verdict"] == "BLOCKED"


def test_cli_main_resolve_path(tmp_path, monkeypatch, capsys):
    from automation import extract_verdict as ev

    log = tmp_path / "pa_resolve.log"
    log.write_text("[T1] RESOLVED foo :: PASS\n", encoding="utf-8")
    out = tmp_path / "results"
    rc = ev.main([
        "--task", "T1", "--cmd", "resolve", "--run-id", "rcli",
        "--producer", "tier1_reverify",
        "--out-dir", str(out), "--resolve-log", str(log),
        "--repo-root", str(tmp_path),
    ])
    assert rc == 0
    assert (out / "rcli" / "DONE").exists()
    captured = capsys.readouterr()
    # The CLI prints a json status line; sanity-check it.
    assert "T1" in captured.out and "PASS" in captured.out


def test_validate_result_schema_rejects_extra_key():
    from automation import extract_verdict as ev

    bad = {
        "schema_version": "1.0", "run_id": "r", "task_id": "t",
        "cmd": "resolve", "verdict": "PASS", "verdict_class": None,
        "reason": "ok", "produced_at": "x", "producer": "tier1_reverify",
        "extra_key": "nope",
    }
    with pytest.raises(ValueError, match="schema violation"):
        ev.validate_result_schema(bad)
