"""Tests for ``automation.tier1_reverify`` (Unit 1.8).

Mocks ``subprocess.run`` to avoid invoking the real resolver / verifier
binaries; verifies the wrapper composes the right command + properly
invokes ``extract_verdict`` to materialise the 9-field result.json.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

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


def _fake_run_writes(log_text: str, returncode: int = 0):
    """Return a callable that mimics ``subprocess.run`` by writing
    ``log_text`` to its ``stdout`` file handle and returning rc."""

    def _run(cmd, stdout=None, stderr=None, timeout=None, **kw):
        if hasattr(stdout, "write"):
            stdout.write(log_text)
            stdout.flush()
        cp = MagicMock()
        cp.returncode = returncode
        cp.args = cmd
        return cp

    return _run


def test_resolve_happy_writes_result(tmp_path):
    from automation import tier1_reverify

    resolve_log = tmp_path / "pa_resolve.log"
    resolve_log.write_text("[L1_99_demo] RESOLVED demo_fn :: PASS\n", encoding="utf-8")
    out_dir = tmp_path / "results"

    fake = _fake_run_writes("[L1_99_demo] PASS\n", returncode=0)
    with patch.object(subprocess, "run", side_effect=fake):
        rc = tier1_reverify.main([
            "--task", "L1_99_demo",
            "--cmd", "resolve",
            "--out-dir", str(out_dir),
            "--run-id", "r1",
            "--resolve-log", str(resolve_log),
            "--repo-root", str(tmp_path),
        ])
    assert rc == 0
    run_dir = out_dir / "r1"
    assert (run_dir / "result.json").is_file()
    assert (run_dir / "DONE").is_file()
    assert (run_dir / "log.txt").is_file()
    result = json.loads((run_dir / "result.json").read_text())
    assert result["verdict"] == "PASS"
    assert result["cmd"] == "resolve"
    assert result["producer"] == "tier1_reverify"


def test_phase_a_verify_happy(tmp_path):
    from automation import tier1_reverify

    # Set up tasks/L2/.../meta.yaml so extract_verdict picks it up
    td = tmp_path / "tasks" / "L2" / "L2_77_demo"
    td.mkdir(parents=True)
    (td / "meta.yaml").write_text(
        yaml.safe_dump({"phase_a_prime": {"status": "PASS", "reason": "ok"}}),
        encoding="utf-8",
    )
    out_dir = tmp_path / "results"
    fake = _fake_run_writes("verify ok\n", returncode=0)
    with patch.object(subprocess, "run", side_effect=fake):
        rc = tier1_reverify.main([
            "--task", "L2_77_demo",
            "--cmd", "phase_a_verify",
            "--out-dir", str(out_dir),
            "--run-id", "r2",
            "--repo-root", str(tmp_path),
        ])
    assert rc == 0
    result = json.loads((out_dir / "r2" / "result.json").read_text())
    assert result["verdict"] == "PASS"
    assert result["cmd"] == "phase_a_verify"


def test_subprocess_timeout_yields_blocked(tmp_path):
    from automation import tier1_reverify

    def raise_timeout(cmd, **kw):
        raise subprocess.TimeoutExpired(cmd=cmd, timeout=kw.get("timeout", 0))

    out_dir = tmp_path / "results"
    with patch.object(subprocess, "run", side_effect=raise_timeout):
        rc = tier1_reverify.main([
            "--task", "L1_99_demo",
            "--cmd", "resolve",
            "--out-dir", str(out_dir),
            "--run-id", "r1",
            "--timeout", "1",
            "--repo-root", str(tmp_path),
        ])
    assert rc == 1
    result = json.loads((out_dir / "r1" / "result.json").read_text())
    assert result["verdict"] == "BLOCKED"
    assert "timeout" in result["reason"].lower()


def test_subprocess_nonzero_with_blocked_log(tmp_path):
    """sub-cmd rc != 0 but extract_verdict still derives a verdict."""
    from automation import tier1_reverify

    log = tmp_path / "pa_resolve.log"
    log.write_text(
        "[T1] BLOCKED - :: BLOCKED(no-verified-upstream-entrypoint: shapes-incomplete)\n",
        encoding="utf-8",
    )
    out_dir = tmp_path / "results"
    fake = _fake_run_writes("blocked\n", returncode=2)
    with patch.object(subprocess, "run", side_effect=fake):
        rc = tier1_reverify.main([
            "--task", "T1",
            "--cmd", "resolve",
            "--out-dir", str(out_dir),
            "--run-id", "r1",
            "--resolve-log", str(log),
            "--repo-root", str(tmp_path),
        ])
    assert rc == 0
    result = json.loads((out_dir / "r1" / "result.json").read_text())
    assert result["verdict"] == "BLOCKED"
    assert result["verdict_class"] == "shapes-incomplete"


def test_invalid_cmd_choice_rejected(tmp_path, capsys):
    from automation import tier1_reverify

    with pytest.raises(SystemExit):
        tier1_reverify.main([
            "--task", "T1",
            "--cmd", "banana",
            "--out-dir", str(tmp_path),
        ])


def test_default_run_id_minted(tmp_path):
    from automation import tier1_reverify

    log = tmp_path / "pa_resolve.log"
    log.write_text("[T1] RESOLVED demo :: PASS\n", encoding="utf-8")
    fake = _fake_run_writes("ok\n", returncode=0)
    with patch.object(subprocess, "run", side_effect=fake):
        rc = tier1_reverify.main([
            "--task", "T1",
            "--cmd", "resolve",
            "--out-dir", str(tmp_path / "results"),
            "--resolve-log", str(log),
            "--repo-root", str(tmp_path),
        ])
    assert rc == 0
    # exactly one run_id directory was created
    results = list((tmp_path / "results").iterdir())
    assert len(results) == 1
    assert results[0].name.startswith("tier1-")


def test_sub_command_built_with_sys_executable(tmp_path):
    """The sub_cmd must use sys.executable (not raw 'python3')."""
    from automation import tier1_reverify

    log = tmp_path / "pa_resolve.log"
    log.write_text("[T1] RESOLVED demo :: PASS\n", encoding="utf-8")
    captured_cmds = []

    def capture(cmd, stdout=None, stderr=None, timeout=None, **kw):
        captured_cmds.append(cmd)
        if hasattr(stdout, "write"):
            stdout.write("ok\n")
        cp = MagicMock()
        cp.returncode = 0
        return cp

    with patch.object(subprocess, "run", side_effect=capture):
        tier1_reverify.main([
            "--task", "T1",
            "--cmd", "resolve",
            "--out-dir", str(tmp_path / "results"),
            "--run-id", "r1",
            "--resolve-log", str(log),
            "--repo-root", str(tmp_path),
        ])
    assert len(captured_cmds) >= 1
    # First argument must be sys.executable
    assert captured_cmds[0][0] == sys.executable


def test_e2e_three_files_created(tmp_path):
    from automation import tier1_reverify

    log = tmp_path / "pa_resolve.log"
    log.write_text("[T1] RESOLVED demo :: PASS\n", encoding="utf-8")
    fake = _fake_run_writes("ok\n", returncode=0)
    with patch.object(subprocess, "run", side_effect=fake):
        tier1_reverify.main([
            "--task", "T1",
            "--cmd", "resolve",
            "--out-dir", str(tmp_path / "results"),
            "--run-id", "r1",
            "--resolve-log", str(log),
            "--repo-root", str(tmp_path),
        ])
    rd = tmp_path / "results" / "r1"
    assert (rd / "result.json").exists()
    assert (rd / "_provenance.json").exists()
    assert (rd / "DONE").exists()
    assert (rd / "log.txt").exists()
