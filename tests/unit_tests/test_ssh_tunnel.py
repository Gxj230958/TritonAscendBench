"""Tests for ``automation.lib.ssh_tunnel`` (Unit 1.4).

Uses ``unittest.mock`` to stub ``subprocess.Popen`` and ``requests.get``
so no real ssh process is spawned.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
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


def _fake_popen(alive: bool = True):
    proc = MagicMock()
    proc.poll.return_value = None if alive else 0
    proc.returncode = None if alive else 0
    proc.pid = 99999
    return proc


def test_build_cmd_includes_required_flags():
    from automation.lib import ssh_tunnel

    t = ssh_tunnel.SSHTunnel(
        local_port=18080, remote_host="C.host", remote_port=8080,
        ssh_user="bench",
    )
    cmd = t._build_cmd()
    assert "-N" in cmd
    assert "-L" in cmd
    assert "18080:127.0.0.1:8080" in cmd
    # ControlMaster=no is the critical flag (KTD-3) -- it shows up as
    # the value of a -o argument.
    idx = cmd.index("ControlMaster=no")
    assert cmd[idx - 1] == "-o"
    assert "bench@C.host" in cmd


def test_start_spawns_popen_with_new_session():
    from automation.lib import ssh_tunnel

    with patch.object(ssh_tunnel, "subprocess") as sp:
        sp.Popen.return_value = _fake_popen()
        sp.DEVNULL = -3  # match real value enough for kwargs
        t = ssh_tunnel.SSHTunnel(
            local_port=18080, remote_host="C", remote_port=8080, ssh_user="b",
            probe_interval_s=999,
        )
        t.start()
        kwargs = sp.Popen.call_args.kwargs
        assert kwargs.get("start_new_session") is True
        t.stop()


def test_probe_alive_200(monkeypatch):
    from automation.lib import ssh_tunnel

    t = ssh_tunnel.SSHTunnel(18080, "C", 8080, "b", probe_interval_s=999)

    class FakeResp:
        status_code = 200

    monkeypatch.setattr(ssh_tunnel.requests, "get", lambda *a, **k: FakeResp())
    assert t._probe_once() is True


def test_probe_alive_404_still_ok(monkeypatch):
    """404 means server reachable but path missing -- not a tunnel failure."""
    from automation.lib import ssh_tunnel

    t = ssh_tunnel.SSHTunnel(18080, "C", 8080, "b", probe_interval_s=999)

    class FakeResp:
        status_code = 404

    monkeypatch.setattr(ssh_tunnel.requests, "get", lambda *a, **k: FakeResp())
    assert t._probe_once() is True


def test_probe_dead_5xx(monkeypatch):
    from automation.lib import ssh_tunnel

    t = ssh_tunnel.SSHTunnel(18080, "C", 8080, "b", probe_interval_s=999)

    class FakeResp:
        status_code = 503

    monkeypatch.setattr(ssh_tunnel.requests, "get", lambda *a, **k: FakeResp())
    assert t._probe_once() is False


def test_probe_dead_timeout(monkeypatch):
    from automation.lib import ssh_tunnel

    t = ssh_tunnel.SSHTunnel(18080, "C", 8080, "b", probe_interval_s=999)

    def _raise(*a, **k):
        raise ssh_tunnel.requests.Timeout("timeout")

    monkeypatch.setattr(ssh_tunnel.requests, "get", _raise)
    assert t._probe_once() is False


def test_probe_dead_connection_refused(monkeypatch):
    from automation.lib import ssh_tunnel

    t = ssh_tunnel.SSHTunnel(18080, "C", 8080, "b", probe_interval_s=999)

    def _raise(*a, **k):
        raise ssh_tunnel.requests.ConnectionError("refused")

    monkeypatch.setattr(ssh_tunnel.requests, "get", _raise)
    assert t._probe_once() is False


def test_supervisor_rebuilds_after_threshold(monkeypatch):
    from automation.lib import ssh_tunnel

    proc_alive = _fake_popen(alive=True)
    calls = {"spawn": 0}

    def fake_popen(*a, **k):
        calls["spawn"] += 1
        return proc_alive

    with patch.object(ssh_tunnel.subprocess, "Popen", side_effect=fake_popen):
        # All probes return dead -> threshold reached -> rebuild.
        monkeypatch.setattr(
            ssh_tunnel.requests, "get",
            lambda *a, **k: (_ for _ in ()).throw(ssh_tunnel.requests.ConnectionError("x")),
        )
        t = ssh_tunnel.SSHTunnel(
            18080, "C", 8080, "b",
            probe_interval_s=0,  # tight loop
            fail_threshold=2,
        )
        t.start()
        time.sleep(0.3)
        t.stop()
        # We expect at least one rebuild beyond the initial spawn
        assert calls["spawn"] >= 2


def test_context_manager_calls_stop():
    from automation.lib import ssh_tunnel

    with patch.object(ssh_tunnel.subprocess, "Popen") as popen:
        popen.return_value = _fake_popen()
        with ssh_tunnel.SSHTunnel(18080, "C", 8080, "b", probe_interval_s=999) as t:
            assert t._proc is not None
        # After exit, _proc should be cleared
        assert t._proc is None


def test_is_listening_false_when_port_closed():
    from automation.lib import ssh_tunnel

    # Pick an almost-certainly-closed high port
    assert ssh_tunnel.is_listening(1, timeout=0.5) is False
