"""SSH local-port-forward + tunnel supervisor (Unit 1.4).

Wraps ``ssh -N -L <local>:127.0.0.1:<remote> -o ControlMaster=no ...`` in a
``subprocess.Popen(start_new_session=True)`` process, plus a supervisor
thread that probes ``http://localhost:<local>/<health_path>`` every
``probe_interval_s`` seconds and rebuilds the tunnel after a few
consecutive failures.

HTTP probe semantics:
    requests.get(..., timeout=5)
      status 2xx / 4xx     -> alive (server is reachable -- /health may
                              be 404 if remote does not implement it)
      status 5xx           -> dead
      requests.Timeout     -> dead
      requests.ConnectionError -> dead
"""
from __future__ import annotations

import logging
import socket
import subprocess
import threading
import time

import requests

logger = logging.getLogger("automation.lib.ssh_tunnel")

DEFAULT_PROBE_INTERVAL_S = 30
DEFAULT_FAIL_THRESHOLD = 3
DEFAULT_HEALTH_PATH = "/health"


def is_listening(local_port: int, host: str = "127.0.0.1", timeout: float = 2.0) -> bool:
    try:
        with socket.create_connection((host, local_port), timeout=timeout):
            return True
    except (TimeoutError, OSError):
        return False


class SSHTunnel:
    """Local-port-forward SSH tunnel with health-probe supervisor.

    Example::

        with SSHTunnel(local_port=18080, remote_host="C.example",
                       remote_port=8080, ssh_user="bench") as t:
            response = requests.post(f"http://localhost:{t.local_port}/v1/...")
    """

    def __init__(
        self,
        local_port: int,
        remote_host: str,
        remote_port: int,
        ssh_user: str,
        ssh_key_path: str | None = None,
        health_path: str = DEFAULT_HEALTH_PATH,
        probe_interval_s: int = DEFAULT_PROBE_INTERVAL_S,
        fail_threshold: int = DEFAULT_FAIL_THRESHOLD,
    ) -> None:
        self.local_port = int(local_port)
        self.remote_host = remote_host
        self.remote_port = int(remote_port)
        self.ssh_user = ssh_user
        self.ssh_key_path = ssh_key_path
        self.health_path = health_path
        self.probe_interval_s = int(probe_interval_s)
        self.fail_threshold = int(fail_threshold)

        self._proc: subprocess.Popen | None = None
        self._supervisor: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._fail_count = 0
        self._started_at: float | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def start(self) -> None:
        self._stop_event.clear()
        self._spawn_ssh()
        self._supervisor = threading.Thread(
            target=self._supervisor_loop,
            name=f"ssh-tunnel-supervisor-{self.local_port}",
            daemon=True,
        )
        self._supervisor.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._proc is not None and self._proc.poll() is None:
            try:
                self._proc.terminate()
                try:
                    self._proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self._proc.kill()
                    self._proc.wait(timeout=5)
            except Exception:  # pragma: no cover
                pass
        self._proc = None

    def __enter__(self) -> SSHTunnel:
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.stop()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _spawn_ssh(self) -> None:
        cmd = self._build_cmd()
        logger.debug("spawning ssh tunnel: %s", " ".join(cmd))
        self._proc = subprocess.Popen(
            cmd,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        self._started_at = time.time()
        self._fail_count = 0

    def _build_cmd(self) -> list[str]:
        cmd = [
            "ssh",
            "-N",  # no remote command, only forwarding
            "-L", f"{self.local_port}:127.0.0.1:{self.remote_port}",
            "-o", "ControlMaster=no",
            "-o", "ServerAliveInterval=30",
            "-o", "StrictHostKeyChecking=accept-new",
            "-o", "ExitOnForwardFailure=yes",
        ]
        if self.ssh_key_path:
            cmd += ["-i", self.ssh_key_path]
        cmd += [f"{self.ssh_user}@{self.remote_host}"]
        return cmd

    def _probe_once(self) -> bool:
        url = f"http://localhost:{self.local_port}{self.health_path}"
        try:
            r = requests.get(url, timeout=5)
        except (requests.Timeout, requests.ConnectionError):
            return False
        if 500 <= r.status_code < 600:
            return False
        return True

    def _supervisor_loop(self) -> None:
        while not self._stop_event.wait(self.probe_interval_s):
            if self._proc is None:
                return
            if self._proc.poll() is not None:
                logger.warning("ssh process exited (rc=%s); rebuilding", self._proc.returncode)
                self._spawn_ssh()
                continue
            ok = self._probe_once()
            if ok:
                self._fail_count = 0
                continue
            self._fail_count += 1
            if self._fail_count >= self.fail_threshold:
                logger.warning(
                    "tunnel failed %d times; killing and rebuilding",
                    self._fail_count,
                )
                try:
                    if self._proc.poll() is None:
                        self._proc.terminate()
                        try:
                            self._proc.wait(timeout=5)
                        except subprocess.TimeoutExpired:
                            self._proc.kill()
                except Exception:  # pragma: no cover
                    pass
                self._spawn_ssh()


__all__ = ["SSHTunnel", "is_listening"]
