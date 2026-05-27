"""Unit tests for V9 plan M2.0 -- R19 verified-runner boundary (v1).

Covers:

* Layer A static-checker deny: candidate source importing
  ``safetensors`` / ``torch.load`` / ``torch.serialization`` is rejected.
* Daemon startup self-check: refuse-to-start unless the process is
  running as the provider uid OR has CAP_SETUID. Explicit opt-out via
  ``$TAB_R19_SELF_CHECK_DISABLE=1`` is honoured (loudly).
"""
from __future__ import annotations

import os
from unittest.mock import patch

import pytest

from triton_ascend_bench.static_checker import (
    _DENYLIST_IMPORT_EXACT,
    check_solution,
)
from scripts.automation.bench_evaluator_daemon import _r19_startup_self_check


# ─────────────────────────────────────────────────────────────────────
# Layer A -- static_checker denylist additions
# ─────────────────────────────────────────────────────────────────────


def test_denylist_includes_safetensors_load():
    """The full V9 M2.0 deny set is wired in."""
    expected_additions = {
        "safetensors",
        "safetensors.torch",
        "safetensors.torch.load_file",
        "safetensors.torch.save_file",
        "torch.load",
        "torch.serialization",
        "torch.serialization.load",
    }
    missing = expected_additions - _DENYLIST_IMPORT_EXACT
    assert not missing, f"V9 M2.0 deny entries missing: {missing}"


def _solution_src(import_line: str) -> str:
    """Wrap a single import line into a complete candidate solution."""
    return (
        f"{import_line}\n"
        "import torch\n"
        "\n"
        "def solution(*args, **kwargs):\n"
        "    return None\n"
    )


def _findings_msgs(report) -> list[str]:
    return [f.message for f in report.findings]


def test_static_checker_rejects_safetensors_import():
    src = _solution_src("import safetensors")
    report = check_solution(src)
    assert report.blocked, "expected static_checker to block `import safetensors`"
    assert any("safetensors" in m for m in _findings_msgs(report))


def test_static_checker_rejects_safetensors_torch_load_file():
    src = _solution_src("from safetensors.torch import load_file")
    report = check_solution(src)
    assert report.blocked, (
        "expected static_checker to block "
        "`from safetensors.torch import load_file`"
    )
    assert any("safetensors" in m for m in _findings_msgs(report))


def test_static_checker_rejects_torch_load():
    src = _solution_src("from torch import load as _l")
    report = check_solution(src)
    assert report.blocked, "expected static_checker to block `from torch import load`"


def test_static_checker_allows_plain_math_import():
    """The historical baseline must keep working -- `import math` is allowed."""
    src = _solution_src("import math")
    report = check_solution(src)
    # `import math` is harmless; assert no V9 M2.0 deny finding specifically.
    safetensors_findings = [
        m for m in _findings_msgs(report)
        if "safetensors" in m or "torch.load" in m
    ]
    assert safetensors_findings == []


# ─────────────────────────────────────────────────────────────────────
# Daemon startup self-check -- fail-closed on uid mismatch
# ─────────────────────────────────────────────────────────────────────


def test_self_check_opt_out_via_env():
    """$TAB_R19_SELF_CHECK_DISABLE=1 makes the check pass with a warning."""
    with patch.dict(os.environ, {"TAB_R19_SELF_CHECK_DISABLE": "1"}, clear=False):
        ok, reason = _r19_startup_self_check()
        assert ok is True
        assert "disabled" in reason.lower()


def test_self_check_passes_when_uid_matches_provider():
    """uid match -> pass."""
    current_uid = os.geteuid()
    env = {
        "TAB_PROVIDER_UID": str(current_uid),
    }
    env.pop("TAB_R19_SELF_CHECK_DISABLE", None)
    with patch.dict(os.environ, env, clear=False):
        os.environ.pop("TAB_R19_SELF_CHECK_DISABLE", None)
        ok, reason = _r19_startup_self_check()
    assert ok is True
    assert "provider uid" in reason


def test_self_check_fails_when_uid_mismatch_and_no_cap_setuid():
    """uid mismatch + no CAP_SETUID -> refuse to start.

    The current process is unprivileged (dev machine, uid != 0,
    no CAP_SETUID in the user-level env), so setting
    $TAB_PROVIDER_UID to a NEVER-equal value reliably triggers the
    failure path.
    """
    # Pick a uid the current process cannot possibly be.
    impossible_uid = (os.geteuid() + 1) if os.geteuid() != 0 else 65534
    env = {"TAB_PROVIDER_UID": str(impossible_uid)}
    env.pop("TAB_R19_SELF_CHECK_DISABLE", None)
    with patch.dict(os.environ, env, clear=False):
        os.environ.pop("TAB_R19_SELF_CHECK_DISABLE", None)
        ok, reason = _r19_startup_self_check()
    # Reason mentions either the uid mismatch OR the missing CAP_SETUID.
    assert ok is False
    assert "R19" in reason


def test_self_check_rejects_garbled_provider_uid():
    """$TAB_PROVIDER_UID=not_a_number -> fail-closed."""
    env = {"TAB_PROVIDER_UID": "not_a_number"}
    env.pop("TAB_R19_SELF_CHECK_DISABLE", None)
    with patch.dict(os.environ, env, clear=False):
        os.environ.pop("TAB_R19_SELF_CHECK_DISABLE", None)
        ok, reason = _r19_startup_self_check()
    assert ok is False
    assert "not an integer" in reason
