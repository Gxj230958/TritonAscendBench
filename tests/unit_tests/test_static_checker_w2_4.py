"""W2.4 — Layer E runtime audit-hook tests for static_checker.

`install_runtime_audit_hook` installs the `sys.addaudithook` runtime
fallback the candidate child uses (specs/ast_denylist.md §Layer E). The
hook itself raises `RuntimeAuditViolation` on a blocked event; the
installer is idempotent.
"""
from __future__ import annotations

import pytest

from ascendbench.integrity import static_checker as sc
from ascendbench.integrity.static_checker import (
    RuntimeAuditViolation,
    _runtime_audit_hook,
    install_runtime_audit_hook,
)


def test_install_runtime_audit_hook_is_idempotent(monkeypatch):
    """First install returns True; a second call is a no-op (False).

    The module-level guard is monkeypatched back to False so this test
    can exercise the first-install path without leaking a permanent
    audit hook into the test process (CPython has no removeaudithook).
    """
    installed = []
    monkeypatch.setattr(sc, "_AUDIT_HOOK_INSTALLED", False)
    monkeypatch.setattr(sc.sys, "addaudithook",
                        lambda fn: installed.append(fn))

    first = install_runtime_audit_hook()
    second = install_runtime_audit_hook()

    assert first is True
    assert second is False
    # addaudithook called exactly once despite two install calls.
    assert len(installed) == 1
    assert installed[0] is _runtime_audit_hook


def test_audit_hook_blocks_exec():
    """A blocked audit event raises RuntimeAuditViolation."""
    with pytest.raises(RuntimeAuditViolation):
        _runtime_audit_hook("exec", (object(),))


def test_audit_hook_blocks_subprocess_spawn():
    with pytest.raises(RuntimeAuditViolation):
        _runtime_audit_hook("subprocess.Popen", ("/bin/sh", None, None))


def test_audit_hook_blocks_denied_runtime_import():
    """A runtime `import` of a denied module is blocked."""
    with pytest.raises(RuntimeAuditViolation):
        _runtime_audit_hook("import", ("ctypes", None, (), (), ()))
    with pytest.raises(RuntimeAuditViolation):
        # submodule of a denied root is blocked too.
        _runtime_audit_hook("import", ("importlib.util", None, (), (), ()))


def test_audit_hook_allows_benign_import():
    """A benign import does not raise."""
    _runtime_audit_hook("import", ("torch", None, (), (), ()))
    _runtime_audit_hook("import", ("math", None, (), (), ()))


def test_audit_hook_ignores_unrelated_events():
    """An event that is not on the blocklist passes through silently."""
    _runtime_audit_hook("object.__getattr__", (object(), "x"))
    _runtime_audit_hook("open", ("/etc/hostname", "r", 0))
