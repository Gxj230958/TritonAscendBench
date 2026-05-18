"""U2: content-addressed pool vendoring tests.

Covers: h_<sha12> prefix REQUIRED, content-addressed dedup, idempotent
re-vendor, Apache/provenance header, NOTICE update, importable in a
clean torch+triton env.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
_spec = importlib.util.spec_from_file_location(
    "phase_a_prime", REPO / "scripts" / "phase_a_prime.py"
)
pap = importlib.util.module_from_spec(_spec)
sys.modules["phase_a_prime"] = pap
_spec.loader.exec_module(pap)


def test_pool_path_has_h_prefix():
    """Package segment MUST be `h_<sha12>` (bare sha12 may start with a
    digit = invalid Python package segment)."""
    p = pap.pool_module_path("0abc12345678", "vllm.foo.bar")
    assert p.parent.name == "h_0abc12345678"
    assert p.parent.name.isidentifier()  # valid Python pkg segment
    assert p.name == "vllm_foo_bar.py"


def test_vendor_is_content_addressed_and_idempotent(tmp_path, monkeypatch):
    monkeypatch.setattr(pap, "FROZEN_DEPS_POOL", tmp_path / "_frozen_deps_pool")
    monkeypatch.setattr(pap, "REPO_ROOT", tmp_path)
    (tmp_path / "NOTICE").write_text(
        "All vendored / adapted code retains its original copyright notices.\n"
    )
    src = "def helper(x):\n    return x + 1\n"
    p1 = pap.vendor_symbol("vllm.q.utils", src, "vLLM")
    assert p1.exists()
    assert "Apache-2.0" in p1.read_text()
    assert "Commit SHA:" in p1.read_text()
    # Re-vendor identical content -> same path, written once.
    mtime = p1.stat().st_mtime_ns
    p2 = pap.vendor_symbol("vllm.q.utils", src, "vLLM")
    assert p2 == p1
    assert p1.stat().st_mtime_ns == mtime  # not rewritten


def test_vendor_distinct_content_distinct_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(pap, "FROZEN_DEPS_POOL", tmp_path / "pool")
    monkeypatch.setattr(pap, "REPO_ROOT", tmp_path)
    (tmp_path / "NOTICE").write_text("x\n")
    a = pap.vendor_symbol("m", "def a(): return 1\n", "vLLM")
    b = pap.vendor_symbol("m", "def a(): return 2\n", "vLLM")
    assert a.parent != b.parent  # different content -> different h_ dir


def test_notice_updated_idempotently(tmp_path, monkeypatch):
    monkeypatch.setattr(pap, "FROZEN_DEPS_POOL", tmp_path / "pool")
    monkeypatch.setattr(pap, "REPO_ROOT", tmp_path)
    notice = tmp_path / "NOTICE"
    notice.write_text(
        "All vendored / adapted code retains its original copyright notices.\n"
    )
    pap.vendor_symbol("m", "x = 1\n", "vLLM")
    pap.vendor_symbol("m2", "y = 2\n", "vLLM")
    text = notice.read_text()
    assert pap._NOTICE_PA_MARKER in text
    assert text.count(pap._NOTICE_PA_MARKER) == 1  # idempotent


def test_vendored_module_imports_clean(tmp_path, monkeypatch):
    """A pool entry must import in a torch+triton-only env (no
    vllm/sglang resolution)."""
    monkeypatch.setattr(pap, "FROZEN_DEPS_POOL", tmp_path / "pool")
    monkeypatch.setattr(pap, "REPO_ROOT", tmp_path)
    (tmp_path / "NOTICE").write_text("x\n")
    p = pap.vendor_symbol("m", "VALUE = 42\ndef f():\n    return VALUE\n", "vLLM")
    spec = importlib.util.spec_from_file_location("_pooltest", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.f() == 42
