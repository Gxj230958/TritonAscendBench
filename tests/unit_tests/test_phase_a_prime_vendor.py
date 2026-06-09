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
    "phase_a_prime", REPO / "scripts" / "verify" / "standalone_freeze_verifier.py"
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


def test_vendor_triton_helper_strips_vllm_reexport(tmp_path, monkeypatch):
    """_vendor_triton_helper_module rewrites the upstream module's
    `from vllm.triton_utils import tl, triton` re-export to public triton
    imports, so the pooled module carries no vllm reference."""
    monkeypatch.setattr(pap, "FROZEN_DEPS_POOL", tmp_path / "pool")
    monkeypatch.setattr(pap, "REPO_ROOT", tmp_path)
    (tmp_path / "NOTICE").write_text("x\n")
    src = tmp_path / "triton_helpers.py"
    src.write_text(
        "from vllm.triton_utils import tl, triton\n\n\n"
        "@triton.jit\n"
        "def fast_exp(x):\n"
        "    return tl.math.exp2(x)\n"
    )
    dotted = pap._vendor_triton_helper_module("vllm.x.triton_helpers", src, "vllm")
    assert dotted.startswith("ascendbench.frozen.deps.h_")
    assert dotted.endswith(".vllm_x_triton_helpers")
    # Locate the written pool file and assert it is vllm-free + has public triton.
    sha_dir = next(p for p in dotted.split(".") if p.startswith("h_"))
    pooled = (tmp_path / "pool" / sha_dir / "vllm_x_triton_helpers.py").read_text()
    assert "from vllm" not in pooled
    assert "import triton" in pooled
    assert "import triton.language as tl" in pooled
    assert "def fast_exp" in pooled


def test_vendor_triton_helper_missing_source_returns_none(tmp_path, monkeypatch):
    """Unreadable upstream source yields None (rewriter then falls
    through to leak, never a half-vendored pool)."""
    monkeypatch.setattr(pap, "FROZEN_DEPS_POOL", tmp_path / "pool")
    monkeypatch.setattr(pap, "REPO_ROOT", tmp_path)
    out = pap._vendor_triton_helper_module(
        "vllm.x.missing", tmp_path / "does_not_exist.py", "vllm"
    )
    assert out is None


def test_packaging_classified_class_d():
    """`packaging` is admitted as a Class-D stable PyPI dep (used by the
    mamba SSD tasks); doubtful => A, so this must be explicit."""
    assert "packaging" in pap.CLASS_D_PKGS


def test_vendor_package_atomic_and_importable(tmp_path, monkeypatch):
    """vendor_package writes a complete h_<sha>/<pkg>/ with inert __init__
    + each module; partial dirs are NOT treated as cached."""
    monkeypatch.setattr(pap, "FROZEN_DEPS_POOL", tmp_path / "pool")
    monkeypatch.setattr(pap, "REPO_ROOT", tmp_path)
    (tmp_path / "NOTICE").write_text("x\n")
    mods = {"a": "VALUE = 1\n", "b": "from .a import VALUE\nDOUBLE = VALUE * 2\n"}
    prefix = pap.vendor_package("pkg", mods, "vLLM")
    assert prefix.startswith("ascendbench.frozen.deps.h_")
    assert prefix.endswith(".pkg")
    sha_dir = next(p for p in prefix.split(".") if p.startswith("h_"))
    pkgdir = tmp_path / "pool" / sha_dir / "pkg"
    assert (pkgdir / "__init__.py").exists()
    assert (pkgdir / "a.py").exists() and (pkgdir / "b.py").exists()
    assert "Commit SHA:" in (pkgdir / "a.py").read_text()
    # Partial-write must NOT be accepted as cached: delete b.py, re-vendor
    # should rewrite it.
    (pkgdir / "b.py").unlink()
    prefix2 = pap.vendor_package("pkg", mods, "vLLM")
    assert prefix2 == prefix
    assert (pkgdir / "b.py").exists()


def test_rewrite_fla_module_fail_closed_on_unknown_vllm():
    """_rewrite_fla_module_source returns None if a module carries an
    unsanctioned vllm/sglang import (fail-closed -> no half-severed pool)."""
    bad = "from vllm.triton_utils import triton\nfrom vllm.v1.worker.x import Y\n"
    assert pap._rewrite_fla_module_source(bad) is None
    good = "from vllm.triton_utils import triton\nfrom .utils import z\nX = 1\n"
    out = pap._rewrite_fla_module_source(good)
    assert out is not None
    assert "import triton" in out
    assert "from .utils import z" in out          # relative preserved
    assert "from vllm" not in out


def test_rewrite_fla_module_tldevice_triple():
    """The 3-symbol triton_utils line expands to public triton + tl +
    libdevice (op.py's import)."""
    out = pap._rewrite_fla_module_source(
        "from vllm.triton_utils import tl, tldevice, triton\n"
    )
    assert "import triton\n" in out
    assert "import triton.language as tl\n" in out
    assert "import triton.language.extra.libdevice as tldevice\n" in out


def test_pool_import_re_captures_module_paths():
    """_POOL_IMPORT_RE captures the exact shipped pool MODULE dotted-path
    from both vendoring emitters (package submodule + single-file helper),
    multiline `from ... import (` forms, and bare `import ...`."""
    src = (
        "import torch\n"
        "from ascendbench.frozen.deps.h_a9f952fcc731.fla_ops.index import x\n"
        "from ascendbench.frozen.deps.h_a9f952fcc731.fla_ops.utils import (a,\n"
        "    b)\n"
        "from ascendbench.frozen.deps.h_119bf34e8937.vllm_helpers import fast_exp\n"
        "import ascendbench.frozen.deps.h_deadbeef1234.m as sm\n"
        "from vllm.triton_utils import tl\n"
    )
    got = sorted(set(pap._POOL_IMPORT_RE.findall(src)))
    assert got == [
        "ascendbench.frozen.deps.h_119bf34e8937.vllm_helpers",
        "ascendbench.frozen.deps.h_a9f952fcc731.fla_ops.index",
        "ascendbench.frozen.deps.h_a9f952fcc731.fla_ops.utils",
        "ascendbench.frozen.deps.h_deadbeef1234.m",
    ]


def test_promote_frozen_deps_is_shipped_not_classification(tmp_path):
    """promote_staged_candidate records frozen_deps = the pool modules the
    standalone source ACTUALLY imports, NOT the (potentially huge/noisy)
    CLASS_A classification closure. Regression guard for the provenance
    fix: a task whose classifier walked a big vllm.* closure but whose
    rewrite only pooled one FLA module must record only that module."""
    task_dir = tmp_path / "tasks" / "level3" / "L3_demo"
    task_dir.mkdir(parents=True)
    (task_dir / "meta.yaml").write_text(
        "task_id: L3_demo\nupstream:\n  file_sha256: deadbeef\n"
    )
    # Staged standalone: imports ONE pooled module; classification claims
    # several unrelated CLASS_A symbols (the noise we must NOT record).
    staged = tmp_path / "staged_kernel.py"
    staged.write_text(
        "import torch\n"
        "from ascendbench.frozen.deps.h_a9f952fcc731.fla_ops.index "
        "import prepare_chunk_indices\n"
        "def run(inputs):\n    return {}\n"
    )
    classification = pap.TaskClassification(
        task_id="L3_demo",
        symbols=[
            pap.SymbolClass(".index.prepare_chunk_indices", ".index",
                            "prepare_chunk_indices", pap.CLASS_A),
            pap.SymbolClass("vllm.config.VllmConfig", "vllm.config",
                            "VllmConfig", pap.CLASS_A),
            pap.SymbolClass("vllm._C.foo", "vllm._C", "foo", pap.CLASS_A),
        ],
    )
    s2 = pap.S2Result(
        task_id="L3_demo", status="PASS", reason="bit-exact",
        s2_gpu_run_id="s2-demo", audit_leaks=[],
        s2_original_upstream_closure=[],
    )
    ok, _ = pap.promote_staged_candidate(staged, task_dir, s2, classification)
    assert ok
    import yaml as _yaml
    meta = _yaml.safe_load((task_dir / "meta.yaml").read_text())
    # ONLY the shipped pool module is recorded -- no vllm.config / vllm._C noise.
    assert meta["frozen_deps"] == [
        "ascendbench.frozen.deps.h_a9f952fcc731.fla_ops.index"
    ]
