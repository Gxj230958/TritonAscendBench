"""U5: standalone rewrite + meta.yaml dual-hash tests.

Covers: atomic temp-then-replace (no half-written kernel.py),
dual-hash semantics (upstream.file_sha256 preserved, new
phase_a_prime.standalone_sha256), provenance header extension,
jit-only launcher template raises NotImplementedError (no fake PASS).
"""
from __future__ import annotations

import hashlib
import importlib.util
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parent.parent.parent
_spec = importlib.util.spec_from_file_location(
    "phase_a_prime", REPO / "scripts" / "phase_a_prime.py"
)
pap = importlib.util.module_from_spec(_spec)
sys.modules["phase_a_prime"] = pap
_spec.loader.exec_module(pap)


def test_atomic_write_returns_correct_sha(tmp_path):
    td = tmp_path
    text = "def run(inputs):\n    return {}\n"
    sha = pap.atomic_write_kernel(td, text)
    assert (td / "kernel.py").read_text() == text
    assert sha == hashlib.sha256(text.encode()).hexdigest()
    # No leftover temp file (atomic replace).
    assert not (td / "kernel.py.tmp").exists()


def test_dual_hash_preserves_upstream_sha(tmp_path):
    """upstream.file_sha256 is original provenance -- UNCHANGED.
    phase_a_prime.standalone_sha256 is the new on-disk hash."""
    td = tmp_path
    meta = {
        "task_id": "t",
        "upstream": {
            "repo": "vllm-project/vllm",
            "commit_sha": "abc",
            "file_sha256": "ORIGINAL_UPSTREAM_SHA",
        },
    }
    (td / "meta.yaml").write_text(yaml.safe_dump(meta))
    pap.update_meta_phase_a_prime(
        td, status="PASS", reason="", s2_gpu_run_id="run-1",
        standalone_sha256="NEW_STANDALONE_SHA", frozen_deps=["h_abc/m.py"],
        class_b_symbols=[{"sym": "vllm.logger.init_logger", "proof": "noop"}],
        reference_path="native", pinned_semantic_config=None,
    )
    m = yaml.safe_load((td / "meta.yaml").read_text())
    # original provenance untouched
    assert m["upstream"]["file_sha256"] == "ORIGINAL_UPSTREAM_SHA"
    # new standalone hash present
    assert m["phase_a_prime"]["standalone_sha256"] == "NEW_STANDALONE_SHA"
    assert m["phase_a_prime"]["status"] == "PASS"
    assert m["phase_a_prime"]["s2_gpu_run_id"] == "run-1"
    assert m["frozen_deps"] == ["h_abc/m.py"]
    assert m["classB_symbols"][0]["sym"] == "vllm.logger.init_logger"
    assert m["reference_path"] == "native"
    assert not (td / "meta.yaml.tmp").exists()  # atomic


def test_standalone_header_has_phase_a_prime_block():
    h = pap.standalone_provenance_header(
        task_id="t", kernel_name="k", source_path="vllm/x.py", repo="vLLM",
        commit_sha="abc", standalone_sha256="deadbeef",
        frozen_deps=["h_abc/m.py"],
        class_b_symbols=[{"sym": "vllm.logger.init_logger", "proof": "noop sink"}],
        reference_path="triton_fallback",
        pinned_semantic_config={"rule": "upstream-default", "resolved_value": "X"},
    )
    assert h.strip().startswith('"""')
    assert h.rstrip().endswith('"""')
    assert "Phase A' standalone:" in h
    assert "reference_path: triton_fallback" in h
    assert "frozen_deps:" in h
    assert "h_abc/m.py" in h
    assert "classB_symbols:" in h
    assert "pinned_semantic_config:" in h
    assert "Commit SHA: abc" in h  # base header preserved


def test_jit_launcher_template_raises_not_pass():
    """The generated jit-only launcher is an explicit
    NotImplementedError stub -- it can NEVER masquerade as a PASS
    (plan §invariant 1: no fake completion)."""
    src = "import torch\n" + pap.JIT_LAUNCHER_TEMPLATE
    ns: dict = {}
    exec(compile(src, "<t>", "exec"), ns)
    import pytest

    with pytest.raises(NotImplementedError, match="jit-only launcher"):
        ns["run"]({})


def test_rewrite_refuses_without_s2_pass(tmp_path):
    """FIX 5: perform_standalone_rewrite must REFUSE to rewrite when
    the S2 verdict is not PASS (UNRUN/BLOCKED) -- no fake-PASS rewrite
    (plan §invariant 3)."""
    td = tmp_path
    (td / "kernel.py").write_text(
        "from vllm.triton_utils import tl, triton\n@triton.jit\n"
        "def k(x): pass\ndef run(i): return {}\n"
    )
    tc = pap.classify_task("t", (td / "kernel.py").read_text())
    unrun = pap.S2Result(task_id="t", status="UNRUN", reason="no gpu")
    ok, msg = pap.perform_standalone_rewrite(td, unrun, tc)
    assert ok is False
    assert "refusing to rewrite" in msg
    blocked = pap.S2Result(task_id="t", status="BLOCKED", reason="leak")
    ok2, _ = pap.perform_standalone_rewrite(td, blocked, tc)
    assert ok2 is False


def test_rewrite_operation_end_to_end_on_pass_fixture(tmp_path):
    """FIX 5: the rewrite OPERATION is invocable + works end-to-end
    when an S2-PASS verdict is supplied (simulated -- S2-gpu is UNRUN
    on this no-GPU host). Asserts: vllm/sglang imports rewritten to
    the standalone surface, atomic write (no .tmp leftover), dual-hash
    meta written (upstream.file_sha256 preserved, new
    standalone_sha256 == on-disk)."""
    td = tmp_path
    kernel = (
        "import torch\n"
        "from vllm.triton_utils import tl, triton\n"
        "from vllm.logger import init_logger\n"
        "logger = init_logger(__name__)\n"
        "@triton.jit\ndef k(x): pass\n"
        "def run(inputs):\n    return {}\n"
    )
    (td / "kernel.py").write_text(kernel)
    meta = {
        "task_id": "t",
        "upstream": {"repo": "vllm-project/vllm", "commit_sha": "abc",
                     "file_sha256": "ORIGINAL_UPSTREAM_SHA"},
    }
    (td / "meta.yaml").write_text(yaml.safe_dump(meta))
    tc = pap.classify_task("t", kernel)

    # Simulate the S2-gpu-authoritative PASS verdict (the only thing
    # we cannot produce here is a *real* GPU run -- the rewrite
    # OPERATION itself is fully exercised).
    s2_pass = pap.S2Result(
        task_id="t", status="PASS", reason="bit-exact",
        s2_gpu_run_id="s2-t-sim",
    )
    ok, msg = pap.perform_standalone_rewrite(td, s2_pass, tc)
    assert ok is True, msg

    new_src = (td / "kernel.py").read_text()
    # vllm/sglang imports are gone; standalone surface present.
    assert "from vllm.triton_utils import" not in new_src
    assert "from vllm.logger import" not in new_src
    assert "import triton.language as tl" in new_src
    assert "from tasks._frozen_shim import init_logger" in new_src
    assert not (td / "kernel.py.tmp").exists()  # atomic

    m = yaml.safe_load((td / "meta.yaml").read_text())
    # dual-hash: upstream preserved, new standalone == on-disk
    assert m["upstream"]["file_sha256"] == "ORIGINAL_UPSTREAM_SHA"
    disk_sha = hashlib.sha256(new_src.encode()).hexdigest()
    assert m["phase_a_prime"]["standalone_sha256"] == disk_sha
    assert m["phase_a_prime"]["status"] == "PASS"
    assert m["phase_a_prime"]["s2_gpu_run_id"] == "s2-t-sim"


def test_cmd_rewrite_invocable_and_honest_no_gpu(tmp_path, monkeypatch):
    """FIX 5: the `rewrite` subcommand is invocable end-to-end. With a
    no-GPU verdict NOTHING is rewritten (honest); with an injected
    PASS verify_fn the task IS rewritten (operation wired, not
    orphaned)."""
    import argparse

    # Build a minimal task tree the real _iter_task_dirs can find.
    lvl = tmp_path / "tasks" / "level1" / "L1_x"
    lvl.mkdir(parents=True)
    (lvl / "kernel.py").write_text(
        "import torch\nfrom vllm.triton_utils import tl, triton\n"
        "@triton.jit\ndef k(x): pass\ndef run(i): return {}\n"
    )
    (lvl / "meta.yaml").write_text(
        yaml.safe_dump({"task_id": "L1_x",
                        "upstream": {"file_sha256": "U"}})
    )
    (lvl / "shapes.yaml").write_text("public_shapes: []\n")
    monkeypatch.setattr(pap, "TASKS_ROOT", tmp_path / "tasks")

    # 1. Default (no-GPU) verdict -> UNRUN -> 0 rewritten (honest).
    monkeypatch.setattr(pap, "_cuda_available", lambda: False)
    rc = pap.cmd_rewrite(argparse.Namespace(task="L1_x"))
    assert rc == 0
    assert "vllm.triton_utils" in (lvl / "kernel.py").read_text()  # untouched

    # 2. Injected PASS verify_fn -> the rewrite OPERATION fires.
    def _fake_pass(td):
        return pap.S2Result(task_id=td.name, status="PASS",
                            reason="bit-exact", s2_gpu_run_id="rid")

    rc2 = pap.cmd_rewrite(argparse.Namespace(task="L1_x"),
                          verify_fn=_fake_pass)
    assert rc2 == 0
    assert "from vllm.triton_utils import" not in (lvl / "kernel.py").read_text()
    m = yaml.safe_load((lvl / "meta.yaml").read_text())
    assert m["phase_a_prime"]["status"] == "PASS"
