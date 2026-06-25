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
    "phase_a_prime", REPO / "scripts" / "verify" / "standalone_freeze_verifier.py"
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
    assert "from ascendbench.frozen.shim import init_logger" in new_src
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


def test_rewrite_severs_batch_invariant_shim_imports():
    """The batch_invariant family (L2_13/L2_14) leaks vllm.envs,
    mem_utils.get_max_shared_memory_bytes, torch_utils.is_torch_equal_or_newer,
    and platform_utils.num_compute_units. All four redirect to the
    ascendbench.frozen.shim surface so the standalone has NO vllm import left."""
    import re
    src = (
        "import torch\n"
        "import vllm.envs as envs\n"
        "from vllm.platforms import current_platform\n"
        "from vllm.triton_utils import tl, triton\n"
        "from vllm.utils.mem_utils import get_max_shared_memory_bytes\n"
        "from vllm.utils.platform_utils import num_compute_units\n"
        "from vllm.utils.torch_utils import is_torch_equal_or_newer\n"
        "def bmm_batch_invariant(a, b):\n    return a @ b\n"
    )
    cls = pap.TaskClassification(task_id="L2_20_bmm", symbols=[])
    out = pap.rewrite_imports_to_standalone(src, cls, task_dir=None)
    assert not [l for l in out.splitlines()
                if re.search(r"^\s*(from|import)\s+vllm", l)], out
    assert "import ascendbench.frozen.shim.envs as envs" in out
    assert "from ascendbench.frozen.shim import get_max_shared_memory_bytes" in out
    assert "from ascendbench.frozen.shim import is_torch_equal_or_newer" in out
    assert "from ascendbench.frozen.shim import num_compute_units" in out


def test_rewrite_refuses_unallowlisted_torch_utils_symbol():
    """A non-allowlisted vllm.utils.torch_utils symbol is NOT silently
    shimmed -- it must surface as a leak (fail-closed), not be defaulted."""
    src = "from vllm.utils.torch_utils import some_unvetted_helper\nx = 1\n"
    out = pap.rewrite_imports_to_standalone(
        src, pap.TaskClassification(task_id="T", symbols=[]), task_dir=None
    )
    # left untouched -> the severed-import audit will catch the leak
    assert "from vllm.utils.torch_utils import some_unvetted_helper" in out


def test_envs_shim_exposes_only_verified_inert_flag():
    """ascendbench.frozen.shim.envs exposes ONLY the verified-inert
    VLLM_BATCH_INVARIANT (verbatim upstream default False). Compute-path
    semantic knobs (Class-S: VLLM_TUNED_CONFIG_FOLDER / VLLM_ROCM_FP8_PADDING)
    and any other flag LOUD-FAIL -- the loud-fail is the Class-S gate that
    keeps a task honestly BLOCKED instead of silently defaulting a knob."""
    import importlib

    import pytest
    REPO = Path(__file__).resolve().parent.parent.parent
    sys.path.insert(0, str(REPO))
    envs = importlib.import_module("ascendbench.frozen.shim.envs")
    assert envs.VLLM_BATCH_INVARIANT is False
    for semantic in ("VLLM_TUNED_CONFIG_FOLDER", "VLLM_ROCM_FP8_PADDING",
                     "VLLM_SOME_UNVETTED_FLAG"):
        with pytest.raises(AttributeError):
            getattr(envs, semantic)


def test_rewrite_severs_real_l2_13_and_l2_14_kernels():
    """Real-kernel evidence (not synthetic): rewriting the actual
    L2_20_bmm / L2_14 kernel.py leaves ZERO direct vllm imports."""
    import re
    REPO = Path(__file__).resolve().parent.parent.parent
    for tid in ("L2_20_bmm", "L2_21_matmul_kernel_persistent"):
        kps = list(REPO.glob(f"tasks/level*/{tid}/kernel.py"))
        if not kps:
            continue  # corpus task absent in this checkout
        src = kps[0].read_text()
        out = pap.rewrite_imports_to_standalone(
            src, pap.TaskClassification(task_id=tid, symbols=[]), task_dir=None
        )
        leaks = [l for l in out.splitlines()
                 if re.search(r"^\s*(from|import)\s+vllm(\.|\s|$)", l)]
        assert not leaks, f"{tid}: {leaks}"


def test_rewrite_customop_inert_shim_and_failclosed_guards():
    """CustomOp (used only by an unrelated @register class; verified entry
    is a free function) redirects to the inert shim. Fail-closed: refuse
    if other symbols share the line (PluggableLayer) or the module
    references op_registry/op_registry_oot."""
    cls = pap.TaskClassification(task_id="T", symbols=[])
    rw = lambda s: pap.rewrite_imports_to_standalone(s, cls, task_dir=None)
    ok = rw("from vllm.model_executor.custom_op import CustomOp\n"
            "@CustomOp.register('x')\nclass Y(CustomOp):\n    pass\n")
    assert "from ascendbench.frozen.shim import CustomOp" in ok
    assert "from vllm.model_executor.custom_op" not in ok
    # refuse: extra symbol
    multi = rw("from vllm.model_executor.custom_op import CustomOp, PluggableLayer\n")
    assert "import CustomOp, PluggableLayer" in multi  # left intact
    # refuse: module reads the registry
    reg = rw("from vllm.model_executor.custom_op import CustomOp\nz = op_registry['k']\n")
    assert "from vllm.model_executor.custom_op import CustomOp\n" in reg  # intact


def test_customop_shim_is_nn_module_with_noop_register():
    from pathlib import Path

    import torch
    REPO = Path(__file__).resolve().parent.parent.parent
    sys.path.insert(0, str(REPO))
    import importlib
    shim = importlib.import_module("ascendbench.frozen.shim")
    assert issubclass(shim.CustomOp, torch.nn.Module)

    @shim.CustomOp.register("op_x")
    class _Demo(shim.CustomOp):
        pass

    assert _Demo.name == "op_x"  # register sets .name and returns the class


def _write_min_meta(td: Path, entrypoint: str = "kernel.py::k") -> None:
    (td / "meta.yaml").write_text(
        yaml.safe_dump(
            {
                "oracle": {"gpu_triton": entrypoint},
                "interface": {"param_name_map": {}},
            }
        )
    )


def test_staged_candidate_copies_task_local_relative_siblings(tmp_path):
    """A staged severed candidate that still imports a task-local sibling
    via ``from .X import`` must have ``X.py`` copied into the staging dir,
    so the severed package (__path__=[staging_dir]) resolves it instead of
    crashing with ``ModuleNotFoundError: No module named
    '_pa_pkg__pa_kernel.base'`` (the L1_31 regression)."""
    td = tmp_path / "task"
    td.mkdir()
    # Sibling stub the kernel imports at module scope (off the graded path).
    (td / "base.py").write_text(
        "import torch.nn as nn\n\n\nclass RotaryEmbeddingBase(nn.Module):\n"
        "    pass\n"
    )
    (td / "kernel.py").write_text(
        "import torch\n"
        "from vllm.triton_utils import tl, triton\n"
        "from .base import RotaryEmbeddingBase\n\n\n"
        "def k(x):\n    return x\n"
    )
    _write_min_meta(td)
    tc = pap.classify_task("task", (td / "kernel.py").read_text())
    staged = pap.build_staged_candidate(td, tc)
    # The relative import survives the rewrite (no pool/shim rule matches a
    # task-local sibling) ...
    assert "from .base import RotaryEmbeddingBase" in staged.read_text()
    # ... so its source file must be staged alongside kernel.py.
    assert (staged.parent / "base.py").is_file()
    assert (staged.parent / "base.py").read_text() == (
        td / "base.py"
    ).read_text()


def test_staged_candidate_follows_transitive_sibling_closure(tmp_path):
    """Staging the sibling closure is transitive: if a copied sibling itself
    imports another task-local sibling, that one is staged too (else the bug
    just moves one hop)."""
    td = tmp_path / "task"
    td.mkdir()
    (td / "leaf.py").write_text("VALUE = 1\n")
    (td / "base.py").write_text("from .leaf import VALUE\n")
    (td / "kernel.py").write_text(
        "import torch\n"
        "from vllm.triton_utils import tl, triton\n"
        "from .base import VALUE\n\n\n"
        "def k(x):\n    return x\n"
    )
    _write_min_meta(td)
    tc = pap.classify_task("task", (td / "kernel.py").read_text())
    staged = pap.build_staged_candidate(td, tc)
    assert (staged.parent / "base.py").is_file()
    assert (staged.parent / "leaf.py").is_file()


def test_staged_candidate_does_not_synthesise_missing_sibling(tmp_path):
    """Fail-closed: a relative import with NO task-local file is left to
    surface as the real failure -- nothing is fabricated into staging."""
    td = tmp_path / "task"
    td.mkdir()
    (td / "kernel.py").write_text(
        "import torch\n"
        "from vllm.triton_utils import tl, triton\n"
        "from .nonexistent import X\n\n\n"
        "def k(x):\n    return x\n"
    )
    _write_min_meta(td)
    tc = pap.classify_task("task", (td / "kernel.py").read_text())
    staged = pap.build_staged_candidate(td, tc)
    assert not (staged.parent / "nonexistent.py").exists()


def test_staged_candidate_ignores_upward_relative_import(tmp_path):
    """Level-2 (``from ..X``) imports are NOT followed: the severed loader
    synthesises only a single-dir package, so an upward import cannot be
    satisfied by a flat copy and must surface as the real failure
    (fail-closed)."""
    td = tmp_path / "task"
    td.mkdir()
    (td / "base.py").write_text("X = 1\n")
    (td / "kernel.py").write_text(
        "import torch\n"
        "from vllm.triton_utils import tl, triton\n"
        "from ..base import X\n\n\n"
        "def k(x):\n    return x\n"
    )
    _write_min_meta(td)
    tc = pap.classify_task("task", (td / "kernel.py").read_text())
    staged = pap.build_staged_candidate(td, tc)
    # base.py lives in task_dir but the import is level-2 (../base) which a
    # flat staging copy cannot satisfy -> NOT staged (fail-closed).
    assert not (staged.parent / "base.py").exists()
