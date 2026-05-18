"""U1: symbol-level reachable-closure classifier tests.

Covers the plan's enumerated scenarios: T0 happy path, tl/triton
re-export -> B-rewrite-public, relative-import -> A, VLLM_TUNED -> S,
torch.ops._C + no triton -> BLOCKED, side-effect-only top-level NOT
dropped, cycle / depth -> T3 (no silent truncate), no third
"silently dropped" state.
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


def test_t0_no_external_imports():
    """A torch+triton-only kernel -> tier T0, zero SEVERABLE symbols.

    torch/triton imports are now recorded as CLASS_TORCH_TRITON (so a
    symbol is NEVER silently dropped, plan FIX 4) but they are
    handled-and-non-severable, so the tier is still T0 and there is
    nothing to vendor/shim/pin."""
    src = (
        "import torch\nimport triton\nimport triton.language as tl\n"
        "@triton.jit\ndef k(x): pass\n"
        "def run(inputs): return {}\n"
    )
    tc = pap.classify_task("t0", src)
    assert tc.tier == "T0"
    severable = [s for s in tc.symbols if s.cls != pap.CLASS_TORCH_TRITON]
    assert severable == []
    assert tc.closure_size == 0
    # torch/triton ARE visible (not silently dropped) and explicitly
    # classified as the handled torch+triton category.
    assert all(s.cls == pap.CLASS_TORCH_TRITON for s in tc.symbols)
    assert not tc.blocked


def test_tl_triton_reexport_is_b_rewrite_public():
    """`from vllm.triton_utils import tl, triton` -> B-rewrite-public
    with a proof; NOT shimmed, NOT Class-A."""
    src = (
        "import torch\nfrom vllm.triton_utils import tl, triton\n"
        "@triton.jit\ndef k(x): pass\n"
        "def run(inputs): return {}\n"
    )
    tc = pap.classify_task("reexport", src)
    syms = {s.name: s for s in tc.symbols}
    assert syms["tl"].cls == pap.CLASS_B_REWRITE_PUBLIC
    assert syms["triton"].cls == pap.CLASS_B_REWRITE_PUBLIC
    assert syms["tl"].proof  # proof string required for B
    assert "public" in syms["tl"].proof


def test_has_triton_gate_stays_class_a():
    """`HAS_TRITON` from the same module GATES behaviour -> Class A,
    never B (plan: triton_utils not wholesale shimmed)."""
    src = (
        "import torch\nfrom vllm.triton_utils import HAS_TRITON\n"
        "@triton.jit\ndef k(x): pass\ndef run(i): return {}\n"
    )
    tc = pap.classify_task("hastriton", src)
    s = {x.name: x for x in tc.symbols}["HAS_TRITON"]
    assert s.cls == pap.CLASS_A


def test_relative_import_is_class_a():
    """`from ..utils import input_guard` -> Class A (resolves into
    upstream; relative import is never B)."""
    src = (
        "import torch\nfrom ..utils import input_guard\n"
        "@triton.jit\ndef k(x): pass\ndef run(i): return {}\n"
    )
    tc = pap.classify_task("rel", src)
    rel = [s for s in tc.symbols if s.name == "input_guard"]
    assert rel and rel[0].cls == pap.CLASS_A


def test_vllm_envs_is_class_s():
    """`from vllm import envs` / VLLM_TUNED_CONFIG_FOLDER -> Class S
    (env/file-driven config), NOT B."""
    src = (
        "import torch\nfrom vllm.envs import VLLM_TUNED_CONFIG_FOLDER\n"
        "@triton.jit\ndef k(x): pass\ndef run(i): return {}\n"
    )
    tc = pap.classify_task("envs", src)
    s = {x.name: x for x in tc.symbols}["VLLM_TUNED_CONFIG_FOLDER"]
    assert s.cls == pap.CLASS_S


def test_class_c_no_triton_path_is_blocked():
    """torch.ops._C accel + NO @triton.jit anywhere -> BLOCKED
    (no-triton-fallback ⇒ BLOCKED, plan §invariant 4)."""
    src = (
        "import torch\nfrom vllm._custom_ops import scaled_fp8\n"
        "def run(i):\n    return {'y': torch.ops._C.foo(i['x'])}\n"
    )
    tc = pap.classify_task("accelonly", src)
    assert tc.blocked
    assert "no reachable triton fallback" in tc.blocked_reason.lower()


def test_side_effect_only_toplevel_not_dropped():
    """A bare module-scope side-effect statement (logger=init_logger,
    current_platform read) MUST enter the closure even though no
    entrypoint references it (plan U1 FIX)."""
    src = (
        "import torch\n"
        "from vllm.logger import init_logger\n"
        "from vllm.platforms import current_platform\n"
        "logger = init_logger(__name__)\n"
        "FP8_DTYPE = current_platform.fp8_dtype()\n"
        "@triton.jit\ndef k(x): pass\ndef run(i): return {}\n"
    )
    tc = pap.classify_task("sideeffect", src)
    names = {s.name: s for s in tc.symbols}
    assert "init_logger" in names
    assert names["init_logger"].cls == pap.CLASS_B_INERT_SHIM
    assert "current_platform" in names
    assert names["current_platform"].cls == pap.CLASS_A  # platform read -> A


def test_no_third_silently_dropped_state():
    """Every external symbol lands in exactly one of A/B/C/S/D -- there
    is no undecidable / silently-dropped third state."""
    src = (
        "import torch\nfrom vllm.logger import init_logger\n"
        "from vllm.triton_utils import tl, triton\n"
        "from vllm.platforms import current_platform\n"
        "@triton.jit\ndef k(x): pass\ndef run(i): return {}\n"
    )
    tc = pap.classify_task("alltyped", src)
    valid = {
        pap.CLASS_A, pap.CLASS_B_REWRITE_PUBLIC, pap.CLASS_B_INERT_SHIM,
        pap.CLASS_C, pap.CLASS_S, pap.CLASS_D, pap.CLASS_TORCH_TRITON,
    }
    for s in tc.symbols:
        assert s.cls in valid, f"{s.symbol} got undecidable class {s.cls!r}"


def test_einops_is_class_d_not_silently_dropped():
    """`from einops import rearrange` (a real third-party import, e.g.
    L1_13:33) MUST be classified Class-D with a proof -- NOT silently
    ignored (plan FIX 4: every symbol -> A|B-proof|C|S|D|BLOCKED-cand,
    no silent-drop third state)."""
    src = (
        "import torch\nfrom einops import rearrange\n"
        "from vllm.triton_utils import tl, triton\n"
        "@triton.jit\ndef k(x): pass\ndef run(i): return {}\n"
    )
    tc = pap.classify_task("einops", src)
    by_name = {s.name: s for s in tc.symbols}
    assert "rearrange" in by_name, "einops.rearrange was silently dropped"
    s = by_name["rearrange"]
    assert s.cls == pap.CLASS_D
    assert s.proof and "einops" in s.proof
    assert not s.blocked_reason  # einops IS allowlisted -> not blocked


def test_unaudited_thirdparty_is_blocked_candidate():
    """An unknown non-vllm/sglang third-party package (NOT on the
    Class-D allowlist) must surface as a BLOCKED-candidate, never be
    silently ignored."""
    src = (
        "import torch\nimport scipy.linalg\n"
        "@triton.jit\ndef k(x): pass\ndef run(i): return {}\n"
    )
    tc = pap.classify_task("scipy", src)
    sci = [s for s in tc.symbols if s.module.startswith("scipy")]
    assert sci, "scipy import was silently dropped"
    assert tc.blocked_candidate
    assert any("scipy" in s.blocked_reason for s in sci if s.blocked_reason)


def test_import_alias_resolves_to_original_name():
    """`from vllm import _custom_ops as ops` must be classified +
    recorded under the ORIGINAL name `vllm._custom_ops` (Class-C),
    NOT the local alias `vllm.ops` (plan FIX 4 / L3_02:26)."""
    src = (
        "import torch\nfrom vllm import _custom_ops as ops\n"
        "@triton.jit\ndef k(x): pass\ndef run(i): return {}\n"
    )
    tc = pap.classify_task("alias", src)
    names = {s.symbol for s in tc.symbols}
    assert "vllm._custom_ops" in names, names
    assert "vllm.ops" not in names, "alias mis-recorded as vllm.ops"
    s = {x.symbol: x for x in tc.symbols}["vllm._custom_ops"]
    assert s.cls == pap.CLASS_C  # _custom_ops is an optional-accel fork
    assert "local alias `ops`" in s.proof


def test_multi_kernel_pipeline_is_t3():
    """>=2 @triton.jit (multi-kernel pipeline) -> T3."""
    src = (
        "import torch\nimport triton\n"
        "@triton.jit\ndef k1(x): pass\n"
        "@triton.jit\ndef k2(x): pass\n"
        "def run(i): return {}\n"
    )
    tc = pap.classify_task("pipeline", src)
    assert tc.tier == "T3"


def test_syntax_error_kernel_is_blocked_not_crash():
    """A kernel.py that doesn't parse -> BLOCKED + T3, never an
    uncaught crash."""
    tc = pap.classify_task("broken", "def run(:\n  pass")
    assert tc.blocked
    assert tc.tier == "T3"


def test_classify_is_deterministic():
    """Re-running classify on the same source is byte-identical
    (plan U1 verification: deterministic histogram)."""
    src = (
        "import torch\nfrom vllm.triton_utils import tl, triton\n"
        "from vllm.logger import init_logger\nlogger = init_logger(__name__)\n"
        "@triton.jit\ndef k(x): pass\ndef run(i): return {}\n"
    )
    a = pap.classify_task("d", src)
    b = pap.classify_task("d", src)
    assert sorted((s.symbol, s.cls) for s in a.symbols) == sorted(
        (s.symbol, s.cls) for s in b.symbols
    )
    assert a.tier == b.tier
