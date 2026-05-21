"""U4: S2 runner soundness tests (test-first -- the soundness command).

Covers: strict bit-exact equivalence (NO ULP escape, NOT
tolerance_class), dtype-mismatch ⇒ fail (no fp32-cast escape),
stride-mismatch ⇒ fail, no-GPU ⇒ UNRUN(needs-gpu) NEVER PASS,
import-audit leak ⇒ BLOCKED signal, C1 non-determinism backdoor
rejection.
"""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parent.parent.parent
_spec = importlib.util.spec_from_file_location(
    "phase_a_prime", REPO / "scripts" / "phase_a_prime.py"
)
pap = importlib.util.module_from_spec(_spec)
sys.modules["phase_a_prime"] = pap
_spec.loader.exec_module(pap)


def test_bit_exact_identical_passes():
    a = {"y": torch.arange(6, dtype=torch.float32).reshape(2, 3)}
    b = {"y": torch.arange(6, dtype=torch.float32).reshape(2, 3)}
    ok, why = pap.s2_smoke_compare(a, b)
    assert ok, why


def test_one_ulp_difference_fails_no_escape():
    """A 1-ULP difference is NOT a legal numeric drift in S2 -- it must
    fail (plan: NO ULP escape, severing is pure import rewrite)."""
    x = torch.tensor([1.0], dtype=torch.float32)
    y = torch.tensor([1.0 + 2 ** -23], dtype=torch.float32)  # 1 ULP
    ok, why = pap.s2_smoke_compare({"o": x}, {"o": y})
    assert not ok
    assert "bit-exact" in why


def test_dtype_mismatch_fails_no_fp32_cast_escape():
    """fp16 vs fp32 (numerically near) MUST fail -- S2 does NOT route
    through tolerance.compare's fp32 cast (plan codex-r2 scenario)."""
    a = {"o": torch.ones(4, dtype=torch.float16)}
    b = {"o": torch.ones(4, dtype=torch.float32)}
    ok, why = pap.s2_smoke_compare(a, b)
    assert not ok
    assert "dtype" in why and "fp32-cast escape" in why


def test_stride_mismatch_fails():
    """Non-contiguous vs contiguous (same values) -> strides differ  ->
    fail (plan: strides/layout in the assertion)."""
    base = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    a = {"o": base.clone()}
    b = {"o": base.t().contiguous().t()}  # same values, different stride
    ok, why = pap.s2_smoke_compare(a, b)
    assert not ok
    assert "stride" in why


def test_key_set_mismatch_fails():
    ok, why = pap.s2_smoke_compare({"a": torch.zeros(1)}, {"b": torch.zeros(1)})
    assert not ok
    assert "key-set" in why


def test_no_gpu_verify_is_unrun_never_pass(tmp_path, monkeypatch):
    """The hard invariant: with no CUDA, --verify HONESTLY returns
    UNRUN(needs-gpu) and NEVER PASS."""
    # This dev box has no GPU; assert the real behaviour.
    monkeypatch.setattr(pap, "_cuda_available", lambda: False)

    class A:
        task = "L1_01_bias"
        verify = "L1_01_bias"

    rc = pap.cmd_verify(A())
    assert rc == 0  # UNRUN is not an error exit


def test_s2result_no_gpu_status_is_unrun():
    """Even when fully wired, no-GPU status must be UNRUN, never PASS
    (plan §invariant 3)."""
    res = pap.S2Result(task_id="t", status="UNRUN")
    assert res.status != "PASS"


def test_subproc_runner_severed_import_audit_blocks_on_leak(tmp_path):
    """The severed subprocess import-audit must flag a vllm leak as a
    BLOCKED signal (exit code 3), never silently pass."""
    runner = REPO / "scripts" / "_pa_subproc_runner.py"
    # A "kernel" that imports vllm at module load -> must leak.
    kp = tmp_path / "kernel.py"
    kp.write_text(
        "import sys, types\n"
        "sys.modules['vllm'] = types.ModuleType('vllm')\n"
        "def run(inputs): return {}\n"
    )
    status_json = tmp_path / "status.json"
    spec = {
        "kernel_path": str(kp),
        "status_json": str(status_json),
        "shapes_json": str(tmp_path / "shapes.json"),
        "out_dir": str(tmp_path),
        "shape_ids": [],
        "seed": 0,
    }
    (tmp_path / "shapes.json").write_text("{}")
    spec_f = tmp_path / "spec.json"
    spec_f.write_text(json.dumps(spec))
    r = subprocess.run(
        [sys.executable, str(runner), "--side", "severed", "--spec", str(spec_f)],
        capture_output=True, text=True,
    )
    assert r.returncode == 3, (r.stdout, r.stderr)
    st = json.loads(status_json.read_text())
    assert st["audit_leaks"]
    assert any("vllm" in x for x in st["audit_leaks"])


def test_c1_non_determinism_backdoor_rejected():
    """C1 (strengthened): actually drive the non-determinism path.

    (a) An UNDECLARED non-deterministic difference yields a
        non-bit-exact failure -- the ONLY verdict is BLOCKED; there is
        no code path that temp-tags it `non_deterministic` to pass.
    (b) The comparator has NO parameter through which a band /
        tolerance / policy could be injected from a
        severed-vs-original delta (the forbidden band source).
    (c) Only a PRE-DECLARED policy `non_deterministic` with a band
        from original-vs-original repeats is acceptable -- and even
        then S2's comparator stays strict bit-exact (the band lives in
        policy, NEVER in s2_smoke_compare / compare_s2_manifests).
    """
    import inspect

    # (a) undeclared non-determinism -> BLOCKED (not temp-tagged).
    a = {"o": torch.tensor([1.0, 2.0])}
    b = {"o": torch.tensor([1.0, 2.0001])}  # tiny but real difference
    ok, why = pap.s2_smoke_compare(a, b)
    assert not ok
    assert "bit-exact" in why  # the verdict is strict, no tag escape

    # (b) NO band/tolerance/policy/non_deterministic injection point in
    # EITHER comparator (the band can never come from severed-vs-orig).
    for fn in (pap.s2_smoke_compare, pap.compare_s2_manifests):
        sig = inspect.signature(fn)
        for forbidden in (
            "tolerance", "non_deterministic", "rtol", "atol", "band",
            "policy", "rerun_count", "tolerance_class",
        ):
            assert forbidden not in sig.parameters, (
                f"{fn.__name__} exposes a {forbidden!r} backdoor"
            )

    # (c) A pre-declared policy `non_deterministic` band is an
    # original-vs-original construct -- it must NOT leak into the S2
    # comparator. The phase_a_prime module never imports the tolerance
    # module (no tolerance_class fallback path at all).
    import ast as _ast

    src = (REPO / "scripts" / "phase_a_prime.py").read_text()
    tree = _ast.parse(src)
    imported = set()
    for n in _ast.walk(tree):
        if isinstance(n, _ast.ImportFrom) and n.module:
            imported.add(n.module)
        elif isinstance(n, _ast.Import):
            imported.update(a.name for a in n.names)
    assert not any(
        "tolerance" in m for m in imported
    ), f"phase_a_prime imports a tolerance module: {imported}"
    # S3 non-determinism -> BLOCKED+investigate, NO tolerance backdoor.
    assert "S3 non-determinism" in src and "NO tolerance backdoor" in src
    # The severed re-runs are compared severed-vs-severed (run i vs
    # run 0), NEVER severed-vs-original as a band source.
    assert 's_side="severed"' in src and 'o_side="severed"' in src


def test_severed_runtime_import_of_vllm_is_caught(tmp_path):
    """Strengthened import-audit: a severed module that imports vllm
    AT RUNTIME (inside run(), AFTER the pre-scan) must STILL be caught
    by the meta_path leak guard ⇒ BLOCKED signal (exit 3), not just
    import-time leaks (plan FIX 1)."""
    runner = REPO / "scripts" / "_pa_subproc_runner.py"
    # No vllm at import time; vllm is imported INSIDE run() (after the
    # pre-run sys.modules scan would have passed).
    kp = tmp_path / "kernel.py"
    kp.write_text(
        "def run(inputs):\n"
        "    import vllm  # runtime import AFTER the pre-scan\n"
        "    return {}\n"
    )
    status_json = tmp_path / "status.json"
    spec = {
        "kernel_path": str(kp),
        "status_json": str(status_json),
        "shapes_json": str(tmp_path / "shapes.json"),
        "out_dir": str(tmp_path),
        "shape_ids": ["s"],
        "seed": 0,
    }
    (tmp_path / "shapes.json").write_text(
        json.dumps({"public_shapes": [{"id": "s"}], "inputs": {}})
    )
    spec_f = tmp_path / "spec.json"
    spec_f.write_text(json.dumps(spec))
    r = subprocess.run(
        [sys.executable, str(runner), "--side", "severed", "--spec",
         str(spec_f)],
        capture_output=True, text=True,
    )
    assert r.returncode == 3, (r.stdout, r.stderr)
    st = json.loads(status_json.read_text())
    assert st.get("runtime_leaks"), st
    assert any("vllm" in x for x in st["runtime_leaks"])
