"""GpuBackend dev-box behaviour + backend routing.

On the dev box there is no CUDA, so the GpuBackend guards must return safe
values (never crash), and routing `backend="gpu"` must reach GpuBackend — the
real CUDA execution is validated on the H200 daemon.
"""
from __future__ import annotations

from pathlib import Path

from triton_ascend_bench import inputs as _inputs
from triton_ascend_bench import process_isolation as pi
from triton_ascend_bench.backends import get_backend
from triton_ascend_bench.backends.gpu import GpuBackend


def test_get_backend_gpu_returns_gpu_backend():
    b = get_backend("gpu")
    assert isinstance(b, GpuBackend)
    assert b.name == "gpu"


def test_gpu_backend_dev_guards_no_cuda():
    """Without CUDA the guards return safe values, never crash."""
    b = GpuBackend()
    assert b.event_timer() is None          # no CUDA -> host-only timing
    assert b.synchronize() is None          # no-op
    assert b.device_name() == "<no cuda>"


def test_gpu_backend_profile_call_runs_fn_once():
    """profile_call runs fn exactly once and returns its output; signals are
    unknown without a profiler."""
    b = GpuBackend()
    calls = {"n": 0}

    def fn(inp):
        calls["n"] += 1
        return {"y": inp["x"] * 2}

    out = b.profile_call(fn, {"x": 3.0})
    assert calls["n"] == 1
    assert out["output"] == {"y": 6.0}
    assert out["kernel_event_present"] is None
    assert out["fallback_signal"] is None


_DOC = {
    "size_knobs": {"B": 1, "D": 4},
    "public_shapes": [{"id": "P1", "knobs": {"B": 1, "D": 4}}],
    "inputs": {"x": {"dims": ["B", "D"], "dtype": "float32", "dist": "randn"}},
}


def test_run_candidate_isolated_routes_gpu_backend(tmp_path):
    """run_candidate_isolated(backend='gpu') routes the child to GpuBackend.
    Without CUDA its stubs degrade to host timing + CPU inputs, so the
    candidate still runs — proving the routing, not CUDA execution."""
    sol = Path(tmp_path) / "solution.py"
    sol.write_text(
        "def npu_triton_impl(inputs):\n    return {'y': inputs['x'] * 2.0}\n")
    correctness = {"P1": _inputs.build_inputs(_DOC, "P1", seed=0)}
    res = pi.run_candidate_isolated(
        sol, correctness, pass_kind="correctness", backend="gpu")
    scr = res.per_shape["P1"]
    assert scr.ran, scr.error
