"""GPU anti-hack provenance flag in _apply_runtime_diagnostics.

The GPU profiler trace parser is not yet implemented (its diagnostics are
all-None until the H200 parser lands), so a GPU run's anti-hack gate is
inactive. The result must be marked `gpu_antihack_unverified` instead of
passing as if the gate had run — while NPU and a fully-signalled GPU run must
NOT get the flag.
"""
from __future__ import annotations

from types import SimpleNamespace

from triton_ascend_bench.eval import _apply_runtime_diagnostics


def _scr(ke=None, sn=None, fs=None):
    return SimpleNamespace(
        kernel_event_present=ke, stream_nondefault=sn, fallback_signal=fs)


def _two(timing, correctness=None):
    return SimpleNamespace(
        timing=SimpleNamespace(per_shape=timing),
        correctness=SimpleNamespace(per_shape=correctness or {}))


def _result(shape_ids):
    return SimpleNamespace(
        shape_results=[SimpleNamespace(shape_id=s) for s in shape_ids],
        performance_flags=[], fallback_detected=False,
        performance_status="ok")


def test_gpu_all_none_diagnostics_flagged_unverified():
    """A GPU run whose profiler returned all-None signals is marked unverified."""
    res = _result(["P1"])
    _apply_runtime_diagnostics(res, _two({"P1": _scr()}), backend="gpu")
    assert "gpu_antihack_unverified" in res.performance_flags


def test_npu_all_none_diagnostics_not_flagged():
    """NPU stub all-None is the documented dev path — no GPU provenance flag."""
    res = _result(["P1"])
    _apply_runtime_diagnostics(res, _two({"P1": _scr()}), backend="npu")
    assert "gpu_antihack_unverified" not in res.performance_flags


def test_gpu_all_signals_present_not_flagged():
    """A fully-signalled GPU run (real H200 parser) clears the provenance flag."""
    res = _result(["P1"])
    two = _two({"P1": _scr(ke=True, sn=False, fs=False)})
    _apply_runtime_diagnostics(res, two, backend="gpu")
    assert "gpu_antihack_unverified" not in res.performance_flags


def test_gpu_partial_signals_flagged_unverified():
    """A partial parser (any signal still None) stays unverified (future-proof)."""
    res = _result(["P1"])
    two = _two({"P1": _scr(ke=True, sn=None, fs=False)})
    _apply_runtime_diagnostics(res, two, backend="gpu")
    assert "gpu_antihack_unverified" in res.performance_flags


def test_gpu_flag_idempotent_across_shapes():
    """Multiple unverified shapes add the flag exactly once."""
    res = _result(["P1", "P2"])
    two = _two({"P1": _scr(), "P2": _scr()})
    _apply_runtime_diagnostics(res, two, backend="gpu")
    assert res.performance_flags.count("gpu_antihack_unverified") == 1
