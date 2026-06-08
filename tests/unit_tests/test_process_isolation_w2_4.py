"""W2.4 — runtime-defense plumbing tests for process_isolation.

Pins the W2.4 child-side detection contract on top of W1.0/W1.3:

  * the candidate child profiles one call per shape and ships up the
    `kernel_event_present` / `stream_nondefault` / `fallback_signal`
    diagnostic carrier fields (W1.0 status.json schema);
  * on the dev machine (no NPU) the profiler is a documented stub, so
    the three signals decode to `None` ("not measured here") rather than
    a fabricated value;
  * the child installs the Layer E `sys.addaudithook` hook at start-up.

Dev machine has no NPU — these verify the uplink plumbing and the stub
boundary, not NPU numeric reality.
"""
from __future__ import annotations

from pathlib import Path

from ascendbench.input_spec import inputs as _inputs
from ascendbench.integrity import process_isolation as pi

_SHAPES_DOC = {
    "size_knobs": {"B": 2, "L": 4, "D": 8},
    "public_shapes": [
        {"id": "P1", "name": "tiny", "knobs": {"B": 1, "L": 2, "D": 4}},
    ],
    "inputs": {
        "x": {"dims": ["B", "L", "D"], "dtype": "float32", "dist": "randn"},
        "w": {"dims": ["D"], "dtype": "float32", "dist": "ones"},
        "eps": {"scalar": 1.0e-5},
    },
}

_SOLUTION_SRC = '''
import torch


def npu_triton_impl(inputs):
    x = inputs["x"]
    w = inputs["w"]
    eps = inputs["eps"]
    return {"y": (x + w) * (1.0 + eps)}
'''


def _write_solution(tmp_path: Path) -> Path:
    sol = tmp_path / "solution.py"
    sol.write_text(_SOLUTION_SRC)
    return sol


def _timing_payload(shape_id: str, n_warmup: int, n_repeat: int) -> list:
    return (
        [_inputs.build_inputs(_SHAPES_DOC, shape_id, seed=1 + w)
         for w in range(n_warmup)]
        + [_inputs.build_inputs(_SHAPES_DOC, shape_id, seed=1000 + t)
           for t in range(n_repeat)]
    )


def test_correctness_pass_carries_diagnostic_fields(tmp_path):
    """The correctness pass populates the W2.4 diagnostic carrier fields
    on the per-shape result (stubbed → None on the dev machine)."""
    sol = _write_solution(tmp_path)
    correctness = {"P1": _inputs.build_inputs(_SHAPES_DOC, "P1", seed=0)}
    res = pi.run_candidate_isolated(sol, correctness, pass_kind="correctness")

    scr = res.per_shape["P1"]
    assert scr.ran, scr.error
    # The fields EXIST on the result object (they are decoded from
    # status.json) — dev-machine stub leaves them None.
    assert hasattr(scr, "kernel_event_present")
    assert hasattr(scr, "stream_nondefault")
    assert hasattr(scr, "fallback_signal")
    assert scr.kernel_event_present is None
    assert scr.stream_nondefault is None
    assert scr.fallback_signal is None


def test_timing_pass_carries_diagnostic_fields(tmp_path):
    """The timing pass profiles one MEASURED trial per shape (R3-5: bound
    to a runner-secret sampled measured trial, never a warm-up call) and
    carries the diagnostics up. Every measured trial is still recorded."""
    sol = _write_solution(tmp_path)
    timing = {"P1": _timing_payload("P1", n_warmup=2, n_repeat=4)}
    res = pi.run_candidate_isolated(
        sol, timing, pass_kind="timing", n_warmup=2,
        sampled_indices={"P1": [1]})

    scr = res.per_shape["P1"]
    assert scr.ran, scr.error
    assert scr.fallback_signal is None  # dev stub
    # The profiled measured trial is itself a timed trial — all 4 recorded.
    assert len(scr.durations) == 4


def test_timing_pass_diagnostics_without_warmup(tmp_path):
    """With no warm-up trials the child still profiles a measured trial
    (R3-5) — diagnostics travel up and every measured duration is kept."""
    sol = _write_solution(tmp_path)
    timing = {"P1": _timing_payload("P1", n_warmup=0, n_repeat=3)}
    res = pi.run_candidate_isolated(
        sol, timing, pass_kind="timing", n_warmup=0,
        sampled_indices={"P1": [2]})

    scr = res.per_shape["P1"]
    assert scr.ran, scr.error
    assert len(scr.durations) == 3
    assert scr.fallback_signal is None


def test_two_pass_diagnostics_round_trip(tmp_path):
    """run_two_pass carries the diagnostic fields on both passes."""
    sol = _write_solution(tmp_path)
    correctness = {"P1": _inputs.build_inputs(_SHAPES_DOC, "P1", seed=0)}
    timing = {"P1": _timing_payload("P1", n_warmup=1, n_repeat=3)}
    two = pi.run_two_pass(sol, correctness, timing, n_warmup=1)

    for res in (two.correctness, two.timing):
        scr = res.per_shape["P1"]
        assert scr.ran, scr.error
        assert scr.fallback_signal is None


def test_child_script_installs_layer_e_audit_hook():
    """The embedded child script wires the Layer E install function in."""
    assert "install_runtime_audit_hook" in pi._CHILD
    # The profiler delegate is wired in too.
    assert "profile_call" in pi._CHILD
