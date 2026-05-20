"""Unit tests for the W1.0 child<->parent process-isolation protocol.

Pins the frozen protocol contract that downstream W items depend on:

  * the runner (trusted parent) materialises every shape's inputs and
    injects them down the IPC as safetensors payload + JSON sidecar;
  * the candidate child rebuilds the full mixed inputs dict (tensors +
    ``scalar`` values) from that payload — it never calls ``build_inputs``,
    never receives ``repo_root`` / the shapes document;
  * ``status.json`` carries a schema version and the per-shape diagnostics
    fields (``durations`` / ``kernel_event_present`` / ``stream_nondefault``
    / ``fallback_signal`` / ``sampled_outputs``) — placeholders here, filled
    by W1.3 / W2.4;
  * timing-pass per-trial input injection works (``t=0..N-1``).
"""
from __future__ import annotations

import json
from pathlib import Path

import torch

from triton_ascend_bench import inputs as _inputs
from triton_ascend_bench import process_isolation as pi

# A shapes doc with a mixed inputs block: two tensors + a float scalar +
# a scalar_knob (resolves to an int from the knobs).
_SHAPES_DOC = {
    "size_knobs": {"B": 2, "L": 4, "D": 8},
    "public_shapes": [
        {"id": "P1", "name": "tiny", "knobs": {"B": 1, "L": 2, "D": 4}},
    ],
    "hidden_shapes": [
        {"id": "H1", "name": "h_odd", "knobs": {"B": 3, "L": 5, "D": 7}},
    ],
    "inputs": {
        "x": {"dims": ["B", "L", "D"], "dtype": "float32", "dist": "randn"},
        "w": {"dims": ["D"], "dtype": "float32", "dist": "ones"},
        "eps": {"scalar": 1.0e-5},
        "n_rows": {"scalar_knob": "B"},
    },
}

# A candidate that consumes BOTH the tensor inputs and the scalar inputs,
# proving the child rebuilt a full mixed dict from safetensors + sidecar.
_SOLUTION_SRC = '''
import torch


def npu_triton_impl(inputs):
    x = inputs["x"]
    w = inputs["w"]
    eps = inputs["eps"]          # float scalar from the sidecar
    n_rows = inputs["n_rows"]    # int scalar_knob result from the sidecar
    assert isinstance(eps, float), type(eps)
    assert isinstance(n_rows, int), type(n_rows)
    y = (x + w) * (1.0 + eps) + float(n_rows)
    return {"y": y}
'''


def _write_solution(tmp_path: Path) -> Path:
    sol = tmp_path / "solution.py"
    sol.write_text(_SOLUTION_SRC)
    return sol


def test_child_rebuilds_mixed_inputs_correctness_pass(tmp_path):
    """Correctness pass: one inputs dict per shape, child rebuilds tensors
    + scalars and runs the candidate to completion."""
    sol = _write_solution(tmp_path)
    built = _inputs.build_inputs(_SHAPES_DOC, "P1", seed=0)
    materialised = {"P1": built}

    res = pi.run_candidate_isolated(sol, materialised)

    assert res.load_error is None, res.load_error
    assert res.ran_any
    scr = res.per_shape["P1"]
    assert scr.ran, scr.error
    assert "y" in scr.outputs
    # Recompute the expected output in the trusted process and compare.
    expected = (built["x"] + built["w"]) * (1.0 + built["eps"]) \
        + float(built["n_rows"])
    assert torch.allclose(scr.outputs["y"], expected)


def test_child_handles_hidden_shape_inputs(tmp_path):
    """Hidden-shape inputs are materialised by the runner identically —
    the child has no way to tell public from hidden."""
    sol = _write_solution(tmp_path)
    built = _inputs.build_inputs(_SHAPES_DOC, "H1", seed=0)
    res = pi.run_candidate_isolated(sol, {"H1": built})

    assert res.ran_any
    assert res.per_shape["H1"].ran, res.per_shape["H1"].error


def test_timing_pass_per_trial_injection(tmp_path):
    """Timing pass: a per-trial sequence (t=0..N-1) is injected; each
    trial gets a fresh runner-built inputs dict."""
    sol = _write_solution(tmp_path)
    n = 4
    trials = [
        _inputs.build_inputs(_SHAPES_DOC, "P1", seed=100 + t)
        for t in range(n)
    ]
    res = pi.run_candidate_isolated(sol, {"P1": trials})

    assert res.ran_any
    scr = res.per_shape["P1"]
    assert scr.ran, scr.error
    # The output safetensors corresponds to trial t=0.
    built0 = trials[0]
    expected = (built0["x"] + built0["w"]) * (1.0 + built0["eps"]) \
        + float(built0["n_rows"])
    assert torch.allclose(scr.outputs["y"], expected)


def test_status_json_has_schema_version_and_diagnostics(tmp_path):
    """status.json (decoded into CandidateRunResult / ShapeCandidateResult)
    carries the schema version and the per-shape diagnostics fields."""
    sol = _write_solution(tmp_path)
    built = _inputs.build_inputs(_SHAPES_DOC, "P1", seed=0)
    res = pi.run_candidate_isolated(sol, {"P1": built})

    # schema version round-trips from the child's status.json.
    assert res.schema_version == pi.STATUS_SCHEMA_VERSION

    scr = res.per_shape["P1"]
    # diagnostics carrier fields exist on the dataclass (placeholders;
    # filled by W1.3 / W2.4) — downstream W items pin these names.
    assert isinstance(scr.durations, list)
    assert scr.durations == []
    assert scr.kernel_event_present is None
    assert scr.stream_nondefault is None
    assert scr.fallback_signal is None
    assert scr.sampled_outputs is None


def test_child_args_never_carry_repo_root_or_shapes(tmp_path, monkeypatch):
    """The downstream child args must NOT expose repo_root / the shapes
    document / a seed — the child runs in the sanitised minimal set.

    R3-4: the child is now spawned via `subprocess.Popen` (so the runner
    can `os.killpg` the whole process group on timeout) — the spy wraps
    `Popen`, not `subprocess.run`."""
    captured = {}
    real_popen = pi.subprocess.Popen

    def _spy(cmd, **kw):
        # cmd = [python, child_py, json_args]
        captured["args"] = json.loads(cmd[2])
        return real_popen(cmd, **kw)

    monkeypatch.setattr(pi.subprocess, "Popen", _spy)

    sol = _write_solution(tmp_path)
    built = _inputs.build_inputs(_SHAPES_DOC, "P1", seed=0)
    pi.run_candidate_isolated(sol, {"P1": built})

    args = captured["args"]
    assert "repo_root" not in args
    assert "shapes_yaml" not in args
    assert "shape_ids" not in args
    assert "seed" not in args
    assert args["protocol_version"] == pi.PROTOCOL_VERSION
    assert args["inputs_mode"] == "single"
    # Only the sanitised minimal set: solution + injected inputs.
    assert Path(args["solution"]).name == "solution.py"
    assert args["shapes"][0]["shape_id"] == "P1"
    assert args["shapes"][0]["input_tensors"].endswith(".safetensors")
    assert args["shapes"][0]["input_sidecar"].endswith(".json")


def test_split_merge_inputs_roundtrip():
    """inputs.split_inputs / merge_inputs are exact inverses for a mixed
    dict (the IPC primitive both directions rely on)."""
    built = _inputs.build_inputs(_SHAPES_DOC, "P1", seed=0)
    tensors, sidecar = _inputs.split_inputs(built)

    # tensors-only on one side, JSON scalars on the other.
    assert set(tensors) == {"x", "w"}
    assert set(sidecar) == {"eps", "n_rows"}
    assert sidecar["eps"] == built["eps"]
    assert sidecar["n_rows"] == built["n_rows"]
    # sidecar must be JSON-serialisable.
    json.dumps(sidecar)

    merged = _inputs.merge_inputs(tensors, sidecar)
    assert set(merged) == set(built)
    assert torch.equal(merged["x"], built["x"])
    assert merged["eps"] == built["eps"]


def test_split_inputs_rejects_non_json_scalar():
    """A non-tensor, non-JSON value must fail loudly — the IPC has no
    pickle, so unexpected object types cannot cross it."""
    import pytest

    bad = {"obj": object()}
    with pytest.raises(TypeError):
        _inputs.split_inputs(bad)
