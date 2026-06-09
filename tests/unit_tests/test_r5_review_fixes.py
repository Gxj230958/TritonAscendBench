"""R5 codex-review regression tests — four review findings.

Pins the R5 review findings so a regression re-surfaces as a test
failure:

  * R5-1 — the correctness pass profiles and grades ONE candidate call,
    not two. ``_record_diag`` runs ``fn(inp)`` exactly once and returns
    that call's output; the stored correctness output IS the profiled
    call's output. A stateful candidate cannot behave honestly on a
    diagnosed call and serve a different (cheating) output on a
    separately-graded call — there is no second call.
  * R5-2 — the timing pass's profiled measured trial runs ``fn`` exactly
    once: that single invocation is host-timed, profiled, AND its return
    value is the sampled output reverified for that trial. No second,
    untimed call re-opens the sampled-output cheat path.
  * R5-3 — a map-form ``cost_model`` key must be a real peak-table
    compute class / resource. A typo'd key (e.g. ``"vecotr"``) is
    rejected statically by ``parse_cost_model``, not silently carried
    until ``compute_t_ref`` fails closed at scoring time.
  * R5-4 — ``TaskEvalResult.task_measurement()`` exposes a
    ``(baseline_ns, measured_ns)`` provenance pair from the SAME
    representative shape, never spliced from two different shapes.

Dev machine has no NPU; the candidate stateful-cheat tests count calls
inside the candidate process to prove single-invocation, not NPU
numeric reality.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import torch

from ascendbench.input_spec import inputs as _inputs
from ascendbench.integrity import process_isolation as pi
from ascendbench.perf.cost_model import CostModelError, parse_cost_model
from ascendbench.engine.eval import ShapeRunResult, TaskEvalResult

_SHAPES_DOC = {
    "size_knobs": {"B": 2, "L": 4, "D": 8},
    "public_shapes": [
        {"id": "P1", "name": "tiny", "knobs": {"B": 1, "L": 2, "D": 4}},
    ],
    "inputs": {
        "x": {"dims": ["B", "L", "D"], "dtype": "float32", "dist": "randn"},
    },
}

# A stateful candidate: it counts how many times `npu_triton_impl` is
# invoked *in this process* and stamps the count into the output tensor.
# If the runner calls it more than once per graded input, the stored /
# sampled output reveals the extra call — the cheat-path the R5 fixes
# close. `npu_triton_impl` returns the input scaled by the call ordinal,
# so a 2nd call produces a strictly different tensor than the 1st.
_STATEFUL_SOLUTION = '''
import torch

_CALLS = {"n": 0}


def npu_triton_impl(inputs):
    _CALLS["n"] += 1
    x = inputs["x"]
    # The output encodes the per-process call ordinal: call k -> x * k.
    return {"y": x * float(_CALLS["n"])}
'''


def _write_stateful(tmp_path: Path) -> Path:
    sol = tmp_path / "solution.py"
    sol.write_text(_STATEFUL_SOLUTION)
    return sol


def _timing_payload(shape_id: str, n_warmup: int, n_repeat: int) -> list:
    return (
        [_inputs.build_inputs(_SHAPES_DOC, shape_id, seed=1 + w)
         for w in range(n_warmup)]
        + [_inputs.build_inputs(_SHAPES_DOC, shape_id, seed=1000 + t)
           for t in range(n_repeat)]
    )


# --------------------------------------------------------------------------
# R5-1 — correctness pass profiles + grades ONE call.
# --------------------------------------------------------------------------

def test_r5_1_correctness_output_is_the_profiled_call(tmp_path):
    """The correctness pass invokes the candidate exactly once per shape.

    The stateful candidate stamps its per-process call ordinal into the
    output (call k -> x * k). With ONE call the stored output is x * 1.
    Before R5-1 the diagnostic call + the graded call were two separate
    invocations, so the stored output would be x * 2 — the second call.
    """
    sol = _write_stateful(tmp_path)
    correctness = {"P1": _inputs.build_inputs(_SHAPES_DOC, "P1", seed=0)}
    res = pi.run_candidate_isolated(sol, correctness, pass_kind="correctness")

    scr = res.per_shape["P1"]
    assert scr.ran, scr.error
    x = correctness["P1"]["x"]
    # Exactly one call -> output is x * 1, NOT x * 2 (a 2nd call).
    assert torch.equal(scr.outputs["y"], x * 1.0), (
        "correctness output must be the single profiled call's output")


def test_r5_1_correctness_per_trial_t0_is_single_call(tmp_path):
    """In per-trial correctness mode the t=0 trial — the profiled one —
    is still a single invocation; its stored output is x_t0 * 1."""
    sol = _write_stateful(tmp_path)
    # Per-trial correctness payload: t=0 is the first call in-process.
    correctness = {
        "P1": [_inputs.build_inputs(_SHAPES_DOC, "P1", seed=t)
               for t in range(3)]
    }
    res = pi.run_candidate_isolated(sol, correctness, pass_kind="correctness")
    scr = res.per_shape["P1"]
    assert scr.ran, scr.error
    x0 = correctness["P1"][0]["x"]
    # t=0 is the very first call -> output x0 * 1.
    assert torch.equal(scr.outputs["y"], x0 * 1.0)


# --------------------------------------------------------------------------
# R5-2 — timing pass profiled trial is a single timed+sampled call.
# --------------------------------------------------------------------------

def test_r5_2_timing_sampled_output_is_the_timed_call(tmp_path):
    """The timing pass's profiled measured trial runs the candidate once.

    Sampling the profiled trial: its shipped-up sampled output must be the
    output of the SAME call that was host-timed and profiled. The stateful
    candidate stamps its call ordinal, so the sampled output for measured
    trial t is x_t * (ordinal of that call). Before R5-2 the profiled
    trial did a profiler call THEN an untimed re-run, so its sampled
    output reflected a different call than the timed one.

    With no warm-up, n_repeat=3, and the profiled trial sampled: the
    profiled trial is the first sampled measured index. We sample every
    measured trial so each sampled output is checkable, and assert the
    profiled trial's output is consistent with a single call per trial
    (call ordinal == measured index + 1, no skipped/extra calls).
    """
    sol = _write_stateful(tmp_path)
    timing = {"P1": _timing_payload("P1", n_warmup=0, n_repeat=3)}
    # Sample all three measured trials; the profiler binds to the first.
    res = pi.run_candidate_isolated(
        sol, timing, pass_kind="timing", n_warmup=0,
        sampled_indices={"P1": [0, 1, 2]})

    scr = res.per_shape["P1"]
    assert scr.ran, scr.error
    assert scr.sampled_outputs is not None
    # Measured inputs are payload[0:], one fresh call each. With exactly
    # one call per measured trial the per-process ordinal for measured
    # index t is t+1, so the sampled output for trial t is x_t * (t+1).
    meas_inputs = timing["P1"]
    for t in range(3):
        key = f"{t}::y"
        assert key in scr.sampled_outputs, key
        expected = meas_inputs[t]["x"] * float(t + 1)
        assert torch.equal(scr.sampled_outputs[key], expected), (
            f"trial {t}: sampled output must be the single timed call's "
            f"output (one call per measured trial, no extra re-run)")
    # Three measured trials, three durations — the profiled one included.
    assert len(scr.durations) == 3


# --------------------------------------------------------------------------
# R5-3 — map-form cost_model key validation.
# --------------------------------------------------------------------------

def test_r5_3_flops_map_rejects_typo_key():
    """A typo'd ``flops`` map key (e.g. ``vecotr``) is rejected at parse
    time with a CostModelError, not silently carried to scoring."""
    with pytest.raises(CostModelError, match="not a valid compute class"):
        parse_cost_model({
            "flops": {"vecotr": "B*N"},
            "bytes": {"hbm": "B*N"},
        })


def test_r5_3_bytes_map_rejects_typo_key():
    """A typo'd ``bytes`` map key (e.g. ``hmb``) is rejected at parse
    time with a CostModelError."""
    with pytest.raises(CostModelError, match="not a valid resource"):
        parse_cost_model({
            "flops": {"vector": "B*N"},
            "bytes": {"hmb": "B*N"},
        })


def test_r5_3_map_form_accepts_every_valid_key():
    """Every legal peak-table compute class / resource is accepted as a
    map key — the validation rejects typos, not the real key set."""
    model = parse_cost_model({
        "flops": {"fp16_cube": "B", "bf16_cube": "B",
                  "int8": "B", "vector": "B"},
        "bytes": {"hbm": "B", "interconnect": "B"},
    })
    assert set(model.flops) == {"fp16_cube", "bf16_cube", "int8", "vector"}
    assert set(model.bytes) == {"hbm", "interconnect"}


def test_r5_3_validate_tasks_catches_typo_map_key(tmp_path, monkeypatch):
    """The validator's cost_model check surfaces the typo'd map key —
    `parse_cost_model` raises and `_validate_cost_model` reports it."""
    import importlib.util
    import sys

    repo = Path(__file__).resolve().parents[2]
    spec = importlib.util.spec_from_file_location(
        "validate_tasks_r5", repo / "scripts" / "corpus" / "validate_tasks.py")
    vt = importlib.util.module_from_spec(spec)
    sys.modules["validate_tasks_r5"] = vt
    spec.loader.exec_module(vt)

    errs = vt._validate_cost_model(
        "t_typo",
        {"cost_model": {"flops": {"vecotr": "B"}, "bytes": {"hbm": "B"}}},
        tmp_path)
    assert errs, "validator must report the invalid map key"
    assert any("invalid cost_model" in e and "vecotr" in e for e in errs), errs


# --------------------------------------------------------------------------
# R5-4 — task_measurement provenance pair from one shape.
# --------------------------------------------------------------------------

def test_r5_4_provenance_pair_from_same_shape():
    """task_measurement() takes (baseline_ns, measured_ns) from the first
    shape that has BOTH a t_ref and a t_npu — not spliced across shapes."""
    result = TaskEvalResult(task_id="t", static_checks_passed=True)
    # Shape S1 has a t_ref but no t_npu (no usable timing).
    s1 = ShapeRunResult("S1", True, None, None, None)
    s1.t_ref = 10.0
    s1.t_npu = None
    # Shape S2 has both -> it is the representative shape.
    s2 = ShapeRunResult("S2", True, None, None, None)
    s2.t_ref = 20.0
    s2.t_npu = 40.0
    result.shape_results = [s1, s2]

    tm = result.task_measurement()
    # The pair MUST be S2's, never spliced (S1.t_ref=10, S2.t_npu=40).
    # baseline_ns is t_ref(s) converted to ns = 20.0 * 1e9; measured_ns
    # is t_npu (ns).
    assert tm.baseline_ns == 20.0e9
    assert tm.measured_ns == 40.0


def test_r5_4_provenance_pair_none_when_no_shape_has_both():
    """When no single shape carries both t_ref and t_npu the provenance
    pair is (None, None) — a half-pair is never fabricated."""
    result = TaskEvalResult(task_id="t", static_checks_passed=True)
    s1 = ShapeRunResult("S1", True, None, None, None)
    s1.t_ref = 10.0
    s1.t_npu = None
    s2 = ShapeRunResult("S2", True, None, None, None)
    s2.t_ref = None
    s2.t_npu = 40.0
    result.shape_results = [s1, s2]

    tm = result.task_measurement()
    assert tm.baseline_ns is None
    assert tm.measured_ns is None
