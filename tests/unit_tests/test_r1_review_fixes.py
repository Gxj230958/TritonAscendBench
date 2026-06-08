"""R1 codex-review regression tests — cross-W integration gaps.

Covers the four cross-W-item gaps the R1 review found that the per-W-item
suites did not pin:

  * R1-1 — a clean two-pass run emits ``performance_status == "ok"``
    (the field was never promoted off its ``"error"`` default).
  * R1-3 — the timing-arbitration NPU event-timer channel is wired
    end-to-end: the timing-pass child collects an ``event_ns`` series and
    the trusted parent feeds it to ``arbitrate_timing``. On the dev
    machine there is no NPU so the series is ``None`` and arbitration
    degrades to host-only — but the channel is connected.
  * R1-5 — a scalar-only inputs block does not leak the ``__empty__``
    safetensors placeholder to the candidate.

Dev machine has no NPU — these verify plumbing/connectivity and the
status-promotion contract, not NPU numeric reality.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
import yaml

from ascendbench.integrity import process_isolation as pi
from ascendbench.corpus.dataset import TaskDef
from ascendbench.engine.eval import evaluate
from ascendbench.integrity.process_isolation import (
    CandidateRunResult,
    ShapeCandidateResult,
    TwoPassResult,
)

REPO = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------
# A self-contained task whose oracle is a pure-torch elementwise op. The
# candidate cannot run on the dev machine (no triton/NPU — the static
# checker rightly demands a real @triton.jit kernel), so the genuine
# two-pass *execution* is simulated by monkeypatching `run_two_pass` with
# a synthetic clean result: candidate ran, correct outputs, clean
# host-timing series, no NPU event timer (dev-machine `event_ns=None`).
# This pins the eval-level status-promotion / event-channel wiring, which
# is exactly what the R1 review found broken.
# ---------------------------------------------------------------------------

_ORACLE_SRC = (
    "import torch\n"
    "def run(inputs):\n"
    "    x = inputs['x']\n"
    "    return {'y': x + 1.0}\n"
)
# A static-clean candidate stub: it has the @triton.jit / tl.load / tl.store
# shape the Layer-D positive-AST check requires, so `evaluate()` passes the
# static gate and reaches phase-2b. Its body is never executed — the
# two-pass run is monkeypatched.
_SOLUTION_SRC = (
    "import triton\n"
    "import triton.language as tl\n"
    "\n"
    "@triton.jit\n"
    "def _k(x_ptr, y_ptr, n, BLOCK: tl.constexpr):\n"
    "    o = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)\n"
    "    m = o < n\n"
    "    tl.store(y_ptr + o, tl.load(x_ptr + o, mask=m) + 1.0, mask=m)\n"
    "\n"
    "def npu_triton_impl(inputs):\n"
    "    x = inputs['x']\n"
    "    y = x.new_empty(x.shape)\n"
    "    _k[(1,)](x, y, x.numel(), BLOCK=64)\n"
    "    return {'y': y}\n"
)


def _make_task(td: Path, *, inputs_spec: dict) -> tuple[TaskDef, Path]:
    """Materialise a task dir + a static-clean solution under ``td``."""
    kernel_src = "@triton.jit\ndef k(i):\n    return {}\n"
    (td / "kernel.py").write_text(kernel_src)
    (td / "oracle_pytorch.py").write_text(_ORACLE_SRC)
    (td / "contract.md").write_text("# Contract — R1 task\n")
    shapes_doc = {
        "size_knobs": {"B": 4, "N": 8},
        "public_shapes": [
            {"id": "P1", "name": "tiny", "knobs": {"B": 2, "N": 4}},
            {"id": "P2", "name": "small", "knobs": {"B": 3, "N": 5}},
        ],
        "inputs": inputs_spec,
    }
    (td / "shapes.yaml").write_text(yaml.safe_dump(shapes_doc))
    meta = {
        "schema_version": "1.0", "task_id": "L1_99_r1",
        "level": "L1", "idx": 99, "name": "r1", "repo": "vLLM",
        "dtype_class": "VV", "category": "norm", "npu_difficulty": "low",
        "upstream": {"file_sha256": hashlib.sha256(
            kernel_src.encode()).hexdigest()},
        "interface": {}, "oracle": {"gpu_triton": "kernel.py::placeholder"},
        "tolerance_class": "elementwise",
        "oracle_simplified": "simple_pytorch_ok", "runner_checks": {},
        "is_canary": False, "is_hidden_only": False,
    }
    (td / "meta.yaml").write_text(yaml.safe_dump(meta))
    task = TaskDef(
        task_id="L1_99_r1", level=1, task_dir=td, meta=meta,
        public_shapes=[], contract_md="", kernel_py=kernel_src,
        kernel_py_sha256=hashlib.sha256(kernel_src.encode()).hexdigest(),
    )
    sol = td / "solution.py"
    sol.write_text(_SOLUTION_SRC)
    return task, sol


_TENSOR_INPUTS = {
    "x": {"dims": ["B", "N"], "dtype": "float32", "dist": "randn"},
}


def _clean_two_pass_factory(task: TaskDef, *, event_ns=None):
    """Build a `run_two_pass` replacement that returns a synthetic CLEAN
    two-pass result for the task's shapes — the candidate "ran", produced
    the oracle-correct output, and recorded a clean ~5 ms host-timing
    series. `event_ns` (per the dev-machine contract) is None.
    """
    import importlib.util

    import torch

    from ascendbench.input_spec import inputs as _inputs

    omod_spec = importlib.util.spec_from_file_location(
        "_r1_oracle", task.task_dir / "oracle_pytorch.py")
    omod = importlib.util.module_from_spec(omod_spec)
    omod_spec.loader.exec_module(omod)
    shapes_doc = yaml.safe_load((task.task_dir / "shapes.yaml").read_text())
    pub, _hid = _inputs.shape_ids(shapes_doc)

    def _fake_run_two_pass(solution_path, correctness_inputs,
                           timing_inputs, **kw):
        corr_shapes: dict[str, ShapeCandidateResult] = {}
        time_shapes: dict[str, ShapeCandidateResult] = {}
        for sid in pub:
            # Correct output = oracle on the seed-0 correctness input.
            golden = omod.run(correctness_inputs[sid])
            corr_shapes[sid] = ShapeCandidateResult(
                shape_id=sid, ran=True,
                outputs={k: v for k, v in golden.items()
                         if torch.is_tensor(v)})
            # A clean ~5 ms host-timing series (well above the sub-µs
            # quarantine floor, finite, positive).
            time_shapes[sid] = ShapeCandidateResult(
                shape_id=sid, ran=True,
                durations=[5_000_000.0] * 8,
                event_ns=event_ns)
        return TwoPassResult(
            correctness=CandidateRunResult(
                ran_any=True, timed_out=False, load_error=None,
                per_shape=corr_shapes),
            timing=CandidateRunResult(
                ran_any=True, timed_out=False, load_error=None,
                per_shape=time_shapes, pass_kind="timing"),
            scrubbed=[])

    return _fake_run_two_pass


# ---------------------------------------------------------------------------
# R1-1 — a clean two-pass run promotes performance_status to "ok".
# ---------------------------------------------------------------------------

def test_clean_run_emits_performance_status_ok(tmp_path, monkeypatch):
    """A two-pass run with a correct candidate + a clean host-timing
    series emits ``performance_status == "ok"`` — the field is promoted
    off its default ``"error"`` once the candidate ran cleanly."""
    task, sol = _make_task(tmp_path, inputs_spec=_TENSOR_INPUTS)
    monkeypatch.setattr(
        pi, "run_two_pass", _clean_two_pass_factory(task))

    result = evaluate("L1_99_r1", sol, task_def=task)

    assert result.error_reason is None, result.error_reason
    # The candidate ran cleanly: correctness ok, runtime ok.
    assert result.correctness_status == "ok", (
        result.correctness_status, result.shape_results)
    assert result.runtime_status == "ok", result.runtime_status
    # R1-1: performance_status must be promoted to "ok" — never stuck at
    # the "error" default after a clean run.
    assert result.performance_status == "ok", (
        f"clean run did not promote performance_status: "
        f"{result.performance_status!r}")
    # A clean run carries no downgrade flags.
    assert result.performance_flags == [], result.performance_flags


def test_quarantine_still_downgrades_after_ok_promotion(tmp_path,
                                                        monkeypatch):
    """R1-1 must not break the downgrade path: a sub-µs timing series
    still downgrades the (now-promoted) ``performance_status`` to
    ``quarantined`` — the promotion is downgrade-friendly, not a clobber.
    """
    task, sol = _make_task(tmp_path, inputs_spec=_TENSOR_INPUTS)
    factory = _clean_two_pass_factory(task)

    def _subus_two_pass(*a, **kw):
        tp = factory(*a, **kw)
        for scr in tp.timing.per_shape.values():
            scr.durations = [200.0] * 8        # sub-µs ⇒ quarantine
        return tp

    monkeypatch.setattr(pi, "run_two_pass", _subus_two_pass)
    result = evaluate("L1_99_r1", sol, task_def=task)
    assert result.performance_status == "quarantined", (
        f"sub-µs run should downgrade, got {result.performance_status!r}")


def test_unran_task_keeps_performance_status_error():
    """A task that never reached phase-2b grading keeps the default
    ``performance_status == "error"`` — the R1-1 promotion is gated on a
    real run, it does not blanket-set "ok"."""
    from ascendbench.engine.eval import TaskEvalResult

    fresh = TaskEvalResult(task_id="x", static_checks_passed=True)
    assert fresh.performance_status == "error"


# ---------------------------------------------------------------------------
# R1-3 — the timing-arbitration event-timer channel is wired end-to-end.
# ---------------------------------------------------------------------------

def test_evaluate_carries_event_ns_channel(tmp_path, monkeypatch):
    """The per-shape ``event_ns`` field exists and travels through
    ``evaluate()``. On the dev machine (no NPU) it is ``None`` — the
    host-only degradation — but the channel from the timing-pass child
    up to the parent's arbitration is connected."""
    task, sol = _make_task(tmp_path, inputs_spec=_TENSOR_INPUTS)
    monkeypatch.setattr(
        pi, "run_two_pass", _clean_two_pass_factory(task, event_ns=None))

    result = evaluate("L1_99_r1", sol, task_def=task)

    assert result.error_reason is None, result.error_reason
    assert result.shape_results, "no shape results"
    for r in result.shape_results:
        # The field is present on every shape result (R1-3 schema).
        assert hasattr(r, "event_ns")
        # Dev machine: no NPU event timer ⇒ None (host-only arbitration).
        assert r.event_ns is None, (
            f"{r.shape_id}: dev machine should have no event_ns, "
            f"got {r.event_ns!r}")
    # The status.json schema version was bumped for the new event_ns
    # carrier — the timing-pass child speaks v2.
    assert pi.STATUS_SCHEMA_VERSION >= 2


def test_evaluate_event_drift_invalidates_timing(tmp_path, monkeypatch):
    """R1-3 end-to-end: when the timing-pass child reports an `event_ns`
    series that drifts > 10% from host and the W2.4 profiler flags a
    non-default stream, `evaluate()` lands `runtime_status='invalid_timing'`
    + the `invalid_stream` flag — the path that was unreachable before the
    event channel crossed the IPC."""
    task, sol = _make_task(tmp_path, inputs_spec=_TENSOR_INPUTS)
    # event_ns ~= half the host series ⇒ ~50% drift.
    factory = _clean_two_pass_factory(task, event_ns=[2_500_000.0] * 8)

    def _drift_two_pass(*a, **kw):
        tp = factory(*a, **kw)
        # Profiler verdict: work ran on a non-default stream.
        for scr in tp.timing.per_shape.values():
            scr.stream_nondefault = True
        return tp

    monkeypatch.setattr(pi, "run_two_pass", _drift_two_pass)
    result = evaluate("L1_99_r1", sol, task_def=task)

    assert result.runtime_status == "invalid_timing", (
        f"event-drift + non-default stream should invalidate timing, "
        f"got runtime_status={result.runtime_status!r}")
    assert "invalid_stream" in result.performance_flags, (
        result.performance_flags)


def test_arbitrate_timing_invalid_stream_path_reachable():
    """R1-3: with an `event_ns` series + a profiler-detected non-default
    stream, `arbitrate_timing` reaches the `invalid_timing` /
    `invalid_stream` verdict — the path that was previously unreachable
    because `event_ns` never crossed the IPC."""
    from ascendbench.perf import timing

    host = [10_000_000] * 6              # 10 ms host durations
    # Event timer reads ~half the host time -> ~50% drift, well over 10%.
    event = [5_000_000] * 6
    arb = timing.arbitrate_timing(
        host, event_ns=event, stream_nondefault=True)
    assert arb.runtime_status == "invalid_timing"
    assert "invalid_stream" in arb.performance_flags

    # Same drift but no non-default stream -> only the soft drift flag.
    arb2 = timing.arbitrate_timing(
        host, event_ns=event, stream_nondefault=False)
    assert arb2.runtime_status is None
    assert "host_event_drift" in arb2.performance_flags


# ---------------------------------------------------------------------------
# R1-2 — a scalar-form cost_model produces a scorable t_ref.
#
# The scalar form normalises to a one-entry map keyed by its companion
# `flops_class` / `bytes_resource` — a real peak-table class/resource —
# so `compute_t_ref` (which only accepts real peak keys) can score it.
# Before R1-2 the scalar form normalised to a synthetic `"default"` key
# the peak table never had, so `t_ref` was uncomputable.
# ---------------------------------------------------------------------------

def test_scalar_cost_model_t_ref_path_connected():
    """End-to-end: a scalar-form `cost_model` → `parse_cost_model` →
    `compute_t_ref` → `shape_efficiency`. The companion `flops_class` /
    `bytes_resource` make the normalised map key a real peak-table
    class/resource, so the Roofline path actually computes a number."""
    from ascendbench.perf.cost_model import parse_cost_model
    from ascendbench.perf.scoring import compute_t_ref, shape_efficiency

    # Scalar form: a single FLOPs formula on the fp16 cube, a single
    # bytes formula on HBM — each with its required companion field.
    model = parse_cost_model({
        "flops": "2*B*N",
        "flops_class": "fp16_cube",
        "bytes": "B*N*dtype_size",
        "bytes_resource": "hbm",
        "gpu_xcheck_tol": 0.05,
    })
    # The normalised maps key REAL peak-table classes/resources — no
    # synthetic "default" key.
    assert set(model.flops) == {"fp16_cube"}
    assert set(model.bytes) == {"hbm"}

    # Synthetic peak table with a known π for fp16_cube / β for hbm.
    knobs = {"B": 4, "N": 8}            # W = 2*4*8 = 64 FLOP
    t_ref = compute_t_ref(
        model, knobs,
        compute_peaks={"fp16_cube": 64.0},   # 64 FLOP / 64 FLOP/s = 1.0 s
        resource_peaks={"hbm": 1e30},        # bytes term ~ 0
        dtype_size=2.0)
    assert t_ref == 1.0, t_ref

    # The per-shape efficiency e = min(P_CAP, t_ref / t_npu) is then
    # computable — the whole "scalar cost_model → t_ref → e" chain works.
    e = shape_efficiency(t_ref=t_ref, t_npu=2.0)
    assert e == 0.5


def test_scalar_cost_model_unknown_class_still_fails_closed():
    """A scalar `flops_class` the peak table does not carry still
    fail-closes in `compute_t_ref` — the companion field is validated at
    parse time, but a peak table missing that class is a scoring-time
    fail-closed (RooflineError), never silent bad data."""
    from ascendbench.perf.cost_model import parse_cost_model
    from ascendbench.perf.scoring import RooflineError, compute_t_ref

    model = parse_cost_model({
        "flops": "B", "flops_class": "int8",
        "bytes": "B", "bytes_resource": "hbm",
    })
    # int8 is a legal companion class, but this peak table omits it.
    with pytest.raises(RooflineError, match="int8"):
        compute_t_ref(model, {"B": 2},
                      compute_peaks={"fp16_cube": 1e12},
                      resource_peaks={"hbm": 1e12})


# ---------------------------------------------------------------------------
# R1-5 — a scalar-only inputs block does not leak the __empty__ sentinel.
# ---------------------------------------------------------------------------

# A candidate that FAILS LOUDLY if it receives the `__empty__` placeholder
# key — the run is graded by the child, so a leaked sentinel surfaces as a
# per-shape error the test can assert on.
_SENTINEL_GUARD_SOLUTION = '''
import torch


def npu_triton_impl(inputs):
    if "__empty__" in inputs:
        raise AssertionError(
            "__empty__ sentinel leaked into the candidate inputs")
    # Echo the scalars back as a tensor so the output dict is well-formed.
    return {"y": torch.tensor([float(inputs["alpha"]),
                               float(inputs["beta"])])}
'''


def test_scalar_only_inputs_materialise_with_placeholder(tmp_path):
    """Fixture sanity: an all-scalar inputs block has no tensors, so the
    runner materialises a `{"__empty__": zeros(0)}` safetensors payload
    (safetensors refuses an empty dict)."""
    from safetensors.torch import load_file

    tdir = tmp_path / "inp"
    tdir.mkdir()
    tp, _sp = pi._materialise_inputs({"alpha": 1.5, "beta": 7}, tdir, "P1")
    raw = load_file(tp)
    assert "__empty__" in raw, (
        "scalar-only block should carry the __empty__ placeholder")


def test_scalar_only_inputs_strip_empty_sentinel(tmp_path):
    """A scalar-only inputs block reaches the candidate child with NO
    `__empty__` key. This runs the real child subprocess
    (`run_candidate_isolated`): the candidate raises if it sees the
    placeholder, so a leak surfaces as a per-shape error. A clean run
    proves `_merge` stripped the sentinel."""
    sol = tmp_path / "solution.py"
    sol.write_text(_SENTINEL_GUARD_SOLUTION)

    # All-scalar inputs for one shape — no tensor values at all.
    scalar_inputs = {"P1": {"alpha": 1.5, "beta": 7}}
    res = pi.run_candidate_isolated(sol, scalar_inputs, pass_kind="correctness")

    scr = res.per_shape.get("P1")
    assert scr is not None, f"no result for P1: {res.load_error}"
    assert scr.error is None, (
        f"scalar-only candidate hit an error (sentinel leak?): {scr.error}")
    assert scr.ran, "scalar-only candidate did not run"
    assert "y" in scr.outputs


def test_child_merge_strips_empty_sentinel(tmp_path):
    """Unit-level: the real child `_merge` rebuilds a scalar-only inputs
    dict with the `__empty__` placeholder removed."""
    tp, sp = pi._materialise_inputs(
        {"alpha": 1.5, "beta": 7, "flag": True}, tmp_path, "chk")
    merged = _run_child_merge(tp, sp)
    assert "__empty__" not in merged, (
        f"__empty__ sentinel leaked into candidate inputs: {sorted(merged)}")
    assert merged == {"alpha": 1.5, "beta": 7, "flag": True}


def _run_child_merge(tensors_path: str, sidecar_path: str) -> dict:
    """Execute the candidate child's real `_merge` from the `_CHILD`
    source so the test pins the actual code path, not a copy."""
    import ast
    import json

    from safetensors.torch import load_file as _lf

    # Parse the _CHILD source and pull out `_merge` + its `_to_device` helper.
    # `_BACKEND=None` makes `_to_device` a no-op (dev box), so the test pins
    # the real sentinel-stripping path without an accelerator.
    tree = ast.parse(pi._CHILD)
    defs = {n.name: n for n in tree.body
            if isinstance(n, ast.FunctionDef) and n.name in ("_to_device", "_merge")}
    ns: dict = {"load_file": _lf, "json": json, "Path": Path, "_BACKEND": None}
    exec(  # noqa: S102 — running our own vetted child snippet, not input.
        compile(ast.Module(body=[defs["_to_device"], defs["_merge"]],
                           type_ignores=[]), "<child-merge>", "exec"),
        ns)
    return ns["_merge"](tensors_path, sidecar_path)
