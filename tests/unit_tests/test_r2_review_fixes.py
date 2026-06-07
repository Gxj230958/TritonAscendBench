"""R2 codex-review regression tests — four MUST-FIX gaps.

Pins the four R2 review findings so a regression re-surfaces as a test
failure:

  * R2-1 — a timing pass that timed out / crashed / shipped no per-shape
    timing data must NOT yield an ok/ok/ok verdict with an empty timing
    series. `performance_status` is promoted to "ok" only when the
    timing pass really succeeded.
  * R2-2 — the NPU event timer calls `start.elapsed_time(end)` (the
    PyTorch / torch_npu order), not the reversed `end.elapsed_time(start)`.
  * R2-3 — Roofline scoring threads the task's real representative
    `dtype_size` into `compute_t_ref` / `FormulaEnv`, so a float32 task
    scores with `dtype_size = 4`, not the fp16 default `2.0`.
  * R2-4 — `analyze` takes the per-task `task_status` from trusted repo
    metadata under the suite's top-level `mode`, never verbatim from the
    result — a hand-edited `unsupported` cannot drop a failing task out
    of the scored denominator.

Dev machine has no NPU — the R2-2 test mocks `torch.npu.Event`.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path
from unittest import mock

import yaml

from triton_ascend_bench import process_isolation as pi
from triton_ascend_bench.dataset import TaskDef
from triton_ascend_bench.eval import evaluate
from triton_ascend_bench.process_isolation import (
    CandidateRunResult,
    ShapeCandidateResult,
    TwoPassResult,
)

REPO = Path(__file__).resolve().parents[2]


# ===========================================================================
# Shared task fixture (mirrors test_r1_review_fixes — a pure-torch oracle +
# a static-clean candidate stub; the two-pass run is monkeypatched).
# ===========================================================================

_ORACLE_SRC = (
    "import torch\n"
    "def run(inputs):\n"
    "    x = inputs['x']\n"
    "    return {'y': x + 1.0}\n"
)
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

_TENSOR_INPUTS = {
    "x": {"dims": ["B", "N"], "dtype": "float32", "dist": "randn"},
}


def _make_task(td: Path, *, inputs_spec: dict) -> tuple[TaskDef, Path]:
    """Materialise a task dir + a static-clean solution under ``td``."""
    kernel_src = "@triton.jit\ndef k(i):\n    return {}\n"
    (td / "kernel.py").write_text(kernel_src)
    (td / "oracle_pytorch.py").write_text(_ORACLE_SRC)
    (td / "contract.md").write_text("# Contract — R2 task\n")
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
        "schema_version": "1.0", "task_id": "L1_99_r2",
        "level": "L1", "idx": 99, "name": "r2", "repo": "vLLM",
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
        task_id="L1_99_r2", level=1, task_dir=td, meta=meta,
        public_shapes=[], contract_md="", kernel_py=kernel_src,
        kernel_py_sha256=hashlib.sha256(kernel_src.encode()).hexdigest(),
    )
    sol = td / "solution.py"
    sol.write_text(_SOLUTION_SRC)
    return task, sol


def _two_pass_factory(task: TaskDef, *, timing_kind: str = "clean"):
    """Build a `run_two_pass` replacement. ``timing_kind`` selects the
    timing pass's outcome:

    * ``"clean"``       — correct candidate, clean ~5 ms host-timing series.
    * ``"timeout"``     — the timing-pass child timed out (no per-shape data).
    * ``"crash"``       — the timing-pass child failed to load (load_error).
    * ``"empty"``       — the timing-pass child ran but shipped NO durations.

    The correctness pass is always clean — R2-1 is about the timing pass
    failing INDEPENDENTLY of a clean correctness pass.
    """
    import importlib.util

    import torch

    from triton_ascend_bench import inputs as _inputs

    omod_spec = importlib.util.spec_from_file_location(
        "_r2_oracle", task.task_dir / "oracle_pytorch.py")
    omod = importlib.util.module_from_spec(omod_spec)
    omod_spec.loader.exec_module(omod)
    shapes_doc = yaml.safe_load((task.task_dir / "shapes.yaml").read_text())
    pub, _hid = _inputs.shape_ids(shapes_doc)

    def _fake_run_two_pass(solution_path, correctness_inputs,
                           timing_inputs, **kw):
        corr_shapes: dict[str, ShapeCandidateResult] = {}
        time_shapes: dict[str, ShapeCandidateResult] = {}
        for sid in pub:
            golden = omod.run(correctness_inputs[sid])
            corr_shapes[sid] = ShapeCandidateResult(
                shape_id=sid, ran=True,
                outputs={k: v for k, v in golden.items()
                         if torch.is_tensor(v)})
            if timing_kind == "clean":
                time_shapes[sid] = ShapeCandidateResult(
                    shape_id=sid, ran=True, durations=[5_000_000.0] * 8)
            elif timing_kind == "empty":
                # The child ran but produced NO host-timing series.
                time_shapes[sid] = ShapeCandidateResult(
                    shape_id=sid, ran=True, durations=[])
            # "timeout" / "crash": no per-shape timing records at all.

        if timing_kind == "timeout":
            timing = CandidateRunResult(
                ran_any=False, timed_out=True, load_error=None,
                per_shape={}, pass_kind="timing")
        elif timing_kind == "crash":
            timing = CandidateRunResult(
                ran_any=False, timed_out=False,
                load_error="ImportError: triton not installed",
                per_shape={}, pass_kind="timing")
        else:
            timing = CandidateRunResult(
                ran_any=True, timed_out=False, load_error=None,
                per_shape=time_shapes, pass_kind="timing")

        return TwoPassResult(
            correctness=CandidateRunResult(
                ran_any=True, timed_out=False, load_error=None,
                per_shape=corr_shapes),
            timing=timing,
            scrubbed=[])

    return _fake_run_two_pass


# ===========================================================================
# R2-1 — a failed timing pass must NOT promote performance_status to "ok".
# ===========================================================================

def test_r2_1_clean_timing_promotes_performance_status(tmp_path, monkeypatch):
    """Baseline: a CLEAN timing pass still promotes performance_status to
    "ok" — the R2-1 gate must not break the happy path."""
    task, sol = _make_task(tmp_path, inputs_spec=_TENSOR_INPUTS)
    monkeypatch.setattr(
        pi, "run_two_pass", _two_pass_factory(task, timing_kind="clean"))

    result = evaluate("L1_99_r2", sol, task_def=task)

    assert result.error_reason is None, result.error_reason
    assert result.correctness_status == "ok"
    assert result.runtime_status == "ok"
    assert result.performance_status == "ok"


def test_r2_1_timing_timeout_does_not_promote(tmp_path, monkeypatch):
    """A timing pass that timed out must NOT yield ok/ok/ok with an empty
    timing series — `performance_status` stays "error" and the timeout
    surfaces on `runtime_status`."""
    task, sol = _make_task(tmp_path, inputs_spec=_TENSOR_INPUTS)
    monkeypatch.setattr(
        pi, "run_two_pass", _two_pass_factory(task, timing_kind="timeout"))

    result = evaluate("L1_99_r2", sol, task_def=task)

    assert result.error_reason is None, result.error_reason
    # Correctness pass succeeded — that verdict stays.
    assert result.correctness_status == "ok"
    # R2-1: the timing pass failed ⇒ performance NOT promoted.
    assert result.performance_status == "error", (
        f"timing timeout must not promote performance_status, "
        f"got {result.performance_status!r}")
    # The timing-pass timeout is reflected on runtime_status.
    assert result.runtime_status == "timeout", result.runtime_status
    # No per-shape timing data was produced.
    assert all(not r.durations for r in result.shape_results)


def test_r2_1_timing_crash_does_not_promote(tmp_path, monkeypatch):
    """A timing pass that crashed (load_error) must NOT promote
    performance_status — it stays "error", runtime_status becomes "error"."""
    task, sol = _make_task(tmp_path, inputs_spec=_TENSOR_INPUTS)
    monkeypatch.setattr(
        pi, "run_two_pass", _two_pass_factory(task, timing_kind="crash"))

    result = evaluate("L1_99_r2", sol, task_def=task)

    assert result.error_reason is None, result.error_reason
    assert result.correctness_status == "ok"
    assert result.performance_status == "error", (
        f"timing crash must not promote performance_status, "
        f"got {result.performance_status!r}")
    assert result.runtime_status == "error", result.runtime_status


def test_r2_1_timing_empty_durations_does_not_promote(tmp_path, monkeypatch):
    """A timing pass that ran but shipped EMPTY per-shape durations must
    NOT promote performance_status — an ok/ok/ok with no timing series is
    exactly the bug R2-1 closes."""
    task, sol = _make_task(tmp_path, inputs_spec=_TENSOR_INPUTS)
    monkeypatch.setattr(
        pi, "run_two_pass", _two_pass_factory(task, timing_kind="empty"))

    result = evaluate("L1_99_r2", sol, task_def=task)

    assert result.error_reason is None, result.error_reason
    assert result.correctness_status == "ok"
    assert result.performance_status == "error", (
        f"empty timing series must not promote performance_status, "
        f"got {result.performance_status!r}")
    # Empty timing ⇒ no usable durations on any shape.
    assert all(not r.durations for r in result.shape_results)


# ===========================================================================
# R2-2 — the NPU event timer calls start.elapsed_time(end), not reversed.
# ===========================================================================

def test_r2_2_event_timer_calls_elapsed_time_on_start_event():
    """The NPU `EventTimer` must call `elapsed_time` on the START event
    with the END event as the argument — the PyTorch / torch_npu API
    order `start.elapsed_time(end)`. The reversed form returns a wrong /
    negative duration on a real 910B.

    R6-1: the timer is a context manager that BRACKETS one externally-made
    call — it never runs the candidate itself. The caller invokes the
    candidate inside the `with` block, exactly once, so a measured trial
    is one candidate call (host-timed AND event-bracketed). This test
    asserts the record ORDER and the `elapsed_time` argument order on the
    mocked Event objects.

    Dev machine has no NPU, so `torch.npu` and the candidate `fn` are
    mocked.
    """
    from triton_ascend_bench.backends.npu import NpuBackend

    backend = NpuBackend()

    # The two mocked events the timer will record(). `record_order` lets
    # us confirm `ev_start` is recorded before `ev_end`.
    record_order: list[str] = []

    def _make_event(name: str):
        ev = mock.MagicMock(name=name)
        ev._tab_name = name
        ev.record.side_effect = lambda n=name: record_order.append(n)
        # elapsed_time on the start event returns 3 ms.
        ev.elapsed_time.return_value = 3.0
        return ev

    ev_start = _make_event("start")
    ev_end = _make_event("end")
    events = iter([ev_start, ev_end])

    fake_npu = mock.MagicMock()
    fake_npu.Event.side_effect = lambda *a, **kw: next(events)

    # Force `_NPU_AVAILABLE` so the real timer (not the dev-machine None)
    # is returned, and patch `torch.npu` with the mock.
    with mock.patch("triton_ascend_bench.backends.npu._NPU_AVAILABLE", True), \
            mock.patch("torch.npu", fake_npu, create=True):
        timer = backend.event_timer()
        assert timer is not None, "timer must be live when _NPU_AVAILABLE"

        called: list[str] = []
        # R6-1: the candidate runs ONCE, inside the bracket, made by the
        # caller — the timer does not invoke it.
        with timer:
            called.append("ran")
        ns = timer.elapsed_ns()

    # The candidate fn ran exactly once, between the two record() calls.
    assert called == ["ran"]
    assert record_order == ["start", "end"], record_order

    # R2-2: elapsed_time was invoked on the START event, NOT the end event.
    ev_start.elapsed_time.assert_called_once_with(ev_end)
    ev_end.elapsed_time.assert_not_called()

    # 3 ms -> 3e6 ns.
    assert ns == 3_000_000, ns


# ===========================================================================
# R2-3 — Roofline scoring threads the task's real dtype_size into the
# cost-model formula evaluation (a float32 task uses 4, not the fp16 2.0).
# ===========================================================================

def test_r2_3_float32_inputs_resolve_dtype_size_4():
    """`_representative_dtype_size` reads the first declared tensor input's
    dtype — a float32 primary operand resolves to 4 bytes."""
    from triton_ascend_bench.eval import _representative_dtype_size

    doc = {
        "inputs": {
            "x": {"dims": ["B", "N"], "dtype": "float32", "dist": "randn"},
            "w": {"dims": ["N"], "dtype": "float32", "dist": "ones"},
        }
    }
    assert _representative_dtype_size(doc) == 4.0


def test_r2_3_dtype_size_skips_scalars_and_covers_dtypes():
    """The rule skips `scalar` / `scalar_knob` entries and resolves the
    first real tensor input; a scalar-only block yields None (fp16
    default applies)."""
    from triton_ascend_bench.eval import _representative_dtype_size

    # First entry is a scalar — skipped; the int64 tensor is representative.
    doc = {
        "inputs": {
            "eps": {"scalar": 1e-5},
            "idx": {"dims": ["B"], "dtype": "int64", "dist": "arange"},
        }
    }
    assert _representative_dtype_size(doc) == 8.0

    # fp16 tensor -> 2 bytes.
    doc_fp16 = {"inputs": {"x": {"dims": ["B"], "dtype": "float16"}}}
    assert _representative_dtype_size(doc_fp16) == 2.0

    # Scalar-only inputs block -> None (caller falls back to fp16 default).
    doc_scalar = {"inputs": {"eps": {"scalar": 1.0}}}
    assert _representative_dtype_size(doc_scalar) is None


def test_r2_3_roofline_uses_real_dtype_size(tmp_path, monkeypatch):
    """End-to-end: a float32 task whose `cost_model.bytes` references
    `dtype_size` is scored with `dtype_size = 4`. The disclosed
    `m_per_resource` must reflect the float32 element size, not the fp16
    default — proof `eval` threaded the real dtype_size into FormulaEnv /
    compute_t_ref.
    """
    from triton_ascend_bench import eval as _eval
    from triton_ascend_bench import peak_table as _pt
    from triton_ascend_bench.cost_model import parse_cost_model

    task, sol = _make_task(tmp_path, inputs_spec=_TENSOR_INPUTS)
    # A cost model whose bytes formula is purely `... * dtype_size`, so the
    # disclosed M_r scales linearly with the element size.
    task = TaskDef(
        task_id=task.task_id, level=task.level, task_dir=task.task_dir,
        meta=task.meta, public_shapes=task.public_shapes,
        contract_md=task.contract_md, kernel_py=task.kernel_py,
        kernel_py_sha256=task.kernel_py_sha256,
        cost_model=parse_cost_model({
            "flops": "2*B*N", "flops_class": "fp16_cube",
            "bytes": "B*N*dtype_size", "bytes_resource": "hbm",
        }),
    )

    # A fully-verified synthetic peak table so the Roofline path runs.
    monkeypatch.setattr(
        _pt, "load_peak_table",
        lambda require_verified=True: ({"fp16_cube": 1e12}, {"hbm": 1e12}))
    monkeypatch.setattr(
        pi, "run_two_pass", _two_pass_factory(task, timing_kind="clean"))

    result = evaluate(task.task_id, sol, task_def=task)
    assert result.error_reason is None, result.error_reason

    graded = [r for r in result.shape_results if r.roofline_error is None]
    assert graded, [r.roofline_error for r in result.shape_results]
    for r in graded:
        # P1: B=2, N=4 -> B*N = 8 elements. With float32 dtype_size=4,
        # m_per_resource['hbm'] = 8 * 4 = 32, NOT 8 * 2 = 16 (the fp16
        # default the bug used).
        knob_bn = {"P1": 2 * 4, "P2": 3 * 5}[r.shape_id]
        assert r.m_per_resource["hbm"] == knob_bn * 4.0, (
            f"{r.shape_id}: dtype_size should be 4 (float32), "
            f"got M_r={r.m_per_resource['hbm']!r} for B*N={knob_bn}")


# ===========================================================================
# R2-4 — analyze takes task_status from repo authority, not the result.
# ===========================================================================

def _run_tab(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "triton_ascend_bench.cli", *args],
        capture_output=True, text=True, cwd=str(REPO))


def _n_eligible_upstream_oracle() -> int:
    """Count of tasks that are BOTH score-eligible AND
    upstream_oracle_required — exactly the set the self-reported
    denominator drops relative to the full score-eligible set. Derived
    from the live repo metadata so the test does not hard-code a number
    that drifts as the corpus changes."""
    from triton_ascend_bench.dataset import iter_all_tasks
    from triton_ascend_bench.eligibility import (
        compute_score_eligible,
        compute_upstream_oracle_required,
    )

    tasks = iter_all_tasks()
    eligible = compute_score_eligible(tasks)
    upstream = compute_upstream_oracle_required(tasks)
    return len(eligible & upstream)


def test_r2_4_self_reported_denominator_excludes_upstream_oracle(tmp_path):
    """R2-4: in self-reported mode the denominator is the fixed
    repo-derived score-eligible set MINUS the upstream_oracle_required
    set. An `upstream_oracle_required` task's result is dropped from the
    scored set regardless of the `task_status` it reports."""
    verified = {
        "mode": "self-reported",
        "results": [
            {"task_id": "L1_01_bias",
             "static_checks_passed": True, "correctness_status": "ok",
             "runtime_status": "ok", "performance_status": "ok",
             "performance_flags": [], "fallback_detected": False,
             "aggregate_score": 1.0, "s_t": 1.0, "task_status": "evaluated"},
            # upstream_oracle_required — has no provider golden in
            # self-reported mode; excluded by the fixed formula.
            {"task_id": "L2_03_act_and_mul",
             "static_checks_passed": True, "correctness_status": "fail",
             "runtime_status": "error", "performance_status": "error",
             "performance_flags": [], "fallback_detected": False,
             "aggregate_score": 0.0, "s_t": None,
             "task_status": "evaluated"},
        ],
    }
    (tmp_path / "v.json").write_text(json.dumps(verified))
    out = tmp_path / "score.json"
    r = _run_tab("analyze", "--verified-results", str(tmp_path / "v.json"),
                 "--out", str(out))
    assert r.returncode == 0, r.stderr
    doc = json.loads(out.read_text())

    assert doc["mode"] == "self-reported"
    # The upstream_oracle_required task is excluded from the scored set.
    assert doc["upstream_oracle_excluded_count"] == 1
    assert doc["n_native_measurements"] == 1
    # The self-reported denominator drops every score-eligible
    # upstream_oracle_required task — exactly that intersection.
    assert (doc["n_score_eligible_tasks"]
            - doc["n_scored_denominator_tasks"]) == _n_eligible_upstream_oracle()


def test_r2_4_bad_result_cannot_drop_failing_task_via_unsupported(tmp_path):
    """R2-4: a hand-edited result that marks a genuinely score-eligible,
    non-upstream-oracle task `unsupported` must NOT remove it from the
    scored denominator. The result's `task_status` is consistency-checked
    (counted as a mismatch) but the task is still measured/scored.
    """
    verified = {
        "mode": "self-reported",
        "results": [
            # A score-eligible non-upstream-oracle task FAILED — and the
            # result lies, claiming "unsupported" to escape the geomean.
            # WA.4: use a RECONCILED task (`param_constraints.scaffold:
            # false`, semantically clean) so the repo-authoritative status
            # is genuinely "evaluated" — a scaffold task would itself be
            # "blocked" by the WA.4 gate, which is a different code path.
            {"task_id": "L2_04_silu_and_mul",
             "static_checks_passed": True, "correctness_status": "fail",
             "runtime_status": "error", "performance_status": "error",
             "performance_flags": [], "fallback_detected": False,
             "aggregate_score": 0.0, "s_t": None,
             "task_status": "unsupported"},
        ],
    }
    (tmp_path / "v.json").write_text(json.dumps(verified))
    out = tmp_path / "score.json"
    r = _run_tab("analyze", "--verified-results", str(tmp_path / "v.json"),
                 "--out", str(out))
    assert r.returncode == 0, r.stderr
    doc = json.loads(out.read_text())

    # R2-4: the cheat fails — the task is still a measurement (not dropped
    # to non_evaluated), so it contributes to the scored set.
    assert doc["n_native_measurements"] == 1
    assert doc["non_evaluated_task_count"] == 0, (
        "a non-upstream-oracle task must not be droppable via a "
        "hand-edited task_status=unsupported")
    # The lie is recorded as a consistency mismatch.
    assert doc["task_status_mismatch_count"] == 1
    # The denominator is the fixed repo-derived set, not shrunk by the lie.
    assert doc["n_scored_denominator_tasks"] > 0


def test_r2_4_verified_mode_denominator_is_full_eligible_set(tmp_path):
    """R2-4: in verified mode the denominator is the fixed full
    score-eligible set |S| — the upstream_oracle_required tasks have
    offline golden there and are NOT excluded."""
    verified = {
        "mode": "verified",
        "results": [
            {"task_id": "L2_03_act_and_mul",
             "static_checks_passed": True, "correctness_status": "ok",
             "runtime_status": "ok", "performance_status": "ok",
             "performance_flags": [], "fallback_detected": False,
             "aggregate_score": 1.0, "s_t": 1.0, "task_status": "evaluated"},
        ],
    }
    (tmp_path / "v.json").write_text(json.dumps(verified))
    out = tmp_path / "score.json"
    r = _run_tab("analyze", "--verified-results", str(tmp_path / "v.json"),
                 "--out", str(out))
    assert r.returncode == 0, r.stderr
    doc = json.loads(out.read_text())

    assert doc["mode"] == "verified"
    # No exclusion in verified mode — the upstream-oracle task is scored.
    assert doc["upstream_oracle_excluded_count"] == 0
    assert doc["n_native_measurements"] == 1
    # Denominator == the full score-eligible set.
    assert doc["n_scored_denominator_tasks"] == doc["n_score_eligible_tasks"]


def test_r2_4_missing_mode_defaults_to_self_reported(tmp_path):
    """R2-4: a suite JSON with no top-level `mode` defaults to the
    stricter self-reported denominator — a malformed doc cannot widen
    the scored set to the verified |S|."""
    verified = {
        "results": [
            {"task_id": "L1_01_bias",
             "static_checks_passed": True, "correctness_status": "ok",
             "runtime_status": "ok", "performance_status": "ok",
             "performance_flags": [], "fallback_detected": False,
             "aggregate_score": 1.0, "s_t": 1.0},
        ],
    }
    (tmp_path / "v.json").write_text(json.dumps(verified))
    out = tmp_path / "score.json"
    r = _run_tab("analyze", "--verified-results", str(tmp_path / "v.json"),
                 "--out", str(out))
    assert r.returncode == 0, r.stderr
    doc = json.loads(out.read_text())

    assert doc["mode"] == "self-reported"
    # Self-reported denominator excludes the score-eligible upstream-oracle
    # tasks — a malformed (mode-less) doc gets the stricter denominator.
    assert (doc["n_score_eligible_tasks"]
            - doc["n_scored_denominator_tasks"]) == _n_eligible_upstream_oracle()
