"""W4.3 (R3-1 / R3-2) — mode-aware golden routing + task_status three-state.

Pins the W4.3 contract on top of W2.1:

  * a non-``upstream_oracle_required`` task uses the in-process PyTorch
    oracle as golden and is ``task_status = "evaluated"`` in both modes;
  * an ``upstream_oracle_required`` task graded ``self-reported`` has no
    provider golden ⇒ ``task_status = "unsupported"`` with null perf;
  * an ``upstream_oracle_required`` task graded ``verified`` loads the
    offline ``golden/`` artifacts — complete ⇒ ``"evaluated"``,
    missing/incomplete ⇒ ``"blocked"`` (fail-closed) with null perf;
  * on the dev machine (no ``golden/`` directory) every verified-mode
    upstream task is ``blocked`` — the expected pre-Phase-4 state;
  * ``analyze`` derives the three states from trusted repo state
    (meta + the presence of golden artifacts), never from the result.

Dev machine has no NPU — these verify the routing plumbing and the
fail-closed contract, not NPU numeric reality.
"""
from __future__ import annotations

import json
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
import torch
import yaml
from safetensors.torch import save_file

from ascendbench.engine import eval as _eval
from ascendbench.perf import scoring
from ascendbench.integrity import static_checker
from ascendbench.corpus.dataset import TaskDef
from ascendbench.release_gate.eligibility import (
    compute_verified_blocked,
    golden_complete,
)
from ascendbench.engine.eval import _resolve_golden, evaluate

REPO = Path(__file__).resolve().parents[2]


@pytest.fixture
def static_pass(monkeypatch):
    """Make the static checker pass so the golden-routing block (the W4.3
    contract under test) is reached. The static gate is orthogonal to
    golden routing and is pinned by its own test module — a synthetic
    plain-torch solution would otherwise be blocked before routing runs.
    """
    # An empty CheckReport has no `block`-severity finding ⇒ not blocked.
    monkeypatch.setattr(_eval.static_checker, "check_solution",
                        lambda *a, **k: static_checker.CheckReport())


# --- L2_03_act_and_mul: a real upstream_oracle_required task. -----
_UPSTREAM_TASK = "L2_03_act_and_mul"


# ---------------------------------------------------------------------------
# Synthetic upstream task with a self-contained oracle + shapes + golden.
# ---------------------------------------------------------------------------
_ORACLE_SRC = textwrap.dedent('''\
    import torch


    def run(inputs):
        return {"y": inputs["x"] * 2.0}
''')

_SOLUTION_SRC = textwrap.dedent('''\
    import torch


    def npu_triton_impl(inputs):
        return {"y": inputs["x"] * 2.0}
''')

_SHAPES = {
    "size_knobs": {"B": 4},
    "public_shapes": [
        {"id": "P1", "name": "tiny", "knobs": {"B": 2}},
        {"id": "P2", "name": "small", "knobs": {"B": 4}},
    ],
    "hidden_shapes": [
        {"id": "H1", "name": "hid", "knobs": {"B": 3}},
    ],
    "inputs": {
        "x": {"dims": ["B"], "dtype": "float32", "dist": "randn"},
    },
}

# WA.4: a reconciled (`scaffold: false`), semantically-valid
# `param_constraints` block for `_SHAPES` — these W4.3 tests exercise the
# `oracle_simplified` x `mode` status routing, so the WA.4 param_constraints
# gate must PASS here (it would otherwise mark every task `blocked`).
# R4-1: a `scaffold: false` block must be schema-closed (all five semantic
# subsections present); the unused ones are empty containers.
_PARAM_CONSTRAINTS = {
    "scaffold": False,
    "dims": [],
    "dtypes": {"x": "float32"},
    "dists": {"x": {"kind": "randn"}},
    "scalars": {},
    "tensor_invariants": [],
}

_META = {
    "task_id": "SYN_upstream",
    "level": "L1",
    "tolerance_class": "elementwise_exact_or_inf_safe",
    "oracle_simplified": "upstream_oracle_required",
    "runner_checks": {},
    "upstream": {"commit_sha": "deadbeef"},
    "param_constraints": _PARAM_CONSTRAINTS,
}


def _make_synthetic_task(tmp_path: Path, *, oracle_tier: str,
                         with_golden: bool, partial_golden: bool = False):
    """Build a synthetic task dir + return (TaskDef, solution_path).

    `oracle_tier` sets `meta.oracle_simplified`. `with_golden` writes the
    offline `golden/<shape_id>.safetensors` artifacts; `partial_golden`
    omits one hidden shape's artifact to exercise the incomplete-set path.
    """
    tdir = tmp_path / "task"
    tdir.mkdir()
    (tdir / "oracle_pytorch.py").write_text(_ORACLE_SRC)
    (tdir / "shapes.yaml").write_text(yaml.safe_dump(_SHAPES))
    (tdir / "contract.md").write_text("# synthetic\n")
    (tdir / "kernel.py").write_text("# frozen\n")
    meta = dict(_META, oracle_simplified=oracle_tier)
    (tdir / "meta.yaml").write_text(yaml.safe_dump(meta))

    if with_golden:
        gdir = tdir / "golden"
        gdir.mkdir()
        # Golden = oracle output (x*2) on the seed-0 correctness input.
        # `partial_golden` omits H1 to exercise the incomplete-set path.
        from ascendbench.input_spec import inputs as _inputs
        for sid in ("P1", "P2", "H1"):
            if partial_golden and sid == "H1":
                continue
            built = _inputs.build_inputs(_SHAPES, sid, seed=0)
            save_file({"y": built["x"] * 2.0},
                      str(gdir / f"{sid}.safetensors"))

    task = TaskDef(
        task_id="SYN_upstream",
        level=1,
        task_dir=tdir,
        meta=meta,
        public_shapes=[],
        contract_md="# synthetic\n",
        kernel_py="# frozen\n",
        kernel_py_sha256="0" * 64,
        param_constraints=_PARAM_CONSTRAINTS,
    )
    sol = tmp_path / "solution.py"
    sol.write_text(_SOLUTION_SRC)
    return task, sol


# ---------------------------------------------------------------------------
# _resolve_golden routing.
# ---------------------------------------------------------------------------

def test_resolve_golden_non_upstream_uses_oracle(tmp_path):
    """A non-upstream task routes to the in-process PyTorch oracle."""
    task, _ = _make_synthetic_task(
        tmp_path, oracle_tier="simple_pytorch_ok", with_golden=False)
    status, kind, offline = _resolve_golden(task, "verified", ["P1"])
    assert status == scoring.TASK_STATUS_EVALUATED
    assert kind == "oracle"
    assert offline is None


def test_resolve_golden_upstream_self_reported_unsupported(tmp_path):
    """An upstream task in self-reported mode has no provider golden."""
    task, _ = _make_synthetic_task(
        tmp_path, oracle_tier="upstream_oracle_required", with_golden=False)
    status, kind, offline = _resolve_golden(
        task, "self-reported", ["P1", "P2"])
    assert status == "unsupported"
    assert kind == "none"
    assert offline is None


def test_resolve_golden_upstream_verified_no_golden_blocked(tmp_path):
    """An upstream task in verified mode with no golden/ dir is blocked."""
    task, _ = _make_synthetic_task(
        tmp_path, oracle_tier="upstream_oracle_required", with_golden=False)
    status, kind, _ = _resolve_golden(
        task, "verified", ["P1", "P2", "H1"])
    assert status == "blocked"
    assert kind == "none"


def test_resolve_golden_upstream_verified_partial_golden_blocked(tmp_path):
    """An incomplete golden/ set (missing a graded shape) is blocked."""
    task, _ = _make_synthetic_task(
        tmp_path, oracle_tier="upstream_oracle_required",
        with_golden=True, partial_golden=True)
    status, _, _ = _resolve_golden(
        task, "verified", ["P1", "P2", "H1"])
    assert status == "blocked"


def test_resolve_golden_upstream_verified_complete_golden_evaluated(tmp_path):
    """A complete golden/ set in verified mode routes to the offline golden."""
    task, _ = _make_synthetic_task(
        tmp_path, oracle_tier="upstream_oracle_required", with_golden=True)
    status, kind, offline = _resolve_golden(
        task, "verified", ["P1", "P2", "H1"])
    assert status == scoring.TASK_STATUS_EVALUATED
    assert kind == "offline"
    assert set(offline.keys()) == {"P1", "P2", "H1"}
    assert torch.is_tensor(offline["P1"]["y"])


# ---------------------------------------------------------------------------
# evaluate() end-to-end: task_status three-state + null perf.
# ---------------------------------------------------------------------------

def test_evaluate_upstream_self_reported_is_unsupported(tmp_path, static_pass):
    """An upstream task graded self-reported ⇒ task_status="unsupported",
    perf fields null, candidate never run."""
    task, sol = _make_synthetic_task(
        tmp_path, oracle_tier="upstream_oracle_required", with_golden=False)
    result = evaluate("SYN_upstream", sol, task_def=task,
                      mode="self-reported")
    assert result.task_status == "unsupported"
    assert result.s_t is None
    assert result.shape_results == []
    tm = result.task_measurement()
    assert tm.task_status == "unsupported"
    assert tm.measured_ns is None and tm.baseline_ns is None
    assert tm.s_t is None


def test_evaluate_upstream_verified_no_golden_is_blocked(tmp_path, static_pass):
    """An upstream task graded verified with no golden/ ⇒ blocked,
    perf fields null (fail-closed)."""
    task, sol = _make_synthetic_task(
        tmp_path, oracle_tier="upstream_oracle_required", with_golden=False)
    result = evaluate("SYN_upstream", sol, task_def=task, mode="verified")
    assert result.task_status == "blocked"
    assert result.s_t is None
    assert result.shape_results == []
    assert result.error_reason is not None
    assert "golden" in result.error_reason


def test_evaluate_upstream_verified_complete_golden_is_evaluated(tmp_path, static_pass):
    """An upstream task graded verified with a complete golden/ set is
    evaluated against the offline golden (the candidate matches it)."""
    task, sol = _make_synthetic_task(
        tmp_path, oracle_tier="upstream_oracle_required", with_golden=True)
    result = evaluate("SYN_upstream", sol, task_def=task, mode="verified")
    assert result.task_status == scoring.TASK_STATUS_EVALUATED
    graded = {r.shape_id for r in result.shape_results}
    assert graded == {"P1", "P2", "H1"}
    # The candidate (x*2) matches the offline golden (x*2).
    assert all(r.correctness_passed for r in result.shape_results), [
        (r.shape_id, r.correctness_reason) for r in result.shape_results
    ]


def test_evaluate_non_upstream_is_evaluated(tmp_path, static_pass):
    """A non-upstream task is evaluated normally in both modes (oracle)."""
    task, sol = _make_synthetic_task(
        tmp_path, oracle_tier="simple_pytorch_ok", with_golden=False)
    for mode in ("self-reported", "verified"):
        result = evaluate("SYN_upstream", sol, task_def=task, mode=mode)
        assert result.task_status == scoring.TASK_STATUS_EVALUATED, mode
        assert result.shape_results, mode


def test_real_upstream_task_self_reported_unsupported(tmp_path, static_pass):
    """The real `L2_03_act_and_mul` (upstream_oracle_required) is
    `unsupported` in self-reported mode — no synthetic fixture needed."""
    sol = tmp_path / "solution.py"
    sol.write_text(_SOLUTION_SRC)
    result = evaluate(_UPSTREAM_TASK, sol, mode="self-reported")
    assert result.task_status == "unsupported"
    assert result.s_t is None


def test_real_upstream_task_verified_blocked_on_dev_machine(tmp_path, static_pass):
    """The real upstream task is `blocked` in verified mode on the dev
    machine — no `golden/` directory exists (pre-Phase-4 state)."""
    sol = tmp_path / "solution.py"
    sol.write_text(_SOLUTION_SRC)
    result = evaluate(_UPSTREAM_TASK, sol, mode="verified")
    assert result.task_status == "blocked"


# ---------------------------------------------------------------------------
# eligibility: golden completeness + verified-blocked set.
# ---------------------------------------------------------------------------

def test_golden_complete_false_without_golden_dir(tmp_path):
    task, _ = _make_synthetic_task(
        tmp_path, oracle_tier="upstream_oracle_required", with_golden=False)
    assert golden_complete(task) is False


def test_golden_complete_false_on_partial_set(tmp_path):
    task, _ = _make_synthetic_task(
        tmp_path, oracle_tier="upstream_oracle_required",
        with_golden=True, partial_golden=True)
    assert golden_complete(task) is False


def test_golden_complete_true_on_full_set(tmp_path):
    task, _ = _make_synthetic_task(
        tmp_path, oracle_tier="upstream_oracle_required", with_golden=True)
    assert golden_complete(task) is True


def test_compute_verified_blocked_includes_all_upstream_on_dev_machine():
    """On the dev machine no task ships a `golden/` directory, so every
    `upstream_oracle_required` task is verified-blocked."""
    from ascendbench.corpus.dataset import iter_all_tasks
    from ascendbench.release_gate.eligibility import (
        compute_upstream_oracle_required,
    )

    tasks = iter_all_tasks()
    upstream = compute_upstream_oracle_required(tasks)
    blocked = compute_verified_blocked(tasks)
    assert upstream, "expected some upstream_oracle_required tasks"
    assert blocked == upstream, (
        "every upstream task should be blocked pre-Phase-4 "
        f"(missing: {upstream - blocked})"
    )


# ---------------------------------------------------------------------------
# analyze() derives the three states from trusted repo state.
# ---------------------------------------------------------------------------

def _run_tab(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "ascendbench.engine.cli", *args],
        cwd=REPO, capture_output=True, text=True,
    )


def test_analyze_verified_mode_marks_upstream_blocked(tmp_path):
    """In verified mode `analyze` assigns `task_status="blocked"` to the
    upstream tasks (golden/ missing on the dev machine), derived from repo
    state — NOT from the result's reported `task_status`."""
    from ascendbench.corpus.dataset import iter_all_tasks
    from ascendbench.release_gate.eligibility import (
        compute_score_eligible,
        compute_verified_blocked,
    )

    tasks = iter_all_tasks()
    eligible = compute_score_eligible(tasks)
    blocked = compute_verified_blocked(tasks) & eligible
    if not blocked:
        return  # no eligible upstream task — nothing to assert
    a_blocked = sorted(blocked)[0]

    # The result claims "evaluated" — analyze must override to "blocked".
    suite = {
        "mode": "verified",
        "results": [
            {"task_id": a_blocked,
             "correctness_status": "ok", "runtime_status": "ok",
             "performance_status": "ok", "performance_flags": [],
             "fallback_detected": False,
             "aggregate_score": 1.0, "s_t": 1.0,
             "task_status": "evaluated"},
        ],
    }
    suite_path = tmp_path / "suite.json"
    out_path = tmp_path / "score.json"
    suite_path.write_text(json.dumps(suite))

    r = _run_tab("analyze", "--verified-results", str(suite_path),
                 "--submitter", "t", "--out", str(out_path))
    assert r.returncode == 0, f"analyze failed: {r.stdout} {r.stderr}"
    doc = json.loads(out_path.read_text())
    assert doc["mode"] == "verified"
    # The result claimed "evaluated" but repo state says "blocked" — a
    # counted mismatch (analyze never trusts the result's task_status).
    assert doc["task_status_mismatch_count"] == 1
