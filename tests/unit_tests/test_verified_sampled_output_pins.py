"""Pinned regressions: verified-mode sampled-output cheat-path closure.

Pins the three R4 review findings so a regression re-surfaces as a test
failure:

  * R4-1 — for an ``upstream_oracle_required`` task graded in verified
    mode the sampled-output cheat-path closure is ENFORCED, not skipped:
    the runner forces every runner-secret sampled timing trial onto the
    shape's fixed seed-0 input (the offline golden's input) so the
    shipped-up timing output can be reverified against the offline
    ``golden/<shape_id>.safetensors`` artifact. A candidate fast-but-
    wrong only in the timed pass is caught with ``measured_outputs_drift``.
  * R4-2 — ``eval._resolve_golden`` and ``eligibility.golden_complete``
    share ONE "golden usable" predicate (``golden_artifact_usable``):
    path exists AND the safetensors file loads. A corrupt artifact is
    fail-closed in BOTH, so ``evaluate()`` cannot block a task that
    ``compute_verified_blocked()`` calls evaluated (or vice versa).
  * R4-3 — ``validate_tasks._validate_golden`` strictly REQUIRES a
    complete ``golden/`` set for every ``upstream_oracle_required`` task;
    a task that never produced a ``golden/`` directory is a validation
    error, no longer silently clean.

Dev machine has no NPU / golden artifact; these use mock golden to
verify the routing + cheat-path logic, not NPU numeric reality.
"""
from __future__ import annotations

import hashlib
import importlib.util
import sys
import textwrap
from pathlib import Path

import pytest
import torch
import yaml
from safetensors.torch import save_file

from ascendbench.engine import eval as _eval
from ascendbench.integrity import process_isolation as pi
from ascendbench.integrity import static_checker
from ascendbench.corpus.dataset import TaskDef
from ascendbench.release_gate.eligibility import (
    golden_artifact_usable,
    golden_complete,
)
from ascendbench.engine.eval import evaluate

REPO = Path(__file__).resolve().parents[2]

_spec = importlib.util.spec_from_file_location(
    "validate_tasks", REPO / "scripts" / "corpus" / "validate_tasks.py"
)
vt = importlib.util.module_from_spec(_spec)
sys.modules["validate_tasks"] = vt
_spec.loader.exec_module(vt)


# ===========================================================================
# R4-1 — offline-golden sampled-output cheat-path closure.
# ===========================================================================

# A correct candidate: y = x * 2 every call (matches the offline golden).
_HONEST_SOLUTION = textwrap.dedent('''\
    import torch


    def npu_triton_impl(inputs):
        return {"y": inputs["x"] * 2.0}
''')

# A cheating candidate: correct on the first call of a process (the
# correctness pass makes exactly one call per shape), wrong afterwards
# (every timed call returns x*3). Because the runner forces the sampled
# timing trials onto the seed-0 golden input, a sampled trial's shipped-up
# output is x*3 on the seed-0 input — which mismatches the offline golden
# (x*2). The call counter lives in a module global; each pass is a fresh
# subprocess so it resets to 0 per pass.
_CHEAT_SOLUTION = textwrap.dedent('''\
    import torch

    _calls = {"n": 0}


    def npu_triton_impl(inputs):
        _calls["n"] += 1
        if _calls["n"] == 1:
            # correctness pass: one call per shape -> honest answer.
            return {"y": inputs["x"] * 2.0}
        # every later (timed) call -> fast-but-wrong answer.
        return {"y": inputs["x"] * 3.0}
''')

_ORACLE_SRC = textwrap.dedent('''\
    import torch


    def run(inputs):
        return {"y": inputs["x"] * 2.0}
''')

# One shape only -> a small, fast end-to-end grading run.
_SHAPES = {
    "size_knobs": {"B": 4},
    "public_shapes": [
        {"id": "P1", "name": "tiny", "knobs": {"B": 4}},
    ],
    "hidden_shapes": [],
    "inputs": {
        "x": {"dims": ["B"], "dtype": "float32", "dist": "randn"},
    },
}

# WA.4: a reconciled (`scaffold: false`), semantically-valid
# `param_constraints` for `_SHAPES` — these R4-1 tests exercise the
# offline-golden routing, so the WA.4 param_constraints gate must PASS.
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
    "task_id": "SYN_r4_upstream",
    "level": "L1",
    "tolerance_class": "elementwise_exact_or_inf_safe",
    "oracle_simplified": "upstream_oracle_required",
    "runner_checks": {},
    "upstream": {"commit_sha": "deadbeef"},
    "param_constraints": _PARAM_CONSTRAINTS,
}


@pytest.fixture
def static_pass(monkeypatch):
    """Make the static checker pass so the golden-routing + grading block
    (the R4-1 contract under test) is reached."""
    monkeypatch.setattr(_eval.static_checker, "check_solution",
                        lambda *a, **k: static_checker.CheckReport())


def _make_upstream_task(tmp_path: Path, *, solution_src: str,
                        corrupt_golden: bool = False):
    """Build a synthetic upstream_oracle_required task with a complete
    mock offline golden + return (TaskDef, solution_path).

    The golden = the oracle output (x*2) on the seed-0 input — i.e. the
    same input the verified runner uses for the correctness pass and
    (R4-1) forces the sampled timing trials onto.
    """
    from ascendbench.input_spec import inputs as _inputs

    tdir = tmp_path / "task"
    tdir.mkdir()
    (tdir / "oracle_pytorch.py").write_text(_ORACLE_SRC)
    (tdir / "shapes.yaml").write_text(yaml.safe_dump(_SHAPES))
    (tdir / "contract.md").write_text("# synthetic\n")
    (tdir / "kernel.py").write_text("# frozen\n")
    (tdir / "meta.yaml").write_text(yaml.safe_dump(_META))

    gdir = tdir / "golden"
    gdir.mkdir()
    for sid in ("P1",):
        if corrupt_golden:
            (gdir / f"{sid}.safetensors").write_bytes(b"not-safetensors")
            continue
        built = _inputs.build_inputs(_SHAPES, sid, seed=0)
        save_file({"y": built["x"] * 2.0}, str(gdir / f"{sid}.safetensors"))

    task = TaskDef(
        task_id="SYN_r4_upstream",
        level=1,
        task_dir=tdir,
        meta=_META,
        public_shapes=[],
        contract_md="# synthetic\n",
        kernel_py="# frozen\n",
        kernel_py_sha256="0" * 64,
        param_constraints=_PARAM_CONSTRAINTS,
    )
    sol = tmp_path / "solution.py"
    sol.write_text(solution_src)
    return task, sol


# --- _apply_golden_fixed_inputs: only sampled measured trials swapped. ----

def test_apply_golden_fixed_inputs_swaps_only_sampled_measured():
    """`_apply_golden_fixed_inputs` overwrites exactly the runner-secret
    sampled measured trials; warm-up + non-sampled trials keep their
    fresh per-trial inputs."""
    n_warmup = 2
    # 2 warm-up + 4 measured trials. Each "input" is a sentinel string.
    timing_inputs = {"P1": [f"fresh_{i}" for i in range(6)]}
    sampled_indices = {"P1": [1, 3]}  # measured indices (0-based)
    fixed = {"P1": "FIXED_GOLDEN_INPUT"}

    out = pi._apply_golden_fixed_inputs(
        timing_inputs, sampled_indices, fixed, n_warmup)

    payload = out["P1"]
    # positions 0,1 are warm-up -> never swapped.
    assert payload[0] == "fresh_0"
    assert payload[1] == "fresh_1"
    # measured index = position - n_warmup; sampled measured idx 1,3 ->
    # positions 3,5 swapped to the fixed golden input.
    assert payload[2] == "fresh_2"          # measured idx 0, not sampled
    assert payload[3] == "FIXED_GOLDEN_INPUT"  # measured idx 1, sampled
    assert payload[4] == "fresh_4"          # measured idx 2, not sampled
    assert payload[5] == "FIXED_GOLDEN_INPUT"  # measured idx 3, sampled
    # original list not mutated.
    assert timing_inputs["P1"][3] == "fresh_3"


def test_apply_golden_fixed_inputs_skips_shapes_without_fixed():
    """A shape with no fixed-input entry is left untouched."""
    timing_inputs = {"P1": ["a", "b", "c"], "P2": ["d", "e", "f"]}
    out = pi._apply_golden_fixed_inputs(
        timing_inputs, {"P1": [0]}, {"P1": "FIX"}, n_warmup=1)
    assert out["P2"] == ["d", "e", "f"]


# --- _verify_sampled_outputs_offline: match vs drift. ---------------------

def _tol_cls():
    from ascendbench.correctness import tolerance
    pol = tolerance.load_policy()
    return next(iter(pol.values()))


def test_verify_sampled_outputs_offline_matching_passes():
    """Sampled timing outputs that match the offline golden ⇒ no drift."""
    golden = {"y": torch.arange(8, dtype=torch.float32)}
    # Every sampled trial shipped up the golden tensor (it ran on the
    # forced seed-0 input).
    sampled = {}
    for t in (0, 2, 5):
        sampled[f"{t}::y"] = golden["y"].clone()
    drift = _eval._verify_sampled_outputs_offline(
        sampled, golden, _tol_cls())
    assert drift["checked"] == 3
    assert drift["mismatch"] is False


def test_verify_sampled_outputs_offline_drift_detected():
    """A sampled timing output that does NOT match the offline golden ⇒
    mismatch — the offline-golden cheat-path closure (R4-1)."""
    golden = {"y": torch.arange(8, dtype=torch.float32)}
    sampled = {}
    for t in (0, 1):
        # Trial t=1 shipped a wrong (fast-but-wrong) output.
        bad = golden["y"].clone() + (100.0 if t == 1 else 0.0)
        sampled[f"{t}::y"] = bad
    drift = _eval._verify_sampled_outputs_offline(
        sampled, golden, _tol_cls())
    assert drift["checked"] == 2
    assert drift["mismatch"] is True


# --- end-to-end: an honest upstream verified task is clean. ---------------

def test_evaluate_upstream_verified_honest_no_drift(tmp_path, static_pass):
    """An honest candidate (x*2 every call) graded against a complete
    offline golden in verified mode: evaluated, correct, and the sampled
    timing outputs reverify clean — no measured_outputs_drift."""
    task, sol = _make_upstream_task(tmp_path, solution_src=_HONEST_SOLUTION)
    result = evaluate("SYN_r4_upstream", sol, task_def=task, mode="verified")

    from ascendbench.perf import scoring
    assert result.task_status == scoring.TASK_STATUS_EVALUATED
    assert result.shape_results, "expected the shape to be graded"
    assert all(r.correctness_passed for r in result.shape_results), [
        (r.shape_id, r.correctness_reason) for r in result.shape_results
    ]
    assert "measured_outputs_drift" not in result.performance_flags
    # The sampled-output check actually ran against the offline golden.
    assert any(r.sampled_outputs_checked > 0 for r in result.shape_results), (
        "the offline-golden sampled-output cheat-path closure must run"
    )


def test_evaluate_upstream_verified_cheat_detected(tmp_path, static_pass):
    """The R4-1 regression guard: a candidate honest in the correctness
    pass but fast-but-wrong in the timing pass is CAUGHT — the sampled
    timing trials (forced onto the seed-0 golden input) mismatch the
    offline golden ⇒ measured_outputs_drift + performance_status invalid.

    Pre-fix the offline-golden path skipped the sampled-output re-verify
    entirely, so this cheat scored clean."""
    task, sol = _make_upstream_task(tmp_path, solution_src=_CHEAT_SOLUTION)
    result = evaluate("SYN_r4_upstream", sol, task_def=task, mode="verified")

    from ascendbench.perf import scoring
    assert result.task_status == scoring.TASK_STATUS_EVALUATED
    # The sampled timing outputs (x*3 on the seed-0 input) drift from the
    # offline golden (x*2) — the cheat is detected.
    assert "measured_outputs_drift" in result.performance_flags, (
        f"cheat not caught; flags={result.performance_flags}"
    )
    assert result.performance_status == "suspicious"


# ===========================================================================
# R4-2 — shared "golden usable" predicate (path exists AND loads).
# ===========================================================================

def test_golden_artifact_usable_true_on_real_safetensors(tmp_path):
    p = tmp_path / "g.safetensors"
    save_file({"y": torch.zeros(4)}, str(p))
    assert golden_artifact_usable(p) is True


def test_golden_artifact_usable_false_on_missing(tmp_path):
    assert golden_artifact_usable(tmp_path / "nope.safetensors") is False


def test_golden_artifact_usable_false_on_corrupt(tmp_path):
    """A file that exists but is not a loadable safetensors is NOT usable
    — fail-closed (R4-2)."""
    p = tmp_path / "bad.safetensors"
    p.write_bytes(b"this is not a safetensors file")
    assert golden_artifact_usable(p) is False


def test_resolve_golden_and_golden_complete_agree_on_corrupt(tmp_path):
    """R4-2: a corrupt golden artifact makes BOTH `eval._resolve_golden`
    block the task AND `eligibility.golden_complete` report it incomplete
    — the two authorities cannot disagree."""
    task, _ = _make_upstream_task(
        tmp_path, solution_src=_HONEST_SOLUTION, corrupt_golden=True)

    # eval side: a corrupt artifact ⇒ blocked.
    status, kind, offline = _eval._resolve_golden(task, "verified", ["P1"])
    assert status == "blocked"
    assert kind == "none"
    assert offline is None

    # eligibility side: the SAME corrupt artifact ⇒ not complete.
    assert golden_complete(task) is False


def test_resolve_golden_and_golden_complete_agree_on_complete(tmp_path):
    """R4-2: a complete + loadable golden set ⇒ BOTH `_resolve_golden`
    evaluates and `golden_complete` is True."""
    task, _ = _make_upstream_task(
        tmp_path, solution_src=_HONEST_SOLUTION, corrupt_golden=False)

    from ascendbench.perf import scoring
    status, kind, offline = _eval._resolve_golden(task, "verified", ["P1"])
    assert status == scoring.TASK_STATUS_EVALUATED
    assert kind == "offline"
    assert set(offline) == {"P1"}
    assert golden_complete(task) is True


def test_evaluate_corrupt_golden_is_blocked(tmp_path, static_pass):
    """End-to-end: a corrupt offline golden ⇒ evaluate() returns
    task_status='blocked' (fail-closed), perf fields null."""
    task, sol = _make_upstream_task(
        tmp_path, solution_src=_HONEST_SOLUTION, corrupt_golden=True)
    result = evaluate("SYN_r4_upstream", sol, task_def=task, mode="verified")
    assert result.task_status == "blocked"
    assert result.s_t is None
    assert result.shape_results == []
    assert result.error_reason is not None


# ===========================================================================
# R4-3 — upstream_oracle_required tasks strictly require a complete golden/.
# ===========================================================================

_DEF_KERNEL = "@triton.jit\ndef the_kernel(i):\n    return {}\n"
_DEF_ORACLE = "def run(i):\n    return {}\n"


def _write_validatable_task(td: Path, *, oracle_simplified: str) -> None:
    """Materialise a minimal schema-valid task dir for validate_tasks."""
    (td / "kernel.py").write_text(_DEF_KERNEL)
    sha = hashlib.sha256(_DEF_KERNEL.encode()).hexdigest()
    (td / "oracle_pytorch.py").write_text(_DEF_ORACLE)
    knobs = {"B": 2, "N": 4}
    (td / "shapes.yaml").write_text(
        yaml.safe_dump({
            "size_knobs": knobs,
            "public_shapes": [{"id": "P1", "name": "tiny", "knobs": knobs}],
            "hidden_shapes": [{"id": "H1", "name": "hid", "knobs": knobs}],
        })
    )
    (td / "contract.md").write_text("# Contract — L1_01 demo\n")
    meta = {
        "schema_version": "1.0", "task_id": "L1_01_demo", "level": "L1",
        "idx": 1, "name": "k", "repo": "vLLM", "dtype_class": "VV",
        "category": "norm", "npu_difficulty": "low",
        "upstream": {"file_sha256": sha},
        "interface": {}, "oracle": {"gpu_triton": vt.GPU_TRITON_PLACEHOLDER},
        "tolerance_class": "_t",
        "oracle_simplified": oracle_simplified, "runner_checks": {},
        "is_canary": False, "is_hidden_only": False,
        "cost_model": {
            "flops": "B*N", "flops_class": "vector",
            "bytes": "B*N", "bytes_resource": "hbm",
            # this minimal fixture declares no tensor inputs, so B/N have no input home; declare them
            # row-only so the U7 cost-symbol-universe check (validate_tasks) sees them as legal, not homeless.
            "row_only_params": ["B", "N"],
        },
        # WA.1/WA.4: `param_constraints` is a required top-level meta key.
        # Use a reconciled (`scaffold: false`) schema-closed block — WA.4's
        # semantic gate rejects `scaffold: true`, and R4-1 requires all five
        # semantic subsections (empty containers here); this fixture is not
        # testing param_constraints (its checks live in the WA.1/WA.4 suites).
        "param_constraints": {"scaffold": False, "dims": [], "dtypes": {},
                              "dists": {}, "scalars": {},
                              "tensor_invariants": []},
    }
    (td / "meta.yaml").write_text(yaml.safe_dump(meta))


def test_validate_golden_required_for_upstream_no_dir(tmp_path):
    """R4-3: an upstream_oracle_required task with NO golden/ dir is a
    validation error — the static 'missing provider golden' gate. Pre-fix
    this validated clean (the gate was absent)."""
    _write_validatable_task(
        tmp_path, oracle_simplified="upstream_oracle_required")
    errs = vt.validate_task(tmp_path, {"_t"}, strict=False)
    assert any("no golden/ directory" in e for e in errs), errs


def test_validate_golden_required_for_upstream_incomplete_dir(tmp_path):
    """An upstream task whose golden/ does not cover every eval shape is a
    validation error."""
    _write_validatable_task(
        tmp_path, oracle_simplified="upstream_oracle_required")
    golden = tmp_path / "golden"
    golden.mkdir()
    (golden / "P1.safetensors").write_bytes(b"\x00")  # H1 missing
    errs = vt.validate_task(tmp_path, {"_t"}, strict=False)
    assert any("golden" in e and "H1" in e for e in errs), errs


def test_validate_golden_complete_dir_passes(tmp_path):
    """An upstream task with a complete golden/ set of LOADABLE safetensors
    passes the golden check (only the golden axis is asserted here).

    R7-2: the validator now requires each artifact to be a loadable
    safetensors (the shared `golden_artifact_usable` predicate), so the
    fixture writes real safetensors, not placeholder bytes."""
    _write_validatable_task(
        tmp_path, oracle_simplified="upstream_oracle_required")
    golden = tmp_path / "golden"
    golden.mkdir()
    save_file({"y": torch.zeros(4)}, str(golden / "P1.safetensors"))
    save_file({"y": torch.zeros(4)}, str(golden / "H1.safetensors"))
    errs = vt.validate_task(tmp_path, {"_t"}, strict=False)
    assert not any("golden" in e for e in errs), errs


def test_validate_golden_not_required_for_non_upstream(tmp_path):
    """A non-upstream_oracle_required task needs no golden/ — the golden
    requirement applies ONLY to the upstream tier."""
    _write_validatable_task(
        tmp_path, oracle_simplified="simple_pytorch_ok")
    errs = vt.validate_task(tmp_path, {"_t"}, strict=False)
    assert not any("golden" in e for e in errs), errs
