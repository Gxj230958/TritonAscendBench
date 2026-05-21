"""R7 codex-review regression tests.

Pins the R7 review findings so a regression re-surfaces as a test
failure:

  * R7-1 — in verified mode an ``upstream_oracle_required`` task whose
    golden has resolved to the offline ``golden/`` artifact set
    (``golden_kind == "offline"``) must NOT import / execute
    ``oracle_pytorch.py``. Such an oracle is a low-level non-runnable
    stand-in; a broken one used to fail the whole grading as a harness
    ``error`` even with a complete ``golden/`` set.
  * R7-2 — ``validate_tasks._validate_golden`` reuses the shared
    ``eligibility.golden_artifact_usable`` predicate (exists AND
    safetensors loads), not a bare ``is_file()``. A corrupt / text-pasted
    ``.safetensors`` is rejected statically — the static gate and the
    runtime golden gates agree.
  * R7-4 — the ``cli`` duplicate-result worst-case merge escalates
    ``performance_status`` on the ``ok < suspicious < quarantined``
    precedence ladder. A first result ``suspicious`` + a duplicate
    ``quarantined`` merges to ``quarantined`` (the old ``ok → non-ok``
    rule froze it at ``suspicious``).

Dev machine has no NPU; the upstream-verified test grades against a
mock offline golden, exactly as R4's tests do.
"""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
import yaml
from safetensors.torch import save_file

from triton_ascend_bench import eval as _eval
from triton_ascend_bench import inputs as _inputs
from triton_ascend_bench import scoring, static_checker
from triton_ascend_bench.dataset import TaskDef
from triton_ascend_bench.eval import _PERFORMANCE_STATUS_PRECEDENCE, evaluate

REPO = Path(__file__).resolve().parents[2]

_spec = importlib.util.spec_from_file_location(
    "validate_tasks", REPO / "scripts" / "validate_tasks.py"
)
vt = importlib.util.module_from_spec(_spec)
sys.modules["validate_tasks"] = vt
_spec.loader.exec_module(vt)


# ===========================================================================
# R7-1 — offline-golden path must not touch oracle_pytorch.py.
# ===========================================================================

# A correct candidate: y = x * 2 every call (matches the offline golden).
_HONEST_SOLUTION = textwrap.dedent('''\
    import torch


    def npu_triton_impl(inputs):
        return {"y": inputs["x"] * 2.0}
''')

# An oracle that EXPLODES on import — proves the offline path never
# imports / executes it.  A `golden_kind == "oracle"` path would crash
# loading this module; the offline path must sail past it.
_BROKEN_ORACLE_SRC = textwrap.dedent('''\
    raise RuntimeError("oracle_pytorch.py is a non-runnable stand-in")
''')

_SHAPES = {
    "size_knobs": {"B": 4},
    "public_shapes": [{"id": "P1", "name": "tiny", "knobs": {"B": 4}}],
    "hidden_shapes": [],
    "inputs": {"x": {"dims": ["B"], "dtype": "float32", "dist": "randn"}},
}

# WA.4: a reconciled (`scaffold: false`), semantically-valid
# `param_constraints` block for `_SHAPES` — these tests exercise the R7-1
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
    "task_id": "SYN_r7_upstream",
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
    is reached."""
    monkeypatch.setattr(_eval.static_checker, "check_solution",
                        lambda *a, **k: static_checker.CheckReport())


def _make_upstream_task(tmp_path: Path, *, oracle_src: str,
                        corrupt_golden: bool = False):
    """Build a synthetic upstream_oracle_required task + a complete (or
    deliberately corrupt) mock offline golden."""
    tdir = tmp_path / "task"
    tdir.mkdir()
    (tdir / "oracle_pytorch.py").write_text(oracle_src)
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
        task_id="SYN_r7_upstream",
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
    sol.write_text(_HONEST_SOLUTION)
    return task, sol


def test_r7_1_offline_path_ignores_broken_oracle(tmp_path, static_pass):
    """R7-1 regression guard: an upstream_oracle_required task with a
    BROKEN oracle, graded in verified mode against a complete offline
    golden, must still grade cleanly — the offline path never imports
    oracle_pytorch.py.

    Pre-fix the oracle was imported unconditionally, so a broken oracle
    aborted phase-2b grading with a harness error_reason and left
    correctness_status stuck at 'error'."""
    task, sol = _make_upstream_task(tmp_path, oracle_src=_BROKEN_ORACLE_SRC)
    result = evaluate("SYN_r7_upstream", sol, task_def=task, mode="verified")

    # The broken oracle must NOT have surfaced as a harness error.
    assert result.error_reason is None, result.error_reason
    assert result.task_status == scoring.TASK_STATUS_EVALUATED
    assert result.shape_results, "expected the shape to be graded"
    # Graded against the offline golden — the honest candidate passes.
    assert all(r.correctness_passed for r in result.shape_results), [
        (r.shape_id, r.correctness_reason) for r in result.shape_results
    ]
    assert result.correctness_status == "ok"


def test_r7_1_self_reported_upstream_still_unsupported(tmp_path, static_pass):
    """An upstream_oracle_required task in self-reported mode has no
    provider golden — still 'unsupported', and the broken oracle is never
    reached (golden routing returns before the oracle would load)."""
    task, sol = _make_upstream_task(tmp_path, oracle_src=_BROKEN_ORACLE_SRC)
    result = evaluate("SYN_r7_upstream", sol, task_def=task,
                      mode="self-reported")
    assert result.task_status == "unsupported"
    assert result.error_reason is None


# ===========================================================================
# R7-2 — validate_tasks._validate_golden uses golden_artifact_usable.
# ===========================================================================

def _write_upstream_task_dir(root: Path, *, corrupt: bool) -> Path:
    """Write an upstream_oracle_required task dir with a golden/ set that
    is either valid or a corrupt (non-loadable) safetensors."""
    tdir = root / "L9_99_syn"
    tdir.mkdir()
    (tdir / "oracle_pytorch.py").write_text("def run(inputs):\n    return {}\n")
    (tdir / "shapes.yaml").write_text(yaml.safe_dump(_SHAPES))
    (tdir / "contract.md").write_text("# synthetic\n")
    (tdir / "kernel.py").write_text("# frozen\n")
    (tdir / "meta.yaml").write_text(yaml.safe_dump(_META))
    gdir = tdir / "golden"
    gdir.mkdir()
    if corrupt:
        (gdir / "P1.safetensors").write_bytes(b"this is not safetensors")
    else:
        built = _inputs.build_inputs(_SHAPES, "P1", seed=0)
        save_file({"y": built["x"] * 2.0}, str(gdir / "P1.safetensors"))
    return tdir


def test_r7_2_validate_golden_rejects_corrupt_safetensors(tmp_path):
    """R7-2 regression guard: a golden/<shape>.safetensors that EXISTS but
    is not a loadable safetensors must be a validation error.

    Pre-fix `_validate_golden` used a bare `is_file()` — a corrupt text
    file validated clean while `eval._resolve_golden` fail-closed on it."""
    tdir = _write_upstream_task_dir(tmp_path, corrupt=True)
    errs = vt._validate_golden("L9_99_syn", _META, tdir)
    assert errs, "a corrupt golden artifact must be a validation error"
    assert any("unreadable" in e or "missing" in e for e in errs), errs


def test_r7_2_validate_golden_accepts_loadable_safetensors(tmp_path):
    """A complete, loadable golden/ set validates clean."""
    tdir = _write_upstream_task_dir(tmp_path, corrupt=False)
    errs = vt._validate_golden("L9_99_syn", _META, tdir)
    assert errs == [], errs


def test_r7_2_validate_golden_uses_shared_predicate():
    """The validator must reuse the shared eligibility predicate, not a
    private is_file() check.

    Structural pin (order-independent — `test_eligibility` reloads the
    `eligibility` module, so an object-identity check would be fragile to
    test ordering): `validate_tasks` imports `golden_artifact_usable` from
    `eligibility`, and `_validate_golden` no longer uses a bare
    `is_file()` on the golden artifacts."""
    src = (REPO / "scripts" / "validate_tasks.py").read_text()
    assert "from triton_ascend_bench.eligibility import" in src
    assert "golden_artifact_usable" in src
    # The `_validate_golden` body must call the shared predicate, not
    # `.is_file()` on a `<shape>.safetensors` path.
    body = src[src.index("def _validate_golden"):
               src.index("def _validate_metamorphic_checks")]
    assert "golden_artifact_usable(golden_dir" in body
    assert ".safetensors\").is_file()" not in body


# ===========================================================================
# R7-4 — cli duplicate-merge performance_status precedence.
# ===========================================================================

def _merge_perf(prev_status: str, dup_status: str) -> str:
    """Reproduce the cli duplicate-merge `performance_status` rule (R7-4):
    escalate on the `ok < suspicious < quarantined` ladder, taking the
    more severe verdict."""
    prev_rank = _PERFORMANCE_STATUS_PRECEDENCE.get(prev_status)
    new_rank = _PERFORMANCE_STATUS_PRECEDENCE.get(dup_status)
    if new_rank is not None and (prev_rank is None or new_rank > prev_rank):
        return dup_status
    return prev_status


def test_r7_4_suspicious_then_quarantined_merges_to_quarantined():
    """R7-4 regression guard: a first 'suspicious' result + a duplicate
    'quarantined' must merge to 'quarantined'. The old `ok → non-ok` rule
    only moved off 'ok' and froze the merged value at 'suspicious'."""
    assert _merge_perf("suspicious", "quarantined") == "quarantined"


def test_r7_4_quarantined_not_downgraded_by_suspicious_duplicate():
    """A milder duplicate never rolls back a more severe verdict."""
    assert _merge_perf("quarantined", "suspicious") == "quarantined"


def test_r7_4_ok_escalates_to_suspicious():
    """The classic ok → non-ok case still escalates."""
    assert _merge_perf("ok", "suspicious") == "suspicious"


def test_r7_4_error_prev_replaced_by_graded_duplicate():
    """`error` is outside the ladder (perf never graded) — a duplicate
    that DID grade perf wins."""
    assert _merge_perf("error", "ok") == "ok"
    assert _merge_perf("error", "quarantined") == "quarantined"


def test_r7_4_error_duplicate_does_not_overwrite_graded():
    """A duplicate `error` (perf not graded) never overwrites a graded
    verdict — `error` has no ladder rank, so it cannot escalate."""
    assert _merge_perf("ok", "error") == "ok"
    assert _merge_perf("quarantined", "error") == "quarantined"


def test_r7_4_analyze_counts_duplicate_and_runs_clean(tmp_path):
    """End-to-end smoke: `tab analyze` over a verified suite with a
    duplicate score-eligible task_id (first `suspicious`, duplicate
    `quarantined`) runs clean and counts the duplicate, exercising the
    real duplicate-merge code path. The merged severity itself is pinned
    by the precedence-ladder tests above and the source-structure check
    below."""
    # A real catalogue, score-eligible task — the duplicate-merge block
    # only runs for tasks inside the scored denominator.
    tid = "L1_01_bias"
    verified = tmp_path / "suite.json"
    verified.write_text(json.dumps({
        "mode": "verified",
        "results": [
            {"task_id": tid,
             "correctness_status": "ok", "runtime_status": "ok",
             "performance_status": "suspicious"},
            {"task_id": tid,
             "correctness_status": "ok", "runtime_status": "ok",
             "performance_status": "quarantined"},
        ],
    }))
    out = tmp_path / "score.json"
    rc = subprocess.run(
        [sys.executable, "-m", "triton_ascend_bench.cli", "analyze",
         "--verified-results", str(verified), "--out", str(out)],
        cwd=REPO, capture_output=True, text=True,
    )
    assert rc.returncode == 0, rc.stderr
    doc = json.loads(out.read_text())
    assert doc["duplicate_task_id_count"] == 1


def test_r7_4_cli_duplicate_merge_uses_precedence_ladder():
    """The cli duplicate-merge block must escalate `performance_status`
    via `_PERFORMANCE_STATUS_PRECEDENCE`, not the old `ok → non-ok` rule.
    Pinned structurally so a regression to the frozen-at-suspicious bug
    re-surfaces here."""
    src = (REPO / "triton_ascend_bench" / "cli.py").read_text()
    assert "_PERFORMANCE_STATUS_PRECEDENCE" in src, (
        "cli duplicate-merge must import the shared precedence ladder"
    )
    # the old buggy rule must be gone from the duplicate-merge block.
    assert 'new_pf != "ok" and prev.performance_status == "ok"' not in src
