"""W2.5 — canary task framework tests.

Covers the four W2.5 deliverables:

1. Discovery enumerator — `dataset.iter_canary_tasks()` finds the sample
   `C1_identity_trap` canary; `iter_all_tasks()` does NOT (it skips
   `_`-prefixed dirs), and the canary is never score-eligible.
2. `source_task_id` binding — the canary binds to a real level1/2/3 task and
   `as_task_def()` produces a TaskDef the grading machinery accepts.
3. `check_canaries` trap detection — `run_canary_suite` resolves the bound
   solution and judges it: an honest observation does NOT trigger, an
   identity-cheat observation (passes the identity trap) DOES trigger.
4. `validate_tasks` `_canary` entry point — a well-formed canary validates;
   a malformed one (missing files, bad `is_canary`, missing / invalid
   `source_task_id`) is reported as an error.

No test is skipped. The trap's oracle-vs-candidate compare needs a real
NPU/triton run (Phase 4); on the dev machine the `run_canary_suite` grader is
injected with a deterministic stub so the binding + judgement wiring is
exercised end-to-end without a GPU (plan §8).
"""
from __future__ import annotations

import importlib.util
import shutil
import sys
from pathlib import Path

import yaml

from ascendbench.integrity.adversarial import (
    aggregate_canary_findings,
    check_canaries,
    run_canary_suite,
)
from ascendbench.corpus.dataset import (
    CanaryTaskDef,
    iter_all_tasks,
    iter_canary_tasks,
    load_canary_task,
)
from ascendbench.release_gate.eligibility import compute_score_eligible

REPO = Path(__file__).resolve().parent.parent.parent
CANARY_ID = "C1_identity_trap"

# load validate_tasks.py as a module (it is a script, not a package member).
_spec = importlib.util.spec_from_file_location(
    "validate_tasks", REPO / "scripts" / "corpus" / "validate_tasks.py"
)
vt = importlib.util.module_from_spec(_spec)
sys.modules["validate_tasks"] = vt
_spec.loader.exec_module(vt)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Discovery enumerator
# ─────────────────────────────────────────────────────────────────────────────
def test_iter_canary_tasks_discovers_sample_canary():
    """The dedicated enumerator finds the sample canary."""
    canaries = iter_canary_tasks()
    ids = {c.canary_id for c in canaries}
    assert CANARY_ID in ids


def test_canary_not_in_iter_all_tasks():
    """`iter_all_tasks()` skips `_`-prefixed dirs — no canary leaks in."""
    all_ids = {t.task_id for t in iter_all_tasks()}
    assert CANARY_ID not in all_ids


def test_canary_not_score_eligible():
    """A canary is never in the score-eligible denominator (DESIGN §6.8)."""
    eligible = compute_score_eligible(iter_all_tasks())
    assert CANARY_ID not in eligible
    # And the dedicated enumerator's canaries share no id with the scored set.
    canary_ids = {c.canary_id for c in iter_canary_tasks()}
    assert canary_ids.isdisjoint(eligible)


# ─────────────────────────────────────────────────────────────────────────────
# 2. source_task_id binding
# ─────────────────────────────────────────────────────────────────────────────
def test_canary_binds_to_existing_real_task():
    """`source_task_id` names a real level1/2/3 task that actually exists."""
    canary = load_canary_task(CANARY_ID)
    assert isinstance(canary, CanaryTaskDef)
    assert canary.canary_trap_type == "identity_hack"
    real_ids = {t.task_id for t in iter_all_tasks()}
    assert canary.source_task_id in real_ids


def test_canary_as_task_def_carries_canary_dir():
    """`as_task_def()` adapts the canary into a TaskDef rooted at the canary
    directory — the grading machinery uses the canary's own oracle/shapes."""
    canary = load_canary_task(CANARY_ID)
    td = canary.as_task_def()
    assert td.task_id == CANARY_ID
    assert td.task_dir == canary.task_dir
    assert (td.task_dir / "oracle_pytorch.py").is_file()
    # a canary has no kernel.py / own solution.
    assert td.kernel_py == ""


def test_canary_expected_honest_pass_is_false_for_identity_trap():
    """C1's oracle is an identity, so an honest bias kernel must NOT pass it."""
    canary = load_canary_task(CANARY_ID)
    assert canary.expected_honest_pass is False


# ─────────────────────────────────────────────────────────────────────────────
# 3. check_canaries trap detection via run_canary_suite
# ─────────────────────────────────────────────────────────────────────────────
def _make_solutions_root(tmp_path: Path, source_task_id: str) -> Path:
    """A solutions root with a `solution.py` for the canary's bound task."""
    root = tmp_path / "solutions"
    sd = root / source_task_id
    sd.mkdir(parents=True)
    (sd / "solution.py").write_text("# placeholder bound solution\n")
    return root


def test_run_canary_suite_honest_solution_not_triggered(tmp_path):
    """An honest observation (does NOT pass the identity trap) ⇒ no catch."""
    canary = load_canary_task(CANARY_ID)
    root = _make_solutions_root(tmp_path, canary.source_task_id)

    seen: dict[str, Path] = {}

    def honest_grader(c, sol_path: Path) -> bool:
        # Record that the runner resolved the bound solution path.
        seen[c.canary_id] = sol_path
        return False  # honest kernel applies the bias ⇒ ≠ identity oracle.

    findings = run_canary_suite(root, grader=honest_grader)
    assert seen[CANARY_ID] == root / canary.source_task_id / "solution.py"
    by_id = {f.canary_id: f for f in findings}
    assert by_id[CANARY_ID].triggered is False
    agg = aggregate_canary_findings(findings)
    assert agg.canary_caught == 0
    assert agg.suspicious is False


def test_run_canary_suite_identity_cheat_triggers(tmp_path):
    """An identity-cheat observation (PASSES the identity trap) ⇒ caught.

    The runner resolves the canary's bound `source_task_id` to the
    submission's solution, grades it, and `check_canaries` flags
    `got != expected_honest_pass` — escalating the submission to suspicious.
    """
    canary = load_canary_task(CANARY_ID)
    root = _make_solutions_root(tmp_path, canary.source_task_id)

    def cheat_grader(c, sol_path: Path) -> bool:
        # An identity hack returns its input unchanged ⇒ matches the
        # identity oracle ⇒ "passes" the canary.
        return True

    findings = run_canary_suite(root, grader=cheat_grader)
    by_id = {f.canary_id: f for f in findings}
    assert by_id[CANARY_ID].triggered is True
    agg = aggregate_canary_findings(findings)
    assert agg.canary_caught == 1
    assert agg.suspicious is True


def test_run_canary_suite_missing_bound_solution_skipped(tmp_path):
    """A canary whose bound solution is absent yields no finding."""
    root = tmp_path / "empty_solutions"
    root.mkdir()
    findings = run_canary_suite(root, grader=lambda c, p: True)
    assert findings == []


def test_run_canary_suite_default_grader_path_runs(tmp_path):
    """The default grader path (`eval.evaluate`) is wired and returns findings.

    The dev machine has no NPU/triton, so the candidate cannot truly execute —
    but the binding (`source_task_id` → bound solution → `evaluate`) must still
    resolve and produce a finding list without raising (plan §8 wiring test).
    """
    canary = load_canary_task(CANARY_ID)
    root = _make_solutions_root(tmp_path, canary.source_task_id)
    findings = run_canary_suite(root)  # default grader = eval.evaluate
    # one canary, bound solution present ⇒ exactly one finding.
    assert {f.canary_id for f in findings} == {CANARY_ID}


def test_run_canary_suite_default_grader_threads_backend(tmp_path, monkeypatch):
    """`run_canary_suite(backend=...)` routes the trap run to that backend.

    Regression: `tab run-suite --backend gpu` must grade canaries on the same
    device as the scored tasks, not silently fall back to npu.
    """
    from types import SimpleNamespace

    import ascendbench.engine.eval as _eval
    canary = load_canary_task(CANARY_ID)
    root = _make_solutions_root(tmp_path, canary.source_task_id)
    seen: dict[str, str] = {}

    def fake_evaluate(task_id, solution_path, *, task_def=None,
                      backend="npu", **kw):
        seen["backend"] = backend
        return SimpleNamespace(correctness_status="error")

    monkeypatch.setattr(_eval, "evaluate", fake_evaluate)
    run_canary_suite(root, backend="gpu")
    assert seen["backend"] == "gpu"


def test_check_canaries_flags_identity_cheat_directly():
    """`check_canaries` itself: an observed pass on C1 (whose honest expected
    is False) is a triggered finding."""
    canary = load_canary_task(CANARY_ID)
    findings = check_canaries(
        submitted_results={CANARY_ID: True},               # cheat passed it
        canary_truth={CANARY_ID: canary.expected_honest_pass},  # honest = False
    )
    assert len(findings) == 1
    assert findings[0].triggered is True


# ─────────────────────────────────────────────────────────────────────────────
# 4. validate_tasks _canary entry point
# ─────────────────────────────────────────────────────────────────────────────
def test_validate_canary_sample_is_clean():
    """The shipped sample canary validates with zero errors."""
    policy = vt._load_policy_classes()
    real_ids = vt._real_task_ids()
    errs = vt.validate_canary_task(
        vt.CANARY_ROOT / CANARY_ID, policy, real_ids)
    assert errs == [], errs


def test_run_canary_validation_passes():
    """`run_canary_validation` over the whole `_canary/` dir is clean."""
    assert vt.run_canary_validation(strict=False) == []


def _copy_canary(tmp_path: Path) -> Path:
    """A writable copy of the sample canary for malformed-mutation tests."""
    dst = tmp_path / CANARY_ID
    shutil.copytree(vt.CANARY_ROOT / CANARY_ID, dst)
    return dst


def _validate(tmp_path: Path, canary_dir: Path) -> list[str]:
    policy = vt._load_policy_classes()
    real_ids = vt._real_task_ids()
    return vt.validate_canary_task(canary_dir, policy, real_ids)


def test_validate_canary_missing_source_task_id_errors(tmp_path):
    """A canary with no `source_task_id` is rejected."""
    cdir = _copy_canary(tmp_path)
    meta = yaml.safe_load((cdir / "meta.yaml").read_text())
    del meta["source_task_id"]
    (cdir / "meta.yaml").write_text(yaml.safe_dump(meta))
    errs = _validate(tmp_path, cdir)
    assert any("source_task_id" in e for e in errs), errs


def test_validate_canary_invalid_source_task_id_errors(tmp_path):
    """A `source_task_id` pointing at a non-existent task is rejected."""
    cdir = _copy_canary(tmp_path)
    meta = yaml.safe_load((cdir / "meta.yaml").read_text())
    meta["source_task_id"] = "L9_99_does_not_exist"
    (cdir / "meta.yaml").write_text(yaml.safe_dump(meta))
    errs = _validate(tmp_path, cdir)
    assert any("does not name an existing" in e for e in errs), errs


def test_validate_canary_bad_is_canary_errors(tmp_path):
    """`is_canary` must be true for a `_canary/` task."""
    cdir = _copy_canary(tmp_path)
    meta = yaml.safe_load((cdir / "meta.yaml").read_text())
    meta["is_canary"] = False
    (cdir / "meta.yaml").write_text(yaml.safe_dump(meta))
    errs = _validate(tmp_path, cdir)
    assert any("is_canary" in e for e in errs), errs


def test_validate_canary_unknown_trap_type_errors(tmp_path):
    """`canary_trap_type` must be one of the 10 trap classes."""
    cdir = _copy_canary(tmp_path)
    meta = yaml.safe_load((cdir / "meta.yaml").read_text())
    meta["canary_trap_type"] = "totally_made_up_trap"
    (cdir / "meta.yaml").write_text(yaml.safe_dump(meta))
    errs = _validate(tmp_path, cdir)
    assert any("canary_trap_type" in e for e in errs), errs


def test_validate_canary_missing_file_errors(tmp_path):
    """A canary missing a required file is rejected."""
    cdir = _copy_canary(tmp_path)
    (cdir / "oracle_pytorch.py").unlink()
    errs = _validate(tmp_path, cdir)
    assert any("missing files" in e for e in errs), errs


def test_validate_canary_bad_expected_honest_pass_errors(tmp_path):
    """`expected_honest_pass` must be a bool."""
    cdir = _copy_canary(tmp_path)
    meta = yaml.safe_load((cdir / "meta.yaml").read_text())
    meta["expected_honest_pass"] = "yes"
    (cdir / "meta.yaml").write_text(yaml.safe_dump(meta))
    errs = _validate(tmp_path, cdir)
    assert any("expected_honest_pass" in e for e in errs), errs


# ─────────────────────────────────────────────────────────────────────────────
# main scan also validates _canary (no --task filter)
# ─────────────────────────────────────────────────────────────────────────────
def test_validate_tasks_canary_only_mode(monkeypatch):
    """`validate_tasks.py --canary` exits 0 on the clean sample canary."""
    monkeypatch.setattr(sys, "argv", ["validate_tasks.py", "--canary"])
    assert vt.main() == 0
