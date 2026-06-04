"""validate_tasks.py --release-gate + phase-aware hash tests.

Covers: all-PASS ⇒ exit 0; one eligible BLOCKED ⇒ nonzero + named;
non-eligible BLOCKED ⇒ not blocking; missing status (UNRUN) ⇒ blocks;
non-vacuity guard (empty / mismatched count ⇒ HARD FAIL, never
vacuous-pass); mixed rollout passes normal schema/hash mode while
release-gate independently blocks.

Release predicate: a task is releasable iff `phase_a_prime.status == PASS`
(S2 severance) over the C2-aligned eligible set. `cost_model.status` is
NOT a release blocker -- it is ncu-evidenced performance metadata. The
`_pass()` helper therefore sets only `phase_a_prime.status`.
"""
from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass, field
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parent.parent.parent
_spec = importlib.util.spec_from_file_location(
    "validate_tasks", REPO / "scripts" / "validate_tasks.py"
)
vt = importlib.util.module_from_spec(_spec)
sys.modules["validate_tasks"] = vt
_spec.loader.exec_module(vt)


@dataclass
class _T:
    task_id: str
    meta: dict = field(default_factory=dict)
    task_dir: Path = Path("/nonexistent")


def _pass(tid):
    """A releasable task: `phase_a_prime.status` is PASS (S2 severance).
    `cost_model.status` is NOT gated, so it is intentionally omitted."""
    return _T(tid, {"phase_a_prime": {"status": "PASS"}})


def test_all_eligible_pass_exit_zero():
    tasks = [_pass("a"), _pass("b")]
    rc = vt.run_release_gate(tasks=tasks, eligible_ids={"a", "b"})
    assert rc == 0


def test_one_eligible_blocked_nonzero_and_named(capsys):
    tasks = [
        _pass("a"),
        _T("b", {"phase_a_prime": {"status": "BLOCKED"}}),
    ]
    rc = vt.run_release_gate(tasks=tasks, eligible_ids={"a", "b"})
    assert rc == 1
    err = capsys.readouterr().err
    assert "b" in err and "NOT releasable" in err


def test_non_eligible_blocked_does_not_block():
    """A BLOCKED task NOT in the eligible set must not trigger the
    release gate (only the eligible set has teeth)."""
    tasks = [
        _pass("a"),
        _T("stub", {"phase_a_prime": {"status": "BLOCKED"}}),
    ]
    rc = vt.run_release_gate(tasks=tasks, eligible_ids={"a"})
    assert rc == 0


def test_missing_status_unrun_blocks():
    """status missing / UNRUN ⇒ treated as != PASS ⇒ blocks."""
    tasks = [_pass("a"), _T("b", {})]
    rc = vt.run_release_gate(tasks=tasks, eligible_ids={"a", "b"})
    assert rc == 1


def test_vacuous_empty_is_hard_fail(capsys):
    """Empty eligible set ⇒ HARD FAIL (never vacuous-pass)."""
    rc = vt.run_release_gate(tasks=[], eligible_ids=set())
    assert rc == 2
    assert "HARD FAIL" in capsys.readouterr().err


def test_examined_count_mismatch_hard_fail(capsys):
    """eligible_ids references a task not present in `tasks` (bad
    --task / wrong root) ⇒ examined != expected ⇒ HARD FAIL."""
    tasks = [_pass("a")]
    rc = vt.run_release_gate(tasks=tasks, eligible_ids={"a", "missing"})
    assert rc == 2
    assert "non-vacuity guard" in capsys.readouterr().err


def test_generator_input_not_vacuous_pass():
    """FIX 2: a one-shot GENERATOR of one eligible task with no
    phase_a_prime must return 1 (NOT a vacuous 0). The function
    iterates `tasks` twice (examined, then blocking); without
    materialisation the 2nd pass on a generator is empty -> a false
    'releasable'. `tasks = list(tasks)` defends this."""

    def _one_task_gen():
        yield _T("only", {})  # eligible, no phase_a_prime -> not PASS

    rc = vt.run_release_gate(tasks=_one_task_gen(), eligible_ids={"only"})
    assert rc == 1, (
        "generator input vacuously passed -- the 2nd iteration was "
        "empty (materialisation missing)"
    )


def test_generator_input_all_pass_still_works():
    """Sanity: a generator of all-PASS tasks still correctly returns 0
    (materialisation does not break the happy path)."""

    def _gen():
        yield _pass("a")
        yield _pass("b")

    rc = vt.run_release_gate(tasks=_gen(), eligible_ids={"a", "b"})
    assert rc == 0


def test_phase_aware_hash_uses_standalone_when_present(tmp_path):
    """A Phase-A'-processed task: standalone_sha256 compared to
    on-disk kernel.py; upstream.file_sha256 may differ (it is original
    provenance, NOT on-disk)."""
    td = tmp_path
    kernel = "def run(i):\n    return {}\n"
    (td / "kernel.py").write_text(kernel)
    import hashlib

    sha = hashlib.sha256(kernel.encode()).hexdigest()
    (td / "oracle_pytorch.py").write_text("def run(i):\n    return {}\n")
    (td / "shapes.yaml").write_text("public_shapes: []\n")
    (td / "contract.md").write_text("c\n")
    meta = {
        "schema_version": "1.0", "task_id": "t", "level": "L1", "idx": 1,
        "name": "k", "repo": "vLLM", "dtype_class": "VV",
        "category": "x", "npu_difficulty": "low",
        "upstream": {"file_sha256": "OLD_UPSTREAM_DIFFERENT"},
        "interface": {}, "oracle": {}, "tolerance_class": "_t",
        "oracle_simplified": "simple_pytorch_ok", "runner_checks": {},
        "is_canary": False, "is_hidden_only": False,
        "phase_a_prime": {"standalone_sha256": sha},
    }
    (td / "meta.yaml").write_text(yaml.safe_dump(meta))
    # Add a @triton.jit so the triton-jit check doesn't fire.
    (td / "kernel.py").write_text("@triton.jit\n" + kernel)
    sha2 = hashlib.sha256(("@triton.jit\n" + kernel).encode()).hexdigest()
    meta["phase_a_prime"]["standalone_sha256"] = sha2
    (td / "meta.yaml").write_text(yaml.safe_dump(meta))
    errs = vt.validate_task(td, {"_t"}, strict=False)
    # No sha-mismatch error (phase-aware path compared standalone_sha256).
    assert not any("sha256 mismatch" in e for e in errs), errs


def test_legacy_task_uses_upstream_hash(tmp_path):
    """An unprocessed/legacy task (no phase_a_prime) -> legacy compare
    of upstream.file_sha256 (mixed rollout still all-green)."""
    td = tmp_path
    kernel = "@triton.jit\ndef run(i):\n    return {}\n"
    (td / "kernel.py").write_text(kernel)
    import hashlib

    sha = hashlib.sha256(kernel.encode()).hexdigest()
    (td / "oracle_pytorch.py").write_text("def run(i):\n    return {}\n")
    (td / "shapes.yaml").write_text("public_shapes: []\n")
    (td / "contract.md").write_text("c\n")
    meta = {
        "schema_version": "1.0", "task_id": "t", "level": "L1", "idx": 1,
        "name": "k", "repo": "vLLM", "dtype_class": "VV",
        "category": "x", "npu_difficulty": "low",
        "upstream": {"file_sha256": sha},
        "interface": {}, "oracle": {}, "tolerance_class": "_t",
        "oracle_simplified": "simple_pytorch_ok", "runner_checks": {},
        "is_canary": False, "is_hidden_only": False,
    }
    (td / "meta.yaml").write_text(yaml.safe_dump(meta))
    errs = vt.validate_task(td, {"_t"}, strict=False)
    assert not any("sha256 mismatch" in e for e in errs), errs


def test_neither_hash_is_validation_error(tmp_path):
    """FIX 5: a task with NEITHER phase_a_prime.standalone_sha256 NOR
    upstream.file_sha256 populated must be a validation ERROR (not a
    silent pass = a task with zero integrity check)."""
    td = tmp_path
    (td / "kernel.py").write_text("@triton.jit\ndef run(i):\n    return {}\n")
    (td / "oracle_pytorch.py").write_text("def run(i):\n    return {}\n")
    (td / "shapes.yaml").write_text("public_shapes: []\n")
    (td / "contract.md").write_text("c\n")
    meta = {
        "schema_version": "1.0", "task_id": "t", "level": "L1", "idx": 1,
        "name": "k", "repo": "vLLM", "dtype_class": "VV",
        "category": "x", "npu_difficulty": "low",
        "upstream": {},  # NO file_sha256
        "interface": {}, "oracle": {}, "tolerance_class": "_t",
        "oracle_simplified": "simple_pytorch_ok", "runner_checks": {},
        "is_canary": False, "is_hidden_only": False,
        # NO phase_a_prime block at all
    }
    (td / "meta.yaml").write_text(yaml.safe_dump(meta))
    errs = vt.validate_task(td, {"_t"}, strict=False)
    assert any("no integrity hash" in e for e in errs), errs


def test_one_hash_present_is_not_error(tmp_path):
    """Sanity: a legacy task WITH upstream.file_sha256 (and no
    phase_a_prime) is NOT flagged by the neither-hash check."""
    import hashlib

    td = tmp_path
    kernel = "@triton.jit\ndef run(i):\n    return {}\n"
    (td / "kernel.py").write_text(kernel)
    sha = hashlib.sha256(kernel.encode()).hexdigest()
    (td / "oracle_pytorch.py").write_text("def run(i):\n    return {}\n")
    (td / "shapes.yaml").write_text("public_shapes: []\n")
    (td / "contract.md").write_text("c\n")
    meta = {
        "schema_version": "1.0", "task_id": "t", "level": "L1", "idx": 1,
        "name": "k", "repo": "vLLM", "dtype_class": "VV",
        "category": "x", "npu_difficulty": "low",
        "upstream": {"file_sha256": sha},
        "interface": {}, "oracle": {}, "tolerance_class": "_t",
        "oracle_simplified": "simple_pytorch_ok", "runner_checks": {},
        "is_canary": False, "is_hidden_only": False,
    }
    (td / "meta.yaml").write_text(yaml.safe_dump(meta))
    errs = vt.validate_task(td, {"_t"}, strict=False)
    assert not any("no integrity hash" in e for e in errs), errs


def test_mixed_rollout_normal_mode_green_gate_blocks():
    """Mixed: one rewritten (has standalone) + one legacy -> normal
    schema/hash mode would pass BOTH; release-gate independently
    blocks if any eligible is not PASS (two checks don't cross-fire)."""
    tasks = [
        _pass("rewritten"),
        _T("legacy", {}),  # no phase_a_prime -> not PASS
    ]
    rc = vt.run_release_gate(tasks=tasks, eligible_ids={"rewritten", "legacy"})
    assert rc == 1  # release-gate blocks (legacy not PASS)
    # The normal schema/hash mode is a SEPARATE codepath (validate_task)
    # -- tested green above for both legacy and rewritten shapes.
