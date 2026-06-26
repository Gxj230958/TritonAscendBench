"""G7 — NPU software-stack provenance + offline-golden completeness report.

G7b: ``software_hash`` ({cann, torch_npu, triton_ascend}) is read from
version_matrix.toml and emitted into the NPU round_result (None on GPU).
G7a: the pre-run golden-completeness report partitions upstream_oracle_required
tasks using the SAME predicate the eval loader uses.
"""
from __future__ import annotations

from types import SimpleNamespace

from ascendbench.policies.version_matrix import (
    npu_software_hash,
    software_hash_complete,
)
from scripts.provenance.golden_completeness_report import report_completeness


# ─────────────────────────────────────────────────────────────────────────────
# G7b — software_hash
# ─────────────────────────────────────────────────────────────────────────────
def test_npu_software_hash_has_scoring_spec_keys():
    sw = npu_software_hash()
    assert set(sw) == {"cann", "torch_npu", "triton_ascend"}
    # All declared in version_matrix.toml [npu] — non-empty for the shipped file.
    assert sw["cann"] and sw["torch_npu"] and sw["triton_ascend"]


def test_software_hash_complete_true_for_full_stack():
    assert software_hash_complete(
        {"cann": "8.5.0", "torch_npu": "2.8.0", "triton_ascend": "3.2.0"})


def test_software_hash_complete_false_on_missing_key():
    assert not software_hash_complete(
        {"cann": "8.5.0", "torch_npu": "", "triton_ascend": "3.2.0"})
    assert not software_hash_complete({"cann": "8.5.0"})


def test_round_result_emits_software_hash_only_for_npu():
    from ascendbench.engine.cli import _round_result_json
    from ascendbench.engine.eval import TaskEvalResult

    r = TaskEvalResult(task_id="T", static_checks_passed=True)
    npu = _round_result_json(r, backend="npu")
    gpu = _round_result_json(r, backend="gpu")
    assert npu["software_hash"] == npu_software_hash()
    assert gpu["software_hash"] is None


# ─────────────────────────────────────────────────────────────────────────────
# G7a — golden completeness report
# ─────────────────────────────────────────────────────────────────────────────
def _task(task_id, tier):
    return SimpleNamespace(task_id=task_id, oracle_tier=tier)


def test_completeness_filters_to_upstream_oracle_required():
    tasks = [
        _task("L1_01", "simple_pytorch_ok"),       # not upstream — ignored
        _task("L2_23", "upstream_oracle_required"),
        _task("L2_24", "upstream_oracle_required"),
    ]
    # L2_23 complete, L2_24 missing.
    rep = report_completeness(tasks, lambda t: t.task_id == "L2_23")
    assert rep["complete"] == ["L2_23"]
    assert rep["missing"] == ["L2_24"]


def test_completeness_all_present():
    tasks = [_task("A", "upstream_oracle_required"),
             _task("B", "upstream_oracle_required")]
    rep = report_completeness(tasks, lambda t: True)
    assert rep["missing"] == []
    assert rep["complete"] == ["A", "B"]


def test_completeness_ignores_non_upstream_even_if_incomplete():
    """A non-upstream task with no golden must NOT appear as missing — it uses
    the in-process oracle, not an offline golden."""
    tasks = [_task("L1_01", "formula_exact_manual")]
    rep = report_completeness(tasks, lambda t: False)
    assert rep == {"complete": [], "missing": []}


def test_completeness_output_sorted():
    tasks = [_task("Z", "upstream_oracle_required"),
             _task("A", "upstream_oracle_required")]
    rep = report_completeness(tasks, lambda t: True)
    assert rep["complete"] == ["A", "Z"]


# ─────────────────────────────────────────────────────────────────────────────
# G7 — verify_version_matrix rejects empty NPU software_hash values (gate)
# ─────────────────────────────────────────────────────────────────────────────
def test_verify_version_matrix_passes_on_shipped_file():
    from scripts.provenance.verify_version_matrix import main as vvm_main

    assert vvm_main([]) == 0  # the bundled matrix is complete


def test_verify_version_matrix_rejects_empty_npu_software(tmp_path, capsys):
    """An empty npu.cann / torch_npu / triton_ascend must fail the gate even
    though the key is present (G7 caveat)."""
    from ascendbench.policies import policy_path
    from scripts.provenance.verify_version_matrix import main as vvm_main

    src = policy_path("version_matrix.toml").read_text()
    bad = src.replace('cann = "8.5.0"', 'cann = ""', 1)
    p = tmp_path / "vm_bad.toml"
    p.write_text(bad)

    assert vvm_main([str(p)]) == 1
    assert "empty NPU software_hash value npu.cann" in capsys.readouterr().err
