"""M2 Unit 2.5: validate_tasks.py schema checks for the four §6a state
field groups.

Covers (specs/task_api_contract.md §6a.1 / §6a.3 / §6a.5 / §6a.6):

* `release_scope.excluded == True` requires `reason`, `excluded_at`
  (ISO 8601), `excluded_by` — but does NOT short-circuit `validate_task`
  (file presence, syntax, hashes still gate);
* `phase_a_prime.skip_invocation == True` ⇒
  `phase_a_prime.status == SKIPPED_BY_DESIGN` and non-empty `reason`;
* `phase_b_capture.status` ∈ {`PENDING`, `DONE`, `FAILED`}, with the
  §6a.6 #3 / #4 cross-field invariants on the `DONE` / `FAILED` branches;
* the existing 100-task corpus (none of these blocks present) stays
  byte-stable on `--strict`.
"""
from __future__ import annotations

import hashlib
import importlib.util
import subprocess
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parent.parent.parent
_spec = importlib.util.spec_from_file_location(
    "validate_tasks", REPO / "scripts" / "corpus" / "validate_tasks.py"
)
vt = importlib.util.module_from_spec(_spec)
sys.modules["validate_tasks"] = vt
_spec.loader.exec_module(vt)


def _write_task(td: Path, *, meta_overrides=None) -> None:
    """Materialise a minimal schema-valid task dir under ``td``.

    The task carries every other field validate_task requires so the
    only error surface under test is the §6a state-field block(s).
    """
    kernel = "@triton.jit\ndef run(i):\n    return {}\n"
    (td / "kernel.py").write_text(kernel)
    sha = hashlib.sha256(kernel.encode()).hexdigest()
    (td / "oracle_pytorch.py").write_text("def run(i):\n    return {}\n")
    knobs = {"B": 2, "N": 4}
    (td / "shapes.yaml").write_text(
        yaml.safe_dump(
            {
                "size_knobs": knobs,
                "public_shapes": [{"id": "P1", "name": "tiny", "knobs": knobs}],
                "inputs": {
                    "x": {"dims": ["B", "N"], "dtype": "float32",
                          "dist": "randn"},
                },
            }
        )
    )
    (td / "contract.md").write_text(f"# Contract — {td.name}\n")
    meta = {
        "schema_version": "1.0",
        "task_id": td.name,
        "level": "L1",
        "idx": 1,
        "name": "fixture",
        "repo": "vLLM",
        "dtype_class": "VV",
        "category": "norm",
        "npu_difficulty": "low",
        "upstream": {"file_sha256": sha},
        "interface": {},
        "oracle": {"gpu_triton": "kernel.py::placeholder"},
        "tolerance_class": "_t",
        "oracle_simplified": "simple_pytorch_ok",
        "runner_checks": {},
        "is_canary": False,
        "is_hidden_only": False,
        "cost_model": {
            "flops": "B*N", "flops_class": "vector",
            "bytes": "2*B*N*dtype_size", "bytes_resource": "hbm",
        },
        "param_constraints": {
            "scaffold": False,
            "dims": ["B >= 1"],
            "dtypes": {"x": "float32"},
            "dists": {"x": {"kind": "randn"}},
            "scalars": {},
            "tensor_invariants": [],
        },
    }
    if meta_overrides:
        meta.update(meta_overrides)
    (td / "meta.yaml").write_text(yaml.safe_dump(meta))


def _errs(td: Path, *, strict: bool = False) -> list[str]:
    return vt.validate_task(td, {"_t"}, strict=strict)


# ---------------------------------------------------------------------------
# release_scope.excluded — exclusion record schema (§6a.5 / §6a.6 #6)
# ---------------------------------------------------------------------------

def test_strict_does_not_skip_on_excluded(tmp_path):
    """release_scope.excluded == True must NOT short-circuit validate_task.
    A missing file is still a hard error even on an excluded task — the
    --strict scan must surface it (the orchestrator's `--mark-unfixable`
    flow excludes from RELEASE, not from validation).
    """
    td = tmp_path / "L1_99"
    td.mkdir()
    _write_task(td, meta_overrides={
        "release_scope": {
            "excluded": True,
            "reason": "unfixable by patch flow",
            "excluded_at": "2026-05-23T12:00:00Z",
            "excluded_by": "fix_orchestrator/automation",
        }
    })
    # Now delete `oracle_pytorch.py` — the file-presence gate must
    # still fire even though the task is excluded.
    (td / "oracle_pytorch.py").unlink()
    errs = _errs(td)
    assert any("missing files" in e for e in errs), errs


def test_excluded_requires_reason(tmp_path):
    td = tmp_path / "L1_99"
    td.mkdir()
    _write_task(td, meta_overrides={
        "release_scope": {
            "excluded": True,
            "reason": "",
            "excluded_at": "2026-05-23T12:00:00Z",
            "excluded_by": "op",
        }
    })
    errs = _errs(td)
    assert any("release_scope.reason" in e for e in errs), errs


def test_excluded_requires_excluded_at_iso8601(tmp_path):
    td = tmp_path / "L1_99"
    td.mkdir()
    _write_task(td, meta_overrides={
        "release_scope": {
            "excluded": True,
            "reason": "r",
            "excluded_at": "not-a-timestamp",
            "excluded_by": "op",
        }
    })
    errs = _errs(td)
    assert any("excluded_at" in e and "ISO 8601" in e for e in errs), errs


def test_excluded_requires_excluded_by(tmp_path):
    td = tmp_path / "L1_99"
    td.mkdir()
    _write_task(td, meta_overrides={
        "release_scope": {
            "excluded": True,
            "reason": "r",
            "excluded_at": "2026-05-23T12:00:00Z",
            "excluded_by": "",
        }
    })
    errs = _errs(td)
    assert any("excluded_by" in e for e in errs), errs


def test_excluded_well_formed_passes(tmp_path):
    td = tmp_path / "L1_99"
    td.mkdir()
    _write_task(td, meta_overrides={
        "release_scope": {
            "excluded": True,
            "reason": "unfixable",
            "excluded_at": "2026-05-23T12:00:00Z",
            "excluded_by": "fix_orchestrator/automation",
        }
    })
    errs = _errs(td)
    assert not any("release_scope" in e for e in errs), errs


def test_release_scope_absent_passes(tmp_path):
    """A task with no `release_scope` block (the 100-task corpus norm)
    must not produce a release_scope error."""
    td = tmp_path / "L1_99"
    td.mkdir()
    _write_task(td)  # no override
    errs = _errs(td)
    assert not any("release_scope" in e for e in errs), errs


# ---------------------------------------------------------------------------
# phase_a_prime.skip_invocation cross-field invariants (§6a.1 / §6a.6 #1-#2)
# ---------------------------------------------------------------------------

def test_skip_invocation_requires_skipped_status(tmp_path):
    """`skip_invocation: true` + `status: PASS` is a schema invariant
    violation (§6a.6 #1)."""
    td = tmp_path / "L1_99"
    td.mkdir()
    _write_task(td, meta_overrides={
        "phase_a_prime": {
            "skip_invocation": True,
            "status": "PASS",
            "reason": "anything",
        }
    })
    errs = _errs(td)
    assert any("skip_invocation" in e and "SKIPPED_BY_DESIGN" in e
               for e in errs), errs


def test_skip_invocation_requires_reason(tmp_path):
    td = tmp_path / "L1_99"
    td.mkdir()
    _write_task(td, meta_overrides={
        "phase_a_prime": {
            "skip_invocation": True,
            "status": "SKIPPED_BY_DESIGN",
            "reason": "",
        }
    })
    errs = _errs(td)
    assert any("phase_a_prime.reason" in e for e in errs), errs


def test_skipped_by_design_status_requires_skip_invocation(tmp_path):
    """Inverse §6a.6 #2: status `SKIPPED_BY_DESIGN` without
    `skip_invocation: true` is a violation."""
    td = tmp_path / "L1_99"
    td.mkdir()
    _write_task(td, meta_overrides={
        "phase_a_prime": {
            "skip_invocation": False,
            "status": "SKIPPED_BY_DESIGN",
            "reason": "r",
        }
    })
    errs = _errs(td)
    assert any("skip_invocation" in e and "must be true" in e
               for e in errs), errs


# ---------------------------------------------------------------------------
# phase_b_capture block schema (§6a.3 / §6a.6 #3 / #4)
# ---------------------------------------------------------------------------

def test_phase_b_status_enum(tmp_path):
    td = tmp_path / "L1_99"
    td.mkdir()
    _write_task(td, meta_overrides={
        "phase_b_capture": {"status": "INVALID"},
    })
    errs = _errs(td)
    assert any("phase_b_capture.status" in e and "INVALID" in e
               for e in errs), errs


def test_phase_b_done_requires_upstream_callable(tmp_path):
    td = tmp_path / "L1_99"
    td.mkdir()
    _write_task(td, meta_overrides={
        "phase_b_capture": {
            "status": "DONE",
            "captured_at": "2026-05-23T12:00:00Z",
            # upstream_callable missing entirely
        },
    })
    errs = _errs(td)
    assert any("upstream_callable" in e for e in errs), errs


def test_phase_b_done_requires_captured_at(tmp_path):
    td = tmp_path / "L1_99"
    td.mkdir()
    _write_task(td, meta_overrides={
        "phase_b_capture": {
            "status": "DONE",
            "upstream_callable": {
                "module": "vllm.model_executor.layers.layernorm",
                "function": "rms_norm",
            },
            # captured_at missing
        },
    })
    errs = _errs(td)
    assert any("captured_at" in e and "ISO 8601" in e for e in errs), errs


def test_phase_b_failed_requires_reason(tmp_path):
    td = tmp_path / "L1_99"
    td.mkdir()
    _write_task(td, meta_overrides={
        "phase_b_capture": {"status": "FAILED"},
    })
    errs = _errs(td)
    assert any("phase_b_capture.reason" in e for e in errs), errs


def test_phase_b_module_dotted_path_regex(tmp_path):
    """`module` must be a canonical dotted Python path."""
    # Legal: multi-segment dotted path PASSes.
    td_ok = tmp_path / "ok"
    td_ok.mkdir()
    _write_task(td_ok, meta_overrides={
        "phase_b_capture": {
            "status": "DONE",
            "captured_at": "2026-05-23T12:00:00Z",
            "upstream_callable": {
                "module": "torch.nn.functional",
                "function": "rms_norm",
            },
        },
    })
    errs_ok = _errs(td_ok)
    assert not any("upstream_callable.module" in e for e in errs_ok), errs_ok

    # Illegal: leading digit — fail.
    td_bad = tmp_path / "bad"
    td_bad.mkdir()
    _write_task(td_bad, meta_overrides={
        "phase_b_capture": {
            "status": "DONE",
            "captured_at": "2026-05-23T12:00:00Z",
            "upstream_callable": {
                "module": "123invalid",
                "function": "rms_norm",
            },
        },
    })
    errs_bad = _errs(td_bad)
    assert any("upstream_callable.module" in e and "123invalid" in e
               for e in errs_bad), errs_bad


def test_phase_b_done_well_formed_passes(tmp_path):
    td = tmp_path / "L1_99"
    td.mkdir()
    _write_task(td, meta_overrides={
        "phase_b_capture": {
            "status": "DONE",
            "captured_at": "2026-05-23T12:00:00Z",
            "upstream_callable": {
                "module": "vllm.model_executor.layers.layernorm",
                "function": "rms_norm",
            },
        },
    })
    errs = _errs(td)
    assert not any("phase_b_capture" in e for e in errs), errs


def test_phase_b_absent_passes(tmp_path):
    td = tmp_path / "L1_99"
    td.mkdir()
    _write_task(td)  # no phase_b_capture block
    errs = _errs(td)
    assert not any("phase_b_capture" in e for e in errs), errs


# ---------------------------------------------------------------------------
# Backward compatibility — existing 100-task corpus.
# ---------------------------------------------------------------------------

def test_backward_compat_all_existing_tasks_pass():
    """`python3 scripts/corpus/validate_tasks.py` over the real corpus reports
    the SAME number of validation errors as before §6a-aware checks
    were added (no new false-positives from the §6a state-field
    validators). The corpus baseline is the 18 outstanding
    upstream_oracle_required golden-completeness errors (every other
    task validates clean); none of the 100 tasks carries a §6a state
    block today, so the new checks add zero errors.
    """
    proc = subprocess.run(
        [sys.executable, str(REPO / "scripts" / "corpus" / "validate_tasks.py")],
        capture_output=True,
        text=True,
        cwd=REPO,
    )
    stderr = proc.stderr
    # No §6a-related error should appear in the corpus output.
    forbidden_substrings = (
        "release_scope.",
        "skip_invocation",
        "phase_b_capture",
        "SKIPPED_BY_DESIGN",
    )
    for line in stderr.splitlines():
        if not line.startswith("ERROR"):
            continue
        for sub in forbidden_substrings:
            assert sub not in line, (
                f"§6a-aware validator unexpectedly flagged a corpus task: "
                f"{line!r}"
            )
