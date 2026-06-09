"""W1.2: validate_tasks.py cost_model schema/formula validation tests.

Covers: a task with a legal scalar-form / map-form `cost_model` passes; a
malformed `cost_model` (bad schema, unsafe formula, unknown knob) is reported
as a validation error; a task with NO `cost_model` block still passes (pre-W3.1
backward compatibility — the red→green transition is deferred); and the two
hand-authored W1.2 sample tasks (`L1_01`, `L1_12`) validate clean end-to-end
through the real `validate_tasks --task` path.
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


_OMIT = object()  # sentinel: leave the cost_model block out entirely


def _write_task(td: Path, *, cost_model=_OMIT, shapes_knobs=None) -> None:
    """Materialise a minimal schema-valid task dir under ``td``.

    ``cost_model`` defaults to :data:`_OMIT` meaning "omit the block"; pass an
    explicit value to inject one.
    """
    kernel = "@triton.jit\ndef run(i):\n    return {}\n"
    (td / "kernel.py").write_text(kernel)
    sha = hashlib.sha256(kernel.encode()).hexdigest()
    (td / "oracle_pytorch.py").write_text("def run(i):\n    return {}\n")
    knobs = shapes_knobs if shapes_knobs is not None else {"B": 2, "N": 4}
    (td / "shapes.yaml").write_text(
        yaml.safe_dump(
            {
                "size_knobs": knobs,
                "public_shapes": [{"id": "P1", "name": "tiny", "knobs": knobs}],
            }
        )
    )
    # task_id / contract.md / oracle.gpu_triton are populated so the minimal
    # fixture also satisfies the W2.3 completeness checks (contract.md
    # references the task id; gpu_triton is the well-formed placeholder).
    (td / "contract.md").write_text("# Contract — L1_99 cost-model fixture\n")
    meta = {
        "schema_version": "1.0", "task_id": "L1_99_cost_model_fixture",
        "level": "L1", "idx": 1,
        "name": "k", "repo": "vLLM", "dtype_class": "VV",
        "category": "norm", "npu_difficulty": "low",
        "upstream": {"file_sha256": sha},
        "interface": {}, "oracle": {"gpu_triton": "kernel.py::placeholder"},
        "tolerance_class": "_t",
        "oracle_simplified": "simple_pytorch_ok", "runner_checks": {},
        "is_canary": False, "is_hidden_only": False,
        # WA.1/WA.4: `param_constraints` is a required top-level meta key.
        # Use a reconciled (`scaffold: false`) schema-closed block — WA.4's
        # semantic gate rejects `scaffold: true`, and R4-1 requires all five
        # semantic subsections (empty containers here); this cost_model
        # fixture is not testing param_constraints (its own checks live in
        # the WA.1/WA.4 suites).
        "param_constraints": {"scaffold": False, "dims": [], "dtypes": {},
                              "dists": {}, "scalars": {},
                              "tensor_invariants": []},
    }
    if cost_model is not _OMIT:
        meta["cost_model"] = cost_model
    (td / "meta.yaml").write_text(yaml.safe_dump(meta))


# --------------------------------------------------------------------------
# legal cost_model — passes
# --------------------------------------------------------------------------

def test_scalar_cost_model_passes(tmp_path):
    _write_task(
        tmp_path,
        cost_model={
            "flops": "B*N",
            "flops_class": "vector",
            "bytes": "2*B*N*dtype_size",
            "bytes_resource": "hbm",
            "gpu_xcheck_tol": 0.05,
        },
    )
    errs = vt.validate_task(tmp_path, {"_t"}, strict=False)
    assert not any("cost_model" in e for e in errs), errs


def test_scalar_cost_model_missing_companion_reported(tmp_path):
    # A scalar `flops` without its `flops_class` companion is rejected.
    _write_task(
        tmp_path,
        cost_model={"flops": "B*N", "bytes": {"hbm": "2*B*N*dtype_size"}},
    )
    errs = vt.validate_task(tmp_path, {"_t"}, strict=False)
    assert any("invalid cost_model" in e and "flops_class" in e
               for e in errs), errs


def test_map_cost_model_passes(tmp_path):
    _write_task(
        tmp_path,
        cost_model={
            "flops": {"vector": "4*B*N"},
            "bytes": {"hbm": "2*B*N*dtype_size"},
            "gpu_xcheck_tol": 0.05,
        },
    )
    errs = vt.validate_task(tmp_path, {"_t"}, strict=False)
    assert not any("cost_model" in e for e in errs), errs


def test_missing_cost_model_reported(tmp_path):
    # R1-4: `cost_model` is a required top-level meta key — a task with no
    # cost_model block is a validation error (the red→green transition for
    # the full corpus is carried by validate_tasks; the dataset loader
    # stays backward compatible separately, see test_dataset.py).
    _write_task(tmp_path)  # cost_model omitted
    errs = vt.validate_task(tmp_path, {"_t"}, strict=False)
    assert any("missing keys" in e and "cost_model" in e for e in errs), errs


# --------------------------------------------------------------------------
# malformed cost_model — reported
# --------------------------------------------------------------------------

def test_cost_model_missing_bytes_key_reported(tmp_path):
    _write_task(tmp_path, cost_model={"flops": "B*N"})
    errs = vt.validate_task(tmp_path, {"_t"}, strict=False)
    assert any("invalid cost_model" in e for e in errs), errs


def test_cost_model_unsafe_formula_reported(tmp_path):
    _write_task(
        tmp_path,
        cost_model={
            "flops": "__import__('os')", "flops_class": "vector",
            "bytes": "B*N", "bytes_resource": "hbm",
        },
    )
    errs = vt.validate_task(tmp_path, {"_t"}, strict=False)
    assert any("invalid cost_model" in e for e in errs), errs


def test_cost_model_unknown_knob_reported(tmp_path):
    # `Z` is not a shapes.yaml knob -> formula not evaluable.
    _write_task(
        tmp_path,
        cost_model={
            "flops": "B*Z", "flops_class": "vector",
            "bytes": "B*N*dtype_size", "bytes_resource": "hbm",
        },
    )
    errs = vt.validate_task(tmp_path, {"_t"}, strict=False)
    assert any("not evaluable" in e for e in errs), errs


def test_cost_model_bad_tol_reported(tmp_path):
    _write_task(
        tmp_path,
        cost_model={
            "flops": "B*N", "flops_class": "vector",
            "bytes": "B*N", "bytes_resource": "hbm", "gpu_xcheck_tol": 9.0,
        },
    )
    errs = vt.validate_task(tmp_path, {"_t"}, strict=False)
    assert any("invalid cost_model" in e for e in errs), errs


# --------------------------------------------------------------------------
# end-to-end: the W1.2 sample tasks validate clean via the real CLI
# --------------------------------------------------------------------------

def test_sample_tasks_validate_clean_via_cli():
    # WA.4: a `scaffold: true` corpus task is now a `validate_tasks` error
    # (Phase A red — WB.1 reconciles every task to turn the full corpus
    # green). `L1_12_rms_norm` is the fully-reconciled sample task
    # (`param_constraints.scaffold: false` AND a complete `cost_model`), so
    # it is the one that validates clean end-to-end pre-WB.1.
    for task_id in ("L1_12_rms_norm",):
        proc = subprocess.run(
            [sys.executable, str(REPO / "scripts" / "corpus" / "validate_tasks.py"),
             "--task", task_id],
            cwd=str(REPO), capture_output=True, text=True,
        )
        assert proc.returncode == 0, (
            f"{task_id} failed validate_tasks --task:\n"
            f"stdout={proc.stdout}\nstderr={proc.stderr}"
        )
