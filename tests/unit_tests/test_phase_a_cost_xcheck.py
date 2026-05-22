"""W4.2: cost-model GPU cross-check (scripts/phase_a_cost_xcheck.py) tests.

Covers: analytic W_c / M_r evaluation reuses cost_model.FormulaEnv; the
tolerance reconciliation logic (within / over tol, the no-kernel-launch
gate, zero-measured edge); the Nsight Compute DRAM-bytes parser; the
fail-closed behaviour when ``ncu`` is unavailable or yields no metric;
the atomic meta.yaml writer; the L1_21 (and generic non-score-eligible)
skip special case; dry-run never writes / never PASSes. The real ``ncu``
profiling pass is exercised through a mocked ``measure_kernel_bytes`` /
mocked subprocess -- the dev machine has no CUDA or Nsight Compute, so
the live profiler path is verified on the GPU host, not here.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parent.parent.parent
_spec = importlib.util.spec_from_file_location(
    "phase_a_cost_xcheck", REPO / "scripts" / "phase_a_cost_xcheck.py"
)
xc = importlib.util.module_from_spec(_spec)
sys.modules["phase_a_cost_xcheck"] = xc
_spec.loader.exec_module(xc)

from triton_ascend_bench.cost_model import parse_cost_model  # noqa: E402
from triton_ascend_bench.dataset import ShapeSpec, TaskDef, load_task  # noqa: E402


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def _make_task(
    tmp_path: Path,
    *,
    cost_model_block: dict,
    size_knobs: dict | None = None,
    public_shapes: list[dict] | None = None,
    inputs: dict | None = None,
    oracle_src: str = "def run(inputs):\n    return {}\n",
    shapes_marker: str = "",
) -> TaskDef:
    """Build a minimal on-disk task and load it via dataset.load_task-style
    construction. Returns a TaskDef whose task_dir is under ``tmp_path``."""
    task_dir = tmp_path / "L9_99_synthetic"
    task_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "task_id": "L9_99_synthetic",
        "level": "L9",
        "oracle": {"gpu_triton": "kernel.py::run"},
        "tolerance_class": "exact",
        "cost_model": cost_model_block,
    }
    (task_dir / "meta.yaml").write_text(yaml.safe_dump(meta, sort_keys=False))
    shapes = {
        "size_knobs": size_knobs or {},
        "public_shapes": public_shapes or [{"id": "P1", "name": "p1"}],
    }
    if inputs:
        shapes["inputs"] = inputs
    shapes_txt = yaml.safe_dump(shapes, sort_keys=False)
    if shapes_marker:
        shapes_txt += f"\n# {shapes_marker}\n"
    (task_dir / "shapes.yaml").write_text(shapes_txt)
    (task_dir / "contract.md").write_text("synthetic")
    (task_dir / "kernel.py").write_text("def run(inputs):\n    return {}\n")
    (task_dir / "oracle_pytorch.py").write_text(oracle_src)

    cm = parse_cost_model(cost_model_block)
    return TaskDef(
        task_id="L9_99_synthetic",
        level=9,
        task_dir=task_dir,
        meta=meta,
        public_shapes=[
            ShapeSpec(
                id=s["id"], name=s.get("name", s["id"]),
                params={k: v for k, v in s.items() if k not in {"id", "name"}},
            )
            for s in (public_shapes or [{"id": "P1", "name": "p1"}])
        ],
        contract_md="synthetic",
        kernel_py="def run(inputs):\n    return {}\n",
        kernel_py_sha256="0" * 64,
        cost_model=cm,
    )


# --------------------------------------------------------------------------
# analytic evaluation -- reuses cost_model.FormulaEnv
# --------------------------------------------------------------------------

def test_evaluate_analytic_scalar_form():
    cm = parse_cost_model(
        {"flops": "4*B*D", "flops_class": "vector",
         "bytes": "2*B*D * dtype_size", "bytes_resource": "hbm"}
    )
    flops, bytes_ = xc.evaluate_analytic(cm, {"B": 4.0, "D": 64.0}, 2.0)
    assert flops == {"vector": 4 * 4 * 64}
    assert bytes_ == {"hbm": 2 * 4 * 64 * 2}


def test_evaluate_analytic_map_form_and_dtype_default():
    cm = parse_cost_model(
        {"flops": {"fp16_cube": "2*M*N*K", "vector": "M*N"},
         "bytes": {"hbm": "(M*K + K*N) * dtype_size"}}
    )
    # dtype_size=None -> FormulaEnv supplies the fp16 default (2.0).
    flops, bytes_ = xc.evaluate_analytic(cm, {"M": 2.0, "N": 3.0, "K": 4.0}, None)
    assert flops == {"fp16_cube": 2 * 2 * 3 * 4, "vector": 2 * 3}
    assert bytes_ == {"hbm": (2 * 4 + 4 * 3) * 2.0}


def test_representative_dtype_size_first_tensor_input():
    doc = {"inputs": {"s": {"scalar": 1}, "x": {"dims": ["B"], "dtype": "float32"}}}
    assert xc._representative_dtype_size(doc) == 4.0
    # scalar-only inputs -> None (caller uses fp16 default)
    assert xc._representative_dtype_size({"inputs": {"s": {"scalar": 3}}}) is None


def test_shape_knobs_merges_size_knobs_with_row_override():
    doc = {
        "size_knobs": {"B": 32, "D": 1024},
        "public_shapes": [{"id": "P1", "name": "p1", "knobs": {"B": 4}}],
    }
    assert xc._shape_knobs(doc, "P1") == {"B": 4.0, "D": 1024.0}
    assert xc._shape_knobs(doc, "missing") is None


# --------------------------------------------------------------------------
# tolerance reconciliation
# --------------------------------------------------------------------------

def test_relative_deviation_edges():
    assert xc._relative_deviation(100.0, 100.0) == 0.0
    assert xc._relative_deviation(105.0, 100.0) == pytest.approx(0.05)
    # zero measured: only a zero analytic reconciles
    assert xc._relative_deviation(0.0, 0.0) == 0.0
    assert xc._relative_deviation(1.0, 0.0) == float("inf")


def test_reconcile_shape_within_tol_passes():
    sx = xc.reconcile_shape(
        "P1", {"vector": 256.0}, {"hbm": 1024.0},
        measured_bytes=1030.0, kernel_launched=True, tol=0.05,
    )
    assert sx.within_tol is True
    assert sx.bytes_deviation == pytest.approx(6 / 1030.0)


def test_reconcile_shape_over_tol_fails():
    sx = xc.reconcile_shape(
        "P1", {"vector": 256.0}, {"hbm": 1024.0},
        measured_bytes=2000.0, kernel_launched=True, tol=0.05,
    )
    assert sx.within_tol is False
    assert "deviation" in sx.note


def test_reconcile_shape_no_kernel_launch_blocks_even_if_bytes_match():
    # bytes are within tol, but ncu profiled no kernel launch => the kernel
    # did not actually run on the GPU, so the cost model cannot be certified.
    sx = xc.reconcile_shape(
        "P1", {"vector": 256.0}, {"hbm": 1024.0},
        measured_bytes=1024.0, kernel_launched=False, tol=0.05,
    )
    assert sx.within_tol is False
    assert "kernel launch" in sx.note


def test_reconcile_shape_not_measured_is_advisory():
    sx = xc.reconcile_shape(
        "P1", {"vector": 1.0}, {"hbm": 1.0},
        measured_bytes=None, kernel_launched=None, tol=0.05,
    )
    assert sx.within_tol is False
    assert "not measured" in sx.note


# --------------------------------------------------------------------------
# Nsight Compute DRAM-bytes parser
# --------------------------------------------------------------------------

# A representative `ncu --csv` capture: a couple of banner lines, then the
# CSV header, then one read + one write metric row per profiled launch.
_NCU_CSV_OK = (
    '==PROF== Connected to process 1234\n'
    '==PROF== Profiling "vecadd_kernel" - 0: 0%....50%....100%\n'
    '"ID","Kernel Name","Metric Name","Metric Unit","Metric Value"\n'
    '"0","vecadd_kernel","dram__bytes_read.sum","byte","2048"\n'
    '"0","vecadd_kernel","dram__bytes_write.sum","byte","1024"\n'
)
# Two launches, mixed units (Kbyte must be scaled to bytes).
_NCU_CSV_TWO_LAUNCHES = (
    '"ID","Kernel Name","Metric Name","Metric Unit","Metric Value"\n'
    '"0","k_a","dram__bytes_read.sum","Kbyte","1"\n'
    '"0","k_a","dram__bytes_write.sum","byte","500"\n'
    '"1","k_b","dram__bytes_read.sum","byte","250"\n'
    '"1","k_b","dram__bytes_write.sum","byte","250"\n'
)


def test_parse_ncu_dram_bytes_sums_read_and_write():
    total, launches = xc._parse_ncu_dram_bytes(_NCU_CSV_OK)
    assert total == 2048 + 1024
    assert launches == 1


def test_parse_ncu_dram_bytes_two_launches_and_unit_scaling():
    total, launches = xc._parse_ncu_dram_bytes(_NCU_CSV_TWO_LAUNCHES)
    # 1 Kbyte -> 1000 bytes, + 500 + 250 + 250
    assert total == 1000 + 500 + 250 + 250
    assert launches == 2


def test_parse_ncu_dram_bytes_no_header_fails_closed():
    with pytest.raises(xc.NsightUnavailableError, match="no parseable CSV"):
        xc._parse_ncu_dram_bytes("==PROF== nothing useful here\n")


def test_parse_ncu_dram_bytes_no_metric_rows_fails_closed():
    csv_no_dram = (
        '"ID","Kernel Name","Metric Name","Metric Unit","Metric Value"\n'
        '"0","k","sm__cycles_elapsed.avg","cycle","999"\n'
    )
    with pytest.raises(xc.NsightUnavailableError, match="no dram__bytes"):
        xc._parse_ncu_dram_bytes(csv_no_dram)


def test_parse_ncu_dram_bytes_unknown_unit_fails_closed():
    csv_bad_unit = (
        '"ID","Kernel Name","Metric Name","Metric Unit","Metric Value"\n'
        '"0","k","dram__bytes_read.sum","furlongs","5"\n'
    )
    with pytest.raises(xc.NsightUnavailableError, match="unrecognised unit"):
        xc._parse_ncu_dram_bytes(csv_bad_unit)


# --------------------------------------------------------------------------
# fail-closed: ncu unavailable / profiling failure -> BLOCKED, never faked
# --------------------------------------------------------------------------

def test_measure_kernel_bytes_no_ncu_raises(tmp_path, monkeypatch):
    """When ncu is not on PATH the measured pass fails closed -- it raises
    NsightUnavailableError rather than approximating HBM traffic."""
    task = _make_task(
        tmp_path,
        cost_model_block={
            "flops": "1*B", "flops_class": "vector",
            "bytes": "1000*B", "bytes_resource": "hbm",
        },
        size_knobs={"B": 1},
    )
    monkeypatch.setattr(xc, "_require_cuda", lambda: None)
    monkeypatch.setattr(xc, "_find_ncu", lambda: None)
    with pytest.raises(xc.NsightUnavailableError, match="not on PATH"):
        xc.measure_kernel_bytes(task, {}, "P1")


def test_cross_check_blocks_shape_when_ncu_unavailable(tmp_path, monkeypatch):
    """A NsightUnavailableError from the GPU pass BLOCKS the shape with an
    explicit reason -- the task verdict is BLOCKED, never a fabricated PASS."""
    task = _make_task(
        tmp_path,
        cost_model_block={
            "flops": "1*B", "flops_class": "vector",
            "bytes": "1000*B", "bytes_resource": "hbm",
        },
        size_knobs={"B": 1},
    )

    def _no_ncu(*_a, **_k):
        raise xc.NsightUnavailableError(
            "Nsight Compute (ncu) is not on PATH"
        )

    monkeypatch.setattr(xc, "measure_kernel_bytes", _no_ncu)
    v = xc.cross_check_task(task, dry_run=False)
    assert v.status == xc.STATUS_BLOCKED
    assert v.shapes[0].blocked is True
    assert v.shapes[0].within_tol is False
    assert v.shapes[0].measured_bytes is None
    assert "BLOCKED" in v.shapes[0].note
    assert "could not be measured" in v.reason


def test_cross_check_blocks_when_ncu_profiles_no_launch(tmp_path, monkeypatch):
    """ncu running but profiling zero launches is also a fail-closed BLOCK --
    a non-running kernel cannot be cross-checked."""
    task = _make_task(
        tmp_path,
        cost_model_block={
            "flops": "1*B", "flops_class": "vector",
            "bytes": "1000*B", "bytes_resource": "hbm",
        },
        size_knobs={"B": 1},
    )

    def _no_launch(*_a, **_k):
        raise xc.NsightUnavailableError("ncu profiled no kernel launch")

    monkeypatch.setattr(xc, "measure_kernel_bytes", _no_launch)
    v = xc.cross_check_task(task, dry_run=False)
    assert v.status == xc.STATUS_BLOCKED
    assert v.shapes[0].blocked is True


# --------------------------------------------------------------------------
# per-task driver -- dry-run / GPU path (mocked) / verdict
# --------------------------------------------------------------------------

def test_cross_check_dry_run_evaluates_analytic_never_passes(tmp_path):
    task = _make_task(
        tmp_path,
        cost_model_block={
            "flops": "4*B*D", "flops_class": "vector",
            "bytes": "2*B*D * dtype_size", "bytes_resource": "hbm",
        },
        size_knobs={"B": 4, "D": 64},
    )
    v = xc.cross_check_task(task, dry_run=True)
    assert v.dry_run is True
    # dry-run NEVER certifies PASS
    assert v.status == xc.STATUS_BLOCKED
    # but the analytic formulas were evaluated
    assert v.shapes[0].analytic_flops == {"vector": 4 * 4 * 64}
    assert v.shapes[0].measured_bytes is None


def test_cross_check_gpu_pass_when_bytes_within_tol(tmp_path, monkeypatch):
    task = _make_task(
        tmp_path,
        cost_model_block={
            "flops": "1*B", "flops_class": "vector",
            "bytes": "1000*B", "bytes_resource": "hbm",
        },
        size_knobs={"B": 1},
    )

    def _fake_measure(_task, _doc, _sid):
        # analytic bytes = 1000; return 1010 -> 1% deviation, within 0.05.
        return 1010.0, True

    monkeypatch.setattr(xc, "measure_kernel_bytes", _fake_measure)
    v = xc.cross_check_task(task, dry_run=False)
    assert v.status == xc.STATUS_PASS
    assert v.shapes[0].within_tol is True


def test_cross_check_gpu_blocked_when_bytes_exceed_tol(tmp_path, monkeypatch):
    task = _make_task(
        tmp_path,
        cost_model_block={
            "flops": "1*B", "flops_class": "vector",
            "bytes": "1000*B", "bytes_resource": "hbm",
        },
        size_knobs={"B": 1},
    )

    def _fake_measure(_task, _doc, _sid):
        return 5000.0, True  # 400% deviation

    monkeypatch.setattr(xc, "measure_kernel_bytes", _fake_measure)
    v = xc.cross_check_task(task, dry_run=False)
    assert v.status == xc.STATUS_BLOCKED
    assert "exceeded" in v.reason


# --------------------------------------------------------------------------
# L1_21 / non-score-eligible special case
# --------------------------------------------------------------------------

def test_l1_21_non_score_eligible_is_skipped(tmp_path, monkeypatch):
    """A stub-oracle task (raise NotImplementedError) is non-score-eligible
    and must be SKIPPED, never BLOCKED -- exactly the L1_21 placeholder
    cost_model case."""
    task = _make_task(
        tmp_path,
        cost_model_block={
            "flops": "1", "flops_class": "vector",
            "bytes": "1", "bytes_resource": "hbm",
        },
        oracle_src="def run(inputs):\n    raise NotImplementedError\n",
    )

    def _boom(*_a, **_k):  # the GPU pass must NOT be reached for a skip
        raise AssertionError("measure_kernel_bytes called for a skipped task")

    monkeypatch.setattr(xc, "measure_kernel_bytes", _boom)
    v = xc.cross_check_task(task, dry_run=False)
    assert v.status == xc.STATUS_SKIPPED
    assert v.shapes == []


def test_real_l1_21_task_is_skipped():
    """The actual L1_21_gumbel_sample task in the corpus is skipped."""
    task = load_task("L1_21_gumbel_sample")
    v = xc.cross_check_task(task, dry_run=True)
    assert v.status == xc.STATUS_SKIPPED


def test_phase_c_manual_shapes_also_skipped(tmp_path):
    task = _make_task(
        tmp_path,
        cost_model_block={
            "flops": "1", "flops_class": "vector",
            "bytes": "1", "bytes_resource": "hbm",
        },
        shapes_marker="PHASE-C-MANUAL",
    )
    v = xc.cross_check_task(task, dry_run=True)
    assert v.status == xc.STATUS_SKIPPED


# --------------------------------------------------------------------------
# atomic meta.yaml writer
# --------------------------------------------------------------------------

def test_update_meta_writes_status_and_evidence(tmp_path, monkeypatch):
    task = _make_task(
        tmp_path,
        cost_model_block={
            "flops": "1*B", "flops_class": "vector",
            "bytes": "1000*B", "bytes_resource": "hbm",
        },
        size_knobs={"B": 1},
    )
    monkeypatch.setattr(
        xc, "measure_kernel_bytes", lambda *_: (1010.0, True)
    )
    v = xc.cross_check_task(task, dry_run=False)
    xc.update_meta_cost_model(task.task_dir, v)

    meta = yaml.safe_load((task.task_dir / "meta.yaml").read_text())
    cm = meta["cost_model"]
    assert cm["status"] == xc.STATUS_PASS
    assert cm["xcheck_run_id"] == v.run_id
    assert "xchecked_at" in cm
    # analytic flops/bytes formulas are left untouched
    assert cm["flops"] == "1*B"
    assert cm["bytes"] == "1000*B"
    # per-shape evidence is recorded
    ev = cm["xcheck_evidence"]
    assert ev[0]["shape_id"] == "P1"
    assert ev[0]["within_tol"] is True
    assert ev[0]["measured_bytes"] == 1010.0
    # measured via the ncu DRAM counters; FLOPs are honestly not hard-checked
    assert ev[0]["measured_via"] == "ncu_dram_bytes"
    assert ev[0]["flops_checked"] is False
    assert ev[0]["kernel_launched"] is True
    assert ev[0]["blocked"] is False


def test_update_meta_is_atomic_and_idempotent(tmp_path, monkeypatch):
    task = _make_task(
        tmp_path,
        cost_model_block={
            "flops": "1*B", "flops_class": "vector",
            "bytes": "1000*B", "bytes_resource": "hbm",
        },
        size_knobs={"B": 1},
    )
    monkeypatch.setattr(
        xc, "measure_kernel_bytes", lambda *_: (1010.0, True)
    )
    v1 = xc.cross_check_task(task, dry_run=False)
    xc.update_meta_cost_model(task.task_dir, v1)
    # re-run: overwrites in place, no stale .tmp left behind
    v2 = xc.cross_check_task(task, dry_run=False)
    xc.update_meta_cost_model(task.task_dir, v2)
    assert not (task.task_dir / "meta.yaml.tmp").exists()
    meta = yaml.safe_load((task.task_dir / "meta.yaml").read_text())
    assert meta["cost_model"]["xcheck_run_id"] == v2.run_id


def test_update_meta_refuses_dry_run_verdict(tmp_path):
    task = _make_task(
        tmp_path,
        cost_model_block={
            "flops": "1", "flops_class": "vector",
            "bytes": "1", "bytes_resource": "hbm",
        },
    )
    v = xc.cross_check_task(task, dry_run=True)
    with pytest.raises(RuntimeError, match="dry-run"):
        xc.update_meta_cost_model(task.task_dir, v)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def test_cli_requires_task_or_all():
    with pytest.raises(SystemExit):
        xc.build_parser().parse_args([])


def test_cli_dry_run_and_apply_mutually_exclusive():
    rc = xc.main(["--all", "--dry-run", "--apply"])
    assert rc == 2


def test_cli_dry_run_full_sweep_returns_zero_and_writes_nothing():
    rc = xc.main(["--all", "--dry-run"])
    assert rc == 0
