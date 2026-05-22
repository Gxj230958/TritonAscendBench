"""Tests for scripts/phase_b_golden_capture.py -- offline golden capture
for upstream_oracle_required tasks (Phase B', GPU host).

Covers the dev-machine-runnable surface: the 18-task enumeration (the
score-eligible upstream_oracle_required set), the public + hidden shape
enumeration, the golden artifact path / naming convention, the Phase A'
promote precondition gate, and the --dry-run
path. The actual GPU capture-child run is exercised through a mocked
``_run_capture_child`` (the dev machine has no CUDA -- the real GPU run
is the user's responsibility on a GPU host).

These tests never skip: every path here is host-independent.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import torch
import yaml
from safetensors.torch import load_file, save_file

REPO = Path(__file__).resolve().parents[2]

# Load the script as a file-local module instance (it lives under
# scripts/, not the importable package).
_spec = importlib.util.spec_from_file_location(
    "phase_b_golden_capture",
    REPO / "scripts" / "phase_b_golden_capture.py",
)
pbgc = importlib.util.module_from_spec(_spec)
sys.modules["phase_b_golden_capture"] = pbgc
_spec.loader.exec_module(pbgc)

from triton_ascend_bench.dataset import TaskDef  # noqa: E402

# --- fixtures: synthetic on-disk tasks --------------------------------

_PROMOTED_KERNEL = '''\
"""Synthetic promoted standalone kernel for tests."""
import torch


def run(inputs: dict) -> dict:
    x = inputs["x"]
    return {"out": x * 2.0}
'''

_UNPROMOTED_KERNEL = '''\
"""Synthetic pre-promote kernel: no top-level run()."""
import torch


def placeholder_kernel(x):
    return x
'''

_SHAPES_YAML = """\
size_knobs: {B: 4, D: 8}
public_shapes:
  - {id: P1, name: tiny,  knobs: {B: 1, D: 4}}
  - {id: P2, name: small, knobs: {B: 2, D: 8}}
hidden_shapes:
  - {id: H1, name: h_odd, knobs: {B: 3, D: 13}}
inputs:
  x: {dims: [B, D], dtype: float32, dist: randn}
"""


def _write_task(
    root: Path,
    task_id: str,
    *,
    oracle_simplified: str = "upstream_oracle_required",
    phase_a_status: str | None = "PASS",
    kernel_src: str = _PROMOTED_KERNEL,
) -> Path:
    """Materialise a synthetic task directory and return its path."""
    task_dir = root / task_id
    task_dir.mkdir(parents=True)
    meta: dict = {
        "task_id": task_id,
        "level": "L1",
        "oracle_simplified": oracle_simplified,
        "oracle": {"gpu_triton": "kernel.py::placeholder"},
        "tolerance_class": "vector_reduce",
        "tolerance": {"class": "vector_reduce"},
        "upstream": {"commit_sha": "deadbeef"},
    }
    if phase_a_status is not None:
        meta["phase_a_prime"] = {"status": phase_a_status}
    (task_dir / "meta.yaml").write_text(yaml.safe_dump(meta, sort_keys=False))
    (task_dir / "shapes.yaml").write_text(_SHAPES_YAML)
    (task_dir / "contract.md").write_text("# synthetic\n")
    (task_dir / "kernel.py").write_text(kernel_src)
    (task_dir / "oracle_pytorch.py").write_text(
        "def run(inputs):\n    raise NotImplementedError\n")
    return task_dir


def _load(task_dir: Path) -> TaskDef:
    """Load a synthetic task dir into a TaskDef (bypasses level scan)."""
    from triton_ascend_bench import dataset

    meta = yaml.safe_load((task_dir / "meta.yaml").read_text())
    return TaskDef(
        task_id=task_dir.name,
        level=1,
        task_dir=task_dir,
        meta=meta,
        public_shapes=[],
        contract_md=(task_dir / "contract.md").read_text(),
        kernel_py=(task_dir / "kernel.py").read_text(),
        kernel_py_sha256=dataset._sha256_text(
            (task_dir / "kernel.py").read_text()),
    )


# --- shape enumeration ------------------------------------------------

def test_enumerate_shapes_covers_public_and_hidden(tmp_path):
    """_enumerate_shapes returns the public + hidden eval shape split."""
    task = _load(_write_task(tmp_path, "L1_99_demo"))
    public, hidden = pbgc._enumerate_shapes(task)
    assert public == ["P1", "P2"]
    assert hidden == ["H1"]


def test_task_capture_result_all_shape_ids_order():
    """all_shape_ids is public ids then hidden ids, in order."""
    r = pbgc.TaskCaptureResult(
        task_id="t", public_shapes=["P1", "P2"], hidden_shapes=["H1"])
    assert r.all_shape_ids == ["P1", "P2", "H1"]


# --- 18-task enumeration (real repo) ----------------------------------

def test_select_all_finds_exactly_the_score_eligible_uor_set():
    """--all selects exactly the SCORE-ELIGIBLE upstream_oracle_required
    tasks (18 today), derived from trusted repo meta -- the same
    score-eligible filter gpu_pipeline step 20 (phase_a_cost_xcheck)
    applies. A non-score-eligible upstream_oracle_required task
    (L1_21_gumbel_sample: stub oracle / placeholder kernel) has no
    capturable offline golden and is excluded from the capture sweep."""
    from triton_ascend_bench import eligibility
    from triton_ascend_bench.dataset import iter_all_tasks

    all_tasks = list(iter_all_tasks())
    selected = pbgc._select_tasks(None)
    expected = {
        t.task_id for t in all_tasks
        if eligibility.is_upstream_oracle_required(t)
        and eligibility.is_score_eligible(t)
    }
    assert {t.task_id for t in selected} == expected
    # Every selected task is upstream_oracle_required AND score-eligible.
    for t in selected:
        assert eligibility.is_upstream_oracle_required(t)
        assert eligibility.is_score_eligible(t)
    # The corpus has 19 upstream_oracle_required tasks; exactly one
    # (L1_21_gumbel_sample) is non-score-eligible and excluded -> 18.
    assert len(selected) == 18
    uor = eligibility.compute_upstream_oracle_required(all_tasks)
    assert len(uor) == 19
    excluded = uor - {t.task_id for t in selected}
    assert excluded == {"L1_21_gumbel_sample"}


def test_select_all_excludes_non_score_eligible_uor():
    """L1_21_gumbel_sample is upstream_oracle_required but non-score-
    eligible (stub oracle) -- it must NOT appear in the capture sweep."""
    selected_ids = {t.task_id for t in pbgc._select_tasks(None)}
    assert "L1_21_gumbel_sample" not in selected_ids


def test_select_task_rejects_non_score_eligible_uor():
    """--task on a non-score-eligible upstream_oracle_required task is an
    honest hard error: a stub-oracle task has no capturable golden."""
    from triton_ascend_bench import eligibility
    from triton_ascend_bench.dataset import iter_all_tasks

    non_se_uor = next(
        (t for t in iter_all_tasks()
         if eligibility.is_upstream_oracle_required(t)
         and not eligibility.is_score_eligible(t)),
        None,
    )
    assert non_se_uor is not None, (
        "expected at least one non-score-eligible upstream_oracle_required "
        "task (L1_21_gumbel_sample) in the corpus")
    with pytest.raises(ValueError, match="NOT score-eligible"):
        pbgc._select_tasks(non_se_uor.task_id)


def test_select_all_is_sorted():
    """--all returns the tasks sorted by task_id for stable reporting."""
    selected = pbgc._select_tasks(None)
    ids = [t.task_id for t in selected]
    assert ids == sorted(ids)


def test_select_task_rejects_non_upstream_oracle_required():
    """--task on a PyTorch-oracle task is a category error: such a task's
    golden is generated live, never pre-stored."""
    from triton_ascend_bench import eligibility
    from triton_ascend_bench.dataset import iter_all_tasks

    non_uor = next(
        t for t in iter_all_tasks()
        if not eligibility.is_upstream_oracle_required(t))
    with pytest.raises(ValueError, match="not an upstream_oracle_required"):
        pbgc._select_tasks(non_uor.task_id)


def test_select_task_accepts_upstream_oracle_required():
    """--task on a real upstream_oracle_required task resolves to it."""
    from triton_ascend_bench import eligibility
    from triton_ascend_bench.dataset import iter_all_tasks

    uor = next(
        t for t in iter_all_tasks()
        if eligibility.is_upstream_oracle_required(t))
    selected = pbgc._select_tasks(uor.task_id)
    assert [t.task_id for t in selected] == [uor.task_id]


# --- promote precondition --------------------------------------------

def test_precondition_passes_for_promoted_kernel(tmp_path):
    """A PASS phase_a_prime status + a top-level run() => precondition ok."""
    task = _load(_write_task(tmp_path, "L1_99_demo"))
    assert pbgc._check_promote_precondition(task) is None


def test_precondition_fails_when_not_promoted(tmp_path):
    """A missing / non-PASS phase_a_prime.status is an honest hard error
    -- golden capture must run AFTER Phase A' promote (step 10)."""
    task = _load(_write_task(tmp_path, "L1_99_demo", phase_a_status=None))
    err = pbgc._check_promote_precondition(task)
    assert err is not None and "not 'PASS'" in err

    task2 = _load(_write_task(tmp_path, "L1_98_demo",
                              phase_a_status="BLOCKED"))
    err2 = pbgc._check_promote_precondition(task2)
    assert err2 is not None and "not 'PASS'" in err2


def test_precondition_fails_when_kernel_lacks_run(tmp_path):
    """A PASS status but a kernel.py with no top-level run() is a
    meta/kernel mismatch -- caught before the GPU run."""
    task = _load(_write_task(tmp_path, "L1_99_demo",
                             kernel_src=_UNPROMOTED_KERNEL))
    err = pbgc._check_promote_precondition(task)
    assert err is not None and "no top-level 'run'" in err


# --- dry-run ----------------------------------------------------------

def test_dry_run_enumerates_without_promote_gate_or_write(tmp_path):
    """--dry-run enumerates shapes with no promote gate, no GPU, no write
    -- it works even for an unpromoted task and creates no golden/ dir."""
    task = _load(_write_task(tmp_path, "L1_99_demo", phase_a_status=None,
                             kernel_src=_UNPROMOTED_KERNEL))
    result = pbgc.capture_task(task, dry_run=True)
    assert result.precondition_error is None
    assert result.all_shape_ids == ["P1", "P2", "H1"]
    assert all(not s.captured and not s.skipped for s in result.shapes)
    assert not (task.task_dir / "golden").exists()


def test_dry_run_all_18_tasks_enumerates_162_shapes():
    """The documented dev-machine verification: --dry-run --all enumerates
    the 18 score-eligible upstream_oracle_required tasks and their full
    eval shape set."""
    tasks = pbgc._select_tasks(None)
    results = [pbgc.capture_task(t, dry_run=True) for t in tasks]
    assert len(results) == 18
    assert all(r.precondition_error is None for r in results)
    total = sum(len(r.all_shape_ids) for r in results)
    # 162 = the score-eligible upstream_oracle_required eval shape count
    # (the full 19-task corpus has 166; L1_21_gumbel_sample's 4 shapes
    # are excluded with the task).
    assert total == 162
    # Every task has both public and hidden eval shapes.
    for r in results:
        assert r.public_shapes, f"{r.task_id} has no public shapes"
        assert r.hidden_shapes, f"{r.task_id} has no hidden shapes"


# --- GPU capture path (mocked child) ----------------------------------

def test_capture_writes_golden_with_correct_path_and_naming(
        tmp_path, monkeypatch):
    """capture_task writes golden/<shape_id>.safetensors -- the exact
    layout eval._resolve_golden / eligibility.golden_complete read."""
    task = _load(_write_task(tmp_path, "L1_99_demo"))

    def _fake_child(t, shape_ids, golden_dir, timeout_s):
        # Mimic the real child: write a flat tensor-dict per shape.
        for sid in shape_ids:
            save_file({"out": torch.ones(3)},
                      str(golden_dir / f"{sid}.safetensors"))
        return {"shapes": {sid: {"captured": True, "keys": ["out"]}
                           for sid in shape_ids},
                "fatal": None}

    monkeypatch.setattr(pbgc, "_run_capture_child", _fake_child)
    result = pbgc.capture_task(task, dry_run=False)

    assert result.ok
    golden_dir = task.task_dir / "golden"
    for sid in ("P1", "P2", "H1"):
        artifact = golden_dir / f"{sid}.safetensors"
        assert artifact.is_file()
        # Loadable flat tensor dict -- the golden_artifact_usable contract.
        loaded = load_file(str(artifact))
        assert "out" in loaded and torch.is_tensor(loaded["out"])


def test_capture_result_satisfies_eligibility_golden_complete(
        tmp_path, monkeypatch):
    """A fully-captured task satisfies eligibility.golden_complete -- the
    runtime gate eval and validate_tasks both consume."""
    from triton_ascend_bench import eligibility

    task = _load(_write_task(tmp_path, "L1_99_demo"))

    def _fake_child(t, shape_ids, golden_dir, timeout_s):
        for sid in shape_ids:
            save_file({"out": torch.ones(2)},
                      str(golden_dir / f"{sid}.safetensors"))
        return {"shapes": {sid: {"captured": True} for sid in shape_ids},
                "fatal": None}

    monkeypatch.setattr(pbgc, "_run_capture_child", _fake_child)
    pbgc.capture_task(task, dry_run=False)
    assert eligibility.golden_complete(task)


def test_capture_is_idempotent_skips_existing(tmp_path, monkeypatch):
    """A shape with an already-usable artifact is skipped (not re-run)."""
    task = _load(_write_task(tmp_path, "L1_99_demo"))
    golden_dir = task.task_dir / "golden"
    golden_dir.mkdir()
    # Pre-seed P1's golden.
    save_file({"out": torch.ones(1)}, str(golden_dir / "P1.safetensors"))

    captured_shapes: list[str] = []

    def _fake_child(t, shape_ids, golden_dir, timeout_s):
        captured_shapes.extend(shape_ids)
        for sid in shape_ids:
            save_file({"out": torch.ones(1)},
                      str(golden_dir / f"{sid}.safetensors"))
        return {"shapes": {sid: {"captured": True} for sid in shape_ids},
                "fatal": None}

    monkeypatch.setattr(pbgc, "_run_capture_child", _fake_child)
    result = pbgc.capture_task(task, dry_run=False)

    # P1 was skipped; only P2/H1 went to the child.
    assert sorted(captured_shapes) == ["H1", "P2"]
    skipped = {s.shape_id for s in result.shapes if s.skipped}
    assert skipped == {"P1"}


def test_capture_force_recaptures_existing(tmp_path, monkeypatch):
    """--force re-captures every shape even when artifacts already exist."""
    task = _load(_write_task(tmp_path, "L1_99_demo"))
    golden_dir = task.task_dir / "golden"
    golden_dir.mkdir()
    save_file({"out": torch.ones(1)}, str(golden_dir / "P1.safetensors"))

    captured_shapes: list[str] = []

    def _fake_child(t, shape_ids, golden_dir, timeout_s):
        captured_shapes.extend(shape_ids)
        for sid in shape_ids:
            save_file({"out": torch.ones(1)},
                      str(golden_dir / f"{sid}.safetensors"))
        return {"shapes": {sid: {"captured": True} for sid in shape_ids},
                "fatal": None}

    monkeypatch.setattr(pbgc, "_run_capture_child", _fake_child)
    result = pbgc.capture_task(task, dry_run=False, force=True)
    assert sorted(captured_shapes) == ["H1", "P1", "P2"]
    assert all(s.captured for s in result.shapes)


def test_capture_honours_promote_gate_before_gpu_run(tmp_path, monkeypatch):
    """A non-promoted task fails the precondition and never reaches the
    GPU child -- no fabricated golden."""
    task = _load(_write_task(tmp_path, "L1_99_demo", phase_a_status=None))

    def _fake_child(*a, **k):
        raise AssertionError("child must not run for an unpromoted task")

    monkeypatch.setattr(pbgc, "_run_capture_child", _fake_child)
    result = pbgc.capture_task(task, dry_run=False)
    assert not result.ok
    assert result.precondition_error is not None
    assert not (task.task_dir / "golden").exists() or not list(
        (task.task_dir / "golden").glob("*.safetensors"))


def test_capture_surfaces_child_fatal_as_honest_error(tmp_path, monkeypatch):
    """A whole-child failure (e.g. no CUDA) surfaces as an honest
    precondition error -- never a silently-clean run."""
    task = _load(_write_task(tmp_path, "L1_99_demo"))

    def _fake_child(t, shape_ids, golden_dir, timeout_s):
        return {"shapes": {}, "fatal": "torch.cuda.is_available() is False"}

    monkeypatch.setattr(pbgc, "_run_capture_child", _fake_child)
    result = pbgc.capture_task(task, dry_run=False)
    assert not result.ok
    assert "cuda" in (result.precondition_error or "").lower()


def test_capture_surfaces_per_shape_error(tmp_path, monkeypatch):
    """A per-shape kernel failure is reported per shape, others still
    capture -- a partial failure is honest, not fatal."""
    task = _load(_write_task(tmp_path, "L1_99_demo"))

    def _fake_child(t, shape_ids, golden_dir, timeout_s):
        shapes = {}
        for sid in shape_ids:
            if sid == "H1":
                shapes[sid] = {"captured": False,
                               "error": "RuntimeError: synthetic kernel fail"}
            else:
                save_file({"out": torch.ones(1)},
                          str(golden_dir / f"{sid}.safetensors"))
                shapes[sid] = {"captured": True}
        return {"shapes": shapes, "fatal": None}

    monkeypatch.setattr(pbgc, "_run_capture_child", _fake_child)
    result = pbgc.capture_task(task, dry_run=False)
    assert not result.ok
    errs = {s.shape_id: s.error for s in result.shapes if s.error}
    assert set(errs) == {"H1"}
    assert "synthetic kernel fail" in errs["H1"]


# --- CLI / constants --------------------------------------------------

def test_golden_seed_matches_eval_correctness_seed():
    """The capture seed MUST be 0 -- byte-identical to eval.evaluate's
    correctness_inputs (build_inputs seed=0) and the offline-golden
    cheat-path closure's fixed seed-0 input."""
    assert pbgc.GOLDEN_SEED == 0


def test_cli_requires_all_or_task():
    """The CLI mandates exactly one of --all / --task."""
    with pytest.raises(SystemExit):
        pbgc.build_parser().parse_args([])


def test_cli_all_and_task_mutually_exclusive():
    """--all and --task cannot be combined."""
    with pytest.raises(SystemExit):
        pbgc.build_parser().parse_args(["--all", "--task", "L1_99"])


def test_cli_dry_run_and_force_flags_parse():
    """--dry-run / --force / --timeout parse onto the namespace."""
    ns = pbgc.build_parser().parse_args(
        ["--all", "--dry-run", "--force", "--timeout", "120"])
    assert ns.all and ns.dry_run and ns.force and ns.timeout == 120


def test_main_dry_run_all_returns_zero(capsys):
    """`main(['--all', '--dry-run'])` enumerates cleanly and exits 0."""
    rc = pbgc.main(["--all", "--dry-run"])
    assert rc == 0
