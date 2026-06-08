"""Unit 2.4 tests for ``scripts/phase_b_golden_capture.py`` -- the
path-A / path-B dual-path capture flow + ``meta.phase_b_capture``
state writes.

Covers:
  * ``_resolve_capture_path`` selects the right path from
    ``phase_a_prime.status`` (+ ``upstream_callable`` for path B).
  * ``_check_promote_precondition`` accepts SKIPPED_BY_DESIGN tasks
    when the upstream_callable block is canonical.
  * ``capture_task`` persists ``meta.phase_b_capture`` (status =
    DONE / FAILED / PENDING) on every terminal capture outcome with
    the documented field set.
  * The capture-meta block is written atomically (no .tmp residue
    after a successful run; the file is loadable mid-run).
  * The promote gate (``phase_b_capture_required == true`` ⇒ status
    must be DONE) is enforced by ``_check_phase_b_required_gate``.
  * ``_select_tasks`` additionally picks up SKIPPED_BY_DESIGN tasks
    so the defer-to-golden flow reaches them.
  * ``captured_at`` is a parseable ISO 8601 UTC timestamp.

These tests never spawn the real GPU capture child; the
``_run_capture_child`` call site is monkey-patched.
"""
from __future__ import annotations

import importlib.util
import sys
from datetime import datetime
from pathlib import Path

import pytest
import torch
import yaml
from safetensors.torch import save_file

REPO = Path(__file__).resolve().parents[2]

_spec = importlib.util.spec_from_file_location(
    "phase_b_golden_capture",
    REPO / "scripts" / "phase_b_golden_capture.py",
)
pbgc = importlib.util.module_from_spec(_spec)
sys.modules["phase_b_golden_capture"] = pbgc
_spec.loader.exec_module(pbgc)

from ascendbench.corpus.dataset import TaskDef  # noqa: E402

_PROMOTED_KERNEL = '''\
"""Synthetic promoted standalone kernel for path-A tests."""
import torch


def run(inputs: dict) -> dict:
    return {"out": inputs["x"] * 2.0}
'''

_SHAPES_YAML = """\
size_knobs: {B: 4, D: 8}
public_shapes:
  - {id: P1, name: tiny,  knobs: {B: 1, D: 4}}
  - {id: P2, name: small, knobs: {B: 2, D: 8}}
hidden_shapes:
  - {id: H1, name: hidden, knobs: {B: 3, D: 13}}
inputs:
  x: {dims: [B, D], dtype: float32, dist: randn}
"""


def _write_task(
    root: Path,
    task_id: str = "L1_99_demo",
    *,
    phase_a_prime: dict | None = None,
    phase_b_capture: dict | None = None,
    phase_b_capture_required: bool | None = None,
    oracle_simplified: str = "upstream_oracle_required",
    kernel_src: str = _PROMOTED_KERNEL,
) -> Path:
    """Materialise a synthetic task directory; return its path."""
    task_dir = root / task_id
    task_dir.mkdir(parents=True)
    meta: dict = {
        "task_id": task_id,
        "level": "L1",
        "oracle_simplified": oracle_simplified,
        "oracle": {"gpu_triton": "kernel.py::placeholder"},
        "tolerance_class": "vector_reduce",
    }
    if phase_a_prime is not None:
        meta["phase_a_prime"] = phase_a_prime
    if phase_b_capture is not None:
        meta["phase_b_capture"] = phase_b_capture
    if phase_b_capture_required is not None:
        meta["phase_b_capture_required"] = phase_b_capture_required
    (task_dir / "meta.yaml").write_text(
        yaml.safe_dump(meta, sort_keys=False, allow_unicode=True)
    )
    (task_dir / "shapes.yaml").write_text(_SHAPES_YAML)
    (task_dir / "contract.md").write_text("# synthetic\n")
    (task_dir / "kernel.py").write_text(kernel_src)
    (task_dir / "oracle_pytorch.py").write_text(
        "def run(inputs):\n    raise NotImplementedError\n"
    )
    return task_dir


def _load(task_dir: Path) -> TaskDef:
    """Build a TaskDef from a synthetic task dir."""
    from ascendbench.corpus import dataset

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


def _read_meta(task_dir: Path) -> dict:
    return yaml.safe_load((task_dir / "meta.yaml").read_text()) or {}


# --- _resolve_capture_path --------------------------------------------

def test_resolve_capture_path_promoted_kernel(tmp_path):
    """phase_a_prime.status=PASS -> path A (promoted-kernel)."""
    td = _write_task(tmp_path, phase_a_prime={"status": "PASS"})
    meta = _read_meta(td)
    path, err, callable_ = pbgc._resolve_capture_path(meta)
    assert path == pbgc.CAPTURE_PATH_PROMOTED_KERNEL
    assert err is None
    assert callable_ is None


def test_resolve_capture_path_upstream_callable(tmp_path):
    """SKIPPED_BY_DESIGN + canonical upstream_callable -> path B."""
    td = _write_task(
        tmp_path,
        phase_a_prime={
            "status": "SKIPPED_BY_DESIGN", "skip_invocation": True,
            "reason": "fixture",
        },
        phase_b_capture={
            "status": "PENDING",
            "upstream_callable": {
                "module": "vllm.model_executor.layers.layernorm",
                "function": "rms_norm",
            },
        },
    )
    meta = _read_meta(td)
    path, err, callable_ = pbgc._resolve_capture_path(meta)
    assert path == pbgc.CAPTURE_PATH_UPSTREAM_CALLABLE
    assert err is None
    assert callable_ == {
        "module": "vllm.model_executor.layers.layernorm",
        "function": "rms_norm",
    }


def test_resolve_capture_path_blocked_no_callable(tmp_path):
    """SKIPPED_BY_DESIGN without upstream_callable -> BLOCKED_no_callable."""
    td = _write_task(
        tmp_path,
        phase_a_prime={"status": "SKIPPED_BY_DESIGN", "reason": "x"},
    )
    meta = _read_meta(td)
    path, err, _ = pbgc._resolve_capture_path(meta)
    assert path == "BLOCKED_no_callable"
    assert "upstream_callable" in (err or "")


def test_resolve_capture_path_blocked_phase_a_not_ready(tmp_path):
    """Any other phase_a_prime.status -> BLOCKED_phase_a_prime_not_ready."""
    td = _write_task(tmp_path, phase_a_prime={"status": "BLOCKED"})
    meta = _read_meta(td)
    path, err, _ = pbgc._resolve_capture_path(meta)
    assert path == "BLOCKED_phase_a_prime_not_ready"
    assert "not 'PASS' or 'SKIPPED_BY_DESIGN'" in (err or "")


# --- _check_promote_precondition (path B) -----------------------------

def test_precondition_accepts_skipped_by_design_with_callable(tmp_path):
    """A SKIPPED_BY_DESIGN task with a canonical upstream_callable
    block passes the precondition without inspecting kernel.py."""
    td = _write_task(
        tmp_path,
        phase_a_prime={
            "status": "SKIPPED_BY_DESIGN", "skip_invocation": True,
            "reason": "no standalone invoker",
        },
        phase_b_capture={
            "status": "PENDING",
            "upstream_callable": {
                "module": "vllm.model_executor.layers.layernorm",
                "function": "rms_norm",
            },
        },
        # Use an UNPROMOTED kernel (no top-level run): path B must
        # not depend on kernel.py at all.
        kernel_src='def placeholder(x):\n    return x\n',
    )
    task = _load(td)
    assert pbgc._check_promote_precondition(task) is None


def test_precondition_rejects_skipped_by_design_without_callable(tmp_path):
    """SKIPPED_BY_DESIGN without upstream_callable -> precondition fails."""
    td = _write_task(
        tmp_path,
        phase_a_prime={"status": "SKIPPED_BY_DESIGN", "reason": "x"},
    )
    task = _load(td)
    err = pbgc._check_promote_precondition(task)
    assert err is not None
    assert "upstream_callable" in err


# --- meta.phase_b_capture writes --------------------------------------

def test_capture_done_writes_full_block(tmp_path, monkeypatch):
    """A clean capture writes phase_b_capture.status=DONE with
    captured_at + (when path B) the upstream_callable block."""
    td = _write_task(
        tmp_path,
        phase_a_prime={"status": "PASS"},
    )
    task = _load(td)

    def _fake_child(t, shape_ids, golden_dir, timeout_s, **_kw):
        for sid in shape_ids:
            save_file({"out": torch.ones(1)},
                      str(golden_dir / f"{sid}.safetensors"))
        return {"shapes": {sid: {"captured": True} for sid in shape_ids},
                "fatal": None}

    monkeypatch.setattr(pbgc, "_run_capture_child", _fake_child)
    result = pbgc.capture_task(task, dry_run=False)
    assert result.ok

    meta_after = _read_meta(td)
    pbc = meta_after["phase_b_capture"]
    assert pbc["status"] == pbgc.PHASE_B_CAPTURE_STATUS_DONE
    assert "captured_at" in pbc
    # captured_at must be a parseable ISO 8601 timestamp.
    parsed = datetime.fromisoformat(pbc["captured_at"].replace("Z", "+00:00"))
    assert parsed.tzinfo is not None


def test_capture_done_writes_upstream_callable_for_path_b(
        tmp_path, monkeypatch):
    """Path B: a DONE record also surfaces the upstream_callable
    block (orchestrator's authored intent)."""
    upstream = {
        "module": "vllm.model_executor.layers.layernorm",
        "function": "rms_norm",
    }
    td = _write_task(
        tmp_path,
        phase_a_prime={
            "status": "SKIPPED_BY_DESIGN", "skip_invocation": True,
            "reason": "x",
        },
        phase_b_capture={
            "status": "PENDING", "upstream_callable": upstream,
        },
        kernel_src='def placeholder(x):\n    return x\n',
    )
    task = _load(td)

    captured_paths: list[str] = []

    def _fake_child(t, shape_ids, golden_dir, timeout_s, **kw):
        captured_paths.append(kw.get("entrypoint_source", "?"))
        assert kw.get("upstream_callable") == upstream, kw
        for sid in shape_ids:
            save_file({"out": torch.ones(1)},
                      str(golden_dir / f"{sid}.safetensors"))
        return {"shapes": {sid: {"captured": True} for sid in shape_ids},
                "fatal": None}

    monkeypatch.setattr(pbgc, "_run_capture_child", _fake_child)
    result = pbgc.capture_task(task, dry_run=False)
    assert result.ok
    assert captured_paths == [pbgc.CAPTURE_PATH_UPSTREAM_CALLABLE]

    meta_after = _read_meta(td)
    pbc = meta_after["phase_b_capture"]
    assert pbc["status"] == pbgc.PHASE_B_CAPTURE_STATUS_DONE
    assert pbc["upstream_callable"] == upstream


def test_capture_failed_writes_reason(tmp_path, monkeypatch):
    """A child failure persists status=FAILED with a non-empty
    reason capped at 200 chars."""
    td = _write_task(tmp_path, phase_a_prime={"status": "PASS"})
    task = _load(td)

    def _fake_child(t, shape_ids, golden_dir, timeout_s, **_kw):
        return {
            "shapes": {},
            "fatal": "import-failed: " + ("x" * 500),
        }

    monkeypatch.setattr(pbgc, "_run_capture_child", _fake_child)
    result = pbgc.capture_task(task, dry_run=False)
    assert not result.ok

    meta_after = _read_meta(td)
    pbc = meta_after["phase_b_capture"]
    assert pbc["status"] == pbgc.PHASE_B_CAPTURE_STATUS_FAILED
    assert isinstance(pbc.get("reason"), str) and pbc["reason"]
    assert len(pbc["reason"]) <= 200


def test_capture_precondition_failure_writes_failed_state(
        tmp_path, monkeypatch):
    """A precondition failure (unpromoted task) records status=FAILED
    so the orchestrator backlog sweep sees it (rule a -> BACKLOG_
    UNEXPECTED)."""
    td = _write_task(tmp_path, phase_a_prime={"status": "BLOCKED"})
    task = _load(td)

    def _fake_child(*a, **k):
        raise AssertionError("child must not run on a failed precondition")

    monkeypatch.setattr(pbgc, "_run_capture_child", _fake_child)
    pbgc.capture_task(task, dry_run=False)
    meta_after = _read_meta(td)
    pbc = meta_after["phase_b_capture"]
    assert pbc["status"] == pbgc.PHASE_B_CAPTURE_STATUS_FAILED
    assert pbc["reason"]


def test_meta_atomic_write_leaves_no_tmp_residue(tmp_path, monkeypatch):
    """The phase_b_capture write goes through atomic tmp+replace --
    no ``meta.yaml.tmp`` is left behind after a clean run."""
    td = _write_task(tmp_path, phase_a_prime={"status": "PASS"})
    task = _load(td)

    def _fake_child(t, shape_ids, golden_dir, timeout_s, **_kw):
        for sid in shape_ids:
            save_file({"out": torch.ones(1)},
                      str(golden_dir / f"{sid}.safetensors"))
        return {"shapes": {sid: {"captured": True} for sid in shape_ids},
                "fatal": None}

    monkeypatch.setattr(pbgc, "_run_capture_child", _fake_child)
    pbgc.capture_task(task, dry_run=False)
    leftovers = list(td.glob("meta.yaml.tmp*")) + list(td.glob("*.yaml.tmp"))
    assert not leftovers, f"atomic write left tmp residue: {leftovers}"
    # And the canonical meta.yaml is still well-formed YAML.
    yaml.safe_load((td / "meta.yaml").read_text())


# --- _check_phase_b_required_gate -------------------------------------

def test_promote_gate_blocks_when_required_and_pending(tmp_path):
    """phase_b_capture_required=true + status=PENDING -> gate blocks
    (the orchestrator must keep the task in BACKLOG_PENDING_GOLDEN)."""
    td = _write_task(
        tmp_path,
        phase_a_prime={"status": "SKIPPED_BY_DESIGN", "reason": "x"},
        phase_b_capture={
            "status": "PENDING",
            "upstream_callable": {"module": "m.x", "function": "f"},
        },
        phase_b_capture_required=True,
    )
    meta = _read_meta(td)
    err = pbgc._check_phase_b_required_gate(meta)
    assert err is not None
    assert "BACKLOG_PENDING_GOLDEN" in err


def test_promote_gate_allows_when_required_and_done(tmp_path):
    """phase_b_capture_required=true + status=DONE -> gate allows."""
    td = _write_task(
        tmp_path,
        phase_a_prime={"status": "SKIPPED_BY_DESIGN", "reason": "x"},
        phase_b_capture={
            "status": "DONE",
            "upstream_callable": {"module": "m.x", "function": "f"},
            "captured_at": "2026-05-23T12:00:00Z",
        },
        phase_b_capture_required=True,
    )
    meta = _read_meta(td)
    assert pbgc._check_phase_b_required_gate(meta) is None


def test_promote_gate_no_op_when_not_required(tmp_path):
    """phase_b_capture_required absent / false -> gate is no-op."""
    td = _write_task(tmp_path, phase_a_prime={"status": "PASS"})
    meta = _read_meta(td)
    assert pbgc._check_phase_b_required_gate(meta) is None


# --- _select_tasks SKIPPED_BY_DESIGN additive inclusion ---------------

def test_select_task_accepts_skipped_by_design_non_uor(tmp_path,
                                                       monkeypatch):
    """``_select_tasks(task_id=<SKIPPED_BY_DESIGN task id>)`` accepts
    even when the task is not score-eligible (the orchestrator has
    explicitly marked the task as defer-to-golden so it must reach
    capture). This is the additive inclusion path; existing UOR +
    score-eligible filter is preserved for non-SKIPPED tasks."""
    td = _write_task(
        tmp_path,
        task_id="L1_50_demo",
        phase_a_prime={
            "status": "SKIPPED_BY_DESIGN", "skip_invocation": True,
            "reason": "fixture",
        },
        phase_b_capture={
            "status": "PENDING",
            "upstream_callable": {"module": "m.x", "function": "f"},
        },
    )
    fake_task = _load(td)

    # Stub the corpus loaders so ``_select_tasks`` sees our task.
    monkeypatch.setattr(pbgc, "load_task", lambda tid: fake_task)
    monkeypatch.setattr(pbgc.eligibility, "is_upstream_oracle_required",
                        lambda t: True)
    monkeypatch.setattr(pbgc.eligibility, "is_score_eligible",
                        lambda t: False)

    out = pbgc._select_tasks("L1_50_demo")
    assert [t.task_id for t in out] == ["L1_50_demo"]


# --- captured_at format -----------------------------------------------

def test_iso8601_captured_at_format(tmp_path, monkeypatch):
    """captured_at written by capture is parseable with
    ``datetime.fromisoformat`` (after rewriting the Z suffix as
    ``+00:00`` per the Python stdlib convention)."""
    td = _write_task(tmp_path, phase_a_prime={"status": "PASS"})
    task = _load(td)

    def _fake_child(t, shape_ids, golden_dir, timeout_s, **_kw):
        for sid in shape_ids:
            save_file({"out": torch.ones(1)},
                      str(golden_dir / f"{sid}.safetensors"))
        return {"shapes": {sid: {"captured": True} for sid in shape_ids},
                "fatal": None}

    monkeypatch.setattr(pbgc, "_run_capture_child", _fake_child)
    pbgc.capture_task(task, dry_run=False)
    pbc = _read_meta(td)["phase_b_capture"]
    ts = pbc["captured_at"]
    assert ts.endswith("Z")
    parsed = datetime.fromisoformat(ts.replace("Z", "+00:00"))
    assert parsed.year >= 2025  # sanity range
    assert parsed.tzinfo is not None
    assert parsed.utcoffset().total_seconds() == 0


# --- vendored upstream sys.path constants -----------------------------

def test_vendored_upstream_sys_paths_includes_python_root():
    """The path-B child's sys.path additions MUST include the inner
    ``infer/sglang/python`` root (matches _pa_oracle_check_runner.py
    :4-6), not the parent ``infer/sglang``."""
    extras = pbgc._vendored_upstream_sys_paths()
    assert any(p.endswith("/infer/vllm") for p in extras), extras
    assert any(p.endswith("/infer/sglang/python") for p in extras), extras
    # The parent infer/sglang root by itself is NOT canonical.
    assert not any(p.endswith("/infer/sglang") and not p.endswith(
        "/infer/sglang/python") for p in extras), extras


# --- runtime error reason capping -------------------------------------

def test_capture_failed_short_reason_persists_verbatim(
        tmp_path, monkeypatch):
    """A short fatal reason is persisted verbatim (no truncation)."""
    td = _write_task(tmp_path, phase_a_prime={"status": "PASS"})
    task = _load(td)
    short = "RuntimeError: synthetic kernel fail"

    def _fake_child(t, shape_ids, golden_dir, timeout_s, **_kw):
        return {"shapes": {}, "fatal": short}

    monkeypatch.setattr(pbgc, "_run_capture_child", _fake_child)
    pbgc.capture_task(task, dry_run=False)
    pbc = _read_meta(td)["phase_b_capture"]
    assert pbc["status"] == pbgc.PHASE_B_CAPTURE_STATUS_FAILED
    assert short in pbc["reason"]


# --- _run_capture_child validation ------------------------------------

def test_run_capture_child_rejects_invalid_entrypoint_source(tmp_path):
    """_run_capture_child raises ValueError on an unknown entrypoint
    source (defensive guard against orchestrator typos)."""
    td = _write_task(tmp_path, phase_a_prime={"status": "PASS"})
    task = _load(td)
    with pytest.raises(ValueError, match="invalid entrypoint_source"):
        pbgc._run_capture_child(
            task, ["P1"], td / "golden", 60,
            entrypoint_source="garbage",
        )


def test_run_capture_child_rejects_path_b_without_callable(tmp_path):
    """_run_capture_child raises when path B is selected but no
    upstream_callable is provided."""
    td = _write_task(tmp_path, phase_a_prime={"status": "PASS"})
    task = _load(td)
    with pytest.raises(ValueError, match="upstream-callable path requires"):
        pbgc._run_capture_child(
            task, ["P1"], td / "golden", 60,
            entrypoint_source=pbgc.CAPTURE_PATH_UPSTREAM_CALLABLE,
            upstream_callable=None,
        )
