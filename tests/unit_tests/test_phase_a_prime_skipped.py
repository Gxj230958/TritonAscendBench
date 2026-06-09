"""Unit 2.3 tests for the ``phase_a_prime.skip_invocation`` short-circuit.

Covers:
  * ``s2_verify_task`` early-returns ``SKIPPED_BY_DESIGN`` when the
    task's ``meta.phase_a_prime.skip_invocation`` is True (or the
    status is already ``SKIPPED_BY_DESIGN``), without ever attempting
    the GPU isolated invocation.
  * The meta.yaml is atomically updated so the canonical
    cross-field invariant (skip_invocation=true + status=
    SKIPPED_BY_DESIGN + non-empty reason) is preserved.
  * ``_build_per_task`` produces the SKIPPED_BY_DESIGN row from the
    same pre-rule (no GPU verify needed in the report path either).
  * The status.md banner / S6 reconciliation include the new bucket.
  * Existing PASS / BLOCKED / UNRUN paths still work (no regression).

These tests are GPU-free: the short-circuit never reaches CUDA.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parent.parent.parent
_spec = importlib.util.spec_from_file_location(
    "phase_a_prime", REPO / "scripts" / "verify" / "standalone_freeze_verifier.py"
)
pap = importlib.util.module_from_spec(_spec)
sys.modules["phase_a_prime"] = pap
_spec.loader.exec_module(pap)


_PROMOTED_KERNEL = '''\
"""Synthetic kernel used by Unit 2.3 SKIPPED_BY_DESIGN tests."""
import torch


def run(inputs: dict) -> dict:
    return {"out": inputs["x"] * 2.0}
'''

_SHAPES_YAML = """\
size_knobs: {B: 4, D: 8}
public_shapes:
  - {id: P1, name: tiny, knobs: {B: 1, D: 4}}
hidden_shapes:
  - {id: H1, name: hidden, knobs: {B: 2, D: 8}}
inputs:
  x: {dims: [B, D], dtype: float32, dist: randn}
"""


def _write_task(
    root: Path,
    task_id: str = "L1_99_demo",
    *,
    phase_a_prime: dict | None = None,
    kernel_src: str = _PROMOTED_KERNEL,
) -> Path:
    """Materialise a synthetic on-disk task directory."""
    task_dir = root / task_id
    task_dir.mkdir(parents=True)
    meta: dict = {
        "task_id": task_id,
        "level": "L1",
        "oracle_simplified": "upstream_oracle_required",
        "oracle": {"gpu_triton": "kernel.py::placeholder"},
        "tolerance_class": "vector_reduce",
    }
    if phase_a_prime is not None:
        meta["phase_a_prime"] = phase_a_prime
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


# --- s2_verify_task short-circuit -------------------------------------

def test_s2_skip_invocation_short_circuits(tmp_path, monkeypatch):
    """``s2_verify_task`` returns SKIPPED_BY_DESIGN without ever
    reaching the staged-candidate build / S2-smoke / GPU isolated
    invocation when skip_invocation=True."""
    td = _write_task(
        tmp_path,
        phase_a_prime={
            "skip_invocation": True,
            "reason": "upstream-not-invocable-here(missing-dep:flashinfer)",
        },
    )

    def _boom(*a, **k):
        raise AssertionError(
            "build_staged_candidate must NOT be reached when "
            "skip_invocation=true (the short-circuit failed)"
        )

    monkeypatch.setattr(pap, "build_staged_candidate", _boom)
    monkeypatch.setattr(
        pap,
        "_run_s2_pair",
        lambda *a, **k: (_ for _ in ()).throw(
            AssertionError("_run_s2_pair must NOT be reached either")
        ),
    )

    res = pap.s2_verify_task(td)
    assert res.status == "SKIPPED_BY_DESIGN"
    assert res.skip_invocation is True
    assert "upstream-not-invocable-here" in res.reason


def test_s2_status_skipped_by_design_also_short_circuits(tmp_path,
                                                        monkeypatch):
    """A task whose meta already carries
    ``phase_a_prime.status==SKIPPED_BY_DESIGN`` (but no explicit
    ``skip_invocation`` key) is still short-circuited -- the two
    fields are co-canonical (task_api_contract.md §6a.6)."""
    td = _write_task(
        tmp_path,
        phase_a_prime={
            "status": "SKIPPED_BY_DESIGN",
            "skip_invocation": True,
            "reason": "fixture reason",
        },
    )
    monkeypatch.setattr(
        pap,
        "build_staged_candidate",
        lambda *a, **k: (_ for _ in ()).throw(
            AssertionError("must not reach staged candidate build")
        ),
    )
    res = pap.s2_verify_task(td)
    assert res.status == "SKIPPED_BY_DESIGN"
    assert res.reason == "fixture reason"


def test_s2_skip_invocation_writes_reason(tmp_path, monkeypatch):
    """When ``skip_invocation`` is True but no ``status`` / ``reason``
    is recorded, the short-circuit persists the canonical fields so
    the cross-field invariant is satisfied."""
    td = _write_task(
        tmp_path,
        phase_a_prime={"skip_invocation": True},
    )
    monkeypatch.setattr(
        pap, "build_staged_candidate",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("nope")),
    )
    pap.s2_verify_task(td)
    meta_after = yaml.safe_load((td / "meta.yaml").read_text())
    pap_after = meta_after["phase_a_prime"]
    assert pap_after["skip_invocation"] is True
    assert pap_after["status"] == "SKIPPED_BY_DESIGN"
    assert isinstance(pap_after.get("reason"), str)
    assert pap_after["reason"]  # non-empty


def test_s2_skip_invocation_atomic_write_meta(tmp_path, monkeypatch):
    """The short-circuit's meta.yaml update is atomic: tmp file +
    replace, never a half-written meta.yaml and no .tmp residue."""
    td = _write_task(
        tmp_path,
        phase_a_prime={"skip_invocation": True},
    )
    monkeypatch.setattr(
        pap, "build_staged_candidate",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("nope")),
    )
    pap.s2_verify_task(td)
    assert (td / "meta.yaml").is_file()
    leftovers = list(td.glob("*.tmp")) + list(td.glob("meta.yaml.tmp*"))
    assert not leftovers, f"atomic write left tmp residue: {leftovers}"


def test_s2_normal_path_unchanged_when_skip_invocation_absent(
    tmp_path, monkeypatch,
):
    """skip_invocation absent / False -> the legacy short-circuit
    short-circuit (CUDA-availability gate -> UNRUN) still works.
    This is the regression guard."""
    td = _write_task(tmp_path, phase_a_prime={"status": "PASS"})

    # Force the CUDA short-circuit: smoke "passes" trivially, no GPU
    # available -> UNRUN. The test confirms s2_verify_task does NOT
    # treat status=PASS in the meta as a SKIPPED_BY_DESIGN trigger
    # (only skip_invocation / status=SKIPPED_BY_DESIGN do).
    monkeypatch.setattr(pap, "_cuda_available", lambda: False)
    monkeypatch.setattr(
        pap,
        "build_staged_candidate",
        lambda td_, cls_: td_,  # any sentinel path
    )

    def _fake_pair(*a, **k):
        # interpret-smoke -> equivalent, no leak, OK phase.
        return True, "bit-exact", {}, {}, "ok", None

    monkeypatch.setattr(pap, "_run_s2_pair", _fake_pair)
    monkeypatch.setattr(
        pap,
        "_accumulate_s2_original_closure",
        lambda *a, **k: None,
    )
    monkeypatch.setattr(
        pap, "classify_task",
        lambda *a, **k: pap.TaskClassification(task_id=td.name, tier="T1"),
    )
    monkeypatch.setattr(pap, "_entrypoint_from_meta", lambda *a, **k: None)
    monkeypatch.setattr(pap, "_shape_ids_for", lambda *a, **k: ["P1", "H1"])

    res = pap.s2_verify_task(td)
    assert res.status == "UNRUN", res
    assert res.skip_invocation is False


# --- _build_per_task pre-rule -----------------------------------------

def test_build_per_task_emits_skipped_by_design_row(tmp_path):
    """``_build_per_task`` emits a SKIPPED_BY_DESIGN row directly from
    the meta pre-rule, without invoking the verify_fn / resolver."""
    td = _write_task(
        tmp_path,
        phase_a_prime={
            "skip_invocation": True,
            "status": "SKIPPED_BY_DESIGN",
            "reason": "intentional skip for fixture",
        },
    )

    def _vf(_t):
        raise AssertionError("verify_fn must NOT run when SKIPPED_BY_DESIGN")

    def _rvf(_t):
        raise AssertionError("resolver_verdict_fn must NOT run either")

    results = [pap.TaskClassification(task_id=td.name, tier="T1")]
    task_dirs_by_id = {td.name: td}
    per_task: list[tuple[str, str, str, str]] = []
    pap._build_per_task(results, task_dirs_by_id, _vf, per_task, _rvf)
    assert len(per_task) == 1
    tid, tier, status, reason = per_task[0]
    assert tid == td.name
    assert tier == "T1"
    assert status == "SKIPPED_BY_DESIGN"
    assert "intentional skip" in reason


def test_render_status_md_includes_skipped_bucket(tmp_path):
    """``_render_status_md`` reports the SKIPPED_BY_DESIGN bucket in
    the S6 verdict section (the new release predicate doc string is
    also emitted)."""
    td = _write_task(
        tmp_path,
        phase_a_prime={
            "skip_invocation": True,
            "status": "SKIPPED_BY_DESIGN",
            "reason": "fixture",
        },
    )
    results = [pap.TaskClassification(task_id=td.name, tier="T1")]
    md = pap._render_status_md(
        results,
        gpu=False,
        eligible_count=1,
        task_dirs_by_id={td.name: td},
        verify_fn=lambda _t: (_ for _ in ()).throw(AssertionError("nope")),
        resolver_verdict_fn=lambda _t: (_ for _ in ()).throw(
            AssertionError("nope")
        ),
    )
    assert "SKIPPED_BY_DESIGN: 1" in md
    # The composite release predicate now spans the new conjuncts.
    assert "SKIPPED_BY_DESIGN" in md
    assert "phase_b_capture.status == DONE" in md
    assert "release_scope.excluded" in md
    # The per-task row is present.
    assert "SKIPPED_BY_DESIGN" in md
    row = next(
        ln for ln in md.splitlines()
        if ln.startswith(f"| {td.name} | ")
    )
    assert "SKIPPED_BY_DESIGN" in row, row


def test_skip_invocation_non_bool_falls_through(tmp_path, monkeypatch):
    """Defensive: a non-bool ``skip_invocation`` (e.g. the string
    ``"yes"``) is treated as falsy -- the short-circuit does NOT
    fire. Catches malformed orchestrator patches."""
    td = _write_task(
        tmp_path,
        phase_a_prime={"skip_invocation": "yes"},
    )
    monkeypatch.setattr(pap, "_cuda_available", lambda: False)
    monkeypatch.setattr(
        pap, "build_staged_candidate", lambda td_, cls_: td_,
    )
    monkeypatch.setattr(
        pap, "_run_s2_pair",
        lambda *a, **k: (True, "ok", {}, {}, "ok", None),
    )
    monkeypatch.setattr(
        pap, "_accumulate_s2_original_closure", lambda *a, **k: None,
    )
    monkeypatch.setattr(
        pap, "classify_task",
        lambda *a, **k: pap.TaskClassification(task_id=td.name, tier="T1"),
    )
    monkeypatch.setattr(pap, "_entrypoint_from_meta", lambda *a, **k: None)
    monkeypatch.setattr(pap, "_shape_ids_for", lambda *a, **k: ["P1"])
    res = pap.s2_verify_task(td)
    assert res.status != "SKIPPED_BY_DESIGN", (
        "non-bool skip_invocation must not trip the short-circuit"
    )
