"""W1.5: composite release gate (DESIGN-V2 §4.4).

The release predicate is COMPOSITE::

    release_status == PASS
        <=> phase_a_prime.status == PASS  AND  cost_model.status == PASS

`cost_model.status` is `meta.cost_model.status` (written by the W4.2
GPU cross-check: PASS / BLOCKED). A MISSING `cost_model.status` is
treated as NOT PASS, so the gate stays red until Phase 4.

This file proves three things:
  * `cost_model.status == BLOCKED`         ⇒ release blocked;
  * `phase_a_prime PASS ∧ cost_model PASS` ⇒ NOT blocked;
  * a missing `cost_model.status`          ⇒ blocked;
  * the THREE release-status authorities (`validate_tasks.py
    --release-gate`, `cli.py` analyze's `release_blocking`, and the
    shared predicate behind `phase_a_prime.py`'s S6 verdict) yield an
    IDENTICAL blocking set for the same task corpus.
"""
from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import types
from dataclasses import dataclass, field
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parent.parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

# validate_tasks.py is a script, not an importable package module.
_spec = importlib.util.spec_from_file_location(
    "validate_tasks", REPO / "scripts" / "validate_tasks.py"
)
vt = importlib.util.module_from_spec(_spec)
sys.modules["validate_tasks"] = vt
_spec.loader.exec_module(vt)

# phase_a_prime.py is likewise a script.
_pa_spec = importlib.util.spec_from_file_location(
    "phase_a_prime", REPO / "scripts" / "phase_a_prime.py"
)
pap = importlib.util.module_from_spec(_pa_spec)
sys.modules["phase_a_prime"] = pap
_pa_spec.loader.exec_module(pap)

from triton_ascend_bench.eligibility import (  # noqa: E402
    compute_release_blocking,
    release_status_pass,
)


@dataclass
class _T:
    """Minimal duck-typed task: exposes `task_id`, `meta`, `task_dir`."""

    task_id: str
    meta: dict = field(default_factory=dict)
    task_dir: Path = Path("/nonexistent")


def _meta(pap: str | None, cm: str | None) -> dict:
    """Build a meta dict with optional phase_a_prime / cost_model status.

    `None` for either status omits that status key entirely (a missing
    status, the realistic pre-W4.2 state for `cost_model`)."""
    m: dict = {}
    if pap is not None:
        m["phase_a_prime"] = {"status": pap}
    if cm is not None:
        m["cost_model"] = {"status": cm}
    return m


# ── release_status_pass: the composite single-task predicate ─────────
def test_release_status_pass_both_pass():
    assert release_status_pass(_meta("PASS", "PASS")) is True


def test_release_status_pass_cost_model_blocked():
    assert release_status_pass(_meta("PASS", "BLOCKED")) is False


def test_release_status_pass_cost_model_missing():
    """The realistic pre-W4.2 state: phase_a_prime PASS but no
    cost_model.status at all ⇒ NOT PASS (gate stays red)."""
    assert release_status_pass(_meta("PASS", None)) is False


def test_release_status_pass_phase_a_prime_blocked():
    assert release_status_pass(_meta("BLOCKED", "PASS")) is False


def test_release_status_pass_phase_a_prime_missing():
    assert release_status_pass(_meta(None, "PASS")) is False


def test_release_status_pass_both_missing():
    assert release_status_pass({}) is False
    assert release_status_pass(None) is False


# ── compute_release_blocking: the shared blocking-set computation ────
def test_compute_release_blocking_picks_composite_failures():
    tasks = [
        _T("ok", _meta("PASS", "PASS")),          # releasable
        _T("cm_blocked", _meta("PASS", "BLOCKED")),  # cost_model fails
        _T("cm_missing", _meta("PASS", None)),       # cost_model missing
        _T("pap_blocked", _meta("BLOCKED", "PASS")),  # phase_a_prime fails
    ]
    eligible = {"ok", "cm_blocked", "cm_missing", "pap_blocked"}
    blocking = compute_release_blocking(tasks, eligible)
    assert blocking == ["cm_blocked", "cm_missing", "pap_blocked"]


def test_compute_release_blocking_ignores_non_eligible():
    """A composite-failing task NOT in the eligible set must not block
    -- only the eligible set has teeth."""
    tasks = [
        _T("ok", _meta("PASS", "PASS")),
        _T("stub", _meta("BLOCKED", "BLOCKED")),  # not eligible
    ]
    blocking = compute_release_blocking(tasks, {"ok"})
    assert blocking == []


# ── authority 1: validate_tasks.py --release-gate ────────────────────
def test_release_gate_cost_model_blocked_blocks(capsys):
    """cost_model.status == BLOCKED ⇒ release gate fails (rc 1)."""
    tasks = [
        _T("a", _meta("PASS", "PASS")),
        _T("b", _meta("PASS", "BLOCKED")),
    ]
    rc = vt.run_release_gate(tasks=tasks, eligible_ids={"a", "b"})
    assert rc == 1
    err = capsys.readouterr().err
    assert "b" in err and "NOT releasable" in err


def test_release_gate_both_pass_releasable():
    """phase_a_prime PASS ∧ cost_model PASS for every eligible task
    ⇒ release gate passes (rc 0)."""
    tasks = [
        _T("a", _meta("PASS", "PASS")),
        _T("b", _meta("PASS", "PASS")),
    ]
    rc = vt.run_release_gate(tasks=tasks, eligible_ids={"a", "b"})
    assert rc == 0


def test_release_gate_cost_model_missing_blocks():
    """Missing cost_model.status (pre-W4.2) ⇒ NOT PASS ⇒ gate fails."""
    tasks = [
        _T("a", _meta("PASS", "PASS")),
        _T("b", _meta("PASS", None)),
    ]
    rc = vt.run_release_gate(tasks=tasks, eligible_ids={"a", "b"})
    assert rc == 1


# ── authority 2: cli.py analyze release_blocking ─────────────────────
def _eligible_task_dir(base: Path, tid: str) -> Path:
    """Create a real on-disk task dir that `compute_score_eligible`
    counts as score-eligible (non-stub oracle, non-PHASE-C-MANUAL
    shapes)."""
    d = base / tid
    d.mkdir(parents=True, exist_ok=True)
    (d / "oracle_pytorch.py").write_text("def run(inputs):\n    return {}\n")
    (d / "shapes.yaml").write_text("public_shapes: []\n")
    return d


def _run_cli_analyze(tmp_path, monkeypatch, tasks):
    """Drive `cli._cmd_analyze` with `dataset.iter_all_tasks` patched to
    return `tasks`; return the parsed score_report.json."""
    from triton_ascend_bench import cli
    from triton_ascend_bench import dataset as ds

    monkeypatch.setattr(ds, "iter_all_tasks", lambda: tasks)

    verified = tmp_path / "verified.json"
    verified.write_text(json.dumps({"results": []}))
    out = tmp_path / "score.json"
    args = types.SimpleNamespace(
        verified_results=str(verified),
        hidden_results=None,
        out=str(out),
        submitter="w1.5-test",
    )
    rc = cli._cmd_analyze(args)
    assert rc == 0
    return json.loads(out.read_text())


def test_cli_analyze_cost_model_blocked_blocks(tmp_path, monkeypatch):
    tasks = [
        _T("a", _meta("PASS", "PASS"), _eligible_task_dir(tmp_path, "a")),
        _T("b", _meta("PASS", "BLOCKED"),
           _eligible_task_dir(tmp_path, "b")),
    ]
    doc = _run_cli_analyze(tmp_path, monkeypatch, tasks)
    assert doc["release_blocked"] is True
    assert doc["release_blocked_tasks"] == ["b"]


def test_cli_analyze_both_pass_not_blocked(tmp_path, monkeypatch):
    tasks = [
        _T("a", _meta("PASS", "PASS"), _eligible_task_dir(tmp_path, "a")),
        _T("b", _meta("PASS", "PASS"), _eligible_task_dir(tmp_path, "b")),
    ]
    doc = _run_cli_analyze(tmp_path, monkeypatch, tasks)
    assert doc["release_blocked"] is False
    assert doc["release_blocked_tasks"] == []


def test_cli_analyze_cost_model_missing_blocks(tmp_path, monkeypatch):
    tasks = [
        _T("a", _meta("PASS", "PASS"), _eligible_task_dir(tmp_path, "a")),
        _T("b", _meta("PASS", None), _eligible_task_dir(tmp_path, "b")),
    ]
    doc = _run_cli_analyze(tmp_path, monkeypatch, tasks)
    assert doc["release_blocked"] is True
    assert doc["release_blocked_tasks"] == ["b"]


# ── three-authority parity: identical blocking set ───────────────────
def test_three_authorities_identical_blocking_set(tmp_path, monkeypatch):
    """The same task corpus must produce an IDENTICAL blocking list
    from all three release-status authorities. They are identical BY
    CONSTRUCTION because all three call `compute_release_blocking`;
    this test pins that contract end-to-end.

    `phase_a_prime.py`'s S6 verdict uses `_release_blocker_count`,
    which is `len(compute_release_blocking(iter_all_tasks(), eligible))`
    -- so the shared `compute_release_blocking` IS the phase_a_prime
    authority and is exercised directly here as the third path.
    """
    corpus = [
        _T("t_ok", _meta("PASS", "PASS"),
           _eligible_task_dir(tmp_path, "t_ok")),
        _T("t_cm_blocked", _meta("PASS", "BLOCKED"),
           _eligible_task_dir(tmp_path, "t_cm_blocked")),
        _T("t_cm_missing", _meta("PASS", None),
           _eligible_task_dir(tmp_path, "t_cm_missing")),
        _T("t_pap_blocked", _meta("BLOCKED", "PASS"),
           _eligible_task_dir(tmp_path, "t_pap_blocked")),
    ]
    eligible_ids = {t.task_id for t in corpus}
    expected = ["t_cm_blocked", "t_cm_missing", "t_pap_blocked"]

    # Authority 3 (the shared predicate, also behind phase_a_prime S6).
    shared = compute_release_blocking(corpus, eligible_ids)
    assert shared == expected

    # Authority 1: validate_tasks.py --release-gate. It internally
    # computes the SAME blocking list; a non-empty one ⇒ rc 1.
    rc = vt.run_release_gate(tasks=list(corpus), eligible_ids=eligible_ids)
    assert rc == 1  # blocking non-empty (3 tasks)

    # Authority 2: cli.py analyze emits the blocking list verbatim.
    doc = _run_cli_analyze(tmp_path, monkeypatch, corpus)
    cli_blocking = doc["release_blocked_tasks"]

    # The three authorities agree on the SAME blocking set.
    assert cli_blocking == shared == expected


def test_three_authorities_agree_on_all_pass(tmp_path, monkeypatch):
    """Parity also holds on the happy path: every authority reports an
    EMPTY blocking set when every eligible task is composite-PASS."""
    corpus = [
        _T("a", _meta("PASS", "PASS"), _eligible_task_dir(tmp_path, "a")),
        _T("b", _meta("PASS", "PASS"), _eligible_task_dir(tmp_path, "b")),
    ]
    eligible_ids = {"a", "b"}

    shared = compute_release_blocking(corpus, eligible_ids)
    assert shared == []

    rc = vt.run_release_gate(tasks=list(corpus), eligible_ids=eligible_ids)
    assert rc == 0

    doc = _run_cli_analyze(tmp_path, monkeypatch, corpus)
    assert doc["release_blocked_tasks"] == shared == []


# ── authority 3: phase_a_prime.py S6 verdict report ──────────────────
def _placeholder_task() -> Path:
    """A minimal task whose `oracle.gpu_triton` points at a callable
    that does NOT exist -> pre-filtered (no smoke run needed)."""
    td = Path(tempfile.mkdtemp(prefix="w1_5_pa_"))
    (td / "kernel.py").write_text("import torch\ndef ent(x):\n    return x\n")
    (td / "meta.yaml").write_text(
        yaml.safe_dump(
            {
                "task_id": "FX",
                "upstream": {"file_sha256": "ORIG"},
                "oracle": {"gpu_triton": "kernel.py::placeholder"},
            }
        )
    )
    (td / "shapes.yaml").write_text(
        yaml.safe_dump(
            {
                "size_knobs": {"D": 4},
                "public_shapes": [{"id": "P1", "knobs": {"D": 4}}],
                "inputs": {
                    "x": {"dims": ["D"], "dtype": "float32",
                          "dist": "randn"},
                },
            }
        )
    )
    return td


def test_phase_a_prime_s6_report_states_composite_predicate():
    """The S6 verdict section must (W1.5):
      * carry the COMPOSITE-predicate blocker count it is handed; and
      * spell out `release_status == PASS <=> phase_a_prime.status ==
        PASS AND cost_model.status == PASS`.
    The blocker count is supplied by `_release_blocker_count`, which is
    `len(compute_release_blocking(...))` -- the SAME shared predicate
    the other two authorities use, so the report's count cannot drift.
    """
    td = _placeholder_task()
    tc = pap.classify_task(td.name, (td / "kernel.py").read_text())

    def _rvf(task_dir):
        # genuine no-callable placeholder -> not a false UNRUN.
        return (
            "BLOCKED(no-verified-upstream-entrypoint: no-host-wrapper)",
            "no host wrapper (placeholder)",
        )

    md = pap._render_status_md(
        [tc], gpu=False, eligible_count=7,
        task_dirs_by_id={td.name: td}, resolver_verdict_fn=_rvf,
    )
    # The composite-predicate blocker count is carried verbatim.
    assert "release_status != PASS`: **7**" in md
    # The composite predicate is spelled out (not the old
    # phase_a_prime-only predicate).
    assert "phase_a_prime.status == PASS" in md
    assert "cost_model.status == PASS" in md
    # Parity with validate_tasks.py --release-gate is asserted in-text.
    assert "compute_release_blocking" in md


def test_phase_a_prime_release_blocker_count_uses_composite(monkeypatch):
    """`_release_blocker_count` must compute the COMPOSITE blocking set
    (cost_model.status folded in), not just phase_a_prime.status."""
    tasks = [
        _T("ok", _meta("PASS", "PASS"), Path("/x/ok")),
        _T("cm_blocked", _meta("PASS", "BLOCKED"), Path("/x/cm")),
    ]
    from triton_ascend_bench import dataset as ds
    from triton_ascend_bench import eligibility as elig

    monkeypatch.setattr(ds, "iter_all_tasks", lambda: tasks)
    # Both tasks score-eligible (bypass on-disk file reads).
    monkeypatch.setattr(
        elig, "compute_score_eligible", lambda ts: {t.task_id for t in ts}
    )
    # 1 blocker: `cm_blocked` (phase_a_prime PASS but cost_model BLOCKED)
    # -- a phase_a_prime-only predicate would wrongly count 0.
    assert pap._release_blocker_count() == 1
