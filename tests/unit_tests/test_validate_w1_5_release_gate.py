"""W1.5: release gate keys on the two real correctness proofs.

The release predicate is the S2 severance gate over the C2-aligned
eligible set::

    release_status == PASS
        <=> phase_a_prime.status == PASS          (S2 severance)

    release-ready eligible set
        = compute_score_eligible INTERSECT is_provenance_eligible
          (C2 oracle_kernel_equivalence VERIFIED / CEILING_ACCEPTED)

`cost_model.status` is NOT a release blocker -- it is ncu-evidenced
PERFORMANCE/access metadata, not a correctness proof (conditional
access, L2 reuse, tile geometry, and the HBM burst floor make a
portable analytic bytes formula structurally unable to hit per-shape
tol everywhere). Its PASS / BLOCKED status stays recorded as metadata
but never gates release.

This file proves:
  * `cost_model.status == BLOCKED` / missing  ⇒ does NOT block release;
  * `phase_a_prime PASS`                        ⇒ NOT blocked;
  * a C2-FAILED task is EXCLUDED from the release-ready eligible set
    (not a blocker -- it never enters the gate);
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
    compute_release_ready_eligible,
    release_status_pass,
)

# A provenance block that makes `is_provenance_eligible` True (C2
# oracle_kernel_equivalence VERIFIED). Tasks carrying this are part of
# the release-ready eligible set; tasks without it are C2-EXCLUDED.
_C2_VERIFIED = {
    "provenance": {"golden": {"oracle_kernel_equivalence": "VERIFIED"}}
}


@dataclass
class _T:
    """Minimal duck-typed task: exposes `task_id`, `meta`, `task_dir`."""

    task_id: str
    meta: dict = field(default_factory=dict)
    task_dir: Path = Path("/nonexistent")


def _meta(pap: str | None, cm: str | None, c2: bool = False) -> dict:
    """Build a meta dict with optional phase_a_prime / cost_model status.

    `None` for either status omits that status key entirely. `cm` is the
    (non-gating) cost_model status -- it is folded in only to prove it
    never moves the release verdict. `c2=True` adds a C2-VERIFIED
    provenance block so the task is in the release-ready eligible set."""
    m: dict = {}
    if pap is not None:
        m["phase_a_prime"] = {"status": pap}
    if cm is not None:
        m["cost_model"] = {"status": cm}
    if c2:
        m.update(_C2_VERIFIED)
    return m


# ── release_status_pass: the S2-severance single-task predicate ──────
def test_release_status_pass_phase_a_prime_pass():
    assert release_status_pass(_meta("PASS", "PASS")) is True


def test_release_status_pass_cost_model_blocked_does_not_block():
    """cost_model.status == BLOCKED is evidenced performance metadata,
    NOT a release blocker ⇒ phase_a_prime PASS still releases."""
    assert release_status_pass(_meta("PASS", "BLOCKED")) is True


def test_release_status_pass_cost_model_missing_does_not_block():
    """A missing cost_model.status does NOT block release -- only the S2
    phase_a_prime gate matters."""
    assert release_status_pass(_meta("PASS", None)) is True


def test_release_status_pass_phase_a_prime_blocked():
    assert release_status_pass(_meta("BLOCKED", "PASS")) is False


def test_release_status_pass_phase_a_prime_missing():
    assert release_status_pass(_meta(None, "PASS")) is False


def test_release_status_pass_both_missing():
    assert release_status_pass({}) is False
    assert release_status_pass(None) is False


# ── compute_release_blocking: the shared blocking-set computation ────
def test_compute_release_blocking_picks_only_s2_failures():
    """Only S2 phase_a_prime failures block -- cost_model BLOCKED /
    missing is evidenced metadata and never enters the blocking set."""
    tasks = [
        _T("ok", _meta("PASS", "PASS")),          # releasable
        _T("cm_blocked", _meta("PASS", "BLOCKED")),  # cost_model: not a blocker
        _T("cm_missing", _meta("PASS", None)),       # cost_model: not a blocker
        _T("pap_blocked", _meta("BLOCKED", "PASS")),  # phase_a_prime fails
    ]
    eligible = {"ok", "cm_blocked", "cm_missing", "pap_blocked"}
    blocking = compute_release_blocking(tasks, eligible)
    assert blocking == ["pap_blocked"]


def test_compute_release_blocking_ignores_non_eligible():
    """A non-PASS task NOT in the eligible set must not block -- only the
    eligible set has teeth. A C2-FAILED task is EXCLUDED from the eligible
    set, so it never blocks release."""
    tasks = [
        _T("ok", _meta("PASS", "PASS")),
        _T("stub", _meta("BLOCKED", "BLOCKED")),  # not eligible (C2-excluded)
    ]
    blocking = compute_release_blocking(tasks, {"ok"})
    assert blocking == []


# ── authority 1: validate_tasks.py --release-gate ────────────────────
def test_release_gate_phase_a_prime_blocked_blocks(capsys):
    """phase_a_prime.status != PASS ⇒ release gate fails (rc 1)."""
    tasks = [
        _T("a", _meta("PASS", "PASS")),
        _T("b", _meta("BLOCKED", "PASS")),
    ]
    rc = vt.run_release_gate(tasks=tasks, eligible_ids={"a", "b"})
    assert rc == 1
    err = capsys.readouterr().err
    assert "b" in err and "NOT releasable" in err


def test_release_gate_phase_a_prime_pass_releasable():
    """phase_a_prime PASS for every eligible task ⇒ gate passes (rc 0)
    regardless of cost_model.status (here BLOCKED / missing)."""
    tasks = [
        _T("a", _meta("PASS", "BLOCKED")),
        _T("b", _meta("PASS", None)),
    ]
    rc = vt.run_release_gate(tasks=tasks, eligible_ids={"a", "b"})
    assert rc == 0


def test_release_gate_cost_model_does_not_block():
    """cost_model BLOCKED / missing is evidenced metadata, NOT a release
    blocker ⇒ gate stays green while phase_a_prime PASSes."""
    tasks = [
        _T("a", _meta("PASS", "PASS")),
        _T("b", _meta("PASS", None)),
    ]
    rc = vt.run_release_gate(tasks=tasks, eligible_ids={"a", "b"})
    assert rc == 0


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


# ── C2 release-ready eligible set: a C2-FAILED task is EXCLUDED ───────
def test_release_ready_eligible_excludes_c2_failed(tmp_path):
    """`compute_release_ready_eligible` = score-eligible INTERSECT
    is_provenance_eligible (C2 VERIFIED / CEILING_ACCEPTED). A C2-FAILED
    task (no provenance block) is EXCLUDED from the eligible set, not a
    blocker -- it never enters the release gate at all."""
    tasks = [
        _T("c2_ok", _meta("PASS", "PASS", c2=True),
           _eligible_task_dir(tmp_path, "c2_ok")),
        _T("c2_failed", _meta("PASS", "PASS"),  # no provenance ⇒ C2-excluded
           _eligible_task_dir(tmp_path, "c2_failed")),
    ]
    eligible = compute_release_ready_eligible(tasks)
    assert eligible == {"c2_ok"}
    # The C2-FAILED task is absent from the eligible set, so the shared
    # blocking computation over that set never sees it -- not a blocker.
    assert compute_release_blocking(tasks, eligible) == []


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


def test_cli_analyze_phase_a_prime_blocked_blocks(tmp_path, monkeypatch):
    # Both tasks C2-aligned (in the release-ready eligible set); `b` is a
    # genuine S2 failure (phase_a_prime BLOCKED) -> the one blocker.
    tasks = [
        _T("a", _meta("PASS", "PASS", c2=True), _eligible_task_dir(tmp_path, "a")),
        _T("b", _meta("BLOCKED", "PASS", c2=True),
           _eligible_task_dir(tmp_path, "b")),
    ]
    doc = _run_cli_analyze(tmp_path, monkeypatch, tasks)
    assert doc["release_blocked"] is True
    assert doc["release_blocked_tasks"] == ["b"]


def test_cli_analyze_phase_a_prime_pass_not_blocked(tmp_path, monkeypatch):
    tasks = [
        _T("a", _meta("PASS", "PASS", c2=True), _eligible_task_dir(tmp_path, "a")),
        _T("b", _meta("PASS", "PASS", c2=True), _eligible_task_dir(tmp_path, "b")),
    ]
    doc = _run_cli_analyze(tmp_path, monkeypatch, tasks)
    assert doc["release_blocked"] is False
    assert doc["release_blocked_tasks"] == []


def test_cli_analyze_cost_model_does_not_block(tmp_path, monkeypatch):
    """cost_model BLOCKED / missing is evidenced metadata, NOT a release
    blocker ⇒ analyze reports no blocking while phase_a_prime PASSes."""
    tasks = [
        _T("a", _meta("PASS", "BLOCKED", c2=True), _eligible_task_dir(tmp_path, "a")),
        _T("b", _meta("PASS", None, c2=True), _eligible_task_dir(tmp_path, "b")),
    ]
    doc = _run_cli_analyze(tmp_path, monkeypatch, tasks)
    assert doc["release_blocked"] is False
    assert doc["release_blocked_tasks"] == []


def test_cli_analyze_c2_failed_excluded_not_blocker(tmp_path, monkeypatch):
    """cli analyze keys the release gate on the C2-aligned release-ready
    eligible set (`compute_release_ready_eligible`), NOT the score-only
    set. A C2-FAILED task (score-eligible on disk but NO provenance, so
    `is_provenance_eligible` is False) is EXCLUDED from the gate -- it
    never enters the blocking set EVEN WHEN its phase_a_prime status is
    BLOCKED. Under the old score-only gate that task would have been
    counted as a blocker; under the C2-aligned gate it is excluded.
    Only the C2-aligned S2 failure (`s2_blocked`) is the genuine blocker
    -- the negative control (`c2_failed`) is BLOCKED yet must not appear.
    """
    tasks = [
        # C2-aligned, S2 PASS -> releasable.
        _T("c2_ok", _meta("PASS", "PASS", c2=True),
           _eligible_task_dir(tmp_path, "c2_ok")),
        # C2-FAILED (no provenance) AND S2 BLOCKED. The old score-only
        # gate would count it; the C2-aligned gate EXCLUDES it.
        _T("c2_failed", _meta("BLOCKED", "PASS"),
           _eligible_task_dir(tmp_path, "c2_failed")),
        # C2-aligned, S2 BLOCKED -> the genuine release blocker.
        _T("s2_blocked", _meta("BLOCKED", "PASS", c2=True),
           _eligible_task_dir(tmp_path, "s2_blocked")),
    ]
    doc = _run_cli_analyze(tmp_path, monkeypatch, tasks)
    # Only the C2-aligned S2 failure blocks; the C2-FAILED task is absent
    # despite being phase_a_prime BLOCKED (C2 exclusion, not S2 pass).
    assert doc["release_blocked"] is True
    assert doc["release_blocked_tasks"] == ["s2_blocked"]


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

    Only S2 phase_a_prime failures block; cost_model BLOCKED / missing
    is evidenced metadata and is absent from every authority's blocking
    set.
    """
    # Every corpus task is C2-aligned (in the release-ready eligible set);
    # only the S2 phase_a_prime failure blocks.
    corpus = [
        _T("t_ok", _meta("PASS", "PASS", c2=True),
           _eligible_task_dir(tmp_path, "t_ok")),
        _T("t_cm_blocked", _meta("PASS", "BLOCKED", c2=True),
           _eligible_task_dir(tmp_path, "t_cm_blocked")),
        _T("t_cm_missing", _meta("PASS", None, c2=True),
           _eligible_task_dir(tmp_path, "t_cm_missing")),
        _T("t_pap_blocked", _meta("BLOCKED", "PASS", c2=True),
           _eligible_task_dir(tmp_path, "t_pap_blocked")),
    ]
    eligible_ids = {t.task_id for t in corpus}
    # cost_model BLOCKED / missing does NOT block -- only S2 phase_a_prime.
    expected = ["t_pap_blocked"]

    # Authority 3 (the shared predicate, also behind phase_a_prime S6).
    shared = compute_release_blocking(corpus, eligible_ids)
    assert shared == expected

    # Authority 1: validate_tasks.py --release-gate. It internally
    # computes the SAME blocking list; a non-empty one ⇒ rc 1.
    rc = vt.run_release_gate(tasks=list(corpus), eligible_ids=eligible_ids)
    assert rc == 1  # blocking non-empty (1 task: t_pap_blocked)

    # Authority 2: cli.py analyze emits the blocking list verbatim.
    doc = _run_cli_analyze(tmp_path, monkeypatch, corpus)
    cli_blocking = doc["release_blocked_tasks"]

    # The three authorities agree on the SAME blocking set.
    assert cli_blocking == shared == expected


def test_three_authorities_agree_on_all_pass(tmp_path, monkeypatch):
    """Parity also holds on the happy path: every authority reports an
    EMPTY blocking set when every eligible task is composite-PASS."""
    corpus = [
        _T("a", _meta("PASS", "PASS", c2=True), _eligible_task_dir(tmp_path, "a")),
        _T("b", _meta("PASS", "PASS", c2=True), _eligible_task_dir(tmp_path, "b")),
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


def test_phase_a_prime_s6_report_states_s2_predicate():
    """The S6 verdict section must (W1.5):
      * carry the blocker count it is handed; and
      * spell out the S2 release predicate `release_status == PASS <=>
        phase_a_prime.status == PASS`, with cost_model.status NOT in the
        gate.
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
    # The blocker count is carried verbatim.
    assert "release_status != PASS`: **7**" in md
    # The S2 release predicate is spelled out.
    assert "phase_a_prime.status == PASS" in md
    # cost_model.status is NOT a gate clause -- the report states it is
    # evidenced metadata, never `cost_model.status == PASS` as a predicate.
    assert "cost_model.status == PASS" not in md
    # Parity with validate_tasks.py --release-gate is asserted in-text.
    assert "compute_release_blocking" in md


def test_phase_a_prime_release_blocker_count_uses_s2(monkeypatch):
    """`_release_blocker_count` must compute the S2 blocking set
    (phase_a_prime.status only) over the C2-aligned release-ready
    eligible set. cost_model.status is NOT folded in -- it is evidenced
    metadata, not a release blocker."""
    tasks = [
        _T("ok", _meta("PASS", "PASS", c2=True), Path("/x/ok")),
        _T("cm_blocked", _meta("PASS", "BLOCKED", c2=True), Path("/x/cm")),
        _T("pap_blocked", _meta("BLOCKED", "PASS", c2=True), Path("/x/pap")),
    ]
    from triton_ascend_bench import dataset as ds
    from triton_ascend_bench import eligibility as elig

    monkeypatch.setattr(ds, "iter_all_tasks", lambda: tasks)
    # All tasks score-eligible (bypass on-disk file reads);
    # `compute_release_ready_eligible` further intersects with C2
    # provenance, which every task here carries (c2=True).
    monkeypatch.setattr(
        elig, "compute_score_eligible", lambda ts: {t.task_id for t in ts}
    )
    # 1 blocker: `pap_blocked` (phase_a_prime BLOCKED). `cm_blocked`
    # (phase_a_prime PASS, cost_model BLOCKED) does NOT block.
    assert pap._release_blocker_count() == 1


def test_phase_a_prime_release_blocker_count_excludes_c2_failed(monkeypatch):
    """`_release_blocker_count` keys on the C2-aligned release-ready set
    (`compute_release_ready_eligible` = score-eligible INTERSECT
    `is_provenance_eligible`), NOT the score-only set. A C2-FAILED task
    (no provenance) is EXCLUDED from the gate even when its phase_a_prime
    status is BLOCKED -- under the old score-only gate it would have been
    counted; under the C2-aligned gate it is excluded. Only the
    C2-aligned S2 failure is counted -- authority parity with cli analyze
    / validate_tasks."""
    tasks = [
        # C2-aligned, S2 PASS -> not a blocker.
        _T("c2_ok", _meta("PASS", "PASS", c2=True), Path("/x/ok")),
        # C2-FAILED (no provenance) AND S2 BLOCKED -> EXCLUDED, not
        # counted (the negative control vs the old score-only gate).
        _T("c2_failed", _meta("BLOCKED", "PASS"), Path("/x/c2f")),
        # C2-aligned, S2 BLOCKED -> the genuine blocker.
        _T("s2_blocked", _meta("BLOCKED", "PASS", c2=True), Path("/x/s2")),
    ]
    from triton_ascend_bench import dataset as ds
    from triton_ascend_bench import eligibility as elig

    monkeypatch.setattr(ds, "iter_all_tasks", lambda: tasks)
    # All score-eligible (bypass on-disk reads); the C2 intersection is
    # what excludes `c2_failed` from the release-ready eligible set.
    monkeypatch.setattr(
        elig, "compute_score_eligible", lambda ts: {t.task_id for t in ts}
    )
    # Exactly 1 blocker: the C2-aligned S2 failure. `c2_failed` is
    # C2-excluded (never a blocker) despite being phase_a_prime BLOCKED.
    assert pap._release_blocker_count() == 1
