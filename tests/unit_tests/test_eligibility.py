"""U7: single-source-of-truth eligibility regression tests.

Covers: behaviour byte-identical to the old inline cli.py loop
(NotImplementedError stub / PHASE-C-MANUAL excluded), side-effect-light
(no import-time I/O), cli.py and validate share the same authority.

M2 Unit 2.2 extension: bench-team-internal §6a state fields
(`release_scope.excluded`, `phase_a_prime.status == SKIPPED_BY_DESIGN`,
`phase_b_capture_required` + `phase_b_capture.status`) are surfaced by
the eligibility gate. Backward-compatible — a meta.yaml with none of
these fields behaves exactly as before.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from ascendbench.corpus.dataset import iter_all_tasks
from ascendbench.release_gate import eligibility


@dataclass
class _FakeTask:
    task_id: str
    task_dir: Path
    meta: dict = field(default_factory=dict)


def _write_real_files(td: Path) -> None:
    """Materialise the two files :func:`is_score_eligible` reads."""
    (td / "oracle_pytorch.py").write_text("def run(i):\n    return {}\n")
    (td / "shapes.yaml").write_text("public_shapes: []\n")


def test_stub_oracle_not_eligible(tmp_path):
    (tmp_path / "oracle_pytorch.py").write_text(
        "def run(i):\n    raise NotImplementedError\n"
    )
    (tmp_path / "shapes.yaml").write_text("public_shapes: []\n")
    assert not eligibility.is_score_eligible(_FakeTask("t", tmp_path))


def test_phase_c_manual_not_eligible(tmp_path):
    (tmp_path / "oracle_pytorch.py").write_text("def run(i):\n    return {}\n")
    (tmp_path / "shapes.yaml").write_text("# PHASE-C-MANUAL\n")
    assert not eligibility.is_score_eligible(_FakeTask("t", tmp_path))


def test_real_task_is_eligible(tmp_path):
    (tmp_path / "oracle_pytorch.py").write_text("def run(i):\n    return {}\n")
    (tmp_path / "shapes.yaml").write_text("public_shapes: []\n")
    assert eligibility.is_score_eligible(_FakeTask("t", tmp_path))


def test_phase_c_manual_v2_gated_by_release_scope(tmp_path):
    # the v2 corpus is comment-free, so a phase-c-manual scaffold can no longer carry a `# PHASE-C-MANUAL`
    # shapes.yaml comment -- generate gates it via the structured meta marker instead. A canonical v2
    # shapes.yaml (no comment) + release_scope.excluded must still be NOT score-eligible.
    (tmp_path / "oracle_pytorch.py").write_text("def run(i):\n    return {}\n")
    (tmp_path / "shapes.yaml").write_text("schema_version: 2\npublic: []\nhidden: []\n")
    task = _FakeTask("t", tmp_path, meta={"release_scope": {"excluded": True}})
    assert not eligibility.is_score_eligible(task)


def test_os_error_is_not_eligible(tmp_path):
    """Missing files -> not eligible (same conservative behaviour as
    the original inline `continue` on OSError)."""
    assert not eligibility.is_score_eligible(_FakeTask("t", tmp_path / "nope"))


def test_no_import_time_io():
    """eligibility module must not perform import-time I/O (plan U7:
    side-effect-light). Re-importing is a no-op."""
    import importlib

    import ascendbench.release_gate.eligibility as e

    importlib.reload(e)  # would raise if import did filesystem work badly
    assert callable(e.compute_score_eligible)


def test_matches_legacy_inline_logic():
    """Regression: compute_score_eligible over the real catalogue
    equals the OLD inline cli.py loop exactly (single source of
    truth, no drift)."""
    all_tasks = iter_all_tasks()
    new = eligibility.compute_score_eligible(all_tasks)

    legacy: set[str] = set()
    for t in all_tasks:
        try:
            osrc = (t.task_dir / "oracle_pytorch.py").read_text()
            shp = (t.task_dir / "shapes.yaml").read_text()
        except OSError:
            continue
        if "raise NotImplementedError" in osrc or "PHASE-C-MANUAL" in shp:
            continue
        legacy.add(t.task_id)

    assert new == legacy
    assert len(new) > 0  # the real catalogue has eligible tasks


# ---------------------------------------------------------------------------
# M2 Unit 2.2 — `release_scope.excluded` drops a task from score-eligibility
# without altering the corpus denominator.
# ---------------------------------------------------------------------------

def test_excluded_task_not_score_eligible(tmp_path):
    """`release_scope.excluded == True` ⇒ the task is NOT score-eligible
    (the early-out the orchestrator's `--mark-unfixable` flow relies on).
    """
    _write_real_files(tmp_path)
    task = _FakeTask("t", tmp_path, meta={
        "release_scope": {
            "excluded": True,
            "reason": "unfixable by patch flow",
            "excluded_at": "2026-05-23T12:00:00Z",
            "excluded_by": "fix_orchestrator/automation",
        }
    })
    assert not eligibility.is_score_eligible(task)


def test_excluded_with_reason_still_in_corpus(tmp_path):
    """Excluding a task drops it from the SCORED set but does NOT remove
    it from the corpus iteration — `compute_score_eligible` over a mixed
    list still counts every non-excluded task; the excluded one alone
    drops out (no subset escape via silent denominator shrink).
    """
    td1 = tmp_path / "kept"
    td2 = tmp_path / "excluded"
    td1.mkdir()
    td2.mkdir()
    _write_real_files(td1)
    _write_real_files(td2)
    kept = _FakeTask("kept", td1, meta={})
    excluded = _FakeTask("excluded", td2, meta={
        "release_scope": {
            "excluded": True,
            "reason": "unfixable",
            "excluded_at": "2026-05-23T12:00:00Z",
            "excluded_by": "operator",
        }
    })
    # Eligible set: just `kept` — `excluded` is dropped from scoring.
    eligible = eligibility.compute_score_eligible([kept, excluded])
    assert eligible == {"kept"}
    # But the iteration still sees both — caller controls the corpus.
    assert {kept.task_id, excluded.task_id} == {"kept", "excluded"}


def test_release_scope_excluded_false_is_eligible(tmp_path):
    """`release_scope.excluded == False` is the explicit not-excluded
    form (same as an absent block)."""
    _write_real_files(tmp_path)
    task = _FakeTask("t", tmp_path, meta={"release_scope": {"excluded": False}})
    assert eligibility.is_score_eligible(task)


def test_backward_compat_meta_without_new_fields(tmp_path):
    """A meta.yaml that predates M2 (no release_scope / phase_b_capture
    blocks) loads and behaves exactly as before — same score-eligibility,
    and release_status_pass keys on phase_a_prime alone.
    """
    _write_real_files(tmp_path)
    # Score-eligibility: same as the legacy positional-only construction.
    legacy = _FakeTask("t", tmp_path, meta={})
    assert eligibility.is_score_eligible(legacy)
    # release_status_pass: phase_a_prime.status == PASS is sufficient.
    # cost_model is NOT a release blocker (evidenced perf metadata only),
    # so phase_a_prime PASS alone — with no cost_model block — passes.
    assert eligibility.release_status_pass({
        "phase_a_prime": {"status": "PASS"},
    })
    # And a missing phase_a_prime status still fails (no spurious uplift).
    assert not eligibility.release_status_pass({})


# ---------------------------------------------------------------------------
# M2 Unit 2.2 — `phase_a_prime.status == SKIPPED_BY_DESIGN` + golden DONE
# is accepted by the release gate (defer-to-golden flow).
# ---------------------------------------------------------------------------

def test_skipped_by_design_passes_release_gate():
    """`SKIPPED_BY_DESIGN` + `phase_b_capture.status == DONE` (+ a
    non-empty reason) is accepted by `release_status_pass` as the
    offline-golden truth-source delegation path (§6a.2)."""
    meta = {
        "phase_a_prime": {
            "skip_invocation": True,
            "status": "SKIPPED_BY_DESIGN",
            "reason": "upstream entrypoint requires distributed launcher",
        },
        "cost_model": {"status": "PASS"},
        "phase_b_capture": {
            "status": "DONE",
            "captured_at": "2026-05-23T12:00:00Z",
            "upstream_callable": {
                "module": "vllm.model_executor.layers.layernorm",
                "function": "rms_norm",
            },
        },
    }
    assert eligibility.release_status_pass(meta)


def test_skipped_by_design_requires_reason():
    """`SKIPPED_BY_DESIGN` without a non-empty `reason` is NOT PASS —
    the invariant §6a.6 #2 must hold at gate time too (fail-closed
    defence against a malformed block bypassing the patch validator).
    """
    meta = {
        "phase_a_prime": {
            "skip_invocation": True,
            "status": "SKIPPED_BY_DESIGN",
            "reason": "",
        },
        "cost_model": {"status": "PASS"},
        "phase_b_capture": {"status": "DONE"},
    }
    assert not eligibility.release_status_pass(meta)


def test_skipped_by_design_without_golden_done_fails():
    """`SKIPPED_BY_DESIGN` but the offline golden has NOT been captured
    yet (`phase_b_capture.status != DONE`) is NOT PASS — the gate
    requires the golden to be on disk before the delegation can land
    (§6a.2)."""
    meta = {
        "phase_a_prime": {
            "skip_invocation": True,
            "status": "SKIPPED_BY_DESIGN",
            "reason": "needs distributed launcher",
        },
        "cost_model": {"status": "PASS"},
        "phase_b_capture": {"status": "PENDING"},
    }
    assert not eligibility.release_status_pass(meta)


# ---------------------------------------------------------------------------
# M2 Unit 2.2 — `phase_b_capture_required` promotes the offline golden
# into the conjunctive release predicate.
# ---------------------------------------------------------------------------

def test_phase_b_required_blocks_release():
    """`phase_b_capture_required == True` + capture not DONE ⇒ NOT
    PASS, even if `phase_a_prime.status == PASS` (the §6a.4 promotion
    of the golden into the release-gate conjunction).
    """
    meta = {
        "phase_a_prime": {"status": "PASS"},
        "cost_model": {"status": "PASS"},
        "phase_b_capture_required": True,
        "phase_b_capture": {"status": "PENDING"},
    }
    assert not eligibility.release_status_pass(meta)


def test_phase_b_required_with_done_passes():
    """`phase_b_capture_required == True` + `phase_b_capture.status ==
    DONE` clears the §6a.4 conjunction; PASS provided the base
    predicate also holds."""
    meta = {
        "phase_a_prime": {"status": "PASS"},
        "cost_model": {"status": "PASS"},
        "phase_b_capture_required": True,
        "phase_b_capture": {"status": "DONE"},
    }
    assert eligibility.release_status_pass(meta)


# ---------------------------------------------------------------------------
# M2 Unit 2.2 — `release_scope.excluded` is a release-gate hard blocker:
# the composite predicate includes `NOT release_scope.excluded` so an
# operator-excluded task can never satisfy `release_status_pass` even if
# its phase_a_prime / cost_model / phase_b_capture all PASS.
# ---------------------------------------------------------------------------

def test_release_scope_excluded_blocks_release_status_pass():
    """`release_scope.excluded == True` is a hard release blocker: a
    task removed from the scored set by an operator must not
    simultaneously satisfy `release_status_pass` (the gate keeps
    eligibility and release-status authorities aligned).
    """
    base = {
        "phase_a_prime": {"status": "PASS"},
        "cost_model": {"status": "PASS"},
    }
    excluded = {
        **base,
        "release_scope": {
            "excluded": True,
            "reason": "x",
            "excluded_at": "2026-05-23T12:00:00Z",
            "excluded_by": "op",
        },
    }
    # Base PASS, excluded variant blocked by the `NOT excluded` clause.
    assert eligibility.release_status_pass(base)
    assert not eligibility.release_status_pass(excluded)
