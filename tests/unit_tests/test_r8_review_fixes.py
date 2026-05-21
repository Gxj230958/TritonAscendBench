"""R8 codex-review regression tests.

Pins the two R8 MUST-FIX findings so a regression re-surfaces as a test
failure:

  * R8-1 — in verified mode a ``task_status="blocked"`` task MUST stay in
    the fixed ``|S|`` scoring denominator (DESIGN §9 / W4.3 ④). Scoring
    distinguishes ``unsupported`` (out of the scored set — no provider
    golden for this mode) from ``blocked`` (in the fixed ``|S|``, scored
    as a failure ``ε``). A ``blocked`` task must NOT shrink ``|S|`` — that
    is the no-subset-escape rule; BLOCKED is the publish-gate's target.
  * R8-2 — the sampled-output anti-cheat (``_verify_sampled_outputs``) is
    fail-closed: an oracle re-run exception is a verification failure
    (``mismatch=True``), and "sampled outputs shipped up but NONE got
    verified" never leaves ``performance_status="ok"``.

Dev machine has no NPU; these verify the scoring/anti-cheat plumbing.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from triton_ascend_bench import eval as _eval
from triton_ascend_bench import inputs as _inputs
from triton_ascend_bench.scoring import (
    TaskMeasurement,
    capped_geomean_native,
    fastp_at_threshold,
    native_coverage,
)

REPO = Path(__file__).resolve().parents[2]


def _run_tab(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "triton_ascend_bench.cli", *args],
        cwd=REPO, capture_output=True, text=True,
    )


# ---------------------------------------------------------------------------
# R8-1 — verified-mode `blocked` stays in the fixed |S| denominator.
# ---------------------------------------------------------------------------
class TestR8_1BlockedInDenominator:
    def test_codex_repro_1_evaluated_1_blocked_fixed_denom(self):
        """codex repro: 1 evaluated + 1 verified-blocked eligible task.

        The score must be computed against the FIXED |S| = 2 — geomean of
        {s_t=1.0 (evaluated), ε (blocked)} = sqrt(1.0 * 0.01) = 0.1. If
        `blocked` were dropped from the denominator the score would be
        1.0 (a subset-escape) — exactly the R8-1 cheat.
        """
        ms = [
            TaskMeasurement(task_id="ok", s_t=1.0),
            TaskMeasurement(task_id="blk", s_t=1.0, task_status="blocked"),
        ]
        score = capped_geomean_native(ms, s_native_size=2)
        assert score == pytest.approx(0.1), (
            "blocked task escaped the fixed |S| denominator (R8-1)")

    def test_blocked_does_not_widen_score_vs_unsupported(self):
        """A `blocked` task and an `unsupported` task in the same slot must
        score differently: `unsupported` leaves |S| (score 1.0), `blocked`
        stays as ε (score 0.1). They are NOT interchangeable."""
        evaluated_only = [TaskMeasurement(task_id="a", s_t=1.0)]
        with_blocked = evaluated_only + [
            TaskMeasurement(task_id="b", s_t=1.0, task_status="blocked")]
        with_unsupported = evaluated_only + [
            TaskMeasurement(task_id="b", s_t=1.0, task_status="unsupported")]

        assert capped_geomean_native(
            with_unsupported, s_native_size=2) == pytest.approx(1.0)
        assert capped_geomean_native(
            with_blocked, s_native_size=2) == pytest.approx(0.1)

    def test_blocked_counts_against_coverage_and_fastp(self):
        """`blocked` stays in the |S| denominator for coverage and FastP —
        a blocked task is neither an `ok` (coverage) nor threshold-met."""
        ms = [
            TaskMeasurement(task_id="a", correctness_status="ok", s_t=2.0),
            TaskMeasurement(task_id="b", correctness_status="ok", s_t=2.0,
                            task_status="blocked"),
        ]
        # coverage: only `a` is an ok; |S| stays 2 -> 1/2.
        assert native_coverage(ms, s_native_size=2) == pytest.approx(0.5)
        # FastP@1.5: only `a` meets it; |S| stays 2 -> 1/2.
        assert fastp_at_threshold(
            ms, threshold=1.5, s_native_size=2) == pytest.approx(0.5)

    def test_analyze_verified_blocked_uses_fixed_denominator(self, tmp_path):
        """End-to-end: `analyze` in verified mode keeps a repo-authoritative
        `blocked` task inside `n_scored_denominator_tasks` (the fixed |S|).

        The verified-mode denominator must equal the score-eligible set
        size — a `blocked` task does not shrink it (W4.3 ④)."""
        from triton_ascend_bench.dataset import iter_all_tasks
        from triton_ascend_bench.eligibility import (
            compute_score_eligible,
            compute_verified_blocked,
        )

        tasks = iter_all_tasks()
        eligible = compute_score_eligible(tasks)
        blocked = compute_verified_blocked(tasks) & eligible
        if not blocked:
            pytest.skip("no eligible verified-blocked task in the corpus")
        a_blocked = sorted(blocked)[0]
        # WA.4: pick an `an_evaluated` whose `param_constraints` is RECONCILED
        # (`scaffold: false` + semantically clean) — a scaffold task would
        # itself be `blocked` by the WA.4 gate, so `blocked_in_denominator_
        # count` would no longer be exactly 1. The reconciled corpus tasks
        # are the WA.1 sample tasks.
        by_id = {t.task_id: t for t in tasks}

        def _pc_ok(tid: str) -> bool:
            t = by_id.get(tid)
            pc = getattr(t, "param_constraints", None) if t else None
            return isinstance(pc, dict) and pc.get("scaffold") is False

        evaluated_candidates = sorted(
            tid for tid in (eligible - blocked) if _pc_ok(tid))
        if not evaluated_candidates:
            pytest.skip("no reconciled (non-scaffold) eligible task yet "
                        "(WB.1 backfill incomplete)")
        an_evaluated = evaluated_candidates[0]

        suite = {
            "mode": "verified",
            "results": [
                {"task_id": an_evaluated,
                 "correctness_status": "ok", "runtime_status": "ok",
                 "performance_status": "ok", "performance_flags": [],
                 "fallback_detected": False,
                 "aggregate_score": 1.0, "s_t": 1.0,
                 "task_status": "evaluated"},
                {"task_id": a_blocked,
                 "correctness_status": "ok", "runtime_status": "ok",
                 "performance_status": "ok", "performance_flags": [],
                 "fallback_detected": False,
                 "aggregate_score": 1.0, "s_t": 1.0,
                 "task_status": "evaluated"},
            ],
        }
        suite_path = tmp_path / "suite.json"
        out_path = tmp_path / "score.json"
        suite_path.write_text(json.dumps(suite))

        r = _run_tab("analyze", "--verified-results", str(suite_path),
                     "--submitter", "t", "--out", str(out_path))
        assert r.returncode == 0, f"analyze failed: {r.stdout} {r.stderr}"
        doc = json.loads(out_path.read_text())
        # The fixed |S| — `blocked` did not shrink it.
        assert doc["mode"] == "verified"
        assert doc["n_scored_denominator_tasks"] == len(eligible)
        # The blocked task is reported as in-denominator (no-subset-escape).
        assert doc["blocked_in_denominator_count"] == 1


# ---------------------------------------------------------------------------
# R8-2 — sampled-output anti-cheat is fail-closed on oracle re-run errors.
# ---------------------------------------------------------------------------
_SHAPES_DOC = {
    "size_knobs": {"D": 8},
    "public_shapes": [{"id": "P1", "name": "tiny", "knobs": {"D": 4}}],
    "inputs": {
        "x": {"dims": ["D"], "dtype": "float32", "dist": "randn"},
    },
}


class _GoodOracle:
    @staticmethod
    def run(inputs):
        return {"y": inputs["x"] * 2.0}


class _ThrowingOracle:
    """Oracle whose re-run always raises — simulates an oracle that cannot
    be re-evaluated on a sampled timing input."""

    @staticmethod
    def run(inputs):
        raise RuntimeError("oracle re-run blew up")


def _tol_cls():
    from triton_ascend_bench import tolerance
    pol = tolerance.load_policy()
    return next(iter(pol.values()))


class TestR8_2SampledOutputFailClosed:
    def test_oracle_rerun_exception_is_mismatch(self):
        """R8-2: an oracle re-run that raises must yield `mismatch=True`
        (fail-closed) — never a silently skipped trial."""
        meas_seed = 1000
        sampled = {}
        for t in (0, 1):
            inp = _inputs.build_inputs(_SHAPES_DOC, "P1", seed=meas_seed + t)
            sampled[f"{t}::y"] = inp["x"] * 2.0

        drift = _eval._verify_sampled_outputs(
            sampled, _SHAPES_DOC, "P1", meas_seed, _ThrowingOracle(),
            _tol_cls())
        assert drift["mismatch"] is True, (
            "oracle re-run exception must fail closed (R8-2)")
        # Every trial errored — none were genuinely verified.
        assert drift["checked"] == 0
        assert drift["errors"] == 2

    def test_partial_oracle_failure_still_fails_closed(self):
        """One good trial + one whose oracle raises still fails closed —
        the mismatch from the erroring trial wins."""
        meas_seed = 1000

        class _FlakyOracle:
            """Re-run succeeds on the first call, raises on every later
            one — `_verify_sampled_outputs` iterates trials in sorted
            order, so trial t=0 is verified and t=1 fails closed."""
            _calls = 0

            @classmethod
            def run(cls, inputs):
                cls._calls += 1
                if cls._calls == 1:
                    return {"y": inputs["x"] * 2.0}
                raise RuntimeError("flaky oracle")

        sampled = {}
        for t in (0, 1):
            inp = _inputs.build_inputs(_SHAPES_DOC, "P1", seed=meas_seed + t)
            sampled[f"{t}::y"] = inp["x"] * 2.0

        drift = _eval._verify_sampled_outputs(
            sampled, _SHAPES_DOC, "P1", meas_seed, _FlakyOracle,
            _tol_cls())
        assert drift["mismatch"] is True, "partial oracle failure must fail closed"
        assert drift["checked"] == 1
        assert drift["errors"] == 1

    def test_good_oracle_still_passes(self):
        """A fully reverifiable matching set is still a clean pass — the
        fail-closed change does not over-trigger."""
        meas_seed = 1000
        sampled = {}
        for t in (0, 1, 2):
            inp = _inputs.build_inputs(_SHAPES_DOC, "P1", seed=meas_seed + t)
            sampled[f"{t}::y"] = inp["x"] * 2.0

        drift = _eval._verify_sampled_outputs(
            sampled, _SHAPES_DOC, "P1", meas_seed, _GoodOracle(), _tol_cls())
        assert drift["mismatch"] is False
        assert drift["checked"] == 3
        assert drift["errors"] == 0

    def test_sampled_present_zero_verified_flips_drift(self, monkeypatch):
        """R8-2 (end-to-end): a timing pass that shipped up sampled outputs
        but where NONE could be verified (`checked == 0`) must NOT leave
        the run at `performance_status="ok"` — the anti-cheat could not
        run, so the run is held (`measured_outputs_drift` flag + escalation
        off `ok`).

        The dev machine has no NPU, so the real timing pass ships no
        sampled outputs; this test injects a deterministic `TwoPassResult`
        whose timing pass DOES carry sampled outputs and a clean per-shape
        host-timing series, plus a fail-closed "0 verified" verifier — so
        the R8-2 call-site branch is genuinely exercised, not skipped.
        """
        from triton_ascend_bench import process_isolation as _pi
        from triton_ascend_bench import static_checker

        # Make the static checker pass so grading is reached.
        monkeypatch.setattr(
            static_checker, "check_solution",
            lambda *a, **k: static_checker.CheckReport(findings=[]))

        task = "L1_01_bias_kernel"
        solution = REPO / "solutions" / task / "solution.py"
        if not solution.exists():
            pytest.skip(f"no solution for {task}")

        # The candidate's correct output for a shape — taken from the
        # in-process oracle so correctness grades as a pass and the perf
        # verdict gets promoted off "error" to "ok" (then R8-2 escalates).
        def _fake_run_two_pass(solution_path, correctness_inputs,
                               timing_inputs, **kw):
            import importlib.util
            spec = importlib.util.spec_from_file_location(
                "_o", REPO / "tasks" / "level1"
                / task / "oracle_pytorch.py")
            omod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(omod)
            corr, tim = {}, {}
            for sid, inp in correctness_inputs.items():
                out = omod.run(inp)
                corr[sid] = _pi.ShapeCandidateResult(
                    shape_id=sid, ran=True, outputs=out)
                # Timing pass: clean durations + a sampled output payload
                # so the R8-2 "sampled present" branch is reached. The
                # durations are well above 1 µs so timing arbitration does
                # NOT quarantine the run — the only thing that can move
                # `performance_status` off `ok` here is the R8-2 fix.
                sampled = {f"0::{k}": v for k, v in out.items()
                           if hasattr(v, "shape")}
                tim[sid] = _pi.ShapeCandidateResult(
                    shape_id=sid, ran=True, outputs=out,
                    durations=[5_000.0, 5_100.0, 5_050.0],
                    sampled_outputs=sampled)
            return _pi.TwoPassResult(
                correctness=_pi.CandidateRunResult(
                    ran_any=True, timed_out=False, load_error=None,
                    per_shape=corr),
                timing=_pi.CandidateRunResult(
                    ran_any=True, timed_out=False, load_error=None,
                    per_shape=tim, pass_kind="timing"),
            )

        monkeypatch.setattr(_pi, "run_two_pass", _fake_run_two_pass)

        # Fail-closed "0 verified" verifier — the outcome a throwing oracle
        # produces (every sampled trial unverifiable).
        def _zero_verified(sampled, *a, **k):
            return {"checked": 0, "mismatch": False,
                    "errors": len(sampled or {})}

        monkeypatch.setattr(_eval, "_verify_sampled_outputs", _zero_verified)

        result = _eval.evaluate(task, solution, mode="self-reported")

        # The timing pass shipped sampled outputs — the branch ran.
        assert result.shape_results, "no shapes graded"
        assert any(r.durations for r in result.shape_results), (
            "fake timing pass should have carried durations")
        # R8-2: sampled outputs present but 0 verified — must NOT stay ok.
        assert "measured_outputs_drift" in result.performance_flags, (
            "sampled outputs present but 0 verified must flag drift (R8-2)")
        assert result.performance_status != "ok", (
            "sampled outputs present but 0 verified must not stay "
            "performance_status=ok (R8-2)")
