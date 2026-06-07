#!/usr/bin/env python3
"""Full GPU regression for triton-ascend-bench.

Runs EVERY bench stage over all tasks on real hardware (the H200 daemon via the
file mailbox), each with a FRESH batch-id, FAILS LOUDLY on a partial/stale/
reused run, compares each stage against a baseline to detect regressions,
cleans up ONLY the pull-meta churn this run introduced, and emits a visual
markdown report. The one command to run after any code change before declaring
"verified" (see the full-regression memory rule).

Stages (dependency order = how an operator enters the bench: extract -> verify
-> rank; each later stage depends on the earlier ones):
  1. unit          dev: pytest tests/unit_tests/ (failed+errors vs baseline)
  2. s2            daemon: severance verify (EXTRACTION -- the severed kernel is
                   bit-exact; cert-stage ALSO reruns cost_model_xcheck + golden)
  3. c2            daemon: run-oracle (CORRECTNESS -- oracle-equiv; verdicts in
                   r10/) -- depends on a correctly extracted kernel
  4. gpu-rank      daemon: run-gpu-rank (PERFORMANCE -- Roofline ranking) --
                   ranks only correctness-passing shapes, and reads S2's cert-
                   stage-updated cost_model, so it runs after s2 + c2
  -> cleanup (after the daemon stages, BEFORE release-gate): revert only the
     cert-meta churn and drop only the stale untracked tasks THIS run dragged in.
  5. release-gate  dev: validate_tasks --release-gate

Report -> docs/reviews/<run-id>-full-regression.md

Safety invariants (every one was a real footgun found in review):
  - REFUSES to run if the working tree under tasks/ is dirty at start (else
    cleanup's git-checkout would silently revert the developer's own edits).
  - A start-of-run snapshot means cleanup touches ONLY paths this run created;
    never a pre-existing dirty/untracked path.
  - REFUSES a stage whose results/<batch>/ already exists (a reused batch-id
    would serve a prior run's stale verdicts -- the bc2full1 false-PASS trap).
  - A daemon stage is PASS only if orchestrator rc==0 AND every submitted task
    produced a verdict (no truncated denominator from a DEAD/timeout partial).

Usage:
  python3 scripts/full_regression.py --run-id 0607T0900 \
      [--tasks-file F] [--stages unit,gpu-rank,c2,s2,release-gate] \
      [--unit-baseline 121] [--prev-rank-batch B] [--c2-baseline B] [--dry-run]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
_LOCAL_LOOP = os.environ.get("LOOP_MAILBOX_LOCAL")
RESULTS = (Path(_LOCAL_LOOP) if _LOCAL_LOOP else REPO.parent / "loop_mailbox") \
    / "results"
ORCH = ["python3", str(REPO / "scripts" / "loop" / "orchestrator.py")]
PASS_OVERALL = ("VERIFIED", "CEILING_ACCEPTED")  # oracle-equiv never emits MIXED


@dataclass
class StageResult:
    name: str
    batch_id: str | None
    status: str            # PASS | FAIL | ERROR | INCONCLUSIVE | SKIPPED
    summary: str
    metrics: dict = field(default_factory=dict)
    regressions: list = field(default_factory=list)


def _run(cmd, timeout=None) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=str(REPO), capture_output=True, text=True,
                          timeout=timeout)


def _load_json(path: Path):
    try:
        return json.loads(path.read_text())
    except (ValueError, OSError):
        return None


def _load_tasks(tasks_file) -> list[str]:
    if tasks_file:
        return [t.strip() for t in Path(tasks_file).read_text().splitlines()
                if t.strip()]
    tasks = []
    for lvl in ("level1", "level2", "level3"):
        for d in sorted((REPO / "tasks" / lvl).glob("L[123]_*")):
            if d.is_dir() and (d / "meta.yaml").is_file():
                tasks.append(d.name)
    return tasks


def _verdicts(batch_id: str, sub: str) -> dict:
    d = RESULTS / batch_id / sub
    out = {}
    if d.is_dir():
        for f in sorted(d.glob("*.json")):
            j = _load_json(f)
            if j is not None:
                out[f.stem] = j
    return out


def _tasks_dirty() -> list[str]:
    cp = _run(["git", "status", "--porcelain", "tasks/"])
    return [ln for ln in cp.stdout.splitlines() if ln.strip()]


# --- daemon stage driver --------------------------------------------------

def _run_daemon(subcmd, batch_id, tasks, sub, stage_timeout) -> tuple:
    """Run an orchestrator daemon batch. Returns (rc, verdicts, missing). Inner
    --timeout derived from the subprocess stage_timeout so the daemon wait does
    not outlive (or undershoot) the outer kill."""
    inner = max(600, stage_timeout - 600)
    cmd = [*ORCH, subcmd, "--batch-id", batch_id, "--tasks", *tasks,
           "--timeout", str(inner), "--interval", "30"]
    cp = _run(cmd, timeout=stage_timeout)
    v = _verdicts(batch_id, sub)
    missing = sorted(set(tasks) - set(v))
    return cp.returncode, v, missing


def _daemon_guard(rc, missing, n_v) -> tuple:
    """Common partial/stale guard. Returns (failed, reason)."""
    if rc != 0:
        return True, f"orchestrator rc={rc} (DEAD/timeout/refused)"
    if missing:
        return True, f"{len(missing)} submitted tasks produced no verdict " \
                     f"(partial run): {missing[:5]}"
    if n_v == 0:
        return True, "0 verdicts collected (tunnel down / wrong results dir)"
    return False, ""


# --- stages ---------------------------------------------------------------

def stage_unit(args, _tasks) -> StageResult:
    cp = _run(["python3", "-m", "pytest", "tests/unit_tests/", "-q",
               "-p", "no:cacheprovider"], timeout=1800)
    out = cp.stdout + cp.stderr

    def _n(word):
        m = re.search(rf"(\d+) {word}", out)
        return int(m.group(1)) if m else 0
    failed, errors, skipped, passed = (
        _n("failed"), _n("errors?"), _n("skipped"), _n("passed"))
    bad = failed + errors
    base = args.unit_baseline
    ok = passed > 0 and bad <= base + 2   # +2 for known flaky (flock/resolve)
    regs = []
    if not ok:
        regs.append(f"failed+errors={bad} exceeds baseline {base}+2")
    if skipped:
        regs.append(f"{skipped} skipped (zero-skip policy)")
    return StageResult(
        "unit", None, "PASS" if ok and not skipped else "FAIL",
        f"{failed} failed, {errors} errors, {skipped} skipped, {passed} passed "
        f"(baseline {base})",
        {"failed": failed, "errors": errors, "skipped": skipped,
         "passed": passed}, regs)


def stage_gpu_rank(args, tasks) -> StageResult:
    bid = f"brank{args.run_id}"
    rc, v, missing = _run_daemon("run-gpu-rank", bid, tasks, "gpu_rank",
                                 args.stage_timeout)
    failed, reason = _daemon_guard(rc, missing, len(v))
    if failed:
        return StageResult("gpu-rank", bid, "FAIL", reason, {}, [reason])
    ranked = sum(1 for d in v.values()
                 if d.get("overall") in ("RANKED", "FLOOR"))
    # timing-not-fired only counts for tasks that SHOULD have timed (ranked);
    # an empty series on a BLOCKED/NOT_RANKED task is correctness, not env.
    no_timing = [t for t, d in v.items()
                 if d.get("overall") in ("RANKED", "FLOOR")
                 and not any(d.get("shape_t_npu_ns", {}).values())]
    prev = {t: d.get("overall")
            for t, d in _verdicts(args.prev_rank_batch, "gpu_rank").items()}
    flips = [f"{t}: {prev[t]}->{d.get('overall')}" for t, d in v.items()
             if prev.get(t) and d.get("overall") != prev[t]]
    regs = flips + ([f"{len(no_timing)} ranked tasks timing not fired "
                     f"(env-fix broken)"] if no_timing else [])
    ok = ranked == len(v) and not no_timing and not flips
    status = "PASS" if ok else "FAIL"
    if ok and not prev:
        status = "INCONCLUSIVE"
        regs.append(f"baseline {args.prev_rank_batch} empty -- no flip check")
    return StageResult(
        "gpu-rank", bid, status,
        f"{ranked}/{len(v)} ranked, {len(no_timing)} timing-missing, "
        f"{len(flips)} flips vs {args.prev_rank_batch}",
        {"ranked": ranked, "total": len(v)}, regs)


def stage_c2(args, tasks) -> StageResult:
    bid = f"bc2{args.run_id}"
    rc, v, missing = _run_daemon("run-oracle", bid, tasks, "r10",
                                 args.stage_timeout)
    failed, reason = _daemon_guard(rc, missing, len(v))
    if failed:
        return StageResult("c2", bid, "FAIL", reason, {}, [reason])
    nonpass = {t: (d.get("overall") or "UNPARSEABLE")
               for t, d in v.items() if d.get("overall") not in PASS_OVERALL}
    base = {t: d.get("overall")
            for t, d in _verdicts(args.c2_baseline, "r10").items()}
    # regression-relative: a task non-PASS in the baseline too is not a
    # regression of THIS change (e.g. upstream-bug L2_23/24).
    regs = [f"{t}: {base.get(t)}->{o}" for t, o in nonpass.items()
            if base.get(t) in PASS_OVERALL]
    status = "FAIL" if regs else ("INCONCLUSIVE" if not base else "PASS")
    return StageResult(
        "c2", bid, status,
        f"{len(v)-len(nonpass)}/{len(v)} PASS, {len(nonpass)} non-PASS "
        f"({len(regs)} NEW vs {args.c2_baseline})",
        {"pass": len(v) - len(nonpass), "total": len(v)},
        regs or ([f"baseline {args.c2_baseline} empty"] if not base else []))


def stage_s2(args, tasks) -> StageResult:
    bid = f"bs2{args.run_id}"
    rc, v, missing = _run_daemon("run", bid, tasks, "", args.stage_timeout)
    failed, reason = _daemon_guard(rc, missing, len(v))
    if failed:
        return StageResult("s2", bid, "FAIL", reason, {}, [reason])
    nonpass = [t for t, d in v.items()
               if str(d.get("verdict_status", "")).upper() != "PASS"]
    return StageResult(
        "s2", bid, "PASS" if not nonpass else "FAIL",
        f"{len(v)-len(nonpass)}/{len(v)} severance PASS (cert-stage also "
        f"attempted cost_model_xcheck + golden)",
        {"pass": len(v) - len(nonpass), "total": len(v)},
        [f"{t}: not PASS" for t in nonpass[:10]])


def stage_release_gate(args, _tasks) -> StageResult:
    cp = _run(["python3", "scripts/validate_tasks.py", "--release-gate"],
              timeout=600)
    out = (cp.stdout + cp.stderr).strip()
    ok = "release-gate OK" in out
    last = out.splitlines()[-1][:120] if out else "no output"
    return StageResult("release-gate", None, "PASS" if ok else "FAIL",
                       last, {}, [] if ok else [last])


# --- cleanup (snapshot-scoped) --------------------------------------------

def cleanup_pollution(snapshot: set, dry_run: bool) -> str:
    """Revert/remove ONLY the tasks/ churn THIS run introduced (paths absent
    from the start-of-run snapshot). Pre-existing dirty/untracked paths are
    never touched -- that is how a developer's own uncommitted work survives."""
    now = {ln[3:].strip() for ln in _tasks_dirty()}
    run_delta = now - snapshot
    if not run_delta:
        return "no run-introduced churn"
    untracked, modified = [], []
    for rel in sorted(run_delta):
        porc = _run(["git", "status", "--porcelain", rel]).stdout
        (untracked if porc.startswith("??") else modified).append(rel)
    dropped = []
    for rel in untracked:
        d = REPO / rel.rstrip("/")
        if d.is_dir() and not (d / "shapes.yaml").is_file():
            if not dry_run:
                shutil.rmtree(d, ignore_errors=True)
            dropped.append(d.name)
    if modified and not dry_run:
        cp = _run(["git", "checkout", "--", *modified])
        if cp.returncode != 0:
            raise RuntimeError(
                f"git checkout failed (rc={cp.returncode}): "
                f"{cp.stderr.strip()[:120]}")
    pre = "[dry-run] would revert" if dry_run else "reverted"
    return f"{pre} {len(modified)} run-churned meta, dropped {len(dropped)} " \
           f"stale untracked tasks ({', '.join(dropped) or 'none'})"


# --- report ---------------------------------------------------------------

def write_report(run_id, results, cleanup_msg, commit) -> Path:
    statuses = [r.status for r in results]
    overall = ("GREEN" if statuses and all(s == "PASS" for s in statuses)
               else "RED" if ("FAIL" in statuses or "ERROR" in statuses)
               else "AMBER")
    lines = [
        f"# Full GPU regression -- run {run_id}", "",
        f"**Overall: {overall}** | commit `{commit}` | "
        f"{statuses.count('PASS')}/{len(results)} PASS", "",
        "| Stage | Result | Batch | Summary | Regressions |",
        "|-------|--------|-------|---------|-------------|",
    ]
    for r in results:
        regs = ("; ".join(r.regressions)[:90] or "-") if r.regressions else "-"
        lines.append(f"| {r.name} | {r.status} | `{r.batch_id or '-'}` | "
                     f"{r.summary} | {regs} |")
    lines += ["", f"**Cleanup:** {cleanup_msg}", "",
              "Fresh batch-id per stage (preflight refuses a pre-existing "
              "results dir); a stage is PASS only on rc==0 + full task "
              "coverage. C2 verdicts read from results/<batch>/r10/."]
    out = REPO / "docs" / "reviews" / f"{run_id}-full-regression.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n")
    return out


# --- main -----------------------------------------------------------------

def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--run-id", required=True,
                   help="unique stamp for fresh batch-ids, e.g. 0607T0900")
    p.add_argument("--tasks-file", default=None)
    p.add_argument("--stages", default="unit,s2,c2,gpu-rank,release-gate")
    p.add_argument("--unit-baseline", type=int, default=121)
    p.add_argument("--prev-rank-batch", default="brankall6")
    p.add_argument("--c2-baseline", default="bc2apply")
    p.add_argument("--stage-timeout", type=int, default=9000)
    p.add_argument("--dry-run", action="store_true",
                   help="skip the destructive cleanup (report only)")
    args = p.parse_args(argv)

    if not re.fullmatch(r"[A-Za-z0-9]+", args.run_id):
        print("run-id must be alphanumeric (used verbatim in batch-id)",
              file=sys.stderr)
        return 2

    stages = [s.strip() for s in args.stages.split(",") if s.strip()]
    tasks = _load_tasks(args.tasks_file)

    # Preflight: refuse a dirty tasks/ tree (cleanup would clobber it) and refuse
    # any daemon stage whose results dir already exists (a reused batch-id would
    # serve stale verdicts -- the exact false-PASS this tool prevents).
    snapshot = {ln[3:].strip() for ln in _tasks_dirty()}
    if snapshot and "s2" in stages and not args.dry_run:
        print("REFUSING: tasks/ has uncommitted changes; cleanup after s2 would "
              "revert them. Commit/stash first, or pass --dry-run.",
              file=sys.stderr)
        for ln in list(snapshot)[:10]:
            print(f"  dirty: {ln}", file=sys.stderr)
        return 2
    for st, sub in (("gpu-rank", f"brank{args.run_id}"),
                    ("c2", f"bc2{args.run_id}"), ("s2", f"bs2{args.run_id}")):
        if st in stages and (RESULTS / sub).exists():
            print(f"REFUSING: results/{sub} already exists -- reusing run-id "
                  f"{args.run_id!r} would serve stale verdicts. Use a fresh "
                  f"run-id.", file=sys.stderr)
            return 2

    print(f"full regression run-id={args.run_id}: {len(tasks)} tasks, "
          f"stages={stages}")
    handlers = {"unit": stage_unit, "gpu-rank": stage_gpu_rank, "c2": stage_c2,
                "s2": stage_s2, "release-gate": stage_release_gate}
    results = []
    cleanup_msg = "skipped (no s2)"
    polluted = cleaned = False
    cleanup_ok = True

    def _cleanup():
        nonlocal cleanup_msg, cleaned, cleanup_ok
        try:
            cleanup_msg = cleanup_pollution(snapshot, args.dry_run)
        except Exception as e:
            cleanup_msg = (f"FAILED ({type(e).__name__}: {e}) -- tree may be "
                           "polluted; the release-gate result is unreliable")
            cleanup_ok = False
        cleaned = True
        print(f"    cleanup: {cleanup_msg}", flush=True)

    try:
        for s in stages:
            # s2 is the only pull-meta polluting stage. Clean its churn right
            # before the release gate reads the tree -- but AFTER gpu-rank, so
            # the ranking still saw S2's cert-stage-updated cost_model.status.
            if s == "release-gate" and polluted and not cleaned:
                _cleanup()
            print(f"=== stage: {s} ===", flush=True)
            try:
                h = handlers.get(s)
                r = (h(args, tasks) if h
                     else StageResult(s, None, "SKIPPED", "unknown stage"))
            except Exception as e:
                r = StageResult(s, None, "ERROR", f"{type(e).__name__}: {e}")
            results.append(r)
            print(f"    {r.status}: {r.summary}", flush=True)
            # Incremental report: refresh the md after every stage so partial
            # results are visible mid-run (cat the report any time) instead of
            # waiting for the full ~2-3h run to finish.
            _c = _run(["git", "rev-parse", "--short", "HEAD"]).stdout.strip()
            write_report(args.run_id, results, cleanup_msg, _c)
            if s == "s2":
                polluted = True
        if polluted and not cleaned:  # no release-gate stage -> clean at end
            _cleanup()
    finally:
        commit = _run(["git", "rev-parse", "--short", "HEAD"]).stdout.strip()
        report = write_report(args.run_id, results, cleanup_msg, commit)
        print(f"\nreport: {report}")
    return 0 if all(r.status == "PASS" for r in results) and cleanup_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
