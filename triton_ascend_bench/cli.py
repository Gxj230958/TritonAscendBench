"""cli.py -- `tab` argparse entrypoint.

Subcommands (all driven via `tab <subcommand> ...`):
    init-solution        scaffold an empty solution dir
    run                  evaluate one (task, solution) locally
    run-suite            evaluate many tasks
    package              tar.gz a solution dir into a submission
    submit               POST a package to the leaderboard
    validate-submission  validate a submission tar.gz against the schema
    validate-result      verify a previously emitted round_result.json
    static-check         run static_checker only on a submission
    analyze              compute final scores from verified-runner outputs

Only `tab` is exposed in pyproject.toml [project.scripts]; sub-aliases were
removed (per codex P0-2) -- use `tab <subcommand>` everywhere instead.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import tarfile
import tempfile
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(prog="tab", description=__doc__)
    parser.add_argument("--verbose", "-v", action="store_true")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_init = sub.add_parser("init-solution")
    p_init.add_argument("task_id")
    p_init.add_argument("--out", default=None,
                        help="solution dir to scaffold (default: solutions/<task_id>)")

    p_run = sub.add_parser("run", help="evaluate one (task, solution)")
    p_run.add_argument("task_id")
    p_run.add_argument("--solution", required=True)
    p_run.add_argument("--backend", default="npu", choices=["npu", "gpu"])
    p_run.add_argument("--out", default="round_result.json")

    p_suite = sub.add_parser("run-suite", help="evaluate many tasks")
    # codex round-7 P1: argparse `choices` rejects garbage like `abc` / `-1`.
    p_suite.add_argument(
        "--level", default="all",
        choices=["all", "L1", "L2", "L3", "1", "2", "3"],
    )
    p_suite.add_argument("--solutions-root", required=False)
    p_suite.add_argument("--submission", help="path to submission.tar.gz")
    p_suite.add_argument("--backend", default="npu", choices=["npu", "gpu"])
    p_suite.add_argument("--mode", default="self-reported",
                         choices=["self-reported", "verified"])
    p_suite.add_argument("--shapes-overlay",
                         help="hidden-shape overlay dir (verified mode)")
    p_suite.add_argument("--out", default="suite_results.json")

    p_pkg = sub.add_parser("package")
    p_pkg.add_argument("solution_dir")
    p_pkg.add_argument("--out", default="submission.tar.gz")

    p_sub = sub.add_parser("submit")
    p_sub.add_argument("package")
    p_sub.add_argument("--endpoint", required=True)

    p_vs = sub.add_parser("validate-submission")
    p_vs.add_argument("submission")
    p_vs.add_argument("--out")

    p_vr = sub.add_parser("validate-result")
    p_vr.add_argument("path")

    p_sc = sub.add_parser("static-check")
    p_sc.add_argument("--submission")
    p_sc.add_argument("--solution")
    p_sc.add_argument("--task-id")
    p_sc.add_argument("--out")

    p_an = sub.add_parser("analyze")
    p_an.add_argument("--verified-results", required=True)
    p_an.add_argument("--hidden-results")
    p_an.add_argument("--submitter")
    p_an.add_argument("--out", required=True)

    args = parser.parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    dispatch = {
        "init-solution":       _cmd_init_solution,
        "run":                 _cmd_run,
        "run-suite":           _cmd_run_suite,
        "package":             _cmd_package,
        "submit":              _cmd_submit,
        "validate-submission": _cmd_validate_submission,
        "validate-result":     _cmd_validate_result,
        "static-check":        _cmd_static_check,
        "analyze":             _cmd_analyze,
    }
    return dispatch[args.cmd](args)


def _cmd_init_solution(args) -> int:
    from .dataset import load_task

    task = load_task(args.task_id)
    out = Path(args.out) if args.out else Path(f"solutions/{args.task_id}")
    out.mkdir(parents=True, exist_ok=True)
    (out / "solution.py").write_text(
        f'"""Solution for {args.task_id}.\n\n'
        f"See tasks/level{task.level}/{args.task_id}/contract.md for the\n"
        f"required semantics, I/O dict keys, tolerance class and shape\n"
        f"coverage. The runner (process_isolation) calls the entrypoint\n"
        f"below in an isolated subprocess and compares every returned\n"
        f"tensor against the oracle with tolerance.compare.\n"
        f'"""\n'
        f"from __future__ import annotations\n\n"
        f"import torch\n"
        f"import triton\n"
        f"import triton.language as tl\n\n\n"
        f"# TODO: implement the NPU triton-ascend kernel(s) for this task.\n\n\n"
        f"def npu_triton_impl(inputs: dict) -> dict:\n"
        f'    """Entrypoint. `inputs` is built from shapes.yaml; return a\n'
        f"    dict of output tensors with the keys the contract specifies.\n"
        f'    """\n'
        f"    raise NotImplementedError(\n"
        f'        "implement npu_triton_impl for {args.task_id}; "\n'
        f'        "see contract.md"\n'
        f"    )\n"
    )
    print(f"scaffolded {out}/solution.py")
    return 0


def _cmd_run(args) -> int:
    from .eval import evaluate

    result = evaluate(args.task_id, Path(args.solution), backend=args.backend)
    out = {
        "task_id": result.task_id,
        "static_checks_passed": result.static_checks_passed,
        "aggregate_score": result.aggregate_score,
        "coverage": result.coverage,
        "rank_eligible": result.rank_eligible,
        "correctness_status": result.correctness_status,
        "runtime_status": result.runtime_status,
        "performance_status": result.performance_status,
        "performance_flags": result.performance_flags,
        "fallback_detected": result.fallback_detected,
        "findings": [
            {"layer": f.layer, "severity": f.severity, "message": f.message}
            for f in result.static_findings
        ],
    }
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(
        f"score={result.aggregate_score:.4f}  coverage={result.coverage:.2f}  "
        f"static_ok={result.static_checks_passed}  -> {args.out}"
    )
    return 0 if result.static_checks_passed else 1


def _cmd_run_suite(args) -> int:
    """Evaluate many tasks -> suite_results.json.

    Minimal v0.1: walks all tasks, looks up `solutions_root/<task_id>/solution.py`
    (or unpacks `submission.tar.gz` to a temp dir first), and calls
    `eval.evaluate` per task. Aggregates into per-task results.

    Returns 0 on completion (even with failed tasks); >0 only on
    catastrophic errors. The actual scoring is downstream in `analyze`.
    """
    from .dataset import iter_all_tasks
    from .eval import evaluate

    sol_root: Path | None = None
    tmp_root: tempfile.TemporaryDirectory | None = None
    try:
        if args.submission:
            tmp_root = tempfile.TemporaryDirectory(prefix="tab-suite-")
            try:
                sol_root = _unpack_submission(
                    Path(args.submission), Path(tmp_root.name)
                )
            except (tarfile.TarError, ValueError) as exc:
                payload = {
                    "error": "tar extraction failed",
                    "reason": str(exc),
                    "results": [],
                }
                Path(args.out).write_text(json.dumps(payload, indent=2))
                return 1
        elif args.solutions_root:
            sol_root = Path(args.solutions_root)
        else:
            print("--solutions-root or --submission required", file=sys.stderr)
            return 1

        per_task: list[dict] = []
        all_tasks = iter_all_tasks()
        if args.level and args.level != "all":
            try:
                want_level = int(str(args.level).lstrip("L"))
                all_tasks = [t for t in all_tasks if t.level == want_level]
            except ValueError:
                pass

        for task in all_tasks:
            sol_py = sol_root / task.task_id / "solution.py"
            if not sol_py.exists():
                per_task.append({
                    "task_id": task.task_id,
                    "skipped": True,
                    "reason": f"missing solution.py at {sol_py}",
                    "correctness_status": "error",
                    "runtime_status": "error",
                    "performance_status": "error",
                })
                continue
            try:
                r = evaluate(task.task_id, sol_py, backend=args.backend)
                per_task.append({
                    "task_id": r.task_id,
                    "static_checks_passed": r.static_checks_passed,
                    "aggregate_score": r.aggregate_score,
                    "coverage": r.coverage,
                    "rank_eligible": r.rank_eligible,
                    "correctness_status": r.correctness_status,
                    "runtime_status": r.runtime_status,
                    "performance_status": r.performance_status,
                    "performance_flags": r.performance_flags,
                    "fallback_detected": r.fallback_detected,
                    "findings": [
                        {"layer": f.layer, "severity": f.severity, "message": f.message}
                        for f in r.static_findings
                    ],
                })
            except Exception as exc:
                per_task.append({
                    "task_id": task.task_id,
                    "error": f"{type(exc).__name__}: {exc}",
                    "correctness_status": "error",
                })

        out_doc = {
            "mode": args.mode,
            "backend": args.backend,
            "shapes_overlay": args.shapes_overlay,
            "n_tasks": len(per_task),
            "results": per_task,
        }
        Path(args.out).write_text(json.dumps(out_doc, indent=2))
        print(f"run-suite: {len(per_task)} task(s) -> {args.out}")
        return 0
    finally:
        if tmp_root is not None:
            tmp_root.cleanup()


def _cmd_package(args) -> int:
    """Tar.gz a solution dir into `out`.

    Warns if the directory has no `manifest.yaml` -- the resulting tarball
    will not pass `validate-submission` (codex round-3 R-P0-2 follow-up).
    """
    src = Path(args.solution_dir)
    if not src.is_dir():
        print(f"not a directory: {src}", file=sys.stderr)
        return 1
    out = Path(args.out)
    with tarfile.open(out, "w:gz") as tf:
        tf.add(str(src), arcname=src.name)
    has_manifest = (src / "manifest.yaml").exists()
    has_solution = bool(list(src.rglob("solution.py")))
    print(f"packaged {src} -> {out}")
    if not has_manifest:
        print(
            f"  warning: no manifest.yaml in {src} -- submission will fail "
            "validate-submission. See manifest.SolutionManifest for required fields.",
            file=sys.stderr,
        )
    if not has_solution:
        print(
            f"  warning: no solution.py in {src} -- runner will skip all tasks.",
            file=sys.stderr,
        )
    return 0


def _cmd_submit(args) -> int:
    """POST a package to the leaderboard endpoint.

    codex round-4 P1: must NOT accept a submission that validate-submission
    would reject. We run the same validator first and refuse to ship if
    it fails.
    """
    pkg = Path(args.package)
    if not pkg.is_file():
        print(f"not a file: {pkg}", file=sys.stderr)
        return 1
    if not args.endpoint:
        print("--endpoint required", file=sys.stderr)
        return 1

    # Reuse the validate-submission codepath. Pass --out=None so the
    # JSON is suppressed; we only care about the exit semantics.
    class _Args:
        submission = str(pkg)
        out = None
    rc = _cmd_validate_submission(_Args())
    if rc != 0:
        print(
            f"submit: refusing to ship -- validate-submission failed (rc={rc}) "
            f"on {pkg}",
            file=sys.stderr,
        )
        return 1

    print(
        f"submit: validated {pkg} -> endpoint={args.endpoint} "
        "(actual HTTP POST is the leaderboard service's responsibility at v0.1)",
        file=sys.stderr,
    )
    return 0


def _cmd_validate_submission(args) -> int:
    """Validate a submission tarball: presence, shape, and manifest schema.

    Per codex round-3 R-P0-2: this command must actually parse manifest.yaml
    and reject malformed/incomplete manifests, not just verify file existence.
    """
    import yaml as _yaml

    from .manifest import SolutionManifest

    pkg = Path(args.submission)
    if not pkg.is_file():
        print(f"not a file: {pkg}", file=sys.stderr)
        return 1

    payload: dict = {"ok": False, "errors": []}
    with tempfile.TemporaryDirectory(prefix="tab-vs-") as tmp:
        try:
            unpack_dir = _unpack_submission(pkg, Path(tmp))
        except (tarfile.TarError, ValueError) as exc:
            payload["errors"].append(f"tar extraction failed: {exc}")
            _emit(payload, args.out)
            return 1

        # Look for manifest.yaml: either at unpack_dir or one level up.
        candidates = list(unpack_dir.rglob("manifest.yaml"))
        payload["manifest_present"] = bool(candidates)
        sol_files = list(unpack_dir.rglob("solution.py"))
        payload["solution_count"] = len(sol_files)

        if not candidates:
            payload["errors"].append("manifest.yaml not found in submission")
        else:
            doc: object | None
            try:
                doc = _yaml.safe_load(candidates[0].read_text())
            except _yaml.YAMLError as exc:
                payload["errors"].append(f"manifest.yaml parse error: {exc}")
                doc = None
            # codex round-4 REGRESSION-P0: empty / non-dict YAML must fail.
            if doc is None:
                payload["errors"].append(
                    "manifest.yaml is empty or yields no mapping"
                )
                payload["manifest_schema_ok"] = False
            elif not isinstance(doc, dict):
                payload["errors"].append(
                    f"manifest.yaml top-level must be a mapping, got "
                    f"{type(doc).__name__}"
                )
                payload["manifest_schema_ok"] = False
            else:
                try:
                    SolutionManifest.from_dict(doc)
                    payload["manifest_schema_ok"] = True
                except (KeyError, TypeError) as exc:
                    payload["errors"].append(
                        f"manifest.yaml schema error: missing/invalid {exc}"
                    )
                    payload["manifest_schema_ok"] = False

        if len(sol_files) == 0:
            payload["errors"].append("no solution.py inside submission")

        payload["ok"] = not payload["errors"]
        _emit(payload, args.out)
        return 0 if payload["ok"] else 1


def _cmd_validate_result(args) -> int:
    """Validate a round_result.json against the v0.1 schema.

    codex round-4 P1 + round-5 P1: must require all five status fields
    AND validate their enum values + container types per scoring_spec.md.
    """
    from .scoring import VALID_CORRECTNESS, VALID_PERFORMANCE, VALID_RUNTIME

    p = Path(args.path)
    if not p.exists():
        print(f"not found: {p}", file=sys.stderr)
        return 1
    doc = json.loads(p.read_text())
    if not isinstance(doc, dict):
        print(
            f"result must be a JSON object (mapping), got "
            f"{type(doc).__name__}",
            file=sys.stderr,
        )
        return 1
    required = {
        "task_id", "static_checks_passed", "aggregate_score",
        "correctness_status", "runtime_status", "performance_status",
        "performance_flags", "fallback_detected",
    }
    missing = required - set(doc.keys())
    if missing:
        print(f"missing keys: {sorted(missing)}", file=sys.stderr)
        return 1

    errors: list[str] = []
    # codex round-9 P1: task_id must be a non-empty string.
    tid = doc.get("task_id")
    if not isinstance(tid, str) or not tid:
        errors.append(f"task_id must be a non-empty string, got {tid!r}")
    # codex round-7 P1: full type checks. `bool` is a subclass of `int` so
    # we must reject `aggregate_score: true` and `static_checks_passed: 1`.
    score = doc.get("aggregate_score")
    if (
        isinstance(score, bool)
        or not isinstance(score, (int, float))
        or not math.isfinite(float(score))
    ):
        errors.append(
            f"aggregate_score must be a finite number, got {score!r}"
        )
    if not isinstance(doc.get("static_checks_passed"), bool):
        errors.append(
            "static_checks_passed must be a bool, got "
            f"{type(doc.get('static_checks_passed')).__name__}"
        )
    # codex round-7 P1: unhashable status values would crash `in {...}`
    # because we used `not in`. Guard with try/except.
    for field, valid in (
        ("correctness_status", VALID_CORRECTNESS),
        ("runtime_status", VALID_RUNTIME),
        ("performance_status", VALID_PERFORMANCE),
    ):
        val = doc.get(field)
        try:
            invalid = val not in valid
        except TypeError:
            invalid = True
        if invalid:
            errors.append(
                f"invalid {field}: {val!r} "
                f"(must be one of {sorted(valid)})"
            )
    flags = doc.get("performance_flags")
    if not isinstance(flags, list) or not all(isinstance(f, str) for f in flags):
        errors.append(
            "performance_flags must be a list of strings, got "
            f"{type(flags).__name__}"
        )
    if not isinstance(doc.get("fallback_detected"), bool):
        errors.append(
            "fallback_detected must be a bool, got "
            f"{type(doc.get('fallback_detected')).__name__}"
        )

    if errors:
        for e in errors:
            print(e, file=sys.stderr)
        return 1
    print(f"result OK: {doc.get('task_id')} score={doc.get('aggregate_score')}")
    return 0


def _cmd_static_check(args) -> int:
    from .static_checker import check_solution

    sol_sources: list[tuple[str, str]] = []        # (path-label, source)
    tmp: tempfile.TemporaryDirectory | None = None
    try:
        if args.solution:
            sol_sources.append((args.solution, Path(args.solution).read_text()))
        elif args.submission:
            tmp = tempfile.TemporaryDirectory(prefix="tab-sc-")
            try:
                unpack_dir = _unpack_submission(
                    Path(args.submission), Path(tmp.name)
                )
            except (tarfile.TarError, ValueError) as exc:
                payload = {
                    "any_blocked": True,
                    "error": "tar extraction failed",
                    "reason": str(exc),
                    "files": [],
                }
                _emit(payload, args.out)
                return 1
            for p in unpack_dir.rglob("solution.py"):
                sol_sources.append((str(p), p.read_text()))
            if not sol_sources:
                payload = {
                    "any_blocked": True,
                    "error": "no solution.py inside submission",
                    "files": [],
                }
                _emit(payload, args.out)
                return 1
        else:
            print("--solution or --submission required", file=sys.stderr)
            return 1

        required: list[str] = []
        if args.task_id:
            from .dataset import load_task

            required = load_task(args.task_id).meta.get(
                "runner_checks", {}
            ).get("static_checks", [])

        per_file: list[dict] = []
        any_blocked = False
        for label, src in sol_sources:
            report = check_solution(src, required_checks=required)
            any_blocked = any_blocked or report.blocked
            per_file.append({
                "path": label,
                "blocked": report.blocked,
                "findings": [
                    {"layer": f.layer, "severity": f.severity,
                     "message": f.message, "line": f.line}
                    for f in report.findings
                ],
            })

        payload = {"any_blocked": any_blocked, "files": per_file}
        _emit(payload, args.out)
        return 0 if not any_blocked else 1
    finally:
        if tmp is not None:
            tmp.cleanup()


def _cmd_analyze(args) -> int:
    """Read run-suite output -> compute final scores -> score_report.json.

    codex round-3 N-P0-1 hardening:
    - Reads explicit `correctness_status` / `runtime_status` /
      `performance_status` / `fallback_detected` fields from each result.
    - Does NOT infer correctness from `static_checks_passed` (static check
      proves the code is well-formed, not that it produces correct outputs).
    - The scored catalogue is ALL tasks (this is a GPU-triton  ->
      NPU-triton migration benchmark; there is no AscendC-native
      subset). A result whose task_id is not in the catalogue is
      ignored -- it cannot inflate the score (denominator is the
      catalogue size). `*_native` metric names are retained for output
      schema stability and now denote the NPU-triton track.
    """
    from .dataset import iter_all_tasks
    from .scoring import (
        EPSILON,
        P_CAP,
        TaskMeasurement,
        aggregate_perf,
        native_coverage,
    )

    verified = json.loads(Path(args.verified_results).read_text())
    if not isinstance(verified, dict):
        payload = {
            "error": "verified-results must be a JSON object",
            "got_type": type(verified).__name__,
        }
        Path(args.out).write_text(json.dumps(payload, indent=2))
        return 1
    hidden = (
        json.loads(Path(args.hidden_results).read_text())
        if args.hidden_results else {"results": []}
    )
    if not isinstance(hidden, dict):
        hidden = {"results": []}

    # codex round-10 P1 #3: results may be None / scalar -- coerce safely.
    raw_results = verified.get("results")
    if not isinstance(raw_results, list):
        verified = {**verified, "results": []}

    # The scored catalogue is the full task set (no AscendC-native
    # subset post-removal). The scored denominator is the
    # *score-eligible* set, frozen from trusted repo metadata (codex
    # P0-2/P0-3): a task counts only if it has a real working oracle
    # (not a NotImplementedError stub) and is not PHASE-C-MANUAL.
    # Counting stub/manual tasks as failures would systematically
    # depress valid submissions. The submission JSON cannot influence
    # this set.
    # Single source of truth: triton_ascend_bench.eligibility (plan U7,
    # shared with scripts/validate_tasks.py to prevent drift).
    from .eligibility import compute_score_eligible

    all_tasks = iter_all_tasks()
    catalogue_ids = {t.task_id for t in all_tasks}
    score_eligible_ids = compute_score_eligible(all_tasks)
    s_native_size = len(score_eligible_ids)

    # Phase A' S6 release gate (plan U7 / R6): if ANY score-eligible
    # task has `phase_a_prime.status != PASS`, the benchmark is NOT
    # releasable. The diagnostic score still computes (C2: coherent
    # with perf_status=unavailable) -- only the *releasable leaderboard*
    # is refused.
    release_blocking: list[str] = []
    for t in all_tasks:
        if t.task_id not in score_eligible_ids:
            continue
        pap = (t.meta or {}).get("phase_a_prime") or {}
        if pap.get("status") != "PASS":
            release_blocking.append(t.task_id)
    release_blocking.sort()

    # Build TaskMeasurement only for results whose task_id is in the
    # catalogue. The submission JSON is input, not authority: a result
    # for an unknown task_id is ignored (cannot inflate the score --
    # denominator is the catalogue size, not the result count).
    task_measurements: list[TaskMeasurement] = []
    seen_ids: set[str] = set()
    non_catalogue_count = 0
    non_eligible_count = 0
    missing_task_id_count = 0
    duplicate_task_id_count = 0
    for r in verified.get("results", []):
        # codex round-9 P1: result element must be a dict, not e.g. int.
        if not isinstance(r, dict):
            missing_task_id_count += 1
            continue
        tid = r.get("task_id")
        if not tid or not isinstance(tid, str):
            missing_task_id_count += 1
            continue
        if tid not in catalogue_ids:
            non_catalogue_count += 1
            continue
        if tid not in score_eligible_ids:
            # known task but not score-eligible (stub / manual oracle):
            # excluded from the denominator, not counted as a failure.
            non_eligible_count += 1
            continue
        if tid in seen_ids:
            # codex round-16/17 P1: duplicate worst-case propagation
            # covers ALL status fields, not just correctness/aggregate.
            duplicate_task_id_count += 1
            for prev in task_measurements:
                if prev.task_id != tid:
                    continue
                new_corr = r.get("correctness_status", "error")
                if new_corr != "ok" and prev.correctness_status == "ok":
                    prev.correctness_status = new_corr
                new_rt = r.get("runtime_status", "error")
                if new_rt != "ok" and prev.runtime_status == "ok":
                    prev.runtime_status = new_rt
                new_pf = r.get("performance_status", "error")
                if new_pf != "ok" and prev.performance_status == "ok":
                    prev.performance_status = new_pf
                if r.get("fallback_detected"):
                    prev.fallback_detected = True
                new_score = r.get("aggregate_score")
                if (
                    isinstance(new_score, (int, float))
                    and math.isfinite(float(new_score))
                    and float(new_score) < (prev.baseline_ns or 0.0)
                ):
                    prev.baseline_ns = float(new_score)
                break
            continue
        seen_ids.add(tid)

        s = r.get("aggregate_score", EPSILON)
        # codex round-6 #5: reject non-finite scores at the analyze boundary.
        if isinstance(s, (int, float)) and math.isfinite(float(s)):
            baseline = float(s)
        else:
            baseline = EPSILON
        # codex round-11 P1: performance_flags must be list[str]; non-list
        # / non-string elements are coerced to [].
        raw_flags = r.get("performance_flags")
        if isinstance(raw_flags, list) and all(isinstance(f, str) for f in raw_flags):
            flags = list(raw_flags)
        else:
            flags = []
        # s_t = baseline / measured -> set measured=1 so baseline maps to s_t.
        task_measurements.append(TaskMeasurement(
            task_id=tid,
            correctness_status=r.get("correctness_status", "error"),
            runtime_status=r.get("runtime_status", "error"),
            performance_status=r.get("performance_status", "error"),
            performance_flags=flags,
            fallback_detected=bool(r.get("fallback_detected", False)),
            measured_ns=1.0,
            baseline_ns=baseline,
        ))

    # Denominator = score-eligible set (frozen from repo metadata).
    # Fall back to measurement count only when there is no eligible set
    # (CI smoke with synthetic task_ids).
    effective_denom = s_native_size if s_native_size > 0 else len(task_measurements)
    # Honest perf aggregate: returns perf_status=unavailable + null
    # metrics when no task has a real baseline (codex P0-1) instead of
    # a fabricated ε-collapsed number.
    perf = aggregate_perf(task_measurements, eligible_size=effective_denom)
    score_doc = {
        "submitter": args.submitter,
        "n_score_eligible_tasks": s_native_size,
        "n_native_measurements": len(task_measurements),
        "non_catalogue_task_count": non_catalogue_count,
        "non_eligible_task_count": non_eligible_count,
        "missing_task_id_count": missing_task_id_count,
        "duplicate_task_id_count": duplicate_task_id_count,
        "perf_status": perf["perf_status"],
        "perf_status_reason": perf["perf_status_reason"],
        "capped_geomean_native": perf["capped_geomean_native"],
        "fastp_native_at_1_0": perf["fastp_native_at_1_0"],
        "fastp_native_at_1_5": perf["fastp_native_at_1_5"],
        "native_coverage": native_coverage(
            task_measurements, s_native_size=effective_denom
        ),
        "p_cap": P_CAP,
        "epsilon": EPSILON,
        "hidden_results_n": len(hidden.get("results") if isinstance(hidden.get("results"), list) else []),
        # Phase A' S6 (plan U7 / C2): release_blocked and
        # perf_status=unavailable coexist coherently -- the diagnostic
        # score is still emitted; only the releasable leaderboard is
        # refused while any eligible task is not PASS.
        "release_blocked": bool(release_blocking),
        "release_blocked_tasks": release_blocking,
    }
    Path(args.out).write_text(json.dumps(score_doc, indent=2))
    print(
        f"analyze: {len(task_measurements)} measurement(s) of "
        f"{effective_denom} score-eligible; perf_status={perf['perf_status']}"
        f" -> {args.out}"
    )
    return 0


# ─────────────────────────────────────────────────────────────────────────────
# helpers
# ─────────────────────────────────────────────────────────────────────────────
def _unpack_submission(pkg: Path, dest: Path) -> Path:
    """Unpack tarball; return the directory holding `solution.py` files.

    Defenses:
    - Reject path-traversal entry names (absolute or with `..`).
    - codex round-7 P1: reject any symlink/hardlink entry outright.
    - tarfile `filter="data"` blocks symlink-out at extraction time as a
      second line of defense.
    """
    with tarfile.open(pkg, "r:gz") as tf:
        for m in tf.getmembers():
            if m.name.startswith("/") or ".." in Path(m.name).parts:
                raise ValueError(f"unsafe tar entry: {m.name!r}")
            if m.issym() or m.islnk():
                raise ValueError(
                    f"symlink/hardlink entries not allowed in submission: "
                    f"{m.name!r} -> {m.linkname!r}"
                )
            if m.isdev() or m.ischr() or m.isblk() or m.isfifo():
                raise ValueError(
                    f"device/fifo entries not allowed in submission: {m.name!r}"
                )
        tf.extractall(dest, filter="data")
    entries = [p for p in dest.iterdir() if p.is_dir()]
    return entries[0] if len(entries) == 1 else dest


def _emit(payload: dict, out_path: str | None) -> None:
    text = json.dumps(payload, indent=2)
    if out_path:
        Path(out_path).write_text(text)
    else:
        print(text)


if __name__ == "__main__":
    sys.exit(main())
