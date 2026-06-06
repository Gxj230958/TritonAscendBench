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

logger = logging.getLogger(__name__)


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
    # W2.1: single-task `tab run` is the bring-up / smoke entrypoint and must
    # be able to run verified mode so W2.1 / W4.3 single-task tests can grade
    # hidden shapes. self-reported grades public shapes only; verified also
    # grades the task's sealed `shapes.yaml.hidden_shapes` section.
    p_run.add_argument("--mode", default="self-reported",
                       choices=["self-reported", "verified"])
    p_run.add_argument("--peak-table", default=None,
                       help="peak table TOML (default 910B; H200 self-check "
                            "passes infra/peak_table_h200.toml)")
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
    # W2.1: `--shapes-overlay` is DEPRECATED and has no effect. The hidden
    # shapes' single source of truth is each task's own
    # `shapes.yaml.hidden_shapes` section (DESIGN §3.3 / §5.2); verified mode
    # reads hidden shapes from there. The flag is kept only so old invocations
    # do not hard-error on an unknown argument — any value is ignored.
    p_suite.add_argument("--shapes-overlay",
                         help=argparse.SUPPRESS)
    p_suite.add_argument("--peak-table", default=None,
                         help="peak table TOML (default 910B; H200 self-check "
                              "passes infra/peak_table_h200.toml)")
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
    p_an.add_argument("--peak-table", default=None,
                      help="peak table TOML for authoritative roofline/absolute "
                           "checks (default 910B)")
    p_an.add_argument("--out", required=True)

    # V9 plan M-CLI: provenance pipeline subcommands. Each is a thin
    # dispatcher that calls the underlying script's ``main(argv)`` so
    # the CLI surface and the script surface stay in lock-step.
    p_census = sub.add_parser(
        "census", help="static source provenance census (M0a.1)"
    )
    p_census.add_argument("--task")
    p_census.add_argument("--task-list")
    p_census.add_argument("--all", action="store_true")
    p_census.add_argument("--report")
    p_census.add_argument("--json-out")

    p_pyref = sub.add_parser(
        "oracle-pyref-precompute",
        help="dev/CPU oracle_pytorch ref precompute (M2.1a)",
    )
    p_pyref.add_argument("--task")
    p_pyref.add_argument("--all", action="store_true")
    p_pyref.add_argument("--workers", type=int)
    p_pyref.add_argument("--report")
    p_pyref.add_argument("--json-out")

    p_apply = sub.add_parser(
        "apply-equivalence-verdict",
        help="dev sync of GPU R10 verdict into meta.yaml (M2.1c)",
    )
    p_apply.add_argument("--verdict-dir")
    p_apply.add_argument("--report")
    p_apply.add_argument("--dry-run", action="store_true")

    p_r12 = sub.add_parser(
        "verify-commit-chain",
        help="R12 SHA-chain integrity check (M2.0.5)",
    )
    p_r12.add_argument("--task")
    p_r12.add_argument("--task-list")
    p_r12.add_argument("--all", action="store_true")
    p_r12.add_argument("--report")
    p_r12.add_argument("--json-out")

    p_r13c = sub.add_parser(
        "r13c-gate",
        help="R13c go/no-go release planning gate (M0a.2)",
    )
    p_r13c.add_argument("--json-out")

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
        # V9 plan M-CLI
        "census":                       _cmd_census,
        "oracle-pyref-precompute":      _cmd_oracle_pyref_precompute,
        "apply-equivalence-verdict":    _cmd_apply_equivalence_verdict,
        "verify-commit-chain":          _cmd_verify_commit_chain,
        "r13c-gate":                    _cmd_r13c_gate,
    }
    return dispatch[args.cmd](args)


def _argv_from(args, *keys: str) -> list[str]:
    """Materialise ``args`` into a flat ``--key value`` argv list,
    skipping ``None`` / ``False`` entries so the underlying script's
    argparse sees only what the user actually passed.
    """
    out: list[str] = []
    for k in keys:
        v = getattr(args, k.replace("-", "_"), None)
        if v is None or v is False:
            continue
        if v is True:
            out.append(f"--{k}")
            continue
        out.extend([f"--{k}", str(v)])
    return out


def _cmd_census(args) -> int:
    from pathlib import Path as _P
    import sys as _sys
    _sys.path.insert(0, str(_P(__file__).resolve().parent.parent))
    from scripts.source_provenance_census import main as _main  # noqa: PLC0415

    return _main(_argv_from(args, "task", "task-list", "all", "report", "json-out"))


def _cmd_oracle_pyref_precompute(args) -> int:
    from pathlib import Path as _P
    import sys as _sys
    _sys.path.insert(0, str(_P(__file__).resolve().parent.parent))
    from scripts.oracle_pyref_precompute import main as _main  # noqa: PLC0415

    return _main(
        _argv_from(args, "task", "all", "workers", "report", "json-out")
    )


def _cmd_apply_equivalence_verdict(args) -> int:
    from pathlib import Path as _P
    import sys as _sys
    _sys.path.insert(0, str(_P(__file__).resolve().parent.parent))
    from scripts.apply_oracle_equivalence_verdict import main as _main  # noqa: PLC0415

    return _main(
        _argv_from(args, "verdict-dir", "report", "dry-run")
    )


def _cmd_verify_commit_chain(args) -> int:
    from pathlib import Path as _P
    import sys as _sys
    _sys.path.insert(0, str(_P(__file__).resolve().parent.parent))
    from scripts.verify_upstream_commit_chain import main as _main  # noqa: PLC0415

    return _main(
        _argv_from(args, "task", "task-list", "all", "report", "json-out")
    )


def _cmd_r13c_gate(args) -> int:
    from pathlib import Path as _P
    import sys as _sys
    _sys.path.insert(0, str(_P(__file__).resolve().parent.parent))
    from scripts.provenance_release_planning_gate import main as _main  # noqa: PLC0415

    return _main(_argv_from(args, "json-out"))


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


def _metamorphic_result_json(m) -> dict:
    """Serialise one MetamorphicResult for a round_result.json (W2.4).

    Metamorphic results live under each shape's ``metamorphic_results``
    array — distinct from the submission-level ``canary_findings`` array.
    """
    return {
        "name": m.name,
        "passed": m.passed,
        "detail": m.detail,
    }


def _shape_result_json(r) -> dict:
    """Serialise one ShapeRunResult for a round_result.json (W1.4/W2.4).

    Discloses the Roofline provenance (t_ref / t_npu / per-class W /
    per-resource M / efficiency), the durations-derived stats
    (median / p99 / cv / trials) DESIGN §6.6 requires, and the W2.4
    per-shape ``metamorphic_results`` (the metamorphic detection path —
    canary findings are emitted separately at submission level).
    """
    return {
        "shape_id": r.shape_id,
        "correctness_passed": r.correctness_passed,
        "correctness_reason": r.correctness_reason,
        # baseline_ns/measured_ns are ns. baseline_ns = t_ref(s)->ns
        # (set in _apply_roofline); the raw t_ref disclosure stays in seconds.
        "baseline_ns": r.baseline_ns,
        "measured_ns": r.t_npu,
        "t_ref": r.t_ref,
        "t_npu": r.t_npu,
        "efficiency": r.efficiency,
        "w_per_class": dict(r.w_per_class),
        "m_per_resource": dict(r.m_per_resource),
        "median": r.median,
        "p99": r.p99,
        "cv": r.cv,
        "trials": r.trials,
        "roofline_error": r.roofline_error,
        # W2.4: metamorphic path — distinct schema from canary findings.
        "metamorphic_results": [
            _metamorphic_result_json(m) for m in r.metamorphic_results
        ],
    }


def _round_result_json(result, canary_findings=None) -> dict:
    """Assemble the round_result.json payload from a TaskEvalResult (W1.4/W2.4).

    Emits the task-level Roofline ``s_t``, the DESIGN §9 provenance pair
    (``baseline_ns`` = t_ref / ``measured_ns`` = t_npu), the orthogonal
    ``task_status`` field, per-shape Roofline + timing detail with each
    shape's W2.4 ``metamorphic_results``, and a submission-level
    ``canary_findings`` array (distinct schema; empty for a real task —
    canary traps are separate ``_canary`` tasks, W2.5).
    """
    tm = result.task_measurement()
    from .adversarial import canary_finding_json

    return {
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
        # W1.4: task-level Roofline score + provenance + evaluability.
        "s_t": result.s_t,
        "baseline_ns": tm.baseline_ns,
        "measured_ns": tm.measured_ns,
        "task_status": result.task_status,
        # R9-1: diagnostic — whether the Roofline benchmark data was
        # complete for this task. `analyze` recomputes this authoritatively
        # from repo metadata, so the emitted value is informational only.
        "roofline_data_complete": result.roofline_data_complete,
        # Whether every referenced peak is absolute_verified (gates absolute
        # efficiency claims). Recomputed authoritatively by `analyze`.
        "absolute_verified": result.absolute_verified,
        "shape_results": [_shape_result_json(r) for r in result.shape_results],
        # W2.4: canary findings — their OWN schema, never folded into the
        # per-shape metamorphic_results.
        "canary_findings": [
            canary_finding_json(c) for c in (canary_findings or [])
        ],
        "findings": [
            {"layer": f.layer, "severity": f.severity, "message": f.message}
            for f in result.static_findings
        ],
    }


def _cmd_run(args) -> int:
    from .eval import evaluate

    # W2.1: thread `--mode` through. verified grades the task's sealed
    # hidden shapes alongside the public ones; self-reported grades public
    # only.
    result = evaluate(args.task_id, Path(args.solution),
                      backend=args.backend, mode=args.mode,
                      peak_table_path=args.peak_table)
    out = _round_result_json(result)
    Path(args.out).write_text(json.dumps(out, indent=2))
    s_t_str = f"{result.s_t:.4f}" if result.s_t is not None else "n/a"
    print(
        f"score={result.aggregate_score:.4f}  s_t={s_t_str}  "
        f"coverage={result.coverage:.2f}  mode={args.mode}  "
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
                # W2.1: `--mode` decides whether hidden shapes are graded.
                r = evaluate(task.task_id, sol_py, backend=args.backend,
                             mode=args.mode, peak_table_path=args.peak_table)
                # W1.4: same round_result schema as `tab run` — task-level
                # s_t + baseline_ns/measured_ns provenance + per-shape detail.
                per_task.append(_round_result_json(r))
            except Exception as exc:
                per_task.append({
                    "task_id": task.task_id,
                    "error": f"{type(exc).__name__}: {exc}",
                    "correctness_status": "error",
                })

        # W2.5: run the dedicated canary suite. Canary trap tasks live under
        # `tasks/_canary/`, which `iter_all_tasks()` skips — they never enter
        # `per_task` / the scored denominator. The independent enumerator
        # `iter_canary_tasks()` discovers them; each canary is run with the
        # submission's solution for its bound `source_task_id` and judged by
        # `check_canaries`. The findings join the W2.4 escalation chain below.
        from .adversarial import canary_finding_json, run_canary_suite

        try:
            suite_canary_findings = run_canary_suite(
                sol_root, backend=args.backend)
        except Exception as exc:
            logger.warning("canary suite error: %s: %s",
                           type(exc).__name__, exc)
            suite_canary_findings = []

        # W2.4: submission-level canary escalation. Pool every (real-task)
        # result's `canary_findings` plus the dedicated canary-suite findings
        # and aggregate — `canary_caught > 0` flips the whole submission to
        # `suspicious` for manual review (DESIGN §6.8).
        canary = _aggregate_suite_canaries(
            per_task,
            extra_findings=[canary_finding_json(f) for f in suite_canary_findings],
        )

        # W2.1: the deprecated `--shapes-overlay` flag is intentionally NOT
        # emitted — hidden shapes are sealed inside each task's shapes.yaml,
        # there is no overlay artifact in the suite contract.
        out_doc = {
            "mode": args.mode,
            "backend": args.backend,
            "n_tasks": len(per_task),
            "results": per_task,
            # W2.4 canary escalation chain.
            "canary_caught": canary["canary_caught"],
            "suspicious": canary["suspicious"],
            "canary_findings": canary["canary_findings"],
        }
        Path(args.out).write_text(json.dumps(out_doc, indent=2))
        susp = " SUSPICIOUS" if canary["suspicious"] else ""
        print(f"run-suite: {len(per_task)} task(s){susp} -> {args.out}")
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
    """Validate a round_result.json against the schema.

    codex round-4 P1 + round-5 P1: must require all five status fields
    AND validate their enum values + container types per scoring_spec.md.

    W1.4 schema migration: accepts the new producer fields — task-level
    ``s_t``, the DESIGN §9 provenance pair ``baseline_ns`` / ``measured_ns``,
    the orthogonal ``task_status`` field, and per-shape ``shape_results``
    (each with Roofline ``t_ref`` / ``t_npu`` / ``efficiency`` and the
    ``median`` / ``p99`` / ``cv`` / ``trials`` stats). When
    ``task_status != "evaluated"`` the task carries no valid perf data —
    ``s_t`` / ``baseline_ns`` / ``measured_ns`` and the per-shape timing
    detail are allowed to be ``null`` and only ``task_status`` is checked.
    """
    from .scoring import (
        VALID_CORRECTNESS,
        VALID_PERFORMANCE,
        VALID_RUNTIME,
        VALID_TASK_STATUS,
    )

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

    # W1.4: task_status is optional for back-compat (older results predate
    # it); when present it must be a valid enum. Default to "evaluated".
    task_status = doc.get("task_status", "evaluated")
    if task_status not in VALID_TASK_STATUS:
        errors.append(
            f"invalid task_status: {task_status!r} "
            f"(must be one of {sorted(VALID_TASK_STATUS)})"
        )
        task_status = "evaluated"   # continue checks under a sane default

    # W1.4: optional numeric perf fields. They may be a finite number or
    # null — for an "evaluated" task null means timing was unavailable; for
    # unsupported/blocked null is the expected "no valid perf data" state.
    # Either way the validator only type-checks the non-null case.
    def _check_optional_number(name: str) -> None:
        val = doc.get(name)
        if val is None:
            return                          # null allowed (any task_status)
        if (isinstance(val, bool)
                or not isinstance(val, (int, float))
                or not math.isfinite(float(val))):
            errors.append(
                f"{name} must be a finite number or null, got {val!r}"
            )

    for _num in ("s_t", "baseline_ns", "measured_ns"):
        if _num in doc:
            _check_optional_number(_num)

    # W1.4: per-shape detail — a list of objects when present.
    shape_results = doc.get("shape_results")
    if shape_results is not None and not isinstance(shape_results, list):
        errors.append(
            f"shape_results must be a list, got {type(shape_results).__name__}"
        )
    elif isinstance(shape_results, list):
        for i, sr in enumerate(shape_results):
            if not isinstance(sr, dict):
                errors.append(f"shape_results[{i}] must be an object")
                continue
            for _sf in ("t_ref", "t_npu", "efficiency", "median", "p99",
                        "cv"):
                v = sr.get(_sf)
                if v is None:
                    continue          # null allowed (timing may be absent)
                if (isinstance(v, bool)
                        or not isinstance(v, (int, float))
                        or not math.isfinite(float(v))):
                    errors.append(
                        f"shape_results[{i}].{_sf} must be a finite "
                        f"number or null, got {v!r}"
                    )
            trials = sr.get("trials")
            if trials is not None and (
                isinstance(trials, bool) or not isinstance(trials, int)
            ):
                errors.append(
                    f"shape_results[{i}].trials must be an int or null"
                )
            # W2.4: per-shape metamorphic_results — a list of objects.
            mr = sr.get("metamorphic_results")
            if mr is not None and not isinstance(mr, list):
                errors.append(
                    f"shape_results[{i}].metamorphic_results must be a list"
                )

    # W2.4: submission-level canary findings — a list of objects, kept
    # distinct from per-shape metamorphic_results. `canary_caught` /
    # `suspicious` must be consistent when present.
    canary_findings = doc.get("canary_findings")
    if canary_findings is not None and not isinstance(canary_findings, list):
        errors.append("canary_findings must be a list, got "
                       f"{type(canary_findings).__name__}")
    cc = doc.get("canary_caught")
    if cc is not None and (isinstance(cc, bool) or not isinstance(cc, int)):
        errors.append(f"canary_caught must be an int or null, got {cc!r}")
    susp = doc.get("suspicious")
    if susp is not None and not isinstance(susp, bool):
        errors.append(f"suspicious must be a bool or null, got {susp!r}")
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

    R2-4 (W4.3 ④ -- denominator authority): the per-task ``task_status``
    is taken from TRUSTED REPO METADATA, never verbatim from the result.
    The suite JSON's top-level ``mode`` is the sole authority:

    * ``verified``      -- the denominator is the fixed score-eligible
      set ``|S|``; every task is ``task_status = "evaluated"`` (verified
      mode has provider golden for every task, incl. the
      ``upstream_oracle_required`` ones).
    * ``self-reported`` -- the denominator is the score-eligible set
      MINUS the fixed ``upstream_oracle_required`` set (no provider
      golden in this mode); those tasks are forced to
      ``task_status = "unsupported"``, the rest to ``"evaluated"``.

    Both the score-eligible set and the ``upstream_oracle_required`` set
    are derived from repo ``meta`` and frozen — they do not shrink/grow
    with the results' contents. A result's own ``task_status`` is only
    cross-checked for consistency and reported as a mismatch count; it
    never moves a task into or out of the denominator. This closes the
    cheat where a hand-edited result marks a failing task ``unsupported``
    to drop it from coverage + geomean.
    """
    from .dataset import iter_all_tasks
    from .eval import _PERFORMANCE_STATUS_PRECEDENCE
    from .scoring import (
        EPSILON,
        P_CAP,
        TASK_STATUS_EVALUATED,
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
    from .eligibility import (
        compute_score_eligible,
        compute_upstream_oracle_required,
        compute_verified_blocked,
    )

    all_tasks = iter_all_tasks()
    catalogue_ids = {t.task_id for t in all_tasks}
    score_eligible_ids = compute_score_eligible(all_tasks)

    # R2-4 (W4.3 ④): the suite JSON's TOP-LEVEL `mode` is the sole
    # denominator authority. An invalid / missing `mode` defaults to
    # "self-reported" — the stricter denominator (it excludes the
    # upstream-oracle tasks), so a malformed suite doc cannot widen the
    # scored set.
    mode = verified.get("mode")
    if mode not in ("self-reported", "verified"):
        mode = "self-reported"

    # The upstream-oracle-required set is frozen from repo `meta` —
    # NEVER from the results. In self-reported mode these tasks have no
    # provider golden, so they are excluded from that mode's denominator
    # and forced to task_status="unsupported"; in verified mode they have
    # offline golden and score normally (task_status="evaluated"), unless
    # the golden artifact set is missing/incomplete (R3-2 — "blocked").
    upstream_oracle_ids = compute_upstream_oracle_required(all_tasks)
    if mode == "self-reported":
        scored_denominator_ids = score_eligible_ids - upstream_oracle_ids
    else:
        # R2-4: verified-mode denominator is the fixed |S| — a "blocked"
        # task_status does NOT move a task out of the scored set.
        scored_denominator_ids = set(score_eligible_ids)
    s_native_size = len(scored_denominator_ids)

    # R3-2: in verified mode, an `upstream_oracle_required` task whose
    # offline `golden/` artifacts are missing/incomplete is `blocked`.
    # Derived purely from trusted repo state (the presence of the golden
    # files), never from the result JSON. Computed only for verified mode
    # (self-reported never inspects golden/).
    verified_blocked_ids: set[str] = (
        compute_verified_blocked(all_tasks) if mode == "verified" else set()
    )

    def _param_constraints_failed(tid: str) -> bool:
        """True iff `tid`'s `param_constraints` fails the shared WA.4 helper.

        Derived purely from trusted repo state (`meta.param_constraints` +
        the task's `shapes.yaml`) — never from the result JSON. A missing /
        `scaffold: true` / schema-illegal / semantically-violated block is a
        failure. A task absent from the repo (no `task_dir`) is treated as
        not-failed here so the existing self-reported / verified verdicts
        still own its status.
        """
        import yaml

        from . import param_constraints as _pc

        task = task_by_id.get(tid)
        if task is None:
            return False
        shapes_doc: dict | None = None
        shapes_path = task.task_dir / "shapes.yaml"
        try:
            if shapes_path.is_file():
                loaded = yaml.safe_load(shapes_path.read_text())
                if isinstance(loaded, dict):
                    shapes_doc = loaded
        except (OSError, yaml.YAMLError) as e:  # noqa: BLE001
            logger.warning("param_constraints gate: shapes.yaml unreadable "
                            "for %s: %s", tid, e)
        result = _pc.check_param_constraints(
            getattr(task, "param_constraints", None), shapes_doc, tid=tid)
        return not result.ok

    def _authoritative_task_status(tid: str) -> str:
        """The repo-derived task_status for `tid` under the suite's mode.

        WA.4 ②(b): the SINGLE authoritative arbiter — all `task_status`
        verdicts are decided here, from trusted repo state (meta + golden
        artifacts + `param_constraints`), never from the result JSON. The
        fixed priority (high → low) is delegated to the shared
        `param_constraints.resolve_task_status`:

        1. `is_canary` ⇒ "evaluated" (exempt from the param_constraints gate);
        2. self-reported `upstream_oracle_required` ⇒ "unsupported";
        3. verified-mode incomplete offline golden ⇒ "blocked";
        4. `param_constraints` fails the shared helper ⇒ "blocked"
           (stays in the fixed denominator, scored as ε — not dropped);
        5. otherwise ⇒ "evaluated".
        """
        from . import param_constraints as _pc

        task = task_by_id.get(tid)
        is_canary = bool(
            (getattr(task, "meta", None) or {}).get("is_canary") is True)
        return _pc.resolve_task_status(
            is_canary=is_canary,
            self_reported_unsupported=(
                mode == "self-reported" and tid in upstream_oracle_ids),
            verified_golden_incomplete=(
                mode == "verified" and tid in verified_blocked_ids),
            param_constraints_failed=_param_constraints_failed(tid),
        )

    # R9-1: a task's Roofline data is "complete" iff repo state can score
    # it — it has a `cost_model` AND every π/β that cost model references
    # is verified (positive-finite) in the 910B peak table. This is
    # derived purely from trusted repo metadata (`meta.cost_model` + the
    # `infra/` peak table), NEVER from the result JSON: a submitter must
    # not be able to mark its own task "data-incomplete" to escape the ε
    # denominator (no-subset-escape). When the peak table itself cannot
    # be loaded, no task is scorable — every task is data-incomplete.
    task_by_id = {t.task_id: t for t in all_tasks}
    try:
        from .peak_table import load_peak_provenance, load_peak_table

        # Optional flag (default None → built-in table); tolerate an args object
        # that predates it rather than failing the whole peak load.
        _peak_table_path = getattr(args, "peak_table", None)
        _compute_peaks, _resource_peaks = load_peak_table(_peak_table_path)
        _compute_prov, _resource_prov = load_peak_provenance(_peak_table_path)
        _peak_table_ok = True
    except Exception as e:
        logger.warning("peak table unloadable — perf data incomplete: %s", e)
        _compute_peaks, _resource_peaks = {}, {}
        _compute_prov, _resource_prov = {}, {}
        _peak_table_ok = False

    def _authoritative_roofline_data_complete(tid: str) -> bool:
        """True iff the repo can score `tid`'s Roofline (cost model + all
        referenced π/β verified). Mirrors `eval._apply_roofline`'s
        per-task verification — the SAME judgement, just repo-derived."""
        if not _peak_table_ok:
            return False
        task = task_by_id.get(tid)
        cost_model = getattr(task, "cost_model", None) if task else None
        if cost_model is None:
            return False
        for compute_class in cost_model.flops:
            pi = _compute_peaks.get(compute_class)
            if pi is None or not math.isfinite(pi) or pi <= 0:
                return False
        for resource in cost_model.bytes:
            beta = _resource_peaks.get(resource)
            if beta is None or not math.isfinite(beta) or beta <= 0:
                return False
        return True

    def _authoritative_absolute_verified(tid: str) -> bool:
        """True iff every peak `tid` references is absolute_verified (vendor-
        official / on-device-measured). Gates absolute efficiency claims, not
        ranking. Mirrors `eval._apply_roofline`, repo-derived."""
        from .peak_table import is_absolute_verified
        if not _peak_table_ok:
            return False
        task = task_by_id.get(tid)
        cost_model = getattr(task, "cost_model", None) if task else None
        if cost_model is None:
            return False
        for c in cost_model.flops:
            if not is_absolute_verified(
                    _compute_peaks.get(c, 0.0), _compute_prov.get(c, "unspecified")):
                return False
        for res in cost_model.bytes:
            if not is_absolute_verified(
                    _resource_peaks.get(res, 0.0),
                    _resource_prov.get(res, "unspecified")):
                return False
        return True

    # S6 release gate: if ANY score-eligible task has
    # `release_status != PASS`, the benchmark is NOT releasable, where
    # `release_status == PASS  <=>  phase_a_prime.status PASS` (S2
    # severance / golden correctness). cost_model is NOT gated here --
    # it is ncu-evidenced PERFORMANCE/access metadata, not a correctness
    # proof, so its PASS / BLOCKED_* status stays recorded but never
    # blocks release. The two real correctness gates are C2
    # oracle-equivalence (`is_provenance_eligible`) + S2 severance. The
    # diagnostic score still computes (C2: coherent with
    # perf_status=unavailable) -- only the *releasable leaderboard* is
    # refused. `compute_release_blocking` is the SAME shared predicate
    # `validate_tasks.py --release-gate` and `phase_a_prime.py`'s S6
    # verdict use -> identical blocking set.
    #
    # The blocking set keys on the C2-aligned release-ready eligible set
    # (`compute_release_ready_eligible` = score-eligible INTERSECT
    # `is_provenance_eligible`), NOT the score-only set: a C2-FAILED task
    # is EXCLUDED from the gate (it never enters), not counted as a
    # blocker. This matches `validate_tasks.py --release-gate` and
    # `phase_a_prime.py`'s S6 authority.
    from .eligibility import (
        compute_release_blocking,
        compute_release_ready_eligible,
    )

    release_ready_eligible_ids = compute_release_ready_eligible(all_tasks)
    release_blocking = compute_release_blocking(
        all_tasks, release_ready_eligible_ids
    )

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
    # R2-4: count results whose self-reported `task_status` disagrees with
    # the repo-derived authoritative value. Pure diagnostics — a mismatch
    # never moves a task into / out of the scored denominator.
    task_status_mismatch_count = 0
    # R2-4: results for `upstream_oracle_required` tasks excluded from the
    # self-reported denominator by the fixed repo-derived formula (these
    # tasks have no provider golden in that mode).
    upstream_oracle_excluded_count = 0
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
        if tid not in scored_denominator_ids:
            # R2-4 (W4.3 ④): score-eligible but excluded from THIS mode's
            # denominator by the fixed repo-derived formula — a
            # self-reported `upstream_oracle_required` task (no provider
            # golden). Its result is not scored and is not a failure.
            # The result's reported `task_status` is still cross-checked.
            upstream_oracle_excluded_count += 1
            if (r.get("task_status", TASK_STATUS_EVALUATED)
                    != _authoritative_task_status(tid)):
                task_status_mismatch_count += 1
            continue
        if tid in seen_ids:
            # codex round-16/17 P1: duplicate worst-case propagation
            # covers ALL status fields, not just correctness/aggregate.
            # W1.4: the per-task Roofline score is `s_t` (lower = worse);
            # `task_status` worst-case follows DESIGN — a blocked/unsupported
            # duplicate must not be masked by an evaluated one.
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
                # R7-4: merge `performance_status` on the SAME
                # `ok < suspicious < quarantined` precedence ladder
                # `eval._raise_performance_status` uses, not the old
                # `ok → non-ok` rule. A first result `suspicious` plus a
                # duplicate `quarantined` must merge to `quarantined`; the
                # old rule only moved off `ok` and froze at `suspicious`.
                # `error` is outside the ladder (perf never graded) — a
                # duplicate that did grade perf still wins via the
                # `prev` not-in-ladder branch below.
                new_pf = r.get("performance_status", "error")
                prev_rank = _PERFORMANCE_STATUS_PRECEDENCE.get(
                    prev.performance_status)
                new_rank = _PERFORMANCE_STATUS_PRECEDENCE.get(new_pf)
                if new_rank is not None and (
                        prev_rank is None or new_rank > prev_rank):
                    prev.performance_status = new_pf
                if r.get("fallback_detected"):
                    prev.fallback_detected = True
                # R2-4: `task_status` is repo-authoritative — a duplicate
                # result's reported `task_status` does NOT change the
                # merged value (the cheat W4.3 ④ closes). `prev.task_status`
                # was already set from `_authoritative_task_status(tid)`.
                # We still cross-check the duplicate's reported value.
                if (r.get("task_status", TASK_STATUS_EVALUATED)
                        != prev.task_status):
                    task_status_mismatch_count += 1
                # W1.4: keep the smaller (worse) Roofline s_t.
                new_st = r.get("s_t")
                if (isinstance(new_st, (int, float))
                        and not isinstance(new_st, bool)
                        and math.isfinite(float(new_st))):
                    if (prev.s_t is None
                            or float(new_st) < prev.s_t):
                        prev.s_t = float(new_st)
                break
            continue
        seen_ids.add(tid)

        # codex round-11 P1: performance_flags must be list[str]; non-list
        # / non-string elements are coerced to [].
        raw_flags = r.get("performance_flags")
        if isinstance(raw_flags, list) and all(isinstance(f, str) for f in raw_flags):
            flags = list(raw_flags)
        else:
            flags = []

        # W1.4: assemble from the real Roofline producer fields — no more
        # `measured_ns=1.0` schema-shim. `s_t` is the per-task Roofline
        # score (geomean of per-shape e); `task_status` decides whether
        # scoring includes the task at all. A non-finite / absent `s_t`
        # leaves it None (scoring then treats the task as perf-unavailable,
        # not a fail). baseline_ns/measured_ns carry the §9 provenance pair.
        raw_st = r.get("s_t")
        if (isinstance(raw_st, (int, float)) and not isinstance(raw_st, bool)
                and math.isfinite(float(raw_st))):
            s_t = float(raw_st)
        else:
            s_t = None
        # R2-4: `task_status` is taken from the repo-derived authority for
        # the suite's `mode`, NOT verbatim from the result. The result's
        # own `task_status` is only consistency-checked — a mismatch is
        # counted and reported but does not change the scored denominator.
        task_status = _authoritative_task_status(tid)
        reported_status = r.get("task_status", TASK_STATUS_EVALUATED)
        if reported_status != task_status:
            task_status_mismatch_count += 1
        task_measurements.append(TaskMeasurement(
            task_id=tid,
            correctness_status=r.get("correctness_status", "error"),
            runtime_status=r.get("runtime_status", "error"),
            performance_status=r.get("performance_status", "error"),
            performance_flags=flags,
            fallback_detected=bool(r.get("fallback_detected", False)),
            measured_ns=_finite_or_none(r.get("measured_ns")),
            baseline_ns=_finite_or_none(r.get("baseline_ns")),
            s_t=s_t,
            task_status=task_status,
            # R9-1: repo-authoritative — derived from `meta.cost_model` +
            # the 910B peak table, never the result JSON. A task whose
            # Roofline data is incomplete is honestly excluded from the
            # perf geomean by `aggregate_perf`, not ε-collapsed.
            roofline_data_complete=_authoritative_roofline_data_complete(tid),
            # Repo-authoritative absolute_verified (gates absolute efficiency
            # claims, not ranking).
            absolute_verified=_authoritative_absolute_verified(tid),
        ))

    # R2-4: denominator = the mode-aware scored set, frozen from repo
    # metadata (`scored_denominator_ids`): verified ⇒ |S|; self-reported
    # ⇒ |S| − upstream_oracle_required. Fall back to the measurement
    # count only when there is no eligible set at all (CI smoke with
    # synthetic task_ids).
    effective_denom = s_native_size if s_native_size > 0 else len(task_measurements)
    # Honest perf aggregate: returns perf_status=unavailable + null
    # metrics when no task has a real baseline (codex P0-1) instead of
    # a fabricated ε-collapsed number.
    perf = aggregate_perf(task_measurements, eligible_size=effective_denom)

    # W2.4: submission-level canary escalation. `analyze` re-aggregates the
    # canary findings from the trusted suite JSON (a `run-suite` doc may
    # already carry `canary_caught`; we recompute from `canary_findings`
    # so the verdict is authoritative even on hand-assembled inputs).
    # `canary_caught > 0` flips `suspicious`, holding the submission for
    # manual review (DESIGN §6.8 / §8).
    canary = _aggregate_suite_canaries(verified.get("results", []))
    suite_findings = verified.get("canary_findings")
    if isinstance(suite_findings, list):
        from .adversarial import (
            CanaryFinding,
            aggregate_canary_findings,
            canary_finding_json,
        )
        extra = [
            CanaryFinding(
                canary_id=cf.get("canary_id", "?"),
                triggered=bool(cf.get("triggered", False)),
                severity=cf.get("severity", "soft"),
                message=cf.get("message", ""),
            )
            for cf in suite_findings if isinstance(cf, dict)
        ]
        if extra:
            agg = aggregate_canary_findings(extra)
            canary = {
                "canary_caught": canary["canary_caught"] + agg.canary_caught,
                "suspicious": canary["suspicious"] or agg.suspicious,
                "canary_findings": canary["canary_findings"]
                + [canary_finding_json(c) for c in agg.findings],
            }

    score_doc = {
        "submitter": args.submitter,
        # R2-4: the mode the denominator authority resolved to.
        "mode": mode,
        # W2.4 canary escalation chain — suspicious gates manual review.
        "canary_caught": canary["canary_caught"],
        "suspicious": canary["suspicious"],
        "canary_findings": canary["canary_findings"],
        # `n_score_eligible_tasks` is the FULL repo-derived score-eligible
        # set (real working oracle, not PHASE-C-MANUAL) — the release-gate
        # blocking set is a subset of it.
        "n_score_eligible_tasks": len(score_eligible_ids),
        # R2-4: the mode-aware scored denominator — verified: |S|;
        # self-reported: |S| minus the upstream_oracle_required set. Frozen
        # from repo metadata; the perf / coverage aggregates use this.
        "n_scored_denominator_tasks": s_native_size,
        "n_native_measurements": len(task_measurements),
        "non_catalogue_task_count": non_catalogue_count,
        "non_eligible_task_count": non_eligible_count,
        # R2-4: results dropped from the self-reported denominator by the
        # fixed upstream_oracle_required formula (repo-derived).
        "upstream_oracle_excluded_count": upstream_oracle_excluded_count,
        # R2-4: results whose self-reported `task_status` disagreed with
        # the repo-derived authority (diagnostic only — does not move the
        # denominator).
        "task_status_mismatch_count": task_status_mismatch_count,
        "missing_task_id_count": missing_task_id_count,
        "duplicate_task_id_count": duplicate_task_id_count,
        # W4.3 ④ / R8-1: `non_evaluated` counts every task_status that is
        # not "evaluated" (diagnostic). The two are NOT treated alike by
        # scoring: `blocked` tasks STAY in the fixed |S| denominator and
        # are scored as failures (ε); only `unsupported` tasks leave the
        # scored set. `blocked_in_denominator_count` makes the
        # no-subset-escape behaviour explicit in the report.
        "non_evaluated_task_count": sum(
            1 for m in task_measurements if not m.is_evaluated
        ),
        "blocked_in_denominator_count": sum(
            1 for m in task_measurements
            if m.task_status == "blocked"
        ),
        "perf_status": perf["perf_status"],
        "perf_status_reason": perf["perf_status_reason"],
        "capped_geomean_native": perf["capped_geomean_native"],
        "fastp_native_at_1_0": perf["fastp_native_at_1_0"],
        "fastp_native_at_1_5": perf["fastp_native_at_1_5"],
        # R9-1: tasks honestly excluded from the perf geomean because
        # their Roofline benchmark data is not yet complete (missing
        # cost model / referenced peak still an [unverified] placeholder)
        # — NOT ε-collapsed against the candidate's score.
        "roofline_data_incomplete_count": perf.get(
            "roofline_data_incomplete_count", 0),
        # How many ranked tasks have an absolute_verified Roofline (vendor-
        # official / measured peaks). The rest rank but their "% of Roofline"
        # is pending verified peaks (academic-spec / estimate).
        "absolute_verified_count": sum(
            1 for m in task_measurements if m.absolute_verified),
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


def _aggregate_suite_canaries(per_task: list, extra_findings: list | None = None) -> dict:
    """W2.4: roll every result's ``canary_findings`` into a submission verdict.

    Pools the per-result ``canary_findings`` arrays, rebuilds
    :class:`adversarial.CanaryFinding` objects, and aggregates them via
    :func:`adversarial.aggregate_canary_findings`. ``canary_caught`` is
    the count of triggered canaries; any triggered canary makes the whole
    submission ``suspicious`` (DESIGN §6.8 / §8).

    `extra_findings` (W2.5) are canary-finding dicts produced by the
    dedicated `_canary/`-suite runner — they are pooled alongside the
    per-result findings so the dedicated canary path joins the same
    submission-level escalation chain.

    Returns ``{"canary_caught": int, "suspicious": bool,
    "canary_findings": list[dict]}``.
    """
    from .adversarial import (
        CanaryFinding,
        aggregate_canary_findings,
        canary_finding_json,
    )

    def _to_finding(cf: dict) -> CanaryFinding:
        return CanaryFinding(
            canary_id=cf.get("canary_id", "?"),
            triggered=bool(cf.get("triggered", False)),
            severity=cf.get("severity", "soft"),
            message=cf.get("message", ""),
        )

    findings: list = []
    for r in per_task:
        if not isinstance(r, dict):
            continue
        for cf in r.get("canary_findings", []) or []:
            if not isinstance(cf, dict):
                continue
            findings.append(_to_finding(cf))
    for cf in extra_findings or []:
        if isinstance(cf, dict):
            findings.append(_to_finding(cf))
    agg = aggregate_canary_findings(findings)
    return {
        "canary_caught": agg.canary_caught,
        "suspicious": agg.suspicious,
        "canary_findings": [canary_finding_json(c) for c in agg.findings],
    }


def _finite_or_none(value) -> float | None:
    """Coerce a JSON value to a finite float, or ``None``.

    W1.4: round_result.json perf fields (``s_t`` / ``baseline_ns`` /
    ``measured_ns``) may legitimately be ``null`` (no timing, or a
    non-"evaluated" task). A ``bool`` or a non-finite number is rejected
    to ``None`` so it cannot smuggle a NaN into scoring.
    """
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    f = float(value)
    return f if math.isfinite(f) else None


def _emit(payload: dict, out_path: str | None) -> None:
    text = json.dumps(payload, indent=2)
    if out_path:
        Path(out_path).write_text(text)
    else:
        print(text)


if __name__ == "__main__":
    sys.exit(main())
