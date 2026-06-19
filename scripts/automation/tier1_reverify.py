"""Tier 1 manual reverify wrapper (Unit 1.8).

User-facing CLI invoked on the GPU machine. Internally dispatches to:

  --cmd resolve         -> scripts/verify/resolve_upstream_entrypoints.py --task <T> --apply
  --cmd phase_a_verify  -> scripts/verify/standalone_freeze_verifier.py --verify <T>

then runs ``scripts/automation/extract_verdict.py`` to materialise the
9-field ``result.json`` + ``_provenance.json`` sidecar + ``DONE``
marker under ``<out_dir>/<run_id>/``.

Always invokes both sub-commands as fresh ``subprocess.run`` calls (no
in-process import) to avoid leaking the resolver's process-global
``_RESOLVER_CACHE_MEM`` across runs (KTD-5).
"""
from __future__ import annotations

# ---- KTD-11 sys.path bootstrap (CLI entry-point) -------------------------
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "scripts"))
# --------------------------------------------------------------------------

import argparse  # noqa: E402
import logging  # noqa: E402
import subprocess  # noqa: E402
import time  # noqa: E402
import uuid  # noqa: E402
from datetime import datetime, timezone  # noqa: E402

from automation import extract_verdict  # noqa: E402

logger = logging.getLogger("automation.tier1_reverify")

DEFAULT_TIMEOUT_S = 1800


def _now_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _mint_run_id() -> str:
    return f"tier1-{_now_id()}-{uuid.uuid4().hex[:8]}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--task", required=True, dest="task_id")
    parser.add_argument("--cmd", required=True, choices=["resolve", "phase_a_verify"])
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT_S)
    parser.add_argument("--repo-root", type=Path, default=_REPO_ROOT)
    parser.add_argument(
        "--resolve-log", type=Path, default=None,
        help="Path to pa_resolve.log; only used for --cmd resolve.",
    )
    args = parser.parse_args(argv)

    run_id = args.run_id or _mint_run_id()
    result_dir = args.out_dir / run_id
    result_dir.mkdir(parents=True, exist_ok=True)
    log_path = result_dir / "log.txt"

    repo = args.repo_root
    if args.cmd == "resolve":
        sub_cmd = [
            sys.executable,
            str(repo / "scripts" / "verify" / "resolve_upstream_entrypoints.py"),
            "--task", args.task_id, "--apply",
        ]
    else:  # phase_a_verify
        sub_cmd = [
            sys.executable,
            str(repo / "scripts" / "verify" / "standalone_freeze_verifier.py"),
            "--verify", args.task_id,
        ]

    started = time.time()
    timed_out = False
    with open(log_path, "w", encoding="utf-8") as log_fh:
        try:
            subprocess.run(
                sub_cmd,
                stdout=log_fh,
                stderr=subprocess.STDOUT,
                timeout=args.timeout,
            )
        except subprocess.TimeoutExpired:
            timed_out = True
            log_fh.write(f"\n*** TIMEOUT after {args.timeout}s\n")
        except FileNotFoundError as exc:
            log_fh.write(f"\n*** sub-command not found: {exc}\n")
    elapsed = time.time() - started

    if timed_out:
        # Synthesize a BLOCKED result + finish.
        extract_verdict.write_verdict(
            task_id=args.task_id, cmd=args.cmd, run_id=run_id,
            producer="tier1_reverify", out_dir=args.out_dir,
            verdict="BLOCKED", verdict_class="unexpected-blocked",
            reason=f"tier1 wrapper timeout after {args.timeout}s",
            raw_subclass="timeout",
            extra_provenance={"timed_out": True, "elapsed_s": elapsed},
        )
        return 1

    # Otherwise: run extract_verdict to materialize result.json from
    # the produced artifacts (resolver log / meta.yaml).
    extract_argv = [
        "--task", args.task_id,
        "--cmd", args.cmd,
        "--run-id", run_id,
        "--producer", "tier1_reverify",
        "--out-dir", str(args.out_dir),
        "--stdout-log", str(log_path),
        "--repo-root", str(repo),
    ]
    if args.cmd == "resolve":
        resolve_log = (
            args.resolve_log
            or repo / "results" / "gpu_pipeline" / "pa_resolve.log"
        )
        extract_argv += ["--resolve-log", str(resolve_log)]

    ev_rc = extract_verdict.main(extract_argv)
    if ev_rc != 0:
        logger.error("extract_verdict failed (rc=%d)", ev_rc)
        return 1

    done_marker = result_dir / "DONE"
    print(f"DONE: {result_dir / 'result.json'}", flush=True)
    return 0 if done_marker.exists() else 1


if __name__ == "__main__":
    raise SystemExit(main())
