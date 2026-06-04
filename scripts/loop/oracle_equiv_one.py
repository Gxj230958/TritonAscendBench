#!/usr/bin/env python3
"""Single-task C2 oracle-equiv: R10.a then R10.b in ONE process.

The loop daemon's oracle-equiv action runs this once per task instead of a
bash `R10.a && R10.b`. Doing both in-process buys three things the bash form
could not:

* the daemon timeout kills exactly this process -- no orphaned GPU python
  from a `bash -c` whose grandchild outlives the killed shell;
* a real business gate -- R10.b runs only when R10.a returns ALIGNED_READY,
  so a SIGNATURE_MISMATCH / RUNTIME_ERROR / NEEDS_BRIDGE task can't fall
  through to a comparison against stale or half-written refs;
* same-host ref capture, so R10.b's rebuilt inputs match (H2).

Stale ``r10_pyref`` is wiped first so a half-written prior run never leaks
into R10.b. Writes ``<output-dir>/<task>.json`` as a TaskVerdict (the same
schema R10.b emits), so the daemon parser and apply_oracle_equivalence_verdict
consume it unchanged.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from dataclasses import asdict
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent.parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from oracle_pyref_precompute import _audit_task  # noqa: E402
from oracle_equivalence_gpu_runner import (  # noqa: E402
    REPO_ROOT,
    ShapeVerdict,
    TaskVerdict,
    _verify_one_task,
)


def _blocked(task_id: str, overall: str, reason: str) -> TaskVerdict:
    v = TaskVerdict(task_id=task_id)
    v.overall = overall
    v.shapes.append(
        ShapeVerdict(shape_id="*", status=overall, blocker_reason=reason)
    )
    return v


def run_one(task_dir: Path, gpu: str, infer_root: Path) -> TaskVerdict:
    task_id = task_dir.name
    # Wipe stale refs so a half-written prior run can't leak into R10.b.
    rd = task_dir / "r10_pyref"
    if rd.exists():
        shutil.rmtree(rd)
    # R10.a: capture oracle ref on this host (H2 input parity).
    audit = _audit_task(task_dir)
    if audit.status != "ALIGNED_READY":
        # Gate: never run R10.b on a task R10.a could not align -- surface
        # R10.a's status, not a fake precision FAILED.
        detail = audit.runtime_error or audit.suggested_fix or ""
        return _blocked(task_id, audit.status or "RUNTIME_ERROR",
                        f"R10.a not aligned: {audit.status} {detail}".strip())
    # R10.b: GPU kernel-vs-oracle compare.
    return _verify_one_task(task_dir, infer_root, gpu)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    p.add_argument("--task", required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--gpu", default="cuda")
    p.add_argument("--infer-root", type=Path, default=REPO_ROOT.parent / "infer")
    args = p.parse_args(argv)

    task_dir = None
    for lvl in ("level1", "level2", "level3"):
        cand = REPO_ROOT / "tasks" / lvl / args.task
        if cand.is_dir():
            task_dir = cand
            break
    if task_dir is None:
        print(f"task {args.task} not found under tasks/", file=sys.stderr)
        return 2

    args.output_dir.mkdir(parents=True, exist_ok=True)
    v = run_one(task_dir, args.gpu, args.infer_root)
    verdict_path = args.output_dir / f"{args.task}.json"
    verdict_path.write_text(json.dumps(asdict(v), indent=2, default=str))
    print(f"oracle-equiv {args.task}: {v.overall}")

    # Daemon-side auto-apply: persist the verdict into meta.provenance.golden HERE,
    # where the just-wiped+regenerated r10_pyref refs live, so apply can hash them
    # for golden_sha256 without a multi-GB pull to the authoring host. Fail-closed:
    # only with full shape coverage; _apply_one carries the per-status guards
    # (S2-fresh + kernel_sha256, golden_sha256-only-for-VERIFIED, known-bug -> BLOCKED).
    try:
        import yaml  # noqa: PLC0415

        sd = yaml.safe_load((task_dir / "shapes.yaml").read_text()) or {}
        want = {
            s.get("id")
            for grp in ("public_shapes", "hidden_shapes")
            for s in (sd.get(grp) or [])
            if s.get("id")
        }
        got = {s.shape_id for s in v.shapes if s.shape_id not in ("*", None)}
        if want and not want.issubset(got):
            print(
                f"  auto-apply {args.task} SKIPPED: shape coverage {sorted(got)} "
                f"misses {sorted(want - got)}",
                file=sys.stderr,
            )
        else:
            from apply_oracle_equivalence_verdict import _apply_one  # noqa: PLC0415

            ar = _apply_one(verdict_path)
            print(
                f"  auto-apply {args.task}: {ar.overall} "
                f"promoted_to_truth={getattr(ar, 'promoted_to_truth', None)}"
            )
    except Exception as e:  # never let persistence crash the grade
        print(f"  auto-apply {args.task} SKIPPED (error): {e}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
