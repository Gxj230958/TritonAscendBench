#!/usr/bin/env python3
"""Dev-side R10 stage-3: apply GPU equivalence verdict to meta.yaml.

After the GPU-host operator runs scripts/gpu_pipeline/oracle_equivalence_verify.sh,
verdict JSONs land at /tmp/r10_verdict/<task_id>.json (or wherever the
operator rsyncs them back to). This script reads those verdicts and
writes provenance.golden.* fields into each task's meta.yaml.

Architectural shift: when oracle_kernel_equivalence == VERIFIED, the
corpus oracle_pytorch.py is promoted to truth source for that task.
Existing top-level `provenance:` keys (added_in / authors / last_modified)
are preserved; this only adds/updates the nested `provenance.golden:` block.

Manual operator workflow:
  # On dev machine after rsync /tmp/r10_verdict/ back:
  python3 scripts/oracle/apply_oracle_equivalence_verdict.py --verdict-dir /tmp/r10_verdict/
  # OR:
  tab apply-equivalence-verdict --verdict-dir /tmp/r10_verdict/
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from ascendbench._util.atomic import atomic_write_yaml  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("apply_oracle_equivalence_verdict")


def _utc_iso() -> str:
    """Single-source UTC timestamp emitter (mirrors automation/state_machine.iso_utc_now)."""
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _ref_sha256(task_dir: Path) -> str | None:
    """Compute aggregated SHA-256 of all r10_pyref/ref_*.safetensors files
    (sorted by filename for determinism). Returns None if no ref files."""
    ref_dir = task_dir / "r10_pyref"
    if not ref_dir.is_dir():
        return None
    refs = sorted(ref_dir.glob("ref_*.safetensors"))
    if not refs:
        return None
    h = hashlib.sha256()
    for f in refs:
        h.update(f.name.encode("utf-8"))
        h.update(f.read_bytes())
    return h.hexdigest()


@dataclass
class ApplyResult:
    task_id: str
    overall: str
    meta_updated: bool = False
    skipped_reason: str = ""
    promoted_to_truth: bool = False  # oracle_pytorch.py promoted to truth source
    diff_summary: dict[str, dict[str, float]] = field(default_factory=dict)


def _find_task_dir(task_id: str) -> Path | None:
    for level in ("level1", "level2", "level3"):
        td = REPO_ROOT / "tasks" / level / task_id
        if td.is_dir():
            return td
    return None


def _apply_one(verdict_path: Path) -> ApplyResult:
    """Read one verdict JSON;write provenance.golden into the task's meta.yaml."""
    try:
        verdict = json.loads(verdict_path.read_text())
    except Exception as e:
        return ApplyResult(
            task_id=verdict_path.stem,
            overall="VERDICT_PARSE_ERROR",
            skipped_reason=str(e),
        )

    task_id = verdict.get("task_id") or verdict_path.stem
    res = ApplyResult(task_id=task_id, overall=verdict.get("overall", ""))

    td = _find_task_dir(task_id)
    if td is None:
        res.skipped_reason = "task dir not found"
        return res

    meta_path = td / "meta.yaml"
    if not meta_path.is_file():
        res.skipped_reason = "meta.yaml missing"
        return res

    try:
        meta = yaml.safe_load(meta_path.read_text()) or {}
    except Exception as e:
        res.skipped_reason = f"meta.yaml parse: {e}"
        return res

    # Preserve existing top-level provenance keys (added_in / authors / last_modified)
    prov = meta.get("provenance")
    if not isinstance(prov, dict):
        prov = {}

    # Build provenance.golden block
    golden: dict = dict(prov.get("golden") or {})

    overall = res.overall
    # A known upstream-kernel-bug task can never be VERIFIED: a passing tolerance
    # run would falsely validate a buggy kernel against a correct oracle. Force
    # BLOCKED regardless of the equivalence verdict (kept in the denominator).
    if (meta.get("upstream") or {}).get("known_kernel_bug") is True:
        overall = "BLOCKED_UPSTREAM_BUG"
        res.overall = overall
    golden["oracle_kernel_equivalence"] = overall
    golden["verified_at"] = _utc_iso()
    golden["tolerance_class_used"] = meta.get("tolerance_class") or ""

    # Per-shape diff summary
    shapes = verdict.get("shapes") or []
    per_shape: dict[str, dict[str, float | str]] = {}
    for s in shapes:
        sid = s.get("shape_id", "")
        entry: dict[str, float | str] = {"status": s.get("status", "")}
        if s.get("max_abs_diff"):
            entry["max_abs"] = max(s["max_abs_diff"].values(), default=0.0)
        if s.get("max_rel_diff"):
            entry["max_rel"] = max(s["max_rel_diff"].values(), default=0.0)
        if s.get("blocker_reason"):
            entry["blocker"] = s["blocker_reason"][:200]
        per_shape[sid] = entry
    if per_shape:
        golden["equivalence_per_shape"] = per_shape
        # Store flat diff_summary for easy grep
        res.diff_summary = {
            sid: {
                k: v
                for k, v in entry.items()
                if k in ("max_abs", "max_rel") and isinstance(v, (int, float))
            }
            for sid, entry in per_shape.items()
        }

    # Clear stale ``golden_sha256`` whenever
    # the new verdict is NOT ``VERIFIED``. Without this, switching from
    # VERIFIED -> FAILED / BLOCKED / WRAPPER_*/ ORACLE_* would leave the
    # previous successful run's hash hanging in meta -- a real provenance
    # contamination: downstream gates would see ``oracle_kernel_equivalence
    # != VERIFIED`` paired with a populated ``golden_sha256`` and could
    # mis-trust it. Wipe before re-attesting only when verified.
    if overall != "VERIFIED":
        golden.pop("golden_sha256", None)

    if overall == "VERIFIED":
        # Masking root fix (plan P0.2-A): a C2 VERIFIED only PROMOTES to
        # truth-source / fully-usable when the S2 severance gate is a FRESH PASS
        # for the CURRENT kernel.py (status PASS + standalone_sha256 matches the
        # on-disk file). A standalone that S2 rejects -- or whose S2 status is
        # stale vs the current kernel.py -- is cross-inconsistent: record the raw
        # C2 verdict but do NOT promote. (`compute_usability` enforces the same
        # conjunction independently; this keeps the golden provenance honest.)
        from ascendbench.corpus.shape_manifest import shape_manifest_sha256_for_task
        from ascendbench.release_gate.eligibility import fresh_s2_pass

        _kp = td / "kernel.py"
        _ksha = (
            hashlib.sha256(_kp.read_bytes()).hexdigest() if _kp.is_file() else None
        )
        # Shape-set freshness: do NOT promote C2 VERIFIED to truth over a shapes.yaml that drifted
        # from the one S2 verified (a stale shape set would otherwise be silently re-attested).
        _smanifest = shape_manifest_sha256_for_task(td)
        if fresh_s2_pass(meta, _ksha, _smanifest):
            golden["cross_check"] = "S2_FRESH"
            sha = _ref_sha256(td)
            if sha:
                golden["golden_sha256"] = sha
            golden["golden_source"] = "oracle_pytorch_verified"
            res.promoted_to_truth = True
        else:
            golden["cross_check"] = "CROSS_INCONSISTENT"
            golden["golden_source"] = "verified_s2_not_fresh"
            golden.pop("golden_sha256", None)
            res.promoted_to_truth = False
    elif overall == "CEILING_ACCEPTED":
        # Strict tolerance.compare() fails only at large shapes, by a PROVEN dtype-ULP
        # ceiling on an allowlisted task (CPU-fp32-oracle cannot certify GPU tensor-core
        # reduction order at scale). Accepted as a status DISTINCT from VERIFIED -- no
        # golden_sha256 promotion -- but recorded with its own source so a stale
        # 'oracle_pytorch_verified' never lingers and downstream never conflates it with a
        # strict pass. See ascendbench/policies/ceiling_acceptance.yaml.
        golden["golden_source"] = "dtype_ceiling_accepted"
        # NOT a strict pass -> never carry a golden_sha256 (a task that was VERIFIED before a
        # new shape pushed it to CEILING_ACCEPTED would otherwise leave a stale one that
        # downstream reads as verified-truth).
        golden.pop("golden_sha256", None)
        res.promoted_to_truth = False
    elif overall == "FAILED" or overall == "MIXED":
        golden["golden_source"] = "fallback_to_gpu_capture_pending"
        # Caller-side note: this task needs source-2 GPU capture
    elif overall == "LAUNCH_BLOCKED":
        golden["golden_source"] = "blocked_kernel_launch"
        # Operator must triage launch_harness or upstream import
    elif overall == "WRAPPER_CONTRACT_MISMATCH":
        # shapes.yaml.inputs.dims disagrees with kernel wrapper unpack
        # rank. Patching shapes is a candidate fix; oracle helper
        # appears rank-agnostic. Route to scripts/auto_patch_shapes
        # workflow or hand-patch and re-run R10.a + R10.b.
        golden["golden_source"] = "blocked_wrapper_contract_mismatch"
    elif overall == "ORACLE_KERNEL_LAYOUT_MISMATCH":
        # Oracle helper and wrapper disagree on the input rank; patching
        # shapes would break one to fix the other. Force source (2).
        golden["golden_source"] = "blocked_oracle_kernel_layout_mismatch"
    elif overall == "SKIPPED":
        golden["golden_source"] = "skipped_no_pyref_artifacts"
    elif overall == "INPUT_DRIFT":
        # Explicit source so a prior 'oracle_pytorch_verified' doesn't linger.
        golden["golden_source"] = "blocked_input_drift"
    elif overall == "SHAPE_COVERAGE_GAP":
        # A declared shape was never graded; the C2 coverage gate refuses VERIFIED.
        golden["golden_source"] = "blocked_shape_coverage_gap"
    elif overall not in ("VERIFIED", "CEILING_ACCEPTED"):
        # Final fallback: any novel / unrecognised non-VERIFIED overall MUST overwrite
        # golden_source (assignment, NOT setdefault -- `golden` is seeded from the existing block)
        # so a prior 'oracle_pytorch_verified' can never linger. golden_sha256 was already cleared
        # above for every non-VERIFIED overall.
        golden["golden_source"] = "blocked_unrecognized_verdict"

    prov["golden"] = golden
    # Preserve other top-level provenance keys (added_in / authors / last_modified)
    meta["provenance"] = prov

    try:
        atomic_write_yaml(meta_path, meta)
        res.meta_updated = True
    except Exception as e:
        res.skipped_reason = f"meta.yaml write: {e}"
    return res


def _emit_report(
    results: list[ApplyResult], report_path: Path, verdict_dir: Path
) -> None:
    """Write a human-readable summary report."""
    from collections import Counter

    tally = Counter(r.overall for r in results)
    promoted = sum(1 for r in results if r.promoted_to_truth)

    lines: list[str] = [
        "---",
        f"date: {_utc_iso()[:10]}",
        "tool: scripts/oracle/apply_oracle_equivalence_verdict.py",
        "purpose: V9 M2.1c — apply R10 GPU verdict to meta.yaml provenance.golden",
        f"verdict-dir: {verdict_dir}",
        "---",
        "",
        "# Oracle Equivalence Verdict Apply Report",
        "",
        f"Tasks processed: **{len(results)}**  ",
        f"Promoted to truth source (`oracle_pytorch_verified`): **{promoted}**",
        "",
        "## Overall verdict tally",
        "",
        "| overall | count |",
        "|---|---|",
    ]
    for overall, n in tally.most_common():
        lines.append(f"| `{overall}` | {n} |")
    lines.append("")

    for status in (
        "VERIFIED",
        "CEILING_ACCEPTED",
        "FAILED",
        "MIXED",
        "LAUNCH_BLOCKED",
        "SKIPPED",
        "VERDICT_PARSE_ERROR",
    ):
        sub = [r for r in results if r.overall == status]
        if not sub:
            continue
        lines.append(f"## {status} ({len(sub)})")
        lines.append("")
        for r in sub:
            lines.append(f"### `{r.task_id}`")
            if r.promoted_to_truth:
                lines.append("- **oracle_pytorch.py promoted to truth source**")
            if r.diff_summary:
                lines.append("- per-shape numeric diff:")
                for sid, d in r.diff_summary.items():
                    parts = []
                    if "max_abs" in d:
                        parts.append(f"max_abs={d['max_abs']:.3e}")
                    if "max_rel" in d:
                        parts.append(f"max_rel={d['max_rel']:.3e}")
                    lines.append(f"  - `{sid}`: {', '.join(parts)}")
            if r.skipped_reason:
                lines.append(f"- skipped reason: `{r.skipped_reason[:200]}`")
            if not r.meta_updated:
                lines.append("- meta.yaml **not** updated")
            lines.append("")

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    p.add_argument(
        "--verdict-dir",
        type=Path,
        default=Path("/tmp/r10_verdict"),
        help="directory containing per-task verdict JSON files",
    )
    p.add_argument(
        "--report",
        type=Path,
        default=REPO_ROOT
        / "build"
        / "oracle-equivalence-applied.md",
        help="human-readable summary report path",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="parse verdicts but do not write meta.yaml",
    )
    args = p.parse_args(argv)

    if not args.verdict_dir.is_dir():
        log.error("verdict dir does not exist: %s", args.verdict_dir)
        return 2

    verdict_files = sorted(args.verdict_dir.glob("*.json"))
    if not verdict_files:
        log.error("no verdict json files in %s", args.verdict_dir)
        return 2

    log.info(
        "applying %d verdict file(s) from %s%s",
        len(verdict_files),
        args.verdict_dir,
        " [DRY-RUN]" if args.dry_run else "",
    )

    results: list[ApplyResult] = []
    for vf in verdict_files:
        if args.dry_run:
            # Just parse + classify without writing
            try:
                v = json.loads(vf.read_text())
                results.append(
                    ApplyResult(
                        task_id=v.get("task_id", vf.stem),
                        overall=v.get("overall", ""),
                        meta_updated=False,
                        promoted_to_truth=(v.get("overall") == "VERIFIED"),
                    )
                )
            except Exception as e:
                results.append(
                    ApplyResult(
                        task_id=vf.stem,
                        overall="VERDICT_PARSE_ERROR",
                        skipped_reason=str(e),
                    )
                )
        else:
            results.append(_apply_one(vf))
        if results[-1].meta_updated:
            log.info("  %s: %s → meta.yaml updated", results[-1].task_id, results[-1].overall)
        elif args.dry_run:
            log.info("  %s: %s [dry-run, no write]", results[-1].task_id, results[-1].overall)
        else:
            log.warning(
                "  %s: %s SKIPPED (%s)",
                results[-1].task_id,
                results[-1].overall,
                results[-1].skipped_reason or "no meta write",
            )

    # Summary
    from collections import Counter

    tally = Counter(r.overall for r in results)
    promoted = sum(1 for r in results if r.promoted_to_truth)
    log.info("--- summary ---")
    for overall, n in tally.most_common():
        log.info("  %s: %d", overall, n)
    log.info("  promoted-to-truth (oracle_pytorch_verified): %d", promoted)

    _emit_report(results, args.report, args.verdict_dir)
    log.info("report -> %s", args.report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
