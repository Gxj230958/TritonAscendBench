#!/usr/bin/env python3
"""auto_patch_shapes_from_wrapper.py -- V9 P6 (codex review 2026-05-27).

Auto-fix WRAPPER_CONTRACT_MISMATCH tasks by rewriting
``shapes.yaml.inputs[<name>].dims`` to the dim-name list the kernel
host wrapper unpacks. Confidence-gated:

* HIGH:   wrapper exposes a direct ``B, T, H, K = k.shape`` unpack
          AND every dim name (``B``/``T``/``H``/``K`` / ...) is a key
          in the task's ``shapes.yaml.size_knobs``. Auto-applied by
          ``--apply`` (default is dry-run).
* MEDIUM: wrapper exposes a direct unpack BUT some dim name is
          missing from the knobs universe. Surfaced in the triage
          backlog report; operator picks a knob mapping then
          re-runs.
* LOW:    wrapper uses starred unpack (``*k.shape, u.shape[-1]``) or
          branch unpacks with conflicting dim-name sets. dim_names
          extraction doesn't get a unique answer; manual review.

Outputs:

* per-task verdicts -> ``/tmp/auto_patch_shapes.json``
* human report      -> ``docs/triage_backlog/2026-05-27-auto-patch-shapes.md``
* (when ``--apply``) shapes.yaml rewritten atomically via the
  ``automation.lib.atomic`` helpers; dry-run prints the diff only.
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.automation.lib.atomic import atomic_write_yaml  # noqa: E402
from ascendbench.integrity.wrapper_contract import (  # noqa: E402
    extract_wrapper_contract,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("auto_patch_shapes_from_wrapper")


@dataclass
class PatchVerdict:
    task_id: str
    fn_name: str | None = None
    confidence: str = "LOW"     # HIGH / MEDIUM / LOW
    patched: bool = False       # True iff --apply succeeded
    per_input: dict[str, dict] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)


def _read_oracle_keys(oracle_path: Path) -> list[str]:
    if not oracle_path.exists():
        return []
    src = oracle_path.read_text()
    return sorted(
        set(re.findall(r'inputs\[["\']([^"\']+)["\']\]', src))
        | set(re.findall(r'inputs\.get\(["\']([^"\']+)["\']', src))
    )


def _knobs_universe(shapes_doc: dict) -> set[str]:
    """Return the set of knob names available for substitution into
    ``dims`` (i.e. the symbols that ``ascendbench.input_spec.inputs.
    _resolve_dims`` knows how to evaluate)."""
    out: set[str] = set()
    out.update((shapes_doc.get("size_knobs") or {}).keys())
    # Public shape knobs may override / extend the global knobs.
    for s in shapes_doc.get("public_shapes") or []:
        out.update((s.get("knobs") or {}).keys())
    return out


def _resolve_kernel_entrypoint(
    meta: dict, kernel_path: Path, task_id: str, oracle_input_keys: list[str]
) -> str | None:
    """Shim over the shared resolver -- all three R10 scripts (R10.a /
    R10.b / auto-patch) now route through one implementation that
    rejects @triton.jit picks AND gates the resolve_all.json cache by
    jaccard overlap (the P5.2 + P6 fix codex flagged on 2026-05-27)."""
    from ascendbench.corpus.kernel_resolver import (  # noqa: PLC0415
        resolve_kernel_entrypoint,
    )

    return resolve_kernel_entrypoint(
        meta,
        kernel_path,
        task_id,
        oracle_input_keys,
        resolve_cache_path=REPO_ROOT / "results" / "resolve_all.json",
    )


def audit_task(task_dir: Path) -> PatchVerdict:
    task_id = task_dir.name
    verdict = PatchVerdict(task_id=task_id)

    meta_path = task_dir / "meta.yaml"
    shapes_path = task_dir / "shapes.yaml"
    kernel_path = task_dir / "kernel.py"
    oracle_path = task_dir / "oracle_pytorch.py"
    if not all(p.is_file() for p in (meta_path, shapes_path, kernel_path)):
        verdict.notes.append("missing meta.yaml/shapes.yaml/kernel.py")
        return verdict

    try:
        meta = yaml.safe_load(meta_path.read_text()) or {}
        shapes = yaml.safe_load(shapes_path.read_text()) or {}
    except yaml.YAMLError as e:
        verdict.notes.append(f"yaml parse: {e}")
        return verdict

    oracle_keys = _read_oracle_keys(oracle_path)
    fn = _resolve_kernel_entrypoint(meta, kernel_path, task_id, oracle_keys)
    verdict.fn_name = fn
    if fn is None:
        verdict.notes.append("kernel entrypoint not resolvable")
        return verdict

    contract = extract_wrapper_contract(kernel_path, fn)
    if contract is None or not contract.accepted_ranks:
        verdict.notes.append(
            "wrapper carries no rank constraints "
            "(may use starred unpack only)"
        )
        return verdict

    inputs_block = shapes.get("inputs") or {}
    knobs = _knobs_universe(shapes)

    proposals: dict[str, dict] = {}
    any_high = False
    any_medium = False

    for param_name, accepted in contract.accepted_ranks.items():
        existing = inputs_block.get(param_name)
        existing_dims = (existing or {}).get("dims") if isinstance(existing, dict) else None
        if not isinstance(existing_dims, list):
            existing_dims = None

        candidate_names = contract.dim_names.get(param_name, [])
        if not candidate_names:
            # Starred unpack only -- can't infer dim names automatically.
            proposals[param_name] = {
                "confidence": "LOW",
                "current_dims": existing_dims,
                "proposed_dims": None,
                "reason": (
                    "wrapper has rank constraint "
                    f"{sorted(accepted)} but no direct-unpack dim names "
                    "(starred unpack); manual"
                ),
            }
            continue

        # Pick the first observed dim-name tuple as canonical.
        chosen = list(candidate_names[0])
        if existing_dims is not None and len(existing_dims) == len(chosen):
            proposals[param_name] = {
                "confidence": "OK",
                "current_dims": existing_dims,
                "proposed_dims": chosen,
                "reason": "current dims already match wrapper rank",
            }
            continue

        unknown = [n for n in chosen if n not in knobs]
        if unknown:
            any_medium = True
            proposals[param_name] = {
                "confidence": "MEDIUM",
                "current_dims": existing_dims,
                "proposed_dims": chosen,
                "reason": (
                    f"dim names not in task knobs universe: {unknown}; "
                    "operator must add knob entries or map names"
                ),
            }
            continue

        any_high = True
        proposals[param_name] = {
            "confidence": "HIGH",
            "current_dims": existing_dims,
            "proposed_dims": chosen,
            "reason": "wrapper unpack matches knobs; auto-patch candidate",
        }

    verdict.per_input = proposals
    if any_high and not any_medium:
        verdict.confidence = "HIGH"
    elif any_high or any_medium:
        verdict.confidence = "MEDIUM"
    else:
        verdict.confidence = "LOW"
    return verdict


def apply_patch(task_dir: Path, verdict: PatchVerdict) -> bool:
    shapes_path = task_dir / "shapes.yaml"
    shapes = yaml.safe_load(shapes_path.read_text()) or {}
    inputs = shapes.setdefault("inputs", {})
    changed = False
    for name, prop in verdict.per_input.items():
        if prop.get("confidence") != "HIGH":
            continue
        proposed = prop.get("proposed_dims")
        if not proposed:
            continue
        entry = inputs.get(name)
        if not isinstance(entry, dict):
            entry = {}
            inputs[name] = entry
        if entry.get("dims") == proposed:
            continue
        entry["dims"] = proposed
        changed = True
    if changed:
        atomic_write_yaml(shapes_path, shapes)
    return changed


def _emit_report(verdicts: list[PatchVerdict], report_path: Path) -> None:
    from collections import Counter

    tally = Counter(v.confidence for v in verdicts)
    lines = [
        "---",
        "tool: scripts/auto_patch_shapes_from_wrapper.py",
        "purpose: V9 P6 — auto-patch shapes.yaml dims from kernel host wrapper unpacks",
        "---",
        "",
        "# Auto-patch shapes from wrapper contract",
        "",
        f"Tasks scanned: **{len(verdicts)}**",
        "",
        "## Confidence tally",
        "",
        "| confidence | count |",
        "|---|---|",
    ]
    for c, n in tally.most_common():
        lines.append(f"| `{c}` | {n} |")
    lines.append("")

    for confidence in ("HIGH", "MEDIUM", "LOW"):
        subset = [v for v in verdicts if v.confidence == confidence]
        if not subset:
            continue
        lines.append(f"## {confidence} ({len(subset)})")
        lines.append("")
        for v in subset:
            lines.append(f"### `{v.task_id}` ({v.fn_name})")
            for name, prop in v.per_input.items():
                lines.append(
                    f"- `{name}`: {prop.get('confidence')} "
                    f"current={prop.get('current_dims')} "
                    f"proposed={prop.get('proposed_dims')} "
                    f"-- {prop.get('reason')}"
                )
            for note in v.notes:
                lines.append(f"- note: {note}")
            lines.append("")

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines))


def _iter_task_dirs() -> list[Path]:
    out = []
    for level in ("level1", "level2", "level3"):
        ld = REPO_ROOT / "tasks" / level
        if not ld.is_dir():
            continue
        for td in sorted(ld.iterdir()):
            if td.is_dir() and not td.name.startswith("_"):
                out.append(td)
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    p.add_argument("--task", help="single task id")
    p.add_argument("--all", action="store_true", help="all tasks")
    p.add_argument(
        "--apply",
        action="store_true",
        help="actually rewrite shapes.yaml for HIGH-confidence cases",
    )
    p.add_argument("--json-out", default="/tmp/auto_patch_shapes.json")
    p.add_argument(
        "--report",
        default=str(
            REPO_ROOT / "docs" / "triage_backlog"
            / "2026-05-27-auto-patch-shapes.md"
        ),
    )
    args = p.parse_args(argv)
    if not (args.task or args.all):
        p.error("either --task or --all required")

    task_dirs = _iter_task_dirs()
    if args.task:
        task_dirs = [td for td in task_dirs if td.name == args.task]
        if not task_dirs:
            log.error("task %s not found", args.task)
            return 2

    verdicts: list[PatchVerdict] = []
    n_applied = 0
    for td in task_dirs:
        v = audit_task(td)
        if args.apply and v.confidence == "HIGH":
            v.patched = apply_patch(td, v)
            if v.patched:
                n_applied += 1
        verdicts.append(v)
        log.info(
            "  %s: confidence=%s%s",
            v.task_id, v.confidence,
            " [PATCHED]" if v.patched else "",
        )

    from collections import Counter
    tally = Counter(v.confidence for v in verdicts)
    log.info("--- summary ---")
    for c, n in tally.most_common():
        log.info("  %s: %d", c, n)
    if args.apply:
        log.info("  patched: %d", n_applied)

    Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.json_out).write_text(
        json.dumps([asdict(v) for v in verdicts], indent=2)
    )
    _emit_report(verdicts, Path(args.report))
    log.info("json -> %s", args.json_out)
    log.info("report -> %s", args.report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
