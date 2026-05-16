#!/usr/bin/env python3
"""validate_tasks.py — schema validator for every task in tasks/.

Per specs/task_api_contract.md §5–7 + R14-P1-2.

Checks:
- All 6 mandatory files present
- meta.yaml has required top-level fields and nested structures
- tolerance_class references an entry in tolerance_policy.yaml
- oracle_simplified ∈ {simple_pytorch_ok, formula_exact_manual, upstream_oracle_required}
- kernel.py and oracle_pytorch.py parse (ast.parse) — no SyntaxError
- kernel.py contains @triton.jit
- file_sha256 matches the on-disk kernel.py if set
- in_s_native ⇒ oracle.ascendc.available

Exit non-zero on first error (or all errors with --strict).
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
TASKS_ROOT = REPO_ROOT / "tasks"
POLICY_FILE = REPO_ROOT / "triton_ascend_bench" / "tolerance_policy.yaml"

REQUIRED_FILES = {
    "kernel.py",
    "oracle_pytorch.py",
    "meta.yaml",
    "shapes.yaml",
    "contract.md",
    "ascendc_baseline.yaml",
}

REQUIRED_META_TOP_LEVEL = {
    "schema_version", "task_id", "level", "idx", "name",
    "repo", "dtype_class", "category", "npu_difficulty",
    "upstream", "interface", "oracle", "tolerance_class",
    "oracle_simplified", "runner_checks",
    "is_canary", "is_hidden_only", "in_s_native",
}

VALID_LEVELS = {"L1", "L2", "L3", "appendix", "canary"}
VALID_REPOS = {
    "vLLM", "SGLang", "FBGEMM", "Bytedance_internal", "PrimusTurbo",
}
VALID_DTYPE_CLASS = {"VV", "CV"}
VALID_DIFFICULTY = {"low", "medium", "high"}
VALID_ORACLE_TIERS = {
    "simple_pytorch_ok", "formula_exact_manual", "upstream_oracle_required"
}

PLACEHOLDER_PATTERNS = ("<frozen-snapshot>", "<TBD>", "<set-by-", "<fill-after-")


def _load_policy_classes() -> set[str]:
    doc = yaml.safe_load(POLICY_FILE.read_text())
    return set(doc.get("classes", {}).keys())


def _file_sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def validate_task(task_dir: Path, policy_classes: set[str], strict: bool) -> list[str]:
    errs: list[str] = []
    tid = task_dir.name

    # Step 1: files
    have = {p.name for p in task_dir.iterdir() if p.is_file()}
    missing = REQUIRED_FILES - have
    if missing:
        errs.append(f"{tid}: missing files {sorted(missing)}")
        return errs

    meta = yaml.safe_load((task_dir / "meta.yaml").read_text())

    # Step 2: top-level keys
    missing_keys = REQUIRED_META_TOP_LEVEL - set(meta.keys())
    if missing_keys:
        errs.append(f"{tid}: meta.yaml missing keys {sorted(missing_keys)}")
        if strict and missing_keys:
            return errs

    # Step 3: enum checks
    if meta.get("level") not in VALID_LEVELS:
        errs.append(f"{tid}: level {meta.get('level')!r} not in {VALID_LEVELS}")
    if meta.get("repo") not in VALID_REPOS:
        errs.append(f"{tid}: repo {meta.get('repo')!r} not in {VALID_REPOS}")
    if meta.get("dtype_class") not in VALID_DTYPE_CLASS:
        errs.append(f"{tid}: dtype_class {meta.get('dtype_class')!r} invalid")
    if meta.get("npu_difficulty") not in VALID_DIFFICULTY:
        errs.append(f"{tid}: npu_difficulty {meta.get('npu_difficulty')!r} invalid")
    if meta.get("oracle_simplified") not in VALID_ORACLE_TIERS:
        errs.append(
            f"{tid}: oracle_simplified {meta.get('oracle_simplified')!r} "
            f"not in {VALID_ORACLE_TIERS}"
        )

    # Step 4: tolerance class exists in policy
    tol = meta.get("tolerance_class")
    if tol not in policy_classes:
        errs.append(
            f"{tid}: tolerance_class {tol!r} not in tolerance_policy.yaml "
            f"({sorted(policy_classes)})"
        )

    # Step 5: in_s_native ⇒ AscendC baseline available **and** frozen.
    # Per codex round-2 N-P0-2: an `available: true` baseline with
    # `status: placeholder_*` is not actually frozen and cannot serve as
    # the FastP denominator.
    if meta.get("in_s_native") is True:
        ascendc = (meta.get("oracle") or {}).get("ascendc") or {}
        if not ascendc.get("available"):
            errs.append(
                f"{tid}: in_s_native=true but oracle.ascendc.available is "
                f"{ascendc.get('available')!r}"
            )
        baseline_path = task_dir / "ascendc_baseline.yaml"
        if baseline_path.exists():
            baseline = yaml.safe_load(baseline_path.read_text()) or {}
            status = baseline.get("status", "")
            if str(status).startswith("placeholder"):
                errs.append(
                    f"{tid}: in_s_native=true but ascendc_baseline.yaml "
                    f"status={status!r} (not frozen). Either flip "
                    f"`in_s_native: false` until baseline is measured, "
                    f"or freeze the baseline first."
                )
            src_sha = (baseline.get("source") or {}).get("sha256", "")
            if any(p in str(src_sha) for p in PLACEHOLDER_PATTERNS):
                errs.append(
                    f"{tid}: ascendc_baseline source.sha256 still a placeholder: "
                    f"{src_sha!r}"
                )

    # Step 6: Python syntax gate (G11 fix). A schema-valid task whose
    # kernel.py / oracle_pytorch.py does not even parse can never compile
    # or be evaluated, so it must fail validation — not slip through.
    kernel_text = (task_dir / "kernel.py").read_text()
    for fname, src in (
        ("kernel.py", kernel_text),
        ("oracle_pytorch.py", (task_dir / "oracle_pytorch.py").read_text()),
    ):
        try:
            ast.parse(src, filename=str(task_dir / fname))
        except SyntaxError as e:
            errs.append(
                f"{tid}: {fname} SyntaxError at line {e.lineno}: {e.msg}"
            )

    # Step 7: kernel.py contains @triton.jit
    if "@triton.jit" not in kernel_text:
        errs.append(f"{tid}: kernel.py contains no @triton.jit")

    # Step 8: file_sha256 must match (and never be a placeholder)
    declared_sha = (meta.get("upstream") or {}).get("file_sha256", "")
    if any(p in str(declared_sha) for p in PLACEHOLDER_PATTERNS):
        errs.append(
            f"{tid}: upstream.file_sha256 still a placeholder: {declared_sha!r}"
        )
    actual_sha = _file_sha256(task_dir / "kernel.py")
    if declared_sha and not any(p in declared_sha for p in PLACEHOLDER_PATTERNS):
        if declared_sha != actual_sha:
            errs.append(
                f"{tid}: kernel.py sha256 mismatch: "
                f"meta has {declared_sha[:16]}…, actual {actual_sha[:16]}…"
            )

    return errs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", help="Validate a single task id (skip the rest)")
    parser.add_argument("--strict", action="store_true",
                        help="Stop on first task error")
    args = parser.parse_args()

    policy_classes = _load_policy_classes()

    all_errs: list[str] = []
    for level in ("level1", "level2", "level3"):
        level_dir = TASKS_ROOT / level
        if not level_dir.is_dir():
            continue
        for task_dir in sorted(level_dir.iterdir()):
            if not task_dir.is_dir() or task_dir.name.startswith("_"):
                continue
            if args.task and task_dir.name != args.task:
                continue
            errs = validate_task(task_dir, policy_classes, args.strict)
            all_errs.extend(errs)
            for e in errs:
                print(f"ERROR: {e}", file=sys.stderr)
            if args.strict and errs:
                return 1

    if all_errs:
        print(f"\n{len(all_errs)} validation error(s)", file=sys.stderr)
        return 1
    print("All tasks validated successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
