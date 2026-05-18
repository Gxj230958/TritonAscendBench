#!/usr/bin/env python3
"""validate_tasks.py -- schema validator for every task in tasks/.

Per specs/task_api_contract.md §5-7 + R14-P1-2.

Checks:
- All 6 mandatory files present
- meta.yaml has required top-level fields and nested structures
- tolerance_class references an entry in tolerance_policy.yaml
- oracle_simplified ∈ {simple_pytorch_ok, formula_exact_manual, upstream_oracle_required}
- kernel.py and oracle_pytorch.py parse (ast.parse) -- no SyntaxError
- kernel.py contains @triton.jit
- file_sha256 matches the on-disk kernel.py if set

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
    # ascendc_baseline.yaml removed: GPU-triton -> NPU-triton *migration*
    # benchmark; perf is not measured vs hand-written AscendC native.
}

REQUIRED_META_TOP_LEVEL = {
    "schema_version", "task_id", "level", "idx", "name",
    "repo", "dtype_class", "category", "npu_difficulty",
    "upstream", "interface", "oracle", "tolerance_class",
    "oracle_simplified", "runner_checks",
    "is_canary", "is_hidden_only",   # `in_s_native` removed (no AscendC)
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

    # Step 5 removed: the in_s_native ⇒ AscendC-baseline-frozen check is
    # obsolete. This is a GPU-triton -> NPU-triton migration benchmark;
    # there is no AscendC native baseline (see scoring_spec.md). The
    # performance reference is deferred (correctness is the live scored
    # axis); no per-task baseline artifact is required.

    # Step 6: Python syntax gate (G11 fix). A schema-valid task whose
    # kernel.py / oracle_pytorch.py does not even parse can never compile
    # or be evaluated, so it must fail validation -- not slip through.
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

    # Step 8: PHASE-AWARE hash integrity (plan U5 codex-cep-r2 must-fix).
    # If the task has been Phase-A'-processed it carries
    # meta.phase_a_prime.standalone_sha256 -- compare THAT to the on-disk
    # kernel.py. Otherwise (unprocessed / PoC / legacy: kernel.py is
    # still the verbatim upstream freeze) keep the legacy behaviour and
    # compare meta.upstream.file_sha256. `upstream.file_sha256` is
    # ALWAYS preserved as the original provenance hash. This makes a
    # mixed rollout (some rewritten, some legacy) all-green under the
    # normal schema/hash mode (release gate is a SEPARATE mode, U7).
    actual_sha = _file_sha256(task_dir / "kernel.py")
    pap = meta.get("phase_a_prime") or {}
    standalone_sha = pap.get("standalone_sha256", "")
    if standalone_sha and not any(
        p in str(standalone_sha) for p in PLACEHOLDER_PATTERNS
    ):
        if standalone_sha != actual_sha:
            errs.append(
                f"{tid}: phase_a_prime.standalone_sha256 mismatch: "
                f"meta has {str(standalone_sha)[:16]}…, actual {actual_sha[:16]}…"
            )
    else:
        declared_sha = (meta.get("upstream") or {}).get("file_sha256", "")
        if any(p in str(declared_sha) for p in PLACEHOLDER_PATTERNS):
            errs.append(
                f"{tid}: upstream.file_sha256 still a placeholder: {declared_sha!r}"
            )
        elif not declared_sha:
            # FIX 5: NEITHER phase_a_prime.standalone_sha256 NOR
            # upstream.file_sha256 is populated -> there is NO
            # on-disk-comparable integrity hash at all. The old code
            # silently passed this (a task with zero integrity check).
            # Every task MUST have at least one comparable hash.
            errs.append(
                f"{tid}: no integrity hash -- neither "
                f"phase_a_prime.standalone_sha256 nor "
                f"upstream.file_sha256 is populated (every task must "
                f"have at least one on-disk-comparable hash)"
            )
        elif declared_sha != actual_sha:
            errs.append(
                f"{tid}: kernel.py sha256 mismatch: "
                f"meta has {declared_sha[:16]}…, actual {actual_sha[:16]}…"
            )

    return errs


def _load_eligible_ids() -> tuple[set[str], int]:
    """Return (score_eligible_ids, catalogue_expected_eligible_count).

    Single source of truth: triton_ascend_bench.eligibility (shared
    with cli.py -- plan U7, prevents drift). The expected count is
    derived independently from iter_all_tasks() so the non-vacuity
    guard can detect a wrong task-root / empty fixture / compute bug.
    """
    sys.path.insert(0, str(REPO_ROOT))
    from triton_ascend_bench.dataset import iter_all_tasks
    from triton_ascend_bench.eligibility import compute_score_eligible

    all_tasks = iter_all_tasks()
    eligible = compute_score_eligible(all_tasks)
    return eligible, len(eligible)


def run_release_gate(tasks=None, eligible_ids=None) -> int:
    """`--release-gate` mode (plan U7): SEPARATE from schema/hash mode.

    `eligibility  intersect  {phase_a_prime.status != PASS}` non-empty ⇒ nonzero
    exit + the blocking list. PLUS a non-vacuity guard (codex-cep-r3
    must-fix): if the examined eligible count == 0, OR != the
    catalogue-expected eligible count, this is a HARD FAIL -- never a
    vacuous pass ("no tasks so nothing BLOCKED so releasable" is the
    degenerate path S6 / no-subset-escape forbids).

    `tasks` / `eligible_ids` are injectable for unit tests; in
    production they are loaded from the real catalogue.
    """
    if tasks is None or eligible_ids is None:
        eligible_ids, expected = _load_eligible_ids()
        from triton_ascend_bench.dataset import iter_all_tasks

        tasks = iter_all_tasks()
    else:
        expected = len(eligible_ids)

    # MATERIALISE before BOTH passes (codex FIX 2 [confirmed]): the
    # function iterates `tasks` once for `examined` and again for
    # `blocking`. If `tasks` is a one-shot generator the 2nd pass is
    # empty -> `blocking` is [] -> a vacuous return 0 ("releasable")
    # even with eligible non-PASS tasks. list() makes it re-iterable.
    tasks = list(tasks)

    if expected == 0:
        print(
            "RELEASE-GATE HARD FAIL: catalogue-expected eligible count is 0 "
            "(bad task root / empty fixture / eligibility compute bug). "
            "Refusing to vacuously pass.",
            file=sys.stderr,
        )
        return 2

    all_tasks = tasks
    examined = {t.task_id for t in all_tasks if t.task_id in eligible_ids}
    if len(examined) == 0 or len(examined) != expected:
        print(
            f"RELEASE-GATE HARD FAIL: examined eligible count "
            f"{len(examined)} != catalogue-expected {expected} "
            f"(non-vacuity guard; never vacuous-pass).",
            file=sys.stderr,
        )
        return 2

    blocking: list[str] = []
    for t in all_tasks:
        if t.task_id not in eligible_ids:
            continue
        pap = (t.meta or {}).get("phase_a_prime") or {}
        if pap.get("status") != "PASS":
            blocking.append(t.task_id)
    blocking.sort()

    if blocking:
        print(
            f"RELEASE-GATE FAIL: {len(blocking)} score-eligible task(s) "
            f"with phase_a_prime.status != PASS -- benchmark NOT releasable:",
            file=sys.stderr,
        )
        for tid in blocking:
            print(f"  - {tid}", file=sys.stderr)
        return 1

    print(
        f"release-gate OK: all {expected} score-eligible tasks "
        f"phase_a_prime.status == PASS."
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", help="Validate a single task id (skip the rest)")
    parser.add_argument("--strict", action="store_true",
                        help="Stop on first task error")
    parser.add_argument(
        "--release-gate", action="store_true",
        help="Phase A' S6 release gate (SEPARATE from schema/hash mode): "
             "eligible  intersect  {phase_a_prime.status != PASS} non-empty ⇒ fail; "
             "non-vacuity guarded (never vacuous-pass).",
    )
    args = parser.parse_args()

    if args.release_gate:
        return run_release_gate()

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
