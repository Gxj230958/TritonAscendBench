#!/usr/bin/env python3
"""phase_a_freeze.py — Phase A batch freeze: vendor upstream GPU triton sources.

Iterates over tasks/level{1,2,3}/<task_id>/, resolves each task's
upstream.path against /data1/code/agentic-rl/kernel/infer/{vllm,sglang}/,
locates the file that defines the kernel function named in meta.yaml,
vendors the FULL upstream source file as kernel.py (with provenance
header), and updates meta.yaml's commit_sha / file_sha256 / vendored
snapshot pointer.

Whole-file vendoring (not per-function extraction) is deliberate for
Phase A: it's simpler, preserves the original Apache-2.0 license header
verbatim, and lets the static checker — which already accepts the full
upstream form for the 5 existing PoCs — operate on the same canonical
input.

Tasks whose upstream.path is shorthand (`vLLM 通用`, `同上`, bare repo
name) or whose kernel function cannot be found are reported as
`needs_manual` and left untouched. Existing 5 PoCs (kernel.py already
hand-curated) are skipped via SKIP_TASKS.

Usage:
    python scripts/phase_a_freeze.py --limit 30        # freeze first 30
    python scripts/phase_a_freeze.py --task L1_02_...  # one task
    python scripts/phase_a_freeze.py --dry-run         # report only
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
TASKS_ROOT = REPO_ROOT / "tasks"
VENDORED_ROOT = REPO_ROOT / "ascendbench" / "frozen" / "snapshots"
INFER_ROOT = REPO_ROOT.parent / "infer"

VLLM_ROOT = INFER_ROOT / "vllm"
SGLANG_ROOT = INFER_ROOT / "sglang" / "python"
FBGEMM_ROOT = REPO_ROOT / "external" / "fbgemm"

VLLM_COMMIT = "18f6bf5a214fd51d0cb08fcf1b59edc5629e7c19"
SGLANG_COMMIT = "409d350fb6f6a1e7c7546e39028f811092a8e489"
FBGEMM_COMMIT = "bf6dce360a4fe133bc779e2fd036277678509f95"

# 5 existing PoCs — kernel.py is hand-curated; leave alone.
SKIP_TASKS = {
    "L1_01_bias",
    "L1_12_rms_norm",
    "L2_04_silu_and_mul",
    "L2_17_w8a8_block_int8_matmul",
    "L3_16_fwd",
}

AMBIGUOUS_PATH_MARKERS = ("通用", "同上", "TODO", "<")


@dataclass
class ResolveResult:
    task_id: str
    repo: str
    kernel_name: str
    raw_path: str
    status: str            # "ok" | "needs_manual" | "skip" | "error"
    source_file: Path | None = None
    commit_sha: str = ""
    file_sha256: str = ""
    diagnostics: list[str] = field(default_factory=list)


def _strip_md(s: str) -> str:
    return s.strip().strip("`").strip()


def _repo_root(repo: str) -> Path:
    if repo == "vLLM":
        return VLLM_ROOT
    if repo == "SGLang":
        return SGLANG_ROOT
    if repo == "FBGEMM":
        return FBGEMM_ROOT
    return INFER_ROOT


def _commit_sha(repo: str) -> str:
    if repo == "vLLM":
        return VLLM_COMMIT
    if repo == "SGLang":
        return SGLANG_COMMIT
    if repo == "FBGEMM":
        return FBGEMM_COMMIT
    return "snapshot-no-git"


def _file_contains_kernel(path: Path, kernel_name: str) -> bool:
    """True iff `path` is a .py file declaring a function named `kernel_name`."""
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except (OSError, UnicodeDecodeError):
        return False
    # Cheap pre-filter so we don't AST-parse every file.
    if f"def {kernel_name}" not in text:
        return False
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return False
    for n in ast.walk(tree):
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == kernel_name:
            return True
    return False


def resolve_source(task_id: str, repo: str, raw_path: str, kernel_name: str) -> ResolveResult:
    res = ResolveResult(
        task_id=task_id, repo=repo, kernel_name=kernel_name,
        raw_path=raw_path, status="error", commit_sha=_commit_sha(repo),
    )
    cleaned = _strip_md(raw_path)

    # Shorthand path: keep a diagnostic but still try whole-tree search by
    # kernel function name (resolver step 3 below).
    shorthand = (
        not cleaned
        or any(m in cleaned for m in AMBIGUOUS_PATH_MARKERS)
        or cleaned in {"vLLM", "SGLang", "FBGEMM"}
        or " " in cleaned
        or "/" not in cleaned
    )
    if shorthand:
        res.diagnostics.append(f"shorthand path: {raw_path!r}, falling back to whole-tree search")

    # Build candidate list. SGLang paths in the markdown often start
    # `sglang/srt/...` but the on-disk prefix is `sglang/python/sglang/srt/`.
    bases: list[Path] = []
    if repo == "vLLM":
        bases = [VLLM_ROOT]
    elif repo == "SGLang":
        bases = [SGLANG_ROOT, INFER_ROOT / "sglang"]
    elif repo == "FBGEMM":
        bases = [FBGEMM_ROOT]
    else:
        bases = [VLLM_ROOT, SGLANG_ROOT, FBGEMM_ROOT]

    candidates: list[Path] = []
    for base in bases:
        # Direct join.
        candidates.append(base / cleaned)
        # SGLang quirk: markdown writes "sglang/srt/..." → on disk "python/sglang/srt/...".
        if repo == "SGLang" and cleaned.startswith("sglang/"):
            candidates.append(SGLANG_ROOT / cleaned)
        # Strip leading repo-name segment if present.
        head, _, tail = cleaned.partition("/")
        if head.lower() in {"vllm", "sglang"} and tail:
            candidates.append(base / tail)

    # 1) Exact .py hit
    file_hits: list[Path] = []
    for c in candidates:
        if c.is_file() and c.suffix == ".py":
            if _file_contains_kernel(c, kernel_name):
                file_hits.append(c)

    # 2) Directory hit → glob inside and grep for the kernel def
    if not file_hits:
        seen_dirs: set[Path] = set()
        for c in candidates:
            if c.is_dir() and c not in seen_dirs:
                seen_dirs.add(c)
                for py in sorted(c.rglob("*.py")):
                    if _file_contains_kernel(py, kernel_name):
                        file_hits.append(py)
                        break

    # 3) Whole-base fallback. When the markdown path is shorthand
    # (`同上`, `vLLM sample`, ...), search all known repos by kernel name.
    if not file_hits:
        search_bases = bases if not shorthand else [VLLM_ROOT, SGLANG_ROOT, FBGEMM_ROOT]
        for base in search_bases:
            for py in base.rglob("*.py"):
                # Skip third-party / tests to reduce false positives.
                rel = py.relative_to(base).as_posix()
                if rel.startswith(("tests/", "benchmarks/", "examples/")):
                    continue
                if _file_contains_kernel(py, kernel_name):
                    file_hits.append(py)
                    break
            if file_hits:
                break

    if not file_hits:
        if res.status != "needs_manual":
            res.status = "needs_manual"
            res.diagnostics.append(
                f"kernel def {kernel_name!r} not found under {[str(b) for b in bases]}"
            )
        return res

    if len(file_hits) > 1:
        res.diagnostics.append(
            f"multiple matches: {[str(p) for p in file_hits]}; using first"
        )

    res.source_file = file_hits[0]
    res.status = "ok"
    return res


def _provenance_header(
    task_id: str, kernel_name: str, source_path: str, repo: str,
    commit_sha: str, file_sha256: str,
) -> str:
    return (
        f'"""Task {task_id} — frozen upstream GPU triton source.\n\n'
        f"Vendored verbatim from the upstream repository for benchmark use\n"
        f"under the Apache-2.0 license. The original license header (below)\n"
        f"is preserved; see NOTICE at the repository root for attribution.\n\n"
        f"Kernel function: `{kernel_name}`\n"
        f"Source path: `{source_path}`\n"
        f"Upstream repo: {repo}\n"
        f"Commit SHA: {commit_sha}\n"
        f"File SHA-256 (this vendored copy): {file_sha256}\n"
        f"Frozen by: scripts/phase_a_freeze.py\n"
        f'"""\n\n'
    )


def freeze_task(task_id: str, dry_run: bool) -> ResolveResult:
    # Locate task dir.
    task_dir: Path | None = None
    for level in ("level1", "level2", "level3"):
        c = TASKS_ROOT / level / task_id
        if c.is_dir():
            task_dir = c
            break
    if task_dir is None:
        return ResolveResult(
            task_id=task_id, repo="", kernel_name="", raw_path="",
            status="error", diagnostics=[f"task dir not found"],
        )

    if task_id in SKIP_TASKS:
        return ResolveResult(
            task_id=task_id, repo="", kernel_name="", raw_path="",
            status="skip", diagnostics=["existing PoC, hand-curated"],
        )

    meta = yaml.safe_load((task_dir / "meta.yaml").read_text())
    repo = meta.get("repo", "")
    raw_path = meta["upstream"]["path"]
    kernel_name = meta["name"]

    res = resolve_source(task_id, repo, raw_path, kernel_name)
    if res.status != "ok" or res.source_file is None:
        return res

    upstream_bytes = res.source_file.read_bytes()
    upstream_text = upstream_bytes.decode("utf-8", errors="replace")
    new_sha = hashlib.sha256(upstream_bytes).hexdigest()

    base = _repo_root(repo)
    try:
        rel_src = res.source_file.relative_to(base).as_posix()
    except ValueError:
        rel_src = res.source_file.as_posix()

    header = _provenance_header(
        task_id=task_id, kernel_name=kernel_name,
        source_path=rel_src, repo=repo, commit_sha=res.commit_sha,
        file_sha256=new_sha,
    )
    new_kernel_text = header + upstream_text
    new_kernel_sha = hashlib.sha256(new_kernel_text.encode("utf-8")).hexdigest()
    res.file_sha256 = new_kernel_sha

    # Vendored snapshot (raw, no header).
    sha7 = res.commit_sha[:7]
    VENDORED_ROOT.mkdir(parents=True, exist_ok=True)
    vendored_dst = VENDORED_ROOT / f"{task_id}_{sha7}.py"

    if dry_run:
        res.diagnostics.append(f"would-write kernel.py ({len(new_kernel_text)} bytes)")
        res.diagnostics.append(f"would-write vendored {vendored_dst.name}")
        return res

    # Write kernel.py and vendored snapshot.
    (task_dir / "kernel.py").write_text(new_kernel_text)
    vendored_dst.write_bytes(upstream_bytes)

    # Update meta.yaml.
    meta["upstream"]["commit_sha"] = res.commit_sha
    meta["upstream"]["file_sha256"] = new_kernel_sha
    meta["upstream"]["function_hash"] = new_kernel_sha
    meta["upstream"]["vendored_snapshot"] = (
        vendored_dst.relative_to(REPO_ROOT).as_posix()
    )
    # Bump benchmark_version to mark this task as no-longer-scaffold.
    meta["benchmark_version"] = "0.1.0-frozen"
    (task_dir / "meta.yaml").write_text(
        yaml.safe_dump(meta, sort_keys=False, allow_unicode=True)
    )
    return res


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, default=30,
                    help="Stop after N successful freezes (default 30)")
    ap.add_argument("--task", help="Freeze one task only (overrides --limit)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--report", help="Write JSON summary to this path")
    args = ap.parse_args()

    if not VLLM_ROOT.is_dir():
        print(f"missing vLLM source: {VLLM_ROOT}", file=sys.stderr)
        return 1
    if not SGLANG_ROOT.is_dir():
        print(f"missing SGLang source: {SGLANG_ROOT}", file=sys.stderr)
        return 1

    if args.task:
        task_ids = [args.task]
    else:
        task_ids = []
        for level in ("level1", "level2", "level3"):
            for d in sorted((TASKS_ROOT / level).iterdir()):
                if d.is_dir() and not d.name.startswith("_"):
                    task_ids.append(d.name)

    results: list[ResolveResult] = []
    n_ok = 0
    for tid in task_ids:
        if not args.task and args.limit and n_ok >= args.limit:
            break
        res = freeze_task(tid, dry_run=args.dry_run)
        results.append(res)
        flag = {
            "ok": "✓", "skip": "·", "needs_manual": "?", "error": "✗"
        }.get(res.status, "?")
        diag = f" ({res.diagnostics[0]})" if res.diagnostics else ""
        print(f"  {flag} {tid:55s} [{res.status}]{diag}")
        if res.status == "ok":
            n_ok += 1

    counts: dict[str, int] = {}
    for r in results:
        counts[r.status] = counts.get(r.status, 0) + 1

    print()
    print(f"== Summary ==  scanned={len(results)}  ok={counts.get('ok', 0)}  "
          f"skip={counts.get('skip', 0)}  needs_manual={counts.get('needs_manual', 0)}  "
          f"error={counts.get('error', 0)}")

    if args.report:
        Path(args.report).write_text(json.dumps(
            [
                {
                    "task_id": r.task_id, "status": r.status, "repo": r.repo,
                    "kernel_name": r.kernel_name, "raw_path": r.raw_path,
                    "source_file": str(r.source_file) if r.source_file else None,
                    "commit_sha": r.commit_sha, "file_sha256": r.file_sha256,
                    "diagnostics": r.diagnostics,
                }
                for r in results
            ], indent=2,
        ))
        print(f"  report → {args.report}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
