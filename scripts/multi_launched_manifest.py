#!/usr/bin/env python3
"""Emit the P2 slice scope from code: tasks whose kernel.py launches Triton
kernels that are NOT reachable from the task's TRUE graded path (entrypoint +
launcher). Those unreachable launches are the droppable sibling operators a
single-operator slice removes; a task whose every launch is graded-path-reachable
(target + its real helpers) is a legitimate multi-launch operator and is NOT in
scope.

The predicate is graded-path REACHABILITY, never text launch-detection: a true
launch is the AST shape ``name[grid](args)`` (``Call(func=Subscript(Name)))``,
robust to nested-bracket grids (the L3_01 ``reduce_segments`` false-positive a
naive ``name[grid](`` regex hits). Reachability walks the call graph from the
graded root through host wrappers down to the JIT launches.

Usage: ``python3 scripts/multi_launched_manifest.py [--json]`` -> the scope list.
"""
from __future__ import annotations

import argparse
import ast
import glob
import json
from pathlib import Path

import yaml

BENCH = Path(__file__).resolve().parents[1]


def _func_defs(tree: ast.AST) -> dict[str, ast.AST]:
    return {
        n.name: n
        for n in ast.walk(tree)
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
    }


def _is_jit(fn: ast.AST) -> bool:
    return any("triton.jit" in ast.unparse(d) for d in fn.decorator_list)


def _launches_in(fn: ast.AST) -> set[str]:
    """Names launched as ``name[grid](...)`` inside ``fn`` (AST, nested-bracket safe)."""
    out: set[str] = set()
    for c in ast.walk(fn):
        if isinstance(c, ast.Call) and isinstance(c.func, ast.Subscript):
            v = c.func.value
            if isinstance(v, ast.Name):
                out.add(v.id)
    return out


def _calls_in(fn: ast.AST) -> set[str]:
    """Plain-call + launched names referenced inside ``fn`` (call-graph edges)."""
    out: set[str] = set()
    for c in ast.walk(fn):
        if isinstance(c, ast.Call):
            f = c.func
            if isinstance(f, ast.Name):
                out.add(f.id)
            elif isinstance(f, ast.Subscript) and isinstance(f.value, ast.Name):
                out.add(f.value.id)
    return out


def _reachable(roots: set[str], defs: dict[str, ast.AST]) -> set[str]:
    seen: set[str] = set()
    stack = [r for r in roots if r in defs]
    while stack:
        name = stack.pop()
        if name in seen:
            continue
        seen.add(name)
        for callee in _calls_in(defs[name]):
            if callee in defs and callee not in seen:
                stack.append(callee)
    return seen


def _graded_roots(td: Path, defs: dict[str, ast.AST]) -> set[str]:
    """The true graded entry function name(s) in kernel.py."""
    meta = yaml.safe_load((td / "meta.yaml").read_text()) or {}
    ep = str((meta.get("oracle") or {}).get("gpu_triton") or "")
    launcher = td / "_pa_launcher.py"
    # Launcher-driven (incl. the 20 ambiguous `kernel.py::run + launcher` tasks):
    # the launcher's run() references the graded kernel.py functions -- use those.
    if launcher.is_file() and "def run" in launcher.read_text():
        try:
            ltree = ast.parse(launcher.read_text())
            run = _func_defs(ltree).get("run")
            if run is not None:
                refs = _calls_in(run) | {
                    n.id for n in ast.walk(run) if isinstance(n, ast.Name)
                }
                roots = {r for r in refs if r in defs}
                if roots:
                    return roots
        except SyntaxError:
            pass
    sym = ep.split("::", 1)[1] if "::" in ep else ep
    if sym in defs:
        return {sym}
    return set()  # unresolved -> caller treats as not-sliceable


def analyse(td: Path) -> dict | None:
    kp = td / "kernel.py"
    try:
        tree = ast.parse(kp.read_text())
    except SyntaxError:
        return None
    defs = _func_defs(tree)
    jits = {n for n, fn in defs.items() if _is_jit(fn)}
    launched = {j for j in jits if any(j in _launches_in(d) for d in defs.values())}
    roots = _graded_roots(td, defs)
    reach = _reachable(roots, defs)
    reachable_launched = launched & reach
    droppable = sorted(launched - reach)
    return {
        "task": td.name,
        "roots": sorted(roots),
        "launched": sorted(launched),
        "reachable_launched": sorted(reachable_launched),
        "droppable": droppable,
        "in_scope": bool(droppable) and bool(roots),
        "unresolved_roots": not roots,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", action="store_true", help="full per-task JSON")
    args = ap.parse_args()
    recs = []
    for kp in sorted(glob.glob(str(BENCH / "tasks/level*/*/kernel.py"))):
        r = analyse(Path(kp).parent)
        if r:
            recs.append(r)
    scope = [r["task"] for r in recs if r["in_scope"]]
    unresolved = [r["task"] for r in recs if r["unresolved_roots"]]
    if args.json:
        print(json.dumps({"scope": scope, "records": recs}, indent=1))
    else:
        for t in scope:
            print(t)
    import sys

    print(
        f"# scope (droppable-sibling) = {len(scope)} tasks; "
        f"unresolved-root = {len(unresolved)} {unresolved}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
