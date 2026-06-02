#!/usr/bin/env python3
"""P2 operator-slicer: rewrite a task's kernel.py to ONLY the graded operator's
reachable closure, dropping the unrelated launched sibling kernels that
whole-module vendoring left behind (e.g. the 21-kernel cutlass-MoE module copied
into 9 tasks).

Closure = the top-level statements (functions / classes / module constants /
imports) transitively referenced from the TRUE graded roots (entrypoint +
launcher launched-JITs, via scripts.multi_launched_manifest). Everything else --
the unreachable launched siblings and their exclusive helpers -- is dropped.

FAIL-CLOSED: the slice is produced ONLY if (a) roots resolve and (b) the sliced
source has NO undefined free name (every Name load is closure-defined, imported,
or a builtin). Otherwise returns None and the caller keeps the whole file. This
NEVER touches the committed kernel.py; promotion is a separate S2/C2-gated step.
"""
from __future__ import annotations

import argparse
import ast
import builtins
import glob
from pathlib import Path

BENCH = Path(__file__).resolve().parents[1]
_BUILTINS = set(dir(builtins))
# names the standalone gets from its own import rewrites / runtime
_ALWAYS_OK = {"triton", "tl", "torch", "math", "__name__", "__file__", "tlx"}


def _toplevel(tree: ast.Module) -> list[ast.stmt]:
    return list(tree.body)


def _defined_names(stmt: ast.stmt) -> set[str]:
    """Top-level names a statement binds."""
    out: set[str] = set()
    if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        out.add(stmt.name)
    elif isinstance(stmt, ast.Assign):
        for t in stmt.targets:
            out |= {n.id for n in ast.walk(t) if isinstance(n, ast.Name)}
    elif isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
        out.add(stmt.target.id)
    elif isinstance(stmt, (ast.Import, ast.ImportFrom)):
        for a in stmt.names:
            out.add((a.asname or a.name).split(".")[0])
    return out


def _refs(node: ast.AST) -> set[str]:
    """All Name loads referenced anywhere under node (body, decorators, defaults,
    bases, annotations)."""
    return {n.id for n in ast.walk(node) if isinstance(n, ast.Name)}


def slice_source(src: str, td: Path) -> str | None:
    import sys

    sys.path.insert(0, str(BENCH / "scripts"))
    from multi_launched_manifest import _func_defs, _graded_roots

    try:
        tree = ast.parse(src)
    except SyntaxError:
        return None
    defs = _func_defs(tree)
    roots = _graded_roots(td, defs)
    if not roots:
        return None

    body = _toplevel(tree)
    # Statement-level dependency graph: each top-level statement is a node that
    # DEFINES names and REFERENCES names. A statement is kept iff it is reachable
    # from the graded roots, so a sibling-only dispatch table (Assign that names
    # the dropped kernels but is never used by the graded operator) is itself
    # dropped -- it does not drag its siblings back in.
    name2stmt: dict[str, list[int]] = {}
    stmt_refs: list[set[str]] = []
    for i, s in enumerate(body):
        for nm in _defined_names(s):
            name2stmt.setdefault(nm, []).append(i)
        stmt_refs.append(_refs(s))

    # Unconditionally-kept statements: imports + module-scope side-effecting code
    # (If/Try/With/Expr that isn't a def/class/assign). These RUN at import, so
    # their references must resolve -- seed the closure with them so a module
    # constant they read (e.g. `_is_cuda`) is pulled in, not dangled.
    always_keep = {
        i
        for i, s in enumerate(body)
        if isinstance(s, (ast.Import, ast.ImportFrom))
        or not isinstance(
            s,
            (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Assign, ast.AnnAssign),
        )
    }
    reached: set[int] = set()
    stack: list[int] = list(always_keep)
    for r in roots:  # seed: statements defining a graded root + always-kept
        stack += name2stmt.get(r, [])
    while stack:
        i = stack.pop()
        if i in reached:
            continue
        reached.add(i)
        for ref in stmt_refs[i]:
            for j in name2stmt.get(ref, []):
                if j not in reached:
                    stack.append(j)

    kept_idx = reached | always_keep
    if len(kept_idx) == len(body):
        return None  # nothing dropped -> not a slice candidate

    kept = [body[i] for i in sorted(kept_idx)]
    new_tree = ast.Module(body=kept, type_ignores=[])
    sliced = ast.unparse(ast.fix_missing_locations(new_tree))

    # FAIL-CLOSED free-name audit: every Name load in the sliced module must be
    # closure-defined, imported, a builtin, or an always-ok runtime name.
    stree = ast.parse(sliced)
    bound: set[str] = set(_ALWAYS_OK) | _BUILTINS
    for s in stree.body:
        bound |= _defined_names(s)
    # collect module-level + function-local free names: a name LOADED that is
    # neither module-bound, a builtin, nor locally bound inside the function is a
    # genuine dangling reference. Local binding = every Store-context Name
    # (assigns, for/with/comprehension/walrus targets) + args + nested def/class
    # names + except handlers -- so loop/comprehension vars are not false leaks.
    def _local_bound(fn: ast.AST) -> set[str]:
        b: set[str] = set()
        for n in ast.walk(fn):
            if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store):
                b.add(n.id)
            elif isinstance(n, ast.arg):
                b.add(n.arg)
            elif isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                b.add(n.name)
            elif isinstance(n, ast.ExceptHandler) and n.name:
                b.add(n.name)
            elif isinstance(n, (ast.Import, ast.ImportFrom)):
                for a in n.names:
                    b.add((a.asname or a.name).split(".")[0])
        return b

    def _loads(node: ast.AST) -> set[str]:
        return {
            n.id
            for n in ast.walk(node)
            if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)
        }

    free: set[str] = set()
    for s in stree.body:
        if isinstance(s, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            local = _local_bound(s)
            for r in _loads(s):
                if r not in bound and r not in local:
                    free.add(r)
        else:
            for r in _loads(s):
                if r not in bound:
                    free.add(r)
    if free:
        return None  # undefined free name(s) -> decline (keep whole file)
    return sliced if sliced.endswith("\n") else sliced + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("task", nargs="?", help="task id (or omit to summarise scope)")
    ap.add_argument("--write", metavar="PATH", help="write sliced kernel.py to PATH")
    args = ap.parse_args()
    if args.task:
        td = next(iter(glob.glob(str(BENCH / f"tasks/level*/{args.task}"))), None)
        if not td:
            print(f"no such task {args.task}")
            return 1
        td = Path(td)
        src = (td / "kernel.py").read_text()
        sliced = slice_source(src, td)
        if sliced is None:
            print(f"{args.task}: DECLINED (unresolved roots / nothing-to-drop / free-name)")
            return 2
        print(
            f"{args.task}: sliced {len(src.splitlines())} -> {len(sliced.splitlines())} lines"
        )
        if args.write:
            Path(args.write).write_text(sliced)
            print(f"  wrote {args.write}")
        return 0
    # summary over the manifest scope
    import sys

    sys.path.insert(0, str(BENCH / "scripts"))
    from multi_launched_manifest import analyse

    ok = dec = 0
    for kp in sorted(glob.glob(str(BENCH / "tasks/level*/*/kernel.py"))):
        td = Path(kp).parent
        r = analyse(td)
        if not (r and r["in_scope"]):
            continue
        s = slice_source((td / "kernel.py").read_text(), td)
        if s is None:
            dec += 1
            print(f"  DECLINE {td.name}")
        else:
            ok += 1
            print(
                f"  OK      {td.name}: "
                f"{len((td / 'kernel.py').read_text().splitlines())}->{len(s.splitlines())}"
            )
    print(f"# sliceable {ok}, declined {dec} of the manifest scope")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
