#!/usr/bin/env python3
"""build_oracle_digest.py — compact oracle↔kernel digest for review.

For each task whose oracle_pytorch.py is a working reference (not a
NotImplementedError stub), emit a compact block:

  - oracle: helper def signature(s) + docstring + run() body
  - kernel: @triton.jit def name + arg list + the constexpr "optional
    path" toggles (HAS_*/USE_*/IS_*/*_SOFTPLUS/...) + launcher wrapper
    signature

This lets a reviewer judge math/I-O fidelity without cat-ing whole
files (which exhausts the reviewer's output budget).
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
TASKS = REPO / "tasks"

_TOGGLE_PREFIXES = ("HAS_", "USE_", "IS_", "NO_", "APPLY_", "SAVE_")
_TOGGLE_SUBSTR = ("SOFTPLUS", "CAUSAL", "SPLIT", "MODE", "ALIBI",
                  "BIAS", "INITSTATES", "VARLEN", "SCALE")


def _oracle_summary(src: str) -> str:
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return "  <oracle parse error>"
    out: list[str] = []
    mod_doc = ast.get_docstring(tree)
    for n in tree.body:
        if isinstance(n, ast.FunctionDef):
            args = []
            a = n.args
            for x in (list(a.posonlyargs) + list(a.args)):
                args.append(x.arg)
            if a.vararg:
                args.append("*" + a.vararg.arg)
            if a.kwonlyargs:
                args += ["*"] + [x.arg for x in a.kwonlyargs]
            if a.kwarg:
                args.append("**" + a.kwarg.arg)
            doc = (ast.get_docstring(n) or "").strip()
            out.append(f"  def {n.name}({', '.join(args)})")
            if doc and n.name != "run":
                for line in doc.splitlines():
                    out.append(f"    | {line.rstrip()}")
            if n.name == "run":
                seg = ast.get_source_segment(src, n) or ""
                for line in seg.splitlines():
                    out.append(f"    > {line}")
    return "\n".join(out) if out else "  <no functions>"


def _kernel_summary(src: str, kernel_name: str) -> str:
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return "  <kernel parse error>"
    lines: list[str] = []
    for n in ast.walk(tree):
        if isinstance(n, ast.FunctionDef) and n.name == kernel_name:
            a = n.args
            allargs = [x.arg for x in (list(a.posonlyargs) + list(a.args)
                                       + list(a.kwonlyargs))]
            # constexpr toggles
            toggles = [
                arg for arg in allargs
                if arg.upper() == arg and (
                    any(arg.startswith(p) for p in _TOGGLE_PREFIXES)
                    or any(s in arg for s in _TOGGLE_SUBSTR))
            ]
            tensor_like = [
                arg for arg in allargs
                if arg.endswith(("_ptr", "_ptrs")) or arg in (
                    "A", "B", "C", "As", "Bs")
            ]
            lines.append(f"  @triton.jit {kernel_name}")
            lines.append(f"    ptr/tensor args: {tensor_like}")
            lines.append(f"    optional-path toggles: {toggles}")
            break
    # launcher wrapper (python fn that calls the kernel)
    for n in ast.walk(tree):
        if (isinstance(n, ast.FunctionDef) and n.name != kernel_name):
            seg = ast.get_source_segment(src, n) or ""
            if kernel_name in seg and "@triton.jit" not in seg[:60]:
                a = n.args
                wargs = [x.arg for x in (list(a.posonlyargs) + list(a.args))]
                lines.append(f"  launcher {n.name}({', '.join(wargs)})")
                break
    return "\n".join(lines) if lines else f"  <{kernel_name} not found>"


def main() -> int:
    import yaml
    blocks: list[str] = []
    n = 0
    for level in ("level1", "level2", "level3"):
        for d in sorted((TASKS / level).iterdir()):
            if not d.is_dir() or d.name.startswith("_"):
                continue
            op = d / "oracle_pytorch.py"
            kp = d / "kernel.py"
            if not op.exists():
                continue
            osrc = op.read_text()
            if "raise NotImplementedError" in osrc:
                continue
            meta = yaml.safe_load((d / "meta.yaml").read_text())
            kname = meta.get("name", "")
            cat = meta.get("category", "?")
            tier = meta.get("oracle_simplified", "?")
            tol = meta.get("tolerance_class", "?")
            ksrc = kp.read_text() if kp.exists() else ""
            blocks.append(
                f"================================================================\n"
                f"TASK {d.name}  [cat={cat} tier={tier} tol={tol}]\n"
                f"--- ORACLE ---\n{_oracle_summary(osrc)}\n"
                f"--- KERNEL ({kname}) ---\n{_kernel_summary(ksrc, kname)}\n"
            )
            n += 1
    out = Path("/tmp/_oracle_digest.txt")
    out.write_text(f"# Oracle↔Kernel digest — {n} working-ref tasks\n\n"
                   + "\n".join(blocks))
    print(f"wrote {out} ({n} tasks, {out.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
