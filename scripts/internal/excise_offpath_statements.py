#!/usr/bin/env python3
"""Byte-faithful purity slicer: rewrite a task's kernel.py to ONLY the graded
operator's reachable closure + the authored run/launcher harness, by EXCISING the
source byte-ranges of off-path top-level statements (siblings, dead wrappers, dead
classes, vendored test functions, unused loggers/constants, and off-path
module-level side-effect blocks). Every KEPT statement stays byte-identical to the
frozen upstream source -- no ast.unparse reformatting.

Closure rule (graded-path reachability):
  roots = graded entry symbol(s) (meta.oracle.gpu_triton kernel.py::SYM) + the
  kernel.py names the launcher's run() references. K0 = closure of roots over
  Name refs, with all imports always kept. A module-level side-effect statement
  (If/Try/Expr/With/For/While) is KEPT iff it references >=1 KEPT non-import
  def/class/assign symbol (it is part of the kept operator's init); otherwise it
  is off-path and dropped together with its now-unreachable closure. Statements
  with no top-level-symbol reference (module docstring, __future__) are kept.

FAIL-CLOSED gates (any failure => DECLINE, original returned unchanged):
  1. kept-statement byte-span equality (the primary correctness proof);
  2. symtable free-name audit (no dangling Name);
  3. out-of-drop-span scan: no dropped symbol name appears anywhere in the slice
     (catches string / torch.ops / getattr references the Name closure misses).
"""
from __future__ import annotations

import argparse
import ast
import builtins
import glob
import sys
from pathlib import Path

import yaml

BENCH = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BENCH / "scripts" / "internal"))
from list_multi_launch_tasks import _func_defs  # noqa: E402
from slice_single_operator_kernel import _defined_names, _refs, _toplevel  # noqa: E402

_BUILTINS = set(dir(builtins))
_ALWAYS_OK = {"triton", "tl", "torch", "math", "__name__", "__file__", "tlx"}

# The audit-identified 25 DIRTY tasks (docs/reviews/2026-06-12-kernel-purity-audit.md).
# Writing is refused for any task NOT in this allowlist unless --force, so a no-arg
# run can never touch the 76 already-clean tasks.
DIRTY_TASKS = frozenset({
    # L1_07 / L1_08 added 2026-06-13: vendored `if __name__=="__main__"` self-test block
    # (the audit's reachability had kept it because it calls the graded operator).
    "L1_07_dequantize_k_cache_fast", "L1_08_dequantize_k_cache_paged",
    "L1_02_compute_slot_mappings", "L1_09_quantize_k_cache_fast", "L1_13_fused_softcap",
    "L1_16_apply_grammar_bitmask", "L1_19_triton_mrope_forward",
    "L1_22_transform_index_page_table_decode", "L2_01_layer_norm_fwd",
    "L2_02_layer_norm_fwd_1pass", "L2_08_gumbel_sample", "L2_12_pack_bitmatrix",
    "L2_13_silu_mul_fp8_quant_deep_gemm", "L2_15_chunked_lora_shrink", "L2_16_sgemm_lora_a",
    "L2_18_w8a8_triton_block_scaled_mm", "L2_28_moe_align_block_size_stage2",
    "L2_31_scaled_mm", "L3_03_chunked_lora_expand", "L3_04_qkv_lora_b",
    "L3_05_gate_up_lora_b", "L3_06_sgemm_lora_b", "L3_12_selective_state_update",
    "L3_13_kernel_unified_attention_3d", "L3_14_kernel_paged_attention_2d",
    "L3_22_chunk_gated_delta_rule_fwd_kernel_h_blockdim64", "L3_29_recompute_w_u_fwd",
})

_DEFCLASSASSIGN = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Assign, ast.AnnAssign)


def _launcher_run(td: Path):
    lp = td / "_pa_launcher.py"
    if not lp.is_file():
        return None
    src = lp.read_text()
    if "def run" not in src:
        return None
    try:
        return _func_defs(ast.parse(src)).get("run")
    except SyntaxError:
        return None


def _launcher_refs(td: Path) -> set[str]:
    run = _launcher_run(td)
    if run is None:
        return set()
    return {n.id for n in ast.walk(run) if isinstance(n, ast.Name)}


def _strict_roots(td: Path, defs: dict) -> set[str]:
    """Graded-path roots matching the runner's _resolve_entrypoint:
      * meta.oracle.gpu_triton / interface.entrypoint = kernel.py::SYM -> add SYM.
      * launcher-graded (a _pa_launcher.py with run() exists): the launcher's run()
        is exec'd into kernel.py's module and SHADOWS kernel.py's own run(); the live
        roots are the kernel.py names the launcher's run references -- NOT the dead
        kernel.py run().
      * no launcher: kernel.py's own run() is the authored standalone harness -> add it.
    """
    meta = yaml.safe_load((td / "meta.yaml").read_text()) or {}
    roots: set[str] = set()
    for ep in (str((meta.get("oracle") or {}).get("gpu_triton") or ""),
               str((meta.get("interface") or {}).get("entrypoint") or "")):
        if "::" in ep:
            f, sym = ep.split("::", 1)
            if f.endswith("kernel.py") and sym in defs:
                roots.add(sym)
    has_launcher = _launcher_run(td) is not None
    if has_launcher:
        roots |= {r for r in _launcher_refs(td) if r in defs}
    elif "run" in defs:
        roots.add("run")
    return roots


def _is_main_guard(stmt: ast.stmt) -> bool:
    """True for a top-level ``if __name__ == "__main__":`` block. Such a block is a
    vendored self-test / demo harness -- NEVER executed when the module is imported
    for grading -- so it (and any symbol reachable only through it, e.g. a `*_ref`
    reference impl) is always off-path and must be dropped, regardless of whether it
    references the graded operator."""
    if not isinstance(stmt, ast.If):
        return False
    t = stmt.test
    if not isinstance(t, ast.Compare) or not all(isinstance(o, ast.Eq) for o in t.ops):
        return False  # only `==` (not `!=`, which would invert the guard)
    cmps = [t.left, *t.comparators]
    names = {n.id for n in cmps if isinstance(n, ast.Name)}
    consts = {c.value for c in cmps if isinstance(c, ast.Constant)}
    return "__name__" in names and "__main__" in consts


def _bound_names_deep(stmt: ast.stmt) -> set[str]:
    """Top-level names a statement binds at MODULE scope, recursing into module-level
    control-flow blocks (If/Try/With/For/While) -- so a name conditionally defined by
    `if TRITON3: def softplus(...)` is attributed to that block. Does NOT recurse into
    function/class bodies (those bind local, not module, names)."""
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
    elif isinstance(stmt, (ast.If, ast.Try, ast.With, ast.AsyncWith, ast.For, ast.AsyncFor, ast.While)):
        for sub in ast.iter_child_nodes(stmt):
            if isinstance(sub, ast.stmt):
                out |= _bound_names_deep(sub)
    return out


def _self_module_attr_refs(stmt: ast.stmt) -> set[str]:
    """Resolve self-module reflection: ``_kmod = sys.modules[...]`` then
    ``_kmod.<attr>`` -- the Name-load closure misses ``<attr>`` because it is an
    attribute, not a bare Name (the L1_02 ``_compute_slot_mappings_kernel`` launch).
    Returns the attribute names accessed off any sys.modules-bound alias in ``stmt``."""
    aliases: set[str] = set()
    for n in ast.walk(stmt):
        if isinstance(n, ast.Assign) and isinstance(n.value, ast.Subscript):
            base = n.value.value
            if isinstance(base, ast.Attribute) and base.attr == "modules" \
                    and isinstance(base.value, ast.Name) and base.value.id == "sys":
                for t in n.targets:
                    if isinstance(t, ast.Name):
                        aliases.add(t.id)
    if not aliases:
        return set()
    out: set[str] = set()
    for n in ast.walk(stmt):
        if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name) and n.value.id in aliases:
            out.add(n.attr)
    return out


def _stmt_line_span(stmt: ast.stmt) -> tuple[int, int]:
    """1-indexed inclusive [start, end] source line range, decorator-aware."""
    start = stmt.lineno
    for d in getattr(stmt, "decorator_list", []) or []:
        start = min(start, d.lineno)
    return start, stmt.end_lineno


def compute_drop_set(src: str, td: Path) -> tuple[list[int], dict] | None:
    """Return (dropped_stmt_indices, info) or None on unresolved roots / syntax err."""
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return None
    body = _toplevel(tree)
    defs = _func_defs(tree)
    # DIRECT top-level bound names (shallow): the only launcher refs eligible as roots,
    # so a kernel.py module CONSTANT the launcher uses as a launch kwarg (e.g. NUM_WARPS)
    # is kept, WITHOUT pulling in a demo block merely because it binds a same-named local
    # (e.g. `num_tokens` inside `if __name__=='__main__'`).
    direct_top = set()
    for s in body:
        direct_top |= _defined_names(s)
    roots = _strict_roots(td, defs) | (_launcher_refs(td) & direct_top)
    if not roots:
        return None

    # `if __name__ == "__main__"` self-test/demo blocks are NEVER on the graded path ->
    # exclude them from reachability ENTIRELY: do not register the local names they bind
    # (so a kept function referencing a same-named local, e.g. `result`, cannot pull the
    # block back in -- the L1_22 collision), and force them into the drop-set below.
    main_guards = {i for i, s in enumerate(body) if _is_main_guard(s)}

    name2stmt: dict[str, list[int]] = {}
    stmt_refs: list[set[str]] = []
    for i, s in enumerate(body):
        if i not in main_guards:
            for nm in _bound_names_deep(s):  # deep: catches names bound in module-level If/Try blocks
                name2stmt.setdefault(nm, []).append(i)
        stmt_refs.append(_refs(s) | _self_module_attr_refs(s))  # + sys.modules self-alias attrs

    is_import = [isinstance(s, (ast.Import, ast.ImportFrom)) for s in body]
    is_dca = [isinstance(s, _DEFCLASSASSIGN) for s in body]
    is_side = [not is_import[i] and not is_dca[i] for i in range(len(body))]
    # top-level symbols bound by def/class/assign (NOT imports)
    dca_names: set[str] = set()
    for i, s in enumerate(body):
        if is_dca[i]:
            dca_names |= _defined_names(s)

    def closure(seed_idx: set[int]) -> set[int]:
        reached: set[int] = set()
        stack = list(seed_idx)
        while stack:
            i = stack.pop()
            if i in reached:
                continue
            reached.add(i)
            for ref in stmt_refs[i]:
                for j in name2stmt.get(ref, []):
                    if j not in reached:
                        stack.append(j)
        return reached

    # K0: imports always kept; seed from root-defining statements (NOT side effects).
    seed = {i for i in range(len(body)) if is_import[i]}
    for r in roots:
        seed |= set(name2stmt.get(r, []))
    kept = closure(seed)

    # Fixpoint: keep a side-effect statement iff it references a KEPT non-import
    # def/class/assign symbol (it is load-bearing init for kept code). A side
    # effect with no def/class/assign reference (module docstring) is kept too.
    changed = True
    while changed:
        changed = False
        kept_dca_names = {nm for i in kept if is_dca[i] for nm in _defined_names(body[i])}
        for i in range(len(body)):
            if i in kept or not is_side[i] or i in main_guards:
                continue  # never keep a __main__ self-test block
            refs_dca = stmt_refs[i] & dca_names  # off/on-path top-level symbols it touches
            keep_it = (not refs_dca) or bool(refs_dca & kept_dca_names)
            if keep_it:
                add = closure(kept | {i})
                if add - kept:
                    kept |= add
                    changed = True
                elif i not in kept:
                    kept.add(i)
                    changed = True

    kept -= main_guards  # safety: a __main__ guard is never kept, even if a closure grazed it
    considerable = {i for i in range(len(body)) if is_dca[i] or is_side[i]}
    dropped = sorted(considerable - kept)
    info = {
        "roots": sorted(roots),
        "dropped_names": [
            (body[i].name if isinstance(body[i], (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
             else ("<side-effect>" if is_side[i] else ",".join(sorted(_defined_names(body[i])))))
            for i in dropped
        ],
        "body": body,
        "dropped_idx": dropped,
    }
    return dropped, info


def excise(src: str, td: Path) -> tuple[str, dict]:
    """Return (sliced_src, report). report['status'] in OK / CLEAN_ALREADY / DECLINE."""
    res = compute_drop_set(src, td)
    if res is None:
        return src, {"status": "DECLINE", "reason": "unresolved-roots-or-syntax"}
    dropped, info = res
    if not dropped:
        return src, {"status": "CLEAN_ALREADY", "dropped": []}

    body = info["body"]
    lines = src.splitlines(keepends=True)
    drop_lines: set[int] = set()
    for i in dropped:
        a, b = _stmt_line_span(body[i])
        drop_lines.update(range(a, b + 1))
    sliced = "".join(ln for n, ln in enumerate(lines, 1) if n not in drop_lines)
    if not sliced.endswith("\n"):
        sliced += "\n"

    rep = {"status": "OK", "dropped": info["dropped_names"], "roots": info["roots"],
           "n_lines": f"{len(lines)}->{len(sliced.splitlines())}"}

    # GATE 1: kept-statement byte-span equality (PRIMARY proof) -- EVERY retained
    # top-level statement (defs, classes, assigns, imports, AND kept If/Try/side-effect
    # blocks like L3_12's conditional softplus) must be byte-identical to the original.
    kept_idx = [i for i in range(len(body)) if i not in set(dropped)]
    orig_segs = [ast.get_source_segment(src, body[i]) for i in kept_idx]
    try:
        sbody = _toplevel(ast.parse(sliced))
    except SyntaxError as e:
        return src, {"status": "DECLINE", "reason": f"sliced-syntax-error: {e}"}
    sliced_segs = [ast.get_source_segment(sliced, s) for s in sbody]
    if orig_segs != sliced_segs:
        return src, {"status": "DECLINE", "reason": "kept-statement byte-span mismatch"}

    # GATE 2: free-name audit (symtable)
    import symtable
    try:
        top = symtable.symtable(sliced, "<sliced>", "exec")
    except SyntaxError as e:
        return src, {"status": "DECLINE", "reason": f"symtable: {e}"}
    module_bound = {s.get_name() for s in top.get_symbols()
                    if s.is_assigned() or s.is_imported() or s.is_namespace()}
    ok = module_bound | _BUILTINS | _ALWAYS_OK
    free: set[str] = set()

    def _audit(tbl):
        for s in tbl.get_symbols():
            if s.is_global() and not s.is_assigned() and s.get_name() not in ok:
                free.add(s.get_name())
        for c in tbl.get_children():
            _audit(c)

    _audit(top)
    if free:
        return src, {"status": "DECLINE", "reason": f"free-names: {sorted(free)}"}

    # GATE 3: out-of-drop-span registry/string scan over the sliced AST (NOT raw text,
    # so docstrings/comments that merely mention a dropped name are not false positives).
    # The real hazard is a dropped symbol reached from kept code by string/attribute:
    #   torch.ops.<dropped>(...) / getattr(x, "<dropped>") / a string literal that EXACTLY
    #   equals a dropped symbol name (registry key). Name refs are already caught by GATE 2.
    import re as _re
    dropped_syms = {n for n in info["dropped_names"]
                    if n != "<side-effect>" and _re.fullmatch(r"[A-Za-z_]\w*", n)}
    stree = ast.parse(sliced)
    # collect docstring node ids to exclude module/func/class docstrings
    doc_ids = set()
    for node in ast.walk(stree):
        body_attr = getattr(node, "body", None)
        if isinstance(body_attr, list) and body_attr and isinstance(body_attr[0], ast.Expr) \
                and isinstance(getattr(body_attr[0], "value", None), ast.Constant) \
                and isinstance(body_attr[0].value.value, str):
            doc_ids.add(id(body_attr[0].value))
    survivors: set[str] = set()
    for node in ast.walk(stree):
        if isinstance(node, ast.Attribute) and node.attr in dropped_syms:
            survivors.add(node.attr)            # x.<dropped> / torch.ops.<dropped>
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) not in doc_ids:
            if node.value in dropped_syms:       # exact-string registry key
                survivors.add(node.value)
    if survivors:
        return src, {"status": "DECLINE", "reason": f"dropped-symbol reached by string/attr in kept code: {sorted(survivors)}"}

    return sliced, rep


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("task", nargs="?", help="task id; omit to run the 25 DIRTY tasks")
    ap.add_argument("--write-staging", action="store_true", help="write slice to .cache/purity_slices/<T>/kernel.py")
    ap.add_argument("--write-tasks", action="store_true", help="write slice INTO tasks/<T>/kernel.py (working tree)")
    ap.add_argument("--all", action="store_true", help="iterate the whole corpus (report only)")
    ap.add_argument("--force", action="store_true", help="allow writing a non-DIRTY task")
    args = ap.parse_args()

    if args.task:
        tasks = [args.task]
    elif args.all:
        tasks = [Path(p).parent.name for p in sorted(glob.glob(str(BENCH / "tasks/level*/*/kernel.py")))]
    else:
        tasks = sorted(DIRTY_TASKS)  # default scope = the 25 dirty tasks (never the 76 clean)

    writing = args.write_staging or args.write_tasks

    n_ok = n_decline = n_clean = 0
    for t in tasks:
        td = next((Path(p) for p in glob.glob(str(BENCH / f"tasks/level*/{t}"))), None)
        if td is None:
            continue
        src = (td / "kernel.py").read_text()
        sliced, rep = excise(src, td)
        st = rep["status"]
        if st == "OK":
            n_ok += 1
            print(f"OK       {t:<50} {rep['n_lines']:<14} drops {len(rep['dropped'])}: {rep['dropped']}")
            if writing and t not in DIRTY_TASKS and not args.force:
                print(f"  REFUSED to write {t}: not in the 25 DIRTY allowlist (use --force to override)")
                continue
            if args.write_staging:
                out = BENCH / ".cache" / "purity_slices" / t / "kernel.py"
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_text(sliced)
            if args.write_tasks:
                (td / "kernel.py").write_text(sliced)
                print(f"  wrote slice INTO {td / 'kernel.py'}")
        elif st == "CLEAN_ALREADY":
            n_clean += 1
            print(f"CLEAN    {t}")
        else:
            n_decline += 1
            print(f"DECLINE  {t:<50} {rep.get('reason')}")
    if not args.task:
        print(f"\n# OK={n_ok}  CLEAN_ALREADY={n_clean}  DECLINE={n_decline}  of {len(tasks)}")
    return 0 if n_decline == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
