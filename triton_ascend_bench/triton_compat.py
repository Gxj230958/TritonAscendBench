"""Triton-version compatibility transforms applied at kernel LOAD time.

A frozen upstream kernel is byte-immutable on disk, but some upstream sources use
constructs a newer/different Triton than the upstream's own target rejects at
compile time. The only such case in this bench is `tl.device_assert` debug
instrumentation guarded by Python control flow over a *vectorized* value
(``if (cur_batch == 0 and cur_head == 0): tl.device_assert(...)`` in SGLang's MLA
decode stage1): Triton 3.6.0 evaluates the vector ``cur_head == 0`` through
``Tensor.__bool__`` and raises "Boolean value of Tensor with more than one value
is ambiguous".

`tl.device_assert` is debug-only instrumentation (it only aborts when
``TRITON_DEBUG`` is set and writes NO output), so removing it is behaviour-
preserving for output grading. This transform strips ONLY ``tl.device_assert``
expression statements inside ``@triton.jit`` bodies, applied to the in-memory
source the runner execs -- the kernel.py on disk is never modified. It does NOT
touch ``tl.static_assert`` (a compile-time check), ``tl.assume``, or Python
``assert`` statements.
"""
from __future__ import annotations

import ast


def _attr_chain(node: ast.AST) -> str:
    """Dotted name of an attribute/name expression (e.g. ``tl.device_assert``)."""
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
    return ".".join(reversed(parts))


def _is_device_assert(stmt: ast.stmt) -> bool:
    return (
        isinstance(stmt, ast.Expr)
        and isinstance(stmt.value, ast.Call)
        and _attr_chain(stmt.value.func)
        in ("tl.device_assert", "triton.language.device_assert", "device_assert")
    )


#: Exact decorator chains that mark a Triton kernel (a bare ``@jit`` re-export, or
#: the dotted ``triton.jit`` / ``triton.language.jit`` forms) -- NOT a loose
#: ``endswith("jit")`` which would also match an unrelated ``@myjit``.
_JIT_DECORATORS = ("jit", "triton.jit", "triton.language.jit", "tl.jit")


def _has_jit_decorator(fn: ast.AST) -> bool:
    if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return False
    for d in fn.decorator_list:
        target = d.func if isinstance(d, ast.Call) else d
        if _attr_chain(target) in _JIT_DECORATORS:
            return True
    return False


def strip_jit_device_asserts(src: str) -> str | None:
    """Return ``src`` with every ``tl.device_assert(...)`` statement removed from
    ``@triton.jit`` function bodies, or ``None`` if there is nothing to strip (so
    the caller keeps the unmodified load path -- a no-op for the 100+ tasks that
    carry no device_assert).

    A guard ``if`` whose body becomes empty after removal (and has no ``else``) is
    dropped; any other compound body that would be left empty gets a ``pass`` so the
    result stays valid. The scrub recurses into EVERY nested statement list
    (``if``/``for``/``while``/``with`` bodies, ``try`` handlers/else/finally,
    ``match`` cases), so a device_assert anywhere in the jit body is removed; if a
    construct were somehow missed the assert simply survives and fails to compile
    (fail-closed, never a silent wrong result).
    """
    if "device_assert" not in src:
        return None
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return None

    removed = 0

    def scrub_stmts(stmts: list[ast.stmt]) -> list[ast.stmt]:
        nonlocal removed
        out: list[ast.stmt] = []
        for s in stmts:
            if _is_device_assert(s):
                removed += 1
                continue
            scrub_children(s)
            if isinstance(s, ast.If) and not s.body:
                # The guard's body was only device_assert(s). Drop the whole
                # guard if there is no else; otherwise keep it with a pass body
                # (an `if` with an else still needs a non-empty body).
                if not s.orelse:
                    continue
                s.body = [ast.Pass()]
            out.append(s)
        return out

    def scrub_children(node: ast.AST) -> None:
        for field, value in ast.iter_fields(node):
            if isinstance(value, list) and value and all(
                isinstance(x, ast.stmt) for x in value
            ):
                new = scrub_stmts(value)
                # Pass-fill a non-If compound body that emptied (for/while/with/
                # def/try need a body); an If body is left empty here so the
                # CALLER's scrub_stmts can drop the whole guard.
                if not new and field in ("body", "finalbody") and not isinstance(
                    node, ast.If
                ):
                    new = [ast.Pass()]
                setattr(node, field, new)
            elif isinstance(value, list):
                # lists of non-stmt nodes (ExceptHandler, match_case) -- descend.
                for item in value:
                    if isinstance(item, ast.AST):
                        scrub_children(item)
            elif isinstance(value, ast.AST):
                scrub_children(value)

    for node in ast.walk(tree):
        if _has_jit_decorator(node):
            node.body = scrub_stmts(node.body) or [ast.Pass()]

    if removed == 0:
        return None
    ast.fix_missing_locations(tree)
    return ast.unparse(tree)


def load_with_compat_strip(mod, kernel_src: str, file_tag: str) -> bool:
    """Exec a device_assert-stripped ``kernel_src`` into module ``mod`` so a
    frozen kernel whose debug-assert won't compile on this Triton still loads.

    Returns True if the strip applied (the caller must NOT then call
    ``exec_module``), or False if there was nothing to strip (the caller keeps
    its normal ``spec.loader.exec_module`` path -- a no-op for the 100+ tasks
    without a device_assert).

    The stripped source is exec'd under a synthetic ``co_filename`` whose lines
    are registered in ``linecache``, so Triton's ``inspect.getsource`` on the
    ``@triton.jit`` functions reads the SAME stripped lines the bytecode was
    compiled from -- pointing it at the on-disk file instead would desync line
    numbers (the file still has the assert) and mis-resolve the kernel's globals.
    """
    stripped = strip_jit_device_asserts(kernel_src)
    if stripped is None:
        return False
    import hashlib
    import linecache

    # Key the synthetic co_filename on the stripped CONTENT, not just the path, so
    # a re-load of the same path with different content never reads a stale entry.
    digest = hashlib.sha256(stripped.encode()).hexdigest()[:12]
    tag = f"<triton-compat:{file_tag}:{digest}>"
    linecache.cache[tag] = (
        len(stripped),
        None,
        stripped.splitlines(keepends=True),
        tag,
    )
    exec(compile(stripped, tag, "exec"), mod.__dict__)
    return True
