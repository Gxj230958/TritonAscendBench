"""static_checker.py — Layer A–F static defense for solution.py.

Per `specs/ast_denylist.md` v0 (R11-P1-8 / R12-P1-6 / R14-P1-5):

    Layer A — Import denylist
    Layer B — AST node-type allowlist (positive)
    Layer C — Call dotted-name allowlist
    Layer D — Attribute dotted-name denylist
    Layer E — Runtime audit hook (`install_runtime_audit_hook`; the
              runner installs it inside the isolated candidate child)
    Layer F — Source string denylist (case-insensitive)

Returns a CheckReport with `(layer, severity, message, line)` findings.
"""
from __future__ import annotations

import ast
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

try:
    import tomllib  # Python 3.11+
except ModuleNotFoundError:  # pragma: no cover — runtime fallback for 3.10
    import tomli as tomllib  # type: ignore[no-redef]

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_VERSION_MATRIX = REPO_ROOT / "infra" / "version_matrix.toml"


@dataclass
class Finding:
    layer: str
    severity: str          # "block" | "warn"
    message: str
    line: int | None = None


@dataclass
class CheckReport:
    findings: list[Finding] = field(default_factory=list)

    @property
    def blocked(self) -> bool:
        return any(f.severity == "block" for f in self.findings)

    def add(self, layer: str, sev: str, msg: str, line: int | None = None) -> None:
        self.findings.append(Finding(layer, sev, msg, line))


# ─────────────────────────────────────────────────────────────────────────────
# Layer A — Import denylist (specs/ast_denylist.md §Layer A)
# ─────────────────────────────────────────────────────────────────────────────
_DENYLIST_IMPORT_EXACT = {
    "functools.partial", "functools.partialmethod",   # round-21 R-P0-1
    "exec", "eval", "compile", "__import__",
    "importlib", "importlib.util", "importlib.import_module",
    "ctypes", "ctypes.CDLL", "cffi",
    "torch.utils.cpp_extension",
    "torch.utils.cpp_extension.load",
    "torch.utils.cpp_extension.load_inline",
    "subprocess", "socket", "requests", "urllib", "http", "pickle",
    "threading", "multiprocessing", "asyncio", "concurrent.futures",
    "os", "os.path", "pathlib", "mmap", "fcntl", "shutil", "tempfile",
    "glob",
    "time", "datetime",
    "torch.cuda.Event", "torch.npu.Event", "torch_npu.npu.Event",
    "torch.profiler", "torch_npu.profiler",
    "oracle_pytorch", "triton_ascend_bench", "tasks",
    # V9 plan M2.0 (R19 verified-runner boundary, v1 minimum-viable):
    # candidate source MUST NOT import the safetensors loader or
    # torch deserialization primitives. The verified runner holds the
    # sealed-golden tensors in process memory; if a candidate could
    # call `safetensors.torch.load_file` on a sealed path it would
    # short-circuit the R14 file-mode + R18 signature gates. The full
    # process-isolation model (uid switching + MADV_DONTFORK) ships
    # in the security follow-up plan; this Layer A static deny is the
    # immediate v1 defence.
    "safetensors", "safetensors.torch",
    "safetensors.torch.load_file", "safetensors.torch.save_file",
    "torch.load", "torch.serialization",
    "torch.serialization.load",
}
_DENYLIST_IMPORT_PREFIX = {
    "vllm_ascend.npu_",
    "sgl_kernel_npu",
    "torch_npu.npu_",
}

# Default import allowlist for solution.py (project pin-locked).
# `operator` removed (round-4 codex: `from operator import attrgetter`
# exposes reflection).  `functools` allowed but `partial` calls are
# audited via RHS-walk in Assign (round-4 codex: `partial(open, ...)`).
_DEFAULT_IMPORT_ALLOWLIST = {
    "torch",
    "torch_npu",
    "triton",
    "triton.language",
    "math",
    "typing",
    "dataclasses",
    "functools",
    "numbers",
    "warnings",
    # `__future__` is a frozen stdlib module of compiler-flag objects
    # only (no callables touching FS/network/reflection). `from
    # __future__ import annotations` is standard in the project's own
    # oracle/scaffold templates; blocking it was a pure false positive
    # with zero attack surface (FP fix, not a security relaxation).
    "__future__",
}


# ─────────────────────────────────────────────────────────────────────────────
# Layer B — AST node-type allowlist (specs/ast_denylist.md §Layer B)
# ast.MatMult intentionally NOT in allowlist (R14-P1-5).
# ─────────────────────────────────────────────────────────────────────────────
_AST_ALLOW: set[type] = {
    ast.Module, ast.Expr, ast.Expression,
    ast.Import, ast.ImportFrom, ast.alias, ast.keyword,
    ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Return,
    ast.arguments, ast.arg,
    ast.If, ast.For, ast.While, ast.Break, ast.Continue, ast.Pass,
    ast.Try, ast.ExceptHandler,
    ast.Assert, ast.Raise,           # host wrappers may guard shapes
    ast.Constant, ast.Name, ast.Load, ast.Store, ast.Del,
    ast.BinOp, ast.UnaryOp, ast.BoolOp, ast.Compare, ast.Lambda,
    ast.IfExp, ast.Subscript, ast.Slice, ast.Tuple, ast.List, ast.Dict, ast.Set,
    ast.Starred,
    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow,
    ast.LShift, ast.RShift, ast.BitOr, ast.BitXor, ast.BitAnd,
    ast.USub, ast.UAdd, ast.Not, ast.Invert,
    ast.Eq, ast.NotEq, ast.Lt, ast.Gt, ast.LtE, ast.GtE,
    ast.In, ast.NotIn, ast.Is, ast.IsNot,
    ast.And, ast.Or,
    ast.Attribute, ast.Call,
    ast.AugAssign, ast.Assign, ast.AnnAssign, ast.AugStore,
    ast.Index,    # 3.10 compat; harmless on 3.11+
}


# ─────────────────────────────────────────────────────────────────────────────
# Layer C — Call dotted-name allowlist for non-trivial calls.
#
# Free identifiers (Name calls) and local function calls are allowed
# implicitly. Dotted calls (Attribute calls) must match one of these
# prefixes OR equal one of these exact names.
# ─────────────────────────────────────────────────────────────────────────────
_CALL_ALLOW_EXACT = {
    "torch.empty", "torch.zeros", "torch.ones", "torch.full",
    "torch.empty_like", "torch.zeros_like", "torch.ones_like",
    "torch.tensor", "torch.as_tensor", "torch.arange",
    "torch.finfo", "torch.iinfo",
    "triton.cdiv", "triton.next_power_of_2",
    "triton.heuristics", "triton.autotune", "triton.jit", "triton.Config",
    "tl.load", "tl.store", "tl.dot", "tl.sum", "tl.max", "tl.min",
    "tl.exp", "tl.log", "tl.sqrt", "tl.rsqrt", "tl.sigmoid", "tl.where",
    "tl.arange", "tl.zeros", "tl.full", "tl.cdiv",
    "tl.program_id", "tl.constexpr", "tl.static_range",
    "tl.maximum", "tl.minimum", "tl.multiple_of", "tl.max_contiguous",
    "tl.num_programs", "tl.advance",
    "triton.language.load", "triton.language.store",   # explicit alias
}
# Methods whose result is itself a Tensor (Tensor-ness propagates).
_TENSOR_RETURNING_METHOD_SUFFIX = {
    ".reshape", ".view", ".contiguous", ".transpose", ".permute",
    ".unsqueeze", ".squeeze",
    ".to", ".new_empty", ".new_zeros", ".new_ones", ".new_full",
    ".repeat_interleave",
    ".clone",
}

# Methods whose result is a scalar (int / bool / tuple of ints).
# These break the Tensor-ness chain — e.g. `A.numel()` returns int.
_SCALAR_RETURNING_METHOD_SUFFIX = {
    ".size", ".stride", ".numel", ".dim", ".is_contiguous",
}

_CALL_ALLOW_METHOD_SUFFIX = (
    _TENSOR_RETURNING_METHOD_SUFFIX | _SCALAR_RETURNING_METHOD_SUFFIX
)

# Read-only attribute suffixes that produce non-Tensor scalars.
_PROP_ALLOW_SUFFIX = {
    ".dtype", ".device", ".shape", ".ndim", ".element_ty",
}


# ─────────────────────────────────────────────────────────────────────────────
# Layer D — Attribute dotted-name denylist.
# Anything starting with these prefixes (anywhere in source) is rejected
# even if reached via dynamic attribute lookup.
# ─────────────────────────────────────────────────────────────────────────────
_ATTR_DENY_PREFIXES = (
    "torch.ops.",
    "torch.classes.",
    "torch.fx.",
    "torch.compile",
    "torch.nn.functional",
    "torch.nn.Module",
    "torch.cuda.stream",
    "torch.npu.stream",
    "torch_npu.npu.stream",
    "torch_npu.npu_",
    "sgl_kernel_npu",
    "vllm_ascend.npu_",
)
_ATTR_DENY_METHOD_SUFFIXES = (
    ".cpu", ".numpy", ".item", ".tolist", ".detach",
)

# Method names that are unsafe regardless of receiver base (codex round-6 #1).
# Catches `inputs["x"].matmul(...)`, `f = x.cpu`, etc. — the base is a
# runtime Tensor, so dotted-prefix checks can't see it. Anything in this
# set blocks at the Attribute node itself.
_BANNED_METHOD_NAMES = {
    # Eager-path callables (would defeat the migration benchmark).
    "matmul", "mm", "bmm", "einsum",
    "softmax", "log_softmax",
    "addmm", "addbmm",
    # Host transfer / reflection / introspection.
    "cpu", "numpy", "item", "tolist", "detach",
    "share_memory_", "data_ptr",
    "untyped_storage", "storage",
}
# Names that are dangerous when *referenced* (any use), because they enable
# reflection/eval primitives that defeat all other checks.
# Round-4 codex: `open` moved here because `partial(open, ...)` is a bypass.
_BANNED_REFLECT_NAMES = {
    "exec", "eval", "compile", "__import__",
    "getattr", "setattr", "delattr",
    "globals", "locals", "vars",
    "breakpoint",
    "type",                        # type(x).__getattribute__ bypass
    "open",
    "attrgetter", "itemgetter", "methodcaller",   # from operator
    "partial", "partialmethod",                    # from functools — round-20
}

# Names that are forbidden only as call targets — they are commonly used
# as variable names in ML code (e.g. `def f(input):`).
_BANNED_CALL_ONLY = {
    "input", "hasattr", "dir",
}

# Builtins that force PyTorch tensors into eager numerical evaluation.
# Banned as Call targets in host-side code regardless of receiver.
# (Codex round-7: `sum(x)`, `int(x)`, `float(x)`, `list(x)` etc.)
# Always-banned: these have no legitimate use in solution.py and only
# appear when an attacker is laundering Tensor operations.
_BANNED_EAGER_BUILTINS = {
    "iter", "next", "divmod",
}

# Tensor-context-banned: legitimate as `float("inf")` etc. but illegal
# when applied to a Tensor-typed local. The type-flow pass below
# rejects these when the arg is a known Tensor Name.
_TENSOR_COERCE_BUILTINS = {
    "sum", "min", "max", "abs", "round", "pow",
    "int", "float", "bool", "complex",
    "list", "tuple", "set", "dict", "frozenset",
    "any", "all", "sorted", "reversed",   # codex round-9 NEW-P0 #4
}

# Backward-compat alias — used in some places that check both.
_BANNED_BUILTIN_CALLS = _BANNED_REFLECT_NAMES | _BANNED_CALL_ONLY

# Banned identifiers anywhere in source — these expose reflection / dunder
# escape hatches that bypass attribute denylists.
_BANNED_DUNDER_NAMES = {
    "__builtins__", "__import__",
    "__getattribute__", "__getattr__", "__setattr__", "__delattr__",
    "__class__", "__bases__", "__mro__", "__subclasses__",
    "__dict__", "__globals__", "__code__", "__closure__",
    "__loader__", "__spec__", "__package__",
    "__class_getitem__",
}

# Banned tl.* calls — these are inline-asm / unsafe escape hatches.
# Both `tl.X` and `triton.language.X` forms must be denied so that
# `from triton.language import inline_asm_elementwise as asm` is caught.
_BANNED_TL_CALLS = {
    "tl.inline_asm_elementwise",
    "tl.extern_elementwise",
    "tl.inline_asm",
    "triton.language.inline_asm_elementwise",
    "triton.language.extern_elementwise",
    "triton.language.inline_asm",
}

# Layer C extension — torch.* paths permitted beyond `_CALL_ALLOW_EXACT`.
# Used by the round-4 alias-canonicalization check to allow `Tensor.foo`
# attribute access (which the AST parser sees as `torch.X.foo`).
_TORCH_DOTTED_ALLOW_PREFIXES = (
    "torch.dtype",
    "torch.device",
    "torch.float16",
    "torch.float32",
    "torch.bfloat16",
    "torch.int8",
    "torch.int16",
    "torch.int32",
    "torch.int64",
    "torch.bool",
    "torch.uint8",
)


# ─────────────────────────────────────────────────────────────────────────────
# Layer F — Source string denylist (case-insensitive substring).
# ─────────────────────────────────────────────────────────────────────────────
_STRING_DENY_PATTERNS = (
    r"\boracle\b",
    r"\bgolden\b",
    r"\bhidden_shape\b",
    r"\bhidden_set\b",
    r"\bcheat\b",
    r"\banswer_key\b",
)


# ─────────────────────────────────────────────────────────────────────────────
# Layer F+ — Required ops (selected from meta.yaml.runner_checks.static_checks)
# ─────────────────────────────────────────────────────────────────────────────
REQUIRED_OPS = {
    "cube_arithmetic_present": [("tl.dot",)],
    "online_softmax_present": [("tl.exp",)],
}


def load_allowlist(version_matrix_path: Path = DEFAULT_VERSION_MATRIX) -> set[str]:
    """Load import allowlist from version_matrix.toml or fall back to defaults."""
    if not version_matrix_path.exists():
        return set(_DEFAULT_IMPORT_ALLOWLIST)
    data = tomllib.loads(version_matrix_path.read_text())
    extra = data.get("import_allowlist", {}).get("modules", [])
    return set(_DEFAULT_IMPORT_ALLOWLIST) | set(extra)


def check_solution(
    source: str,
    required_checks: list[str] | None = None,
    allowlist: set[str] | None = None,
) -> CheckReport:
    """Run all layers on `source` and return findings."""
    report = CheckReport()
    if allowlist is None:
        allowlist = load_allowlist()

    # Layer F: source string denylist (cheap, run first)
    for pat in _STRING_DENY_PATTERNS:
        m = re.search(pat, source, re.IGNORECASE)
        if m:
            line = source.count("\n", 0, m.start()) + 1
            report.add("F-string-deny", "block",
                       f"banned substring matched: {pat}", line=line)

    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        report.add("syntax", "block", f"SyntaxError: {e}", line=e.lineno)
        return report

    # Layer B: AST node allowlist
    for node in ast.walk(tree):
        t = type(node)
        if t in {ast.Load, ast.Store, ast.Del}:
            continue
        if t not in _AST_ALLOW:
            report.add(
                "B-ast-node",
                "block",
                f"disallowed AST node: {t.__name__}",
                line=getattr(node, "lineno", None),
            )

    # Pass 1a: build the alias canonicalization table (first binding wins).
    # Wildcard imports are rejected outright at the same pass.
    aliases: dict[str, str] = {}
    all_bindings: dict[str, list[str]] = {}    # name -> all observed qualifieds

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                local = a.asname or a.name.split(".")[0]
                if local not in aliases:
                    aliases[local] = a.name
                all_bindings.setdefault(local, []).append(a.name)
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            for a in node.names:
                if a.name == "*":
                    report.add(
                        "A-wildcard-import", "block",
                        f"`from {mod or '?'} import *` is forbidden — "
                        "introduces untracked symbols",
                        getattr(node, "lineno", None),
                    )
                    continue
                local = a.asname or a.name
                qualified = f"{mod}.{a.name}" if mod else a.name
                if local not in aliases:
                    aliases[local] = qualified
                all_bindings.setdefault(local, []).append(qualified)

    def _canonical(n: ast.AST | None) -> str:
        """Return canonical dotted name, resolving aliases from the table."""
        if n is None:
            return ""
        if isinstance(n, ast.Name):
            return aliases.get(n.id, n.id)
        if isinstance(n, ast.Attribute):
            base = _canonical(n.value)
            return f"{base}.{n.attr}" if base else n.attr
        return ""

    def _is_dotted_banned(dotted: str) -> str | None:
        """Return reason if this dotted-name reference (not just a call)
        is banned. Used for Assign-RHS scans and bare Name reference checks.

        Only `_BANNED_REFLECT_NAMES` are rejected here; `_BANNED_CALL_ONLY`
        names (`input`, `hasattr`) are *legitimate variable names* and
        only banned when actually invoked as Call.func.
        """
        if not dotted:
            return None
        if dotted in _DENYLIST_IMPORT_EXACT:
            return f"banned import {dotted!r}"
        if any(dotted.startswith(p) for p in _DENYLIST_IMPORT_PREFIX):
            return f"banned import prefix {dotted!r}"
        if any(dotted.startswith(p) for p in _ATTR_DENY_PREFIXES):
            return f"banned attribute path {dotted!r}"
        if dotted in _BANNED_TL_CALLS:
            return f"banned tl call {dotted!r}"
        if dotted in _BANNED_REFLECT_NAMES:
            return f"banned reflection name {dotted!r}"
        # `torch.X` references outside the explicit Layer-C allowlist are
        # rejected — high-level torch eager ops (matmul, mm, einsum,
        # nn.*, ops.*) must not be reachable from solution.py.
        if dotted.startswith("torch."):
            if dotted not in _CALL_ALLOW_EXACT and not any(
                dotted.startswith(p) for p in _TORCH_DOTTED_ALLOW_PREFIXES
            ):
                return f"non-allowlisted torch reference {dotted!r}"
        return None

    # Pass 1c: simple Tensor-flow type tracking for host-side eager bans
    # (codex round-7 #1). Per non-kernel function, infer which local
    # Names are Tensors, then reject BinOp / Subscript / for-iter on them.
    _TENSOR_CTORS = (
        "torch.empty", "torch.empty_like",
        "torch.zeros", "torch.zeros_like",
        "torch.ones", "torch.ones_like",
        "torch.full", "torch.full_like",
        "torch.tensor", "torch.as_tensor", "torch.arange",
    )

    def _rhs_yields_tensor(value: ast.AST, tensors: set[str], containers: set[str] | None = None) -> bool:
        """Round-9: broadened — Lambda body, Tuple/List/Dict literals,
        IfExp arms, comprehensions, all walked for Tensor references."""
        if isinstance(value, ast.Call):
            fname = _canonical(value.func) or _dotted_name(value.func)
            if fname in _TENSOR_CTORS or any(
                fname.startswith(p) for p in _TENSOR_CTORS
            ):
                return True
            if isinstance(value.func, ast.Attribute):
                # codex round-10 P1 #2: respect scalar-returning methods —
                # `x.numel()` / `.size()` / `.dim()` produce ints, not Tensors.
                suffix = "." + value.func.attr
                if suffix in _SCALAR_RETURNING_METHOD_SUFFIX or suffix in _PROP_ALLOW_SUFFIX:
                    return False
                if _expr_contains_tensor(value.func.value, tensors, containers):
                    return True
            if isinstance(value.func, ast.Name):
                for arg in value.args + [kw.value for kw in value.keywords]:
                    if _expr_contains_tensor(arg, tensors, containers):
                        return True
            # codex round-9/13/17/18: `(lambda v: ...)(x)` — bind formals,
            # *args, **kwargs, defaults, and keywords.
            if isinstance(value.func, ast.Lambda):
                bound = set(tensors)
                bound_c = set(containers or set())
                lam = value.func
                pos_formals = list(lam.args.posonlyargs) + list(lam.args.args)
                defaults = list(lam.args.defaults)
                n_defaults = len(defaults)
                # Positional binding.
                for i, formal in enumerate(pos_formals):
                    actual: ast.AST | None = None
                    if i < len(value.args):
                        actual = value.args[i]
                    else:
                        default_idx = i - (len(pos_formals) - n_defaults)
                        if 0 <= default_idx < n_defaults:
                            actual = defaults[default_idx]
                    if actual is None:
                        continue
                    if _expr_contains_tensor(actual, tensors, containers):
                        bound.add(formal.arg)
                    if isinstance(actual, ast.Name) and actual.id in (containers or set()):
                        bound_c.add(formal.arg)
                # *args formal — bound if any positional actual is Tensor/container.
                if lam.args.vararg is not None:
                    for actual in value.args[len(pos_formals):]:
                        if _expr_contains_tensor(actual, tensors, containers):
                            bound.add(lam.args.vararg.arg)
                            break
                # Keyword-only formals and **kwargs.
                kwonly_formals = list(lam.args.kwonlyargs)
                kw_defaults = list(lam.args.kw_defaults)
                for kw in value.keywords:
                    if kw.arg is None:
                        # **kwargs unpacking — if dict literal contains Tensor.
                        if _expr_contains_tensor(kw.value, tensors, containers):
                            if lam.args.kwarg is not None:
                                bound.add(lam.args.kwarg.arg)
                            for formal in pos_formals + kwonly_formals:
                                bound.add(formal.arg)
                        continue
                    # Match named kwarg to a formal.
                    for formal in pos_formals + kwonly_formals:
                        if formal.arg == kw.arg:
                            if _expr_contains_tensor(kw.value, tensors, containers):
                                bound.add(formal.arg)
                            if isinstance(kw.value, ast.Name) and kw.value.id in (containers or set()):
                                bound_c.add(formal.arg)
                            break
                # Apply kwonly defaults.
                for i, formal in enumerate(kwonly_formals):
                    if i < len(kw_defaults) and kw_defaults[i] is not None:
                        if _expr_contains_tensor(kw_defaults[i], tensors, containers):
                            bound.add(formal.arg)
                return _expr_contains_tensor(lam.body, bound, bound_c)
            return False
        if isinstance(value, ast.Attribute):
            suffix = "." + value.attr
            if suffix in _PROP_ALLOW_SUFFIX:
                return False
            if _expr_contains_tensor(value.value, tensors, containers):
                return True
        if isinstance(value, ast.Name) and value.id in tensors:
            return True
        if isinstance(value, ast.Subscript):
            base = value.value
            # Container subscript → Tensor element (round-10 fix for varargs).
            if isinstance(base, ast.Name) and containers and base.id in containers:
                return True
            return _expr_contains_tensor(value.value, tensors, containers)
        # codex round-9 R-P0-1: containers and IfExp propagate Tensor-ness.
        if isinstance(value, (ast.Tuple, ast.List, ast.Set)):
            return any(_expr_contains_tensor(e, tensors, containers) for e in value.elts)
        if isinstance(value, ast.Dict):
            return any(_expr_contains_tensor(v, tensors, containers) for v in value.values)
        if isinstance(value, ast.IfExp):
            return (
                _expr_contains_tensor(value.body, tensors, containers)
                or _expr_contains_tensor(value.orelse, tensors, containers)
            )
        if isinstance(value, ast.Lambda):
            return _expr_contains_tensor(value.body, tensors, containers)
        return False

    def _expr_contains_tensor(
        node: ast.AST | None,
        tensors: set[str],
        containers: set[str] | None = None,
    ) -> bool:
        """Return True iff this expression evaluates to a Tensor reference.

        Context-aware: `.shape` / `.dtype` / `.device` peel a Tensor down
        to non-Tensor scalars, so `q.shape[-1]` is NOT a Tensor expr.
        Method calls like `.reshape(-1)` keep the Tensor-ness.
        Round-10: subscript on a *container* (`inputs: dict`) unwraps to
        Tensor (assumes the contract's container holds Tensors).
        """
        if node is None:
            return False
        containers = containers or set()
        if isinstance(node, ast.Name):
            return node.id in tensors
        if isinstance(node, ast.Attribute):
            suffix = "." + node.attr
            if suffix in _PROP_ALLOW_SUFFIX:
                return False
            # codex round-13 NEW-P0 #1: `b.x` on container `b` yields Tensor.
            if isinstance(node.value, ast.Name) and node.value.id in containers:
                return True
            return _expr_contains_tensor(node.value, tensors, containers)
        if isinstance(node, ast.Subscript):
            # Container subscript → Tensor element.
            base = node.value
            if isinstance(base, ast.Name) and base.id in containers:
                return True
            return _expr_contains_tensor(node.value, tensors, containers)
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Attribute):
                suffix = "." + node.func.attr
                if suffix in _PROP_ALLOW_SUFFIX or suffix in _SCALAR_RETURNING_METHOD_SUFFIX:
                    return False
                if suffix in _TENSOR_RETURNING_METHOD_SUFFIX:
                    return _expr_contains_tensor(node.func.value, tensors, containers)
            # codex round-20 NEW-P0 #3: nested callable lambda
            # `(lambda f: f(x))(lambda z: z+z)` — when actual is a
            # Lambda passed to formal `f`, audit f-as-lambda when `f(x)`
            # is called in outer lambda body.
            if isinstance(node.func, ast.Lambda):
                outer_lam = node.func
                outer_formals = list(outer_lam.args.posonlyargs) + list(outer_lam.args.args)
                # Map formal name → bound lambda actual.
                lambda_bound: dict[str, ast.Lambda] = {}
                for i, formal in enumerate(outer_formals):
                    if i < len(node.args) and isinstance(node.args[i], ast.Lambda):
                        lambda_bound[formal.arg] = node.args[i]
                # If body contains `Call(Name(f), [arg])` where f is a
                # bound lambda actual, inline that lambda body.
                if lambda_bound:
                    for inner_call in ast.walk(outer_lam.body):
                        if (
                            isinstance(inner_call, ast.Call)
                            and isinstance(inner_call.func, ast.Name)
                            and inner_call.func.id in lambda_bound
                        ):
                            inlined = ast.Call(
                                func=lambda_bound[inner_call.func.id],
                                args=list(inner_call.args),
                                keywords=list(inner_call.keywords),
                            )
                            if _expr_contains_tensor(inlined, tensors, containers):
                                return True
            # Round-10/19: `(lambda z: z + z)(x)` — bind ALL formal
            # categories (positional, keyword, *args, defaults).
            if isinstance(node.func, ast.Lambda):
                lam = node.func
                bound = set(tensors)
                pos_formals = list(lam.args.posonlyargs) + list(lam.args.args)
                kwonly_formals = list(lam.args.kwonlyargs)
                defaults = list(lam.args.defaults)
                n_def = len(defaults)
                for i, formal in enumerate(pos_formals):
                    actual: ast.AST | None = None
                    if i < len(node.args):
                        actual = node.args[i]
                    else:
                        d_idx = i - (len(pos_formals) - n_def)
                        if 0 <= d_idx < n_def:
                            actual = defaults[d_idx]
                    if actual is not None and _expr_contains_tensor(actual, tensors, containers):
                        bound.add(formal.arg)
                for i, formal in enumerate(kwonly_formals):
                    if i < len(lam.args.kw_defaults) and lam.args.kw_defaults[i] is not None:
                        if _expr_contains_tensor(lam.args.kw_defaults[i], tensors, containers):
                            bound.add(formal.arg)
                for kw in node.keywords:
                    if kw.arg is None:
                        if _expr_contains_tensor(kw.value, tensors, containers):
                            for f in pos_formals + kwonly_formals:
                                bound.add(f.arg)
                        continue
                    for f in pos_formals + kwonly_formals:
                        if f.arg == kw.arg and _expr_contains_tensor(
                            kw.value, tensors, containers
                        ):
                            bound.add(f.arg)
                            break
                if lam.args.vararg is not None and any(
                    _expr_contains_tensor(a, tensors, containers)
                    for a in node.args[len(pos_formals):]
                ):
                    bound.add(lam.args.vararg.arg)
                return _expr_contains_tensor(lam.body, bound, containers)
            # Generic call: if any arg is a Tensor, propagate conservatively.
            for arg in node.args:
                if _expr_contains_tensor(arg, tensors, containers):
                    return True
            for kw in node.keywords:
                if _expr_contains_tensor(kw.value, tensors, containers):
                    return True
            return False
        if isinstance(node, ast.BinOp):
            return (
                _expr_contains_tensor(node.left, tensors, containers)
                or _expr_contains_tensor(node.right, tensors, containers)
            )
        if isinstance(node, ast.UnaryOp):
            return _expr_contains_tensor(node.operand, tensors, containers)
        if isinstance(node, (ast.Tuple, ast.List, ast.Set)):
            return any(_expr_contains_tensor(e, tensors, containers) for e in node.elts)
        if isinstance(node, ast.Dict):
            return any(_expr_contains_tensor(v, tensors, containers) for v in node.values)
        if isinstance(node, ast.Compare):
            return (
                _expr_contains_tensor(node.left, tensors, containers)
                or any(_expr_contains_tensor(c, tensors, containers) for c in node.comparators)
            )
        if isinstance(node, ast.IfExp):
            return (
                _expr_contains_tensor(node.body, tensors, containers)
                or _expr_contains_tensor(node.orelse, tensors, containers)
            )
        if isinstance(node, ast.Starred):
            return _expr_contains_tensor(node.value, tensors, containers)
        return False

    # Pass 0: collect names that *might* hold a "cpu"-bearing string at any
    # observed assignment / default argument. Conservative-union: once a
    # name has ever been bound to a cpu-bearing string (literal or via
    # static `str + str` folding), every later reference is treated as
    # cpu. Covers default args (`def f(x, d="cpu")`) and string
    # concatenation (`"c" + "pu"`) — codex round-9 R-P0-2.
    cpu_aliases: set[str] = set()
    # codex round-26: any-string name→value, supports fragment concat
    # like `c = "c"; pu = "pu"; device=c+pu`.
    str_aliases: dict[str, str] = {}
    # codex round-26: forward-declared so _eval_str_const can reference
    # them before the post-tree-walk population pass populates them.
    _global_callable_dotted: dict[str, str] = {}

    # Forward declarations; populated AFTER _eval_str_const is defined.
    function_returns: dict[str, str] = {}
    callable_aliases: dict[str, set[str]] = {}
    lambda_aliases: dict[str, list[ast.Lambda]] = {}

    def _const_value(n: ast.AST) -> object:
        """Return any Python literal value the expression folds to."""
        if isinstance(n, ast.Constant):
            return n.value
        # Try string-folding (handles chr() / "+" / Mod / etc.)
        s = _eval_str_const(n)
        if s is not None:
            return s
        return None

    def _eval_str_const(n: ast.AST) -> str | None:
        """Constant-fold a string-typed expression. Round-11/21 broadened."""
        if isinstance(n, ast.Constant) and isinstance(n.value, str):
            return n.value
        # codex round-21: known local cpu-bearing Name.
        if isinstance(n, ast.Name) and n.id in cpu_aliases:
            return "cpu"
        # codex round-26: any-string fragment Name (e.g. `c = "c"`).
        if isinstance(n, ast.Name) and n.id in str_aliases:
            return str_aliases[n.id]
        if isinstance(n, ast.BinOp):
            lhs = _eval_str_const(n.left)
            rhs = _eval_str_const(n.right)
            if isinstance(n.op, ast.Add):
                if lhs is not None and rhs is not None:
                    return lhs + rhs
            if isinstance(n.op, ast.Mult):
                for s, k in ((lhs, n.right), (rhs, n.left)):
                    if (
                        s is not None
                        and isinstance(k, ast.Constant)
                        and isinstance(k.value, int)
                        and k.value > 0
                    ):
                        return s * k.value
            # codex round-11/12: `"%s" % args` / `"%c%c%c" % (99,112,117)`
            # printf-style formatting with int or string args.
            if isinstance(n.op, ast.Mod) and isinstance(n.right, (ast.Tuple, ast.Constant)):
                fmt = _eval_str_const(n.left)
                if fmt is not None:
                    if isinstance(n.right, ast.Constant):
                        args_tup: tuple = (n.right.value,)
                    else:
                        args_tup = tuple(
                            _const_value(e) if isinstance(e, ast.expr) else None
                            for e in n.right.elts
                        )
                    if all(a is not None for a in args_tup):
                        try:
                            return fmt % args_tup
                        except (TypeError, ValueError):
                            return None
        if isinstance(n, ast.BoolOp):
            # codex round-28: `"" or "cpu"` / `cond and "cpu"` — any operand
            # folding to a cpu-bearing string wins.
            folded_any = None
            for v in n.values:
                f = _eval_str_const(v)
                if f is not None:
                    if "cpu" in f.lower():
                        return f
                    folded_any = f
            return folded_any
        if isinstance(n, ast.JoinedStr):
            joined = []
            for v in n.values:
                if isinstance(v, ast.Constant) and isinstance(v.value, str):
                    joined.append(v.value)
                else:
                    return None
            return "".join(joined)
        if isinstance(n, ast.Subscript):
            container = n.value
            idx = n.slice
            if isinstance(container, (ast.Tuple, ast.List)) and isinstance(idx, ast.Constant):
                if isinstance(idx.value, int) and 0 <= idx.value < len(container.elts):
                    return _eval_str_const(container.elts[idx.value])
            # codex round-11: `"xcpu"[1:]` — slice a literal string.
            base_str = _eval_str_const(container)
            if base_str is not None:
                if isinstance(idx, ast.Constant) and isinstance(idx.value, int):
                    try:
                        return base_str[idx.value]
                    except IndexError:
                        return None
                if isinstance(idx, ast.Slice):
                    def _const_int(slot: ast.AST | None) -> int | None:
                        if slot is None:
                            return None
                        if isinstance(slot, ast.Constant) and isinstance(slot.value, int):
                            return slot.value
                        return -1
                    lo = _const_int(idx.lower)
                    hi = _const_int(idx.upper)
                    st = _const_int(idx.step)
                    if -1 not in (
                        (lo if lo is not None else 0),
                        (hi if hi is not None else 0),
                        (st if st is not None else 1),
                    ):
                        return base_str[lo:hi:st]
        if isinstance(n, ast.IfExp):
            # Union of arms — if either branch is a known cpu string,
            # treat the expression as cpu-bearing.
            b = _eval_str_const(n.body)
            o = _eval_str_const(n.orelse)
            for s in (b, o):
                if s is not None and "cpu" in s.lower():
                    return s
            return b if b is not None else o
        if isinstance(n, ast.Call):
            fname = _canonical(n.func) or _dotted_name(n.func)
            # codex round-26: resolve local callable aliases (e.g. `s = str`).
            if not fname or fname == getattr(n.func, "id", ""):
                if isinstance(n.func, ast.Name) and n.func.id in _global_callable_dotted:
                    fname = _global_callable_dotted[n.func.id]
            # codex round-27: `(lambda: (lambda: "cpu"))()()` — recursively
            # unwrap nested IIFE: inner Call returns a Lambda; outer Call
            # of it gives Lambda.body.
            if isinstance(n.func, ast.Call):
                _cur: ast.AST = n.func
                while isinstance(_cur, ast.Call) and isinstance(_cur.func, ast.Lambda):
                    _cur = _cur.func.body
                if isinstance(_cur, ast.Lambda):
                    folded = _eval_str_const(_cur.body)
                    if folded is not None:
                        return folded
                # codex round-27: unwrapped result is a callable (e.g.
                # `torch.device`); synthesize an outer call and re-fold.
                elif not isinstance(_cur, ast.Call):
                    synth = ast.Call(func=_cur, args=list(n.args), keywords=list(n.keywords))
                    folded = _eval_str_const(synth)
                    if folded is not None:
                        return folded
            if fname == "str" and n.args:
                return _eval_str_const(n.args[0])
            # codex round-26: `torch.device("cpu")` returns CPU device.
            if fname == "torch.device" and n.args:
                inner = _eval_str_const(n.args[0])
                if inner is not None:
                    return inner
            # codex round-12: `format(99, "c")` returns "c"-encoded chr.
            if fname == "format" and len(n.args) >= 1:
                v = _const_value(n.args[0])
                spec = _eval_str_const(n.args[1]) if len(n.args) >= 2 else ""
                if v is not None and spec is not None:
                    try:
                        return format(v, spec)
                    except (ValueError, TypeError):
                        return None
            # codex round-12: `bytes((99,112,117)).decode()` / `str(bytes(...))`.
            if fname == "bytes" and n.args:
                arg0 = n.args[0]
                if isinstance(arg0, (ast.Tuple, ast.List)):
                    ints = []
                    for e in arg0.elts:
                        if isinstance(e, ast.Constant) and isinstance(e.value, int):
                            ints.append(e.value)
                        else:
                            ints = None
                            break
                    if ints is not None:
                        try:
                            return bytes(ints).decode("utf-8", errors="replace")
                        except Exception:                              # noqa: BLE001
                            return None
            # codex round-12: `chr(99)` — chr of literal int.
            if fname == "chr" and n.args:
                if isinstance(n.args[0], ast.Constant) and isinstance(n.args[0].value, int):
                    try:
                        return chr(n.args[0].value)
                    except (ValueError, OverflowError):
                        return None
            # codex round-18/19: bind Lambda formals (positional + kw +
            # *args + **kwargs + defaults) to actuals before evaluating
            # body string. Also handles `(lambda *a: a[0])("cpu")` and
            # `(lambda d: d)(**{"d": "cpu"})`.
            if isinstance(n.func, ast.Lambda):
                lam = n.func
                pos_formals = list(lam.args.posonlyargs) + list(lam.args.args)
                kwonly_formals = list(lam.args.kwonlyargs)
                bindings: dict[str, str] = {}
                defaults = list(lam.args.defaults)
                n_def = len(defaults)
                for i, formal in enumerate(pos_formals):
                    actual: ast.AST | None = None
                    if i < len(n.args):
                        actual = n.args[i]
                    else:
                        d_idx = i - (len(pos_formals) - n_def)
                        if 0 <= d_idx < n_def:
                            actual = defaults[d_idx]
                    if actual is None:
                        continue
                    folded = _eval_str_const(actual)
                    if folded is not None:
                        bindings[formal.arg] = folded
                # kwonly defaults
                for i, formal in enumerate(kwonly_formals):
                    if i < len(lam.args.kw_defaults) and lam.args.kw_defaults[i] is not None:
                        folded = _eval_str_const(lam.args.kw_defaults[i])
                        if folded is not None:
                            bindings[formal.arg] = folded
                # Named kwargs.
                for kw in n.keywords:
                    if kw.arg is not None:
                        folded = _eval_str_const(kw.value)
                        if folded is not None:
                            bindings[kw.arg] = folded
                    else:
                        # **{"d": "cpu"} — Dict literal unpacking.
                        if isinstance(kw.value, ast.Dict):
                            for k_node, v_node in zip(kw.value.keys, kw.value.values):
                                if isinstance(k_node, ast.Constant) and isinstance(k_node.value, str):
                                    folded = _eval_str_const(v_node)
                                    if folded is not None:
                                        bindings[k_node.value] = folded
                # *args[i] = position i — body `a[0]` resolves to first positional.
                if lam.args.vararg is not None and lam.args.vararg.arg:
                    vararg_name = lam.args.vararg.arg
                    folded_vararg = [_eval_str_const(a) for a in n.args[len(pos_formals):]]

                    def _fold_body(body: ast.AST) -> str | None:
                        if isinstance(body, ast.Name) and body.id in bindings:
                            return bindings[body.id]
                        if (
                            isinstance(body, ast.Subscript)
                            and isinstance(body.value, ast.Name)
                            and body.value.id == vararg_name
                            and isinstance(body.slice, ast.Constant)
                            and isinstance(body.slice.value, int)
                            and 0 <= body.slice.value < len(folded_vararg)
                        ):
                            return folded_vararg[body.slice.value]
                        return _eval_str_const(body)
                    return _fold_body(lam.body)

                def _fold_body(body: ast.AST) -> str | None:
                    if isinstance(body, ast.Name) and body.id in bindings:
                        return bindings[body.id]
                    return _eval_str_const(body)
                return _fold_body(lam.body)
            # codex round-20: call to lambda alias `dev = lambda d="cpu": d; dev()`.
            if isinstance(n.func, ast.Name) and n.func.id in lambda_aliases:
                for lam in lambda_aliases[n.func.id]:
                    synthetic_call = ast.Call(
                        func=lam, args=list(n.args), keywords=list(n.keywords),
                    )
                    folded = _eval_str_const(synthetic_call)
                    if folded is not None:
                        return folded
            # codex round-11/15/19: call to a local function whose body
            # folds to a constant string. Also bind actual call args to
            # formals so `def dev(d): return d; dev("cpu")` folds.
            if isinstance(n.func, ast.Name):
                target = n.func.id
                visited: set[str] = set()
                while target in callable_aliases and target not in visited:
                    visited.add(target)
                    candidates = callable_aliases.get(target, set())
                    if not candidates:
                        break
                    cpu_match = None
                    for c in candidates:
                        if c in function_returns and "cpu" in function_returns[c].lower():
                            cpu_match = c
                            break
                    target = cpu_match if cpu_match else next(iter(candidates))
                # codex round-29: try caller-actual-binding BEFORE
                # falling back to the cached default-bound return; otherwise
                # `def dev(d="npu"): return d; dev("cpu")` would yield "npu".
                _cached = function_returns.get(target)
                # Round-19: fold the callee body with caller's actual args.
                if target in module_level_fns:
                    fnd = module_level_fns[target]
                    bindings: dict[str, str] = {}
                    pos_formals = list(fnd.args.posonlyargs) + list(fnd.args.args)
                    defaults = list(fnd.args.defaults)
                    n_def = len(defaults)
                    for i, formal in enumerate(pos_formals):
                        actual = None
                        if i < len(n.args):
                            actual = n.args[i]
                        else:
                            d_idx = i - (len(pos_formals) - n_def)
                            if 0 <= d_idx < n_def:
                                actual = defaults[d_idx]
                        if actual is not None:
                            folded = _eval_str_const(actual)
                            if folded is not None:
                                bindings[formal.arg] = folded
                    for i, formal in enumerate(fnd.args.kwonlyargs):
                        if i < len(fnd.args.kw_defaults) and fnd.args.kw_defaults[i] is not None:
                            folded = _eval_str_const(fnd.args.kw_defaults[i])
                            if folded is not None:
                                bindings[formal.arg] = folded
                    # codex round-20: *args / **kwargs unpacking.
                    vararg_folded: list[str | None] = []
                    if fnd.args.vararg is not None:
                        vararg_folded = [
                            _eval_str_const(a)
                            for a in n.args[len(pos_formals):]
                        ]
                    for kw in n.keywords:
                        if kw.arg is not None:
                            folded = _eval_str_const(kw.value)
                            if folded is not None:
                                bindings[kw.arg] = folded
                        else:
                            # `dev(**{"d": "cpu"})`
                            if isinstance(kw.value, ast.Dict):
                                for k_node, v_node in zip(kw.value.keys, kw.value.values):
                                    if isinstance(k_node, ast.Constant) and isinstance(k_node.value, str):
                                        f2 = _eval_str_const(v_node)
                                        if f2 is not None:
                                            bindings[k_node.value] = f2
                    vararg_name = fnd.args.vararg.arg if fnd.args.vararg else None

                    def _fold_in_callee_body(body: ast.AST) -> str | None:
                        if isinstance(body, ast.Name) and body.id in bindings:
                            return bindings[body.id]
                        if (
                            vararg_name is not None
                            and isinstance(body, ast.Subscript)
                            and isinstance(body.value, ast.Name)
                            and body.value.id == vararg_name
                            and isinstance(body.slice, ast.Constant)
                            and isinstance(body.slice.value, int)
                            and 0 <= body.slice.value < len(vararg_folded)
                        ):
                            return vararg_folded[body.slice.value]
                        # codex round-21/22: recursively substitute Name
                        # in BinOp (Add/Mod/Mult).
                        if isinstance(body, ast.BinOp):
                            l = _fold_in_callee_body(body.left)
                            r = _fold_in_callee_body(body.right)
                            if isinstance(body.op, ast.Add) and l is not None and r is not None:
                                return l + r
                            if isinstance(body.op, ast.Mult):
                                if l is not None and isinstance(body.right, ast.Constant) and isinstance(body.right.value, int) and body.right.value > 0:
                                    return l * body.right.value
                                if r is not None and isinstance(body.left, ast.Constant) and isinstance(body.left.value, int) and body.left.value > 0:
                                    return r * body.left.value
                            if isinstance(body.op, ast.Mod) and l is not None:
                                # "%s" % d → substitute d via bindings
                                if isinstance(body.right, ast.Tuple):
                                    args = tuple(_fold_in_callee_body(e) for e in body.right.elts)
                                else:
                                    args = (_fold_in_callee_body(body.right),)
                                if all(a is not None for a in args):
                                    try:
                                        return l % args
                                    except (TypeError, ValueError):
                                        return None
                        if isinstance(body, ast.BinOp) and isinstance(body.op, ast.Add):
                            l = _fold_in_callee_body(body.left)
                            r = _fold_in_callee_body(body.right)
                            if l is not None and r is not None:
                                return l + r
                        return _eval_str_const(body)

                    # Walk body returns and substitute. Prefer cpu-bearing
                    # branch if multiple Returns exist (codex round-29).
                    _any_found = None
                    for inner in ast.walk(fnd):
                        if isinstance(inner, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)) and inner is not fnd:
                            continue
                        if isinstance(inner, ast.Return) and inner.value is not None:
                            folded = _fold_in_callee_body(inner.value)
                            if folded is None:
                                continue
                            if "cpu" in folded.lower():
                                return folded
                            if _any_found is None:
                                _any_found = folded
                    if _any_found is not None:
                        return _any_found
                # codex round-29: caller-actual-binding didn't fold; fall
                # back to cached default-bound return (if any).
                if _cached is not None:
                    return _cached
            if isinstance(n.func, ast.Lambda):
                return _eval_str_const(n.func.body)
            # Calls to known string methods on string constants.
            if isinstance(n.func, ast.Attribute):
                base = _eval_str_const(n.func.value)
                if base is not None:
                    method = n.func.attr
                    try:
                        if method in {"upper", "lower", "strip", "lstrip", "rstrip"}:
                            return getattr(base, method)()
                        if method == "replace" and len(n.args) == 2:
                            a = _eval_str_const(n.args[0])
                            b = _eval_str_const(n.args[1])
                            if a is not None and b is not None:
                                return base.replace(a, b)
                        if method == "join":
                            if n.args and isinstance(n.args[0], (ast.Tuple, ast.List)):
                                parts = [
                                    _eval_str_const(e) for e in n.args[0].elts
                                ]
                                if all(p is not None for p in parts):
                                    return base.join(parts)  # type: ignore[arg-type]
                    except (TypeError, ValueError):
                        return None
        return None

    # Populate function_returns now that _eval_str_const is defined.
    # codex round-13/14: module-level scope only (class methods are
    # scoped to instances and must NOT shadow module functions).
    # Track via fixpoint so cross-fn returns (`def a: return b()`)
    # resolve to the leaf constant.
    module_level_fns: dict[str, ast.FunctionDef | ast.AsyncFunctionDef] = {}
    # Walk only the tree.body — skip nested in ClassDef or other fn bodies.
    for sub in tree.body if hasattr(tree, "body") else []:
        if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)):
            module_level_fns[sub.name] = sub
    # Module-level lambda aliases will be folded later (after
    # `lambda_aliases` is collected) via the same fixpoint loop.

    def _fold_fn_return(fn_node) -> str | None:
        """codex round-15/18: fold ALL top-level Returns. Also bind
        default arg values into the body string folding, so
        `def dev(d="cpu"): return d` resolves to "cpu"."""
        cpu_found = None
        any_found = None

        def _walk_skip_nested(node):
            for child in ast.iter_child_nodes(node):
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
                    continue
                if isinstance(child, ast.Return) and child.value is not None:
                    yield child
                yield from _walk_skip_nested(child)

        # codex round-18: build a name → string-default map for the fn.
        default_bindings: dict[str, str] = {}
        pos_formals = list(fn_node.args.posonlyargs) + list(fn_node.args.args)
        defaults = list(fn_node.args.defaults)
        n_def = len(defaults)
        for i, default in enumerate(defaults):
            idx = len(pos_formals) - n_def + i
            if 0 <= idx < len(pos_formals):
                folded = _eval_str_const(default)
                if folded is not None:
                    default_bindings[pos_formals[idx].arg] = folded

        def _fold_with_bindings(node: ast.AST) -> str | None:
            if isinstance(node, ast.Name) and node.id in default_bindings:
                return default_bindings[node.id]
            return _eval_str_const(node)

        for inner in _walk_skip_nested(fn_node):
            folded = _fold_with_bindings(inner.value)
            if folded is None:
                continue
            if any_found is None:
                any_found = folded
            if "cpu" in folded.lower():
                cpu_found = folded
                break
        return cpu_found if cpu_found is not None else any_found

    # Fixpoint iteration: function_returns may grow as nested calls
    # resolve.
    for _ in range(max(8, len(module_level_fns) * 2)):
        grew = False
        for name, fnd in module_level_fns.items():
            folded = _fold_fn_return(fnd)
            if folded is not None:
                if name not in function_returns or (
                    "cpu" in folded.lower()
                    and "cpu" not in function_returns[name].lower()
                ):
                    function_returns[name] = folded
                    grew = True
        if not grew:
            break

    def _mark_if_cpu(name: str, val: str | None) -> None:
        if val is not None and "cpu" in val.lower():
            cpu_aliases.add(name)
        if val is not None:
            str_aliases[name] = val

    for sub in ast.walk(tree):
        if isinstance(sub, ast.Assign):
            v = _eval_str_const(sub.value)
            for tgt in sub.targets:
                if isinstance(tgt, ast.Name):
                    _mark_if_cpu(tgt.id, v)
        elif isinstance(sub, ast.AnnAssign) and sub.value is not None:
            v = _eval_str_const(sub.value)
            if isinstance(sub.target, ast.Name):
                _mark_if_cpu(sub.target.id, v)
        elif isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)):
            args = sub.args
            pos_args = list(args.posonlyargs) + list(args.args)
            n_pos = len(pos_args)
            n_defaults = len(args.defaults)
            for i, default in enumerate(args.defaults):
                v = _eval_str_const(default)
                idx = n_pos - n_defaults + i
                if 0 <= idx < n_pos:
                    _mark_if_cpu(pos_args[idx].arg, v)
            for i, default in enumerate(args.kw_defaults):
                if default is None:
                    continue
                v = _eval_str_const(default)
                if i < len(args.kwonlyargs):
                    _mark_if_cpu(args.kwonlyargs[i].arg, v)

    # Back-compat: `string_constants` dict kept as fallback for direct lookup.
    string_constants: dict[str, str] = {n: "cpu" for n in cpu_aliases}

    kernel_fn_ids: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for deco in node.decorator_list:
                if _dotted_name(deco) in {"triton.jit", "triton.autotune"} or (
                    isinstance(deco, ast.Call)
                    and _dotted_name(deco.func) in {"triton.jit", "triton.autotune"}
                ):
                    kernel_fn_ids.add(id(node))

    # codex round-26/27: name → dotted alias map for `mkdev = torch.device`
    # / `s = str` / `f = torch.empty` / chained `a=torch; dev=a.device`.
    # Used by _value_is_cpu / _eval_str_const to resolve Call.func canonical.
    _DOTTED_OF_INTEREST = {"str", "format", "bytes", "chr", "torch.device"}

    def _resolve_dotted_with_aliases(node: ast.AST) -> str:
        """Resolve a dotted name using both `aliases` (import) and
        `_global_callable_dotted` (local rebinding)."""
        if isinstance(node, ast.Name):
            return (
                _global_callable_dotted.get(node.id)
                or aliases.get(node.id, node.id)
            )
        if isinstance(node, ast.Attribute):
            base = _resolve_dotted_with_aliases(node.value)
            return f"{base}.{node.attr}" if base else node.attr
        # codex round-29: `(torch,)[0]` / `[torch][0]` — subscript on
        # literal tuple/list with constant index.
        if isinstance(node, ast.Subscript):
            container = node.value
            if isinstance(container, (ast.Tuple, ast.List)) and isinstance(node.slice, ast.Constant):
                if isinstance(node.slice.value, int):
                    idx = node.slice.value
                    if -len(container.elts) <= idx < len(container.elts):
                        return _resolve_dotted_with_aliases(container.elts[idx])
        return ""

    _assign_count = sum(
        1 for _n in ast.walk(tree) if isinstance(_n, (ast.Assign, ast.AnnAssign))
    )
    for _ in range(max(8, _assign_count)):
        prev = dict(_global_callable_dotted)
        for _s in ast.walk(tree):
            targets, value = (
                (_s.targets, _s.value)
                if isinstance(_s, ast.Assign)
                else (([_s.target], _s.value)
                      if isinstance(_s, ast.AnnAssign) and _s.value is not None
                      else ([], None))
            )
            if value is None:
                continue
            rhs_dotted = _resolve_dotted_with_aliases(value)
            if rhs_dotted and (
                rhs_dotted in _DOTTED_OF_INTEREST
                or rhs_dotted == "torch"
                or rhs_dotted.startswith("torch.")
            ):
                for tgt in targets:
                    if isinstance(tgt, ast.Name) and tgt.id != rhs_dotted:
                        _global_callable_dotted[tgt.id] = rhs_dotted
            # codex round-28: tuple/list destructure `(a,) = (torch,)`.
            if isinstance(value, (ast.Tuple, ast.List)):
                for tgt in targets:
                    if isinstance(tgt, (ast.Tuple, ast.List)) and len(tgt.elts) == len(value.elts):
                        for lhs_elt, rhs_elt in zip(tgt.elts, value.elts):
                            if not isinstance(lhs_elt, ast.Name):
                                continue
                            r_dotted = _resolve_dotted_with_aliases(rhs_elt)
                            if r_dotted and (
                                r_dotted in _DOTTED_OF_INTEREST
                                or r_dotted == "torch"
                                or r_dotted.startswith("torch.")
                            ):
                                _global_callable_dotted[lhs_elt.id] = r_dotted
        if _global_callable_dotted == prev:
            break

    # codex round-26: name → Tuple/List literal map (any scope).
    _global_tuple_literals: dict[str, ast.Tuple | ast.List] = {}
    for s in ast.walk(tree):
        if isinstance(s, ast.Assign) and isinstance(s.value, (ast.Tuple, ast.List)):
            for tgt in s.targets:
                if isinstance(tgt, ast.Name):
                    _global_tuple_literals[tgt.id] = s.value
        elif isinstance(s, ast.AnnAssign) and isinstance(s.value, (ast.Tuple, ast.List)) and isinstance(s.target, ast.Name):
            _global_tuple_literals[s.target.id] = s.value

    # codex round-25/26: name → Dict literal map (any scope) for **kw folding.
    # Tracks: direct dict literal Assign/AnnAssign, `dict(...)` Call assigns,
    # AugAssign `|=` merges, and Subscript-style mutations `kw["k"] = v`.
    _global_dict_literals: dict[str, ast.Dict] = {}

    def _ensure_dict(name: str) -> ast.Dict:
        d = _global_dict_literals.get(name)
        if d is None:
            d = ast.Dict(keys=[], values=[])
            _global_dict_literals[name] = d
        return d

    def _merge_into(name: str, other: ast.Dict) -> None:
        d = _ensure_dict(name)
        for k_node, v_node in zip(other.keys, other.values):
            replaced = False
            for i, existing in enumerate(d.keys):
                if (
                    existing is not None and k_node is not None
                    and isinstance(existing, ast.Constant) and isinstance(k_node, ast.Constant)
                    and existing.value == k_node.value
                ):
                    d.values[i] = v_node
                    replaced = True
                    break
            if not replaced:
                d.keys.append(k_node)
                d.values.append(v_node)

    def _resolve_to_dict_literal(expr: ast.AST) -> ast.Dict | None:
        """Best-effort: resolve expr to a Dict literal (None if unknown)."""
        if isinstance(expr, ast.Dict):
            # codex round-29: `{**{...}}` — keys may include None (the `**`).
            if any(k is None for k in expr.keys):
                synth_keys: list[ast.AST] = []
                synth_values: list[ast.AST] = []
                for k_node, v_node in zip(expr.keys, expr.values):
                    if k_node is None:
                        inner = _resolve_to_dict_literal(v_node)
                        if inner is not None:
                            synth_keys.extend(inner.keys)
                            synth_values.extend(inner.values)
                    else:
                        synth_keys.append(k_node)
                        synth_values.append(v_node)
                return ast.Dict(keys=synth_keys, values=synth_values)
            return expr
        if isinstance(expr, ast.Name):
            return _global_dict_literals.get(expr.id)
        # codex round-29: `{} | {"device": "cpu"}` BitOr.
        if isinstance(expr, ast.BinOp) and isinstance(expr.op, ast.BitOr):
            left_d = _resolve_to_dict_literal(expr.left)
            right_d = _resolve_to_dict_literal(expr.right)
            if left_d is not None or right_d is not None:
                merged_keys = list((left_d.keys if left_d else []) + (right_d.keys if right_d else []))
                merged_values = list((left_d.values if left_d else []) + (right_d.values if right_d else []))
                return ast.Dict(keys=merged_keys, values=merged_values)
        if isinstance(expr, ast.Call):
            # `dict(device="cpu")` or `dict(**{"device":"cpu"})` or
            # `dict({"device":"cpu"})` (positional Dict arg).
            fn = _canonical(expr.func) or _dotted_name(expr.func)
            if fn == "dict":
                # codex round-29: positional Dict arg.
                keys: list[ast.AST] = []
                values: list[ast.AST] = []
                for a in expr.args:
                    inner = _resolve_to_dict_literal(a)
                    if inner is not None:
                        keys.extend(inner.keys)
                        values.extend(inner.values)
                for kw in expr.keywords:
                    if kw.arg is not None:
                        keys.append(ast.Constant(value=kw.arg))
                        values.append(kw.value)
                    else:
                        inner = _resolve_to_dict_literal(kw.value)
                        if inner is not None:
                            keys.extend(inner.keys)
                            values.extend(inner.values)
                return ast.Dict(keys=keys, values=values)
        if isinstance(expr, ast.Subscript):
            # codex round-27: handle Name bound to tuple/list of dicts.
            if isinstance(expr.value, ast.Name) and isinstance(expr.slice, ast.Constant):
                tup = _global_tuple_literals.get(expr.value.id)
                if tup is not None and isinstance(expr.slice.value, int):
                    idx = expr.slice.value
                    if -len(tup.elts) <= idx < len(tup.elts):
                        return _resolve_to_dict_literal(tup.elts[idx])
            container = _resolve_to_dict_literal(expr.value)
            if container is None and isinstance(expr.value, ast.Name):
                container = _global_dict_literals.get(expr.value.id)
            if container is None:
                return None
            idx_n = expr.slice
            if isinstance(idx_n, ast.Constant):
                for k_node, v_node in zip(container.keys, container.values):
                    if isinstance(k_node, ast.Constant) and k_node.value == idx_n.value:
                        return _resolve_to_dict_literal(v_node)
        return None

    for s in ast.walk(tree):
        # Direct Dict assignment.
        if isinstance(s, ast.Assign) and isinstance(s.value, ast.Dict):
            for tgt in s.targets:
                if isinstance(tgt, ast.Name):
                    _global_dict_literals[tgt.id] = s.value
        elif isinstance(s, ast.AnnAssign) and isinstance(s.value, ast.Dict) and isinstance(s.target, ast.Name):
            _global_dict_literals[s.target.id] = s.value
        # codex round-28: Name→Name dict alias (`kw2 = kw1`).
        elif isinstance(s, ast.Assign) and isinstance(s.value, ast.Name) and s.value.id in _global_dict_literals:
            for tgt in s.targets:
                if isinstance(tgt, ast.Name):
                    _global_dict_literals[tgt.id] = _global_dict_literals[s.value.id]
        elif isinstance(s, ast.AnnAssign) and isinstance(s.value, ast.Name) and isinstance(s.target, ast.Name) and s.value.id in _global_dict_literals:
            _global_dict_literals[s.target.id] = _global_dict_literals[s.value.id]
        # `kw = dict(device="cpu")` / `dict(**{...})` / `dict({...})` Call.
        elif isinstance(s, ast.Assign) and isinstance(s.value, ast.Call):
            synth = _resolve_to_dict_literal(s.value)
            if synth is not None:
                for tgt in s.targets:
                    if isinstance(tgt, ast.Name):
                        _global_dict_literals[tgt.id] = synth
        # codex round-29: `kw = {} | {"k": v}` BitOr.
        elif isinstance(s, ast.Assign) and isinstance(s.value, ast.BinOp) and isinstance(s.value.op, ast.BitOr):
            synth = _resolve_to_dict_literal(s.value)
            if synth is not None:
                for tgt in s.targets:
                    if isinstance(tgt, ast.Name):
                        _global_dict_literals[tgt.id] = synth
        # `kw["k"] = v` subscript-assign mutation.
        elif isinstance(s, ast.Assign) and len(s.targets) == 1:
            tgt = s.targets[0]
            if isinstance(tgt, ast.Subscript) and isinstance(tgt.value, ast.Name) and isinstance(tgt.slice, ast.Constant):
                d = _ensure_dict(tgt.value.id)
                d.keys.append(ast.Constant(value=tgt.slice.value))
                d.values.append(s.value)
        # `kw |= {"k": v}` AugAssign merge.
        elif isinstance(s, ast.AugAssign) and isinstance(s.op, ast.BitOr) and isinstance(s.target, ast.Name):
            other = s.value if isinstance(s.value, ast.Dict) else _global_dict_literals.get(
                s.value.id if isinstance(s.value, ast.Name) else "", None
            )
            if other is not None:
                _merge_into(s.target.id, other)

    # codex round-24/25 NEW-P0 #1: cpu_transfer_aliases / custom_method_aliases
    # collected at module level (aggregate across all functions).
    _global_cpu_transfer_aliases: set[str] = set()
    _global_custom_method_aliases: set[str] = set()

    _CPU_TRANSFER_ATTR_PREFIXES = (
        "torch.empty", "torch.zeros", "torch.ones", "torch.full",
        "torch.tensor", "torch.as_tensor", "torch.arange",
    )

    # codex round-26: track module-level fn → set of possible return exprs.
    _global_fn_returns: dict[str, list[ast.AST]] = {}
    # codex round-27: track fn → set of formals that flow to a CPU-sensitive
    # position (device kwarg / .to() positional / **kw unpack).
    _cpu_tainted_formals: dict[str, set[str]] = {}
    # codex round-28: same, but indexed by class method name (any class).
    _cpu_tainted_method_formals: dict[str, set[str]] = {}
    _method_pos_formals: dict[str, list[str]] = {}
    # codex round-27: track fn → ordered list of positional formal names.
    _fn_pos_formals: dict[str, list[str]] = {}
    # codex round-27: track fn → set of formal names overall.
    _fn_all_formals: dict[str, set[str]] = {}

    def _walk_fn_local(fn_node):
        """Yield nodes inside `fn_node` but NOT inside nested function/class
        defs (codex round-28 P1: avoid over-tainting outer formals)."""
        for child in ast.iter_child_nodes(fn_node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef)):
                continue
            yield child
            for sub in ast.walk(child):
                if sub is child:
                    continue
                yield sub

    def _scan_fn_formal_taint(fn_node) -> tuple[set[str], list[str], set[str]]:
        pos = [a.arg for a in (list(fn_node.args.posonlyargs) + list(fn_node.args.args))]
        all_f = set(pos) | {a.arg for a in fn_node.args.kwonlyargs}
        if fn_node.args.vararg:
            all_f.add(fn_node.args.vararg.arg)
        if fn_node.args.kwarg:
            all_f.add(fn_node.args.kwarg.arg)
        sensitive: set[str] = set()
        for sub in _walk_fn_local(fn_node):
            if not isinstance(sub, ast.Call):
                continue
            for kw in sub.keywords:
                if kw.arg == "device" and isinstance(kw.value, ast.Name) and kw.value.id in all_f:
                    sensitive.add(kw.value.id)
                if kw.arg is None and isinstance(kw.value, ast.Name) and kw.value.id in all_f:
                    sensitive.add(kw.value.id)
            if isinstance(sub.func, ast.Attribute) and sub.func.attr == "to":
                for a in sub.args:
                    if isinstance(a, ast.Name) and a.id in all_f:
                        sensitive.add(a.id)
                for kw in sub.keywords:
                    if isinstance(kw.value, ast.Name) and kw.value.id in all_f:
                        sensitive.add(kw.value.id)
            for a in sub.args:
                if isinstance(a, ast.Starred) and isinstance(a.value, ast.Name) and a.value.id in all_f:
                    sensitive.add(a.value.id)
        return sensitive, pos, all_f

    # First pass: scan module-level fns and methods.
    for _fn in ast.walk(tree):
        if isinstance(_fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
            rets: list[ast.AST] = []
            for r in ast.walk(_fn):
                if isinstance(r, ast.Return) and r.value is not None:
                    rets.append(r.value)
            _global_fn_returns[_fn.name] = rets
            sensitive, pos, all_f = _scan_fn_formal_taint(_fn)
            _fn_pos_formals[_fn.name] = pos
            _fn_all_formals[_fn.name] = all_f
            if sensitive:
                _cpu_tainted_formals[_fn.name] = sensitive

    # codex round-28: class method formal taint (skip self/cls).
    for _cls in ast.walk(tree):
        if not isinstance(_cls, ast.ClassDef):
            continue
        for _m in _cls.body:
            if not isinstance(_m, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            sensitive_m, pos_m, _ = _scan_fn_formal_taint(_m)
            # First positional is `self` / `cls` — drop from pos list for
            # caller-supplied actuals.
            caller_pos = pos_m[1:] if pos_m else []
            if sensitive_m:
                _cpu_tainted_method_formals[_m.name] = (
                    _cpu_tainted_method_formals.get(_m.name, set()) | sensitive_m
                )
                _method_pos_formals[_m.name] = caller_pos

    # codex round-29: helper fn aliases (`f = B`) for tainted-fn name
    # resolution at call sites.
    _global_fn_aliases: dict[str, str] = {}

    def _resolve_fn_alias(name: str) -> str:
        seen = set()
        cur = name
        while cur in _global_fn_aliases and cur not in seen:
            seen.add(cur)
            cur = _global_fn_aliases[cur]
        return cur

    def _is_tainted_fn(name: str) -> bool:
        return _resolve_fn_alias(name) in _cpu_tainted_formals

    # codex round-29: collect callable-Name → fn-name aliases.
    for s in ast.walk(tree):
        if isinstance(s, ast.Assign) and isinstance(s.value, ast.Name):
            if s.value.id in _global_fn_returns or s.value.id in _cpu_tainted_formals:
                for tgt in s.targets:
                    if isinstance(tgt, ast.Name):
                        _global_fn_aliases[tgt.id] = s.value.id
        # `f = (B,)[0]` Subscript on tuple/list literal.
        if isinstance(s, ast.Assign) and isinstance(s.value, ast.Subscript):
            container = s.value.value
            if isinstance(container, (ast.Tuple, ast.List)) and isinstance(s.value.slice, ast.Constant):
                idx = s.value.slice.value
                if isinstance(idx, int) and 0 <= idx < len(container.elts):
                    elt = container.elts[idx]
                    if isinstance(elt, ast.Name) and (
                        elt.id in _global_fn_returns or elt.id in _cpu_tainted_formals
                    ):
                        for tgt in s.targets:
                            if isinstance(tgt, ast.Name):
                                _global_fn_aliases[tgt.id] = elt.id

    # codex round-28/29: transitive taint — true fixpoint (no hard cap).
    # If fn A calls fn B with formal `d` at B's tainted slot, A's formal
    # `d` becomes tainted too. Also propagates to class methods.
    def _propagate_one_fn(fn_node, current_set: set[str], result_dict: dict[str, set[str]], pos_formals_dict: dict[str, list[str]]) -> set[str]:
        all_f = _fn_all_formals.get(fn_node.name, set())
        new = set(current_set)
        for sub in _walk_fn_local(fn_node):
            if not isinstance(sub, ast.Call):
                continue
            # Direct fn call (possibly aliased).
            callee = None
            if isinstance(sub.func, ast.Name):
                callee = _resolve_fn_alias(sub.func.id)
            if callee and callee in _cpu_tainted_formals:
                callee_pos = _fn_pos_formals.get(callee, [])
                callee_tainted = _cpu_tainted_formals[callee]
                for i, actual in enumerate(sub.args):
                    if i < len(callee_pos) and callee_pos[i] in callee_tainted:
                        if isinstance(actual, ast.Name) and actual.id in all_f:
                            new.add(actual.id)
                for kw in sub.keywords:
                    if kw.arg in callee_tainted and isinstance(kw.value, ast.Name) and kw.value.id in all_f:
                        new.add(kw.value.id)
            # Method call.
            if isinstance(sub.func, ast.Attribute) and sub.func.attr in _cpu_tainted_method_formals:
                callee_pos = _method_pos_formals.get(sub.func.attr, [])
                callee_tainted = _cpu_tainted_method_formals[sub.func.attr]
                for i, actual in enumerate(sub.args):
                    if i < len(callee_pos) and callee_pos[i] in callee_tainted:
                        if isinstance(actual, ast.Name) and actual.id in all_f:
                            new.add(actual.id)
                for kw in sub.keywords:
                    if kw.arg in callee_tainted and isinstance(kw.value, ast.Name) and kw.value.id in all_f:
                        new.add(kw.value.id)
        return new

    # True fixpoint (no cap — `if not grew: break` always terminates).
    grew = True
    _iter = 0
    while grew and _iter < 100:
        grew = False
        _iter += 1
        for _fn in ast.walk(tree):
            if isinstance(_fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
                current = _cpu_tainted_formals.get(_fn.name, set())
                new = _propagate_one_fn(_fn, current, _cpu_tainted_formals, _fn_pos_formals)
                if new != current:
                    _cpu_tainted_formals[_fn.name] = new
                    grew = True
            if isinstance(_fn, ast.ClassDef):
                for _m in _fn.body:
                    if not isinstance(_m, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        continue
                    current_m = _cpu_tainted_method_formals.get(_m.name, set())
                    new_m = _propagate_one_fn(_m, current_m, _cpu_tainted_method_formals, _method_pos_formals)
                    if new_m != current_m:
                        _cpu_tainted_method_formals[_m.name] = new_m
                        grew = True

    def _resolve_subscript_container(rhs: ast.Subscript) -> ast.AST | None:
        """Resolve `name["k"]` / `name[0]` to the element node if container literal known."""
        container = rhs.value
        if isinstance(container, ast.Name):
            d = _global_dict_literals.get(container.id)
            if d is not None:
                container = d
            else:
                t = _global_tuple_literals.get(container.id)
                if t is not None:
                    container = t
        if isinstance(container, ast.Dict) and isinstance(rhs.slice, ast.Constant):
            for k_node, v_node in zip(container.keys, container.values):
                if isinstance(k_node, ast.Constant) and k_node.value == rhs.slice.value:
                    return v_node
        if isinstance(container, (ast.Tuple, ast.List)) and isinstance(rhs.slice, ast.Constant):
            if isinstance(rhs.slice.value, int):
                idx = rhs.slice.value
                if -len(container.elts) <= idx < len(container.elts):
                    return container.elts[idx]
        return None

    def _rhs_is_cpu_transfer(rhs: ast.AST, seen: set[int] | None = None) -> bool:
        """Module-level: does this RHS produce a CPU-transfer callable?"""
        if seen is None:
            seen = set()
        if id(rhs) in seen:
            return False
        seen.add(id(rhs))
        if isinstance(rhs, ast.Attribute):
            if rhs.attr == "to" and isinstance(rhs.value, (ast.Name, ast.Subscript, ast.Attribute)):
                return True
            r_dotted = _canonical(rhs) or _dotted_name(rhs)
            return r_dotted.startswith(_CPU_TRANSFER_ATTR_PREFIXES)
        if isinstance(rhs, ast.Name):
            return rhs.id in _global_cpu_transfer_aliases
        if isinstance(rhs, ast.IfExp):
            return _rhs_is_cpu_transfer(rhs.body, seen) or _rhs_is_cpu_transfer(rhs.orelse, seen)
        if isinstance(rhs, ast.BoolOp):
            return any(_rhs_is_cpu_transfer(v, seen) for v in rhs.values)
        if isinstance(rhs, ast.Subscript):
            container = rhs.value
            if isinstance(container, (ast.Tuple, ast.List)):
                return any(_rhs_is_cpu_transfer(e, seen) for e in container.elts)
            if isinstance(container, ast.Dict):
                return any(_rhs_is_cpu_transfer(v, seen) for v in container.values if v is not None)
            # codex round-26: container is a Name bound to a Dict literal.
            resolved = _resolve_subscript_container(rhs)
            if resolved is not None:
                return _rhs_is_cpu_transfer(resolved, seen)
            # codex round-26: tuple/list literal bound to a Name.
            if isinstance(container, ast.Name):
                tup = _global_tuple_literals.get(container.id)
                if (
                    tup is not None
                    and isinstance(rhs.slice, ast.Constant)
                    and isinstance(rhs.slice.value, int)
                ):
                    idx = rhs.slice.value
                    if -len(tup.elts) <= idx < len(tup.elts):
                        return _rhs_is_cpu_transfer(tup.elts[idx], seen)
        if isinstance(rhs, ast.Call):
            # codex round-26/27: unwrap Call(Lambda)/Call(Name fn)/Call(Call).
            for ret in _call_possible_returns(rhs, seen):
                if _rhs_is_cpu_transfer(ret, seen):
                    return True
        return False

    def _call_possible_returns(c: ast.Call, seen: set[int]) -> list[ast.AST]:
        """Best-effort: possible AST return nodes of evaluating `c`."""
        if isinstance(c.func, ast.Lambda):
            return [c.func.body]
        if isinstance(c.func, ast.Name):
            return list(_global_fn_returns.get(c.func.id, []))
        if isinstance(c.func, ast.Call):
            outs: list[ast.AST] = []
            for inner_ret in _call_possible_returns(c.func, seen):
                if isinstance(inner_ret, ast.Lambda):
                    outs.append(inner_ret.body)
                elif isinstance(inner_ret, ast.Name):
                    outs.extend(_global_fn_returns.get(inner_ret.id, []))
                elif isinstance(inner_ret, ast.Call):
                    outs.extend(_call_possible_returns(inner_ret, seen))
            return outs
        return []

    def _rhs_is_custom_method(rhs: ast.AST, seen: set[int] | None = None) -> bool:
        """Module-level: does this RHS bind to an attribute of a custom (non-Tensor) class?"""
        if seen is None:
            seen = set()
        if id(rhs) in seen:
            return False
        seen.add(id(rhs))
        if isinstance(rhs, ast.Attribute):
            suffix = "." + rhs.attr
            if suffix not in _CALL_ALLOW_METHOD_SUFFIX:
                return False
            # Conservative: receiver is a Name not known as module / alias / Tensor.
            if isinstance(rhs.value, ast.Name) and rhs.value.id not in {
                "tl", "triton", "torch", "torch_npu"
            }:
                return True
            # `b.h.reshape` / `objs[0].reshape` — attribute on Attribute / Subscript.
            if isinstance(rhs.value, (ast.Attribute, ast.Subscript)):
                return True
        if isinstance(rhs, ast.Name):
            return rhs.id in _global_custom_method_aliases
        if isinstance(rhs, ast.IfExp):
            return _rhs_is_custom_method(rhs.body, seen) or _rhs_is_custom_method(rhs.orelse, seen)
        if isinstance(rhs, ast.BoolOp):
            return any(_rhs_is_custom_method(v, seen) for v in rhs.values)
        if isinstance(rhs, ast.Subscript):
            container = rhs.value
            if isinstance(container, (ast.Tuple, ast.List)):
                return any(_rhs_is_custom_method(e, seen) for e in container.elts)
            if isinstance(container, ast.Dict):
                return any(_rhs_is_custom_method(v, seen) for v in container.values if v is not None)
            resolved = _resolve_subscript_container(rhs)
            if resolved is not None:
                return _rhs_is_custom_method(resolved, seen)
        if isinstance(rhs, ast.Call):
            for ret in _call_possible_returns(rhs, seen):
                if _rhs_is_custom_method(ret, seen):
                    return True
        return False

    def _lhs_names(lhs: ast.AST) -> list[str]:
        """All bare Name LHS, recursing through Tuple/List/Starred."""
        if isinstance(lhs, ast.Name):
            return [lhs.id]
        if isinstance(lhs, (ast.Tuple, ast.List)):
            out: list[str] = []
            for elt in lhs.elts:
                out.extend(_lhs_names(elt))
            return out
        if isinstance(lhs, ast.Starred):
            return _lhs_names(lhs.value)
        return []

    def _walk_destructure(lhs: ast.AST, rhs: ast.AST,
                          ct_set: set[str], cm_set: set[str]) -> None:
        """Recursive element-wise matching of (Tuple/List) LHS↔RHS."""
        # codex round-27: resolve Name RHS → bound Tuple/List literal.
        if isinstance(rhs, ast.Name):
            bound = _global_tuple_literals.get(rhs.id)
            if bound is not None:
                rhs = bound
        if isinstance(lhs, (ast.Tuple, ast.List)) and isinstance(rhs, (ast.Tuple, ast.List)):
            # Locate Starred LHS, if any.
            star_idx = None
            for i, lhs_elt in enumerate(lhs.elts):
                if isinstance(lhs_elt, ast.Starred):
                    star_idx = i
                    break
            if star_idx is None:
                for lhs_elt, rhs_elt in zip(lhs.elts, rhs.elts):
                    _walk_destructure(lhs_elt, rhs_elt, ct_set, cm_set)
                return
            # Split RHS into [head | starred | tail].
            head_lhs = lhs.elts[:star_idx]
            tail_lhs = lhs.elts[star_idx + 1:]
            n_tail = len(tail_lhs)
            head_rhs = rhs.elts[:star_idx]
            tail_rhs = rhs.elts[len(rhs.elts) - n_tail:] if n_tail else []
            star_rhs = rhs.elts[star_idx:len(rhs.elts) - n_tail] if n_tail else rhs.elts[star_idx:]
            for l, r in zip(head_lhs, head_rhs):
                _walk_destructure(l, r, ct_set, cm_set)
            for l, r in zip(tail_lhs, tail_rhs):
                _walk_destructure(l, r, ct_set, cm_set)
            star_name_node = lhs.elts[star_idx].value
            if isinstance(star_name_node, ast.Name):
                # Bind starred to a synthesized list of remaining RHS elts.
                synth = ast.List(elts=list(star_rhs), ctx=ast.Load())
                _global_tuple_literals[star_name_node.id] = synth
                # Taint directly if any elt is cpu-transfer / custom.
                if any(_rhs_is_cpu_transfer(e) for e in star_rhs):
                    ct_set.add(star_name_node.id)
                if any(_rhs_is_custom_method(e) for e in star_rhs):
                    cm_set.add(star_name_node.id)
            return
        # Leaf: lhs is a Name (or Starred of Name).
        for nm in _lhs_names(lhs):
            if _rhs_is_cpu_transfer(rhs):
                ct_set.add(nm)
            if _rhs_is_custom_method(rhs):
                cm_set.add(nm)

    def _collect_assign_targets(node) -> tuple[list[ast.AST], ast.AST | None]:
        if isinstance(node, ast.Assign):
            return (node.targets, node.value)
        if isinstance(node, ast.AnnAssign) and node.value is not None:
            return ([node.target], node.value)
        return ([], None)

    # Fixpoint: iterate until no new aliases are discovered (bounded by
    # assignment count to avoid pathological cases).
    _max_passes = max(8, sum(1 for _n in ast.walk(tree) if isinstance(_n, (ast.Assign, ast.AnnAssign))))
    for _ in range(_max_passes):
        prev_ct = len(_global_cpu_transfer_aliases)
        prev_cm = len(_global_custom_method_aliases)
        for s in ast.walk(tree):
            targets, value = _collect_assign_targets(s)
            if not value:
                continue
            for tgt in targets:
                _walk_destructure(
                    tgt, value,
                    _global_cpu_transfer_aliases,
                    _global_custom_method_aliases,
                )
        if len(_global_cpu_transfer_aliases) == prev_ct and len(_global_custom_method_aliases) == prev_cm:
            break

    # codex round-18 NEW-P0 #6: full dict-like head set including
    # OrderedDict / DefaultDict (typed) — used by annotation walkers
    # and type-alias detector.
    _DICT_LIKE_ANNOTATION_HEADS = {
        "dict", "Dict", "typing.Dict",
        "Mapping", "typing.Mapping",
        "MutableMapping", "typing.MutableMapping",
        "OrderedDict", "typing.OrderedDict",
        "DefaultDict", "typing.DefaultDict",
        "ChainMap", "typing.ChainMap",
        "Counter", "typing.Counter",
        "collections.OrderedDict", "collections.defaultdict",
        "collections.ChainMap", "collections.Counter",
    }

    # codex round-11/16: class-bases & type-alias container detection.
    # Round-16 P1: take a transitive closure over subclass chains
    # (`class D(dict); class My(D)` → My is dict-like).
    dict_subclasses: set[str] = set()
    list_tensor_subclasses: set[str] = set()
    type_alias_to_container: set[str] = set()

    # First pass: gather direct base names per class.
    class_bases: dict[str, list[ast.AST]] = {}
    for sub in ast.walk(tree):
        if isinstance(sub, ast.ClassDef):
            class_bases[sub.name] = list(sub.bases)
            for base in sub.bases:
                base_name = _canonical(base) or _dotted_name(base)
                if isinstance(base, ast.Subscript):
                    head = _canonical(base.value) or _dotted_name(base.value)
                    if head in {
                        "list", "List", "typing.List",
                        "tuple", "Tuple", "typing.Tuple",
                    } and any(
                        isinstance(n2, ast.Name) and n2.id == "Tensor"
                        or isinstance(n2, ast.Attribute) and n2.attr == "Tensor"
                        for n2 in ast.walk(base.slice)
                    ):
                        list_tensor_subclasses.add(sub.name)
                    if head in _DICT_LIKE_ANNOTATION_HEADS:
                        dict_subclasses.add(sub.name)
                if base_name in _DICT_LIKE_ANNOTATION_HEADS:
                    dict_subclasses.add(sub.name)

    # Transitive closure: if base name is itself a dict_subclass, this
    # class is also a dict_subclass.
    for _ in range(max(8, len(class_bases) * 2)):
        grew = False
        for cls_name, bases in class_bases.items():
            if cls_name in dict_subclasses:
                continue
            for base in bases:
                base_name = (_canonical(base) or _dotted_name(base) or "").split(".")[-1]
                if base_name in dict_subclasses:
                    dict_subclasses.add(cls_name)
                    grew = True
                    break
        if not grew:
            break

    # type-alias scan (module-level `TensorList = List[Tensor]` etc.)
    for sub in ast.walk(tree):
        if isinstance(sub, ast.Assign) and len(sub.targets) == 1 and isinstance(sub.targets[0], ast.Name):
            tgt = sub.targets[0].id
            val = sub.value
            if isinstance(val, ast.Subscript):
                head = _canonical(val.value) or _dotted_name(val.value)
                if head in _DICT_LIKE_ANNOTATION_HEADS:
                    type_alias_to_container.add(tgt)
                elif head in {
                    "list", "List", "typing.List",
                    "tuple", "Tuple", "typing.Tuple",
                    "Sequence", "typing.Sequence",
                } and any(
                    (isinstance(n2, ast.Name) and n2.id == "Tensor")
                    or (isinstance(n2, ast.Attribute) and n2.attr == "Tensor")
                    or (isinstance(n2, ast.Constant) and isinstance(n2.value, str)
                        and ("Tensor" in n2.value))
                    for n2 in ast.walk(val.slice)
                ):
                    type_alias_to_container.add(tgt)

    # codex round-10 NEW-P0 #2: shallow cross-function Tensor-arg
    # propagation. Two passes:
    # (1) Scan each non-kernel function: collect its Tensor-typed params.
    # (2) For each Call to a top-level function whose actual arg is in
    #     the caller's Tensor set, mark the callee's corresponding formal
    #     parameter Tensor-tainted.
    # codex round-15/16: fn_defs / class_defs include nested functions
    # but exclude class methods (class methods should not shadow
    # top-level fn names).
    fn_defs: dict[str, ast.FunctionDef] = {}
    class_defs: dict[str, ast.ClassDef] = {}

    decorated_unsafely: set[str] = set()
    TRUSTED_DECORATORS = {"triton.jit", "triton.autotune", "triton.heuristics"}

    def _walk_fns_classes(node: ast.AST, in_class: bool = False) -> None:
        if isinstance(node, ast.ClassDef):
            if not in_class:
                class_defs[node.name] = node
            for child in ast.iter_child_nodes(node):
                _walk_fns_classes(child, in_class=True)
            return
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if not in_class:
                # codex round-17 NEW-P0 #4: prefer LAST-binding (nested
                # shadow). Nested defs are visited after top-level by
                # this DFS — overwrite intentionally.
                fn_defs[node.name] = node                        # type: ignore[assignment]
                # codex round-17 NEW-P0 #3: detect non-trusted decorators —
                # the decorated function is REPLACED at runtime, so the
                # audit on its original body is unsafe.
                for deco in node.decorator_list:
                    dname = _canonical(deco) or _dotted_name(deco)
                    if isinstance(deco, ast.Call):
                        dname = _canonical(deco.func) or _dotted_name(deco.func)
                    if dname not in TRUSTED_DECORATORS:
                        decorated_unsafely.add(node.name)
            for child in ast.iter_child_nodes(node):
                _walk_fns_classes(child, in_class=False)
            return
        for child in ast.iter_child_nodes(node):
            _walk_fns_classes(child, in_class=in_class)

    _walk_fns_classes(tree)

    _CROSS_FN_TENSOR_PARAMS: dict[str, set[str]] = {}
    _CROSS_FN_CONTAINER_PARAMS: dict[str, set[str]] = {}

    # Bootstrap: each function's "annotated Tensor params" set.
    fn_initial_tensors: dict[str, set[str]] = {}
    for name, fnd in fn_defs.items():
        ts: set[str] = set()
        args_all = (
            list(fnd.args.posonlyargs)
            + list(fnd.args.args)
            + list(fnd.args.kwonlyargs)
        )
        for a in args_all:
            if a.annotation is not None:
                anno_text = _canonical(a.annotation) or _dotted_name(a.annotation)
                if isinstance(a.annotation, ast.Constant) and isinstance(a.annotation.value, str):
                    anno_text = a.annotation.value
                if anno_text in {"torch.Tensor", "Tensor", "'torch.Tensor'"}:
                    ts.add(a.arg)
        fn_initial_tensors[name] = ts

    # codex round-13/14/16: collect callable aliases — `f = h` /
    # `f = lambda z: ...`. Round-16 NEW-P0 #4: also handle AnnAssign,
    # destructuring tuple/list, for-target, IIFE Call results.
    lambda_aliases: dict[str, list[ast.Lambda]] = {}

    def _record_callable_binding(name: str, value: ast.AST) -> None:  # noqa: C901
        # codex round-20 NEW-P0 #5: Starred, BoolOp aliases.
        if isinstance(value, ast.Starred):
            _record_callable_binding(name, value.value)
            return
        if isinstance(value, ast.BoolOp):
            for v in value.values:
                _record_callable_binding(name, v)
            return
        # Direct callable: function name.
        if isinstance(value, ast.Name):
            if value.id in fn_defs:
                callable_aliases.setdefault(name, set()).add(value.id)
            elif value.id in callable_aliases:
                callable_aliases.setdefault(name, set()).update(
                    callable_aliases[value.id]
                )
        if isinstance(value, ast.Lambda):
            lambda_aliases.setdefault(name, []).append(value)
        # IIFE result `(lambda: h)()` — recurse on body.
        if isinstance(value, ast.Call) and isinstance(value.func, ast.Lambda):
            _record_callable_binding(name, value.func.body)
        # codex round-19/21 NEW-P0: computed alias forms.
        # `g = (h,)[0]` / `g = {"f": h}["f"]` / `g = h if cond else other` /
        # `g = ((h,) or (h,))[0]` (BoolOp inside Subscript).
        if isinstance(value, ast.Subscript):
            # Recurse into the container expression — covers BoolOp/IfExp.
            _record_callable_binding(name, value.value)
            container = value.value
            if isinstance(container, (ast.Tuple, ast.List)):
                for elt in container.elts:
                    _record_callable_binding(name, elt)
            elif isinstance(container, ast.Dict):
                for v in container.values:
                    _record_callable_binding(name, v)
        if isinstance(value, ast.IfExp):
            _record_callable_binding(name, value.body)
            _record_callable_binding(name, value.orelse)
        if isinstance(value, (ast.Tuple, ast.List)):
            for elt in value.elts:
                _record_callable_binding(name, elt)
        if isinstance(value, ast.Dict):
            for v in value.values:
                _record_callable_binding(name, v)

    for sub in ast.walk(tree):
        if isinstance(sub, ast.Assign):
            rhs = sub.value
            for tgt in sub.targets:
                if isinstance(tgt, ast.Name):
                    _record_callable_binding(tgt.id, rhs)
                elif isinstance(tgt, (ast.Tuple, ast.List)) and isinstance(rhs, (ast.Tuple, ast.List)):
                    # codex round-20: handle `*_, g = (h,)` — Starred
                    # consumes leading elements, last Name targets last
                    # Tuple element (conservatively bind every element
                    # to every Name target).
                    has_starred = any(isinstance(e, ast.Starred) for e in tgt.elts)
                    if has_starred:
                        for tgt_elt in tgt.elts:
                            if isinstance(tgt_elt, ast.Starred) and isinstance(tgt_elt.value, ast.Name):
                                for rhs_elt in rhs.elts:
                                    _record_callable_binding(tgt_elt.value.id, rhs_elt)
                            elif isinstance(tgt_elt, ast.Name):
                                for rhs_elt in rhs.elts:
                                    _record_callable_binding(tgt_elt.id, rhs_elt)
                    else:
                        for i, elt in enumerate(tgt.elts):
                            if isinstance(elt, ast.Name) and i < len(rhs.elts):
                                _record_callable_binding(elt.id, rhs.elts[i])
        elif isinstance(sub, ast.AnnAssign) and isinstance(sub.target, ast.Name) and sub.value is not None:
            _record_callable_binding(sub.target.id, sub.value)
        elif isinstance(sub, ast.For) and isinstance(sub.target, ast.Name):
            # `for f in (h,) | [h] | {h}`: bind f to each element.
            if isinstance(sub.iter, (ast.Tuple, ast.List, ast.Set)):
                for elt in sub.iter.elts:
                    _record_callable_binding(sub.target.id, elt)

    # codex round-14 NEW-P0 #4: fold lambda-alias bodies into
    # function_returns so `dev = lambda: "cpu"; x.to(dev())` is caught.
    for name, lams in lambda_aliases.items():
        for lam in lams:
            folded = _eval_str_const(lam.body)
            if folded is None:
                continue
            if "cpu" in folded.lower():
                function_returns[name] = folded
                break
            if name not in function_returns:
                function_returns[name] = folded

    # codex round-11 R-P0-1: iterate to fixed point so multi-hop
    # `h1(x) -> h2(y) -> ...` propagates. Also include local Assign
    # aliases (`y = x`) and keyword actuals, plus container subscript
    # results (`inputs["x"]`).
    def _fn_local_tensors(caller_fn: ast.FunctionDef) -> tuple[set[str], set[str]]:
        local_tensors = set(fn_initial_tensors.get(caller_fn.name, set()))
        local_containers: set[str] = set()
        # Pull annotated containers + npu_triton_impl convention.
        args_all = (
            list(caller_fn.args.posonlyargs)
            + list(caller_fn.args.args)
            + list(caller_fn.args.kwonlyargs)
        )
        for a in args_all:
            if a.annotation is None:
                continue
            anno_text = _canonical(a.annotation) or _dotted_name(a.annotation)
            if isinstance(a.annotation, ast.Subscript):
                head = _canonical(a.annotation.value) or _dotted_name(a.annotation.value)
                if head in _DICT_LIKE_ANNOTATION_HEADS:
                    local_containers.add(a.arg)
            if anno_text in dict_subclasses or anno_text in type_alias_to_container:
                local_containers.add(a.arg)
        if caller_fn.name == "npu_triton_impl":
            pos = list(caller_fn.args.posonlyargs) + list(caller_fn.args.args)
            if pos:
                local_containers.add(pos[0].arg)
        if caller_fn.args.vararg is not None:
            local_containers.add(caller_fn.args.vararg.arg)
        if caller_fn.args.kwarg is not None:
            local_containers.add(caller_fn.args.kwarg.arg)
        # Add cross-fn-propagated tensors discovered so far.
        local_tensors |= _CROSS_FN_TENSOR_PARAMS.get(caller_fn.name, set())
        # Forward-prop within body (fixpoint within the function):
        # codex round-12 R-P0-2: include container alias propagation
        # (`c = inputs`), Tuple destructuring on Tensor RHS, AnnAssign.
        for _ in range(8):
            grew = False
            for s in ast.walk(caller_fn):
                if isinstance(s, ast.Assign):
                    rhs = s.value
                    # codex round-13 R-P0-1: propagate Tensor-ness from
                    # method calls (`x.contiguous()`), IIFE lambdas, and
                    # local helper calls that return Tensor.
                    rhs_tensor = (
                        (isinstance(rhs, ast.Name) and rhs.id in local_tensors)
                        or (
                            isinstance(rhs, ast.Subscript)
                            and isinstance(rhs.value, ast.Name)
                            and rhs.value.id in local_containers
                        )
                        or _rhs_yields_tensor(rhs, local_tensors, local_containers)
                    )
                    rhs_container = (
                        isinstance(rhs, ast.Name) and rhs.id in local_containers
                    )
                    # codex round-12/13: Tuple/List RHS only triggers
                    # destructure path when its elements actually contain
                    # a Tensor — `dims = (1, 2)` is NOT a Tensor tuple.
                    if isinstance(rhs, (ast.Tuple, ast.List)):
                        if any(
                            _expr_contains_tensor(e, local_tensors, local_containers)
                            for e in rhs.elts
                        ):
                            rhs_tensor = True
                    # codex round-13 NEW-P0 #1: subscript / attribute
                    # write of Tensor into a local → that local becomes
                    # a Tensor container.
                    if _expr_contains_tensor(rhs, local_tensors, local_containers):
                        for tgt in s.targets:
                            if isinstance(tgt, ast.Subscript) and isinstance(tgt.value, ast.Name):
                                if tgt.value.id not in local_containers:
                                    local_containers.add(tgt.value.id)
                                    grew = True
                            elif isinstance(tgt, ast.Attribute) and isinstance(tgt.value, ast.Name):
                                if tgt.value.id not in local_containers:
                                    local_containers.add(tgt.value.id)
                                    grew = True

                    def _spread(target: ast.expr, is_t: bool, is_c: bool) -> None:
                        nonlocal grew
                        if isinstance(target, ast.Name):
                            if is_t and target.id not in local_tensors:
                                local_tensors.add(target.id)
                                grew = True
                            if is_c and target.id not in local_containers:
                                local_containers.add(target.id)
                                grew = True
                        elif isinstance(target, (ast.Tuple, ast.List)):
                            # Destructure: if RHS is a Tuple/List literal,
                            # zip elements; otherwise propagate type to
                            # all targets (conservative).
                            if isinstance(rhs, (ast.Tuple, ast.List)):
                                for i, sub_tgt in enumerate(target.elts):
                                    if i < len(rhs.elts):
                                        sub_rhs = rhs.elts[i]
                                        elem_t = (
                                            isinstance(sub_rhs, ast.Name)
                                            and sub_rhs.id in local_tensors
                                        ) or (
                                            isinstance(sub_rhs, ast.Subscript)
                                            and isinstance(sub_rhs.value, ast.Name)
                                            and sub_rhs.value.id in local_containers
                                        )
                                        elem_c = (
                                            isinstance(sub_rhs, ast.Name)
                                            and sub_rhs.id in local_containers
                                        )
                                        _spread(sub_tgt, elem_t, elem_c)
                            else:
                                for sub_tgt in target.elts:
                                    _spread(sub_tgt, is_t, is_c)

                    if rhs_tensor or rhs_container:
                        for tgt in s.targets:
                            _spread(tgt, rhs_tensor, rhs_container)
                elif isinstance(s, ast.AnnAssign) and s.value is not None and isinstance(s.target, ast.Name):
                    rhs = s.value
                    if isinstance(rhs, ast.Name) and rhs.id in local_tensors:
                        if s.target.id not in local_tensors:
                            local_tensors.add(s.target.id)
                            grew = True
                    elif isinstance(rhs, ast.Name) and rhs.id in local_containers:
                        if s.target.id not in local_containers:
                            local_containers.add(s.target.id)
                            grew = True
            if not grew:
                break
        return local_tensors, local_containers

    # codex round-12 R-P0-1: raise cap to a hard fixpoint; iteration is
    # bounded by `len(fn_defs)` (call graph depth) and call sites.
    _MAX_ITERATIONS = max(64, len(fn_defs) * 4)
    _hit_cap = True
    for _iteration in range(_MAX_ITERATIONS):
        changed = False
        for caller_name, caller_fn in fn_defs.items():
            local_tensors, local_containers = _fn_local_tensors(caller_fn)
            for sub in ast.walk(caller_fn):
                if not isinstance(sub, ast.Call):
                    continue
                if not isinstance(sub.func, ast.Name):
                    continue
                # Round-13/14: resolve callable alias chain — expand to all
                # possible callees via UNION semantics. Any binding that
                # ever pointed at a real fn_def is propagated.
                seen: set[str] = set()
                worklist = [sub.func.id]
                resolved: set[str] = set()
                while worklist:
                    name = worklist.pop()
                    if name in seen:
                        continue
                    seen.add(name)
                    if name in fn_defs:
                        resolved.add(name)
                    for alias_target in callable_aliases.get(name, ()):
                        worklist.append(alias_target)
                if not resolved:
                    continue
                # Apply propagation to ALL resolved callees.
                def _actual_is_tensor(actual: ast.AST) -> bool:
                    return _expr_contains_tensor(
                        actual, local_tensors, local_containers
                    )

                def _actual_is_container(actual: ast.AST) -> bool:
                    """codex round-16/17: passing a container (directly,
                    via Subscript chain, IIFE Lambda, or wrapped in
                    Tuple/List/Dict) to a helper taints the formal as
                    container."""
                    # Direct Name reference.
                    if isinstance(actual, ast.Name) and actual.id in local_containers:
                        return True
                    # Walk wrapping expressions.
                    for sub in ast.walk(actual):
                        if isinstance(sub, ast.Name) and sub.id in local_containers:
                            return True
                    return False

                for callee_name in resolved:
                    callee_fn = fn_defs.get(callee_name)
                    if callee_fn is None or id(callee_fn) in kernel_fn_ids:
                        continue
                    # codex round-17 NEW-P0 #2: bind callee's defaults
                    # to formals so `def h(z=x): return z + z` is audited.
                    pos_args_callee = list(callee_fn.args.posonlyargs) + list(callee_fn.args.args)
                    defaults_callee = list(callee_fn.args.defaults)
                    n_def = len(defaults_callee)
                    for i, default in enumerate(defaults_callee):
                        idx = len(pos_args_callee) - n_def + i
                        if 0 <= idx < len(pos_args_callee):
                            formal = pos_args_callee[idx].arg
                            if _expr_contains_tensor(default, local_tensors, local_containers):
                                _CROSS_FN_TENSOR_PARAMS.setdefault(callee_name, set()).add(formal)
                                changed = True
                            if _actual_is_container(default):
                                _CROSS_FN_CONTAINER_PARAMS.setdefault(callee_name, set()).add(formal)
                                changed = True
                    # codex round-23: kw-only defaults containing Tensor.
                    kwonly_callee = list(callee_fn.args.kwonlyargs)
                    for i, default in enumerate(callee_fn.args.kw_defaults):
                        if default is None or i >= len(kwonly_callee):
                            continue
                        formal = kwonly_callee[i].arg
                        if _expr_contains_tensor(default, local_tensors, local_containers):
                            _CROSS_FN_TENSOR_PARAMS.setdefault(callee_name, set()).add(formal)
                            changed = True
                    pos_formals = list(callee_fn.args.posonlyargs) + list(callee_fn.args.args)
                    kwonly_formals = list(callee_fn.args.kwonlyargs)
                    for i, actual in enumerate(sub.args):
                        if i >= len(pos_formals):
                            break
                        formal_name = pos_formals[i].arg
                        if _actual_is_tensor(actual):
                            existing = _CROSS_FN_TENSOR_PARAMS.setdefault(callee_name, set())
                            if formal_name not in existing:
                                existing.add(formal_name)
                                changed = True
                        if _actual_is_container(actual):
                            existing_c = _CROSS_FN_CONTAINER_PARAMS.setdefault(callee_name, set())
                            if formal_name not in existing_c:
                                existing_c.add(formal_name)
                                changed = True
                    for kw in sub.keywords:
                        # codex round-18 NEW-P0 #5: `**{"z": x}` unpacking.
                        if kw.arg is None:
                            # ** unpacking — if value contains Tensor/container,
                            # taint ALL formals conservatively.
                            if _expr_contains_tensor(kw.value, local_tensors, local_containers):
                                for f in pos_formals + kwonly_formals:
                                    _CROSS_FN_TENSOR_PARAMS.setdefault(callee_name, set()).add(f.arg)
                                changed = True
                            if _actual_is_container(kw.value):
                                for f in pos_formals + kwonly_formals:
                                    _CROSS_FN_CONTAINER_PARAMS.setdefault(callee_name, set()).add(f.arg)
                                changed = True
                            continue
                        if any(f.arg == kw.arg for f in pos_formals + kwonly_formals):
                            if _actual_is_tensor(kw.value):
                                existing = _CROSS_FN_TENSOR_PARAMS.setdefault(callee_name, set())
                                if kw.arg not in existing:
                                    existing.add(kw.arg)
                                    changed = True
                            if _actual_is_container(kw.value):
                                existing_c = _CROSS_FN_CONTAINER_PARAMS.setdefault(callee_name, set())
                                if kw.arg not in existing_c:
                                    existing_c.add(kw.arg)
                                    changed = True
        if not changed:
            _hit_cap = False
            break
    if _hit_cap:
        # Pathological call graph (or genuine cycle) — fail closed by
        # tainting every parameter of every helper that receives a Tensor
        # actual at any visited call site.
        for caller_fn in fn_defs.values():
            local_tensors, local_containers = _fn_local_tensors(caller_fn)
            for sub in ast.walk(caller_fn):
                if not isinstance(sub, ast.Call) or not isinstance(sub.func, ast.Name):
                    continue
                callee_fn = fn_defs.get(sub.func.id)
                if callee_fn is None or id(callee_fn) in kernel_fn_ids:
                    continue
                callee_args = list(callee_fn.args.posonlyargs) + list(callee_fn.args.args)
                for formal in callee_args:
                    _CROSS_FN_TENSOR_PARAMS.setdefault(callee_fn.name, set()).add(
                        formal.arg
                    )

    for fn_node in ast.walk(tree):
        if not isinstance(fn_node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if id(fn_node) in kernel_fn_ids:
            continue                                  # don't audit kernels here

        tensors: set[str] = set()
        # Args annotated as torch.Tensor (or as opaque container dict) are
        # Tensor-tainted. codex round-9 NEW-P0 #1: `inputs: dict` is the
        # canonical Phase-2.B contract entry — `inputs["x"]` is a Tensor.
        all_args = (
            list(fn_node.args.posonlyargs)
            + list(fn_node.args.args)
            + list(fn_node.args.kwonlyargs)
        )
        # Tensor-like annotations: bare Tensor (`x: torch.Tensor`) or
        # forward-ref string ('torch.Tensor'). Container annotations
        # (`inputs: dict`, `xs: List[torch.Tensor]`) are handled
        # separately and seed the `containers` set rather than `tensors`.
        _TENSOR_ANNOTATIONS = {
            "torch.Tensor", "Tensor", "'torch.Tensor'",
        }
        # Always-Tensor-container heads — dict-like contracts hold Tensors
        # by convention in this benchmark; treat opaque even without a
        # slice annotation.
        _ALWAYS_TENSOR_CONTAINER_HEADS = set(_DICT_LIKE_ANNOTATION_HEADS)
        # Conditional container heads — `list[int]` / `tuple[int, int]`
        # is NOT a Tensor container, but `list[torch.Tensor]` is. Require
        # an explicit `torch.Tensor` element in the slice annotation.
        _CONDITIONAL_CONTAINER_HEADS = {
            "list", "List", "typing.List",
            "tuple", "Tuple", "typing.Tuple",
            "Sequence", "typing.Sequence",
            "Iterable", "typing.Iterable",
            "Collection", "typing.Collection",
        }
        _CONTAINER_HEADS = _ALWAYS_TENSOR_CONTAINER_HEADS | _CONDITIONAL_CONTAINER_HEADS
        containers = getattr(fn_node, "_tab_containers", None) or set()
        def _slice_has_tensor(slice_node: ast.AST) -> bool:
            for sub in ast.walk(slice_node):
                if isinstance(sub, ast.Name) and sub.id == "Tensor":
                    return True
                if isinstance(sub, ast.Attribute) and sub.attr == "Tensor":
                    return True
                if isinstance(sub, ast.Constant) and isinstance(sub.value, str):
                    # codex round-11 P1: word-boundary match, not substring.
                    parts = re.split(r"[,\s\[\]\(\)\|]+", sub.value)
                    if any(p.strip("'\"") in {"Tensor", "torch.Tensor"} for p in parts):
                        return True
            return False

        def _annotation_is_tensor_container(anno_node: ast.AST | None) -> bool:
            """Round-11: deep walk annotations including
            `Optional/Union/T | None / List[Tensor] / dict subclasses /
            type aliases`."""
            if anno_node is None:
                return False
            if isinstance(anno_node, ast.Name):
                if anno_node.id in dict_subclasses or anno_node.id in list_tensor_subclasses:
                    return True
                if anno_node.id in type_alias_to_container:
                    return True
                if anno_node.id in _ALWAYS_TENSOR_CONTAINER_HEADS:
                    return True
                return False
            if isinstance(anno_node, ast.Attribute):
                head = _canonical(anno_node) or _dotted_name(anno_node)
                return head in _ALWAYS_TENSOR_CONTAINER_HEADS
            if isinstance(anno_node, ast.Subscript):
                head = _canonical(anno_node.value) or _dotted_name(anno_node.value)
                if head in {"Optional", "typing.Optional", "Union", "typing.Union"}:
                    return _slice_has_container(anno_node.slice)
                if head in _ALWAYS_TENSOR_CONTAINER_HEADS:
                    return True
                if head in _CONDITIONAL_CONTAINER_HEADS and _slice_has_tensor(anno_node.slice):
                    return True
                return False
            if isinstance(anno_node, ast.BinOp) and isinstance(anno_node.op, ast.BitOr):
                return (
                    _annotation_is_tensor_container(anno_node.left)
                    or _annotation_is_tensor_container(anno_node.right)
                )
            if isinstance(anno_node, ast.Constant) and isinstance(anno_node.value, str):
                tokens = re.split(r"[,\s\[\]\(\)\|]+", anno_node.value)
                has_tensor_token = any(
                    t.strip("'\"") in {"Tensor", "torch.Tensor"} for t in tokens
                )
                head = anno_node.value.split("[")[0].strip().strip("'\"")
                # Conditional containers (List/Tuple/Sequence/...) require
                # an explicit Tensor token in the slice. Word-boundary
                # match avoids "NotTensor" false-positives.
                if head in _CONDITIONAL_CONTAINER_HEADS and has_tensor_token:
                    return True
                if head in _ALWAYS_TENSOR_CONTAINER_HEADS:
                    return True
                if head in dict_subclasses or head in type_alias_to_container:
                    return True
            return False

        def _slice_has_container(slice_node: ast.AST) -> bool:
            if isinstance(slice_node, ast.Tuple):
                return any(_annotation_is_tensor_container(e) for e in slice_node.elts)
            return _annotation_is_tensor_container(slice_node)

        def _annotation_is_tensor(anno_node: ast.AST | None) -> bool:
            """Recursive Tensor-ness check for annotations. Handles
            `torch.Tensor`, `Tensor`, `Optional[torch.Tensor]`,
            `Union[torch.Tensor, None]`, `torch.Tensor | None`,
            string forward-refs."""
            if anno_node is None:
                return False
            if isinstance(anno_node, ast.Name):
                return anno_node.id in _TENSOR_ANNOTATIONS
            if isinstance(anno_node, ast.Attribute):
                head = _canonical(anno_node) or _dotted_name(anno_node)
                return head in _TENSOR_ANNOTATIONS
            if isinstance(anno_node, ast.Subscript):
                head = _canonical(anno_node.value) or _dotted_name(anno_node.value)
                if head in {"Optional", "typing.Optional", "Union", "typing.Union"}:
                    if isinstance(anno_node.slice, ast.Tuple):
                        return any(_annotation_is_tensor(e) for e in anno_node.slice.elts)
                    return _annotation_is_tensor(anno_node.slice)
            if isinstance(anno_node, ast.BinOp) and isinstance(anno_node.op, ast.BitOr):
                return _annotation_is_tensor(anno_node.left) or _annotation_is_tensor(anno_node.right)
            if isinstance(anno_node, ast.Constant) and isinstance(anno_node.value, str):
                tokens = re.split(r"[,\s\[\]\(\)\|]+", anno_node.value)
                return any(t.strip("'\"") in _TENSOR_ANNOTATIONS for t in tokens)
            return False

        for arg in all_args:
            anno_node = arg.annotation
            if anno_node is None:
                continue
            if _annotation_is_tensor_container(anno_node):
                containers.add(arg.arg)
                continue
            if _annotation_is_tensor(anno_node):
                tensors.add(arg.arg)
                continue
            # Legacy fallthroughs (kept for direct text matches not covered
            # by the recursive walker).
            anno = _canonical(anno_node) or _dotted_name(anno_node)
            if anno in _TENSOR_ANNOTATIONS:
                tensors.add(arg.arg)
        # codex round-10 NEW-P0 #1: contract entry name `npu_triton_impl`
        # — first arg is always a Tensor container (`inputs`), even when
        # the submitter omits the annotation or annotates as `object`.
        if fn_node.name == "npu_triton_impl":
            pos = list(fn_node.args.posonlyargs) + list(fn_node.args.args)
            if pos:
                containers.add(pos[0].arg)
        # codex round-8 R-P0-1: `*args` / `**kwargs` are opaque containers.
        if fn_node.args.vararg is not None:
            containers.add(fn_node.args.vararg.arg)
        if fn_node.args.kwarg is not None:
            containers.add(fn_node.args.kwarg.arg)
        # codex round-10/16: cross-function tensor + container propagation.
        cross_fn_tensor_params = _CROSS_FN_TENSOR_PARAMS.get(fn_node.name, set())
        cross_fn_container_params = _CROSS_FN_CONTAINER_PARAMS.get(fn_node.name, set())
        for arg in all_args:
            if arg.arg in cross_fn_tensor_params:
                tensors.add(arg.arg)
            if arg.arg in cross_fn_container_params:
                containers.add(arg.arg)

        # Forward-propagate Tensor type via Assigns inside this function.
        # Tuple-unpacking propagates Tensor-ness to ALL targets if the
        # RHS is a Tensor-yielding call.
        def _spread_tensor(target: ast.expr) -> None:
            if isinstance(target, ast.Name):
                tensors.add(target.id)
            elif isinstance(target, (ast.Tuple, ast.List)):
                for elt in target.elts:
                    _spread_tensor(elt)
            elif isinstance(target, ast.Starred):
                _spread_tensor(target.value)
            elif isinstance(target, (ast.Subscript, ast.Attribute)):
                # codex round-13 NEW-P0 #1: writing Tensor into a slot /
                # attribute promotes the base name to a Tensor container.
                if isinstance(target.value, ast.Name):
                    containers.add(target.value.id)

        # Iterate to a fixed point — same-pass forward propagation may
        # reveal new Tensor names that change later RHS evaluation.
        for _ in range(8):
            grew = False
            for sub in ast.walk(fn_node):
                if isinstance(sub, ast.Assign) and _rhs_yields_tensor(sub.value, tensors, containers):
                    snap_t = len(tensors)
                    snap_c = len(containers)
                    for tgt in sub.targets:
                        _spread_tensor(tgt)
                    if len(tensors) > snap_t or len(containers) > snap_c:
                        grew = True
            if not grew:
                break

        for sub in ast.walk(fn_node):
            if isinstance(sub, ast.Assign):
                if _rhs_yields_tensor(sub.value, tensors, containers):
                    for tgt in sub.targets:
                        _spread_tensor(tgt)
                # codex round-14/15/16: container alias propagation.
                # Round-16 NEW-P0 #2: handle any RHS containing a
                # container reference (not just direct Name), via
                # `_expr_contains_container`.
                def _root_name(node: ast.AST) -> str | None:
                    cur = node
                    while isinstance(cur, (ast.Subscript, ast.Attribute)):
                        cur = cur.value
                    if isinstance(cur, ast.Name):
                        return cur.id
                    return None

                def _expr_contains_container(
                    node: ast.AST, ctx: set[str] | None = None
                ) -> bool:
                    """Round-17/18 broadened: also walks BinOp (multiply
                    list), IIFE Lambda defaults."""
                    cs = ctx if ctx is not None else containers
                    if isinstance(node, ast.Name):
                        return node.id in cs
                    if isinstance(node, ast.Call):
                        if isinstance(node.func, ast.Lambda):
                            bound = set(cs)
                            lam = node.func
                            pos_formals = list(lam.args.posonlyargs) + list(lam.args.args)
                            defaults = list(lam.args.defaults)
                            n_def = len(defaults)
                            for i, formal in enumerate(pos_formals):
                                actual: ast.AST | None = None
                                if i < len(node.args):
                                    actual = node.args[i]
                                else:
                                    d_idx = i - (len(pos_formals) - n_def)
                                    if 0 <= d_idx < n_def:
                                        actual = defaults[d_idx]
                                if actual is not None and _expr_contains_container(actual, cs):
                                    bound.add(formal.arg)
                            for kw in node.keywords:
                                if kw.arg is None:
                                    continue
                                if _expr_contains_container(kw.value, cs):
                                    for f in pos_formals:
                                        if f.arg == kw.arg:
                                            bound.add(f.arg)
                            return _expr_contains_container(lam.body, bound)
                        # codex round-19: include keyword args for `dict(a=inputs)`.
                        if any(_expr_contains_container(a, cs) for a in node.args):
                            return True
                        if any(_expr_contains_container(kw.value, cs) for kw in node.keywords):
                            return True
                        return False
                    if isinstance(node, ast.Subscript):
                        return _expr_contains_container(node.value, cs)
                    if isinstance(node, ast.IfExp):
                        return (
                            _expr_contains_container(node.body, cs)
                            or _expr_contains_container(node.orelse, cs)
                        )
                    if isinstance(node, (ast.Tuple, ast.List, ast.Set)):
                        return any(_expr_contains_container(e, cs) for e in node.elts)
                    if isinstance(node, ast.Dict):
                        return any(_expr_contains_container(v, cs) for v in node.values)
                    # codex round-18: `[inputs] * 1` / `list1 + list2`.
                    if isinstance(node, ast.BinOp):
                        return (
                            _expr_contains_container(node.left, cs)
                            or _expr_contains_container(node.right, cs)
                        )
                    return False

                if _expr_contains_container(sub.value):
                    for tgt in sub.targets:
                        # codex round-18: Tuple/List destructure of a
                        # container-bearing RHS → all elements are
                        # containers (typical: `(c,) = box`).
                        if isinstance(tgt, (ast.Tuple, ast.List)):
                            for elt in tgt.elts:
                                if isinstance(elt, ast.Name):
                                    containers.add(elt.id)
                        elif isinstance(tgt, ast.Name):
                            containers.add(tgt.id)
                        else:
                            root = _root_name(tgt)
                            if root:
                                containers.add(root)
            elif isinstance(sub, ast.AnnAssign) and sub.value is not None:
                anno = _canonical(sub.annotation) or _dotted_name(sub.annotation)
                if isinstance(sub.annotation, ast.Constant) and isinstance(sub.annotation.value, str):
                    anno = sub.annotation.value
                if (
                    anno in {"torch.Tensor", "Tensor", "'torch.Tensor'"}
                    and isinstance(sub.target, ast.Name)
                ):
                    tensors.add(sub.target.id)
                if _rhs_yields_tensor(sub.value, tensors, containers) and isinstance(sub.target, ast.Name):
                    tensors.add(sub.target.id)
                # codex round-17/19/20: AnnAssign with any container-
                # bearing RHS uses the same recursive walker as Assign.
                if isinstance(sub.target, ast.Name):
                    # Reuse the broad walker — handles `box: object = [inputs]`.
                    def _rhs_has_container(v: ast.AST) -> bool:
                        return any(
                            isinstance(s, ast.Name) and s.id in containers
                            for s in ast.walk(v)
                        )
                    if _rhs_has_container(sub.value):
                        containers.add(sub.target.id)
            elif isinstance(sub, ast.For):
                # `for row in tensor:` / `for _, row in enumerate(tensor):`
                if _expr_contains_tensor(sub.iter, tensors):
                    _spread_tensor(sub.target)
                # codex round-23: `for g in s` where s is bound to a
                # local callable container. Guard against being called
                # in the module-level walker scope.
                _per_fn_callable_box = locals().get("callable_container_names_in_fn") or set()
                _per_fn_callable_instance = locals().get("callable_instance_names_in_fn")
                if (
                    isinstance(sub.iter, ast.Name)
                    and sub.iter.id in _per_fn_callable_box
                    and isinstance(sub.target, ast.Name)
                    and _per_fn_callable_instance is not None
                ):
                    _per_fn_callable_instance.add(sub.target.id)
                # codex round-17/18/19: For-target gets container if
                # iter contains container in any element (incl. IfExp).
                def _for_target_container(target: ast.expr) -> None:
                    if isinstance(target, ast.Name):
                        containers.add(target.id)
                    elif isinstance(target, (ast.Tuple, ast.List)):
                        # `for (c,) in [(inputs,)]:` — destructure to
                        # each element.
                        for elt in target.elts:
                            _for_target_container(elt)

                def _iter_has_container(it: ast.AST) -> bool:
                    # Broad walker — covers Call (e.g. `dict(a=inputs)`),
                    # Subscript, etc. anything that references a known
                    # container Name.
                    for s in ast.walk(it):
                        if isinstance(s, ast.Name) and s.id in containers:
                            return True
                    return False

                if _iter_has_container(sub.iter):
                    _for_target_container(sub.target)

        if not tensors and not containers:
            continue

        # codex round-15/17: collect instance assignments. Round-17
        # NEW-P0 #5: also track `f = factory()` where factory returns
        # a class instance — any local fn call result is conservatively
        # treated as a callable-helper unless it's a known-safe call.
        # codex round-23: names bound to `x.to` / `torch.empty` etc.
        # so that `move = x.to; move("cpu")` triggers CPU transfer check.
        cpu_transfer_aliases: set[str] = set()
        # codex round-23: names bound to allowlisted method on
        # non-Tensor receiver (e.g. `f = a.reshape` where `a` is custom).
        custom_method_aliases: set[str] = set()
        # codex round-23: names bound to local tuple/list/set of callables
        # (e.g. `s = (h,); for g in s:`).
        callable_container_names_in_fn: set[str] = set()
        instance_names_in_fn: set[str] = set()
        # Names known to come from local Call returning a class instance.
        callable_instance_names_in_fn: set[str] = set()
        SAFE_CALL_RETURNS = {
            "torch.empty", "torch.empty_like",
            "torch.zeros", "torch.zeros_like",
            "torch.ones", "torch.ones_like",
            "torch.full", "torch.full_like",
            "torch.tensor", "torch.as_tensor", "torch.arange",
            "triton.cdiv", "triton.next_power_of_2",
        }
        for s in ast.walk(fn_node):
            if isinstance(s, ast.Assign) and isinstance(s.value, ast.Call):
                if isinstance(s.value.func, ast.Name) and s.value.func.id in class_defs:
                    for tgt in s.targets:
                        if isinstance(tgt, ast.Name):
                            instance_names_in_fn.add(tgt.id)
                else:
                    fname = _canonical(s.value.func) or _dotted_name(s.value.func)
                    if fname and fname not in SAFE_CALL_RETURNS and not fname.startswith(
                        ("tl.", "triton.", "torch.")
                    ):
                        for tgt in s.targets:
                            if isinstance(tgt, ast.Name):
                                callable_instance_names_in_fn.add(tgt.id)
            # codex round-23: `move = x.to` / `alloc = torch.empty`.
            if isinstance(s, ast.Assign) and isinstance(s.value, ast.Attribute):
                v = s.value
                # `x.to` where x is Tensor.
                if (
                    v.attr == "to"
                    and isinstance(v.value, ast.Name)
                    and v.value.id in tensors
                ):
                    for tgt in s.targets:
                        if isinstance(tgt, ast.Name):
                            cpu_transfer_aliases.add(tgt.id)
                # `torch.empty` / `torch.zeros` etc.
                dotted = _canonical(v) or _dotted_name(v)
                if dotted in _TENSOR_CTORS or dotted.startswith(("torch.empty", "torch.zeros", "torch.ones", "torch.full")):
                    for tgt in s.targets:
                        if isinstance(tgt, ast.Name):
                            cpu_transfer_aliases.add(tgt.id)
                # codex round-23: `f = a.reshape` (custom-class method).
                suffix = "." + v.attr
                if (
                    suffix in _CALL_ALLOW_METHOD_SUFFIX
                    and isinstance(v.value, ast.Name)
                    and v.value.id not in tensors
                    and v.value.id not in containers
                    and v.value.id not in {"tl", "triton", "torch", "torch_npu"}
                    and v.value.id not in aliases
                ):
                    for tgt in s.targets:
                        if isinstance(tgt, ast.Name):
                            custom_method_aliases.add(tgt.id)
            # codex round-23: `s = (h,) / [h] / {h}` containing callables
            if isinstance(s, ast.Assign) and isinstance(s.value, (ast.Tuple, ast.List, ast.Set)):
                has_callable = False
                for elt in s.value.elts:
                    if isinstance(elt, ast.Name) and (
                        elt.id in fn_defs
                        or elt.id in callable_aliases
                        or elt.id in lambda_aliases
                    ):
                        has_callable = True
                        break
                if has_callable:
                    for tgt in s.targets:
                        if isinstance(tgt, ast.Name):
                            callable_container_names_in_fn.add(tgt.id)
            # codex round-24: destructure `(f,) = (a.reshape,)` or
            # `[f, g] = [x.to, y.cpu]` — match element-wise from Tuple/List
            # RHS into Tuple/List LHS targets.
            if (
                isinstance(s, ast.Assign)
                and isinstance(s.value, (ast.Tuple, ast.List))
            ):
                for tgt in s.targets:
                    if isinstance(tgt, (ast.Tuple, ast.List)) and len(tgt.elts) == len(s.value.elts):
                        for lhs_elt, rhs_elt in zip(tgt.elts, s.value.elts):
                            if not isinstance(lhs_elt, ast.Name):
                                continue
                            if isinstance(rhs_elt, ast.Attribute):
                                # `(move,) = (x.to,)`
                                if (
                                    rhs_elt.attr == "to"
                                    and isinstance(rhs_elt.value, ast.Name)
                                    and rhs_elt.value.id in tensors
                                ):
                                    cpu_transfer_aliases.add(lhs_elt.id)
                                # `(f,) = (a.reshape,)`
                                rsuffix = "." + rhs_elt.attr
                                if (
                                    rsuffix in _CALL_ALLOW_METHOD_SUFFIX
                                    and isinstance(rhs_elt.value, ast.Name)
                                    and rhs_elt.value.id not in tensors
                                    and rhs_elt.value.id not in containers
                                    and rhs_elt.value.id not in {"tl", "triton", "torch", "torch_npu"}
                                    and rhs_elt.value.id not in aliases
                                ):
                                    custom_method_aliases.add(lhs_elt.id)
                            elif isinstance(rhs_elt, ast.Name):
                                # `(f,) = (g,)` — propagate alias kind.
                                if rhs_elt.id in custom_method_aliases:
                                    custom_method_aliases.add(lhs_elt.id)
                                if rhs_elt.id in cpu_transfer_aliases:
                                    cpu_transfer_aliases.add(lhs_elt.id)

        # Now scan for forbidden ops. Round-10: thread `containers` set
        # so that `inputs["x"]` (Subscript on container) unwraps to
        # Tensor and only fails when used in a forbidden op.
        for sub in ast.walk(fn_node):
            if isinstance(sub, ast.BinOp):
                for side in (sub.left, sub.right):
                    if _expr_contains_tensor(side, tensors, containers):
                        report.add(
                            "G-tensor-binop", "block",
                            "host-side Tensor BinOp (eager evaluation; "
                            "use a triton kernel)",
                            getattr(sub, "lineno", None),
                        )
                        break
            elif isinstance(sub, ast.Compare):
                if _expr_contains_tensor(sub.left, tensors, containers) or any(
                    _expr_contains_tensor(c, tensors, containers)
                    for c in sub.comparators
                ):
                    report.add(
                        "G-tensor-compare", "block",
                        "host-side Tensor comparison",
                        getattr(sub, "lineno", None),
                    )
            elif isinstance(sub, ast.UnaryOp):
                if _expr_contains_tensor(sub.operand, tensors, containers):
                    report.add(
                        "G-tensor-unary", "block",
                        "host-side Tensor unary op",
                        getattr(sub, "lineno", None),
                    )
            elif isinstance(sub, ast.BoolOp):
                if any(
                    _expr_contains_tensor(v, tensors, containers)
                    for v in sub.values
                ):
                    report.add(
                        "G-tensor-boolop", "block",
                        "host-side Tensor boolean op",
                        getattr(sub, "lineno", None),
                    )
            elif isinstance(sub, ast.AugAssign):
                # codex round-23: also audit RHS (`y += x`).
                if (
                    _expr_contains_tensor(sub.target, tensors, containers)
                    or _expr_contains_tensor(sub.value, tensors, containers)
                ):
                    report.add(
                        "G-tensor-augassign", "block",
                        "host-side Tensor augmented-assign",
                        getattr(sub, "lineno", None),
                    )
            elif isinstance(sub, ast.Subscript):
                base = sub.value
                # `.shape[i]` etc. — fine.
                if isinstance(base, ast.Attribute) and (
                    "." + base.attr in _PROP_ALLOW_SUFFIX
                ):
                    pass
                # Subscript on a container is legitimate (extract Tensor).
                elif isinstance(base, ast.Name) and base.id in containers:
                    pass
                elif _expr_contains_tensor(base, tensors, containers):
                    report.add(
                        "G-tensor-subscript", "block",
                        "host-side Tensor subscripting",
                        getattr(sub, "lineno", None),
                    )
            elif isinstance(sub, ast.For):
                if _expr_contains_tensor(sub.iter, tensors, containers):
                    report.add(
                        "G-tensor-iter", "block",
                        "host-side iteration over Tensor expression",
                        getattr(sub, "lineno", None),
                    )
                # codex round-23: `for g in s` where s is callable_container_box.
                if (
                    isinstance(sub.iter, ast.Name)
                    and sub.iter.id in callable_container_names_in_fn
                    and isinstance(sub.target, ast.Name)
                ):
                    callable_instance_names_in_fn.add(sub.target.id)
            # codex round-23: Tensor truthiness in control flow.
            elif isinstance(sub, (ast.If, ast.While, ast.Assert)):
                test_node = sub.test if isinstance(sub, ast.Assert) else sub.test
                if _expr_contains_tensor(test_node, tensors, containers):
                    report.add(
                        "G-tensor-truthiness", "block",
                        "host-side Tensor truthiness (if/while/assert)",
                        getattr(sub, "lineno", None),
                    )
            elif isinstance(sub, ast.IfExp):
                if _expr_contains_tensor(sub.test, tensors, containers):
                    report.add(
                        "G-tensor-truthiness", "block",
                        "host-side Tensor truthiness (ternary if-expr)",
                        getattr(sub, "lineno", None),
                    )
            elif isinstance(sub, ast.Call):
                fname = _canonical(sub.func) or _dotted_name(sub.func)
                # codex round-12 NEW-P0 #2 + round-14: callable-object,
                # computed call target, and method on non-Tensor receiver
                # — fail closed if any actual arg evaluates to a Tensor.
                func = sub.func
                # Triton kernel launch idiom: `_fwd_kernel[grid](args...)`.
                # Subscript on a Name that is a @triton.jit fn is legitimate.
                # codex round-15 P1: Triton kernel launch
                # `kernel[grid](...)` — kernel may be the original
                # @triton.jit fn OR a callable alias to one.
                def _resolves_to_kernel(name: str) -> bool:
                    seen: set[str] = set()
                    cur = name
                    while cur not in seen:
                        seen.add(cur)
                        if cur in fn_defs and id(fn_defs[cur]) in kernel_fn_ids:
                            return True
                        if cur in callable_aliases:
                            targets = callable_aliases[cur]
                            if any(
                                t in fn_defs and id(fn_defs[t]) in kernel_fn_ids
                                for t in targets
                            ):
                                return True
                            cur = next(iter(targets))
                        else:
                            break
                    return False

                is_triton_kernel_launch = (
                    isinstance(func, ast.Subscript)
                    and isinstance(func.value, ast.Name)
                    and _resolves_to_kernel(func.value.id)
                )
                # Lambda IIFE is audited by the dedicated lambda body
                # check below; do NOT include it in the indirect-call ban.
                func_is_indirect = (
                    isinstance(func, ast.Call)
                    or (
                        isinstance(func, ast.Attribute)
                        and isinstance(func.value, ast.Call)
                    )
                    or isinstance(func, (ast.IfExp, ast.BoolOp, ast.Tuple, ast.List))
                    or (isinstance(func, ast.Subscript) and not is_triton_kernel_launch)
                )
                # `a.reshape(tensor)` where `a` is not a Tensor (custom
                # class). round-15 R-P0-2: also covers
                # `b.h.reshape(tensor)` / `objs[0].reshape(tensor)` —
                # any non-Tensor / non-module receiver.
                def _receiver_is_safe(rcv: ast.AST) -> bool:
                    if isinstance(rcv, ast.Name):
                        return (
                            rcv.id in {"tl", "triton", "torch", "torch_npu"}
                            or rcv.id in aliases
                            or rcv.id in tensors
                            or rcv.id in containers
                        )
                    if _expr_contains_tensor(rcv, tensors, containers):
                        return True
                    canon = _canonical(rcv)
                    if canon in {"triton.language"}:
                        return True
                    return False

                func_is_custom_method = (
                    isinstance(func, ast.Attribute)
                    and not _receiver_is_safe(func.value)
                )
                # codex round-15 NEW-P0 #1: class constructor / instance
                # __call__ with Tensor actual.
                func_is_class_init = (
                    isinstance(func, ast.Name) and func.id in class_defs
                )
                # codex round-17 NEW-P0 #3: decorator-replaced fn —
                # `@deco def h(z): ...; h(tensor)` may invoke decorator-
                # returned eager callable.
                def _resolves_to_decorated(name: str) -> bool:
                    seen_a: set[str] = set()
                    worklist = [name]
                    while worklist:
                        cur_a = worklist.pop()
                        if cur_a in seen_a:
                            continue
                        seen_a.add(cur_a)
                        if cur_a in decorated_unsafely:
                            return True
                        if cur_a in callable_aliases:
                            worklist.extend(callable_aliases[cur_a])
                    return False

                func_is_decorated_unsafely = (
                    isinstance(func, ast.Name)
                    and _resolves_to_decorated(func.id)
                )
                # Track instances of classes — `f = H()`. If a later
                # `f(tensor)` call resolves to such a name, fail-closed.
                func_is_instance_call = (
                    isinstance(func, ast.Name)
                    and (
                        func.id in instance_names_in_fn
                        or func.id in callable_instance_names_in_fn
                        or func.id in custom_method_aliases
                        or func.id in _global_custom_method_aliases
                    )
                )
                if func_is_indirect or func_is_custom_method or func_is_class_init or func_is_instance_call or func_is_decorated_unsafely:
                    for actual in sub.args + [kw.value for kw in sub.keywords]:
                        if _expr_contains_tensor(actual, tensors, containers):
                            report.add(
                                "G-callable-helper", "block",
                                "Tensor passed to indirect / instance-method / "
                                "custom-class helper — eager-eval bypass risk",
                                getattr(sub, "lineno", None),
                            )
                            break
                if fname in _TENSOR_COERCE_BUILTINS:
                    for arg in sub.args:
                        if _expr_contains_tensor(arg, tensors, containers):
                            report.add(
                                "G-tensor-coerce", "block",
                                f"host-side {fname}(...) — forces "
                                "Tensor → Python; use a triton kernel",
                                getattr(sub, "lineno", None),
                            )
                            break
                # codex round-20/21 NEW-P0 #2: IIFE outer lambda receives
                # a callable lambda — via positional, keyword, or default.
                if isinstance(sub.func, ast.Lambda):
                    outer_lam = sub.func
                    outer_formals_for_callable = (
                        list(outer_lam.args.posonlyargs) + list(outer_lam.args.args)
                    )
                    # codex round-22: resolve Name → Lambda (via
                    # lambda_aliases/fn_defs). kw-only defaults + **dict.
                    def _resolve_to_lambda(v: ast.AST) -> ast.Lambda | None:
                        if isinstance(v, ast.Lambda):
                            return v
                        if isinstance(v, ast.Name):
                            if v.id in lambda_aliases and lambda_aliases[v.id]:
                                return lambda_aliases[v.id][0]
                            if v.id in fn_defs:
                                fnd = fn_defs[v.id]
                                for inner in ast.walk(fnd):
                                    if isinstance(inner, ast.Return) and inner.value is not None:
                                        return ast.Lambda(
                                            args=fnd.args, body=inner.value
                                        )
                        # codex round-23: computed forms.
                        if isinstance(v, ast.Subscript):
                            container = v.value
                            if isinstance(container, (ast.Tuple, ast.List, ast.Set)):
                                for elt in container.elts:
                                    r = _resolve_to_lambda(elt)
                                    if r is not None:
                                        return r
                            elif isinstance(container, ast.Dict):
                                for val in container.values:
                                    r = _resolve_to_lambda(val)
                                    if r is not None:
                                        return r
                        if isinstance(v, ast.IfExp):
                            for branch in (v.body, v.orelse):
                                r = _resolve_to_lambda(branch)
                                if r is not None:
                                    return r
                        if isinstance(v, ast.BoolOp):
                            for val in v.values:
                                r = _resolve_to_lambda(val)
                                if r is not None:
                                    return r
                        return None

                    lambda_actuals: dict[str, ast.Lambda] = {}
                    pos_n = len(outer_formals_for_callable)
                    defaults_outer = list(outer_lam.args.defaults)
                    kw_defaults_outer = list(outer_lam.args.kw_defaults)
                    n_def = len(defaults_outer)
                    kwonly_formals_outer = list(outer_lam.args.kwonlyargs)
                    for i, fl in enumerate(outer_formals_for_callable):
                        if i < len(sub.args):
                            resolved = _resolve_to_lambda(sub.args[i])
                            if resolved is not None:
                                lambda_actuals[fl.arg] = resolved
                                continue
                        d_idx = i - (pos_n - n_def)
                        if 0 <= d_idx < n_def:
                            resolved = _resolve_to_lambda(defaults_outer[d_idx])
                            if resolved is not None:
                                lambda_actuals[fl.arg] = resolved
                    for i, fl in enumerate(kwonly_formals_outer):
                        if i < len(kw_defaults_outer) and kw_defaults_outer[i] is not None:
                            resolved = _resolve_to_lambda(kw_defaults_outer[i])
                            if resolved is not None:
                                lambda_actuals[fl.arg] = resolved
                    for kw in sub.keywords:
                        if kw.arg is not None:
                            resolved = _resolve_to_lambda(kw.value)
                            if resolved is not None:
                                lambda_actuals[kw.arg] = resolved
                        elif isinstance(kw.value, ast.Dict):
                            for k_node, v_node in zip(kw.value.keys, kw.value.values):
                                if isinstance(k_node, ast.Constant) and isinstance(k_node.value, str):
                                    resolved = _resolve_to_lambda(v_node)
                                    if resolved is not None:
                                        lambda_actuals[k_node.value] = resolved
                    for formal_name, actual_lam in lambda_actuals.items():
                        for body_call in ast.walk(outer_lam.body):
                            if (
                                isinstance(body_call, ast.Call)
                                and isinstance(body_call.func, ast.Name)
                                and body_call.func.id == formal_name
                            ):
                                # Build bound set for inner lambda.
                                bound_t = set(tensors)
                                bound_c = set(containers)
                                inner_formals = list(actual_lam.args.args)
                                for j, f2 in enumerate(inner_formals):
                                    if j < len(body_call.args):
                                        a2 = body_call.args[j]
                                        if _expr_contains_tensor(a2, tensors, containers):
                                            bound_t.add(f2.arg)
                                # Audit actual_lam body for forbidden ops.
                                for inner_n in ast.walk(actual_lam.body):
                                    if isinstance(inner_n, ast.BinOp):
                                        if any(
                                            _expr_contains_tensor(s, bound_t, bound_c)
                                            for s in (inner_n.left, inner_n.right)
                                        ):
                                            report.add(
                                                "G-nested-lambda-op", "block",
                                                "nested callable lambda performs eager Tensor op",
                                                getattr(sub, "lineno", None),
                                            )
                                            break
                                # Also: fold cpu string check for body call.
                                synthetic_call = ast.Call(
                                    func=actual_lam, args=body_call.args,
                                    keywords=body_call.keywords,
                                )
                                folded = _eval_str_const(synthetic_call)
                                if folded is not None and "cpu" in folded.lower():
                                    report.add(
                                        "D-cpu-transfer", "block",
                                        "nested callable lambda returns cpu string",
                                        getattr(sub, "lineno", None),
                                    )
                # codex round-13/14/15/20: `f = lambda; f(tensor)` —
                # round-20: full binding (kw/*args/defaults) + full op
                # categories (BinOp/Compare/UnaryOp/BoolOp/Subscript/coerce).
                if isinstance(sub.func, ast.Name) and sub.func.id in lambda_aliases:
                    for lam in lambda_aliases[sub.func.id]:
                        bound_t = set(tensors)
                        bound_c = set(containers)
                        pos_formals = list(lam.args.posonlyargs) + list(lam.args.args)
                        kwonly_formals = list(lam.args.kwonlyargs)
                        defaults = list(lam.args.defaults)
                        n_def = len(defaults)
                        for i, formal in enumerate(pos_formals):
                            actual_arg: ast.AST | None = None
                            if i < len(sub.args):
                                actual_arg = sub.args[i]
                            else:
                                d_idx = i - (len(pos_formals) - n_def)
                                if 0 <= d_idx < n_def:
                                    actual_arg = defaults[d_idx]
                            if actual_arg is not None:
                                if _expr_contains_tensor(actual_arg, tensors, containers):
                                    bound_t.add(formal.arg)
                                if isinstance(actual_arg, ast.Name) and actual_arg.id in containers:
                                    bound_c.add(formal.arg)
                        for kw in sub.keywords:
                            if kw.arg is None:
                                if _expr_contains_tensor(kw.value, tensors, containers):
                                    for f in pos_formals + kwonly_formals:
                                        bound_t.add(f.arg)
                                continue
                            for f in pos_formals + kwonly_formals:
                                if f.arg == kw.arg and _expr_contains_tensor(
                                    kw.value, tensors, containers
                                ):
                                    bound_t.add(f.arg)
                                    break
                        # Walk body for forbidden ops.
                        for inner_n in ast.walk(lam.body):
                            forbidden = False
                            if isinstance(inner_n, ast.BinOp):
                                forbidden = any(
                                    _expr_contains_tensor(s, bound_t, bound_c)
                                    for s in (inner_n.left, inner_n.right)
                                )
                            elif isinstance(inner_n, ast.Compare):
                                forbidden = _expr_contains_tensor(inner_n.left, bound_t, bound_c) or any(
                                    _expr_contains_tensor(c, bound_t, bound_c)
                                    for c in inner_n.comparators
                                )
                            elif isinstance(inner_n, ast.UnaryOp):
                                forbidden = _expr_contains_tensor(inner_n.operand, bound_t, bound_c)
                            elif isinstance(inner_n, ast.BoolOp):
                                forbidden = any(
                                    _expr_contains_tensor(v, bound_t, bound_c)
                                    for v in inner_n.values
                                )
                            elif isinstance(inner_n, ast.Subscript):
                                base = inner_n.value
                                if isinstance(base, ast.Attribute) and "." + base.attr in _PROP_ALLOW_SUFFIX:
                                    pass
                                elif isinstance(base, ast.Name) and base.id in bound_c:
                                    pass
                                elif _expr_contains_tensor(base, bound_t, bound_c):
                                    forbidden = True
                            if forbidden:
                                report.add(
                                    "G-lambda-alias-op", "block",
                                    "lambda alias performs eager Tensor op",
                                    getattr(sub, "lineno", None),
                                )
                                break
                    continue
                # codex round-11/12 P1: audit IIFE-lambda body under bound
                # formals. Recursively handles nested IIFE-lambdas
                # (`(lambda z: (lambda w: w+w)(z))(tensor)`).
                if isinstance(sub.func, ast.Lambda):
                    bound_tensors = set(tensors)
                    bound_containers = set(containers)
                    lam_node = sub.func
                    formals = list(lam_node.args.posonlyargs) + list(lam_node.args.args)
                    defaults = list(lam_node.args.defaults)
                    n_def = len(defaults)
                    # Positional + defaults.
                    for i, formal in enumerate(formals):
                        actual: ast.AST | None = None
                        if i < len(sub.args):
                            actual = sub.args[i]
                        else:
                            d_idx = i - (len(formals) - n_def)
                            if 0 <= d_idx < n_def:
                                actual = defaults[d_idx]
                        if actual is None:
                            continue
                        if _expr_contains_tensor(actual, tensors, containers):
                            bound_tensors.add(formal.arg)
                        if isinstance(actual, ast.Name) and actual.id in containers:
                            bound_containers.add(formal.arg)
                    # codex round-18/23: keyword + *args + **kwargs.
                    kwonly_formals = list(lam_node.args.kwonlyargs)
                    # codex round-23: kw-only defaults containing Tensor.
                    for i, formal in enumerate(kwonly_formals):
                        if i < len(lam_node.args.kw_defaults) and lam_node.args.kw_defaults[i] is not None:
                            if _expr_contains_tensor(
                                lam_node.args.kw_defaults[i], tensors, containers
                            ):
                                bound_tensors.add(formal.arg)
                    for kw in sub.keywords:
                        if kw.arg is None:
                            # **dict unpacking — taint all formals.
                            if _expr_contains_tensor(kw.value, tensors, containers):
                                for f in formals + kwonly_formals:
                                    bound_tensors.add(f.arg)
                            continue
                        for f in formals + kwonly_formals:
                            if f.arg == kw.arg:
                                if _expr_contains_tensor(kw.value, tensors, containers):
                                    bound_tensors.add(f.arg)
                                break
                    if lam_node.args.vararg is not None:
                        for extra in sub.args[len(formals):]:
                            if _expr_contains_tensor(extra, tensors, containers):
                                bound_tensors.add(lam_node.args.vararg.arg)
                                break
                    if lam_node.args.kwarg is not None:
                        for kw in sub.keywords:
                            if kw.arg is None and _expr_contains_tensor(
                                kw.value, tensors, containers
                            ):
                                bound_tensors.add(lam_node.args.kwarg.arg)
                                break
                    body = sub.func.body
                    forbidden = False
                    for inner in ast.walk(body):
                        if isinstance(inner, ast.BinOp):
                            for side in (inner.left, inner.right):
                                if _expr_contains_tensor(side, bound_tensors, bound_containers):
                                    forbidden = True
                                    break
                        elif isinstance(inner, ast.Compare):
                            if _expr_contains_tensor(inner.left, bound_tensors, bound_containers) or any(
                                _expr_contains_tensor(c, bound_tensors, bound_containers)
                                for c in inner.comparators
                            ):
                                forbidden = True
                        elif isinstance(inner, ast.UnaryOp):
                            if _expr_contains_tensor(inner.operand, bound_tensors, bound_containers):
                                forbidden = True
                        elif isinstance(inner, ast.BoolOp):
                            if any(
                                _expr_contains_tensor(v, bound_tensors, bound_containers)
                                for v in inner.values
                            ):
                                forbidden = True
                        elif isinstance(inner, ast.Subscript):
                            base = inner.value
                            if isinstance(base, ast.Attribute) and (
                                "." + base.attr in _PROP_ALLOW_SUFFIX
                            ):
                                pass
                            elif isinstance(base, ast.Name) and base.id in bound_containers:
                                pass
                            elif _expr_contains_tensor(base, bound_tensors, bound_containers):
                                forbidden = True
                        if forbidden:
                            break
                    if forbidden:
                        report.add(
                            "G-tensor-iife-lambda", "block",
                            "inline lambda performs eager Tensor op on its bound arg",
                            getattr(sub, "lineno", None),
                        )
                    # Codex round-12 R-P0-4: recurse into nested IIFE-lambdas
                    # within the body — `(lambda z: (lambda w: w+w)(z))(x)`.
                    for inner_call in ast.walk(body):
                        if isinstance(inner_call, ast.Call) and isinstance(
                            inner_call.func, ast.Lambda
                        ):
                            for actual in inner_call.args:
                                if _expr_contains_tensor(
                                    actual, bound_tensors, bound_containers
                                ):
                                    # block the inner IIFE too via the
                                    # same body-scan logic on the nested lambda.
                                    nbody = inner_call.func.body
                                    nbound = set(bound_tensors)
                                    for i, formal in enumerate(inner_call.func.args.args):
                                        if i < len(inner_call.args) and _expr_contains_tensor(
                                            inner_call.args[i],
                                            bound_tensors,
                                            bound_containers,
                                        ):
                                            nbound.add(formal.arg)
                                    nforbidden = False
                                    for ninner in ast.walk(nbody):
                                        if isinstance(ninner, ast.BinOp):
                                            for s2 in (ninner.left, ninner.right):
                                                if _expr_contains_tensor(s2, nbound, bound_containers):
                                                    nforbidden = True
                                                    break
                                        if nforbidden:
                                            break
                                    if nforbidden:
                                        report.add(
                                            "G-tensor-iife-lambda", "block",
                                            "nested inline lambda performs eager "
                                            "Tensor op",
                                            getattr(inner_call, "lineno", None),
                                        )

    # Pass 1b: build alias taint table — union of all observed bindings.
    # codex round-6 #4: any banned binding ⇒ alias tainted, regardless of
    # which binding `first-wins` aliases happened to pick.
    tainted_aliases: dict[str, str] = {}
    for local, qualifieds in all_bindings.items():
        # Conflict between distinct bindings is itself conservative-taint.
        if len(set(qualifieds)) > 1:
            tainted_aliases[local] = (
                f"conflicting bindings for {local!r}: {qualifieds!r}"
            )
            continue
        reason = _is_dotted_banned(qualifieds[0])
        if reason:
            tainted_aliases[local] = reason

    def _scan_rhs_for_banned(value: ast.AST) -> str | None:
        """Walk an RHS expression and return the first banned reason found.

        Covers banned dotted references AND captures of arbitrary `obj.<attr>`
        where `<attr>` is neither a method allowlist nor a property allowlist
        entry — that's almost certainly a callable being bound to a local
        name (codex round-5 NEW-P0 #5).
        """
        for sub in ast.walk(value):
            if isinstance(sub, (ast.Name, ast.Attribute)):
                reason = _is_dotted_banned(_canonical(sub))
                if reason:
                    return reason
            if isinstance(sub, ast.Attribute):
                suffix = "." + sub.attr
                base_canonical = _canonical(sub.value)
                # If the base resolves to a known module namespace, the
                # whole reference is already validated by `_is_dotted_banned`
                # / `_check_call_allow_either` elsewhere — skip the
                # bound-method capture rule here to avoid false positives
                # on legitimate `triton.cdiv`, `tl.exp`, `torch.empty`.
                from_known_module = (
                    base_canonical in {"tl", "triton", "torch", "torch_npu"}
                    or base_canonical.startswith(("tl.", "triton.", "torch.", "torch_npu."))
                )
                if (
                    not from_known_module
                    and suffix not in _CALL_ALLOW_METHOD_SUFFIX
                    and suffix not in _PROP_ALLOW_SUFFIX
                ):
                    return (
                        f"non-allowlisted method/attribute reference {suffix!r} "
                        f"(bound-call alias)"
                    )
        return None

    def _taint_targets(targets: list[ast.expr], reason: str) -> None:
        """Taint Name targets in arbitrary destructuring patterns.

        Covers `f = x`, `f, = (x,)`, `[f] = (x,)`, `*f, = (x,)`,
        `for f in ...`, `f: T = x`, function default `def f(x=...)`.
        """
        for tgt in targets:
            if isinstance(tgt, ast.Name):
                tainted_aliases[tgt.id] = reason
            elif isinstance(tgt, (ast.Tuple, ast.List)):
                _taint_targets(list(tgt.elts), reason)
            elif isinstance(tgt, ast.Starred):
                _taint_targets([tgt.value], reason)

    # (Pass-1 already pre-tainted the aliased imports; do not re-add.)

    # Pass 2: taint LHS of any binding form whose RHS contains a banned ref.
    # codex round-5 NEW-P0 #2: must cover AnnAssign / For / starred / defaults
    # in addition to plain Assign.
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            reason = _scan_rhs_for_banned(node.value)
            if reason:
                _taint_targets(list(node.targets), reason)
        elif isinstance(node, ast.AnnAssign):
            if node.value is not None:
                reason = _scan_rhs_for_banned(node.value)
                if reason and isinstance(node.target, ast.Name):
                    tainted_aliases[node.target.id] = reason
        elif isinstance(node, ast.For):
            reason = _scan_rhs_for_banned(node.iter)
            if reason:
                _taint_targets([node.target], reason)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            args = node.args
            all_defaults = list(args.defaults) + [
                d for d in args.kw_defaults if d is not None
            ]
            for default in all_defaults:
                reason = _scan_rhs_for_banned(default)
                if reason:
                    # Find the matching arg name. `defaults` line up with
                    # the *trailing* positional args; `kw_defaults` line
                    # up with `kwonlyargs`.
                    n_pos = len(args.posonlyargs) + len(args.args)
                    n_defaults = len(args.defaults)
                    pos_args = args.posonlyargs + args.args
                    for i, d in enumerate(args.defaults):
                        if d is default:
                            idx = n_pos - n_defaults + i
                            if 0 <= idx < len(pos_args):
                                tainted_aliases[pos_args[idx].arg] = reason
                    for i, d in enumerate(args.kw_defaults):
                        if d is default and i < len(args.kwonlyargs):
                            tainted_aliases[args.kwonlyargs[i].arg] = reason

    # Track positive AST signals
    triton_jit_found = False
    tl_load_found = False
    tl_store_found = False
    used_ops: set[str] = set()

    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            _check_imports(node, allowlist, report)
        elif isinstance(node, ast.Name):
            # Forbid bare references to dunder reflection trap-doors.
            if node.id in _BANNED_DUNDER_NAMES:
                report.add("D-dunder-deny", "block",
                           f"banned dunder name: {node.id}",
                           getattr(node, "lineno", None))
            # Bare-name reference to reflection-class builtin (`eval`,
            # `type`, etc.) is rejected even before it's called, so
            # `e = eval` triggers at the `eval` Name and at the alias.
            # `input` / `open` are NOT in this set — they are legitimate
            # variable names in ML code.
            if node.id in _BANNED_REFLECT_NAMES:
                report.add(
                    "D-banned-builtin", "block",
                    f"banned reflection name: {node.id}",
                    getattr(node, "lineno", None),
                )
            if node.id in tainted_aliases:
                report.add(
                    "A-tainted-name", "block",
                    f"name {node.id!r} is tainted: {tainted_aliases[node.id]}",
                    getattr(node, "lineno", None),
                )
        elif isinstance(node, ast.Attribute):
            # Use canonical (alias-resolved) name for attribute checks so
            # that `nn.functional.softmax` (with `from torch import nn`)
            # is checked as `torch.nn.functional.softmax`.
            dotted = _canonical(node) or _dotted_name(node)
            if dotted:
                _check_attribute_deny(dotted, report, node.lineno)
            if node.attr in _BANNED_DUNDER_NAMES:
                report.add("D-dunder-deny", "block",
                           f"banned dunder attribute: .{node.attr}",
                           node.lineno)
            # Codex round-6 #1: receiver-agnostic banned method names.
            # `inputs["x"].matmul(y)`, `Q.cpu()`, etc.
            if node.attr in _BANNED_METHOD_NAMES:
                report.add("D-banned-method", "block",
                           f"banned method attribute: .{node.attr} "
                           f"(unsafe regardless of receiver)",
                           node.lineno)
        elif isinstance(node, ast.Subscript):
            # Catch `__builtins__["eval"]` and `globals()["x"]`.
            base = _dotted_name(node.value)
            if base in _BANNED_DUNDER_NAMES or base in _BANNED_BUILTIN_CALLS:
                report.add(
                    "D-subscript-deny", "block",
                    f"banned subscript on {base!r}",
                    getattr(node, "lineno", None),
                )
        elif isinstance(node, ast.Call):
            # codex round-12 NEW-P0 #1: Attribute call on non-Name receiver
            # — `inputs["x"].add(inputs["y"])` / `(lambda:x)().cpu()` —
            # check the method name receiver-agnostically. If method is
            # in `_BANNED_METHOD_NAMES`, block. Otherwise, if method is
            # not in our allowlist AND receiver is a Tensor expression,
            # fail closed.
            if isinstance(node.func, ast.Attribute) and not isinstance(
                node.func.value, (ast.Name, ast.Attribute)
            ):
                method_name = node.func.attr
                if method_name in _BANNED_METHOD_NAMES:
                    report.add(
                        "D-banned-method", "block",
                        f"banned method on non-trivial receiver: .{method_name}",
                        node.lineno,
                    )
                # Receiver-agnostic check: if not allowlisted method, deny.
                if (
                    "." + method_name not in _CALL_ALLOW_METHOD_SUFFIX
                    and "." + method_name not in _PROP_ALLOW_SUFFIX
                ):
                    report.add(
                        "C-method-on-expr", "block",
                        f"method `.{method_name}` on non-trivial receiver — "
                        "not on method allowlist",
                        node.lineno,
                    )
            # Canonical (alias-resolved) name catches `matmul(...)` after
            # `from torch import matmul` as `torch.matmul`.
            fname = _canonical(node.func) or _dotted_name(node.func)
            if fname in _BANNED_BUILTIN_CALLS:
                report.add("D-banned-builtin", "block",
                           f"banned builtin call: {fname}", node.lineno)
            if fname in _BANNED_TL_CALLS:
                report.add("D-banned-tl", "block",
                           f"banned triton call: {fname}", node.lineno)
            # Codex round-7 #1: bare-call eager builtins force Tensor
            # evaluation regardless of receiver type.
            if fname in _BANNED_EAGER_BUILTINS:
                report.add(
                    "D-eager-builtin", "block",
                    f"banned eager builtin call: {fname}(...) "
                    "(forces Tensor evaluation on host)",
                    node.lineno,
                )
            # Codex round-7/8: `.to("cpu")` / `.to(device="cpu")` /
            # `torch.empty(..., device="cpu")` — explicit host transfer.
            # Round-8 R-P0-2: also resolve string-constant locals
            # (`d = "cpu"; x.to(d)`).
            def _value_is_cpu(expr: ast.AST) -> str | None:
                static = _eval_str_const(expr)
                if static is not None and "cpu" in static.lower():
                    return static
                if isinstance(expr, ast.Name) and expr.id in cpu_aliases:
                    return "cpu"                          # conservative
                return None

            # Any Tensor-returning method call OR a torch.* allocator can
            # potentially target CPU (codex round-9 NEW-P0 #3:
            # `x.new_empty(..., device='cpu')`).
            is_tensor_method_call = isinstance(node.func, ast.Attribute) and (
                "." + node.func.attr in _TENSOR_RETURNING_METHOD_SUFFIX
            )
            # codex round-23: also catch CPU transfer through aliases
            # — `move = x.to; move("cpu")` / `alloc = torch.empty;
            # alloc(..., device="cpu")`. Track names bound to tensor
            # methods / torch allocators in cpu_transfer_aliases.
            # cpu_transfer_aliases is a per-fn local set; this Call
            # handler runs in the main AST walk where it may not exist.
            is_cpu_transfer_alias = (
                isinstance(node.func, ast.Name)
                and node.func.id in _global_cpu_transfer_aliases
            )
            # codex round-25 NEW-P0: computed call target — IfExp/BoolOp/
            # Subscript whose branches all resolve to cpu-transfer.
            if not is_cpu_transfer_alias and isinstance(node.func, (ast.IfExp, ast.BoolOp, ast.Subscript, ast.Call)):
                if _rhs_is_cpu_transfer(node.func):
                    is_cpu_transfer_alias = True
            if (
                fname.endswith(".to") or fname.startswith("torch.")
                or fname == "to" or is_tensor_method_call
                or is_cpu_transfer_alias
            ):
                # codex round-23: positional args, including Starred
                # unpack of literal tuple `(*("cpu",))`.
                for arg in node.args:
                    if isinstance(arg, ast.Starred):
                        starred_container: ast.AST | None = arg.value
                        # codex round-27: resolve Name → bound Tuple/List.
                        if isinstance(starred_container, ast.Name):
                            starred_container = _global_tuple_literals.get(starred_container.id)
                        if isinstance(starred_container, (ast.Tuple, ast.List)):
                            for elt in starred_container.elts:
                                hit = _value_is_cpu(elt)
                                if hit:
                                    report.add(
                                        "D-cpu-transfer", "block",
                                        f"banned host-transfer in starred unpack {hit!r}",
                                        node.lineno,
                                    )
                        continue
                    hit = _value_is_cpu(arg)
                    if hit:
                        report.add(
                            "D-cpu-transfer", "block",
                            f"banned host-transfer arg {hit!r}",
                            node.lineno,
                        )
                for kw in node.keywords:
                    if kw.arg == "device":
                        hit = _value_is_cpu(kw.value)
                        if hit:
                            report.add(
                                "D-cpu-transfer", "block",
                                f"banned device kwarg device={hit!r}",
                                node.lineno,
                            )
                    elif kw.arg is None:
                        # codex round-23/25: **kw unpacking — accept either
                        # a direct Dict literal, a bound dict-literal Name,
                        # `dict(device="cpu")`, or `outer["k"]` nested.
                        resolved = _resolve_to_dict_literal(kw.value)
                        if resolved is not None:
                            for k_node, v_node in zip(resolved.keys, resolved.values):
                                if (
                                    isinstance(k_node, ast.Constant)
                                    and isinstance(k_node.value, str)
                                    and k_node.value == "device"
                                ):
                                    hit = _value_is_cpu(v_node)
                                    if hit:
                                        report.add(
                                            "D-cpu-transfer", "block",
                                            f"banned device kwarg via **dict {hit!r}",
                                            node.lineno,
                                        )
            # codex round-28/29: Attribute call to class method whose formal
            # flows to a CPU-sensitive position. (Uses _audit_helper_call
            # defined below for vararg/kwarg consistency.)
            _method_call_match = (
                isinstance(node.func, ast.Attribute)
                and node.func.attr in _cpu_tainted_method_formals
            )

            # codex round-27/29: helper fn propagation. Call to a fn
            # (or alias) whose formal flows to a CPU-sensitive position.
            # codex round-29: vararg/kwarg formal handling.
            def _audit_helper_call(fn_def_node, tainted: set[str], pos_formals: list[str], node, label: str) -> None:
                vararg_name = fn_def_node.args.vararg.arg if fn_def_node and fn_def_node.args.vararg else None
                kwarg_name = fn_def_node.args.kwarg.arg if fn_def_node and fn_def_node.args.kwarg else None
                for i, actual in enumerate(node.args):
                    if isinstance(actual, ast.Starred):
                        continue
                    if i < len(pos_formals):
                        if pos_formals[i] in tainted:
                            hit = _value_is_cpu(actual)
                            if hit:
                                report.add(
                                    "D-cpu-transfer", "block",
                                    f"cpu string flows into {label} formal {pos_formals[i]!r}",
                                    node.lineno,
                                )
                    else:
                        if vararg_name and vararg_name in tainted:
                            hit = _value_is_cpu(actual)
                            if hit:
                                report.add(
                                    "D-cpu-transfer", "block",
                                    f"cpu string flows into {label} *{vararg_name}",
                                    node.lineno,
                                )
                for kw in node.keywords:
                    if kw.arg is not None:
                        if kw.arg in tainted:
                            hit = _value_is_cpu(kw.value)
                            if hit:
                                report.add(
                                    "D-cpu-transfer", "block",
                                    f"cpu string flows into {label} kwarg {kw.arg!r}",
                                    node.lineno,
                                )
                        elif kwarg_name and kwarg_name in tainted:
                            hit = _value_is_cpu(kw.value)
                            if hit:
                                report.add(
                                    "D-cpu-transfer", "block",
                                    f"cpu string flows via {kw.arg}=… into {label} **{kwarg_name}",
                                    node.lineno,
                                )
                    elif kw.arg is None:
                        d = _resolve_to_dict_literal(kw.value)
                        if d is not None:
                            for k_node, v_node in zip(d.keys, d.values):
                                if isinstance(k_node, ast.Constant):
                                    key_name = k_node.value
                                    if key_name in tainted or (kwarg_name and kwarg_name in tainted):
                                        hit = _value_is_cpu(v_node)
                                        if hit:
                                            report.add(
                                                "D-cpu-transfer", "block",
                                                f"cpu string flows into {label} **kw {key_name!r}",
                                                node.lineno,
                                            )

            def _find_fn_def_by_name(name: str):
                for nd in ast.walk(tree):
                    if isinstance(nd, (ast.FunctionDef, ast.AsyncFunctionDef)) and nd.name == name:
                        return nd
                return None

            def _find_method_def_by_name(name: str):
                for nd in ast.walk(tree):
                    if isinstance(nd, ast.ClassDef):
                        for m in nd.body:
                            if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef)) and m.name == name:
                                return m
                return None

            # codex round-28/29: closure helper — `maker()("cpu")` or
            # `factory()("cpu")` where factory→maker→inner; recursively
            # unwrap fn returns to find a Name in _cpu_tainted_formals.
            def _resolve_closure_returns(c: ast.Call, depth: int = 0) -> list[str]:
                """Walk Call(Call(...)) returning tainted-fn names reachable."""
                if depth > 10:
                    return []
                if not isinstance(c.func, ast.Name):
                    if isinstance(c.func, ast.Call):
                        return _resolve_closure_returns(c.func, depth + 1)
                    return []
                out: list[str] = []
                for r in _global_fn_returns.get(c.func.id, []):
                    if isinstance(r, ast.Name):
                        nm = _resolve_fn_alias(r.id)
                        if nm in _cpu_tainted_formals:
                            out.append(nm)
                        # Maker returns alias `f = inner; return f`.
                        elif r.id in _global_fn_returns:
                            # Treat r as another callable returning something.
                            pass
                    elif isinstance(r, ast.Call):
                        out.extend(_resolve_closure_returns(r, depth + 1))
                return out

            if isinstance(node.func, ast.Call):
                for tainted_fn_name in _resolve_closure_returns(node.func):
                    callee_pos = _fn_pos_formals.get(tainted_fn_name, [])
                    callee_tainted = _cpu_tainted_formals[tainted_fn_name]
                    fn_def = _find_fn_def_by_name(tainted_fn_name)
                    _audit_helper_call(fn_def, callee_tainted, callee_pos, node, f"closure-returned {tainted_fn_name}()")

            if isinstance(node.func, ast.Name) and (
                node.func.id in _cpu_tainted_formals
                or _resolve_fn_alias(node.func.id) in _cpu_tainted_formals
            ):
                fn_name = _resolve_fn_alias(node.func.id)
                tainted = _cpu_tainted_formals[fn_name]
                pos_formals = _fn_pos_formals.get(fn_name, [])
                fn_def = _find_fn_def_by_name(fn_name)
                _audit_helper_call(fn_def, tainted, pos_formals, node, f"{fn_name}()")

            if _method_call_match:
                method_name = node.func.attr
                tainted_m = _cpu_tainted_method_formals[method_name]
                pos_m = _method_pos_formals.get(method_name, [])
                m_def = _find_method_def_by_name(method_name)
                _audit_helper_call(m_def, tainted_m, pos_m, node, f"method {method_name}()")

            # codex round-27: IIFE Lambda with cpu-tainted formal —
            # `(lambda x, **kw: x.to(**kw))(inputs["x"], device="cpu")`.
            # Also handles `pick()(5, device="cpu")` where `pick` returns
            # a Lambda — we resolve via `_global_fn_returns`.
            _iife_lambda: ast.Lambda | None = None
            if isinstance(node.func, ast.Lambda):
                _iife_lambda = node.func
            elif isinstance(node.func, ast.Call) and isinstance(node.func.func, ast.Name):
                rets = _global_fn_returns.get(node.func.func.id, [])
                for r in rets:
                    if isinstance(r, ast.Lambda):
                        _iife_lambda = r
                        break
            if _iife_lambda is not None:
                lam = _iife_lambda
                pos_formals_lam = [a.arg for a in (list(lam.args.posonlyargs) + list(lam.args.args))]
                all_formals_lam = set(pos_formals_lam) | {a.arg for a in lam.args.kwonlyargs}
                if lam.args.vararg:
                    all_formals_lam.add(lam.args.vararg.arg)
                if lam.args.kwarg:
                    all_formals_lam.add(lam.args.kwarg.arg)
                sensitive_lam: set[str] = set()
                for sub in ast.walk(lam.body) if isinstance(lam.body, ast.AST) else []:
                    if isinstance(sub, ast.Call):
                        for kw in sub.keywords:
                            if kw.arg == "device" and isinstance(kw.value, ast.Name) and kw.value.id in all_formals_lam:
                                sensitive_lam.add(kw.value.id)
                            if kw.arg is None and isinstance(kw.value, ast.Name) and kw.value.id in all_formals_lam:
                                sensitive_lam.add(kw.value.id)
                        if isinstance(sub.func, ast.Attribute) and sub.func.attr == "to":
                            for a in sub.args:
                                if isinstance(a, ast.Name) and a.id in all_formals_lam:
                                    sensitive_lam.add(a.id)
                        for a in sub.args:
                            if isinstance(a, ast.Starred) and isinstance(a.value, ast.Name) and a.value.id in all_formals_lam:
                                sensitive_lam.add(a.value.id)
                        # codex round-29: call to tainted helper fn with
                        # formal as actual at a tainted slot.
                        if isinstance(sub.func, ast.Name):
                            callee = _resolve_fn_alias(sub.func.id)
                            if callee in _cpu_tainted_formals:
                                callee_pos = _fn_pos_formals.get(callee, [])
                                callee_t = _cpu_tainted_formals[callee]
                                for i, a in enumerate(sub.args):
                                    if i < len(callee_pos) and callee_pos[i] in callee_t:
                                        if isinstance(a, ast.Name) and a.id in all_formals_lam:
                                            sensitive_lam.add(a.id)
                                for kw in sub.keywords:
                                    if kw.arg in callee_t and isinstance(kw.value, ast.Name) and kw.value.id in all_formals_lam:
                                        sensitive_lam.add(kw.value.id)
                        # Call to tainted method.
                        if isinstance(sub.func, ast.Attribute) and sub.func.attr in _cpu_tainted_method_formals:
                            callee_pos = _method_pos_formals.get(sub.func.attr, [])
                            callee_t = _cpu_tainted_method_formals[sub.func.attr]
                            for i, a in enumerate(sub.args):
                                if i < len(callee_pos) and callee_pos[i] in callee_t:
                                    if isinstance(a, ast.Name) and a.id in all_formals_lam:
                                        sensitive_lam.add(a.id)
                            for kw in sub.keywords:
                                if kw.arg in callee_t and isinstance(kw.value, ast.Name) and kw.value.id in all_formals_lam:
                                    sensitive_lam.add(kw.value.id)
                if sensitive_lam:
                    for i, actual in enumerate(node.args):
                        if i < len(pos_formals_lam) and pos_formals_lam[i] in sensitive_lam:
                            hit = _value_is_cpu(actual)
                            if hit:
                                report.add(
                                    "D-cpu-transfer", "block",
                                    f"cpu string flows into IIFE lambda formal {pos_formals_lam[i]!r}",
                                    node.lineno,
                                )
                    for kw in node.keywords:
                        if kw.arg is not None and kw.arg in sensitive_lam:
                            hit = _value_is_cpu(kw.value)
                            if hit:
                                report.add(
                                    "D-cpu-transfer", "block",
                                    f"cpu string flows into IIFE lambda kwarg {kw.arg!r}",
                                    node.lineno,
                                )
                        elif kw.arg is None:
                            # **kw passing to lambda **kw formal
                            # Look at the dict and check if any key matches sensitive.
                            d = _resolve_to_dict_literal(kw.value)
                            if d is not None:
                                for k_node, v_node in zip(d.keys, d.values):
                                    if isinstance(k_node, ast.Constant):
                                        hit = _value_is_cpu(v_node)
                                        if hit:
                                            report.add(
                                                "D-cpu-transfer", "block",
                                                f"cpu string flows via **kw into IIFE lambda",
                                                node.lineno,
                                            )
                    # If lambda has **kw formal AND keywords contain device="cpu",
                    # treat as flow.
                    if lam.args.kwarg and lam.args.kwarg.arg in sensitive_lam:
                        for kw in node.keywords:
                            if kw.arg is not None:
                                hit = _value_is_cpu(kw.value)
                                if hit:
                                    report.add(
                                        "D-cpu-transfer", "block",
                                        f"cpu string flows via {kw.arg}=… into IIFE **{lam.args.kwarg.arg}",
                                        node.lineno,
                                    )
                    # codex round-28: IIFE Lambda positional / kw-only
                    # defaults that themselves are "cpu" — formal binds
                    # to the default when actual is omitted.
                    pos_formals_with_defaults = pos_formals_lam[
                        len(pos_formals_lam) - len(lam.args.defaults):
                    ]
                    for i, formal_name in enumerate(pos_formals_with_defaults):
                        if formal_name in sensitive_lam:
                            actual_pos_idx = pos_formals_lam.index(formal_name)
                            actual_supplied = actual_pos_idx < len(node.args) or any(
                                kw.arg == formal_name for kw in node.keywords
                            )
                            if not actual_supplied:
                                hit = _value_is_cpu(lam.args.defaults[i])
                                if hit:
                                    report.add(
                                        "D-cpu-transfer", "block",
                                        f"cpu string flows from default of {formal_name!r} in IIFE lambda",
                                        node.lineno,
                                    )
                    for i, formal in enumerate(lam.args.kwonlyargs):
                        if formal.arg not in sensitive_lam:
                            continue
                        if i >= len(lam.args.kw_defaults) or lam.args.kw_defaults[i] is None:
                            continue
                        actual_supplied = any(kw.arg == formal.arg for kw in node.keywords)
                        if actual_supplied:
                            continue
                        hit = _value_is_cpu(lam.args.kw_defaults[i])
                        if hit:
                            report.add(
                                "D-cpu-transfer", "block",
                                f"cpu string flows from kw-only default {formal.arg!r} in IIFE lambda",
                                node.lineno,
                            )

            # Codex round-5 #1 + round-6 #2: any non-trivial Call.func
            # (anything beyond a plain Name or simple Attribute) is
            # walked for banned references. Covers Subscript / Tuple /
            # List / IfExp / BoolOp / Lambda / Call nesting.
            if not isinstance(node.func, (ast.Name, ast.Attribute)):
                reason = _scan_rhs_for_banned(node.func)
                if reason:
                    report.add(
                        "D-indirect-call", "block",
                        f"indirect call target hides banned reference: {reason}",
                        node.lineno,
                    )
            # Codex round-6 #3: deep-walk Call args + kwargs so that
            # `(lambda fs: fs[0](...))((torch.matmul,))` is caught.
            for arg in list(node.args) + [kw.value for kw in node.keywords]:
                reason = _scan_rhs_for_banned(arg)
                if reason:
                    report.add(
                        "D-banned-arg", "block",
                        f"call argument is banned: {reason}",
                        node.lineno,
                    )
            # codex round-12 NEW-P0 #2: instance method / callable-object
            # hides eager helper. `H().h(x)` / `H()(x)` — Call.func is
            # neither plain Name nor plain Attribute(Name). Fail closed
            # when any actual arg contains a Tensor expression we can see
            # in this scope. We check this by deferring the tensor-flow
            # audit's _expr_contains_tensor — but at parse time we don't
            # have the per-function tensors set, so plant a placeholder
            # finding here that the per-function audit can re-validate.
            if isinstance(node.func, ast.Call):
                # Direct callable-object pattern: `H()(...)` — block if
                # any arg is even remotely callable-tensor-ish; the
                # per-function audit upgrades this below.
                pass
            if isinstance(node.func, ast.Attribute) and isinstance(
                node.func.value, ast.Call
            ):
                # `H().h(x)` — same.
                pass
            raw_fname = _dotted_name(node.func)
            if fname:
                # Pass both canonical and raw so Layer-C allowlist (which
                # is written in `tl.X` form) accepts `triton.language.X`
                # callers without losing canonical-name ban checks.
                _check_call_allow_either(
                    fname, raw_fname, report, node.lineno
                )
                if raw_fname.startswith("tl.") or fname.startswith("triton.language."):
                    used_ops.add(raw_fname or fname)
                if raw_fname == "tl.load" or fname == "triton.language.load":
                    tl_load_found = True
                if raw_fname == "tl.store" or fname == "triton.language.store":
                    tl_store_found = True
        elif isinstance(node, ast.FunctionDef):
            for deco in node.decorator_list:
                name = _dotted_name(deco) or _dotted_name(
                    deco.func if isinstance(deco, ast.Call) else None
                )
                if name in {"triton.jit", "triton.autotune"}:
                    triton_jit_found = True

    # Layer D: positive AST presence (must have a JIT'd kernel doing real I/O)
    if not triton_jit_found:
        report.add("D-positive-AST", "block", "no @triton.jit kernel found")
    if not tl_load_found:
        report.add("D-positive-AST", "block", "no tl.load found")
    if not tl_store_found:
        report.add("D-positive-AST", "block", "no tl.store found")

    # Layer F+: required ops
    for check_name in required_checks or []:
        ops = REQUIRED_OPS.get(check_name)
        if not ops:
            continue
        for needed in ops:
            if not any(op in used_ops for op in needed):
                report.add(
                    "F-required-op",
                    "block",
                    f"required by {check_name}: one of {needed}",
                )

    return report


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def _dotted_name(node: ast.AST | None) -> str:
    """Best-effort dotted name for a Call/Attribute/Name node."""
    if node is None:
        return ""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = _dotted_name(node.value)
        return f"{prefix}.{node.attr}" if prefix else node.attr
    return ""


def _check_imports(
    node: ast.Import | ast.ImportFrom, allowlist: set[str], report: CheckReport
) -> None:
    line = node.lineno
    if isinstance(node, ast.Import):
        names = [a.name for a in node.names]
    else:
        names = [node.module or ""]
        # also flag `from torch.utils import cpp_extension`
        if node.module:
            for a in node.names:
                names.append(f"{node.module}.{a.name}")

    for name in names:
        if not name:
            continue
        if name in _DENYLIST_IMPORT_EXACT:
            report.add("A-banned-import", "block",
                       f"banned import: {name}", line)
            continue
        if any(name.startswith(p) for p in _DENYLIST_IMPORT_PREFIX):
            report.add("A-banned-import", "block",
                       f"banned import (prefix): {name}", line)
            continue
        base = name.split(".")[0]
        if name not in allowlist and base not in allowlist:
            report.add("A-non-allowlist", "block",
                       f"non-allowlisted import: {name}", line)


def _check_attribute_deny(
    dotted: str, report: CheckReport, line: int | None
) -> None:
    if any(dotted.startswith(p) for p in _ATTR_DENY_PREFIXES):
        report.add("D-attr-deny", "block",
                   f"banned attribute path: {dotted}", line)
        return
    for suf in _ATTR_DENY_METHOD_SUFFIXES:
        if dotted.endswith(suf):
            report.add("D-attr-deny", "block",
                       f"banned tensor method: {dotted}", line)
            return


def _check_call_allow(
    fname: str, report: CheckReport, line: int | None
) -> None:
    """Single-name variant. Prefer `_check_call_allow_either` so raw and
    canonical forms both get a chance against the allowlist."""
    _check_call_allow_either(fname, fname, report, line)


def _check_call_allow_either(
    canonical: str,
    raw: str,
    report: CheckReport,
    line: int | None,
) -> None:
    """Layer C: dotted call must match allowlist under EITHER name.

    Free identifiers (bare names) — only allow if the canonical resolves
    to a free identifier too (so `matmul()` after `from torch import matmul`
    is rejected because its canonical is `torch.matmul`).
    """
    canonical = canonical or raw
    raw = raw or canonical
    # Bare name (no dot) is OK only if neither form is dotted.
    if "." not in canonical and "." not in raw:
        return
    # Allowlisted under either form.
    if canonical in _CALL_ALLOW_EXACT or raw in _CALL_ALLOW_EXACT:
        return
    for suf in _CALL_ALLOW_METHOD_SUFFIX:
        if canonical.endswith(suf) or raw.endswith(suf):
            return
    # Triton dotted calls — must be on the explicit allowlist.
    for f in (canonical, raw):
        if f.startswith("triton.") or f.startswith("tl."):
            report.add(
                "C-tl-deny", "block",
                f"unknown triton call (not on Layer C allowlist): {f}",
                line,
            )
            return
    report.add(
        "C-call-deny", "block",
        f"call not in Layer C allowlist: canonical={canonical!r} raw={raw!r}",
        line,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Layer E — Runtime audit hook (specs/ast_denylist.md §Layer E, DESIGN §6.4)
# ─────────────────────────────────────────────────────────────────────────────
# The static layers (A–D/F) scan the candidate source AST. Layer E is the
# *runtime fallback* line: a `sys.addaudithook` callback that traps
# dynamically-constructed dangerous operations the static pass cannot see
# (e.g. an `exec` reached via an alias, a `subprocess` spawned through a
# data-dependent path). The verified runner installs it inside the
# isolated candidate child *before* candidate code runs.

# Audit events that abort the candidate process when raised. CPython emits
# these for the corresponding dangerous operations regardless of how the
# call was syntactically spelled (alias, getattr, exec'd string …), so the
# hook catches what the static AST scan misses.
_AUDIT_BLOCKED_EVENTS = frozenset({
    "exec",                 # exec()/eval() of a code object
    "compile",              # compile() of a source string
    "subprocess.Popen",     # any subprocess spawn
    "os.system",            # shell-out
    "os.exec",              # exec-family
    "os.spawn",             # spawn-family
    "os.fork",              # fork
    "ctypes.dlopen",        # native library load
    "ctypes.dlsym",         # native symbol resolution
    "ctypes.call_function", # raw native call
    "import",               # __import__ — see _AUDIT_IMPORT_DENY below
})

# Modules a runtime `import` audit event must refuse. The Layer A static
# import denylist already rejects these in source; this is the runtime
# net for an import reached dynamically (importlib through an alias, an
# `__import__` built from a data-dependent name).
_AUDIT_IMPORT_DENY = frozenset({
    "ctypes", "cffi", "subprocess", "importlib",
    "oracle_pytorch", "tasks",
})


class RuntimeAuditViolation(RuntimeError):
    """Raised by the Layer E audit hook when a candidate trips a blocked
    runtime operation. Uncaught, it aborts the isolated candidate process;
    the runner records the candidate as failed."""


def _runtime_audit_hook(event: str, audit_args: tuple) -> None:
    """`sys.addaudithook` callback — the Layer E runtime fallback.

    Raises :class:`RuntimeAuditViolation` for a blocked audit event. A
    raising audit hook aborts the offending operation (and, uncaught, the
    process) — this is the intended fail-closed behaviour inside the
    isolated candidate child. Never installed in the trusted parent.
    """
    if event in _AUDIT_BLOCKED_EVENTS and event != "import":
        raise RuntimeAuditViolation(
            f"Layer E: blocked runtime operation {event!r}"
        )
    if event == "import":
        # audit_args for "import" is
        # (module, filename, sys.path, sys.meta_path, sys.path_hooks).
        module = audit_args[0] if audit_args else ""
        root = str(module).split(".", 1)[0]
        if root in _AUDIT_IMPORT_DENY:
            raise RuntimeAuditViolation(
                f"Layer E: blocked runtime import {module!r}"
            )


# Module-level guard so a process installs the hook at most once.
_AUDIT_HOOK_INSTALLED = False


def install_runtime_audit_hook() -> bool:
    """Install the Layer E `sys.addaudithook` runtime-defense hook.

    Called by `process_isolation`'s candidate child at start-up, *before*
    any candidate code loads. Idempotent: a second call is a no-op and
    returns ``False``; the first install returns ``True``. Audit hooks
    cannot be removed once installed (CPython has no `removeaudithook`),
    so the idempotency guard avoids stacking duplicates across re-entry.

    Returns ``True`` if this call installed the hook, ``False`` if it was
    already installed.
    """
    global _AUDIT_HOOK_INSTALLED
    if _AUDIT_HOOK_INSTALLED:
        return False
    sys.addaudithook(_runtime_audit_hook)
    _AUDIT_HOOK_INSTALLED = True
    return True
