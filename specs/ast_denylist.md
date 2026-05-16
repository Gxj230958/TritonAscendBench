# AST Denylist / Allowlist

> v0 — per v4.13 §5b.3 + R11-P1-8 + R12-P1-6 + R14-P1-5

## Strategy

**Positive AST node allowlist** + **dynamic AttrLookup denylist** + **runtime import hook**.

This is stricter than blacklist alone — anything not explicitly allowed is rejected.

## Layer A — Import Denylist

```python
DENYLIST_IMPORT = {
    # Code execution
    "exec", "eval", "compile", "__import__",
    "importlib", "importlib.util", "importlib.import_module",

    # Native code loading
    "ctypes", "ctypes.CDLL", "cffi",
    "torch.utils.cpp_extension", "torch.utils.cpp_extension.load",
    "torch.utils.cpp_extension.load_inline",

    # IPC / Network
    "subprocess", "socket", "requests", "urllib", "http", "pickle",
    "threading", "multiprocessing", "asyncio", "concurrent.futures",

    # FS / IO / mmap
    "os", "os.path", "pathlib", "mmap", "fcntl", "shutil", "tempfile",
    "glob", "sys.path", "sys.modules", "sys.meta_path",

    # Timing / profiling manipulation
    "time", "datetime", "monotonic",
    "torch.cuda.Event", "torch.npu.Event", "torch_npu.npu.Event",
    "torch.profiler", "torch_npu.profiler",

    # Oracle / harness imports
    "oracle_pytorch", "triton_ascend_bench", "tasks",
}
```

## Layer B — AST Node Allowlist (R14-P1-5)

```python
AST_ALLOW = {
    # Module structure
    ast.Module, ast.Expr, ast.Expression,
    ast.Import, ast.ImportFrom, ast.alias, ast.keyword,
    # Function / class
    ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Return,
    ast.arguments, ast.arg,
    # Control flow
    ast.If, ast.For, ast.While, ast.Break, ast.Continue, ast.Pass,
    ast.Try, ast.ExceptHandler,
    # Expression / literal
    ast.Constant, ast.Name, ast.Load, ast.Store, ast.Del,
    ast.BinOp, ast.UnaryOp, ast.BoolOp, ast.Compare, ast.Lambda,
    ast.IfExp, ast.Subscript, ast.Slice, ast.Tuple, ast.List, ast.Dict, ast.Set,
    # Operators
    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow,
    ast.LShift, ast.RShift, ast.BitOr, ast.BitXor, ast.BitAnd,
    # NOTE: ast.MatMult (= `@`) is REMOVED per R14-P1-5 (conflicts with denylist)
    ast.USub, ast.UAdd, ast.Not, ast.Invert,
    ast.Eq, ast.NotEq, ast.Lt, ast.Gt, ast.LtE, ast.GtE, ast.In, ast.NotIn, ast.Is, ast.IsNot,
    # Calls / attributes (validated against NAME ALLOWLIST below)
    ast.Attribute, ast.Call,
    ast.AugAssign, ast.Assign, ast.AnnAssign,
}
```

Any AST node type not in this set → reject.

## Layer C — Call NAME Allowlist (dotted path)

Allowed only:
- `torch.empty`, `torch.zeros`, `torch.ones`, `torch.full`, `torch.empty_like`, `torch.tensor`, `torch.as_tensor`
- `Tensor.reshape`, `Tensor.view`, `Tensor.contiguous`, `Tensor.transpose`, `Tensor.permute`, `Tensor.unsqueeze`, `Tensor.squeeze`, `Tensor.size`, `Tensor.shape`, `Tensor.stride`, `Tensor.numel`, `Tensor.dim`
- `Tensor.to` (dtype/device cast only)
- `triton.cdiv`, `triton.next_power_of_2`, `triton.heuristics`, `triton.autotune`, `triton.jit`, `triton.Config`
- `tl.load`, `tl.store`, `tl.dot`, `tl.sum`, `tl.max`, `tl.min`, `tl.exp`, `tl.log`, `tl.where`, `tl.arange`, `tl.zeros`, `tl.full`, `tl.cdiv`, `tl.program_id`, `tl.constexpr`, `tl.static_range`

## Layer D — Attribute Lookup Denylist Patterns

Always reject (even if reached via getattr):
- `torch.ops.*`, `torch.classes.*`, `torch.fx.*`, `torch.compile`
- `torch.nn.functional.*`
- `Tensor.cpu`, `Tensor.numpy`, `Tensor.item`, `Tensor.tolist`, `Tensor.detach`
- `torch.cuda.stream`, `torch.npu.stream`, `torch_npu.npu.stream`
- `sgl_kernel_npu.*`, `vllm_ascend.npu_*`, `torch_npu.npu_*`

## Layer E — Runtime Audit Hook

```python
# sys.audit hook (Python 3.8+)
def audit_hook(event, args):
    if event in {"exec", "compile", "subprocess.*", "socket.*", "open"}:
        log_violation(event, args)
        raise ASTViolationError(event)

sys.addaudithook(audit_hook)
```

## Layer F — String Literal Denylist

Reject any source code containing these substrings (case-insensitive regex):
- `oracle`, `golden`, `hidden_shape`, `hidden_set`, `answer`, `cheat`

## Validator Workflow

```
1. tab validate-solution <path-to-solution.py>
2. AST parse → walk all nodes
3. Check each node type against AST_ALLOW (Layer B)
4. For Call/Attribute nodes, check dotted path against Layers C/D
5. Check imports against Layer A
6. Regex scan source for Layer F patterns
7. (At runtime) Install Layer E audit hook
8. Any violation → reject + return precise location
```
