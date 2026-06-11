# AST 拒绝列表 / 允许列表

> English: [../../reference/ast_denylist.md](../../reference/ast_denylist.md)

## 策略

**正向 AST 节点允许列表** + **动态 AttrLookup 拒绝列表** + **运行时 import 钩子**。

这比单独使用黑名单更严格——任何未被显式允许的内容都会被拒绝。

## A 层 —— Import 拒绝列表

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
    "oracle_pytorch", "ascendbench", "tasks",
}
```

## B 层 —— AST 节点允许列表

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
    # NOTE: ast.MatMult (= `@`) is REMOVED (conflicts with denylist)
    ast.USub, ast.UAdd, ast.Not, ast.Invert,
    ast.Eq, ast.NotEq, ast.Lt, ast.Gt, ast.LtE, ast.GtE, ast.In, ast.NotIn, ast.Is, ast.IsNot,
    # Calls / attributes (validated against NAME ALLOWLIST below)
    ast.Attribute, ast.Call,
    ast.AugAssign, ast.Assign, ast.AnnAssign,
}
```

任何不在此集合中的 AST 节点类型 → 拒绝。

## C 层 —— 调用 NAME 允许列表(点分路径)

仅允许:
- `torch.empty`、`torch.zeros`、`torch.ones`、`torch.full`、`torch.empty_like`、`torch.tensor`、`torch.as_tensor`
- `Tensor.reshape`、`Tensor.view`、`Tensor.contiguous`、`Tensor.transpose`、`Tensor.permute`、`Tensor.unsqueeze`、`Tensor.squeeze`、`Tensor.size`、`Tensor.shape`、`Tensor.stride`、`Tensor.numel`、`Tensor.dim`
- `Tensor.to`(仅限 dtype/device 转换)
- `triton.cdiv`、`triton.next_power_of_2`、`triton.heuristics`、`triton.autotune`、`triton.jit`、`triton.Config`
- `tl.load`、`tl.store`、`tl.dot`、`tl.sum`、`tl.max`、`tl.min`、`tl.exp`、`tl.log`、`tl.where`、`tl.arange`、`tl.zeros`、`tl.full`、`tl.cdiv`、`tl.program_id`、`tl.constexpr`、`tl.static_range`

## D 层 —— 属性查找拒绝列表模式

始终拒绝(即使通过 getattr 访问):
- `torch.ops.*`、`torch.classes.*`、`torch.fx.*`、`torch.compile`
- `torch.nn.functional.*`
- `Tensor.cpu`、`Tensor.numpy`、`Tensor.item`、`Tensor.tolist`、`Tensor.detach`
- `torch.cuda.stream`、`torch.npu.stream`、`torch_npu.npu.stream`
- `sgl_kernel_npu.*`、`vllm_ascend.npu_*`、`torch_npu.npu_*`

## E 层 —— 运行时审计钩子

```python
# sys.audit hook (Python 3.8+)
def audit_hook(event, args):
    if event in {"exec", "compile", "subprocess.*", "socket.*", "open"}:
        log_violation(event, args)
        raise ASTViolationError(event)

sys.addaudithook(audit_hook)
```

## F 层 —— 字符串字面量拒绝列表

拒绝任何包含以下子串的源代码(大小写不敏感的正则匹配):
- `oracle`、`golden`、`hidden_shape`、`hidden_set`、`answer`、`cheat`

## 校验器工作流

```
1. tab static-check <path-to-solution.py>
2. AST parse → walk all nodes
3. Check each node type against AST_ALLOW (Layer B)
4. For Call/Attribute nodes, check dotted path against Layers C/D
5. Check imports against Layer A
6. Regex scan source for Layer F patterns
7. (At runtime) Install Layer E audit hook
8. Any violation → reject + return precise location
```
