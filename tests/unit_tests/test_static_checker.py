"""Unit tests for ascendbench.integrity.static_checker."""
from __future__ import annotations

import textwrap

import pytest

from ascendbench.integrity.static_checker import check_solution


def _src(s: str) -> str:
    return textwrap.dedent(s).lstrip()


VALID_BASE = _src(
    """
    import torch
    import triton
    import triton.language as tl

    @triton.jit
    def my_kernel(x_ptr, y_ptr, n, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offs < n
        x = tl.load(x_ptr + offs, mask=mask, other=0.0)
        tl.store(y_ptr + offs, x, mask=mask)
    """
)


def test_minimal_valid_kernel_passes():
    report = check_solution(VALID_BASE)
    assert not report.blocked, [f"{f.layer}: {f.message}" for f in report.findings]


def test_missing_triton_jit_blocks():
    src = _src(
        """
        import torch
        def foo(x): return x
        """
    )
    report = check_solution(src)
    assert report.blocked
    assert any("triton.jit" in f.message for f in report.findings)


def test_banned_ctypes_import_blocks():
    src = VALID_BASE + "\nimport ctypes\n"
    report = check_solution(src)
    assert report.blocked
    assert any(f.layer == "A-banned-import" for f in report.findings)


def test_banned_eval_call_blocks():
    src = VALID_BASE + "\nresult = eval('1+1')\n"
    report = check_solution(src)
    assert report.blocked
    assert any(f.layer == "D-banned-builtin" for f in report.findings)


def test_torch_nn_functional_blocked():
    """P0-1: F.softmax / torch.nn.functional.* must be rejected."""
    src = _src(
        """
        import torch
        import torch.nn.functional as F
        import triton
        import triton.language as tl

        @triton.jit
        def k(x_ptr, BLOCK: tl.constexpr):
            tl.load(x_ptr)
            tl.store(x_ptr, 0)

        def helper(x):
            return F.softmax(x)
        """
    )
    report = check_solution(src)
    assert report.blocked
    assert any("F.softmax" in f.message for f in report.findings) or any(
        "non-allowlist" in f.message.lower() for f in report.findings
    )


def test_string_oracle_substring_blocked():
    """Layer F: source containing 'oracle' as identifier must be rejected."""
    src = VALID_BASE + "\n# the oracle answer is 42\n"
    report = check_solution(src)
    assert report.blocked
    assert any(f.layer == "F-string-deny" for f in report.findings)


def test_from_torch_nn_functional_import_blocked():
    """`from torch.nn.functional import softmax` must reject.

    Previous fix only checked base module; this taints the imported alias so
    bare `softmax(...)` still fails.
    """
    src = _src(
        """
        import torch
        from torch.nn.functional import softmax
        import triton
        import triton.language as tl

        @triton.jit
        def k(x_ptr, BLOCK: tl.constexpr):
            tl.load(x_ptr); tl.store(x_ptr, 0)

        def helper(x):
            return softmax(x)
        """
    )
    report = check_solution(src)
    assert report.blocked
    assert any("softmax" in f.message for f in report.findings)


def test_dunder_getattribute_bypass_blocked():
    """type(x).__getattribute__(x, 'cpu')() must reject."""
    src = _src(
        """
        import torch
        import triton
        import triton.language as tl

        @triton.jit
        def k(x_ptr, BLOCK: tl.constexpr):
            tl.load(x_ptr); tl.store(x_ptr, 0)

        def attack(x):
            return type(x).__getattribute__(x, "cpu")()
        """
    )
    report = check_solution(src)
    assert report.blocked


def test_builtins_subscript_bypass_blocked():
    """__builtins__['eval'](...) must reject."""
    src = _src(
        """
        import torch
        import triton
        import triton.language as tl

        @triton.jit
        def k(x_ptr, BLOCK: tl.constexpr):
            tl.load(x_ptr); tl.store(x_ptr, 0)

        def attack():
            return __builtins__["eval"]("1+1")
        """
    )
    report = check_solution(src)
    assert report.blocked


def test_assign_aliased_eval_blocked():
    """`e = eval; e('1+1')` via Assign alias must block."""
    src = _src(
        """
        import triton
        import triton.language as tl

        @triton.jit
        def k(x_ptr, BLOCK: tl.constexpr):
            tl.load(x_ptr); tl.store(x_ptr, 0)

        def attack():
            e = eval
            return e("1+1")
        """
    )
    report = check_solution(src)
    assert report.blocked


def test_assign_aliased_f_softmax_blocked():
    """`sf = F.softmax` via Assign alias must block."""
    src = _src(
        """
        import torch
        import torch.nn.functional as F
        import triton
        import triton.language as tl

        @triton.jit
        def k(x_ptr, BLOCK: tl.constexpr):
            tl.load(x_ptr); tl.store(x_ptr, 0)

        def attack(x):
            sf = F.softmax
            return sf(x)
        """
    )
    report = check_solution(src)
    assert report.blocked


def test_assign_aliased_tl_inline_asm_blocked():
    """`asm = tl.inline_asm_elementwise` via alias."""
    src = _src(
        """
        import triton
        import triton.language as tl

        @triton.jit
        def k(x_ptr, BLOCK: tl.constexpr):
            asm = tl.inline_asm_elementwise
            y = asm("nop", "=r,r", [tl.load(x_ptr)], dtype=tl.float32)
            tl.store(x_ptr, y)
        """
    )
    report = check_solution(src)
    assert report.blocked


def test_from_triton_language_inline_asm_alias_blocked():
    """`from triton.language import inline_asm as asm`."""
    src = _src(
        """
        import triton
        import triton.language as tl
        from triton.language import inline_asm_elementwise as asm

        @triton.jit
        def k(x_ptr, BLOCK: tl.constexpr):
            tl.load(x_ptr); tl.store(x_ptr, 0)

        def attack(x):
            return asm("", "", [x], dtype=tl.float32)
        """
    )
    report = check_solution(src)
    assert report.blocked


def test_from_torch_import_matmul_blocked():
    """`from torch import matmul; matmul(x,y)` must block."""
    src = _src(
        """
        import torch
        from torch import matmul
        import triton
        import triton.language as tl

        @triton.jit
        def k(x_ptr, BLOCK: tl.constexpr):
            tl.load(x_ptr); tl.store(x_ptr, 0)

        def attack(x, y):
            return matmul(x, y)
        """
    )
    report = check_solution(src)
    assert report.blocked


def test_module_alias_inline_asm_blocked():
    """`import triton.language as lang;
    asm = lang.inline_asm_elementwise` must canonicalize and block."""
    src = _src(
        """
        import triton
        import triton.language as lang

        @triton.jit
        def k(x_ptr, BLOCK: lang.constexpr):
            lang.load(x_ptr); lang.store(x_ptr, 0)

        def attack(x):
            asm = lang.inline_asm_elementwise
            return asm("nop", "=r,r", [x], dtype=lang.float32)
        """
    )
    report = check_solution(src)
    assert report.blocked


def test_tuple_unpacking_alias_blocked():
    """`asm = (tl.inline_asm_elementwise,)[0]` via Subscript."""
    src = _src(
        """
        import triton
        import triton.language as tl

        @triton.jit
        def k(x_ptr, BLOCK: tl.constexpr):
            tl.load(x_ptr); tl.store(x_ptr, 0)

        def attack(x):
            asm = (tl.inline_asm_elementwise,)[0]
            return asm("nop", "=r,r", [x], dtype=tl.float32)
        """
    )
    report = check_solution(src)
    assert report.blocked


def test_partial_open_blocked():
    """`from functools import partial; partial(open, ...)`."""
    src = _src(
        """
        from functools import partial
        import triton
        import triton.language as tl

        @triton.jit
        def k(x_ptr, BLOCK: tl.constexpr):
            tl.load(x_ptr); tl.store(x_ptr, 0)

        def attack():
            p = partial(open, "/etc/passwd")
            return p().read()
        """
    )
    report = check_solution(src)
    assert report.blocked


def test_from_torch_nn_alias_softmax_blocked():
    """`from torch import nn; nn.functional.softmax`."""
    src = _src(
        """
        from torch import nn
        import triton
        import triton.language as tl

        @triton.jit
        def k(x_ptr, BLOCK: tl.constexpr):
            tl.load(x_ptr); tl.store(x_ptr, 0)

        def attack(x):
            sf = nn.functional.softmax
            return sf(x)
        """
    )
    report = check_solution(src)
    assert report.blocked


def test_operator_attrgetter_blocked():
    """`from operator import attrgetter` reflection."""
    src = _src(
        """
        from operator import attrgetter
        import triton
        import triton.language as tl

        @triton.jit
        def k(x_ptr, BLOCK: tl.constexpr):
            tl.load(x_ptr); tl.store(x_ptr, 0)

        def attack(x):
            g = attrgetter("__class__")
            return g(x)
        """
    )
    report = check_solution(src)
    assert report.blocked


_ATTACK_PREFIX = _src(
    """
    import torch
    import triton
    import triton.language as tl

    @triton.jit
    def k(x_ptr, BLOCK: tl.constexpr):
        v = tl.load(x_ptr)
        tl.store(x_ptr, v)
    """
)


def test_pinned_tuple_indexed_call_blocked():
    """#1: `(torch.matmul,)[0](x,y)` indirect call."""
    src = _ATTACK_PREFIX + _src(
        """
        def attack(x, y):
            return (torch.matmul,)[0](x, y)
        """
    )
    report = check_solution(src)
    assert report.blocked


def test_dict_indexed_call_blocked():
    """#1: `{0: torch.matmul}[0](x,y)`."""
    src = _ATTACK_PREFIX + _src(
        """
        def attack(x, y):
            return {0: torch.matmul}[0](x, y)
        """
    )
    report = check_solution(src)
    assert report.blocked


def test_destructuring_assign_blocked():
    """#2: `f, = (torch.matmul,); f(x,y)`."""
    src = _ATTACK_PREFIX + _src(
        """
        def attack(x, y):
            f, = (torch.matmul,)
            return f(x, y)
        """
    )
    report = check_solution(src)
    assert report.blocked


def test_for_loop_target_blocked():
    """#2: `for f in (torch.matmul,): f(x,y)`."""
    src = _ATTACK_PREFIX + _src(
        """
        def attack(x, y):
            for f in (torch.matmul,):
                return f(x, y)
        """
    )
    report = check_solution(src)
    assert report.blocked


def test_annassign_blocked():
    """#2: `f: object = torch.matmul; f(x,y)`."""
    src = _ATTACK_PREFIX + _src(
        """
        def attack(x, y):
            f: object = torch.matmul
            return f(x, y)
        """
    )
    report = check_solution(src)
    assert report.blocked


def test_default_arg_blocked():
    """#2: `def attack(x, y, f=torch.matmul):`."""
    src = _ATTACK_PREFIX + _src(
        """
        def attack(x, y, f=torch.matmul):
            return f(x, y)
        """
    )
    report = check_solution(src)
    assert report.blocked


def test_wildcard_import_torch_blocked():
    """#3: `from torch import *` is forbidden."""
    src = _src(
        """
        from torch import *
        import triton
        import triton.language as tl

        @triton.jit
        def k(x_ptr, BLOCK: tl.constexpr):
            tl.load(x_ptr); tl.store(x_ptr, 0)

        def attack(x, y):
            return matmul(x, y)
        """
    )
    report = check_solution(src)
    assert report.blocked
    assert any("wildcard" in f.message.lower() or "import *" in f.message
               for f in report.findings)


def test_wildcard_import_tl_blocked():
    src = _src(
        """
        import triton
        import triton.language as tl
        from triton.language import *

        @triton.jit
        def k(x_ptr, BLOCK: tl.constexpr):
            tl.load(x_ptr); tl.store(x_ptr, 0)

        def attack(x):
            return inline_asm_elementwise("nop", "=r,r", [x], dtype=tl.float32)
        """
    )
    report = check_solution(src)
    assert report.blocked


def test_nested_shadow_first_binding_wins():
    """#4: nested import cannot overwrite top-level alias."""
    src = _ATTACK_PREFIX + _src(
        """
        from torch import matmul as f

        def unused_shadow():
            from triton import cdiv as f
            return f

        def attack(x, y):
            return f(x, y)
        """
    )
    report = check_solution(src)
    assert report.blocked


def test_bound_method_alias_blocked():
    """#5: `f = x.matmul; f(y)` (Tensor.method binding)."""
    src = _ATTACK_PREFIX + _src(
        """
        def attack(x, y):
            f = x.matmul
            return f(y)
        """
    )
    report = check_solution(src)
    assert report.blocked


def test_subscript_base_matmul_blocked():
    """#1: `inputs['x'].matmul(y)` (non-Name base)."""
    src = _ATTACK_PREFIX + _src(
        """
        def attack(inputs):
            return inputs["x"].matmul(inputs["y"])
        """
    )
    report = check_solution(src)
    assert report.blocked


def test_subscript_base_cpu_blocked():
    """#1: `inputs['x'].cpu()` host-transfer."""
    src = _ATTACK_PREFIX + _src(
        """
        def attack(inputs):
            return inputs["x"].cpu()
        """
    )
    report = check_solution(src)
    assert report.blocked


def test_ifexp_call_target_blocked():
    """#2: `(matmul if cond else empty)(x,y)`."""
    src = _ATTACK_PREFIX + _src(
        """
        def attack(x, y):
            return (torch.matmul if True else torch.empty)(x, y)
        """
    )
    report = check_solution(src)
    assert report.blocked


def test_lambda_call_target_blocked():
    """#2: `(lambda: torch.matmul)()(x,y)`."""
    src = _ATTACK_PREFIX + _src(
        """
        def attack(x, y):
            return (lambda: torch.matmul)()(x, y)
        """
    )
    report = check_solution(src)
    assert report.blocked


def test_higher_order_call_arg_blocked():
    """#3: `(lambda fs: fs[0](x,y))((torch.matmul,))`."""
    src = _ATTACK_PREFIX + _src(
        """
        def attack(x, y):
            return (lambda fs: fs[0](x, y))((torch.matmul,))
        """
    )
    report = check_solution(src)
    assert report.blocked


def test_first_binding_safe_then_banned_blocked():
    """#4: safe `f` rebound to banned `f` must taint."""
    src = _ATTACK_PREFIX + _src(
        """
        from triton import cdiv as f
        from torch import matmul as f

        def attack(x, y):
            return f(x, y)
        """
    )
    report = check_solution(src)
    assert report.blocked


def test_tensor_binop_blocked():
    """#1: `x + y` on Tensor-annotated args."""
    src = _src(
        """
        import torch
        import triton
        import triton.language as tl

        @triton.jit
        def k(x_ptr, BLOCK: tl.constexpr):
            tl.load(x_ptr); tl.store(x_ptr, 0)

        def attack(x: torch.Tensor, y: torch.Tensor):
            return x + y
        """
    )
    report = check_solution(src)
    assert report.blocked


def test_tensor_subscript_blocked():
    """#1: `x[:]` on Tensor."""
    src = _src(
        """
        import torch
        import triton
        import triton.language as tl

        @triton.jit
        def k(x_ptr, BLOCK: tl.constexpr):
            tl.load(x_ptr); tl.store(x_ptr, 0)

        def attack(x: torch.Tensor):
            return x[:]
        """
    )
    report = check_solution(src)
    assert report.blocked


def test_tensor_iter_blocked():
    """#1: `for row in x:` on Tensor."""
    src = _src(
        """
        import torch
        import triton
        import triton.language as tl

        @triton.jit
        def k(x_ptr, BLOCK: tl.constexpr):
            tl.load(x_ptr); tl.store(x_ptr, 0)

        def attack(x: torch.Tensor):
            for row in x:
                return row
        """
    )
    report = check_solution(src)
    assert report.blocked


def test_tensor_sum_coerce_blocked():
    """#1: `sum(x)` / `float(x)` on Tensor."""
    src = _src(
        """
        import torch
        import triton
        import triton.language as tl

        @triton.jit
        def k(x_ptr, BLOCK: tl.constexpr):
            tl.load(x_ptr); tl.store(x_ptr, 0)

        def attack(x: torch.Tensor):
            return float(x)
        """
    )
    report = check_solution(src)
    assert report.blocked


def test_to_cpu_string_blocked():
    """#2: `.to('cpu')`."""
    src = _src(
        """
        import torch
        import triton
        import triton.language as tl

        @triton.jit
        def k(x_ptr, BLOCK: tl.constexpr):
            tl.load(x_ptr); tl.store(x_ptr, 0)

        def attack(x):
            return x.to("cpu")
        """
    )
    report = check_solution(src)
    assert report.blocked


def test_to_cpu_kwarg_blocked():
    """#2: `.to(device='cpu')`."""
    src = _src(
        """
        import torch
        import triton
        import triton.language as tl

        @triton.jit
        def k(x_ptr, BLOCK: tl.constexpr):
            tl.load(x_ptr); tl.store(x_ptr, 0)

        def attack(x):
            return x.to(device="cpu")
        """
    )
    report = check_solution(src)
    assert report.blocked


def test_torch_empty_device_cpu_blocked():
    """#2: `torch.empty(..., device='cpu')`."""
    src = _src(
        """
        import torch
        import triton
        import triton.language as tl

        @triton.jit
        def k(x_ptr, BLOCK: tl.constexpr):
            tl.load(x_ptr); tl.store(x_ptr, 0)

        def attack(x):
            return torch.empty(x.shape, device="cpu")
        """
    )
    report = check_solution(src)
    assert report.blocked


def test_iter_next_builtins_blocked():
    """#1: bare `iter()` / `next()` calls in solution."""
    src = _src(
        """
        import torch
        import triton
        import triton.language as tl

        @triton.jit
        def k(x_ptr, BLOCK: tl.constexpr):
            tl.load(x_ptr); tl.store(x_ptr, 0)

        def attack(x):
            it = iter(x)
            return next(it)
        """
    )
    report = check_solution(src)
    assert report.blocked


def test_string_annotation_blocked():
    """`x: 'torch.Tensor'` (forward ref str annotation)."""
    src = _ATTACK_PREFIX + _src(
        """
        def attack(x: "torch.Tensor", y: "torch.Tensor"):
            return x + y
        """
    )
    report = check_solution(src)
    assert report.blocked


def test_varargs_attack_blocked():
    """`def f(*args, **kwargs): x = args[0]; x + y`."""
    src = _ATTACK_PREFIX + _src(
        """
        def attack(*args, **kwargs):
            x = args[0]
            y = args[1]
            return x + y
        """
    )
    report = check_solution(src)
    assert report.blocked


def test_reshape_binop_blocked():
    """`x.reshape(-1) + y.reshape(-1)` (Tensor method + BinOp)."""
    src = _ATTACK_PREFIX + _src(
        """
        def attack(x: torch.Tensor, y: torch.Tensor):
            return x.reshape(-1) + y.reshape(-1)
        """
    )
    report = check_solution(src)
    assert report.blocked


def test_tuple_return_unpacking_blocked():
    """`a, b = get_pair(x, y); a + b`."""
    src = _ATTACK_PREFIX + _src(
        """
        def get_pair(x: torch.Tensor, y: torch.Tensor):
            return x, y

        def attack(x: torch.Tensor, y: torch.Tensor):
            a, b = get_pair(x, y)
            return a + b
        """
    )
    report = check_solution(src)
    assert report.blocked


def test_enumerate_tensor_blocked():
    """`for _, row in enumerate(x):`."""
    src = _ATTACK_PREFIX + _src(
        """
        def attack(x: torch.Tensor):
            for _, row in enumerate(x):
                return row
        """
    )
    report = check_solution(src)
    assert report.blocked


def test_local_string_cpu_to_blocked():
    """`d = 'cpu'; x.to(d)`."""
    src = _ATTACK_PREFIX + _src(
        """
        def attack(x):
            d = "cpu"
            return x.to(d)
        """
    )
    report = check_solution(src)
    assert report.blocked


def test_local_string_empty_cpu_blocked():
    """`d = 'cpu'; torch.empty(..., device=d)`."""
    src = _ATTACK_PREFIX + _src(
        """
        def attack(x):
            d = "cpu"
            return torch.empty(x.shape, device=d)
        """
    )
    report = check_solution(src)
    assert report.blocked


def test_lambda_call_taint_blocked():
    src = _ATTACK_PREFIX + _src(
        """
        def attack(x: torch.Tensor, y: torch.Tensor):
            a = (lambda: x)()
            b = (lambda: y)()
            return a + b
        """
    )
    assert check_solution(src).blocked


def test_tuple_destructure_assign_blocked():
    src = _ATTACK_PREFIX + _src(
        """
        def attack(x: torch.Tensor, y: torch.Tensor):
            a, b = (x, y)
            return a + b
        """
    )
    assert check_solution(src).blocked


def test_ifexp_assign_blocked():
    src = _ATTACK_PREFIX + _src(
        """
        def attack(x: torch.Tensor, y: torch.Tensor, cond):
            z = x if cond else y
            return z + x
        """
    )
    assert check_solution(src).blocked


def test_default_arg_cpu_blocked():
    src = _ATTACK_PREFIX + _src(
        """
        def attack(x, d="cpu"):
            return x.to(d)
        """
    )
    assert check_solution(src).blocked


def test_binop_cpu_concat_blocked():
    src = _ATTACK_PREFIX + _src(
        """
        def attack(x):
            d = "c" + "pu"
            return torch.empty(x.shape, device=d)
        """
    )
    assert check_solution(src).blocked


def test_dict_inputs_contract_blocked():
    """`def npu_triton_impl(inputs: dict)` must
    treat `inputs` as a Tensor-container per the contract."""
    src = _ATTACK_PREFIX + _src(
        """
        def npu_triton_impl(inputs: dict) -> dict:
            return {"out": inputs["x"] + inputs["y"]}
        """
    )
    assert check_solution(src).blocked


def test_tensor_compare_blocked():
    src = _ATTACK_PREFIX + _src(
        """
        def attack(x: torch.Tensor, y: torch.Tensor):
            return x == y
        """
    )
    assert check_solution(src).blocked


def test_tensor_unary_blocked():
    src = _ATTACK_PREFIX + _src(
        """
        def attack(x: torch.Tensor):
            return -x
        """
    )
    assert check_solution(src).blocked


def test_tensor_boolop_blocked():
    src = _ATTACK_PREFIX + _src(
        """
        def attack(x: torch.Tensor, y: torch.Tensor):
            return x and y
        """
    )
    assert check_solution(src).blocked


def test_new_empty_cpu_blocked():
    """#3: `.new_empty(..., device='cpu')`."""
    src = _ATTACK_PREFIX + _src(
        """
        def attack(x: torch.Tensor):
            return x.new_empty(x.shape, device="cpu")
        """
    )
    assert check_solution(src).blocked


def test_any_all_sorted_reversed_on_tensor_blocked():
    """#4: any/all/sorted/reversed eager coercions."""
    for fn_name in ("any", "all", "sorted", "reversed"):
        src = _ATTACK_PREFIX + _src(
            f"""
            def attack(x: torch.Tensor):
                return {fn_name}(x)
            """
        )
        assert check_solution(src).blocked, (
            f"`{fn_name}(tensor)` should be blocked but passed"
        )


def test_str_call_cpu_blocked():
    src = _ATTACK_PREFIX + _src(
        """
        def attack(x: torch.Tensor):
            return x.to(str("cpu"))
        """
    )
    assert check_solution(src).blocked


def test_tuple_index_cpu_blocked():
    src = _ATTACK_PREFIX + _src(
        """
        def attack(x: torch.Tensor):
            return x.new_empty(x.shape, device=("gpu", "cpu")[1])
        """
    )
    assert check_solution(src).blocked


def test_mult_cpu_blocked():
    src = _ATTACK_PREFIX + _src(
        """
        def attack(x: torch.Tensor):
            return torch.empty(x.shape, device="cpu" * 1)
        """
    )
    assert check_solution(src).blocked


def test_lambda_cpu_blocked():
    src = _ATTACK_PREFIX + _src(
        """
        def attack(x: torch.Tensor):
            return x.to((lambda: "cpu")())
        """
    )
    assert check_solution(src).blocked


def test_iife_lambda_tensor_arg_blocked():
    """`(lambda z: z+z)(tensor)` — IIFE binding."""
    src = _ATTACK_PREFIX + _src(
        """
        def attack(x: torch.Tensor):
            return (lambda z: z + z)(x)
        """
    )
    assert check_solution(src).blocked


def test_typing_mapping_blocked():
    """`Mapping[str, torch.Tensor]`."""
    src = _ATTACK_PREFIX + _src(
        """
        from typing import Mapping

        def attack(inputs: Mapping[str, torch.Tensor]):
            return inputs["x"] + inputs["y"]
        """
    )
    assert check_solution(src).blocked


def test_typing_list_tensor_blocked():
    src = _ATTACK_PREFIX + _src(
        """
        from typing import List

        def attack(xs: List[torch.Tensor]):
            return xs[0] + xs[1]
        """
    )
    assert check_solution(src).blocked


def test_bare_list_tensor_blocked():
    src = _ATTACK_PREFIX + _src(
        """
        def attack(xs: list[torch.Tensor]):
            return xs[0] + xs[1]
        """
    )
    assert check_solution(src).blocked


def test_npu_triton_impl_unannotated_blocked():
    """#1: missing annotation still treated as container."""
    src = _ATTACK_PREFIX + _src(
        """
        def npu_triton_impl(inputs) -> dict:
            return {"out": inputs["x"] + inputs["y"]}
        """
    )
    assert check_solution(src).blocked


def test_npu_triton_impl_object_annotated_blocked():
    src = _ATTACK_PREFIX + _src(
        """
        def npu_triton_impl(inputs: object) -> dict:
            return {"out": inputs["x"] + inputs["y"]}
        """
    )
    assert check_solution(src).blocked


def test_helper_propagation_blocked():
    """#2: cross-function helper Tensor arg propagation."""
    src = _ATTACK_PREFIX + _src(
        """
        def add(z):
            return z + z

        def attack(x: torch.Tensor):
            return add(x)
        """
    )
    assert check_solution(src).blocked


def test_legit_dict_lookup_not_blocked():
    """#1: `inputs["x"]` then `torch.empty_like(x)` legit."""
    src = _ATTACK_PREFIX + _src(
        """
        def npu_triton_impl(inputs: dict) -> dict:
            x = inputs["x"]
            y = torch.empty_like(x)
            return {"out": y}
        """
    )
    report = check_solution(src)
    assert not report.blocked, [f"{f.layer}:{f.message}" for f in report.findings]


def test_scalar_method_not_blocked():
    """#2: `.numel()`/`.size()`/`.dim()` return int."""
    src = _ATTACK_PREFIX + _src(
        """
        def attack(x: torch.Tensor):
            n = x.numel()
            return n + 1
        """
    )
    report = check_solution(src)
    assert not report.blocked, [f"{f.layer}:{f.message}" for f in report.findings]


def test_helper_rename_alias_blocked():
    src = _ATTACK_PREFIX + _src(
        """
        def h(z): return z + z
        def attack(x: torch.Tensor):
            y = x
            return h(y)
        """
    )
    assert check_solution(src).blocked


def test_helper_kwarg_blocked():
    src = _ATTACK_PREFIX + _src(
        """
        def h(z): return z + z
        def attack(x: torch.Tensor):
            return h(z=x)
        """
    )
    assert check_solution(src).blocked


def test_container_to_helper_blocked():
    src = _ATTACK_PREFIX + _src(
        """
        def h(z): return z + z
        def npu_triton_impl(inputs):
            return {"out": h(inputs["x"])}
        """
    )
    assert check_solution(src).blocked


def test_twohop_helper_blocked():
    src = _ATTACK_PREFIX + _src(
        """
        def h2(z): return z + z
        def h1(y): return h2(y)
        def attack(x: torch.Tensor):
            return h1(x)
        """
    )
    assert check_solution(src).blocked


def test_optional_list_tensor_blocked():
    src = _ATTACK_PREFIX + _src(
        """
        def attack(xs: list[torch.Tensor] | None):
            return xs[0] + xs[1]
        """
    )
    assert check_solution(src).blocked


def test_typing_optional_list_tensor_blocked():
    src = _ATTACK_PREFIX + _src(
        """
        from typing import Optional, List
        def attack(xs: Optional[List[torch.Tensor]]):
            return xs[0] + xs[1]
        """
    )
    assert check_solution(src).blocked


def test_type_alias_container_blocked():
    src = _ATTACK_PREFIX + _src(
        """
        from typing import List
        TensorList = List["Tensor"]
        def attack(xs: TensorList):
            return xs[0] + xs[1]
        """
    )
    assert check_solution(src).blocked


def test_dict_subclass_blocked():
    src = _ATTACK_PREFIX + _src(
        """
        class MyDict(dict): pass
        def attack(inputs: MyDict):
            return inputs["x"] + inputs["y"]
        """
    )
    assert check_solution(src).blocked


def test_percent_format_cpu_blocked():
    src = _ATTACK_PREFIX + _src(
        """
        def attack(x: torch.Tensor):
            return x.to("%s%s%s" % ("c", "p", "u"))
        """
    )
    assert check_solution(src).blocked


def test_slice_literal_cpu_blocked():
    src = _ATTACK_PREFIX + _src(
        """
        def attack(x: torch.Tensor):
            return x.to("xcpu"[1:])
        """
    )
    assert check_solution(src).blocked


def test_function_return_cpu_blocked():
    src = _ATTACK_PREFIX + _src(
        """
        def dev(): return "cpu"
        def attack(x: torch.Tensor):
            return x.to(dev())
        """
    )
    assert check_solution(src).blocked


def test_chr_concat_cpu_blocked():
    src = _ATTACK_PREFIX + _src(
        """
        def attack(x: torch.Tensor):
            return x.to(chr(99) + chr(112) + chr(117))
        """
    )
    assert check_solution(src).blocked


def test_iife_lambda_scalar_metadata_not_blocked():
    """`(lambda z: z.numel())(x)` returns int — legit."""
    src = _ATTACK_PREFIX + _src(
        """
        def attack(x: torch.Tensor):
            n = (lambda z: z.numel())(x)
            return n + 1
        """
    )
    report = check_solution(src)
    assert not report.blocked, [f"{f.layer}:{f.message}" for f in report.findings]


def test_forward_ref_not_tensor_not_blocked():
    """`xs: 'List[NotTensor]'` must NOT be Tensor container."""
    src = _ATTACK_PREFIX + _src(
        """
        from typing import List
        def attack(xs: "List[NotTensor]"):
            return xs[0] + xs[1]
        """
    )
    report = check_solution(src)
    assert not report.blocked, [f"{f.layer}:{f.message}" for f in report.findings]


def test_sorted_with_key_lambda_not_blocked():
    """`sorted(vals, key=lambda z: z)` legit."""
    src = _ATTACK_PREFIX + _src(
        """
        def attack(vals):
            return sorted(vals, key=lambda z: z)
        """
    )
    report = check_solution(src)
    assert not report.blocked, [f"{f.layer}:{f.message}" for f in report.findings]


def test_11hop_helper_blocked():
    """fixpoint must handle >8-hop chains."""
    helpers = "\n".join(
        f"def h{i}(z): return h{i+1}(z)" for i in range(1, 10)
    ) + "\ndef h10(z): return z + z\n"
    src = _ATTACK_PREFIX + helpers + _src(
        """
        def attack(x: torch.Tensor):
            return h1(x)
        """
    )
    assert check_solution(src).blocked


def test_reshape_arg_to_helper_blocked():
    src = _ATTACK_PREFIX + _src(
        """
        def h(z): return z + z
        def attack(x: torch.Tensor):
            return h(x.reshape(-1))
        """
    )
    assert check_solution(src).blocked


def test_container_alias_blocked():
    src = _ATTACK_PREFIX + _src(
        """
        def h(z): return z + z
        def npu_triton_impl(inputs):
            c = inputs
            return h(c["x"])
        """
    )
    assert check_solution(src).blocked


def test_destructure_subscript_blocked():
    src = _ATTACK_PREFIX + _src(
        """
        def h(z): return z + z
        def npu_triton_impl(inputs):
            (y,) = (inputs["x"],)
            return h(y)
        """
    )
    assert check_solution(src).blocked


def test_percent_c_cpu_blocked():
    src = _ATTACK_PREFIX + _src(
        """
        def attack(x: torch.Tensor):
            return x.to("%c%c%c" % (99, 112, 117))
        """
    )
    assert check_solution(src).blocked


def test_chr_in_fn_return_blocked():
    src = _ATTACK_PREFIX + _src(
        """
        def dev():
            return chr(99) + chr(112) + chr(117)
        def attack(x: torch.Tensor):
            return x.to(dev())
        """
    )
    assert check_solution(src).blocked


def test_nested_iife_lambda_blocked():
    src = _ATTACK_PREFIX + _src(
        """
        def attack(x: torch.Tensor):
            return (lambda z: (lambda w: w + w)(z))(x)
        """
    )
    assert check_solution(src).blocked


def test_tensor_method_on_subscript_blocked():
    src = _ATTACK_PREFIX + _src(
        """
        def npu_triton_impl(inputs):
            return {"out": inputs["x"].add(inputs["y"])}
        """
    )
    assert check_solution(src).blocked


def test_instance_method_helper_blocked():
    src = _ATTACK_PREFIX + _src(
        """
        class H:
            def h(self, z): return z + z

        def npu_triton_impl(inputs):
            return {"out": H().h(inputs["x"])}
        """
    )
    assert check_solution(src).blocked


def test_callable_object_blocked():
    src = _ATTACK_PREFIX + _src(
        """
        class H:
            def __call__(self, z): return z + z

        def npu_triton_impl(inputs):
            return {"out": H()(inputs["x"])}
        """
    )
    assert check_solution(src).blocked


def test_optional_pipe_tensor_blocked():
    src = _ATTACK_PREFIX + _src(
        """
        def attack(x: torch.Tensor | None):
            return x + x
        """
    )
    assert check_solution(src).blocked


def test_typing_optional_tensor_blocked():
    src = _ATTACK_PREFIX + _src(
        """
        from typing import Optional
        def attack(x: Optional[torch.Tensor]):
            return x + x
        """
    )
    assert check_solution(src).blocked


def test_forward_ref_optional_tensor_blocked():
    src = _ATTACK_PREFIX + _src(
        """
        def attack(xs: "Optional[List[torch.Tensor]]"):
            return xs[0] + xs[1]
        """
    )
    assert check_solution(src).blocked


def test_iife_lambda_local_alias_blocked():
    src = _ATTACK_PREFIX + _src(
        """
        def h(z): return z + z
        def attack(x: torch.Tensor):
            y = (lambda v: v.contiguous())(x)
            return h(y)
        """
    )
    assert check_solution(src).blocked


def test_multi_branch_cpu_blocked():
    src = _ATTACK_PREFIX + _src(
        """
        def dev(cond):
            if cond:
                return "cpu"
            return "npu"
        def attack(x: torch.Tensor):
            return x.to(dev(True))
        """
    )
    assert check_solution(src).blocked


def test_callable_named_alias_blocked():
    src = _ATTACK_PREFIX + _src(
        """
        def h(z): return z + z
        def npu_triton_impl(inputs):
            f = h
            return {"out": f(inputs["x"])}
        """
    )
    assert check_solution(src).blocked


def test_lambda_alias_blocked():
    src = _ATTACK_PREFIX + _src(
        """
        def npu_triton_impl(inputs):
            f = lambda z: z + z
            return {"out": f(inputs["x"])}
        """
    )
    assert check_solution(src).blocked


def test_dict_slot_assign_blocked():
    src = _ATTACK_PREFIX + _src(
        """
        def npu_triton_impl(inputs):
            d = {}
            d["x"] = inputs["x"]
            return {"out": d["x"] + d["x"]}
        """
    )
    assert check_solution(src).blocked


def test_attr_assign_blocked():
    src = _ATTACK_PREFIX + _src(
        """
        class B: pass
        def npu_triton_impl(inputs):
            b = B()
            b.x = inputs["x"]
            return {"out": b.x + b.x}
        """
    )
    assert check_solution(src).blocked


def test_scalar_tuple_helper_not_blocked():
    """`dims = (1, 2); h(dims)` must NOT be tainted."""
    src = _ATTACK_PREFIX + _src(
        """
        def h(dims):
            return dims[0] + dims[1]
        def attack(x: torch.Tensor):
            dims = (1, 2)
            return h(dims)
        """
    )
    report = check_solution(src)
    assert not report.blocked, [f"{f.layer}:{f.message}" for f in report.findings]


def test_unreachable_rebind_blocked():
    src = _ATTACK_PREFIX + _src(
        """
        def h(z): return z + z
        def safe(z): return z
        def npu_triton_impl(inputs):
            f = h
            if False:
                f = safe
            return {"out": f(inputs["x"])}
        """
    )
    assert check_solution(src).blocked


def test_dict_callable_slot_blocked():
    src = _ATTACK_PREFIX + _src(
        """
        def h(z): return z + z
        def npu_triton_impl(inputs):
            d = {}
            d["f"] = h
            return {"out": d["f"](inputs["x"])}
        """
    )
    assert check_solution(src).blocked


def test_tuple_indexed_call_blocked():
    src = _ATTACK_PREFIX + _src(
        """
        def h(z): return z + z
        def npu_triton_impl(inputs):
            return {"out": (h,)[0](inputs["x"])}
        """
    )
    assert check_solution(src).blocked


def test_ifexp_callable_blocked():
    src = _ATTACK_PREFIX + _src(
        """
        def h(z): return z + z
        def safe(z): return z
        def npu_triton_impl(inputs):
            return {"out": (h if True else safe)(inputs["x"])}
        """
    )
    assert check_solution(src).blocked


def test_custom_class_method_blocked():
    src = _ATTACK_PREFIX + _src(
        """
        class H:
            def reshape(self, z): return z + z

        def npu_triton_impl(inputs):
            a = H()
            return {"out": a.reshape(inputs["x"])}
        """
    )
    assert check_solution(src).blocked


def test_container_alias_audit_blocked():
    src = _ATTACK_PREFIX + _src(
        """
        def npu_triton_impl(inputs):
            c = inputs
            return {"out": c["x"] + c["x"]}
        """
    )
    assert check_solution(src).blocked


def test_nested_container_blocked():
    src = _ATTACK_PREFIX + _src(
        """
        def npu_triton_impl(inputs):
            outer = {}
            outer["k"] = inputs
            return {"out": outer["k"]["x"] + outer["k"]["x"]}
        """
    )
    assert check_solution(src).blocked


def test_fn_to_fn_cpu_blocked():
    src = _ATTACK_PREFIX + _src(
        """
        def dev(): return inner()
        def inner(): return "cpu"
        def attack(x: torch.Tensor):
            return x.to(dev())
        """
    )
    assert check_solution(src).blocked


def test_class_method_does_not_shadow_blocked():
    src = _ATTACK_PREFIX + _src(
        """
        def dev(): return "cpu"
        class C:
            def dev(self): return "npu"
        def attack(x: torch.Tensor):
            return x.to(dev())
        """
    )
    assert check_solution(src).blocked


def test_lambda_alias_returns_cpu_blocked():
    src = _ATTACK_PREFIX + _src(
        """
        dev = lambda: "cpu"
        def attack(x: torch.Tensor):
            return x.to(dev())
        """
    )
    assert check_solution(src).blocked


def test_lambda_alias_multi_binding_blocked():
    src = _ATTACK_PREFIX + _src(
        """
        def npu_triton_impl(inputs):
            f = lambda z: z.contiguous()
            f = lambda z: z + z
            return {"out": f(inputs["x"])}
        """
    )
    assert check_solution(src).blocked


def test_nested_attr_method_blocked():
    src = _ATTACK_PREFIX + _src(
        """
        class H:
            def reshape(self, z): return z + z
        class B: pass
        def npu_triton_impl(inputs):
            b = B()
            b.h = H()
            return {"out": b.h.reshape(inputs["x"])}
        """
    )
    assert check_solution(src).blocked


def test_list_indexed_method_blocked():
    src = _ATTACK_PREFIX + _src(
        """
        class H:
            def reshape(self, z): return z + z
        def npu_triton_impl(inputs):
            objs = [H()]
            return {"out": objs[0].reshape(inputs["x"])}
        """
    )
    assert check_solution(src).blocked


def test_instance_call_method_blocked():
    src = _ATTACK_PREFIX + _src(
        """
        class H:
            def __call__(self, z): return z + z
        def npu_triton_impl(inputs):
            f = H()
            return {"out": f(inputs["x"])}
        """
    )
    assert check_solution(src).blocked


def test_class_init_eager_blocked():
    src = _ATTACK_PREFIX + _src(
        """
        class H:
            def __init__(self, z):
                self.out = z + z
        def npu_triton_impl(inputs):
            return {"out": H(inputs["x"]).out}
        """
    )
    assert check_solution(src).blocked


def test_class_method_shadow_blocked():
    src = _ATTACK_PREFIX + _src(
        """
        def h(z): return z + z
        class C:
            def h(self, z): return z
        def npu_triton_impl(inputs):
            return {"out": h(inputs["x"])}
        """
    )
    assert check_solution(src).blocked


def test_dev_alias_to_inner_blocked():
    src = _ATTACK_PREFIX + _src(
        """
        def inner(): return "cpu"
        dev = inner
        def attack(x: torch.Tensor):
            return x.to(dev())
        """
    )
    assert check_solution(src).blocked


def test_deeper_nested_container_blocked():
    src = _ATTACK_PREFIX + _src(
        """
        def npu_triton_impl(inputs):
            root = {}
            root["a"] = {}
            root["a"]["k"] = inputs
            return {"out": root["a"]["k"]["x"] + root["a"]["k"]["x"]}
        """
    )
    assert check_solution(src).blocked


def test_whole_container_helper_blocked():
    src = _ATTACK_PREFIX + _src(
        """
        def h(z):
            return z["x"] + z["x"]
        def npu_triton_impl(inputs):
            return {"out": h(inputs)}
        """
    )
    assert check_solution(src).blocked


def test_lambda_alias_container_blocked():
    src = _ATTACK_PREFIX + _src(
        """
        def npu_triton_impl(inputs):
            c = (lambda z: z)(inputs)
            return {"out": c["x"] + c["x"]}
        """
    )
    assert check_solution(src).blocked


def test_nested_lambda_container_blocked():
    src = _ATTACK_PREFIX + _src(
        """
        def npu_triton_impl(inputs):
            root = {}
            root["a"] = {}
            root["a"]["k"] = (lambda z: z)(inputs)
            return {"out": root["a"]["k"]["x"] + root["a"]["k"]["x"]}
        """
    )
    assert check_solution(src).blocked


def test_local_helper_blocked():
    src = _ATTACK_PREFIX + _src(
        """
        def npu_triton_impl(inputs):
            def h(z): return z + z
            return {"out": h(inputs["x"])}
        """
    )
    assert check_solution(src).blocked


def test_tuple_destructure_callable_blocked():
    src = _ATTACK_PREFIX + _src(
        """
        def h(z): return z + z
        def npu_triton_impl(inputs):
            (f,) = (h,)
            return {"out": f(inputs["x"])}
        """
    )
    assert check_solution(src).blocked


def test_annassign_callable_blocked():
    src = _ATTACK_PREFIX + _src(
        """
        def h(z): return z + z
        def npu_triton_impl(inputs):
            f: object = h
            return {"out": f(inputs["x"])}
        """
    )
    assert check_solution(src).blocked


def test_for_target_callable_blocked():
    src = _ATTACK_PREFIX + _src(
        """
        def h(z): return z + z
        def npu_triton_impl(inputs):
            for f in (h,):
                return {"out": f(inputs["x"])}
        """
    )
    assert check_solution(src).blocked


def test_iife_result_callable_blocked():
    src = _ATTACK_PREFIX + _src(
        """
        def h(z): return z + z
        def npu_triton_impl(inputs):
            f = (lambda: h)()
            return {"out": f(inputs["x"])}
        """
    )
    assert check_solution(src).blocked


def test_kernel_alias_chain_not_blocked():
    """`k1 = k; k2 = k1; k2[grid](...)` legit."""
    src = _ATTACK_PREFIX + _src(
        """
        def npu_triton_impl(inputs):
            k1 = k
            k2 = k1
            out = torch.empty_like(inputs["x"])
            k2[(1,)](inputs["x"], BLOCK=1)
            return {"out": out}
        """
    )
    report = check_solution(src)
    assert not report.blocked, [f"{f.layer}:{f.message}" for f in report.findings]


def test_transitive_dict_subclass_blocked():
    src = _ATTACK_PREFIX + _src(
        """
        class D(dict): pass
        class My(D): pass
        def attack(inputs: My):
            return inputs["x"] + inputs["y"]
        """
    )
    assert check_solution(src).blocked


def test_box_tuple_container_blocked():
    src = _ATTACK_PREFIX + _src(
        """
        def npu_triton_impl(inputs):
            box = (inputs,)
            c = box[0]
            return {"out": c["x"] + c["x"]}
        """
    )
    assert check_solution(src).blocked


def test_box_dict_container_blocked():
    src = _ATTACK_PREFIX + _src(
        """
        def npu_triton_impl(inputs):
            box = {"i": inputs}
            c = box["i"]
            return {"out": c["x"] + c["x"]}
        """
    )
    assert check_solution(src).blocked


def test_for_container_iter_blocked():
    src = _ATTACK_PREFIX + _src(
        """
        def npu_triton_impl(inputs):
            for c in (inputs,):
                return {"out": c["x"] + c["x"]}
        """
    )
    assert check_solution(src).blocked


def test_annassign_container_blocked():
    src = _ATTACK_PREFIX + _src(
        """
        def npu_triton_impl(inputs):
            c: object = inputs
            return {"out": c["x"] + c["x"]}
        """
    )
    assert check_solution(src).blocked


def test_helper_iife_container_blocked():
    src = _ATTACK_PREFIX + _src(
        """
        def h(z):
            return z["x"] + z["x"]
        def npu_triton_impl(inputs):
            return {"out": h((lambda z: z)(inputs))}
        """
    )
    assert check_solution(src).blocked


def test_lambda_default_tensor_blocked():
    src = _ATTACK_PREFIX + _src(
        """
        def attack(x: torch.Tensor):
            return (lambda z=x: z + z)()
        """
    )
    assert check_solution(src).blocked


def test_fn_default_tensor_blocked():
    src = _ATTACK_PREFIX + _src(
        """
        def attack(x: torch.Tensor):
            def h(z=x): return z + z
            return h()
        """
    )
    assert check_solution(src).blocked


def test_decorator_replace_blocked():
    src = _ATTACK_PREFIX + _src(
        """
        def deco(fn): return lambda z: z + z
        @deco
        def h(z): return z
        def attack(x: torch.Tensor):
            return h(x)
        """
    )
    assert check_solution(src).blocked


def test_name_shadow_nested_blocked():
    src = _ATTACK_PREFIX + _src(
        """
        def h(a): return a
        def npu_triton_impl(inputs):
            def h(z): return z + z
            return {"out": h(inputs["x"])}
        """
    )
    assert check_solution(src).blocked


def test_factory_returns_instance_blocked():
    src = _ATTACK_PREFIX + _src(
        """
        class H:
            def __call__(self, z): return z + z
        def make(): return H()
        def npu_triton_impl(inputs):
            f = make()
            return {"out": f(inputs["x"])}
        """
    )
    assert check_solution(src).blocked


def test_mutable_mapping_subclass_blocked():
    src = _ATTACK_PREFIX + _src(
        """
        from typing import MutableMapping
        class My(MutableMapping[str, torch.Tensor]):
            pass
        def attack(inputs: My):
            return inputs["x"] + inputs["x"]
        """
    )
    assert check_solution(src).blocked


def test_lambda_kw_blocked():
    src = _ATTACK_PREFIX + _src(
        """
        def attack(x: torch.Tensor):
            return (lambda z: z + z)(z=x)
        """
    )
    assert check_solution(src).blocked


def test_lambda_starargs_blocked():
    src = _ATTACK_PREFIX + _src(
        """
        def attack(x: torch.Tensor):
            return (lambda *a: a[0] + a[0])(x)
        """
    )
    assert check_solution(src).blocked


def test_lambda_default_return_container_blocked():
    src = _ATTACK_PREFIX + _src(
        """
        def npu_triton_impl(inputs):
            c = (lambda z=inputs: z)()
            return {"out": c["x"] + c["x"]}
        """
    )
    assert check_solution(src).blocked


def test_cpu_lambda_arg_blocked():
    src = _ATTACK_PREFIX + _src(
        """
        def attack(x: torch.Tensor):
            return x.to((lambda d: d)("cpu"))
        """
    )
    assert check_solution(src).blocked


def test_cpu_lambda_default_blocked():
    src = _ATTACK_PREFIX + _src(
        """
        def attack(x: torch.Tensor):
            return x.to((lambda d="cpu": d)())
        """
    )
    assert check_solution(src).blocked


def test_cpu_fn_default_blocked():
    src = _ATTACK_PREFIX + _src(
        """
        def dev(d="cpu"): return d
        def attack(x: torch.Tensor):
            return x.to(dev())
        """
    )
    assert check_solution(src).blocked


def test_decorator_alias_blocked():
    src = _ATTACK_PREFIX + _src(
        """
        def deco(fn): return lambda z: z + z
        @deco
        def h(z): return z
        def attack(x: torch.Tensor):
            g = h
            return g(x)
        """
    )
    assert check_solution(src).blocked


def test_decorator_for_target_blocked():
    src = _ATTACK_PREFIX + _src(
        """
        def deco(fn): return lambda z: z + z
        @deco
        def h(z): return z
        def attack(x: torch.Tensor):
            for g in (h,):
                return g(x)
        """
    )
    assert check_solution(src).blocked


def test_box_destructure_container_blocked():
    src = _ATTACK_PREFIX + _src(
        """
        def npu_triton_impl(inputs):
            box = [inputs]
            (c,) = box
            return {"out": c["x"] + c["x"]}
        """
    )
    assert check_solution(src).blocked


def test_box_for_container_blocked():
    src = _ATTACK_PREFIX + _src(
        """
        def npu_triton_impl(inputs):
            box = [inputs]
            for c in box:
                return {"out": c["x"] + c["x"]}
        """
    )
    assert check_solution(src).blocked


def test_box_mult_container_blocked():
    src = _ATTACK_PREFIX + _src(
        """
        def npu_triton_impl(inputs):
            box = [inputs] * 1
            c = box[0]
            return {"out": c["x"] + c["x"]}
        """
    )
    assert check_solution(src).blocked


def test_kwargs_unpack_blocked():
    src = _ATTACK_PREFIX + _src(
        """
        def h(z): return z + z
        def attack(x: torch.Tensor):
            return h(**{"z": x})
        """
    )
    assert check_solution(src).blocked


def test_typing_ordereddict_blocked():
    src = _ATTACK_PREFIX + _src(
        """
        from typing import OrderedDict
        def attack(inputs: OrderedDict[str, torch.Tensor]):
            return inputs["x"] + inputs["x"]
        """
    )
    assert check_solution(src).blocked


def test_typing_defaultdict_blocked():
    src = _ATTACK_PREFIX + _src(
        """
        from typing import DefaultDict
        def attack(inputs: DefaultDict[str, torch.Tensor]):
            return inputs["x"] + inputs["x"]
        """
    )
    assert check_solution(src).blocked


def test_type_alias_mutable_mapping_blocked():
    src = _ATTACK_PREFIX + _src(
        """
        from typing import MutableMapping
        TensorMap = MutableMapping[str, "torch.Tensor"]
        def attack(inputs: TensorMap):
            return inputs["x"] + inputs["x"]
        """
    )
    assert check_solution(src).blocked


def test_cross_fn_lambda_kw_blocked():
    src = _ATTACK_PREFIX + _src(
        """
        def h(z): return z + z
        def attack(x: torch.Tensor):
            return h((lambda z: z)(z=x))
        """
    )
    assert check_solution(src).blocked


def test_dev_fn_arg_cpu_blocked():
    src = _ATTACK_PREFIX + _src(
        """
        def dev(d): return d
        def attack(x: torch.Tensor):
            return x.to(dev("cpu"))
        """
    )
    assert check_solution(src).blocked


def test_lambda_starargs_idx_cpu_blocked():
    src = _ATTACK_PREFIX + _src(
        """
        def attack(x: torch.Tensor):
            return x.to((lambda *a: a[0])("cpu"))
        """
    )
    assert check_solution(src).blocked


def test_lambda_kwonly_default_cpu_blocked():
    src = _ATTACK_PREFIX + _src(
        """
        def attack(x: torch.Tensor):
            return x.to((lambda *, d="cpu": d)())
        """
    )
    assert check_solution(src).blocked


def test_lambda_kw_unpack_cpu_blocked():
    src = _ATTACK_PREFIX + _src(
        """
        def attack(x: torch.Tensor):
            return x.to((lambda d: d)(**{"d": "cpu"}))
        """
    )
    assert check_solution(src).blocked


def test_fn_kwonly_default_cpu_blocked():
    src = _ATTACK_PREFIX + _src(
        """
        def dev(*, d="cpu"): return d
        def attack(x: torch.Tensor):
            return x.to(dev())
        """
    )
    assert check_solution(src).blocked


def test_ifexp_container_blocked():
    src = _ATTACK_PREFIX + _src(
        """
        def npu_triton_impl(inputs):
            c: object = inputs if True else inputs
            return {"out": c["x"] + c["x"]}
        """
    )
    assert check_solution(src).blocked


def test_for_ifexp_container_blocked():
    src = _ATTACK_PREFIX + _src(
        """
        def npu_triton_impl(inputs):
            for c in ([inputs] if True else []):
                return {"out": c["x"] + c["x"]}
        """
    )
    assert check_solution(src).blocked


def test_for_tuple_destructure_blocked():
    src = _ATTACK_PREFIX + _src(
        """
        def npu_triton_impl(inputs):
            for (c,) in [(inputs,)]:
                return {"out": c["x"] + c["x"]}
        """
    )
    assert check_solution(src).blocked


def test_dict_kwarg_constructor_blocked():
    src = _ATTACK_PREFIX + _src(
        """
        def npu_triton_impl(inputs):
            box = dict(a=inputs)
            return {"out": box["a"]["x"] + box["a"]["x"]}
        """
    )
    assert check_solution(src).blocked


def test_decorator_tuple_idx_blocked():
    src = _ATTACK_PREFIX + _src(
        """
        def deco(fn): return lambda z: z + z
        @deco
        def h(z): return z
        def attack(x: torch.Tensor):
            g = (h,)[0]
            return g(x)
        """
    )
    assert check_solution(src).blocked


def test_decorator_dict_get_blocked():
    src = _ATTACK_PREFIX + _src(
        """
        def deco(fn): return lambda z: z + z
        @deco
        def h(z): return z
        def attack(x: torch.Tensor):
            g = {"f": h}["f"]
            return g(x)
        """
    )
    assert check_solution(src).blocked


def test_decorator_ifexp_alias_blocked():
    src = _ATTACK_PREFIX + _src(
        """
        def deco(fn): return lambda z: z + z
        @deco
        def h(z): return z
        def attack(x: torch.Tensor):
            g = h if True else h
            return g(x)
        """
    )
    assert check_solution(src).blocked


def test_cross_fn_ordereddict_blocked():
    src = _ATTACK_PREFIX + _src(
        """
        from typing import OrderedDict
        def h(inputs): return inputs["x"] + inputs["x"]
        def attack(inputs: OrderedDict[str, torch.Tensor]):
            return h(inputs)
        """
    )
    assert check_solution(src).blocked


def test_cross_fn_defaultdict_blocked():
    src = _ATTACK_PREFIX + _src(
        """
        from typing import DefaultDict
        def h(inputs): return inputs["x"] + inputs["x"]
        def attack(inputs: DefaultDict[str, torch.Tensor]):
            return h(inputs)
        """
    )
    assert check_solution(src).blocked


def test_partial_alias_blocked():
    src = _ATTACK_PREFIX + _src(
        """
        from functools import partial
        def h(z): return z + z
        def attack(x: torch.Tensor):
            p = partial
            q = p(h, x)
            return q()
        """
    )
    assert check_solution(src).blocked


def test_lambda_alias_subscript_blocked():
    src = _ATTACK_PREFIX + _src(
        """
        def attack(x: torch.Tensor):
            f = lambda z: z[0]
            return f(x)
        """
    )
    assert check_solution(src).blocked


def test_lambda_alias_kw_blocked():
    src = _ATTACK_PREFIX + _src(
        """
        def attack(x: torch.Tensor):
            f = lambda z: z + z
            return f(z=x)
        """
    )
    assert check_solution(src).blocked


def test_lambda_alias_default_cpu_blocked():
    src = _ATTACK_PREFIX + _src(
        """
        def attack(x: torch.Tensor):
            dev = lambda d="cpu": d
            return x.to(dev())
        """
    )
    assert check_solution(src).blocked


def test_nested_callable_lambda_blocked():
    src = _ATTACK_PREFIX + _src(
        """
        def attack(x: torch.Tensor):
            return (lambda f: f(x))(lambda z: z + z)
        """
    )
    assert check_solution(src).blocked


def test_nested_callable_lambda_cpu_blocked():
    src = _ATTACK_PREFIX + _src(
        """
        def attack(x: torch.Tensor):
            return x.to((lambda f: f())(lambda: "cpu"))
        """
    )
    assert check_solution(src).blocked


def test_cpu_fn_kwargs_unpack_blocked():
    src = _ATTACK_PREFIX + _src(
        """
        def dev(d): return d
        def attack(x: torch.Tensor):
            return x.to(dev(**{"d": "cpu"}))
        """
    )
    assert check_solution(src).blocked


def test_cpu_fn_vararg_blocked():
    src = _ATTACK_PREFIX + _src(
        """
        def dev(*a): return a[0]
        def attack(x: torch.Tensor):
            return x.to(dev("cpu"))
        """
    )
    assert check_solution(src).blocked


def test_starred_alias_blocked():
    src = _ATTACK_PREFIX + _src(
        """
        def deco(fn): return lambda z: z + z
        @deco
        def h(z): return z
        def attack(x: torch.Tensor):
            *_, g = (h,)
            return g(x)
        """
    )
    assert check_solution(src).blocked


def test_boolop_alias_blocked():
    src = _ATTACK_PREFIX + _src(
        """
        def deco(fn): return lambda z: z + z
        @deco
        def h(z): return z
        def attack(x: torch.Tensor):
            g = h or h
            return g(x)
        """
    )
    assert check_solution(src).blocked


def test_annassign_box_list_blocked():
    src = _ATTACK_PREFIX + _src(
        """
        def npu_triton_impl(inputs):
            box: object = [inputs]
            return {"out": box[0]["x"] + box[0]["x"]}
        """
    )
    assert check_solution(src).blocked


def test_for_dict_call_container_blocked():
    src = _ATTACK_PREFIX + _src(
        """
        def npu_triton_impl(inputs):
            for box in [dict(a=inputs)]:
                return {"out": box["a"]["x"] + box["a"]["x"]}
        """
    )
    assert check_solution(src).blocked


def test_partial_alias_chain_blocked():
    src = _ATTACK_PREFIX + _src(
        """
        from functools import partial as p
        def h(z): return z + z
        def attack(x: torch.Tensor):
            q = p
            return q(h, x)()
        """
    )
    assert check_solution(src).blocked


def test_iife_kw_lambda_blocked():
    src = _ATTACK_PREFIX + _src(
        """
        def attack(x: torch.Tensor):
            return (lambda f: f(x))(f=lambda z: z + z)
        """
    )
    assert check_solution(src).blocked


def test_iife_default_lambda_blocked():
    src = _ATTACK_PREFIX + _src(
        """
        def attack(x: torch.Tensor):
            return (lambda f=(lambda z: z + z): f(x))()
        """
    )
    assert check_solution(src).blocked


def test_subscript_boolop_blocked():
    src = _ATTACK_PREFIX + _src(
        """
        def h(z): return z + z
        def attack(x: torch.Tensor):
            g = ((h,) or (h,))[0]
            return g(x)
        """
    )
    assert check_solution(src).blocked


def test_cpu_local_alias_blocked():
    src = _ATTACK_PREFIX + _src(
        """
        def dev(d): return d
        def attack(x: torch.Tensor):
            c = "cpu"
            return x.to(dev(c))
        """
    )
    assert check_solution(src).blocked


def test_cpu_binop_concat_body_blocked():
    src = _ATTACK_PREFIX + _src(
        """
        def dev(d): return d + ""
        def attack(x: torch.Tensor):
            return x.to(dev("cpu"))
        """
    )
    assert check_solution(src).blocked


def test_iife_name_actual_blocked():
    src = _ATTACK_PREFIX + _src(
        """
        def h(z): return z + z
        def attack(x: torch.Tensor):
            return (lambda f: f(x))(h)
        """
    )
    assert check_solution(src).blocked


def test_iife_kw_unpack_dict_lambda_blocked():
    src = _ATTACK_PREFIX + _src(
        """
        def attack(x: torch.Tensor):
            return (lambda f: f(x))(**{"f": lambda z: z + z})
        """
    )
    assert check_solution(src).blocked


def test_iife_kwonly_default_lambda_blocked():
    src = _ATTACK_PREFIX + _src(
        """
        def attack(x: torch.Tensor):
            return (lambda *, f=(lambda z: z + z): f(x))()
        """
    )
    assert check_solution(src).blocked


def test_set_container_for_blocked():
    src = _ATTACK_PREFIX + _src(
        """
        def h(z): return z + z
        def attack(x: torch.Tensor):
            for g in {h}:
                return g(x)
        """
    )
    assert check_solution(src).blocked


def test_cpu_mod_format_body_blocked():
    src = _ATTACK_PREFIX + _src(
        """
        def dev(d): return "%s" % d
        def attack(x: torch.Tensor):
            c = "cpu"
            return x.to(dev(c))
        """
    )
    assert check_solution(src).blocked


def test_cpu_mult_body_blocked():
    src = _ATTACK_PREFIX + _src(
        """
        def dev(d): return d * 1
        def attack(x: torch.Tensor):
            return x.to(dev("cpu"))
        """
    )
    assert check_solution(src).blocked


def test_cpu_to_alias_blocked():
    src = _ATTACK_PREFIX + _src(
        """
        def attack(x: torch.Tensor):
            move = x.to
            return move("cpu")
        """
    )
    assert check_solution(src).blocked


def test_cpu_to_kwargs_unpack_blocked():
    src = _ATTACK_PREFIX + _src(
        """
        def attack(x: torch.Tensor):
            return x.to(**{"device": "cpu"})
        """
    )
    assert check_solution(src).blocked


def test_cpu_to_starred_blocked():
    src = _ATTACK_PREFIX + _src(
        """
        def attack(x: torch.Tensor):
            return x.to(*("cpu",))
        """
    )
    assert check_solution(src).blocked


def test_local_callable_container_for_blocked():
    src = _ATTACK_PREFIX + _src(
        """
        def h(z): return z + z
        def attack(x: torch.Tensor):
            s = (h,)
            for g in s:
                return g(x)
        """
    )
    assert check_solution(src).blocked


def test_fn_kwonly_default_tensor_blocked():
    src = _ATTACK_PREFIX + _src(
        """
        def attack(x: torch.Tensor):
            def h(*, z=x): return z + z
            return h()
        """
    )
    assert check_solution(src).blocked


def test_tensor_if_truthiness_blocked():
    src = _ATTACK_PREFIX + _src(
        """
        def attack(x: torch.Tensor):
            if x:
                return x
            return x
        """
    )
    assert check_solution(src).blocked


def test_tensor_aug_rhs_blocked():
    src = _ATTACK_PREFIX + _src(
        """
        def attack(x: torch.Tensor):
            y = 0
            y += x
            return y
        """
    )
    assert check_solution(src).blocked


def test_custom_method_alias_blocked():
    src = _ATTACK_PREFIX + _src(
        """
        class H:
            def reshape(self, z): return z + z
        def npu_triton_impl(inputs):
            a = H()
            f = a.reshape
            return {"out": f(inputs["x"])}
        """
    )
    assert check_solution(src).blocked


def test_cpu_alias_multi_fn_blocked():
    src = _ATTACK_PREFIX + _src(
        """
        def attack(x: torch.Tensor):
            move = x.to
            return move("cpu")
        def npu_triton_impl(inputs):
            return {"out": attack(inputs["x"])}
        """
    )
    assert check_solution(src).blocked


def test_subscript_cpu_to_blocked():
    src = _ATTACK_PREFIX + _src(
        """
        def npu_triton_impl(inputs):
            move = inputs["x"].to
            return {"out": move("cpu")}
        """
    )
    assert check_solution(src).blocked


def test_iife_subscript_lambda_blocked():
    src = _ATTACK_PREFIX + _src(
        """
        def npu_triton_impl(inputs):
            x = inputs["x"]
            return {"out": ({"f": lambda z: z.cpu()}["f"])(x)}
        """
    )
    assert check_solution(src).blocked


def test_lambda_ifexp_truthiness_blocked():
    src = _ATTACK_PREFIX + _src(
        """
        def npu_triton_impl(inputs):
            x = inputs["x"]
            cond = True
            return {"out": (lambda z: z.cpu() if cond else z)(x)}
        """
    )
    assert check_solution(src).blocked


def test_lambda_kwonly_default_tensor_blocked():
    src = _ATTACK_PREFIX + _src(
        """
        def npu_triton_impl(inputs):
            x = inputs["x"]
            f = lambda *, z=x: z.cpu()
            return {"out": f()}
        """
    )
    assert check_solution(src).blocked


def test_destructure_to_cpu_blocked():
    src = _ATTACK_PREFIX + _src(
        """
        def npu_triton_impl(inputs):
            a = inputs["x"]
            (f,) = (a.to,)
            return {"out": f("cpu")}
        """
    )
    assert check_solution(src).blocked


def test_destructure_torch_empty_blocked():
    src = _ATTACK_PREFIX + _src(
        """
        def npu_triton_impl(inputs):
            (f,) = (torch.empty,)
            return {"out": f(5, device="cpu")}
        """
    )
    assert check_solution(src).blocked


def test_starred_destructure_blocked():
    src = _ATTACK_PREFIX + _src(
        """
        def npu_triton_impl(inputs):
            a = inputs["x"]
            (f, *rest) = (a.to,)
            return {"out": f("cpu")}
        """
    )
    assert check_solution(src).blocked


def test_nested_destructure_blocked():
    src = _ATTACK_PREFIX + _src(
        """
        def npu_triton_impl(inputs):
            a = inputs["x"]
            ((f,),) = ((a.to,),)
            return {"out": f("cpu")}
        """
    )
    assert check_solution(src).blocked


def test_annassign_torch_empty_blocked():
    src = _ATTACK_PREFIX + _src(
        """
        def npu_triton_impl(inputs):
            f: object = torch.empty
            return {"out": f(5, device="cpu")}
        """
    )
    assert check_solution(src).blocked


def test_pinned_ifexp_callable_blocked():
    src = _ATTACK_PREFIX + _src(
        """
        def npu_triton_impl(inputs):
            return {"out": (torch.empty if True else torch.zeros)(5, device="cpu")}
        """
    )
    assert check_solution(src).blocked


def test_subscript_custom_method_blocked():
    src = _ATTACK_PREFIX + _src(
        """
        class H:
            def reshape(self, z): return z
        def npu_triton_impl(inputs):
            a = H()
            g = (a.reshape,)[0]
            return {"out": g(inputs["x"])}
        """
    )
    assert check_solution(src).blocked


def test_ifexp_custom_method_blocked():
    src = _ATTACK_PREFIX + _src(
        """
        class H:
            def reshape(self, z): return z
            def view(self, z): return z
        def npu_triton_impl(inputs):
            a = H()
            b = H()
            g = a.reshape if True else b.view
            return {"out": g(inputs["x"])}
        """
    )
    assert check_solution(src).blocked


def test_annassign_custom_method_blocked():
    src = _ATTACK_PREFIX + _src(
        """
        class H:
            def reshape(self, z): return z
        def npu_triton_impl(inputs):
            a = H()
            f: object = a.reshape
            return {"out": f(inputs["x"])}
        """
    )
    assert check_solution(src).blocked


def test_module_custom_method_blocked():
    src = _ATTACK_PREFIX + _src(
        """
        class H:
            def reshape(self, z): return z
        a = H()
        f = a.reshape
        def npu_triton_impl(inputs):
            return {"out": f(inputs["x"])}
        """
    )
    assert check_solution(src).blocked


def test_bound_dict_kwargs_blocked():
    src = _ATTACK_PREFIX + _src(
        """
        def npu_triton_impl(inputs):
            kw = {"device": "cpu"}
            return {"out": torch.empty(5, **kw)}
        """
    )
    assert check_solution(src).blocked


def test_dict_call_kwargs_blocked():
    src = _ATTACK_PREFIX + _src(
        """
        def npu_triton_impl(inputs):
            return {"out": torch.empty(5, **dict(device="cpu"))}
        """
    )
    assert check_solution(src).blocked


def test_nested_subscript_kwargs_blocked():
    src = _ATTACK_PREFIX + _src(
        """
        def npu_triton_impl(inputs):
            outer = {"x": {"device": "cpu"}}
            return {"out": torch.empty(5, **outer["x"])}
        """
    )
    assert check_solution(src).blocked


def test_call_returns_callable_blocked():
    src = _ATTACK_PREFIX + _src(
        """
        def npu_triton_impl(inputs):
            return {"out": (lambda: torch.empty)()(5, device="cpu")}
        """
    )
    assert check_solution(src).blocked


def test_fn_returns_callable_blocked():
    src = _ATTACK_PREFIX + _src(
        """
        def pick():
            return torch.empty
        def npu_triton_impl(inputs):
            return {"out": pick()(5, device="cpu")}
        """
    )
    assert check_solution(src).blocked


def test_box_subscript_inline_blocked():
    src = _ATTACK_PREFIX + _src(
        """
        def npu_triton_impl(inputs):
            box = {"f": torch.empty}
            return {"out": box["f"](5, device="cpu")}
        """
    )
    assert check_solution(src).blocked


def test_box_subscript_to_name_blocked():
    src = _ATTACK_PREFIX + _src(
        """
        def npu_triton_impl(inputs):
            box = {"f": torch.empty}
            f = box["f"]
            return {"out": f(5, device="cpu")}
        """
    )
    assert check_solution(src).blocked


def test_dict_call_binding_blocked():
    src = _ATTACK_PREFIX + _src(
        """
        def npu_triton_impl(inputs):
            kw = dict(device="cpu")
            return {"out": torch.empty(5, **kw)}
        """
    )
    assert check_solution(src).blocked


def test_dict_index_mutation_blocked():
    src = _ATTACK_PREFIX + _src(
        """
        def npu_triton_impl(inputs):
            kw = {}
            kw["device"] = "cpu"
            return {"out": torch.empty(5, **kw)}
        """
    )
    assert check_solution(src).blocked


def test_dict_or_mutation_blocked():
    src = _ATTACK_PREFIX + _src(
        """
        def npu_triton_impl(inputs):
            kw = {}
            kw |= {"device": "cpu"}
            return {"out": torch.empty(5, **kw)}
        """
    )
    assert check_solution(src).blocked


def test_starred_tail_destructure_blocked():
    src = _ATTACK_PREFIX + _src(
        """
        def npu_triton_impl(inputs):
            (*rest,) = (torch.empty,)
            return {"out": rest[0](5, device="cpu")}
        """
    )
    assert check_solution(src).blocked


def test_mkdev_alias_blocked():
    src = _ATTACK_PREFIX + _src(
        """
        def npu_triton_impl(inputs):
            x = inputs["x"]
            mkdev = torch.device
            return {"out": x.to(mkdev("cpu"))}
        """
    )
    assert check_solution(src).blocked


def test_str_alias_device_blocked():
    src = _ATTACK_PREFIX + _src(
        """
        def npu_triton_impl(inputs):
            s = str
            return {"out": torch.empty(5, device=s("cpu"))}
        """
    )
    assert check_solution(src).blocked


def test_str_concat_device_blocked():
    src = _ATTACK_PREFIX + _src(
        """
        def npu_triton_impl(inputs):
            c = "c"
            pu = "pu"
            return {"out": torch.empty(5, device=c + pu)}
        """
    )
    assert check_solution(src).blocked


def test_alias_chain_device_blocked():
    src = _ATTACK_PREFIX + _src(
        """
        def npu_triton_impl(inputs):
            x = inputs["x"]
            a = torch.device
            b = a
            return {"out": x.to(b("cpu"))}
        """
    )
    assert check_solution(src).blocked


def test_module_alias_chain_blocked():
    src = _ATTACK_PREFIX + _src(
        """
        def npu_triton_impl(inputs):
            x = inputs["x"]
            a = torch
            dev = a.device
            return {"out": x.to(dev("cpu"))}
        """
    )
    assert check_solution(src).blocked


def test_nested_lambda_iife_blocked():
    src = _ATTACK_PREFIX + _src(
        """
        def npu_triton_impl(inputs):
            return {"out": ((lambda: (lambda: torch.empty))()())(5, device="cpu")}
        """
    )
    assert check_solution(src).blocked


def test_fn_returns_lambda_blocked():
    src = _ATTACK_PREFIX + _src(
        """
        def pick():
            return lambda *a, **kw: torch.empty(*a, **kw)
        def npu_triton_impl(inputs):
            return {"out": pick()(5, device="cpu")}
        """
    )
    assert check_solution(src).blocked


def test_nested_lambda_device_blocked():
    src = _ATTACK_PREFIX + _src(
        """
        def npu_triton_impl(inputs):
            x = inputs["x"]
            return {"out": x.to((lambda: torch.device)()("cpu"))}
        """
    )
    assert check_solution(src).blocked


def test_helper_alloc_cpu_blocked():
    src = _ATTACK_PREFIX + _src(
        """
        def alloc(d):
            return torch.empty(5, device=d)
        def npu_triton_impl(inputs):
            return {"out": alloc("cpu")}
        """
    )
    assert check_solution(src).blocked


def test_helper_move_cpu_blocked():
    src = _ATTACK_PREFIX + _src(
        """
        def move(x, d):
            return x.to(d)
        def npu_triton_impl(inputs):
            return {"out": move(inputs["x"], "cpu")}
        """
    )
    assert check_solution(src).blocked


def test_iife_helper_to_kw_blocked():
    src = _ATTACK_PREFIX + _src(
        """
        def npu_triton_impl(inputs):
            return {"out": (lambda x, **kw: x.to(**kw))(inputs["x"], device="cpu")}
        """
    )
    assert check_solution(src).blocked


def test_iife_helper_empty_kw_blocked():
    src = _ATTACK_PREFIX + _src(
        """
        def npu_triton_impl(inputs):
            return {"out": (lambda *a, **kw: torch.empty(*a, **kw))(5, device="cpu")}
        """
    )
    assert check_solution(src).blocked


def test_bound_tuple_starred_blocked():
    src = _ATTACK_PREFIX + _src(
        """
        def npu_triton_impl(inputs):
            box = (torch.empty,)
            (*rest,) = box
            return {"out": rest[0](5, device="cpu")}
        """
    )
    assert check_solution(src).blocked


def test_starred_args_bound_blocked():
    src = _ATTACK_PREFIX + _src(
        """
        def npu_triton_impl(inputs):
            x = inputs["x"]
            args = ("cpu",)
            return {"out": x.to(*args)}
        """
    )
    assert check_solution(src).blocked


def test_subscript_tuple_kwargs_blocked():
    src = _ATTACK_PREFIX + _src(
        """
        def npu_triton_impl(inputs):
            box = ({"device": "cpu"},)
            return {"out": torch.empty(5, **box[0])}
        """
    )
    assert check_solution(src).blocked


def test_transitive_helper_blocked():
    src = _ATTACK_PREFIX + _src(
        """
        def B(d):
            return torch.empty(5, device=d)
        def A(d):
            return B(d)
        def npu_triton_impl(inputs):
            return {"out": A("cpu")}
        """
    )
    assert check_solution(src).blocked


def test_closure_helper_blocked():
    src = _ATTACK_PREFIX + _src(
        """
        def maker():
            def inner(d):
                return torch.empty(5, device=d)
            return inner
        def npu_triton_impl(inputs):
            return {"out": maker()("cpu")}
        """
    )
    assert check_solution(src).blocked


def test_class_method_helper_blocked():
    src = _ATTACK_PREFIX + _src(
        """
        class H:
            def reshape(self, d):
                return torch.empty(5, device=d)
        def npu_triton_impl(inputs):
            h = H()
            return {"out": h.reshape(d="cpu")}
        """
    )
    assert check_solution(src).blocked


def test_lambda_positional_default_cpu_blocked():
    src = _ATTACK_PREFIX + _src(
        """
        def npu_triton_impl(inputs):
            x = inputs["x"]
            return {"out": (lambda x, d="cpu": x.to(d))(x)}
        """
    )
    assert check_solution(src).blocked


def test_pinned_lambda_kwonly_default_cpu_blocked():
    src = _ATTACK_PREFIX + _src(
        """
        def npu_triton_impl(inputs):
            x = inputs["x"]
            return {"out": (lambda x, *, d="cpu": x.to(d))(x)}
        """
    )
    assert check_solution(src).blocked


def test_boolop_cpu_blocked():
    src = _ATTACK_PREFIX + _src(
        """
        def npu_triton_impl(inputs):
            x = inputs["x"]
            return {"out": x.to("" or "cpu")}
        """
    )
    assert check_solution(src).blocked


def test_dict_name_alias_blocked():
    src = _ATTACK_PREFIX + _src(
        """
        def npu_triton_impl(inputs):
            kw1 = {"device": "cpu"}
            kw2 = kw1
            return {"out": torch.empty(5, **kw2)}
        """
    )
    assert check_solution(src).blocked


def test_tuple_destructure_module_alias_blocked():
    src = _ATTACK_PREFIX + _src(
        """
        def npu_triton_impl(inputs):
            x = inputs["x"]
            (a,) = (torch,)
            dev = a.device
            return {"out": x.to(dev("cpu"))}
        """
    )
    assert check_solution(src).blocked


def test_method_transitive_blocked():
    src = _ATTACK_PREFIX + _src(
        """
        def B(d):
            return torch.empty(5, device=d)
        class H:
            def reshape(self, d):
                return B(d)
        def npu_triton_impl(inputs):
            h = H()
            return {"out": h.reshape(d="cpu")}
        """
    )
    assert check_solution(src).blocked


def test_iife_calls_helper_blocked():
    src = _ATTACK_PREFIX + _src(
        """
        def B(d):
            return torch.empty(5, device=d)
        def npu_triton_impl(inputs):
            return {"out": (lambda d: B(d))("cpu")}
        """
    )
    assert check_solution(src).blocked


def test_iife_calls_helper_default_blocked():
    src = _ATTACK_PREFIX + _src(
        """
        def B(d):
            return torch.empty(5, device=d)
        def npu_triton_impl(inputs):
            return {"out": (lambda d="cpu": B(d))()}
        """
    )
    assert check_solution(src).blocked


def test_vararg_taint_blocked():
    src = _ATTACK_PREFIX + _src(
        """
        def move(x, *a):
            return x.to(*a)
        def npu_triton_impl(inputs):
            return {"out": move(inputs["x"], "cpu")}
        """
    )
    assert check_solution(src).blocked


def test_kwarg_taint_blocked():
    src = _ATTACK_PREFIX + _src(
        """
        def alloc(**kw):
            return torch.empty(5, **kw)
        def npu_triton_impl(inputs):
            return {"out": alloc(device="cpu")}
        """
    )
    assert check_solution(src).blocked


def test_helper_alias_direct_blocked():
    src = _ATTACK_PREFIX + _src(
        """
        def B(d):
            return torch.empty(5, device=d)
        def npu_triton_impl(inputs):
            f = B
            return {"out": f("cpu")}
        """
    )
    assert check_solution(src).blocked


def test_helper_alias_subscript_blocked():
    src = _ATTACK_PREFIX + _src(
        """
        def B(d):
            return torch.empty(5, device=d)
        def npu_triton_impl(inputs):
            f = (B,)[0]
            return {"out": f("cpu")}
        """
    )
    assert check_solution(src).blocked


def test_multi_closure_blocked():
    src = _ATTACK_PREFIX + _src(
        """
        def maker():
            def inner(d):
                return torch.empty(5, device=d)
            return inner
        def factory():
            return maker()
        def npu_triton_impl(inputs):
            return {"out": factory()("cpu")}
        """
    )
    assert check_solution(src).blocked


def test_dict_unpack_in_dict_blocked():
    src = _ATTACK_PREFIX + _src(
        """
        def npu_triton_impl(inputs):
            return {"out": torch.empty(5, **{**{"device":"cpu"}})}
        """
    )
    assert check_solution(src).blocked


def test_dict_positional_blocked():
    src = _ATTACK_PREFIX + _src(
        """
        def npu_triton_impl(inputs):
            kw = dict({"device":"cpu"})
            return {"out": torch.empty(5, **kw)}
        """
    )
    assert check_solution(src).blocked


def test_dict_bitor_blocked():
    src = _ATTACK_PREFIX + _src(
        """
        def npu_triton_impl(inputs):
            kw = {} | {"device": "cpu"}
            return {"out": torch.empty(5, **kw)}
        """
    )
    assert check_solution(src).blocked


def test_fn_default_with_actual_blocked():
    src = _ATTACK_PREFIX + _src(
        """
        def dev(d="npu"):
            return d
        def npu_triton_impl(inputs):
            x = inputs["x"]
            return {"out": x.to(dev("cpu"))}
        """
    )
    assert check_solution(src).blocked


def test_tuple_subscript_torch_blocked():
    src = _ATTACK_PREFIX + _src(
        """
        def npu_triton_impl(inputs):
            x = inputs["x"]
            a = (torch,)[0]
            dev = a.device
            return {"out": x.to(dev("cpu"))}
        """
    )
    assert check_solution(src).blocked


def test_legit_input_param_name_allowed():
    """`input` is a common parameter name and must NOT be blocked
    just because it shadows the Python `input()` builtin."""
    src = _src(
        """
        import torch
        import triton
        import triton.language as tl

        @triton.jit
        def k(x_ptr, BLOCK: tl.constexpr):
            tl.load(x_ptr); tl.store(x_ptr, 0)

        def host(input: torch.Tensor):
            return input.contiguous()
        """
    )
    report = check_solution(src)
    assert not report.blocked, [f"{f.layer}:{f.message}" for f in report.findings]


def test_tl_inline_asm_blocked():
    """tl.inline_asm_elementwise must reject (was warn)."""
    src = _src(
        """
        import torch
        import triton
        import triton.language as tl

        @triton.jit
        def k(x_ptr, BLOCK: tl.constexpr):
            x = tl.load(x_ptr)
            y = tl.inline_asm_elementwise("nop", "=r,r", [x], dtype=tl.float32)
            tl.store(x_ptr, y)
        """
    )
    report = check_solution(src)
    assert report.blocked


def test_tensor_cpu_method_blocked():
    """Layer D: x.cpu() / x.numpy() must be rejected."""
    src = _src(
        """
        import torch
        import triton
        import triton.language as tl

        @triton.jit
        def k(x_ptr, BLOCK: tl.constexpr):
            tl.load(x_ptr); tl.store(x_ptr, 0)

        def helper(x):
            return x.cpu().numpy()
        """
    )
    report = check_solution(src)
    assert report.blocked
    assert any(f.layer == "D-attr-deny" for f in report.findings)


def test_non_allowlisted_module_blocks():
    """An import not in any list and not in denylist still blocks."""
    src = VALID_BASE + "\nimport jinja2\n"
    report = check_solution(src)
    assert report.blocked
    assert any(
        f.layer == "A-non-allowlist" or "non-allowlisted" in f.message.lower()
        for f in report.findings
    )


def test_required_op_cube_present():
    src = _src(
        """
        import torch
        import triton
        import triton.language as tl

        @triton.jit
        def k(a, b, c, BLOCK: tl.constexpr):
            x = tl.load(a)
            y = tl.load(b)
            z = tl.dot(x, y)
            tl.store(c, z)
        """
    )
    report = check_solution(src, required_checks=["cube_arithmetic_present"])
    # tl.dot is present → no F-required-op finding.
    assert not any(f.layer == "F-required-op" for f in report.findings), [
        f.message for f in report.findings
    ]


def test_required_op_cube_missing():
    # No tl.dot used.
    report = check_solution(VALID_BASE, required_checks=["cube_arithmetic_present"])
    assert report.blocked
    assert any(f.layer == "F-required-op" for f in report.findings)


@pytest.mark.parametrize("task_id", [
    "L1_01_bias",
    "L1_12_rms_norm",
    "L2_04_silu_and_mul",
    "L2_17_w8a8_block_int8_matmul",
    "L3_16_fwd",
])
def test_real_poc_kernels_pass_static(task_id):
    """Every PoC task's reference kernel.py must pass static checks."""
    from ascendbench.corpus.dataset import load_task

    task = load_task(task_id)
    required = task.meta.get("runner_checks", {}).get("static_checks", [])
    # The static_checks list in meta.yaml is by name; we only enforce ones
    # we have rules for. Strip the ones not relevant to the kernel itself
    # (those are runner-side).
    enforced = [c for c in required if c in {
        "cube_arithmetic_present",
        "online_softmax_present",
    }]
    # The Phase A' synthesised entrypoint (HOLE B) appended below the
    # marker is bench-generated scaffolding (reflection / dict-comp
    # helpers), not candidate-authored kernel code -- production static-
    # checks the candidate solution.py, never this vendored reference.
    # Check only the kernel body the candidate actually writes.
    kernel_src = task.kernel_py
    idx = kernel_src.find("# ── Phase A' synthesised standalone entrypoint")
    if idx != -1:
        kernel_src = kernel_src[:idx]
    report = check_solution(kernel_src, required_checks=enforced)
    assert not report.blocked, (
        f"{task_id} kernel.py blocked: "
        + ", ".join(f"{f.layer}:{f.message}" for f in report.findings)
    )
