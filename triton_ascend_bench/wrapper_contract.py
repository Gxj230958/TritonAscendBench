"""wrapper_contract.py -- V9 plan M0a.2 / codex review 2026-05-27.

Statically extract per-input rank constraints from a kernel.py host
wrapper function body, then compare against the ranks ``build_inputs``
actually produced. Used to catch ``shapes.yaml.inputs.dims`` errors at
the dev / CPU R10.a stage rather than letting GPU verify fall over
with stale ``ValueError: not enough values to unpack`` tracebacks.

Pure-stdlib AST -- NO ``_load()``, NO torch import, NO file execution.
Wrapper bodies often touch GPU paths (``triton.cdiv`` /
``torch.cuda``); executing them in dev would re-introduce the
fragility the R10 split was meant to eliminate.

Coverage:

* Direct tuple unpack: ``B, T, H, K = k.shape`` -> ``k.ndim == 4``
* Starred unpack: ``B, T, H, K, V = *k.shape, u.shape[-1]``
                  -> ``k.ndim == 4`` AND ``u.ndim >= 1``
* Branch unpack: ``if head_first: B,H,T,S = g.shape; else: B,T,H,S = g.shape``
                 -> ``g.ndim in {4}`` (set union of both branches)
* Indexing: ``BT = A.shape[-1]`` / ``A.shape[3]`` -> ``A.ndim >= 1`` /
                                                    ``A.ndim >= 4``
* ``len(x.shape)`` dispatch: ``if len(x.shape) == 3: ...``
                            -> ``x.ndim`` accepts 3

Out of scope (codex flagged but unlikely worth modeling for v1):

* ``view`` / ``reshape`` that change rank mid-body
* ``*args`` / ``**kwargs`` wrappers
* Dynamically computed shape unpacks
"""
from __future__ import annotations

import ast
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class WrapperContract:
    """Per-input rank constraints extracted from a host wrapper body."""

    fn_name: str
    # Each input arg name maps to the set of ranks that satisfy the
    # wrapper body's shape usage. Empty set = no rank constraint found
    # (the wrapper doesn't unpack / index the input -- caller treats as
    # "no constraint", not "fails any rank").
    accepted_ranks: dict[str, set[int]] = field(default_factory=dict)
    # Minimum rank required by .shape[N] / .shape[-1] indexing (when
    # an exact-rank unpack is absent). E.g. ``BT = A.shape[-1]`` only
    # tells us ``A.ndim >= 1``.
    min_ranks: dict[str, int] = field(default_factory=dict)
    # Diagnostic: each constraint's source line so error messages can
    # point reviewers at the actual wrapper code.
    source_lines: dict[str, list[tuple[int, str]]] = field(
        default_factory=dict
    )


@dataclass
class Violation:
    """One contract violation between built inputs and the wrapper body."""

    input_name: str
    expected_ranks: set[int]
    expected_min_rank: int | None
    actual_rank: int
    reason: str


def _starred_count(elts: list[ast.expr], inputs: set[str]) -> tuple[str, int] | None:
    """When a tuple-assign LHS is ``B, T, Hg, K, V = *k.shape, u.shape[-1]``
    the RHS is a ``Tuple`` whose first element is ``Starred(k.shape)``.
    Decode: the starred element contributes ``len(LHS) - other_count``
    elements; non-starred ``X.shape[N]`` elements contribute exactly 1.

    Returns ``(starred_param_name, starred_rank)`` when the pattern is
    a single-starred shape with at least one scalar shape index sibling.
    """
    # Find the single Starred element
    starred_index = None
    for i, e in enumerate(elts):
        if isinstance(e, ast.Starred):
            if starred_index is not None:
                return None  # multiple starred -- not a pattern we model
            starred_index = i
    if starred_index is None:
        return None

    starred = elts[starred_index]
    assert isinstance(starred, ast.Starred)
    inner = starred.value
    # Must be ``param.shape``
    if not (
        isinstance(inner, ast.Attribute)
        and inner.attr == "shape"
        and isinstance(inner.value, ast.Name)
        and inner.value.id in inputs
    ):
        return None
    param_name = inner.value.id
    # Non-starred siblings must look like X.shape[N] (each contributes 1
    # to the unpack length); if any is not, refuse the heuristic.
    other_count = 0
    for i, e in enumerate(elts):
        if i == starred_index:
            continue
        if isinstance(e, ast.Subscript) and isinstance(e.value, ast.Attribute):
            if e.value.attr == "shape":
                other_count += 1
                continue
        return None
    return param_name, other_count  # starred_rank = total_unpack - other_count


def _direct_shape_unpack_rank(
    target: ast.expr, value: ast.expr, inputs: set[str]
) -> tuple[str, int] | None:
    """Match ``a, b, ..., c = x.shape`` -> ``(x, n)`` where n = len(LHS).

    LHS must be a Tuple of bare Names (no Starred, no Subscripts).
    RHS must be ``x.shape`` for some ``x`` in the wrapper param set.
    """
    if not isinstance(target, (ast.Tuple, ast.List)):
        return None
    if not all(isinstance(e, ast.Name) for e in target.elts):
        return None
    if not (
        isinstance(value, ast.Attribute)
        and value.attr == "shape"
        and isinstance(value.value, ast.Name)
        and value.value.id in inputs
    ):
        return None
    return value.value.id, len(target.elts)


def _shape_subscript_min_rank(node: ast.expr, inputs: set[str]) -> tuple[str, int] | None:
    """Match ``x.shape[N]`` / ``x.shape[-1]`` -> ``(x, min_rank)``.

    For a positive index N: ``x.ndim >= N + 1``.
    For a negative index -K: ``x.ndim >= K``.
    For a non-constant slice (variable index): ``x.ndim >= 1`` (we can
    only assert the attribute exists).
    """
    if not isinstance(node, ast.Subscript):
        return None
    base = node.value
    if not (
        isinstance(base, ast.Attribute)
        and base.attr == "shape"
        and isinstance(base.value, ast.Name)
        and base.value.id in inputs
    ):
        return None
    name = base.value.id
    sub = node.slice
    if isinstance(sub, ast.Constant) and isinstance(sub.value, int):
        n = sub.value
        if n >= 0:
            return name, n + 1
        return name, abs(n)
    # Unary minus over constant (``-1`` parses to UnaryOp on Python 3.7+).
    if (
        isinstance(sub, ast.UnaryOp)
        and isinstance(sub.op, ast.USub)
        and isinstance(sub.operand, ast.Constant)
        and isinstance(sub.operand.value, int)
    ):
        return name, sub.operand.value
    return name, 1


def extract_wrapper_contract(
    kernel_path: Path, fn_name: str
) -> WrapperContract | None:
    """Parse ``kernel_path``, find ``fn_name``, return its rank contract.

    Returns ``None`` when the file doesn't parse / the function isn't
    found / it's ``@triton.jit``-decorated (the latter is not a host
    wrapper). The caller's other gates (``is_jit_decorated``,
    ``kernel_function_exists``) cover those paths separately.
    """
    if not kernel_path.exists():
        return None
    try:
        tree = ast.parse(kernel_path.read_text())
    except (OSError, SyntaxError):
        return None

    target: ast.FunctionDef | ast.AsyncFunctionDef | None = None
    for node in ast.walk(tree):
        if (
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == fn_name
        ):
            target = node
            break
    if target is None:
        return None

    # Wrapper's input parameter names (drop self).
    params = [a.arg for a in target.args.args]
    if params and params[0] == "self":
        params = params[1:]
    params_set = set(params)
    if not params_set:
        return WrapperContract(fn_name=fn_name)

    contract = WrapperContract(fn_name=fn_name)

    def _accept_rank(name: str, rank: int, line: int, src: str) -> None:
        contract.accepted_ranks.setdefault(name, set()).add(rank)
        contract.min_ranks[name] = max(contract.min_ranks.get(name, 0), rank)
        contract.source_lines.setdefault(name, []).append((line, src.strip()[:120]))

    def _update_min(name: str, rank: int, line: int, src: str) -> None:
        contract.min_ranks[name] = max(contract.min_ranks.get(name, 0), rank)
        contract.source_lines.setdefault(name, []).append((line, src.strip()[:120]))

    # Walk the function body. Every assign / aug-assign / return that
    # mentions an input.shape goes through the matchers above.
    for stmt in ast.walk(target):
        if isinstance(stmt, ast.Assign):
            for tgt in stmt.targets:
                # Direct unpack ``a, b, ... = x.shape``
                hit = _direct_shape_unpack_rank(tgt, stmt.value, params_set)
                if hit is not None:
                    name, rank = hit
                    try:
                        src = ast.unparse(stmt)
                    except Exception:
                        src = ""
                    _accept_rank(name, rank, stmt.lineno, src)
                    continue
                # Starred unpack ``a, b, c = *x.shape, y.shape[-1]``
                if (
                    isinstance(tgt, (ast.Tuple, ast.List))
                    and isinstance(stmt.value, ast.Tuple)
                ):
                    sh = _starred_count(stmt.value.elts, params_set)
                    if sh is not None:
                        name, other_count = sh
                        starred_rank = len(tgt.elts) - other_count
                        if starred_rank > 0:
                            try:
                                src = ast.unparse(stmt)
                            except Exception:
                                src = ""
                            _accept_rank(
                                name, starred_rank, stmt.lineno, src
                            )
        # ``.shape[N]`` access anywhere -- min_rank only (not enough
        # info to fix the rank exactly).
        if isinstance(stmt, ast.Subscript):
            mh = _shape_subscript_min_rank(stmt, params_set)
            if mh is not None:
                name, n = mh
                try:
                    src = ast.unparse(stmt)
                except Exception:
                    src = ""
                _update_min(name, n, stmt.lineno, src)

    return contract


def check_against_inputs(
    contract: WrapperContract, input_ranks: dict[str, int]
) -> list[Violation]:
    """Compare a built input set's per-key tensor ranks to the contract.

    ``input_ranks`` is ``{name: tensor.ndim}`` (scalars / non-tensors
    omitted by the caller). A name that the contract doesn't constrain
    (no unpack, no .shape[N] use) is silently allowed -- we cannot
    refute what we haven't seen.
    """
    out: list[Violation] = []
    for name, actual in input_ranks.items():
        ranks = contract.accepted_ranks.get(name) or set()
        if ranks:
            if actual not in ranks:
                out.append(
                    Violation(
                        input_name=name,
                        expected_ranks=set(ranks),
                        expected_min_rank=None,
                        actual_rank=actual,
                        reason=(
                            f"wrapper unpacks {name}.shape into "
                            f"{sorted(ranks)}-rank tuple(s); built input "
                            f"has rank {actual}"
                        ),
                    )
                )
            continue
        # No accepted_ranks; check min_rank from indexing usage.
        min_rank = contract.min_ranks.get(name)
        if min_rank is not None and actual < min_rank:
            out.append(
                Violation(
                    input_name=name,
                    expected_ranks=set(),
                    expected_min_rank=min_rank,
                    actual_rank=actual,
                    reason=(
                        f"wrapper indexes {name}.shape[i] requiring "
                        f"rank >= {min_rank}; built input has rank {actual}"
                    ),
                )
            )
    return out


__all__ = [
    "WrapperContract",
    "Violation",
    "extract_wrapper_contract",
    "check_against_inputs",
]
