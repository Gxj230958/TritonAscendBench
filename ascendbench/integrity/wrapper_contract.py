"""wrapper_contract.py

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

Out of scope (unlikely worth modeling for v1):

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
    # (auto_patch_shapes): for each input, the list of dim-name
    # tuples observed in *direct* (non-starred) ``a, b, c = x.shape``
    # unpacks. Each tuple is the LHS variable name sequence in source
    # order, e.g. ``B, T, H, K = k.shape`` -> ``('B', 'T', 'H', 'K')``.
    # The auto-patch tool reads the first entry as the canonical
    # ``shapes.yaml.inputs[name].dims`` value. Branch unpack populates
    # multiple entries; starred unpack contributes nothing here (the
    # starred slot has no source-level name).
    dim_names: dict[str, list[tuple[str, ...]]] = field(default_factory=dict)


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
) -> tuple[str, int, tuple[str, ...]] | None:
    """Match ``a, b, ..., c = x.shape`` -> ``(x, n, names)``.

    ``n`` = ``len(LHS)``; ``names`` = source-order tuple of the LHS
    Name ids (e.g. ``('B', 'T', 'H', 'K')``).

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
    names = tuple(e.id for e in target.elts)
    return value.value.id, len(target.elts), names


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


# Tensor methods that produce an alias of the same rank as the
# receiver. Oracle helpers commonly do ``kf = k.to(fp32)`` then unpack
# ``kf.shape`` -- without alias tracking we'd miss that contract entirely.
_RANK_PRESERVING_METHODS: frozenset[str] = frozenset({
    "to", "contiguous", "detach", "clone", "cpu", "cuda",
    "float", "half", "double", "bfloat16", "int", "long", "byte",
    "type_as", "requires_grad_", "pin_memory",
})


def _is_rank_preserving_alias(
    value: ast.expr, params: set[str]
) -> str | None:
    """Match ``x.METHOD(...)`` where METHOD preserves rank. Returns the
    aliased param name, or ``None`` if value is not such a call."""
    if not isinstance(value, ast.Call):
        return None
    func = value.func
    if not (isinstance(func, ast.Attribute) and func.attr in _RANK_PRESERVING_METHODS):
        return None
    base = func.value
    if isinstance(base, ast.Name) and base.id in params:
        return base.id
    return None


def _is_genexp_alias_unpack(
    value: ast.expr, params: set[str]
) -> list[str] | None:
    """Match ``(t.METHOD(...) for t in (a, b, c))`` -- the common oracle
    pattern ``kf, wf, uf = (t.to(fp32) for t in (k, w, u))``.

    Returns the list of param names in iteration order (so the caller
    can zip them with the multi-target tuple LHS), or ``None``.
    """
    if not isinstance(value, ast.GeneratorExp):
        return None
    elt = value.elt
    if not isinstance(elt, ast.Call):
        return None
    func = elt.func
    if not (
        isinstance(func, ast.Attribute)
        and func.attr in _RANK_PRESERVING_METHODS
    ):
        return None
    loop_base = func.value
    if not isinstance(loop_base, ast.Name):
        return None
    if len(value.generators) != 1:
        return None
    gen = value.generators[0]
    if not (
        isinstance(gen.target, ast.Name)
        and gen.target.id == loop_base.id
    ):
        return None
    iter_node = gen.iter
    if not isinstance(iter_node, (ast.Tuple, ast.List)):
        return None
    names: list[str] = []
    for e in iter_node.elts:
        if not (isinstance(e, ast.Name) and e.id in params):
            return None
        names.append(e.id)
    return names


def _lhs_starred_unpack_min_rank(
    target: ast.expr, value: ast.expr, inputs: set[str]
) -> tuple[str, int] | None:
    """Match LHS starred unpack ``*lead, n, _ = x.shape``.

    Returns ``(x, min_rank)`` where ``min_rank`` = number of
    non-starred fixed slots. ``*lead, n, _ = Af.shape`` means
    ``Af.ndim >= 2`` (the starred slot can absorb 0 elements).
    """
    if not isinstance(target, (ast.Tuple, ast.List)):
        return None
    starred = [e for e in target.elts if isinstance(e, ast.Starred)]
    if len(starred) != 1:
        return None
    if not (
        isinstance(value, ast.Attribute)
        and value.attr == "shape"
        and isinstance(value.value, ast.Name)
        and value.value.id in inputs
    ):
        return None
    fixed_count = sum(
        1 for e in target.elts if not isinstance(e, ast.Starred)
    )
    return value.value.id, fixed_count


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

    # ``alias_map``: name -> originating param via rank-preserving call
    # (``kf = k.to(fp32)``) or generator-exp alias unpack
    # (``kf,wf,uf = (t.to(fp32) for t in (k,w,u))``). Built in a
    # pre-pass so subsequent .shape unpacks on ``kf`` are credited to
    # the original param ``k``.
    alias_map: dict[str, str] = {p: p for p in params_set}
    for stmt in ast.walk(target):
        if not isinstance(stmt, ast.Assign):
            continue
        if len(stmt.targets) != 1:
            continue
        tgt = stmt.targets[0]
        # Simple alias: ``kf = k.to(fp32)``
        base = _is_rank_preserving_alias(stmt.value, set(alias_map))
        if base is not None and isinstance(tgt, ast.Name):
            alias_map[tgt.id] = alias_map[base]
            continue
        # Generator-exp alias unpack: ``kf,wf,uf = (t.to(...) for t in (k,w,u))``
        if isinstance(tgt, (ast.Tuple, ast.List)):
            genexp_bases = _is_genexp_alias_unpack(stmt.value, set(alias_map))
            if genexp_bases is not None and len(genexp_bases) == len(tgt.elts):
                for alias_node, base_name in zip(tgt.elts, genexp_bases):
                    if isinstance(alias_node, ast.Name):
                        alias_map[alias_node.id] = alias_map[base_name]
    aliased_inputs = set(alias_map)

    contract = WrapperContract(fn_name=fn_name)

    def _accept_rank(
        name: str,
        rank: int,
        line: int,
        src: str,
        names: tuple[str, ...] | None = None,
    ) -> None:
        contract.accepted_ranks.setdefault(name, set()).add(rank)
        contract.min_ranks[name] = max(contract.min_ranks.get(name, 0), rank)
        contract.source_lines.setdefault(name, []).append((line, src.strip()[:120]))
        if names is not None:
            existing = contract.dim_names.setdefault(name, [])
            if names not in existing:
                existing.append(names)

    def _update_min(name: str, rank: int, line: int, src: str) -> None:
        contract.min_ranks[name] = max(contract.min_ranks.get(name, 0), rank)
        contract.source_lines.setdefault(name, []).append((line, src.strip()[:120]))

    # Walk the function body. Every assign / aug-assign / return that
    # mentions an aliased-input.shape goes through the matchers above.
    # Constraints are credited to the ORIGINAL param (via alias_map),
    # not the local alias name -- so ``kf = k.to(fp32); B,T,D = kf.shape``
    # is treated as a constraint on ``k``.
    for stmt in ast.walk(target):
        if isinstance(stmt, ast.Assign):
            for tgt in stmt.targets:
                hit = _direct_shape_unpack_rank(tgt, stmt.value, aliased_inputs)
                if hit is not None:
                    alias, rank, names = hit
                    original = alias_map.get(alias, alias)
                    try:
                        src = ast.unparse(stmt)
                    except Exception:
                        src = ""
                    _accept_rank(original, rank, stmt.lineno, src, names)
                    continue
                if (
                    isinstance(tgt, (ast.Tuple, ast.List))
                    and isinstance(stmt.value, ast.Tuple)
                ):
                    sh = _starred_count(stmt.value.elts, aliased_inputs)
                    if sh is not None:
                        alias, other_count = sh
                        original = alias_map.get(alias, alias)
                        starred_rank = len(tgt.elts) - other_count
                        if starred_rank > 0:
                            try:
                                src = ast.unparse(stmt)
                            except Exception:
                                src = ""
                            _accept_rank(
                                original, starred_rank, stmt.lineno, src
                            )
                # LHS starred unpack: ``*lead, n, _ = Af.shape``
                lhs = _lhs_starred_unpack_min_rank(
                    tgt, stmt.value, aliased_inputs
                )
                if lhs is not None:
                    alias, min_rank = lhs
                    original = alias_map.get(alias, alias)
                    try:
                        src = ast.unparse(stmt)
                    except Exception:
                        src = ""
                    _update_min(original, min_rank, stmt.lineno, src)
        if isinstance(stmt, ast.Subscript):
            mh = _shape_subscript_min_rank(stmt, aliased_inputs)
            if mh is not None:
                alias, n = mh
                original = alias_map.get(alias, alias)
                try:
                    src = ast.unparse(stmt)
                except Exception:
                    src = ""
                _update_min(original, n, stmt.lineno, src)

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


def extract_oracle_helper_contract(oracle_path: Path) -> WrapperContract | None:
    """Single shared
    extractor for the oracle's internal helper contract.

    Oracles by convention have one top-level helper function (any
    ``def`` that is NOT named ``run``) that takes tensor args and a
    thin ``run(inputs)`` shim that calls it. This helper carries the
    real tensor-shape contract -- ``B, T, D = kf.shape`` etc. -- so
    that's what R10.a / R10.b both consult to decide whether a
    wrapper-vs-oracle rank conflict is fundamental
    (ORACLE_KERNEL_LAYOUT_MISMATCH) or just a shapes.yaml typo
    (WRAPPER_CONTRACT_MISMATCH).

    Returns ``None`` when the oracle file has no non-``run`` def, or
    if the file does not parse. Both R10.a and R10.b previously
    inlined near-identical AST walks; consolidating here is the
    single-source fix for the Medium-severity duplication.
    """
    if not oracle_path.exists():
        return None
    try:
        tree = ast.parse(oracle_path.read_text())
    except (OSError, SyntaxError):
        return None
    helper_name: str | None = None
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name != "run":
            helper_name = node.name
            break
    if helper_name is None:
        return None
    return extract_wrapper_contract(oracle_path, helper_name)


__all__ = [
    "WrapperContract",
    "Violation",
    "extract_wrapper_contract",
    "extract_oracle_helper_contract",
    "check_against_inputs",
]
