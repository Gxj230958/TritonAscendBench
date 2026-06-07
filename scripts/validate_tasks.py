#!/usr/bin/env python3
"""validate_tasks.py -- schema validator for every task in tasks/.

Per specs/task_api_contract.md §5-7 + R14-P1-2.

Checks:
- All 6 mandatory files present
- meta.yaml has required top-level fields and nested structures
- tolerance_class references an entry in tolerance_policy.yaml
- oracle_simplified ∈ {simple_pytorch_ok, formula_exact_manual, upstream_oracle_required}
- kernel.py and oracle_pytorch.py parse (ast.parse) -- no SyntaxError
- kernel.py contains @triton.jit
- oracle_pytorch.py exports run(); contract.md is consistent with meta.yaml
- upstream_oracle_required tasks: oracle body has no high-level torch ops
- kernel.py exports the resolved gpu_triton entrypoint (phase-aware)
- upstream_oracle_required tasks: a complete golden/ artifact set is required
- runner_checks.metamorphic_checks names are in the known-check registry
- cost_model schema + formula evaluability (W1.2)
- param_constraints required + schema-shape valid (WA.1; canary tasks exempt)
- file_sha256 matches the on-disk kernel.py if set

Exit non-zero on first error (or all errors with --strict).
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import logging
import re
import sys
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import yaml

logger = logging.getLogger("validate_tasks")

REPO_ROOT = Path(__file__).resolve().parent.parent
TASKS_ROOT = REPO_ROOT / "tasks"
CANARY_ROOT = TASKS_ROOT / "_canary"
POLICY_FILE = REPO_ROOT / "triton_ascend_bench" / "tolerance_policy.yaml"

sys.path.insert(0, str(REPO_ROOT))
from triton_ascend_bench import param_constraints  # noqa: E402
from triton_ascend_bench.adversarial import (  # noqa: E402
    KNOWN_METAMORPHIC_CHECKS,
)
from triton_ascend_bench.cost_model import (  # noqa: E402
    CostModelError,
    parse_cost_model,
    validate_formulas,
)
from triton_ascend_bench.eligibility import (  # noqa: E402
    golden_artifact_usable,
    is_score_eligible,
)

REQUIRED_FILES = {
    "kernel.py",
    "oracle_pytorch.py",
    "meta.yaml",
    "shapes.yaml",
    "contract.md",
    # ascendc_baseline.yaml removed: GPU-triton -> NPU-triton *migration*
    # benchmark; perf is not measured vs hand-written AscendC native.
}

REQUIRED_META_TOP_LEVEL = {
    "schema_version", "task_id", "level", "idx", "name",
    "repo", "dtype_class", "category", "npu_difficulty",
    "upstream", "interface", "oracle", "tolerance_class",
    "oracle_simplified", "runner_checks",
    "is_canary", "is_hidden_only",   # `in_s_native` removed (no AscendC)
    # W1.2 / R1-4: `cost_model` is a required top-level key — a missing block
    # is a validation error. The red→green transition for the full corpus is
    # carried by this validator layer (Phase 1-2 the corpus validator is red
    # until W3.1 backfills all 100 tasks' cost_model). The dataset *loader*
    # stays backward compatible (`TaskDef.cost_model = None` on a missing
    # block) so shared loader paths are not broken — the two layers do not
    # contradict: only `validate_tasks` enforces presence.
    "cost_model",
    # WA.1: `param_constraints` (the operator input contract, six subsections
    # — `specs/task_api_contract.md §5.2`) is a required top-level key, with
    # the SAME loader-lenient / validator-strict split as `cost_model`. A
    # missing block is a validation error. `is_canary` tasks are EXEMPT —
    # `_param_constraints_required` removes the key for them before the
    # missing-key check (see `validate_task` Step 2). The full-corpus
    # red→green transition is carried by WB.1, never by the loader.
    "param_constraints",
}

# `param_constraints` is required for every real task EXCEPT canary tasks
# (`is_canary: true`) — a canary reuses its source task's solution, is a
# deliberately-crafted trap, and is never score-eligible. Removing the key
# from the required set for canaries keeps the W2.5 canary detection chain
# open. Every other top-level key in `REQUIRED_META_TOP_LEVEL` stays required.
_CANARY_EXEMPT_REQUIRED_KEYS = frozenset({"param_constraints"})

VALID_LEVELS = {"L1", "L2", "L3", "appendix", "canary"}
VALID_REPOS = {
    "vLLM", "SGLang", "FBGEMM", "Bytedance_internal", "PrimusTurbo",
}
VALID_DTYPE_CLASS = {"VV", "CV"}
VALID_DIFFICULTY = {"low", "medium", "high"}
VALID_ORACLE_TIERS = {
    "simple_pytorch_ok", "formula_exact_manual", "upstream_oracle_required"
}

PLACEHOLDER_PATTERNS = ("<frozen-snapshot>", "<TBD>", "<set-by-", "<fill-after-")

# `meta.oracle.gpu_triton` carries this sentinel until the dual-anchor
# resolver / phase_a_prime fills the verified upstream entrypoint
# (specs/task_api_contract.md §2). While it is still the placeholder the
# kernel-export check only validates the field FORMAT -- it does not yet
# require kernel.py to export anything (otherwise the 94/100 unresolved
# tasks would all false-positive). See `_validate_kernel_export`.
GPU_TRITON_PLACEHOLDER = "kernel.py::placeholder"

# High-level torch ops forbidden in the body of an `upstream_oracle_required`
# oracle (specs/task_api_contract.md §6, L195). Such a task's oracle must be
# expanded into low-level primitives -- a one-line `F.softmax` would make the
# oracle an *independent* (and possibly diverging) truth source.
#
# The check is AST-based and **alias-resolving** (W2.3 / R9-3): it parses the
# module's `import` / `from ... import` statements to map every local name to
# its canonical dotted path, then resolves each *call*'s function to a
# canonical `torch...`-rooted symbol. A high-level op is therefore caught no
# matter how it is imported — `import torch.nn.functional as G; G.silu(...)`,
# `from torch.nn.functional import silu as s; s(...)`, `torch.softmax(...)` —
# not just the handful of hard-coded `F.` / `torch.` qualifiers the old
# narrow leaf-name match recognised. A bare `import torch.nn.functional` with
# no high-level call is still fine (L1_13 / L1_14 import F but hand-roll the
# math).
#
# R12-1: the high-level-op family covers BOTH forms of every op — the
# lowercase `torch.nn.functional` *function* (`silu`) AND the Capitalised
# `torch.nn` *nn.Module class* (`SiLU`). The latter is invoked as a
# construct-then-call expression — `torch.nn.SiLU()(x)`,
# `from torch.nn import SiLU as S; S()(x)` — whose outer `Call`'s `func` is
# itself a `Call` (the constructor), not an attribute chain. Both forms are
# semantically equivalent and equally forbidden in an oracle body.
#
# `_HIGH_LEVEL_OP_NAMES` is the canonical *leaf* op-name family — a symbol is
# forbidden when it resolves to `torch.<name>` or `torch.nn.functional.<name>`
# for one of these names. These are the lowercase `torch.nn.functional`
# *function* leaves.
_HIGH_LEVEL_OP_NAMES = frozenset({
    "silu", "gelu", "softmax", "log_softmax", "sigmoid", "relu",
    "leaky_relu", "elu", "tanh", "mish", "hardswish", "hardsigmoid",
    "layer_norm", "group_norm", "batch_norm", "rms_norm",
    "scaled_dot_product_attention", "linear", "embedding",
    "cross_entropy", "softplus", "glu",
})

# Capitalised `torch.nn` *module/class* leaves — the `nn.Module` form of the
# same high-level ops (R12-1). `torch.nn.SiLU()(x)` is semantically identical
# to `torch.nn.functional.silu(x)`: both are forbidden in an
# `upstream_oracle_required` oracle body. The detection covers the
# construct-then-call pattern `torch.nn.SiLU()(x)` and import aliases
# (`from torch.nn import SiLU as S; S()(x)`).
_HIGH_LEVEL_NN_CLASS_NAMES = frozenset({
    "SiLU", "GELU", "Softmax", "LogSoftmax", "Sigmoid", "ReLU",
    "LeakyReLU", "ELU", "Tanh", "Mish", "Hardswish", "Hardsigmoid",
    "LayerNorm", "GroupNorm", "BatchNorm1d", "BatchNorm2d", "BatchNorm3d",
    "RMSNorm", "MultiheadAttention", "Linear", "Embedding",
    "CrossEntropyLoss", "Softplus", "GLU",
})

# Canonical module roots whose lowercase `.<op>` attribute is a high-level op.
_HIGH_LEVEL_OP_MODULES = frozenset({
    "torch",
    "torch.nn.functional",
})

# specs/task_api_contract.md §6a.3 — canonical dotted Python module path
# (non-empty sequence of valid Python identifiers separated by single dots).
# Same pattern the patch validator (`scripts/automation/lib/file_blocks.py`)
# enforces at write time; pre-compiled so the post-write corpus validator
# does not regress to a different shape.
_CANONICAL_DOTTED_PATH_RE = re.compile(
    r"^[a-zA-Z_][a-zA-Z0-9_]*(\.[a-zA-Z_][a-zA-Z0-9_]*)*$"
)
# §6a.3 — `upstream_callable.function` is a single Python identifier.
_PY_IDENTIFIER_RE = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*$")

# §6a.3 / §6a.5 — `*_at` and `captured_at` are ISO 8601 UTC timestamps.
# The bench writer (`state_machine.iso_utc_now`) emits the trailing-`Z` form
# `YYYY-MM-DDTHH:MM:SSZ`; readers accept either the `Z` form or the numeric
# offset form per the ISO 8601 standard.
_PHASE_B_CAPTURE_STATUS_VALUES = frozenset({"PENDING", "DONE", "FAILED"})


def _is_iso8601_utc_timestamp(value) -> bool:
    """True iff ``value`` parses as an ISO 8601 timestamp.

    Accepts both the bench-canonical trailing-`Z` form
    (``YYYY-MM-DDTHH:MM:SSZ``) and the numeric-offset form
    (``YYYY-MM-DDTHH:MM:SS+0000``). Uses :func:`datetime.fromisoformat`
    on a normalised string — ``Z`` is replaced with ``+00:00`` because
    Python <3.11 does not accept the trailing-`Z` literal. No new
    dependency is introduced.
    """
    if not isinstance(value, str) or not value:
        return False
    normalised = value[:-1] + "+00:00" if value.endswith("Z") else value
    try:
        datetime.fromisoformat(normalised)
    except ValueError:
        return False
    return True

# Canonical module roots under which a Capitalised `.<Class>` attribute names
# a high-level `nn.Module` class (the module/class form).
_HIGH_LEVEL_NN_CLASS_MODULES = frozenset({
    "torch.nn",
})

# Known metamorphic-transformation check names. The authoritative registry
# lives in `triton_ascend_bench.adversarial.KNOWN_METAMORPHIC_CHECKS` (W2.4)
# and is imported above — `validate_tasks` owns only the *validation* of
# `meta.runner_checks.metamorphic_checks` against that single source of
# truth, no longer a local bootstrap copy.


def _load_policy_classes() -> set[str]:
    doc = yaml.safe_load(POLICY_FILE.read_text())
    return set(doc.get("classes", {}).keys())


def _file_sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _shapes_knob_names(task_dir: Path) -> set[str]:
    """Collect every size-knob name a task's shapes.yaml declares.

    The cost-model formula universe is ``size_knobs`` keys plus every per-row
    ``knobs`` key (public + hidden). A formula may legitimately reference any
    of them, so the validator checks formulas against this full set.
    """
    names: set[str] = set()
    shapes_path = task_dir / "shapes.yaml"
    if not shapes_path.is_file():
        return names
    doc = yaml.safe_load(shapes_path.read_text()) or {}
    names.update((doc.get("size_knobs") or {}).keys())
    for section in ("public_shapes", "hidden_shapes"):
        for row in doc.get(section, []) or []:
            if isinstance(row, dict):
                names.update((row.get("knobs") or {}).keys())
    return names


def _validate_cost_model(tid: str, meta: dict, task_dir: Path) -> list[str]:
    """Validate ``meta.yaml.cost_model``.

    R1-4: ``cost_model`` is a **required** top-level key — its absence is
    already reported by the :data:`REQUIRED_META_TOP_LEVEL` check (Step 2).
    This function only validates the block when present: schema (scalar /
    map form, with the scalar form's required companion ``flops_class`` /
    ``bytes_resource``), then every formula is confirmed parseable +
    evaluable under the shared cost_model evaluator against the task's
    shapes.yaml knob universe. A ``None`` block is skipped here (the
    required-key check owns that error) so it is not double-reported.
    """
    raw = meta.get("cost_model")
    if raw is None:
        return []
    try:
        model = parse_cost_model(raw)
    except CostModelError as e:
        return [f"{tid}: invalid cost_model: {e}"]
    try:
        validate_formulas(model, _shapes_knob_names(task_dir))
    except CostModelError as e:
        return [f"{tid}: cost_model formula not evaluable: {e}"]
    return []


def _validate_param_constraints(tid: str, meta: dict) -> list[str]:
    """Validate ``meta.yaml.param_constraints`` — schema-shape layer (WA.1).

    Checks only the **shape** of the operator input contract: that the block
    is a mapping, that every key is one of the six known subsections, and
    that each present subsection has the right container type and the coarse
    per-element structure (`specs/task_api_contract.md §5.2`). The
    five-subsection *semantic* checks — ``dims`` predicates against the knob
    universe, materialised-tensor ``tensor_invariants``, etc. — are run by
    :func:`_validate_param_constraints_semantic` (WA.4).

    The schema-shape logic is the shared
    :func:`triton_ascend_bench.param_constraints.validate_schema_shape`, so
    the validator and the eval gate cannot drift. A missing block is reported
    by the :data:`REQUIRED_META_TOP_LEVEL` check (Step 2) and skipped here so
    it is not double-reported.

    Args:
        tid: Task id (for error-message prefixing).
        meta: The parsed ``meta.yaml`` mapping.

    Returns:
        A list of human-readable schema-shape error strings; empty when the
        block is absent or schema-shape-valid.
    """
    raw = meta.get("param_constraints")
    if raw is None:
        return []
    if not isinstance(raw, dict):
        return [f"{tid}: param_constraints must be a mapping, "
                f"got {type(raw).__name__}"]
    return param_constraints.validate_schema_shape(raw, tid=tid)


def _validate_param_constraints_semantic(
    tid: str, meta: dict, task_dir: Path,
) -> list[str]:
    """Validate ``meta.yaml.param_constraints`` semantics — WA.4 ①.

    Runs the shared :func:`triton_ascend_bench.param_constraints.
    check_param_constraints` helper (the SAME helper ``eval`` and the
    ``cli.py`` status resolver consult) against the task's ``shapes.yaml``.
    The helper's fixed decision order is: scaffold short-circuit → schema
    shape → five-subsection semantics.

    A missing block is reported by :data:`REQUIRED_META_TOP_LEVEL` (Step 2)
    and the schema-shape errors by :func:`_validate_param_constraints`
    (Step 7b'); to avoid double-reporting, this step suppresses the helper's
    *missing* verdict and only adds its semantic / scaffold reasons that the
    schema-shape pass did not already surface.

    Args:
        tid: Task id (for error-message prefixing).
        meta: The parsed ``meta.yaml`` mapping.
        task_dir: The task directory (to load ``shapes.yaml``).

    Returns:
        A list of semantic / scaffold error strings; empty when the block is
        absent, semantically clean, or its only problem is a schema-shape one
        already reported by Step 7b'.
    """
    raw = meta.get("param_constraints")
    if not isinstance(raw, dict):
        # Missing / non-mapping: reported elsewhere — do not double-report.
        return []
    # A non-score-eligible task (stub oracle / PHASE-C-MANUAL, e.g.
    # L2_08_gumbel_sample) is an acknowledged special case excluded from the
    # scored denominator. Exempt it from the scaffold / semantic gate — the
    # SAME `is_score_eligible` predicate `_validate_golden`, `eval`, and
    # `eligibility` consult. L1_21 keeps `param_constraints.scaffold: true`
    # (honest — genuinely un-reconciled, kernel is a stub) without
    # permanently blocking `--release-gate`.
    task_like = SimpleNamespace(task_id=tid, task_dir=task_dir, meta=meta)
    if not is_score_eligible(task_like):
        return []
    shapes_doc: dict | None = None
    shapes_path = task_dir / "shapes.yaml"
    if shapes_path.is_file():
        loaded = yaml.safe_load(shapes_path.read_text())
        if isinstance(loaded, dict):
            shapes_doc = loaded
    result = param_constraints.check_param_constraints(
        raw, shapes_doc, tid=tid)
    if result.ok or result.missing:
        return []
    # A schema-shape failure short-circuits the helper before the semantic
    # layer; Step 7b' already owns those reasons, so drop them here. The
    # scaffold reason and every semantic reason are WA.4's to report.
    schema_errs = set(param_constraints.validate_schema_shape(raw, tid=tid))
    return [r for r in result.reasons if r not in schema_errs]


def _module_exports(src: str, name: str) -> bool:
    """True if a parseable Python module exports a top-level ``name``.

    "Exports" means a module-level ``def``/``class``/assignment binding the
    name -- the same notion an ``from mod import name`` would resolve. Used to
    confirm ``oracle_pytorch.py`` exposes ``run`` and ``kernel.py`` exposes the
    resolved gpu_triton entrypoint, without importing (and thus executing) the
    module.
    """
    try:
        tree = ast.parse(src)
    except SyntaxError:
        # syntax errors are reported by the dedicated Step-6 gate; treat an
        # unparseable module as "does not export" so we do not crash here.
        return False
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if node.name == name:
                return True
        elif isinstance(node, ast.Assign):
            for tgt in node.targets:
                if isinstance(tgt, ast.Name) and tgt.id == name:
                    return True
        elif isinstance(node, ast.AnnAssign):
            if isinstance(node.target, ast.Name) and node.target.id == name:
                return True
    return False


def _resolve_import_aliases(tree: ast.AST) -> dict[str, str]:
    """Map every locally-bound name to its canonical dotted path (R9-3).

    Walks the module's ``import`` / ``from ... import`` statements so a
    call's function can later be resolved to a canonical ``torch...``
    symbol regardless of the alias the oracle chose. Examples of the
    bindings produced:

    * ``import torch``                          → ``{"torch": "torch"}``
    * ``import torch.nn.functional as G``       → ``{"G": "torch.nn.functional"}``
    * ``import torch.nn.functional``            → ``{"torch": "torch"}`` (the
      bound name is the *top* package ``torch``)
    * ``from torch.nn import functional as F``  → ``{"F": "torch.nn.functional"}``
    * ``from torch.nn.functional import silu``  → ``{"silu": "torch.nn.functional.silu"}``
    * ``from torch.nn.functional import silu as s`` → ``{"s": "torch.nn.functional.silu"}``
    * ``from torch import softmax as sm``       → ``{"sm": "torch.softmax"}``

    Only ``torch``-rooted imports are recorded — any other module is
    irrelevant to the high-level-op check.
    """
    aliases: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                mod = alias.name                     # e.g. torch.nn.functional
                if not (mod == "torch" or mod.startswith("torch.")):
                    continue
                if alias.asname:
                    # `import torch.nn.functional as G` binds G → that module.
                    aliases[alias.asname] = mod
                else:
                    # `import torch.nn.functional` binds the TOP package name.
                    aliases[mod.split(".")[0]] = mod.split(".")[0]
        elif isinstance(node, ast.ImportFrom):
            base = node.module or ""
            if not (base == "torch" or base.startswith("torch.")):
                continue
            for alias in node.names:
                local = alias.asname or alias.name
                aliases[local] = f"{base}.{alias.name}"
    return aliases


def _canonical_call_path(fn: ast.AST, aliases: dict[str, str]) -> str | None:
    """Resolve a ``Call``'s function node to a canonical dotted path.

    Walks the attribute chain (``G.silu`` / ``torch.nn.functional.silu``)
    down to a root :class:`ast.Name`, then rewrites the root through the
    import-alias map. ``G.silu`` with ``G → torch.nn.functional`` resolves
    to ``torch.nn.functional.silu``; a bare ``silu()`` with
    ``silu → torch.nn.functional.silu`` resolves to that directly. Returns
    ``None`` when the chain does not bottom out at a known torch alias.
    """
    parts: list[str] = []
    cur = fn
    while isinstance(cur, ast.Attribute):
        parts.append(cur.attr)
        cur = cur.value
    if not isinstance(cur, ast.Name):
        return None
    root = cur.id
    if root not in aliases:
        return None
    parts.append(root)
    parts.reverse()
    # parts == [root, attr1, attr2, ...]; rewrite the root via the alias.
    resolved_root = aliases[root]
    tail = parts[1:]
    return ".".join([resolved_root, *tail]) if tail else resolved_root


def _is_high_level_canonical(canonical: str) -> bool:
    """True if a canonical ``torch``-rooted path names a high-level op.

    Covers both op forms (R12-1):

    * lowercase ``torch.nn.functional`` / ``torch`` *function* leaf —
      ``torch.nn.functional.silu``, ``torch.softmax``;
    * Capitalised ``torch.nn`` *nn.Module class* leaf —
      ``torch.nn.SiLU``, ``torch.nn.Softmax``.
    """
    module, _, leaf = canonical.rpartition(".")
    if module in _HIGH_LEVEL_OP_MODULES and leaf in _HIGH_LEVEL_OP_NAMES:
        return True
    if (
        module in _HIGH_LEVEL_NN_CLASS_MODULES
        and leaf in _HIGH_LEVEL_NN_CLASS_NAMES
    ):
        return True
    return False


def _high_level_oracle_ops(src: str) -> list[str]:
    """Return any forbidden high-level torch ops *called* in an oracle body.

    AST-based and **alias-resolving** (R9-3 / R12-1): every ``import`` is
    parsed into a canonical-path alias map, then each ``Call`` is resolved
    to a canonical ``torch``-rooted dotted path. A call is a forbidden
    high-level op when its canonical path names one of the high-level op
    families (see :func:`_is_high_level_canonical`).

    Two call shapes are recognised:

    * **functional form** — the call's ``func`` is an attribute chain /
      bare name resolving to a ``torch.nn.functional`` / ``torch``
      lowercase function: ``F.silu(x)``, ``torch.softmax(x)``,
      ``import torch.nn.functional as G; G.silu(x)``,
      ``from torch.nn.functional import silu as s; s(x)``.
    * **module/class form** (R12-1) — the call's ``func`` is *itself a
      ``Call``* (the ``nn.Module`` constructor), i.e. the
      construct-then-call expression ``torch.nn.SiLU()(x)`` or
      ``from torch.nn import SiLU as S; S()(x)``. The constructor's own
      ``func`` is resolved and checked against the Capitalised
      ``torch.nn`` class family.

    A bare ``import torch.nn.functional`` with no high-level call still
    does not trip the check (specs/task_api_contract.md §6 / L195).
    """
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return []
    aliases = _resolve_import_aliases(tree)
    hits: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        # Functional form: func is an attribute chain / bare name.
        canonical = _canonical_call_path(node.func, aliases)
        if canonical is not None and _is_high_level_canonical(canonical):
            hits.append(canonical)
            continue
        # Module/class form (R12-1): `torch.nn.SiLU()(x)` — the outer
        # call's func is the constructor Call `torch.nn.SiLU()`. Resolve
        # the constructor's own func and check the nn.Module class family.
        if isinstance(node.func, ast.Call):
            ctor = _canonical_call_path(node.func.func, aliases)
            if ctor is not None and _is_high_level_canonical(ctor):
                hits.append(ctor)
    return sorted(set(hits))


def _validate_oracle_and_contract(tid: str, meta: dict, task_dir: Path) -> list[str]:
    """oracle_pytorch.py / contract.md consistency checks.

    1. ``oracle_pytorch.py`` must export ``run`` (the §3 contract entrypoint).
    2. ``contract.md`` must reference the task -- its short task id (``L1_12``)
       must appear -- so the human doc cannot silently belong to another task.
    3. For ``oracle_simplified == "upstream_oracle_required"`` the oracle body
       must not *call* high-level torch ops (§6 / L195).
    """
    errs: list[str] = []

    oracle_src = (task_dir / "oracle_pytorch.py").read_text()
    if not _module_exports(oracle_src, "run"):
        errs.append(f"{tid}: oracle_pytorch.py does not export 'run'")

    # contract.md <-> meta.yaml consistency: the short task id (the level-prefix
    # part, e.g. L1_12) must appear in the human-readable contract.
    contract_text = (task_dir / "contract.md").read_text()
    short_id = "_".join(str(meta.get("task_id", "")).split("_")[:2])
    if short_id and short_id not in contract_text:
        errs.append(
            f"{tid}: contract.md does not reference task id {short_id!r} "
            f"(human doc / meta.yaml inconsistency)"
        )

    if meta.get("oracle_simplified") == "upstream_oracle_required":
        hits = _high_level_oracle_ops(oracle_src)
        if hits:
            errs.append(
                f"{tid}: upstream_oracle_required oracle_pytorch.py calls "
                f"high-level torch op(s) {hits} -- oracle body must be "
                f"expanded into low-level primitives (spec L195)"
            )
    return errs


def _validate_kernel_export(tid: str, meta: dict, task_dir: Path) -> list[str]:
    """kernel.py exports the resolved gpu_triton entrypoint -- PHASE-AWARE.

    `meta.oracle.gpu_triton` starts as `kernel.py::placeholder` and the
    dual-anchor resolver / phase_a_prime later rewrites it to the verified
    upstream entrypoint (`kernel.py::<callable>`).

    Phase gate:
      * gpu_triton still the placeholder AND phase_a_prime not PASS
        -> only the FIELD FORMAT is checked (`kernel.py::<symbol>`); the
        callable need not exist yet. (94/100 tasks are here today; enforcing
        export would be a corpus-wide false-positive.)
      * gpu_triton resolved to a real entrypoint, OR phase_a_prime.status
        == PASS -> kernel.py MUST actually export the named callable.
    """
    errs: list[str] = []
    gpu_triton = ((meta.get("oracle") or {}).get("gpu_triton")) or ""

    # field-format check applies in every phase. Two accepted module parts:
    #   * ``kernel.py::<symbol>`` -- the upstream callable resolved in-module
    #     (the common case: 94/100 placeholders + tasks with a host wrapper).
    #   * ``_pa_launcher.py::run`` -- the jit-only convention. A bare @triton.jit
    #     kernel.py has no host wrapper to call, so the task ships an authored
    #     ``_pa_launcher.py`` exposing ``run(inputs)`` that drives the in-module
    #     jit; the resolver / phase_a_prime splice it onto kernel.py at S2/oracle
    #     time (siblings L2_06, L2_26 ship this way). Accept it here so the
    #     static validator stays consistent with the actual resolution pipeline.
    _ALLOWED_MODULE_PARTS = ("kernel.py", "_pa_launcher.py")
    if "::" not in str(gpu_triton):
        errs.append(
            f"{tid}: meta.oracle.gpu_triton {gpu_triton!r} is not in "
            f"'kernel.py::<symbol>' or '_pa_launcher.py::run' form"
        )
        return errs
    module_part, _, symbol = str(gpu_triton).partition("::")
    if module_part not in _ALLOWED_MODULE_PARTS or not symbol:
        errs.append(
            f"{tid}: meta.oracle.gpu_triton {gpu_triton!r} is not in "
            f"'kernel.py::<symbol>' or '_pa_launcher.py::run' form"
        )
        return errs

    pap_status = (meta.get("phase_a_prime") or {}).get("status")
    is_resolved = gpu_triton != GPU_TRITON_PLACEHOLDER
    if not is_resolved and pap_status != "PASS":
        # still placeholder, not yet phase-A'-processed: format-only.
        return errs

    # resolved (or phase_a_prime PASS): the callable must really be exported.
    # For the jit-only launcher form the exporting module is _pa_launcher.py
    # (which must exist and export ``run``); otherwise it is kernel.py.
    if module_part == "_pa_launcher.py":
        # The launcher convention is fixed: the entrypoint is ALWAYS ``run``.
        # Reject any other symbol so a meta cannot point at an arbitrary
        # launcher function (e.g. _pa_launcher.py::fake_pass).
        if symbol != "run":
            errs.append(
                f"{tid}: _pa_launcher.py entrypoint must be 'run', got "
                f"{symbol!r} (meta.oracle.gpu_triton={gpu_triton!r})"
            )
            return errs
        launcher = task_dir / "_pa_launcher.py"
        if not launcher.is_file():
            errs.append(
                f"{tid}: meta.oracle.gpu_triton points at _pa_launcher.py::"
                f"{symbol} but _pa_launcher.py is missing"
            )
        elif not _module_exports(launcher.read_text(), symbol):
            errs.append(
                f"{tid}: _pa_launcher.py does not export the resolved gpu_triton "
                f"entrypoint {symbol!r} (meta.oracle.gpu_triton={gpu_triton!r})"
            )
        return errs
    kernel_src = (task_dir / "kernel.py").read_text()
    if not _module_exports(kernel_src, symbol):
        errs.append(
            f"{tid}: kernel.py does not export the resolved gpu_triton "
            f"entrypoint {symbol!r} (meta.oracle.gpu_triton={gpu_triton!r})"
        )
    return errs


def _validate_golden(tid: str, meta: dict, task_dir: Path) -> list[str]:
    """golden/ layout completeness for upstream_oracle_required tasks.

    R4-3: an `upstream_oracle_required` task has NO PyTorch oracle that
    can serve as golden — its only golden source is the offline-collected
    `golden/<shape_id>.safetensors` artifact set (W4.3 Phase-4 GPU
    capture). This is the static "missing provider golden" gate: such a
    task MUST ship a complete `golden/` directory covering every public +
    hidden eval shape. A missing `golden/` directory, or a `golden/` that
    does not cover every eval shape, is a validation error.

    This is NOT phase-gated on the directory's prior existence (the old
    "validate only if golden/ exists" rule let a task that NEVER produced
    a `golden/` validate clean forever — the promised static gate would
    then be absent). Like R1-4's `cost_model` requirement: the full
    corpus is expected RED on this check until W4.3 Phase 4 collects the
    artifacts; the validator layer carries that red→green transition.

    The golden requirement applies ONLY to **score-eligible**
    `upstream_oracle_required` tasks. A non-score-eligible task (stub
    oracle / `PHASE-C-MANUAL` shapes — `L2_08_gumbel_sample` today) has a
    placeholder kernel and an unimplemented oracle stub: it can never
    reach a promoted state, so an offline golden can never be captured
    for it. Enforcing the golden gate on such a task would keep
    `--release-gate` permanently RED on a task that is, by design, an
    acknowledged non-score-eligible special case excluded from the
    scored denominator. The exemption uses the SAME
    `eligibility.is_score_eligible` predicate `eval` / `cli analyze` /
    `phase_b_golden_capture` consult — so the static validator and the
    runtime eligibility logic cannot drift. Score-eligible
    `upstream_oracle_required` tasks (18 today) still require a complete
    `golden/` set, unchanged.

    This is a `validate_tasks` schema-mode check -- it does NOT alter the
    binary `release_status` predicate (DESIGN §4.4).
    """
    if meta.get("oracle_simplified") != "upstream_oracle_required":
        return []

    # A non-score-eligible upstream_oracle_required task (stub oracle /
    # PHASE-C-MANUAL) has no capturable offline golden — exempt it from
    # the golden completeness gate, same predicate eval / eligibility /
    # phase_b_golden_capture use.
    task_like = SimpleNamespace(task_id=tid, task_dir=task_dir, meta=meta)
    if not is_score_eligible(task_like):
        return []

    errs: list[str] = []
    shapes_path = task_dir / "shapes.yaml"
    expected: list[str] = []
    if shapes_path.is_file():
        doc = yaml.safe_load(shapes_path.read_text()) or {}
        for section in ("public_shapes", "hidden_shapes"):
            for row in doc.get(section, []) or []:
                if isinstance(row, dict) and row.get("id"):
                    expected.append(str(row["id"]))

    golden_dir = task_dir / "golden"
    if not golden_dir.is_dir():
        # R4-3: a `golden/` directory is REQUIRED for an
        # `upstream_oracle_required` task — its absence is the "missing
        # provider golden" error, no longer a silently-clean pre-W4.3
        # state.
        errs.append(
            f"{tid}: upstream_oracle_required task has no golden/ "
            f"directory (the offline upstream-kernel golden is the only "
            f"golden source for this tier — a complete "
            f"golden/<shape_id>.safetensors set is required; collected by "
            f"W4.3 Phase 4)"
        )
        return errs

    # R7-2: an artifact must be USABLE, not merely present. Reuse the
    # shared `eligibility.golden_artifact_usable` predicate (R4-2) — the
    # exact "exists AND safetensors loads" check `eval._resolve_golden`
    # and `eligibility.golden_complete` apply at runtime. A bare
    # `is_file()` check let a corrupt / truncated / text-pasted
    # `.safetensors` validate clean while the runtime golden gates
    # fail-closed on it: the static validator and the runtime loader
    # would then disagree on whether the same task is gradeable.
    unusable = [
        sid for sid in expected
        if not golden_artifact_usable(golden_dir / f"{sid}.safetensors")
    ]
    if unusable:
        errs.append(
            f"{tid}: golden/ present but missing or unreadable safetensors "
            f"for eval shape(s) {sorted(unusable)} (expected a loadable "
            f"golden/<shape_id>.safetensors covering every public + hidden "
            f"eval shape)"
        )
    return errs


def _validate_release_scope(tid: str, meta: dict) -> list[str]:
    """`release_scope` block shape (specs/task_api_contract.md §6a.5).

    Absent / well-formed blocks pass. When ``release_scope.excluded`` is
    ``True``:

    * ``reason`` must be a non-empty string;
    * ``excluded_at`` must be a parseable ISO 8601 UTC timestamp;
    * ``excluded_by`` must be a non-empty string identifying the actor.

    Cross-field invariant §6a.6 #6 — the patch validator (Layer 5)
    gates writes; this corpus-side check guards against a malformed
    block that somehow reached disk past the patch validator.
    """
    rs = meta.get("release_scope")
    if rs is None:
        return []
    errs: list[str] = []
    if not isinstance(rs, dict):
        errs.append(
            f"{tid}: meta.yaml release_scope must be a mapping "
            f"(got {type(rs).__name__})"
        )
        return errs
    excluded = rs.get("excluded")
    if excluded is None or excluded is False:
        return errs
    if excluded is not True:
        errs.append(
            f"{tid}: meta.yaml release_scope.excluded must be a bool "
            f"(got {excluded!r})"
        )
        return errs
    reason = rs.get("reason")
    if not isinstance(reason, str) or not reason.strip():
        errs.append(
            f"{tid}: meta.yaml release_scope.excluded is true but "
            f"release_scope.reason is missing or empty"
        )
    excluded_at = rs.get("excluded_at")
    if not _is_iso8601_utc_timestamp(excluded_at):
        errs.append(
            f"{tid}: meta.yaml release_scope.excluded_at "
            f"{excluded_at!r} is not a valid ISO 8601 UTC timestamp"
        )
    excluded_by = rs.get("excluded_by")
    if not isinstance(excluded_by, str) or not excluded_by.strip():
        errs.append(
            f"{tid}: meta.yaml release_scope.excluded_by must be a "
            f"non-empty string identifying the actor"
        )
    return errs


def _validate_phase_a_skip_invocation(tid: str, meta: dict) -> list[str]:
    """`phase_a_prime.skip_invocation` cross-field invariants.

    specs/task_api_contract.md §6a.1 + §6a.6 #1:

      ``skip_invocation == true``
          ⇒ ``status == "SKIPPED_BY_DESIGN"``
          ⇒ ``reason`` is a non-empty string

    Plus the inverse §6a.6 #2 — ``status == "SKIPPED_BY_DESIGN"`` ⇒
    ``skip_invocation == true`` and a non-empty ``reason``. Absent
    blocks pass.
    """
    pap = meta.get("phase_a_prime")
    if not isinstance(pap, dict):
        return []
    errs: list[str] = []
    skip = pap.get("skip_invocation")
    status = pap.get("status")
    reason = pap.get("reason")
    reason_ok = isinstance(reason, str) and bool(reason.strip())

    if skip is True:
        if status != "SKIPPED_BY_DESIGN":
            errs.append(
                f"{tid}: meta.yaml phase_a_prime.skip_invocation is true "
                f"but phase_a_prime.status is {status!r} (must be "
                f"'SKIPPED_BY_DESIGN'; invariant §6a.6 #1)"
            )
        if not reason_ok:
            errs.append(
                f"{tid}: meta.yaml phase_a_prime.skip_invocation is true "
                f"but phase_a_prime.reason is missing or empty"
            )
    elif skip is not None and skip is not False:
        errs.append(
            f"{tid}: meta.yaml phase_a_prime.skip_invocation must be a "
            f"bool (got {skip!r})"
        )

    if status == "SKIPPED_BY_DESIGN":
        if skip is not True:
            errs.append(
                f"{tid}: meta.yaml phase_a_prime.status is "
                f"'SKIPPED_BY_DESIGN' but skip_invocation is {skip!r} "
                f"(must be true; invariant §6a.6 #2)"
            )
        if not reason_ok:
            errs.append(
                f"{tid}: meta.yaml phase_a_prime.status is "
                f"'SKIPPED_BY_DESIGN' but phase_a_prime.reason is "
                f"missing or empty"
            )
    return errs


def _validate_phase_b_capture(tid: str, meta: dict) -> list[str]:
    """`phase_b_capture` block shape + cross-field invariants.

    specs/task_api_contract.md §6a.3 + §6a.6 #3/#4:

    * ``status`` ∈ {``PENDING``, ``DONE``, ``FAILED``};
    * ``DONE`` ⇒ ``upstream_callable.module`` is a canonical dotted
      Python path, ``upstream_callable.function`` is a valid Python
      identifier, ``captured_at`` is a parseable ISO 8601 UTC
      timestamp;
    * ``FAILED`` ⇒ ``reason`` is a non-empty string.

    Absent / well-formed blocks pass.
    """
    pbc = meta.get("phase_b_capture")
    if pbc is None:
        return []
    errs: list[str] = []
    if not isinstance(pbc, dict):
        errs.append(
            f"{tid}: meta.yaml phase_b_capture must be a mapping "
            f"(got {type(pbc).__name__})"
        )
        return errs
    status = pbc.get("status")
    if status is None:
        # block present but no status — treat as no capture attempted.
        return errs
    if status not in _PHASE_B_CAPTURE_STATUS_VALUES:
        errs.append(
            f"{tid}: meta.yaml phase_b_capture.status {status!r} not in "
            f"{sorted(_PHASE_B_CAPTURE_STATUS_VALUES)}"
        )
        return errs

    uc = pbc.get("upstream_callable")
    if status == "DONE":
        if not isinstance(uc, dict):
            errs.append(
                f"{tid}: meta.yaml phase_b_capture.status is 'DONE' but "
                f"phase_b_capture.upstream_callable is missing or not a "
                f"mapping"
            )
        else:
            module = uc.get("module")
            function = uc.get("function")
            if not isinstance(module, str) or not _CANONICAL_DOTTED_PATH_RE.fullmatch(module):
                errs.append(
                    f"{tid}: meta.yaml phase_b_capture.upstream_callable."
                    f"module {module!r} is not a canonical dotted Python "
                    f"path"
                )
            if not isinstance(function, str) or not _PY_IDENTIFIER_RE.fullmatch(function):
                errs.append(
                    f"{tid}: meta.yaml phase_b_capture.upstream_callable."
                    f"function {function!r} is not a valid Python "
                    f"identifier"
                )
        captured_at = pbc.get("captured_at")
        if not _is_iso8601_utc_timestamp(captured_at):
            errs.append(
                f"{tid}: meta.yaml phase_b_capture.status is 'DONE' but "
                f"phase_b_capture.captured_at {captured_at!r} is not a "
                f"valid ISO 8601 UTC timestamp"
            )
    elif status == "FAILED":
        reason = pbc.get("reason")
        if not isinstance(reason, str) or not reason.strip():
            errs.append(
                f"{tid}: meta.yaml phase_b_capture.status is 'FAILED' "
                f"but phase_b_capture.reason is missing or empty"
            )
    return errs


def _validate_provenance_golden(tid: str, meta: dict) -> list[str]:
    """V9 plan M1.2 -- closed-enum + shape check on ``provenance.golden:``.

    Delegates to ``triton_ascend_bench.provenance.validate_provenance_golden``,
    which is the single source of truth for the schema. Grace period:
    when the nested ``golden:`` block is absent the validator returns
    cleanly; when present, every closed enum and hash-length field
    must conform.
    """
    # Import lazily so this script stays importable in environments
    # where the package's optional deps aren't installed (the
    # cli/dataset path imports torch transitively; this validator
    # script is run from CI on minimal envs).
    from triton_ascend_bench.provenance import (  # noqa: PLC0415
        validate_provenance_golden,
    )

    raw_errs = validate_provenance_golden(meta)
    return [f"{tid}: meta.yaml {e}" for e in raw_errs]


def _validate_metamorphic_checks(tid: str, meta: dict) -> list[str]:
    """`runner_checks.metamorphic_checks` names must be in the known registry.

    Each declared check name (string item, or dict item's ``name``) must be a
    member of :data:`KNOWN_METAMORPHIC_CHECKS`. An unknown name is rejected at
    validation time rather than silently skipped at runtime (the W2.4 runtime
    path can only execute checks it has a transformation for).
    """
    rc = meta.get("runner_checks") or {}
    declared = rc.get("metamorphic_checks") or []
    errs: list[str] = []
    for item in declared:
        if isinstance(item, str):
            name = item
        elif isinstance(item, dict):
            name = item.get("name")
        else:
            errs.append(
                f"{tid}: metamorphic_checks entry {item!r} is neither a "
                f"name string nor a {{name: ...}} mapping"
            )
            continue
        if name not in KNOWN_METAMORPHIC_CHECKS:
            errs.append(
                f"{tid}: metamorphic_checks declares unknown check {name!r} "
                f"(not in the known-check registry)"
            )
    return errs


# ─────────────────────────────────────────────────────────────────────────────
# Canary task validation (W2.5 — dedicated `_canary` entry point)
# ─────────────────────────────────────────────────────────────────────────────
# The main scan walks ONLY `tasks/level{1,2,3}/` (and skips every `_`-prefixed
# dir). A malformed canary would therefore have no static gate at all — so
# canary tasks get this separate validator, run by default and via `--canary`.

# A canary has NO `kernel.py` and NO submitter solution — `source_task_id`
# binds it to a real task whose solution the runner reuses. `_trap_intent.md`
# is maintainer-only and not required.
CANARY_REQUIRED_FILES = {
    "meta.yaml", "shapes.yaml", "contract.md", "oracle_pytorch.py",
}

# Canary-only meta keys (W2.5 canary schema), on top of a small shared core.
CANARY_REQUIRED_META = {
    "schema_version", "task_id", "level", "name",
    "tolerance_class", "oracle_simplified", "oracle",
    "is_canary", "is_hidden_only",
    "canary_trap_type", "source_task_id", "expected_honest_pass",
}

# The 10 canary trap classes (DESIGN-V2 §8 / specs/canary_task_design.md).
VALID_CANARY_TRAP_TYPES = {
    "identity_hack", "zero_out_hack", "cache_reuse_hack",
    "non_default_stream_hack", "torch_fallback_hack", "shape_route_hack",
    "lookup_table_hack", "async_no_sync_hack", "partial_compute_hack",
    "alibi_attention_hack",
}


def _real_task_ids() -> set[str]:
    """Every real level1/2/3 task id (the legal `source_task_id` targets)."""
    ids: set[str] = set()
    for level in ("level1", "level2", "level3"):
        level_dir = TASKS_ROOT / level
        if not level_dir.is_dir():
            continue
        for d in level_dir.iterdir():
            if d.is_dir() and not d.name.startswith("_"):
                ids.add(d.name)
    return ids


def validate_canary_task(
    canary_dir: Path, policy_classes: set[str], real_task_ids: set[str]
) -> list[str]:
    """Validate one `tasks/_canary/<canary_id>/` against the canary schema.

    Checks (W2.5):
      * the canary required files are present (no `kernel.py` — a canary has
        no own solution);
      * meta.yaml carries every canary-schema key;
      * `is_canary` is true and `level` is `canary`;
      * `canary_trap_type` is one of the 10 trap classes;
      * `source_task_id` points at an EXISTING real level1/2/3 task — an
        invalid binding is rejected here (a canary cannot be run otherwise);
      * `expected_honest_pass` is a bool;
      * `tolerance_class` is a known policy class;
      * `oracle_pytorch.py` parses and exports `run`.
    """
    errs: list[str] = []
    cid = canary_dir.name

    have = {p.name for p in canary_dir.iterdir() if p.is_file()}
    missing = CANARY_REQUIRED_FILES - have
    if missing:
        errs.append(f"canary {cid}: missing files {sorted(missing)}")
        return errs

    try:
        meta = yaml.safe_load((canary_dir / "meta.yaml").read_text())
    except yaml.YAMLError as e:
        return [f"canary {cid}: meta.yaml is not valid YAML: {e}"]
    if not isinstance(meta, dict):
        return [f"canary {cid}: meta.yaml is not a mapping"]

    missing_keys = CANARY_REQUIRED_META - set(meta.keys())
    if missing_keys:
        errs.append(f"canary {cid}: meta.yaml missing keys {sorted(missing_keys)}")

    if meta.get("is_canary") is not True:
        errs.append(
            f"canary {cid}: meta.is_canary must be true "
            f"(got {meta.get('is_canary')!r})"
        )
    if meta.get("level") != "canary":
        errs.append(
            f"canary {cid}: meta.level must be 'canary' "
            f"(got {meta.get('level')!r})"
        )

    trap = meta.get("canary_trap_type")
    if trap not in VALID_CANARY_TRAP_TYPES:
        errs.append(
            f"canary {cid}: canary_trap_type {trap!r} not in "
            f"{sorted(VALID_CANARY_TRAP_TYPES)}"
        )

    src = meta.get("source_task_id")
    if not src:
        errs.append(f"canary {cid}: source_task_id is empty")
    elif src not in real_task_ids:
        errs.append(
            f"canary {cid}: source_task_id {src!r} does not name an existing "
            f"level1/2/3 task — the canary cannot be bound to a solution"
        )

    if not isinstance(meta.get("expected_honest_pass"), bool):
        errs.append(
            f"canary {cid}: expected_honest_pass must be a bool "
            f"(got {meta.get('expected_honest_pass')!r})"
        )

    tol = meta.get("tolerance_class")
    if tol not in policy_classes:
        errs.append(
            f"canary {cid}: tolerance_class {tol!r} not in tolerance_policy.yaml"
        )

    oracle_src = (canary_dir / "oracle_pytorch.py").read_text()
    try:
        ast.parse(oracle_src, filename=str(canary_dir / "oracle_pytorch.py"))
    except SyntaxError as e:
        errs.append(
            f"canary {cid}: oracle_pytorch.py SyntaxError at line {e.lineno}: {e.msg}"
        )
    else:
        if not _module_exports(oracle_src, "run"):
            errs.append(f"canary {cid}: oracle_pytorch.py does not export 'run'")

    return errs


def run_canary_validation(strict: bool) -> list[str]:
    """Validate every canary under `tasks/_canary/` — the W2.5 entry point."""
    if not CANARY_ROOT.is_dir():
        return []
    policy_classes = _load_policy_classes()
    real_ids = _real_task_ids()
    all_errs: list[str] = []
    for canary_dir in sorted(CANARY_ROOT.iterdir()):
        if not canary_dir.is_dir():
            continue
        errs = validate_canary_task(canary_dir, policy_classes, real_ids)
        all_errs.extend(errs)
        for e in errs:
            print(f"ERROR: {e}", file=sys.stderr)
        if strict and errs:
            break
    return all_errs


def validate_task(task_dir: Path, policy_classes: set[str], strict: bool) -> list[str]:
    errs: list[str] = []
    tid = task_dir.name

    # Step 1: files
    have = {p.name for p in task_dir.iterdir() if p.is_file()}
    missing = REQUIRED_FILES - have
    if missing:
        errs.append(f"{tid}: missing files {sorted(missing)}")
        return errs

    meta = yaml.safe_load((task_dir / "meta.yaml").read_text())

    # Step 2: top-level keys. `is_canary` tasks are exempt from a subset of
    # required keys (WA.1: `param_constraints` — a canary reuses its source
    # task's solution and is never score-eligible). For a canary, the exempt
    # keys are dropped from the required set before the missing-key check.
    required_keys = REQUIRED_META_TOP_LEVEL
    if meta.get("is_canary") is True:
        required_keys = REQUIRED_META_TOP_LEVEL - _CANARY_EXEMPT_REQUIRED_KEYS
    missing_keys = required_keys - set(meta.keys())
    if missing_keys:
        errs.append(f"{tid}: meta.yaml missing keys {sorted(missing_keys)}")
        if strict and missing_keys:
            return errs

    # Step 3: enum checks
    if meta.get("level") not in VALID_LEVELS:
        errs.append(f"{tid}: level {meta.get('level')!r} not in {VALID_LEVELS}")
    if meta.get("repo") not in VALID_REPOS:
        errs.append(f"{tid}: repo {meta.get('repo')!r} not in {VALID_REPOS}")
    if meta.get("dtype_class") not in VALID_DTYPE_CLASS:
        errs.append(f"{tid}: dtype_class {meta.get('dtype_class')!r} invalid")
    if meta.get("npu_difficulty") not in VALID_DIFFICULTY:
        errs.append(f"{tid}: npu_difficulty {meta.get('npu_difficulty')!r} invalid")
    if meta.get("oracle_simplified") not in VALID_ORACLE_TIERS:
        errs.append(
            f"{tid}: oracle_simplified {meta.get('oracle_simplified')!r} "
            f"not in {VALID_ORACLE_TIERS}"
        )

    # Step 4: tolerance class exists in policy
    tol = meta.get("tolerance_class")
    if tol not in policy_classes:
        errs.append(
            f"{tid}: tolerance_class {tol!r} not in tolerance_policy.yaml "
            f"({sorted(policy_classes)})"
        )

    # Step 5 removed: the in_s_native ⇒ AscendC-baseline-frozen check is
    # obsolete. This is a GPU-triton -> NPU-triton migration benchmark;
    # there is no AscendC native baseline (see scoring_spec.md). The
    # performance reference is deferred (correctness is the live scored
    # axis); no per-task baseline artifact is required.

    # Step 6: Python syntax gate (G11 fix). A schema-valid task whose
    # kernel.py / oracle_pytorch.py does not even parse can never compile
    # or be evaluated, so it must fail validation -- not slip through.
    kernel_text = (task_dir / "kernel.py").read_text()
    for fname, src in (
        ("kernel.py", kernel_text),
        ("oracle_pytorch.py", (task_dir / "oracle_pytorch.py").read_text()),
    ):
        try:
            ast.parse(src, filename=str(task_dir / fname))
        except SyntaxError as e:
            errs.append(
                f"{tid}: {fname} SyntaxError at line {e.lineno}: {e.msg}"
            )

    # Step 7: kernel.py contains @triton.jit
    if "@triton.jit" not in kernel_text:
        errs.append(f"{tid}: kernel.py contains no @triton.jit")

    # Step 7b: cost_model schema + formula check (W1.2). Optional pre-W3.1;
    # validated strictly when present. Accepts both the scalar and the map
    # form for `flops` / `bytes`; confirms every formula is parseable and
    # evaluable under the shared cost_model evaluator.
    errs.extend(_validate_cost_model(tid, meta, task_dir))

    # Step 7b': param_constraints schema-shape check (WA.1). Presence is the
    # REQUIRED_META_TOP_LEVEL check's job (Step 2, with the canary exemption);
    # this only validates the SHAPE of a present block (six subsections,
    # container types).
    errs.extend(_validate_param_constraints(tid, meta))

    # Step 7b'': param_constraints five-subsection semantic check (WA.4 ①).
    # Runs the shared `param_constraints.check_param_constraints` helper —
    # the same pass/fail authority `eval` and `cli analyze` consult — so the
    # `dims` predicates, `dtypes` / `dists` agreement, `scalars` value
    # domains and materialised-tensor `tensor_invariants` are enforced here.
    # `is_canary` tasks are exempt (canary reuses its source solution and is
    # never score-eligible — keeps the W2.5 canary detection chain open).
    if meta.get("is_canary") is not True:
        errs.extend(
            _validate_param_constraints_semantic(tid, meta, task_dir))

    # Step 7c: oracle_pytorch.py / contract.md consistency (W2.3). oracle
    # exports `run`; contract.md references the task id; an
    # upstream_oracle_required oracle body has no high-level torch ops.
    errs.extend(_validate_oracle_and_contract(tid, meta, task_dir))

    # Step 7d: kernel.py exports the resolved gpu_triton entrypoint (W2.3),
    # PHASE-AWARE -- format-only while gpu_triton is still the placeholder.
    errs.extend(_validate_kernel_export(tid, meta, task_dir))

    # Step 7e: golden/ layout completeness for SCORE-ELIGIBLE
    # upstream_oracle_required tasks (W2.3). A non-score-eligible
    # upstream_oracle_required task (stub oracle / PHASE-C-MANUAL) has no
    # capturable offline golden and is exempt.
    errs.extend(_validate_golden(tid, meta, task_dir))

    # Step 7f: metamorphic_checks names must be in the known-check registry
    # (W2.3). The authoritative registry is built later by W2.4.
    errs.extend(_validate_metamorphic_checks(tid, meta))

    # Step 7g: bench-team-internal §6a state fields. All four blocks are
    # OPTIONAL with explicit defaults — the 100-task corpus authored before
    # M2 has none of them and stays clean. When present, the shape /
    # cross-field invariants from specs/task_api_contract.md §6a.5 / §6a.1 /
    # §6a.3 are enforced (the patch validator gates writes at Layer 5;
    # these checks defend against any malformed block that bypassed it).
    # `release_scope.excluded == true` does NOT short-circuit `validate_task`
    # — file presence, syntax, hashes, and every other gate still run, so a
    # task pulled out of release scope cannot silently rot.
    errs.extend(_validate_release_scope(tid, meta))
    errs.extend(_validate_phase_a_skip_invocation(tid, meta))
    errs.extend(_validate_phase_b_capture(tid, meta))

    # Step 7h: V9 plan M1.2 -- closed-enum + shape check on the nested
    # ``provenance.golden:`` sub-block. Top-level ``provenance:`` keys
    # (``added_in`` / ``authors`` / ``last_modified``) are NOT touched
    # here; this gate is for the V9 trust-chain fields populated by
    # R10/R5 capture (oracle equivalence verdict, captured golden
    # SHA, signed manifest, reviewer attestation, ...). Grace-period
    # semantics: an absent ``golden:`` block is allowed (legacy task
    # / pre-R10). When the block IS present, every closed-enum value
    # must conform -- a malformed value here is a release-blocking
    # error because the release gate reads these fields directly.
    errs.extend(_validate_provenance_golden(tid, meta))

    # Step 8: PHASE-AWARE hash integrity (plan U5 codex-cep-r2 must-fix).
    # If the task has been Phase-A'-processed it carries
    # meta.phase_a_prime.standalone_sha256 -- compare THAT to the on-disk
    # kernel.py. Otherwise (unprocessed / PoC / legacy: kernel.py is
    # still the verbatim upstream freeze) keep the legacy behaviour and
    # compare meta.upstream.file_sha256. `upstream.file_sha256` is
    # ALWAYS preserved as the original provenance hash. This makes a
    # mixed rollout (some rewritten, some legacy) all-green under the
    # normal schema/hash mode (release gate is a SEPARATE mode, U7).
    actual_sha = _file_sha256(task_dir / "kernel.py")
    pap = meta.get("phase_a_prime") or {}
    standalone_sha = pap.get("standalone_sha256", "")
    if standalone_sha and not any(
        p in str(standalone_sha) for p in PLACEHOLDER_PATTERNS
    ):
        if standalone_sha != actual_sha:
            errs.append(
                f"{tid}: phase_a_prime.standalone_sha256 mismatch: "
                f"meta has {str(standalone_sha)[:16]}…, actual {actual_sha[:16]}…"
            )
    else:
        declared_sha = (meta.get("upstream") or {}).get("file_sha256", "")
        if any(p in str(declared_sha) for p in PLACEHOLDER_PATTERNS):
            errs.append(
                f"{tid}: upstream.file_sha256 still a placeholder: {declared_sha!r}"
            )
        elif not declared_sha:
            # FIX 5: NEITHER phase_a_prime.standalone_sha256 NOR
            # upstream.file_sha256 is populated -> there is NO
            # on-disk-comparable integrity hash at all. The old code
            # silently passed this (a task with zero integrity check).
            # Every task MUST have at least one comparable hash.
            errs.append(
                f"{tid}: no integrity hash -- neither "
                f"phase_a_prime.standalone_sha256 nor "
                f"upstream.file_sha256 is populated (every task must "
                f"have at least one on-disk-comparable hash)"
            )
        elif declared_sha != actual_sha:
            errs.append(
                f"{tid}: kernel.py sha256 mismatch: "
                f"meta has {declared_sha[:16]}…, actual {actual_sha[:16]}…"
            )

    return errs


def _load_eligible_ids() -> tuple[set[str], int]:
    """Return (score_eligible_ids, catalogue_expected_eligible_count).

    Single source of truth: triton_ascend_bench.eligibility (shared
    with cli.py -- plan U7, prevents drift). The expected count is
    derived independently from iter_all_tasks() so the non-vacuity
    guard can detect a wrong task-root / empty fixture / compute bug.
    """
    sys.path.insert(0, str(REPO_ROOT))
    from triton_ascend_bench.dataset import iter_all_tasks
    from triton_ascend_bench.eligibility import compute_release_ready_eligible

    # Release-ready eligible = score-eligible INTERSECT C2 provenance-eligible
    # (is_provenance_eligible). Gating the release set on this is what makes C2
    # a REAL release gate: a C2-FAILED / un-provenanced task is excluded from
    # the released corpus, so it cannot satisfy the S2-only release_status_pass
    # and sneak through. (compute_score_eligible alone has no C2 clause.)
    all_tasks = list(iter_all_tasks())
    eligible = compute_release_ready_eligible(all_tasks)
    return eligible, len(eligible)


def run_release_gate(tasks=None, eligible_ids=None) -> int:
    """`--release-gate` mode (plan U7): SEPARATE from schema/hash mode.

    `eligibility  intersect  {release_status != PASS}` non-empty ⇒ nonzero
    exit + the blocking list, where the COMPOSITE release predicate is::

        release_status == PASS  <=>  phase_a_prime.status == PASS   (S2 severance)

    over the score-eligible set, which is already intersected with
    `is_provenance_eligible` (C2 oracle-equivalence VERIFIED / CEILING).
    So the gate keys on the two real CORRECTNESS proofs -- C2 precision +
    S2 severance -- not on `cost_model.status`. cost_model is an
    ncu-evidenced PERFORMANCE/access model whose per-shape status is
    recorded as metadata but is NOT a release blocker (conditional access,
    L2 reuse, tile geometry, and the HBM burst floor make a portable
    analytic formula structurally unable to hit per-shape tol everywhere;
    gating on it would pin release red on a performance technicality or
    invite dishonest shape fitting). The blocking set is computed by the
    shared `eligibility.compute_release_blocking` -- the SAME function
    `cli.py` analyze and `phase_a_prime.py`'s S6 verdict use, so all three
    authorities yield an identical blocking set.

    PLUS a non-vacuity guard (codex-cep-r3 must-fix): if the examined
    eligible count == 0, OR != the catalogue-expected eligible count,
    this is a HARD FAIL -- never a vacuous pass ("no tasks so nothing
    BLOCKED so releasable" is the degenerate path S6 / no-subset-escape
    forbids).

    `tasks` / `eligible_ids` are injectable for unit tests; in
    production they are loaded from the real catalogue.
    """
    if tasks is None or eligible_ids is None:
        eligible_ids, expected = _load_eligible_ids()
        from triton_ascend_bench.dataset import iter_all_tasks

        tasks = iter_all_tasks()
    else:
        expected = len(eligible_ids)

    # MATERIALISE before BOTH passes (codex FIX 2 [confirmed]): the
    # function iterates `tasks` once for `examined` and again for
    # `blocking`. If `tasks` is a one-shot generator the 2nd pass is
    # empty -> `blocking` is [] -> a vacuous return 0 ("releasable")
    # even with eligible non-PASS tasks. list() makes it re-iterable.
    tasks = list(tasks)

    if expected == 0:
        print(
            "RELEASE-GATE HARD FAIL: catalogue-expected eligible count is 0 "
            "(bad task root / empty fixture / eligibility compute bug). "
            "Refusing to vacuously pass.",
            file=sys.stderr,
        )
        return 2

    all_tasks = tasks
    examined = {t.task_id for t in all_tasks if t.task_id in eligible_ids}
    if len(examined) == 0 or len(examined) != expected:
        print(
            f"RELEASE-GATE HARD FAIL: examined eligible count "
            f"{len(examined)} != catalogue-expected {expected} "
            f"(non-vacuity guard; never vacuous-pass).",
            file=sys.stderr,
        )
        return 2

    # W1.5: composite release predicate -- shared with cli.py analyze
    # and phase_a_prime.py's S6 verdict (identical blocking set).
    from triton_ascend_bench.eligibility import compute_release_blocking

    blocking = compute_release_blocking(all_tasks, eligible_ids)

    if blocking:
        print(
            f"RELEASE-GATE FAIL: {len(blocking)} release-ready "
            f"(score-eligible AND C2-aligned) task(s) with "
            f"release_status != PASS (S2 phase_a_prime.status == PASS) "
            f"-- benchmark NOT releasable:",
            file=sys.stderr,
        )
        for tid in blocking:
            print(f"  - {tid}", file=sys.stderr)
        return 1

    print(
        f"release-gate OK: all {expected} release-ready "
        f"(score-eligible AND C2-aligned) task(s) release_status == PASS "
        f"(phase_a_prime.status == PASS / S2 severance). "
        f"cost_model is evidenced performance metadata, NOT a release blocker."
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", help="Validate a single task id (skip the rest)")
    parser.add_argument("--strict", action="store_true",
                        help="Stop on first task error")
    parser.add_argument(
        "--release-gate", action="store_true",
        help="S6 release gate (SEPARATE from schema/hash mode): over the "
             "release-ready set (score-eligible AND C2-aligned), "
             "{release_status != PASS} non-empty ⇒ fail, where "
             "release_status == PASS <=> phase_a_prime.status == PASS "
             "(S2 severance). cost_model is evidenced performance metadata, "
             "NOT a release blocker; non-vacuity guarded (never vacuous-pass).",
    )
    parser.add_argument(
        "--canary", action="store_true",
        help="Validate ONLY the `tasks/_canary/` trap tasks against the "
             "canary schema (W2.5). The main scan walks only level1/2/3, so "
             "this is the dedicated static gate for canary tasks; it also "
             "runs by default at the end of a full scan.",
    )
    args = parser.parse_args()

    if args.release_gate:
        return run_release_gate()

    if args.canary:
        # Dedicated canary-only mode (W2.5).
        canary_errs = run_canary_validation(args.strict)
        if canary_errs:
            print(f"\n{len(canary_errs)} canary validation error(s)",
                  file=sys.stderr)
            return 1
        print("All canary tasks validated successfully.")
        return 0

    policy_classes = _load_policy_classes()

    all_errs: list[str] = []
    for level in ("level1", "level2", "level3"):
        level_dir = TASKS_ROOT / level
        if not level_dir.is_dir():
            continue
        for task_dir in sorted(level_dir.iterdir()):
            if not task_dir.is_dir() or task_dir.name.startswith("_"):
                continue
            if args.task and task_dir.name != args.task:
                continue
            errs = validate_task(task_dir, policy_classes, args.strict)
            all_errs.extend(errs)
            for e in errs:
                print(f"ERROR: {e}", file=sys.stderr)
            if args.strict and errs:
                return 1

    # W2.5: a full scan (no `--task` filter) also validates the `_canary`
    # trap tasks — malformed canaries must not slip past the static gate.
    if not args.task:
        all_errs.extend(run_canary_validation(args.strict))

    if all_errs:
        print(f"\n{len(all_errs)} validation error(s)", file=sys.stderr)
        return 1
    print("All tasks validated successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
