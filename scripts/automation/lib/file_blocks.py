"""LLM output META + FILE block parser and 5-layer validator (Unit 1.6).

Output protocol (parsed globally; META and FILE blocks may interleave;
prose outside blocks is ignored)::

    ### META
    {json}
    ### END

    ### FILE: tasks/<level>/<task_id>/<name>
    <full file content>
    ### END

Validation pipeline:
  Layer 1 -- path whitelist (Layer 1)
  Layer 2 -- import leak (oracle_pytorch.py must not import vllm/sglang)
  Layer 3 -- stub-signature gate (function body == empty_like / pass /
             raise NotImplementedError)
  Layer 4 -- META x FILE set consistency by ``decision``
  Layer 5 -- content-level semantic validation (per-decision; AST scan
             for rewrite-primitives, canonical dotted-path / Python
             identifier checks for defer-to-*; YAML required-field
             checks for defer-to-*)

Two-stage caller contract:
  ``validate_patch(...)`` -- runs Layer 1..5, returns errors (no FS writes).
  ``apply_patch(...)``    -- atomic writes only; caller already snapshotted.
"""
from __future__ import annotations

import ast
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from automation.lib import atomic

logger = logging.getLogger("automation.lib.file_blocks")


# ---------------------------------------------------------------------------
# Module-level constants (mirror scripts/corpus/validate_tasks.py:147-167)
# ---------------------------------------------------------------------------

_HIGH_LEVEL_OP_NAMES = frozenset({
    "silu", "gelu", "softmax", "log_softmax", "sigmoid", "relu",
    "leaky_relu", "elu", "tanh", "mish", "hardswish", "hardsigmoid",
    "layer_norm", "group_norm", "batch_norm", "rms_norm",
    "scaled_dot_product_attention", "linear", "embedding",
    "cross_entropy", "softplus", "glu",
})

_HIGH_LEVEL_NN_CLASS_NAMES = frozenset({
    "SiLU", "GELU", "Softmax", "LogSoftmax", "Sigmoid", "ReLU",
    "LeakyReLU", "ELU", "Tanh", "Mish", "Hardswish", "Hardsigmoid",
    "LayerNorm", "GroupNorm", "BatchNorm1d", "BatchNorm2d", "BatchNorm3d",
    "RMSNorm", "MultiheadAttention", "Linear", "Embedding",
    "CrossEntropyLoss", "Softplus", "GLU",
})

_CANONICAL_DOTTED_PATH_RE = re.compile(
    r"^[a-zA-Z_][a-zA-Z0-9_]*(\.[a-zA-Z_][a-zA-Z0-9_]*)*$"
)
_PY_IDENTIFIER_RE = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*$")

# Decisions that allow / require a FILE: meta.yaml
_ALL_DECISIONS = frozenset({
    "patch-shapes",
    "tolerance-bump",
    "rewrite-primitives",
    "defer-to-upstream-oracle",
    "defer-to-golden",
})


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class MetaBlock:
    decision: str
    payload: dict[str, Any]


@dataclass
class FileBlock:
    path: Path
    content: str


@dataclass
class ValidationError:
    layer: int
    code: str
    message: str
    path: Path | None = None

    def __str__(self) -> str:
        loc = f" [{self.path}]" if self.path is not None else ""
        return f"L{self.layer} {self.code}{loc}: {self.message}"


class MultipleMetaBlocksError(RuntimeError):
    """Raised by ``parse_meta`` when more than one META block is found."""


# ---------------------------------------------------------------------------
# Parsers
# ---------------------------------------------------------------------------

_META_RE = re.compile(r"^### META\n(.*?)\n### END\n", re.MULTILINE | re.DOTALL)
_FILE_RE = re.compile(r"^### FILE: (.+?)\n(.*?)\n### END\n", re.MULTILINE | re.DOTALL)


def parse_meta(text: str) -> MetaBlock | None:
    """Find the single META block; return None if zero, raise on >1."""
    if not text.endswith("\n"):
        text = text + "\n"
    hits = list(_META_RE.finditer(text))
    if len(hits) == 0:
        return None
    if len(hits) > 1:
        raise MultipleMetaBlocksError(
            f"expected exactly 1 META block, got {len(hits)}"
        )
    raw_json = hits[0].group(1).strip()
    try:
        import json as _j
        payload = _j.loads(raw_json)
    except Exception as exc:
        # Treat parse failure as "missing meta" so caller can re-prompt.
        logger.warning("META JSON parse failed: %s", exc)
        return None
    if not isinstance(payload, dict):
        return None
    decision = str(payload.get("decision", ""))
    return MetaBlock(decision=decision, payload=payload)


def parse_blocks(text: str) -> list[FileBlock]:
    """Return all FILE blocks in their textual order."""
    if not text.endswith("\n"):
        text = text + "\n"
    blocks: list[FileBlock] = []
    for m in _FILE_RE.finditer(text):
        path = Path(m.group(1).strip())
        content = m.group(2)
        # Re-add trailing newline so callers writing the file get the
        # canonical "ends with \n" form back.
        if not content.endswith("\n"):
            content = content + "\n"
        blocks.append(FileBlock(path=path, content=content))
    return blocks


# ---------------------------------------------------------------------------
# Layer 1: path whitelist
# ---------------------------------------------------------------------------

_ALLOWED_FILE_NAMES = frozenset({"meta.yaml", "shapes.yaml", "oracle_pytorch.py"})


def _layer1_path_whitelist(
    blocks: list[FileBlock], task_id: str,
) -> list[ValidationError]:
    errs: list[ValidationError] = []
    for b in blocks:
        if ".." in b.path.parts or b.path.is_absolute():
            errs.append(ValidationError(
                layer=1, code="path_traversal",
                message=f"path must be relative under tasks/<level>/{task_id}/",
                path=b.path,
            ))
            continue
        # Path must look like tasks/<level>/<task_id>/<name>
        parts = b.path.parts
        if len(parts) != 4:
            errs.append(ValidationError(
                layer=1, code="path_depth",
                message=f"expected tasks/<level>/<task_id>/<name>, got {b.path}",
                path=b.path,
            ))
            continue
        if parts[0] != "tasks":
            errs.append(ValidationError(
                layer=1, code="path_root",
                message=f"first segment must be 'tasks', got {parts[0]!r}",
                path=b.path,
            ))
            continue
        if parts[2] != task_id:
            errs.append(ValidationError(
                layer=1, code="path_task_id",
                message=f"task_id mismatch: expected {task_id!r}, got {parts[2]!r}",
                path=b.path,
            ))
            continue
        if parts[3] not in _ALLOWED_FILE_NAMES:
            errs.append(ValidationError(
                layer=1, code="path_filename",
                message=f"filename {parts[3]!r} not in {sorted(_ALLOWED_FILE_NAMES)}",
                path=b.path,
            ))
    return errs


# ---------------------------------------------------------------------------
# Layer 2: import leak (oracle_pytorch.py)
# ---------------------------------------------------------------------------

_FORBIDDEN_IMPORT_RE = re.compile(
    r"^\s*(?:import\s+(?:vllm|sglang)\b|from\s+(?:vllm|sglang)\b)",
    re.MULTILINE,
)


def _layer2_import_leak(
    blocks: list[FileBlock],
) -> list[ValidationError]:
    errs: list[ValidationError] = []
    for b in blocks:
        if b.path.name != "oracle_pytorch.py":
            continue
        m = _FORBIDDEN_IMPORT_RE.search(b.content)
        if m:
            errs.append(ValidationError(
                layer=2, code="forbidden_import",
                message=f"oracle_pytorch.py must not import upstream "
                        f"engine (matched: {m.group(0).strip()!r})",
                path=b.path,
            ))
    return errs


# ---------------------------------------------------------------------------
# Layer 3: stub-signature gate (function body is empty)
# ---------------------------------------------------------------------------

def _layer3_stub_signature(
    blocks: list[FileBlock],
) -> list[ValidationError]:
    errs: list[ValidationError] = []
    for b in blocks:
        if b.path.name != "oracle_pytorch.py":
            continue
        try:
            tree = ast.parse(b.content)
        except SyntaxError:
            # Layer 5 surfaces syntax errors; skip Layer 3 if unparseable.
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                body = node.body
                if not body:
                    continue
                # Drop a leading docstring (Expr/Constant(str)) before
                # checking stub-ness, so an oracle with a real impl is
                # not falsely flagged just because it has a docstring.
                if (
                    isinstance(body[0], ast.Expr)
                    and isinstance(body[0].value, ast.Constant)
                    and isinstance(body[0].value.value, str)
                ):
                    body = body[1:]
                if not body:
                    errs.append(ValidationError(
                        layer=3, code="stub_pass",
                        message=f"function {node.name!r} body is empty",
                        path=b.path,
                    ))
                    continue
                if len(body) == 1:
                    stmt = body[0]
                    # case: pass
                    if isinstance(stmt, ast.Pass):
                        errs.append(ValidationError(
                            layer=3, code="stub_pass",
                            message=f"function {node.name!r} body is only pass",
                            path=b.path,
                        ))
                        continue
                    # case: raise NotImplementedError(...)
                    if isinstance(stmt, ast.Raise) and stmt.exc is not None:
                        name = _exception_name(stmt.exc)
                        if name == "NotImplementedError":
                            errs.append(ValidationError(
                                layer=3, code="stub_not_implemented",
                                message=f"function {node.name!r} only raises "
                                        f"NotImplementedError",
                                path=b.path,
                            ))
                            continue
                    # case: return torch.empty_like(...)
                    if isinstance(stmt, ast.Return) and stmt.value is not None:
                        if _is_call_to_attr(stmt.value, "torch", "empty_like"):
                            errs.append(ValidationError(
                                layer=3, code="stub_empty_like",
                                message=f"function {node.name!r} returns only "
                                        f"torch.empty_like(...)",
                                path=b.path,
                            ))
                            continue
    return errs


def _exception_name(node: ast.expr) -> str:
    if isinstance(node, ast.Call):
        return _exception_name(node.func)
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return ""


def _is_call_to_attr(node: ast.expr, root: str, attr: str) -> bool:
    if not isinstance(node, ast.Call):
        return False
    func = node.func
    if isinstance(func, ast.Attribute) and func.attr == attr:
        base = func.value
        if isinstance(base, ast.Name) and base.id == root:
            return True
    return False


# ---------------------------------------------------------------------------
# Layer 4: META x FILE consistency by decision
# ---------------------------------------------------------------------------

def _filenames(blocks: list[FileBlock]) -> set[str]:
    return {b.path.name for b in blocks}


def validate_meta_file_consistency(
    meta_block: MetaBlock, file_blocks: list[FileBlock],
) -> list[ValidationError]:
    errs: list[ValidationError] = []
    names = _filenames(file_blocks)
    decision = meta_block.decision
    if decision == "patch-shapes":
        if "shapes.yaml" not in names:
            errs.append(ValidationError(
                layer=4, code="missing_shapes_yaml",
                message="patch-shapes requires shapes.yaml FILE",
            ))
    elif decision == "tolerance-bump":
        if "meta.yaml" not in names:
            errs.append(ValidationError(
                layer=4, code="missing_meta_yaml",
                message="tolerance-bump requires meta.yaml FILE",
            ))
        if "oracle_pytorch.py" in names:
            errs.append(ValidationError(
                layer=4, code="unexpected_oracle",
                message="tolerance-bump must NOT include oracle_pytorch.py FILE",
            ))
    elif decision == "rewrite-primitives":
        if "oracle_pytorch.py" not in names:
            errs.append(ValidationError(
                layer=4, code="missing_oracle",
                message="rewrite-primitives requires oracle_pytorch.py FILE",
            ))
    elif decision in ("defer-to-upstream-oracle", "defer-to-golden"):
        if "meta.yaml" not in names:
            errs.append(ValidationError(
                layer=4, code="missing_meta_yaml",
                message=f"{decision} requires meta.yaml FILE",
            ))
        oracle_action = str(meta_block.payload.get("oracle_action", "unchanged"))
        if oracle_action == "rewrite-primitives":
            if "oracle_pytorch.py" not in names:
                errs.append(ValidationError(
                    layer=4, code="missing_oracle_for_rewrite",
                    message=f"{decision} + oracle_action=rewrite-primitives "
                            "requires oracle_pytorch.py FILE",
                ))
        else:  # unchanged
            if "oracle_pytorch.py" in names:
                errs.append(ValidationError(
                    layer=4, code="unexpected_oracle_for_unchanged",
                    message=f"{decision} + oracle_action=unchanged must NOT "
                            "include oracle_pytorch.py FILE",
                ))
    return errs


# ---------------------------------------------------------------------------
# Meta-schema validate
# ---------------------------------------------------------------------------

def validate_meta(
    meta_block: MetaBlock | None,
    expected_decision_set: set[str],
    expected_keys: set[str],
) -> list[ValidationError]:
    errs: list[ValidationError] = []
    if meta_block is None:
        errs.append(ValidationError(
            layer=4, code="missing_meta",
            message="no META block found",
        ))
        return errs
    if meta_block.decision not in expected_decision_set:
        errs.append(ValidationError(
            layer=4, code="bad_decision",
            message=f"decision={meta_block.decision!r} not in "
                    f"{sorted(expected_decision_set)}",
        ))
    missing = expected_keys - set(meta_block.payload.keys())
    if missing:
        errs.append(ValidationError(
            layer=4, code="missing_keys",
            message=f"META missing required keys: {sorted(missing)}",
        ))
    oa = meta_block.payload.get("oracle_action")
    if oa is not None and oa not in ("unchanged", "rewrite-primitives"):
        errs.append(ValidationError(
            layer=4, code="bad_oracle_action",
            message=f"oracle_action must be unchanged|rewrite-primitives, "
                    f"got {oa!r}",
        ))
    # Forward-compat warning: unknown keys do not fail; logger only.
    unknown = set(meta_block.payload.keys()) - expected_keys - {
        "decision", "oracle_action", "rationale", "evidence",
        "missing_params", "upstream_callable",
    }
    for k in unknown:
        logger.debug("META has unknown forward-compat key: %r", k)
    return errs


# ---------------------------------------------------------------------------
# Layer 5: content-level semantic validation
# ---------------------------------------------------------------------------

def _scan_high_level_ops(
    content: str, path: Path,
) -> list[ValidationError]:
    """AST scan a candidate oracle for any high-level op leaks.

    Detects:
      * ``F.gelu(x)`` / ``torch.nn.functional.gelu(x)`` / ``torch.gelu(x)``
        -> ``_HIGH_LEVEL_OP_NAMES``
      * ``torch.nn.SiLU()(x)`` / ``from torch.nn import SiLU; SiLU()(x)``
        -> ``_HIGH_LEVEL_NN_CLASS_NAMES``
    """
    errs: list[ValidationError] = []
    try:
        tree = ast.parse(content)
    except SyntaxError as exc:
        errs.append(ValidationError(
            layer=5, code="syntax_error",
            message=f"oracle_pytorch.py syntax error: {exc}",
            path=path,
        ))
        return errs

    for node in ast.walk(tree):
        # F.gelu(...) / nn.functional.gelu(...) -- Attribute leaf
        if isinstance(node, ast.Attribute) and isinstance(node.attr, str):
            if node.attr in _HIGH_LEVEL_OP_NAMES:
                errs.append(ValidationError(
                    layer=5, code="high_level_op",
                    message=f"rewrite-primitives oracle must not use "
                            f"high-level op {node.attr!r}",
                    path=path,
                ))
            if node.attr in _HIGH_LEVEL_NN_CLASS_NAMES:
                errs.append(ValidationError(
                    layer=5, code="high_level_nn_class",
                    message=f"rewrite-primitives oracle must not use "
                            f"torch.nn.{node.attr}",
                    path=path,
                ))
        # Plain Name -- e.g. `SiLU()(x)` after `from torch.nn import SiLU`
        if isinstance(node, ast.Name):
            if node.id in _HIGH_LEVEL_NN_CLASS_NAMES:
                errs.append(ValidationError(
                    layer=5, code="high_level_nn_class",
                    message=f"rewrite-primitives oracle must not reference "
                            f"{node.id} (likely from torch.nn import)",
                    path=path,
                ))
    return errs


def validate_meta_content(
    meta_block: MetaBlock,
    file_blocks: list[FileBlock],
    task_id: str,
    repo_root: Path,
) -> list[ValidationError]:
    """Layer 5 -- per-decision content-level validation.

    For ``patch-shapes`` / ``tolerance-bump``: if FILE includes
    ``oracle_pytorch.py``, run the rewrite-primitives high-level op
    scan defensively.

    For ``rewrite-primitives``: require ``oracle_pytorch.py`` AST clean
    of all ``_HIGH_LEVEL_OP_NAMES`` / ``_HIGH_LEVEL_NN_CLASS_NAMES``.

    For ``defer-to-upstream-oracle`` / ``defer-to-golden``: require all
    5 mandatory fields in candidate meta.yaml + canonical dotted path /
    Python identifier checks. If oracle_action=rewrite-primitives, also
    scan candidate oracle.
    """
    errs: list[ValidationError] = []
    decision = meta_block.decision

    # Helper to fetch a FILE block by name
    def _by_name(name: str) -> FileBlock | None:
        for b in file_blocks:
            if b.path.name == name:
                return b
        return None

    if decision in ("patch-shapes", "tolerance-bump"):
        oracle_block = _by_name("oracle_pytorch.py")
        if oracle_block is not None:
            errs.extend(_scan_high_level_ops(oracle_block.content, oracle_block.path))
        return errs

    if decision == "rewrite-primitives":
        oracle_block = _by_name("oracle_pytorch.py")
        if oracle_block is None:
            return errs  # Layer 4 already complained
        errs.extend(_scan_high_level_ops(oracle_block.content, oracle_block.path))
        return errs

    if decision in ("defer-to-upstream-oracle", "defer-to-golden"):
        meta_block_file = _by_name("meta.yaml")
        if meta_block_file is None:
            return errs  # Layer 4 already complained
        try:
            candidate_meta = yaml.safe_load(meta_block_file.content)
        except yaml.YAMLError as exc:
            errs.append(ValidationError(
                layer=5, code="meta_yaml_syntax",
                message=f"candidate meta.yaml YAML syntax error: {exc}",
                path=meta_block_file.path,
            ))
            return errs
        if not isinstance(candidate_meta, dict):
            errs.append(ValidationError(
                layer=5, code="meta_yaml_not_dict",
                message="candidate meta.yaml must be a YAML mapping",
                path=meta_block_file.path,
            ))
            return errs

        # 5 mandatory field gate
        errs.extend(_validate_defer_meta_fields(candidate_meta, meta_block_file.path))

        # Optional: oracle scan for defer-to-* + rewrite-primitives
        oracle_action = str(meta_block.payload.get("oracle_action", "unchanged"))
        if oracle_action == "rewrite-primitives":
            oracle_block = _by_name("oracle_pytorch.py")
            if oracle_block is not None:
                errs.extend(_scan_high_level_ops(oracle_block.content, oracle_block.path))
        return errs

    return errs


def _validate_defer_meta_fields(
    meta: dict, meta_path: Path,
) -> list[ValidationError]:
    errs: list[ValidationError] = []

    # 1) phase_b_capture_required must be bool and == True
    pbcr = meta.get("phase_b_capture_required")
    if not isinstance(pbcr, bool) or pbcr is not True:
        errs.append(ValidationError(
            layer=5, code="bad_phase_b_capture_required",
            message=f"phase_b_capture_required must be True (got {pbcr!r})",
            path=meta_path,
        ))

    # 2) phase_b_capture must be dict
    pbc = meta.get("phase_b_capture")
    if not isinstance(pbc, dict):
        errs.append(ValidationError(
            layer=5, code="bad_phase_b_capture",
            message=f"phase_b_capture must be a dict (got {type(pbc).__name__})",
            path=meta_path,
        ))
    else:
        status = pbc.get("status")
        if status not in ("PENDING", "DONE", "FAILED"):
            errs.append(ValidationError(
                layer=5, code="bad_phase_b_capture_status",
                message=f"phase_b_capture.status must be PENDING/DONE/FAILED, "
                        f"got {status!r}",
                path=meta_path,
            ))
        uc = pbc.get("upstream_callable")
        if not isinstance(uc, dict):
            errs.append(ValidationError(
                layer=5, code="bad_upstream_callable",
                message="phase_b_capture.upstream_callable must be a dict",
                path=meta_path,
            ))
        else:
            module = uc.get("module")
            function = uc.get("function")
            if not isinstance(module, str) or not _CANONICAL_DOTTED_PATH_RE.fullmatch(module or ""):
                errs.append(ValidationError(
                    layer=5, code="bad_upstream_module",
                    message=f"upstream_callable.module is not canonical "
                            f"dotted path: {module!r}",
                    path=meta_path,
                ))
            if not isinstance(function, str) or not _PY_IDENTIFIER_RE.fullmatch(function or ""):
                errs.append(ValidationError(
                    layer=5, code="bad_upstream_function",
                    message=f"upstream_callable.function is not a valid "
                            f"Python identifier: {function!r}",
                    path=meta_path,
                ))

    # 3) phase_a_prime block
    pap = meta.get("phase_a_prime")
    if not isinstance(pap, dict):
        errs.append(ValidationError(
            layer=5, code="bad_phase_a_prime",
            message=f"phase_a_prime must be a dict (got {type(pap).__name__})",
            path=meta_path,
        ))
    else:
        skip = pap.get("skip_invocation")
        if not isinstance(skip, bool) or skip is not True:
            errs.append(ValidationError(
                layer=5, code="bad_skip_invocation",
                message=f"phase_a_prime.skip_invocation must be True "
                        f"(got {skip!r})",
                path=meta_path,
            ))
        status = pap.get("status")
        if status != "SKIPPED_BY_DESIGN":
            errs.append(ValidationError(
                layer=5, code="bad_phase_a_prime_status",
                message=f"phase_a_prime.status must be SKIPPED_BY_DESIGN "
                        f"(got {status!r})",
                path=meta_path,
            ))
        reason = pap.get("reason")
        if not isinstance(reason, str) or not reason.strip():
            errs.append(ValidationError(
                layer=5, code="bad_phase_a_prime_reason",
                message="phase_a_prime.reason must be a non-empty string",
                path=meta_path,
            ))

    # 4) oracle_simplified
    os_field = meta.get("oracle_simplified")
    if os_field != "upstream_oracle_required":
        errs.append(ValidationError(
            layer=5, code="bad_oracle_simplified",
            message=f"oracle_simplified must be 'upstream_oracle_required' "
                    f"(got {os_field!r})",
            path=meta_path,
        ))
    return errs


# ---------------------------------------------------------------------------
# validate_patch / apply_patch (the public two-stage contract)
# ---------------------------------------------------------------------------

def validate_patch(
    meta_block: MetaBlock | None,
    file_blocks: list[FileBlock],
    task_id: str,
    repo_root: Path,
    *,
    expected_decision_set: set[str],
    expected_keys: set[str],
) -> list[ValidationError]:
    """Run Layer 1..5; return all collected errors. Does NOT write."""
    errs: list[ValidationError] = []
    errs.extend(validate_meta(meta_block, expected_decision_set, expected_keys))
    errs.extend(_layer1_path_whitelist(file_blocks, task_id))
    errs.extend(_layer2_import_leak(file_blocks))
    errs.extend(_layer3_stub_signature(file_blocks))
    if meta_block is not None:
        errs.extend(validate_meta_file_consistency(meta_block, file_blocks))
        errs.extend(validate_meta_content(meta_block, file_blocks, task_id, repo_root))
    return errs


def apply_patch(
    meta_block: MetaBlock,
    file_blocks: list[FileBlock],
    task_id: str,
    repo_root: Path,
) -> None:
    """Write each FILE block atomically into ``repo_root``.

    Caller MUST have already (a) run ``validate_patch`` with empty errors
    and (b) ``snapshot_task_baseline``. This function performs ONLY the
    atomic writes -- on OSError it raises so the caller can rollback.
    """
    for fb in file_blocks:
        target = (repo_root / fb.path).resolve()
        if not str(target).startswith(str(repo_root.resolve())):
            # Defense-in-depth -- Layer 1 should have caught path
            # traversal but never trust.
            raise OSError(f"refusing to write outside repo_root: {target}")
        if fb.path.name == "meta.yaml":
            data = yaml.safe_load(fb.content)
            atomic.atomic_write_yaml(target, data)
        elif fb.path.name == "shapes.yaml":
            data = yaml.safe_load(fb.content)
            atomic.atomic_write_yaml(target, data)
        else:
            atomic.atomic_write_text(target, fb.content)


__all__ = [
    "FileBlock",
    "MetaBlock",
    "MultipleMetaBlocksError",
    "ValidationError",
    "apply_patch",
    "parse_blocks",
    "parse_meta",
    "validate_meta",
    "validate_meta_content",
    "validate_meta_file_consistency",
    "validate_patch",
]
