#!/usr/bin/env python3
"""extract_kernel.py — extract a GPU triton kernel from vLLM/SGLang upstream.

Per v4.13 Phase 1.1 + R7-6. Outputs a clean kernel.py for tasks/level{1,2,3}/<task_id>/.

Usage:
    python scripts/extract_kernel.py \\
        --task L1_12_rms_norm \\
        --kernel _rms_norm_kernel \\
        --from infer/vllm/vllm/model_executor/layers/batch_invariant.py \\
        --out tasks/level1/L1_12_rms_norm/kernel.py

Resolves:
- Extracts the `@triton.jit` kernel function
- Collects helper functions called by it
- Resolves imports
- Records commit SHA + file SHA in metadata

Per v4.13 §extraction_failure_taxonomy:
- constexpr_runtime_dep / autotune_multi_config / runtime_meta_param / deep_helper_chain
  / cpp_stub_dep / dynamic_shape_branch / vendor_specific_op / private_no_source
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import logging
import subprocess
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


class KernelExtractor(ast.NodeVisitor):
    """Walks AST of an upstream file and collects the named kernel + its dependencies."""

    def __init__(self, target_name: str, source: str):
        self.target_name = target_name
        self.source = source
        self.lines = source.splitlines(keepends=True)
        self.collected: dict[str, str] = {}  # name -> source snippet
        self.imports: list[str] = []
        self.found_target = False
        self.has_triton_jit = False

    def visit_Import(self, node: ast.Import) -> None:
        # Capture top-level imports
        snippet = "".join(self.lines[node.lineno - 1 : node.end_lineno])
        self.imports.append(snippet.rstrip())

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        snippet = "".join(self.lines[node.lineno - 1 : node.end_lineno])
        self.imports.append(snippet.rstrip())

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        is_triton_jit = any(
            self._is_triton_jit_decorator(d) for d in node.decorator_list
        )
        is_target = node.name == self.target_name

        if is_target:
            self.found_target = True
            if is_triton_jit:
                self.has_triton_jit = True

        if is_target or is_triton_jit:
            # Capture full source (including decorators) — find start of first decorator
            start_line = (
                min(d.lineno for d in node.decorator_list)
                if node.decorator_list
                else node.lineno
            )
            snippet = "".join(self.lines[start_line - 1 : node.end_lineno])
            self.collected[node.name] = snippet.rstrip() + "\n"

        self.generic_visit(node)

    @staticmethod
    def _is_triton_jit_decorator(deco: ast.expr) -> bool:
        # @triton.jit | @triton.jit(launch_metadata=...) | @triton.autotune(...)
        if isinstance(deco, ast.Attribute) and deco.attr == "jit":
            return True
        if isinstance(deco, ast.Call):
            if isinstance(deco.func, ast.Attribute) and deco.func.attr in {
                "jit",
                "autotune",
            }:
                return True
        return False


def get_git_commit_sha(repo_path: Path) -> str:
    """Get current HEAD commit SHA of upstream repo."""
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_path,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def get_file_sha256(file_path: Path) -> str:
    return hashlib.sha256(file_path.read_bytes()).hexdigest()


def extract(args: argparse.Namespace) -> dict:
    source_file = Path(args.from_path).resolve()
    if not source_file.exists():
        sys.exit(f"ERROR: source file not found: {source_file}")

    source = source_file.read_text()
    tree = ast.parse(source, filename=str(source_file))

    extractor = KernelExtractor(args.kernel, source)
    extractor.visit(tree)

    if not extractor.found_target:
        sys.exit(f"ERROR: kernel '{args.kernel}' not found in {source_file}")

    if not extractor.has_triton_jit:
        logger.warning(
            "Target '%s' is not decorated with @triton.jit", args.kernel
        )

    # Resolve upstream repo + commit SHA
    upstream_repo_root = source_file
    while upstream_repo_root.parent != upstream_repo_root:
        if (upstream_repo_root / ".git").exists():
            break
        upstream_repo_root = upstream_repo_root.parent
    else:
        logger.warning("No upstream git repo found; commit_sha will be 'unknown'")
        commit_sha = "unknown"
    if (upstream_repo_root / ".git").exists():
        try:
            commit_sha = get_git_commit_sha(upstream_repo_root)
        except subprocess.CalledProcessError:
            commit_sha = "unknown"

    file_sha = get_file_sha256(source_file)
    rel_path = source_file.relative_to(upstream_repo_root)

    # Compose output
    header = f'''"""Task {args.task} — GPU triton reference

Source: {rel_path}
Upstream repo: {upstream_repo_root.name}
Commit SHA: {commit_sha}
File SHA-256: {file_sha}

Extracted by tools/extract_kernel.py at Phase 1.1.
Per v4.13 spec, this kernel.py is read-only and hash-verified by `tab` evaluator.
"""
'''

    imports_block = "\n".join(extractor.imports)
    functions_block = "\n\n".join(extractor.collected.values())

    output_text = f"{header}\n{imports_block}\n\n\n{functions_block}\n"

    if args.out:
        out_path = Path(args.out).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(output_text)
        logger.info("Wrote kernel to %s", out_path)
    else:
        sys.stdout.write(output_text)

    return {
        "task_id": args.task,
        "kernel_name": args.kernel,
        "upstream_path": str(rel_path),
        "commit_sha": commit_sha,
        "file_sha256": file_sha,
        "collected_functions": list(extractor.collected.keys()),
        "imports_count": len(extractor.imports),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True, help="Task ID, e.g. L1_12_rms_norm")
    parser.add_argument("--kernel", required=True, help="Kernel function name to extract")
    parser.add_argument(
        "--from",
        dest="from_path",
        required=True,
        help="Upstream Python file path",
    )
    parser.add_argument(
        "--out",
        help="Output kernel.py path (default: stdout)",
    )
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    metadata = extract(args)
    logger.info("Extraction metadata: %s", metadata)
    return 0


if __name__ == "__main__":
    sys.exit(main())
