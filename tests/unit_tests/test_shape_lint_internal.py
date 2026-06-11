"""Shape-lint checks against the internal lint_shapes_against_kernel helper.

INTERNAL: loads scripts/internal/lint_shapes_against_kernel.py, which is not in
the release tarball — this test file is export-ignored alongside it.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[2]


def _load_script(rel_path: str, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, str(REPO / rel_path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_l2_10_shape_lint_passes_after_block_k_fix():
    lint = _load_script(
        "scripts/internal/lint_shapes_against_kernel.py",
        "lint_shapes_against_kernel_test",
    )
    td = REPO / "tasks" / "level2" / "L2_17_w8a8_block_int8_matmul"
    assert lint.lint_task(td) == []


def test_shape_lint_flags_small_int8_tl_dot_block_k(tmp_path):
    lint = _load_script(
        "scripts/internal/lint_shapes_against_kernel.py",
        "lint_shapes_against_kernel_bad_test",
    )
    td = tmp_path / "L2_bad"
    td.mkdir()
    (td / "kernel.py").write_text(
        "import triton.language as tl\n"
        "def wrap():\n"
        "    tl.dot(0, 0)\n"
    )
    (td / "shapes.yaml").write_text(
        yaml.safe_dump(
            {
                "public_shapes": [{"id": "P1", "knobs": {"B": 2, "D": 16, "N": 8}}],
                "hidden_shapes": [],
                "inputs": {
                    "A": {"dims": ["B", "D"], "dtype": "int8", "dist": "randint:0:8"},
                    "B": {"dims": ["N", "D"], "dtype": "int8", "dist": "randint:0:8"},
                    "As": {"dims": ["B", "1"], "dtype": "float32", "dist": "rand"},
                    "Bs": {"dims": ["1", "1"], "dtype": "float32", "dist": "rand"},
                    "block_size": {"scalar_seq": [8, 8]},
                },
            },
            sort_keys=False,
        )
    )
    findings = lint.lint_task(td)
    assert any(f.code == "int8_tl_dot_block_k_lt_32" for f in findings)
