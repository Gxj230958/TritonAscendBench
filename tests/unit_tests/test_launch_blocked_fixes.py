from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import torch
import yaml

from triton_ascend_bench.inputs import build_inputs
from triton_ascend_bench.upstream_shims.vllm_runtime_hygiene import (
    clear_conflicting_vllm_registrations,
    install_vllm_source_version_shim,
)

REPO = Path(__file__).resolve().parents[2]


def _load_script(rel_path: str, module_name: str):
    path = REPO / rel_path
    spec = importlib.util.spec_from_file_location(module_name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_vllm_source_version_shim_falls_back_to_source_tree(tmp_path, monkeypatch):
    vroot = tmp_path / "vllm_src"
    pkg = vroot / "vllm"
    pkg.mkdir(parents=True)
    (pkg / "version.py").write_text('__version__ = "0.9.9-src"\n')

    from importlib import metadata as importlib_metadata

    orig_version = importlib_metadata.version

    def fake_version(name: str) -> str:
        if name == "vllm":
            raise importlib_metadata.PackageNotFoundError
        return f"ok:{name}"

    monkeypatch.setattr(importlib_metadata, "version", fake_version)
    monkeypatch.syspath_prepend(str(vroot))

    assert install_vllm_source_version_shim(vroot) is True
    assert importlib_metadata.version("vllm") == "0.9.9-src"
    assert importlib_metadata.version("other") == "ok:other"

    monkeypatch.setattr(importlib_metadata, "version", orig_version)


def test_clear_conflicting_vllm_registrations_pops_duplicate_names(monkeypatch):
    fake_mod = types.ModuleType("vllm.model_executor.custom_op")
    fake_mod.op_registry = {"dup_op": object(), "keep": object()}
    fake_mod.op_registry_oot = {"dup_op": object(), "keep": object()}

    saved = {
        k: sys.modules.get(k)
        for k in (
            "vllm",
            "vllm.model_executor",
            "vllm.model_executor.custom_op",
        )
    }
    sys.modules["vllm"] = types.ModuleType("vllm")
    sys.modules["vllm.model_executor"] = types.ModuleType("vllm.model_executor")
    sys.modules["vllm.model_executor.custom_op"] = fake_mod
    try:
        src = '@CustomOp.register("dup_op")\nclass X: pass\n'
        removed = clear_conflicting_vllm_registrations(src)
        assert removed == ["dup_op"]
        assert "dup_op" not in fake_mod.op_registry
        assert "dup_op" not in fake_mod.op_registry_oot
        assert "keep" in fake_mod.op_registry
    finally:
        for key, val in saved.items():
            if val is None:
                sys.modules.pop(key, None)
            else:
                sys.modules[key] = val


def test_l2_10_shape_lint_passes_after_block_k_fix():
    lint = _load_script(
        "scripts/lint_shapes_against_kernel.py",
        "lint_shapes_against_kernel_test",
    )
    td = REPO / "tasks" / "level2" / "L2_10_w8a8_block_int8_matmul"
    assert lint.lint_task(td) == []


def test_shape_lint_flags_small_int8_tl_dot_block_k(tmp_path):
    lint = _load_script(
        "scripts/lint_shapes_against_kernel.py",
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


def test_l2_34_shapes_and_oracle_follow_wrapper_contract():
    td = (
        REPO / "tasks" / "level2"
        / "L2_34_silu_mul_static_tensorwise_quant_triton_kernel_for_cutlass_moe"
    )
    shapes = yaml.safe_load((td / "shapes.yaml").read_text())
    inputs = build_inputs(shapes, "P1", seed=0)

    assert inputs["input"].ndim == 3
    assert inputs["output"].dtype == torch.float8_e4m3fn
    assert inputs["masked_m"].ndim == 1

    oracle = _load_script(
        str(td.relative_to(REPO) / "oracle_pytorch.py"),
        "l2_34_oracle_test",
    )
    out = oracle.run(inputs)["output"]
    assert out.shape == inputs["output"].shape
    assert out.dtype == inputs["output"].dtype
