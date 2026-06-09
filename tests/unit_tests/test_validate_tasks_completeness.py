"""W2.3: validate_tasks.py completeness checks.

Covers the four W2.3 validator additions, all phase-aware:

* oracle_pytorch.py exports ``run``; contract.md references the task id;
  an ``upstream_oracle_required`` oracle body has no high-level torch ops.
* kernel-export: ``meta.oracle.gpu_triton`` resolved -> kernel.py must export
  the callable; still ``kernel.py::placeholder`` -> format-only (no
  false-positive on the 94/100 unresolved tasks).
* golden completeness: only enforced once a ``golden/`` dir exists; absent
  ``golden/`` is NOT an error.
* metamorphic check names must be in the known-check registry.

cost_model schema validation is W1.2's job and is covered by
``test_validate_tasks_cost_model.py`` -- not retested here.
"""
from __future__ import annotations

import hashlib
import importlib.util
import sys
from pathlib import Path

import torch
import yaml
from safetensors.torch import save_file

REPO = Path(__file__).resolve().parent.parent.parent
_spec = importlib.util.spec_from_file_location(
    "validate_tasks", REPO / "scripts" / "corpus" / "validate_tasks.py"
)
vt = importlib.util.module_from_spec(_spec)
sys.modules["validate_tasks"] = vt
_spec.loader.exec_module(vt)


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

_DEF_KERNEL = "@triton.jit\ndef the_kernel(i):\n    return {}\n"
_DEF_ORACLE = "def run(i):\n    return {}\n"


def _write_task(
    td: Path,
    *,
    task_id: str = "L1_01_demo",
    kernel_src: str = _DEF_KERNEL,
    oracle_src: str = _DEF_ORACLE,
    contract: str | None = None,
    gpu_triton: str = vt.GPU_TRITON_PLACEHOLDER,
    phase_a_prime: dict | None = None,
    oracle_simplified: str = "simple_pytorch_ok",
    metamorphic_checks: list | None = None,
    param_constraints: dict | None = None,
) -> None:
    """Materialise a minimal schema-valid task dir under ``td``."""
    (td / "kernel.py").write_text(kernel_src)
    sha = hashlib.sha256(kernel_src.encode()).hexdigest()
    (td / "oracle_pytorch.py").write_text(oracle_src)
    knobs = {"B": 2, "N": 4}
    (td / "shapes.yaml").write_text(
        yaml.safe_dump(
            {
                "size_knobs": knobs,
                "public_shapes": [{"id": "P1", "name": "tiny", "knobs": knobs}],
                "hidden_shapes": [{"id": "H1", "name": "hid", "knobs": knobs}],
            }
        )
    )
    short_id = "_".join(task_id.split("_")[:2])
    (td / "contract.md").write_text(
        contract if contract is not None else f"# Contract — {short_id} demo\n"
    )
    rc: dict = {}
    if metamorphic_checks is not None:
        rc["metamorphic_checks"] = metamorphic_checks
    meta = {
        "schema_version": "1.0", "task_id": task_id, "level": "L1", "idx": 1,
        "name": "k", "repo": "vLLM", "dtype_class": "VV",
        "category": "norm", "npu_difficulty": "low",
        "upstream": {"file_sha256": sha},
        "interface": {}, "oracle": {"gpu_triton": gpu_triton},
        "tolerance_class": "_t",
        "oracle_simplified": oracle_simplified, "runner_checks": rc,
        "is_canary": False, "is_hidden_only": False,
        # R1-4: `cost_model` is a required top-level meta key — include a
        # well-formed scalar block so this minimal fixture stays
        # schema-valid (cost_model validation itself is W1.2's job, tested
        # in test_validate_tasks_cost_model.py, not retested here).
        "cost_model": {
            "flops": "B*N", "flops_class": "vector",
            "bytes": "B*N", "bytes_resource": "hbm",
        },
        # WA.1/WA.4: `param_constraints` is a required top-level meta key.
        # Use a reconciled (`scaffold: false`) schema-closed block: WA.4's
        # semantic gate rejects `scaffold: true`, and R4-1 requires all five
        # semantic subsections (empty containers here). These fixtures are
        # not testing param_constraints itself — its own checks live in
        # test_validate_tasks_param_constraints.py and the WA.4 suite.
        "param_constraints": param_constraints if param_constraints is not None
        else {"scaffold": False, "dims": [], "dtypes": {},
              "dists": {}, "scalars": {}, "tensor_invariants": []},
    }
    if phase_a_prime is not None:
        meta["phase_a_prime"] = phase_a_prime
    (td / "meta.yaml").write_text(yaml.safe_dump(meta))


def _run(td: Path) -> list[str]:
    return vt.validate_task(td, {"_t"}, strict=False)


# --------------------------------------------------------------------------
# oracle_pytorch.py exports run / contract.md consistency
# --------------------------------------------------------------------------

def test_oracle_missing_run_reported(tmp_path):
    _write_task(tmp_path, oracle_src="def something_else(i):\n    return {}\n")
    errs = _run(tmp_path)
    assert any("does not export 'run'" in e for e in errs), errs


def test_oracle_run_present_passes(tmp_path):
    _write_task(tmp_path)  # default oracle exports run
    errs = _run(tmp_path)
    assert not any("does not export 'run'" in e for e in errs), errs


def test_contract_missing_task_id_reported(tmp_path):
    _write_task(tmp_path, task_id="L1_07_demo", contract="# Some other task\n")
    errs = _run(tmp_path)
    assert any("does not reference task id" in e for e in errs), errs


def test_contract_referencing_task_id_passes(tmp_path):
    _write_task(tmp_path, task_id="L1_07_demo")  # default contract has L1_07
    errs = _run(tmp_path)
    assert not any("does not reference task id" in e for e in errs), errs


# --------------------------------------------------------------------------
# upstream_oracle_required oracle body must have no high-level torch ops
# --------------------------------------------------------------------------

def test_upstream_oracle_high_level_op_reported(tmp_path):
    bad = (
        "import torch.nn.functional as F\n"
        "def run(i):\n"
        "    return {'y': F.silu(i['x'])}\n"
    )
    _write_task(
        tmp_path, oracle_src=bad, oracle_simplified="upstream_oracle_required"
    )
    errs = _run(tmp_path)
    # R9-3: the hit is reported by its alias-resolved CANONICAL path —
    # `F.silu` resolves to `torch.nn.functional.silu`.
    assert any(
        "high-level torch op" in e and "torch.nn.functional.silu" in e
        for e in errs
    ), errs


def test_upstream_oracle_torch_softmax_reported(tmp_path):
    bad = (
        "import torch\n"
        "def run(i):\n"
        "    return {'y': torch.softmax(i['x'], dim=-1)}\n"
    )
    _write_task(
        tmp_path, oracle_src=bad, oracle_simplified="upstream_oracle_required"
    )
    errs = _run(tmp_path)
    assert any("torch.softmax" in e for e in errs), errs


def test_upstream_oracle_bare_import_no_call_passes(tmp_path):
    # importing F but hand-rolling the math (L1_13 / L1_14 pattern) is fine:
    # the check is on actual CALLS, not imports.
    ok = (
        "import torch.nn.functional as F  # noqa: F401\n"
        "def run(i):\n"
        "    x = i['x']\n"
        "    return {'y': x * (x > 0)}\n"
    )
    _write_task(
        tmp_path, oracle_src=ok, oracle_simplified="upstream_oracle_required"
    )
    errs = _run(tmp_path)
    assert not any("high-level torch op" in e for e in errs), errs


def test_high_level_op_allowed_for_non_upstream_tier(tmp_path):
    # the high-level-op ban only applies to upstream_oracle_required tasks.
    body = (
        "import torch.nn.functional as F\n"
        "def run(i):\n"
        "    return {'y': F.silu(i['x'])}\n"
    )
    _write_task(
        tmp_path, oracle_src=body, oracle_simplified="simple_pytorch_ok"
    )
    errs = _run(tmp_path)
    assert not any("high-level torch op" in e for e in errs), errs


# --------------------------------------------------------------------------
# R9-3: the high-level-op guard resolves import aliases — a non-`F` alias
# or a direct-symbol import must not be able to slip a high-level op past.
# --------------------------------------------------------------------------

def test_upstream_oracle_aliased_functional_import_caught(tmp_path):
    """`import torch.nn.functional as G; G.silu(...)` — the old narrow
    leaf-name match only knew `F`; an arbitrary alias bypassed it."""
    bad = (
        "import torch.nn.functional as G\n"
        "def run(i):\n"
        "    return {'y': G.silu(i['x'])}\n"
    )
    _write_task(
        tmp_path, oracle_src=bad, oracle_simplified="upstream_oracle_required"
    )
    errs = _run(tmp_path)
    assert any(
        "high-level torch op" in e and "torch.nn.functional.silu" in e
        for e in errs
    ), errs


def test_upstream_oracle_from_import_symbol_caught(tmp_path):
    """`from torch.nn.functional import silu as s; s(...)` — a direct
    symbol import with no module qualifier at the call site at all."""
    bad = (
        "from torch.nn.functional import silu as s\n"
        "def run(i):\n"
        "    return {'y': s(i['x'])}\n"
    )
    _write_task(
        tmp_path, oracle_src=bad, oracle_simplified="upstream_oracle_required"
    )
    errs = _run(tmp_path)
    assert any(
        "high-level torch op" in e and "torch.nn.functional.silu" in e
        for e in errs
    ), errs


def test_upstream_oracle_from_torch_nn_functional_alias_caught(tmp_path):
    """`from torch.nn import functional as F; F.softmax(...)` — the alias
    binds the functional module via a `from ... import` form."""
    bad = (
        "from torch.nn import functional as F\n"
        "def run(i):\n"
        "    return {'y': F.softmax(i['x'], dim=-1)}\n"
    )
    _write_task(
        tmp_path, oracle_src=bad, oracle_simplified="upstream_oracle_required"
    )
    errs = _run(tmp_path)
    assert any(
        "high-level torch op" in e and "torch.nn.functional.softmax" in e
        for e in errs
    ), errs


def test_upstream_oracle_unaliased_dotted_import_caught(tmp_path):
    """`import torch.nn.functional` (no alias) → the call site must use the
    full `torch.nn.functional.silu` path; the resolver follows it."""
    bad = (
        "import torch.nn.functional\n"
        "def run(i):\n"
        "    return {'y': torch.nn.functional.gelu(i['x'])}\n"
    )
    _write_task(
        tmp_path, oracle_src=bad, oracle_simplified="upstream_oracle_required"
    )
    errs = _run(tmp_path)
    assert any(
        "high-level torch op" in e and "torch.nn.functional.gelu" in e
        for e in errs
    ), errs


def test_upstream_oracle_non_torch_alias_not_flagged(tmp_path):
    """A `silu`-named call from an unrelated (non-torch) module must not
    trip the check — the resolver only recognises torch-rooted aliases."""
    ok = (
        "from my_local_helpers import silu\n"
        "def run(i):\n"
        "    return {'y': silu(i['x'])}\n"
    )
    _write_task(
        tmp_path, oracle_src=ok, oracle_simplified="upstream_oracle_required"
    )
    errs = _run(tmp_path)
    assert not any("high-level torch op" in e for e in errs), errs


# --------------------------------------------------------------------------
# R12-1: the high-level-op guard covers the `torch.nn` module/class form —
# `torch.nn.SiLU()(x)` is semantically equivalent to
# `torch.nn.functional.silu(x)` and must be rejected just the same.
# --------------------------------------------------------------------------

def test_upstream_oracle_nn_module_class_form_caught(tmp_path):
    """`torch.nn.SiLU()(x)` — the construct-then-call nn.Module form. The
    old leaf-name match only knew the lowercase `functional` function."""
    bad = (
        "import torch\n"
        "def run(i):\n"
        "    return {'y': torch.nn.SiLU()(i['x'])}\n"
    )
    _write_task(
        tmp_path, oracle_src=bad, oracle_simplified="upstream_oracle_required"
    )
    errs = _run(tmp_path)
    assert any(
        "high-level torch op" in e and "torch.nn.SiLU" in e for e in errs
    ), errs


def test_upstream_oracle_nn_class_import_alias_caught(tmp_path):
    """`from torch.nn import SiLU as S; S()(x)` — the nn.Module class
    imported under an alias, then constructed and called."""
    bad = (
        "from torch.nn import SiLU as S\n"
        "def run(i):\n"
        "    return {'y': S()(i['x'])}\n"
    )
    _write_task(
        tmp_path, oracle_src=bad, oracle_simplified="upstream_oracle_required"
    )
    errs = _run(tmp_path)
    assert any(
        "high-level torch op" in e and "torch.nn.SiLU" in e for e in errs
    ), errs


def test_upstream_oracle_nn_submodule_alias_class_form_caught(tmp_path):
    """`from torch import nn; nn.Softmax(dim=-1)(x)` — the `nn` submodule
    bound by name, then a class attribute constructed and called."""
    bad = (
        "from torch import nn\n"
        "def run(i):\n"
        "    return {'y': nn.Softmax(dim=-1)(i['x'])}\n"
    )
    _write_task(
        tmp_path, oracle_src=bad, oracle_simplified="upstream_oracle_required"
    )
    errs = _run(tmp_path)
    assert any(
        "high-level torch op" in e and "torch.nn.Softmax" in e for e in errs
    ), errs


def test_upstream_oracle_nn_tanh_class_form_caught(tmp_path):
    """`torch.nn.Tanh()(x)` — `tanh`/`Tanh` is a composite activation
    (R11 ruling), forbidden in both function and module/class form."""
    bad = (
        "import torch\n"
        "def run(i):\n"
        "    return {'y': torch.nn.Tanh()(i['x'])}\n"
    )
    _write_task(
        tmp_path, oracle_src=bad, oracle_simplified="upstream_oracle_required"
    )
    errs = _run(tmp_path)
    assert any(
        "high-level torch op" in e and "torch.nn.Tanh" in e for e in errs
    ), errs


def test_upstream_oracle_functional_silu_still_caught(tmp_path):
    """Regression: the functional lowercase form `torch.nn.functional.silu`
    must still be caught after the module/class-form extension (R12-1)."""
    bad = (
        "import torch.nn.functional as F\n"
        "def run(i):\n"
        "    return {'y': F.silu(i['x'])}\n"
    )
    _write_task(
        tmp_path, oracle_src=bad, oracle_simplified="upstream_oracle_required"
    )
    errs = _run(tmp_path)
    assert any(
        "high-level torch op" in e and "torch.nn.functional.silu" in e
        for e in errs
    ), errs


def test_upstream_oracle_primitive_math_not_flagged(tmp_path):
    """Basic math primitives (`torch.exp` / `torch.sqrt` / `torch.rsqrt`)
    are allowed (R11 ruling) and must not false-positive — neither the
    function nor the module/class detector should flag them."""
    ok = (
        "import torch\n"
        "def run(i):\n"
        "    x = i['x']\n"
        "    return {'y': torch.exp(x) * torch.rsqrt(torch.sqrt(x) + 1.0)}\n"
    )
    _write_task(
        tmp_path, oracle_src=ok, oracle_simplified="upstream_oracle_required"
    )
    errs = _run(tmp_path)
    assert not any("high-level torch op" in e for e in errs), errs


# --------------------------------------------------------------------------
# kernel-export (phase-aware)
# --------------------------------------------------------------------------

def test_placeholder_gpu_triton_no_export_required(tmp_path):
    # gpu_triton still placeholder + no phase_a_prime: kernel.py need NOT
    # export anything -- this is the 94/100 unresolved-task case.
    _write_task(
        tmp_path,
        kernel_src="@triton.jit\ndef whatever(i):\n    return {}\n",
        gpu_triton=vt.GPU_TRITON_PLACEHOLDER,
    )
    errs = _run(tmp_path)
    assert not any("gpu_triton entrypoint" in e for e in errs), errs


def test_resolved_gpu_triton_missing_export_reported(tmp_path):
    # gpu_triton resolved to kernel.py::real_entry but kernel.py exports a
    # differently-named function -> error.
    _write_task(
        tmp_path,
        kernel_src="@triton.jit\ndef other_name(i):\n    return {}\n",
        gpu_triton="kernel.py::real_entry",
    )
    errs = _run(tmp_path)
    assert any("gpu_triton entrypoint" in e and "real_entry" in e for e in errs), errs


def test_resolved_gpu_triton_present_export_passes(tmp_path):
    _write_task(
        tmp_path,
        kernel_src="@triton.jit\ndef real_entry(i):\n    return {}\n",
        gpu_triton="kernel.py::real_entry",
    )
    errs = _run(tmp_path)
    assert not any("gpu_triton entrypoint" in e for e in errs), errs


def test_phase_a_prime_pass_forces_export_check(tmp_path):
    # phase_a_prime.status == PASS forces the export check even if gpu_triton
    # were still the placeholder string.
    _write_task(
        tmp_path,
        kernel_src="@triton.jit\ndef other_name(i):\n    return {}\n",
        gpu_triton="kernel.py::expected_entry",
        phase_a_prime={"status": "PASS"},
    )
    errs = _run(tmp_path)
    assert any("gpu_triton entrypoint" in e for e in errs), errs


def test_malformed_gpu_triton_field_reported(tmp_path):
    _write_task(tmp_path, gpu_triton="not_a_valid_ref")
    errs = _run(tmp_path)
    assert any("gpu_triton" in e and "kernel.py::<symbol>" in e for e in errs), errs


# --------------------------------------------------------------------------
# golden completeness (R4-3: strictly required for upstream_oracle_required)
# --------------------------------------------------------------------------

def test_no_golden_dir_is_an_error_for_upstream_task(tmp_path):
    # R4-3: an upstream_oracle_required task with NO golden/ dir is a
    # validation error — the offline upstream-kernel golden is the only
    # golden source for this tier, so a complete golden/ set is required
    # (the static "missing provider golden" gate). The full corpus is
    # expected RED on this check until W4.3 Phase 4 collects the
    # artifacts; the validator layer carries the red→green transition.
    _write_task(tmp_path, oracle_simplified="upstream_oracle_required")
    errs = _run(tmp_path)
    assert any("no golden/ directory" in e for e in errs), errs


def test_no_golden_dir_not_an_error_for_non_upstream_task(tmp_path):
    # The golden requirement applies ONLY to upstream_oracle_required
    # tasks — a plain PyTorch-oracle task needs no golden/ directory.
    _write_task(tmp_path, oracle_simplified="simple_pytorch_ok")
    errs = _run(tmp_path)
    assert not any("golden" in e for e in errs), errs


def test_golden_dir_present_incomplete_reported(tmp_path):
    # golden/ exists but is missing the hidden-shape safetensors.
    _write_task(tmp_path, oracle_simplified="upstream_oracle_required")
    golden = tmp_path / "golden"
    golden.mkdir()
    (golden / "P1.safetensors").write_bytes(b"\x00")  # H1 missing
    errs = _run(tmp_path)
    assert any("golden" in e and "H1" in e for e in errs), errs


def test_golden_dir_complete_passes(tmp_path):
    # R7-2: each artifact must be a LOADABLE safetensors (the shared
    # `golden_artifact_usable` predicate), so the fixture writes real
    # safetensors, not placeholder bytes.
    _write_task(tmp_path, oracle_simplified="upstream_oracle_required")
    golden = tmp_path / "golden"
    golden.mkdir()
    save_file({"y": torch.zeros(4)}, str(golden / "P1.safetensors"))
    save_file({"y": torch.zeros(4)}, str(golden / "H1.safetensors"))
    errs = _run(tmp_path)
    assert not any("golden" in e for e in errs), errs


def test_golden_not_checked_for_non_upstream_tier(tmp_path):
    # a non-upstream_oracle_required task with a stray golden/ dir is not
    # subject to the golden completeness check.
    _write_task(tmp_path, oracle_simplified="simple_pytorch_ok")
    (tmp_path / "golden").mkdir()
    errs = _run(tmp_path)
    assert not any("golden" in e for e in errs), errs


# --------------------------------------------------------------------------
# golden gate is exempt for NON-score-eligible upstream tasks
# --------------------------------------------------------------------------

_STUB_ORACLE = "def run(i):\n    raise NotImplementedError\n"


def test_golden_not_required_for_non_score_eligible_upstream_task(tmp_path):
    # A non-score-eligible upstream_oracle_required task (stub oracle —
    # `raise NotImplementedError`) has a placeholder kernel and can never
    # reach a promoted state, so no offline golden can ever be captured
    # for it. It is an acknowledged special case and must NOT be held RED
    # by the golden completeness gate (which would keep --release-gate
    # permanently red). No golden/ dir, no error.
    _write_task(
        tmp_path,
        oracle_src=_STUB_ORACLE,
        oracle_simplified="upstream_oracle_required",
    )
    errs = _run(tmp_path)
    assert not any("golden/ directory" in e for e in errs), errs


def test_golden_not_required_for_phase_c_manual_upstream_task(tmp_path):
    # The other non-score-eligible path: a PHASE-C-MANUAL shapes.yaml is
    # likewise exempt from the golden gate. The size_knobs are kept so
    # the unrelated cost_model formula check still passes (this test
    # isolates the golden dimension).
    _write_task(
        tmp_path,
        oracle_simplified="upstream_oracle_required",
    )
    (tmp_path / "shapes.yaml").write_text(
        yaml.safe_dump(
            {
                "comment": "PHASE-C-MANUAL",
                "size_knobs": {"B": 2, "N": 4},
                "public_shapes": [],
                "hidden_shapes": [],
            }
        )
    )
    errs = _run(tmp_path)
    assert not any("golden/ directory" in e for e in errs), errs


def test_golden_still_required_for_score_eligible_upstream_task(tmp_path):
    # Guard: the exemption is strictly scoped to non-score-eligible
    # tasks. A SCORE-eligible upstream_oracle_required task (real oracle,
    # non-PHASE-C-MANUAL shapes) still REQUIRES a complete golden/ set —
    # the exemption must not silently weaken the gate for the 18
    # score-eligible upstream tasks.
    _write_task(
        tmp_path,
        oracle_src=_DEF_ORACLE,  # real `run`, not a stub
        oracle_simplified="upstream_oracle_required",
    )
    errs = _run(tmp_path)
    assert any("no golden/ directory" in e for e in errs), errs


# --------------------------------------------------------------------------
# metamorphic check name registry
# --------------------------------------------------------------------------

def test_unknown_metamorphic_check_reported(tmp_path):
    _write_task(
        tmp_path,
        metamorphic_checks=[{"name": "totally_made_up_check"}],
    )
    errs = _run(tmp_path)
    assert any("unknown check" in e for e in errs), errs


def test_known_metamorphic_check_passes(tmp_path):
    _write_task(
        tmp_path,
        metamorphic_checks=[{"name": "differential_perturbation_general"}],
    )
    errs = _run(tmp_path)
    assert not any("metamorphic" in e for e in errs), errs


def test_known_metamorphic_check_string_form_passes(tmp_path):
    # the registry check accepts both the dict and the bare-string item form.
    _write_task(
        tmp_path,
        metamorphic_checks=["row_independence"],
    )
    errs = _run(tmp_path)
    assert not any("metamorphic" in e for e in errs), errs


def test_empty_metamorphic_checks_passes(tmp_path):
    # 95/100 tasks declare an empty metamorphic_checks list -- not an error.
    _write_task(tmp_path, metamorphic_checks=[])
    errs = _run(tmp_path)
    assert not any("metamorphic" in e for e in errs), errs


# --------------------------------------------------------------------------
# param_constraints scaffold exemption for non-score-eligible tasks
# --------------------------------------------------------------------------

_SCAFFOLD_BLOCK = {
    "scaffold": True, "dims": [], "dtypes": {},
    "dists": {}, "scalars": {}, "tensor_invariants": [],
}


def test_scaffold_not_reported_for_non_score_eligible_task(tmp_path):
    # A non-score-eligible task (stub oracle -- `raise NotImplementedError`)
    # with `param_constraints.scaffold: true` is an acknowledged special
    # case (L2_08_gumbel_sample). The
    # `_validate_param_constraints_semantic` scaffold gate must exempt it --
    # SAME `is_score_eligible` predicate as the golden exemption -- so
    # the task keeps `scaffold: true` honestly (genuinely un-reconciled)
    # without holding `--release-gate` permanently RED.
    _write_task(
        tmp_path,
        oracle_src=_STUB_ORACLE,
        param_constraints=_SCAFFOLD_BLOCK,
    )
    errs = _run(tmp_path)
    assert not any("scaffold" in e for e in errs), errs


def test_scaffold_still_reported_for_score_eligible_task(tmp_path):
    # Guard: the exemption is strictly scoped to non-score-eligible
    # tasks. A SCORE-eligible task (real `run` oracle) carrying
    # `scaffold: true` is still reported -- the exemption must not silently
    # let real, scored tasks ship un-reconciled `param_constraints`.
    _write_task(
        tmp_path,
        oracle_src=_DEF_ORACLE,  # real `run` -> score-eligible
        param_constraints=_SCAFFOLD_BLOCK,
    )
    errs = _run(tmp_path)
    assert any("scaffold" in e for e in errs), errs
