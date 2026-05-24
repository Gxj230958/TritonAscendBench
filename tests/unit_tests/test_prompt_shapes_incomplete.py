"""Tests for ``automation.prompts.shapes_incomplete`` (Unit 1.10)."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import yaml

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPTS = _REPO_ROOT / "scripts"


@pytest.fixture(autouse=True)
def _ensure_scripts_on_path():
    inserted = False
    if str(_SCRIPTS) not in sys.path:
        sys.path.insert(0, str(_SCRIPTS))
        inserted = True
    yield
    if inserted:
        try:
            sys.path.remove(str(_SCRIPTS))
        except ValueError:
            pass


def _scaffold(tmp_path: Path, task_id: str = "L1_06_swiglu", level: str = "L1") -> Path:
    td = tmp_path / "tasks" / level / task_id
    td.mkdir(parents=True)
    (td / "meta.yaml").write_text(
        yaml.safe_dump({"schema_version": "1.0", "name": "swiglu", "kind": "triton"}),
        encoding="utf-8",
    )
    (td / "shapes.yaml").write_text("public_shapes: []\n", encoding="utf-8")
    (td / "oracle_pytorch.py").write_text("def oracle(x):\n    return x\n", encoding="utf-8")
    vd = tmp_path / "tasks" / "_vendored"
    vd.mkdir(parents=True)
    (vd / f"{task_id}_wrapper.py").write_text(
        "def swiglu(x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:\n"
        "    ...\n",
        encoding="utf-8",
    )
    infer = tmp_path / "infer" / "vllm"
    infer.mkdir(parents=True)
    (infer / "model.py").write_text(
        "from x import y\n# call site: swiglu(x, gate)\nout = swiglu(x, gate)\n",
        encoding="utf-8",
    )
    return td


def test_build_prompt_happy(tmp_path):
    from automation.prompts import shapes_incomplete

    _scaffold(tmp_path)
    messages = shapes_incomplete.build_prompt("L1_06_swiglu", repo_root=tmp_path)
    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"
    assert "patch-shapes" in messages[0]["content"]
    assert "### META" in messages[0]["content"]
    assert "L1_06_swiglu" in messages[1]["content"]
    assert "swiglu" in messages[1]["content"]


def test_build_prompt_includes_high_level_op_warning(tmp_path):
    from automation.prompts import shapes_incomplete

    _scaffold(tmp_path)
    messages = shapes_incomplete.build_prompt("L1_06_swiglu", repo_root=tmp_path)
    sys_content = messages[0]["content"]
    # At least the famous gelu / silu / softmax names must be listed.
    for name in ("silu", "gelu", "softmax", "layer_norm", "rms_norm"):
        assert name in sys_content


def test_build_prompt_includes_caller_evidence(tmp_path):
    from automation.prompts import shapes_incomplete

    _scaffold(tmp_path)
    messages = shapes_incomplete.build_prompt("L1_06_swiglu", repo_root=tmp_path)
    assert "infer/vllm/model.py" in messages[1]["content"]


def test_build_prompt_missing_wrapper_raises(tmp_path):
    from automation.prompts import shapes_incomplete

    task_id = "L1_06_swiglu"
    td = tmp_path / "tasks" / "L1" / task_id
    td.mkdir(parents=True)
    (td / "meta.yaml").write_text("schema_version: 1.0\n", encoding="utf-8")
    # No _vendored/ dir at all.
    with pytest.raises(shapes_incomplete.WrapperSourceMissingError):
        shapes_incomplete.build_prompt(task_id, repo_root=tmp_path)


def test_build_prompt_no_caller_does_not_block(tmp_path):
    from automation.prompts import shapes_incomplete

    task_id = "L1_06_swiglu"
    td = tmp_path / "tasks" / "L1" / task_id
    td.mkdir(parents=True)
    (td / "meta.yaml").write_text("schema_version: 1.0\n", encoding="utf-8")
    (td / "shapes.yaml").write_text("public_shapes: []\n", encoding="utf-8")
    vd = tmp_path / "tasks" / "_vendored"
    vd.mkdir(parents=True)
    (vd / f"{task_id}_w.py").write_text(
        "def swiglu(x, y): ...\n", encoding="utf-8",
    )
    # No infer/ dir; should still build.
    msgs = shapes_incomplete.build_prompt(task_id, repo_root=tmp_path)
    assert "no caller hits" in msgs[1]["content"] or "no infer" in msgs[1]["content"]


def test_build_prompt_task_dir_missing_raises(tmp_path):
    from automation.prompts import shapes_incomplete

    with pytest.raises(FileNotFoundError):
        shapes_incomplete.build_prompt("L1_99_unknown", repo_root=tmp_path)


def test_build_prompt_under_32k_chars(tmp_path):
    """Rough token cap: the prompt should be << 32k chars (~8k tokens)."""
    from automation.prompts import shapes_incomplete

    _scaffold(tmp_path)
    messages = shapes_incomplete.build_prompt("L1_06_swiglu", repo_root=tmp_path)
    total = sum(len(m["content"]) for m in messages)
    assert total < 32_000


def test_build_prompt_missing_param_evidence_passed_through(tmp_path):
    from automation.prompts import shapes_incomplete

    _scaffold(tmp_path)
    messages = shapes_incomplete.build_prompt(
        "L1_06_swiglu", repo_root=tmp_path,
        missing_params_evidence="param 'gate' has no value at L342",
    )
    assert "param 'gate' has no value at L342" in messages[1]["content"]


def test_shared_protocol_lists_all_required_meta_keys():
    """OUTPUT_PROTOCOL must enumerate every key validated by ``validate_patch``.

    ``fix_orchestrator.py`` calls ``file_blocks.validate_patch`` with
    ``expected_keys={"decision", "missing_params", "rationale", "evidence"}``;
    if the system prompt drops any of these, the LLM will ship an
    underspecified META, the validator will reject the attempt, and the
    retry loop will burn token budget. This test pins the contract so a
    future edit to either side keeps them in sync.
    """
    from automation.prompts import shared

    for key in ("decision", "missing_params", "rationale", "evidence"):
        assert key in shared.OUTPUT_PROTOCOL, (
            f"OUTPUT_PROTOCOL is missing required META key {key!r}; it must "
            "mention every key that fix_orchestrator.validate_patch expects."
        )
