"""Tests for ``automation.prompts.oracle_mismatch`` (Unit 2.6).

Validates the structural contract of the chat-messages list returned by
``build_prompt``: the system message exposes the 3-path decision tree
and the shared OUTPUT_PROTOCOL meta keys, and the user message carries
the wrapper signature, the current oracle source, and the mismatch /
log excerpts.
"""
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


def _scaffold(
    tmp_path: Path,
    task_id: str = "L1_06_rms_norm",
    level: str = "level1",
    oracle_body: str | None = None,
) -> Path:
    td = tmp_path / "tasks" / level / task_id
    td.mkdir(parents=True)
    (td / "meta.yaml").write_text(
        yaml.safe_dump(
            {"schema_version": "1.0", "name": "rms_norm", "kind": "triton"}
        ),
        encoding="utf-8",
    )
    (td / "shapes.yaml").write_text("public_shapes: []\n", encoding="utf-8")
    body = oracle_body or (
        "import torch\n"
        "def oracle(x: torch.Tensor) -> torch.Tensor:\n"
        "    return x * torch.sigmoid(x)\n"
    )
    (td / "oracle_pytorch.py").write_text(body, encoding="utf-8")

    vd = tmp_path / "tasks" / "_vendored"
    vd.mkdir(parents=True)
    (vd / f"{task_id}_wrapper.py").write_text(
        "import torch\n"
        "def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:\n"
        "    ...\n",
        encoding="utf-8",
    )
    return td


def test_build_prompt_returns_two_messages(tmp_path):
    from automation.prompts import oracle_mismatch

    _scaffold(tmp_path)
    messages = oracle_mismatch.build_prompt(
        "L1_06_rms_norm", tmp_path,
    )
    assert isinstance(messages, list)
    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"


def test_user_prompt_contains_wrapper_signature(tmp_path):
    """The vendored wrapper signature must be spliced into the user
    message; without it the LLM has no contract to mirror in a
    rewrite-primitives oracle."""
    from automation.prompts import oracle_mismatch

    _scaffold(tmp_path)
    messages = oracle_mismatch.build_prompt(
        "L1_06_rms_norm", tmp_path,
    )
    user_content = messages[1]["content"]
    assert "def rms_norm(" in user_content
    assert "weight: torch.Tensor" in user_content


def test_user_prompt_contains_oracle_source(tmp_path):
    """The full current oracle_pytorch.py body must appear so the LLM
    can decide whether the formula is wrong or only the tolerance is."""
    from automation.prompts import oracle_mismatch

    _scaffold(
        tmp_path,
        oracle_body=(
            "import torch\n"
            "def oracle(x: torch.Tensor) -> torch.Tensor:\n"
            "    return x * torch.sigmoid(x)  # marker_for_test\n"
        ),
    )
    messages = oracle_mismatch.build_prompt(
        "L1_06_rms_norm", tmp_path,
    )
    user_content = messages[1]["content"]
    assert "marker_for_test" in user_content
    assert "def oracle(" in user_content


def test_decision_tree_in_user_prompt(tmp_path):
    """All three canonical decisions must be enumerated -- the LLM picks
    exactly one and the validator (file_blocks.py:_ALL_DECISIONS) only
    knows these three strings for the oracle-mismatch verdict.
    """
    from automation.prompts import oracle_mismatch

    _scaffold(tmp_path)
    messages = oracle_mismatch.build_prompt(
        "L1_06_rms_norm", tmp_path,
    )
    # Decision tree lives in the system message; assert all 3 canonical
    # decision strings appear so the LLM sees the full menu.
    sys_content = messages[0]["content"]
    for decision in (
        "tolerance-bump",
        "rewrite-primitives",
        "defer-to-upstream-oracle",
    ):
        assert decision in sys_content, (
            f"decision {decision!r} must be described in the system prompt"
        )


def test_mismatch_evidence_truncated_to_max_lines(tmp_path):
    """Mismatch evidence lists longer than MAX_MISMATCH_EVIDENCE_LINES
    must be truncated with an explicit ``(truncated; ...)`` marker so
    the LLM does not silently lose context."""
    from automation.prompts import oracle_mismatch

    _scaffold(tmp_path)
    n_lines = oracle_mismatch.MAX_MISMATCH_EVIDENCE_LINES
    evidence = [f"mismatch[{i}]: |a-b|={i*1e-3:.4e}" for i in range(n_lines + 5)]
    messages = oracle_mismatch.build_prompt(
        "L1_06_rms_norm", tmp_path,
        mismatch_evidence=evidence,
    )
    user_content = messages[1]["content"]
    # First MAX_MISMATCH_EVIDENCE_LINES lines kept
    assert evidence[0] in user_content
    assert evidence[n_lines - 1] in user_content
    # Beyond the cap should be gone
    assert evidence[n_lines + 4] not in user_content
    # And the truncation marker explicit
    assert "truncated" in user_content
    assert "5 more line" in user_content


def test_missing_oracle_file_raises(tmp_path):
    """If oracle_pytorch.py is missing under the task dir we fail loud --
    asking the LLM to fix an oracle we cannot show is meaningless and
    would burn token budget on a hallucinated baseline."""
    from automation.prompts import oracle_mismatch

    task_id = "L1_06_rms_norm"
    td = tmp_path / "tasks" / "level1" / task_id
    td.mkdir(parents=True)
    (td / "meta.yaml").write_text("schema_version: '1.0'\nname: rms_norm\n", encoding="utf-8")
    # Intentionally do not create oracle_pytorch.py
    with pytest.raises(FileNotFoundError):
        oracle_mismatch.build_prompt(task_id, tmp_path)


def test_oracle_mismatch_prompt_lists_per_decision_keys(tmp_path):
    """The oracle_mismatch system prompt must enumerate the per-decision
    META keys for all THREE oracle-mismatch decisions: ``tolerance-bump``,
    ``rewrite-primitives``, and ``defer-to-upstream-oracle``. Each
    decision has its own decision-specific field
    (``tolerance_delta`` / ``primitives_changed`` / ``upstream_callable``)
    -- the table guards against the LLM omitting them."""
    from automation.prompts import oracle_mismatch

    _scaffold(tmp_path)
    messages = oracle_mismatch.build_prompt(
        "L1_06_rms_norm", tmp_path,
    )
    sys_content = messages[0]["content"]
    # Block grammar invariants (META + FILE syntax must appear).
    assert "### META" in sys_content
    assert "### FILE" in sys_content
    # Common keys.
    for key in ("decision", "rationale", "evidence"):
        assert key in sys_content
    # The three per-decision specific fields and decision labels.
    assert "tolerance-bump" in sys_content
    assert "tolerance_delta" in sys_content
    assert "rewrite-primitives" in sys_content
    assert "primitives_changed" in sys_content
    assert "defer-to-upstream-oracle" in sys_content
    assert "upstream_callable" in sys_content


def test_task_dir_missing_raises(tmp_path):
    """Unknown task_id -> FileNotFoundError (no task dir under tasks/)."""
    from automation.prompts import oracle_mismatch

    with pytest.raises(FileNotFoundError):
        oracle_mismatch.build_prompt("L1_99_unknown", tmp_path)
