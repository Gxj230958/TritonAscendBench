"""Schema-doc text assertions for M2 Unit 2.1.

The four bench-team-internal field groups are defined in
``docs/reference/task_api_contract.md`` §6a. This test guards against accidental
deletion / rename of those documented schema sections — a code reviewer
adding the corresponding validator / eligibility logic must be able to
point at the spec, and `git grep` for these field names must hit the spec
in addition to the code.

The test is intentionally textual (not a YAML / JSON-schema parse) — the
spec is human-readable Markdown, and the goal here is presence + structure
of the documented contract, not machine-parseable schema enforcement.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

# Repository root is two levels above the tests/unit_tests/ directory.
_REPO_ROOT = Path(__file__).resolve().parents[2]
_SPEC = _REPO_ROOT / "docs" / "internal" / "specs" / "task_api_contract_state_fields.md"


@pytest.fixture(scope="module")
def spec_text() -> str:
    assert _SPEC.is_file(), f"missing spec file: {_SPEC}"
    return _SPEC.read_text(encoding="utf-8")


def test_section_6a_present(spec_text: str) -> None:
    """The new §6a section header must exist."""
    assert "## 6a. Task State Fields" in spec_text, (
        "the internal task-state-fields doc must declare §6a "
        "'Task State Fields (bench-team-internal writes)'"
    )


@pytest.mark.parametrize(
    "field_name",
    [
        "phase_a_prime.skip_invocation",
        "phase_b_capture",
        "phase_b_capture_required",
        "release_scope",
    ],
)
def test_field_name_documented(spec_text: str, field_name: str) -> None:
    """Each of the four field groups must be name-mentioned in the spec."""
    assert field_name in spec_text, (
        f"the internal task-state-fields doc must mention '{field_name}'"
    )


@pytest.mark.parametrize(
    "subsection_header",
    [
        "### 6a.1 `phase_a_prime.skip_invocation`",
        "### 6a.2 `phase_a_prime.status` — `SKIPPED_BY_DESIGN` enum extension",
        "### 6a.3 `phase_b_capture` block",
        "### 6a.4 `phase_b_capture_required`",
        "### 6a.5 `release_scope` block",
        "### 6a.6 Cross-field invariants",
    ],
)
def test_subsection_header_present(spec_text: str, subsection_header: str) -> None:
    """Every documented field group must have its own subsection header."""
    assert subsection_header in spec_text, (
        f"the internal task-state-fields doc must contain subsection header: "
        f"{subsection_header!r}"
    )


def _slice_section(spec_text: str, header: str) -> str:
    """Return the body between ``header`` and the next ``###`` / ``##`` line."""
    start = spec_text.index(header)
    rest = spec_text[start + len(header):]
    # next header at any level
    m = re.search(r"^#{2,3} ", rest, flags=re.MULTILINE)
    end = m.start() if m else len(rest)
    return rest[:end]


@pytest.mark.parametrize(
    "header",
    [
        "### 6a.1 `phase_a_prime.skip_invocation`",
        "### 6a.4 `phase_b_capture_required`",
    ],
)
def test_boolean_field_documents_type_and_default(spec_text: str, header: str) -> None:
    """Boolean field subsections must document type / default / who-writes."""
    body = _slice_section(spec_text, header)
    assert "`bool`" in body, f"{header}: missing 'bool' type marker"
    assert "default" in body.lower(), f"{header}: missing 'default' marker"
    assert "false" in body.lower(), (
        f"{header}: boolean default must be documented (false)"
    )
    assert "written by" in body.lower(), (
        f"{header}: must document the writer (orchestrator / phase runner)"
    )


def test_phase_b_capture_block_documents_subfields(spec_text: str) -> None:
    """The phase_b_capture block must document all five subfields."""
    body = _slice_section(spec_text, "### 6a.3 `phase_b_capture` block")
    for subfield in (
        "phase_b_capture.status",
        "phase_b_capture.upstream_callable.module",
        "phase_b_capture.upstream_callable.function",
        "phase_b_capture.reason",
        "phase_b_capture.captured_at",
    ):
        assert subfield in body, (
            f"phase_b_capture subsection must document subfield '{subfield}'"
        )
    # status enum must be enumerated
    for enum_value in ("PENDING", "DONE", "FAILED"):
        assert enum_value in body, (
            f"phase_b_capture.status enum must list '{enum_value}'"
        )


def test_release_scope_block_documents_subfields(spec_text: str) -> None:
    """The release_scope block must document all four subfields."""
    body = _slice_section(spec_text, "### 6a.5 `release_scope` block")
    for subfield in (
        "release_scope.excluded",
        "release_scope.reason",
        "release_scope.excluded_at",
        "release_scope.excluded_by",
    ):
        assert subfield in body, (
            f"release_scope subsection must document subfield '{subfield}'"
        )
    # The 'required when excluded == true' invariant must be stated.
    assert "required when" in body.lower(), (
        "release_scope subsection must state the 'required when excluded "
        "is true' invariant for reason / excluded_at / excluded_by"
    )


def test_phase_a_prime_status_enum_extension(spec_text: str) -> None:
    """phase_a_prime.status must list SKIPPED_BY_DESIGN as the new value."""
    body = _slice_section(
        spec_text,
        "### 6a.2 `phase_a_prime.status` — `SKIPPED_BY_DESIGN` enum extension",
    )
    for existing_value in ("PASS", "BLOCKED", "UNRUN"):
        assert existing_value in body, (
            f"phase_a_prime.status subsection must list existing value "
            f"'{existing_value}'"
        )
    assert "SKIPPED_BY_DESIGN" in body, (
        "phase_a_prime.status subsection must list the new value "
        "'SKIPPED_BY_DESIGN'"
    )


def test_cross_field_invariants_documented(spec_text: str) -> None:
    """The §6a.6 invariants must tie the four field groups together."""
    body = _slice_section(spec_text, "### 6a.6 Cross-field invariants")
    # skip_invocation <=> SKIPPED_BY_DESIGN bidirectional invariant
    assert "skip_invocation" in body
    assert "SKIPPED_BY_DESIGN" in body
    # phase_b_capture DONE => upstream_callable + captured_at
    assert "captured_at" in body
    assert "upstream_callable" in body
    # release_scope excluded => reason + excluded_at + excluded_by
    assert "excluded_at" in body
    assert "excluded_by" in body
    # phase_b_capture_required + status != DONE => BLOCKED
    assert "phase_b_capture_required" in body
    assert "BLOCKED" in body


def test_iso_8601_format_documented(spec_text: str) -> None:
    """The ISO 8601 timestamp format must be explicit in §6a."""
    section_start = spec_text.index("## 6a. Task State Fields")
    try:
        section_end = spec_text.index("## 7. Validator")
    except ValueError:
        section_end = len(spec_text)  # §6a moved to the internal state-fields doc (runs to EOF)
    section_6a = spec_text[section_start:section_end]
    # The Z (UTC) form documented as the writer's format.
    assert "YYYY-MM-DDTHH:MM:SSZ" in section_6a, (
        "§6a must document the ISO 8601 UTC timestamp format "
        "(YYYY-MM-DDTHH:MM:SSZ) used by iso_utc_now()"
    )


def test_validator_section_references_6a() -> None:
    """§7 of the PUBLIC contract must explain that §6a fields are NOT in REQUIRED_META_TOP_LEVEL."""
    public = (_REPO_ROOT / "docs" / "reference" / "task_api_contract.md").read_text(encoding="utf-8")
    body = _slice_section(public, "## 7. Validator")
    assert "REQUIRED_META_TOP_LEVEL" in body, (
        "§7 must reference REQUIRED_META_TOP_LEVEL"
    )
    # The §6a fields are explicitly optional / not in REQUIRED_META_TOP_LEVEL.
    assert "§6a" in body or "phase_b_capture" in body, (
        "§7 must explain that §6a's new fields are NOT in "
        "REQUIRED_META_TOP_LEVEL"
    )


def test_no_chinese_in_spec(spec_text: str) -> None:
    """Spec doc is English-only (CLAUDE.md project rule)."""
    chinese_chars = re.findall(r"[一-龥]", spec_text)
    assert not chinese_chars, (
        f"the internal task-state-fields doc must be English-only "
        f"(found {len(chinese_chars)} CJK chars: {chinese_chars[:10]!r}...)"
    )


def test_no_process_noise_in_new_section(spec_text: str) -> None:
    """§6a must not contain process noise (codex / round-N / V-numbered)."""
    section_start = spec_text.index("## 6a. Task State Fields")
    try:
        section_end = spec_text.index("## 7. Validator")
    except ValueError:
        section_end = len(spec_text)  # §6a moved to the internal state-fields doc (runs to EOF)
    section_6a = spec_text[section_start:section_end]
    forbidden = re.findall(
        r"\b(codex|claude|gemini|round[-_ ]?[0-9]+|"
        r"[RHDFP][0-9]+\b|sanity findings)",
        section_6a,
        flags=re.IGNORECASE,
    )
    assert not forbidden, (
        f"§6a must not contain process-noise tokens; found: {forbidden[:10]!r}"
    )
