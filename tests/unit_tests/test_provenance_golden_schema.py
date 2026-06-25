"""Unit tests for V9 plan M1.2 -- nested ``provenance.golden:`` schema.

Covers:

* ``parse_provenance_golden`` returns ``None`` when the nested block is
  absent (grace period).
* Closed-enum validation rejects out-of-range values.
* Hash-length validation rejects wrong-length / non-hex strings.
* Type validation rejects bad bool / numeric / list-of-string fields.
* Operational extras (``verified_at`` / ``equivalence_per_shape`` /
  ``golden_source`` / ``tolerance_class_used``) round-trip through
  ``ProvenanceGoldenSpec.extra``.
* Backward compatibility: meta with only the top-level legacy
  ``provenance`` keys (``added_in`` / ``authors`` / ``last_modified``)
  produces ``provenance_golden = None``.
"""
from __future__ import annotations

import pytest

from ascendbench.corpus.provenance import (
    VALID_TOLERANCE_SOURCE,
    ProvenanceGoldenSpec,
    parse_provenance_golden,
    validate_provenance_golden,
)

# ─────────────────────────────────────────────────────────────────────
# parse_provenance_golden -- grace period + typed view
# ─────────────────────────────────────────────────────────────────────


def test_parse_returns_none_when_no_provenance():
    """Meta without any provenance key returns None (grace period)."""
    assert parse_provenance_golden({}) is None


def test_parse_returns_none_when_provenance_is_legacy_top_level_only():
    """Legacy provenance block (no nested golden:) returns None."""
    meta = {
        "provenance": {
            "added_in": "v0.1",
            "authors": ["alice", "bob"],
            "last_modified": "2026-05-26",
        }
    }
    assert parse_provenance_golden(meta) is None


def test_parse_returns_spec_with_nested_golden():
    """Nested block parses into populated ProvenanceGoldenSpec."""
    meta = {
        "provenance": {
            "added_in": "v0.1",
            "golden": {
                "source": "upstream_author_test",
                "tolerance_source": "upstream_test_explicit",
                "atol": 1e-5,
                "rtol": 1e-5,
                "oracle_kernel_equivalence": "VERIFIED",
                "gpu_kernel_capture_status": "DONE",
                "gpu_kernel_deterministic": True,
                "reviewers": ["alice", "bob"],
                "reconstructability_risk": "low",
                "test_capture_sha": "a" * 40,
                "golden_sha256": "f" * 64,
            },
        }
    }
    spec = parse_provenance_golden(meta)
    assert isinstance(spec, ProvenanceGoldenSpec)
    assert spec.source == "upstream_author_test"
    assert spec.tolerance_source == "upstream_test_explicit"
    assert spec.atol == pytest.approx(1e-5)
    assert spec.rtol == pytest.approx(1e-5)
    assert spec.oracle_kernel_equivalence == "VERIFIED"
    assert spec.gpu_kernel_capture_status == "DONE"
    assert spec.gpu_kernel_deterministic is True
    assert spec.reviewers == ("alice", "bob")
    assert spec.reconstructability_risk == "low"
    assert spec.test_capture_sha == "a" * 40
    assert spec.golden_sha256 == "f" * 64
    assert spec.extra == {}


def test_parse_preserves_operational_extras():
    """Operational fields round-trip through ``extra`` dict."""
    meta = {
        "provenance": {
            "golden": {
                "source": "gpu_kernel_rerun",
                "verified_at": "2026-05-27T08:30:00Z",
                "tolerance_class_used": "elementwise",
                "equivalence_per_shape": {"P1": {"status": "VERIFIED"}},
                "golden_source": "oracle_pytorch_verified",
            }
        }
    }
    spec = parse_provenance_golden(meta)
    assert spec is not None
    assert spec.source == "gpu_kernel_rerun"
    assert "verified_at" in spec.extra
    assert spec.extra["verified_at"] == "2026-05-27T08:30:00Z"
    assert spec.extra["tolerance_class_used"] == "elementwise"
    assert spec.extra["equivalence_per_shape"] == {"P1": {"status": "VERIFIED"}}
    assert spec.extra["golden_source"] == "oracle_pytorch_verified"


def test_parse_malformed_reviewers_kept_visible():
    """Non-list reviewers leaves spec.reviewers empty + flagged in extra."""
    meta = {"provenance": {"golden": {"reviewers": "not_a_list"}}}
    spec = parse_provenance_golden(meta)
    assert spec is not None
    assert spec.reviewers == ()
    assert "__malformed_reviewers" in spec.extra


def test_parse_returns_none_when_golden_is_not_a_dict():
    """A scalar/list ``golden:`` value yields None (validator catches the shape)."""
    assert parse_provenance_golden(
        {"provenance": {"golden": "not_a_dict"}}
    ) is None
    assert parse_provenance_golden(
        {"provenance": {"golden": ["a", "b"]}}
    ) is None


# ─────────────────────────────────────────────────────────────────────
# validate_provenance_golden -- closed enums + shape
# ─────────────────────────────────────────────────────────────────────


def test_validate_empty_meta_clean():
    """Absent provenance block: no errors (grace period)."""
    assert validate_provenance_golden({}) == []


def test_validate_legacy_top_level_clean():
    """Legacy provenance with only top-level keys: no errors."""
    assert validate_provenance_golden(
        {"provenance": {"added_in": "v0.1", "authors": ["alice"]}}
    ) == []


def test_validate_rejects_bad_source_enum():
    meta = {"provenance": {"golden": {"source": "wrong_value"}}}
    errs = validate_provenance_golden(meta)
    assert any("source" in e for e in errs)


def test_validate_rejects_bad_tolerance_source_enum():
    meta = {
        "provenance": {
            "golden": {"tolerance_source": "made_up_value"}
        }
    }
    errs = validate_provenance_golden(meta)
    assert any("tolerance_source" in e for e in errs)


def test_validate_rejects_bad_oracle_kernel_equivalence_enum():
    meta = {
        "provenance": {
            "golden": {"oracle_kernel_equivalence": "MAYBE_PROBABLY"}
        }
    }
    errs = validate_provenance_golden(meta)
    assert any("oracle_kernel_equivalence" in e for e in errs)


def test_validate_rejects_bad_gpu_capture_status():
    meta = {
        "provenance": {
            "golden": {"gpu_kernel_capture_status": "WORKING_ON_IT"}
        }
    }
    errs = validate_provenance_golden(meta)
    assert any("gpu_kernel_capture_status" in e for e in errs)


def test_validate_rejects_bad_reconstructability_risk():
    meta = {
        "provenance": {
            "golden": {"reconstructability_risk": "extreme"}
        }
    }
    errs = validate_provenance_golden(meta)
    assert any("reconstructability_risk" in e for e in errs)


def test_validate_rejects_non_bool_gpu_deterministic():
    meta = {
        "provenance": {
            "golden": {"gpu_kernel_deterministic": "yes"}
        }
    }
    errs = validate_provenance_golden(meta)
    assert any("gpu_kernel_deterministic" in e for e in errs)


def test_validate_rejects_short_golden_sha256():
    meta = {
        "provenance": {
            "golden": {"golden_sha256": "abcd1234"}
        }
    }
    errs = validate_provenance_golden(meta)
    assert any("golden_sha256" in e for e in errs)


def test_validate_rejects_non_hex_golden_sha256():
    meta = {
        "provenance": {
            "golden": {"golden_sha256": "z" * 64}
        }
    }
    errs = validate_provenance_golden(meta)
    assert any("golden_sha256" in e for e in errs)


def test_validate_rejects_short_test_capture_sha():
    meta = {
        "provenance": {
            "golden": {"test_capture_sha": "a" * 39}
        }
    }
    errs = validate_provenance_golden(meta)
    assert any("test_capture_sha" in e for e in errs)


def test_validate_rejects_bool_for_atol():
    meta = {"provenance": {"golden": {"atol": True}}}
    errs = validate_provenance_golden(meta)
    assert any("atol" in e for e in errs)


def test_validate_rejects_string_for_atol():
    meta = {"provenance": {"golden": {"atol": "tight"}}}
    errs = validate_provenance_golden(meta)
    assert any("atol" in e for e in errs)


def test_validate_rejects_non_list_reviewers():
    meta = {"provenance": {"golden": {"reviewers": "alice"}}}
    errs = validate_provenance_golden(meta)
    assert any("reviewers" in e for e in errs)


def test_validate_rejects_non_string_reviewer_entry():
    meta = {"provenance": {"golden": {"reviewers": ["alice", 42]}}}
    errs = validate_provenance_golden(meta)
    assert any("reviewers[1]" in e for e in errs)


def test_validate_rejects_non_dict_golden_block():
    meta = {"provenance": {"golden": "fine"}}
    errs = validate_provenance_golden(meta)
    assert any("golden must be a mapping" in e for e in errs)


def test_validate_accepts_full_valid_block():
    """A complete and well-formed block produces zero errors."""
    meta = {
        "provenance": {
            "added_in": "v0.1",
            "authors": ["alice"],
            "last_modified": "2026-05-27",
            "golden": {
                "source": "upstream_author_test",
                "tolerance_source": "upstream_test_explicit",
                "atol": 1e-5,
                "rtol": 1e-5,
                "ref_path": "tasks/L1_12/upstream_ref.py",
                "test_file_path": "vllm/tests/test_rms_norm.py",
                "test_capture_sha": "1" * 40,
                "golden_sha256": "2" * 64,
                "oracle_kernel_equivalence": "VERIFIED",
                "gpu_kernel_capture_status": "DONE",
                "gpu_kernel_deterministic": True,
                "upstream_closure_fingerprint": "3" * 64,
                "reviewers": ["alice", "bob"],
                "reconstructability_risk": "low",
                # operational extras (open):
                "verified_at": "2026-05-27T08:30:00Z",
                "tolerance_class_used": "elementwise",
                "golden_source": "oracle_pytorch_verified",
            },
        }
    }
    assert validate_provenance_golden(meta) == []


def test_validate_partial_block_with_oracle_status_only():
    """A partially-populated block (during pipeline progress) is valid
    as long as every present value conforms. R10 stage-3 typically
    writes only ``oracle_kernel_equivalence`` first; the rest fills
    in later as stages complete."""
    meta = {
        "provenance": {
            "golden": {"oracle_kernel_equivalence": "VERIFIED"}
        }
    }
    assert validate_provenance_golden(meta) == []


def test_validate_rejects_status_enum_swap():
    """oracle_kernel_equivalence and gpu_kernel_capture_status have
    distinct enums -- using one's value for the other is rejected."""
    # "DONE" is a gpu_kernel_capture_status value, NOT oracle_kernel_equivalence
    meta = {
        "provenance": {"golden": {"oracle_kernel_equivalence": "DONE"}}
    }
    errs = validate_provenance_golden(meta)
    assert any("oracle_kernel_equivalence" in e for e in errs)
    # Conversely, "VERIFIED" is oracle_kernel_equivalence, NOT gpu_kernel_capture_status
    meta2 = {
        "provenance": {"golden": {"gpu_kernel_capture_status": "VERIFIED"}}
    }
    errs2 = validate_provenance_golden(meta2)
    assert any("gpu_kernel_capture_status" in e for e in errs2)


def test_validate_partial_block_with_only_one_valid_field():
    meta = {
        "provenance": {
            "golden": {
                "tolerance_source": next(iter(VALID_TOLERANCE_SOURCE)),
            }
        }
    }
    assert validate_provenance_golden(meta) == []


# ─────────────────────────────────────────────────────────────────────
# Loader integration -- dataset.TaskDef.provenance_golden field
# ─────────────────────────────────────────────────────────────────────


def test_loader_returns_typed_spec_when_nested_block_present():
    """After M2.1c apply has written a nested provenance.golden block,
    TaskDef.provenance_golden surfaces as a typed ProvenanceGoldenSpec
    (not None) so consumers can read fields directly without re-parsing
    the raw meta dict."""
    from ascendbench.corpus.dataset import load_task

    td = load_task("L1_12_rms_norm")
    prov = td.meta.get("provenance") or {}
    has_nested = isinstance(prov.get("golden"), dict)
    if not has_nested:
        # Pre-M2.1c state: the field stays None.
        assert td.provenance_golden is None
    else:
        # Post-M2.1c state: typed view populated.
        assert td.provenance_golden is not None
        # Top-level legacy keys still reachable.
        assert isinstance(prov, dict)
