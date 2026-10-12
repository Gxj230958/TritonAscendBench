import pytest

from scripts.internal.streaming_evidence import merge_attempts


def test_supplement_pass_keeps_primary_failure_and_origin():
    row = {"evidence_root": "suite", "required_shapes": 2, "complete": False,
           "attempts": [{"complete": False, "category": "candidate_numerical_failure"}]}
    extra = {"required_shapes": 2,
             "attempts": [{"complete": True, "category": "complete_verified"}]}
    merged = merge_attempts(row, extra)
    assert merged["complete"] is True
    assert [a["evidence_root"] for a in merged["attempts"]] == ["suite", "supplement"]
    assert merged["attempts"][0]["category"] == "candidate_numerical_failure"
    assert len(row["attempts"]) == 1


def test_supplement_cannot_shrink_required_coverage():
    row = {"evidence_root": "replay", "required_shapes": 2,
           "attempts": [], "complete": False}
    with pytest.raises(RuntimeError, match="shape coverage"):
        merge_attempts(row, {"required_shapes": 1, "attempts": []})
