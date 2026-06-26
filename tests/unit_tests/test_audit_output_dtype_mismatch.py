"""G6(iii) — report-only output-dtype-mismatch audit over GPU C2 verdicts.

The audit must flag ONLY shapes that are status=VERIFIED yet carry a non-empty
``bad_dtype_keys`` — the "passes on values but output dtype != reference" set
that would flip under the migration output-dtype gate. It must never flag an
already-non-VERIFIED shape (FAILED / *_BLOCKED / CEILING_ACCEPTED), and it must
never mutate a verdict or drop a task.
"""
from __future__ import annotations

from scripts.oracle.audit_output_dtype_mismatch import audit_verdicts


def _verdict(task_id, shapes):
    return {"task_id": task_id, "overall": "MIXED", "shapes": shapes}


def _shape(shape_id, status, bad_dtype_keys=None):
    return {"shape_id": shape_id, "status": status,
            "bad_dtype_keys": bad_dtype_keys or []}


def test_verified_with_bad_dtype_is_flagged():
    v = _verdict("L1_01", [_shape("P1", "VERIFIED", ["out"])])
    flagged = audit_verdicts([v])
    assert len(flagged) == 1
    assert flagged[0].task_id == "L1_01"
    assert flagged[0].shape_id == "P1"
    assert flagged[0].bad_dtype_keys == ["out"]


def test_verified_clean_is_not_flagged():
    v = _verdict("L1_02", [_shape("P1", "VERIFIED", [])])
    assert audit_verdicts([v]) == []


def test_failed_shape_with_bad_dtype_not_flagged():
    """A FAILED shape cannot 'flip' — it is already non-VERIFIED."""
    v = _verdict("L1_03", [_shape("P1", "FAILED", ["out"])])
    assert audit_verdicts([v]) == []


def test_ceiling_accepted_with_bad_dtype_not_flagged():
    """CEILING_ACCEPTED already fail-closes on bad dtype elsewhere; it is not a
    hidden VERIFIED pass, so the audit leaves it alone."""
    v = _verdict("L2_03", [_shape("P1", "CEILING_ACCEPTED", ["out"])])
    assert audit_verdicts([v]) == []


def test_mixed_shapes_only_verified_bad_flagged():
    v = _verdict("L3_01", [
        _shape("P1", "VERIFIED", ["a"]),       # flagged
        _shape("P2", "VERIFIED", []),          # clean
        _shape("P3", "FAILED", ["b"]),         # non-VERIFIED
        _shape("H1", "VERIFIED", ["c", "d"]),  # flagged
    ])
    flagged = audit_verdicts([v])
    assert {f.shape_id for f in flagged} == {"P1", "H1"}
    assert sorted(f.bad_dtype_keys for f in flagged) == [["a"], ["c", "d"]]


def test_multiple_tasks_counted():
    vs = [
        _verdict("T1", [_shape("P1", "VERIFIED", ["x"])]),
        _verdict("T2", [_shape("P1", "VERIFIED", [])]),
        _verdict("T3", [_shape("P1", "VERIFIED", ["y"])]),
    ]
    flagged = audit_verdicts(vs)
    assert {f.task_id for f in flagged} == {"T1", "T3"}


def test_missing_fields_do_not_crash():
    """A verdict missing shapes / a shape missing bad_dtype_keys is tolerated
    (empty -> not flagged), never an exception."""
    assert audit_verdicts([{"task_id": "T"}]) == []
    assert audit_verdicts([{"task_id": "T", "shapes": [{"status": "VERIFIED"}]}]) == []
