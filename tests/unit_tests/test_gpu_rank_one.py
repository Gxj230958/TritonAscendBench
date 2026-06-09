"""Unit tests for the GPU ranking outcome logic (gpu_rank_one).

These cover the pure decision logic the daemon ranking path relies on, which
the full GPU path (torch.cuda + triton do_bench) never exercises on a dev box.
The logic itself is GPU-free; importing ``serialize_verdict`` pulls the runner
module (which imports torch), so this file needs torch installed (the dev box
ships torch CPU) even though ``_classify_overall`` is torch-free.

  - ``_classify_overall``: the task-level RANKED / FLOOR / NOT_RANKED decision,
    including the FLOOR-gated-on-correctness invariant (a fully failed
    FLOOR_DOMINATED task must be NOT_RANKED, never masked as FLOOR) and the
    per-shape MIXED ranking rule.
  - ``serialize_verdict``: the default oracle-equiv writer drops the ranking-
    only ShapeVerdict fields when timing is off, so its JSON schema stays
    stable (no spurious ``t_npu_*: null`` / ``timing_method: null``). The drop
    gate keys on ``timing_method``, not ``t_npu_ns`` -- pinned below.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[2]
for _p in (_REPO / "scripts", _REPO / "scripts" / "loop", _REPO / "scripts" / "oracle"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from gpu_rank_one import _classify_overall  # noqa: E402
from run_oracle_equivalence_gpu import (  # noqa: E402
    ShapeVerdict,
    TaskVerdict,
    serialize_verdict,
)

# --- _classify_overall -----------------------------------------------------

@pytest.mark.parametrize("cm_status,correctness,n_ranked,expected", [
    # FLOOR_DOMINATED only counts as FLOOR when correctness passed.
    ("FLOOR_DOMINATED", "VERIFIED", 0, "FLOOR"),
    ("FLOOR_DOMINATED", "CEILING_ACCEPTED", 0, "FLOOR"),
    ("FLOOR_DOMINATED", "MIXED", 0, "FLOOR"),
    # F3 regression: a failed FLOOR_DOMINATED task must NOT be masked as FLOOR.
    ("FLOOR_DOMINATED", "FAILED", 0, "NOT_RANKED"),
    ("FLOOR_DOMINATED", "LAUNCH_BLOCKED", 0, "NOT_RANKED"),
    # Normal ranking: correctness pass + at least one ranked shape.
    ("PASS", "VERIFIED", 5, "RANKED"),
    ("BLOCKED", "VERIFIED", 5, "RANKED"),
    (None, "CEILING_ACCEPTED", 3, "RANKED"),
    # F2: a MIXED task ranks its passing shapes.
    ("PASS", "MIXED", 4, "RANKED"),
    # Correctness pass but nothing ranked -> NOT_RANKED.
    ("PASS", "VERIFIED", 0, "NOT_RANKED"),
    ("PASS", "MIXED", 0, "NOT_RANKED"),
    # Correctness fail -> NOT_RANKED regardless of n_ranked.
    ("PASS", "FAILED", 0, "NOT_RANKED"),
    ("PASS", "LAUNCH_BLOCKED", 5, "NOT_RANKED"),
])
def test_classify_overall(cm_status, correctness, n_ranked, expected):
    assert _classify_overall(cm_status, correctness, n_ranked) == expected


def test_floor_failed_is_not_ranked_regression():
    """FLOOR_DOMINATED + correctness FAILED must surface NOT_RANKED, not FLOOR
    -- otherwise a real correctness failure is silently masked."""
    assert _classify_overall("FLOOR_DOMINATED", "FAILED", 0) == "NOT_RANKED"
    assert _classify_overall("FLOOR_DOMINATED", "VERIFIED", 0) == "FLOOR"


# --- serialize_verdict -----------------------------------------------------

def _shape(sid, **kw):
    return ShapeVerdict(shape_id=sid, **kw)


def test_serialize_drops_ranking_fields_when_timing_off():
    """Default oracle-equiv output omits the ranking-only shape fields."""
    v = TaskVerdict(task_id="t", overall="VERIFIED", shapes=[_shape("P1")])
    sh = serialize_verdict(v)["shapes"][0]
    for k in ("t_npu_ns", "t_npu_min_ns", "t_npu_max_ns", "timing_method"):
        assert k not in sh
    assert sh["shape_id"] == "P1"  # correctness fields stay


def test_serialize_keeps_ranking_fields_when_timed():
    """A ranking verdict (timing_method set) keeps the ranking fields."""
    sv = _shape("P1")
    sv.timing_method = "do_bench"
    sv.t_npu_ns = 5000.0
    v = TaskVerdict(task_id="t", overall="VERIFIED", shapes=[sv])
    sh = serialize_verdict(v)["shapes"][0]
    assert sh["t_npu_ns"] == 5000.0
    assert sh["timing_method"] == "do_bench"


def test_serialize_gate_keys_on_timing_method_not_t_npu():
    """The drop gate must key on timing_method, NOT t_npu_ns. The event_fallback
    path sets t_npu_ns before timing_method (runner ~835/838), so a shape can
    carry a t_npu_ns while timing_method is still None; that orphan must still
    drop all four ranking fields, else the default oracle-equiv output would
    leak ``timing_method: null``. A t_npu_ns-keyed gate would pass every other
    serialize test but regress here."""
    sv = _shape("P1")
    sv.t_npu_ns = 5000.0          # set, but timing_method deliberately None
    assert sv.timing_method is None
    v = TaskVerdict(task_id="t", overall="VERIFIED", shapes=[sv])
    sh = serialize_verdict(v)["shapes"][0]
    for k in ("t_npu_ns", "t_npu_min_ns", "t_npu_max_ns", "timing_method"):
        assert k not in sh


def test_serialize_mixed_shapes_independent():
    """Per-shape: a timed shape keeps fields, an untimed one drops them."""
    timed = _shape("P1")
    timed.timing_method = "do_bench"
    timed.t_npu_ns = 1.0
    untimed = _shape("P2")
    v = TaskVerdict(task_id="t", overall="MIXED", shapes=[timed, untimed])
    shapes = serialize_verdict(v)["shapes"]
    assert "t_npu_ns" in shapes[0]
    assert "t_npu_ns" not in shapes[1]


def test_serialize_empty_shapes_ok():
    v = TaskVerdict(task_id="t", overall="SKIPPED", shapes=[])
    assert serialize_verdict(v)["shapes"] == []
