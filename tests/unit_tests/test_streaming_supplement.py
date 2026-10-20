from scripts.internal.streaming_supplement import select_candidates


def candidate(name, correct=None, shapes=0, blocked=False):
    return {"path": name, "correctness": correct, "runtime": "ok",
            "correct_shapes": shapes, "static_blocked": blocked}


def test_selects_every_compatible_historical_pass():
    rows = [candidate("a", "ok", 12), candidate("b", "ok", 15),
            candidate("rejected", "ok", 20, True), candidate("partial", shapes=100)]
    assert [r["path"] for r in select_candidates(rows)] == ["b", "a"]


def test_selects_strongest_remaining_source_without_historical_pass():
    rows = [candidate("small", shapes=2), candidate("large", shapes=9)]
    assert [r["path"] for r in select_candidates(rows)] == ["large"]


def test_no_static_compatible_candidate_is_explicit():
    assert select_candidates([candidate("rejected", "ok", 20, True)]) == []
