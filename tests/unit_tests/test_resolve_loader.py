"""Tests for ``automation.lib.resolve_loader`` -- the unified loader for
resolver artefacts (aggregate JSON / JSONL / legacy plain text).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

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


def test_load_aggregate_json_returns_normalized_records(tmp_path):
    from automation.lib.resolve_loader import load_resolve_artifact

    p = tmp_path / "resolve.json"
    payload = {
        "_provenance": {"tool_version": "0.1.0"},
        "verdicts": [
            {
                "task_id": "T1",
                "status": "BLOCKED(no-verified-upstream-entrypoint: shapes-incomplete)",
                "reason": "missing param X",
            },
            {
                "task_id": "T2",
                "status": "PASS",
                "reason": "",
            },
        ],
    }
    p.write_text(json.dumps(payload), encoding="utf-8")
    records = load_resolve_artifact(p)
    assert len(records) == len(payload["verdicts"])
    assert records[0]["task_id"] == "T1"
    assert records[0]["verdict_class"] == "shapes-incomplete"
    assert records[0]["reason"] == "missing param X"
    assert records[1]["verdict_class"] == "PASS"


def test_load_jsonl_returns_normalized_records(tmp_path):
    from automation.lib.resolve_loader import load_resolve_artifact

    p = tmp_path / "resolve.jsonl"
    lines = [
        json.dumps({
            "task_id": "A",
            "status": "BLOCKED(no-verified-upstream-entrypoint: oracle-mismatch)",
        }),
        json.dumps({
            "task_id": "B",
            "status": "PASS",
            "reason": "ok",
        }),
    ]
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    records = load_resolve_artifact(p)
    assert len(records) == 2
    assert records[0]["task_id"] == "A"
    assert records[0]["verdict_class"] == "oracle-mismatch"
    assert records[1]["verdict_class"] == "PASS"


def test_load_legacy_text_returns_normalized_records(tmp_path):
    from automation.lib.resolve_loader import load_resolve_artifact

    p = tmp_path / "pa_resolve.log"
    p.write_text(
        "[T1] RESOLVED my_kernel :: PASS\n"
        "[T2] BLOCKED - :: BLOCKED(no-verified-upstream-entrypoint: shapes-incomplete)\n",
        encoding="utf-8",
    )
    records = load_resolve_artifact(p)
    assert len(records) == 2
    by_id = {r["task_id"]: r for r in records}
    assert by_id["T1"]["verdict_class"] == "PASS"
    assert by_id["T2"]["verdict_class"] == "shapes-incomplete"


def test_normalize_pass_status(tmp_path):
    from automation.lib.resolve_loader import load_resolve_artifact

    p = tmp_path / "r.json"
    p.write_text(json.dumps({
        "verdicts": [{"task_id": "X", "status": "PASS"}],
    }), encoding="utf-8")
    rec = load_resolve_artifact(p)[0]
    assert rec["verdict_class"] == "PASS"


def test_normalize_blocked_shapes_incomplete(tmp_path):
    from automation.lib.resolve_loader import load_resolve_artifact

    p = tmp_path / "r.json"
    p.write_text(json.dumps({
        "verdicts": [{
            "task_id": "X",
            "status": "BLOCKED(no-verified-upstream-entrypoint: shapes-incomplete)",
        }],
    }), encoding="utf-8")
    rec = load_resolve_artifact(p)[0]
    assert rec["verdict_class"] == "shapes-incomplete"


def test_normalize_smoke_fail(tmp_path):
    from automation.lib.resolve_loader import load_resolve_artifact

    p = tmp_path / "r.json"
    p.write_text(json.dumps({
        "verdicts": [{
            "task_id": "X",
            "status": "BLOCKED(no-verified-upstream-entrypoint: smoke-fail)",
        }],
    }), encoding="utf-8")
    rec = load_resolve_artifact(p)[0]
    assert rec["verdict_class"] == "smoke-fail"


def test_normalize_unknown_status_falls_back_to_unexpected_blocked(tmp_path):
    from automation.lib.resolve_loader import load_resolve_artifact

    p = tmp_path / "r.json"
    p.write_text(json.dumps({
        "verdicts": [{"task_id": "X", "status": "BLOCKED(weird-form)"}],
    }), encoding="utf-8")
    rec = load_resolve_artifact(p)[0]
    assert rec["verdict_class"] == "unexpected-blocked"


def test_load_real_resolve_all_json():
    """Cross-check against the actually-checked-in resolve_all.json.

    Expected distribution (computed at task time):
      PASS                          = 3
      shapes-incomplete             = 56
      upstream-not-invocable-here   = 38
      oracle-mismatch               = 2
      smoke-fail                    = 1
      total                         = 100
    """
    from automation.lib.resolve_loader import load_resolve_artifact

    p = _REPO_ROOT / "resolve_all.json"
    if not p.is_file():
        pytest.skip(f"resolve_all.json not present at {p}")
    records = load_resolve_artifact(p)
    assert len(records) == 100, f"expected 100 records, got {len(records)}"
    from collections import Counter
    counts = Counter(r["verdict_class"] for r in records)
    assert counts["PASS"] == 3
    assert counts["shapes-incomplete"] == 56
    assert counts["upstream-not-invocable-here"] == 38
    assert counts["oracle-mismatch"] == 2
    assert counts["smoke-fail"] == 1


def test_load_empty_file_raises(tmp_path):
    from automation.lib.resolve_loader import load_resolve_artifact

    p = tmp_path / "empty.log"
    p.write_text("", encoding="utf-8")
    with pytest.raises(ValueError):
        load_resolve_artifact(p)


def test_load_missing_file_raises(tmp_path):
    from automation.lib.resolve_loader import load_resolve_artifact

    p = tmp_path / "missing.log"
    with pytest.raises(FileNotFoundError):
        load_resolve_artifact(p)


def test_load_aggregate_json_with_no_verdicts_key_falls_through(tmp_path):
    """A JSON file without a 'verdicts' key parses cleanly but produces
    zero records (JSONL fallback also gets 0 because the whole file is
    one JSON object, not one-per-line, and plain-text parser sees no
    legacy lines either)."""
    from automation.lib.resolve_loader import load_resolve_artifact

    p = tmp_path / "weird.json"
    p.write_text(json.dumps({"foo": "bar"}), encoding="utf-8")
    records = load_resolve_artifact(p)
    assert records == []


def test_load_corrupt_lines_skipped_silently(tmp_path):
    """JSONL with garbage lines mixed in: garbage skipped, valid records
    returned."""
    from automation.lib.resolve_loader import load_resolve_artifact

    p = tmp_path / "mixed.log"
    p.write_text(
        "this is not json\n"
        + json.dumps({
            "task_id": "T1",
            "status": "BLOCKED(no-verified-upstream-entrypoint: shapes-incomplete)",
        })
        + "\n"
        + "[T2] BLOCKED - :: BLOCKED(no-verified-upstream-entrypoint: oracle-mismatch)\n",
        encoding="utf-8",
    )
    records = load_resolve_artifact(p)
    # JSONL pass picks up T1 (one valid line); the legacy plain-text fallback
    # only runs when the JSONL path returns nothing, so T2 is dropped in the
    # mixed file. We just assert at least the valid JSONL record survives.
    by_id = {r["task_id"]: r for r in records}
    assert "T1" in by_id
    assert by_id["T1"]["verdict_class"] == "shapes-incomplete"


def test_load_aggregate_json_preserves_reason(tmp_path):
    """The 'reason' field from the aggregate JSON must survive normalisation
    so downstream evidence extractors can mine it."""
    from automation.lib.resolve_loader import load_resolve_artifact

    p = tmp_path / "r.json"
    payload = {
        "verdicts": [{
            "task_id": "T1",
            "status": "BLOCKED(no-verified-upstream-entrypoint: shapes-incomplete)",
            "reason": "missing param 'foo' for entrypoint 'bar'",
        }],
    }
    p.write_text(json.dumps(payload), encoding="utf-8")
    rec = load_resolve_artifact(p)[0]
    assert "missing param 'foo'" in rec["reason"]
