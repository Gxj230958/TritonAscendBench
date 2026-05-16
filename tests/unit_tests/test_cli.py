"""Smoke tests for `tab` CLI subcommands.

codex round-2 R-P0-2: validate-submission / package / static-check / analyze
must all be executable, not stubbed return-2.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent.parent


def _run_tab(*args: str, check: bool = False) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "triton_ascend_bench.cli", *args],
        cwd=REPO, capture_output=True, text=True, check=check,
    )


def test_tab_help_lists_all_subcommands():
    r = _run_tab("--help")
    out = r.stdout + r.stderr
    for cmd in [
        "init-solution", "run", "run-suite", "package", "submit",
        "validate-submission", "validate-result", "static-check", "analyze",
    ]:
        assert cmd in out, f"`tab --help` does not list `{cmd}`"


def test_init_solution_and_static_check_smoke(tmp_path):
    sol_dir = tmp_path / "sol"
    r = _run_tab("init-solution", "L1_12_rms_norm_kernel", "--out", str(sol_dir))
    assert r.returncode == 0
    assert (sol_dir / "solution.py").exists()

    # Skeleton solution should fail static check (no @triton.jit).
    out_path = tmp_path / "static.json"
    r = _run_tab(
        "static-check",
        "--solution", str(sol_dir / "solution.py"),
        "--task-id", "L1_12_rms_norm_kernel",
        "--out", str(out_path),
    )
    assert r.returncode == 1, f"skeleton must fail static-check: {r.stdout} {r.stderr}"
    doc = json.loads(out_path.read_text())
    assert doc["any_blocked"]


def test_package_and_validate_submission_smoke(tmp_path):
    sol_dir = tmp_path / "sol"
    _run_tab("init-solution", "L1_12_rms_norm_kernel", "--out", str(sol_dir))
    pkg = tmp_path / "sub.tar.gz"
    r = _run_tab("package", str(sol_dir), "--out", str(pkg))
    assert r.returncode == 0
    assert pkg.exists()

    # validate-submission returns 1 because no manifest.yaml — fine, it ran.
    out_path = tmp_path / "valid.json"
    r = _run_tab("validate-submission", str(pkg), "--out", str(out_path))
    assert r.returncode in (0, 1), f"unexpected rc={r.returncode}"
    doc = json.loads(out_path.read_text())
    assert "solution_count" in doc


def test_static_check_on_submission_unpacks(tmp_path):
    sol_dir = tmp_path / "sol"
    _run_tab("init-solution", "L1_12_rms_norm_kernel", "--out", str(sol_dir))
    pkg = tmp_path / "sub.tar.gz"
    _run_tab("package", str(sol_dir), "--out", str(pkg))

    out_path = tmp_path / "static.json"
    r = _run_tab(
        "static-check", "--submission", str(pkg), "--out", str(out_path)
    )
    # Skeleton solution → any_blocked true → rc 1.
    assert r.returncode == 1
    doc = json.loads(out_path.read_text())
    assert doc["any_blocked"]
    assert doc["files"][0]["blocked"]


def test_run_suite_skips_missing_solutions(tmp_path):
    # Empty solutions root → every task gets `skipped: missing solution.py`.
    out_path = tmp_path / "suite.json"
    r = _run_tab(
        "run-suite", "--level", "1", "--solutions-root", str(tmp_path / "no_sols"),
        "--out", str(out_path),
    )
    assert r.returncode == 0
    doc = json.loads(out_path.read_text())
    assert doc["n_tasks"] > 0
    assert all(r.get("skipped") for r in doc["results"])


def test_analyze_smoke(tmp_path):
    """analyze must emit the standard fields. Results whose task_id is
    not in the catalogue are ignored (post-AscendC: scored catalogue is
    the full task set; unknown ids cannot inflate the score)."""
    verified = {
        "results": [
            {"task_id": "t1",
             "static_checks_passed": True, "correctness_status": "ok",
             "runtime_status": "ok", "performance_status": "ok",
             "aggregate_score": 1.5},
            {"task_id": "t2",
             "static_checks_passed": False, "correctness_status": "fail",
             "runtime_status": "ok", "performance_status": "ok",
             "aggregate_score": 0.01},
        ]
    }
    (tmp_path / "v.json").write_text(json.dumps(verified))
    out_path = tmp_path / "score.json"
    r = _run_tab(
        "analyze",
        "--verified-results", str(tmp_path / "v.json"),
        "--out", str(out_path),
    )
    assert r.returncode == 0
    doc = json.loads(out_path.read_text())
    assert "capped_geomean_native" in doc
    assert "fastp_native_at_1_0" in doc
    # Fake task_ids cannot inflate scores; both are non-catalogue.
    assert doc["n_native_measurements"] == 0
    assert doc["non_catalogue_task_count"] == 2


def test_analyze_does_not_use_static_check_as_correctness(tmp_path):
    """codex round-3 N-P0-1 regression test.

    A result with `static_checks_passed=True` but no real correctness
    (status fields set to error) must NOT show up as coverage=1.0.
    """
    verified = {
        "results": [
            {"task_id": "t1",
             "static_checks_passed": True,         # static-only path
             "correctness_status": "error",         # not yet run
             "runtime_status": "error",
             "performance_status": "error",
             "aggregate_score": 0.01, "coverage": 0.0},
        ]
    }
    (tmp_path / "v.json").write_text(json.dumps(verified))
    out_path = tmp_path / "score.json"
    r = _run_tab(
        "analyze",
        "--verified-results", str(tmp_path / "v.json"),
        "--out", str(out_path),
    )
    assert r.returncode == 0
    doc = json.loads(out_path.read_text())
    # NOT 1.0 — the single result is `correctness_status=error`, not `ok`.
    assert doc["native_coverage"] == 0.0


def test_analyze_filters_to_catalogue(tmp_path):
    """Only catalogue-known task_ids count. An unknown task_id cannot
    inflate the score regardless of the claimed aggregate_score."""
    verified = {
        "results": [
            {"task_id": "t1",
             "static_checks_passed": True, "correctness_status": "ok",
             "runtime_status": "ok", "performance_status": "ok",
             "aggregate_score": 2.0},
            {"task_id": "t2",
             "static_checks_passed": True, "correctness_status": "ok",
             "runtime_status": "ok", "performance_status": "ok",
             "aggregate_score": 100.0},
        ]
    }
    (tmp_path / "v.json").write_text(json.dumps(verified))
    out_path = tmp_path / "score.json"
    r = _run_tab(
        "analyze",
        "--verified-results", str(tmp_path / "v.json"),
        "--out", str(out_path),
    )
    assert r.returncode == 0
    doc = json.loads(out_path.read_text())
    # Neither t1 nor t2 is in the catalogue → 0 measurements, both
    # counted as non-catalogue (the 100.0 score cannot leak in).
    assert doc["n_native_measurements"] == 0
    assert doc["non_catalogue_task_count"] == 2


def test_validate_submission_rejects_missing_manifest(tmp_path):
    """codex round-3 R-P0-2: missing manifest.yaml must fail validation."""
    sol_dir = tmp_path / "sol"
    _run_tab("init-solution", "L1_12_rms_norm_kernel", "--out", str(sol_dir))
    pkg = tmp_path / "sub.tar.gz"
    _run_tab("package", str(sol_dir), "--out", str(pkg))

    out_path = tmp_path / "valid.json"
    r = _run_tab("validate-submission", str(pkg), "--out", str(out_path))
    assert r.returncode == 1
    doc = json.loads(out_path.read_text())
    assert doc["ok"] is False
    assert not doc["manifest_present"]
    assert any("manifest" in e for e in doc["errors"])


def test_validate_submission_rejects_malformed_yaml(tmp_path):
    """codex round-3 R-P0-2: malformed manifest.yaml content must reject."""
    import tarfile
    sol_dir = tmp_path / "sol"
    _run_tab("init-solution", "L1_12_rms_norm_kernel", "--out", str(sol_dir))
    (sol_dir / "manifest.yaml").write_text("not: [valid")  # malformed
    pkg = tmp_path / "sub.tar.gz"
    with tarfile.open(pkg, "w:gz") as tf:
        tf.add(str(sol_dir), arcname=sol_dir.name)

    out_path = tmp_path / "valid.json"
    r = _run_tab("validate-submission", str(pkg), "--out", str(out_path))
    assert r.returncode == 1
    doc = json.loads(out_path.read_text())
    assert doc["ok"] is False
    assert any("parse error" in e or "schema error" in e for e in doc["errors"])


def test_validate_submission_rejects_empty_manifest(tmp_path):
    """codex round-4 REGRESSION-P0: empty manifest.yaml (yaml -> None) must reject."""
    import tarfile
    sol_dir = tmp_path / "sol"
    _run_tab("init-solution", "L1_12_rms_norm_kernel", "--out", str(sol_dir))
    (sol_dir / "manifest.yaml").write_text("")           # empty
    pkg = tmp_path / "sub.tar.gz"
    with tarfile.open(pkg, "w:gz") as tf:
        tf.add(str(sol_dir), arcname=sol_dir.name)

    out_path = tmp_path / "valid.json"
    r = _run_tab("validate-submission", str(pkg), "--out", str(out_path))
    assert r.returncode == 1
    doc = json.loads(out_path.read_text())
    assert doc["ok"] is False
    assert any("empty" in e.lower() for e in doc["errors"])


def test_submit_refuses_invalid_package(tmp_path):
    """codex round-4 P1: submit must run validate-submission first."""
    sol_dir = tmp_path / "sol"
    _run_tab("init-solution", "L1_12_rms_norm_kernel", "--out", str(sol_dir))
    pkg = tmp_path / "sub.tar.gz"
    _run_tab("package", str(sol_dir), "--out", str(pkg))
    # sol has no manifest.yaml so the package is invalid.
    r = _run_tab(
        "submit", str(pkg), "--endpoint", "https://example.com"
    )
    assert r.returncode == 1
    assert "refusing" in r.stderr.lower() or "failed" in r.stderr.lower()


def test_validate_result_rejects_old_schema(tmp_path):
    """codex round-4 P1: result JSON without the five status fields must reject."""
    old = {
        "task_id": "t1",
        "static_checks_passed": True,
        "aggregate_score": 1.0,
    }
    (tmp_path / "r.json").write_text(json.dumps(old))
    r = _run_tab("validate-result", str(tmp_path / "r.json"))
    assert r.returncode == 1
    assert "missing keys" in r.stderr.lower()


def test_validate_result_accepts_full_schema(tmp_path):
    full = {
        "task_id": "t1",
        "static_checks_passed": True,
        "aggregate_score": 1.0,
        "correctness_status": "ok",
        "runtime_status": "ok",
        "performance_status": "ok",
        "performance_flags": [],
        "fallback_detected": False,
    }
    (tmp_path / "r.json").write_text(json.dumps(full))
    r = _run_tab("validate-result", str(tmp_path / "r.json"))
    assert r.returncode == 0


def test_round6_analyze_rejects_non_dict(tmp_path):
    """codex round-6 P1: `verified_results` must be a JSON object."""
    (tmp_path / "v.json").write_text("[]")     # list, not dict
    out_path = tmp_path / "score.json"
    r = _run_tab(
        "analyze", "--verified-results", str(tmp_path / "v.json"),
        "--out", str(out_path),
    )
    assert r.returncode == 1
    doc = json.loads(out_path.read_text())
    assert "error" in doc


def test_round10_validate_result_rejects_non_dict_toplevel(tmp_path):
    """codex round-10 P1 #3: top-level JSON must be a mapping."""
    (tmp_path / "r.json").write_text("[]")
    r = _run_tab("validate-result", str(tmp_path / "r.json"))
    assert r.returncode == 1
    assert "mapping" in r.stderr.lower() or "object" in r.stderr.lower()


def test_round10_analyze_rejects_non_list_results(tmp_path):
    """codex round-10 P1 #3: results: null / scalar must not iterate."""
    bad = {"results": None}
    (tmp_path / "v.json").write_text(json.dumps(bad))
    out_path = tmp_path / "score.json"
    r = _run_tab(
        "analyze", "--verified-results", str(tmp_path / "v.json"),
        "--out", str(out_path),
    )
    assert r.returncode == 0
    assert "TypeError" not in r.stderr


def test_round9_analyze_rejects_non_dict_result_entry(tmp_path):
    """codex round-9 P1: `results: [1]` (non-dict element) must not crash."""
    bad = {"results": [1, "string", None, {"task_id": "t1"}]}
    (tmp_path / "v.json").write_text(json.dumps(bad))
    out_path = tmp_path / "score.json"
    r = _run_tab(
        "analyze", "--verified-results", str(tmp_path / "v.json"),
        "--out", str(out_path),
    )
    assert r.returncode == 0
    assert "AttributeError" not in r.stderr
    doc = json.loads(out_path.read_text())
    assert doc["missing_task_id_count"] >= 3


def test_round9_validate_result_rejects_non_string_task_id(tmp_path):
    """codex round-9 P1: task_id must be a non-empty string."""
    bad = {
        "task_id": [],                       # list, not string
        "static_checks_passed": True,
        "aggregate_score": 1.0,
        "correctness_status": "ok",
        "runtime_status": "ok",
        "performance_status": "ok",
        "performance_flags": [],
        "fallback_detected": False,
    }
    (tmp_path / "r.json").write_text(json.dumps(bad))
    r = _run_tab("validate-result", str(tmp_path / "r.json"))
    assert r.returncode == 1
    assert "task_id" in r.stderr


def test_round8_analyze_handles_missing_task_id(tmp_path):
    """codex round-8 P1: result without task_id must not KeyError."""
    verified = {
        "results": [
            {
                # task_id missing
                "correctness_status": "ok",
                "runtime_status": "ok",
                "performance_status": "ok",
                "performance_flags": [],
                "fallback_detected": False,
                "aggregate_score": 1.0,
            }
        ]
    }
    (tmp_path / "v.json").write_text(json.dumps(verified))
    out_path = tmp_path / "score.json"
    r = _run_tab(
        "analyze", "--verified-results", str(tmp_path / "v.json"),
        "--out", str(out_path),
    )
    assert r.returncode == 0
    assert "Traceback" not in r.stderr
    assert "KeyError" not in r.stderr
    doc = json.loads(out_path.read_text())
    assert doc["missing_task_id_count"] == 1


def test_round7_run_suite_rejects_invalid_level(tmp_path):
    """codex round-7 P1: --level must reject `abc` / `-1`."""
    out_path = tmp_path / "suite.json"
    r = _run_tab(
        "run-suite", "--level", "abc",
        "--solutions-root", str(tmp_path / "no_sols"),
        "--out", str(out_path),
    )
    assert r.returncode != 0
    assert "invalid choice" in r.stderr.lower() or "choices" in r.stderr.lower()


def test_round7_validate_result_rejects_bool_as_score(tmp_path):
    """codex round-7 P1: `aggregate_score: true` (bool, not int)."""
    bad = {
        "task_id": "t1",
        "static_checks_passed": True,
        "aggregate_score": True,             # bool not number
        "correctness_status": "ok",
        "runtime_status": "ok",
        "performance_status": "ok",
        "performance_flags": [],
        "fallback_detected": False,
    }
    (tmp_path / "r.json").write_text(json.dumps(bad))
    r = _run_tab("validate-result", str(tmp_path / "r.json"))
    assert r.returncode == 1
    assert "aggregate_score" in r.stderr


def test_round7_validate_result_rejects_str_static_checks_passed(tmp_path):
    """codex round-7 P1: `static_checks_passed: 'yes'`."""
    bad = {
        "task_id": "t1",
        "static_checks_passed": "yes",
        "aggregate_score": 1.0,
        "correctness_status": "ok",
        "runtime_status": "ok",
        "performance_status": "ok",
        "performance_flags": [],
        "fallback_detected": False,
    }
    (tmp_path / "r.json").write_text(json.dumps(bad))
    r = _run_tab("validate-result", str(tmp_path / "r.json"))
    assert r.returncode == 1
    assert "static_checks_passed" in r.stderr


def test_round7_validate_result_handles_unhashable_status(tmp_path):
    """codex round-7 P1: status as list should not crash."""
    bad = {
        "task_id": "t1",
        "static_checks_passed": True,
        "aggregate_score": 1.0,
        "correctness_status": [],
        "runtime_status": "ok",
        "performance_status": "ok",
        "performance_flags": [],
        "fallback_detected": False,
    }
    (tmp_path / "r.json").write_text(json.dumps(bad))
    r = _run_tab("validate-result", str(tmp_path / "r.json"))
    assert r.returncode == 1                          # rejected, not crashed
    assert "correctness_status" in r.stderr


def test_round7_validate_result_rejects_non_string_flags(tmp_path):
    """codex round-7 P1: performance_flags must be list[str]."""
    bad = {
        "task_id": "t1",
        "static_checks_passed": True,
        "aggregate_score": 1.0,
        "correctness_status": "ok",
        "runtime_status": "ok",
        "performance_status": "ok",
        "performance_flags": [1, {}],                  # invalid contents
        "fallback_detected": False,
    }
    (tmp_path / "r.json").write_text(json.dumps(bad))
    r = _run_tab("validate-result", str(tmp_path / "r.json"))
    assert r.returncode == 1
    assert "performance_flags" in r.stderr


def test_round7_tar_symlink_rejected(tmp_path):
    """codex round-7 P1: symlink/hardlink entries must be refused."""
    import tarfile

    sol_dir = tmp_path / "sol"
    sol_dir.mkdir()
    (sol_dir / "solution.py").write_text("# ok\n")
    # Build a tarball with a symlink entry pointing to solution.py.
    pkg = tmp_path / "sub.tar.gz"
    with tarfile.open(pkg, "w:gz") as tf:
        tf.add(str(sol_dir / "solution.py"), arcname="sol/solution.py")
        link = tarfile.TarInfo("sol/payload.py")
        link.type = tarfile.SYMTYPE
        link.linkname = "solution.py"
        tf.addfile(link)
    out_path = tmp_path / "valid.json"
    r = _run_tab("validate-submission", str(pkg), "--out", str(out_path))
    assert r.returncode == 1
    doc = json.loads(out_path.read_text())
    assert any(
        "symlink" in e.lower() or "hardlink" in e.lower() for e in doc["errors"]
    )


def test_round6_validate_result_rejects_nan_score(tmp_path):
    """codex round-6 NEW-P0 #5: aggregate_score must be finite."""
    bad = {
        "task_id": "t1",
        "static_checks_passed": True,
        "aggregate_score": float("nan"),
        "correctness_status": "ok",
        "runtime_status": "ok",
        "performance_status": "ok",
        "performance_flags": [],
        "fallback_detected": False,
    }
    # Use python's json with allow_nan to embed NaN.
    (tmp_path / "r.json").write_text(
        json.dumps(bad, allow_nan=True)
    )
    r = _run_tab("validate-result", str(tmp_path / "r.json"))
    assert r.returncode == 1
    assert "finite" in r.stderr.lower()


def test_validate_result_rejects_invalid_status_values(tmp_path):
    """codex round-5 P1: enum + type validation across all status fields."""
    bad = {
        "task_id": "t1",
        "static_checks_passed": True,
        "aggregate_score": 1.0,
        "correctness_status": "ok",
        "runtime_status": "not_a_status",     # invalid enum
        "performance_status": "not_a_status", # invalid enum
        "performance_flags": "not-a-list",    # wrong type
        "fallback_detected": "false",          # str not bool
    }
    (tmp_path / "r.json").write_text(json.dumps(bad))
    r = _run_tab("validate-result", str(tmp_path / "r.json"))
    assert r.returncode == 1
    err = r.stderr
    assert "runtime_status" in err
    assert "performance_status" in err
    assert "performance_flags" in err
    assert "fallback_detected" in err


def test_validate_submission_accepts_well_formed(tmp_path):
    """A valid manifest.yaml + solution.py tarball passes validation."""
    import tarfile
    import yaml as _yaml

    sol_dir = tmp_path / "sol"
    _run_tab("init-solution", "L1_12_rms_norm_kernel", "--out", str(sol_dir))
    manifest = {
        "submission_id": "01HXXX",
        "task_id": "L1_12_rms_norm_kernel",
        "submitter": {
            "org_identity": "sha256:abc",
            "contact": "alice@example.com",
        },
        "triton_ascend_version": "3.2.0",
        "cann_version": "8.5.0",
    }
    (sol_dir / "manifest.yaml").write_text(_yaml.safe_dump(manifest))
    pkg = tmp_path / "sub.tar.gz"
    with tarfile.open(pkg, "w:gz") as tf:
        tf.add(str(sol_dir), arcname=sol_dir.name)

    out_path = tmp_path / "valid.json"
    r = _run_tab("validate-submission", str(pkg), "--out", str(out_path))
    assert r.returncode == 0
    doc = json.loads(out_path.read_text())
    assert doc["ok"] is True
    assert doc["manifest_schema_ok"] is True


def test_tab_run_does_not_crash_on_skeleton_solution(tmp_path):
    """codex round-2 N-P0-1 regression: `tab run` must not raise TypeError."""
    sol_dir = tmp_path / "sol"
    _run_tab("init-solution", "L1_12_rms_norm_kernel", "--out", str(sol_dir))
    # Skeleton solution will fail static check but the CLI must exit
    # cleanly with rc=1, not crash with TypeError.
    out_path = tmp_path / "run.json"
    r = _run_tab(
        "run", "L1_12_rms_norm_kernel",
        "--solution", str(sol_dir / "solution.py"),
        "--backend", "gpu",                                # avoid NPU import
        "--out", str(out_path),
    )
    assert r.returncode in (0, 1), f"crash: {r.stderr}"
    assert "TypeError" not in r.stderr
    assert out_path.exists()
