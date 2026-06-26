"""G1 parse-in-parent: the trusted-PARENT offline parse of a child-recorded NPU
profiler raw dir (`_parse_npu_kernel_event`). The in-child CSV parse is blocked
by the graded sandbox (exec), so the parent parses the raw dir and fills
kernel_event_present / fallback_signal. Tests the anti-forge validation (all
reachable on dev) + the analyse success/fallback paths (mocked torch_npu).
"""
from __future__ import annotations

import sys
import types

from ascendbench.integrity.process_isolation import _parse_npu_kernel_event


def _raw(root, name="t_x", asc="host_ascend_pt", make_out=True):
    """Build <root>/<name>/<asc>/ASCEND_PROFILER_OUTPUT and return <root>/<name>."""
    rd = root / name
    out = rd / asc / "ASCEND_PROFILER_OUTPUT"
    out.mkdir(parents=True)
    return rd, out


def _mock_analyse(monkeypatch, fn):
    """Inject a fake torch_npu.profiler.profiler.analyse (dev has no torch_npu)."""
    tn = types.ModuleType("torch_npu")
    prof = types.ModuleType("torch_npu.profiler")
    profprof = types.ModuleType("torch_npu.profiler.profiler")
    profprof.analyse = fn
    monkeypatch.setitem(sys.modules, "torch_npu", tn)
    monkeypatch.setitem(sys.modules, "torch_npu.profiler", prof)
    monkeypatch.setitem(sys.modules, "torch_npu.profiler.profiler", profprof)


# ── anti-forge / validation (no analyse reached) ─────────────────────────────
def test_none_inputs_unverified():
    assert _parse_npu_kernel_event(None, "/tmp") == (None, None)
    assert _parse_npu_kernel_event("/tmp/x", None) == (None, None)


def test_raw_dir_outside_root_is_unverified(tmp_path):
    """A raw dir that resolves OUTSIDE the parent-owned root is rejected (a
    candidate cannot point the parent at an attacker-controlled trace)."""
    root = tmp_path / "root"
    root.mkdir()
    outside = tmp_path / "outside"
    rd, _ = _raw(outside)
    assert _parse_npu_kernel_event(str(rd), str(root)) == (None, None)


def test_zero_trace_dirs_unverified(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    rd = root / "t_x"
    rd.mkdir()  # no *_ascend_pt inside
    assert _parse_npu_kernel_event(str(rd), str(root)) == (None, None)


def test_multiple_trace_dirs_unverified(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    rd = root / "t_x"
    rd.mkdir()
    (rd / "a_ascend_pt").mkdir()
    (rd / "b_ascend_pt").mkdir()
    assert _parse_npu_kernel_event(str(rd), str(root)) == (None, None)


# ── analyse success / fallback (mocked torch_npu) ────────────────────────────
def test_kernel_rows_present_is_true(tmp_path, monkeypatch):
    root = tmp_path / "root"
    root.mkdir()
    rd, out = _raw(root)
    _mock_analyse(monkeypatch,
                  lambda p, **k: (out / "kernel_details.csv").write_text(
                      "Name,Type\nmul2,AI_CORE\n"))
    assert _parse_npu_kernel_event(str(rd), str(root)) == (True, False)


def test_zero_rows_is_fallback(tmp_path, monkeypatch):
    root = tmp_path / "root"
    root.mkdir()
    rd, out = _raw(root)
    # analyse writes a header-only CSV (no kernel ran -> CPU fallback)
    _mock_analyse(monkeypatch,
                  lambda p, **k: (out / "kernel_details.csv").write_text("Name,Type\n"))
    assert _parse_npu_kernel_event(str(rd), str(root)) == (False, True)


def test_analyse_produces_no_csv_is_unverified(tmp_path, monkeypatch):
    root = tmp_path / "root"
    root.mkdir()
    rd, _out = _raw(root)
    _mock_analyse(monkeypatch, lambda p, **k: None)  # parse produced nothing
    assert _parse_npu_kernel_event(str(rd), str(root)) == (None, None)


def test_preexisting_forged_csv_is_deleted_before_parse(tmp_path, monkeypatch):
    """A candidate-forged kernel_details.csv must NOT be trusted: the parent
    deletes it and requires its own analyse to regenerate. Here analyse writes
    nothing, so the forged rows are gone -> unverified, not a forged True."""
    root = tmp_path / "root"
    root.mkdir()
    rd, out = _raw(root)
    (out / "kernel_details.csv").write_text("Name,Type\nFORGED,AI_CORE\n")
    _mock_analyse(monkeypatch, lambda p, **k: None)  # does not regenerate it
    assert _parse_npu_kernel_event(str(rd), str(root)) == (None, None)


def test_analyse_raising_is_unverified(tmp_path, monkeypatch):
    root = tmp_path / "root"
    root.mkdir()
    rd, _out = _raw(root)
    def _boom(p, **k):
        raise RuntimeError("parser blew up")
    _mock_analyse(monkeypatch, _boom)
    assert _parse_npu_kernel_event(str(rd), str(root)) == (None, None)


# ── path-safety hardening: rd==root, symlink trace dir, parse timeout ────────
def test_rd_equals_root_unverified(tmp_path):
    """raw_dir == prof_root must be rejected (the raw dir must be a STRICT subdir
    of the parent-owned root, never the root itself)."""
    root = tmp_path / "root"
    (root / "host_ascend_pt" / "ASCEND_PROFILER_OUTPUT").mkdir(parents=True)
    assert _parse_npu_kernel_event(str(root), str(root)) == (None, None)


def test_symlinked_trace_dir_unverified(tmp_path):
    """A *_ascend_pt that is a SYMLINK (redirecting the parse outside the
    parent-owned tree) must be rejected even though it is a dir."""
    root = tmp_path / "root"
    root.mkdir()
    rd = root / "t_x"
    rd.mkdir()
    external = tmp_path / "external_ascend_pt"
    (external / "ASCEND_PROFILER_OUTPUT").mkdir(parents=True)
    (rd / "host_ascend_pt").symlink_to(external, target_is_directory=True)
    assert _parse_npu_kernel_event(str(rd), str(root)) == (None, None)


def test_analyse_hang_times_out_unverified(tmp_path, monkeypatch):
    """A parse that hangs past the timeout is treated as a failure (unverified ->
    fail-closed), not a hang of the runner."""
    import ascendbench.integrity.process_isolation as pi

    root = tmp_path / "root"
    root.mkdir()
    rd, out = _raw(root)

    def _hang(p, **k):
        import time
        time.sleep(5)  # longer than the patched timeout below
        (out / "kernel_details.csv").write_text("Name\nmul2\n")

    _mock_analyse(monkeypatch, _hang)
    monkeypatch.setattr(pi, "_ANALYSE_TIMEOUT_S", 0.3)
    assert _parse_npu_kernel_event(str(rd), str(root)) == (None, None)


def test_run_with_timeout_basic():
    from ascendbench.integrity.process_isolation import _run_with_timeout
    assert _run_with_timeout(lambda: None, 2.0) is True

    def _boom():
        raise RuntimeError("x")
    assert _run_with_timeout(_boom, 2.0) is False

    import time
    assert _run_with_timeout(lambda: time.sleep(2.0), 0.2) is False


def test_nested_output_dir_symlink_unverified(tmp_path, monkeypatch):
    """A REAL *_ascend_pt dir whose intermediate ASCEND_PROFILER_OUTPUT is a
    SYMLINK to an external dir must be rejected — the parse/delete must not
    follow it outside the parent-owned tree."""
    root = tmp_path / "root"
    root.mkdir()
    rd = root / "t_x"
    trace = rd / "host_ascend_pt"
    trace.mkdir(parents=True)
    external = tmp_path / "external_out"
    external.mkdir()
    (external / "kernel_details.csv").write_text("Name,Type\nEXT,AI_CORE\n")
    (trace / "ASCEND_PROFILER_OUTPUT").symlink_to(external, target_is_directory=True)
    # analyse would never even run; if it did, it must not read the external CSV.
    _mock_analyse(monkeypatch, lambda p, **k: None)
    assert _parse_npu_kernel_event(str(rd), str(root)) == (None, None)
