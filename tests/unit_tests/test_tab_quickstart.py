"""Behavior + boundary tests for the tab_quickstart.sh wrapper.

These tests exercise the operator-facing one-shot wrapper without
touching real SSH, real GPU, or the real acceptance gate. The strategy
is to copy the production tab_quickstart.sh into a tmpdir alongside
stub `setup_machine_a_gpu.sh` and `run_acceptance_gate.sh` files that
record their argv into a log; the wrapper resolves RUNNERS_DIR off its
own location, so it picks up the stubs transparently. SSH is mocked
through a PATH shim that prepends a tmpdir containing a fake `ssh`
executable.

The release-boundary assertion at the bottom mirrors
test_release_boundary.py-style export-ignore coverage but stays inline
to keep the wrapper's own boundary self-evident in one file.
"""
from __future__ import annotations

import json
import os
import shutil
import stat
import subprocess
import textwrap
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_REAL_SCRIPT = (
    _REPO_ROOT / "scripts" / "automation" / "runners" / "tab_quickstart.sh"
)
_REAL_CONF_EXAMPLE = (
    _REPO_ROOT
    / "scripts"
    / "automation"
    / "runners"
    / "tab_quickstart.conf.example"
)


# ---------------------------------------------------------------------------
# Test harness helpers
# ---------------------------------------------------------------------------
def _bash() -> str:
    path = shutil.which("bash")
    if path is None:  # pragma: no cover -- sandbox without bash is a hard fail
        pytest.skip("bash not available on this host")
    return path


def _make_executable(path: Path) -> None:
    mode = path.stat().st_mode
    path.chmod(mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def _write_stub_runner(path: Path, recorder_log: Path, exit_rc: int = 0) -> None:
    """Drop a stub runner that records its argv as JSON lines."""
    body = (
        "#!/usr/bin/env bash\n"
        f"# Stub runner -- records argv as a JSON line, exits {exit_rc}.\n"
        "set -eu\n"
        f"python3 - \"$@\" <<'PY' >> {recorder_log!s}\n"
        "import json, sys\n"
        'print(json.dumps({"script": sys.argv[0], "argv": sys.argv[1:]}))\n'
        "PY\n"
        f"exit {exit_rc}\n"
    )
    path.write_text(body, encoding="utf-8")
    _make_executable(path)


def _write_path_shim(
    bin_dir: Path,
    name: str,
    recorder_log: Path,
    exit_rc: int = 0,
    stdout: str = "",
) -> None:
    """Drop a PATH-shim binary (e.g. ssh) that records argv + emits stdout.

    `stdout` is emitted verbatim (newlines preserved) via a single-quoted
    heredoc so shell metacharacters are not interpreted.
    """
    path = bin_dir / name
    body = (
        "#!/usr/bin/env bash\n"
        f"python3 - \"$@\" <<'PY' >> {recorder_log!s}\n"
        "import json, sys, os\n"
        "print(json.dumps({\n"
        '  "bin": os.path.basename(sys.argv[0]),\n'
        '  "argv": sys.argv[1:],\n'
        "}))\n"
        "PY\n"
        "cat <<'STDOUT_EOF'\n"
        f"{stdout}"
        "STDOUT_EOF\n"
        f"exit {exit_rc}\n"
    )
    path.write_text(body, encoding="utf-8")
    _make_executable(path)


@pytest.fixture
def harness(tmp_path: Path) -> dict:
    """Build a sandboxed copy of tab_quickstart.sh with stub runners.

    Layout:
      tmp_path/
        runners/                       <-- copy of tab_quickstart.sh + stubs
          tab_quickstart.sh
          setup_machine_a_gpu.sh       <-- stub
          run_acceptance_gate.sh       <-- stub
        bin/                           <-- PATH shim
          ssh                          <-- stub (default: succeed)
        gpfs/                          <-- $TAB_GPFS_ROOT
        conf/tab.conf                  <-- TAB_CONFIG target
        log/runner.log                 <-- argv recorder for stubs
        log/ssh.log                    <-- argv recorder for ssh shim
        home/                          <-- HOME override (no real conf)
    """
    runners = tmp_path / "runners"
    bin_dir = tmp_path / "bin"
    gpfs = tmp_path / "gpfs"
    conf_dir = tmp_path / "conf"
    log_dir = tmp_path / "log"
    home_dir = tmp_path / "home"
    for d in (runners, bin_dir, gpfs, conf_dir, log_dir, home_dir):
        d.mkdir(parents=True, exist_ok=True)

    shutil.copy(_REAL_SCRIPT, runners / "tab_quickstart.sh")
    _make_executable(runners / "tab_quickstart.sh")

    runner_log = log_dir / "runner.log"
    ssh_log = log_dir / "ssh.log"
    runner_log.touch()
    ssh_log.touch()

    _write_stub_runner(runners / "setup_machine_a_gpu.sh", runner_log)
    _write_stub_runner(runners / "run_acceptance_gate.sh", runner_log)
    _write_path_shim(bin_dir, "ssh", ssh_log, exit_rc=0, stdout="OK\n")

    conf_file = conf_dir / "tab.conf"
    conf_file.write_text(
        textwrap.dedent(
            f"""\
            TAB_GPFS_ROOT="{gpfs}"
            TAB_C_HOST="user@stub-host"
            TAB_C_CLI_CMD='stub-cli --print {{prompt}}'
            """
        ),
        encoding="utf-8",
    )

    return {
        "tmp": tmp_path,
        "script": runners / "tab_quickstart.sh",
        "runners": runners,
        "bin_dir": bin_dir,
        "gpfs": gpfs,
        "conf_file": conf_file,
        "runner_log": runner_log,
        "ssh_log": ssh_log,
        "home_dir": home_dir,
    }


def _run(
    harness: dict,
    *args: str,
    env_extra: dict | None = None,
    stdin: str | None = None,
    use_conf: bool = True,
    use_ssh_shim: bool = True,
) -> subprocess.CompletedProcess:
    env = {
        # Minimal env -- avoid leaking the developer's real ~/.tab-quickstart.conf.
        "HOME": str(harness["home_dir"]),
        "PATH": (
            f"{harness['bin_dir']}:{os.environ.get('PATH', '')}"
            if use_ssh_shim
            else os.environ.get("PATH", "")
        ),
    }
    if use_conf:
        env["TAB_CONFIG"] = str(harness["conf_file"])
    else:
        # Point TAB_CONFIG at a missing file so the wrapper skips sourcing.
        env["TAB_CONFIG"] = str(harness["tmp"] / "nonexistent.conf")
    if env_extra:
        env.update(env_extra)
    return subprocess.run(
        [_bash(), str(harness["script"]), *args],
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
        input=stdin,
        env=env,
    )


def _read_argv_log(log_path: Path) -> list[dict]:
    rows = []
    for ln in log_path.read_text(encoding="utf-8").splitlines():
        ln = ln.strip()
        if not ln:
            continue
        rows.append(json.loads(ln))
    return rows


# ---------------------------------------------------------------------------
# 1. Help + dispatch
# ---------------------------------------------------------------------------
def test_tab_quickstart_help_lists_all_subcommands(harness):
    # Help works even without any conf loaded -- prove that path too.
    proc = _run(harness, "help", use_conf=False)
    assert proc.returncode == 0, proc.stderr
    for cmd in ("setup", "verify-c", "resolve", "smoke", "full", "status", "clean"):
        assert cmd in proc.stdout, f"help missing subcommand: {cmd}"
    # Help also surfaces the conf file path and the example path.
    assert "CONFIG_FILE" in proc.stdout
    assert "tab_quickstart.conf.example" in proc.stdout


def test_tab_quickstart_unknown_subcommand_fatal_rc2(harness):
    proc = _run(harness, "bogus")
    assert proc.returncode == 2, (proc.returncode, proc.stderr)
    assert "unknown subcommand" in proc.stderr


# ---------------------------------------------------------------------------
# 2. Config loading
# ---------------------------------------------------------------------------
def test_tab_quickstart_missing_required_var_fails(harness):
    # No conf, no env vars set for any of the three required keys.
    proc = _run(harness, "status", use_conf=False)
    assert proc.returncode != 0
    assert "TAB_GPFS_ROOT not set" in proc.stderr


def test_tab_quickstart_loads_conf_file_from_env(harness):
    # The harness conf supplies the three required vars; status must run.
    proc = _run(harness, "status")
    assert proc.returncode == 0, (proc.stderr, proc.stdout)
    assert "TAB_GPFS_ROOT" in proc.stdout
    assert str(harness["gpfs"]) in proc.stdout


def test_tab_quickstart_env_overrides_conf(harness):
    # conf sets TAB_GPFS_ROOT to harness["gpfs"]; env points elsewhere.
    override = harness["tmp"] / "env-root"
    override.mkdir()
    proc = _run(harness, "status", env_extra={"TAB_GPFS_ROOT": str(override)})
    assert proc.returncode == 0, proc.stderr
    assert str(override) in proc.stdout
    # Sanity: conf's GPFS path should NOT appear as the active value
    # in the config block (it may still appear in derived defaults if
    # the operator left them at the default, but env override wins).
    config_block = proc.stdout.split("== state ==", 1)[0]
    assert str(override) in config_block, (
        "env override TAB_GPFS_ROOT must surface in the config block"
    )


def test_tab_quickstart_cli_cmd_must_contain_prompt_placeholder(harness):
    # Override TAB_C_CLI_CMD to something missing {prompt}.
    proc = _run(harness, "status", env_extra={"TAB_C_CLI_CMD": "no-placeholder-cli"})
    assert proc.returncode == 2, (proc.returncode, proc.stderr)
    assert "must contain literal '{prompt}'" in proc.stderr


# ---------------------------------------------------------------------------
# 3. setup
# ---------------------------------------------------------------------------
def test_tab_quickstart_setup_delegates_to_setup_machine_a(harness):
    proc = _run(harness, "setup")
    assert proc.returncode == 0, (proc.stderr, proc.stdout)
    rows = _read_argv_log(harness["runner_log"])
    assert len(rows) == 1, rows
    argv = rows[0]["argv"]
    assert "--gpfs-root" in argv
    gpfs_idx = argv.index("--gpfs-root")
    assert argv[gpfs_idx + 1] == str(harness["gpfs"])
    assert "--skip-install" not in argv


def test_tab_quickstart_setup_skip_install_passes_flag(harness):
    proc = _run(harness, "setup", env_extra={"TAB_SETUP_SKIP_INSTALL": "1"})
    assert proc.returncode == 0, proc.stderr
    rows = _read_argv_log(harness["runner_log"])
    assert len(rows) == 1, rows
    argv = rows[0]["argv"]
    assert "--skip-install" in argv


# ---------------------------------------------------------------------------
# 4. verify-c
# ---------------------------------------------------------------------------
def test_tab_quickstart_verify_c_runs_ssh_echo_and_cli_version(harness):
    proc = _run(harness, "verify-c")
    assert proc.returncode == 0, (proc.stderr, proc.stdout)
    rows = _read_argv_log(harness["ssh_log"])
    # Two SSH invocations: 'echo OK' probe + CLI --version probe.
    assert len(rows) == 2, rows
    # Both calls target TAB_C_HOST.
    for row in rows:
        argv = row["argv"]
        assert "user@stub-host" in argv, argv
    # First call must end in 'echo OK'.
    assert any("echo OK" in tok for tok in rows[0]["argv"])
    # Second call must contain '--version' and reference 'stub-cli'.
    second_joined = " ".join(rows[1]["argv"])
    assert "stub-cli --version" in second_joined or "stub-cli" in second_joined
    assert "--version" in second_joined


def test_tab_quickstart_verify_c_extracts_cli_name_past_env_prefix(harness):
    # 'env LANG=C stub-cli ...' must still resolve CLI name = stub-cli.
    proc = _run(
        harness,
        "verify-c",
        env_extra={"TAB_C_CLI_CMD": "env LANG=C stub-cli --print {prompt}"},
    )
    assert proc.returncode == 0, proc.stderr
    rows = _read_argv_log(harness["ssh_log"])
    assert len(rows) == 2
    second_joined = " ".join(rows[1]["argv"])
    assert "stub-cli --version" in second_joined


# ---------------------------------------------------------------------------
# 5. resolve
# ---------------------------------------------------------------------------
def test_tab_quickstart_resolve_overwrite_prompt_fatal_without_tty(harness):
    # With existing resolve log and no --force, the overwrite confirmation
    # used to block on `read`. Now -- when stdin is not a TTY (pipe from
    # subprocess.PIPE here mirrors `tail -f /dev/null | tab_quickstart.sh
    # resolve` in production) -- we fatal-exit with rc=2 so the wrapper
    # never hangs in CI / non-interactive shells.
    rl = harness["gpfs"] / "pa_resolve.log"
    rl.parent.mkdir(parents=True, exist_ok=True)
    rl.write_text("PRE-EXISTING\n", encoding="utf-8")
    proc = _run(harness, "resolve", stdin="N\n")
    assert proc.returncode == 2, (proc.returncode, proc.stdout, proc.stderr)
    assert "stdin is not a TTY" in proc.stderr
    assert rl.read_text(encoding="utf-8") == "PRE-EXISTING\n"


def test_tab_quickstart_resolve_force_skips_prompt(harness, tmp_path):
    rl = harness["gpfs"] / "pa_resolve.log"
    rl.parent.mkdir(parents=True, exist_ok=True)
    rl.write_text("PRE-EXISTING\n", encoding="utf-8")

    # Build a stub bench repo that overrides scripts/resolve_upstream_entrypoints.py
    # to emit deterministic verdict-class lines.
    stub_repo = tmp_path / "stub_bench_repo"
    (stub_repo / "scripts").mkdir(parents=True)
    (stub_repo / "scripts" / "resolve_upstream_entrypoints.py").write_text(
        textwrap.dedent(
            """\
            print('task=a shapes-incomplete')
            print('task=b oracle-mismatch')
            """
        ),
        encoding="utf-8",
    )

    proc = _run(
        harness,
        "resolve",
        "--force",
        env_extra={"TAB_BENCH_REPO": str(stub_repo)},
    )
    assert proc.returncode == 0, (proc.stderr, proc.stdout)
    content = rl.read_text(encoding="utf-8")
    assert "shapes-incomplete" in content
    assert "PRE-EXISTING" not in content


# ---------------------------------------------------------------------------
# 6. smoke / full
# ---------------------------------------------------------------------------
def _seed_resolve_log(harness) -> Path:
    rl = harness["gpfs"] / "pa_resolve.log"
    rl.parent.mkdir(parents=True, exist_ok=True)
    rl.write_text("task=a shapes-incomplete\n", encoding="utf-8")
    return rl


def test_tab_quickstart_smoke_fails_when_resolve_log_missing(harness):
    proc = _run(harness, "smoke")
    assert proc.returncode != 0
    assert "run 'resolve' first" in proc.stderr


def test_tab_quickstart_smoke_passes_phase_1_to_run_gate(harness):
    _seed_resolve_log(harness)
    proc = _run(harness, "smoke")
    assert proc.returncode == 0, (proc.stderr, proc.stdout)
    rows = _read_argv_log(harness["runner_log"])
    assert len(rows) == 1
    argv = rows[0]["argv"]
    assert "--phase" in argv and argv[argv.index("--phase") + 1] == "1"
    assert "--llm-transport" in argv
    assert argv[argv.index("--llm-transport") + 1] == "ssh-cli"
    assert "--llm-ssh-host" in argv
    assert argv[argv.index("--llm-ssh-host") + 1] == "user@stub-host"
    assert "--llm-cli-cmd" in argv
    assert argv[argv.index("--llm-cli-cmd") + 1] == "stub-cli --print {prompt}"


def test_tab_quickstart_full_passes_phase_all(harness):
    _seed_resolve_log(harness)
    proc = _run(harness, "full")
    assert proc.returncode == 0, (proc.stderr, proc.stdout)
    rows = _read_argv_log(harness["runner_log"])
    assert len(rows) == 1
    argv = rows[0]["argv"]
    assert "--phase" in argv and argv[argv.index("--phase") + 1] == "all"


# ---------------------------------------------------------------------------
# 7. status / clean
# ---------------------------------------------------------------------------
def test_tab_quickstart_status_no_state_file_friendly(harness):
    proc = _run(harness, "status")
    assert proc.returncode == 0, proc.stderr
    assert "(no state file yet" in proc.stdout


def test_tab_quickstart_clean_force_removes_dirs(harness):
    gpfs = harness["gpfs"]
    for sub in ("state", "queue", "results", ".phase4-worktree"):
        (gpfs / sub).mkdir(parents=True, exist_ok=True)
        (gpfs / sub / "marker").write_text("x", encoding="utf-8")
    # Also seed logs/ and pa_resolve.log -- they must SURVIVE clean.
    (gpfs / "logs").mkdir(parents=True, exist_ok=True)
    (gpfs / "logs" / "keep.log").write_text("keep", encoding="utf-8")
    (gpfs / "pa_resolve.log").write_text("keep", encoding="utf-8")

    proc = _run(harness, "clean", "--force")
    assert proc.returncode == 0, (proc.stderr, proc.stdout)
    for sub in ("state", "queue", "results", ".phase4-worktree"):
        assert not (gpfs / sub).exists(), f"{sub} should be removed"
    assert (gpfs / "logs" / "keep.log").exists()
    assert (gpfs / "pa_resolve.log").exists()


# ---------------------------------------------------------------------------
# 8. Release boundary -- both new files must be export-ignored.
# ---------------------------------------------------------------------------
def test_tab_quickstart_artifacts_export_ignored():
    """tab_quickstart.{sh,conf.example} + test live under export-ignore patterns."""
    gitattrs = (_REPO_ROOT / ".gitattributes").read_text(encoding="utf-8")
    dockerignore = (_REPO_ROOT / ".dockerignore").read_text(encoding="utf-8")
    # The script + conf example are under scripts/automation/ which is
    # itself an export-ignore directory pattern.
    assert "scripts/automation/" in gitattrs
    assert "scripts/automation/" in dockerignore
    # The test file must be explicitly listed (it does NOT match any of
    # the existing test_*.py globs).
    assert "tests/unit_tests/test_tab_quickstart.py" in gitattrs, (
        "test_tab_quickstart.py must be export-ignored in .gitattributes"
    )
    assert "tests/unit_tests/test_tab_quickstart.py" in dockerignore, (
        "test_tab_quickstart.py must be mirrored in .dockerignore"
    )


def test_tab_quickstart_conf_example_exists_and_documents_required_vars():
    """The sample conf must enumerate all three required variables."""
    assert _REAL_CONF_EXAMPLE.is_file(), f"missing: {_REAL_CONF_EXAMPLE}"
    text = _REAL_CONF_EXAMPLE.read_text(encoding="utf-8")
    for var in ("TAB_GPFS_ROOT", "TAB_C_HOST", "TAB_C_CLI_CMD"):
        assert var in text, f"conf example missing {var}"
    # Must mention the {prompt} placeholder so operators don't omit it.
    assert "{prompt}" in text


def test_tab_quickstart_script_has_no_hardcoded_cli_names():
    """The wrapper must stay CLI-agnostic -- no hardcoded codex/claude/gemini."""
    text = _REAL_SCRIPT.read_text(encoding="utf-8")
    for needle in ("codex", "claude", "gemini"):
        assert needle.lower() not in text.lower(), (
            f"tab_quickstart.sh hardcodes CLI name '{needle}' "
            "-- must be operator-supplied via TAB_C_CLI_CMD"
        )


# ---------------------------------------------------------------------------
# 9. Env vs conf precedence corner cases
# ---------------------------------------------------------------------------
def _write_conf(harness, extra_lines: str) -> None:
    """Rewrite the harness conf with the three required vars + extra_lines."""
    harness["conf_file"].write_text(
        textwrap.dedent(
            f"""\
            TAB_GPFS_ROOT="{harness['gpfs']}"
            TAB_C_HOST="user@stub-host"
            TAB_C_CLI_CMD='stub-cli --print {{prompt}}'
            {extra_lines}
            """
        ),
        encoding="utf-8",
    )


def test_tab_quickstart_empty_env_overrides_conf_nonempty(harness):
    """Empty env var ('' is present, not unset) must trump non-empty conf.

    Conf sets TAB_SETUP_SKIP_INSTALL=1; env sets it to ''. Expected:
    setup runs without --skip-install. Emptiness-based precedence (the
    pre-fix logic) would treat the empty env as "unset" and keep conf's 1.
    """
    _write_conf(harness, 'TAB_SETUP_SKIP_INSTALL=1')
    proc = _run(harness, "setup", env_extra={"TAB_SETUP_SKIP_INSTALL": ""})
    assert proc.returncode == 0, (proc.stderr, proc.stdout)
    rows = _read_argv_log(harness["runner_log"])
    assert len(rows) == 1
    assert "--skip-install" not in rows[0]["argv"], (
        "empty env TAB_SETUP_SKIP_INSTALL must override conf's '1' "
        "(presence-based precedence)"
    )


def test_tab_quickstart_env_only_no_conf_works(harness):
    """Three required vars supplied only via env (no conf file) -> status 0."""
    proc = _run(
        harness,
        "status",
        use_conf=False,
        env_extra={
            "TAB_GPFS_ROOT": str(harness["gpfs"]),
            "TAB_C_HOST": "user@env-only-host",
            "TAB_C_CLI_CMD": "env-only-cli --print {prompt}",
        },
    )
    assert proc.returncode == 0, (proc.stderr, proc.stdout)
    assert "user@env-only-host" in proc.stdout
    assert "env-only-cli --print {prompt}" in proc.stdout


def test_tab_quickstart_verbose_empty_env_overrides_conf(harness):
    """Empty env TAB_VERBOSE must disable conf-supplied TAB_VERBOSE=1."""
    _write_conf(harness, 'TAB_VERBOSE=1')
    proc = _run(harness, "status", env_extra={"TAB_VERBOSE": ""})
    assert proc.returncode == 0, (proc.stderr, proc.stdout)
    # `set -x` would echo every command to stderr with a leading '+ '.
    # Status itself emits nothing to stderr in the happy path, so absence
    # of any '+ ' prefix line is a clean signal.
    trace_lines = [
        ln for ln in proc.stderr.splitlines() if ln.startswith("+ ")
    ]
    assert not trace_lines, (
        f"empty env TAB_VERBOSE should disable set -x; got trace: {trace_lines[:3]}"
    )


# ---------------------------------------------------------------------------
# 10. CLI cmd early-failure corner cases (quote-depth + exactly-once)
# ---------------------------------------------------------------------------
def test_tab_quickstart_cli_cmd_quoted_double_prompt_early_fail(harness):
    """`cli "{prompt}"` -- double-quoted placeholder -> rc=2 + 'do NOT wrap'."""
    proc = _run(
        harness,
        "status",
        env_extra={"TAB_C_CLI_CMD": 'cli "{prompt}"'},
    )
    assert proc.returncode == 2, (proc.returncode, proc.stderr)
    assert "do NOT wrap" in proc.stderr


def test_tab_quickstart_cli_cmd_quoted_single_prompt_early_fail(harness):
    """`cli '{prompt}'` -- single-quoted placeholder -> rc=2."""
    proc = _run(
        harness,
        "status",
        env_extra={"TAB_C_CLI_CMD": "cli '{prompt}'"},
    )
    assert proc.returncode == 2, (proc.returncode, proc.stderr)
    assert "do NOT wrap" in proc.stderr


def test_tab_quickstart_cli_cmd_outer_quotes_early_fail(harness):
    """`cli "prefix {prompt} suffix"` -- placeholder inside outer quotes -> rc=2."""
    proc = _run(
        harness,
        "status",
        env_extra={"TAB_C_CLI_CMD": 'cli "prefix {prompt} suffix"'},
    )
    assert proc.returncode == 2, (proc.returncode, proc.stderr)
    assert "do NOT wrap" in proc.stderr


def test_tab_quickstart_cli_cmd_two_placeholders_early_fail(harness):
    """`cli {prompt} {prompt}` -- two placeholders -> rc=2 + 'exactly once'."""
    proc = _run(
        harness,
        "status",
        env_extra={"TAB_C_CLI_CMD": "cli {prompt} {prompt}"},
    )
    assert proc.returncode == 2, (proc.returncode, proc.stderr)
    assert "exactly once" in proc.stderr


# ---------------------------------------------------------------------------
# 11. Status corrupt / partial state-file fallbacks
# ---------------------------------------------------------------------------
def test_tab_quickstart_status_corrupt_json_graceful(harness):
    """Corrupt JSON state file -> status exits 0 with a warning, no crash."""
    state_path = harness["gpfs"] / "state" / "triage_state.json"
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text("{not valid json at all", encoding="utf-8")
    proc = _run(
        harness,
        "status",
        env_extra={"TAB_STATE_PATH": str(state_path)},
    )
    assert proc.returncode == 0, (proc.stderr, proc.stdout)
    # Either "could not parse" (json decode) or the generic warning fallback.
    assert ("could not parse" in proc.stdout) or ("warning" in proc.stdout), (
        proc.stdout
    )


def test_tab_quickstart_status_missing_schema_version_graceful(harness):
    """State file with no schema_version key -> status 0, total tasks: 0."""
    state_path = harness["gpfs"] / "state" / "triage_state.json"
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(
        json.dumps({"tasks": {}}),
        encoding="utf-8",
    )
    proc = _run(
        harness,
        "status",
        env_extra={"TAB_STATE_PATH": str(state_path)},
    )
    assert proc.returncode == 0, (proc.stderr, proc.stdout)
    assert "total tasks: 0" in proc.stdout
    # The wrapper should surface that schema_version is missing.
    assert "schema_version" in proc.stdout


# ---------------------------------------------------------------------------
# 12. Quickstart markdown documentation contract
# ---------------------------------------------------------------------------
_QUICKSTART_MD = (
    _REPO_ROOT / "docs" / "release" / "internal-acceptance-quickstart.md"
)


def test_tab_quickstart_md_has_tldr_section():
    """Quickstart md must announce a TL;DR one-shot wrapper section."""
    assert _QUICKSTART_MD.is_file(), f"missing: {_QUICKSTART_MD}"
    text = _QUICKSTART_MD.read_text(encoding="utf-8")
    assert "TL;DR" in text and "一键封装" in text, (
        "quickstart.md must contain the 'TL;DR -- 一键封装' header"
    )


def test_tab_quickstart_md_lists_all_7_subcommands():
    """The TL;DR section must enumerate all 7 wrapper subcommands."""
    text = _QUICKSTART_MD.read_text(encoding="utf-8")
    # Slice from the TL;DR header up to the next ## header so we test the
    # block, not random occurrences elsewhere in the doc.
    tldr_idx = text.find("TL;DR")
    assert tldr_idx >= 0, "TL;DR header not found"
    rest = text[tldr_idx:]
    next_h = rest.find("\n## ", 1)
    tldr_block = rest[:next_h] if next_h > 0 else rest
    for cmd in ("setup", "verify-c", "resolve", "smoke", "full", "status", "clean"):
        assert cmd in tldr_block, (
            f"TL;DR block missing subcommand '{cmd}'"
        )


def test_tab_quickstart_help_lists_runs_on_per_subcommand(harness):
    """`help` output must show a 'Runs on:' marker for each of the 7 subcommands."""
    proc = _run(harness, "help", use_conf=False)
    assert proc.returncode == 0, proc.stderr
    # At least 7 lines (one per subcommand) must contain the 'Runs on:' marker.
    runs_on_lines = [ln for ln in proc.stdout.splitlines() if "Runs on:" in ln]
    assert len(runs_on_lines) >= 7, (
        f"help must annotate each of the 7 subcommands with 'Runs on:'; "
        f"got {len(runs_on_lines)} occurrence(s):\n" + "\n".join(runs_on_lines)
    )
    # Each subcommand must appear on a 'Runs on:' line so the mapping is
    # explicit rather than positional.
    for cmd in ("setup", "verify-c", "resolve", "smoke", "full", "status", "clean"):
        assert any(cmd in ln for ln in runs_on_lines), (
            f"subcommand '{cmd}' has no 'Runs on:' annotation in help"
        )


def test_tab_quickstart_help_explains_single_machine_mode(harness):
    """help text must mention the single-machine (A=B) deployment mode."""
    proc = _run(harness, "help", use_conf=False)
    assert proc.returncode == 0, proc.stderr
    text = proc.stdout
    assert ("single-machine" in text) or ("A=B" in text), (
        "help must explain that single-machine mode (A=B) runs all subcommands "
        f"on one host; got:\n{text}"
    )


def test_tab_quickstart_md_has_machine_target_table():
    """Quickstart md must include the per-command machine target table."""
    text = _QUICKSTART_MD.read_text(encoding="utf-8")
    assert "哪条命令在哪台机器跑" in text, (
        "quickstart.md must contain the '哪条命令在哪台机器跑' section heading"
    )
    # The table header must enumerate all three machines.
    assert "A(GPU 机)" in text, "machine target table must label A column"
    assert "B(CPU" in text, "machine target table must label B column"


def test_tab_quickstart_md_references_conf_example_path():
    """Quickstart md must point operators at the conf example file."""
    text = _QUICKSTART_MD.read_text(encoding="utf-8")
    assert "scripts/automation/runners/tab_quickstart.conf.example" in text, (
        "quickstart.md must reference scripts/automation/runners/"
        "tab_quickstart.conf.example so operators know where to copy from"
    )


# ---------------------------------------------------------------------------
# 13. Config file resolution priority
# ---------------------------------------------------------------------------
# The wrapper resolves the config file from three sources (high to low):
#   1. $TAB_CONFIG env var (explicit override)
#   2. <repo-root>/tab_quickstart.conf  (recommended; gitignored)
#   3. ~/.tab-quickstart.conf            (legacy fallback)
# REPO_ROOT inside the wrapper is derived as $SCRIPT_DIR/../../.., so for
# the priority tests we must mirror the real three-level layout
#   <fake_repo>/scripts/automation/runners/tab_quickstart.sh
# under each test's tmp_path. That way the wrapper's REPO_ROOT resolves
# back into the per-test tmp_path -- not into the pytest shared tmp dir
# (which would let parallel runs/leftover repo-local conf files step on
# each other).
def _make_fake_bench_repo(tmp_path: Path) -> Path:
    """Build a fake bench-repo layout that mirrors the real
    ``<repo>/scripts/automation/runners/`` location so the wrapper's
    ``REPO_ROOT="$SCRIPT_DIR/../../.."`` resolves to ``tmp_path``.

    Stubs ``setup_machine_a_gpu.sh`` and ``run_acceptance_gate.sh`` are
    also dropped beside the script so delegated subcommands have a
    no-op target if a test ever exercises them. The priority tests
    themselves only call ``status``, which does not delegate.

    Returns the fake repo root (== ``tmp_path``).
    """
    runners_dir = tmp_path / "scripts" / "automation" / "runners"
    runners_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(_REAL_SCRIPT, runners_dir / "tab_quickstart.sh")
    _make_executable(runners_dir / "tab_quickstart.sh")
    for runner in ("setup_machine_a_gpu.sh", "run_acceptance_gate.sh"):
        stub = runners_dir / runner
        stub.write_text(
            "#!/usr/bin/env bash\n"
            "echo \"[stub] $0 $*\"\n"
            "exit 0\n",
            encoding="utf-8",
        )
        _make_executable(stub)
    return tmp_path


def _fake_repo_script(fake_repo: Path) -> Path:
    """Return the tab_quickstart.sh path inside the fake-layout repo."""
    return fake_repo / "scripts" / "automation" / "runners" / "tab_quickstart.sh"


def _make_fake_home(tmp_path: Path) -> Path:
    """Sibling tmp dir for HOME so home/repo are independent.

    Living under ``tmp_path.parent`` keeps it inside the pytest tmpdir
    tree (auto-cleaned) while keeping HOME outside the fake repo root.
    """
    fake_home = tmp_path.parent / f"{tmp_path.name}_home"
    fake_home.mkdir(exist_ok=True)
    return fake_home


def _run_fake_repo(
    fake_repo: Path,
    fake_home: Path,
    *args: str,
    env_extra: dict | None = None,
) -> subprocess.CompletedProcess:
    """Invoke tab_quickstart.sh inside a fake-layout repo.

    Builds a minimal env (HOME + PATH + optional overrides) and -- unless
    the caller explicitly sets TAB_CONFIG via env_extra -- leaves
    TAB_CONFIG unset so the wrapper's repo-local / home fallback chain
    is exercised.
    """
    env = {
        "HOME": str(fake_home),
        "PATH": os.environ.get("PATH", ""),
    }
    if env_extra:
        env.update(env_extra)
    return subprocess.run(
        [_bash(), str(_fake_repo_script(fake_repo)), *args],
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
        env=env,
    )


def test_fake_bench_repo_layout_resolves_repo_root_correctly(tmp_path):
    """Sanity: fake-layout REPO_ROOT must resolve to tmp_path, not the
    pytest shared tmp dir (regression guard for the priority tests).
    """
    fake_repo = _make_fake_bench_repo(tmp_path)
    fake_home = _make_fake_home(tmp_path)
    (fake_repo / "tab_quickstart.conf").write_text(
        textwrap.dedent(
            """\
            TAB_GPFS_ROOT="/x"
            TAB_C_HOST="user@host"
            TAB_C_CLI_CMD='cli {prompt}'
            """
        ),
        encoding="utf-8",
    )
    proc = _run_fake_repo(fake_repo, fake_home, "status")
    assert proc.returncode == 0, (proc.stderr, proc.stdout)
    # CONFIG_FILE must point inside fake_repo (per-test tmp), not pytest's
    # shared /tmp/pytest-of-* root.
    expected = str(fake_repo / "tab_quickstart.conf")
    assert expected in proc.stdout, proc.stdout
    assert "[repo-local]" in proc.stdout, proc.stdout


def test_config_priority_env_overrides_repo_local(tmp_path):
    """$TAB_CONFIG env wins over <repo-root>/tab_quickstart.conf."""
    fake_repo = _make_fake_bench_repo(tmp_path)
    fake_home = _make_fake_home(tmp_path)
    repo_local = fake_repo / "tab_quickstart.conf"
    repo_local.write_text(
        textwrap.dedent(
            """\
            TAB_GPFS_ROOT="/from-repo"
            TAB_C_HOST="user@repo-host"
            TAB_C_CLI_CMD='repo-cli --print {prompt}'
            """
        ),
        encoding="utf-8",
    )
    env_conf = tmp_path / "env.conf"
    env_conf.write_text(
        textwrap.dedent(
            """\
            TAB_GPFS_ROOT="/from-env"
            TAB_C_HOST="user@env-host"
            TAB_C_CLI_CMD='env-cli --print {prompt}'
            """
        ),
        encoding="utf-8",
    )
    proc = _run_fake_repo(
        fake_repo,
        fake_home,
        "status",
        env_extra={"TAB_CONFIG": str(env_conf)},
    )
    assert proc.returncode == 0, (proc.stderr, proc.stdout)
    config_block = proc.stdout.split("== state ==", 1)[0]
    assert "/from-env" in config_block, config_block
    assert "/from-repo" not in config_block, config_block
    assert "[TAB_CONFIG env]" in config_block, config_block


def test_config_priority_repo_local_overrides_home(tmp_path):
    """<repo-root>/tab_quickstart.conf wins over ~/.tab-quickstart.conf."""
    fake_repo = _make_fake_bench_repo(tmp_path)
    fake_home = _make_fake_home(tmp_path)
    repo_local = fake_repo / "tab_quickstart.conf"
    repo_local.write_text(
        textwrap.dedent(
            """\
            TAB_GPFS_ROOT="/from-repo"
            TAB_C_HOST="user@repo-host"
            TAB_C_CLI_CMD='repo-cli --print {prompt}'
            """
        ),
        encoding="utf-8",
    )
    home_conf = fake_home / ".tab-quickstart.conf"
    home_conf.write_text(
        textwrap.dedent(
            """\
            TAB_GPFS_ROOT="/from-home"
            TAB_C_HOST="user@home-host"
            TAB_C_CLI_CMD='home-cli --print {prompt}'
            """
        ),
        encoding="utf-8",
    )
    proc = _run_fake_repo(fake_repo, fake_home, "status")
    assert proc.returncode == 0, (proc.stderr, proc.stdout)
    config_block = proc.stdout.split("== state ==", 1)[0]
    assert "/from-repo" in config_block, config_block
    assert "/from-home" not in config_block, config_block
    assert "[repo-local]" in config_block, config_block


def test_config_source_indicator_repo_local(tmp_path):
    """Status output surfaces the [repo-local] source tag."""
    fake_repo = _make_fake_bench_repo(tmp_path)
    fake_home = _make_fake_home(tmp_path)
    repo_local = fake_repo / "tab_quickstart.conf"
    repo_local.write_text(
        textwrap.dedent(
            """\
            TAB_GPFS_ROOT="/fake-gpfs"
            TAB_C_HOST="user@stub-host"
            TAB_C_CLI_CMD='stub-cli --print {prompt}'
            """
        ),
        encoding="utf-8",
    )
    proc = _run_fake_repo(fake_repo, fake_home, "status")
    assert proc.returncode == 0, (proc.stderr, proc.stdout)
    assert "[repo-local]" in proc.stdout, proc.stdout
    assert "(loaded)" in proc.stdout, proc.stdout


def test_config_source_indicator_home_legacy(tmp_path):
    """Status surfaces '[home (legacy fallback)]' when only home conf exists."""
    fake_repo = _make_fake_bench_repo(tmp_path)
    fake_home = _make_fake_home(tmp_path)
    # Per-test fake_repo is unique, so no leftover repo-local conf can leak.
    assert not (fake_repo / "tab_quickstart.conf").exists()
    home_conf = fake_home / ".tab-quickstart.conf"
    home_conf.write_text(
        textwrap.dedent(
            """\
            TAB_GPFS_ROOT="/fake-gpfs"
            TAB_C_HOST="user@stub-host"
            TAB_C_CLI_CMD='stub-cli --print {prompt}'
            """
        ),
        encoding="utf-8",
    )
    proc = _run_fake_repo(fake_repo, fake_home, "status")
    assert proc.returncode == 0, (proc.stderr, proc.stdout)
    assert "[home (legacy fallback)]" in proc.stdout, proc.stdout


def test_gitignore_contains_tab_quickstart_conf_rules():
    """Repo .gitignore must list both tab_quickstart conf patterns."""
    gitignore = (_REPO_ROOT / ".gitignore").read_text(encoding="utf-8")
    lines = gitignore.splitlines()
    # Match exact rule lines (the leading slash on the first entry pins
    # the match to repo root; the second entry covers the legacy name
    # anywhere it might be accidentally placed).
    assert "/tab_quickstart.conf" in lines, (
        ".gitignore must list '/tab_quickstart.conf' (repo-root-only)"
    )
    assert ".tab-quickstart.conf" in lines, (
        ".gitignore must list '.tab-quickstart.conf' (legacy name, any dir)"
    )


# ---------------------------------------------------------------------------
# 14. Release-gate subcommands (goldens / verify-all / release-gate / pass-all)
#
# These subcommands wrap three repo-local python producers:
#   phase_b_golden_capture.py --all
#   phase_a_prime.py --run --tier <tier>
#   validate_tasks.py --release-gate
#
# To test them in isolation we stand up a *stub* bench repo under tmp_path
# whose ``scripts/<producer>.py`` files just record argv as JSON lines.
# The wrapper picks the stub via TAB_BENCH_REPO env (set per-test); no
# real GPU / NPZ / corpus is touched.
# ---------------------------------------------------------------------------


def _stub_python_script(
    path: Path, recorder_log: Path, exit_rc: int = 0
) -> None:
    """Drop a stub python producer that records argv + exits ``exit_rc``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    body = textwrap.dedent(
        f"""\
        import json, pathlib, sys
        pathlib.Path({str(recorder_log)!r}).open("a").write(
            json.dumps({{"script": sys.argv[0], "argv": sys.argv[1:]}}) + "\\n"
        )
        sys.exit({exit_rc})
        """
    )
    path.write_text(body, encoding="utf-8")


def _stub_bench_repo(
    tmp_path: Path,
    runner_log: Path,
    *,
    phase_b_present: bool = True,
    goldens_rc: int = 0,
    verify_rc: int = 0,
    release_gate_rc: int = 0,
) -> Path:
    """Build a stub bench repo with stub python producers + at least one
    ``meta.yaml`` carrying ``phase_b_capture_required: true`` so the
    goldens sanity grep finds something to capture (unless
    ``phase_b_present=False`` -- the warn-when-no-tasks path).

    Returns the stub repo root; tests point TAB_BENCH_REPO at it.
    """
    repo = tmp_path / "stub_bench"
    (repo / "scripts").mkdir(parents=True, exist_ok=True)
    _stub_python_script(
        repo / "scripts" / "phase_b_golden_capture.py",
        runner_log,
        exit_rc=goldens_rc,
    )
    _stub_python_script(
        repo / "scripts" / "phase_a_prime.py",
        runner_log,
        exit_rc=verify_rc,
    )
    _stub_python_script(
        repo / "scripts" / "validate_tasks.py",
        runner_log,
        exit_rc=release_gate_rc,
    )

    # Plant a meta.yaml the cmd_goldens sanity grep can find.
    if phase_b_present:
        meta_dir = repo / "tasks" / "level1" / "L1_stub_task"
        meta_dir.mkdir(parents=True, exist_ok=True)
        (meta_dir / "meta.yaml").write_text(
            "phase_b_capture_required: true\n",
            encoding="utf-8",
        )

    return repo


# --- 14a. help / status surface checks for the 4 new subcommands -----------


def test_goldens_subcommand_listed_in_help(harness):
    """`help` must enumerate the new ``goldens`` subcommand."""
    proc = _run(harness, "help", use_conf=False)
    assert proc.returncode == 0, proc.stderr
    assert "goldens" in proc.stdout, (
        "help must list 'goldens' subcommand"
    )
    # Anchored on a 'Runs on:' line so the machine target is explicit.
    runs_on_lines = [
        ln for ln in proc.stdout.splitlines() if "Runs on:" in ln
    ]
    assert any("goldens" in ln for ln in runs_on_lines), (
        "'goldens' must appear on a 'Runs on:' line in help"
    )


def test_verify_all_subcommand_listed_in_help(harness):
    """`help` must enumerate the new ``verify-all`` subcommand."""
    proc = _run(harness, "help", use_conf=False)
    assert proc.returncode == 0, proc.stderr
    assert "verify-all" in proc.stdout
    runs_on_lines = [
        ln for ln in proc.stdout.splitlines() if "Runs on:" in ln
    ]
    assert any("verify-all" in ln for ln in runs_on_lines), (
        "'verify-all' must appear on a 'Runs on:' line in help"
    )


def test_release_gate_subcommand_listed_in_help(harness):
    """`help` must enumerate the new ``release-gate`` subcommand."""
    proc = _run(harness, "help", use_conf=False)
    assert proc.returncode == 0, proc.stderr
    assert "release-gate" in proc.stdout
    runs_on_lines = [
        ln for ln in proc.stdout.splitlines() if "Runs on:" in ln
    ]
    assert any("release-gate" in ln for ln in runs_on_lines), (
        "'release-gate' must appear on a 'Runs on:' line in help"
    )


def test_pass_all_subcommand_listed_in_help(harness):
    """`help` must enumerate the new ``pass-all`` subcommand."""
    proc = _run(harness, "help", use_conf=False)
    assert proc.returncode == 0, proc.stderr
    assert "pass-all" in proc.stdout
    runs_on_lines = [
        ln for ln in proc.stdout.splitlines() if "Runs on:" in ln
    ]
    assert any("pass-all" in ln for ln in runs_on_lines), (
        "'pass-all' must appear on a 'Runs on:' line in help"
    )


def test_help_lists_runs_on_per_subcommand_includes_new_four(harness):
    """At least 11 lines (7 existing + 4 new) carry the 'Runs on:' marker."""
    proc = _run(harness, "help", use_conf=False)
    assert proc.returncode == 0, proc.stderr
    runs_on_lines = [
        ln for ln in proc.stdout.splitlines() if "Runs on:" in ln
    ]
    assert len(runs_on_lines) >= 11, (
        f"help must annotate all 11 subcommands with 'Runs on:'; "
        f"got {len(runs_on_lines)}:\n" + "\n".join(runs_on_lines)
    )


# --- 14b. behaviour: each subcommand actually invokes its producer ---------


def test_goldens_calls_phase_b_golden_capture(harness, tmp_path):
    """`goldens` must invoke ``scripts/phase_b_golden_capture.py --all``."""
    runner_log = harness["runner_log"]
    repo = _stub_bench_repo(tmp_path, runner_log)
    proc = _run(harness, "goldens", env_extra={"TAB_BENCH_REPO": str(repo)})
    assert proc.returncode == 0, (proc.stderr, proc.stdout)
    rows = _read_argv_log(runner_log)
    assert len(rows) == 1, f"expected one producer invocation, got: {rows}"
    assert rows[0]["script"].endswith("phase_b_golden_capture.py"), rows[0]
    assert rows[0]["argv"] == ["--all"], rows[0]


def test_goldens_forwards_dry_run_arg(harness, tmp_path):
    """Extra args after `goldens` must forward verbatim to the producer."""
    runner_log = harness["runner_log"]
    repo = _stub_bench_repo(tmp_path, runner_log)
    proc = _run(
        harness, "goldens", "--dry-run",
        env_extra={"TAB_BENCH_REPO": str(repo)},
    )
    assert proc.returncode == 0, (proc.stderr, proc.stdout)
    rows = _read_argv_log(runner_log)
    assert len(rows) == 1
    assert rows[0]["argv"] == ["--all", "--dry-run"], rows[0]


def test_verify_all_calls_phase_a_prime(harness, tmp_path):
    """`verify-all` must invoke ``scripts/phase_a_prime.py --run --tier all``."""
    runner_log = harness["runner_log"]
    repo = _stub_bench_repo(tmp_path, runner_log)
    proc = _run(harness, "verify-all", env_extra={"TAB_BENCH_REPO": str(repo)})
    assert proc.returncode == 0, (proc.stderr, proc.stdout)
    rows = _read_argv_log(runner_log)
    assert len(rows) == 1, rows
    assert rows[0]["script"].endswith("phase_a_prime.py"), rows[0]
    assert rows[0]["argv"] == ["--run", "--tier", "all"], rows[0]


def test_verify_all_accepts_explicit_tier(harness, tmp_path):
    """`verify-all T2` must forward ``--tier T2`` to phase_a_prime."""
    runner_log = harness["runner_log"]
    repo = _stub_bench_repo(tmp_path, runner_log)
    proc = _run(
        harness, "verify-all", "T2",
        env_extra={"TAB_BENCH_REPO": str(repo)},
    )
    assert proc.returncode == 0, (proc.stderr, proc.stdout)
    rows = _read_argv_log(runner_log)
    assert len(rows) == 1
    assert rows[0]["argv"] == ["--run", "--tier", "T2"], rows[0]


def test_verify_all_invalid_tier_fails_loud(harness, tmp_path):
    """`verify-all T9` must fatal-exit (rc=2) with a clear message
    and must NOT invoke phase_a_prime.py.
    """
    runner_log = harness["runner_log"]
    repo = _stub_bench_repo(tmp_path, runner_log)
    proc = _run(
        harness, "verify-all", "T9",
        env_extra={"TAB_BENCH_REPO": str(repo)},
    )
    assert proc.returncode == 2, (proc.returncode, proc.stderr)
    assert "T0|T1|T2|T3|all" in proc.stderr
    # And the producer must NOT have been invoked.
    assert _read_argv_log(runner_log) == []


def test_verify_all_rejects_extra_args(harness, tmp_path):
    """`verify-all T2 --dry-run` must fatal (rc=2) instead of silently
    dropping the extra arg. Guards a real footgun: phase_a_prime.py has
    no `--dry-run`, so a silent drop would let an operator think they
    were dry-running when in fact the producer is running for real.
    """
    runner_log = harness["runner_log"]
    repo = _stub_bench_repo(tmp_path, runner_log)
    proc = _run(
        harness, "verify-all", "T2", "--dry-run",
        env_extra={"TAB_BENCH_REPO": str(repo)},
    )
    assert proc.returncode == 2, (proc.returncode, proc.stderr)
    assert "unexpected extra args" in proc.stderr
    assert _read_argv_log(runner_log) == []


def test_release_gate_calls_validate_tasks(harness, tmp_path):
    """`release-gate` must invoke ``validate_tasks.py --release-gate``."""
    runner_log = harness["runner_log"]
    repo = _stub_bench_repo(tmp_path, runner_log)
    proc = _run(
        harness, "release-gate",
        env_extra={"TAB_BENCH_REPO": str(repo)},
    )
    assert proc.returncode == 0, (proc.stderr, proc.stdout)
    rows = _read_argv_log(runner_log)
    assert len(rows) == 1, rows
    assert rows[0]["script"].endswith("validate_tasks.py"), rows[0]
    assert rows[0]["argv"] == ["--release-gate"], rows[0]


def test_release_gate_failure_propagates(harness, tmp_path):
    """A non-zero validate_tasks exit must surface as a `[fail]` rc=1."""
    runner_log = harness["runner_log"]
    repo = _stub_bench_repo(tmp_path, runner_log, release_gate_rc=3)
    proc = _run(
        harness, "release-gate",
        env_extra={"TAB_BENCH_REPO": str(repo)},
    )
    assert proc.returncode == 1, (proc.returncode, proc.stderr, proc.stdout)
    assert "release-gate FAIL" in proc.stderr


def test_release_gate_rejects_extra_args(harness, tmp_path):
    """`release-gate --foo` must fatal (rc=2); the subcommand takes no
    arguments and a silent drop would mask a typo.
    """
    runner_log = harness["runner_log"]
    repo = _stub_bench_repo(tmp_path, runner_log)
    proc = _run(
        harness, "release-gate", "--foo",
        env_extra={"TAB_BENCH_REPO": str(repo)},
    )
    assert proc.returncode == 2, (proc.returncode, proc.stderr)
    assert "unexpected extra args" in proc.stderr
    assert _read_argv_log(runner_log) == []


# --- 14c. pass-all chain + fail-fast ----------------------------------------


def test_pass_all_chains_three_steps(harness, tmp_path):
    """`pass-all` must invoke the three producers in order: goldens ->
    verify-all -> release-gate.
    """
    runner_log = harness["runner_log"]
    repo = _stub_bench_repo(tmp_path, runner_log)
    proc = _run(harness, "pass-all", env_extra={"TAB_BENCH_REPO": str(repo)})
    assert proc.returncode == 0, (proc.stderr, proc.stdout)
    rows = _read_argv_log(runner_log)
    assert len(rows) == 3, f"expected 3 producer calls, got: {rows}"
    # Order must be goldens -> verify-all -> release-gate.
    assert rows[0]["script"].endswith("phase_b_golden_capture.py"), rows[0]
    assert rows[1]["script"].endswith("phase_a_prime.py"), rows[1]
    assert rows[2]["script"].endswith("validate_tasks.py"), rows[2]
    # And each got the expected argv.
    assert rows[0]["argv"] == ["--all"], rows[0]
    assert rows[1]["argv"] == ["--run", "--tier", "all"], rows[1]
    assert rows[2]["argv"] == ["--release-gate"], rows[2]


def test_pass_all_fail_fast_stops_after_goldens_failure(harness, tmp_path):
    """If `goldens` fails, `pass-all` must NOT invoke verify-all or
    release-gate. Verifies the fail-fast contract.
    """
    runner_log = harness["runner_log"]
    repo = _stub_bench_repo(tmp_path, runner_log, goldens_rc=7)
    proc = _run(harness, "pass-all", env_extra={"TAB_BENCH_REPO": str(repo)})
    # set -e + the `fail` helper turn a goldens failure into rc=1 here.
    assert proc.returncode == 1, (proc.returncode, proc.stderr, proc.stdout)
    rows = _read_argv_log(runner_log)
    # Exactly one producer invocation (the failed goldens); no follow-ons.
    assert len(rows) == 1, f"fail-fast violated, got: {rows}"
    assert rows[0]["script"].endswith("phase_b_golden_capture.py"), rows[0]


def test_pass_all_forwards_dry_run_to_goldens_only(harness, tmp_path):
    """Args after `pass-all` forward to goldens; verify-all + release-gate
    get no extra args even though pass-all received them.
    """
    runner_log = harness["runner_log"]
    repo = _stub_bench_repo(tmp_path, runner_log)
    proc = _run(
        harness, "pass-all", "--dry-run",
        env_extra={"TAB_BENCH_REPO": str(repo)},
    )
    assert proc.returncode == 0, (proc.stderr, proc.stdout)
    rows = _read_argv_log(runner_log)
    assert len(rows) == 3
    assert rows[0]["argv"] == ["--all", "--dry-run"], rows[0]
    assert rows[1]["argv"] == ["--run", "--tier", "all"], rows[1]
    assert rows[2]["argv"] == ["--release-gate"], rows[2]


# --- 14d. log file landing --------------------------------------------------


def test_goldens_writes_log_under_gpfs_logs(harness, tmp_path):
    """`goldens` must write its log to $TAB_GPFS_ROOT/logs/phase_b_golden_*.log."""
    runner_log = harness["runner_log"]
    repo = _stub_bench_repo(tmp_path, runner_log)
    proc = _run(harness, "goldens", env_extra={"TAB_BENCH_REPO": str(repo)})
    assert proc.returncode == 0, (proc.stderr, proc.stdout)
    logs = list((harness["gpfs"] / "logs").glob("phase_b_golden_*.log"))
    assert len(logs) == 1, f"expected one phase_b_golden log file, got: {logs}"


def test_verify_all_writes_log_under_gpfs_logs(harness, tmp_path):
    """`verify-all` must write its log to $TAB_GPFS_ROOT/logs/phase_a_verify_*.log."""
    runner_log = harness["runner_log"]
    repo = _stub_bench_repo(tmp_path, runner_log)
    proc = _run(harness, "verify-all", env_extra={"TAB_BENCH_REPO": str(repo)})
    assert proc.returncode == 0, (proc.stderr, proc.stdout)
    logs = list((harness["gpfs"] / "logs").glob("phase_a_verify_*.log"))
    assert len(logs) == 1, f"expected one phase_a_verify log file, got: {logs}"


def test_release_gate_writes_log_under_gpfs_logs(harness, tmp_path):
    """`release-gate` must write its log to $TAB_GPFS_ROOT/logs/release_gate_*.log."""
    runner_log = harness["runner_log"]
    repo = _stub_bench_repo(tmp_path, runner_log)
    proc = _run(
        harness, "release-gate",
        env_extra={"TAB_BENCH_REPO": str(repo)},
    )
    assert proc.returncode == 0, (proc.stderr, proc.stdout)
    logs = list((harness["gpfs"] / "logs").glob("release_gate_*.log"))
    assert len(logs) == 1, f"expected one release_gate log file, got: {logs}"


# --- 14e. goldens warn-when-no-phase-b-required-tasks -----------------------


def test_goldens_warns_when_no_phase_b_required_tasks(harness, tmp_path):
    """If no task has ``phase_b_capture_required: true``, `goldens` must
    warn and return rc=0 without invoking the producer.
    """
    runner_log = harness["runner_log"]
    # phase_b_present=False -> no meta.yaml plants the required field.
    repo = _stub_bench_repo(tmp_path, runner_log, phase_b_present=False)
    proc = _run(harness, "goldens", env_extra={"TAB_BENCH_REPO": str(repo)})
    assert proc.returncode == 0, (proc.stderr, proc.stdout)
    # The warn line must surface to the operator. step / warn both go to
    # stdout in the wrapper, so check the combined stream.
    combined = proc.stdout + proc.stderr
    assert "nothing to capture" in combined, combined
    # And the producer must NOT have been invoked.
    assert _read_argv_log(runner_log) == [], (
        "phase_b_golden_capture.py must NOT run when zero tasks need capture"
    )


# --- 14f. status surfaces release-gate artefacts ----------------------------


def test_status_reports_release_gate_artefacts(harness, tmp_path):
    """`status` must surface the new release-gate artefacts block:
    phase B captured tally, phase A' status report, latest release-gate log.
    """
    runner_log = harness["runner_log"]
    repo = _stub_bench_repo(tmp_path, runner_log)
    # Plant a stale release_gate log + a phase_a_prime_status.md to give
    # status real lookup targets.
    (harness["gpfs"] / "logs").mkdir(parents=True, exist_ok=True)
    (harness["gpfs"] / "logs" / "release_gate_20260101_120000.log").write_text(
        "stale\n", encoding="utf-8"
    )
    (repo / "docs" / "plans").mkdir(parents=True, exist_ok=True)
    (repo / "docs" / "plans" / "phase_a_prime_status.md").write_text(
        "stub status\n", encoding="utf-8"
    )

    proc = _run(harness, "status", env_extra={"TAB_BENCH_REPO": str(repo)})
    assert proc.returncode == 0, (proc.stderr, proc.stdout)
    assert "release-gate artefacts" in proc.stdout
    assert "phase B golden tensors" in proc.stdout
    assert "phase_a_prime_status.md" in proc.stdout
    assert "release_gate_20260101_120000.log" in proc.stdout


def test_status_release_gate_block_handles_cold_start_repo(harness, tmp_path):
    """On a cold-start repo with no $TAB_GPFS_ROOT/logs dir, the
    release-gate artefacts block must still emit the 'latest release-gate log'
    line (with the friendly '(none ...)' placeholder), so the status report
    keeps a fixed shape across rehearsal stages.
    """
    runner_log = harness["runner_log"]
    repo = _stub_bench_repo(tmp_path, runner_log)
    # Important: do NOT create the harness logs/ directory -- this is the
    # cold-start path. The harness fixture only creates gpfs/ itself.
    assert not (harness["gpfs"] / "logs").exists(), (
        "harness fixture must not pre-create gpfs/logs for this test"
    )

    proc = _run(harness, "status", env_extra={"TAB_BENCH_REPO": str(repo)})
    assert proc.returncode == 0, (proc.stderr, proc.stdout)
    assert "release-gate artefacts" in proc.stdout
    # The phase B + phase A' lines must still render, and the latest-log
    # line must always appear (with the friendly placeholder when missing).
    assert "phase B golden tensors" in proc.stdout
    assert "latest release-gate log" in proc.stdout, (
        "status must always emit the 'latest release-gate log' line "
        "even on cold-start repos with no logs dir"
    )
    assert "(none -- run 'release-gate' or 'pass-all')" in proc.stdout
