"""Smoke + lint tests for scripts/automation/runners/*.sh.

These tests guard the three-machine runner scripts and the INTERNAL
acceptance runbook. They are intentionally fast and do not require any
GPU / GPFS / SSH infrastructure -- they only verify that the scripts
are syntactically valid bash, expose a --help line, contain no
process-noise tokens, and live under paths matched by the release
boundary in .gitattributes.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_RUNNERS_DIR = _REPO_ROOT / "scripts" / "automation" / "runners"
_RUNBOOK = _REPO_ROOT / "docs" / "release" / "internal-acceptance-runbook.md"

_BASH_SCRIPTS = [
    _RUNNERS_DIR / "setup_machine_a_gpu.sh",
    _RUNNERS_DIR / "setup_machine_b_cpu.sh",
    _RUNNERS_DIR / "run_machine_a_daemon.sh",
    _RUNNERS_DIR / "run_machine_b_orchestrator.sh",
    _RUNNERS_DIR / "run_acceptance_gate.sh",
]


def _bash_executable() -> str:
    path = shutil.which("bash")
    if path is None:  # pragma: no cover - sandbox without bash should fail loud
        pytest.skip("bash not available on this host")
    return path


def test_all_5_bash_scripts_exist():
    """All five runner scripts must exist and have the executable bit set."""
    for script in _BASH_SCRIPTS:
        assert script.is_file(), f"missing runner script: {script}"
        mode = script.stat().st_mode
        assert mode & 0o111, f"{script} must be chmod +x (mode={oct(mode)})"


@pytest.mark.parametrize("script", _BASH_SCRIPTS, ids=lambda p: p.name)
def test_bash_syntax_valid(script: Path):
    """`bash -n` must accept every runner script."""
    bash = _bash_executable()
    result = subprocess.run(
        [bash, "-n", str(script)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, (
        f"bash -n failed for {script.name}: rc={result.returncode}\n"
        f"stderr={result.stderr}"
    )


def _run_help(script: Path) -> subprocess.CompletedProcess[str]:
    bash = _bash_executable()
    return subprocess.run(
        [bash, str(script), "--help"],
        capture_output=True,
        text=True,
        check=False,
        timeout=10,
        env={**os.environ, "PATH": os.environ.get("PATH", "")},
    )


def test_setup_machine_a_help_smoke():
    proc = _run_help(_RUNNERS_DIR / "setup_machine_a_gpu.sh")
    assert proc.returncode == 0, f"--help should exit 0, got {proc.returncode}: {proc.stderr}"
    assert "Usage:" in proc.stdout, "expected Usage: line in --help output"
    assert "--gpfs-root" in proc.stdout, "expected --gpfs-root flag documented"


def test_setup_machine_b_help_smoke():
    proc = _run_help(_RUNNERS_DIR / "setup_machine_b_cpu.sh")
    assert proc.returncode == 0, f"--help should exit 0, got {proc.returncode}: {proc.stderr}"
    assert "Usage:" in proc.stdout
    assert "--a-host" in proc.stdout
    assert "--c-host" in proc.stdout


def test_run_daemon_help_smoke():
    proc = _run_help(_RUNNERS_DIR / "run_machine_a_daemon.sh")
    assert proc.returncode == 0, f"--help should exit 0, got {proc.returncode}: {proc.stderr}"
    assert "Usage:" in proc.stdout
    assert "--lease-seconds" in proc.stdout


def test_run_orchestrator_help_smoke():
    proc = _run_help(_RUNNERS_DIR / "run_machine_b_orchestrator.sh")
    assert proc.returncode == 0, f"--help should exit 0, got {proc.returncode}: {proc.stderr}"
    assert "Usage:" in proc.stdout
    assert "--local-port" in proc.stdout
    assert "--apply" in proc.stdout


def test_run_acceptance_gate_help_smoke():
    proc = _run_help(_RUNNERS_DIR / "run_acceptance_gate.sh")
    assert proc.returncode == 0, f"--help should exit 0, got {proc.returncode}: {proc.stderr}"
    assert "Usage:" in proc.stdout
    assert "--phase" in proc.stdout


def test_runbook_exists_and_contains_phases():
    """The INTERNAL acceptance runbook must exist and cover all 4 phases."""
    assert _RUNBOOK.is_file(), f"missing runbook: {_RUNBOOK}"
    text = _RUNBOOK.read_text(encoding="utf-8")
    for needle in ("Phase 1", "Phase 2", "Phase 3", "Phase 4"):
        assert needle in text, f"runbook missing '{needle}' heading"
    # Sanity: top-level INTERNAL marker so reviewers spot the boundary.
    assert "INTERNAL" in text, "runbook must declare itself INTERNAL"


def test_runner_scripts_in_export_ignore_scope():
    """All runner scripts + runbook must fall under .gitattributes
    export-ignore patterns so they never ship in the release tarball."""
    gitattrs = (_REPO_ROOT / ".gitattributes").read_text(encoding="utf-8")
    # The automation tree is excluded as a directory pattern; the runbook
    # is excluded by the internal-*.md glob under docs/release/.
    assert "scripts/automation/" in gitattrs, (
        "scripts/automation/ must be export-ignored in .gitattributes"
    )
    assert "docs/release/internal-*.md" in gitattrs, (
        "docs/release/internal-*.md must be export-ignored in .gitattributes"
    )
    # Sanity: every runner script lives under scripts/automation/.
    for script in _BASH_SCRIPTS:
        rel = script.relative_to(_REPO_ROOT)
        assert str(rel).startswith("scripts/automation/"), (
            f"{rel} not covered by the scripts/automation/ export-ignore pattern"
        )
    # And the runbook is matched by internal-*.md under docs/release/.
    runbook_rel = _RUNBOOK.relative_to(_REPO_ROOT)
    assert str(runbook_rel).startswith("docs/release/internal-"), (
        f"{runbook_rel} not covered by docs/release/internal-*.md"
    )


# CJK Unified Ideographs basic block U+4E00..U+9FFF, expressed as \u
# escapes so this test file itself is pure ASCII (and therefore exempt
# from its own no-Chinese assertion, which would otherwise self-match
# on a literal CJK character class).
_CHINESE_RANGE = re.compile("[\u4e00-\u9fff]")


@pytest.mark.parametrize(
    "path",
    [*_BASH_SCRIPTS, _RUNBOOK],
    ids=lambda p: p.name,
)
def test_runner_scripts_no_chinese(path: Path):
    """Runner scripts and the INTERNAL runbook must be English-only."""
    text = path.read_text(encoding="utf-8")
    matches = _CHINESE_RANGE.findall(text)
    assert not matches, f"{path.name}: found {len(matches)} CJK chars; runner artifacts are English-only"


# Process-noise tokens that must not appear in shipped artifacts.
# Allowed exceptions: api version paths like /v1/, hardware names like
# V100/A100, the literal acceptance "Phase N" headings, and the legacy
# "v1/chat/completions" OpenAI path.
_NOISE_PATTERNS = [
    re.compile(r"\bcodex\b", re.IGNORECASE),
    re.compile(r"\bclaude\b", re.IGNORECASE),
    re.compile(r"\bgemini\b", re.IGNORECASE),
    re.compile(r"\bsanity findings\b", re.IGNORECASE),
    re.compile(r"\bself-review\b", re.IGNORECASE),
    re.compile(r"\bRound-\d+\b"),
    re.compile(r"\bround[-_ ]?\d+\b", re.IGNORECASE),
    # Internal milestone tags like M1.11 / M2.9 are noise when referenced
    # outside of the plan doc; the runbook talks about Phase N (1-4),
    # never milestone labels.
    re.compile(r"\bM\d+\.\d+\b"),
]


@pytest.mark.parametrize(
    "path",
    [*_BASH_SCRIPTS, _RUNBOOK],
    ids=lambda p: p.name,
)
def test_runner_scripts_no_process_noise(path: Path):
    """Runner artifacts must not contain process / agent / round tokens."""
    text = path.read_text(encoding="utf-8")
    hits: list[str] = []
    for pat in _NOISE_PATTERNS:
        for m in pat.finditer(text):
            hits.append(f"{pat.pattern} -> {m.group(0)!r}")
    assert not hits, f"{path.name}: process-noise hits:\n  " + "\n  ".join(hits)


# ---------------------------------------------------------------------------
# Behavior tests
#
# These tests verify the *runner contracts* that earlier smoke tests miss:
# canonical state-field naming, dated artefact filenames, per-phase state
# isolation, single tunnel lifecycle, flock-based daemon serialization,
# resolve-log preflight, and so on. Each test inspects the script source
# (the runners are bash, not python, and bash unit-testing with PID-level
# fidelity is more trouble than it's worth) plus tiny harness scripts that
# exercise the relevant bash blocks in isolation -- no real GPFS / SSH.
# ---------------------------------------------------------------------------


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_acceptance_gate_state_field_name_correct():
    """Phase 1/2/4 must compare canonical ``state`` field, not ``status``.

    Canonical field is ``state`` per
    scripts/automation/state_machine.py COMMON_FIELDS. The orchestrator
    never writes a ``status`` key on task entries; the prior runner draft
    would have produced ``DONE count=0`` every time.
    """
    text = _read(_RUNNERS_DIR / "run_acceptance_gate.sh")
    # The runner must read t.get("state") in the embedded python helpers,
    # never t.get("status"). Allow the substring inside a code-comment
    # that mentions the legacy field, but the executable python helper
    # body must use "state".
    assert 't.get("state")' in text or 't.get(\\"state\\")' in text or '.get("state")' in text, (
        "acceptance gate must use canonical 'state' field"
    )
    # Hard ban on t.get("status") in any executable python helper.
    # (The helper bodies are inside python3 -<<'PY' here-docs.)
    forbidden = re.findall(r't\.get\(["\']status["\']\)', text)
    assert not forbidden, (
        f"acceptance gate references legacy 'status' field: {forbidden!r}"
    )


def test_acceptance_gate_dated_artifact_filenames():
    """Phase 3 must check ``backlog-report-<ISO_DATE>.{md,json}`` and
    ``summary-metrics-<ISO_DATE>.{md,json}``, NOT the undated form.

    Producers (backlog_report.py:536-548, summary_metrics.py:537-538)
    only ever write the dated form; the undated form will always be
    missing.
    """
    text = _read(_RUNNERS_DIR / "run_acceptance_gate.sh")
    assert "EXPECT_DATE=" in text, "must capture EXPECT_DATE for filename match"
    assert "backlog-report-$EXPECT_DATE.md" in text
    assert "backlog-report-$EXPECT_DATE.json" in text
    assert "summary-metrics-$EXPECT_DATE.md" in text
    assert "summary-metrics-$EXPECT_DATE.json" in text
    # The runner must also pass --date-iso to the writers so the
    # date in the filename agrees with what the runner checks for.
    assert "--date-iso \"$EXPECT_DATE\"" in text, (
        "runner must pin date with --date-iso so the dated filename matches"
    )
    # And the legacy undated check must be gone.
    bad_patterns = [
        '"$TRIAGE_OUT/backlog-report.md"',
        '"$TRIAGE_OUT/backlog-report.json"',
        '"$TRIAGE_OUT/summary-metrics.md"',
        '"$TRIAGE_OUT/summary-metrics.json"',
    ]
    for bad in bad_patterns:
        assert bad not in text, f"acceptance gate still references undated artefact: {bad}"


def test_acceptance_gate_phase4_uses_worktree_isolation():
    """Phase 4 must run Tier 2 in a `git worktree`, not in the primary
    checkout. Sharing the task tree would let Tier 1 + Tier 2 stomp on
    each other's meta.yaml mutations and pollute the equivalence diff.
    """
    text = _read(_RUNNERS_DIR / "run_acceptance_gate.sh")
    assert "git worktree add" in text, (
        "Phase 4 must isolate Tier 2 in a git worktree to avoid meta.yaml pollution"
    )
    assert "git worktree remove" in text, (
        "Phase 4 must clean up the worktree on EXIT"
    )
    # Per-tier state roots and results roots must also differ.
    assert "state-phase4-tier1" in text
    assert "state-phase4-tier2" in text
    assert "phase4-tier1" in text and "phase4-tier2" in text
    # Tier 2 must pass --repo-root pointing at the worktree.
    assert "--repo-root \"$PHASE4_WORKTREE\"" in text


def test_acceptance_gate_per_phase_state_roots():
    """Phases 1, 2, 3 must each use their own state root so phase N's
    DONE count cannot bleed into phase N+1's verdict.
    """
    text = _read(_RUNNERS_DIR / "run_acceptance_gate.sh")
    for phase in ("phase1", "phase2", "phase3"):
        state_var = f"$GPFS_ROOT/state-{phase}"
        assert state_var in text, (
            f"acceptance gate must allocate a dedicated state root for {phase}"
        )
    # Init the state fresh (--init-empty-state) so a phase never inherits
    # an unrelated prior triage_state.json.
    assert text.count("--init-empty-state") >= 4, (
        "every phase must pass --init-empty-state to start clean"
    )


def test_acceptance_gate_single_tunnel_lifecycle():
    """The whole acceptance gate runs under one SSH tunnel.

    Earlier drafts called ``start_tunnel`` inside every phase, leaking
    one SSH process per phase. We verify the helper is called exactly
    once and only from ``main`` (or top-level after main).
    """
    text = _read(_RUNNERS_DIR / "run_acceptance_gate.sh")
    # `start_tunnel` definition (once) + one invocation in main (once).
    # No invocation inside any phase{1,2,3,4} function body.
    invocation_lines = [
        ln for ln in text.splitlines()
        if ln.strip().startswith("start_tunnel")
        and "start_tunnel()" not in ln
    ]
    assert len(invocation_lines) == 1, (
        f"start_tunnel must be invoked exactly once (in main); "
        f"got {len(invocation_lines)} invocations: {invocation_lines}"
    )
    # And the EXIT trap must close the tunnel.
    assert "trap cleanup SIGTERM SIGINT EXIT" in text, (
        "acceptance gate must install an EXIT trap to close the tunnel"
    )
    assert 'kill -TERM "$TUNNEL_PID"' in text


def test_acceptance_gate_resolve_log_preflight():
    """Empty / missing --resolve-log must fail before opening the
    tunnel, mirroring the orchestrator runner."""
    text = _read(_RUNNERS_DIR / "run_acceptance_gate.sh")
    assert '[[ ! -s "$RESOLVE_LOG" ]]' in text
    # The preflight check must appear before start_tunnel is invoked.
    pre_idx = text.find('[[ ! -s "$RESOLVE_LOG" ]]')
    invoke_idx = text.find("\nmain\n")
    assert pre_idx > 0 and invoke_idx > 0 and pre_idx < invoke_idx, (
        "resolve-log preflight must run before main() (and thus before the tunnel)"
    )


def test_orchestrator_runner_health_path_configurable_and_4xx_ok():
    """The orchestrator runner must default --health-path to /v1/models
    and accept any HTTP 2xx/3xx/4xx response as 'tunnel alive'."""
    text = _read(_RUNNERS_DIR / "run_machine_b_orchestrator.sh")
    assert 'HEALTH_PATH="/v1/models"' in text
    assert "--health-path" in text
    # Probe must use curl -w '%{http_code}', not the old `curl -sf`
    # (which incorrectly treats 4xx as fatal).
    assert "%{http_code}" in text
    assert 'curl -sf -o /dev/null --max-time 2 "$LLM_ENDPOINT/health"' not in text, (
        "must not use the legacy /health endpoint + -sf probe"
    )
    # Acceptance: any 2xx/3xx/4xx counts as alive.
    assert "code >= 200 && code < 500" in text


def test_orchestrator_runner_supervises_both_children():
    """Orchestrator runner must run the orchestrator in the background
    and watch both tunnel + orchestrator; either process dying triggers
    SIGTERM to the other.
    """
    text = _read(_RUNNERS_DIR / "run_machine_b_orchestrator.sh")
    # Orchestrator is backgrounded and PID is captured.
    assert 'ORCH_PID=$!' in text
    assert 'TUNNEL_PID=$!' in text
    # Supervisor loop on both PIDs.
    assert 'while kill -0 "$ORCH_PID" 2>/dev/null && kill -0 "$TUNNEL_PID" 2>/dev/null' in text
    # Trap kills both children, not just the tunnel.
    assert 'kill -TERM "$ORCH_PID"' in text
    assert 'kill -TERM "$TUNNEL_PID"' in text


def test_orchestrator_runner_resolve_log_preflight():
    """`--resolve-log` missing or empty must fail with rc=2 before any
    SSH or python is launched.
    """
    text = _read(_RUNNERS_DIR / "run_machine_b_orchestrator.sh")
    assert '[[ ! -s "$RESOLVE_LOG" ]]' in text
    # Preflight must appear before start_tunnel.
    pre_idx = text.find('[[ ! -s "$RESOLVE_LOG" ]]')
    tunnel_idx = text.find("start_tunnel\n")
    assert pre_idx > 0 and tunnel_idx > 0 and pre_idx < tunnel_idx


def test_daemon_runner_uses_flock_for_serialization():
    """Daemon runner must acquire an exclusive `flock` on the per-host
    lock file, so two concurrent invocations on the same host cannot
    both pass the single-daemon-per-host check.
    """
    text = _read(_RUNNERS_DIR / "run_machine_a_daemon.sh")
    assert "flock" in text, "daemon runner must use flock(1)"
    assert 'exec 9>"$HOST_LOCK"' in text, (
        "daemon runner must hold flock via fd 9 for the script lifetime"
    )
    assert "flock -n 9" in text, (
        "daemon runner must use non-blocking flock so the second invocation fails fast"
    )
    # And cleanup must not unconditionally remove the PID file.
    assert '$on_disk" == "$DAEMON_PID"' in text or 'on_disk" == "$DAEMON_PID"' in text, (
        "cleanup must only remove the PID file when it still records our own PID"
    )


def test_daemon_runner_flock_serializes_concurrent_invocations(tmp_path: Path):
    """End-to-end: two concurrent invocations on the same `flock` target
    cannot both proceed past the lock. This exercises the flock
    primitive directly (we do not start real daemons or touch GPFS)."""
    flock_bin = shutil.which("flock")
    if flock_bin is None:
        pytest.skip("flock(1) not installed on this host")
    lock_file = tmp_path / "host.lock"
    # First holder takes the lock for 1.5s.
    holder = subprocess.Popen(
        [flock_bin, "-x", str(lock_file), "sleep", "1.5"],
    )
    try:
        # Second invocation must fail immediately with non-zero rc.
        contender = subprocess.run(
            [flock_bin, "-n", str(lock_file), "true"],
            capture_output=True,
            timeout=5,
        )
        assert contender.returncode != 0, (
            f"second flock must fail while holder is alive, rc={contender.returncode}"
        )
    finally:
        holder.wait(timeout=5)
    # After holder exits, the lock is free again.
    third = subprocess.run(
        [flock_bin, "-n", str(lock_file), "true"],
        capture_output=True,
        timeout=5,
    )
    assert third.returncode == 0, (
        f"flock must be reacquirable after holder exits, rc={third.returncode}"
    )


def test_acceptance_gate_threads_budget_env_to_orchestrator():
    """Each phase must pass BENCH_AUTOMATION_BUDGET_JSON to its
    orchestrator invocation, so the LLM cost stanza in the report has
    real data to read (NICE-TO-HAVE 14).
    """
    text = _read(_RUNNERS_DIR / "run_acceptance_gate.sh")
    # 5 invocations: phase 1, phase 2, phase 3, phase 4 tier 1, phase 4 tier 2.
    count = text.count("BENCH_AUTOMATION_BUDGET_JSON=")
    assert count >= 5, (
        f"acceptance gate must pass BENCH_AUTOMATION_BUDGET_JSON for every "
        f"orchestrator invocation; got {count} occurrences"
    )
    # Helper must read the JSON and emit a budget stanza per phase.
    assert "append_budget_stanza" in text


def test_orchestrator_writes_budget_when_env_set(tmp_path: Path):
    """fix_orchestrator must dump LLMClient.budget to the path named by
    BENCH_AUTOMATION_BUDGET_JSON when set; missing env var must be a
    silent no-op (back-compat with operator-driven runs).
    """
    # Inspect the source: argparse-level wiring is tested via subprocess
    # by other tests; here we just verify the dump block exists and
    # references the canonical 4 fields.
    text = _read(_REPO_ROOT / "scripts" / "automation" / "fix_orchestrator.py")
    assert 'os.environ.get("BENCH_AUTOMATION_BUDGET_JSON"' in text
    for field in (
        "total_calls",
        "prompt_tokens",
        "completion_tokens",
        "estimated_cost_usd",
    ):
        assert f'"{field}"' in text, f"budget dump must include {field}"


def test_acceptance_gate_health_probe_rejects_5xx():
    """The acceptance gate health probe must accept [200, 500) only.

    Earlier drafts allowed [200, 600), which would mark a server that
    only ever responds 500/502/503 as "tunnel alive" -- producing
    cascading downstream failures once the orchestrator tried to issue
    real chat requests. Behavioural verification via a fake HTTP
    server on a real socket so we exercise the same curl-driven
    semantics the runner uses at runtime.
    """
    import http.server
    import socket
    import threading

    text = _read(_RUNNERS_DIR / "run_acceptance_gate.sh")
    # Static: the contract code is "code >= 200 && code < 500", not
    # "< 600". The 5xx-as-alive branch is gone.
    assert "code >= 200 && code < 500" in text, (
        "acceptance gate must accept only [200, 500) as 'tunnel alive'"
    )
    assert "code >= 200 && code < 600" not in text, (
        "acceptance gate must NOT accept 5xx as 'tunnel alive'"
    )

    # Behavioural: a server that always returns 500 must be treated as
    # unreachable by the same curl invocation the runner uses.
    curl = shutil.which("curl")
    if curl is None:
        pytest.skip("curl(1) not installed on this host")

    class _Always500(http.server.BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802 - stdlib signature
            self.send_response(500)
            self.end_headers()
            self.wfile.write(b"boom\n")

        def log_message(self, *args, **kwargs):  # silence test noise
            pass

    # Pick a free port.
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]

    srv = http.server.HTTPServer(("127.0.0.1", port), _Always500)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    try:
        proc = subprocess.run(
            [
                curl, "-s", "-o", "/dev/null", "-w", "%{http_code}",
                "--max-time", "2",
                f"http://127.0.0.1:{port}/v1/models",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )
        code_str = (proc.stdout or "").strip() or "000"
        code = int(code_str) if code_str.isdigit() else 0
        # The runner's accept condition: [200, 500). 500 must FAIL it.
        accepted = 200 <= code < 500
        assert not accepted, (
            f"curl returned http={code} which the new gate must reject "
            f"(prior 5xx-as-alive bug would have accepted this)"
        )
    finally:
        srv.shutdown()
        srv.server_close()


def test_acceptance_gate_passes_task_ids_to_orchestrator():
    """Phase 2 must pass --task-ids (NOT just BENCH_AUTOMATION_TASK_IDS
    env) to fix_orchestrator so the stratified 5+5+(up to 5) allowlist
    is actually honoured.
    """
    text = _read(_RUNNERS_DIR / "run_acceptance_gate.sh")
    # The orchestrator CLI takes a comma-separated --task-ids flag and
    # the runner must pass it. The previous draft set
    # BENCH_AUTOMATION_TASK_IDS as a plain env var that the orchestrator
    # never read.
    assert "--task-ids" in text, (
        "acceptance gate must pass --task-ids to fix_orchestrator "
        "(env-var-only path was a no-op)"
    )
    assert '--task-ids "$task_ids"' in text, (
        "the resolved stratified id list must be threaded into --task-ids"
    )
    # The legacy env-var-only path must be gone.
    assert "BENCH_AUTOMATION_TASK_IDS=" not in text, (
        "the deprecated BENCH_AUTOMATION_TASK_IDS env var path must be "
        "removed in favour of the explicit --task-ids CLI flag"
    )
    # And the orchestrator must accept that flag.
    orch_text = _read(_REPO_ROOT / "scripts" / "automation" / "fix_orchestrator.py")
    assert '"--task-ids"' in orch_text, (
        "fix_orchestrator must declare --task-ids in its arg parser"
    )


def test_acceptance_gate_phase2_uses_shared_select_tasks_cli():
    """Phase 2 stratified sampler must invoke
    ``scripts/automation/select_tasks.py`` (which routes through the
    shared resolve loader) rather than embedding its own line-parser.
    This keeps the JSON / JSONL / legacy-text detection in one place.
    """
    text = _read(_RUNNERS_DIR / "run_acceptance_gate.sh")
    assert "scripts/automation/select_tasks.py" in text, (
        "Phase 2 sampler must call select_tasks.py CLI"
    )
    # And the legacy inline heredoc parser is gone.
    assert "buckets:" not in text or "select_tasks.py" in text, (
        "the legacy buckets-by-substring heredoc must be removed / replaced"
    )


def test_select_tasks_cli_smoke(tmp_path):
    """End-to-end smoke test for the stratified selector: it must
    read an aggregate JSON artefact (the format ``--refresh-cache``
    writes) and emit a CSV of task IDs capped at the requested per-class
    count.
    """
    import json
    import subprocess
    import sys

    p = tmp_path / "resolve.json"
    p.write_text(json.dumps({
        "verdicts": [
            {"task_id": f"S{i}", "status": "BLOCKED(no-verified-upstream-entrypoint: shapes-incomplete)"}
            for i in range(7)
        ] + [
            {"task_id": f"O{i}", "status": "BLOCKED(no-verified-upstream-entrypoint: oracle-mismatch)"}
            for i in range(2)
        ] + [
            {"task_id": f"U{i}", "status": "BLOCKED(no-verified-upstream-entrypoint: upstream-not-invocable-here)"}
            for i in range(10)
        ],
    }), encoding="utf-8")
    cli = _REPO_ROOT / "scripts" / "automation" / "select_tasks.py"
    proc = subprocess.run(
        [sys.executable, str(cli), str(p), "--cap", "5"],
        capture_output=True, text=True, check=True,
    )
    ids = proc.stdout.strip().split(",")
    # 5 shapes + 2 oracle (only 2 in corpus) + 5 upstream = 12.
    assert len(ids) == 12, f"expected 12, got {len(ids)}: {ids}"
    # Per-class capping honoured.
    assert sum(1 for i in ids if i.startswith("S")) == 5
    assert sum(1 for i in ids if i.startswith("O")) == 2
    assert sum(1 for i in ids if i.startswith("U")) == 5


def test_run_orchestrator_tail_pid_cleanup():
    """Orchestrator runner must declare TAIL_PID at module scope, set
    it after launching `tail -f`, and kill it in the cleanup trap so
    the operator-side log follower never survives as an orphan after
    runner exit.
    """
    text = _read(_RUNNERS_DIR / "run_machine_b_orchestrator.sh")

    # Declared at module scope alongside ORCH_PID / TUNNEL_PID.
    assert "TAIL_PID=\"\"" in text, (
        "TAIL_PID must be initialised to empty string at module scope"
    )
    # Captured immediately after the `tail -f` background launch.
    tail_idx = text.find("tail -f")
    assert tail_idx > 0, "expected `tail -f` background launch"
    capture_idx = text.find("TAIL_PID=$!", tail_idx)
    assert capture_idx > tail_idx, (
        "TAIL_PID=$! must immediately follow the `tail -f` launch"
    )
    # cleanup() trap kills TAIL_PID, not just ORCH_PID and TUNNEL_PID.
    assert 'kill -TERM "$TAIL_PID"' in text, (
        "cleanup() must SIGTERM the tail child"
    )
    # And waits on it (no zombies / no orphans).
    assert '"$TAIL_PID"' in text and "wait" in text, (
        "cleanup() must wait on the tail child"
    )

    # Ordering: in the cleanup() body, the kill sequence must be
    # orchestrator -> tunnel -> tail. We do the cheap textual check by
    # locating the three kill-TERM calls inside the cleanup() block.
    cleanup_start = text.find("cleanup() {")
    cleanup_end = text.find("\n}\n", cleanup_start)
    assert cleanup_start > 0 and cleanup_end > cleanup_start, (
        "cleanup() body not found"
    )
    body = text[cleanup_start:cleanup_end]
    orch_kill = body.find('kill -TERM "$ORCH_PID"')
    tunnel_kill = body.find('kill -TERM "$TUNNEL_PID"')
    tail_kill = body.find('kill -TERM "$TAIL_PID"')
    assert 0 < orch_kill < tunnel_kill < tail_kill, (
        f"cleanup() kill ordering must be orch ({orch_kill}) -> "
        f"tunnel ({tunnel_kill}) -> tail ({tail_kill})"
    )


# ---------------------------------------------------------------------------
# LLM transport behaviour (--llm-transport http | ssh-cli)
# ---------------------------------------------------------------------------

def test_orchestrator_runner_declares_transport_flags():
    """run_machine_b_orchestrator.sh must accept --llm-transport /
    --llm-ssh-host / --llm-cli-cmd CLI flags."""
    text = _read(_RUNNERS_DIR / "run_machine_b_orchestrator.sh")
    assert "--llm-transport" in text
    assert "--llm-ssh-host" in text
    assert "--llm-cli-cmd" in text
    # http is the back-compat default; existing call sites omit the flag.
    assert 'LLM_TRANSPORT="http"' in text


def test_orchestrator_runner_ssh_cli_branch_skips_tunnel():
    """ssh-cli mode opens NO ssh -L tunnel and runs NO curl probe.

    The transport guard `if [[ "$LLM_TRANSPORT" == "http" ]]` must wrap
    the `start_tunnel` call and the curl /v1/models probe block, so an
    ssh-cli invocation does not consume a local port or depend on
    curl(1).
    """
    text = _read(_RUNNERS_DIR / "run_machine_b_orchestrator.sh")
    # Transport guard around start_tunnel.
    assert 'if [[ "$LLM_TRANSPORT" == "http" ]]; then' in text, (
        "orchestrator runner must guard tunnel + curl on transport=http"
    )
    # The curl probe must live inside that branch (we check ordering:
    # the guard appears before the curl line).
    guard_idx = text.find('if [[ "$LLM_TRANSPORT" == "http" ]]; then')
    probe_idx = text.find('"$LLM_ENDPOINT$HEALTH_PATH"')
    assert 0 < guard_idx < probe_idx, (
        "curl /v1/models probe must live inside the http guard"
    )
    # ssh-cli branch probes via `ssh ... 'echo CLI_OK'`, not curl.
    assert "echo CLI_OK" in text, (
        "ssh-cli branch must do a plain ssh reachability probe"
    )


def test_orchestrator_runner_passes_transport_flags_to_orchestrator():
    """The orchestrator argv must include --llm-transport and the
    transport-specific flags (--llm-endpoint OR --llm-ssh-host+--llm-cli-cmd).
    """
    text = _read(_RUNNERS_DIR / "run_machine_b_orchestrator.sh")
    assert '--llm-transport "$LLM_TRANSPORT"' in text
    assert '--llm-endpoint "$LLM_ENDPOINT"' in text
    assert '--llm-ssh-host "$LLM_SSH_HOST"' in text
    assert '--llm-cli-cmd "$LLM_CLI_CMD"' in text
    assert '--llm-timeout-seconds "$LLM_TIMEOUT_SECONDS"' in text


def test_orchestrator_runner_validates_prompt_placeholder():
    """ssh-cli mode must reject a template without the literal '{prompt}'."""
    text = _read(_RUNNERS_DIR / "run_machine_b_orchestrator.sh")
    assert '{prompt}' in text, (
        "runner must validate the {prompt} placeholder in --llm-cli-cmd"
    )
    assert 'must contain literal' in text


def test_orchestrator_runner_supervisor_loop_is_transport_aware():
    """In ssh-cli mode the supervisor watches ORCH_PID only (no TUNNEL_PID)."""
    text = _read(_RUNNERS_DIR / "run_machine_b_orchestrator.sh")
    # http branch retains the dual-PID supervisor.
    assert ('while kill -0 "$ORCH_PID" 2>/dev/null '
            '&& kill -0 "$TUNNEL_PID" 2>/dev/null') in text
    # ssh-cli branch supervises ORCH_PID only.
    assert 'while kill -0 "$ORCH_PID" 2>/dev/null; do' in text


def test_acceptance_gate_declares_transport_flags():
    text = _read(_RUNNERS_DIR / "run_acceptance_gate.sh")
    assert "--llm-transport" in text
    assert "--llm-ssh-host" in text
    assert "--llm-cli-cmd" in text
    assert 'LLM_TRANSPORT="http"' in text


def test_acceptance_gate_start_tunnel_is_transport_aware():
    """The single start_tunnel definition must early-return in ssh-cli
    mode (no port forward, no curl probe). Test invariant: invoked
    exactly once from main (matches the existing single-tunnel test).
    """
    text = _read(_RUNNERS_DIR / "run_acceptance_gate.sh")
    # The helper still exists, still called exactly once from main, but
    # branches internally on $LLM_TRANSPORT.
    assert 'if [[ "$LLM_TRANSPORT" == "ssh-cli" ]]; then' in text, (
        "start_tunnel must early-return on transport=ssh-cli"
    )
    # ssh-cli branch does ssh reachability probe, not curl.
    assert "echo CLI_OK" in text
    # The four phase command builders must NOT hardcode the http-only
    # --llm-endpoint flag; only the append_llm_args helper is allowed to
    # emit it (and only in the http branch). We assert exactly one
    # occurrence in the file, inside the helper.
    count = text.count('--llm-endpoint "$LLM_ENDPOINT"')
    assert count == 1, (
        f"--llm-endpoint must appear exactly once (inside the "
        f"append_llm_args helper); got {count} occurrences"
    )
    helper_start = text.find("append_llm_args() {")
    helper_end = text.find("\n}\n", helper_start)
    assert helper_start > 0 and helper_end > helper_start
    helper_body = text[helper_start:helper_end]
    assert '--llm-endpoint "$LLM_ENDPOINT"' in helper_body, (
        "the sole --llm-endpoint reference must live inside append_llm_args"
    )


def test_acceptance_gate_append_llm_args_helper_used_per_phase():
    """append_llm_args must be called from every phase command
    builder so all 4 phases honour the transport flags consistently.
    """
    text = _read(_RUNNERS_DIR / "run_acceptance_gate.sh")
    # 5 invocations total: phase1 cmd, phase2 cmd, phase3 cmd, phase4
    # tier1 cmd, phase4 tier2 cmd.
    count = text.count("append_llm_args ")
    assert count >= 5, (
        f"append_llm_args must be invoked at least 5 times (one per "
        f"orchestrator argv); got {count}"
    )


def test_acceptance_gate_5xx_probe_block_is_http_only():
    """The 500-rejection curl probe block runs only when transport=http.

    In ssh-cli mode there is no curl probe at all, so the
    `code >= 200 && code < 500` filter does not apply. We verify the
    probe lives inside an http-mode branch in start_tunnel.
    """
    text = _read(_RUNNERS_DIR / "run_acceptance_gate.sh")
    # The probe loop body is still present and still rejects 5xx.
    assert "code >= 200 && code < 500" in text
    # Locate the start_tunnel body and confirm the curl call sits AFTER
    # the ssh-cli early-return so it never runs in ssh-cli mode.
    start_tunnel_idx = text.find("start_tunnel() {")
    cli_branch_idx = text.find('if [[ "$LLM_TRANSPORT" == "ssh-cli" ]]; then',
                                start_tunnel_idx)
    curl_idx = text.find("curl -s -o /dev/null -w '%{http_code}'",
                          start_tunnel_idx)
    assert 0 < start_tunnel_idx < cli_branch_idx < curl_idx, (
        "the ssh-cli early-return must precede the curl probe in start_tunnel"
    )


def test_setup_b_declares_transport_flag():
    """setup_machine_b_cpu.sh must accept --llm-transport (and
    optionally --llm-cli-cmd for the probe).
    """
    text = _read(_RUNNERS_DIR / "setup_machine_b_cpu.sh")
    assert "--llm-transport" in text
    assert "--llm-cli-cmd" in text
    assert 'LLM_TRANSPORT="http"' in text


def test_setup_b_imports_use_correct_class_names():
    """setup_machine_b_cpu.sh must import classes that actually exist.

    A historical typo imported a non-existent OpenAIChatClient symbol;
    the canonical exports are LLMClient + SSHCLIClient + create_llm_client.
    """
    text = _read(_RUNNERS_DIR / "setup_machine_b_cpu.sh")
    # No reference to the non-existent symbol.
    assert "OpenAIChatClient" not in text, (
        "setup_machine_b_cpu.sh must not import a non-existent class"
    )
    # The smoke import line must reference real exports.
    assert "from automation.llm_client import" in text
    for needed in ("LLMClient", "SSHCLIClient", "create_llm_client"):
        assert needed in text, f"setup smoke import missing {needed}"


def test_setup_b_ssh_cli_probe_substitutes_prompt_placeholder():
    """In ssh-cli mode the CLI probe substitutes '{prompt}' with
    '--version' to test the CLI without sending a real prompt."""
    text = _read(_RUNNERS_DIR / "setup_machine_b_cpu.sh")
    # The probe substitution is bash-native (no python helper).
    assert '${LLM_CLI_CMD//\\{prompt\\}/--version}' in text, (
        "setup must substitute {prompt} with --version for the probe"
    )
    # And the probe runs only when transport == ssh-cli AND cli-cmd is set.
    assert '[[ "$LLM_TRANSPORT" == "ssh-cli" ]]' in text


# ---------------------------------------------------------------------------
# Machine-target annotations -- every runner's top comment and --help output
# must declare the target machine so operators see it before reading docs.
# ---------------------------------------------------------------------------

_RUNNER_TARGETS = {
    "setup_machine_a_gpu.sh": "Runs on: A",
    "setup_machine_b_cpu.sh": "Runs on: B",
    "run_machine_a_daemon.sh": "Runs on: A",
    "run_machine_b_orchestrator.sh": "Runs on: B",
    "run_acceptance_gate.sh": "Runs on: B",
}


def test_setup_machine_a_gpu_help_declares_runs_on_a():
    proc = _run_help(_RUNNERS_DIR / "setup_machine_a_gpu.sh")
    assert proc.returncode == 0, proc.stderr
    assert "Runs on: A" in proc.stdout, (
        f"setup_machine_a_gpu.sh --help must declare 'Runs on: A'; got:\n{proc.stdout}"
    )


def test_setup_machine_b_cpu_help_declares_runs_on_b():
    proc = _run_help(_RUNNERS_DIR / "setup_machine_b_cpu.sh")
    assert proc.returncode == 0, proc.stderr
    assert "Runs on: B" in proc.stdout, (
        f"setup_machine_b_cpu.sh --help must declare 'Runs on: B'; got:\n{proc.stdout}"
    )


def test_run_machine_a_daemon_help_declares_runs_on_a():
    proc = _run_help(_RUNNERS_DIR / "run_machine_a_daemon.sh")
    assert proc.returncode == 0, proc.stderr
    assert "Runs on: A" in proc.stdout, (
        f"run_machine_a_daemon.sh --help must declare 'Runs on: A'; got:\n{proc.stdout}"
    )


def test_run_machine_b_orchestrator_help_declares_runs_on_b():
    proc = _run_help(_RUNNERS_DIR / "run_machine_b_orchestrator.sh")
    assert proc.returncode == 0, proc.stderr
    assert "Runs on: B" in proc.stdout, (
        f"run_machine_b_orchestrator.sh --help must declare 'Runs on: B'; "
        f"got:\n{proc.stdout}"
    )


def test_run_acceptance_gate_help_declares_runs_on_b():
    proc = _run_help(_RUNNERS_DIR / "run_acceptance_gate.sh")
    assert proc.returncode == 0, proc.stderr
    assert "Runs on: B" in proc.stdout, (
        f"run_acceptance_gate.sh --help must declare 'Runs on: B'; got:\n{proc.stdout}"
    )


@pytest.mark.parametrize(
    "script_name,expected",
    list(_RUNNER_TARGETS.items()),
    ids=list(_RUNNER_TARGETS),
)
def test_all_5_runner_scripts_have_runs_on_header_comment(
    script_name: str, expected: str
):
    """Every runner's top comment block (within first 5 lines) must contain
    the canonical 'Runs on: <target>' marker so reviewers see the target
    without invoking --help."""
    path = _RUNNERS_DIR / script_name
    head = "".join(path.read_text(encoding="utf-8").splitlines(keepends=True)[:5])
    assert expected in head, (
        f"{script_name} header (first 5 lines) must declare '{expected}'; "
        f"got:\n{head}"
    )


def test_runbook_documents_actual_runner_behaviour():
    """The runbook sections 3.3, 5, 6, 7 must reflect the actual
    implementation, not aspirational behaviour. We check for the
    specific contracts the runners now honour: /v1/models probe, flock
    daemon serialization, tunnel-driven orchestrator termination, and
    dated artefact names."""
    text = _read(_RUNBOOK)
    # Section 3.3 -- orchestrator runner contracts.
    assert "/v1/models" in text, "runbook must document the /v1/models probe"
    assert "--health-path" in text, "runbook must document --health-path"
    assert "Resolve-log preflight" in text, "runbook must document preflight"
    # Section 5 -- troubleshooting table.
    assert "flock" in text, "runbook must document the flock-based daemon lock"
    assert "git worktree" in text, "runbook must document Phase 4 worktree isolation"
    # Section 6 -- shutdown: must specify SIGTERM-driven drain.
    assert "SIGTERM" in text, "runbook must specify SIGTERM-driven shutdown"
    # And must call out the no-SIGKILL guarantee for in-flight job drain.
    assert "no `SIGKILL`" in text or "not a SIGKILL" in text, (
        "runbook must document the SIGTERM-only (no-SIGKILL) guarantee"
    )
    # Section 7 -- acceptance criteria (per-phase pass conditions).
    for needle in (
        "backlog-report-<ISO_DATE>",
        "summary-metrics-<ISO_DATE>",
        "verdict_class",
        "5+5+(up to 5)",
    ):
        assert needle in text, f"runbook missing acceptance-criteria detail: {needle}"


# ---------------------------------------------------------------------------
# Doc contract: TAB_C_CLI_CMD configuration guidance
#
# These tests assert that the operator-facing quickstart md and the
# tab_quickstart.conf.example actually document how to fill in the
# TAB_C_CLI_CMD template for the two CLIs operators most often have on
# the C host. The literal CLI tokens that appear in the assertions below
# (e.g. "claude --print {prompt}", "codex exec {prompt}") are
# documentation sample strings -- they are NOT a violation of the
# CLI-agnostic guarantee enforced by
# test_tab_quickstart_script_has_no_hardcoded_cli_names, which scans
# tab_quickstart.sh only. The bench code itself remains CLI-agnostic;
# these strings live exclusively in docs and the conf example so
# operators can see canonical templates instead of guessing.
# ---------------------------------------------------------------------------

_QUICKSTART_MD = (
    _REPO_ROOT / "docs" / "release" / "internal-acceptance-quickstart.md"
)
_CONF_EXAMPLE = (
    _RUNNERS_DIR / "tab_quickstart.conf.example"
)


def test_quickstart_md_has_cli_cmd_config_section():
    """Quickstart md must include a dedicated 'TAB_C_CLI_CMD' config section
    so operators are not left guessing how to fill the template."""
    assert _QUICKSTART_MD.is_file(), f"missing quickstart md: {_QUICKSTART_MD}"
    text = _QUICKSTART_MD.read_text(encoding="utf-8")
    assert "TAB_C_CLI_CMD" in text, "quickstart md must mention TAB_C_CLI_CMD"
    # The section is anchored by a top-level '## TAB_C_CLI_CMD ...' heading
    # so operators can locate the config guidance from the md table of
    # contents.
    heading_lines = [
        ln for ln in text.splitlines()
        if ln.startswith("## ") and "TAB_C_CLI_CMD" in ln
    ]
    assert heading_lines, (
        "quickstart md must contain a dedicated '## TAB_C_CLI_CMD ...' "
        "section so operators can locate the config guidance"
    )


def test_quickstart_md_lists_claude_and_codex_cli_examples():
    """Quickstart md must show canonical TAB_C_CLI_CMD templates for the
    two CLIs operators most often have on the C host."""
    text = _QUICKSTART_MD.read_text(encoding="utf-8")
    # The exact template strings operators copy-paste. These are doc
    # samples, not hardcoded references in code -- see the section
    # comment above.
    assert "claude --print {prompt}" in text, (
        "quickstart md must document the Claude Code CLI template"
    )
    assert "codex exec {prompt}" in text, (
        "quickstart md must document the OpenAI Codex CLI template"
    )


def test_quickstart_md_documents_prompt_placeholder_rules():
    """Quickstart md must document the {prompt} placeholder rules:
    exactly once, no surrounding quotes (including outer quotes)."""
    text = _QUICKSTART_MD.read_text(encoding="utf-8")
    # Must mention the placeholder at least a few times across rules,
    # advanced usage, and the validator's fatal error message.
    placeholder_hits = text.count("{prompt}")
    assert placeholder_hits >= 5, (
        f"quickstart md must reference {{prompt}} placeholder repeatedly "
        f"(rules, examples, errors); got {placeholder_hits} occurrence(s)"
    )
    # Must document the "exactly once" rule. The FATAL error sample
    # block embeds this phrase verbatim.
    assert "exactly once" in text, (
        "quickstart md must document the 'exactly once' placeholder rule"
    )
    # Must document the "no surrounding quotes" rule and explain that
    # the bench wraps with shlex.quote on its own. The FATAL sample block
    # contains "do NOT wrap" verbatim.
    assert "shlex.quote" in text, (
        "quickstart md must explain that bench wraps with shlex.quote"
    )
    assert "do NOT wrap" in text, (
        "quickstart md must document the 'do NOT wrap' rule"
    )
    # Must call out the outer-quote-span trap explicitly. The FATAL
    # sample block uses the phrase "even outer ones".
    assert "outer" in text.lower(), (
        "quickstart md must call out the outer-quote-span placeholder trap"
    )


def test_conf_example_has_cli_cmd_compare_table():
    """tab_quickstart.conf.example must enumerate canonical templates for
    Claude Code CLI and OpenAI Codex CLI so operators can copy-paste."""
    assert _CONF_EXAMPLE.is_file(), f"missing conf example: {_CONF_EXAMPLE}"
    text = _CONF_EXAMPLE.read_text(encoding="utf-8")
    # The doc-sample template strings.
    assert "claude --print {prompt}" in text, (
        "conf example must show the Claude Code CLI template"
    )
    assert "codex exec {prompt}" in text, (
        "conf example must show the OpenAI Codex CLI template"
    )
    # The header that labels the comparison block.
    assert "Common CLI templates" in text or "Common CLI" in text, (
        "conf example must include a labelled comparison block for the "
        "common CLI templates"
    )
    # And the placeholder must be documented as a required token.
    assert "{prompt}" in text
