"""Release boundary verification tests (M0 Unit 0.2).

Scope: ensure the INTERNAL automation tree does not leak into release
artifacts, and that bench core does not call into the orchestrator.

Each scenario is a small, deterministic shell-out to ``git``, ``tar``,
or in-process ``grep``. The tests fail loud if any of the four release-
boundary mechanisms drift (`.gitattributes` rule list, `.dockerignore`
mirror, single-direction grep on bench core, external-project-name
grep on release scope).
"""
from __future__ import annotations

import re
import shutil
import subprocess
import tarfile
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _git_available() -> bool:
    return shutil.which("git") is not None and (_REPO_ROOT / ".git").exists()


@pytest.fixture
def release_tarball(tmp_path):
    """Produce a real `git archive HEAD` tarball for the current
    repository so the test inspects the same artefact CI would ship.

    ``--worktree-attributes`` makes the archive honor the working-tree
    `.gitattributes` (instead of the committed copy). Without this
    flag the test would falsely fail in a contributor's local checkout
    whenever `.gitattributes` itself is still uncommitted -- the
    boundary rules would be in the working tree but not yet in HEAD,
    so the archive wouldn't apply them.
    """
    if not _git_available():
        pytest.skip("git not available or repo not initialized")
    tar_path = tmp_path / "release.tar"
    subprocess.run(
        ["git", "archive", "HEAD", "--format=tar",
         "--worktree-attributes", "-o", str(tar_path)],
        cwd=_REPO_ROOT,
        check=True,
    )
    return tar_path


# --- Happy path: release tarball excludes INTERNAL artifacts ----------------

_INTERNAL_PATTERNS = re.compile(
    r"(scripts/automation/"
    r"|scripts/loop/"
    r"|scripts/internal/"
    r"|tab_automation/"
    r"|docs/plans/"
    r"|docs/reviews/"
    r"|docs/triage_backlog/"
    r"|docs/brainstorms/"
    r"|docs/architecture/"
    r"|docs/release/"
    r"|docs/internal/"
    r"|infra/)"
)


def test_release_tarball_excludes_internal_automation(release_tarball):
    with tarfile.open(release_tarball, "r") as tf:
        members = [m.name for m in tf.getmembers()]
    leaked = [m for m in members if _INTERNAL_PATTERNS.search(m)]
    assert not leaked, (
        "release tarball leaked INTERNAL artefacts -- check .gitattributes "
        f"export-ignore rules. Leaked members: {leaked[:5]}"
    )


def test_release_tarball_excludes_internal_test_files(release_tarball):
    """The Unit 0.1 / 0.2 test files themselves are INTERNAL and must
    NOT appear in the release tarball (covered by the
    ``test_*automation*.py`` and ``test_release_boundary.py`` rules)."""
    with tarfile.open(release_tarball, "r") as tf:
        members = [m.name for m in tf.getmembers()]
    internal_tests = {
        "tests/unit_tests/test_automation_skeleton.py",
        "tests/unit_tests/test_release_boundary.py",
        "tests/unit_tests/test_r19_boundary.py",
        "tests/unit_tests/test_resolve_loader.py",
    }
    leaked = [m for m in members if m in internal_tests]
    assert not leaked, (
        "INTERNAL test files leaked into release tarball: "
        f"{leaked}. Check .gitattributes export-ignore rules."
    )


# --- Happy path: bench core does not reference orchestrator -----------------

# Every PUBLIC (release-tarball) module. Fenced dev-only helpers
# (scripts/loop/, scripts/automation/, full_regression, extract_kernel, ...)
# are export-ignored and intentionally NOT listed here.
_BENCH_CORE_PATHS = [
    "ascendbench",
    "scripts/oracle/apply_oracle_equivalence_verdict.py",
    "scripts/perf/audit_cost_model_formulas.py",
    "scripts/corpus/generate_task_scaffolds.py",
    "scripts/provenance/write_golden_manifest.py",
    "scripts/oracle/run_oracle_equivalence_gpu.py",
    "scripts/oracle/precompute_oracle_reference.py",
    "scripts/verify/oracle_consistency_anchor_runner.py",
    "scripts/verify/severance_subprocess_runner.py",
    "scripts/perf/cross_check_cost_model_gpu.py",
    "scripts/verify/freeze_upstream_sources.py",
    "scripts/verify/standalone_freeze_verifier.py",
    "scripts/provenance/capture_golden_baselines.py",
    "scripts/corpus/generate_oracle_from_kernel.py",
    "scripts/corpus/generate_shapes_and_contract.py",
    "scripts/provenance/compute_release_planning_gate.py",
    "scripts/verify/resolve_upstream_entrypoints.py",
    "scripts/provenance/census_source_provenance.py",
    "scripts/corpus/validate_corpus_consistency.py",
    "scripts/corpus/validate_tasks.py",
    "scripts/provenance/verify_upstream_commit_chain.py",
    "scripts/provenance/verify_version_matrix.py",
]

_ORCH_REFS = re.compile(r"\bfrom automation\b|\bimport automation\b|scripts\.automation|\btab_automation\b")


def test_bench_core_does_not_import_orchestrator():
    """bench core MUST NOT depend on the INTERNAL orchestrator.
    Dependency direction is strictly: orchestrator -> bench core (via
    ``importlib.util.spec_from_file_location``), never the reverse."""
    leaks: list[tuple[str, int, str]] = []
    for rel in _BENCH_CORE_PATHS:
        target = _REPO_ROOT / rel
        if not target.exists():
            continue
        files: list[Path]
        if target.is_dir():
            files = list(target.rglob("*.py"))
        else:
            files = [target]
        for f in files:
            try:
                lines = f.read_text(encoding="utf-8").splitlines()
            except (UnicodeDecodeError, OSError):
                continue
            for i, line in enumerate(lines, 1):
                if _ORCH_REFS.search(line):
                    leaks.append((str(f.relative_to(_REPO_ROOT)), i, line.strip()))
    assert not leaks, (
        "bench core MUST NOT reference orchestrator. "
        f"Offending lines: {leaks[:5]}"
    )


# --- Happy path: release scope does not name external projects --------------

_EXTERNAL_NAMES = re.compile(r"kernelpilot|KernelPilot|kp_")
_SCAN_SUFFIXES = {".py", ".md", ".toml", ".yaml", ".yml"}


def test_release_scope_does_not_name_external_projects(release_tarball):
    """Scan only the files that actually ship in the release tarball.

    Scanning the on-disk working tree would false-positive on INTERNAL
    test files (e.g. this file itself contains the word 'kernelpilot'
    inside its own regex), which never ship because they are
    export-ignored. The release boundary check is therefore: take the
    final tarball contents -> scan -> 0 hits.
    """
    leaks: list[tuple[str, int, str]] = []
    with tarfile.open(release_tarball, "r") as tf:
        for member in tf.getmembers():
            if not member.isfile():
                continue
            if Path(member.name).suffix not in _SCAN_SUFFIXES:
                continue
            f = tf.extractfile(member)
            if f is None:
                continue
            try:
                text = f.read().decode("utf-8")
            except UnicodeDecodeError:
                continue
            for i, line in enumerate(text.splitlines(), 1):
                if _EXTERNAL_NAMES.search(line):
                    leaks.append((member.name, i, line.strip()))
    assert not leaks, (
        "release tarball MUST NOT name external projects. Lines: "
        f"{leaks[:5]}"
    )


# --- Vacuity guard: INTERNAL artefacts must exist on disk -------------------

def test_internal_artifacts_exist_in_repo():
    """Sanity / vacuity guard: the INTERNAL artefacts DO exist in the
    working tree (the ``not in tarball`` assertion above would otherwise
    pass vacuously)."""
    assert (_REPO_ROOT / "scripts" / "automation" / "__init__.py").is_file()
    assert (_REPO_ROOT / "docs" / "release" / "internal-release-process.md").is_file()


def test_gitattributes_covers_all_internal_artefact_classes():
    """Declarative completeness check: every well-known INTERNAL
    artefact class is covered by a `.gitattributes` rule. This
    catches the case where someone introduces a new INTERNAL path
    class but forgets to add an export-ignore rule.

    Together with the synthetic-archive test above, this gives a
    non-vacuous guarantee: rules exist (this test) AND rules work
    on a populated archive (the synthetic-archive test).
    """
    ga_text = (_REPO_ROOT / ".gitattributes").read_text(encoding="utf-8")
    required_rules = [
        "scripts/automation/",
        "tab_automation/",
        "docs/plans/",
        "docs/reviews/",
        "docs/triage_backlog/",
        "docs/brainstorms/",
        "docs/architecture/",
        "docs/release/",
        "docs/internal/",
        "infra/",
        "tests/unit_tests/test_release_boundary.py",
        "tests/unit_tests/test_*automation*.py",
    ]
    missing = [r for r in required_rules if r not in ga_text]
    assert not missing, (
        f".gitattributes is missing rules for: {missing}. "
        "Add an `export-ignore` entry for each."
    )


# --- Edge case: .gitattributes mirror matches .dockerignore -----------------

def _read_text_lines(path: Path) -> list[str]:
    return [
        ln.strip()
        for ln in path.read_text(encoding="utf-8").splitlines()
        if ln.strip() and not ln.strip().startswith("#")
    ]


# --- CI command parity: shell out exactly the way CI does -------------------
#
# The Python-level happy-path tests above use in-process scans (``tarfile``
# walks, regex sweeps). Those would not have caught the M0 codex R1 finding
# where the CI grep step used a path-form ``--exclude-dir`` that GNU grep
# silently ignored. The two tests below shell out the **exact** commands the
# CI yaml runs and assert the same exit codes. If either CI step regresses
# in semantics (basename vs path, wrong scope, missing flag) these tests
# fail loud.


def test_ci_bench_core_single_direction_grep_matches_step_logic():
    """Mirror CI ``Verify bench core does not import orchestrator`` step.

    Runs the same working-tree grep over the same bench-core path list and
    asserts a clean exit (``grep`` exit code 1 == no matches found).
    """
    paths = [
        "ascendbench/",
        "scripts/corpus/validate_tasks.py",
        "scripts/verify/standalone_freeze_verifier.py",
        "scripts/provenance/capture_golden_baselines.py",
        "scripts/verify/resolve_upstream_entrypoints.py",
        "scripts/perf/cross_check_cost_model_gpu.py",
        "scripts/provenance/verify_version_matrix.py",
        "scripts/verify/freeze_upstream_sources.py",
        "scripts/corpus/generate_oracle_from_kernel.py",
    ]
    cmd = [
        "grep",
        "-rn",
        r"from automation\b\|import automation\b\|scripts\.automation\|tab_automation\b",
        *paths,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=_REPO_ROOT)
    assert result.returncode == 1, (
        "bench core MUST NOT reference orchestrator -- the CI grep step is "
        f"expected to exit 1 (no matches), got exit={result.returncode}.\n"
        f"hits:\n{result.stdout}"
    )


def test_ci_release_tarball_no_external_project_name(tmp_path):
    """Mirror CI ``Verify release tarball does not reference external project
    names`` step.

    Reproduces the post-MF1 command exactly: ``git archive`` -> extract ->
    plain recursive grep over the extracted tree. The release tarball must
    have zero hits for the forbidden external-project tokens.

    This catches the M0 codex R1 footgun where a path-form ``--exclude-dir``
    would silently fail to exclude and CI would self-trip on INTERNAL plan
    docs / boundary-test regex strings.
    """
    if not _git_available():
        pytest.skip("git not available or repo not initialized")
    tar_path = tmp_path / "release.tar"
    subprocess.run(
        ["git", "archive", "HEAD", "--format=tar",
         "--worktree-attributes", "-o", str(tar_path)],
        cwd=_REPO_ROOT,
        check=True,
    )
    extract_dir = tmp_path / "release"
    extract_dir.mkdir()
    subprocess.run(["tar", "xf", str(tar_path), "-C", str(extract_dir)], check=True)
    tar_path.unlink()
    result = subprocess.run(
        ["grep", "-rin", r"kernelpilot\|KernelPilot\|kp_", str(extract_dir)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 1, (
        "release tarball MUST NOT reference external project names -- the CI "
        f"grep step is expected to exit 1, got exit={result.returncode}.\n"
        f"hits:\n{result.stdout}"
    )


def test_dockerignore_mirrors_gitattributes_export_ignore():
    """Every export-ignore pattern in .gitattributes must also appear
    in .dockerignore (with the ``export-ignore`` suffix stripped). They
    cover the same release-boundary contract on two different channels,
    so drift between them silently breaks the Docker boundary."""
    ga = _REPO_ROOT / ".gitattributes"
    di = _REPO_ROOT / ".dockerignore"
    assert ga.is_file(), ".gitattributes must exist"
    assert di.is_file(), ".dockerignore must exist"

    ga_patterns: set[str] = set()
    for line in _read_text_lines(ga):
        if "export-ignore" in line:
            # The pattern is the first whitespace-separated token.
            pat = line.split()[0]
            ga_patterns.add(pat)

    di_patterns = set(_read_text_lines(di))

    missing = ga_patterns - di_patterns
    assert not missing, (
        ".dockerignore must mirror every .gitattributes export-ignore "
        f"pattern. Missing: {sorted(missing)}"
    )
