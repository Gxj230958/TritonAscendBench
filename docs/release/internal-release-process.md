# INTERNAL: bench release artifact + release-boundary CI semantics

> **This document is INTERNAL.** The `internal-` filename prefix matches
> the `docs/release/internal-*.md` pattern in `.gitattributes` and
> `.dockerignore`, so this file is automatically excluded from the
> release tarball and Docker build context.

## Why a release boundary

The bench repository has two physically co-located but logically
separate code surfaces:

1. **bench core (released)** -- `triton_ascend_bench/`, the relevant
   parts of `scripts/`, `tests/unit_tests/` excluding internal tests,
   `specs/`, `infra/`, public `docs/`. Ships in:
   - the release tarball produced by `git archive HEAD`,
   - the Docker image built from `infra/Dockerfile.gpu` (etc.),
   - the wheel produced by `pip install -e .` /
     `python -m build` (already scoped via
     `pyproject.toml [tool.setuptools.packages.find]`).
2. **INTERNAL automation (not released)** -- `scripts/automation/`
   plus its tests, plus two design / implementation plan documents
   in `docs/plans/`, plus the `docs/release/internal-*.md` reports
   and `docs/triage_backlog/`.

The release boundary must hold along **three** distribution channels
(git archive, Docker build context, wheel) and must be enforceable by
CI without depending on contributor discipline alone.

## How the boundary is enforced

| Channel              | Mechanism                                        |
|----------------------|--------------------------------------------------|
| `git archive` tarball| `.gitattributes` `export-ignore` rules           |
| Docker build context | `.dockerignore` mirror of `.gitattributes`       |
| pip wheel            | `pyproject.toml [tool.setuptools.packages.find]` already scopes to `triton_ascend_bench*` |
| Single-direction dep | CI `grep` step on the working tree: bench core MUST NOT `from automation ...` |
| External project name| CI `grep` step on the **extracted release tarball**: shipped artefacts MUST NOT name external projects |

All four checks live in `.github/workflows/ci.yml` `unit-tests` job
and run on every push / PR. A failing check turns CI red -- there is
no `continue-on-error` escape hatch.

## Pattern style

* `.gitattributes` and `.dockerignore` use **wildcard** rules so that
  adding a new INTERNAL test file (e.g. `test_fix_orchestrator_m2.py`)
  is automatically excluded; no manual update needed if the file name
  matches one of the existing patterns.
* The external-project-name grep runs on the **release tarball**
  (`git archive | tar x | grep`) rather than the working tree. This
  is intentional: GNU `grep --exclude-dir` matches only the basename
  (not a path), so `--exclude-dir=docs/plans` would NOT exclude
  `docs/plans/foo.md` and CI would self-trip on INTERNAL plan docs.
  Scanning the post-`export-ignore` tarball aligns the check with the
  release contract and removes the need for a parallel exclude list.
* The single-direction grep still runs on the working tree but is
  scoped to an explicit list of bench-core files, so basename vs path
  is not a concern.

## How to verify locally

```bash
# 1. Release tarball must not contain INTERNAL artifacts (path manifest)
#    Explicit rc capture matches CI: 0=found(=FAIL), 1=clean(=OK), 2+=error(=FAIL loud).
#    Bare `tar tf | grep ... && echo FAIL || echo OK` is fail-open: a failing
#    tar plus empty grep input returns 1 → silently prints OK.
set -o pipefail
git archive HEAD --format=tar --worktree-attributes -o /tmp/release.tar
rc=0
tar tf /tmp/release.tar \
    | grep -E '(scripts/automation|tab_automation|docs/plans/2026-05-23-00[12]|docs/release/internal-|docs/triage_backlog)' \
    || rc=$?
[ "$rc" = "1" ] && echo OK || echo "FAIL rc=$rc"
set +o pipefail

# 2. bench core must not reference orchestrator (single-direction, working tree)
#    Same path list and exit-code discipline as CI step
#    `Verify bench core does not import orchestrator`.
#    grep rc: 0=found(=FAIL), 1=clean(=OK), 2+=error(=FAIL loud).
rc=0
grep -rn 'from automation\b\|import automation\b\|scripts\.automation\|tab_automation\b' \
    triton_ascend_bench/ \
    scripts/validate_tasks.py scripts/phase_a_prime.py \
    scripts/phase_b_golden_capture.py scripts/resolve_upstream_entrypoints.py \
    scripts/phase_a_cost_xcheck.py scripts/verify_version_matrix.py \
    scripts/phase_a_freeze.py scripts/phase_b_oracle.py || rc=$?
[ "$rc" = "1" ] && echo OK || echo "FAIL rc=$rc"

# 3. release artefact must not name external projects (post export-ignore)
#    The grep pattern is assembled from concatenated shell strings so
#    that the source of this command -- `.github/workflows/ci.yml`,
#    which DOES ship in the tarball and runs the same grep in CI --
#    never contains the complete tokens as a literal substring and
#    cannot self-trip. This INTERNAL doc itself is excluded from the
#    tarball via `docs/release/internal-*.md` export-ignore, but we
#    still use the split-token style here so the command stays
#    copy-paste consistent with the CI step.
tmp=$(mktemp -d)
git archive HEAD --format=tar --worktree-attributes -o "$tmp/release.tar"
tar xf "$tmp/release.tar" -C "$tmp" && rm "$tmp/release.tar"
tok1='kernel''pilot'
tok2='Kernel''Pilot'
tok3='kp''_'
pat="${tok1}\\|${tok2}\\|${tok3}"
rc=0
grep -rin "$pat" "$tmp" || rc=$?
[ "$rc" = "1" ] && echo OK || echo "FAIL rc=$rc"
```

> NOTE: by default `git archive HEAD` honours the `.gitattributes`
> **baked into the archived tree (`HEAD`)**, not the working tree.
> `--worktree-attributes` switches to reading the **working-tree**
> `.gitattributes`. So a PR that updates the boundary rules but has
> not yet committed them — or has uncommitted edits to the rules —
> is exercised against its own working-tree rules only when the flag
> is passed. Omitting the flag verifies the archive against the
> committed (HEAD) rules, which is the wrong scope for a boundary
> PR. Always pass the flag in local preflight + CI.

> NOTE: GNU `grep --exclude-dir` matches **basename only**, never a
> path. A pattern like `--exclude-dir=docs/plans` is silently a no-op
> (it would match a directory literally named `docs/plans`, of which
> there is none -- the leaf is named `plans`). If you ever need to
> exclude a path-shaped subtree from a working-tree grep, combine
> basename `--exclude-dir=<basename>` with file-level `--exclude=<glob>`,
> or -- better -- scan the release tarball instead.

## Adding new INTERNAL artifacts

When introducing a new INTERNAL file or directory:

1. Place it under one of the patterns already covered by
   `.gitattributes` / `.dockerignore`, **or**
2. Add an explicit new rule to BOTH `.gitattributes` AND
   `.dockerignore` in the same commit, **and**
3. Run the three local verification commands above. There is no
   `--exclude-dir` list to maintain: the external-project grep scans
   the post-`export-ignore` tarball, so a correctly placed INTERNAL
   artefact is automatically out of scope.

Failure mode the boundary protects against: a contributor copies code
from an external project into bench, then `git push` -- without the
CI grep step, the external project name silently lands in the release
artifact. With the grep step (scanning the extracted tarball), CI
fails the push.
