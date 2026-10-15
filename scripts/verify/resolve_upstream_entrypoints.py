#!/usr/bin/env python3
"""resolve_upstream_entrypoints.py -- Phase A' prerequisite: resolve the
upstream entrypoint for the 96 ``kernel.py::placeholder`` metas.

The 96 placeholder tasks vendor the whole upstream file into kernel.py;
the real callable is in there but ``meta.yaml: oracle.gpu_triton`` still
points at the non-existent ``placeholder``. This module finds the host
wrapper (the Python function that launches the ``@triton.jit`` kernel)
and subjects it to a dual anchor before writing it to meta.yaml:

  1. **S2-smoke** (the real ``s2_verify_task``): build the staged
     candidate, run the CPU/``TRITON_INTERPRET=1`` staged-vs-original
     compare. Must progress past entrypoint resolution and the smoke
     compare (``UNRUN(needs-gpu)`` here = smoke passed; a ``BLOCKED``
     with a smoke/severing reason is a genuine per-task issue).
  2. **Oracle-consistency anchor**: invoke the resolved upstream
     entrypoint (via the same strict by-name signature adapter the S2
     runner uses, with ``infer/`` on sys.path so its vllm/sglang
     imports resolve to the frozen upstream source, CPU /
     ``TRITON_INTERPRET=1``) on **>=2 distinct shapes** (small + medium
     when the task has them) and compare every output tensor to
     ``oracle_pytorch.py::run`` for the same inputs within the task's
     ``tolerance_class``, also requiring the candidate output structure
     (arity/set of named tensor outputs) and each matched tensor's
     dtype to match the oracle's. A different structure / dtype / a
     wrong value on any sampled shape ⇒ wrong operator.

This is an empirical anchor, not a formal impossibility proof: a wrong
candidate that coincides with the oracle on the sampled shapes with
matching structure and dtype could still pass.

A resolvable callable that does not match the oracle is the wrong
entrypoint -> it is not written; the task is left ``placeholder`` with
an honest ``BLOCKED(no-verified-upstream-entrypoint: ...)`` reason
recording every candidate tried and why each was rejected.

Idempotent. Touches only ``meta.yaml: oracle.gpu_triton`` (a single
``  gpu_triton: kernel.py::<name>`` line); ``kernel.py`` is never
modified; every other meta field/byte is preserved.
"""
from __future__ import annotations

import argparse
import ast
import datetime
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SCRIPTS_DIR = REPO_ROOT / "scripts"
TASKS_ROOT = REPO_ROOT / "tasks"
INFER_ROOT = REPO_ROOT.parent / "infer"
VLLM_ROOT = INFER_ROOT / "vllm"
SGLANG_ROOT = INFER_ROOT / "sglang" / "python"
_ORACLE_CHECK_RUNNER = SCRIPTS_DIR / "verify" / "oracle_consistency_anchor_runner.py"

# Hard per-call timeout for the oracle-consistency subprocess. Without it,
# one hung candidate (CUDA init deadlock, infinite loop, vllm cold-import
# hang, kernel.py runaway) blocks the entire sweep indefinitely -- and
# parallel workers amplify the damage (one hung worker holds its slot).
# 180s is generous: real cases finish in <30s; the timeout protects
# against pathological hangs, not normal slow imports. A timeout maps
# to upstream-not-invocable-here in the worst-news-first classifier
# (the "timeout" substring matches neither _is_shapes_incomplete nor
# _is_numeric_mismatch, so the fallback bucket is correctly chosen).
#
# Override via TAB_ORACLE_TIMEOUT_S for the rare operator whose CPU-interpret
# anchor is legitimately slow (e.g. the einsum-heavy Mamba2 SSD varlen scan,
# whose median-volume anchor runs the interpreter element-by-element for
# minutes). This only raises the hang guard; it never weakens the numeric
# comparison. The GPU daemon (interpret=False, real CUDA) runs in milliseconds
# and never needs it.
_ORACLE_TIMEOUT_S = int(os.environ.get("TAB_ORACLE_TIMEOUT_S", "180"))

# Canonical resolver-verdict cache (in-repo, deterministic). The status
# generator (`phase_a_prime.py`) reads this path by default. Gitignored
# (machine-derived, never hand-written); regenerate with
# ``python3 scripts/verify/resolve_upstream_entrypoints.py --refresh-cache``
# (equivalent to ``--all --json <CANONICAL_RESOLVE_CACHE>``), which runs
# the full sweep under the current oracle-anchor + S2-smoke.
CANONICAL_RESOLVE_CACHE = REPO_ROOT / "results" / "resolve_all.json"


class StaleResolverCacheError(RuntimeError):
    """Raised when the resolver-verdict cache is stale / missing /
    unprovenanced, or the live/cached infer/ state is not a verifiable
    clean pinned git checkout, instead of consuming (or futilely
    regenerating) an untrustworthy cache. ``phase_a_prime`` aliases its
    ``StaleResolverCacheError`` to this class so one ``except`` catches
    both the code-fp/auto-regen path and the fail-closed infer/ path."""

# The cache JSON embeds a provenance stamp = sha256 of the complete
# resolver-relevant code surface that decides a verdict. The status
# generator validates the stamp against the current on-disk code on
# every read; a mismatch (the cache predates a resolver / anchor /
# runner change) is detected from the artifact itself and the cache is
# auto-regenerated under the current code before any status is derived.
#
# RESOLVER_TOOL_VERSION mirrors pyproject ``version`` so a packaging
# bump alone (no code-byte change) still invalidates an old stamp.
RESOLVER_TOOL_VERSION = "0.1.0"

# The set of files whose bytes decide a per-task verdict / honest
# category. The surface must cover every file a verdict-flip can come
# from, else a change to an uncovered file alters the verdict while the
# cache still validates as "fresh".
#   * resolve_upstream_entrypoints.py  -- classification + anchor
#     ordering + _is_numeric_mismatch (the wrong-operator predicate)
#   * _pa_oracle_check_runner.py       -- the oracle-consistency anchor
#     (>=2-shape / structure / dtype / value reject)
#   * _pa_subproc_runner.py            -- the S2-smoke runner driven by
#     s2_verify_task (the second half of every verdict)
#   * phase_a_prime.py                 -- _honest_category (maps the
#     resolver status+reason to the honest bucket) + s2_verify_task
#   * phase_a_freeze.py                --
#     phase_a_prime.py:58 does ``import phase_a_freeze as paf`` and
#     reuses its commit-pin / provenance / repo-root helpers on the
#     verdict path (paf.REPO_ROOT/TASKS_ROOT/VLLM_ROOT/SGLANG_ROOT at
#     module load, paf._commit_sha + paf._provenance_header in the
#     status flow). A byte change to those helpers can alter what the
#     verdict path sees, so its bytes are hashed too (7 -> 8 files).
#     No other module is imported on the verdict/category path that is
#     not already hashed -- the 4 scripts + 3 package files +
#     phase_a_freeze are the complete code surface
#     (resolve_task -> oracle_consistency_check[_pa_oracle_check_runner]
#     + s2_smoke_with_entrypoint[phase_a_prime.s2_verify_task ->
#     _pa_subproc_runner] -> _honest_category[phase_a_prime]; inputs/
#     tolerance/tolerance_policy feed the comparator; phase_a_freeze
#     supplies the pinned roots/helpers).
#   * ascendbench/input_spec/inputs.py    -- build_inputs() materialises
#     the tensors fed to both the oracle anchor
#     (_pa_oracle_check_runner.py:251) and the S2 runner
#     (_pa_subproc_runner.py:465). Different inputs -> a candidate can
#     match/diverge from the oracle differently -> verdict flip.
#   * ascendbench/correctness/tolerance.py -- compare()/load_policy() are
#     the oracle-mismatch decision (tolerance.py:128/34). A looser/
#     stricter compare flips PASS<->oracle-mismatch.
#   * ascendbench/policies/tolerance_policy.yaml -- the per-class
#     atol/rtol/bit_exact/inf_nan/sample_fraction values load_policy()
#     reads. A tolerance edit (config, not code) flips the verdict
#     identically to a code edit, so its bytes are hashed too.
# Not hashed (audited, not verdict-deciding): ascendbench/dataset.py +
# eligibility.py are imported only by
# phase_a_prime._score_eligible_count(), which feeds the status.md
# score-eligible reporting number, never a per-task resolver
# verdict/category (that is _honest_category(status, reason) -- pure
# string logic over the resolver outcome).
#
# UPSTREAM PROVENANCE. The resolver executes LIVE infer/ bytes --
# ``oracle_consistency_check`` (via ``_pa_oracle_check_runner.py``) and
# the S2 original side (``_pa_subproc_runner.py --side original``) put
# ``infer/vllm`` / ``infer/sglang/python`` on ``sys.path`` and import/run
# them to decide a task's verdict. ``meta.yaml`` only records
# ``upstream.commit_sha``/``file_sha256``; nothing else ties the live
# ``infer/`` tree to that pin, so an edit under ``infer/`` (e.g.
# removing a ``cbor2`` import) can flip a verdict while kernel.py,
# meta.yaml, shapes.yaml, oracle_pytorch.py, the code-fp and the
# per-task input_fp all stay unchanged. Two fields close this:
#   1. per-task ``upstream_fp`` -- sha256 over the sorted
#      ``(relpath, sha256-of-bytes)`` of the actual ``infer/`` files a
#      task's verdict path imported/executed: the UNION of both
#      live-upstream sides, the oracle-consistency subprocess closure
#      AND the S2 *original* subprocess closure (the S2-original side
#      also puts infer/ on sys.path and runs the unmodified upstream
#      kernel.py). Each side snapshots its own ``sys.modules`` under the
#      infer/ roots; their union is the complete per-task closure. A
#      task that imports nothing under infer/ gets the stable
#      empty-closure sentinel (still recorded, never None-by-accident).
#   2. global ``infer_state`` (in ``_provenance``) -- per upstream root
#      actually used: the CLEAN ``git -C <root> rev-parse HEAD`` only,
#      FAIL-CLOSED and clean-required. A non-git / git-missing /
#      rev-parse-or-status-error root, a DIRTY working tree, or a clean
#      checkout at the WRONG commit is recorded as an
#      ``<UNVERIFIABLE:...>`` sentinel: generation records the sentinel
#      and warns, read refuses the cache. The global git-HEAD gate is the primary
#      guarantee; the per-task ``upstream_fp`` union is the precise
#      belt, not a substitute for a clean pin.
# Whole multi-GB trees are NEVER hashed -- only the actual touched
# closure. ``upstream_fp`` is a PER-TASK field on each cached verdict
# (not part of this static code fingerprint).
_PACKAGE_DIR = REPO_ROOT / "ascendbench"
_PROVENANCE_CODE_FILES = (
    SCRIPTS_DIR / "verify" / "resolve_upstream_entrypoints.py",
    SCRIPTS_DIR / "verify" / "oracle_consistency_anchor_runner.py",
    SCRIPTS_DIR / "verify" / "severance_subprocess_runner.py",
    SCRIPTS_DIR / "verify" / "standalone_freeze_verifier.py",
    SCRIPTS_DIR / "verify" / "freeze_upstream_sources.py",
    _PACKAGE_DIR / "input_spec" / "inputs.py",
    _PACKAGE_DIR / "correctness" / "tolerance.py",
    _PACKAGE_DIR / "policies" / "tolerance_policy.yaml",
)

# ADOPT-1: the per-task verdict-deciding INPUT
# files. The code fingerprint above covers verdict-deciding CODE; this
# covers each task's OWN inputs. ``resolve_task`` reads exactly these
# four per task:
#   * kernel.py        -- candidate enumeration/ranking + the real
#                          upstream callable the anchor invokes
#   * meta.yaml         -- gpu_triton entrypoint + upstream
#                          commit_sha/file_sha256 (transitively pins
#                          the frozen upstream -- see REJECT note)
#   * shapes.yaml       -- the shapes the by-name adapter binds; a
#                          changed/added param flips
#                          shapes-incomplete <-> resolved
#   * oracle_pytorch.py -- ``oracle_pytorch.run`` IS the reference the
#                          oracle-consistency anchor compares against;
#                          a changed oracle flips oracle-mismatch
# ``contract.md`` is NOT here (audited: never read on the verdict
# path -- only kernel.py/meta.yaml/shapes.yaml/oracle_pytorch.py are).
# The per-task ``ascendbench/frozen/deps/**`` + ``ascendbench/frozen/shim/**``
# closures are populated ONLY by an ``--apply`` run; this resolver runs
# DRY-RUN (never --apply) so no task's verdict here references them
# (the dirs do not even exist on a dry-run box) -- per the
# disposition, when a task references none it is correct to hash just
# the four task files.
_PER_TASK_INPUT_FILES = (
    "kernel.py",
    "meta.yaml",
    "shapes.yaml",
    "oracle_pytorch.py",
)


def resolver_code_fingerprint() -> str:
    """sha256 over file name + bytes of each ``_PROVENANCE_CODE_FILES``
    entry (fixed order) plus the tool version. Any change -- to the
    resolver, oracle anchor, S2-smoke runner, honest-category mapper,
    input builder, tolerance comparator, tolerance-policy YAML, or the
    tool version -- makes a cache built under older code/config
    detectably stale."""
    h = hashlib.sha256()
    h.update(f"tool:{RESOLVER_TOOL_VERSION}\n".encode())
    for f in _PROVENANCE_CODE_FILES:
        h.update(f"file:{f.name}\n".encode())
        try:
            h.update(f.read_bytes())
        except OSError:
            # A missing resolver-relevant file is itself a provenance
            # CHANGE (the digest differs from any cache built when the
            # file existed) -- never silently treated as "unchanged".
            h.update(b"<MISSING>")
        h.update(b"\n--\n")
    return h.hexdigest()


def task_input_fingerprint(task_dir: Path) -> str:
    """ADOPT-1 -- sha256 over THIS task's
    verdict-deciding INPUT files (a per-task companion to the global
    ``resolver_code_fingerprint``).

    The code fingerprint catches a change to verdict-deciding CODE; it
    does NOT catch a change to a single task's OWN inputs. Without this
    a one-task edit (a fixed shapes.yaml, a corrected oracle, a
    re-frozen kernel.py) would flip THAT task's verdict while the
    global code fingerprint stays identical -- the stale entry would
    be silently consumed. So each cached verdict carries an
    ``input_fp`` validated per-task on read.

    Scheme: sorted ``(relpath, sha256-of-bytes)`` pairs over
    ``_PER_TASK_INPUT_FILES`` -> a single stable sha256. A MISSING
    input file is itself a provenance CHANGE (recorded as ``<MISSING>``
    -- never silently "unchanged"), so deleting/renaming an input also
    invalidates the entry. Deterministic + order-independent."""
    pairs: list[tuple[str, str]] = []
    for name in _PER_TASK_INPUT_FILES:
        p = task_dir / name
        try:
            digest = hashlib.sha256(p.read_bytes()).hexdigest()
        except OSError:
            digest = "<MISSING>"
        pairs.append((name, digest))
    h = hashlib.sha256()
    for rel, digest in sorted(pairs):
        h.update(f"{rel}\0{digest}\n".encode())
    return h.hexdigest()


# The live upstream source roots whose
# IMPORTED CLOSURE a verdict path executes. ``upstream_fp`` (per task)
# hashes the real touched closure under these, UNIONed across the
# oracle UNION S2-original sides (HOLE 3); ``infer_state``
# (global) records their CLEAN git HEAD ONLY and is FAIL-CLOSED
# clean-required (a non-git / DIRTY root => <UNVERIFIABLE:...> sentinel
# => the cache is refused, HOLE 1+2). These ARE the live
# verdict-deciding upstream bytes the old meta-sha claim failed to
# actually enforce.
_INFER_ROOTS = (
    ("vllm", VLLM_ROOT),
    ("sglang", SGLANG_ROOT),
)

# Stable sentinel for a task whose verdict path imported NOTHING under
# infer/ (pure no-callable / syntax-error / no-host-wrapper etc.). It is
# RECORDED (never None-by-accident) so the read side can still validate
# "this task's upstream closure was empty AND is still empty".
_EMPTY_UPSTREAM_CLOSURE_FP = (
    "sha256:empty-upstream-closure:"
    + hashlib.sha256(b"<no-infer-imports>").hexdigest()
)


def upstream_closure_fingerprint(files: list[str] | None) -> str:
    """sha256 over the sorted
    ``(relpath-under-infer-root, sha256-of-bytes)`` pairs of the ACTUAL
    ``infer/`` files a task's verdict path imported/executed.

    ``files`` is the absolute-path list captured at resolve time: the
    UNION (HOLE 3) of the ``sys.modules`` whose
    ``__file__`` is under ``infer/vllm`` / ``infer/sglang/python`` from
    BOTH live-upstream sides -- the oracle-consistency subprocess AND
    the S2 *original* subprocess (it too runs the unmodified upstream
    kernel.py with infer/ on sys.path). An
    EMPTY/None list (the task imported nothing upstream) maps to the
    stable ``_EMPTY_UPSTREAM_CLOSURE_FP`` sentinel -- still a recorded,
    validatable value, NEVER an accidental None. A file that has since
    been deleted hashes as ``<MISSING>`` (a provenance CHANGE, never
    silently "unchanged"). Deterministic + order-independent: identical
    closure bytes -> identical fp regardless of import order.

    This is the precise MINIMAL form
    -- whole multi-GB trees are NEVER hashed,
    only the actual touched closure, so the original precision concern
    is preserved while the live verdict-deciding bytes are validated."""
    if not files:
        return _EMPTY_UPSTREAM_CLOSURE_FP
    roots = [(name, str(p.resolve())) for name, p in _INFER_ROOTS]
    pairs: list[tuple[str, str]] = []
    seen: set[str] = set()
    for fp in files:
        try:
            ap = str(Path(fp).resolve())
        except OSError:
            ap = str(fp)
        rel: str | None = None
        for name, rp in roots:
            if ap == rp or ap.startswith(rp + "/"):
                rel = f"{name}/{ap[len(rp):].lstrip('/')}"
                break
        if rel is None or rel in seen:
            continue
        seen.add(rel)
        try:
            digest = hashlib.sha256(Path(ap).read_bytes()).hexdigest()
        except OSError:
            digest = "<MISSING>"
        pairs.append((rel, digest))
    if not pairs:
        # Every captured path resolved outside the infer/ roots (or
        # vanished) -> treat as empty closure (stable, validatable).
        return _EMPTY_UPSTREAM_CLOSURE_FP
    h = hashlib.sha256()
    h.update(b"upstream-closure-v1\n")
    for rel, digest in sorted(pairs):
        h.update(f"{rel}\0{digest}\n".encode())
    return "sha256:" + h.hexdigest()


# Sentinel prefix marking any infer/ root whose state is not an
# authoritatively-verified clean checkout at the expected HEAD.
# ``build_infer_state`` records it verbatim and ``load_resolver_cache``
# treats its presence (in the cache or the live state) as fail-closed:
# the cache is not usable-as-fresh and is never auto-regenerated into a
# trusted artifact -- a hard ``StaleResolverCacheError`` with an
# explicit pin-and-clean remedy. Rationale: the Phase-A provenance
# design requires a pinned reproducible upstream
# (ascendbench/policies/version_matrix.toml: "Any deviation invalidates
# baseline"), so a non-git / git-broken / DIRTY tree is refused rather
# than accepted as "fresh".
_INFER_UNVERIFIABLE_PREFIX = "<UNVERIFIABLE:"

# Expected per-root pin = the single distinct 40-hex
# ``upstream.commit_sha`` the corpus pins for tasks whose
# ``upstream.repo`` maps to that infer root (vllm-project/vllm -> the
# vllm pin, sgl-project/sglang -> the sglang pin). version_matrix.toml
# records the upstream repo identity only and defers the per-commit pin
# to the corpus, so the meta-consensus here is the pin authority. The
# literal ``snapshot-no-git`` value is a deliberate non-git snapshot
# sentinel (not a commit), excluded from the consensus; a repo mapping
# to no managed infer root (e.g. pytorch/FBGEMM) is irrelevant. If a
# root's consensus is not exactly one distinct real commit (corpus
# disagreement), the expected pin is itself UNVERIFIABLE -> fail-closed
# (we refuse to invent an authority). A clean checkout at the WRONG
# commit is as non-reproducible as a dirty tree and is fail-closed too.
_META_REPO_TO_INFER_ROOT = {
    "vllm-project/vllm": "vllm",
    "sgl-project/sglang": "sglang",
}
# Explicit deliberate "this task is a file snapshot, NOT a git checkout"
# sentinel in meta.yaml -- NOT a commit; excluded from the pin consensus.
_META_SNAPSHOT_SENTINEL = "snapshot-no-git"
_FULL_SHA_RE = re.compile(r"\A[0-9a-f]{40}\Z")


def expected_infer_pins() -> dict[str, str]:
    """The AUTHORITATIVE expected pinned
    commit per infer/ root, derived from the corpus consensus of
    ``tasks/*/*/meta.yaml`` ``upstream.commit_sha`` (version_matrix
    explicitly defers the per-commit pin to meta -- see the comment
    above; it documents repo identity only).

    Returns ``{root_name: expected}`` where ``expected`` is either the
    single distinct real 40-hex commit the corpus pins for that root, OR
    an ``<UNVERIFIABLE:...>`` sentinel when the corpus does not agree on
    exactly one real commit (no tasks, or >1 distinct real commit -- we
    refuse to invent an authority). The ``snapshot-no-git`` sentinel and
    any repo not mapping to a managed infer root are excluded. Cheap
    line-scan parse (no yaml dep needed; meta.yaml ``upstream:`` block is
    a flat 2-space-indented map -- the exact shape every meta.yaml uses).

    A managed repo whose task ``upstream.commit_sha`` is neither a valid
    40-hex SHA nor the explicit ``snapshot-no-git`` sentinel is garbage
    (truncated SHA, branch name, typo, placeholder, ...). It poisons
    that root's expected pin -> ``<UNVERIFIABLE:malformed-commit-sha:
    <value>>`` (fail-closed), taking precedence over any otherwise-valid
    consensus. ``snapshot-no-git`` stays the deliberate non-git
    exclusion (not malformed); only genuinely garbage values trigger
    this.
    """
    by_root: dict[str, set[str]] = {
        name: set() for name in _META_REPO_TO_INFER_ROOT.values()
    }
    # First malformed commit_sha seen per managed root
    # (its presence poisons that root's pin -> UNVERIFIABLE, never
    # silently dropped). Records (meta_relpath, raw_value) for an
    # actionable fail-closed message.
    malformed: dict[str, tuple[str, str]] = {}
    # First blank/absent commit_sha
    # seen per managed root -> distinct ``:blank-commit-sha:`` /
    # ``:absent-commit-sha:`` fail-closed sentinel (not silently
    # skipped). Records (meta_relpath, "blank"|"absent").
    blank: dict[str, tuple[str, str]] = {}
    for meta in sorted(TASKS_ROOT.glob("*/*/meta.yaml")):
        try:
            text = meta.read_text()
        except OSError:
            continue
        # Scope to the `upstream:` block; read its repo + commit_sha.
        # Track whether a ``commit_sha:`` key appeared at all, separately
        # from whether it had a non-blank value: a blank / whitespace /
        # quoted-empty / absent pin on a managed repo fails closed just
        # like a malformed one, never silently skipped.
        in_up = False
        repo: str | None = None
        sha: str | None = None
        sha_key_seen = False
        sha_raw = ""
        for line in text.splitlines():
            if re.match(r"\Aupstream:\s*\Z", line):
                in_up = True
                continue
            if in_up:
                if line and not line[0].isspace():
                    break  # left the upstream block
                mr = re.match(r"\s+repo:\s*(\S+)", line)
                if mr and repo is None:
                    repo = mr.group(1).strip().strip("'\"")
                # Match the KEY whether or not it has a value (blank /
                # whitespace / quoted-empty all count as "seen").
                mk = re.match(r"\s+commit_sha:(.*)$", line)
                if mk and not sha_key_seen:
                    sha_key_seen = True
                    sha_raw = mk.group(1).strip().strip("'\"").strip()
                    if sha_raw:
                        sha = sha_raw
        if repo is None:
            continue
        root = _META_REPO_TO_INFER_ROOT.get(repo)
        if root is None:
            continue  # repo maps to no managed infer root (irrelevant)
        # For a MANAGED repo a
        # commit_sha that is absent / blank / whitespace / quoted-empty
        # is JUST AS UNVERIFIABLE as a malformed one. It is NEVER
        # silently skipped: it poisons this root's pin (fail-closed),
        # taking precedence over any otherwise-valid consensus.
        if sha is None:
            if root not in blank and root not in malformed:
                try:
                    rel = str(meta.relative_to(REPO_ROOT))
                except ValueError:
                    rel = str(meta)
                blank[root] = (
                    rel, "blank" if sha_key_seen else "absent"
                )
            continue
        if sha == _META_SNAPSHOT_SENTINEL:
            continue  # deliberate non-git snapshot, not a commit
        if _FULL_SHA_RE.match(sha):
            by_root[root].add(sha)
        elif root not in malformed:
            # Neither a valid 40-hex SHA nor the
            # deliberate snapshot sentinel -> GARBAGE for a MANAGED
            # repo. Never silently dropped: poison this root's pin.
            try:
                rel = str(meta.relative_to(REPO_ROOT))
            except ValueError:
                rel = str(meta)
            malformed[root] = (rel, sha)
    pins: dict[str, str] = {}
    for name, shas in by_root.items():
        if name in blank:
            # A managed repo whose
            # commit_sha is blank/absent/whitespace/quoted-empty is
            # UNVERIFIABLE -- it FAILS CLOSED with its own distinct
            # sentinel and takes precedence over any otherwise-valid
            # consensus, exactly like a malformed pin. NEVER silently
            # skipped just because some other task pins a valid SHA.
            rel, kind = blank[name]
            pins[name] = (
                f"{_INFER_UNVERIFIABLE_PREFIX}{kind}-commit-sha:"
                f"{name}:in={rel}>"
            )
        elif name in malformed:
            # A malformed managed
            # commit_sha FAILS CLOSED and takes precedence over any
            # otherwise-valid consensus -- never silently ignored.
            rel, bad = malformed[name]
            pins[name] = (
                f"{_INFER_UNVERIFIABLE_PREFIX}malformed-commit-sha:"
                f"{name}:value={bad}:in={rel}>"
            )
        elif len(shas) == 1:
            pins[name] = next(iter(shas))
        elif not shas:
            pins[name] = (
                f"{_INFER_UNVERIFIABLE_PREFIX}no-meta-pin:{name}>"
            )
        else:
            pins[name] = (
                f"{_INFER_UNVERIFIABLE_PREFIX}ambiguous-meta-pin:"
                f"{name}:{len(shas)}-distinct>"
            )
    return pins


def _git_root_state(root: Path, name: str | None = None) -> str:
    """Fail-closed, clean-and-pinned upstream git state for one infer/
    root.

    Returns the resolved ``HEAD`` revision only when the root is a
    verified clean git checkout pinned at the expected upstream commit:

      * ``git rev-parse --is-inside-work-tree`` is ``true`` (a real git
        work tree -- not a bare repo / submodule gitdir edge);
      * ``git rev-parse HEAD`` resolves a real commit;
      * ``git status --porcelain`` is empty (no tracked modifications,
        no untracked files -- a clean checkout);
      * when ``name`` is given, the live HEAD equals the expected pin
        for that root (``expected_infer_pins()`` -- the corpus meta
        consensus). A clean checkout at the wrong commit is as
        non-reproducible as a dirty tree and is fail-closed. ``name`` is
        None only in self-contained unit tests of the non-git/dirty
        sentinels (no managed-root pin applies there).

    Any other outcome (git binary missing, not a git repo, rev-parse /
    status non-zero, a subprocess error/timeout, a non-empty porcelain
    i.e. a dirty tree, or a clean HEAD != the expected pin) returns an
    ``<UNVERIFIABLE:...>`` sentinel, recorded verbatim, which makes the
    cache fail-closed on read. Cheap (a few git plumbing calls per
    root); never hashes file contents (that is ``upstream_fp``'s
    per-task job)."""
    try:
        inside = subprocess.run(
            ["git", "-C", str(root), "rev-parse",
             "--is-inside-work-tree"],
            capture_output=True, text=True, timeout=20,
        )
        if inside.returncode != 0 or inside.stdout.strip() != "true":
            return (
                f"{_INFER_UNVERIFIABLE_PREFIX}not-a-git-work-tree:"
                f"{root.name}>"
            )
        head = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=20,
        )
        if head.returncode != 0 or not head.stdout.strip():
            return (
                f"{_INFER_UNVERIFIABLE_PREFIX}no-resolvable-HEAD:"
                f"{root.name}>"
            )
        rev = head.stdout.strip()
        porc = subprocess.run(
            ["git", "-C", str(root), "status", "--porcelain"],
            capture_output=True, text=True, timeout=30,
        )
        if porc.returncode != 0:
            return (
                f"{_INFER_UNVERIFIABLE_PREFIX}git-status-failed:"
                f"{root.name}>"
            )
        if porc.stdout.strip():
            # HOLE 2: a DIRTY upstream tree is by definition NOT a
            # reproducible pinned state -> refuse it (fail-closed),
            # NEVER record it as a comparable "+dirty" state.
            return (
                f"{_INFER_UNVERIFIABLE_PREFIX}dirty-not-pinned:"
                f"{root.name}>"
            )
        # HOLE 4: a CLEAN checkout at the WRONG
        # commit is NOT a reproducible pinned state. Validate the live
        # HEAD against the AUTHORITATIVE expected pin (meta consensus --
        # version_matrix defers the per-commit pin to meta). The OLD
        # code returned ``rev`` verbatim here and the read only checked
        # cached==live SELF-consistency, so a clean wrong-commit
        # produced a self-consistent cache that validated "fresh". Now
        # a wrong-commit (or an unverifiable expected pin: no/ambiguous
        # meta consensus) is fail-closed exactly like dirty/non-git.
        if name is not None:
            expected = expected_infer_pins().get(name)
            if expected is None or expected.startswith(
                _INFER_UNVERIFIABLE_PREFIX
            ):
                # Cannot establish an authoritative pin for this root
                # (no meta tasks / ambiguous corpus consensus). We
                # refuse to invent an authority -> fail-closed.
                return (
                    f"{_INFER_UNVERIFIABLE_PREFIX}no-authoritative-pin:"
                    f"{name}:{expected}>"
                )
            if rev != expected:
                return (
                    f"{_INFER_UNVERIFIABLE_PREFIX}wrong-commit:"
                    f"{name}:expected={expected}:got={rev}>"
                )
        return rev
    except (OSError, subprocess.SubprocessError):
        # git binary missing / timeout / spawn error: cannot
        # authoritatively determine the infer/ state -> fail-closed.
        return (
            f"{_INFER_UNVERIFIABLE_PREFIX}git-unavailable:"
            f"{root.name}>"
        )


def _infer_state_is_unverifiable(state: dict) -> str | None:
    """If ANY recorded infer/ root state is an
    ``<UNVERIFIABLE:...>`` sentinel, return a short reason; else None.
    Used by ``load_resolver_cache`` to FAIL CLOSED (HOLE 1+2): a cache
    whose embedded infer_state -- OR the CURRENT live infer_state --
    contains an unverifiable root cannot be trusted as fresh."""
    if not isinstance(state, dict):
        return "infer_state-not-a-dict"
    bad = sorted(
        f"{k}={v}"
        for k, v in state.items()
        if isinstance(v, str)
        and v.startswith(_INFER_UNVERIFIABLE_PREFIX)
    )
    return "; ".join(bad) if bad else None


# The UNVERIFIABLE sentinel
# kinds that mean "a managed task DID reference this root, but its pin
# could not be authoritatively resolved" (malformed / blank / absent /
# ambiguous). These roots ARE managed: a task pins them, so a
# bad-pin-AND-missing-on-disk root MUST fail closed (missing-root), not
# be silently omitted. ``no-meta-pin`` is deliberately EXCLUDED: it
# means NO task referenced the root, so there is no live-upstream
# verdict to protect and forcing a redundant ``missing-root`` for a
# root nobody pins would be noise (a root no managed task references --
# e.g. FBGEMM -- never appears in ``expected_infer_pins()`` at all).
_REFERENCED_UNVERIFIABLE_PIN_KINDS = (
    "malformed-commit-sha:",
    "blank-commit-sha:",
    "absent-commit-sha:",
    "ambiguous-meta-pin:",
)


def _managed_root_names() -> set[str]:
    """The infer/ roots a managed corpus task references and that must
    therefore be present-and-pinned on disk: the union of roots with a
    valid 40-hex SHA pin and roots a managed task references but whose
    pin is UNVERIFIABLE for a referenced reason (malformed / blank /
    absent / ambiguous -- ``_REFERENCED_UNVERIFIABLE_PIN_KINDS``).
    ``no-meta-pin`` (no task references the root) is excluded -- a
    redundant ``missing-root`` would be noise, and it still fails closed
    via its own sentinel if the root exists. A bad-pin-and-missing
    managed root thus fails closed (``<UNVERIFIABLE:missing-root:...>``),
    never silently omitted from ``build_infer_state``."""
    managed: set[str] = set()
    for name, sha in expected_infer_pins().items():
        if _FULL_SHA_RE.match(sha):
            managed.add(name)
            continue
        if any(
            kind in sha for kind in _REFERENCED_UNVERIFIABLE_PIN_KINDS
        ):
            managed.add(name)
    return managed


def build_infer_state() -> dict:
    """The repo-global fail-closed infer/ state guard embedded in
    ``_provenance``. One ``<name>: <clean-pinned-git-HEAD>`` entry per
    upstream root that exists on disk -- or an ``<UNVERIFIABLE:...>``
    sentinel for any root that is not a verified clean checkout pinned
    at the expected commit (non-git / git missing / rev-parse/status
    error / DIRTY tree / clean-but-wrong-commit), or a
    ``<UNVERIFIABLE:missing-root:...>`` sentinel for a managed root (one
    the corpus meta-consensus pins -- ``_managed_root_names()``) that is
    ABSENT on this box. An unmanaged root that is absent is correctly
    omitted (no live-upstream verdict to protect).

    Belt-and-suspenders: an infer/ change outside any single task's
    captured ``upstream_fp`` closure (or a whole checkout swap) flips
    the recorded HEAD -> the whole cache is invalidated on read; and an
    unverifiable / dirty / non-git / missing managed infer/ makes the
    cache hard-fail -- a non-reproducible or absent managed upstream
    tree cannot back a trustworthy verdict."""
    managed = _managed_root_names()
    state: dict[str, str] = {}
    for name, root in _INFER_ROOTS:
        if root.is_dir():
            # HOLE 4: pass the root NAME so the live HEAD is
            # validated against the AUTHORITATIVE expected pin (meta
            # consensus), not merely recorded verbatim.
            state[name] = _git_root_state(root, name)
        elif name in managed:
            # A MANAGED root (meta-consensus
            # pins it) that is ABSENT is FAIL-CLOSED, never a silent
            # omission that would self-consistently validate "fresh".
            state[name] = (
                f"{_INFER_UNVERIFIABLE_PREFIX}missing-root:{name}>"
            )
    return state


def build_provenance_stamp() -> dict:
    """The provenance block embedded at the top of the cache JSON.
    ``code_fingerprint`` + ``infer_state`` are the validated-on-read
    fields; the rest is human-auditable context."""
    return {
        "code_fingerprint": resolver_code_fingerprint(),
        "infer_state": build_infer_state(),
        "tool_version": RESOLVER_TOOL_VERSION,
        "generated_at": datetime.datetime.now(
            datetime.timezone.utc
        ).isoformat(),
        "code_files": [f.name for f in _PROVENANCE_CODE_FILES],
        "note": (
            "Regenerate with "
            "`python3 scripts/verify/resolve_upstream_entrypoints.py "
            "--refresh-cache`. NEVER hand-edit. If "
            "code_fingerprint != resolver_code_fingerprint() the cache "
            "is STALE and the status generator auto-regenerates it "
            "under the CURRENT code (a stale cache is never silently "
            "consumed)."
        ),
    }


PLACEHOLDER = "placeholder"

# Marker the strict by-name adapter raises when shapes.yaml omits a
# required no-default param (a benchmark GAP, not a wrong operator).
_SHAPES_INCOMPLETE = "BLOCKED(shapes-incomplete"


def _is_shapes_incomplete(detail: str) -> bool:
    return _SHAPES_INCOMPLETE in (detail or "") or (
        "by-name bind failed" in (detail or "")
        and "ShapesIncompleteError" in (detail or "")
    )


def _primary_blocker_cat(note: str) -> str:
    """Categorise the BEST-RANKED candidate's reject note into that
    candidate's real blocker. Distinguishes ``oracle-consistent-*`` (the
    candidate matched the oracle and only smoke/shapes blocked -- a likely
    dev-host env limit that may run on the GPU) from a genuine wrong-op /
    shapes / invocability failure. The best-ranked candidate is a heuristic
    score-ordered guess at the intended wrapper, not a guarantee. Mirrors
    the exact note phrasings emitted in resolve_task's candidate loop."""
    n = note or ""
    if "oracle-consistent but S2-smoke BLOCKED" in n:
        return "oracle-consistent-smoke-blocked"
    if "oracle-consistent but S2 shapes-incomplete" in n:
        return "oracle-consistent-shapes-incomplete"
    if ": shapes-incomplete" in n:
        return "shapes-incomplete"
    if ": oracle-mismatch" in n:
        return "oracle-mismatch"
    if ": upstream-not-invocable-here" in n:
        return "upstream-not-invocable-here"
    if ": smoke-fail" in n or "S2-smoke BLOCKED" in n:
        return "smoke-fail"
    return "other"


def _is_numeric_mismatch(detail: str) -> bool:
    """True ONLY for a genuine wrong-operator signal: the candidate
    RAN and produced output but the values / shape / dtype / STRUCTURE
    do NOT match the oracle within ``tolerance_class``. An exception
    while invoking the upstream callable (triton-version
    AttributeError, missing dep, etc.) is NOT a wrong-operator -- it is
    an env/upstream limitation that S2's original side hits identically
    (categorised separately so we never mislabel a runnable-elsewhere
    kernel as 'wrong').

    The hardened anchor's
    ``_check_one_shape`` rejects a candidate that produced output of the
    WRONG ARITY/STRUCTURE with ``STRUCTURE MISMATCH: ... ⇒ wrong
    operator`` (e.g. L1_01: candidate emits oracle-outputs-PLUS-extras).
    That is the *strongest* wrong-operator signal -- the candidate ran
    fine, it is just NOT the same operator. The OLD predicate omitted
    this phrasing, so a structure-mismatch fell through to
    ``upstream-not-invocable-here`` and was DISHONESTLY labelled an
    "ENVIRONMENT limitation". A structural/dtype/value mismatch is an
    ``oracle-mismatch`` -- recognise the structure-mismatch + generic
    ``⇒ wrong operator`` marker at the SOURCE so the cache status itself
    is the honest category (the status-gen backstop is defence in
    depth, not the only line)."""
    d = detail or ""
    return (
        "NOT within" in d
        or "no candidate output with shape" in d
        or "oracle produced no tensors" in d
        or "STRUCTURE MISMATCH" in d
        or "⇒ wrong operator" in d
        or "wrong-operator signal" in d
    )


# ───────────────────────────────────────────────────────────────────────────
# Candidate enumeration (AST-only, GPU-free, side-effect-free)
# ───────────────────────────────────────────────────────────────────────────
@dataclass
class Candidate:
    name: str
    is_host_wrapper: bool          # launches a @triton.jit kernel
    launched_jit: list[str] = field(default_factory=list)
    params: list[str] = field(default_factory=list)
    required_params: list[str] = field(default_factory=list)
    score: float = 0.0
    rationale: str = ""


def _jit_decorated_names(tree: ast.Module) -> set[str]:
    """Top-level functions decorated with ``@triton.jit`` (or
    ``@triton.jit(...)``)."""
    out: set[str] = set()
    for n in tree.body:
        if not isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for dec in n.decorator_list:
            d = dec.func if isinstance(dec, ast.Call) else dec
            # triton.jit  /  jit  /  triton.language.jit
            if isinstance(d, ast.Attribute) and d.attr == "jit":
                out.add(n.name)
            elif isinstance(d, ast.Name) and d.id == "jit":
                out.add(n.name)
    return out


def _launched_jit_kernels(fn: ast.AST, jit_names: set[str]) -> list[str]:
    """Names of ``@triton.jit`` kernels launched via the ``kern[grid](...)``
    subscript-call idiom inside ``fn``."""
    launched: list[str] = []
    for node in ast.walk(fn):
        if not isinstance(node, ast.Call):
            continue
        f = node.func
        if isinstance(f, ast.Subscript):
            base = f.value
            if isinstance(base, ast.Name) and base.id in jit_names:
                launched.append(base.id)
            elif isinstance(base, ast.Attribute) and base.attr in jit_names:
                launched.append(base.attr)
    return launched


def _params(fn: ast.AST) -> tuple[list[str], list[str]]:
    """(all positional/keyword param names, required-no-default names).
    ``*args`` / ``**kwargs`` are ignored (the by-name adapter skips
    them)."""
    a = fn.args
    pos = list(a.posonlyargs) + list(a.args) + list(a.kwonlyargs)
    names = [p.arg for p in pos]
    defaults = list(a.defaults)
    n_req_pos = (len(a.posonlyargs) + len(a.args)) - len(defaults)
    required: list[str] = []
    for i, p in enumerate(a.posonlyargs + a.args):
        if i < n_req_pos:
            required.append(p.arg)
    for p, d in zip(a.kwonlyargs, a.kw_defaults, strict=False):
        if d is None:
            required.append(p.arg)
    return names, required


def enumerate_candidates(src: str) -> list[Candidate]:
    """All resolvable entrypoint candidates in ``kernel.py``.

    A *host wrapper* is a top-level Python function that launches a
    ``@triton.jit`` kernel via ``kern[grid](...)``. A *bare jit* kernel
    is a ``@triton.jit`` function itself (only a candidate when no host
    wrapper launches it -- a jit kernel is not directly invocable with
    high-level tensors, but the runner could still resolve the name)."""
    tree = ast.parse(src)
    jit_names = _jit_decorated_names(tree)
    cands: list[Candidate] = []
    launched_anywhere: set[str] = set()
    for n in tree.body:
        if not isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if n.name in jit_names:
            continue  # handled below as bare-jit fallback
        launched = _launched_jit_kernels(n, jit_names)
        if launched:
            launched_anywhere.update(launched)
            names, req = _params(n)
            cands.append(
                Candidate(
                    name=n.name,
                    is_host_wrapper=True,
                    launched_jit=launched,
                    params=names,
                    required_params=req,
                )
            )
    # Bare @triton.jit kernels: only as a last resort, and only those
    # NOT already launched by a host wrapper candidate.
    for n in tree.body:
        if (
            isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
            and n.name in jit_names
        ):
            names, req = _params(n)
            cands.append(
                Candidate(
                    name=n.name,
                    is_host_wrapper=False,
                    launched_jit=[n.name],
                    params=names,
                    required_params=req,
                )
            )
    return cands


# ───────────────────────────────────────────────────────────────────────────
# Ranking signals
# ───────────────────────────────────────────────────────────────────────────
def _provenance_kernel(src: str) -> str | None:
    m = re.search(r"Kernel function:\s*`([^`]+)`", src)
    return m.group(1).strip() if m else None


def _shapes_input_names(task_dir: Path) -> set[str]:
    import yaml

    sp = task_dir / "shapes.yaml"
    if not sp.is_file():
        return set()
    try:
        doc = yaml.safe_load(sp.read_text())
    except (OSError, ValueError):
        return set()
    # input NAMES via the adapter (knob branch == doc["inputs"].keys(); explicit reconstructs them).
    from ascendbench.input_spec.shape_format import input_specs
    return set(input_specs(doc) or {})


def _name_core(task_id: str) -> str:
    """``L2_14_bmm_chunk_fwd`` -> ``bmm_chunk_fwd_kernel``."""
    return re.sub(r"^[LT]\d+_\d+_", "", task_id)


def rank_candidates(
    cands: list[Candidate],
    task_dir: Path,
    src: str,
) -> list[Candidate]:
    """Score each candidate. The decisive proof is still the
    oracle-consistency check; ranking only orders which candidate to
    *try first* so the verifier converges fast.

    Signals (additive):
      * host wrapper                                       (+5.0)
      * every shapes.yaml input that is also a wrapper
        param name (the by-name adapter REQUIRES this)     (+3.0 each)
      * NO required-no-default param missing from shapes
        (a missing one is a guaranteed shapes-incomplete)  (+4.0)
      * launches the provenance "Kernel function:" jit     (+6.0)
      * name relates to the task dir / provenance kernel   (+2.0)
    """
    shp = _shapes_input_names(task_dir)
    prov = _provenance_kernel(src)
    core = _name_core(task_dir.name).lower()
    for c in cands:
        s = 0.0
        why: list[str] = []
        if c.is_host_wrapper:
            s += 5.0
            why.append("host-wrapper")
        pset = set(c.params)
        overlap = shp & pset
        s += 3.0 * len(overlap)
        if overlap:
            why.append(f"shapes∩params={sorted(overlap)}")
        missing_req = [p for p in c.required_params if p not in shp]
        if c.required_params and not missing_req:
            s += 4.0
            why.append("all-required-in-shapes")
        elif missing_req:
            why.append(f"missing-required={missing_req}")
        if prov and prov in c.launched_jit:
            s += 6.0
            why.append(f"launches-provenance-jit({prov})")
        cn = c.name.lower().lstrip("_")
        if prov and (cn in prov.lower() or prov.lower().lstrip("_") in cn):
            s += 2.0
            why.append("name~provenance")
        # task dir core often == "<op>_kernel"; the wrapper is usually
        # the op name WITHOUT the trailing _kernel.
        op = core[: -len("_kernel")] if core.endswith("_kernel") else core
        if op and (op in cn or cn in op):
            s += 2.0
            why.append("name~taskdir")
        c.score = s
        c.rationale = "; ".join(why)
    return sorted(cands, key=lambda c: (-c.score, not c.is_host_wrapper, c.name))


# ───────────────────────────────────────────────────────────────────────────
# meta.yaml surgery -- ONLY the gpu_triton line, byte-preserving
# ───────────────────────────────────────────────────────────────────────────
_GT_RE = re.compile(r"^(\s*gpu_triton:\s*)kernel\.py::(\S+)\s*$", re.M)


def read_gpu_triton(meta_path: Path) -> str | None:
    m = _GT_RE.search(meta_path.read_text())
    return m.group(2) if m else None


def set_gpu_triton(meta_path: Path, new_name: str) -> bool:
    """Replace ONLY the ``gpu_triton: kernel.py::<name>`` line in place.
    Preserves every other byte. Idempotent. Returns True iff written."""
    txt = meta_path.read_text()
    m = _GT_RE.search(txt)
    if not m:
        return False
    if m.group(2) == new_name:
        return False  # already correct -> idempotent no-op
    new_txt = txt[: m.start()] + f"{m.group(1)}kernel.py::{new_name}" + txt[m.end():]
    tmp = meta_path.with_suffix(".yaml.tmp")
    tmp.write_text(new_txt)
    tmp.replace(meta_path)
    return True


# ───────────────────────────────────────────────────────────────────────────
# Oracle-consistency anchor (subprocess, infer/ on path, GPU-free)
# ───────────────────────────────────────────────────────────────────────────
def _gpu_available() -> bool:
    """True when a real CUDA/NPU device is present -- the gate for the
    oracle-consistency GPU fallback. On a no-CUDA dev host this is False, so
    the fallback never engages and resolve behaviour is unchanged."""
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def oracle_consistency_check(
    task_dir: Path, entrypoint: str, interpret: bool = True,
) -> tuple[bool, str, list[str]]:
    """Run the upstream ``entrypoint`` (real frozen upstream code,
    CPU/``TRITON_INTERPRET=1``) on >=2 distinct shapes (small + medium
    when available) and compare every output tensor -- plus the output
    STRUCTURE (arity) and dtype -- to ``oracle_pytorch.run`` for the
    SAME inputs within the task's ``tolerance_class``. Returns
    ``(consistent, detail, upstream_closure)`` where ``upstream_closure``
    is the absolute paths of the ACTUAL ``infer/`` files THIS
    oracle-consistency subprocess imported/executed (the
    real imported-module closure captured in the
    subprocess that ran it). HOLE 3: the caller
    UNIONs this with the S2 *original* subprocess closure before
    hashing the per-task ``upstream_fp`` -- this is the oracle SIDE of
    the complete closure, not the whole of it.

    This is a STRONG anti-fake anchor: a resolvable callable that does
    not match the oracle in value, structure or dtype on the sampled
    shapes is the WRONG operator and is rejected. It is NOT a formal
    impossibility proof -- a wrong candidate coinciding on the sampled
    shapes with matching structure/dtype could in principle pass; we do
    not overclaim it is "impossible" to set a wrong entrypoint.
    """
    work = Path(tempfile.mkdtemp(prefix=f"pa_oracle_{task_dir.name}_"))
    spec = {
        "task_dir": str(task_dir),
        "entrypoint": entrypoint,
        "out_json": str(work / "out.json"),
        "vllm_root": str(VLLM_ROOT),
        "sglang_root": str(SGLANG_ROOT),
        "repo_root": str(REPO_ROOT),
        "interpret": interpret,
    }
    spec_f = work / "spec.json"
    spec_f.write_text(json.dumps(spec))
    try:
        proc = subprocess.run(
            [sys.executable, str(_ORACLE_CHECK_RUNNER), "--spec", str(spec_f)],
            capture_output=True, text=True,
            timeout=_ORACLE_TIMEOUT_S,
        )
    except subprocess.TimeoutExpired:
        # A CPU-interpret timeout is a tooling SPEED skew, not an operator
        # defect: a heavy op (e.g. a per-expert batched GEMM over thousands
        # of program instances x K-loop tiles) is simply too slow to
        # Python-interpret in the per-candidate budget, while the SAME op on
        # the real device runs in milliseconds. When a GPU is present, retry
        # the SAME anchor ONCE on device (interpret=False) with the task's
        # UNCHANGED tolerance -- the same rescue mechanism the completed-run
        # interpret_error / narrow_float_output paths use below. A wrong
        # operator still fails there (tolerance is not relaxed); only the
        # interpret-speed limitation is corrected. interpret=False cannot
        # re-enter this branch (its runner subprocess never times out the
        # same way -- it runs on device, not the interpreter) -> fail-closed.
        # On a no-GPU host the rescue does not engage and the fail-closed
        # BLOCKED verdict is preserved exactly as before.
        if interpret and _gpu_available():
            g_ok, g_why, g_closure = oracle_consistency_check(
                task_dir, entrypoint, interpret=False
            )
            suffix = (
                f" [via GPU re-confirm: CPU-interpret timeout after "
                f"{_ORACLE_TIMEOUT_S}s -> {'PASS' if g_ok else 'FAIL'}]"
            )
            return g_ok, g_why + suffix, g_closure
        return False, (
            f"oracle-check subprocess timeout after {_ORACLE_TIMEOUT_S}s "
            f"(per-candidate hard limit -- a hung subprocess would "
            f"otherwise stall the whole sweep). Treated as "
            f"upstream-not-invocable-here."
        ), []
    out_f = work / "out.json"
    if not out_f.is_file():
        # No result file -> the subprocess crashed before it could
        # capture its closure. An UNKNOWN closure is NOT "empty": return
        # the empty list so the caller records the empty sentinel only
        # when the run genuinely imported nothing; here the verdict
        # itself is a hard error, surfaced to the caller as not-consistent.
        return False, (
            f"oracle-check subprocess produced no result (rc="
            f"{proc.returncode}): {proc.stderr[-400:]}"
        ), []
    try:
        res = json.loads(out_f.read_text())
    except (OSError, ValueError) as e:
        return False, f"oracle-check result unreadable: {e}", []
    closure = res.get("upstream_closure")
    closure = [str(p) for p in closure] if isinstance(closure, list) else []
    _interp_err = bool(res.get("interpret_error"))
    _narrow_out = bool(res.get("narrow_float_output"))
    if interpret and (_interp_err or _narrow_out) and _gpu_available():
        # Two interpret-only tooling skews trigger the SAME GPU rescue:
        #   - interpret_error: the interpret pass threw a triton
        #     InterpreterError (e.g. a grid-stride `tl.range(...).to()`).
        #   - narrow_float_output: the task writes a narrowing float dtype
        #     (fp8/bf16/fp16). The interpreter rounds these stores wrong --
        #     fp32->fp8 mis-rounds both ways (127.2 -> 64.0 instead of 128.0),
        #     fp32->bf16/fp16 truncates toward zero instead of RNE (100.3 ->
        #     100.0 instead of 100.5) -- so the verdict is untrustworthy whether
        #     it PASSED or FAILED (false-FAIL a correct kernel, false-PASS a
        #     wrong one rounding into the oracle value). Re-confirm on the real
        #     device, which rounds correctly, regardless of the interpret verdict.
        # Re-run the SAME anchor ONCE on GPU and adopt its verdict; union both
        # import closures so upstream_fp stays complete. The GPU re-run keeps the
        # task's UNCHANGED tolerance, so it only corrects the rounding model -- a
        # wrong operator still fails there. interpret=False cannot re-enter this
        # branch (fires at most once) -> fail-closed.
        g_ok, g_why, g_closure = oracle_consistency_check(
            task_dir, entrypoint, interpret=False
        )
        merged = sorted(set(closure) | set(g_closure))
        _why = "narrow-float interpret-unreliable" if (_narrow_out and not _interp_err) else (
            "interpret InterpreterError"
        )
        suffix = (
            f" [via GPU re-confirm: {_why} -> "
            f"{'PASS' if g_ok else 'FAIL'}]"
        )
        return g_ok, g_why + suffix, merged
    return bool(res.get("consistent")), str(res.get("detail", "")), closure


# ───────────────────────────────────────────────────────────────────────────
# S2-smoke (the real s2_verify_task) -- entrypoint resolved + smoke pass
# ───────────────────────────────────────────────────────────────────────────
def _load_pap():
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "phase_a_prime", SCRIPTS_DIR / "verify" / "standalone_freeze_verifier.py"
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("phase_a_prime", mod)
    spec.loader.exec_module(mod)
    return mod


def s2_smoke_with_entrypoint(
    task_dir: Path, entrypoint: str, pap,
) -> tuple[str, str, list[str]]:
    """Run the REAL ``s2_verify_task`` (build staged candidate +
    CPU/TRITON_INTERPRET smoke compare) with ``gpu_triton`` set to
    ``entrypoint`` -- AGAINST A TEMP COPY of the task dir, so the real
    ``meta.yaml`` is NEVER mutated during probing. Returns
    ``(status, reason, s2_original_upstream_closure)`` -- status in
    {PASS, UNRUN, BLOCKED}; the third element
    (HOLE 3) is the absolute infer/ paths the S2 *original* side
    imported running the UNMODIFIED upstream kernel.py, which the caller
    UNIONs into the per-task ``upstream_fp`` (the oracle side alone did
    NOT cover an infer/ file imported only on the S2-original path).

    CRASH-SAFETY (anti-fake durability): an earlier design mutated the
    real meta.yaml then restored in ``finally``; a SIGKILL mid-S2
    skipped ``finally`` and left a FAKE unverified entrypoint persisted.
    Probing a temp copy means a kill at ANY point leaves the real
    task dir byte-identical -- the real meta is touched ONLY by the
    explicit ``set_gpu_triton`` on ``--apply``, AFTER both checks pass.

    ``UNRUN`` = smoke PASSED + no GPU here (the honest good outcome on
    this box). ``BLOCKED`` = entrypoint did not resolve / smoke
    mismatch / shapes-incomplete -- a genuine per-task issue, NEVER
    papered over."""
    import shutil

    tmp_root = Path(
        tempfile.mkdtemp(prefix=f"pa_s2probe_{task_dir.name}_")
    )
    tmp_td = tmp_root / task_dir.name
    tmp_td.mkdir()
    # s2_verify_task reads kernel.py + meta.yaml + shapes.yaml, plus
    # _pa_launcher.py for jit-only tasks (the loaders exec it). Copy them
    # byte-exact; only the temp meta's gpu_triton line is edited.
    for fn in ("kernel.py", "meta.yaml", "shapes.yaml", "_pa_launcher.py"):
        src = task_dir / fn
        if src.is_file():
            shutil.copy2(src, tmp_td / fn)
    set_gpu_triton(tmp_td / "meta.yaml", entrypoint)
    try:
        res = pap.s2_verify_task(tmp_td)
        s2_cl = list(
            getattr(res, "s2_original_upstream_closure", []) or []
        )
        return res.status, res.reason, s2_cl
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)


# ───────────────────────────────────────────────────────────────────────────
# Per-task resolution
# ───────────────────────────────────────────────────────────────────────────
@dataclass
class Resolution:
    task_id: str
    resolved: bool
    entrypoint: str | None
    status: str           # UNRUN | PASS | BLOCKED(...)
    reason: str
    tried: list[str] = field(default_factory=list)
    # The PRIMARY (best-ranked) candidate's real blocker, recorded
    # separately from the headline ``status`` bucket. The headline bucket
    # uses worst-news-first across ALL tried candidates, so one sibling's
    # ``shapes-incomplete`` can mask the fact that the INTENDED wrapper was
    # actually oracle-consistent and only smoke/env-blocked (e.g. a dev-host
    # vllm cold-import SIGKILL that would run on the GPU). This field names
    # the intended candidate's status so triage does not mistake a
    # close-to-passing task for a genuine shapes gap. Empty when resolved or
    # no candidate was tried.
    primary_blocker: str = ""
    # The per-task upstream-closure
    # fingerprint: sha256 over the ACTUAL infer/ files THIS task's
    # verdict path imported/executed, UNIONed across BOTH live-upstream
    # sides (oracle-consistency UNION S2-original, HOLE 3) -- or
    # the stable empty-closure sentinel if none. Stamped onto every
    # cached verdict; re-validated on read so a live infer/ edit that
    # flips a verdict can no longer slip past a "fresh" cache. Defaults
    # to the empty sentinel (NEVER None-by-accident).
    upstream_fp: str = _EMPTY_UPSTREAM_CLOSURE_FP


def resolve_task(task_dir: Path, pap, apply: bool) -> Resolution:
    kp = task_dir / "kernel.py"
    meta_path = task_dir / "meta.yaml"
    cur = read_gpu_triton(meta_path)
    src = kp.read_text(encoding="utf-8", errors="replace")

    # Accumulate the UNION of every infer/
    # file ANY live-upstream subprocess imported while deciding THIS
    # task's verdict: each candidate's oracle-consistency subprocess
    # closure AND (HOLE 3) the S2 *original* subprocess
    # closure (returned as ``s2_smoke_with_entrypoint``'s 3rd element).
    # The union is the COMPLETE set of live upstream bytes this task's
    # verdict path executed -- NOT just the oracle side (an infer/ file
    # imported only on the S2-original path was previously invisible).
    # Hashed into ``upstream_fp`` on EVERY return path so the recorded
    # fp reflects the real closure that produced the verdict (empty
    # sentinel if nothing imported).
    _closure_acc: set[str] = set()

    def _fp() -> str:
        return upstream_closure_fingerprint(sorted(_closure_acc))

    if cur is not None and cur != PLACEHOLDER:
        # meta already points at a non-placeholder entrypoint (the
        # pre-existing 5, OR a task an EARLIER --apply run resolved).
        #
        # The OLD code returned
        # ``ALREADY-RESOLVED`` here WITHOUT running the anchor -- so the
        # cache recorded a "resolved" verdict carrying ZERO post-anchor
        # evidence. If that entrypoint had been applied by a
        # PRE-hardening run, the cache (and the docs derived from it)
        # would claim a verification STRENGTH the artifact never
        # evidenced (the stale-cache overclaim). We now RE-VERIFY the
        # already-set entrypoint against the CURRENT (hardened)
        # oracle-consistency anchor so the cached verdict genuinely
        # reflects the post-hardening check and records the real
        # >=2-shape / structure / dtype evidence. meta.yaml is NEVER
        # mutated on this path (cur is already non-placeholder); only
        # the verdict + its honest reason are derived.
        oc_ok, oc_why, oc_cl = oracle_consistency_check(task_dir, cur)
        _closure_acc.update(oc_cl)
        if oc_ok:
            st, why, s2_cl = s2_smoke_with_entrypoint(
                task_dir, cur, pap
            )
            # HOLE 3 -- UNION the S2-original
            # side's imported infer/ closure into the per-task
            # accumulator (the belt now reflects oracle UNION S2-original).
            _closure_acc.update(s2_cl)
            if st in ("UNRUN", "PASS"):
                return Resolution(
                    task_dir.name, True, cur,
                    "UNRUN(needs-gpu)" if st == "UNRUN" else "PASS",
                    f"meta already points at kernel.py::{cur} "
                    f"(untouched) AND re-verified under the CURRENT "
                    f"hardened anchor: oracle-consistent [{oc_why}] AND "
                    f"S2-smoke {st} [{why[:160]}]",
                    upstream_fp=_fp(),
                )
            # entrypoint set but its real S2-smoke now BLOCKS -> honest.
            if _is_shapes_incomplete(why):
                cat = "shapes-incomplete"
            else:
                cat = "smoke-fail"
            return Resolution(
                task_dir.name, False, None,
                f"BLOCKED(no-verified-upstream-entrypoint: {cat})",
                f"meta points at kernel.py::{cur} but under the CURRENT "
                f"hardened anchor it is oracle-consistent yet its REAL "
                f"S2-smoke now {cat} [{why[:200]}]",
                upstream_fp=_fp(),
            )
        # The already-set entrypoint does NOT pass the CURRENT hardened
        # oracle anchor: the honest verdict CHANGES -- never preserve a
        # stale pass to keep the number up (the anti-fake rule).
        if _is_shapes_incomplete(oc_why):
            cat = "shapes-incomplete"
        elif _is_numeric_mismatch(oc_why):
            cat = "oracle-mismatch"
        else:
            cat = "upstream-not-invocable-here"
        return Resolution(
            task_dir.name, False, None,
            f"BLOCKED(no-verified-upstream-entrypoint: {cat})",
            f"meta points at kernel.py::{cur} but it does NOT pass the "
            f"CURRENT hardened oracle-consistency anchor ({cat}) -- the "
            f"honest verdict CHANGES; a stale pre-hardening pass is NOT "
            f"preserved [{oc_why}]",
            upstream_fp=_fp(),
        )

    try:
        cands = rank_candidates(
            enumerate_candidates(src), task_dir, src
        )
    except SyntaxError as e:
        return Resolution(
            task_dir.name, False, None,
            "BLOCKED(no-verified-upstream-entrypoint)",
            f"kernel.py SyntaxError line {e.lineno}: {e.msg}",
            upstream_fp=_fp(),
        )

    if not cands:
        return Resolution(
            task_dir.name, False, None,
            "BLOCKED(no-verified-upstream-entrypoint)",
            "no host wrapper launches a @triton.jit kernel and no bare "
            "@triton.jit kernel is defined (no-host-wrapper)",
            upstream_fp=_fp(),
        )

    tried: list[str] = []
    reject_notes: list[str] = []
    # Only try the host wrappers + (last) a bare jit; cap the breadth so
    # a 21-kernel pipeline does not explode -- the ranking puts the
    # right one first when it exists.
    host = [c for c in cands if c.is_host_wrapper]
    bare = [c for c in cands if not c.is_host_wrapper]
    ordered = host[:6] + (bare[:1] if not host else [])
    # jit-only: if the task ships an authored _pa_launcher.py, try its
    # run(inputs) first (the loaders exec it into the kernel module).
    if (task_dir / "_pa_launcher.py").is_file():
        ordered = [Candidate(name="run", is_host_wrapper=True, score=1000000000.0, rationale="authored-jit-only-launcher"), *ordered]

    for c in ordered:
        tried.append(c.name)
        # Step A: oracle-consistency anchor FIRST (cheaper than full S2;
        # decisively rejects a wrong-but-resolvable callable).
        oc_ok, oc_why, oc_cl = oracle_consistency_check(task_dir, c.name)
        _closure_acc.update(oc_cl)
        if not oc_ok:
            # Distinguish a TRUE wrong-operator (numeric value mismatch
            # / wrong output shape) from an upstream callable that
            # cannot even be invoked in THIS env (triton-version
            # AttributeError, missing 3rd-party dep, shapes-incomplete
            # bind). Only a numeric mismatch is "oracle-mismatch"; the
            # rest are genuine per-task tails that S2 itself would also
            # block on -- categorised honestly, never papered over.
            if _is_shapes_incomplete(oc_why):
                kind = "shapes-incomplete"
            elif _is_numeric_mismatch(oc_why):
                kind = "oracle-mismatch"
            else:
                kind = "upstream-not-invocable-here"
            reject_notes.append(f"{c.name}: {kind} [{oc_why}]")
            continue
        # Step B: the REAL S2-smoke (build staged + CPU/TRITON_INTERPRET
        # staged-vs-original compare). Must reach UNRUN/PASS.
        st, why, s2_cl = s2_smoke_with_entrypoint(
            task_dir, c.name, pap
        )
        # HOLE 3 -- UNION the S2-original side's
        # imported infer/ closure (complete belt = oracle UNION S2-original).
        _closure_acc.update(s2_cl)
        if st in ("UNRUN", "PASS"):
            if apply:
                set_gpu_triton(meta_path, c.name)
            return Resolution(
                task_dir.name, True, c.name,
                "UNRUN(needs-gpu)" if st == "UNRUN" else "PASS",
                f"resolved kernel.py::{c.name} -- oracle-consistent "
                f"[{oc_why}] AND S2-smoke {st} [{why[:160]}]; ranking="
                f"{c.rationale}",
                tried=tried,
                upstream_fp=_fp(),
            )
        if "shapes-incomplete" in (why or ""):
            reject_notes.append(
                f"{c.name}: oracle-consistent but S2 shapes-incomplete "
                f"[{why[:160]}]"
            )
        else:
            reject_notes.append(
                f"{c.name}: oracle-consistent but S2-smoke BLOCKED "
                f"[{why[:160]}]"
            )

    # No candidate oracle-verified + smoke-passed -> honest BLOCKED.
    # Categorise the genuine reason (worst-news-first ordering so the
    # category names the real per-task blocker).
    blob = " | ".join(reject_notes)
    if not reject_notes:
        cat = "no-host-wrapper"
    elif any(": shapes-incomplete" in r for r in reject_notes):
        cat = "shapes-incomplete"
    elif any(": smoke-fail" in r or "S2-smoke BLOCKED" in r for r in reject_notes):
        cat = "smoke-fail"
    elif any(": oracle-mismatch" in r for r in reject_notes):
        cat = "oracle-mismatch"
    elif any(
        ": upstream-not-invocable-here" in r for r in reject_notes
    ):
        cat = "upstream-not-invocable-here"
    else:
        cat = "smoke-fail"
    # The PRIMARY blocker is the BEST-RANKED candidate's own failure
    # (reject_notes[0]; candidates are appended in score-ranked host-first
    # order -- a heuristic best guess, not a guaranteed intended wrapper).
    # ``oracle-consistent-*`` means the best-ranked candidate matched the
    # oracle and only the smoke/shapes step blocked -- the highest-value
    # triage signal (a smoke block is often a dev-host env limit that runs
    # on GPU). Appended at the END of the reason so the candidate-segment
    # parser in per_task_diagnostic._split_segments (which requires the
    # body to start at the first reject note) is unaffected.
    primary = _primary_blocker_cat(reject_notes[0]) if reject_notes else "no-host-wrapper"
    return Resolution(
        task_dir.name, False, None,
        f"BLOCKED(no-verified-upstream-entrypoint: {cat})",
        f"BLOCKED(no-verified-upstream-entrypoint: {cat}): tried "
        f"{tried}; {blob}; primary_blocker={primary}",
        tried=tried,
        primary_blocker=primary,
        upstream_fp=_fp(),
    )


def _iter_task_dirs(only: str | None) -> list[Path]:
    out: list[Path] = []
    for level in ("level1", "level2", "level3"):
        ld = TASKS_ROOT / level
        if not ld.is_dir():
            continue
        for d in sorted(ld.iterdir()):
            if not d.is_dir() or d.name.startswith("_"):
                continue
            if only and d.name != only:
                continue
            out.append(d)
    return out


def _task_dir_for(task_id: str) -> Path | None:
    """Resolve a ``task_id`` to its on-disk task directory (ADOPT-1:
    needed to compute / re-validate a verdict's per-task ``input_fp``).
    Returns ``None`` if no such task dir exists."""
    for level in ("level1", "level2", "level3"):
        cand = TASKS_ROOT / level / task_id
        if cand.is_dir():
            return cand
    return None


def run_resolver(
    only: str | None,
    do_all: bool,
    apply: bool,
    checkpoint_dir: Path | None = None,
    resume: bool = False,
    workers: int = 1,
) -> list[Resolution]:
    """Resolve every (or one) task. ``checkpoint_dir`` persists ONE
    JSON per task as it finishes so a long sweep that is interrupted
    keeps every completed verdict (anti-mechanization: never lose real
    work; ``--resume`` skips tasks already checkpointed).

    ``workers``: thread count. ``workers <= 1`` keeps the sequential
    semantics byte-identical. ``workers >= 2`` runs ``resolve_task``
    calls concurrently in a ``ThreadPoolExecutor`` -- chosen over a
    ProcessPool because every per-task call spends almost all its
    wall-clock inside ``subprocess.run()`` (oracle_check + S2 smoke
    runners), and ``subprocess.run`` releases the GIL while blocking
    on the child. Threads therefore give the same wall-clock parallelism
    as processes here without pickling the ``pap`` module or paying
    fork cost; each task's subprocesses are already isolated (unique
    tempdir per call, no shared mutable state in the runners). Peak
    memory scales ~linearly with ``workers`` because each in-flight
    task holds at least one vllm-importing oracle subprocess
    (~1-2 GB RSS each)."""
    pap = _load_pap()
    dirs = _iter_task_dirs(only if not do_all else None)
    if checkpoint_dir is not None:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Pre-filter to dirs that have the two files resolve_task actually
    # reads (kernel.py + meta.yaml). Keeps the parallel branch's
    # work-unit list identical to the sequential branch's.
    valid = [
        td for td in dirs
        if (td / "kernel.py").is_file() and (td / "meta.yaml").is_file()
    ]

    def _process_one(td: Path) -> Resolution:
        cp = (
            checkpoint_dir / f"{td.name}.json"
            if checkpoint_dir is not None
            else None
        )
        if resume and cp is not None and cp.is_file():
            try:
                d = json.loads(cp.read_text())
                print(f"[{td.name}] (resumed from checkpoint)", flush=True)
                return Resolution(**d)
            except (OSError, ValueError, TypeError):
                pass  # corrupt checkpoint -> recompute
        r = resolve_task(td, pap, apply)
        if cp is not None:
            cp.write_text(json.dumps(r.__dict__, indent=2))
        # Surface primary_blocker on the progress line so a per-task sweep
        # (e.g. the loop daemon, which captures stdout) carries the
        # best-ranked candidate's real status, not just the worst-news-first
        # bucket. Triage reads this to tell a close-to-passing task
        # (oracle-consistent-*) from a genuine shapes gap.
        _pb = f" primary={r.primary_blocker}" if r.primary_blocker else ""
        print(
            f"[{r.task_id}] {'RESOLVED' if r.resolved else 'BLOCKED'} "
            f"{r.entrypoint or '-'} :: {r.status}{_pb}",
            flush=True,
        )
        # For a BLOCKED task, also emit the full reason (the per-candidate
        # reject notes) on its own line so a per-task sweep (loop daemon,
        # which captures each subprocess's stdout) carries the actual
        # blocker detail -- not just the headline bucket -- for triage.
        if not r.resolved and r.reason:
            print(f"[{r.task_id}] reason: {r.reason}", flush=True)
        return r

    if workers <= 1:
        return [_process_one(td) for td in valid]

    from concurrent.futures import ThreadPoolExecutor, as_completed
    by_name: dict[str, Resolution] = {}
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_process_one, td): td for td in valid}
        for fut in as_completed(futs):
            td = futs[fut]
            by_name[td.name] = fut.result()
    # Return in the canonical (sorted-by-task) iteration order so the
    # downstream JSON cache + any stable-diff comparison is insensitive
    # to scheduling. Print order interleaves freely (no consumer cares).
    return [by_name[td.name] for td in valid]


def write_resolver_cache(path: Path, res: list) -> None:
    """Write the resolver-verdict cache WITH the provenance stamp
    embedded.

    Layout: ``{"_provenance": {...}, "verdicts": [ {task...}, ... ]}``.
    The stamp's ``code_fingerprint`` is the validated-on-read field --
    its presence makes freshness verifiable FROM THE ARTIFACT ITSELF
    (no /tmp-mtime / "we just ran it" convention).

    ADOPT-1: EACH verdict also gets an
    ``input_fp`` = ``task_input_fingerprint(task_dir)`` -- the sha256
    of THAT task's own verdict-deciding input files at write time. The
    code fingerprint proves the CODE is unchanged; ``input_fp`` proves
    a SPECIFIC task's INPUTS are unchanged, so a one-task input edit
    invalidates only that entry (read side re-resolves just that task,
    never the whole 101).

    EACH verdict ALSO carries ``upstream_fp``
    -- the sha256 of the ACTUAL infer/ imported closure THIS task's
    verdict path executed, UNIONed across the oracle AND S2-original
    sides (HOLE 3; set by ``resolve_task``; the stable
    empty-closure sentinel if the task imported nothing under infer/).
    The ``_provenance`` block carries the global ``infer_state`` (the
    CLEAN git HEAD per upstream root; HOLE 1+2 fail-closed --
    a non-git / DIRTY root is an ``<UNVERIFIABLE:...>`` sentinel that
    makes the cache REFUSED, never silently trusted). Together they
    close the live-upstream hole: an edit under infer/ that flips a
    verdict (counterexample: removing the cbor2 import in
    infer/vllm/.../hashing.py) changes ``upstream_fp`` (if it is in
    this task's closure) AND/OR ``infer_state`` (the global belt) ->
    the read side detects the stale entry and NEVER silently consumes
    it. A verdict object that predates this field (old checkpoint /
    test stub)
    defaults to the empty sentinel -- a RECORDED, validatable value,
    never an accidental None."""
    path.parent.mkdir(parents=True, exist_ok=True)
    verdicts: list[dict] = []
    for r in res:
        d = dict(r.__dict__) if hasattr(r, "__dict__") else dict(r)
        tid = str(d.get("task_id", ""))
        td = _task_dir_for(tid) if tid else None
        # input_fp is None ONLY when the task dir cannot be located
        # (then the read side treats a missing input_fp as STALE --
        # never silently trusts an unprovenanced per-task verdict).
        d["input_fp"] = (
            task_input_fingerprint(td) if td is not None else None
        )
        # upstream_fp is set by resolve_task on EVERY return path. A
        # verdict object that predates the field (old checkpoint / a
        # test stub) defaults to the empty-closure sentinel -- still a
        # stable, validatable value (never None-by-accident).
        if not d.get("upstream_fp"):
            d["upstream_fp"] = _EMPTY_UPSTREAM_CLOSURE_FP
        verdicts.append(d)
    stamp = build_provenance_stamp()
    # HOLE 1+2 -- cache GENERATION must NOT be
    # silently trusted later when infer/ is dirty / non-git. The
    # embedded ``infer_state`` ALREADY carries the
    # ``<UNVERIFIABLE:...>`` sentinel for any non-clean root (so
    # ``load_resolver_cache`` will hard-fail this artifact). Make that
    # NON-SILENT at generation time too: a dirty/non-git
    # ``--refresh-cache`` is loudly flagged on stderr so the operator
    # sees the cache they just built is non-reproducible and will be
    # REFUSED on read (it is recorded unverifiable, never trusted).
    bad = _infer_state_is_unverifiable(stamp.get("infer_state", {}))
    if bad is not None:
        print(
            "WARNING: resolver cache GENERATED while infer/ is NOT a "
            f"verifiable clean pinned git checkout ({bad}). This cache "
            "is recorded UNVERIFIABLE and will be REFUSED on read "
            "(fail-closed, HOLE 1+2) -- a dirty / non-git "
            "upstream tree is not a reproducible pinned state. Pin "
            "infer/ to the version_matrix commit (commit/stash/clean "
            "local changes) and re-run --refresh-cache to produce a "
            "trustworthy cache.",
            file=sys.stderr, flush=True,
        )
    payload = {
        "_provenance": stamp,
        "verdicts": verdicts,
    }
    path.write_text(json.dumps(payload, indent=2))


@dataclass(frozen=True)
class CacheLoadSnapshot:
    """Single-read provenance of one ``load_resolver_cache`` call.
    ``content_sha`` is the sha256 of the exact ``raw`` bytes that were
    ``json.loads``-parsed; ``code_fp`` / ``infer_state`` are the one
    captured snapshot of ``resolver_code_fingerprint()`` /
    ``build_infer_state()`` used to validate this load. The caller
    memoizes under a key derived from these, avoiding a post-parse
    re-read that could bind verdicts from one cache version under
    another version's key (a TOCTOU: ``--refresh-cache`` /
    ``_writeback_fresh_verdict`` rewrite this exact file)."""

    content_sha: str | None
    code_fp: str
    infer_state: dict | str


def load_resolver_cache(
    path: Path,
    *,
    return_snapshot: bool = False,
):
    """Read a resolver-verdict cache.

    When ``return_snapshot`` is True the return is a 3-tuple
    ``(verdicts, staleness, snapshot)`` where ``snapshot`` is the
    ``CacheLoadSnapshot`` carrying the content-sha / code-fp /
    infer_state from the single file read this call performed (the
    caller memoizes under a key derived from that snapshot). When False
    (the default, kept so the ~30 existing 2-tuple-unpacking callers/
    tests are untouched) the return is the ``(verdicts, staleness)``
    2-tuple.

    ``staleness`` is ``None`` iff the cache is present, parseable, its
    embedded ``code_fingerprint`` equals the current
    ``resolver_code_fingerprint()`` (resolver-relevant code unchanged)
    AND its embedded global ``infer_state`` equals the current
    ``build_infer_state()`` (live upstream git HEAD unchanged, so no
    infer/ checkout swap happened outside any single task's captured
    closure). Otherwise ``staleness`` is a short reason ('missing' /
    'unparseable' / 'no-provenance' / 'fingerprint-mismatch' /
    'infer-state-mismatch' / 'no-infer-state') and ``verdicts`` is
    whatever could still be parsed -- callers must not trust it; the
    contract is auto-regenerate on any non-None staleness.

    The live infer/ UNVERIFIABLE check runs first of all, before the
    cache file is even read: if the live infer/ is UNVERIFIABLE (non-git
    / git-broken / DIRTY / wrong-commit / missing managed root /
    rev-parse|status failure) the function RAISES
    ``StaleResolverCacheError`` immediately with no regen attempt,
    regardless of the cache file's own state -- regenerating under the
    same bad upstream cannot fix it. The same hard-fail applies when the
    cache's embedded ``infer_state`` is an ``<UNVERIFIABLE:...>``
    sentinel (the cache was built against a non-reproducible upstream).
    Regenerable staleness (code-fp / input_fp drift) keeps its
    auto-regen path only when the live infer/ is verifiably clean+pinned.
    The global git-HEAD gate is the primary guarantee; the per-task
    ``upstream_fp`` (validated in ``_resolver_verdict_for``) is the
    precise belt, not a substitute for a verifiable clean pinned
    upstream."""
    # Live-infer UNVERIFIABLE guard runs before the cache file is read,
    # ahead of the missing / unparseable / no-provenance early returns
    # and every regenerable staleness path. An UNVERIFIABLE live infer/
    # (non-git / git-broken / DIRTY / wrong-commit / missing managed
    # root) can never back a trustworthy verdict and regenerating under
    # that same bad upstream cannot fix it, so the first call hard-fails
    # with no regen even when the cache is also missing/unparseable.
    live_state = build_infer_state()
    live_bad = _infer_state_is_unverifiable(live_state)
    if live_bad is not None:
        raise StaleResolverCacheError(
            "infer/ is NOT a verifiable clean git checkout at the "
            f"EXPECTED pinned commit ({live_bad}) -- cannot trust the "
            "resolver cache (a non-git / git-broken / DIRTY / "
            "WRONG-commit / MISSING managed-root upstream tree is not "
            "a reproducible pinned state; cf. "
            "ascendbench/policies/version_matrix.toml 'Any deviation invalidates "
            "baseline'). Pin infer/ to the expected commit (the "
            "tasks/*/meta.yaml upstream.commit_sha consensus -- "
            "checkout that commit / commit / stash / clean any local "
            "changes; restore any missing managed root) and re-run: "
            "python3 scripts/verify/resolve_upstream_entrypoints.py "
            "--refresh-cache"
        )
    # ── SINGLE-SNAPSHOT LOAD ────────
    # Read the cache file bytes EXACTLY ONCE into ``raw_bytes``. The
    # content-sha is sha256 of THOSE bytes; the verdicts are parsed from
    # THOSE bytes; ``code_fp`` is captured ONCE here (``live_state`` was
    # captured once at the top). The caller memoizes under a key derived
    # from THIS one snapshot -- there is NO post-parse re-read (the old
    # ``_resolver_memo_key`` recompute, read #3, is eliminated). A cache
    # rewrite (``--refresh-cache`` / ``_writeback_fresh_verdict``)
    # occurring AFTER this read cannot bind content-A verdicts under a
    # content-B key: the only bytes that ever exist on this load path
    # are ``raw_bytes`` and the sha/parse/key all derive from them.
    code_fp = resolver_code_fingerprint()

    def _ret(verdicts: list[dict], staleness: str | None):
        if not return_snapshot:
            return verdicts, staleness
        return verdicts, staleness, CacheLoadSnapshot(
            content_sha=content_sha,
            code_fp=code_fp,
            infer_state=live_state,
        )

    try:
        raw_bytes = path.read_bytes()
    except OSError:
        content_sha = None
        return _ret([], "missing")
    content_sha = hashlib.sha256(raw_bytes).hexdigest()
    try:
        raw = json.loads(raw_bytes)
    except (ValueError, TypeError):
        return _ret([], "unparseable")
    if isinstance(raw, list):
        # Legacy bare-list cache: NO provenance -> cannot be trusted.
        return _ret(
            [r for r in raw if isinstance(r, dict)], "no-provenance"
        )
    if not isinstance(raw, dict):
        return _ret([], "unparseable")
    prov = raw.get("_provenance")
    verdicts = [
        r for r in raw.get("verdicts", []) if isinstance(r, dict)
    ]
    if not isinstance(prov, dict) or "code_fingerprint" not in prov:
        return _ret(verdicts, "no-provenance")
    # Only the cache-embedded infer_state fail-closed check remains here
    # (it needs the parsed ``prov``); the live-infer guard already ran
    # at the top of this function. A cache built against an unverifiable
    # (dirty / non-git / wrong-commit) infer/ carries no evidence the
    # upstream was a clean pinned checkout, so its verdicts are
    # non-reproducible by construction -> hard-fail, not regenerable
    # staleness, even though the live state is now clean+pinned. A cache
    # that merely PREDATES the guard (no ``infer_state`` key at all) is a
    # regenerable staleness handled below -- only an explicitly-recorded
    # UNVERIFIABLE sentinel hard-fails here.
    if "infer_state" in prov:
        cached_bad = _infer_state_is_unverifiable(
            prov.get("infer_state")
        )
        if cached_bad is not None:
            raise StaleResolverCacheError(
                "the resolver cache was GENERATED while infer/ was not "
                f"a verifiable clean pinned git checkout ({cached_bad}) "
                "-- a dirty-/non-git-/wrong-commit-built cache is "
                "non-reproducible and is NEVER trusted (HOLE "
                "1+2, HOLE 4). With infer/ now pinned & clean, "
                "re-run: python3 "
                "scripts/verify/resolve_upstream_entrypoints.py --refresh-cache"
            )
    # ── REGENERABLE staleness (verifiable clean-pinned infer below) ───
    # A code-fp / no-infer-state /
    # infer-state-mismatch on a VERIFIABLE clean-pinned upstream keeps
    # its auto-regen path -- the read path regenerates / per-task
    # live-re-resolves under the CURRENT code+upstream. A cache built
    # without ``infer_state`` predates the guard -> regenerable. A
    # clean-HEAD MISMATCH means the live infer/ HEAD changed since the
    # cache was built (checkout swap OUTSIDE any task closure) -> the
    # WHOLE cache is stale; NEVER silently consumed.
    if prov.get("code_fingerprint") != code_fp:
        return _ret(verdicts, "fingerprint-mismatch")
    if "infer_state" not in prov:
        return _ret(verdicts, "no-infer-state")
    if prov.get("infer_state") != live_state:
        return _ret(verdicts, "infer-state-mismatch")
    return _ret(verdicts, None)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--task", help="restrict to one task id")
    ap.add_argument("--all", action="store_true", help="all 101 tasks")
    ap.add_argument(
        "--apply", action="store_true",
        help="write the verified entrypoint to meta.yaml "
        "(default: dry-run, no write)",
    )
    ap.add_argument(
        "--json", metavar="PATH",
        help="write a JSON summary to PATH (kept off stdout so progress "
        "lines never corrupt it)",
    )
    ap.add_argument(
        "--checkpoint-dir", metavar="DIR",
        help="persist one JSON per task as it finishes (a long sweep "
        "keeps every completed verdict if interrupted)",
    )
    ap.add_argument(
        "--resume", action="store_true",
        help="skip tasks already present in --checkpoint-dir",
    )
    ap.add_argument(
        "--workers", type=int, default=1, metavar="N",
        help="parallel worker threads (default 1=sequential). Each "
        "in-flight task holds at least one vllm-importing oracle "
        "subprocess (~1-2 GB RSS), so peak memory scales ~linearly. "
        "4-8 is a good range on a 32+ GB machine; higher hits "
        "diminishing returns once cold-import cache saturates.",
    )
    ap.add_argument(
        "--refresh-cache", action="store_true",
        help="Regenerate the "
        "DETERMINISTIC canonical resolver-verdict cache "
        f"({CANONICAL_RESOLVE_CACHE.relative_to(REPO_ROOT)}) by running "
        "the FULL 101-task sweep under the CURRENT (hardened) "
        "oracle-anchor + S2-smoke. Implies --all and "
        "--json <CANONICAL_RESOLVE_CACHE>; the single documented, "
        "reproducible command so the cache is never a stale /tmp "
        "artifact whose freshness cannot be trusted (dry-run -- never "
        "writes meta.yaml).",
    )
    args = ap.parse_args(argv)
    if args.refresh_cache:
        args.all = True
        if not args.json:
            args.json = str(CANONICAL_RESOLVE_CACHE)
            CANONICAL_RESOLVE_CACHE.parent.mkdir(parents=True, exist_ok=True)
    res = run_resolver(
        args.task,
        args.all,
        args.apply,
        Path(args.checkpoint_dir) if args.checkpoint_dir else None,
        args.resume,
        workers=args.workers,
    )
    n_res = sum(1 for r in res if r.resolved and r.status != "ALREADY-RESOLVED")
    n_already = sum(1 for r in res if r.status == "ALREADY-RESOLVED")
    n_blk = sum(1 for r in res if not r.resolved)
    print(
        f"\nresolved={n_res} already-resolved={n_already} "
        f"blocked={n_blk} total={len(res)} "
        f"({'APPLIED' if args.apply else 'DRY-RUN'})"
    )
    if args.json:
        write_resolver_cache(Path(args.json), res)
        print(f"json -> {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
