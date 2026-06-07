"""Phase A' prerequisite: upstream-entrypoint resolver soundness.

The resolver's NON-NEGOTIABLE contract (the anti-fake anchor):

  * the CORRECT host wrapper is picked + written ONLY when it BOTH
    oracle-verifies AND S2-smoke-passes;
  * a resolvable-but-WRONG callable (oracle value mismatch) is REJECTED
    and NEVER written -- the placeholder is left intact with an honest
    BLOCKED(no-verified-upstream-entrypoint: ...) reason;
  * a task with no verifiable candidate keeps ``::placeholder``
    (kernel.py is NEVER touched; only the gpu_triton meta line).

These tests drive the deterministic pick/reject/leave logic with the
two verification steps mocked (the REAL subprocess S2-smoke + oracle
check are exercised by the actual sweep, not unit-mocked).
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent.parent
_spec = importlib.util.spec_from_file_location(
    "resolve_upstream_entrypoints",
    REPO / "scripts" / "resolve_upstream_entrypoints.py",
)
rue = importlib.util.module_from_spec(_spec)
sys.modules["resolve_upstream_entrypoints"] = rue
_spec.loader.exec_module(rue)


# ── kernel.py fixtures ────────────────────────────────────────────────
_KERNEL_WITH_WRAPPER = '''
"""Task fixture — frozen upstream GPU triton source.

Kernel function: `_op_kernel`
Source path: `vllm/x/y.py`
"""
import triton
import triton.language as tl


@triton.jit
def _op_kernel(x_ptr, o_ptr, N: tl.constexpr):
    pass


def op_host(x, scale):
    """Host wrapper that launches the jit kernel."""
    o = x
    _op_kernel[(1,)](x, o, 8)
    return o


def _unrelated_helper(a, b):
    return a + b
'''

_KERNEL_PLACEHOLDER_ONLY = '''
"""Task fixture.

Kernel function: `_thing_kernel`
"""
import triton
import triton.language as tl


@triton.jit
def _thing_kernel(p):
    pass
'''

_META_TMPL = """schema_version: '1.0'
task_id: {tid}
oracle:
  pytorch: oracle_pytorch.py::run
  gpu_triton: kernel.py::placeholder
tolerance_class: elementwise
"""

_SHAPES = """size_knobs: {B: 2}
public_shapes:
- {id: P1, name: tiny, knobs: {B: 1}}
inputs:
  x: {dims: [B], dtype: float32, dist: ones}
  scale: {scalar: 2.0}
"""

_ORACLE = '''
def run(inputs: dict) -> dict:
    return {"o": inputs["x"] * 2.0}
'''


def _make_task(tmp_path: Path, name: str, kernel_src: str) -> Path:
    td = tmp_path / name
    td.mkdir()
    (td / "kernel.py").write_text(kernel_src)
    (td / "meta.yaml").write_text(_META_TMPL.format(tid=name))
    (td / "shapes.yaml").write_text(_SHAPES)
    (td / "oracle_pytorch.py").write_text(_ORACLE)
    return td


# ── candidate enumeration + ranking ───────────────────────────────────
def test_enumerate_finds_host_wrapper_not_helper_or_jit():
    cands = rue.enumerate_candidates(_KERNEL_WITH_WRAPPER)
    by_name = {c.name: c for c in cands}
    assert "op_host" in by_name
    assert by_name["op_host"].is_host_wrapper
    assert by_name["op_host"].launched_jit == ["_op_kernel"]
    # bare jit IS a (last-resort) candidate; the plain helper is NOT.
    assert "_op_kernel" in by_name
    assert not by_name["_op_kernel"].is_host_wrapper
    assert "_unrelated_helper" not in by_name


def test_ranking_prefers_host_wrapper_over_bare_jit():
    ranked = rue.rank_candidates(
        rue.enumerate_candidates(_KERNEL_WITH_WRAPPER),
        Path("/nonexistent/L1_99_op"),
        _KERNEL_WITH_WRAPPER,
    )
    assert ranked[0].name == "op_host"
    assert ranked[0].is_host_wrapper


def test_placeholder_only_kernel_has_only_bare_jit_candidate():
    cands = rue.enumerate_candidates(_KERNEL_PLACEHOLDER_ONLY)
    assert [c.name for c in cands] == ["_thing_kernel"]
    assert not cands[0].is_host_wrapper


# ── meta.yaml surgery: byte-preserving, idempotent ────────────────────
def test_set_gpu_triton_only_touches_that_line(tmp_path):
    td = _make_task(tmp_path, "L1_01_op", _KERNEL_WITH_WRAPPER)
    mp = td / "meta.yaml"
    before = mp.read_text()
    assert rue.read_gpu_triton(mp) == "placeholder"
    assert rue.set_gpu_triton(mp, "op_host") is True
    after = mp.read_text()
    assert rue.read_gpu_triton(mp) == "op_host"
    # Every OTHER line is byte-identical.
    b = [ln for ln in before.splitlines() if "gpu_triton" not in ln]
    a = [ln for ln in after.splitlines() if "gpu_triton" not in ln]
    assert a == b
    # Idempotent: re-setting the same name is a no-op.
    assert rue.set_gpu_triton(mp, "op_host") is False


def test_set_gpu_triton_never_touches_kernel_py(tmp_path):
    td = _make_task(tmp_path, "L1_02_op", _KERNEL_WITH_WRAPPER)
    k_before = (td / "kernel.py").read_text()
    rue.set_gpu_triton(td / "meta.yaml", "op_host")
    assert (td / "kernel.py").read_text() == k_before


# ── the anti-fake decision logic (verification mocked) ────────────────
class _Res:
    def __init__(self, status, reason=""):
        self.status = status
        self.reason = reason


def test_correct_entrypoint_picked_and_applied_when_both_pass(
    tmp_path, monkeypatch
):
    """oracle-consistent + S2-smoke UNRUN  ->  picked AND written."""
    td = _make_task(tmp_path, "L1_03_op", _KERNEL_WITH_WRAPPER)
    # codex r6: oracle_consistency_check now returns a 3-tuple
    # (consistent, detail, upstream_closure). A synthetic fixture task
    # imports nothing under infer/ -> the closure is honestly [] (the
    # resolver maps that to the stable empty-closure sentinel).
    monkeypatch.setattr(
        rue, "oracle_consistency_check",
        lambda d, e: (True, f"shape=P1 :: {e} ✓ within elementwise", []),
    )
    # codex r7 HOLE 3: s2_smoke_with_entrypoint now returns a 3-tuple
    # (status, reason, s2_original_upstream_closure). A synthetic
    # fixture imports nothing under infer/ on the S2-original side ->
    # the closure is honestly [] (resolver maps to empty sentinel).
    monkeypatch.setattr(
        rue, "s2_smoke_with_entrypoint",
        lambda d, e, p: ("UNRUN", "S2-smoke passed; no GPU", []),
    )
    r = rue.resolve_task(td, pap=None, apply=True)
    assert r.resolved
    assert r.entrypoint == "op_host"
    assert r.status == "UNRUN(needs-gpu)"
    # Actually written to meta.
    assert rue.read_gpu_triton(td / "meta.yaml") == "op_host"


def test_oracle_mismatch_rejected_and_not_set(tmp_path, monkeypatch):
    """A resolvable callable whose output does NOT match the oracle is
    the WRONG operator -> NOT written; placeholder stays; honest
    BLOCKED(no-verified-upstream-entrypoint: oracle-mismatch)."""
    td = _make_task(tmp_path, "L1_04_op", _KERNEL_WITH_WRAPPER)

    def _oc(d, e):
        return (
            False, "o: NOT within elementwise (max_abs=3.2e+00)", [],
        )

    monkeypatch.setattr(rue, "oracle_consistency_check", _oc)
    # S2 must NOT even be consulted once oracle rejects.
    monkeypatch.setattr(
        rue, "s2_smoke_with_entrypoint",
        lambda *a: (_ for _ in ()).throw(
            AssertionError("S2 must not run after oracle-mismatch")
        ),
    )
    r = rue.resolve_task(td, pap=None, apply=True)
    assert not r.resolved
    assert r.entrypoint is None
    assert "oracle-mismatch" in r.status
    # The placeholder is UNTOUCHED -- no fake to make the number go up.
    assert rue.read_gpu_triton(td / "meta.yaml") == "placeholder"


def test_oracle_ok_but_smoke_blocked_not_set(tmp_path, monkeypatch):
    """oracle-consistent but the REAL S2-smoke BLOCKS (severing/interpret
    mismatch) -> still NOT written; honest per-task BLOCKED reason."""
    td = _make_task(tmp_path, "L1_05_op", _KERNEL_WITH_WRAPPER)
    monkeypatch.setattr(
        rue, "oracle_consistency_check", lambda d, e: (True, "✓", [])
    )
    monkeypatch.setattr(
        rue, "s2_smoke_with_entrypoint",
        lambda d, e, p: (
            "BLOCKED", "S2-smoke MISMATCH (phase=compare)", [],
        ),
    )
    r = rue.resolve_task(td, pap=None, apply=True)
    assert not r.resolved
    assert rue.read_gpu_triton(td / "meta.yaml") == "placeholder"
    assert "no-verified-upstream-entrypoint" in r.status


def test_no_candidate_leaves_placeholder(tmp_path, monkeypatch):
    """A bare-jit-only kernel whose only candidate fails verification
    keeps ::placeholder (kernel.py never touched)."""
    td = _make_task(
        tmp_path, "L1_06_thing", _KERNEL_PLACEHOLDER_ONLY
    )
    monkeypatch.setattr(
        rue, "oracle_consistency_check",
        lambda d, e: (
            False, "ImportError: upstream not invocable here", [],
        ),
    )
    k_before = (td / "kernel.py").read_text()
    r = rue.resolve_task(td, pap=None, apply=True)
    assert not r.resolved
    assert rue.read_gpu_triton(td / "meta.yaml") == "placeholder"
    assert (td / "kernel.py").read_text() == k_before  # never touched


def test_already_set_entrypoint_reverified_under_current_anchor(
    tmp_path, monkeypatch
):
    """MUST-FIX 1 (codex r2 [confirmed]): a task whose meta ALREADY
    points at a non-placeholder entrypoint (the 5 pre-existing OR a task
    an earlier --apply run resolved) is RE-VERIFIED under the CURRENT
    (hardened) oracle anchor -- it is NOT short-circuited to a
    no-evidence ``ALREADY-RESOLVED``. The OLD behaviour returned
    ``ALREADY-RESOLVED`` WITHOUT running the anchor, so the cache
    recorded a 'resolved' verdict carrying ZERO post-hardening evidence
    (the stale-cache provenance overclaim). meta.yaml is STILL never
    mutated on this path. FAILS ON REVERT to the
    skip-the-anchor short-circuit (the anchor stub below would never be
    called, so r.reason would lack the 're-verified under the CURRENT
    hardened anchor' provenance string)."""
    td = _make_task(tmp_path, "L1_07_op", _KERNEL_WITH_WRAPPER)
    mp = td / "meta.yaml"
    mp.write_text(mp.read_text().replace("::placeholder", "::op_host"))
    meta_before = mp.read_text()
    k_before = (td / "kernel.py").read_text()
    anchor_calls: list[tuple] = []

    def _stub_anchor(task_dir, entrypoint):
        anchor_calls.append((task_dir, entrypoint))
        return True, "anchored on 2/2 shape(s) ['S0', 'S1'] :: ok", []

    monkeypatch.setattr(rue, "oracle_consistency_check", _stub_anchor)
    monkeypatch.setattr(
        rue, "s2_smoke_with_entrypoint",
        lambda *a, **k: (
            "UNRUN", "S2-smoke passed (CPU/TRITON_INTERPRET)", [],
        ),
    )
    r = rue.resolve_task(td, pap=object(), apply=True)
    # The CURRENT hardened anchor WAS run against the already-set
    # entrypoint (anti-stale-cache) -- not skipped.
    assert anchor_calls == [(td, "op_host")], anchor_calls
    assert r.resolved
    assert r.entrypoint == "op_host"
    assert r.status == "UNRUN(needs-gpu)"
    assert "re-verified under the CURRENT" in r.reason, r.reason
    # meta.yaml + kernel.py are STILL byte-identical (never mutated on
    # the already-set path).
    assert mp.read_text() == meta_before
    assert (td / "kernel.py").read_text() == k_before


def test_already_set_entrypoint_failing_current_anchor_changes_verdict(
    tmp_path, monkeypatch
):
    """MUST-FIX 1 anti-fake rule: if an already-set entrypoint does NOT
    pass the CURRENT hardened anchor (e.g. it was applied by a
    pre-hardening run that the stricter anchor now rejects), the honest
    verdict CHANGES to BLOCKED -- a stale pre-hardening pass is NEVER
    preserved just to keep the resolved count up. FAILS ON REVERT to the
    short-circuit (which would still report it resolved)."""
    td = _make_task(tmp_path, "L1_08_op", _KERNEL_WITH_WRAPPER)
    mp = td / "meta.yaml"
    mp.write_text(mp.read_text().replace("::placeholder", "::op_host"))
    monkeypatch.setattr(
        rue, "oracle_consistency_check",
        lambda *a: (
            False, "output: NOT within elementwise (wrong op)", [],
        ),
    )
    r = rue.resolve_task(td, pap=object(), apply=True)
    assert not r.resolved, r
    assert r.entrypoint is None
    assert "oracle-mismatch" in r.status, r.status
    assert "stale pre-hardening pass is NOT preserved" in r.reason


# ── honest categorisation of the genuine per-task tail ────────────────
def test_shapes_incomplete_categorised_distinctly():
    assert rue._is_shapes_incomplete(
        "BLOCKED(shapes-incomplete: missing param 'o'"
    )
    assert not rue._is_shapes_incomplete("o: NOT within elementwise")


def test_numeric_mismatch_only_for_real_value_divergence():
    """An upstream import/triton-version error is NOT a wrong-operator
    signal (S2's original side hits it identically) -- only a true
    value/shape divergence is 'oracle-mismatch'."""
    assert rue._is_numeric_mismatch("o: NOT within elementwise (max_abs=...)")
    assert rue._is_numeric_mismatch(
        "x: no candidate output with shape (4,) dtype torch.float32"
    )
    assert not rue._is_numeric_mismatch(
        "ModuleNotFoundError: No module named 'cbor2'"
    )
    assert not rue._is_numeric_mismatch(
        "AttributeError: module 'triton.language' has no attribute 'float32'"
    )


def test_structure_mismatch_is_wrong_operator_not_env_limitation():
    """MUST-FIX 2 (codex r3 [confirmed]) -- FAIL ON REVERT.

    The hardened oracle-anchor rejects a candidate of the WRONG
    ARITY/STRUCTURE with ``STRUCTURE MISMATCH: ... ⇒ wrong operator``
    (the exact L1_01 signal: candidate emits oracle-outputs-PLUS-11
    extras). That is a wrong operator -> ``_is_numeric_mismatch`` MUST
    be True so the resolver categorises it ``oracle-mismatch``, NEVER
    ``upstream-not-invocable-here`` (which would DISHONESTLY call a
    structural wrong-operator an "ENVIRONMENT limitation"). Reverting
    the predicate (dropping the STRUCTURE-MISMATCH / ``⇒ wrong
    operator`` clauses) makes this FAIL."""
    # The verbatim L1_01 anchor reject note.
    l1_01 = (
        "anchored on 1/2 shape(s) ['H2'] :: shape=H2 "
        "tol=elementwise_exact_or_inf_safe :: logits: ✓ within "
        "elementwise_exact_or_inf_safe (max_abs=0.000e+00); STRUCTURE "
        "MISMATCH: candidate produced 11 flattened tensor outputs but "
        "only 1 matched the oracle's 1; extra unexplained outputs "
        "['expanded_idx_mapping', 'pos', 'num_allowed_token_ids', "
        "'allowed_token_ids'] ⇒ wrong operator (same outputs PLUS "
        "extras must NOT pass the anchor)"
    )
    assert rue._is_numeric_mismatch(l1_01), (
        "a STRUCTURE MISMATCH wrong-operator must be oracle-mismatch, "
        "NOT mislabelled an env limitation"
    )
    # Generic markers also count.
    assert rue._is_numeric_mismatch("... ⇒ wrong operator (...)")
    assert rue._is_numeric_mismatch(
        "dtype mismatch is a wrong-operator signal"
    )
    # A pure env/import failure is still NOT a wrong operator.
    assert not rue._is_numeric_mismatch(
        "ImportError: attempted relative import with no known parent"
    )


def test_provenance_stamp_roundtrip_and_staleness_detection(tmp_path):
    """MUST-FIX 1 (codex r3 [confirmed]) -- FAIL ON REVERT.

    A cache written by ``write_resolver_cache`` carries a provenance
    stamp whose ``code_fingerprint`` equals the CURRENT
    ``resolver_code_fingerprint()`` -> ``load_resolver_cache`` reports
    staleness ``None`` (genuinely fresh). A cache with a TAMPERED
    fingerprint, a bare legacy list, or a missing file MUST report a
    non-None staleness (it is detectably stale FROM THE ARTIFACT
    ITSELF -- no /tmp-mtime convention). Reverting to a bare-list
    write (no stamp) makes the fresh-roundtrip assertion FAIL."""
    import json

    cache = tmp_path / "resolve_all.json"

    class _R:
        def __init__(self):
            self.__dict__ = {"task_id": "T1", "status": "X", "reason": "y"}

    rue.write_resolver_cache(cache, [_R()])
    payload = json.loads(cache.read_text())
    assert "_provenance" in payload and "verdicts" in payload, payload
    assert payload["_provenance"]["code_fingerprint"] == (
        rue.resolver_code_fingerprint()
    )
    verdicts, stale = rue.load_resolver_cache(cache)
    assert stale is None, stale  # genuinely fresh
    # ADOPT-1 (codex r5 [confirmed]): the writer stamps each verdict
    # with a per-task ``input_fp``. ``_R`` is a synthetic task ("T1")
    # with no on-disk dir, so input_fp is None (the read side treats a
    # None/missing input_fp as STALE -- never silently trusts an
    # unprovenanced per-task verdict).
    # codex r6 [confirmed]: the writer ALSO stamps ``upstream_fp``; a
    # verdict object that predates the field (this synthetic stub)
    # defaults to the STABLE empty-closure sentinel -- a recorded,
    # validatable value, NEVER None-by-accident.
    assert verdicts == [
        {
            "task_id": "T1", "status": "X", "reason": "y",
            "input_fp": None,
            "upstream_fp": rue._EMPTY_UPSTREAM_CLOSURE_FP,
        }
    ]

    # Tamper the stamp -> fingerprint-mismatch (the cache predates a
    # resolver/anchor code change). Detected, NOT silently consumed.
    payload["_provenance"]["code_fingerprint"] = "deadbeef" * 8
    cache.write_text(json.dumps(payload))
    _v, stale2 = rue.load_resolver_cache(cache)
    assert stale2 == "fingerprint-mismatch", stale2

    # Legacy bare list (no provenance at all) -> never trusted.
    cache.write_text(json.dumps([{"task_id": "T1"}]))
    _v, stale3 = rue.load_resolver_cache(cache)
    assert stale3 == "no-provenance", stale3

    # Missing file.
    _v, stale4 = rue.load_resolver_cache(tmp_path / "nope.json")
    assert stale4 == "missing", stale4


def test_complete_verdict_surface_is_fingerprinted(tmp_path):
    """MUST-FIX (codex r4 [confirmed]) -- FAIL ON REVERT.

    The resolver VERDICT/category also depends on the input builder,
    the tolerance comparator, and the tolerance-policy YAML:

      * triton_ascend_bench/inputs.py    -- build_inputs() feeds BOTH
        the oracle anchor and the S2 runner;
      * triton_ascend_bench/tolerance.py -- compare()/load_policy()
        ARE the oracle-mismatch decision;
      * triton_ascend_bench/tolerance_policy.yaml -- the per-class
        atol/rtol/bit_exact VALUES the comparator reads.

    A byte change to ANY of these flips a verdict while a cache built
    under the old bytes would still validate as "fresh" unless the
    fingerprint surface covers them. This test asserts (a) each file
    is a MEMBER of ``_PROVENANCE_CODE_FILES`` -- reverting the surface
    expansion drops the membership and FAILS HERE -- and (b)
    byte-modifying each file actually CHANGES
    ``resolver_code_fingerprint()`` (so the cache becomes detectably
    stale). (b) also fails on revert: a file outside the surface does
    not influence the digest, so the changed-digest assertion breaks.
    """
    pkg = REPO / "triton_ascend_bench"
    must_cover = {
        pkg / "inputs.py",
        pkg / "tolerance.py",
        pkg / "tolerance_policy.yaml",
    }
    surface = set(rue._PROVENANCE_CODE_FILES)

    # (a) Membership: reverting the _PROVENANCE_CODE_FILES expansion
    #     removes these and this assertion FAILS (proves fail-on-revert
    #     structurally, no git needed).
    missing = sorted(str(p) for p in (must_cover - surface))
    assert not missing, (
        f"verdict-deciding files NOT in _PROVENANCE_CODE_FILES "
        f"(revert detected -- a change to these would slip past a "
        f"'fresh' cache): {missing}"
    )

    # (b) Behavioural: byte-modifying each newly-added file MUST change
    #     the digest. We mutate the REAL file (saving + restoring bytes
    #     in a finally) so this exercises the exact production path
    #     resolver_code_fingerprint() takes (f.read_bytes()).
    base = rue.resolver_code_fingerprint()
    for target in (
        pkg / "inputs.py",
        pkg / "tolerance.py",
        pkg / "tolerance_policy.yaml",
    ):
        original = target.read_bytes()
        try:
            target.write_bytes(original + b"\n# provenance-probe\n")
            mutated = rue.resolver_code_fingerprint()
            assert mutated != base, (
                f"fingerprint did NOT change after byte-modifying "
                f"{target.name} -- it is NOT in the verdict surface "
                f"(revert detected: a tolerance/inputs edit would slip "
                f"past a 'fresh' cache)"
            )
        finally:
            target.write_bytes(original)

    # Surface restored -> digest back to baseline (no side effects).
    assert rue.resolver_code_fingerprint() == base


# ── ADOPT-2: phase_a_freeze.py is verdict-path CODE (8th file) ────────
def test_phase_a_freeze_is_in_code_fingerprint_surface(tmp_path):
    """ADOPT-2 (codex r5 [confirmed]) -- FAIL ON REVERT.

    ``phase_a_prime.py:58`` does ``import phase_a_freeze as paf`` and
    reuses its commit-pin / repo-root / provenance helpers ON the
    verdict path (paf.REPO_ROOT/TASKS_ROOT/VLLM_ROOT/SGLANG_ROOT at
    module load + paf._commit_sha/_provenance_header in the status
    flow). So ``scripts/phase_a_freeze.py`` is verdict-path CODE and
    MUST be a member of ``_PROVENANCE_CODE_FILES``.

    (a) Membership: dropping the 8th-file addition removes it from the
        surface and FAILS HERE (structural fail-on-revert, no git).
    (b) Behavioural: byte-modifying the REAL file MUST change
        ``resolver_code_fingerprint()`` -- if it were NOT in the
        surface the digest would be unchanged and this assertion
        breaks (the exact revert signature).
    """
    paf = REPO / "scripts" / "phase_a_freeze.py"
    surface = set(rue._PROVENANCE_CODE_FILES)

    assert paf in surface, (
        "scripts/phase_a_freeze.py is NOT in _PROVENANCE_CODE_FILES "
        "(ADOPT-2 reverted -- a byte change to a helper used on the "
        "verdict path would slip past a 'fresh' cache)"
    )
    # The surface is exactly the audited 8 (4 scripts + 3 package
    # files + phase_a_freeze) -- a regression that drops one OR
    # over-hashes an extra file fails here.
    assert len(rue._PROVENANCE_CODE_FILES) == 8, [
        f.name for f in rue._PROVENANCE_CODE_FILES
    ]

    base = rue.resolver_code_fingerprint()
    original = paf.read_bytes()
    try:
        paf.write_bytes(original + b"\n# provenance-probe\n")
        mutated = rue.resolver_code_fingerprint()
        assert mutated != base, (
            "fingerprint did NOT change after byte-modifying "
            "phase_a_freeze.py -- it is NOT in the verdict surface "
            "(ADOPT-2 revert detected)"
        )
    finally:
        paf.write_bytes(original)
    assert rue.resolver_code_fingerprint() == base


# ── ADOPT-1: per-task input_fp invalidates ONLY the changed task ─────
def test_per_task_input_fp_detects_stale_task_input(tmp_path):
    """ADOPT-1 (codex r5 [confirmed]) -- FAIL ON REVERT.

    A cached verdict carries an ``input_fp`` = sha256 over THAT task's
    own verdict-deciding inputs (kernel.py / meta.yaml / shapes.yaml /
    oracle_pytorch.py). Editing ANY of those four flips that task's
    ``task_input_fingerprint`` so a cache built before the edit no
    longer matches -- the stale per-task verdict MUST be detectable
    (and therefore never silently consumed). Reverting the input_fp
    scheme (e.g. hashing nothing / always returning a constant) makes
    the changed-digest assertions FAIL here.
    """
    # A self-contained fake task dir with exactly the 4 input files.
    td = tmp_path / "level1" / "L9_99_probe"
    td.mkdir(parents=True)
    (td / "kernel.py").write_text("# k\n")
    (td / "meta.yaml").write_text("upstream:\n  commit_sha: abc\n")
    (td / "shapes.yaml").write_text("inputs: {}\n")
    (td / "oracle_pytorch.py").write_text("def run(): ...\n")
    (td / "contract.md").write_text("not a verdict input\n")

    base_fp = rue.task_input_fingerprint(td)

    # contract.md is NOT a verdict input -> editing it must NOT change
    # the fingerprint (over-hashing it would be a false provenance
    # claim; this asserts the audited exclusion holds).
    (td / "contract.md").write_text("still not a verdict input -- v2\n")
    assert rue.task_input_fingerprint(td) == base_fp, (
        "contract.md changed the input_fp -- it is NOT a verdict "
        "input and must be excluded (audited exclusion reverted)"
    )

    # Each of the 4 real verdict inputs MUST flip the fingerprint.
    for fn, newtext in (
        ("kernel.py", "# k v2\n"),
        ("meta.yaml", "upstream:\n  commit_sha: DEF\n"),
        ("shapes.yaml", "inputs: {x: {shape: [1]}}\n"),
        ("oracle_pytorch.py", "def run(): return 1\n"),
    ):
        original = (td / fn).read_text()
        before = rue.task_input_fingerprint(td)
        (td / fn).write_text(newtext)
        after = rue.task_input_fingerprint(td)
        assert after != before, (
            f"editing {fn} did NOT change the per-task input_fp -- "
            f"a stale verdict for this task would be silently "
            f"consumed (ADOPT-1 input_fp scheme reverted)"
        )
        (td / fn).write_text(original)

    # Fully restored -> back to baseline (deterministic, no side fx).
    assert rue.task_input_fingerprint(td) == base_fp

    # A MISSING input file is itself a provenance CHANGE (never
    # silently "unchanged").
    (td / "shapes.yaml").unlink()
    assert rue.task_input_fingerprint(td) != base_fp


def test_write_resolver_cache_embeds_per_task_input_fp(tmp_path):
    """ADOPT-1 (codex r5 [confirmed]) -- FAIL ON REVERT.

    ``write_resolver_cache`` MUST stamp each verdict with an
    ``input_fp`` that equals the CURRENT
    ``task_input_fingerprint(task_dir)`` for a real task. A read that
    re-derives the fingerprint and compares is the per-task staleness
    guard; if ``write_resolver_cache`` stops embedding ``input_fp``
    (revert) this assertion FAILS.
    """
    import json

    real_task = "L1_15_merge_state"
    td = rue._task_dir_for(real_task)
    assert td is not None and td.is_dir(), td

    class _R:
        def __init__(self, tid):
            self.__dict__ = {
                "task_id": tid,
                "resolved": False,
                "status": "X",
                "reason": "y",
            }

    cache = tmp_path / "resolve_all.json"
    rue.write_resolver_cache(cache, [_R(real_task)])
    payload = json.loads(cache.read_text())
    v = payload["verdicts"][0]
    assert "input_fp" in v, v
    assert v["input_fp"] == rue.task_input_fingerprint(td), (
        "embedded input_fp does NOT equal the current per-task "
        "fingerprint -- the per-task staleness guard is broken "
        "(ADOPT-1 write-side reverted)"
    )
    # An unknown task id -> input_fp None (read side treats a None /
    # missing input_fp as STALE; it is never silently trusted).
    rue.write_resolver_cache(cache, [_R("L9_99_does_not_exist")])
    payload2 = json.loads(cache.read_text())
    assert payload2["verdicts"][0]["input_fp"] is None


def test_write_resolver_cache_embeds_upstream_fp_and_infer_state(
    tmp_path,
):
    """codex r6 [confirmed] -- FAIL ON REVERT (write side).

    ``write_resolver_cache`` MUST stamp EVERY verdict with an
    ``upstream_fp`` (the per-task imported infer/ closure hash, or the
    stable empty sentinel for a verdict object that predates the field)
    AND embed the GLOBAL ``infer_state`` in ``_provenance``. If the
    write side stops embedding either, a stale cache after a
    verdict-flipping infer/ edit would silently validate fresh -- so
    these assertions FAIL on revert.
    """
    import json

    class _R:
        def __init__(self, tid, ufp=None):
            self.__dict__ = {
                "task_id": tid,
                "resolved": False,
                "status": "X",
                "reason": "y",
            }
            if ufp is not None:
                self.__dict__["upstream_fp"] = ufp

    cache = tmp_path / "resolve_all.json"
    real = "L1_15_merge_state"
    real_ufp = "sha256:" + ("a" * 64)
    rue.write_resolver_cache(
        cache, [_R(real, real_ufp), _R("L9_99_no_fp_field")]
    )
    payload = json.loads(cache.read_text())
    v0 = payload["verdicts"][0]
    v1 = payload["verdicts"][1]
    assert v0["upstream_fp"] == real_ufp, (
        "an explicit per-task upstream_fp was DROPPED on write -- the "
        "live-upstream provenance is not persisted (codex r6 reverted)"
    )
    assert v1["upstream_fp"] == rue._EMPTY_UPSTREAM_CLOSURE_FP, (
        "a verdict with no upstream_fp was NOT defaulted to the stable "
        "empty sentinel -- it would be None-by-accident (never "
        "validatable)"
    )
    assert "infer_state" in payload["_provenance"], (
        "infer_state NOT embedded in _provenance -- the global belt "
        "guard is gone (codex r6 reverted)"
    )


# ── codex r6 [confirmed]: upstream IS now provenance-covered ─────────
# (the EARLIER REJECT was WRONG -- a concrete counterexample overturned
# it; this test asserts the NEW correct invariant, NOT the old one.)
def test_upstream_closure_and_infer_state_are_provenance_validated():
    """codex r6 [confirmed] -- CORRECTED invariant (the prior
    ``test_infer_source_trees_are_deliberately_not_fingerprinted``
    asserted the WRONG invariant and is replaced by THIS).

    The old claim ("meta.commit_sha transitively covers infer/") was
    INSUFFICIENT: the resolver executes LIVE upstream bytes
    (oracle-consistency + S2 original side put infer/ on sys.path and
    import/run it); meta.yaml only RECORDS the pin -- nothing enforced
    live==pin. codex r6 counterexample: removing the ``cbor2`` import in
    ``infer/vllm/vllm/utils/hashing.py`` flips L1_02's verdict
    (missing-dep:cbor2 BLOCKED -> resolvable) with NO change to
    kernel.py / meta.yaml / shapes.yaml / oracle / the code-fp / the
    per-task input_fp -> the stale cache would silently validate fresh.

    The CORRECT invariant (now enforced): the live verdict-deciding
    upstream bytes ARE provenance-validated via
      * a PER-TASK ``upstream_fp`` (precise imported closure -- NOT a
        whole-tree hash; the original cost/precision concern preserved
        by hashing ONLY the touched closure), AND
      * a GLOBAL ``infer_state`` (git HEAD + dirty per upstream root)
        in ``_provenance``.
    Whole multi-GB trees are STILL never hashed (the still-valid cost
    concern); but the hole is closed. (input_fp + meta-sha remain --
    they were never sufficient ALONE for the live-upstream surface.)
    """
    # 1. Whole trees STILL not in the static code fingerprint surface
    #    (the cost concern was real and is preserved).
    infer_root = REPO.parent / "infer"
    for f in rue._PROVENANCE_CODE_FILES:
        rf = str(f.resolve())
        assert "/infer/" not in rf and not rf.startswith(
            str(infer_root.resolve())
        ), (
            f"{f} lives under infer/ -- whole upstream trees are STILL "
            f"not part of the static code fingerprint (cost). Upstream "
            f"is covered by the PER-TASK upstream_fp + GLOBAL "
            f"infer_state instead, NOT by whole-tree hashing."
        )
    # 2. The CORRECTED guarantee exists: the precise-closure fp helper,
    #    the empty-closure sentinel, and the global infer_state builder.
    assert callable(rue.upstream_closure_fingerprint), (
        "upstream_closure_fingerprint MISSING -- the per-task imported "
        "closure is no longer provenance-hashed (codex r6 hole REOPENED)"
    )
    assert isinstance(rue._EMPTY_UPSTREAM_CLOSURE_FP, str) and (
        rue._EMPTY_UPSTREAM_CLOSURE_FP.startswith(
            "sha256:empty-upstream-closure:"
        )
    ), "empty-closure sentinel reverted -- a no-import task's "
    "upstream_fp would be None-by-accident (never validatable)"
    st = rue.build_infer_state()
    assert isinstance(st, dict), "build_infer_state reverted"
    # 3. The provenance stamp embeds infer_state; load validates it.
    stamp = rue.build_provenance_stamp()
    assert "infer_state" in stamp, (
        "infer_state dropped from the provenance stamp -- the GLOBAL "
        "belt-and-suspenders guard (codex r6) is GONE; an infer/ "
        "checkout swap / uncommitted edit OUTSIDE a task's closure "
        "would silently validate fresh"
    )
    # input_fp + meta-sha are RETAINED (complementary, never the sole
    # cover for the live-upstream surface).
    assert "meta.yaml" in rue._PER_TASK_INPUT_FILES


def test_upstream_closure_fingerprint_detects_infer_byte_change(
    tmp_path,
):
    """codex r6 [confirmed] -- FAIL ON REVERT (counterexample closed).

    Editing a file under infer/ that a task's verdict path imports
    (the codex r6 counterexample: a 1-line change to
    ``infer/vllm/vllm/utils/hashing.py``) MUST flip that task's
    ``upstream_fp``. Proven directly + deterministically against a
    writable infer-root surrogate (the real ``infer/`` is root-owned
    and read-only on this box, so the edit is performed on a faithful
    copy mapped to the SAME relpath -- the fingerprint is computed from
    file BYTES at the relpath, exactly as in production).

    Reverting the upstream-closure-provenance logic (e.g. hashing
    nothing / a constant / ignoring file bytes) makes this FAIL.
    """
    # A surrogate vllm root with the exact relpath the counterexample
    # targets.
    fake_vllm = tmp_path / "infer" / "vllm"
    hp = fake_vllm / "vllm" / "utils" / "hashing.py"
    hp.parent.mkdir(parents=True)
    original = "import cbor2\n\n\ndef sha256_cbor(x):\n    return x\n"
    hp.write_text(original)

    # Point the closure hasher at the surrogate root (production reads
    # rue._INFER_ROOTS / VLLM_ROOT identically).
    saved_roots = rue._INFER_ROOTS
    saved_vllm = rue.VLLM_ROOT
    try:
        rue.VLLM_ROOT = fake_vllm
        rue._INFER_ROOTS = (
            ("vllm", fake_vllm),
            ("sglang", rue.SGLANG_ROOT),
        )
        before = rue.upstream_closure_fingerprint([str(hp)])
        assert before != rue._EMPTY_UPSTREAM_CLOSURE_FP, (
            "a real imported infer/ file hashed to the EMPTY sentinel "
            "-- the closure is not actually being hashed (codex r6 "
            "hole REOPENED)"
        )

        # THE codex r6 counterexample edit: remove the cbor2 import
        # (1-line change -> would flip the verdict in production).
        hp.write_text(
            original.replace("import cbor2\n", "# cbor2 removed\n")
        )
        after = rue.upstream_closure_fingerprint([str(hp)])
        assert after != before, (
            "editing infer/vllm/.../hashing.py did NOT change the "
            "per-task upstream_fp -- the codex r6 counterexample is "
            "NOT closed: a stale cached verdict would be silently "
            "consumed after a verdict-flipping infer/ edit "
            "(upstream-closure provenance reverted)"
        )

        # Even a pure-comment 1-byte change must flip it (the
        # disposition's exact fail-on-revert wording: "add a comment to
        # infer/vllm/vllm/utils/hashing.py").
        hp.write_text(original + "# a harmless trailing comment\n")
        commented = rue.upstream_closure_fingerprint([str(hp)])
        assert commented != before, (
            "a comment-only edit to a closure infer/ file did NOT "
            "flip upstream_fp -- closure provenance is byte-exact or "
            "it is nothing"
        )

        # Restore -> back to baseline (deterministic, no side effects).
        hp.write_text(original)
        assert rue.upstream_closure_fingerprint([str(hp)]) == before
    finally:
        rue._INFER_ROOTS = saved_roots
        rue.VLLM_ROOT = saved_vllm


def test_load_resolver_cache_rejects_infer_state_mismatch(tmp_path):
    """codex r6 [confirmed] -- FAIL ON REVERT (global belt).

    ``load_resolver_cache`` MUST report a non-None staleness when the
    embedded global ``infer_state`` no longer equals the CURRENT
    ``build_infer_state()`` (an infer/ checkout swap or uncommitted
    edit OUTSIDE any single task's captured closure). A cache that
    predates the guard (no ``infer_state``) is also untrusted. If the
    infer_state validation is reverted these assertions FAIL (a stale
    cache would be silently consumed).
    """
    import json

    cache = tmp_path / "resolve_all.json"

    # Fresh cache: code_fp + infer_state both current -> trusted.
    rue.write_resolver_cache(cache, [])
    _v, stale = rue.load_resolver_cache(cache)
    assert stale is None, f"fresh cache wrongly stale: {stale}"

    # Tamper ONLY the global infer_state (simulating an infer/ HEAD or
    # dirty change OUTSIDE any task closure) -> WHOLE cache stale.
    payload = json.loads(cache.read_text())
    payload["_provenance"]["infer_state"] = {
        "vllm": "deadbeef-checkout-swapped"
    }
    cache.write_text(json.dumps(payload))
    _v2, stale2 = rue.load_resolver_cache(cache)
    assert stale2 == "infer-state-mismatch", (
        f"infer_state tamper NOT detected ({stale2}) -- a cache built "
        f"under a DIFFERENT live upstream would be silently consumed "
        f"(codex r6 global belt reverted)"
    )

    # A cache with NO infer_state (predates the guard) is untrusted.
    payload2 = json.loads(cache.read_text())
    del payload2["_provenance"]["infer_state"]
    cache.write_text(json.dumps(payload2))
    _v3, stale3 = rue.load_resolver_cache(cache)
    assert stale3 == "no-infer-state", stale3


# ── codex r7 [confirmed] HOLE 1+2+3 fail-on-revert ───────────────────


def test_hole1_nongit_infer_state_is_fail_closed_not_fail_open(
    tmp_path, monkeypatch
):
    """codex r7 [confirmed] HOLE 1 -- FAIL ON REVERT.

    The OLD ``_git_root_state`` recorded ``<not-a-git-repo>`` as a
    NORMAL comparable state, so on a non-git / git-broken infer/ box
    ``load_resolver_cache`` degraded to 'non-git == non-git => fresh'
    (FAIL OPEN -- the live-upstream hole silently reopens). The fix is
    FAIL CLOSED: a non-git / git-missing / rev-parse-error infer/ root
    yields an ``<UNVERIFIABLE:...>`` sentinel and
    ``load_resolver_cache`` RAISES ``StaleResolverCacheError`` with the
    pin-and-clean remedy -- it NEVER returns a regenerable staleness
    that could round-trip back to 'fresh'.

    Reverting to the fail-open ``<not-a-git-repo>`` scheme makes this
    FAIL: the sentinel disappears, the cache validates fresh, and the
    raise never happens.
    """
    cache = tmp_path / "resolve_all.json"

    # 1. Direct: a non-git root yields the UNVERIFIABLE sentinel
    #    (NOT a comparable '<not-a-git-repo>' normal state).
    not_git = tmp_path / "not_a_git_dir"
    not_git.mkdir()
    st = rue._git_root_state(not_git)
    assert st.startswith(rue._INFER_UNVERIFIABLE_PREFIX), (
        f"a non-git infer/ root produced {st!r} -- it must be an "
        f"<UNVERIFIABLE:...> sentinel (HOLE 1 fail-open reverted: a "
        f"non-git box would silently validate the cache as fresh)"
    )

    # 2. Build a cache while infer/ is CLEAN (so the artifact itself is
    #    well-formed), then simulate the LIVE box being non-git and
    #    assert load_resolver_cache FAILS CLOSED (raises), not fresh.
    rue.write_resolver_cache(cache, [])
    _v, stale = rue.load_resolver_cache(cache)
    assert stale is None, f"clean-git baseline wrongly stale: {stale}"

    monkeypatch.setattr(
        rue, "build_infer_state",
        lambda: {
            "vllm": rue._INFER_UNVERIFIABLE_PREFIX + "git-unavailable:vllm>",
            "sglang": rue._INFER_UNVERIFIABLE_PREFIX + "not-a-git-work-tree:python>",
        },
    )
    with pytest.raises(rue.StaleResolverCacheError) as ei:
        rue.load_resolver_cache(cache)
    msg = str(ei.value)
    assert "NOT a verifiable clean git checkout" in msg, msg
    assert "--refresh-cache" in msg and "version_matrix" in msg, msg


def test_hole2_dirty_infer_tree_is_fail_closed_clean_required(
    tmp_path, monkeypatch
):
    """codex r7 [confirmed] HOLE 2 -- FAIL ON REVERT.

    The OLD dirty-state guard only hashed porcelain TEXT and appended a
    ``+dirty`` marker as if a dirty tree were a comparable state -- so
    a cache built dirty survived later byte-edits to the SAME dirty
    file (porcelain text unchanged). The fix is CLEAN-REQUIRED: a
    non-empty ``git status --porcelain`` yields the UNVERIFIABLE
    sentinel; both cache GENERATION (loud stderr + recorded
    unverifiable) and READ fail-closed.

    Reverting to the ``rev+dirty:<hash(porcelain)>`` scheme makes this
    FAIL: a dirty state would be a normal comparable string and the
    cache built/loaded dirty would validate fresh.
    """
    import json

    cache = tmp_path / "resolve_all.json"

    # A real git repo with an UNCOMMITTED change (dirty porcelain).
    import subprocess

    repo = tmp_path / "dirty_infer"
    repo.mkdir()
    for cmd in (
        ["git", "-C", str(repo), "init", "-q"],
        ["git", "-C", str(repo), "config", "user.email", "t@t"],
        ["git", "-C", str(repo), "config", "user.name", "t"],
    ):
        subprocess.run(cmd, capture_output=True)
    (repo / "a.py").write_text("x = 1\n")
    subprocess.run(
        ["git", "-C", str(repo), "add", "a.py"], capture_output=True
    )
    subprocess.run(
        ["git", "-C", str(repo), "commit", "-q", "-m", "init"],
        capture_output=True,
    )
    # Dirty it: byte-edit the tracked file (porcelain now non-empty).
    (repo / "a.py").write_text("x = 2  # uncommitted edit\n")

    st = rue._git_root_state(repo)
    assert st.startswith(rue._INFER_UNVERIFIABLE_PREFIX) and (
        "dirty-not-pinned" in st
    ), (
        f"a DIRTY git infer/ root produced {st!r} -- a dirty tree is "
        f"NOT a reproducible pinned state and MUST be the UNVERIFIABLE "
        f"sentinel (HOLE 2 reverted: porcelain-text scheme would let a "
        f"dirty-built cache validate fresh after a same-file edit)"
    )

    # GENERATION on a dirty infer/: the embedded infer_state is
    # recorded unverifiable, and READ fails closed (a dirty-built
    # cache is NEVER silently trusted later).
    monkeypatch.setattr(
        rue, "build_infer_state",
        lambda: {"vllm": rue._git_root_state(repo)},
    )
    rue.write_resolver_cache(cache, [])
    payload = json.loads(cache.read_text())
    assert rue._infer_state_is_unverifiable(
        payload["_provenance"]["infer_state"]
    ) is not None, (
        "a cache GENERATED while infer/ was dirty did NOT record an "
        "unverifiable infer_state -- a dirty-built cache could be "
        "silently trusted later (HOLE 2 generation guard reverted)"
    )
    with pytest.raises(rue.StaleResolverCacheError) as ei:
        rue.load_resolver_cache(cache)
    assert "DIRTY" in str(ei.value) or "dirty" in str(ei.value), ei.value
    assert "--refresh-cache" in str(ei.value), ei.value


def test_hole3_s2_original_only_infer_import_is_in_upstream_fp(
    tmp_path, monkeypatch
):
    """codex r7 [confirmed] HOLE 3 -- FAIL ON REVERT.

    The per-task ``upstream_fp`` previously hashed ONLY the ORACLE
    subprocess's infer/ closure. The S2 *original* side ALSO imports
    live infer/ (it runs the unmodified upstream kernel.py with infer/
    on sys.path), so an infer/ file imported ONLY on the S2-original
    path was OUTSIDE ``upstream_fp`` -- the belt was incomplete. The
    fix UNIONs the S2-original closure (returned as
    ``s2_smoke_with_entrypoint``'s 3rd element) into the per-task
    accumulator.

    Construction: oracle imports NOTHING under infer/ ([]); the
    S2-original side imports exactly one infer/-rooted file. With the
    fix, that file IS in ``upstream_fp`` (fp != empty sentinel) and a
    byte-edit to it flips ``upstream_fp``. Reverting the union (drop
    the 3rd tuple element / don't ``_closure_acc.update(s2_cl)``) makes
    ``upstream_fp`` collapse to the empty sentinel -> the asserts FAIL.
    """
    # A writable infer surrogate; the resolver byte-hashes by relpath
    # under _INFER_ROOTS exactly as in production.
    fake_vllm = tmp_path / "infer" / "vllm"
    s2only = fake_vllm / "vllm" / "model_executor" / "s2_only_dep.py"
    s2only.parent.mkdir(parents=True)
    original = "MARKER = 'v1'\n"
    s2only.write_text(original)

    td = _make_task(tmp_path, "L1_09_op", _KERNEL_WITH_WRAPPER)

    saved_roots = rue._INFER_ROOTS
    saved_vllm = rue.VLLM_ROOT
    try:
        rue.VLLM_ROOT = fake_vllm
        rue._INFER_ROOTS = (
            ("vllm", fake_vllm),
            ("sglang", rue.SGLANG_ROOT),
        )
        # Oracle imports NOTHING under infer/ ([]).
        monkeypatch.setattr(
            rue, "oracle_consistency_check",
            lambda d, e: (True, "anchored ok", []),
        )
        # S2-original side imported exactly the s2-only infer/ file.
        monkeypatch.setattr(
            rue, "s2_smoke_with_entrypoint",
            lambda d, e, p: ("UNRUN", "S2-smoke passed", [str(s2only)]),
        )
        r = rue.resolve_task(td, pap=None, apply=False)
        assert r.resolved, r
        assert r.upstream_fp != rue._EMPTY_UPSTREAM_CLOSURE_FP, (
            "an infer/ file imported ONLY on the S2-original path is "
            "NOT in the per-task upstream_fp (it collapsed to the "
            "empty sentinel) -- HOLE 3 union reverted: a verdict-"
            "shaping S2-original-only upstream import is invisible"
        )
        fp_before = r.upstream_fp

        # Byte-edit the S2-original-only infer/ file -> upstream_fp
        # MUST flip (the closure genuinely covers the S2-original side).
        s2only.write_text(original.replace("v1", "v2-EDITED"))
        r2 = rue.resolve_task(td, pap=None, apply=False)
        assert r2.upstream_fp != fp_before, (
            "editing an S2-original-only infer/ file did NOT change "
            "upstream_fp -- the S2-original closure is not actually "
            "unioned in (HOLE 3 reverted)"
        )
    finally:
        rue._INFER_ROOTS = saved_roots
        rue.VLLM_ROOT = saved_vllm


# ── codex r8 [confirmed] FIX 1+2 fail-on-revert ──────────────────────


def test_expected_infer_pins_match_meta_consensus():
    """codex r8 [confirmed] FIX 1 -- the AUTHORITATIVE expected pin per
    infer/ root is the corpus meta-consensus (version_matrix defers the
    per-commit pin to tasks/*/meta.yaml). Sanity: each managed root
    resolves to a single real 40-hex commit (NOT an UNVERIFIABLE
    sentinel) on the real corpus."""
    pins = rue.expected_infer_pins()
    assert set(pins) == {"vllm", "sglang"}, pins
    for name, sha in pins.items():
        assert rue._FULL_SHA_RE.match(sha), (
            f"{name} expected pin {sha!r} is not a real commit -- "
            f"meta consensus broke (FIX 1 authority unusable)"
        )


def test_hole4_clean_but_wrong_commit_infer_is_fail_closed(
    tmp_path, monkeypatch
):
    """codex r8 [confirmed] FIX 1 / HOLE 4 -- FAIL ON REVERT.

    The OLD ``_git_root_state`` returned the live clean HEAD verbatim
    and ``load_resolver_cache`` only checked cached==live SELF-
    consistency. So a CLEAN checkout at the WRONG commit produced a
    self-consistent cache that validated "fresh" (trusting
    "clean + same as cache" instead of "clean + pinned at the CORRECT
    commit"). The fix validates the live HEAD against the AUTHORITATIVE
    expected pin (meta consensus); a wrong commit -> a
    ``<UNVERIFIABLE:wrong-commit:...>`` sentinel -> ``load_resolver_cache``
    RAISES.

    Reverting the pin validation (return ``rev`` verbatim, no
    expected-pin check) makes this FAIL: the wrong-commit sentinel
    disappears, build_infer_state stays self-consistent, and the cache
    validates fresh with NO raise.
    """
    cache = tmp_path / "resolve_all.json"

    # Build a well-formed cache while infer/ is the real clean+pinned
    # checkout (the artifact itself is valid).
    rue.write_resolver_cache(cache, [])
    _v, stale = rue.load_resolver_cache(cache)
    assert stale is None, f"clean+pinned baseline wrongly stale: {stale}"

    # 1. Direct: simulate the live HEAD reader returning a WRONG (but
    #    syntactically valid, clean) commit. The expected pin is the
    #    real meta consensus; got != expected -> UNVERIFIABLE.
    real_run = rue.subprocess.run
    wrong_sha = "0" * 40  # clean, resolvable, but NOT the expected pin

    def fake_run(cmd, *a, **k):
        if (
            isinstance(cmd, list)
            and "rev-parse" in cmd
            and "HEAD" in cmd
            and "--is-inside-work-tree" not in cmd
        ):
            class _R:
                returncode = 0
                stdout = wrong_sha + "\n"
                stderr = ""
            return _R()
        return real_run(cmd, *a, **k)

    monkeypatch.setattr(rue.subprocess, "run", fake_run)
    st = rue._git_root_state(rue.VLLM_ROOT, "vllm")
    assert st.startswith(rue._INFER_UNVERIFIABLE_PREFIX) and (
        "wrong-commit" in st
    ), (
        f"a CLEAN infer/ root at the WRONG commit produced {st!r} -- "
        f"it MUST be a <UNVERIFIABLE:wrong-commit:...> sentinel (FIX 1 "
        f"reverted: a clean wrong-commit would be trusted verbatim)"
    )

    # 2. load_resolver_cache must FAIL CLOSED (raise) -- never return a
    #    regenerable staleness that round-trips to fresh, never silently
    #    trust the self-consistent-but-wrong cache.
    with pytest.raises(rue.StaleResolverCacheError) as ei:
        rue.load_resolver_cache(cache)
    msg = str(ei.value)
    assert "WRONG-commit" in msg or "wrong-commit" in msg, msg
    assert "--refresh-cache" in msg, msg


def test_fix2_unverifiable_checked_before_regenerable_no_regen(
    tmp_path, monkeypatch
):
    """codex r8 [confirmed] FIX 2 -- FAIL-CLOSED ORDERING, FAIL ON
    REVERT.

    The OLD order checked ``code_fingerprint`` (a REGENERABLE
    staleness) BEFORE the UNVERIFIABLE infer_state guard. With a
    code-fp mismatch + an UNVERIFIABLE/dirty/wrong-commit infer,
    ``load_resolver_cache`` returned ``"fingerprint-mismatch"`` so the
    caller attempted ONE full regen before a re-read finally raised --
    a pointless "regen-then-raise dance". The fix moves the
    UNVERIFIABLE checks AHEAD of the regenerable checks so the FIRST
    ``load_resolver_cache`` call raises IMMEDIATELY with NO regen
    attempt.

    Construction: a cache with a deliberately WRONG ``code_fingerprint``
    (regenerable staleness present) AND a live infer state forced
    UNVERIFIABLE. With the fix the very first call RAISES (no
    regenerable staleness ever surfaces). Reverting the reorder makes
    the first call return ``"fingerprint-mismatch"`` (regenerable),
    proving the regen-then-raise dance -- the assertion FAILS.
    """
    import json

    cache = tmp_path / "resolve_all.json"
    rue.write_resolver_cache(cache, [])

    # Inject a regenerable staleness: corrupt the code_fingerprint.
    payload = json.loads(cache.read_text())
    payload["_provenance"]["code_fingerprint"] = "deadbeef-wrong-codefp"
    cache.write_text(json.dumps(payload))

    # Force the LIVE infer state UNVERIFIABLE (e.g. dirty/non-git).
    monkeypatch.setattr(
        rue, "build_infer_state",
        lambda: {
            "vllm": rue._INFER_UNVERIFIABLE_PREFIX
            + "dirty-not-pinned:vllm>"
        },
    )

    # The reorder must make load_resolver_cache RAISE on the FIRST call
    # -- never return the regenerable "fingerprint-mismatch" first.
    with pytest.raises(rue.StaleResolverCacheError) as ei:
        rue.load_resolver_cache(cache)
    assert "--refresh-cache" in str(ei.value), ei.value

    # Belt: assert it RAISES (never returns a regenerable staleness
    # tuple) on an UNVERIFIABLE live state -- no caller could
    # regen-then-raise. If the reorder is reverted this returns
    # ("...", "fingerprint-mismatch") instead of raising.
    raised = False
    try:
        rue.load_resolver_cache(cache)
    except rue.StaleResolverCacheError:
        raised = True
    assert raised, (
        "load_resolver_cache returned a regenerable staleness on an "
        "UNVERIFIABLE live infer state -- FIX 2 reorder reverted (the "
        "caller would attempt a futile regen before the re-read raised)"
    )


def test_fix2_regenerable_path_intact_on_clean_pinned_infer(tmp_path):
    """codex r8 [confirmed] FIX 2 -- the regenerable auto-regen path
    MUST stay intact for a code-fp drift on a VERIFIABLE clean-pinned
    infer/ (the legitimate "resolver code changed, regenerate" case).
    A code-fp mismatch with a clean+pinned live infer/ must still
    return the REGENERABLE ``"fingerprint-mismatch"`` (NOT raise) so
    the existing auto-regen behaviour is preserved."""
    import json

    cache = tmp_path / "resolve_all.json"
    rue.write_resolver_cache(cache, [])
    payload = json.loads(cache.read_text())
    payload["_provenance"]["code_fingerprint"] = "deadbeef-old-codefp"
    cache.write_text(json.dumps(payload))

    # Live infer/ here is the real clean+pinned checkout -> NOT
    # UNVERIFIABLE -> the code-fp drift stays REGENERABLE.
    _v, stale = rue.load_resolver_cache(cache)
    assert stale == "fingerprint-mismatch", (
        f"a code-fp drift on a clean+pinned infer/ returned {stale!r} "
        f"-- the regenerable auto-regen path was broken by FIX 2 "
        f"(it must NOT hard-fail; only UNVERIFIABLE infer hard-fails)"
    )


# ── codex r9 [confirmed] EDGE FIXES 1/2/3 -- FAIL ON REVERT ───────────


def test_r9_fix1_missing_managed_root_is_fail_closed_not_omitted(
    tmp_path, monkeypatch
):
    """codex r9 [confirmed] FIX 1 -- FAIL ON REVERT.

    The OLD ``build_infer_state`` only recorded a root when
    ``root.is_dir()``; a MISSING managed root (one the corpus
    meta-consensus pins -- vllm / sglang) was SILENTLY OMITTED. A cache
    generated in that same absent-root state then validated "fresh"
    (omission == omission self-consistency: a live-upstream hole). The
    fix records a ``<UNVERIFIABLE:missing-root:<name>>`` sentinel for an
    absent MANAGED root so build AND load both fail closed.

    Reverting to "only is_dir() roots recorded" makes this FAIL: the
    missing-root sentinel disappears, build_infer_state stays
    self-consistent across the absent state, and load_resolver_cache
    validates fresh with NO raise.
    """
    cache = tmp_path / "resolve_all.json"

    # Baseline: real clean+pinned infer/ -> well-formed cache.
    rue.write_resolver_cache(cache, [])
    _v, stale = rue.load_resolver_cache(cache)
    assert stale is None, f"clean+pinned baseline wrongly stale: {stale}"

    # vllm + sglang are the meta-consensus-pinned (managed) roots.
    managed = rue._managed_root_names()
    assert managed == {"vllm", "sglang"}, (
        f"managed roots changed: {managed} -- meta consensus authority"
    )

    # 1. Simulate the vllm managed root ABSENT (point it at a path that
    #    does not exist; sglang stays the real clean+pinned root).
    missing = tmp_path / "no_such_infer_root"
    monkeypatch.setattr(
        rue, "_INFER_ROOTS",
        (("vllm", missing), ("sglang", rue.SGLANG_ROOT)),
    )
    st = rue.build_infer_state()
    assert st.get("vllm", "").startswith(
        rue._INFER_UNVERIFIABLE_PREFIX
    ) and "missing-root" in st.get("vllm", ""), (
        f"an ABSENT managed root produced {st.get('vllm')!r} -- it MUST "
        f"be a <UNVERIFIABLE:missing-root:...> sentinel (FIX 1 "
        f"reverted: a missing managed root would be silently omitted "
        f"and self-consistently validate 'fresh')"
    )

    # 2. load_resolver_cache must FAIL CLOSED (raise), never fresh.
    with pytest.raises(rue.StaleResolverCacheError) as ei:
        rue.load_resolver_cache(cache)
    msg = str(ei.value)
    assert "MISSING managed-root" in msg or "missing-root" in msg, msg
    assert "--refresh-cache" in msg, msg

    # 3. An UNMANAGED absent root (no meta-consensus pin) stays
    #    correctly OMITTED (not a spurious missing-root sentinel).
    monkeypatch.setattr(
        rue, "_managed_root_names", lambda: set()
    )
    st2 = rue.build_infer_state()
    assert "vllm" not in st2, (
        f"an UNMANAGED absent root was recorded {st2.get('vllm')!r} -- "
        f"only meta-consensus-pinned (managed) roots must fail closed "
        f"on absence; unmanaged absent roots stay omitted"
    )


def test_r9_fix2_unverifiable_before_missing_cache_no_regen(
    tmp_path, monkeypatch
):
    """codex r9 [confirmed] FIX 2 -- FAIL ON REVERT.

    The r8 code computed the LIVE-infer UNVERIFIABLE guard only AFTER
    the ``missing`` / ``unparseable`` / ``no-provenance`` EARLY RETURNS
    in ``load_resolver_cache``. So with an UNVERIFIABLE live infer/ AND
    a missing/unparseable cache, ``load_resolver_cache`` returned the
    regenerable ``"missing"`` -> ``_load_resolver_verdicts`` ran ONE
    full ``run_resolver`` regen under the SAME bad upstream before the
    re-read finally raised: the exact regen-then-raise dance the r8
    comment WRONGLY claimed was eliminated (a self-description
    overclaim). The fix runs the live-infer UNVERIFIABLE guard FIRST OF
    ALL (before the file is even read) so it RAISES IMMEDIATELY with NO
    regen attempt, even when the cache is missing/unparseable.

    Reverting (live check after the missing/unparseable returns) makes
    this FAIL: the first call returns ("...", "missing") (regenerable),
    so the regen path is reached -- the assertions below FAIL.
    """
    # 1. NO cache file at all + UNVERIFIABLE live infer/.
    missing_cache = tmp_path / "does_not_exist.json"
    monkeypatch.setattr(
        rue, "build_infer_state",
        lambda: {
            "vllm": rue._INFER_UNVERIFIABLE_PREFIX
            + "dirty-not-pinned:vllm>"
        },
    )

    # Sentinel: run_resolver MUST NOT be invoked. If the live-infer
    # guard does not precede the "missing" early return, the caller
    # would regen first -- but load_resolver_cache itself must RAISE
    # before ever returning a regenerable "missing".
    called = {"run_resolver": 0}
    real_run_resolver = rue.run_resolver

    def spy_run_resolver(*a, **k):
        called["run_resolver"] += 1
        return real_run_resolver(*a, **k)

    monkeypatch.setattr(rue, "run_resolver", spy_run_resolver)

    with pytest.raises(rue.StaleResolverCacheError) as ei:
        rue.load_resolver_cache(missing_cache)
    msg = str(ei.value)
    assert "--refresh-cache" in msg, msg
    assert called["run_resolver"] == 0, (
        "run_resolver was invoked before load_resolver_cache raised on "
        "an UNVERIFIABLE live infer + MISSING cache -- FIX 2 reverted "
        "(the live-infer guard no longer precedes the 'missing' early "
        "return; the regen-then-raise dance is observable)"
    )

    # 2. UNPARSEABLE cache + UNVERIFIABLE live infer/ -> same: RAISE
    #    before the "unparseable" early return, NO regen.
    bad_cache = tmp_path / "garbage.json"
    bad_cache.write_text("{ this is not valid json :::")
    with pytest.raises(rue.StaleResolverCacheError):
        rue.load_resolver_cache(bad_cache)

    # 3. Drive the FULL _load_resolver_verdicts path to PROVE no regen
    #    is attempted end-to-end (the caller raises straight through).
    #    phase_a_prime's ``_load_resolver_module()`` returns the SAME
    #    sys.modules["resolve_upstream_entrypoints"] == this test's
    #    ``rue``, so patching ``rue`` here patches what pap uses.
    import importlib.util as _ilu

    pap_spec = _ilu.spec_from_file_location(
        "phase_a_prime", REPO / "scripts" / "phase_a_prime.py"
    )
    pap = _ilu.module_from_spec(pap_spec)
    sys.modules["phase_a_prime"] = pap
    pap_spec.loader.exec_module(pap)
    pap._RESOLVER_CACHE_MEM = None
    monkeypatch.setattr(pap, "RESOLVE_CACHE", missing_cache)
    # rue.build_infer_state already monkeypatched UNVERIFIABLE above.
    pap_called = {"run_resolver": 0}

    def pap_spy(*a, **k):
        pap_called["run_resolver"] += 1
        raise AssertionError("run_resolver must NOT be reached")

    monkeypatch.setattr(rue, "run_resolver", pap_spy)
    with pytest.raises(pap.StaleResolverCacheError):
        pap._load_resolver_verdicts()
    assert pap_called["run_resolver"] == 0, (
        "_load_resolver_verdicts attempted a regen on an UNVERIFIABLE "
        "live infer + MISSING cache -- FIX 2 reverted (regen-then-raise "
        "dance the r8 comment wrongly claimed was gone)"
    )


def test_r9_fix2_regenerable_path_intact_on_clean_pinned(tmp_path):
    """codex r9 [confirmed] FIX 2 -- the legitimate REGENERABLE path
    (code-fp drift on a verifiable clean+pinned infer/) MUST still
    auto-regen. With the real clean+pinned infer/ here, a code-fp
    mismatch returns the regenerable ``"fingerprint-mismatch"`` (NOT a
    raise) -- only an UNVERIFIABLE live infer/ hard-fails. An over-strict
    gate that raised here would break the happy regeneration workflow.
    """
    import json

    cache = tmp_path / "resolve_all.json"
    rue.write_resolver_cache(cache, [])
    payload = json.loads(cache.read_text())
    payload["_provenance"]["code_fingerprint"] = "deadbeef-old-codefp"
    cache.write_text(json.dumps(payload))

    _v, stale = rue.load_resolver_cache(cache)
    assert stale == "fingerprint-mismatch", (
        f"a code-fp drift on a clean+pinned infer/ returned {stale!r} "
        f"-- the regenerable auto-regen path was broken (FIX 2 must "
        f"only hard-fail an UNVERIFIABLE live infer/, never the "
        f"legitimate clean-pinned regeneration)"
    )


def test_r9_fix3_malformed_managed_commit_sha_is_fail_closed(
    tmp_path, monkeypatch
):
    """codex r9 [confirmed] FIX 3 -- FAIL ON REVERT.

    The OLD ``expected_infer_pins`` SILENTLY DROPPED a managed repo's
    ``upstream.commit_sha`` that was neither a valid 40-hex SHA nor the
    explicit ``snapshot-no-git`` sentinel (it was neither
    ``continue``-skipped as a snapshot nor added by the 40-hex filter).
    As long as some OTHER task pinned a valid SHA, the garbage value
    vanished without trace -- the silent-ignore anti-pattern. The fix
    poisons that root's expected pin ->
    ``<UNVERIFIABLE:malformed-commit-sha:...>`` (fail-closed), taking
    precedence over any otherwise-valid consensus.

    Reverting (silent drop) makes this FAIL: the malformed value is
    ignored and the root resolves to the valid SHA as if nothing were
    wrong.
    """
    # Build a tiny temp corpus: one VALID-SHA task + one MALFORMED-SHA
    # task for the SAME managed repo, plus a snapshot-no-git task that
    # must STAY the deliberate (non-malformed) exclusion.
    valid_sha = "a" * 40
    tasks_root = tmp_path / "tasks"
    (tasks_root / "level1" / "good").mkdir(parents=True)
    (tasks_root / "level1" / "good" / "meta.yaml").write_text(
        "upstream:\n"
        "  repo: vllm-project/vllm\n"
        f"  commit_sha: {valid_sha}\n"
    )
    (tasks_root / "level1" / "bad").mkdir(parents=True)
    (tasks_root / "level1" / "bad" / "meta.yaml").write_text(
        "upstream:\n"
        "  repo: vllm-project/vllm\n"
        "  commit_sha: not-a-real-sha-deadbeef\n"
    )
    (tasks_root / "level1" / "snap").mkdir(parents=True)
    (tasks_root / "level1" / "snap" / "meta.yaml").write_text(
        "upstream:\n"
        "  repo: sgl-project/sglang\n"
        "  commit_sha: snapshot-no-git\n"
    )

    monkeypatch.setattr(rue, "TASKS_ROOT", tasks_root)
    pins = rue.expected_infer_pins()

    # vllm has a malformed sha alongside a valid one -> UNVERIFIABLE,
    # NEVER silently the valid SHA.
    assert pins["vllm"].startswith(rue._INFER_UNVERIFIABLE_PREFIX) and (
        "malformed-commit-sha" in pins["vllm"]
    ), (
        f"vllm pin = {pins['vllm']!r} -- a malformed managed commit_sha "
        f"MUST poison the pin (<UNVERIFIABLE:malformed-commit-sha:...>) "
        f"and NEVER be silently dropped in favour of the valid SHA "
        f"(FIX 3 reverted)"
    )
    assert valid_sha not in pins["vllm"], (
        "the valid SHA leaked through despite a malformed sibling -- "
        "FIX 3 reverted (silent-ignore: malformed dropped, valid wins)"
    )

    # snapshot-no-git stays the deliberate documented exclusion: NOT
    # malformed -> sglang has no real pin -> the existing no-meta-pin
    # sentinel (NOT malformed-commit-sha).
    assert "malformed-commit-sha" not in pins["sglang"], (
        f"snapshot-no-git was wrongly treated as malformed: "
        f"{pins['sglang']!r} -- it is the deliberate non-git exclusion"
    )
    assert "no-meta-pin" in pins["sglang"], pins["sglang"]

    # codex r10 MUST-FIX 2(b) [confirmed] -- a managed task REFERENCED
    # this root; its pin came back malformed (UNVERIFIABLE). The r9-era
    # assertion here ("must NOT be in the managed set -- it already
    # fails closed via its own sentinel") was the residual double
    # fail-open codex r10 [confirmed]: the "fails closed via its own
    # sentinel" claim ONLY holds when the root EXISTS on disk (so
    # ``_git_root_state`` runs). A bad-pin-AND-missing root skipped BOTH
    # the is_dir() branch AND the (excluded) managed branch -> SILENT
    # OMISSION. The corrected invariant: a referenced-by-a-task root IS
    # managed regardless of whether its pin resolved, so it fails closed
    # (missing-root) when also absent -- never silently omitted.
    assert "vllm" in rue._managed_root_names(), (
        "codex r10 MUST-FIX 2(b) reverted: a root a managed task "
        "references but whose pin is UNVERIFIABLE (malformed) MUST be "
        "in the managed set so a bad-pin-AND-missing root fails closed "
        "(missing-root), never silently omitted from build_infer_state"
    )


def _fresh_pap():
    """A brand-new ``phase_a_prime`` module instance whose
    ``_load_resolver_module()`` resolves to a fresh
    ``resolve_upstream_entrypoints`` -- so this test exercises the
    PRODUCTION memo in isolation WITHOUT depending on the conftest
    autouse memo reset (codex r10 MUST-FIX 1 requires the production
    code, not the test fixture, to be what makes correctness hold)."""
    import importlib.util as _ilu

    rue_spec = _ilu.spec_from_file_location(
        "resolve_upstream_entrypoints",
        REPO / "scripts" / "resolve_upstream_entrypoints.py",
    )
    rue_i = _ilu.module_from_spec(rue_spec)
    sys.modules["resolve_upstream_entrypoints"] = rue_i
    rue_spec.loader.exec_module(rue_i)
    pap_spec = _ilu.spec_from_file_location(
        "phase_a_prime", REPO / "scripts" / "phase_a_prime.py"
    )
    pap_i = _ilu.module_from_spec(pap_spec)
    sys.modules["phase_a_prime"] = pap_i
    pap_spec.loader.exec_module(pap_i)
    return pap_i, rue_i


def test_r10_mustfix1_resolver_memo_is_provenance_keyed_not_bare(
    tmp_path,
):
    """codex r10 MUST-FIX 1 [confirmed PRODUCTION bug] -- FAIL ON
    REVERT, and does NOT rely on the conftest memo reset.

    The OLD ``_RESOLVER_CACHE_MEM`` was a bare unkeyed process-global:
    once populated, ``_load_resolver_verdicts`` returned it verbatim
    even after a within-process change to (a) the RESOLVE_CACHE path,
    (b) the cache file contents, or (c) the code-fp / infer_state
    provenance -- bypassing the entire fail-closed re-validation. The
    conftest autouse reset merely HID this in the suite; it was a
    genuine production leak. The memo is now PROVENANCE-KEYED and
    self-invalidating.

    This test NEVER calls the conftest reset helper -- it asserts the
    PRODUCTION behaviour directly. Reverting to a bare global makes
    every assertion below FAIL (the stale memo is served).
    """
    import json

    pap, rue = _fresh_pap()

    # Two distinct valid caches written by the REAL writer (so each is
    # genuinely fresh under the current code+clean-pinned infer/).
    cache_a = tmp_path / "a.json"
    cache_b = tmp_path / "b.json"

    class _R:
        def __init__(self, tid):
            self.__dict__ = {
                "task_id": tid, "resolved": False,
                "status": "S", "reason": "r",
            }

    rue.write_resolver_cache(cache_a, [_R("TASK_A")])
    rue.write_resolver_cache(cache_b, [_R("TASK_B")])

    # 1. Populate the memo from cache_a (no conftest reset involved).
    pap.RESOLVE_CACHE = cache_a
    pap._RESOLVER_CACHE_MEM = None
    v1 = pap._load_resolver_verdicts()
    assert set(v1) == {"TASK_A"}, v1
    assert pap._RESOLVER_CACHE_MEM is not None
    memo_key_a = pap._RESOLVER_CACHE_MEM[0]

    # 2. PATH CHANGE within the process -> the memo MUST NOT serve
    #    cache_a's verdicts for cache_b. (Bare-global revert: returns
    #    TASK_A here -> assertion FAILS.)
    pap.RESOLVE_CACHE = cache_b
    v2 = pap._load_resolver_verdicts()
    assert set(v2) == {"TASK_B"}, (
        f"stale memo served {set(v2)} after RESOLVE_CACHE was repointed "
        f"-- the bare-unkeyed-global production bug is back (codex r10 "
        f"MUST-FIX 1 reverted)"
    )
    assert pap._RESOLVER_CACHE_MEM[0] != memo_key_a, (
        "memo key did not change after a path change -- not "
        "provenance-keyed"
    )

    # 3. CACHE-FILE CONTENT CHANGE under the SAME path -> the memo MUST
    #    re-load. Rewrite cache_b in place with a different task.
    import time

    time.sleep(0.01)  # ensure a distinct st_mtime_ns on coarse FS
    rue.write_resolver_cache(cache_b, [_R("TASK_B2")])
    v3 = pap._load_resolver_verdicts()
    assert set(v3) == {"TASK_B2"}, (
        f"stale memo served {set(v3)} after the cache FILE was "
        f"regenerated in place -- memo not keyed on the cache file "
        f"identity (codex r10 MUST-FIX 1 reverted)"
    )

    # 3b. codex r11 MUST-FIX 1 [confirmed] -- SAME-SIZE + MTIME-RESTORED
    #     content edit. This is the tamper case ``(st_mtime_ns,
    #     st_size)`` ALONE cannot catch: edit the cache file to a
    #     DIFFERENT verdict-set via an EQUAL-LENGTH in-place byte
    #     substitution (so st_size is byte-identical), then restore the
    #     ORIGINAL mtime. With a stat-only memo key the (mtime, size)
    #     tuple is byte-identical -> the OLD verdicts (TASK_B2) are
    #     served STALE. Only a CONTENT HASH of the cache bytes drops the
    #     memo here. Reverting the key to mtime+size only makes this
    #     assertion FAIL (stale TASK_B2 served). The on-disk JSON layout
    #     (indent / _provenance stamp) is preserved EXACTLY -- we mutate
    #     the raw text by an equal-length token swap, never re-serialize
    #     (re-serializing would change size AND drop the matching
    #     code_fp/infer_state stamp, defeating the scenario).
    st_before = cache_b.stat()
    raw_before = cache_b.read_text()
    assert raw_before.count('"TASK_B2"') == 1, (
        f"test setup broken: expected exactly one '\"TASK_B2\"' token "
        f"in the cache, found {raw_before.count('\"TASK_B2\"')}"
    )
    # "TASK_B2" -> "TASK_XY": both 7 chars, equal-length quoted token
    # -> the file is byte-for-byte the SAME SIZE, only contents differ.
    raw_after = raw_before.replace('"TASK_B2"', '"TASK_XY"')
    assert len(raw_after) == len(raw_before) and raw_after != raw_before
    cache_b.write_text(raw_after)
    # Restore the ORIGINAL atime/mtime so (st_mtime_ns, st_size) is
    # byte-identical to before the edit -- the exact attack a
    # stat-only key cannot see.
    import os as _os3

    _os3.utime(
        cache_b, ns=(st_before.st_atime_ns, st_before.st_mtime_ns)
    )
    st_after = cache_b.stat()
    assert (
        st_after.st_mtime_ns == st_before.st_mtime_ns
        and st_after.st_size == st_before.st_size
    ), (
        "test setup broken: (st_mtime_ns, st_size) is NOT identical "
        "after the restore -- cannot exercise the content-hash leak"
    )
    v3b = pap._load_resolver_verdicts()
    assert set(v3b) == {"TASK_XY"}, (
        f"stale memo served {set(v3b)} after a SAME-SIZE + "
        f"MTIME-RESTORED content edit -- the memo key trusts "
        f"(st_mtime_ns, st_size) alone instead of a CONTENT hash of "
        f"the cache bytes (codex r11 MUST-FIX 1 reverted: any content "
        f"change MUST invalidate)"
    )

    # 4. PROVENANCE CHANGE (code-fp) with the file/path unchanged ->
    #    the memo MUST drop and the cache re-validate fail-closed.
    #    Mutate the on-disk provenance so it is fingerprint-stale, and
    #    suppress regen so the contract is a deterministic hard-fail.
    payload = json.loads(cache_b.read_text())
    payload["_provenance"]["code_fingerprint"] = "deadbeef-stale"
    cache_b.write_text(json.dumps(payload))
    import os as _os

    old = _os.environ.get("PA_NO_CACHE_REGEN")
    _os.environ["PA_NO_CACHE_REGEN"] = "1"
    try:
        with pytest.raises(pap.StaleResolverCacheError):
            pap._load_resolver_verdicts()
    finally:
        if old is None:
            _os.environ.pop("PA_NO_CACHE_REGEN", None)
        else:
            _os.environ["PA_NO_CACHE_REGEN"] = old
    # The stale verdicts were NEVER memoised after the fail-closed
    # raise (a bare global would have kept TASK_B2 and served it).
    assert pap._RESOLVER_CACHE_MEM is None, (
        "memo retained after a fail-closed raise -- a stale provenance "
        "memo can still be served (codex r10 MUST-FIX 1 reverted)"
    )


def test_r10_mustfix1_memo_hit_serves_same_object_when_key_stable(
    tmp_path,
):
    """codex r10 MUST-FIX 1 -- the keyed memo is still a REAL memo: a
    second call with an UNCHANGED path / cache file / provenance MUST
    serve the SAME cached dict (the JSON read + 101-entry rebuild is
    genuinely skipped). Proves the provenance key did not turn the memo
    into a no-op (codex r10 point 6 honesty check)."""
    pap, rue = _fresh_pap()
    cache = tmp_path / "c.json"

    class _R:
        def __init__(self):
            self.__dict__ = {
                "task_id": "T", "resolved": False,
                "status": "S", "reason": "r",
            }

    rue.write_resolver_cache(cache, [_R()])
    pap.RESOLVE_CACHE = cache
    pap._RESOLVER_CACHE_MEM = None
    a = pap._load_resolver_verdicts()
    b = pap._load_resolver_verdicts()
    assert b is a, (
        "a second call with an unchanged key did NOT serve the memoised "
        "object -- the keyed memo degenerated into a no-op (it must "
        "still skip the JSON read+rebuild on a genuine hit)"
    )


def test_r12_mustfix_memo_store_toctou_single_snapshot(tmp_path):
    """codex r12 MUST-FIX [confirmed] -- FAIL ON REVERT (the parse-vs
    -store TOCTOU).

    The load+memoize path used to read the cache file THREE times:
      1. ``_resolver_memo_key`` content-sha (pre-check / live key),
      2. ``load_resolver_cache`` parse,
      3. a POST-PARSE ``_resolver_memo_key`` RECOMPUTE for the store.
    ``--refresh-cache`` / ``_writeback_fresh_verdict`` rewrite this
    EXACT file, so a rewrite landing BETWEEN read #2 (parse) and read
    #3 (store-key recompute) made the memo store content-A's verdicts
    under a key whose ``content_part`` hashed content-B. A later access
    whose live bytes == B would HIT that key and be served the STALE
    content-A verdicts as if fresh -- a genuine TOCTOU.

    This test SIMULATES that race: ``Path.read_bytes`` on the cache
    file is wrapped so that, immediately AFTER the parse read inside
    ``load_resolver_cache`` returns, the file is rewritten in place to
    content-B (a DIFFERENT, still-fresh verdict-set). Any post-parse
    re-read therefore sees content-B.

    SINGLE-SNAPSHOT fix invariant (what this asserts): the stored memo
    key's authoritative ``content_part`` is sha256 of the EXACT bytes
    whose verdicts were stored. With the fix: verdicts == {TASK_A} and
    ``content_part`` == sha256(content-A) -- provably the SAME bytes.
    Reverting the collapse (re-introducing the post-parse recompute)
    makes ``content_part`` == sha256(content-B) while the stored
    verdicts are still {TASK_A} -> the assertion FAILS, exposing the
    A-verdicts-under-B-key stale serve.

    Plus a STRUCTURAL guard: ZERO reads of the cache file occur between
    the parse and the memo store (the eliminated read #3). A revert
    re-introduces a read there and trips the read-count assertion.
    """
    import hashlib
    import json

    pap, rue = _fresh_pap()
    cache = tmp_path / "c.json"

    class _R:
        def __init__(self, tid):
            self.__dict__ = {
                "task_id": tid, "resolved": False,
                "status": "S", "reason": "r",
            }

    # content-A: a genuinely fresh cache (current code + clean-pinned
    # infer/ stamp written by the REAL writer).
    rue.write_resolver_cache(cache, [_R("TASK_A")])
    bytes_a = cache.read_bytes()
    sha_a = hashlib.sha256(bytes_a).hexdigest()

    # content-B: SAME provenance stamp (so it is ALSO "fresh"), only
    # the verdict task_id differs. Re-using A's _provenance keeps B
    # load-valid -- the race is about WHICH bytes the stored key hashes,
    # not about staleness.
    payload_a = json.loads(bytes_a)
    payload_b = dict(payload_a)
    payload_b["verdicts"] = [{
        "task_id": "TASK_B", "resolved": False,
        "status": "S", "reason": "r",
    }]
    bytes_b = json.dumps(payload_b, indent=2).encode()
    sha_b = hashlib.sha256(bytes_b).hexdigest()
    assert sha_a != sha_b

    pap.RESOLVE_CACHE = cache
    pap._RESOLVER_CACHE_MEM = None

    # Instrument Path.read_bytes for THIS cache file only. On a MISS
    # the read order is: #1 ``_resolver_memo_key`` pre-check content
    # -sha (the cheap live-key probe), #2 ``load_resolver_cache`` parse.
    # The single-snapshot fix stores the memo from the read-#2 snapshot
    # with NO further read. We inject the concurrent --refresh-cache
    # rewrite (-> content-B) immediately AFTER read #2 (the parse), so
    # ANY post-parse re-read (the eliminated read #3) observes B. We
    # record reads occurring AFTER that rewrite (== post-parse re-reads:
    # 0 with the fix; the 3-read revert makes it >= 1).
    real_read_bytes = Path.read_bytes
    state = {"reads": 0, "post_rewrite_reads": 0, "rewritten": False}

    def _instrumented_read_bytes(self):
        same = False
        try:
            same = self.resolve() == cache.resolve()
        except OSError:
            same = False
        if not same:
            return real_read_bytes(self)
        state["reads"] += 1
        if state["rewritten"]:
            state["post_rewrite_reads"] += 1
        data = real_read_bytes(self)
        # After read #2 (the load_resolver_cache parse), inject the
        # race: rewrite to content-B before any post-parse re-read.
        if state["reads"] == 2 and not state["rewritten"]:
            cache.write_bytes(bytes_b)
            state["rewritten"] = True
        return data

    Path.read_bytes = _instrumented_read_bytes
    try:
        verdicts = pap._load_resolver_verdicts()
    finally:
        Path.read_bytes = real_read_bytes

    # The memo MUST have been stored.
    assert pap._RESOLVER_CACHE_MEM is not None
    stored_key, stored_verdicts = pap._RESOLVER_CACHE_MEM

    # The verdicts returned/stored come from the bytes the parse read
    # (content-A) -- the rewrite happened AFTER the parse.
    assert set(stored_verdicts) == {"TASK_A"}, (
        f"expected content-A verdicts, got {set(stored_verdicts)}"
    )
    assert set(verdicts) == {"TASK_A"}

    # AUTHORITATIVE INVARIANT: the stored key's content_part (the
    # 3rd tuple element -- sha256 of the bytes the verdicts were
    # derived from) MUST be sha256(content-A), NEVER sha256(content-B).
    # The 3-read revert hashes content-B here while the verdicts are
    # content-A => A-verdicts-under-B-key stale-serve => this FAILS.
    content_part = stored_key[2]
    assert content_part == sha_a, (
        f"memo store TOCTOU: stored key content_part={content_part!r} "
        f"is NOT sha256(content-A)={sha_a!r}. The stored verdicts are "
        f"content-A's but the key hashes other bytes -- a later access "
        f"whose live bytes match that hash would be served STALE "
        f"content-A verdicts as fresh (codex r12 MUST-FIX reverted: "
        f"the post-parse key recompute / 3-read split is back)."
    )
    assert content_part != sha_b, (
        f"memo store TOCTOU: stored key content_part == "
        f"sha256(content-B) while verdicts are content-A -- exactly "
        f"the A-verdicts-under-B-key stale serve the single-snapshot "
        f"fix eliminates (codex r12 MUST-FIX reverted)."
    )

    # STRUCTURAL guard: ZERO cache reads occur AFTER the post-parse
    # rewrite -- i.e. the load path performs NO read once the bytes
    # have been hashed+parsed (the eliminated read #3). Re-introducing
    # the post-parse ``_resolver_memo_key`` recompute trips this.
    assert state["post_rewrite_reads"] == 0, (
        f"{state['post_rewrite_reads']} cache read(s) happened AFTER "
        f"the parse -- the single-snapshot collapse was reverted (a "
        f"post-parse re-read feeds the store key, re-opening the "
        f"TOCTOU)."
    )


def test_r10_mustfix2a_blank_absent_managed_commit_sha_fail_closed(
    tmp_path, monkeypatch
):
    """codex r10 MUST-FIX 2(a) [confirmed] -- FAIL ON REVERT.

    A MANAGED repo whose ``upstream.commit_sha`` is blank / whitespace
    / quoted-empty / an ABSENT key is JUST AS UNVERIFIABLE as a
    malformed one. The OLD code's ``\\s+commit_sha:\\s*(\\S+)`` only
    matched a NON-blank value, so a blank/absent pin left ``sha is
    None`` and was SILENTLY ``continue``-skipped before the malformed
    check -- it vanished if some OTHER task pinned a valid SHA. It now
    poisons the root's pin with a distinct ``blank-commit-sha`` /
    ``absent-commit-sha`` fail-closed sentinel.

    Reverting (silent skip) makes this FAIL: the blank/absent pin is
    ignored and the root resolves to the sibling valid SHA.
    """
    valid_sha = "b" * 40
    tasks_root = tmp_path / "tasks"
    # vllm: a VALID-SHA task + a BLANK-commit_sha task (key present,
    # empty value).
    (tasks_root / "level1" / "good").mkdir(parents=True)
    (tasks_root / "level1" / "good" / "meta.yaml").write_text(
        "upstream:\n"
        "  repo: vllm-project/vllm\n"
        f"  commit_sha: {valid_sha}\n"
    )
    (tasks_root / "level1" / "blank").mkdir(parents=True)
    (tasks_root / "level1" / "blank" / "meta.yaml").write_text(
        "upstream:\n"
        "  repo: vllm-project/vllm\n"
        "  commit_sha:   \n"  # KEY present, value blank/whitespace
    )
    # sglang: a VALID-SHA task + a task with the commit_sha key ABSENT
    # entirely from the upstream block.
    (tasks_root / "level1" / "sgood").mkdir(parents=True)
    (tasks_root / "level1" / "sgood" / "meta.yaml").write_text(
        "upstream:\n"
        "  repo: sgl-project/sglang\n"
        f"  commit_sha: {'c' * 40}\n"
    )
    (tasks_root / "level1" / "nosha").mkdir(parents=True)
    (tasks_root / "level1" / "nosha" / "meta.yaml").write_text(
        "upstream:\n"
        "  repo: sgl-project/sglang\n"  # NO commit_sha key at all
    )

    monkeypatch.setattr(rue, "TASKS_ROOT", tasks_root)
    pins = rue.expected_infer_pins()

    assert pins["vllm"].startswith(rue._INFER_UNVERIFIABLE_PREFIX) and (
        "blank-commit-sha" in pins["vllm"]
    ), (
        f"vllm pin = {pins['vllm']!r} -- a BLANK managed commit_sha MUST "
        f"poison the pin (<UNVERIFIABLE:blank-commit-sha:...>), NEVER be "
        f"silently dropped for the sibling valid SHA (MUST-FIX 2a "
        f"reverted)"
    )
    assert valid_sha not in pins["vllm"], (
        "the sibling valid SHA leaked through despite a blank pin -- "
        "MUST-FIX 2a reverted (silent-skip: blank dropped, valid wins)"
    )
    assert pins["sglang"].startswith(
        rue._INFER_UNVERIFIABLE_PREFIX
    ) and "absent-commit-sha" in pins["sglang"], (
        f"sglang pin = {pins['sglang']!r} -- an ABSENT managed "
        f"commit_sha key MUST poison the pin "
        f"(<UNVERIFIABLE:absent-commit-sha:...>), never silently "
        f"skipped (MUST-FIX 2a reverted)"
    )
    assert ("c" * 40) not in pins["sglang"], (
        "the sibling valid SHA leaked through despite an absent pin"
    )


def test_r10_mustfix2b_badpin_and_missing_root_fail_closed(
    tmp_path, monkeypatch
):
    """codex r10 MUST-FIX 2(b) [confirmed] -- FAIL ON REVERT.

    The r9 ``_managed_root_names()`` only included VALID-SHA-pin roots.
    A root a managed task REFERENCES but whose pin is UNVERIFIABLE
    (malformed/blank/absent/ambiguous) was EXCLUDED -- and if that same
    root was ALSO missing on disk, ``build_infer_state`` skipped BOTH
    the is_dir() branch AND the ``elif name in managed`` branch -> the
    root was SILENTLY OMITTED from the cache provenance (the residual
    double fail-open: "already fails closed via its own sentinel" only
    holds when the root EXISTS so ``_git_root_state`` runs).

    The managed set is now union(has-valid-pin,
    referenced-by-a-task-with-UNVERIFIABLE-pin). A bad-pin-AND-missing
    managed root now fails closed (missing-root). Reverting to the
    valid-SHA-only set makes the silent-omission assertion FAIL.
    """
    import json

    # Corpus: vllm pinned MALFORMED (so its pin is UNVERIFIABLE), sglang
    # pinned VALID.
    tasks_root = tmp_path / "tasks"
    (tasks_root / "level1" / "vbad").mkdir(parents=True)
    (tasks_root / "level1" / "vbad" / "meta.yaml").write_text(
        "upstream:\n"
        "  repo: vllm-project/vllm\n"
        "  commit_sha: totally-not-a-sha\n"
    )
    (tasks_root / "level1" / "sgood").mkdir(parents=True)
    (tasks_root / "level1" / "sgood" / "meta.yaml").write_text(
        "upstream:\n"
        "  repo: sgl-project/sglang\n"
        f"  commit_sha: {'d' * 40}\n"
    )
    monkeypatch.setattr(rue, "TASKS_ROOT", tasks_root)

    pins = rue.expected_infer_pins()
    assert "malformed-commit-sha" in pins["vllm"], pins["vllm"]
    # MUST-FIX 2(b): vllm IS managed despite an UNVERIFIABLE pin (a task
    # references it). Reverting -> vllm absent from the set -> FAIL.
    managed = rue._managed_root_names()
    assert "vllm" in managed, (
        f"managed={managed} -- a root referenced by a managed task with "
        f"an UNVERIFIABLE (malformed) pin MUST be managed (MUST-FIX 2b "
        f"reverted: valid-SHA-only set silently omits it)"
    )

    # Now make vllm ALSO missing on disk; sglang stays the real
    # clean+pinned root. build_infer_state MUST record a fail-closed
    # sentinel for vllm, NOT silently omit it.
    missing = tmp_path / "no_such_vllm"
    monkeypatch.setattr(
        rue, "_INFER_ROOTS",
        (("vllm", missing), ("sglang", rue.SGLANG_ROOT)),
    )
    st = rue.build_infer_state()
    assert "vllm" in st and st["vllm"].startswith(
        rue._INFER_UNVERIFIABLE_PREFIX
    ) and "missing-root" in st["vllm"], (
        f"build_infer_state vllm = {st.get('vllm')!r} -- a "
        f"bad-pin-AND-missing managed root MUST fail closed "
        f"(missing-root), NEVER be silently omitted (MUST-FIX 2b "
        f"reverted)"
    )

    # And the cache then HARD-FAILS on load (never validates fresh on a
    # silently-omitted root).
    cache = tmp_path / "resolve_all.json"
    cache.write_text(json.dumps({
        "_provenance": {
            "code_fingerprint": rue.resolver_code_fingerprint(),
            "infer_state": {"sglang": "x"},
        },
        "verdicts": [],
    }))
    with pytest.raises(rue.StaleResolverCacheError):
        rue.load_resolver_cache(cache)


# ── primary-blocker categorisation (reducer contamination guard) ──────
def test_primary_blocker_cat_distinguishes_oracle_consistent():
    """The best-ranked candidate's reject note is categorised so that an
    intended wrapper that matched the oracle and only smoke/shapes-blocked
    is NOT mistaken for a genuine shapes gap."""
    f = rue._primary_blocker_cat
    assert f("bmm_batch_invariant: oracle-consistent but S2-smoke BLOCKED [x]") \
        == "oracle-consistent-smoke-blocked"
    assert f("w: oracle-consistent but S2 shapes-incomplete [y]") \
        == "oracle-consistent-shapes-incomplete"
    assert f("act_and_mul_triton: shapes-incomplete [missing gateup_output]") \
        == "shapes-incomplete"
    assert f("z: oracle-mismatch [value]") == "oracle-mismatch"
    assert f("k: upstream-not-invocable-here [AttributeError arg_names]") \
        == "upstream-not-invocable-here"


def test_resolution_primary_blocker_round_trips_and_back_compat():
    """primary_blocker serialises with the verdict and an OLD cache dict
    lacking the field still deserialises (default '')."""
    r = rue.Resolution(
        "T", False, None, "BLOCKED(no-verified-upstream-entrypoint: shapes-incomplete)",
        "reason", tried=["a", "b"], primary_blocker="oracle-consistent-smoke-blocked",
    )
    assert r.__dict__["primary_blocker"] == "oracle-consistent-smoke-blocked"
    old = {
        "task_id": "T", "resolved": False, "entrypoint": None,
        "status": "BLOCKED(x)", "reason": "r", "tried": [],
        "upstream_fp": rue._EMPTY_UPSTREAM_CLOSURE_FP,
    }
    assert rue.Resolution(**old).primary_blocker == ""


def test_blocked_reason_format_preserves_diagnostic_segments():
    """Regression: the primary_blocker annotation is appended at the END of
    the BLOCKED reason so per_task_diagnostic._split_segments (which needs
    the body to start at the first reject note) still parses every
    candidate. A prefix would drop the first candidate segment."""
    import importlib.util as _ilu
    _s = _ilu.spec_from_file_location(
        "per_task_diagnostic", REPO / "scripts" / "per_task_diagnostic.py"
    )
    _ptd = _ilu.module_from_spec(_s)
    sys.modules["per_task_diagnostic"] = _ptd
    _s.loader.exec_module(_ptd)
    reason = (
        "BLOCKED(no-verified-upstream-entrypoint: shapes-incomplete): tried "
        "['_dequantize_k_cache_fast', 'dequantize_k_cache_paged']; "
        "_dequantize_k_cache_fast: shapes-incomplete [missing X] | "
        "dequantize_k_cache_paged: oracle-mismatch [val]; "
        "primary_blocker=shapes-incomplete"
    )
    names = [e for e, _, _ in _ptd._split_segments(reason)]
    assert "_dequantize_k_cache_fast" in names  # first candidate NOT dropped
    assert "dequantize_k_cache_paged" in names
