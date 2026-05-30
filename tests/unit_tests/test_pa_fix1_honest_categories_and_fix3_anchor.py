"""FIX 1 + FIX 3 contract tests (codex deep-review [confirmed]).

FIX 1 -- the status generator must emit the TRUE per-task category from
the REAL resolver outcome, NEVER the misleading collapsed
``BLOCKED(no-upstream-entrypoint): no module-level callable exists``
bucket. The six honest categories are distinct and never collapsed:
``no-upstream-entrypoint`` / ``upstream-not-invocable-here(<sub>)`` /
``shapes-incomplete`` / ``oracle-mismatch`` / ``UNRUN(needs-gpu)`` /
``PASS``. These tests FAIL ON REVERT to the pre-resolver short-circuit.

FIX 3 -- the oracle-consistency ANCHOR is hardened: it (a) checks >=2
distinct shapes, (b) rejects EXTRA candidate outputs (structure/arity
must match), (c) requires dtype to match. A wrong candidate that
returns oracle-outputs-PLUS-extras, or a different dtype, or coincides
only on one shape, must be REJECTED. These FAIL ON REVERT to the old
single-shape / extra-output-tolerant / dtype-lax anchor.
"""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parent.parent.parent
SCRIPTS = REPO / "scripts"
_ORACLE_RUNNER = SCRIPTS / "_pa_oracle_check_runner.py"

_spec = importlib.util.spec_from_file_location(
    "phase_a_prime", SCRIPTS / "phase_a_prime.py"
)
pap = importlib.util.module_from_spec(_spec)
sys.modules["phase_a_prime"] = pap
_spec.loader.exec_module(pap)

_ocr_spec = importlib.util.spec_from_file_location(
    "_pa_oracle_check_runner", _ORACLE_RUNNER
)
ocr = importlib.util.module_from_spec(_ocr_spec)
sys.modules["_pa_oracle_check_runner"] = ocr
_ocr_spec.loader.exec_module(ocr)


# ─────────────────────────────────────────────────────────────────────
# FIX 1: honest category derivation from the REAL resolver verdict
# ─────────────────────────────────────────────────────────────────────
def test_honest_category_maps_resolver_status_not_collapsed():
    """``_honest_category`` must map the resolver's REAL status string
    to the SIX distinct honest categories -- never collapse the
    heterogeneous truth into one 'no-upstream-entrypoint' bucket. FAILS
    ON REVERT to a pre-resolver short-circuit (which would emit
    no-upstream-entrypoint for everything whose meta says
    ::placeholder)."""
    # upstream-not-invocable-here with the missing-dep sub-cause -- the
    # ~78-task truth the old generator FALSELY called 'no callable'.
    cat, sub = pap._honest_category(
        "BLOCKED(no-verified-upstream-entrypoint: "
        "upstream-not-invocable-here)",
        "tried ['x']; x: upstream-not-invocable-here "
        "[ModuleNotFoundError: No module named 'pybase64']",
    )
    assert cat == "upstream-not-invocable-here", cat
    assert sub == "missing-dep:pybase64", sub

    # relative-import packaging sub-cause is distinct from missing-dep.
    cat, sub = pap._honest_category(
        "BLOCKED(no-verified-upstream-entrypoint: "
        "upstream-not-invocable-here)",
        "ImportError: attempted relative import with no known parent "
        "package",
    )
    assert cat == "upstream-not-invocable-here"
    assert sub == "relative-import-packaging", sub

    # shapes-incomplete is its OWN category, not collapsed.
    cat, sub = pap._honest_category(
        "BLOCKED(no-verified-upstream-entrypoint: shapes-incomplete)", ""
    )
    assert cat == "shapes-incomplete" and sub == ""

    # oracle-mismatch is its OWN category (a correctly-rejected wrong
    # operator), not 'no callable'.
    cat, _ = pap._honest_category(
        "BLOCKED(no-verified-upstream-entrypoint: oracle-mismatch)", ""
    )
    assert cat == "oracle-mismatch"

    # ONLY a genuine no-host-wrapper is no-upstream-entrypoint.
    cat, _ = pap._honest_category(
        "BLOCKED(no-verified-upstream-entrypoint: no-host-wrapper)",
        "no host wrapper launches a @triton.jit kernel (no-host-wrapper)",
    )
    assert cat == "no-upstream-entrypoint"

    # ALREADY-RESOLVED / UNRUN -> the honest GPU-only blocker.
    assert pap._honest_category("ALREADY-RESOLVED", "")[0] == (
        "UNRUN(needs-gpu)"
    )
    assert pap._honest_category("UNRUN(needs-gpu)", "")[0] == (
        "UNRUN(needs-gpu)"
    )


def test_structure_mismatch_reason_maps_to_oracle_mismatch_not_env():
    """MUST-FIX 2 (codex r3 [confirmed]) -- FAIL ON REVERT.

    L1_01's resolver status string can say
    ``upstream-not-invocable-here`` while its REASON is purely the
    hardened anchor's wrong-operator rejection
    (``STRUCTURE MISMATCH ... ⇒ wrong operator``). Calling a
    structural wrong-operator an "ENVIRONMENT limitation" is dishonest.
    ``_honest_category`` MUST map such a reason to ``oracle-mismatch``
    (the SAME bucket as L1_09), NEVER to
    ``upstream-not-invocable-here(*)``. Reverting the
    ``_is_wrong_operator_reason`` backstop re-mislabels it as the
    prettier env bucket and this FAILS."""
    # The verbatim L1_01 (status, reason) pair from the resolver cache.
    l1_01_status = (
        "BLOCKED(no-verified-upstream-entrypoint: "
        "upstream-not-invocable-here)"
    )
    l1_01_reason = (
        "meta points at kernel.py::apply_logit_bias but it does NOT "
        "pass the CURRENT hardened oracle-consistency anchor "
        "(upstream-not-invocable-here) -- the honest verdict CHANGES; "
        "a stale pre-hardening pass is NOT preserved [anchored on 1/2 "
        "shape(s) ['H2'] :: shape=H2 tol=elementwise_exact_or_inf_safe "
        ":: logits: ✓ within elementwise_exact_or_inf_safe "
        "(max_abs=0.000e+00); STRUCTURE MISMATCH: candidate produced "
        "11 flattened tensor outputs but only 1 matched the oracle's "
        "1; extra unexplained outputs ['expanded_idx_mapping', 'pos', "
        "'num_allowed_token_ids', 'allowed_token_ids'] ⇒ wrong "
        "operator (same outputs PLUS extras must NOT pass the anchor)]"
    )
    cat, sub = pap._honest_category(l1_01_status, l1_01_reason)
    assert cat == "oracle-mismatch", (
        f"a STRUCTURE MISMATCH wrong-operator must be oracle-mismatch "
        f"(L1_09 bucket), NOT the dishonest env label '{cat}'"
    )
    assert cat != "upstream-not-invocable-here", (
        "MUST NEVER map a wrong-operator reason to the prettier "
        "'upstream-not-invocable-here' (env-limitation) bucket"
    )
    assert sub == "", sub

    # Generic wrong-operator markers also reroute.
    assert pap._honest_category(
        "BLOCKED(no-verified-upstream-entrypoint: "
        "upstream-not-invocable-here)",
        "x: NOT within elementwise (max_abs=9.9e+01)",
    )[0] == "oracle-mismatch"

    # Precision: a GENUINE env-import failure (no wrong-op signal) is
    # STILL upstream-not-invocable-here -- the backstop does not
    # over-reroute.
    cat2, sub2 = pap._honest_category(
        "BLOCKED(no-verified-upstream-entrypoint: "
        "upstream-not-invocable-here)",
        "tried ['h']; h: upstream-not-invocable-here "
        "[ModuleNotFoundError: No module named 'zmq']",
    )
    assert cat2 == "upstream-not-invocable-here", cat2
    assert sub2 == "missing-dep:zmq", sub2

    # Precision: a genuine shapes-incomplete (resolver worst-news-first
    # ordering, e.g. L1_30: one candidate oracle-mismatch + another
    # shapes-incomplete -> status is shapes-incomplete) is NOT
    # rerouted; shapes-incomplete is an honest GAP, not a prettier
    # label, and the status string is authoritative there.
    cat3, _ = pap._honest_category(
        "BLOCKED(no-verified-upstream-entrypoint: shapes-incomplete)",
        "act_quant: oracle-mismatch [q: NOT within attention_flash] | "
        "get_valid_kv_indices: shapes-incomplete [missing param ...]",
    )
    assert cat3 == "shapes-incomplete", cat3


def test_resolver_error_is_its_own_category_not_false_no_entrypoint():
    """MUST-FIX 2 (codex r2 [confirmed]) -- FAIL ON REVERT.

    A LIVE cache-miss ``resolve_task()`` that RAISES is reported by
    ``_resolver_verdict_for`` as status
    ``BLOCKED(no-verified-upstream-entrypoint: resolver-error)`` with a
    ``resolve_task raised <Exc>: ...`` reason. ``_honest_category`` MUST
    map this to its OWN distinct ``resolver-error`` category -- a
    resolver CRASH is NOT 'no callable exists'. The OLD code had NO case
    for it and FELL THROUGH to the FALSE ``no-upstream-entrypoint``
    label. Reverting the fix (deleting the resolver-error branch) makes
    this test FAIL because the function would emit the false
    ``no-upstream-entrypoint`` bucket for a resolver crash."""
    # Exact (status, reason) pair _resolver_verdict_for emits on a crash.
    cat, sub = pap._honest_category(
        "BLOCKED(no-verified-upstream-entrypoint: resolver-error)",
        "resolve_task raised RuntimeError: kaboom in the resolver",
    )
    assert cat == "resolver-error", (
        f"a resolver CRASH must be its OWN category, NOT the false "
        f"'{cat}' (collapse-to-misleading-bucket bug recurring in the "
        f"error path)"
    )
    assert cat != "no-upstream-entrypoint", (
        "MUST NEVER fall through to the prettier/false "
        "'no-upstream-entrypoint' label"
    )
    # The subcause carries a short exception summary (distinct, useful).
    assert "RuntimeError" in sub and "kaboom" in sub, sub

    # Also robust when only the reason (not the status token) signals a
    # crash -- still resolver-error, never the false label.
    cat2, _ = pap._honest_category(
        "BLOCKED(something-else)",
        "resolve_task raised ImportError: boom",
    )
    assert cat2 == "resolver-error", cat2

    # Sanity: a GENUINE no-host-wrapper is STILL no-upstream-entrypoint
    # (the fix narrows the false bucket, it does not over-reroute).
    cat3, _ = pap._honest_category(
        "BLOCKED(no-verified-upstream-entrypoint: no-host-wrapper)",
        "no host wrapper launches a @triton.jit kernel (no-host-wrapper)",
    )
    assert cat3 == "no-upstream-entrypoint", cat3


def test_render_status_emits_resolver_error_not_false_no_callable(
    tmp_path,
):
    """MUST-FIX 2 end-to-end -- FAIL ON REVERT. A cache-miss task whose
    LIVE ``resolve_task()`` raises must render as
    ``BLOCKED(resolver-error: ...)`` in status.md, NEVER the false
    ``BLOCKED(no-upstream-entrypoint)``. Reverting the
    ``_honest_category`` fix re-collapses it into the false
    no-upstream-entrypoint bucket and this assertion fails."""
    td = tmp_path / "L_RERR_task"
    td.mkdir()
    (td / "kernel.py").write_text(
        "import triton\n"
        "@triton.jit\n"
        "def _k(X, n):\n    pass\n"
        "def host(x):\n    return {'y': x}\n"
    )
    (td / "meta.yaml").write_text(
        yaml.safe_dump(
            {
                "task_id": "RERR",
                "upstream": {"file_sha256": "S"},
                "oracle": {"gpu_triton": "kernel.py::placeholder"},
            }
        )
    )
    (td / "shapes.yaml").write_text(
        yaml.safe_dump(
            {
                "size_knobs": {"D": 4},
                "public_shapes": [{"id": "P1", "knobs": {"D": 4}}],
                "inputs": {
                    "x": {
                        "dims": ["D"],
                        "dtype": "float32",
                        "dist": "randn",
                    }
                },
            }
        )
    )
    tc = pap.classify_task(td.name, (td / "kernel.py").read_text())
    tc.task_id = "L_RERR"

    # Inject the EXACT (status, reason) pair _resolver_verdict_for emits
    # when the live resolve_task() raises (a resolver crash).
    real_rv = pap._resolver_verdict_for

    def _crash_rv(task_dir):
        return (
            "BLOCKED(no-verified-upstream-entrypoint: resolver-error)",
            "resolve_task raised RuntimeError: deliberate resolver crash",
        )

    pap._resolver_verdict_for = _crash_rv
    try:
        md = pap._render_status_md(
            [tc], gpu=False, eligible_count=1,
            task_dirs_by_id={"L_RERR": td},
        )
    finally:
        pap._resolver_verdict_for = real_rv

    row = next(
        ln for ln in md.splitlines() if ln.startswith("| L_RERR | ")
    )
    assert "BLOCKED(resolver-error:" in row, row
    # The false collapsed bucket must NOT appear for a resolver crash.
    assert "BLOCKED(no-upstream-entrypoint)" not in row, row
    assert "no module-level callable" not in row, row
    # Distinct tally + reconciliation include resolver-error.
    assert "BLOCKED(resolver-error): 1" in md, md
    assert "BLOCKED(no-upstream-entrypoint): 0" in md
    recon = next(
        ln for ln in md.splitlines()
        if ln.startswith("- reconciliation:")
    )
    assert "= 1 " in recon, recon


def test_status_md_states_zero_genuinely_lack_callable_not_collapsed():
    """The regenerated status banner must state the HONEST dominant
    truth: 0 tasks genuinely lack a callable; the dominant blocker is
    the dep-poor dev env (upstream-not-invocable-here), NOT 'no
    entrypoint exists'. FAILS ON REVERT to the old banner that claimed
    '95 placeholder metas ... no such module-level callable exists'.

    MUST-FIX 1 (codex r2 [confirmed]): the numbers are the REAL
    POST-HARDENING sweep (resolver re-verified every already-set
    entrypoint under the CURRENT >=2-shape/structure/dtype anchor; a
    pre-hardening apply no longer preserves a stale pass).

    MUST-FIX 2 (codex r3 [confirmed]): L1_01's anchor reject is a
    STRUCTURE MISMATCH (wrong operator), so it is honestly
    ``oracle-mismatch`` (the SAME bucket as L1_09), NOT the dishonest
    ``upstream-not-invocable-here`` env label. The corrected honest
    breakdown is 78 upstream-not-invocable-here / 2 oracle-mismatch /
    4 UNRUN(needs-gpu) -- reconciling to 101. FAILS ON REVERT to the
    pre-r3 misbucketed 79 / 1."""
    md = (
        REPO / "docs" / "plans" / "phase_a_prime_status.md"
    ).read_text()
    # The misleading collapsed claim must be GONE.
    assert "no such module-level callable exists in" not in md, (
        "the FALSE collapsed claim is back"
    )
    assert "The dominant blocker is 95 placeholder metas" not in md
    assert "The dominant blocker is 96 placeholder metas" not in md
    # The honest categories appear DISTINCTLY in the summary -- the
    # REAL post-hardening numbers (NOT the stale pre-hardening 78/5).
    assert "BLOCKED(no-upstream-entrypoint): 0" in md, (
        "0 tasks genuinely lack a callable -- must say 0"
    )
    assert "BLOCKED(upstream-not-invocable-here): 78" in md, (
        "MUST-FIX 2 (codex r3): L1_01 is a STRUCTURE-MISMATCH wrong "
        "operator -> oracle-mismatch, so upstream-not-invocable-here "
        "is 78 (NOT the misbucketed 79)"
    )
    assert "BLOCKED(upstream-not-invocable-here): 79" not in md, (
        "the misbucketed 79 (L1_01 dishonestly called an env "
        "limitation) must NOT reappear"
    )
    assert "BLOCKED(shapes-incomplete): 17" in md
    assert "BLOCKED(oracle-mismatch): 2" in md, (
        "MUST-FIX 2: oracle-mismatch is L1_09 + L1_01 = 2 (a STRUCTURE "
        "MISMATCH wrong operator is the SAME honest bucket)"
    )
    assert "BLOCKED(resolver-error): 0" in md, (
        "MUST-FIX 2 (r2): the resolver-error bucket is its OWN line"
    )
    assert "UNRUN(needs-gpu): 4" in md, (
        "post-hardening: only 4 tasks survive the genuine hardened "
        "anchor (L1_12/L1_16/L1_26/L2_10), NOT the stale 5"
    )
    assert "UNRUN(needs-gpu): 5" not in md, (
        "the STALE pre-hardening 5 must NOT reappear"
    )
    assert "PASS: 0" in md
    # The dev-env limitation is named honestly, not papered over.
    assert "ENVIRONMENT limitation" in md
    assert "deliberately NOT pip-installed" in md
    # The sub-breakdown of upstream-not-invocable-here is present.
    assert "missing-dep:pybase64 29" in md
    assert "missing-dep:zmq 19" in md
    assert "relative-import-packaging 12" in md
    # Reconciliation line proves the totals add to 101 with the REAL
    # post-hardening numbers.
    assert "(must equal 101 tasks)" in md
    recon = next(
        ln for ln in md.splitlines()
        if ln.startswith("- reconciliation:")
    )
    # MUST-FIX 2 (codex r3): 0+0+78+17+2+0+0+4+0 = 101 (L1_01 is a
    # STRUCTURE-MISMATCH wrong operator -> oracle-mismatch, so the
    # upstream-not-invocable-here term is 78 and oracle-mismatch is 2).
    assert "= 101 " in recon, recon
    assert "+ 78 +" in recon, recon
    assert "+ 79 +" not in recon, (
        "the misbucketed 79 must NOT appear in reconciliation"
    )
    assert recon.count("+ 0 +") >= 1, recon


def test_stale_provenance_cache_is_never_silently_consumed(
    tmp_path, monkeypatch,
):
    """MUST-FIX 1 (codex r3 [confirmed]) -- FAIL ON REVERT.

    Freshness is STRUCTURAL: the cache embeds a provenance
    ``code_fingerprint``. A cache whose stamp does NOT match the
    CURRENT resolver-relevant code (it predates a resolver / anchor /
    runner / honest-category change) MUST NOT be silently consumed --
    status-gen either auto-regenerates under the CURRENT code or
    HARD-FAILS with the explicit ``--refresh-cache`` remedy. With
    auto-regen suppressed (``PA_NO_CACHE_REGEN=1``) a deliberately
    fingerprint-mismatched cache makes ``_load_resolver_verdicts``
    raise ``StaleResolverCacheError`` -- it NEVER returns the stale
    verdicts. Reverting the provenance check (blindly
    ``json.loads`` + use) makes this FAIL because the stale verdicts
    would be returned silently."""
    import importlib.util as _ilu
    import json

    import pytest

    _rs = _ilu.spec_from_file_location(
        "resolve_upstream_entrypoints",
        SCRIPTS / "resolve_upstream_entrypoints.py",
    )
    _rue = _ilu.module_from_spec(_rs)
    sys.modules["resolve_upstream_entrypoints"] = _rue
    _rs.loader.exec_module(_rue)

    # A cache that LOOKS fresh (correct shape) but carries a STALE
    # fingerprint -- exactly a cache built before a resolver/anchor
    # code change. The verdict it carries is deliberately a LIE
    # (oracle-mismatch shown as a passing UNRUN) so that silently
    # consuming it would visibly overclaim.
    stale_cache = tmp_path / "resolve_all.json"
    stale_cache.write_text(
        json.dumps(
            {
                "_provenance": {
                    "code_fingerprint": "stale" * 12,
                    "tool_version": "0.0.0",
                    "generated_at": "1999-01-01T00:00:00+00:00",
                },
                "verdicts": [
                    {
                        "task_id": "L1_01_bias",
                        "status": "UNRUN(needs-gpu)",
                        "reason": "STALE LIE -- overclaimed pass",
                    }
                ],
            }
        )
    )
    # Confirm it IS detected as stale from the artifact itself.
    _v, staleness = _rue.load_resolver_cache(stale_cache)
    assert staleness == "fingerprint-mismatch", staleness

    # Point the status generator at the stale cache; suppress
    # auto-regen so the test is deterministic + fast (no 101-task
    # sweep). The contract: it MUST hard-fail, NEVER return the lie.
    monkeypatch.setattr(pap, "RESOLVE_CACHE", stale_cache)
    monkeypatch.setattr(pap, "_RESOLVER_CACHE_MEM", None)
    monkeypatch.setenv("PA_NO_CACHE_REGEN", "1")

    with pytest.raises(pap.StaleResolverCacheError) as ei:
        pap._load_resolver_verdicts()
    msg = str(ei.value)
    assert "fingerprint-mismatch" in msg, msg
    assert "--refresh-cache" in msg, msg
    # The stale (lying) verdict was NEVER returned.
    assert pap._RESOLVER_CACHE_MEM is None

    # Sanity: a cache written by the REAL writer under the CURRENT
    # code validates fresh and is consumed normally (the check is not
    # over-strict -- a genuinely fresh artifact still works).
    fresh = tmp_path / "fresh.json"

    class _R:
        def __init__(self):
            self.__dict__ = {
                "task_id": "L1_01_bias",
                "status": "BLOCKED(x)",
                "reason": "r",
            }

    _rue.write_resolver_cache(fresh, [_R()])
    monkeypatch.setattr(pap, "RESOLVE_CACHE", fresh)
    monkeypatch.setattr(pap, "_RESOLVER_CACHE_MEM", None)
    got = pap._load_resolver_verdicts()
    assert got["L1_01_bias"]["status"] == "BLOCKED(x)", got
    monkeypatch.setattr(pap, "_RESOLVER_CACHE_MEM", None)


def test_render_status_emits_not_invocable_not_false_no_callable(
    tmp_path,
):
    """End-to-end: a task whose meta says ::placeholder but whose
    kernel.py DOES contain a candidate that merely cannot be invoked in
    this env must be labelled
    ``BLOCKED(upstream-not-invocable-here(<sub>))`` -- NOT the FALSE
    ``BLOCKED(no-upstream-entrypoint): no module-level callable
    exists``. FAILS ON REVERT (the old generator emitted the collapsed
    false bucket here)."""
    td = tmp_path / "L_FAKE_task"
    td.mkdir()
    (td / "kernel.py").write_text(
        "import triton\n"
        "@triton.jit\n"
        "def _k(X, n):\n    pass\n"
        "def host(x):\n    return {'y': x}\n"
    )
    (td / "meta.yaml").write_text(
        yaml.safe_dump(
            {
                "task_id": "FAKE",
                "upstream": {"file_sha256": "S"},
                "oracle": {"gpu_triton": "kernel.py::placeholder"},
            }
        )
    )
    (td / "shapes.yaml").write_text(
        yaml.safe_dump(
            {
                "size_knobs": {"D": 4},
                "public_shapes": [{"id": "P1", "knobs": {"D": 4}}],
                "inputs": {
                    "x": {
                        "dims": ["D"],
                        "dtype": "float32",
                        "dist": "randn",
                    }
                },
            }
        )
    )
    tc = pap.classify_task(td.name, (td / "kernel.py").read_text())
    tc.task_id = "L_FAKE"

    # Inject the resolver verdict so the test is deterministic and does
    # NOT depend on a cache: a candidate WAS found but cannot be
    # invoked here (missing dep). This is the exact ~78-task truth.
    real_rv = pap._resolver_verdict_for

    def _fake_rv(task_dir):
        return (
            "BLOCKED(no-verified-upstream-entrypoint: "
            "upstream-not-invocable-here)",
            "tried ['host']; host: upstream-not-invocable-here "
            "[ModuleNotFoundError: No module named 'cbor2']",
        )

    pap._resolver_verdict_for = _fake_rv
    try:
        md = pap._render_status_md(
            [tc], gpu=False, eligible_count=1,
            task_dirs_by_id={"L_FAKE": td},
        )
    finally:
        pap._resolver_verdict_for = real_rv

    row = next(
        ln for ln in md.splitlines() if ln.startswith("| L_FAKE | ")
    )
    assert "BLOCKED(upstream-not-invocable-here(missing-dep:cbor2))" in (
        row
    ), row
    # The FALSE collapsed claim must NOT appear for this task.
    assert "no module-level callable" not in row, row
    assert "BLOCKED(no-upstream-entrypoint)" not in row, row
    # It is honestly attributed as an ENVIRONMENT limitation.
    assert "ENVIRONMENT limitation" in row, row
    assert "BLOCKED(upstream-not-invocable-here): 1" in md
    assert "BLOCKED(no-upstream-entrypoint): 0" in md


# ─────────────────────────────────────────────────────────────────────
# FIX 3: hardened oracle-consistency anchor
# ─────────────────────────────────────────────────────────────────────
def _mk_anchor_task(
    tmp_path: Path,
    kernel_src: str,
    oracle_src: str,
    n_shapes: int = 2,
) -> Path:
    """A minimal task tree the oracle runner can drive: kernel.py +
    oracle_pytorch.py + meta.yaml + shapes.yaml with ``n_shapes``
    distinct shape rows (so the >=2-shape anchor is exercised)."""
    td = tmp_path / "L_anchor"
    td.mkdir()
    (td / "kernel.py").write_text(kernel_src)
    (td / "oracle_pytorch.py").write_text(oracle_src)
    (td / "meta.yaml").write_text(
        yaml.safe_dump(
            {
                "task_id": "ANCHOR",
                "tolerance_class": "elementwise",
                "oracle": {"gpu_triton": "kernel.py::cand"},
            }
        )
    )
    rows = [
        {"id": f"S{i}", "knobs": {"D": 4 * (i + 1)}}
        for i in range(n_shapes)
    ]
    (td / "shapes.yaml").write_text(
        yaml.safe_dump(
            {
                "size_knobs": {"D": 4},
                "public_shapes": rows,
                "inputs": {
                    "x": {
                        "dims": ["D"],
                        "dtype": "float32",
                        "dist": "randn",
                    }
                },
            }
        )
    )
    return td


def _run_anchor(td: Path, entrypoint: str = "cand") -> tuple[bool, str]:
    work = Path(tempfile.mkdtemp(prefix="pa_anchor_test_"))
    spec = {
        "task_dir": str(td),
        "entrypoint": entrypoint,
        "out_json": str(work / "out.json"),
        "vllm_root": str(REPO.parent / "infer" / "vllm"),
        "sglang_root": str(REPO.parent / "infer" / "sglang" / "python"),
        "repo_root": str(REPO),
    }
    (work / "spec.json").write_text(json.dumps(spec))
    subprocess.run(
        [sys.executable, str(_ORACLE_RUNNER), "--spec",
         str(work / "spec.json")],
        capture_output=True, text=True,
    )
    res = json.loads((work / "out.json").read_text())
    return bool(res.get("consistent")), str(res.get("detail", ""))


def test_anchor_checks_at_least_two_shapes():
    """FIX 3(a): ``_anchor_shape_ids`` must return >=2 distinct shape
    ids when the task enumerates more than one (small + medium) so a
    one-shape coincidence cannot pass. FAILS ON REVERT to the old
    single-smallest-shape ``_smallest_shape_id``."""
    doc = {
        "public_shapes": [
            {"id": "small", "knobs": {"D": 4}},
            {"id": "mid", "knobs": {"D": 16}},
            {"id": "big", "knobs": {"D": 64}},
        ]
    }
    ids = ocr._anchor_shape_ids(doc)
    assert len(ids) >= 2, ids
    assert ids[0] == "small", ids  # smallest first
    assert len(set(ids)) == len(ids), "shapes must be DISTINCT"
    # Single-shape task: honestly check the one (no fake second shape).
    assert ocr._anchor_shape_ids(
        {"public_shapes": [{"id": "only", "knobs": {"D": 4}}]}
    ) == ["only"]


def test_anchor_rejects_extra_candidate_outputs(tmp_path):
    """FIX 3(b): a candidate that returns the oracle's output PLUS an
    extra unexplained tensor is the WRONG operator and must be
    REJECTED. FAILS ON REVERT to the old anchor that matched every
    oracle tensor but ignored extra candidate outputs."""
    td = _mk_anchor_task(
        tmp_path,
        kernel_src=(
            "import torch\n"
            "def cand(x):\n"
            "    # correct y PLUS an extra unexplained output\n"
            "    return {'y': x + 1.0, 'EXTRA': x * 999.0}\n"
        ),
        oracle_src=(
            "import torch\n"
            "def run(inp):\n"
            "    return {'y': inp['x'] + 1.0}\n"
        ),
    )
    ok, detail = _run_anchor(td)
    assert ok is False, detail
    assert "STRUCTURE MISMATCH" in detail, detail


def test_anchor_rejects_dtype_mismatch(tmp_path):
    """FIX 3(c): a candidate whose output value is near the oracle's but
    whose DTYPE differs is a wrong-operator signal and must be
    REJECTED. FAILS ON REVERT to the dtype-lax same-key match."""
    td = _mk_anchor_task(
        tmp_path,
        kernel_src=(
            "import torch\n"
            "def cand(x):\n"
            "    # right values, WRONG dtype (float64 vs oracle float32)\n"
            "    return {'y': (x + 1.0).double()}\n"
        ),
        oracle_src=(
            "import torch\n"
            "def run(inp):\n"
            "    return {'y': inp['x'] + 1.0}\n"
        ),
    )
    ok, detail = _run_anchor(td)
    assert ok is False, detail
    assert "dtype" in detail.lower(), detail


def test_anchor_rejects_single_shape_coincidence(tmp_path):
    """FIX 3(a) end-to-end: a wrong candidate that coincides with the
    oracle ONLY on the smallest shape (D==4) but diverges on the medium
    shape must be REJECTED -- because the anchor now checks >=2 shapes.
    FAILS ON REVERT to single-shape anchoring (which this candidate
    would have fooled)."""
    td = _mk_anchor_task(
        tmp_path,
        kernel_src=(
            "import torch\n"
            "def cand(x):\n"
            "    # coincides with oracle iff numel == 4 (smallest "
            "shape); wrong otherwise.\n"
            "    if x.numel() == 4:\n"
            "        return {'y': x + 1.0}\n"
            "    return {'y': x * -123.0}\n"
        ),
        oracle_src=(
            "import torch\n"
            "def run(inp):\n"
            "    return {'y': inp['x'] + 1.0}\n"
        ),
    )
    ok, detail = _run_anchor(td)
    assert ok is False, detail
    # The smallest shape passed; the medium shape disproved identity.
    assert "S0" in detail or "NOT within" in detail, detail


def test_anchor_passes_genuinely_correct_operator(tmp_path):
    """Sanity / no-overclaim: the hardened anchor still PASSES a
    genuinely correct operator on >=2 shapes with matching structure +
    dtype (it does not over-reject). It is strong, not impossible to
    satisfy."""
    td = _mk_anchor_task(
        tmp_path,
        kernel_src=(
            "import torch\n"
            "def cand(x):\n    return {'y': x + 1.0}\n"
        ),
        oracle_src=(
            "import torch\n"
            "def run(inp):\n    return {'y': inp['x'] + 1.0}\n"
        ),
    )
    ok, detail = _run_anchor(td)
    assert ok is True, detail
    assert "anchored on 2/2 shape(s)" in detail, detail


def test_anchor_strength_is_not_overclaimed_as_proof():
    """No-overclaim applies to the verifier itself (codex finding 6).
    The resolver/runner docstrings must NOT claim a formal
    impossibility / 'PROVES it is the right operator', and MUST carry
    the honest 'not a formal impossibility proof' caveat. FAILS ON
    REVERT to the old 'PROVES it is the right operator' wording."""
    rue_src = (SCRIPTS / "resolve_upstream_entrypoints.py").read_text()
    ocr_src = _ORACLE_RUNNER.read_text()
    assert "PROVES it is the right operator with TWO" not in rue_src
    assert "not a formal impossibility proof" in rue_src.lower()
    assert "could in principle" in rue_src
    assert 'not overclaim it is "impossible"' in rue_src
    assert "not a formal impossibility proof" in ocr_src.lower()
    assert "could in principle still pass" in ocr_src
    # the runner docstring carries the honest no-overclaim caveat
    # (normalise whitespace -- the phrase wraps across lines).
    assert "do not\nnot overclaim" not in ocr_src
    assert "we do\nnot overclaim" in ocr_src or (
        "we do not overclaim" in " ".join(ocr_src.split())
    )


# ── narrowing-float interpreter-skew rescue marker ─────────────────────
# Under TRITON_INTERPRET=1 the interpreter mis-rounds every narrowing float
# store -- fp32->fp8 in both directions (127.2 -> 64.0 instead of 128.0),
# fp32->bf16/fp16 truncated toward zero instead of RNE (100.3 -> 100.0
# instead of 100.5). The anchor flags any such output as
# ``narrow_float_output`` so the caller re-runs on the real device (which
# rounds correctly) and adopts that verdict. The marker keys on output dtype
# alone; the GPU re-run re-applies the full dtype/structure/tolerance check,
# so a wrong operator is still caught there.


def _run_anchor_full(td: Path, entrypoint: str = "cand") -> dict:
    """Like _run_anchor but returns the full result dict (so tests can
    assert on ``narrow_float_output`` / early-return arity safety)."""
    work = Path(tempfile.mkdtemp(prefix="pa_anchor_test_"))
    spec = {
        "task_dir": str(td),
        "entrypoint": entrypoint,
        "out_json": str(work / "out.json"),
        "vllm_root": str(REPO.parent / "infer" / "vllm"),
        "sglang_root": str(REPO.parent / "infer" / "sglang" / "python"),
        "repo_root": str(REPO),
    }
    (work / "spec.json").write_text(json.dumps(spec))
    subprocess.run(
        [sys.executable, str(_ORACLE_RUNNER), "--spec",
         str(work / "spec.json")],
        capture_output=True, text=True,
    )
    return json.loads((work / "out.json").read_text())


def test_fp8_interpreter_skew_sets_rescue_marker():
    """The real L1_30_act_quant task: under TRITON_INTERPRET=1 the
    interpreter's fp32->float8 store mis-rounds, so the CORRECT oracle
    'mismatches' the kernel on ``q`` (float8) while ``scale`` (fp32)
    matches bit-exact. The anchor must set ``narrow_float_output`` so
    the caller can re-run on GPU. FAILS ON REVERT (the float8 skew would
    be reported as a plain oracle-mismatch with no rescue)."""
    td = REPO / "tasks" / "level1" / "L1_30_act_quant"
    if not (td / "kernel.py").is_file():
        import pytest

        pytest.skip("L1_30_act_quant task not present")
    res = _run_anchor_full(td, entrypoint="act_quant")
    assert res.get("consistent") is False, res.get("detail")
    assert res.get("narrow_float_output") is True, res.get("detail")
    # The float8 output (q) is the one that misses; the fp32 scale matches.
    assert "q:" in res.get("detail", "")
    assert "✓ within" in res.get("detail", "")  # scale matched bit-exact


def test_bf16_output_sets_rescue_marker(tmp_path):
    """A bf16 (narrowing-float) output sets ``narrow_float_output`` even when
    the anchor PASSES -- the interpreter's truncated bf16 store is unreliable
    both ways, so the caller must GPU-confirm. FAILS ON REVERT to an fp8-only
    marker that left bf16 anchor verdicts untrustworthy."""
    td = _mk_anchor_task(
        tmp_path,
        kernel_src=(
            "import torch\n"
            "def cand(x):\n    return {'y': (x * 1.0).to(torch.bfloat16)}\n"
        ),
        oracle_src=(
            "import torch\n"
            "def run(inp):\n"
            "    return {'y': (inp['x'] * 1.0).to(torch.bfloat16)}\n"
        ),
    )
    res = _run_anchor_full(td)
    assert res.get("narrow_float_output") is True, res.get("detail")


def test_non_narrow_mismatch_does_not_set_rescue_marker(tmp_path):
    """A numeric mismatch on a NON-narrow (float32) output must NOT set
    ``narrow_float_output`` -- the interpreter rounds fp32 correctly, so a
    wrong fp32 operator gets no GPU rescue. FAILS ON REVERT to a marker that
    ignores output dtype."""
    td = _mk_anchor_task(
        tmp_path,
        kernel_src=(
            "import torch\n"
            "def cand(x):\n    return {'y': x * -123.0}\n"  # wrong fp32
        ),
        oracle_src=(
            "import torch\n"
            "def run(inp):\n    return {'y': inp['x'] + 1.0}\n"
        ),
    )
    res = _run_anchor_full(td)
    assert res.get("consistent") is False, res.get("detail")
    assert not res.get("narrow_float_output"), res.get("detail")


def test_anchor_early_returns_are_arity_safe(tmp_path):
    """Regression: _check_one_shape returns a 4-tuple on EVERY path. A
    non-callable entrypoint takes an early-return branch; main() must
    unpack it without a ValueError and emit the honest 'not callable'
    detail (no ``narrow_float_output``). FAILS ON REVERT to a short
    early-return tuple that crashed main()'s 4-way unpack."""
    td = _mk_anchor_task(
        tmp_path,
        kernel_src="x = 1\n",  # no module-level callable named 'cand'
        oracle_src=(
            "import torch\n"
            "def run(inp):\n    return {'y': inp['x'] + 1.0}\n"
        ),
    )
    res = _run_anchor_full(td)
    assert res.get("consistent") is False, res.get("detail")
    assert not res.get("narrow_float_output"), res.get("detail")
    assert "ValueError" not in res.get("detail", ""), res.get("detail")
    assert "not a module-level callable" in res.get("detail", "")
