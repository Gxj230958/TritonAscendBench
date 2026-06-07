"""Guardrail: corrected-variant proof artifacts are NEVER scored.

The dual-metric policy (SCORE.md) keeps the scored 100-task headline byte-exact
frozen upstream. The fixed-kernel "what-if" lives in non-scored
``tasks/_corrected/`` variants. This test pins the invariant so a back-door
force-fix (promoting a corrected variant into the scored path) fails CI:

1. task discovery (``phase_a_prime._iter_task_dirs``) never returns a
   ``_corrected/`` (or any ``_``-prefixed) task; and
2. every ``tasks/_corrected/<v>/meta.yaml`` carries a ``corrected_variant``
   manifest with ``score_eligible: false`` and ``variant_of`` pointing at a real
   scored task.
"""
from __future__ import annotations

import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))

import phase_a_prime as pap  # noqa: E402


def test_discovery_excludes_corrected_and_underscore_dirs():
    dirs = pap._iter_task_dirs()
    names = [d.name for d in dirs]
    # no _-prefixed dir is ever discovered as a scored task
    assert not any(n.startswith("_") for n in names), names
    # specifically, no corrected variant leaks into the scored set
    assert not any("_corrected" in str(d) for d in dirs), [str(d) for d in dirs]
    # every discovered task lives under level1/2/3
    assert all(d.parent.name in ("level1", "level2", "level3") for d in dirs)


def test_corrected_variants_are_marked_nonscored():
    cdir = REPO / "tasks" / "_corrected"
    if not cdir.is_dir():
        return  # no corrected variants yet -> nothing to enforce
    variants = [d for d in cdir.iterdir() if d.is_dir()]
    scored = {d.name for d in pap._iter_task_dirs()}
    # A corrected variant's original may itself have been excluded from the
    # scored headline (a _deprecated upstream-bug task, e.g. the L2_23/L2_24
    # lightning-attn off-by-one false positives). variant_of must still name a
    # REAL frozen original -- scored OR deprecated -- never a dangling id.
    ddir = REPO / "tasks" / "_deprecated"
    deprecated = {d.name for d in ddir.iterdir() if d.is_dir()} if ddir.is_dir() else set()
    real_originals = scored | deprecated
    for v in variants:
        meta = yaml.safe_load((v / "meta.yaml").read_text())
        cv = meta.get("corrected_variant")
        assert cv is not None, f"{v.name}: missing corrected_variant manifest"
        assert cv.get("score_eligible") is False, f"{v.name}: must be score_eligible:false"
        assert cv.get("release_scope", {}).get("excluded") is True, f"{v.name}: release excluded"
        # variant_of must name a real frozen original (scored or deprecated)
        assert cv.get("variant_of") in real_originals, (
            f"{v.name}: variant_of {cv.get('variant_of')!r} is neither a scored "
            "nor a deprecated task"
        )
