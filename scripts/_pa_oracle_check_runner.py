#!/usr/bin/env python3
"""_pa_oracle_check_runner.py -- the Phase A' oracle-consistency anchor.

Runs in a FRESH subprocess with ``infer/vllm`` + ``infer/sglang/python``
on ``sys.path`` (so the vendored upstream kernel.py's vllm/sglang
imports resolve to the FROZEN upstream source -- exactly like
``_pa_subproc_runner.py --side original``) under
``TRITON_INTERPRET=1`` (CPU, GPU-free).

It (1) imports the task's UNMODIFIED ``kernel.py``, (2) invokes the
candidate ``entrypoint`` via the SAME strict by-name signature adapter
the S2 runner uses (``_pa_subproc_runner._map_inputs_by_name`` -- never
a silent skip+positional-shift), (3) imports the task's
``oracle_pytorch.py`` and calls ``run(inputs)`` on the SAME inputs, and
(4) compares every output tensor within the task's ``tolerance_class``
(``triton_ascend_bench.tolerance.compare``).

FIX 3 (codex finding 6, [adopted]) hardenings:
  (a) the anchor checks **>=2 distinct shapes** (small + median, when
      the task enumerates more than one) -- not only the smallest --
      so a wrong candidate coinciding on one tiny shape is far less
      likely to survive;
  (b) the candidate output STRUCTURE must match the oracle's: every
      candidate flattened tensor output must be consumed by an oracle
      output (EXTRA unexplained candidate outputs ⇒ reject as a wrong
      operator -- a wrong op returning oracle-outputs-PLUS-extras no
      longer passes);
  (c) each matched twin's dtype must EQUAL the oracle's (a
      different-dtype near-match is a wrong-operator signal -- now
      enforced on the same-key match too, not just the fallback).

If the resolved entrypoint's output does NOT match the oracle in value,
structure or dtype on the sampled shapes it is the WRONG operator ->
``consistent: false`` and the caller refuses to write it to meta.yaml
(the anti-fake rule). This is a STRONG empirical anchor, NOT a formal
impossibility proof: a wrong candidate coinciding on the sampled shapes
with matching structure/dtype could in principle still pass -- we do
not overclaim. This script only tests (or disproves) operator
identity; it never edits any file.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import types
from pathlib import Path

THIS = Path(__file__).resolve()
SCRIPTS_DIR = THIS.parent


def _vol(r: dict) -> int:
    k = r.get("knobs", {})
    v = 1
    for x in k.values():
        try:
            v *= max(int(x), 1)
        except (TypeError, ValueError):
            pass
    return v


def _anchor_shape_ids(shapes_doc: dict) -> list[str]:
    """FIX 3(a) (codex finding 6, adopted): the anchor must NOT check
    only the smallest shape -- a wrong candidate can coincide with the
    oracle on one tiny shape. Check at least TWO distinct shapes (small
    + medium) when the task enumerates more than one; a coincidence on
    two independent shapes is far less likely than on one. If only one
    shape exists we honestly check that one (and the softened wording
    documents this is not a formal impossibility proof)."""
    rows = shapes_doc.get("public_shapes", []) + shapes_doc.get(
        "hidden_shapes", []
    )
    rows = [r for r in rows if r.get("id")]
    if not rows:
        return []
    ordered = sorted(rows, key=_vol)
    ids: list[str] = [ordered[0]["id"]]
    if len(ordered) >= 2:
        # medium = the median-volume shape, distinct from the smallest.
        med = ordered[len(ordered) // 2]
        if med["id"] != ids[0]:
            ids.append(med["id"])
        else:
            for r in ordered[1:]:
                if r["id"] != ids[0]:
                    ids.append(r["id"])
                    break
    return ids


def _load(path: Path, name: str):
    # Some kernel.py modules (e.g. those vendored from sglang.srt.* or
    # vllm.v1.*) use sibling-relative imports like ``from .X import Y``.
    # The bare ``spec_from_file_location(name, path)`` form leaves
    # ``__package__`` empty, so any relative import inside kernel.py
    # raises ``ImportError: attempted relative import with no known
    # parent package``. Wrap the load in a synthetic per-call package
    # whose ``__path__`` points at the task dir, so relative imports
    # resolve against the sibling files under that dir; install both the
    # synthetic package and the leaf module into ``sys.modules`` before
    # ``exec_module`` so the resolver itself can find them mid-load.
    pkg = f"_pa_pkg_{name}"
    if pkg not in sys.modules:
        synth = types.ModuleType(pkg)
        synth.__path__ = [str(path.parent)]
        sys.modules[pkg] = synth
    leaf_name = f"{pkg}.{path.stem}"
    spec = importlib.util.spec_from_file_location(leaf_name, path)
    mod = importlib.util.module_from_spec(spec)
    mod.__package__ = pkg
    sys.modules[leaf_name] = mod
    spec.loader.exec_module(mod)
    return mod


def _flatten(obj, prefix=""):
    """Flatten oracle/candidate output dicts into {path: tensor} so a
    structural difference does not crash the comparison."""
    import torch

    out = {}
    if torch.is_tensor(obj):
        out[prefix or "_root"] = obj
    elif isinstance(obj, dict):
        for k, v in obj.items():
            out.update(_flatten(v, f"{prefix}.{k}" if prefix else str(k)))
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            out.update(_flatten(v, f"{prefix}[{i}]"))
    return out


def _check_one_shape(
    task_dir, entrypoint, sid, shapes_doc, tcls, tol_name,
    compare, build_inputs, _map_inputs_by_name,
) -> tuple[bool, str]:
    """Run the anchor on ONE shape. Returns (ok, detail).

    FIX 3(b)+(c) (codex finding 6, adopted): in addition to requiring
    every oracle tensor to find a numeric twin, the candidate output
    STRUCTURE must match the oracle's (same arity of flattened named
    tensor outputs -- EXTRA unexplained candidate outputs ⇒ reject as a
    wrong operator; the old anchor matched every oracle tensor but
    ignored extra candidate outputs, so a wrong op returning
    oracle-outputs-PLUS-extras passed) AND each matched twin's dtype
    must equal the oracle's (a different-dtype near-match is a
    wrong-operator signal; the old anchor only enforced dtype on the
    fallback positional pairing, not the same-key match)."""
    import torch

    inputs = build_inputs(shapes_doc, sid, seed=0)
    kmod = _load(task_dir / "kernel.py", f"_pa_oc_kernel_{sid}")
    fn = getattr(kmod, entrypoint, None)
    if not callable(fn):
        return False, f"entrypoint {entrypoint!r} not a module-level callable"
    try:
        sig, bound, kwargs = _map_inputs_by_name(
            fn, entrypoint, dict(inputs)
        )
    except Exception as e:  # ShapesIncompleteError or introspection
        return False, f"by-name bind failed: {type(e).__name__}: {e}"
    cand_ret = fn(dict(inputs)) if sig is None else fn(**kwargs)
    if cand_ret is None:
        cand_ret = {
            p: inputs[p]
            for p in bound
            if torch.is_tensor(inputs.get(p))
        }
    omod = _load(task_dir / "oracle_pytorch.py", f"_pa_oc_oracle_{sid}")
    oracle_ret = omod.run(dict(inputs))

    cand_t = _flatten(cand_ret)
    orac_t = _flatten(oracle_ret)
    if not orac_t:
        return False, "oracle produced no tensors to compare"

    details: list[str] = []
    all_ok = True
    used: set[int] = set()
    cand_items = list(cand_t.items())
    for ok_, ov in orac_t.items():
        twin = None
        # Same-key match -- FIX 3(c): now ALSO require dtype to match
        # (the old code accepted a same-key twin on shape alone).
        if (
            ok_ in cand_t
            and cand_t[ok_].shape == ov.shape
            and cand_t[ok_].dtype == ov.dtype
        ):
            twin = cand_t[ok_]
            used.add(cand_items.index((ok_, cand_t[ok_])))
        else:
            for i, (_ck, cv) in enumerate(cand_items):
                if (
                    i not in used
                    and cv.shape == ov.shape
                    and cv.dtype == ov.dtype
                ):
                    twin = cv
                    used.add(i)
                    break
        if twin is None:
            all_ok = False
            details.append(
                f"{ok_}: no candidate output with shape "
                f"{tuple(ov.shape)} dtype {ov.dtype} (dtype mismatch is "
                f"a wrong-operator signal)"
            )
            continue
        cmp = compare(twin, ov, tcls)
        if not cmp.passed:
            all_ok = False
            details.append(
                f"{ok_}: NOT within {tol_name} "
                f"(max_abs={cmp.max_abs_diff:.3e} "
                f"max_rel={cmp.max_rel_diff:.3e}: {cmp.reason})"
            )
        else:
            details.append(
                f"{ok_}: ✓ within {tol_name} "
                f"(max_abs={cmp.max_abs_diff:.3e})"
            )
    # FIX 3(b): structure / arity must match -- EXTRA candidate tensor
    # outputs not consumed by any oracle output mean the candidate is
    # producing more than the oracle ⇒ it is NOT the same operator.
    if len(used) != len(cand_items):
        all_ok = False
        extras = [
            ck for i, (ck, _cv) in enumerate(cand_items) if i not in used
        ]
        details.append(
            f"STRUCTURE MISMATCH: candidate produced "
            f"{len(cand_items)} flattened tensor outputs but only "
            f"{len(used)} matched the oracle's {len(orac_t)}; extra "
            f"unexplained outputs {extras[:4]} ⇒ wrong operator "
            f"(same outputs PLUS extras must NOT pass the anchor)"
        )
    return all_ok, (
        f"shape={sid} tol={tol_name} :: " + "; ".join(details[:6])
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True)
    args = ap.parse_args()
    spec = json.loads(Path(args.spec).read_text())

    os.environ["TRITON_INTERPRET"] = "1"
    task_dir = Path(spec["task_dir"])
    out_json = Path(spec["out_json"])
    entrypoint = spec["entrypoint"]

    # infer/ on path FIRST (frozen upstream resolution), then repo root
    # for triton_ascend_bench.* + scripts for the shared adapter.
    for p in (spec["vllm_root"], spec["sglang_root"], spec["repo_root"],
              str(SCRIPTS_DIR)):
        if p not in sys.path:
            sys.path.insert(0, p)

    result = {"consistent": False, "detail": ""}
    try:
        import yaml

        from _pa_subproc_runner import _map_inputs_by_name
        from triton_ascend_bench.inputs import build_inputs
        from triton_ascend_bench.tolerance import compare, load_policy

        shapes_doc = yaml.safe_load(
            (task_dir / "shapes.yaml").read_text()
        )
        meta = yaml.safe_load((task_dir / "meta.yaml").read_text())
        tol_name = str(meta.get("tolerance_class", "elementwise"))
        policy = load_policy()
        tcls = policy.get(tol_name) or policy.get("elementwise")

        # FIX 3(a): anchor on >=2 distinct shapes (small + medium) when
        # the task has them -- a wrong candidate that coincides with the
        # oracle on ONE tiny shape is much less likely to also coincide
        # on a second, independent shape. Honest if only one shape
        # exists (the softened wording says this is not a formal proof).
        sids = _anchor_shape_ids(shapes_doc)
        if not sids:
            result["detail"] = "shapes.yaml has no shape rows"
            out_json.write_text(json.dumps(result))
            return 0

        all_ok = True
        per_shape: list[str] = []
        for sid in sids:
            ok, det = _check_one_shape(
                task_dir, entrypoint, sid, shapes_doc, tcls, tol_name,
                compare, build_inputs, _map_inputs_by_name,
            )
            per_shape.append(det)
            if not ok:
                all_ok = False
                break  # one shape disproves operator identity -> stop
        result["consistent"] = all_ok
        result["detail"] = (
            f"anchored on {len(per_shape)}/{len(sids)} shape(s) "
            f"{sids[:len(per_shape)]} :: " + " || ".join(per_shape)
        )
    except Exception as e:
        import traceback

        result["detail"] = (
            f"{type(e).__name__}: {e} | {traceback.format_exc()[-500:]}"
        )

    # codex r6 [confirmed] -- capture this task's ACTUAL imported
    # upstream closure: every loaded module whose ``__file__`` is under
    # an infer/ root the verdict path just executed. Done in a ``finally``
    # sense (after the check ran, success OR exception) so even a failing
    # candidate that DID import upstream bytes records the closure that
    # produced its verdict. The parent (``oracle_consistency_check``)
    # hashes these paths into the per-task ``upstream_fp``; capturing the
    # PATHS here (in the subprocess that actually imported them) is the
    # only place the real closure is observable.
    roots = [
        str(Path(spec["vllm_root"]).resolve()),
        str(Path(spec["sglang_root"]).resolve()),
    ]
    closure: list[str] = []
    seen: set[str] = set()
    for _name, _mod in list(sys.modules.items()):
        _f = getattr(_mod, "__file__", None)
        if not _f:
            continue
        try:
            _af = str(Path(_f).resolve())
        except OSError:
            continue
        for _r in roots:
            if _af == _r or _af.startswith(_r + "/"):
                if _af not in seen:
                    seen.add(_af)
                    closure.append(_af)
                break
    result["upstream_closure"] = sorted(closure)

    out_json.write_text(json.dumps(result))
    return 0


if __name__ == "__main__":
    sys.exit(main())
