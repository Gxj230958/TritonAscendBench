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
import re
import sys
import types
from pathlib import Path

THIS = Path(__file__).resolve()
SCRIPTS_DIR = THIS.parent

# Header line that the freeze process stamps into every vendored kernel.py
# docstring. Example: ``Source path: `vllm/model_executor/layers/...py```.
_SRC_PATH_RE = re.compile(
    r"Source\s+path\s*:\s*`?([^`\n]+?)`?\s*$",
    re.IGNORECASE | re.MULTILINE,
)
# Match ``from .X import Y`` / ``from ..X import Y`` / ``from . import Y``
# in a kernel.py top-level region. ``\1`` captures the leading dots so the
# loader can know the maximum relative-import depth required.
_REL_IMPORT_RE = re.compile(
    r"^\s*from\s+(\.+)(\w*(?:\.\w+)*)\s+import\s+",
    re.MULTILINE,
)


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


def _extract_source_path(kernel_src: str) -> str | None:
    """Return the upstream ``Source path: <p>`` declared at the top of a
    vendored kernel.py (or ``None`` if the header is missing). Every file
    frozen by ``scripts/phase_a_freeze.py`` stamps this line into the
    module docstring, so it is the authoritative way to identify which
    upstream file the vendored copy was taken from."""
    head = kernel_src[:4000]  # the freeze-stamped header is always near the top
    m = _SRC_PATH_RE.search(head)
    return m.group(1).strip() if m else None


def _max_relative_depth(kernel_src: str) -> int:
    """Return the maximum number of leading dots in any ``from .X import``
    statement at module scope. ``from .X`` -> 1, ``from ..X`` -> 2, etc.
    Returns 0 when the file has no relative imports (the legacy path)."""
    depth = 0
    for m in _REL_IMPORT_RE.finditer(kernel_src):
        depth = max(depth, len(m.group(1)))
    return depth


def _resolve_upstream_sibling(
    source_path: str, upstream_roots: list[Path],
) -> Path | None:
    """Locate the upstream file's PARENT directory under one of the
    configured upstream roots. ``source_path`` is the relpath the
    freeze-stamp recorded (e.g. ``vllm/.../layernorm_guard.py``);
    ``upstream_roots`` is ``[infer/vllm, infer/sglang/python]`` (probed
    in order). Returns the sibling directory on the first hit so a
    ``from .utils`` inside the vendored kernel.py resolves to
    ``<sibling>/utils.py`` -- the same helper file the upstream copy
    sits next to. ``None`` when the file is missing under every root."""
    for root in upstream_roots:
        cand = root / source_path
        if cand.is_file():
            return cand.parent
    return None


def _build_synthetic_pkg_chain(
    leaf_module_name: str,
    task_dir: Path,
    upstream_sibling: Path,
    depth: int,
) -> str:
    """Install a synthetic package chain into ``sys.modules`` deep enough
    to satisfy a kernel.py that uses ``from .X`` (depth==1) or
    ``from ..X`` (depth==2) imports, and return the leaf-package's
    fully qualified name so the caller can set the kernel module's
    ``__package__`` to it.

    The chain mirrors the upstream directory tree, NOT the dotted upstream
    name -- two reasons:
      * the bench MUST NOT register a top-level ``vllm`` / ``sglang``
        package (the severance contract); using a private
        ``_pa_pkg_<name>`` prefix avoids polluting any real upstream
        package namespace, and
      * the only fact a relative import resolves against is the parent
        package's ``__path__`` (which directory list to scan), so wiring
        ``__path__ = [<upstream sibling at the right depth>]`` is
        sufficient regardless of what the synthetic package is *named*.

    For ``depth==1`` the leaf package's ``__path__`` includes BOTH the
    task dir and the upstream sibling dir: ``from .X`` first tries the
    task dir (so a task that ships its own helper wins) then falls back
    to the upstream sibling. For ``depth>=2`` we walk ``depth-1`` levels
    UP the upstream tree to build the grandparent chain; each ancestor
    package's ``__path__`` points at the matching ancestor of the
    upstream sibling so ``from ..Y`` finds ``<grandparent>/Y.py``."""
    base = f"_pa_pkg_{leaf_module_name}"
    # Walk UP `depth-1` steps from the immediate sibling so the chain
    # has enough levels for the deepest ``from ..[..]X`` relative.
    levels: list[Path] = [upstream_sibling]
    cur = upstream_sibling
    for _ in range(max(depth - 1, 0)):
        cur = cur.parent
        levels.append(cur)
    # ``levels[0]`` is the leaf-adjacent dir; ``levels[-1]`` is the root
    # of the synthetic chain. Build the chain root -> leaf so each child
    # appears as ``<parent>.<child>`` in ``sys.modules``.
    parts: list[str] = []
    for i, lvl in enumerate(reversed(levels)):
        parts.append(f"_{i}")
        full = f"{base}." + ".".join(parts)
        if full not in sys.modules:
            pkg_mod = types.ModuleType(full)
            # Each ancestor's __path__ points at the matching ancestor of
            # the upstream sibling -- THIS is what makes relative imports
            # resolve to the upstream tree.
            pkg_mod.__path__ = [str(lvl)]
            sys.modules[full] = pkg_mod
        else:
            # Refresh __path__ so a stale entry from a previous task does
            # not point at the wrong upstream sibling.
            sys.modules[full].__path__ = [str(lvl)]
    # The LEAF package (the one the kernel module belongs to) gets the
    # task dir prepended so a task-local helper, if any, wins over the
    # upstream sibling.
    leaf_pkg = f"{base}." + ".".join(parts)
    sys.modules[leaf_pkg].__path__ = [str(task_dir), str(upstream_sibling)]
    # The top-level synthetic name itself is not a package the kernel
    # references; register a placeholder so ``importlib`` does not get
    # confused about half-built parents.
    if base not in sys.modules:
        top = types.ModuleType(base)
        top.__path__ = []  # opaque -- only the children are addressable
        sys.modules[base] = top
    return leaf_pkg


def _load(path: Path, name: str, upstream_roots: list[Path] | None = None):
    """Import a vendored kernel.py / oracle_pytorch.py.

    The bare ``spec_from_file_location(name, path)`` form leaves
    ``__package__`` empty, so any relative import inside kernel.py raises
    ``ImportError: attempted relative import with no known parent
    package`` -- this is the ~18-task P2 bucket.

    The vendored kernel files frequently use ``from .X import Y`` to
    pull a helper that lives NEXT TO the upstream source. The freeze
    process stamps the upstream relpath into a ``Source path: <p>``
    header in the kernel.py docstring. This loader:

      1. parses that header,
      2. locates the upstream sibling directory under one of the
         configured upstream roots (``infer/vllm`` / ``infer/sglang/
         python``),
      3. builds a per-task synthetic package chain deep enough for the
         deepest relative-import in the file (``.X`` -> 1 level,
         ``..X`` -> 2 levels) whose ``__path__`` mirrors the upstream
         tree at each ancestor,
      4. ``exec_module``s kernel.py as a child of that chain so
         ``from .X`` resolves into the upstream sibling at no extra
         cost.

    Tasks WITHOUT relative imports keep the legacy single-dir synthetic
    package (back-compat: existing PASSes -- L1_12 / L1_16 / L1_26 --
    are unaffected). When a kernel HAS a relative import but the
    upstream sibling cannot be located, the loader raises a clear
    ``RuntimeError`` rather than falling back silently (a silent
    fallback would re-emit the misleading
    ``attempted relative import with no known parent package`` and we
    refuse to hide why)."""
    kernel_src = path.read_text()
    rel_depth = _max_relative_depth(kernel_src)
    pkg = f"_pa_pkg_{name}"
    if rel_depth == 0 or upstream_roots is None:
        # Legacy path: kernel has no relative imports OR caller did not
        # pass upstream roots (e.g. an oracle_pytorch.py that lives in
        # the task dir and never reaches out). Behaviour unchanged.
        if pkg not in sys.modules:
            synth = types.ModuleType(pkg)
            synth.__path__ = [str(path.parent)]
            sys.modules[pkg] = synth
        leaf_name = f"{pkg}.{path.stem}"
        leaf_pkg = pkg
    else:
        source_path = _extract_source_path(kernel_src)
        if not source_path:
            raise RuntimeError(
                f"_load: kernel {path} uses a relative import "
                f"({rel_depth} dot(s)) but its docstring carries no "
                f"``Source path:`` header -- the freeze tool stamps "
                f"that line into every vendored kernel; missing means "
                f"either the freeze metadata was stripped or the kernel "
                f"is not actually a vendored upstream copy. Cannot "
                f"resolve relative imports without the upstream path."
            )
        sibling = _resolve_upstream_sibling(source_path, upstream_roots)
        if sibling is None:
            roots_str = ", ".join(str(r) for r in upstream_roots)
            raise RuntimeError(
                f"_load: upstream sibling not found for {path} -- "
                f"declared source_path={source_path!r} is absent under "
                f"every configured upstream root ({roots_str}); the "
                f"vendored kernel uses ``from .X`` relative imports "
                f"that resolve into that sibling directory, so we "
                f"cannot import the kernel honestly. Either the "
                f"infer/ checkout is missing the file or the "
                f"Source path header is wrong -- failing loud rather "
                f"than silently falling back."
            )
        leaf_pkg = _build_synthetic_pkg_chain(
            leaf_module_name=name,
            task_dir=path.parent,
            upstream_sibling=sibling,
            depth=rel_depth,
        )
        leaf_name = f"{leaf_pkg}.{path.stem}"
    spec = importlib.util.spec_from_file_location(leaf_name, path)
    mod = importlib.util.module_from_spec(spec)
    mod.__package__ = leaf_pkg
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
    upstream_roots: list[Path] | None = None,
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
    # Snapshot inputs before the kernel call so an in-place upstream
    # entrypoint (e.g. apply_logit_bias on L1_01: writes -inf into the
    # supplied ``logits`` tensor and returns None) does not double-apply
    # when the oracle subsequently reads the SAME mapping. Without this
    # clone the oracle saw post-mutation logits, re-applied the same
    # transformations, and produced a doubly-mutated reference -- a
    # spurious value-mismatch that masked a correct kernel as
    # ``oracle-mismatch``. Tensor identity is preserved for non-tensors.
    kernel_inputs = {
        k: (v.clone() if torch.is_tensor(v) else v)
        for k, v in inputs.items()
    }
    kmod = _load(
        task_dir / "kernel.py",
        f"_pa_oc_kernel_{sid}",
        upstream_roots=upstream_roots,
    )
    fn = getattr(kmod, entrypoint, None)
    if not callable(fn):
        return False, f"entrypoint {entrypoint!r} not a module-level callable"
    try:
        sig, bound, kwargs = _map_inputs_by_name(
            fn, entrypoint, dict(kernel_inputs)
        )
    except Exception as e:  # ShapesIncompleteError or introspection
        return False, f"by-name bind failed: {type(e).__name__}: {e}"
    cand_ret = fn(dict(kernel_inputs)) if sig is None else fn(**kwargs)
    if cand_ret is None:
        cand_ret = {
            p: kernel_inputs[p]
            for p in bound
            if torch.is_tensor(kernel_inputs.get(p))
        }
    # oracle_pytorch.py is task-local and has no relative imports onto
    # the upstream tree; passing ``upstream_roots=None`` keeps it on the
    # legacy single-dir loader path.
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
    # sglang's @cached_triton_kernel decorator (common.py:3917-3934) wraps
    # the @triton.jit fn in a CachedKernel object whose __init__ asserts
    # isinstance(fn, triton.runtime.jit.JITFunction). The wrap fires only
    # when is_cuda() is True AND SGLANG_USE_CUSTOM_TRITON_KERNEL_CACHE is
    # unset; on hosts with a real CUDA device the assert path is taken,
    # and the strict isinstance check fails under our import setup
    # (sys.path-layered infer/ + synthetic per-call package in _load),
    # which used to mis-bucket L2_04..L2_09 as upstream-not-invocable-here.
    # Setting the env var to "0" routes the decorator down the `return fn`
    # branch (the native triton cache path) -- correctness is unchanged
    # because we only need to invoke the kernel once for oracle
    # comparison, not benefit from sglang's launch-time caching. Use
    # setdefault so an operator who explicitly exports the variable can
    # still opt into the CachedKernel wrapper.
    os.environ.setdefault("SGLANG_USE_CUSTOM_TRITON_KERNEL_CACHE", "0")
    task_dir = Path(spec["task_dir"])
    out_json = Path(spec["out_json"])
    entrypoint = spec["entrypoint"]

    # infer/ on path FIRST (frozen upstream resolution), then repo root
    # for triton_ascend_bench.* + scripts for the shared adapter.
    for p in (spec["vllm_root"], spec["sglang_root"], spec["repo_root"],
              str(SCRIPTS_DIR)):
        if p not in sys.path:
            sys.path.insert(0, p)

    # Inject the vllm.triton_utils forward shim BEFORE the vendored
    # kernel.py executes. On a host with no active Triton driver, the
    # real upstream vllm/triton_utils/__init__.py would hand out a
    # TritonPlaceholder (no-op .jit, no-op .autotune, placeholder tl.*);
    # the shim re-exports the real triton + an interpreter-aware
    # autotune wrapper so the kernel can be invoked under
    # TRITON_INTERPRET=1 for the oracle-consistency check. We register
    # with force=True so a transitive ``import vllm`` higher in the
    # subprocess (which would otherwise have populated sys.modules with
    # the placeholder regime) is overridden. ImportError is swallowed
    # so a setup without the shim package (e.g. an old editable install)
    # still runs the legacy upstream path -- the resolver will simply
    # report the placeholder failure as it did before, never a fake
    # PASS.
    try:
        from triton_ascend_bench.upstream_shims._inject import (
            register as _register_vllm_triton_utils_shim,
        )

        _register_vllm_triton_utils_shim(force=True)
    except ImportError:
        pass

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

        # Order: vllm probed first, sglang second. Matches the sys.path
        # insertion order so a kernel whose ``Source path`` happens to
        # exist under both roots (rare) binds to the same upstream the
        # rest of the import graph already chose.
        upstream_roots: list[Path] = [
            Path(spec["vllm_root"]),
            Path(spec["sglang_root"]),
        ]

        all_ok = True
        per_shape: list[str] = []
        for sid in sids:
            ok, det = _check_one_shape(
                task_dir, entrypoint, sid, shapes_doc, tcls, tol_name,
                compare, build_inputs, _map_inputs_by_name,
                upstream_roots=upstream_roots,
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
