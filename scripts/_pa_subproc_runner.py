#!/usr/bin/env python3
"""_pa_subproc_runner.py -- Phase A' S2 subprocess entrypoint (U4).

Runs ONE side of the S2 equivalence pair in a fresh subprocess and
serialises its outputs with safetensors (data-only IPC, no pickle --
reuses the ``process_isolation.py`` safetensors-only IPC pattern).

Two modes
---------
``--side original``
    ``sys.path`` is prepended with ``infer/vllm`` and
    ``infer/sglang/python`` (VERIFIED roots: ``vllm/vllm/__init__.py``,
    ``sglang/python/sglang/__init__.py``). Runs the pre-severance
    frozen source.

``--side severed``
    ``sys.path`` is torch + triton + Class-D + pool + shim ONLY. An
    **import audit** asserts ZERO ``vllm`` / ``sglang`` / site-packages
    module resolution. The audit runs in TWO places:

      * a ``sys.meta_path`` finder installed BEFORE the kernel module
        executes -- catches a kernel that imports vllm/sglang *at
        runtime* inside ``run()`` (not just at import time), and
      * a post-run ``sys.modules`` re-scan.

    Any leak (import-time, runtime, or post-run) ⇒ exit code 3 and the
    parent marks the task BLOCKED (plan U4: defends sys.modules
    pollution + runtime-import escape -> "S2 passes but a clean box
    fails").

Output serialisation preserves stride / storage-offset / data_ptr
identity metadata WITHOUT contiguous-coercing the compared tensors, so
the parent can detect a stride / alias / dtype difference. Non-tensor
and nested (list / dict / tuple) outputs are preserved as a structure
descriptor + leaf-tensor table. Inputs mutated in place are captured
(pre-call snapshot vs post-call) so the parent can compare mutated
inputs between the two sides.

The parent (``phase_a_prime.py --verify``) compares the two dumps with
a STRICT bit-exact assertion (no ULP escape, NOT ``tolerance_class``)
including the alias-relationship and mutated-input checks. On a box
with no CUDA the parent never even spawns the GPU-authoritative pair --
it honestly returns ``UNRUN(needs-gpu)``.
"""
from __future__ import annotations

import argparse
import importlib.abc
import importlib.util
import json
import re
import sys
import types
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
INFER_ROOT = REPO_ROOT.parent / "infer"
VLLM_ROOT = INFER_ROOT / "vllm"
SGLANG_ROOT = INFER_ROOT / "sglang" / "python"

# Header line that the freeze process stamps into every vendored
# kernel.py docstring; identical contract to the oracle runner.
_SRC_PATH_RE = re.compile(
    r"Source\s+path\s*:\s*`?([^`\n]+?)`?\s*$",
    re.IGNORECASE | re.MULTILINE,
)
# ``from .X import Y`` / ``from ..X import Y`` / ``from . import Y``.
_REL_IMPORT_RE = re.compile(
    r"^\s*from\s+(\.+)(\w*(?:\.\w+)*)\s+import\s+",
    re.MULTILINE,
)

# Third-party packages permitted on the severed surface (torch+triton
# + Class-D + safetensors/yaml IPC plumbing). Everything else loaded
# from site-packages is a leak.
_SEVERED_ALLOWED_TOP = {
    "torch", "triton", "einops", "numpy", "yaml", "safetensors", "_yaml",
}
_FORBIDDEN_TOP = {"vllm", "sglang"}


class _LeakGuardFinder(importlib.abc.MetaPathFinder):
    """A ``sys.meta_path`` finder that records (and refuses) any
    attempt to resolve a forbidden module DURING execution.

    Installed before the kernel module runs, so a kernel that does
    ``import vllm`` *inside* ``run()`` (after the pre-scan) is still
    caught -- not just import-time imports."""

    def __init__(self) -> None:
        self.runtime_leaks: list[str] = []

    def find_spec(self, fullname, path, target=None):
        top = fullname.split(".")[0]
        if top in _FORBIDDEN_TOP:
            self.runtime_leaks.append(fullname)
            raise ImportError(
                f"Phase A' severed import audit: runtime import of "
                f"forbidden module {fullname!r} (zero vllm/sglang "
                f"resolution is required on the severed surface)"
            )
        # Return None -> defer to the normal finders for allowed modules.
        return None


def _import_audit_modules(baseline: set[str] | None = None) -> list[str]:
    """``sys.modules`` audit. Forbidden:

      * ANY ``vllm`` / ``sglang`` module -- ALWAYS a leak, regardless
        of when it was loaded (the severed surface must resolve ZERO
        vllm/sglang), AND
      * a *kernel-introduced* ``site-packages`` module whose top-level
        is not on the severed allowlist.

    ``baseline`` is the set of module names already present at
    interpreter startup (``.pth`` / editable-install finders, dill,
    etc. that CPython auto-loads before the kernel runs). Those are
    environment noise, NOT a kernel leak, so the site-packages check
    only considers modules loaded AFTER the baseline. vllm/sglang are
    never excused by the baseline."""
    base = baseline or set()
    leaks: list[str] = []
    for name, mod in list(sys.modules.items()):
        top = name.split(".")[0]
        if top in _FORBIDDEN_TOP:
            # vllm/sglang is a leak even if (somehow) pre-baseline.
            leaks.append(name)
            continue
        if name in base:
            continue  # interpreter-startup noise, not a kernel import
        f = getattr(mod, "__file__", None)
        if f and "site-packages" in str(f):
            if top not in _SEVERED_ALLOWED_TOP:
                leaks.append(f"{name} ({f})")
    return leaks


def _capture_infer_closure() -> list[str]:
    """codex r7 [confirmed] HOLE 3 -- snapshot every loaded module whose
    ``__file__`` resolves under an infer/ root (``infer/vllm`` /
    ``infer/sglang/python``). The S2 ``original`` side ALSO puts infer/
    on ``sys.path`` and imports the UNMODIFIED upstream kernel.py, so a
    verdict-affecting infer/ file imported ONLY on the S2-original path
    (not on the oracle-consistency path) was previously OUTSIDE the
    per-task ``upstream_fp`` closure (the oracle runner captured only
    its OWN side). Mirroring the oracle runner's capture here and
    UNIONing it into ``upstream_fp`` makes the per-task belt genuinely
    reflect the COMPLETE live-upstream closure (oracle UNION S2-original).
    Order-independent; absolute resolved paths (the parent maps them to
    infer-rooted relpaths + byte-hashes them, exactly like the oracle
    side)."""
    roots = [str(VLLM_ROOT.resolve()), str(SGLANG_ROOT.resolve())]
    out: list[str] = []
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
                    out.append(_af)
                break
    return sorted(out)


def _write_status(spec: dict, status: dict, side: str) -> None:
    """Write the per-side status JSON. codex r7 [confirmed] HOLE 3: on
    the ``original`` side ALWAYS stamp the live infer/ imported closure
    (success OR error -- a failing original that DID import upstream
    bytes still recorded the closure that shaped the verdict, exactly
    like the oracle runner's finally-sense capture). The severed side
    must resolve ZERO vllm/sglang, so its closure is irrelevant (and is
    not captured)."""
    if side == "original":
        status["upstream_closure"] = _capture_infer_closure()
    Path(spec["status_json"]).write_text(json.dumps(status))


def _storage_id(t) -> int:
    """Process-local-but-stable identity of the UNDERLYING storage
    buffer (NOT the element pointer).

    ``t.data_ptr()`` is ``storage_base + storage_offset*itemsize`` -- two
    views of ONE tensor at different offsets have *different* data_ptr
    but the *same* storage, so a data_ptr key MISSES same-storage views
    (HOLE C). ``untyped_storage().data_ptr()`` is the storage base and
    is identical for every view of the same buffer regardless of
    offset, so it is the correct alias-grouping key."""
    if hasattr(t, "untyped_storage"):
        return int(t.untyped_storage().data_ptr())
    return int(t.storage().data_ptr())


def _tensor_meta(t) -> dict:
    """Identity / layout metadata for a tensor WITHOUT coercing it to
    contiguous (contiguous() would destroy the stride/alias evidence
    the comparator needs)."""
    return {
        "dtype": str(t.dtype),
        "shape": list(t.shape),
        "stride": list(t.stride()),
        "storage_offset": int(t.storage_offset()),
        # data_ptr is the *element* pointer (base + offset*itemsize):
        # kept for diagnostics only -- NOT the alias key (HOLE C).
        "data_ptr": int(t.data_ptr()),
        # storage_id is the UNDERLYING-buffer base pointer: the correct
        # alias-grouping key (same buffer, any offset -> aliased). The
        # parent groups by (storage_id) and records storage_offset +
        # size/stride so same-storage-different-offset views are
        # correctly flagged as aliased.
        "storage_id": _storage_id(t),
        "storage_nbytes": int(t.untyped_storage().nbytes())
        if hasattr(t, "untyped_storage")
        else int(t.storage().nbytes()),
    }


def _encode_tree(obj, leaves: dict, prefix: str):
    """Recursively encode ``obj`` into a JSON-able structure descriptor;
    every leaf tensor is registered in ``leaves`` (keyed by a stable
    path) WITHOUT contiguous-coercion. Non-tensor leaves are inlined."""
    import torch

    if torch.is_tensor(obj):
        key = prefix or "_root"
        meta = _tensor_meta(obj)
        # Keep the tensor as-is for safetensors (safetensors requires
        # contiguous storage on disk, so we serialise a *copy* but the
        # METADATA above is from the ORIGINAL non-contiguous tensor --
        # stride/alias evidence is preserved in `meta`, the bytes are a
        # faithful value snapshot).
        leaves[key] = (obj.detach().cpu(), meta)
        return {"__t__": key}
    if isinstance(obj, dict):
        return {
            "__d__": {
                str(k): _encode_tree(v, leaves, f"{prefix}.{k}")
                for k, v in obj.items()
            }
        }
    if isinstance(obj, (list, tuple)):
        return {
            ("__l__" if isinstance(obj, list) else "__tup__"): [
                _encode_tree(v, leaves, f"{prefix}[{i}]")
                for i, v in enumerate(obj)
            ]
        }
    # Non-tensor scalar / None / str / etc -- inline verbatim.
    if isinstance(obj, (int, float, bool, str)) or obj is None:
        return {"__v__": obj}
    return {"__repr__": repr(obj)}


def _move_to_device(obj, device: str):
    """Recursively move EVERY tensor in ``obj`` (dict / list / tuple /
    bare tensor) onto ``device``, preserving structure and non-tensor
    leaves verbatim.

    BLOCKER 1 (codex r3 [confirmed]): real Triton GPU kernels REQUIRE
    CUDA tensors -- on a GPU host the authoritative S2 pass MUST hand
    the entrypoint CUDA inputs or it no-runs / errors and
    S2-gpu-authoritative is structurally unable to PASS. The
    CPU/``TRITON_INTERPRET=1`` smoke pass passes ``device='cpu'`` so it
    is a no-op there (smoke stays CPU, runs on a no-GPU box).

    The move happens BEFORE the entrypoint is invoked. All stride /
    storage-offset / ``storage_id`` alias evidence is captured by
    ``_tensor_meta`` FROM the resulting on-device tensor objects (the
    same objects the kernel mutates / aliases), so the device move does
    NOT destroy the HOLE C alias/stride evidence -- only the value
    *bytes* are later copied to CPU (``.detach().cpu()``) for
    safetensors serialisation, AFTER the on-device metadata is
    recorded."""
    import torch

    if torch.is_tensor(obj):
        return obj.to(device)
    if isinstance(obj, dict):
        return {k: _move_to_device(v, device) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_move_to_device(v, device) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_move_to_device(v, device) for v in obj)
    return obj


def _input_snapshot(inp: dict) -> dict:
    """Pre-call value+meta snapshot of every input tensor (for
    mutated-input detection)."""
    import torch

    snap: dict = {}
    for k, v in inp.items():
        if torch.is_tensor(v):
            snap[k] = {
                "meta": _tensor_meta(v),
                "bytes": v.detach().cpu().clone(),
            }
    return snap


SHAPES_INCOMPLETE_MARKER = "BLOCKED(shapes-incomplete"


class ShapesIncompleteError(RuntimeError):
    """A required (NO-default) entrypoint parameter has no value in
    shapes.yaml/build_inputs. This is a benchmark GAP (shapes.yaml is
    incomplete), NOT a severing defect -- and CRITICALLY it must NEVER
    be papered over with a silent positional shift (codex r5 ISSUE 1):
    skipping a missing param and shifting the remaining positional args
    can land e.g. ``is_causal=True`` in the ``b_seq_len`` slot, causing
    either a misleading crash OR (far worse) a wrong-but-"successful"
    comparison. We map STRICTLY by name and fail LOUD + explicit."""


def _map_inputs_by_name(fn, entrypoint_name: str, inputs: dict):
    """Map ``inputs`` onto ``fn``'s declared parameters STRICTLY BY NAME
    -- never by silently skip+shift (the codex r5 ISSUE 1 soundness
    hole). For each declared positional/keyword parameter:

      * present in ``inputs``           -> pass it (by keyword, so a
        gap can never positionally shift a later arg).
      * absent but has a real default   -> omit it; the entrypoint's
        own default applies (legitimately optional).
      * absent and NO default           -> hard, explicit
        ``ShapesIncompleteError`` -> the runner surfaces a
        ``BLOCKED(shapes-incomplete: missing param <name> for
        entrypoint <e>)`` (a loud HONEST benchmark gap, never a silent
        mis-call / false PASS).

    ``*args`` / ``**kwargs`` params are skipped (nothing to bind);
    a signature we cannot introspect falls back to ``fn(inputs)``."""
    import inspect

    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return None, [], {(): inputs}  # opaque: caller does fn(inputs)

    kwargs: dict = {}
    bound: list[str] = []
    for name, p in sig.parameters.items():
        if p.kind in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ):
            continue
        if name in inputs:
            kwargs[name] = inputs[name]
            bound.append(name)
        elif p.default is not inspect.Parameter.empty:
            continue  # optional: the entrypoint's own default applies
        else:
            raise ShapesIncompleteError(
                f"{SHAPES_INCOMPLETE_MARKER}: missing param {name!r} "
                f"for entrypoint {entrypoint_name!r} -- shapes.yaml / "
                f"build_inputs provides no value and the parameter has "
                f"no default. Refusing to silently skip+shift positional "
                f"args (would mis-call the entrypoint / risk a "
                f"wrong-but-successful comparison). Fix: add {name!r} to "
                f"the task's shapes.yaml inputs."
            )
    return sig, bound, kwargs


def _signature_adapter(fn, entrypoint_name: str):
    """Wrap a positional-signature upstream entrypoint (e.g.
    ``apply_logit_bias(logits, expanded_idx_mapping, ...)``) as a
    ``run(inputs: dict) -> dict`` callable WITHOUT changing numeric
    logic: it maps ``inputs[param_name]`` onto the entrypoint's declared
    parameters STRICTLY BY NAME (codex r5 ISSUE 1 -- never a silent
    skip+positional-shift), then normalises the return into a dict so
    the two sides' manifests are structurally comparable.

    Many upstream kernels mutate an argument in place and return
    ``None``; in that case the mapped input tensors ARE the outputs, so
    we surface every mapped tensor arg as a named output. This is a
    faithful capture of the unmodified upstream behaviour -- the
    mutated-input + alias machinery in the parent then compares it
    bit-exactly against the severed candidate."""
    import torch

    def _run(inputs: dict) -> dict:
        sig, bound, kwargs = _map_inputs_by_name(
            fn, entrypoint_name, inputs
        )
        if sig is None:
            # Opaque signature: faithful old fallback (fn(inputs)).
            ret = fn(inputs)
            bound = []
        else:
            ret = fn(**kwargs)
        if ret is None:
            # In-place upstream entrypoint: outputs == mutated tensor
            # args (named after their parameter for a stable structure).
            out = {
                p: inputs[p]
                for p in bound
                if torch.is_tensor(inputs.get(p))
            }
            return out or {"_inplace": None}
        if isinstance(ret, dict):
            return ret
        if isinstance(ret, (list, tuple)):
            return {f"_out{i}": v for i, v in enumerate(ret)}
        return {"_out0": ret}

    _run.__pa_entrypoint__ = entrypoint_name  # provenance / debugging
    return _run


def _extract_source_path(kernel_src: str) -> str | None:
    """Return the upstream ``Source path: <p>`` declared at the top of a
    vendored kernel.py (or ``None`` if the header is missing). Identical
    contract to ``_pa_oracle_check_runner._extract_source_path``."""
    head = kernel_src[:4000]
    m = _SRC_PATH_RE.search(head)
    return m.group(1).strip() if m else None


def _max_relative_depth(kernel_src: str) -> int:
    """Return the maximum number of leading dots in any module-scope
    ``from .X import``. Identical contract to the oracle runner."""
    depth = 0
    for m in _REL_IMPORT_RE.finditer(kernel_src):
        depth = max(depth, len(m.group(1)))
    return depth


def _resolve_upstream_sibling(
    source_path: str, upstream_roots: list[Path],
) -> Path | None:
    """Locate the upstream file's parent directory under one of the
    configured upstream roots. Identical contract to the oracle runner."""
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
    ``from ..X`` (depth==2) imports, and return the leaf-package's name.

    Mirrors ``_pa_oracle_check_runner._build_synthetic_pkg_chain`` --
    the runners share the contract on purpose so a verdict on one side
    matches the import surface of the other."""
    base = f"_pa_pkg_{leaf_module_name}"
    levels: list[Path] = [upstream_sibling]
    cur = upstream_sibling
    for _ in range(max(depth - 1, 0)):
        cur = cur.parent
        levels.append(cur)
    parts: list[str] = []
    for i, lvl in enumerate(reversed(levels)):
        parts.append(f"_{i}")
        full = f"{base}." + ".".join(parts)
        if full not in sys.modules:
            pkg_mod = types.ModuleType(full)
            pkg_mod.__path__ = [str(lvl)]
            sys.modules[full] = pkg_mod
        else:
            sys.modules[full].__path__ = [str(lvl)]
    leaf_pkg = f"{base}." + ".".join(parts)
    sys.modules[leaf_pkg].__path__ = [str(task_dir), str(upstream_sibling)]
    if base not in sys.modules:
        top = types.ModuleType(base)
        top.__path__ = []
        sys.modules[base] = top
    return leaf_pkg


def _load_kernel_with_optional_bridge(
    kp: Path, name: str, upstream_roots: list[Path] | None,
):
    """Import a vendored kernel.py with a per-call synthetic package.

    Same contract as ``_pa_oracle_check_runner._load`` -- the original
    side of the S2 pair (``--side original``) needs the same relative-
    import fix the oracle anchor needs, because both load the SAME
    unmodified vendored kernel.py with the SAME ``from .X import Y``
    statements. The severed side never reaches here with relative
    imports (the staged candidate has none), so ``upstream_roots=None``
    is the back-compat path."""
    kernel_src = kp.read_text()
    rel_depth = _max_relative_depth(kernel_src)
    pkg = f"_pa_pkg_{name}"
    if rel_depth == 0 or upstream_roots is None:
        if pkg not in sys.modules:
            synth = types.ModuleType(pkg)
            synth.__path__ = [str(kp.parent)]
            sys.modules[pkg] = synth
        leaf_name = f"{pkg}.{kp.stem}"
        leaf_pkg = pkg
    else:
        source_path = _extract_source_path(kernel_src)
        if not source_path:
            raise RuntimeError(
                f"_load_kernel_with_optional_bridge: kernel {kp} uses a "
                f"relative import ({rel_depth} dot(s)) but its docstring "
                f"carries no ``Source path:`` header -- the freeze tool "
                f"stamps that line into every vendored kernel; missing "
                f"means either the freeze metadata was stripped or the "
                f"kernel is not actually a vendored upstream copy."
            )
        sibling = _resolve_upstream_sibling(source_path, upstream_roots)
        if sibling is None:
            roots_str = ", ".join(str(r) for r in upstream_roots)
            raise RuntimeError(
                f"_load_kernel_with_optional_bridge: upstream sibling "
                f"not found for {kp} -- declared source_path="
                f"{source_path!r} is absent under every configured "
                f"upstream root ({roots_str}); the vendored kernel uses "
                f"``from .X`` relative imports that resolve into that "
                f"sibling directory."
            )
        leaf_pkg = _build_synthetic_pkg_chain(
            leaf_module_name=name,
            task_dir=kp.parent,
            upstream_sibling=sibling,
            depth=rel_depth,
        )
        leaf_name = f"{leaf_pkg}.{kp.stem}"
    spec = importlib.util.spec_from_file_location(leaf_name, kp)
    mod = importlib.util.module_from_spec(spec)
    mod.__package__ = leaf_pkg
    sys.modules[leaf_name] = mod
    spec.loader.exec_module(mod)
    return mod


def _resolve_entrypoint(mod, spec: dict):
    """Resolve the callable this side must invoke with ``inputs``.

    The two S2 sides run DIFFERENT code with the SAME inputs and must
    produce structurally comparable manifests (HOLE B -- the flow was
    circular because BOTH sides required a ``run()`` the real task
    kernel.py never exposes, while ``run()`` only existed AFTER an
    S2-PASS rewrite):

      * ``severed``  -> the STAGED standalone candidate. The candidate
        is the rewrite output PLUS a synthesised ``run(inputs) -> dict``
        (built by ``build_staged_candidate`` into a TEMP dir BEFORE S2;
        the real kernel.py is NEVER touched until S2-PASS promotes it).
        It exposes ``run`` directly.
      * ``original`` -> the UNMODIFIED upstream kernel.py (real
        ``infer/`` on sys.path so its vllm/sglang imports resolve to the
        frozen upstream source). It exposes its real positional
        entrypoint (``meta.yaml: oracle.gpu_triton`` ->
        ``spec['entrypoint']`` = ``kernel.py::<name>``), NOT ``run``.
        We wrap it with the SAME deterministic dict->positional adapter
        the staged candidate uses -- pure argument shuffle, zero numeric
        change -- so original runs unmodified upstream code yet yields a
        comparable manifest.

    Falls back to a bare ``run`` attribute (smoke/GPU fixtures and the
    staged severed module always expose ``run``)."""
    ep = spec.get("entrypoint")
    if ep:
        name = ep.split("::")[-1].strip()
        fn = getattr(mod, name, None)
        if callable(fn):
            # Already a run(inputs)->dict (staged candidate / fixture).
            if name == "run":
                return fn
            return _signature_adapter(fn, name)
    return getattr(mod, "run", None)


def _run_side(args: argparse.Namespace) -> int:
    spec = json.loads(Path(args.spec).read_text())
    side = args.side

    if side == "original":
        # original side: infer/ on path -> pre-severance source.
        for p in (str(VLLM_ROOT), str(SGLANG_ROOT)):
            if p not in sys.path:
                sys.path.insert(0, p)
        # Also expose the bench repo root so the vllm.triton_utils
        # forward shim can be imported below. The infer/ entries above
        # are inserted FIRST so a real ``vllm.X`` submodule still wins
        # over anything in the bench package.
        if str(REPO_ROOT) not in sys.path:
            sys.path.insert(0, str(REPO_ROOT))
        # Inject the vllm.triton_utils forward shim BEFORE loading the
        # vendored kernel.py. Without it, on a host with no active
        # Triton driver upstream hands out a TritonPlaceholder whose
        # .jit / .autotune are no-ops and whose tl.* attributes are
        # placeholder values -- the vendored kernel's
        # ``from vllm.triton_utils import tl, triton`` would silently
        # bind those stubs and the S2-original launch would fail (or
        # worse, silently no-op). force=True overrides a placeholder
        # entry that a transitive ``import vllm`` may have already
        # populated. The severed side intentionally does NOT register
        # the shim: that surface forbids vllm/sglang resolution at all,
        # and the staged candidate has no vllm.triton_utils dependency.
        try:
            from triton_ascend_bench.upstream_shims._inject import (
                register as _register_vllm_triton_utils_shim,
            )

            _register_vllm_triton_utils_shim(force=True)
        except ImportError:
            pass
        guard = None
    else:
        # severed side: torch+triton+Class-D+pool+shim ONLY.
        pool = REPO_ROOT / "tasks" / "_frozen_deps_pool"
        shimp = REPO_ROOT / "tasks"
        for p in (str(pool.parent), str(shimp)):
            if p not in sys.path:
                sys.path.insert(0, p)
        # Expose REPO_ROOT so the rewriter-emitted ``from tasks._frozen_shim
        # import ...`` resolves at IMPORT TIME. Without it the severed kernel
        # is loaded before the later REPO_ROOT insertion (around the run()
        # dispatch) and fails with ``ModuleNotFoundError: No module named
        # 'tasks'``. The leak guard still blocks vllm/sglang at this surface;
        # bench-local imports were already reachable via the post-load
        # REPO_ROOT insertion, so this is not a new leak class.
        if str(REPO_ROOT) not in sys.path:
            sys.path.insert(0, str(REPO_ROOT))
        # Install the runtime import-leak guard BEFORE the kernel runs
        # (catches `import vllm` inside run(), not just import-time).
        guard = _LeakGuardFinder()
        sys.meta_path.insert(0, guard)

    status: dict = {
        "side": side, "ran": False, "error": None,
        "audit_leaks": [], "runtime_leaks": [],
    }

    try:
        import importlib.util

        import torch
        from safetensors.torch import save_file

        # Baseline = every module already present BEFORE the kernel
        # module executes (interpreter / .pth / editable-install
        # finders, dill, torch, triton, safetensors -- legit noise).
        # The site-packages audit only considers the kernel-introduced
        # DELTA; vllm/sglang are forbidden regardless of baseline.
        baseline = set(sys.modules.keys())

        kp = Path(spec["kernel_path"])
        # Original-side kernels are the UNMODIFIED vendored copies that
        # may carry ``from .X import Y`` relative imports onto the
        # upstream sibling helpers. Severed-side kernels are the staged
        # standalone candidates produced by the rewrite, which have no
        # relative imports and MUST stay on the severed surface (the
        # bridge would resolve into infer/ and the leak guard would
        # block it anyway). Passing ``upstream_roots`` only on the
        # original side keeps the behaviour split honest.
        if side == "original":
            kernel_upstream_roots = [VLLM_ROOT, SGLANG_ROOT]
        else:
            kernel_upstream_roots = None
        mod = _load_kernel_with_optional_bridge(
            kp, "_pa_kernel", kernel_upstream_roots,
        )

        if side == "severed":
            # Pre-run sys.modules scan (import-time leaks): audit only
            # what the kernel MODULE introduced vs the baseline.
            leaks = _import_audit_modules(baseline)
            status["audit_leaks"] = leaks
            if guard is not None and guard.runtime_leaks:
                status["runtime_leaks"] = list(guard.runtime_leaks)
            if leaks or (guard is not None and guard.runtime_leaks):
                status["error"] = (
                    f"import audit leak: modules={leaks[:5]} "
                    f"runtime={getattr(guard, 'runtime_leaks', [])[:5]}"
                )
                _write_status(spec, status, side)
                return 3  # BLOCKED signal to parent

        fn = _resolve_entrypoint(mod, spec)
        if not callable(fn):
            status["error"] = (
                "no resolvable entrypoint: severed/staged must expose "
                "run(inputs)->dict; original must expose the upstream "
                f"entrypoint named in spec['entrypoint']={spec.get('entrypoint')!r}"
            )
            _write_status(spec, status, side)
            return 2

        sys.path.insert(0, str(REPO_ROOT))
        from triton_ascend_bench.inputs import build_inputs

        shapes_doc = json.loads(Path(spec["shapes_json"]).read_text())
        out_dir = Path(spec["out_dir"])
        # BLOCKER 1: the parent passes device='cuda' for the
        # S2-gpu-authoritative pass and device='cpu' for the
        # CPU/TRITON_INTERPRET smoke pass. On a GPU host the real
        # Triton kernel needs CUDA tensors; on a no-GPU host the smoke
        # pass keeps device='cpu' (no-op move). We assert the requested
        # CUDA device is actually present so a misconfigured GPU host
        # FAILS LOUDLY rather than silently running CPU tensors (which
        # would make S2-gpu-authoritative vacuously "run" without ever
        # touching the GPU -- a silent-fake we refuse).
        device = str(spec.get("device", "cpu"))
        if device.startswith("cuda") and not torch.cuda.is_available():
            status["error"] = (
                "spec requested device='cuda' for the "
                "S2-gpu-authoritative pass but torch.cuda.is_available() "
                "is False in this subprocess -- refusing to run the "
                "authoritative pass on CPU tensors (would be a silent "
                "fake). The parent must only request cuda when a GPU "
                "is present (it gates on _cuda_available())."
            )
            _write_status(spec, status, side)
            return 2
        manifest: dict = {"shapes": {}}
        for sid in spec["shape_ids"]:
            inp = build_inputs(shapes_doc, sid, seed=spec["seed"])
            # Move inputs onto the target device BEFORE the entrypoint
            # runs (no-op when device='cpu'). All stride/alias evidence
            # is captured from the resulting on-device tensors below.
            inp = _move_to_device(inp, device)
            pre = _input_snapshot(inp)
            out = fn(inp)

            leaves: dict = {}
            tree = _encode_tree(out, leaves, "")

            # Capture EVERY input tensor's storage identity (HOLE C
            # part 2): previously only MUTATED inputs entered leaf_meta,
            # so an output that aliases an UNMODIFIED input was
            # invisible. `__in__<name>` carries the post-call
            # storage_id/offset/stride of the SAME input object the
            # kernel received (storage_id is stable across in-place
            # mutation, so this correctly reflects the alias state at
            # return). The parent groups outputs + ALL inputs by
            # storage identity and asserts the relation is identical
            # between original and severed.
            in_alias_meta = {
                f"__in__{k}": _tensor_meta(v)
                for k, v in inp.items()
                if torch.is_tensor(v)
            }

            # Mutated-input detection: compare post-call input tensors
            # to the pre-call snapshot (in-place writes the kernel did).
            mutated: dict = {}
            for k, v in inp.items():
                if not torch.is_tensor(v) or k not in pre:
                    continue
                post_meta = _tensor_meta(v)
                changed = (
                    post_meta != pre[k]["meta"]
                    or not torch.equal(v.detach().cpu(), pre[k]["bytes"])
                )
                if changed:
                    leaves[f"__mut__{k}"] = (v.detach().cpu(), post_meta)
                    mutated[k] = {
                        "leaf": f"__mut__{k}",
                        "pre_meta": pre[k]["meta"],
                        "post_meta": post_meta,
                    }

            # Persist value bytes via safetensors. We CLONE every leaf
            # to a fresh contiguous buffer: an in-place upstream
            # entrypoint surfaces the SAME storage as both an output
            # leaf and a __mut__ leaf, and safetensors REFUSES tensors
            # that share memory. The on-disk copy is ONLY a value
            # snapshot -- the true stride/storage-offset/ALIAS evidence
            # lives in leaf_meta/in_alias_meta (storage_id), which is
            # captured from the ORIGINAL non-cloned tensors, so cloning
            # here does not weaken the alias check.
            flat: dict = {}
            for lk, (t, _m) in leaves.items():
                flat[lk] = t.contiguous().clone()
            if flat:
                save_file(flat, str(out_dir / f"{side}_{sid}.safetensors"))
            manifest["shapes"][sid] = {
                "tree": tree,
                "leaf_meta": {lk: m for lk, (_t, m) in leaves.items()},
                # ALL input tensors' storage identity (HOLE C part 2):
                # used ONLY for the alias-relationship grouping (no
                # value bytes -- inputs are not value-compared here;
                # mutated inputs are value-compared via __mut__). This
                # makes output<->unmodified-input aliasing visible.
                "in_alias_meta": in_alias_meta,
                "mutated_inputs": mutated,
            }

        # Post-run sys.modules RE-SCAN (plan FIX 1): a kernel could
        # import vllm/sglang DURING run() in a way the meta_path guard
        # somehow missed (e.g. a pre-populated sys.modules entry). Catch
        # it here too -- belt and braces.
        if side == "severed":
            post_leaks = _import_audit_modules(baseline)
            if post_leaks:
                status["audit_leaks"] = (
                    list(status.get("audit_leaks", [])) + post_leaks
                )
                status["error"] = (
                    f"post-run import audit leak: {post_leaks[:5]}"
                )
                _write_status(spec, status, side)
                return 3

        (out_dir / f"{side}_manifest.json").write_text(json.dumps(manifest))
        status["ran"] = True
    except Exception as e:
        status["error"] = f"{type(e).__name__}: {e}"

    if side == "severed" and guard is not None and guard.runtime_leaks:
        # A runtime import leak raised ImportError inside run() -> the
        # except above caught it; still surface as a BLOCKED signal.
        status["runtime_leaks"] = list(guard.runtime_leaks)
        _write_status(spec, status, side)
        return 3

    _write_status(spec, status, side)
    return 0 if status["ran"] else 2


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--side", required=True, choices=["original", "severed"])
    ap.add_argument("--spec", required=True, help="JSON spec file path")
    return _run_side(ap.parse_args())


if __name__ == "__main__":
    sys.exit(main())
