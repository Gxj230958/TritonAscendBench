"""process_isolation.py — run a candidate solution in an isolated process.

Implements the dev-machine-runnable subset of the process-isolation model.
The full three-process Docker/seccomp/cgroup/NPU-device model belongs on
the GPU/NPU training machine; here we enforce the security-critical,
host-independent properties:

  * **Process isolation** — the candidate runs in a separate Python
    subprocess (no shared interpreter state / module cache with the
    trusted oracle), spawned via `subprocess.Popen` with
    `start_new_session=True` so it leads its own process group. On
    timeout — and again before the temp dir is torn down — the runner
    `os.killpg`s the whole group, so forked descendants cannot survive.
  * **Two sandboxes** — a candidate is graded in two *independent*
    subprocess passes: a correctness pass then a timing pass. Between
    them the runner SIGKILLs the correctness pass's whole process group
    and removes its private scratch + Triton compile cache. Device allocations
    die with their owning process. This cuts the correctness-pass → timing-pass
    pre-computation channel and stops a forked descendant from leaking
    state into the timing pass (DESIGN §3.3 / §6.6,
    the process-isolation model "Why Two Candidate Processes").
  * **No golden-output leak** — the child receives ONLY
    a sanitised minimal set: `solution.py` + the inputs the runner
    materialised for it. It never sees the original `task_dir`,
    `oracle_pytorch.py`, `contract.md`, `golden/`, or the `shapes.yaml`
    document. Because the child is handed *materialised inputs* and not
    the shapes document at all, the `hidden_shapes` section is
    structurally absent from the candidate view — this is strictly
    stronger than the DESIGN §3.3 "strip the hidden_shapes section"
    requirement. The repo root IS on the child's `PYTHONPATH` so
    its trusted-shim segment can import the in-repo helper modules; this
    leaks no golden data — the candidate cannot `open()` / walk the repo
    (Layer A/E denylist) and the AST whitelist rejects importing the
    oracle / tasks. When a sanitised public shapes view is needed
    elsewhere, :func:`sanitised_shapes_view` produces one with the
    `hidden_shapes` section removed.
  * **safetensors-only IPC** — both directions of the IPC are
    data-only: inputs flow down as safetensors payloads + JSON sidecar
    manifests; outputs flow up the same way. No pickle → no RCE on load.
  * **Runner-materialised inputs** — every shape's inputs (public
    AND hidden) are built by the trusted parent via `inputs.build_inputs`
    and injected down the IPC. The child does NOT call `build_inputs` and
    is not handed the shapes document; this both keeps hidden-shape
    definitions out of the candidate view and removes the need for the
    child to know the full `shapes_doc`.
  * **Host-authoritative timing** — the timing pass times each
    measured trial with `CLOCK_MONOTONIC_RAW` inside a trusted-shim
    segment of the child, on a *fresh* per-trial inputs dict the runner
    injected. A runner-secret RNG — seeded fresh per evaluation from a
    CSPRNG, strictly distinct from the deterministic input-generation
    seed — selects a subset of trials whose outputs are sent up
    (`sampled_outputs`) so the parent can re-verify them against golden,
    closing the correctness-pass cheat path. The profiler diagnostics are
    bound to one of those runner-secret sampled measured trials, so a
    candidate cannot behave honestly on a profiled call while cheating the
    timed calls.
  * **Timeout** — a hung/over-long candidate is terminated.

NOT enforced here (GPU-machine layer, documented in the spec):
seccomp/cgroup/Docker, NPU device allowlist, trusted-shim alias/data_ptr
checks. NPU event timing / profiler stream detection are per-backend
hooks (`backends/npu.py`) — on the dev machine they are stubbed and
timing degrades to host-only.

Candidate entrypoint convention (per spec §Trusted Shim): the solution
exposes `npu_triton_impl(inputs: dict) -> dict[str, Tensor]`, matching
the oracle's `run(inputs) -> dict`.

----------------------------------------------------------------------
 child<->parent protocol (frozen schema — downstream W items depend
on these exact field names and the schema version).

Downstream child args (parent -> child, the JSON in argv[1]):

    {
      "protocol_version": int,        # == PROTOCOL_VERSION
      "status_schema_version": int,   # == STATUS_SCHEMA_VERSION (the child
      #                                 stamps this onto status.json)
      "pass_kind": "correctness" | "timing",
      "solution":   str,              # abs path to solution.py
      "out_dir":    str,              # abs dir for output safetensors
      "status_json": str,             # abs path the child writes status to
      "inputs_mode": "single" | "per_trial",
      "n_trials":   int,              # per-trial sequence length; 1 for "single"
      # timing-pass knobs (ignored for the correctness pass):
      "n_warmup":   int,              # warm-up iterations (not timed)
      "sample_size": int,             # # of trials whose outputs go up
      "sampled_indices": {            # runner-secret RNG choice, per shape
        "<shape_id>": [int, ...]
      },
      "shapes": [
        {
          "shape_id": str,
          # "single": one inputs payload reference
          "input_tensors": str,       # abs path to <ref>.input.safetensors
          "input_sidecar": str,       # abs path to <ref>.input.json
          # "per_trial": one payload reference per trial t=0..n_trials-1
          "trials": [
            {"t": int, "input_tensors": str, "input_sidecar": str}, ...
          ]
        }, ...
      ]
    }

Upstream status.json (child -> parent):

    {
      "schema_version": int,          # == STATUS_SCHEMA_VERSION
      "load_error": str | None,
      "shapes": {
        "<shape_id>": {
          "ran": bool,
          "error": str | None,
          "keys": [str, ...],          # output tensor keys
          # --- diagnostics (timing-pass carriers) ---
          "durations": [float, ...],   # per-trial host_ns (timing pass)
          "event_ns": [float | None, ...] | None,  # per-trial NPU
          #                              event-timer ns (timing pass; None
          #                              on the dev machine — host-only)
          "kernel_event_present": bool | None,
          "stream_nondefault": bool | None,
          "fallback_signal": bool | None,
          "backend_active": bool | None,  # device backend live when measured
          "npu_prof_raw_dir": str | None,  # schema 4: raw torch_npu profiler
          #                              dir for the trusted PARENT to parse
          #                              offline (the in-child CSV parse is
          #                              blocked by the graded sandbox's exec
          #                              block); the parent fills
          #                              kernel_event_present / fallback_signal.
          "sampled_outputs": str | None  # ref to sampled-output payload
        }, ...
      }
    }
----------------------------------------------------------------------
"""
from __future__ import annotations

import csv
import glob
import json
import logging
import os
import secrets
import signal
import stat
import subprocess
import sys
import tempfile
import threading
from dataclasses import dataclass, field
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from ascendbench.input_spec import inputs as _inputs

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

# Roots the trusted NPU child may load shared libraries / toolchain from. The
# standard CANN install prefix + system lib dirs; a deliberately non-standard
# CANN prefix is added HERE, never auto-trusted from a (possibly contaminated)
# parent env var. See _audited_npu_env.
_NPU_TRUSTED_LIB_ROOTS = (
    "/usr/local/Ascend", "/usr/lib", "/usr/lib64", "/lib", "/lib64",
    "/usr/local/lib", "/usr/local/lib64",
)


def _audited_npu_env(keys: tuple[str, ...]) -> dict[str, str]:
    """Value-level audit of the CANN/Ascend loader/toolchain env vars forwarded
    into the trusted child on NPU (G2). Each ``os.pathsep`` entry must be
    absolute, an existing dir, NOT world-writable, under a trusted root, and
    never under the repo / cwd / a tmp dir. Fail-closed: a failing entry is
    dropped and a var that filters to empty is omitted, so a contaminated parent
    env cannot inject a loader path into the trusted child before grading.
    """
    repo = os.path.realpath(str(REPO_ROOT))
    cwd = os.path.realpath(os.getcwd())
    tmp = os.path.realpath(tempfile.gettempdir())
    roots = tuple(os.path.realpath(r) for r in _NPU_TRUSTED_LIB_ROOTS)

    def _safe(entry: str) -> bool:
        if not entry or not os.path.isabs(entry):
            return False
        p = os.path.realpath(entry)
        if not os.path.isdir(p):
            return False
        if any(p == bad or p.startswith(bad + os.sep) for bad in (repo, cwd, tmp)):
            return False
        try:
            if os.stat(p).st_mode & stat.S_IWOTH:
                return False
        except OSError:
            return False
        return any(p == r or p.startswith(r + os.sep) for r in roots)

    out: dict[str, str] = {}
    for k in keys:
        v = os.environ.get(k)
        if not v:
            continue
        kept = os.pathsep.join(e for e in v.split(os.pathsep) if _safe(e))
        if kept:
            out[k] = kept
    return out


# frozen protocol versions. Bump when the wire schema changes;
# downstream W items pin against these.
PROTOCOL_VERSION = 3        # parent -> child contract
# protocol 3 : the parent sets the `AB_NPU_PROF_ROOT` env on the NPU child — a
#   parent-owned trace root (outside the candidate cwd) the child records the
#   profiler into, so the trust boundary for the raw trace is the parent's.
STATUS_SCHEMA_VERSION = 4   # child -> parent status.json schema
# schema 2 : per-shape `event_ns` added — the timing-pass child's NPU
#   event-timer cross-check series (None on the dev machine; real on the
#   910B). The parent feeds it to `timing.arbitrate_timing(event_ns=...)`.
# schema 3 : per-shape `backend_active` added — whether the child's device
#   backend was live (real 910B/CUDA) vs a dev stub. The trusted parent uses
#   it to fail-close only on a real-runner unverified anti-cheat signal (G4).
# schema 4 : per-shape `npu_prof_raw_dir` added — the raw torch_npu profiler dir
#   the child recorded; the in-child CSV parse is blocked by the graded sandbox
#   (exec), so the trusted parent parses it offline and fills
#   kernel_event_present / fallback_signal (deferred None from the child).

# Timing-pass defaults aligned to docs/reference/timing_arbitration.md (10 warmup /
# 100 trials / 10 samples). Kept overridable so a dev-box smoke can drop to a
# cheaper loop.
DEFAULT_N_WARMUP = 10
DEFAULT_N_TRIALS = 100
DEFAULT_SAMPLE_SIZE = 10


def _default_pass_timeout_s():
    try:
        value = int(os.environ.get("TAB_PASS_TIMEOUT_S", "90"))
        return value if value > 0 else 90
    except ValueError:
        return 90


DEFAULT_PASS_TIMEOUT_S = _default_pass_timeout_s()

# Historical paths, retained for callers that imported the constant. Cleanup
# no longer touches these shared locations; every pass owns its own directories.
_SCRUB_PATHS = (
    "/tmp/triton-ascend-cache",
    "/tmp/scratch",
)


@dataclass
class ShapeCandidateResult:
    shape_id: str
    ran: bool
    outputs: dict[str, torch.Tensor] = field(default_factory=dict)
    error: str | None = None
    # --- diagnostics (filled by the timing pass) ----------
    durations: list[float] = field(default_factory=list)
    # per-trial NPU event-timer ns from the timing pass — the
    # host/event-drift cross-check series. `None` on the dev machine (no
    # NPU); the parent passes it to `timing.arbitrate_timing(event_ns=...)`.
    event_ns: list[float] | None = None
    kernel_event_present: bool | None = None
    stream_nondefault: bool | None = None
    fallback_signal: bool | None = None
    # True iff the device backend was active (real 910B/CUDA) when the signals
    # above were taken; lets G4 fail-close a real-runner unknown vs a dev stub.
    backend_active: bool | None = None
    # schema 4: the raw torch_npu profiler dir the child recorded (the in-child
    # CSV parse is blocked by the graded sandbox). The trusted PARENT parses it
    # offline and OVERWRITES kernel_event_present / fallback_signal above; this
    # field is the child->parent hand-off path only.
    npu_prof_raw_dir: str | None = None
    sampled_outputs: dict[str, torch.Tensor] | None = None


@dataclass
class CandidateRunResult:
    ran_any: bool
    timed_out: bool
    load_error: str | None
    per_shape: dict[str, ShapeCandidateResult] = field(default_factory=dict)
    # schema version of the status.json this result was decoded from.
    schema_version: int = STATUS_SCHEMA_VERSION
    # which pass produced this result ("correctness" | "timing").
    pass_kind: str = "correctness"
    streamed: bool = False


@dataclass
class TwoPassResult:
    """The pair of independent passes plus their isolation receipt.

    `correctness` and `timing` are each a `CandidateRunResult` from a
    separate child subprocess. `scrubbed` records what the runner cleared
    between the two passes (the isolation receipt the tests assert on).
    """

    correctness: CandidateRunResult
    timing: CandidateRunResult
    scrubbed: list[str] = field(default_factory=list)


# Standalone child script. Loads the candidate, rebuilds each shape's mixed
# inputs dict from the runner-materialised safetensors payload + JSON
# sidecar (NEVER calls build_inputs / sees shapes.yaml / repo_root). For a
# correctness pass it calls npu_triton_impl once per shape; for a timing
# pass it runs a trusted-shim segment — warm-up + measured per-trial loop
# with CLOCK_MONOTONIC_RAW host timing — and sends up durations + the
# runner-selected sampled outputs. Never imports the oracle.
#
# After the candidate module loads but before any warmup/graded call the
# child installs the Layer E `sys.addaudithook` runtime-defense hook (a brief
# "compile" phase around the triton warmup; "graded" thereafter), and once per
# shape it profiles one candidate call for the NPU kernel-event /
# non-default-stream / fallback diagnostics (kernel-event source stubbed on dev).
_CHILD = r'''
import contextlib, json, math, sys, time, importlib.util
from ascendbench.integrity.stream_transport import child_loop as _stream_child
from pathlib import Path
import torch
from safetensors.torch import load_file, save_file
# Warm safetensors' lazy save-path import (`numpy.ctypeslib`, lazily pulled in
# on first `np.ctypeslib` access in newer numpy) now, before the Layer E hook —
# a post-hook lazy import fires an `exec` audit event the hook would block.
import numpy.ctypeslib  # noqa: F401

args = json.loads(Path(sys.argv[1]).read_text())
out_dir = Path(args["out_dir"])
inputs_mode = args["inputs_mode"]
pass_kind = args.get("pass_kind", "correctness")
# Required (protocol v2): the parent always stamps it. Fail loud on a missing
# key rather than silently routing a gpu request to the npu default.
backend = args["backend"]


def _fail_child(load_error):
    """Write a status.json carrying an explicit `load_error` and exit non-zero.

    Used when the trusted-shim imports fail (the in-repo `ascendbench` helper
    modules failed to import) — a misconfigured child, not a candidate fault, so
    the parent marks the run an error instead of a silent skip.
    """
    Path(args["status_json"]).write_text(json.dumps(
        {"schema_version": args["status_schema_version"],
         "shapes": {}, "load_error": load_error}))
    sys.exit(1)


# ── Trusted-shim segment ──────────────────────────────────────────
# The trusted-shim segment imports the in-repo `ascendbench`
# helpers — the Layer E runtime audit hook and the NPU profiler / event
# timer. The parent prepends the repository root to this child's
# PYTHONPATH so these resolve to the CURRENT checkout (not a stale
# site-package, and not nothing at all in a plain checkout). A failure
# here means the child is misconfigured: it MUST fail loudly with an
# explicit `load_error`, never be silently swallowed — a silent swallow
# would disable the Layer E audit hook and the NPU profiler without any
# signal.
#
# ORDER MATTERS: every fresh module import emits an `exec` audit event,
# and the Layer E hook blocks `exec`. So all trusted-shim modules — and
# everything the candidate is legitimately allowed to use (`torch`,
# `safetensors`, imported at the top of this script) — MUST be imported
# and cached BEFORE `install_runtime_audit_hook()` runs. Once the hook is
# live a fresh import is blocked; the candidate can only touch the
# already-cached modules. The audit hook is therefore installed below,
# after the candidate `solution.py` module has been loaded.
#
# This does not relax candidate isolation: `solution.py` is still loaded under
# the Layer E audit hook + the static AST whitelist (it cannot fresh-import the
# oracle / tasks or walk the package tree). PYTHONPATH governs only what the
# child process imports, not what the candidate may reach (see the PYTHONPATH
# note in the parent for the two-layer rationale).

# Layer E: the sys.addaudithook runtime-defense hook installer. It
# traps dynamic-import / exec / subprocess / ctypes attempts the static
# pass cannot see. Imported here; *installed* below after every other
# trusted-shim import has been cached.
try:
    from ascendbench.integrity.static_checker import (
        install_runtime_audit_hook,
        set_audit_phase,
    )
except Exception as e:  # noqa: BLE001
    _fail_child("trusted-shim import failed: "
                "ascendbench.integrity.static_checker "
                "(%s: %s)" % (type(e).__name__, e))

# child-side profiler: detect NPU kernel-event presence, non-default
# stream activity, and the derived fallback signal. The NPU measurement
# belongs to backends/npu.py; on the dev machine it is a documented stub.
#
# The same backend exposes the timing event-timer seam — an
# `EventTimer` *context manager* whose NPU `Event.record()` calls bracket
# the same measured `fn(inp)` invocation that host time and the profiler
# see. `_BACKEND.event_timer()` returns a fresh timer on the 910B and
# `None` on the dev machine (no NPU); the timing pass degrades to
# host-only there. Either way the `event_ns` series is sent up so the
# parent's `arbitrate_timing` channel is end-to-end connected.
#
# There is deliberately no callable that re-runs the candidate to take a
# separate event sample — that would execute every measured input twice,
# reopening the stateful-candidate hole. The event timer only brackets a
# call the timing loop already makes.
#
# This import is also a trusted-shim import — a failure is a
# misconfigured child, surfaced via `load_error`, not silently swallowed.
# Both backends guard their accelerator import, so the module loads even
# without torch_npu / CUDA; only the *device* is absent on the dev machine.
try:
    from ascendbench.backends import get_backend
    _BACKEND = get_backend(backend)
except Exception as e:  # noqa: BLE001
    _fail_child("trusted-shim import failed: "
                "ascendbench.backends.%s "
                "(%s: %s)" % (backend, type(e).__name__, e))
_PROFILER = _BACKEND if hasattr(_BACKEND, "profile_call") else None

# Force the accelerator's lazy device init NOW, while dlopen is still allowed:
# CUDA defers its driver/runtime dlopen to the first device op, which would
# otherwise fire inside a graded candidate call and be blocked by the Layer E
# hook below (DeferredCudaCallError). No-op on NPU / the dev machine.
try:
    _BACKEND.ensure_initialized()
except Exception as e:  # noqa: BLE001
    _fail_child("backend init failed: %s (%s: %s)"
                % (backend, type(e).__name__, e))

# The Layer E audit hook is installed after the candidate load (see the install
# site below), not here — installing before the load would block the candidate's
# own module-code `exec`. The candidate body already passed the static AST
# whitelist (Layers A-D/F).


def _sync_device():
    """Flush in-flight device work so host timestamps bracket real execution.
    No-op when the backend's accelerator is absent (dev box)."""
    if _BACKEND is not None:
        try:
            _BACKEND.synchronize()
        except Exception:  # noqa: BLE001
            pass


def _make_event_timer():
    """Return a fresh `EventTimer` for ONE measured call, or `None`.

    `None` on the dev machine (no accelerator) — the caller then uses a
    `contextlib.nullcontext()` and the trial is host-only timed.
    """
    if _BACKEND is None:
        return None
    try:
        return _BACKEND.event_timer()
    except Exception:  # noqa: BLE001 — event timer is a best-effort cross-check.
        return None


def _host_ns():
    return time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW)


def _profile(fn, inp):
    """Profile ONE candidate call; return ``(output, diagnostics)``.

    The candidate `fn(inp)` is invoked exactly once: that single call produces
    both the output the caller consumes and the three diagnostic signals, so
    the profiled call IS the graded/timed call. A stateful candidate cannot be
    honest when profiled and cheat on a separate graded/timed call — there is
    no separate call.

    `_PROFILER.profile_call` runs `fn` once and hands back both `output`
    and the kernel-event / stream / fallback signals. When no profiler is
    importable (dev-machine without the package) `fn` is called directly
    here, once, and the signals stay None ("not measured").

    Returns ``(output, {"kernel_event_present", "stream_nondefault",
    "fallback_signal"})``.
    """
    # backend_active distinguishes a real-runner unknown signal (device present,
    # profiler failed -> fail closed at G4) from a dev-machine stub (no device,
    # signals unmeasurable -> not measured, no flip).
    _active = (_BACKEND.is_available()
               if hasattr(_BACKEND, "is_available") else None)
    if _PROFILER is None:
        output = fn(inp)
        return output, {"kernel_event_present": None,
                        "stream_nondefault": None,
                        "fallback_signal": None,
                        "backend_active": _active,
                        "npu_prof_raw_dir": None}
    res = _PROFILER.profile_call(fn, inp)
    # `npu_prof_raw_dir` (NPU only): the raw profiler dir the child recorded.
    # kernel_event_present / fallback_signal are DEFERRED (None) from the child
    # and OVERWRITTEN by the trusted parent's offline parse of this dir.
    diag = {"kernel_event_present": res.get("kernel_event_present"),
            "stream_nondefault": res.get("stream_nondefault"),
            "fallback_signal": res.get("fallback_signal"),
            "backend_active": _active,
            "npu_prof_raw_dir": res.get("npu_prof_raw_dir")}
    return res.get("output"), diag


def _to_device(d):
    """Move tensor inputs to the backend device before fn(inp). No-op when the
    accelerator is absent (dev box) — inputs stay on CPU."""
    if _BACKEND is None:
        return d
    try:
        dev = _BACKEND.device
    except Exception:  # noqa: BLE001 — accelerator absent on the dev box.
        return d
    return {k: (v.to(dev) if hasattr(v, "to") else v) for k, v in d.items()}


def _merge(tensors_path, sidecar_path):
    """Rebuild a mixed inputs dict from a safetensors payload + JSON sidecar.

    : an all-scalar inputs block has no tensors, but safetensors
    refuses an empty dict, so the runner writes a `{"__empty__": zeros(0)}`
    placeholder. Strip that sentinel here so a scalar-only task's
    `npu_triton_impl` never sees a spurious `__empty__` input key.
    """
    merged = dict(load_file(tensors_path))
    merged.pop("__empty__", None)
    sidecar = json.loads(Path(sidecar_path).read_text())
    for k, v in sidecar.items():
        merged[k] = v
    return _to_device(merged)


spec = importlib.util.spec_from_file_location("_candidate", args["solution"])
# The status.json schema version is its own counter ( bumped it to 2
# for the `event_ns` carrier) — distinct from the parent->child
# `protocol_version`. The runner passes it down explicitly.
status = {"schema_version": args["status_schema_version"],
          "shapes": {}, "load_error": None}
try:
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    fn = getattr(mod, "npu_triton_impl", None)
    if not callable(fn):
        status["load_error"] = "solution.py exposes no npu_triton_impl(inputs)"
        fn = None
except Exception as e:  # noqa: BLE001
    status["load_error"] = f"{type(e).__name__}: {e}"
    fn = None

# Q3: a triton kernel needs the compile warmup; a torch-only candidate does
# not. Read the (vetted) source once, before the hook, to skip the extra call
# for non-triton candidates. Fail-open (warmup) if the source is unreadable.
try:
    _uses_triton = "triton" in Path(args["solution"]).read_text()
except Exception:  # noqa: BLE001
    _uses_triton = True

# Layer E: install the sys.addaudithook runtime-defense hook NOW —
# after the candidate module body loaded (vetted source) but BEFORE any warmup
# or graded `fn(inp)` call. It defaults to the "graded" phase (everything
# blocked); the compile warmup below briefly flips it to the narrow "compile"
# phase. CPython emits an `exec` audit event for module-code execution, so
# installing earlier would have blocked the candidate's own load. The install
# is unconditional and not swallowed.
install_runtime_audit_hook()

# Warmup-compile (Q2 phase-aware, Q3 triton-gated): a triton kernel JIT-compiles
# + dlopens its module on the FIRST call. Run it ONCE per shape under the narrow
# "compile" audit phase (native loads allowed, process/shell escapes still
# blocked) so the graded calls below reuse triton's compiled-kernel cache and
# never dlopen. `finally` restores the "graded" phase even on a warmup error, so
# the lockdown is never left relaxed. Gated to a real accelerator AND a triton
# candidate: the dev box / torch-only paths take NO extra call, keeping the
# exactly-one-graded-call R5/R6 invariant intact. Errors are swallowed — a
# genuine candidate fault resurfaces in the graded passes below.
if (not args.get("stream_fd") and fn is not None and _BACKEND is not None
        and _BACKEND.needs_compile_warmup() and _uses_triton):
    set_audit_phase("compile")
    try:
        for _wshape in args["shapes"]:
            try:
                if inputs_mode == "single":
                    _winp = _merge(_wshape["input_tensors"],
                                   _wshape["input_sidecar"])
                else:
                    _wt = _wshape["trials"][0]
                    _winp = _merge(_wt["input_tensors"], _wt["input_sidecar"])
                fn(_winp)
            except Exception:  # noqa: BLE001
                pass
        _sync_device()
    finally:
        set_audit_phase("graded")


def _new_rec():
    return {
        "ran": False, "error": None, "keys": [],
        "durations": [], "event_ns": None, "kernel_event_present": None,
        "stream_nondefault": None, "fallback_signal": None,
        "backend_active": None,
        "npu_prof_raw_dir": None,
        "sampled_outputs": None,
    }


def _flatten(out):
    """Validate the candidate output dict; return (flat_cpu_dict, error)."""
    if not isinstance(out, dict):
        return None, f"npu_triton_impl returned {type(out).__name__}, expected dict"
    flat = {}
    for k, v in out.items():
        if not torch.is_tensor(v):
            return None, f"output[{k!r}] is {type(v).__name__}, not a Tensor"
        flat[k] = v.detach().contiguous().cpu()
    return flat, None


def _run_one(fn, inp):
    """Call the candidate once; return (flat_outputs, error)."""
    return _flatten(fn(inp))


def _record_diag(rec, inp):
    """Profile ONE candidate call; store the diagnostics, return the output.

    The single profiled `fn(inp)` invocation produces both the diagnostics
    (stored on `rec`) and the candidate output, returned here as
    ``(flat_outputs, error)`` so the caller consumes *this* call's result. The
    caller must not re-invoke `fn` on the same input (single profiled call —
    see `_profile`).
    """
    output, diag = _profile(fn, inp)
    rec["kernel_event_present"] = diag.get("kernel_event_present")
    rec["stream_nondefault"] = diag.get("stream_nondefault")
    rec["fallback_signal"] = diag.get("fallback_signal")
    rec["backend_active"] = diag.get("backend_active")
    rec["npu_prof_raw_dir"] = diag.get("npu_prof_raw_dir")
    return _flatten(output)


def _correctness_shape(fn, shape, rec):
    """Correctness pass for one shape: one call, store t=0 output.

    The stored output is the output of the same call `_record_diag` profiled —
    the candidate is never invoked twice on one input, so the diagnostics and
    the graded correctness output describe the same invocation (single profiled
    call — see `_profile`).
    """
    if inputs_mode == "single":
        inp = _merge(shape["input_tensors"], shape["input_sidecar"])
        flat, err = _record_diag(rec, inp)
        if err is not None:
            rec["error"] = err
            return
        save_file(flat, str(out_dir / (shape["shape_id"] + ".safetensors")))
        rec["ran"] = True
        rec["keys"] = list(flat.keys())
    else:  # per_trial: run every trial's fresh inputs; output from t=0.
        flat0 = None
        for trial in shape["trials"]:
            inp = _merge(trial["input_tensors"], trial["input_sidecar"])
            if trial["t"] == 0:
                # The t=0 trial is the profiled one: its single profiled
                # call also produces the stored correctness output.
                flat, err = _record_diag(rec, inp)
            else:
                flat, err = _run_one(fn, inp)
            if err is not None:
                rec["error"] = err
                return
            if trial["t"] == 0:
                flat0 = flat
        if flat0 is not None:
            save_file(flat0, str(out_dir / (shape["shape_id"] + ".safetensors")))
            rec["ran"] = True
            rec["keys"] = list(flat0.keys())


def _timing_shape(fn, shape, rec):
    """Timing pass for one shape (trusted-shim segment).

    Warm-up on the warm trials (negative-index payloads), then a measured
    per-trial loop: each measured trial gets a FRESH runner-built inputs
    dict, host-timed with CLOCK_MONOTONIC_RAW. The runner-secret sampled
    indices select which trials' outputs are sent up for re-verification.

    The profiler diagnostics (kernel-event / non-default-stream / fallback) are
    bound to a runner-secret sampled measured trial, not a sacrificial warm-up
    call — the candidate cannot predict which, so the profiled call is one of
    the calls being timed and output-reverified (single profiled call — see
    `_profile`). The NPU event timer is a *context manager* whose
    `Event.record()` calls bracket that same one invocation; the candidate is
    never re-run merely to take an event sample.
    """
    sid = shape["shape_id"]
    trials = {t["t"]: t for t in shape.get("trials", [])}
    if not trials:
        rec["error"] = "timing pass requires per_trial inputs"
        return
    sampled_idx = set(args.get("sampled_indices", {}).get(sid, []))

    # Phase a: warm-up — negative trial indices, not timed and not profiled:
    # a sacrificial warm-up is exactly the call a candidate could be honest on
    # while cheating the timed calls. The profiler runs on a runner-secret
    # measured trial in phase b.
    warm_ts = sorted(k for k in trials if k < 0)
    for t in warm_ts:
        inp = _merge(trials[t]["input_tensors"], trials[t]["input_sidecar"])
        flat, err = _run_one(fn, inp)
        if err is not None:
            rec["error"] = err
            return

    # Phase b: measured loop — fresh inputs per trial, host-timed.
    # Each measured trial also records an NPU event-timer ns. The event timer
    # is an `EventTimer` context manager that brackets the same single `fn(inp)`
    # call host time and the profiler see — it does not re-run the candidate.
    # On the dev machine `event_timer()` returns None, so `event_ns` stays an
    # all-None series and arbitration degrades to host-only — but the channel up
    # to the parent is connected either way.
    durations = []
    event_ns = []
    sampled = {}
    keys0 = None
    meas_ts = sorted(k for k in trials if k >= 0)
    # The profiler is bound to a runner-secret sampled measured trial: pick the
    # first sampled measured index when the runner sampled any; otherwise (no
    # sampling configured) fall back to the first measured trial so the channel
    # is still populated. Either way it is a real measured invocation, not a
    # warm-up call.
    profiled_t = None
    sampled_meas = sorted(t for t in meas_ts if t in sampled_idx)
    if sampled_meas:
        profiled_t = sampled_meas[0]
    elif meas_ts:
        profiled_t = meas_ts[0]
    for t in meas_ts:
        inp = _merge(trials[t]["input_tensors"], trials[t]["input_sidecar"])
        # One NPU event timer per measured trial, bracketing the one measured
        # call. `None` on the dev machine ⇒ nullcontext, the trial is host-only
        # timed.
        timer = _make_event_timer()
        bracket = timer if timer is not None else contextlib.nullcontext()
        if t == profiled_t:
            # The profiled call IS the measured invocation: the profiler runs
            # `fn(inp)` exactly once and returns both that call's output and the
            # diagnostic signals. Host-time and event-bracket that same call so
            # the trial is timed, profiled, and output-sampled off one
            # invocation (single profiled call — see `_profile`).
            _sync_device()
            h0 = _host_ns()
            # The post-bracket sync runs in `finally` so an exception from the
            # candidate still drains the device — otherwise its in-flight async
            # work would leak into the next shape in this child.
            try:
                with bracket:
                    out, diag = _profile(fn, inp)
            finally:
                _sync_device()
            h1 = _host_ns()
            rec["kernel_event_present"] = diag.get("kernel_event_present")
            rec["stream_nondefault"] = diag.get("stream_nondefault")
            rec["fallback_signal"] = diag.get("fallback_signal")
            rec["backend_active"] = diag.get("backend_active")
            rec["npu_prof_raw_dir"] = diag.get("npu_prof_raw_dir")
            flat, err = _flatten(out)
        else:
            _sync_device()
            h0 = _host_ns()
            try:
                with bracket:
                    out = fn(inp)
            finally:
                _sync_device()
            h1 = _host_ns()
            flat, err = _flatten(out)
        if err is not None:
            rec["error"] = err
            return
        # G5: the profiled trial's host/event time includes profiler setup /
        # export / parse overhead, so it is NOT a clean latency sample — exclude
        # it from the timing series (t_npu median + event cross-check). Its
        # anti-cheat signals + sampled output are still recorded above, and the
        # candidate cannot tell which trial is profiled, so this is invisible to
        # it. N>1 measured trials leave a clean series; a 1-trial smoke yields no
        # timing sample, which is correct (you cannot cleanly time the profiled
        # call).
        if t != profiled_t:
            durations.append(h1 - h0)
            # NPU event-timer cross-check — read off the SAME bracketed call
            # (None on the dev machine, where `timer` is None).
            if timer is not None:
                try:
                    event_ns.append(timer.elapsed_ns())
                except Exception:  # noqa: BLE001 — cross-check, never fatal.
                    event_ns.append(None)
            else:
                event_ns.append(None)
        if keys0 is None:
            keys0 = list(flat.keys())
        if t in sampled_idx:
            for k, v in flat.items():
                sampled[f"{t}::{k}"] = v

    rec["durations"] = durations
    # All-None / empty series ⇒ no event timer on this host (dev machine).
    rec["event_ns"] = event_ns if any(e is not None for e in event_ns) else None
    rec["keys"] = keys0 or []
    rec["ran"] = True
    if sampled:
        sampled_ref = sid + ".sampled.safetensors"
        save_file(sampled, str(out_dir / sampled_ref))
        rec["sampled_outputs"] = sampled_ref


if args.get("stream_fd"):
    _stream_child(globals())
    sys.exit(0)

for shape in args["shapes"]:
    sid = shape["shape_id"]
    rec = _new_rec()
    if fn is None:
        rec["error"] = "no entrypoint"
        status["shapes"][sid] = rec
        continue
    try:
        if pass_kind == "timing":
            _timing_shape(fn, shape, rec)
        else:
            _correctness_shape(fn, shape, rec)
    except Exception as e:  # noqa: BLE001
        rec["error"] = f"{type(e).__name__}: {e}"
    status["shapes"][sid] = rec

Path(args["status_json"]).write_text(json.dumps(status))
'''


def _materialise_inputs(
    built: dict[str, object], dst_dir: Path, ref: str,
) -> tuple[str, str]:
    """Write a built inputs dict as a safetensors payload + JSON sidecar.

    Returns ``(tensors_path, sidecar_path)`` as absolute strings.
    """
    tensors, sidecar = _inputs.split_inputs(built)
    tensors_path = dst_dir / f"{ref}.input.safetensors"
    sidecar_path = dst_dir / f"{ref}.input.json"
    # safetensors refuses an empty dict; an all-scalar inputs block is legal.
    if tensors:
        save_file(tensors, str(tensors_path))
    else:
        save_file({"__empty__": torch.zeros(0)}, str(tensors_path))
    sidecar_path.write_text(json.dumps(sidecar))
    return str(tensors_path), str(sidecar_path)


def sanitised_shapes_view(shapes_doc: dict) -> dict:
    """The candidate-safe projection of a task's ``shapes.yaml``.

    The trusted runner holds the *complete* ``shapes.yaml`` (public AND
    hidden shapes). Any candidate-facing shapes view must drop the
    ``hidden_shapes`` section — it is the sealed verified-mode eval set
    (DESIGN §3.3 / §5.2) and must never reach the candidate.

    This returns a shallow copy of ``shapes_doc`` with ``hidden_shapes``
    removed; ``size_knobs`` / ``public_shapes`` / ``inputs`` are kept
    verbatim. The candidate child (see :func:`run_candidate_isolated`) is
    actually handed *materialised inputs* and no shapes document at all,
    so it never even sees this view — this helper is the explicit,
    testable sanitiser for any path that does need to expose a public
    shapes view (e.g. a debug bundle), and the contract is the same: the
    hidden section is gone.

    The advisory ``explicit_view`` (Phase-1 readable annotation) is a FLAT
    ``{id: ...}`` map that carries BOTH public and hidden rows, so dropping
    ``hidden_shapes`` alone would leak the hidden shapes through it. Project
    it down to public ids only, keeping the public readability while honouring
    the hidden-is-gone contract.

    U8: delegates to the single dual-format sealing in
    :func:`ascendbench.input_spec.shape_format.sanitised_view`, which applies
    the same knob-format sealing here AND seals an explicit-format doc's
    ``hidden`` row list -- one implementation, no drift.
    """
    from ascendbench.input_spec.shape_format import sanitised_view
    return sanitised_view(shapes_doc)


def run_candidate_isolated(
    solution_path: Path,
    materialised_inputs: dict[str, object],
    *,
    timeout_s: int = DEFAULT_PASS_TIMEOUT_S,
    pass_kind: str = "correctness",
    n_warmup: int = DEFAULT_N_WARMUP,
    sample_size: int = DEFAULT_SAMPLE_SIZE,
    sampled_indices: dict[str, list[int]] | None = None,
    backend: str = "npu",
) -> CandidateRunResult:
    """Spawn the candidate in an isolated subprocess; return its per-shape
    outputs (loaded from safetensors). Never executes candidate code in
    this (trusted) process.

    The runner (this trusted parent) materialises every shape's inputs and
    passes them in via ``materialised_inputs``. The child gets ONLY the
    sanitised minimal set — ``solution.py`` plus the injected inputs — it
    never sees ``repo_root`` / the task dir / the shapes document / the
    oracle.

    Parameters
    ----------
    solution_path:
        Path to the candidate ``solution.py``.
    materialised_inputs:
        Inputs the runner has already built via ``inputs.build_inputs``.
        Two shapes are accepted:

        * **correctness pass** — ``{shape_id: built_inputs_dict}``; one
          inputs dict is injected per shape.
        * **timing pass** — ``{shape_id: [t0_dict, t1_dict, ...]}``; a
          per-trial sequence is injected (``t=0..N-1``). The runner is
          responsible for building each trial's dict with its own
          ``meas_seed + t`` so every trial is a fresh input. The leading
          ``n_warmup`` entries are consumed as warm-up trials (negative
          indices on the wire) and are NOT timed.
    timeout_s:
        Wall-clock budget for the child; on expiry it is SIGKILLed.
    pass_kind:
        ``"correctness"`` (one call/shape, output stored) or ``"timing"``
        (trusted-shim warm-up + measured per-trial loop, host-timed). A
        timing pass requires per-trial inputs.
    n_warmup:
        Timing pass only — leading per-trial entries used for warm-up.
    sample_size:
        Timing pass only — number of measured trials whose outputs the
        child sends up as ``sampled_outputs`` for parent re-verification.
    sampled_indices:
        Timing pass only — explicit ``{shape_id: [trial_idx, ...]}`` map
        of which measured trials to sample. When ``None`` the runner picks
        them with a secret RNG (see :func:`run_two_pass`).
    """
    solution_path = Path(solution_path).resolve()
    if not materialised_inputs:
        return CandidateRunResult(False, False, "no inputs materialised",
                                  pass_kind=pass_kind)

    # Detect mode from the value type; both shapes share one code path.
    sample = next(iter(materialised_inputs.values()))
    per_trial = isinstance(sample, list)
    if pass_kind == "timing" and not per_trial:
        return CandidateRunResult(
            False, False, "timing pass requires per-trial inputs",
            pass_kind=pass_kind)

    # `prof_root` is the NPU profiler trace root: a parent-owned dir OUTSIDE the
    # candidate cwd (`td`), so the trust boundary for the raw trace is the
    # parent's, not the candidate's (G1 parse-in-parent). Auto-cleaned on exit;
    # passed to the NPU child via the AB_NPU_PROF_ROOT env below. (Created for
    # every backend; only wired into the env + parsed for backend == "npu".)
    with tempfile.TemporaryDirectory(prefix="tab-prof-") as prof_root, \
            tempfile.TemporaryDirectory(prefix="tab-cand-") as td:
        tdp = Path(td)
        child_py = tdp / "_child.py"
        child_py.write_text(_CHILD)
        status_json = tdp / "status.json"
        in_dir = tdp / "inputs"
        in_dir.mkdir()

        shapes_arg: list[dict] = []
        n_trials = 1
        for sid, payload in materialised_inputs.items():
            if per_trial:
                trials = []
                if pass_kind == "timing":
                    # Leading n_warmup entries -> negative trial indices
                    # (warm-up, not timed); the rest -> measured trials.
                    n_warm = min(n_warmup, max(len(payload) - 1, 0))
                    for i, built in enumerate(payload):
                        t = i - n_warm
                        tp, sp = _materialise_inputs(
                            built, in_dir, f"{sid}.t{t}")
                        trials.append({"t": t, "input_tensors": tp,
                                       "input_sidecar": sp})
                else:
                    for t, built in enumerate(payload):
                        tp, sp = _materialise_inputs(
                            built, in_dir, f"{sid}.t{t}")
                        trials.append({"t": t, "input_tensors": tp,
                                       "input_sidecar": sp})
                n_trials = max(n_trials, len(trials))
                shapes_arg.append({"shape_id": sid, "trials": trials})
            else:
                tp, sp = _materialise_inputs(payload, in_dir, sid)
                shapes_arg.append({"shape_id": sid, "input_tensors": tp,
                                   "input_sidecar": sp})

        args = {
            "protocol_version": PROTOCOL_VERSION,
            "status_schema_version": STATUS_SCHEMA_VERSION,
            "pass_kind": pass_kind,
            "solution": str(solution_path),
            "out_dir": str(tdp),
            "status_json": str(status_json),
            "inputs_mode": "per_trial" if per_trial else "single",
            "n_trials": n_trials,
            "n_warmup": n_warmup,
            "sample_size": sample_size,
            "sampled_indices": sampled_indices or {},
            "backend": backend,
            "shapes": shapes_arg,
        }
        # Prepend REPO_ROOT to the child's PYTHONPATH so its trusted-shim
        # segment imports the `ascendbench` helpers (Layer E audit hook + NPU
        # profiler / event timer) from this checkout — a plain checkout has
        # nothing on `sys.path`, and a stale site-package install would
        # otherwise shadow the branch under test. Any pre-existing PYTHONPATH
        # is preserved after REPO_ROOT so the checkout wins.
        #
        # This does not loosen candidate isolation: PYTHONPATH governs what the
        # trusted child *process* can import; the Layer E audit hook + static
        # AST whitelist govern what the *candidate* may do (they reject
        # importing the oracle / tasks or walking the repo) — two separate
        # layers.
        _existing_pp = os.environ.get("PYTHONPATH", "")
        child_pythonpath = (
            f"{REPO_ROOT}{os.pathsep}{_existing_pp}"
            if _existing_pp else str(REPO_ROOT))
        env = {
            "PATH": os.environ.get("PATH", ""),
            "HOME": os.environ.get("HOME", ""),
            # No network creds; minimal env. (Full seccomp/cgroup is the
            # GPU-machine layer — see module docstring.)
            "PYTHONHASHSEED": "0",
            "PYTHONDONTWRITEBYTECODE": "1",
            # trusted-shim import path (REPO_ROOT first).
            "PYTHONPATH": child_pythonpath,
        }
        from ascendbench.integrity.streaming import device_environment
        env.update(device_environment(backend))
        for name in ("home", "scratch", "cache"):
            (tdp / name).mkdir()
        env.update(HOME=str(tdp / "home"), TMPDIR=str(tdp / "scratch"),
                   TRITON_CACHE_DIR=str(tdp / "cache"))
        # On NPU the trusted child must load torch_npu + the CANN/driver
        # runtime, which need the Ascend environment (set on the parent via
        # set_env.sh). Forward a FIXED, value-AUDITED allowlist of CANN/driver
        # vars from the parent (_audited_npu_env: only absolute, existing,
        # non-world-writable dirs under a trusted CANN/system root — never
        # repo/cwd/tmp). This governs only the trusted child PROCESS's library
        # loading, NOT candidate capability (the candidate stays under the
        # Layer E audit hook + the static AST whitelist, both unchanged).
        # PATH / PYTHONPATH already carry the parent's CANN additions above.
        if backend == "npu":
            env.update(_audited_npu_env((
                "LD_LIBRARY_PATH", "ASCEND_TOOLKIT_HOME", "ASCEND_HOME_PATH",
                "ASCEND_OPP_PATH", "ASCEND_AICPU_PATH", "CMAKE_PREFIX_PATH",
                "TOOLCHAIN_HOME",
            )))
            # G1 parse-in-parent: tell the child where to RECORD the profiler
            # raw trace (a parent-owned dir outside the candidate cwd). The child
            # cannot parse it (the CSV export's exec is blocked by the graded
            # sandbox); the parent parses each per-shape raw dir below.
            env["AB_NPU_PROF_ROOT"] = prof_root
        # spawn via Popen so we can SIGKILL the whole process group
        # on timeout — `subprocess.run` only waits/kills the direct child,
        # leaving any forked descendant alive to leak state into the next
        # pass. `start_new_session=True` makes the child a process-group
        # leader; its PGID == its PID, and `os.killpg(pgid, SIGKILL)`
        # reaps every descendant.
        timed_out = False
        # Pass args via a file, not argv: the per-trial `shapes` payload scales
        # with n_trials (100 per spec) and overflows ARG_MAX on argv.
        args_path = tdp / "_args.json"
        args_path.write_text(json.dumps(args))
        proc = subprocess.Popen(
            [sys.executable, str(child_py), str(args_path)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=td,
            env=env,
            start_new_session=True,  # own process group → group-killable
        )
        try:
            pgid = os.getpgid(proc.pid)
        except (ProcessLookupError, OSError):
            # Child exited before we could read its PGID — its own PID is
            # the group id by construction (start_new_session=True).
            pgid = proc.pid
        try:
            proc.communicate(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            timed_out = True
            # SIGKILL the entire process group, not just the direct child.
            kill_process_group(pgid)
            try:
                proc.communicate(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
        # Whether or not the child timed out, SIGKILL the whole group
        # before the temp dir is torn down: a fast-exiting child may have
        # left forked descendants behind that would otherwise survive into
        # the next pass and leak compile-cache / device state. killpg on a
        # group with no live members is a harmless no-op.
        kill_process_group(pgid)

        if not status_json.exists():
            return CandidateRunResult(
                False, timed_out,
                "candidate produced no status (crash or timeout)",
                pass_kind=pass_kind)

        status = json.loads(status_json.read_text())
        res = CandidateRunResult(
            ran_any=False, timed_out=timed_out,
            load_error=status.get("load_error"),
            schema_version=int(status.get("schema_version",
                                          STATUS_SCHEMA_VERSION)),
            pass_kind=pass_kind)
        for sid in materialised_inputs:
            rec = status.get("shapes", {}).get(sid, {})
            _ev = rec.get("event_ns")
            scr = ShapeCandidateResult(
                shape_id=sid,
                ran=bool(rec.get("ran")),
                error=rec.get("error"),
                durations=list(rec.get("durations") or []),
                event_ns=(list(_ev) if _ev else None),
                kernel_event_present=rec.get("kernel_event_present"),
                stream_nondefault=rec.get("stream_nondefault"),
                fallback_signal=rec.get("fallback_signal"),
                backend_active=rec.get("backend_active"),
                npu_prof_raw_dir=rec.get("npu_prof_raw_dir"),
            )
            # G1 parse-in-parent: the child DEFERRED kernel_event_present /
            # fallback_signal (None) because its profiler CSV parse is blocked by
            # the graded sandbox. Here in the trusted parent (no hook) we parse
            # the raw trace dir it recorded and OVERWRITE those signals. A None
            # result (no raw dir / escape / parse anomaly) stays None -> G4 fails
            # closed on a real device. The dev stub (no raw dir) is left as-is.
            if backend == "npu" and scr.npu_prof_raw_dir:
                ke, fb = _parse_npu_kernel_event(scr.npu_prof_raw_dir, prof_root)
                scr.kernel_event_present = ke
                scr.fallback_signal = fb
            sampled_ref = rec.get("sampled_outputs")
            if sampled_ref:
                sref = tdp / str(sampled_ref)
                if sref.exists():
                    scr.sampled_outputs = load_file(str(sref))
            if scr.ran and pass_kind != "timing":
                st_file = tdp / f"{sid}.safetensors"
                if st_file.exists():
                    scr.outputs = load_file(str(st_file))
                    res.ran_any = True
                else:
                    scr.ran = False
                    scr.error = "status.ran but safetensors missing"
            elif scr.ran:
                # Timing pass: no per-shape output file; the sampled
                # outputs (if any) carry the data the parent re-verifies.
                res.ran_any = True
            res.per_shape[sid] = scr
        return res


# Generous bound on the trusted-parent NPU profiler parse (normal ~3 s on the
# 910B); a corrupted raw trace that wedges the CANN parser past this is treated
# as a parse failure (unverified -> fail-closed) rather than hanging the runner.
_ANALYSE_TIMEOUT_S = 120


def _run_with_timeout(fn, timeout_s):
    """Run ``fn()`` in a daemon thread; return True iff it completed without an
    exception within ``timeout_s``. A hung ``fn`` leaves the daemon thread
    abandoned (it cannot block process exit) and returns False, so a wedged CANN
    parser cannot hang the runner — the caller fails closed.

    A timed-out parse leaks its abandoned thread + any CANN subprocesses it
    forked until the runner exits (a killpg boundary would be airtight but
    re-imports torch_npu ~10 s per parse — prohibitive at ~2 parses x hundreds
    of shapes). Acceptable: the candidate's FS-write primitives are statically
    blocked (Layers A-D/F), so the raw trace is runner-produced and a hang needs
    a corrupted trace; the verdict still fails closed."""
    done = {"ok": False}

    def _wrap():
        try:
            fn()
            done["ok"] = True
        except Exception:
            done["ok"] = False

    th = threading.Thread(target=_wrap, daemon=True)
    th.start()
    th.join(timeout_s)
    return done["ok"] and not th.is_alive()


def _parse_npu_kernel_event(raw_dir, prof_root, *, synchronous=False):
    """Trusted-PARENT offline parse of a child-recorded NPU profiler raw dir.

    The in-child CSV parse is blocked by the graded sandbox (the profiler's CSV
    export spawns parser subprocesses = `exec`, which Layer E blocks). This runs
    in the parent (no hook), where `torch_npu.profiler.profiler.analyse` can
    spawn its parsers. Returns ``(kernel_event_present, fallback_signal)``:

      * >= 1 kernel row  -> (True,  False)  a real device kernel ran
      * exactly 0 rows   -> (False, True)   confirmed empty-kernel / CPU fallback
      * any anomaly      -> (None,  None)   -> G4 fails closed on a real device

    Anti-forge: ``raw_dir`` MUST resolve UNDER ``prof_root`` (a parent-owned root
    the candidate cannot legitimately reach); there must be exactly one
    ``*_ascend_pt`` trace dir; any pre-existing ``kernel_details.csv`` is deleted
    and MUST be regenerated by THIS parent ``analyse`` call — so a candidate that
    forged the easy CSV (rather than the hard raw CANN binary) gains nothing.
    Never raises; any anomaly returns ``(None, None)`` (unverified -> fail-closed).
    """
    unverified = (None, None)
    if not raw_dir or not prof_root:
        return unverified
    try:
        rd = os.path.realpath(raw_dir)
        root = os.path.realpath(prof_root)
        # rd must be STRICTLY under the trusted root: not the root itself, not
        # escaped (realpath already resolved any symlink/`..`).
        if (rd == root or not os.path.isdir(rd)
                or os.path.commonpath([rd, root]) != root):
            return unverified
        asc = [d for d in glob.glob(os.path.join(rd, "*_ascend_pt"))
               if os.path.isdir(d)]
        if len(asc) != 1:
            return unverified  # zero or multiple trace dirs -> not trustworthy
        # The single trace dir must be a REAL subdir of rd — reject a symlink
        # that would redirect the parent's parse / CSV write outside the
        # parent-owned tree.
        trace = asc[0]
        if os.path.islink(trace):
            return unverified
        trace = os.path.realpath(trace)
        if os.path.commonpath([trace, rd]) != rd:
            return unverified
        kd = os.path.join(trace, "ASCEND_PROFILER_OUTPUT", "kernel_details.csv")
        # No symlink ANYWHERE from the validated trace dir down to the CSV: an
        # intermediate `ASCEND_PROFILER_OUTPUT -> /external` symlink would
        # redirect the delete/read outside the parent-owned tree. realpath
        # resolves every component (incl. a not-yet-existing tail), so requiring
        # realpath(kd) == kd rejects any intermediate-dir or file symlink; the
        # resolved path must still be under rd.
        if (os.path.realpath(kd) != kd
                or os.path.commonpath([os.path.realpath(kd), rd]) != rd):
            return unverified
        # Distrust any child-side CSV — require the parent to regenerate it fresh.
        # (Any symlink at kd was already rejected above, so this removes a real
        # pre-existing file only.)
        if os.path.exists(kd):
            os.remove(kd)
        from torch_npu.profiler.profiler import analyse  # parent-only; no hook
        # Bound the parse: a corrupted raw trace could make CANN parsing hang —
        # a wedged parser must not hang the runner. A hung/raised parse ->
        # unverified -> G4 fails closed.
        if synchronous:
            # Streaming runs this in a separately killable trusted parser group.
            analyse(rd)
        elif not _run_with_timeout(lambda: analyse(rd), _ANALYSE_TIMEOUT_S):
            return unverified
        # Re-validate the WHOLE path right before reading (defense-in-depth: an
        # out-of-model writer could TOCTOU-swap an intermediate component to a
        # symlink between the pre-delete check and here). Require a real,
        # non-symlink CSV freshly produced by THIS parent parse, still under rd.
        if (not os.path.isfile(kd) or os.path.islink(kd)
                or os.path.realpath(kd) != kd
                or os.path.commonpath([os.path.realpath(kd), rd]) != rd):
            return unverified
        with open(kd, newline="") as fh:
            rows = sum(1 for _ in csv.DictReader(fh))
        return (True, False) if rows > 0 else (False, True)
    except Exception:  # any parse anomaly -> unverified -> fail-closed at G4.
        return unverified


def scrub_between_passes() -> list[str]:
    """Compatibility hook: pass-owned resources have already been reclaimed.

    Both entrypoints now give each child a private HOME, scratch and compile
    cache and kill its process group before removing them. Deleting global
    cache paths or touching the parent's device allocator here would interfere
    with other workers and is neither necessary nor safe.
    """
    return []


def _fresh_sampling_seed() -> int:
    """A fresh, unpredictable sampling seed for the runner-secret trial
    selection — generated in the trusted parent for every evaluation.

    The candidate cannot predict which measured trials get output-reverified
    (and which one is profiled) only if the seed driving that choice is
    unpredictable. This seed must therefore be:

      * **fresh per evaluation** — never reused across runs;
      * **drawn from a CSPRNG** — ``secrets`` (stdlib), not a fixed literal
        or a seed derivable from anything the candidate sees;
      * **strictly distinct from the input-generation seed** — the inputs
        are built with the deterministic ``meas_seed + t`` family so they
        are reproducible by the trusted parent; sharing that value here
        would let a submitter pre-compute the sampled set from the input
        seed. The two seed spaces are kept fully separate.

    Never exposed on the wire / in repo-visible logic: only the *resulting*
    ``sampled_indices`` map crosses the IPC, and that is computed parent-side.
    """
    # 63-bit positive int — fits torch.Generator.manual_seed and leaves
    # the sign bit clear.
    return secrets.randbits(63)


def run_two_pass(
    solution_path: Path,
    correctness_inputs: dict[str, object],
    timing_inputs: dict[str, object],
    *,
    timeout_s: int = DEFAULT_PASS_TIMEOUT_S,
    n_warmup: int = DEFAULT_N_WARMUP,
    sample_size: int = DEFAULT_SAMPLE_SIZE,
    golden_fixed_inputs: dict[str, object] | None = None,
    backend: str = "npu",
    result_consumer=None,
) -> TwoPassResult:
    """Dual-sandbox driver — two independent candidate subprocesses.

    1. **Correctness pass** — a fresh subprocess over ``correctness_inputs``
       (one inputs dict per shape). The subprocess is killed on exit.
    2. **Cleanup** — kill the correctness-pass process group and delete
       its owned directories. Each pass has private HOME, compile cache,
       scratch and profiler directories. Shared global paths are never
       deleted; device allocations disappear with their owning process.
    3. **Timing pass** — a *separate* fresh subprocess over
       ``timing_inputs`` (a per-trial sequence per shape). A runner-secret
       RNG picks which measured trials' outputs are sampled and sent up.

    Each child runs in its own process group (``start_new_session=True``);
    a hung child has its whole group SIGKILLed on timeout. The two
    passes never share interpreter / device / cache state — this is what
    the process-isolation model "Why Two Candidate Processes" requires.

    The default InputSource path builds one trial at a time and requires an
    immediate result consumer. Legacy `correctness_inputs` is
    ``{shape_id: built_dict}``; `timing_inputs` is
    ``{shape_id: [t0_dict, ...]}`` whose leading ``n_warmup`` entries are
    warm-up trials.

    The sampled-trial RNG seed is generated fresh per call from a CSPRNG and is
    strictly distinct from the deterministic input-generation seed — no
    caller-supplied / hard-coded seed feeds it, so a submitter cannot
    pre-compute which trials are reverified or profiled.

    ``golden_fixed_inputs`` (``{shape_id: built_dict}``) is the
    offline-golden cheat-path closure for ``upstream_oracle_required``
    tasks graded in verified mode. Such a task has no per-trial PyTorch
    oracle to re-run, so a runner-secret *sampled* timing trial cannot be
    reverified against fresh inputs — and skipping the check entirely
    reopens the correctness-pass→timing-pass cheat (a candidate fast-but-
    wrong only in the timed pass). When ``golden_fixed_inputs`` is given,
    the runner OVERWRITES every runner-secret sampled measured trial's
    input with that shape's fixed input (the seed-0 input the offline
    golden covers). Secret indices do not hide repeated seed-0 content:
    offline sampled inputs remain recognizable and cacheable. Non-sampled
    trials keep fresh per-trial inputs. The caller reverifies sampled outputs
    against the offline golden. ``None`` (the in-process-oracle path)
    leaves ``timing_inputs`` untouched.
    """
    from ascendbench.integrity.streaming import InputSource, run_streaming_two_pass
    if isinstance(correctness_inputs, InputSource):
        return run_streaming_two_pass(
            solution_path, correctness_inputs, result_consumer,
            timeout_s=timeout_s, sample_size=sample_size, backend=backend)

    # Pass 1 — correctness sandbox.
    correctness = run_candidate_isolated(
        solution_path, correctness_inputs,
        timeout_s=timeout_s, pass_kind="correctness", backend=backend)

    # Each isolated call already kills its group and deletes owned directories.
    scrubbed = scrub_between_passes()

    # Pass 2 — timing sandbox. The sampled-trial indices come from a
    # fresh CSPRNG seed the candidate cannot predict.
    rng = torch.Generator()
    rng.manual_seed(_fresh_sampling_seed())
    sampled_indices: dict[str, list[int]] = {}
    for sid, payload in timing_inputs.items():
        if not isinstance(payload, list):
            continue
        n_warm = min(n_warmup, max(len(payload) - 1, 0))
        n_meas = len(payload) - n_warm
        if n_meas <= 0:
            sampled_indices[sid] = []
            continue
        k = min(sample_size, n_meas)
        perm = torch.randperm(n_meas, generator=rng).tolist()
        sampled_indices[sid] = sorted(perm[:k])

    # For an offline-golden task, swap each runner-secret sampled measured
    # trial's input to the shape's fixed (golden-covered) input so the
    # shipped-up sampled output can be reverified against the offline golden.
    # Done after the secret indices are chosen and only on the sampled slots —
    # the candidate cannot distinguish them, and the non-sampled trials keep
    # fresh inputs (anti-cache).
    if golden_fixed_inputs:
        timing_inputs = _apply_golden_fixed_inputs(
            timing_inputs, sampled_indices, golden_fixed_inputs, n_warmup)

    timing = run_candidate_isolated(
        solution_path, timing_inputs,
        timeout_s=timeout_s, pass_kind="timing",
        n_warmup=n_warmup, sample_size=sample_size,
        sampled_indices=sampled_indices, backend=backend)

    return TwoPassResult(correctness=correctness, timing=timing,
                         scrubbed=scrubbed)


def _apply_golden_fixed_inputs(
    timing_inputs: dict[str, object],
    sampled_indices: dict[str, list[int]],
    golden_fixed_inputs: dict[str, object],
    n_warmup: int,
) -> dict[str, object]:
    """Overwrite the runner-secret sampled measured trials with each
    shape's fixed golden-covered input.

    For an ``upstream_oracle_required`` task graded in verified mode the
    only golden is the offline ``golden/<shape_id>.safetensors`` artifact,
    captured on the seed-0 input. To reverify a timed-pass output against
    that golden the *sampled* timed trial must run on exactly that input.

    A measured-trial position ``i`` in a shape's ``timing_inputs`` list
    maps to measured index ``i - n_warm`` (the leading ``n_warm`` entries
    are warm-up trials, negative wire indices — see
    :func:`run_candidate_isolated`). For every measured index in
    ``sampled_indices[sid]`` this replaces that slot with
    ``golden_fixed_inputs[sid]``. The list is shallow-copied so the
    caller's ``timing_inputs`` is not mutated; warm-up trials and the
    non-sampled measured trials keep their fresh per-trial inputs.
    """
    out: dict[str, object] = {}
    for sid, payload in timing_inputs.items():
        if not isinstance(payload, list) or sid not in golden_fixed_inputs:
            out[sid] = payload
            continue
        fixed = golden_fixed_inputs[sid]
        n_warm = min(n_warmup, max(len(payload) - 1, 0))
        sampled = set(sampled_indices.get(sid, []))
        new_payload = list(payload)
        for i in range(len(new_payload)):
            meas_idx = i - n_warm
            if meas_idx >= 0 and meas_idx in sampled:
                new_payload[i] = fixed
        out[sid] = new_payload
    return out


def kill_process_group(pgid: int) -> None:
    """Best-effort SIGKILL of a candidate process group (spec §SIGKILL)."""
    try:
        os.killpg(pgid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass
