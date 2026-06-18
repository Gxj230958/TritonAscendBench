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
    `os.killpg`s the WHOLE group, so forked descendants cannot survive
    .
  * **Two sandboxes ** — a candidate is graded in two *independent*
    subprocess passes: a correctness pass then a timing pass. Between
    them the runner SIGKILLs the correctness pass's whole process group
    and scrubs `/tmp` scratch + the Triton compile cache (device memory
    too, on NPU). This cuts the correctness-pass → timing-pass
    pre-computation channel and stops a forked descendant from leaking
    state into the timing pass (DESIGN §3.3 / §6.6,
    the process-isolation model "Why Two Candidate Processes").
  * **No golden-output leak ** — the child receives ONLY
    a sanitised minimal set: `solution.py` + the inputs the runner
    materialised for it. It never sees the original `task_dir`,
    `oracle_pytorch.py`, `contract.md`, `golden/`, or the `shapes.yaml`
    document. Because the child is handed *materialised inputs* and not
    the shapes document at all, the `hidden_shapes` section is
    structurally absent from the candidate view — this is strictly
    stronger than the DESIGN §3.3 "strip the hidden_shapes section"
    requirement. The repo root IS on the child's `PYTHONPATH`  so
    its trusted-shim segment can import the in-repo helper modules; this
    leaks no golden data — the candidate cannot `open()` / walk the repo
    (Layer A/E denylist) and the AST whitelist rejects importing the
    oracle / tasks. When a sanitised public shapes view is needed
    elsewhere, :func:`sanitised_shapes_view` produces one with the
    `hidden_shapes` section removed.
  * **safetensors-only IPC ** — both directions of the IPC are
    data-only: inputs flow down as safetensors payloads + JSON sidecar
    manifests; outputs flow up the same way. No pickle → no RCE on load.
  * **Runner-materialised inputs ** — every shape's inputs (public
    AND hidden) are built by the trusted parent via `inputs.build_inputs`
    and injected down the IPC. The child does NOT call `build_inputs` and
    is not handed the shapes document; this both keeps hidden-shape
    definitions out of the candidate view and removes the need for the
    child to know the full `shapes_doc`.
  * **Host-authoritative timing ** — the timing pass times each
    measured trial with `CLOCK_MONOTONIC_RAW` inside a trusted-shim
    segment of the child, on a *fresh* per-trial inputs dict the runner
    injected. A runner-secret RNG — seeded fresh per evaluation from a
    CSPRNG, strictly distinct from the deterministic input-generation
    seed  — selects a subset of trials whose outputs are sent up
    (`sampled_outputs`) so the parent can re-verify them against golden,
    closing the correctness-pass cheat path. The  profiler
    diagnostics are bound to one of those runner-secret sampled measured
    trials , so a candidate cannot behave honestly on a profiled
    call while cheating the timed calls.
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
      "pass_kind": "correctness" | "timing",   #
      "solution":   str,              # abs path to solution.py
      "out_dir":    str,              # abs dir for output safetensors
      "status_json": str,             # abs path the child writes status to
      "inputs_mode": "single" | "per_trial",
      "n_trials":   int,              # per-trial sequence length; 1 for "single"
      #  timing-pass knobs (ignored for the correctness pass):
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
          # --- diagnostics ( carriers; populated by /) ---
          "durations": [float, ...],   # per-trial host_ns (timing pass)
          "event_ns": [float | None, ...] | None,  # : per-trial NPU
          #                              event-timer ns (timing pass; None
          #                              on the dev machine — host-only)
          "kernel_event_present": bool | None,
          "stream_nondefault": bool | None,
          "fallback_signal": bool | None,
          "sampled_outputs": str | None  # ref to sampled-output payload
        }, ...
      }
    }
----------------------------------------------------------------------
"""
from __future__ import annotations

import json
import logging
import os
import secrets
import shutil
import signal
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from ascendbench.input_spec import inputs as _inputs

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

#  frozen protocol versions. Bump when the wire schema changes;
# downstream W items pin against these.
PROTOCOL_VERSION = 2        # parent -> child args schema (v2: + backend)
# v2 : per-shape `event_ns` added — the timing-pass child's NPU
# event-timer cross-check series (None on the dev machine; real on the
# 910B). The parent feeds it to `timing.arbitrate_timing(event_ns=...)`.
STATUS_SCHEMA_VERSION = 2   # child -> parent status.json schema

# Timing-pass defaults aligned to docs/reference/timing_arbitration.md (10 warmup /
# 100 trials / 10 samples). Kept overridable so a dev-box smoke can drop to a
# cheaper loop.
DEFAULT_N_WARMUP = 10
DEFAULT_N_TRIALS = 100
DEFAULT_SAMPLE_SIZE = 10

# Triton-Ascend compile-cache / scratch paths scrubbed between the two
# passes so the timing pass cannot read a correctness-pass artifact.
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
    # ---  diagnostics (filled by  timing pass / ) ----------
    durations: list[float] = field(default_factory=list)
    # : per-trial NPU event-timer ns from the timing pass — the
    # host/event-drift cross-check series. `None` on the dev machine (no
    # NPU); the parent passes it to `timing.arbitrate_timing(event_ns=...)`.
    event_ns: list[float] | None = None
    kernel_event_present: bool | None = None
    stream_nondefault: bool | None = None
    fallback_signal: bool | None = None
    sampled_outputs: dict[str, torch.Tensor] | None = None


@dataclass
class CandidateRunResult:
    ran_any: bool
    timed_out: bool
    load_error: str | None
    per_shape: dict[str, ShapeCandidateResult] = field(default_factory=dict)
    # : schema version of the status.json this result was decoded from.
    schema_version: int = STATUS_SCHEMA_VERSION
    # : which pass produced this result ("correctness" | "timing").
    pass_kind: str = "correctness"


@dataclass
class TwoPassResult:
    """ — the pair of independent passes plus their isolation receipt.

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
#  — after the candidate module loads but before any warmup/graded call the
# child installs the Layer E `sys.addaudithook` runtime-defense hook (a brief
# "compile" phase around the triton warmup; "graded" thereafter), and once per
# shape it profiles one candidate call for the NPU kernel-event /
# non-default-stream / fallback diagnostics (kernel-event source stubbed on dev).
_CHILD = r'''
import contextlib, json, math, sys, time, importlib.util
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
    """: write a status.json carrying an explicit `load_error` and
    exit non-zero.

    Used when the trusted-shim segment cannot be set up (the in-repo
    `ascendbench` helper modules failed to import). This is NOT a
    candidate fault — it is a misconfigured child — so it is surfaced
    loudly via `load_error` rather than silently swallowed. The parent
    reads `load_error` off status.json and marks the run an error.
    """
    Path(args["status_json"]).write_text(json.dumps(
        {"schema_version": args["status_schema_version"],
         "shapes": {}, "load_error": load_error}))
    sys.exit(1)


# ── Trusted-shim segment  ──────────────────────────────────────────
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
# This does NOT relax candidate isolation. The candidate `solution.py` is
# still loaded under the Layer E audit hook + the static AST whitelist: it
# cannot fresh-import the oracle / tasks or walk the package tree. "The
# child process can import the package" and "the candidate can reach the
# repository" are two separate layers — the former is PYTHONPATH, the
# latter is the audit hook / AST checker, which are unchanged.

#  Layer E: the sys.addaudithook runtime-defense hook installer. It
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

#  child-side profiler: detect NPU kernel-event presence, non-default
# stream activity, and the derived fallback signal. The NPU measurement
# belongs to backends/npu.py; on the dev machine it is a documented stub.
#
#  / : the same backend exposes the timing event-timer seam — an
# `EventTimer` *context manager* whose NPU `Event.record()` calls bracket
# the SAME measured `fn(inp)` invocation that host time and the profiler
# see. `_BACKEND.event_timer()` returns a fresh timer on the 910B and
# `None` on the dev machine (no NPU); the timing pass degrades to
# host-only there. Either way the `event_ns` series is sent up so the
# parent's `arbitrate_timing` channel is end-to-end connected.
#
# : there is deliberately NO callable that re-runs the candidate to
# take a separate event sample — that executed every measured input
# twice and reopened the  stateful-candidate hole. The event timer
# only brackets a call the timing loop already makes.
#
# : this import is also a trusted-shim import — a failure is a
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

# NOTE: the Layer E audit hook is NOT installed here — it is installed
# below, AFTER the candidate `solution.py` module is loaded but BEFORE any
# graded `fn(inp)` invocation. CPython emits an `exec` audit event for the
# import system's module-code execution, and the hook blocks `exec`; an
# install before the candidate module load would block the load itself.
# The candidate module body already passed the static AST whitelist
# (Layers A-D/F); the audit hook is the runtime net for the *graded*
# `fn(inp)` calls — see the install site after the candidate load below.


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

    The candidate `fn(inp)` is invoked exactly once. That single call
    produces BOTH the output the caller consumes AND the three
    diagnostic signals — the profiler path and the graded/timed path are
    the same invocation. A stateful candidate therefore cannot behave
    honestly on a profiled call and cheat on a separate graded/timed call
    : there is no separate call.

    `_PROFILER.profile_call` runs `fn` once and hands back both `output`
    and the kernel-event / stream / fallback signals. When no profiler is
    importable (dev-machine without the package) `fn` is called directly
    here, once, and the signals stay None ("not measured").

    Returns ``(output, {"kernel_event_present", "stream_nondefault",
    "fallback_signal"})``.
    """
    if _PROFILER is None:
        output = fn(inp)
        return output, {"kernel_event_present": None,
                        "stream_nondefault": None,
                        "fallback_signal": None}
    res = _PROFILER.profile_call(fn, inp)
    diag = {"kernel_event_present": res.get("kernel_event_present"),
            "stream_nondefault": res.get("stream_nondefault"),
            "fallback_signal": res.get("fallback_signal")}
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

#  /  Layer E: install the sys.addaudithook runtime-defense hook NOW —
# after the candidate module body loaded (vetted source) but BEFORE any warmup
# or graded `fn(inp)` call. It defaults to the "graded" phase (everything
# blocked); the compile warmup below briefly flips it to the narrow "compile"
# phase. CPython emits an `exec` audit event for module-code execution, so
# installing earlier would have blocked the candidate's own load. The install
# is unconditional and not swallowed .
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
if (fn is not None and _BACKEND is not None
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

    The single profiled `fn(inp)` invocation produces both the
    diagnostics (stored on `rec`) AND the candidate output, which is
    returned here as ``(flat_outputs, error)`` so the caller consumes
    *this* call's result. The caller MUST NOT re-invoke `fn` on the same
    input — that would let a stateful candidate be honest on the diagnosed
    call and cheat on the separately-graded call .
    """
    output, diag = _profile(fn, inp)
    rec["kernel_event_present"] = diag.get("kernel_event_present")
    rec["stream_nondefault"] = diag.get("stream_nondefault")
    rec["fallback_signal"] = diag.get("fallback_signal")
    return _flatten(output)


def _correctness_shape(fn, shape, rec):
    """Correctness pass for one shape: one call, store t=0 output.

    The stored output is the output of the SAME call that was profiled
     — `_record_diag` profiles `fn(inp)` once and returns that
    call's output. The candidate is never invoked twice on one input, so
    the diagnostics and the graded correctness output describe the same
    invocation .
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

    : the  profiler diagnostics (kernel-event / non-default-stream
    / fallback) are bound to a REAL measured invocation — specifically a
    runner-secret sampled measured trial — not a sacrificial warm-up call.
    The candidate cannot predict which measured trials the runner sampled,
    so it cannot behave honestly on a profiled call and cheat on the
    timing calls: the profiled call IS one of the calls being timed and
    output-reverified.

     / : the profiled measured trial runs `fn` EXACTLY ONCE. That
    single invocation is host-timed, NPU-event-bracketed, profiled, and
    its return value is the output sampled / reverified for that trial —
    there is no second, untimed call. The NPU event timer is a
    *context manager* whose `Event.record()` calls bracket that same one
    invocation; the candidate is never re-run merely to take an event
    sample. A stateful candidate therefore cannot run honestly on the
    timed call and serve a different output on a re-run.
    """
    sid = shape["shape_id"]
    trials = {t["t"]: t for t in shape.get("trials", [])}
    if not trials:
        rec["error"] = "timing pass requires per_trial inputs"
        return
    sampled_idx = set(args.get("sampled_indices", {}).get(sid, []))

    # Phase a: warm-up — negative trial indices, NOT timed. Warm-up calls
    # are no longer profiled: a sacrificial warm-up call is exactly the
    # call a candidate could behave honestly on while cheating the timed
    # calls. The profiler now runs on a runner-secret measured trial in
    # phase b .
    warm_ts = sorted(k for k in trials if k < 0)
    for t in warm_ts:
        inp = _merge(trials[t]["input_tensors"], trials[t]["input_sidecar"])
        flat, err = _run_one(fn, inp)
        if err is not None:
            rec["error"] = err
            return

    # Phase b: measured loop — fresh inputs per trial, host-timed.
    #  / : each measured trial also records an NPU event-timer ns.
    # The event timer is an `EventTimer` context manager that BRACKETS the
    # SAME single `fn(inp)` call host time and the profiler see — it does
    # NOT re-run the candidate. On the dev machine `event_timer()` returns
    # None, so `event_ns` stays an all-None series and arbitration
    # degrades to host-only — but the channel up to the parent is
    # connected either way.
    durations = []
    event_ns = []
    sampled = {}
    keys0 = None
    meas_ts = sorted(k for k in trials if k >= 0)
    # : the profiler is bound to a runner-secret sampled measured
    # trial. Pick the first sampled measured index when the runner sampled
    # any; otherwise (no sampling configured) fall back to the first
    # measured trial so the  channel is still populated. Either way it
    # is a real measured invocation, not a warm-up call.
    profiled_t = None
    sampled_meas = sorted(t for t in meas_ts if t in sampled_idx)
    if sampled_meas:
        profiled_t = sampled_meas[0]
    elif meas_ts:
        profiled_t = meas_ts[0]
    for t in meas_ts:
        inp = _merge(trials[t]["input_tensors"], trials[t]["input_sidecar"])
        # : one NPU event timer per measured trial, bracketing the
        # ONE measured call. `None` on the dev machine ⇒ nullcontext, the
        # trial is host-only timed.
        timer = _make_event_timer()
        bracket = timer if timer is not None else contextlib.nullcontext()
        if t == profiled_t:
            # The profiled call IS the measured invocation: the profiler
            # runs `fn(inp)` exactly once and returns BOTH that call's
            # output and the  signals. Host-time AND event-bracket
            # that same single call so the trial is timed, profiled, and
            # output-sampled off one invocation. The candidate cannot tell
            # this trial apart from any other measured trial, and — because
            # there is no second call — cannot run honestly on the timed
            # call and cheat on a re-run that produces the sampled output
            # .
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
    """: the candidate-safe projection of a task's ``shapes.yaml``.

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
    timeout_s: int = 90,
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

    with tempfile.TemporaryDirectory(prefix="tab-cand-") as td:
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
        # : prepend REPO_ROOT to the child's PYTHONPATH so its
        # trusted-shim segment imports the `ascendbench` helpers
        # (Layer E audit hook + NPU profiler / event timer) from THIS
        # checkout — a plain checkout has nothing on `sys.path`, and a
        # stale site-package install would otherwise shadow the branch
        # under test. Any pre-existing PYTHONPATH is preserved AFTER
        # REPO_ROOT so the checkout wins.
        #
        # This does NOT loosen candidate isolation: the candidate
        # `solution.py` is still graded under the Layer E audit hook + the
        # static AST whitelist (which reject importing the oracle / tasks
        # or walking the repo). PYTHONPATH only governs what the trusted
        # child *process* can import; the audit hook / AST checker govern
        # what the *candidate* may do — two separate, unchanged layers.
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
            # : trusted-shim import path (REPO_ROOT first).
            "PYTHONPATH": child_pythonpath,
        }
        # : spawn via Popen so we can SIGKILL the WHOLE process group
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
            )
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


def scrub_between_passes() -> list[str]:
    """Clear the cross-pass leak channel between the two sandboxes.

    Removes the Triton-Ascend compile cache and `/tmp` scratch dirs that a
    correctness-pass child could have planted pre-computed answers in, and
    empties NPU device memory when an NPU is present. Returns the list of
    paths/resources actually scrubbed (the isolation receipt
    :class:`TwoPassResult` records and the tests assert on).

    Safe to call on the dev machine: the compile-cache / scratch dirs
    simply do not exist and are skipped; the NPU branch is a no-op without
    `torch.npu`.
    """
    scrubbed: list[str] = []
    for path in _SCRUB_PATHS:
        p = Path(path)
        if p.exists():
            try:
                shutil.rmtree(p, ignore_errors=True)
                scrubbed.append(path)
            except OSError as e:
                logger.warning("scrub failed for %s: %s", path, e)
    # Device memory — only meaningful on the NPU machine.
    npu = getattr(torch, "npu", None)
    if npu is not None and getattr(npu, "is_available", lambda: False)():
        try:
            npu.empty_cache()
            scrubbed.append("npu:device_memory")
        except Exception as e:
            logger.warning("npu.empty_cache failed: %s", e)
    return scrubbed


def _fresh_sampling_seed() -> int:
    """: a fresh, unpredictable sampling seed for the runner-secret
    trial selection — generated in the trusted parent for EVERY evaluation.

    The candidate cannot predict which measured trials get output-reverified
    (and, after , which one is profiled) only if the seed driving that
    choice is unpredictable. This seed MUST therefore be:

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
    timeout_s: int = 90,
    n_warmup: int = DEFAULT_N_WARMUP,
    sample_size: int = DEFAULT_SAMPLE_SIZE,
    golden_fixed_inputs: dict[str, object] | None = None,
    backend: str = "npu",
) -> TwoPassResult:
    """ dual-sandbox driver — two independent candidate subprocesses.

    1. **Correctness pass** — a fresh subprocess over ``correctness_inputs``
       (one inputs dict per shape). The subprocess is killed on exit.
    2. **Scrub** — :func:`scrub_between_passes` clears the Triton compile
       cache + `/tmp` scratch + NPU device memory so the timing pass
       cannot read anything the correctness pass left behind. The
       correctness-pass process group is SIGKILLed  before the
       scrub so no forked descendant survives into the timing pass.
    3. **Timing pass** — a *separate* fresh subprocess over
       ``timing_inputs`` (a per-trial sequence per shape). A runner-secret
       RNG picks which measured trials' outputs are sampled and sent up.

    Each child runs in its own process group (``start_new_session=True``);
    a hung child has its WHOLE group SIGKILLed on timeout . The two
    passes never share interpreter / device / cache state — this is the
    the process-isolation model "Why Two Candidate Processes"
    requires.

    `correctness_inputs` is ``{shape_id: built_dict}``; `timing_inputs` is
    ``{shape_id: [t0_dict, ...]}`` whose leading ``n_warmup`` entries are
    warm-up trials.

    : the sampled-trial RNG seed is generated FRESH per call from a
    CSPRNG and is strictly distinct from the deterministic input-generation
    seed — no caller-supplied / hard-coded seed feeds it, so a submitter
    cannot pre-compute which trials are reverified or profiled.

    : ``golden_fixed_inputs`` (``{shape_id: built_dict}``) is the
    offline-golden cheat-path closure for ``upstream_oracle_required``
    tasks graded in verified mode. Such a task has no per-trial PyTorch
    oracle to re-run, so a runner-secret *sampled* timing trial cannot be
    reverified against fresh inputs — and skipping the check entirely
    reopens the correctness-pass→timing-pass cheat (a candidate fast-but-
    wrong only in the timed pass). When ``golden_fixed_inputs`` is given,
    the runner OVERWRITES every runner-secret sampled measured trial's
    input with that shape's fixed input (the seed-0 input the offline
    golden covers). The candidate cannot tell a sampled trial apart from
    any other measured trial (the indices are runner-secret, ), so it
    cannot target the cheat at the non-sampled trials; the non-sampled
    trials keep their fresh per-trial inputs so the cache-reuse defence
    still holds. The caller then reverifies the shipped-up sampled outputs
    against the offline golden. ``None`` (the in-process-oracle path)
    leaves ``timing_inputs`` untouched.
    """
    # Pass 1 — correctness sandbox.
    correctness = run_candidate_isolated(
        solution_path, correctness_inputs,
        timeout_s=timeout_s, pass_kind="correctness", backend=backend)

    # Between passes — the correctness-pass process group was SIGKILLed in
    # `run_candidate_isolated` ; here we scrub the cross-pass
    # pre-computation channel so the timing pass starts clean.
    scrubbed = scrub_between_passes()

    # Pass 2 — timing sandbox. The sampled-trial indices come from a
    # fresh CSPRNG seed  the candidate cannot predict.
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

    # : for an offline-golden task, swap each runner-secret sampled
    # measured trial's input to the shape's fixed (golden-covered) input
    # so the shipped-up sampled output can be reverified against the
    # offline golden. Done AFTER the secret indices are chosen and only
    # on the sampled slots — the candidate cannot distinguish them, and
    # the non-sampled trials keep fresh inputs (anti-cache).
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
    """: overwrite the runner-secret sampled measured trials with each
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
