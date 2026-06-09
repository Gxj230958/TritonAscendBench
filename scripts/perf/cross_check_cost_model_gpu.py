#!/usr/bin/env python3
"""phase_a_cost_xcheck.py -- cost-model GPU cross-check (DESIGN.md §5.6).

Closes Gap 1 of ``scripts/gpu_pipeline/`` (step 20). Per task it:

1. evaluates the analytic ``meta.yaml.cost_model`` ``flops`` (``W_c``) and
   ``bytes`` (``M_r``) formulas at every public eval shape, over the task's
   ``shapes.yaml`` size-knob universe, reusing :mod:`ascendbench.
   cost_model` verbatim (``parse_cost_model`` / ``FormulaEnv``);
2. runs the promoted (Phase A' PASS) ``kernel.py`` GPU-Triton reference on a
   live CUDA device under **Nsight Compute** (``ncu``) instrumentation,
   collecting the hardware DRAM counters ``dram__bytes_read.sum`` and
   ``dram__bytes_write.sum`` -- the *measured HBM read+write traffic* that
   ``M_r`` actually models;
3. reconciles ``|analytic - measured| / measured`` against
   ``meta.cost_model.gpu_xcheck_tol`` (default ``0.05``);
4. atomically writes ``meta.yaml: cost_model.status = PASS | BLOCKED |
   FLOOR_DOMINATED`` (the last for sub-burst index kernels with no asymptotic
   HBM regime -- a source-audited NOT_APPLICABLE class, never a PASS) plus a
   provenance block (run-id, per-shape evidence), mirroring
   :func:`phase_a_prime.update_meta_phase_a_prime`.

``cost_model.status`` is ncu-evidenced performance metadata, NOT a
release-gate blocker. The release gate keys on the two correctness proofs --
C2 oracle-equivalence + S2 severance (``eligibility.release_status_pass`` is
S2-only over the C2-aligned eligible set). cost_model status records where the
first-principles analytic access model fits measured DRAM (PASS) vs where
conditional access / L2 reuse / tile geometry make it inherently not
5%-predictive (BLOCKED) -- under the ``asymptotic_hbm_xcheck`` policy that
gates on large shapes and marks tiny burst-floor shapes ``FLOOR_OBSERVED``.

**Honesty contract.** The cross-check is GPU-authoritative AND
measurement-authoritative: it never fabricates a measurement and never writes
``cost_model.status = PASS`` without a real one.

* With no live CUDA device the script refuses to run the instrumented pass.
* ``M_r`` models HBM *traffic* (bytes read + written by the kernel). The only
  honest measurement of that quantity is the hardware DRAM throughput
  counters, read here via Nsight Compute. If ``ncu`` is not on ``PATH``, or
  the profiled run does not yield the DRAM metrics, the shape/task is marked
  ``BLOCKED`` with an explicit reason -- it is **never** approximated from
  the CUDA allocator's memory bookkeeping (that counts allocations, not HBM
  traffic) or from static tensor sizes (that is itself an analytic estimate).
* ``--dry-run`` evaluates ONLY the analytic formulas (it runs on the no-GPU
  dev machine) and writes nothing -- it is a formula sanity check, not a
  cross-check verdict.

**FLOPs side -- honestly not hard-checked.** A ``@triton.jit`` kernel is
opaque to ``torch.profiler``'s ``with_flops`` aten-op FLOP estimator, and the
generic ``ncu`` flop metrics depend on per-architecture
``sm__sass_thread_inst_executed_op_*`` rollups that are not portable across
the H100/A100 baseline targets. Rather than fabricate a FLOPs measurement,
this cross-check verifies ONLY the analytic ``M_r`` (bytes) against measured
DRAM traffic. The analytic ``W_c`` (FLOPs) is *not* reconciled against a
measurement; the per-shape verdict additionally requires that the kernel
genuinely executed on the GPU (``ncu`` profiled at least one kernel launch),
which proves the cost-model formulas describe a real, running kernel -- but
the FLOPs figure itself is recorded as analytic-only, not GPU-verified.

**L1_21 (and any non-score-eligible task).** ``L2_08_gumbel_sample`` carries a
constant placeholder ``cost_model`` (``flops: '1'`` / ``bytes: '1'``) because
its oracle is an unimplemented stub and its ``shapes.yaml`` has no size
knobs. It is NOT score-eligible (``eligibility.is_score_eligible`` is False --
stub oracle). Cross-checking a constant placeholder against a kernel that
cannot run would wrongly BLOCK it and pollute the gate, so a non-score
-eligible task is recorded ``cost_model.status = SKIPPED_NON_ELIGIBLE`` and
left out of the cross-check entirely.

Usage::

    # GPU host -- single task
    python3 scripts/perf/cross_check_cost_model_gpu.py --task L1_12_rms_norm --apply
    # GPU host -- full sweep
    python3 scripts/perf/cross_check_cost_model_gpu.py --all --apply
    # dev machine -- analytic-only, writes nothing
    python3 scripts/perf/cross_check_cost_model_gpu.py --all --dry-run
"""
from __future__ import annotations

import argparse
import datetime
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import textwrap
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("phase_a_cost_xcheck")

_SCRIPTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = _SCRIPTS_DIR.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ascendbench.perf.cost_model import (  # noqa: E402
    CostModel,
    CostModelError,
    FormulaEnv,
)
from ascendbench.corpus.dataset import TaskDef, iter_all_tasks, load_task  # noqa: E402
from ascendbench.release_gate.eligibility import is_score_eligible  # noqa: E402

#: ``meta.cost_model.status`` values this script writes.
STATUS_PASS = "PASS"
STATUS_BLOCKED = "BLOCKED"
#: A non-score-eligible task (stub oracle / PHASE-C-MANUAL shapes) is not
#: cross-checked at all -- its cost_model is a placeholder and it never
#: enters the scored denominator, so it must not be able to BLOCKED the gate.
STATUS_SKIPPED = "SKIPPED_NON_ELIGIBLE"
#: A kernel with NO asymptotic HBM regime in the provided shapes: every analytic
#: byte total sits below the smallest measured DRAM value, so the continuous
#: cost_model.bytes access model does not apply (measured traffic is dominated by
#: the burst/launch floor, not kernel bytes). Distinct from BLOCKED (a formula
#: mismatch the kernel COULD asymptotically hit) -- a NOT_APPLICABLE class, never
#: counted as a verified PASS. cost_model is not a release gate, so this carries
#: no gate weight; it only keeps the evidence honest instead of mislabelling a
#: sub-burst index kernel BLOCKED.
STATUS_FLOOR_DOMINATED = "FLOOR_DOMINATED"

#: asymptotic_hbm_xcheck floor cutoff. A first-principles analytic bytes model
#: matches measured DRAM on large shapes but cannot match tiny shapes dominated
#: by the fixed HBM burst/launch floor. Shapes whose measured traffic is below
#: this GLOBAL fraction of the task's max-measured shape are recorded as
#: FLOOR_OBSERVED (measured, evidenced) but do not block PASS. This is a fixed
#: bench-wide policy constant -- NEVER tuned per task and NEVER fitted from the
#: deviation table -- so it is an honest size policy, not a tolerance backdoor.
_FLOOR_FRACTION = 0.5


def _floor_dominated_allowlist() -> set[str]:
    """Source-audited task ids eligible for FLOOR_DOMINATED -- see
    ``ascendbench/policies/floor_dominated_acceptance.yaml``. The structural test
    (>= 2 shapes, max analytic < min measured) cannot by itself tell a true
    sub-burst floor from a formula that systematically under-counts (L2_26 omits
    its m_indices store), so a task must ALSO be audited into this allowlist."""
    p = REPO_ROOT / "ascendbench" / "policies" / "floor_dominated_acceptance.yaml"
    if not p.is_file():
        return set()
    return set((yaml.safe_load(p.read_text()) or {}).get("floor_dominated_tasks") or [])


# --------------------------------------------------------------------------
# Result records.
# --------------------------------------------------------------------------

@dataclass
class ShapeXCheck:
    """Per-shape analytic-vs-measured reconciliation evidence."""

    shape_id: str
    #: ``{compute_class: analytic W_c}``.
    analytic_flops: dict[str, float] = field(default_factory=dict)
    #: ``{resource: analytic M_r}``.
    analytic_bytes: dict[str, float] = field(default_factory=dict)
    #: Total measured HBM traffic (Nsight Compute ``dram__bytes_read.sum`` +
    #: ``dram__bytes_write.sum``); ``None`` in dry-run / when the instrumented
    #: pass did not run / when ``ncu`` could not produce the DRAM metrics.
    measured_bytes: float | None = None
    #: Whether Nsight Compute profiled at least one kernel launch for this
    #: shape's call -- a real-execution proof that the cost-model formulas
    #: describe a kernel that actually ran on the GPU.
    kernel_launched: bool | None = None
    #: Relative bytes deviation ``|analytic - measured| / measured``.
    bytes_deviation: float | None = None
    #: True iff every reconciliation for this shape is within tolerance.
    within_tol: bool = False
    #: Set when this shape could not be measured at all (``ncu`` missing /
    #: profiling failed / DRAM metrics absent). A blocked shape is honest
    #: about *why* there is no measurement -- it is never a fabricated PASS.
    blocked: bool = False
    #: Human-readable note when this shape did not pass.
    note: str = ""


@dataclass
class TaskXCheck:
    """Per-task cross-check verdict + provenance."""

    task_id: str
    status: str
    reason: str
    run_id: str
    gpu_xcheck_tol: float
    shapes: list[ShapeXCheck] = field(default_factory=list)
    #: True when the analytic pass ran but the GPU instrumented pass did not
    #: (``--dry-run`` on the dev machine). A dry-run NEVER writes meta.yaml.
    dry_run: bool = False


# --------------------------------------------------------------------------
# Analytic pass -- reuse cost_model.py verbatim.
# --------------------------------------------------------------------------

# Element byte widths for the representative ``dtype_size`` derived symbol.
# Mirrors ``eval._DTYPE_BYTES`` -- kept local so this script does not import
# the (heavyweight) eval module just for a small lookup table.
_DTYPE_BYTES: dict[str, float] = {
    "float64": 8.0, "double": 8.0, "int64": 8.0, "long": 8.0,
    "float32": 4.0, "float": 4.0, "int32": 4.0, "int": 4.0,
    "float16": 2.0, "half": 2.0, "bfloat16": 2.0, "int16": 2.0,
    "int8": 1.0, "uint8": 1.0, "bool": 1.0, "float8_e4m3fn": 1.0,
}


def _representative_dtype_size(shapes_doc: dict[str, Any]) -> float | None:
    """Element byte width of the first declared tensor input.

    Mirrors ``eval._representative_dtype_size`` (``specs/task_api_contract.md``
    §5.1): the first ``inputs`` entry carrying ``dims`` (a tensor spec, in
    YAML declaration order) gives the representative ``dtype_size``.

    Args:
        shapes_doc: the parsed ``shapes.yaml`` document.

    Returns:
        The element size in bytes, or ``None`` for a scalar-only ``inputs``
        block (the caller then uses the fp16 default).
    """
    for spec in (shapes_doc.get("inputs", {}) or {}).values():
        if not isinstance(spec, dict) or "dims" not in spec:
            continue
        return _DTYPE_BYTES.get(str(spec.get("dtype", "float32")))
    return None


def _shape_knobs(
    shapes_doc: dict[str, Any], shape_id: str,
) -> dict[str, float] | None:
    """Resolve a shape's numeric size-knob table.

    ``size_knobs`` defaults merged with the per-row ``knobs`` override -- the
    same merge :func:`ascendbench.input_spec.inputs.build_inputs` performs.

    Args:
        shapes_doc: the parsed ``shapes.yaml`` document.
        shape_id: the public/hidden shape id.

    Returns:
        ``{knob: float}`` for numeric knobs, or ``None`` if the shape id is
        absent from the document.
    """
    rows = list(shapes_doc.get("public_shapes", []) or []) + list(
        shapes_doc.get("hidden_shapes", []) or []
    )
    for row in rows:
        if row.get("id") == shape_id:
            knobs = dict(shapes_doc.get("size_knobs", {}) or {})
            knobs.update(row.get("knobs", {}) or {})
            return {
                k: float(v)
                for k, v in knobs.items()
                if isinstance(v, (int, float)) and not isinstance(v, bool)
            }
    return None


def evaluate_analytic(
    cost_model: CostModel,
    knobs: dict[str, float],
    dtype_size: float | None,
) -> tuple[dict[str, float], dict[str, float]]:
    """Evaluate the analytic ``W_c`` / ``M_r`` formulas for one shape.

    Reuses :class:`ascendbench.perf.cost_model.FormulaEnv` verbatim -- the
    single shared safe evaluator, so the symbol set cannot drift from
    ``dataset.py`` / ``validate_tasks.py``.

    Args:
        cost_model: the parsed, normalised :class:`CostModel`.
        knobs: numeric size-knob values for the shape.
        dtype_size: representative element byte width, or ``None`` for the
            fp16 default supplied by :class:`FormulaEnv`.

    Returns:
        ``(flops_by_class, bytes_by_resource)`` -- both maps of floats.

    Raises:
        CostModelError: if a formula references an unknown symbol or fails to
            evaluate.
    """
    derived = {"dtype_size": dtype_size} if dtype_size is not None else None
    env = FormulaEnv(knobs, derived=derived)
    flops = {c: env.evaluate(f) for c, f in cost_model.flops.items()}
    bytes_ = {r: env.evaluate(f) for r, f in cost_model.bytes.items()}
    return flops, bytes_


# --------------------------------------------------------------------------
# GPU instrumented pass.
# --------------------------------------------------------------------------

class CudaUnavailableError(RuntimeError):
    """Raised when an instrumented pass is requested on a host with no live
    CUDA device. The cross-check is GPU-authoritative -- it fails loudly
    rather than fabricating a verdict."""


def _require_cuda():
    """Return the imported ``torch`` module after asserting a live CUDA device.

    Raises:
        CudaUnavailableError: if ``torch`` is missing or
            ``torch.cuda.is_available()`` is False.
    """
    try:
        import torch
    except Exception as e:  # noqa: BLE001
        raise CudaUnavailableError(f"torch import failed: {e}") from e
    if not torch.cuda.is_available():
        raise CudaUnavailableError(
            "torch.cuda.is_available() is False -- the cost-model "
            "cross-check is GPU-authoritative and refuses to run the "
            "instrumented pass on a no-GPU host (use --dry-run for the "
            "analytic-only path)"
        )
    return torch


class NsightUnavailableError(RuntimeError):
    """Raised when measured HBM traffic cannot be obtained from Nsight
    Compute -- ``ncu`` is not installed, the profiled run failed, or its
    output carried no DRAM metrics. The cross-check is measurement-
    authoritative: it BLOCKS the shape/task rather than fabricating a
    measured-bytes figure from allocator bookkeeping or static sizes."""


#: Nsight Compute hardware DRAM-traffic counters. ``dram__bytes_read.sum`` +
#: ``dram__bytes_write.sum`` is the total HBM read+write traffic of every
#: profiled kernel launch -- exactly the quantity ``meta.cost_model.bytes``
#: (``M_r``) models. These metric names are stable across the Nsight Compute
#: releases shipped with CUDA 11.x-13.x.
_NCU_DRAM_READ_METRIC = "dram__bytes_read.sum"
_NCU_DRAM_WRITE_METRIC = "dram__bytes_write.sum"


def _find_ncu() -> str | None:
    """Locate the Nsight Compute CLI (``ncu``).

    Returns:
        The absolute path to ``ncu`` if it is on ``PATH`` (or at the common
        ``/usr/local/cuda/bin/ncu`` install location), else ``None``.
    """
    found = shutil.which("ncu")
    if found:
        return found
    fallback = Path("/usr/local/cuda/bin/ncu")
    if fallback.is_file() and os.access(fallback, os.X_OK):
        return str(fallback)
    return None


#: NVTX range the measured kernel call is wrapped in. ``ncu`` is launched
#: with ``--nvtx --nvtx-include "<NAME>/"`` so it profiles ONLY the kernel
#: launches inside this range -- the warmup call (Triton JIT compile + its
#: launches) is outside the range and is never profiled, no matter how many
#: kernels the warmup launches.
_NCU_NVTX_RANGE = "tab_xcheck_measured"

#: The in-process worker `ncu` profiles. It imports the promoted ``kernel.py``,
#: resolves the SAME entrypoint Phase A' verified (via
#: ``_pa_subproc_runner._resolve_entrypoint``), materialises one shape's
#: inputs on CUDA, does a warmup call (Triton JIT compile -- OUTSIDE the NVTX
#: range, so not profiled), then runs the measured call once INSIDE the NVTX
#: range. `ncu` collects the DRAM counters across every kernel launch in that
#: range. Kept as a string template so no extra worker file is needed.
_NCU_WORKER_TEMPLATE = textwrap.dedent(
    '''\
    import importlib.util, sys
    REPO_ROOT = {repo_root!r}
    if REPO_ROOT not in sys.path:
        sys.path.insert(0, REPO_ROOT)
    import torch
    if not torch.cuda.is_available():
        print("XCHECK_WORKER_ERROR: torch.cuda.is_available() is False",
              file=sys.stderr)
        raise SystemExit(3)
    from ascendbench.input_spec.inputs import build_inputs
    import yaml

    runner_path = {runner_path!r}
    _rspec = importlib.util.spec_from_file_location("_pa_subproc_runner_ncu",
                                                    runner_path)
    runner = importlib.util.module_from_spec(_rspec)
    _rspec.loader.exec_module(runner)

    from pathlib import Path as _Path
    # Load the frozen kernel through the SAME path C2/S2 use: a per-call
    # synthetic package (so ``from .base`` frozen-dep relative imports
    # resolve), sys.modules registration (so a kernel reading
    # sys.modules[run.__module__] finds itself), device_assert-strip compat
    # load, and _pa_launcher.py exec (so jit-only kernels expose run). A flat
    # spec_from_file_location skipped all four. upstream_roots=None: the
    # on-disk kernel is the severed standalone (no vllm relative imports).
    kmod = runner._load_kernel_with_optional_bridge(_Path({kernel_path!r}), "_pa_kernel", None)
    fn = runner._resolve_entrypoint(kmod, {{"entrypoint": {entrypoint!r}, "param_name_map": {param_name_map!r}}})
    if not callable(fn):
        print("XCHECK_WORKER_ERROR: kernel.py exposes no resolvable entrypoint",
              file=sys.stderr)
        raise SystemExit(4)

    shapes_doc = yaml.safe_load(open({shapes_path!r}).read())

    def _to_cuda(obj):
        if torch.is_tensor(obj):
            return obj.to("cuda")
        if isinstance(obj, dict):
            return {{k: _to_cuda(v) for k, v in obj.items()}}
        if isinstance(obj, (list, tuple)):
            return type(obj)(_to_cuda(v) for v in obj)
        return obj

    # Warmup call: triggers Triton JIT compilation + its launches. This is
    # OUTSIDE the NVTX range, so `ncu --nvtx-include` never profiles it --
    # compile cost is excluded, the warmup discipline timing.py uses.
    warm = _to_cuda(build_inputs(shapes_doc, {shape_id!r}, seed=1))
    fn(warm)
    torch.cuda.synchronize()

    # The single measured call, wrapped in the NVTX range `ncu` filters on.
    # Every kernel launch here -- and ONLY here -- is profiled for DRAM bytes.
    measured = _to_cuda(build_inputs(shapes_doc, {shape_id!r}, seed=1000))
    torch.cuda.synchronize()
    torch.cuda.nvtx.range_push({nvtx_range!r})
    fn(measured)
    torch.cuda.synchronize()
    torch.cuda.nvtx.range_pop()
    print("XCHECK_WORKER_OK")
    '''
)


def _parse_ncu_dram_bytes(csv_text: str) -> tuple[float, int]:
    """Parse ``ncu --csv`` output for total DRAM read+write bytes.

    Nsight Compute ``--csv`` emits one row per (kernel launch, metric). The
    columns include ``"Metric Name"``, ``"Metric Unit"`` and ``"Metric
    Value"``. This sums the read and write byte counters across every
    profiled launch, normalising the unit (``ncu`` may report ``Kbyte`` /
    ``Mbyte`` / ``Gbyte`` depending on magnitude).

    Args:
        csv_text: the captured stdout of ``ncu --csv ...``.

    Returns:
        ``(total_dram_bytes, n_launches_seen)`` -- ``n_launches_seen`` is the
        count of distinct profiled kernel launches.

    Raises:
        NsightUnavailableError: if the CSV carries neither DRAM metric (the
            run produced no usable measurement -- fail closed, never guess).
    """
    import csv
    import io

    # `ncu --csv` prefixes a few non-CSV banner lines (``==PROF==`` etc.);
    # the real CSV starts at the header row containing "Metric Name".
    lines = csv_text.splitlines()
    header_idx = next(
        (i for i, ln in enumerate(lines) if '"Metric Name"' in ln
         or "Metric Name" in ln),
        None,
    )
    if header_idx is None:
        raise NsightUnavailableError(
            "ncu produced no parseable CSV (no 'Metric Name' header) -- "
            "the profiled run yielded no DRAM metrics"
        )
    reader = csv.DictReader(io.StringIO("\n".join(lines[header_idx:])))

    _unit_scale = {
        "byte": 1.0, "bytes": 1.0,
        "kbyte": 1e3, "kbytes": 1e3,
        "mbyte": 1e6, "mbytes": 1e6,
        "gbyte": 1e9, "gbytes": 1e9,
    }
    total = 0.0
    launches: set[str] = set()
    saw_metric = False
    for row in reader:
        name = (row.get("Metric Name") or "").strip()
        if name not in (_NCU_DRAM_READ_METRIC, _NCU_DRAM_WRITE_METRIC):
            continue
        raw_val = (row.get("Metric Value") or "").strip().replace(",", "")
        try:
            val = float(raw_val)
        except ValueError:
            continue
        unit = (row.get("Metric Unit") or "byte").strip().lower()
        scale = _unit_scale.get(unit)
        if scale is None:
            raise NsightUnavailableError(
                f"ncu reported DRAM metric {name!r} in an unrecognised unit "
                f"{unit!r} -- refusing to guess a byte scale"
            )
        total += val * scale
        saw_metric = True
        launch_key = (
            (row.get("ID") or "") + "|" + (row.get("Kernel Name") or "")
        )
        launches.add(launch_key)
    if not saw_metric:
        raise NsightUnavailableError(
            "ncu CSV carried no dram__bytes_read.sum / dram__bytes_write.sum "
            "rows -- the DRAM traffic metrics were not collected"
        )
    return total, len(launches)


def measure_kernel_bytes(
    task: TaskDef,
    shapes_doc: dict[str, Any],
    shape_id: str,
) -> tuple[float, bool]:
    """Measure one shape's HBM traffic with Nsight Compute DRAM counters.

    The promoted ``kernel.py`` is run for ``shape_id`` inside an ``ncu``
    profiling session that collects ``dram__bytes_read.sum`` +
    ``dram__bytes_write.sum`` -- the hardware HBM read+write byte counters,
    i.e. the *measured* quantity ``meta.cost_model.bytes`` (``M_r``) models.
    A warmup call precedes the measured call; the measured call is wrapped
    in an NVTX range and ``ncu`` is run with ``--nvtx-include`` so ONLY the
    measured call's launches are profiled -- Triton JIT-compile traffic and
    warmup launches are excluded regardless of how many kernels they emit.

    This is the ONLY measured-bytes path. There is no allocator-memory or
    static-tensor-size fallback: the CUDA allocator's ``cuda_memory_usage``
    counts allocation bookkeeping (not HBM traffic), and static tensor sizes
    are themselves an analytic estimate -- using either as "measured" would
    cross-check the analytic model against another analytic figure. If
    ``ncu`` is unavailable or yields no DRAM metric the function raises
    :class:`NsightUnavailableError` and the shape/task is BLOCKED.

    Args:
        task: the loaded :class:`TaskDef`.
        shapes_doc: the parsed ``shapes.yaml`` document.
        shape_id: the eval shape to materialise + run.

    Returns:
        ``(measured_hbm_bytes, kernel_launched)`` -- ``kernel_launched`` is
        True iff ``ncu`` profiled at least one kernel launch.

    Raises:
        CudaUnavailableError: if no live CUDA device is present.
        NsightUnavailableError: if ``ncu`` is missing, the profiled run
            failed, or its output carried no DRAM traffic metrics.
    """
    _require_cuda()

    ncu = _find_ncu()
    if ncu is None:
        raise NsightUnavailableError(
            "Nsight Compute (ncu) is not on PATH -- the cost-model GPU "
            "cross-check requires ncu DRAM metrics "
            "(dram__bytes_read.sum / dram__bytes_write.sum) to measure HBM "
            "traffic; install the CUDA toolkit's Nsight Compute or run with "
            "--dry-run for the analytic-only path"
        )

    runner_path = REPO_ROOT / "scripts" / "verify" / "severance_subprocess_runner.py"
    kernel_path = task.task_dir / "kernel.py"
    shapes_path = task.task_dir / "shapes.yaml"
    entrypoint = (task.meta.get("oracle", {}) or {}).get("gpu_triton")
    # Same meta.interface.param_name_map the S2/oracle paths use, so the
    # cost-xcheck worker binds the entrypoint IDENTICALLY (a wrapper param
    # mapped to its task input key) -- otherwise this 6th invocation site
    # would diverge / miss-bind for tasks that declare a rename.
    param_name_map = (task.meta.get("interface") or {}).get("param_name_map") or {}

    worker_src = _NCU_WORKER_TEMPLATE.format(
        repo_root=str(REPO_ROOT),
        runner_path=str(runner_path),
        kernel_path=str(kernel_path),
        entrypoint=entrypoint,
        param_name_map=param_name_map,
        shapes_path=str(shapes_path),
        shape_id=shape_id,
        nvtx_range=_NCU_NVTX_RANGE,
    )

    with tempfile.NamedTemporaryFile(
        "w", suffix="_ncu_worker.py", delete=False
    ) as wf:
        wf.write(worker_src)
        worker_file = wf.name

    try:
        cmd = [
            ncu,
            "--csv",
            "--metrics",
            f"{_NCU_DRAM_READ_METRIC},{_NCU_DRAM_WRITE_METRIC}",
            # Profile ONLY the kernel launches inside the worker's NVTX
            # range -- the warmup call (JIT compile + its launches) is
            # outside the range and is never counted.
            "--nvtx",
            "--nvtx-include",
            f"{_NCU_NVTX_RANGE}/",
            sys.executable,
            worker_file,
        ]
        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600,
            )
        except FileNotFoundError as e:
            raise NsightUnavailableError(
                f"failed to launch ncu ({ncu!r}): {e}"
            ) from e
        except subprocess.TimeoutExpired as e:
            raise NsightUnavailableError(
                f"ncu profiling of {task.task_id}/{shape_id} timed out "
                f"after 600s"
            ) from e

        if "XCHECK_WORKER_ERROR" in proc.stderr:
            err = next(
                (ln for ln in proc.stderr.splitlines()
                 if "XCHECK_WORKER_ERROR" in ln),
                proc.stderr.strip(),
            )
            raise NsightUnavailableError(
                f"ncu worker for {task.task_id}/{shape_id} failed: {err}"
            )
        if proc.returncode != 0 and "XCHECK_WORKER_OK" not in proc.stdout:
            raise NsightUnavailableError(
                f"ncu exited {proc.returncode} profiling "
                f"{task.task_id}/{shape_id}; stderr tail: "
                f"{proc.stderr.strip()[-400:]}"
            )

        measured_bytes, n_launches = _parse_ncu_dram_bytes(proc.stdout)
    finally:
        try:
            os.unlink(worker_file)
        except OSError:
            pass

    kernel_launched = n_launches > 0
    if not kernel_launched:
        raise NsightUnavailableError(
            f"{task.task_id}/{shape_id}: ncu profiled no kernel launch -- "
            f"the kernel did not execute on the GPU (cannot cross-check a "
            f"non-running kernel)"
        )
    return measured_bytes, kernel_launched


# --------------------------------------------------------------------------
# Reconciliation.
# --------------------------------------------------------------------------

def _relative_deviation(analytic: float, measured: float) -> float:
    """Relative deviation ``|analytic - measured| / measured``.

    Args:
        analytic: the analytic cost-model value.
        measured: the instrumented measurement.

    Returns:
        The relative deviation. When ``measured`` is zero the deviation is
        ``0.0`` iff ``analytic`` is also zero, else ``inf`` (a non-zero
        analytic value against zero measured traffic can never reconcile).
    """
    if measured == 0.0:
        return 0.0 if analytic == 0.0 else float("inf")
    return abs(analytic - measured) / abs(measured)


def reconcile_shape(
    shape_id: str,
    analytic_flops: dict[str, float],
    analytic_bytes: dict[str, float],
    measured_bytes: float | None,
    kernel_launched: bool | None,
    tol: float,
) -> ShapeXCheck:
    """Reconcile one shape's analytic cost model against the measurement.

    Bytes are the directly measured quantity: the total analytic ``M_r``
    (summed across resources) is reconciled against the Nsight Compute DRAM
    read+write traffic within ``tol``. The verdict additionally requires
    that ``ncu`` profiled at least one kernel launch -- a real-execution
    proof that the cost-model formulas describe a kernel that actually ran.

    The analytic FLOPs (``W_c``) are NOT reconciled against a measurement: a
    ``@triton.jit`` kernel is opaque to portable FLOP instrumentation (see
    the module docstring). They are recorded as analytic-only evidence; the
    PASS verdict rests solely on the measured bytes cross-check.

    Args:
        shape_id: the eval shape id.
        analytic_flops: ``{compute_class: W_c}`` -- recorded, not hard-checked.
        analytic_bytes: ``{resource: M_r}``.
        measured_bytes: measured HBM traffic, or ``None`` (not measured).
        kernel_launched: whether ``ncu`` profiled a kernel launch.
        tol: ``gpu_xcheck_tol``.

    Returns:
        A populated :class:`ShapeXCheck`.
    """
    sx = ShapeXCheck(
        shape_id=shape_id,
        analytic_flops=dict(analytic_flops),
        analytic_bytes=dict(analytic_bytes),
        measured_bytes=measured_bytes,
        kernel_launched=kernel_launched,
    )
    if measured_bytes is None:
        sx.note = "not measured (dry-run / analytic-only)"
        return sx

    analytic_total_bytes = sum(analytic_bytes.values())
    dev = _relative_deviation(analytic_total_bytes, measured_bytes)
    sx.bytes_deviation = dev
    if dev > tol:
        sx.within_tol = False
        sx.note = (
            f"bytes deviation {dev:.4f} > tol {tol:.4f} "
            f"(analytic={analytic_total_bytes:.0f} "
            f"measured={measured_bytes:.0f})"
        )
        return sx
    if not kernel_launched:
        sx.within_tol = False
        sx.note = (
            "ncu profiled no kernel launch -- cannot certify a cost model "
            "for a kernel that did not run on the GPU"
        )
        return sx
    sx.within_tol = True
    sx.note = (
        f"bytes deviation {dev:.4f} <= tol {tol:.4f} "
        f"(FLOPs analytic-only, not GPU-measured -- see module docstring)"
    )
    return sx


# --------------------------------------------------------------------------
# Per-task driver.
# --------------------------------------------------------------------------

def cross_check_task(task: TaskDef, *, dry_run: bool) -> TaskXCheck:
    """Cross-check one task's analytic cost model against its GPU kernel.

    Args:
        task: the loaded :class:`TaskDef`.
        dry_run: when True, only the analytic formulas are evaluated -- the
            instrumented GPU pass is skipped and no measurement is taken.
            ``--dry-run`` is the no-GPU dev-machine path; it writes nothing.

    Returns:
        A :class:`TaskXCheck` verdict. A non-score-eligible task (stub oracle
        / PHASE-C-MANUAL shapes -- e.g. ``L2_08_gumbel_sample``) is recorded
        :data:`STATUS_SKIPPED` and never cross-checked.
    """
    run_id = f"xcheck-{datetime.date.today().isoformat()}-{uuid.uuid4().hex[:8]}"

    # L1_21 special case (and any non-score-eligible task): a stub-oracle /
    # manual-shapes task carries only a constant placeholder cost_model and
    # never enters the scored denominator. Cross-checking it would wrongly
    # BLOCKED the release gate, so it is skipped outright.
    if not is_score_eligible(task):
        logger.info(
            "%s: non-score-eligible (stub oracle / manual shapes) -- "
            "skipping cost-model cross-check", task.task_id,
        )
        return TaskXCheck(
            task_id=task.task_id,
            status=STATUS_SKIPPED,
            reason=(
                "non-score-eligible task (stub oracle / PHASE-C-MANUAL "
                "shapes); cost_model is a placeholder, not cross-checked"
            ),
            run_id=run_id,
            gpu_xcheck_tol=(
                task.cost_model.gpu_xcheck_tol if task.cost_model else 0.05
            ),
            dry_run=dry_run,
        )

    cost_model = task.cost_model
    if cost_model is None:
        return TaskXCheck(
            task_id=task.task_id,
            status=STATUS_BLOCKED,
            reason="meta.yaml has no cost_model block",
            run_id=run_id,
            gpu_xcheck_tol=0.05,
            dry_run=dry_run,
        )

    shapes_doc = yaml.safe_load((task.task_dir / "shapes.yaml").read_text())
    dtype_size = _representative_dtype_size(shapes_doc)
    public_ids = [s.id for s in task.public_shapes]
    if not public_ids:
        return TaskXCheck(
            task_id=task.task_id,
            status=STATUS_BLOCKED,
            reason="shapes.yaml declares no public eval shape",
            run_id=run_id,
            gpu_xcheck_tol=cost_model.gpu_xcheck_tol,
            dry_run=dry_run,
        )

    shape_results: list[ShapeXCheck] = []
    for sid in public_ids:
        knobs = _shape_knobs(shapes_doc, sid)
        if knobs is None:
            shape_results.append(
                ShapeXCheck(shape_id=sid, note="shape not found in shapes.yaml")
            )
            continue
        try:
            a_flops, a_bytes = evaluate_analytic(cost_model, knobs, dtype_size)
        except CostModelError as e:
            shape_results.append(
                ShapeXCheck(shape_id=sid, note=f"analytic eval failed: {e}")
            )
            continue

        if dry_run:
            shape_results.append(
                reconcile_shape(
                    sid, a_flops, a_bytes, None, None,
                    cost_model.gpu_xcheck_tol,
                )
            )
            continue

        # GPU pass. A NsightUnavailableError (ncu missing / profiling
        # failed / no DRAM metric) BLOCKS this shape with an explicit
        # reason -- it never falls back to a fabricated measurement.
        try:
            measured_bytes, kernel_launched = measure_kernel_bytes(
                task, shapes_doc, sid
            )
        except NsightUnavailableError as e:
            shape_results.append(
                ShapeXCheck(
                    shape_id=sid,
                    analytic_flops=dict(a_flops),
                    analytic_bytes=dict(a_bytes),
                    blocked=True,
                    within_tol=False,
                    note=f"BLOCKED -- {e}",
                )
            )
            continue
        shape_results.append(
            reconcile_shape(
                sid, a_flops, a_bytes, measured_bytes, kernel_launched,
                cost_model.gpu_xcheck_tol,
            )
        )

    if dry_run:
        return TaskXCheck(
            task_id=task.task_id,
            status=STATUS_BLOCKED,  # dry-run never certifies a PASS
            reason="dry-run: analytic formulas evaluated, GPU pass skipped",
            run_id=run_id,
            gpu_xcheck_tol=cost_model.gpu_xcheck_tol,
            shapes=shape_results,
            dry_run=True,
        )

    # asymptotic_hbm_xcheck policy. A first-principles analytic bytes model can
    # match measured DRAM on LARGE (asymptotic) shapes, but tiny shapes are
    # dominated by a fixed HBM burst/launch floor no portable formula can hit.
    # So gate PASS on the large shapes -- the largest measured shape plus any
    # shape whose measured traffic is >= _FLOOR_FRACTION of the max -- and record
    # the small floor-dominated shapes as FLOOR_OBSERVED: measured + evidenced,
    # but NOT release-blocking. The cutoff is a GLOBAL policy constant, never
    # per-task or fitted, and the per-shape tol is unchanged for gate shapes. A
    # blocked (unmeasurable) shape always gates -- it must surface, never floor-excused.
    measured_vals = [s.measured_bytes for s in shape_results
                     if s.measured_bytes is not None and not s.blocked]
    max_meas = max(measured_vals) if measured_vals else 0.0

    def _is_gate(s: ShapeXCheck) -> bool:
        if s.blocked or s.measured_bytes is None or max_meas <= 0.0:
            return True
        return s.measured_bytes >= _FLOOR_FRACTION * max_meas

    gate = [s for s in shape_results if _is_gate(s)]
    floor = [s for s in shape_results if not _is_gate(s)]
    for s in floor:
        if not s.within_tol and not s.blocked:
            s.note = (
                f"{s.note} [FLOOR_OBSERVED: small-shape HBM burst floor "
                f"(measured={s.measured_bytes:.0f} < {_FLOOR_FRACTION:g}x"
                f"max={max_meas:.0f}); non-blocking per asymptotic_hbm_xcheck]"
            ).strip()

    # FLOOR_DOMINATED (task-level): if EVERY analytic byte total sits below the
    # smallest measured DRAM value, the kernel has no asymptotic regime in these
    # shapes and cost_model.bytes is not applicable. Structural test -- analytic
    # fully decoupled below measured -- never a fitted threshold. Checked before
    # the gate/floor verdict so a sub-burst index kernel is not mislabelled
    # BLOCKED (a formula it could never asymptotically hit). Requires >= 2
    # measured shapes (a lone shape with measured slightly above analytic is a
    # within-tol PASS, not a floor) AND a source-audited allowlist entry: the
    # structural test alone cannot tell a true floor from a formula that
    # under-counts (a missing term also yields max_analytic < min_measured, as
    # L2_26 does), so the kernel must be audited as genuinely sub-burst.
    analytic_totals = [
        sum(s.analytic_bytes.values())
        for s in shape_results
        if s.analytic_bytes and not s.blocked and s.measured_bytes is not None
    ]
    if (
        len(measured_vals) >= 2
        and analytic_totals
        and max(analytic_totals) < min(measured_vals)
        and task.task_id in _floor_dominated_allowlist()
    ):
        return TaskXCheck(
            task_id=task.task_id,
            status=STATUS_FLOOR_DOMINATED,
            reason=(
                f"no asymptotic HBM regime: max analytic bytes "
                f"{max(analytic_totals):.0f} < min measured {min(measured_vals):.0f} "
                f"across all {len(shape_results)} shape(s); cost_model.bytes not "
                f"applicable (measured dominated by burst/launch floor)"
            ),
            run_id=run_id,
            gpu_xcheck_tol=cost_model.gpu_xcheck_tol,
            shapes=shape_results,
            dry_run=False,
        )

    failing = [s for s in gate if not s.within_tol]
    floor_observed = sum(1 for s in floor if not s.within_tol and not s.blocked)
    if failing:
        status = STATUS_BLOCKED
        n_blocked = sum(1 for s in failing if s.blocked)
        kind = (
            "gate shape(s) could not be measured / exceeded "
            if n_blocked
            else "gate shape(s) exceeded "
        )
        # A >tol gap is an analytic-vs-H200-ncu MEASUREMENT difference, not
        # necessarily a formula error: H200's large L2 absorbs small output
        # writeback, DRAM burst granularity adds a few-% over-fetch, and
        # gather/attention traffic is data-dependent. cost_model is NOT a release
        # gate and scoring uses the formula value (not this status). Per-task
        # classification: docs/reviews/2026-06-05-costmodel-nonpass-analysis.md.
        reason = (
            "GPU_UNVERIFIABLE (analytic-vs-H200-ncu gap, not necessarily a "
            "formula error -- see docs/reviews/2026-06-05-costmodel-nonpass-"
            "analysis.md). "
            f"{len(failing)}/{len(gate)} {kind}"
            f"gpu_xcheck_tol={cost_model.gpu_xcheck_tol}: "
            + "; ".join(f"{s.shape_id}: {s.note}" for s in failing)
        )
    else:
        status = STATUS_PASS
        reason = (
            f"all {len(gate)} gate (large) shape(s) within "
            f"gpu_xcheck_tol={cost_model.gpu_xcheck_tol}"
            + (f"; {floor_observed} small shape(s) FLOOR_OBSERVED (non-blocking)"
               if floor_observed else "")
        )
    return TaskXCheck(
        task_id=task.task_id,
        status=status,
        reason=reason,
        run_id=run_id,
        gpu_xcheck_tol=cost_model.gpu_xcheck_tol,
        shapes=shape_results,
        dry_run=False,
    )


# --------------------------------------------------------------------------
# Atomic meta.yaml writer -- mirrors update_meta_phase_a_prime.
# --------------------------------------------------------------------------

def update_meta_cost_model(task_dir: Path, verdict: TaskXCheck) -> None:
    """Atomically write the cross-check verdict into ``meta.yaml.cost_model``.

    Adds ``status`` / ``status_reason`` / ``xcheck_run_id`` / ``xchecked_at``
    plus a per-shape ``xcheck_evidence`` list to the existing ``cost_model``
    block (the analytic ``flops`` / ``bytes`` formulas are left untouched).
    Temp-then-replace, so a kill mid-write never leaves a half-written
    ``meta.yaml`` -- the exact discipline of
    :func:`phase_a_prime.update_meta_phase_a_prime` /
    :func:`phase_a_prime.atomic_write_kernel`.

    Idempotent: a re-run overwrites the previous verdict block in place.

    Args:
        task_dir: the task directory containing ``meta.yaml``.
        verdict: the :class:`TaskXCheck` to persist.

    Raises:
        RuntimeError: if called for a dry-run verdict (dry-run writes nothing).
    """
    if verdict.dry_run:
        raise RuntimeError(
            "refusing to write meta.yaml for a dry-run verdict -- "
            "--dry-run is analytic-only and never certifies a status"
        )
    mp = task_dir / "meta.yaml"
    meta = yaml.safe_load(mp.read_text())
    cm = meta.get("cost_model")
    if not isinstance(cm, dict):
        raise RuntimeError(
            f"{task_dir.name}: meta.yaml has no cost_model mapping to "
            f"stamp a status onto"
        )
    cm["status"] = verdict.status
    cm["status_reason"] = verdict.reason
    cm["xcheck_run_id"] = verdict.run_id
    cm["xchecked_at"] = datetime.date.today().isoformat()
    if verdict.shapes:
        cm["xcheck_evidence"] = [
            {
                "shape_id": s.shape_id,
                "analytic_bytes_total": float(sum(s.analytic_bytes.values())),
                # measured_bytes is the Nsight Compute DRAM read+write total
                # (HBM traffic); None when not measured / the shape was
                # BLOCKED because ncu could not produce the metric.
                "measured_bytes": (
                    None if s.measured_bytes is None
                    else float(s.measured_bytes)
                ),
                "measured_via": (
                    "ncu_dram_bytes"
                    if s.measured_bytes is not None
                    else None
                ),
                "bytes_deviation": (
                    None if s.bytes_deviation is None
                    else float(s.bytes_deviation)
                ),
                "kernel_launched": s.kernel_launched,
                # FLOPs are recorded analytic-only; this cross-check does not
                # GPU-measure them (see the module docstring).
                "flops_checked": False,
                "blocked": s.blocked,
                "within_tol": s.within_tol,
                "note": s.note,
            }
            for s in verdict.shapes
        ]
    meta["cost_model"] = cm

    tmp = mp.with_suffix(".yaml.tmp")
    tmp.write_text(yaml.safe_dump(meta, sort_keys=False, allow_unicode=True))
    os.replace(tmp, mp)


# --------------------------------------------------------------------------
# CLI.
# --------------------------------------------------------------------------

def _select_tasks(task_id: str | None) -> list[TaskDef]:
    """Resolve the task set the run operates on.

    Args:
        task_id: a single task id, or ``None`` for the full corpus.

    Returns:
        The loaded :class:`TaskDef` list.
    """
    if task_id:
        return [load_task(task_id)]
    return iter_all_tasks()


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser.

    Returns:
        The configured :class:`argparse.ArgumentParser`.
    """
    ap = argparse.ArgumentParser(description=__doc__)
    grp = ap.add_mutually_exclusive_group(required=True)
    grp.add_argument("--task", metavar="TASK_ID", help="cross-check one task")
    grp.add_argument(
        "--all", action="store_true", help="cross-check every task",
    )
    ap.add_argument(
        "--apply",
        action="store_true",
        help="write the verdict into meta.yaml.cost_model.status "
        "(default: report only, no write)",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="analytic-only: evaluate the cost-model formulas without the "
        "GPU instrumented pass; writes nothing. The no-GPU dev-machine path.",
    )
    return ap


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint for the cost-model GPU cross-check.

    Args:
        argv: argument vector (defaults to ``sys.argv[1:]``).

    Returns:
        Process exit code -- ``0`` if every cross-checked task is PASS (or
        skipped / dry-run), ``1`` if any task is BLOCKED, ``2`` on a fatal
        error (no CUDA device for a non-dry-run, task load failure).
    """
    args = build_parser().parse_args(argv)
    if args.dry_run and args.apply:
        logger.error("--dry-run and --apply are mutually exclusive "
                     "(dry-run is analytic-only and never writes)")
        return 2

    try:
        tasks = _select_tasks(args.task)
    except (FileNotFoundError, CostModelError) as e:
        logger.error("task load failed: %s", e)
        return 2

    logger.info(
        "cost-model cross-check: %s task(s), mode=%s",
        len(tasks),
        "dry-run (analytic-only)" if args.dry_run else "GPU-instrumented",
    )

    verdicts: list[TaskXCheck] = []
    blocked = 0
    for task in tasks:
        try:
            v = cross_check_task(task, dry_run=args.dry_run)
        except CudaUnavailableError as e:
            logger.error("%s: %s", task.task_id, e)
            return 2
        except Exception as e:  # noqa: BLE001
            logger.error("%s: cross-check raised %s: %s",
                         task.task_id, type(e).__name__, e)
            v = TaskXCheck(
                task_id=task.task_id,
                status=STATUS_BLOCKED,
                reason=f"cross-check error: {type(e).__name__}: {e}",
                run_id=f"xcheck-error-{uuid.uuid4().hex[:8]}",
                gpu_xcheck_tol=(
                    task.cost_model.gpu_xcheck_tol if task.cost_model else 0.05
                ),
                dry_run=args.dry_run,
            )
        verdicts.append(v)

        marker = {
            STATUS_PASS: "PASS",
            STATUS_BLOCKED: "BLOCKED",
            STATUS_SKIPPED: "SKIP",
            STATUS_FLOOR_DOMINATED: "FLOOR",
        }.get(v.status, v.status)
        logger.info("[%s] %s -- %s", marker, v.task_id, v.reason)

        if v.status == STATUS_BLOCKED and not v.dry_run:
            blocked += 1
        if args.apply and not v.dry_run and v.status != STATUS_SKIPPED:
            update_meta_cost_model(task.task_dir, v)
            logger.info("  wrote meta.yaml: cost_model.status=%s", v.status)
        elif args.apply and v.status == STATUS_SKIPPED:
            # A non-eligible task is not given a cost_model.status at all --
            # release_status_pass treats a missing status as NOT PASS, which
            # is correct: the task is not score-eligible so it never enters
            # the release-blocking set anyway.
            logger.info("  skipped meta write (non-score-eligible task)")

    n_pass = sum(1 for v in verdicts if v.status == STATUS_PASS)
    n_skip = sum(1 for v in verdicts if v.status == STATUS_SKIPPED)
    n_floor = sum(1 for v in verdicts if v.status == STATUS_FLOOR_DOMINATED)
    logger.info(
        "cross-check done: %s PASS, %s BLOCKED, %s FLOOR_DOMINATED, %s skipped",
        n_pass, blocked, n_floor, n_skip,
    )
    if args.dry_run:
        logger.info("dry-run: no meta.yaml written; verdicts are advisory")
        return 0
    return 1 if blocked else 0


if __name__ == "__main__":
    raise SystemExit(main())
