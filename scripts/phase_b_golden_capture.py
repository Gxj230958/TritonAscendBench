"""phase_b_golden_capture.py -- offline golden capture for
``upstream_oracle_required`` tasks (GPU-host Phase B').

DESIGN.md §4.3 / §5.7: a score-eligible task is one of three
``oracle_simplified`` tiers. The ``simple_pytorch_ok`` /
``formula_exact_manual`` majority (>=80%) has a PyTorch oracle whose
output IS the evaluation-time golden -- the runner re-runs it per shape
with zero pre-stored data. The remaining <=20% are
``upstream_oracle_required``: irreducible fused kernels (online softmax,
chunk-recurrent, ...) whose PyTorch oracle is KNOWN to deviate from the
frozen upstream kernel. Such a task has no provider golden in
``self-reported`` mode; in ``verified`` mode its golden is captured ONCE,
offline, on a CUDA GPU, by running the **promoted frozen upstream
kernel** over the task's COMPLETE eval shape set, and stored as::

    tasks/<level>/<id>/golden/<shape_id>.safetensors

``eval._resolve_golden`` and ``eligibility.golden_complete`` read exactly
this layout -- a flat ``{tensor_key: tensor}`` safetensors dict per
shape. ``scripts/validate_tasks.py::_validate_golden`` enforces it.

This script is the missing producer of those artifacts. It:

  1. enumerates the **score-eligible** ``upstream_oracle_required`` tasks
     (18 today) from trusted repo ``meta`` via
     :func:`eligibility.is_upstream_oracle_required` AND
     :func:`eligibility.is_score_eligible`. The score-eligible filter is
     the SAME policy ``gpu_pipeline`` step 20
     (``phase_a_cost_xcheck.py``) applies: a non-score-eligible task
     (stub oracle / ``PHASE-C-MANUAL`` shapes -- ``L1_21_gumbel_sample``
     today) has no offline golden to capture (its kernel is a
     placeholder and its oracle is an unimplemented stub, so it can
     never reach ``phase_a_prime.status == PASS``), and is an
     acknowledged non-score-eligible special case (Task #20) that must
     not block the full-corpus capture sweep;
  2. for each task enumerates ALL eval shape ids -- public AND sealed
     hidden -- via :func:`inputs.shape_ids` (the W2.1 split: hidden
     shapes are normally verified-only; here they are unsealed so their
     goldens can be captured, and the resulting artifacts stay sealed at
     the same confidentiality level as the hidden shapes themselves);
  3. materialises each shape's inputs with the verified-runner seed
     (``inputs.build_inputs(..., seed=0)`` -- byte-identical to the
     ``correctness_inputs`` ``eval.evaluate`` feeds the candidate, and
     to the seed-0 input the offline-golden cheat-path closure relies
     on, see ``eval._verify_sampled_outputs_offline``);
  4. runs the task's **promoted** standalone ``kernel.py::run`` on the
     GPU inside an isolated subprocess;
  5. writes the output dict to ``golden/<shape_id>.safetensors``
     (safetensors-only -- no pickle, matching the IPC constraint).

PRECONDITION -- runs AFTER Phase A' promote (gpu_pipeline step 10): the
capture uses the severed/promoted standalone ``kernel.py`` (which exposes
``run(inputs)->dict``). A pre-promote ``kernel.py`` still carries live
``from vllm... / from sglang...`` imports and exposes only the upstream
positional entrypoint -- capturing from it would be non-reproducible.
Each task is gated on ``meta.phase_a_prime.status == PASS`` AND the
kernel actually exposing a top-level ``run``; a task that fails the gate
is reported as an HONEST per-task error and skipped, never faked.

CLI::

    python3 scripts/phase_b_golden_capture.py --all          # 18 tasks
    python3 scripts/phase_b_golden_capture.py --task L1_14_...# one task
    python3 scripts/phase_b_golden_capture.py --all --dry-run # enumerate only
    python3 scripts/phase_b_golden_capture.py --all --force   # re-capture

``--dry-run`` enumerates the tasks + shapes and prints what WOULD be
captured without importing torch.cuda, running any kernel, or writing
anything -- it is the dev-machine (no-GPU) verification path. Without
``--dry-run`` a missing CUDA device is an HONEST hard error: this script
never fabricates a golden artifact.

Idempotent: a shape whose ``golden/<shape_id>.safetensors`` already
exists and is loadable is skipped unless ``--force`` is given.
"""
from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("phase_b_golden_capture")

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from triton_ascend_bench import eligibility  # noqa: E402
from triton_ascend_bench.dataset import TaskDef, iter_all_tasks, load_task  # noqa: E402

# The capture seed MUST match the verified runner's correctness-pass seed
# (``eval.evaluate`` builds ``correctness_inputs`` with ``seed=0``). The
# offline-golden cheat-path closure (``eval._verify_sampled_outputs_offline``)
# also reverifies sampled timing trials against this exact seed-0 input.
GOLDEN_SEED = 0

# The promoted standalone kernel exposes ``run(inputs)->dict`` -- it is
# synthesised onto ``kernel.py`` by ``phase_a_prime.synthesise_standalone_run``
# during the Phase A' promote step. This is the entrypoint the capture
# child invokes (NOT the pre-promote upstream positional entrypoint).
PROMOTED_ENTRYPOINT = "run"


# ── Capture-child script ─────────────────────────────────────────────
#
# Run in a fresh subprocess so the promoted kernel's CUDA / triton state
# never pollutes this trusted parent (mirrors the Phase A' S2 isolation
# and process_isolation's subprocess model). The child imports the
# promoted standalone ``kernel.py``, builds each shape's inputs with the
# fixed seed via ``inputs.build_inputs``, moves them onto the GPU, calls
# ``run``, and writes the output dict to ``<shape_id>.safetensors``.
#
# A child failure (no CUDA, kernel error, non-tensor output, ...) is
# surfaced via ``status.json`` so the parent reports it honestly -- the
# child NEVER writes a partial / fabricated artifact.
_CAPTURE_CHILD = r'''
import json, sys, importlib.util
from pathlib import Path

args = json.loads(sys.argv[1])
status = {"shapes": {}, "fatal": None}
status_path = Path(args["status_json"])

try:
    repo_root = args["repo_root"]
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    import torch
    from safetensors.torch import save_file
    from triton_ascend_bench.inputs import build_inputs

    if not torch.cuda.is_available():
        status["fatal"] = ("torch.cuda.is_available() is False -- the "
                           "offline golden capture must run on a real "
                           "CUDA GPU host (refusing to capture on CPU)")
        status_path.write_text(json.dumps(status))
        sys.exit(3)

    device = "cuda"
    shapes_doc = json.loads(Path(args["shapes_json"]).read_text())

    kspec = importlib.util.spec_from_file_location(
        "_pb_kernel", args["kernel_path"])
    kmod = importlib.util.module_from_spec(kspec)
    kspec.loader.exec_module(kmod)
    run_fn = getattr(kmod, args["entrypoint"], None)
    if not callable(run_fn):
        status["fatal"] = (
            "promoted kernel.py exposes no callable %r -- expected the "
            "Phase A' synthesised standalone run(inputs)->dict entrypoint "
            "(is the kernel promoted?)" % args["entrypoint"])
        status_path.write_text(json.dumps(status))
        sys.exit(2)


    def _to_device(value):
        if torch.is_tensor(value):
            return value.to(device)
        return value


    for sid in args["shape_ids"]:
        rec = {"captured": False, "error": None, "keys": []}
        try:
            built = build_inputs(shapes_doc, sid, seed=args["seed"])
            inp = {k: _to_device(v) for k, v in built.items()}
            out = run_fn(inp)
            if not isinstance(out, dict):
                rec["error"] = ("promoted kernel run() returned %s, "
                                "expected dict" % type(out).__name__)
                status["shapes"][sid] = rec
                continue
            flat = {}
            bad = None
            for k, v in out.items():
                if not torch.is_tensor(v):
                    bad = ("output[%r] is %s, not a Tensor -- a golden "
                           "artifact is a flat tensor dict"
                           % (k, type(v).__name__))
                    break
                flat[k] = v.detach().contiguous().cpu()
            if bad is not None:
                rec["error"] = bad
                status["shapes"][sid] = rec
                continue
            if not flat:
                rec["error"] = ("promoted kernel run() produced no output "
                                "tensors -- cannot capture an empty golden")
                status["shapes"][sid] = rec
                continue
            save_file(flat, str(Path(args["out_dir"]) / (sid + ".safetensors")))
            rec["captured"] = True
            rec["keys"] = sorted(flat.keys())
        except Exception as e:  # noqa: BLE001
            rec["error"] = "%s: %s" % (type(e).__name__, e)
        status["shapes"][sid] = rec
except Exception as e:  # noqa: BLE001
    status["fatal"] = "%s: %s" % (type(e).__name__, e)
    status_path.write_text(json.dumps(status))
    sys.exit(1)

status_path.write_text(json.dumps(status))
'''


@dataclass
class ShapeCaptureResult:
    """Per-shape capture outcome.

    Attributes:
        shape_id: the eval shape id.
        captured: True iff a golden artifact was written this run.
        skipped: True iff a usable artifact already existed (idempotent
            skip; mutually exclusive with ``captured``).
        error: an honest failure reason, or ``None`` on success / skip.
    """

    shape_id: str
    captured: bool = False
    skipped: bool = False
    error: str | None = None


@dataclass
class TaskCaptureResult:
    """Per-task capture outcome.

    Attributes:
        task_id: the task id.
        public_shapes: ordered public eval shape ids.
        hidden_shapes: ordered sealed hidden eval shape ids.
        precondition_error: set when Phase A' promote / entrypoint
            preconditions fail -- the task is then skipped, not faked.
        shapes: per-shape results (empty when a precondition failed).
    """

    task_id: str
    public_shapes: list[str] = field(default_factory=list)
    hidden_shapes: list[str] = field(default_factory=list)
    precondition_error: str | None = None
    shapes: list[ShapeCaptureResult] = field(default_factory=list)

    @property
    def all_shape_ids(self) -> list[str]:
        """The complete eval shape set (public + hidden)."""
        return list(self.public_shapes) + list(self.hidden_shapes)

    @property
    def ok(self) -> bool:
        """True iff the task captured cleanly (no precondition error and
        every shape captured or was idempotently skipped)."""
        if self.precondition_error is not None:
            return False
        return all(s.error is None for s in self.shapes)


def _check_promote_precondition(task: TaskDef) -> str | None:
    """Verify a task's kernel is Phase A' promoted -- the capture
    precondition (DESIGN.md §6.10 offline build order; gpu_pipeline
    step 10 runs before step 30).

    Two independent checks, both required:

    * ``meta.phase_a_prime.status == PASS`` -- the atomic-promote step
      (``phase_a_prime.update_meta_phase_a_prime``) only writes ``PASS``
      after a genuine S2-gpu-authoritative verify, so this is the
      authoritative "the standalone kernel.py is promoted" signal.
    * ``kernel.py`` actually exposes a top-level ``run`` -- the promoted
      standalone kernel carries the synthesised ``run(inputs)->dict``
      wrapper (``phase_a_prime.synthesise_standalone_run``). A defensive
      cross-check so a meta/kernel mismatch is caught before the GPU run.

    Args:
        task: the loaded :class:`TaskDef`.

    Returns:
        ``None`` when both checks pass; otherwise an honest error string
        explaining why the task cannot be captured yet.
    """
    pap = (task.meta.get("phase_a_prime") or {})
    status = pap.get("status")
    if status != "PASS":
        return (
            f"phase_a_prime.status is {status!r}, not 'PASS' -- the "
            f"kernel is not Phase A' promoted yet; run gpu_pipeline "
            f"step 10 (10_phase_a_prime.sh) first"
        )

    # Defensive: confirm the on-disk kernel.py really exposes a top-level
    # `run` -- the promoted standalone entrypoint. Static AST parse only,
    # no import / exec (this parent process never runs kernel code).
    import ast

    kernel_path = task.task_dir / "kernel.py"
    try:
        tree = ast.parse(kernel_path.read_text(encoding="utf-8",
                                               errors="replace"))
    except OSError as e:
        return f"kernel.py unreadable: {e}"
    except SyntaxError as e:
        return f"kernel.py SyntaxError line {e.lineno}: {e.msg}"
    has_run = any(
        isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == PROMOTED_ENTRYPOINT
        and node.col_offset == 0
        for node in tree.body
    )
    if not has_run:
        return (
            f"phase_a_prime.status is PASS but kernel.py exposes no "
            f"top-level {PROMOTED_ENTRYPOINT!r} -- the promoted standalone "
            f"entrypoint is missing (meta / kernel mismatch)"
        )
    return None


def _enumerate_shapes(task: TaskDef) -> tuple[list[str], list[str]]:
    """Enumerate a task's public + hidden eval shape ids.

    Uses :func:`inputs.shape_ids` -- the same W2.1 (public, hidden) split
    ``eval.evaluate`` consumes. The hidden shapes are the sealed
    verified-only eval set; the capture must cover them so an
    ``upstream_oracle_required`` task is gradable in verified mode (the
    captured hidden goldens stay sealed at the same confidentiality
    level as the hidden shapes themselves).

    Args:
        task: the loaded :class:`TaskDef`.

    Returns:
        ``(public_shape_ids, hidden_shape_ids)``.
    """
    import yaml

    from triton_ascend_bench import inputs as _inputs

    shapes_doc = yaml.safe_load(
        (task.task_dir / "shapes.yaml").read_text())
    public, hidden = _inputs.shape_ids(shapes_doc)
    return list(public), list(hidden)


def _select_tasks(task_id: str | None) -> list[TaskDef]:
    """Resolve the score-eligible ``upstream_oracle_required`` tasks to
    capture.

    ``--all`` selects only the **score-eligible**
    ``upstream_oracle_required`` tasks (18 today). The score-eligible
    filter (:func:`eligibility.is_score_eligible`) is the SAME policy
    ``gpu_pipeline`` step 20 (``phase_a_cost_xcheck.py``) applies: a
    non-score-eligible ``upstream_oracle_required`` task
    (``L1_21_gumbel_sample`` today -- a stub oracle / placeholder
    kernel) can NEVER reach ``phase_a_prime.status == PASS``, so the
    Phase A' promote precondition is unsatisfiable for it and the
    full-corpus sweep would otherwise hang on it forever. It is the
    Task #20 acknowledged non-score-eligible special case and has no
    offline golden to capture.

    Args:
        task_id: a single task id (``--task``) or ``None`` for the full
            score-eligible ``upstream_oracle_required`` set (``--all``).

    Returns:
        The loaded :class:`TaskDef` list, sorted by ``task_id``.

    Raises:
        ValueError: if ``task_id`` is given but is not an
            ``upstream_oracle_required`` task (capturing a golden for a
            PyTorch-oracle task is a category error -- such a task's
            golden is re-run live, never pre-stored), or is an
            ``upstream_oracle_required`` task that is NOT score-eligible
            (a stub-oracle / ``PHASE-C-MANUAL`` task has no capturable
            golden -- honest hard error, never a faked capture).
    """
    if task_id is not None:
        task = load_task(task_id)
        if not eligibility.is_upstream_oracle_required(task):
            raise ValueError(
                f"{task_id} is not an upstream_oracle_required task "
                f"(oracle_simplified="
                f"{task.meta.get('oracle_simplified')!r}) -- only "
                f"upstream_oracle_required tasks need an offline golden; "
                f"a PyTorch-oracle task's golden is generated live at "
                f"eval time"
            )
        if not eligibility.is_score_eligible(task):
            raise ValueError(
                f"{task_id} is an upstream_oracle_required task but is "
                f"NOT score-eligible (its oracle is a stub or its "
                f"shapes.yaml is PHASE-C-MANUAL) -- such a task has no "
                f"capturable offline golden: its kernel is a placeholder "
                f"and it can never reach phase_a_prime.status == PASS. "
                f"It is an acknowledged non-score-eligible special case "
                f"(Task #20) and is not in the golden capture scope"
            )
        return [task]
    tasks = [
        t for t in iter_all_tasks()
        if eligibility.is_upstream_oracle_required(t)
        and eligibility.is_score_eligible(t)
    ]
    tasks.sort(key=lambda t: t.task_id)
    return tasks


def _run_capture_child(
    task: TaskDef,
    shape_ids: list[str],
    golden_dir: Path,
    timeout_s: int,
) -> dict:
    """Run the promoted kernel on the GPU in an isolated subprocess.

    Spawns :data:`_CAPTURE_CHILD` for ``task``: it imports the promoted
    standalone ``kernel.py``, materialises each shape's inputs with the
    fixed seed, runs ``run`` on CUDA, and writes
    ``golden/<shape_id>.safetensors``. The child runs in a fresh
    interpreter so the kernel's CUDA / triton state never touches this
    trusted parent.

    Args:
        task: the loaded :class:`TaskDef`.
        shape_ids: the eval shape ids to capture this run (a subset of
            the task's full set when some shapes were idempotently
            skipped).
        golden_dir: the task's ``golden/`` directory (created by the
            caller).
        timeout_s: wall-clock budget for the child.

    Returns:
        The child's parsed ``status`` dict: ``{"shapes": {sid: {...}},
        "fatal": str | None}``. A ``fatal`` value is set on a
        whole-child failure (no CUDA, import error, timeout).
    """
    with tempfile.TemporaryDirectory(prefix="tab-golden-") as td:
        tdp = Path(td)
        child_py = tdp / "_capture_child.py"
        child_py.write_text(_CAPTURE_CHILD)
        status_json = tdp / "status.json"
        shapes_json = tdp / "shapes.json"

        import yaml

        shapes_doc = yaml.safe_load(
            (task.task_dir / "shapes.yaml").read_text())
        shapes_json.write_text(json.dumps(shapes_doc))

        args = {
            "repo_root": str(REPO_ROOT),
            "kernel_path": str(task.task_dir / "kernel.py"),
            "entrypoint": PROMOTED_ENTRYPOINT,
            "shapes_json": str(shapes_json),
            "shape_ids": list(shape_ids),
            "seed": GOLDEN_SEED,
            "out_dir": str(golden_dir),
            "status_json": str(status_json),
        }
        proc = subprocess.run(
            [sys.executable, str(child_py), json.dumps(args)],
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
        )
        if not status_json.exists():
            return {
                "shapes": {},
                "fatal": (
                    f"capture child produced no status.json "
                    f"(exit {proc.returncode}); stderr: "
                    f"{proc.stderr.strip()[-500:]}"
                ),
            }
        return json.loads(status_json.read_text())


def capture_task(
    task: TaskDef,
    *,
    dry_run: bool = False,
    force: bool = False,
    timeout_s: int = 600,
) -> TaskCaptureResult:
    """Capture the offline golden for one ``upstream_oracle_required`` task.

    The full per-task flow: check the Phase A' promote precondition,
    enumerate the public + hidden eval shapes, idempotently skip shapes
    that already have a usable artifact (unless ``force``), then run the
    promoted kernel on the GPU and write each shape's
    ``golden/<shape_id>.safetensors``.

    Args:
        task: the loaded :class:`TaskDef` (must be
            ``upstream_oracle_required`` -- the caller guarantees this).
        dry_run: enumerate the task + shapes only -- no precondition GPU
            gate, no kernel run, no file write. The dev-machine path.
        force: re-capture every shape even when a usable artifact exists.
        timeout_s: wall-clock budget for the GPU capture child.

    Returns:
        The :class:`TaskCaptureResult` for this task.
    """
    result = TaskCaptureResult(task_id=task.task_id)
    try:
        public, hidden = _enumerate_shapes(task)
    except Exception as e:
        result.precondition_error = f"shape enumeration failed: {e}"
        return result
    result.public_shapes = public
    result.hidden_shapes = hidden
    all_ids = result.all_shape_ids

    if not all_ids:
        result.precondition_error = (
            "shapes.yaml declares no eval shapes -- nothing to capture")
        return result

    if dry_run:
        # Dev-machine path: enumerate only. No promote gate (it would
        # need meta state we still report), no GPU, no writes.
        for sid in all_ids:
            result.shapes.append(ShapeCaptureResult(shape_id=sid))
        return result

    # --- GPU path: enforce the Phase A' promote precondition ----------
    precond = _check_promote_precondition(task)
    if precond is not None:
        result.precondition_error = precond
        return result

    golden_dir = task.task_dir / "golden"
    golden_dir.mkdir(parents=True, exist_ok=True)

    # Idempotency: skip shapes that already have a usable artifact.
    to_capture: list[str] = []
    for sid in all_ids:
        artifact = golden_dir / f"{sid}.safetensors"
        if not force and eligibility.golden_artifact_usable(artifact):
            result.shapes.append(
                ShapeCaptureResult(shape_id=sid, skipped=True))
        else:
            to_capture.append(sid)

    if not to_capture:
        logger.info("%s: all %d golden artifact(s) already present "
                    "(idempotent skip)", task.task_id, len(all_ids))
        return result

    logger.info("%s: capturing %d/%d shape(s) on GPU%s",
                task.task_id, len(to_capture), len(all_ids),
                " [--force]" if force else "")
    try:
        status = _run_capture_child(
            task, to_capture, golden_dir, timeout_s)
    except subprocess.TimeoutExpired:
        result.precondition_error = (
            f"capture child timed out after {timeout_s}s")
        return result

    fatal = status.get("fatal")
    if fatal is not None:
        # A whole-child failure (no CUDA, import error). Honest hard
        # error -- no artifact is fabricated.
        result.precondition_error = f"GPU capture child failed: {fatal}"
        return result

    child_shapes = status.get("shapes", {})
    for sid in to_capture:
        rec = child_shapes.get(sid)
        if rec is None:
            result.shapes.append(ShapeCaptureResult(
                shape_id=sid,
                error="capture child returned no record for this shape"))
            continue
        if rec.get("captured"):
            result.shapes.append(ShapeCaptureResult(
                shape_id=sid, captured=True))
        else:
            result.shapes.append(ShapeCaptureResult(
                shape_id=sid,
                error=rec.get("error") or "capture did not complete"))
    # Keep the per-shape results in the canonical (public + hidden) order.
    order = {sid: i for i, sid in enumerate(all_ids)}
    result.shapes.sort(key=lambda s: order.get(s.shape_id, len(order)))
    return result


def _report(results: list[TaskCaptureResult], dry_run: bool) -> int:
    """Print a per-task summary and return the process exit code.

    Args:
        results: the per-task capture results.
        dry_run: whether this was a ``--dry-run`` enumeration.

    Returns:
        ``0`` iff every task captured cleanly (or, for ``--dry-run``,
        every task enumerated cleanly); ``1`` otherwise.
    """
    n_shapes = sum(len(r.all_shape_ids) for r in results)
    mode = "DRY-RUN (enumerate only, no GPU)" if dry_run else "GPU capture"
    logger.info("=" * 66)
    logger.info("phase_b_golden_capture -- %s", mode)
    logger.info("upstream_oracle_required tasks: %d   eval shapes: %d",
                len(results), n_shapes)
    logger.info("=" * 66)

    failed = 0
    captured = 0
    skipped = 0
    for r in results:
        pub = len(r.public_shapes)
        hid = len(r.hidden_shapes)
        if r.precondition_error is not None:
            failed += 1
            logger.error("  [FAIL] %s (%d public + %d hidden): %s",
                         r.task_id, pub, hid, r.precondition_error)
            continue
        if dry_run:
            logger.info("  [ENUM] %s: %d public + %d hidden = %d shape(s) "
                        "-> %s", r.task_id, pub, hid, pub + hid,
                        ", ".join(r.all_shape_ids))
            continue
        t_captured = sum(1 for s in r.shapes if s.captured)
        t_skipped = sum(1 for s in r.shapes if s.skipped)
        t_failed = sum(1 for s in r.shapes if s.error is not None)
        captured += t_captured
        skipped += t_skipped
        if t_failed:
            failed += 1
            logger.error("  [FAIL] %s: %d captured, %d skipped, %d FAILED",
                         r.task_id, t_captured, t_skipped, t_failed)
            for s in r.shapes:
                if s.error is not None:
                    logger.error("           %s: %s", s.shape_id, s.error)
        else:
            logger.info("  [ OK ] %s: %d captured, %d already present",
                        r.task_id, t_captured, t_skipped)

    logger.info("=" * 66)
    if dry_run:
        logger.info("dry-run complete: %d task(s), %d eval shape(s) would "
                    "be captured", len(results), n_shapes)
        return 1 if failed else 0
    logger.info("capture complete: %d artifact(s) written, %d already "
                "present, %d task(s) failed",
                captured, skipped, failed)
    return 1 if failed else 0


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    ap = argparse.ArgumentParser(
        description=("Capture offline golden artifacts for "
                     "upstream_oracle_required tasks (Phase B', GPU host)."),
    )
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--all", action="store_true",
        help=("capture every score-eligible upstream_oracle_required "
              "task (18 today)"))
    group.add_argument(
        "--task", metavar="TASK_ID",
        help="capture a single upstream_oracle_required task")
    ap.add_argument(
        "--dry-run", action="store_true",
        help=("enumerate tasks + shapes only; no GPU, no kernel run, no "
              "file write (dev-machine verification path)"))
    ap.add_argument(
        "--force", action="store_true",
        help="re-capture even when a usable golden artifact already exists")
    ap.add_argument(
        "--timeout", type=int, default=600, metavar="SECONDS",
        help="per-task GPU capture-child wall-clock budget (default: 600)")
    return ap


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint.

    Args:
        argv: argument vector (defaults to ``sys.argv[1:]``).

    Returns:
        Process exit code: ``0`` on full success, ``1`` on any failure.
    """
    args = build_parser().parse_args(argv)
    try:
        tasks = _select_tasks(args.task)
    except FileNotFoundError as e:
        logger.error("%s", e)
        return 1
    except ValueError as e:
        logger.error("%s", e)
        return 1

    if not tasks:
        logger.error("no upstream_oracle_required tasks found under "
                     "tasks/ -- nothing to capture")
        return 1

    results = [
        capture_task(t, dry_run=args.dry_run, force=args.force,
                     timeout_s=args.timeout)
        for t in tasks
    ]
    return _report(results, dry_run=args.dry_run)


if __name__ == "__main__":
    sys.exit(main())
