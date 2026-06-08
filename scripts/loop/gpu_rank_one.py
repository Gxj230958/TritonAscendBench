#!/usr/bin/env python3
"""Cost-model ranking for one task on the GPU host — via the TRUSTED operator.

The bench grades untrusted submissions through the candidate sandbox (static
check + Layer E). But validating that a task RANKS uses the task's own
reference operator, which is trusted — so we run it exactly as ``oracle-equiv``
does (``_verify_one_task``: launcher-aware entrypoint resolution, wrapper
contract, upstream deps via infer/), with per-shape timing turned on
(``TAB_RANK_TIMING=1``). The reference operator therefore needs NO candidate
static check (it is not an untrusted submission). We then fold each shape's
measured ``t_npu`` with the cost-model ``t_ref`` into ``e = t_ref/t_npu`` and
report whether the task ranks.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))


@dataclass
class RankVerdict:
    task: str
    overall: str                 # RANKED | FLOOR | NOT_RANKED | BLOCKED | ERROR
    correctness: str | None      # oracle-equiv overall (VERIFIED/FAILED/...)
    cost_model_status: str | None  # meta cost_model.status (PASS/BLOCKED/FLOOR_DOMINATED)
    n_shapes_ranked: int
    n_shapes_total: int
    s_t: float | None            # geomean of per-shape e (FLOOR tasks excluded)
    shape_e: dict                # per-shape e (diagnostic; FLOOR shapes kept but not in s_t)
    shape_t_npu_ns: dict         # per-shape median do_bench device latency (ns)
    shape_t_npu_spread_ns: dict  # per-shape [min, max] timing samples (ns)
    shape_t_ref_ns: dict         # per-shape cost-model floor (ns)
    timing_method: str | None    # do_bench | event_fallback | do_bench+event_fallback (degraded if fallback present)
    device: str | None
    error: str | None


def _err(task, overall, correctness, error, device, n_total=0):
    """Build an early-exit RankVerdict with empty per-shape maps."""
    return RankVerdict(
        task=task, overall=overall, correctness=correctness,
        cost_model_status=None, n_shapes_ranked=0, n_shapes_total=n_total,
        s_t=None, shape_e={}, shape_t_npu_ns={}, shape_t_npu_spread_ns={},
        shape_t_ref_ns={}, timing_method=None, device=device, error=error)


_CORRECTNESS_PASS = ("VERIFIED", "MIXED", "CEILING_ACCEPTED")


def _classify_overall(cm_status: str | None, correctness: str,
                      n_ranked: int) -> str:
    """Task-level ranking outcome (pure, GPU-free -> unit-testable).

    CEILING_ACCEPTED is a valid (dtype-ceiling) correctness pass. A
    FLOOR_DOMINATED task is not rank-comparable, but only counts as FLOOR when
    correctness actually passed -- a fully FAILED FLOOR_DOMINATED task must
    surface as NOT_RANKED, never be masked as FLOOR. A MIXED task ranks its
    passing shapes (per-shape ranking) and is RANKED when at least one shape
    ranked. A FLOOR_DOMINATED+MIXED task is intentionally FLOOR (MIXED is a
    correctness pass); per-shape failures surface via the ``correctness`` field
    and n_shapes_ranked, not via this overall token.
    """
    corr_ok = correctness in _CORRECTNESS_PASS
    if cm_status == "FLOOR_DOMINATED" and corr_ok:
        return "FLOOR"
    if corr_ok and n_ranked > 0:
        return "RANKED"
    return "NOT_RANKED"


def run(task: str, peak_table: str) -> RankVerdict:
    import statistics

    # Enable per-shape ranking timing before the runner module is imported
    # (it reads TAB_RANK_TIMING at import time). Done here, not at module top,
    # so a plain `import gpu_rank_one` (e.g. in unit tests) does not mutate env.
    os.environ.setdefault("TAB_RANK_TIMING", "1")
    import oracle_equivalence_gpu_runner as oeq
    import yaml

    from ascendbench.perf import peak_table as pt
    from ascendbench.perf import scoring
    from ascendbench.corpus.dataset import load_task
    from ascendbench.engine.eval import (
        _representative_dtype_size,
        _shape_knobs,
    )

    try:
        import torch
        device = torch.cuda.get_device_name(0)
    except Exception as e:
        device = f"<{type(e).__name__}>"

    task_dir = None
    for lvl in ("level1", "level2", "level3"):
        cand = REPO_ROOT / "tasks" / lvl / task
        if cand.is_dir():
            task_dir = cand
            break
    if task_dir is None:
        return _err(task, "ERROR", None, "task dir not found", device)

    infer_root = REPO_ROOT.parent / "infer"
    try:
        verdict = oeq._verify_one_task(task_dir, infer_root, "cuda")
    except Exception as e:
        return _err(task, "ERROR", None, f"verify: {type(e).__name__}: {e}",
                    device)

    # Cost-model side: peaks + knobs + dtype -> per-shape t_ref -> e.
    try:
        task_def = load_task(task)
        cost_model = task_def.cost_model
        compute_peaks, resource_peaks = pt.load_peak_table(peak_table)
        shapes_doc = yaml.safe_load((task_dir / "shapes.yaml").read_text()) or {}
        dtype_size = _representative_dtype_size(shapes_doc)
        meta_doc = yaml.safe_load((task_dir / "meta.yaml").read_text()) or {}
        _cm_meta = meta_doc.get("cost_model")
        cm_status = (_cm_meta.get("status")
                     if isinstance(_cm_meta, dict) else None)
    except Exception as e:
        v = _err(task, "BLOCKED", verdict.overall,
                 f"cost-model setup: {type(e).__name__}: {e}", device,
                 n_total=len(verdict.shapes))
        return v

    # FLOOR_DOMINATED: cost_model.bytes is below the DRAM burst floor, so t_ref
    # is not applicable and e is not a comparable efficiency (the kernel is also
    # sub-microsecond, near the CUDA-event resolution). Keep raw e for diagnosis
    # but exclude these from the ranked s_t.
    is_floor = (cm_status == "FLOOR_DOMINATED")

    shape_e: dict = {}
    shape_t_npu: dict = {}
    shape_spread: dict = {}
    shape_t_ref: dict = {}
    timing_methods: set = set()
    es: list[float] = []
    for sv in verdict.shapes:
        if getattr(sv, "timing_method", None):
            timing_methods.add(sv.timing_method)
        e_val = None
        t_ref_ns = None
        if (cost_model is not None and sv.t_npu_ns and sv.t_npu_ns > 0
                and sv.status in ("VERIFIED", "CEILING_ACCEPTED")):
            try:
                knobs = _shape_knobs(shapes_doc, sv.shape_id)
                if knobs is not None:
                    t_ref_ns = scoring.compute_t_ref(
                        cost_model, knobs, compute_peaks, resource_peaks,
                        dtype_size=dtype_size) * 1e9
                    e_val = scoring.shape_efficiency(t_ref_ns, sv.t_npu_ns)
            except Exception:
                e_val = None
        shape_e[sv.shape_id] = e_val
        shape_t_npu[sv.shape_id] = sv.t_npu_ns
        shape_spread[sv.shape_id] = (
            [sv.t_npu_min_ns, sv.t_npu_max_ns]
            if sv.t_npu_min_ns is not None else None)
        shape_t_ref[sv.shape_id] = t_ref_ns
        if e_val is not None and not is_floor:
            es.append(e_val)

    n_ranked = len(es)
    s_t = statistics.geometric_mean(es) if es else None
    overall = _classify_overall(cm_status, verdict.overall, n_ranked)
    return RankVerdict(
        task=task, overall=overall, correctness=verdict.overall,
        cost_model_status=cm_status, n_shapes_ranked=n_ranked,
        n_shapes_total=len(verdict.shapes), s_t=s_t, shape_e=shape_e,
        shape_t_npu_ns=shape_t_npu, shape_t_npu_spread_ns=shape_spread,
        shape_t_ref_ns=shape_t_ref,
        timing_method=("+".join(sorted(timing_methods)) or None),
        device=device, error=None)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    p.add_argument("--task", required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--peak-table", default="infra/peak_table_h200.toml")
    args = p.parse_args(argv)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    v = run(args.task, args.peak_table)
    (args.output_dir / f"{args.task}.json").write_text(
        json.dumps(asdict(v), indent=2, default=str))
    print(f"gpu-rank {args.task}: {v.overall} -- correctness={v.correctness} "
          f"cost_model={v.cost_model_status} timing={v.timing_method} "
          f"ranked={v.n_shapes_ranked}/{v.n_shapes_total} s_t={v.s_t} "
          f"device={v.device} err={v.error}")
    # FLOOR is a valid outcome (correctness-verified, not rank-comparable).
    return 0 if v.overall in ("RANKED", "FLOOR") else 1


if __name__ == "__main__":
    raise SystemExit(main())
