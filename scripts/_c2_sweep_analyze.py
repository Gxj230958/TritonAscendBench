#!/usr/bin/env python3
"""Classify a full-shape C2 sweep: per task, separate a real large-shape oracle
divergence (STRUCTURAL) from the CPU-fp32-vs-GPU-bf16/fp8 tensor-core ceiling (CEILING).
Reads the collected per-task job-wrapper JSONs (verdict_status + stderr_tail log lines
"<task>/<shape>: <STATUS> (max_abs=.. max_rel=.. max_tol_units=..)").

The ONLY reliable structural signal is ``max_tol_units`` = max(|actual-ref| /
(atol + rtol*|ref|)) -- the CO-LOCATED normalized error the GPU runner now emits.
max_abs and max_rel are independent per-tensor maxima at DIFFERENT elements, so gating
on them mis-classifies (e.g. L2_19: max_abs=2 is 1 bf16 ULP on a -478 value, while
max_rel=1.7 is a separate near-zero element -- neither is structural). A FAILED shape
is:
  * CEILING if it only just crosses the band (max_tol_units in (1, STRUCT_TU]) -- a
    1-ULP bf16/fp16/fp8 / tensor-core-accumulation effect at scale.
  * STRUCTURAL if max_tol_units > STRUCT_TU -- a divergence orders beyond the tolerance
    band (L2_23/24 off-by-one: a value wholly wrong, tol_units in the thousands).
Ungraded shapes (CPU-oracle OOM/skip) are reported explicitly, never folded into "pass".

Back-compat: logs without max_tol_units (pre-metric runs) fall back to a coarse
abs+rel heuristic and the task is tagged NEEDS_TU so it is re-run with the metric.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import yaml

# tol-units beyond which a FAILED shape is structural, not a ULP ceiling. A 1-bf16/fp8
# ULP ceiling lands at 1-24 tol-units depending on how tight the task's tolerance is
# (e.g. atomic_scatter atol=1e-4 << bf16 granularity -> ~24 tol-units for 1 ULP). A
# genuine structural divergence (L2_23/24 off-by-one: value wholly wrong) lands in the
# THOUSANDS. 50 cleanly separates the two populations (max gray-zone ceiling ~24, min
# structural shape ~3000). Empirically calibrated on the bc2tu sweep.
STRUCT_TU = 50.0
REPO = Path(__file__).resolve().parent.parent
_LINE = re.compile(
    r"/(\w+):\s+(\w+)\s+\(max_abs=(\S+?)\s+max_rel=(\S+?)"
    r"(?:\s+max_tol_units=(\S+?))?\)"
)


def _f(s: str | None) -> float:
    if s is None:
        return float("nan")
    try:
        return float(s)
    except ValueError:
        return float("nan")


def _expected_shapes(tid: str) -> list[str]:
    lvl = "level1" if tid.startswith("L1") else "level2" if tid.startswith("L2") else "level3"
    sp = REPO / "tasks" / lvl / tid / "shapes.yaml"
    if not sp.exists():
        return []
    d = yaml.safe_load(sp.read_text())
    return [s["id"] for g in ("public_shapes", "hidden_shapes") for s in (d.get(g) or [])]


def analyze(results_dir: Path):
    cats = {"all_pass": [], "ceiling": [], "structural": [], "ungraded": [],
            "needs_tu": [], "blocked": []}
    rows = {}
    for fp in sorted(results_dir.glob("*.json")):
        try:
            v = json.loads(fp.read_text())
        except (ValueError, OSError):
            continue
        # This helper parses the loop_mailbox per-task job-wrapper JSON
        # (task_id + verdict_status + stderr_tail log lines). Skip anything else
        # (e.g. the GPU runner's native TaskVerdict JSON with a `shapes` list).
        if not isinstance(v, dict) or "stderr_tail" not in v:
            continue
        tid = v.get("task_id", fp.stem)
        log = (v.get("stderr_tail", "") or "") + (v.get("stdout_tail", "") or "")
        per = [(sh, st, _f(ma), _f(mr), _f(tu)) for sh, st, ma, mr, tu in _LINE.findall(log)]
        graded = {p[0] for p in per}
        exp = _expected_shapes(tid)
        missing = [s for s in exp if s not in graded] if exp else []
        failed = [p for p in per if p[1] == "FAILED"]
        has_tu = any(p[4] == p[4] for p in failed)  # any non-nan tol_units among fails
        struct = [p for p in failed if p[4] == p[4] and p[4] > STRUCT_TU]
        rows[tid] = (v.get("verdict_status"), per, missing)
        if not per:
            cats["blocked"].append(tid)
        elif struct:
            cats["structural"].append(tid)
        elif failed and not has_tu:
            cats["needs_tu"].append(tid)  # old log without the metric -- re-run
        elif missing:
            cats["ungraded"].append(tid)
        elif not failed:
            cats["all_pass"].append(tid)
        else:
            cats["ceiling"].append(tid)
    return cats, rows


def main() -> int:
    rd = Path(sys.argv[1])
    cats, rows = analyze(rd)
    tot = sum(len(x) for x in cats.values())
    print(f"=== full-shape C2 sweep: {tot} tasks ===")
    for k in ("all_pass", "ceiling", "structural", "needs_tu", "ungraded", "blocked"):
        print(f"  {k:11s}: {len(cats[k])}")
    print()
    for cat in ("structural", "needs_tu", "ungraded", "ceiling", "blocked"):
        if not cats[cat]:
            continue
        print(f"--- {cat.upper()} ({len(cats[cat])}) ---")
        for tid in cats[cat]:
            status, per, missing = rows[tid]
            fails = [p for p in per if p[1] == "FAILED"]
            worst = max(per, key=lambda p: (p[4] if p[4] == p[4] else -1)) if per else None
            wtu = f"{worst[4]:.4g}" if worst else "-"
            extra = f" MISSING={missing}" if missing else ""
            print(f"  {tid:46s} {status:9s} nfail={len(fails)}/{len(per)} "
                  f"worst_tol_units={wtu}{extra}")
            if cat in ("structural", "needs_tu"):
                for sh, st, ma, mr, tu in sorted(per):
                    m = "  <<STRUCT" if (st == "FAILED" and tu == tu and tu > STRUCT_TU) else ""
                    print(f"       {sh:4s} {st:10s} abs={ma:.4g} rel={mr:.4g} tol_units={tu:.4g}{m}")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
