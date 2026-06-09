#!/usr/bin/env python3
"""Report-only auditor for ``meta.cost_model.bytes`` correctness.

Hand-written ``M_r`` formulas can silently omit a tensor term -- L2_26's formula
missed the ``m_indices`` padded-fill store, which only surfaced because ncu
measured it. This auditor catches such omissions WITHOUT needing a GPU: it runs
the task's PyTorch oracle on each public shape, measures the REAL read (inputs)
and write (outputs) tensor bytes, and compares them to the analytic formula.

It flags, per task, two HIGH-PRECISION signals only (weaker read+write / dtype
heuristics were tried but produced too many false positives -- a measured-PASS
mixed-dtype formula or an oracle input that is not fully HBM-read -- so they are
deliberately excluded to keep every flag actionable):
  * MISSING_OUTPUT_TERM -- formula < the write-only lower bound AND ncu measured
    >> formula on an asymptotic gate shape (outputs must be written, so the
    formula is structurally missing an output/fill term -- L2_26's m_indices).
  * UNKNOWN_SYMBOL -- formula references a symbol that is neither a size knob nor
    the derived ``dtype_size``.

It NEVER rewrites meta. A flag is evidence for human review, not an auto-fix: a
missing term may still be data-dependent (L2_26's m_indices size is routing-
dependent), in which case the honest action is annotation, not a fixture-tied
constant. Output writeback absorbed by a large H200 L2 makes measured < formula
(direction OPPOSITE to a missing term), so write-absorb tasks are NOT flagged.
"""
from __future__ import annotations

import argparse
import importlib.util
import re
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parent.parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import torch  # noqa: E402

from ascendbench.perf.cost_model import FormulaEnv, parse_cost_model  # noqa: E402
from ascendbench.input_spec.inputs import build_inputs  # noqa: E402

#: declared-dtype -> element byte width (mirrors phase_a_cost_xcheck dtype_size).
_DTYPE_BYTES = {
    "float64": 8, "int64": 8, "float32": 4, "int32": 4, "uint32": 4,
    "bfloat16": 2, "float16": 2, "int16": 2, "float8_e4m3fn": 1,
    "float8_e5m2": 1, "int8": 1, "uint8": 1, "bool": 1,
}

#: formula function names (ceil/cdiv/...) -- NOT symbols, never flag as unknown.
_FORMULA_FUNCS = {"ceil", "floor", "round", "cdiv", "min", "max", "int", "abs", "sqrt"}


def _tensor_bytes(d: dict) -> float:
    return float(sum(v.element_size() * v.nelement()
                     for v in d.values() if torch.is_tensor(v)))


def _load_oracle_run(task_dir: Path):
    spec = importlib.util.spec_from_file_location(
        f"_audit_oracle_{task_dir.name}", str(task_dir / "oracle_pytorch.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.run


def _find_task_dir(tid: str) -> Path | None:
    for lvl in ("level1", "level2", "level3"):
        d = REPO / "tasks" / lvl / tid
        if d.is_dir():
            return d
    return None


def audit_task(tid: str) -> dict:
    d = _find_task_dir(tid)
    if d is None:
        return {"task": tid, "error": "task dir not found"}
    meta = yaml.safe_load((d / "meta.yaml").read_text())
    cm_raw = (meta or {}).get("cost_model")
    if not cm_raw or not cm_raw.get("bytes"):
        return {"task": tid, "error": "no cost_model.bytes"}
    cost_model = parse_cost_model(cm_raw)
    shapes_doc = yaml.safe_load((d / "shapes.yaml").read_text())
    inputs_decl = shapes_doc.get("inputs", {})
    # representative dtype_size = first declared input's width (matches xcheck).
    first_dtype = next((v.get("dtype") for v in inputs_decl.values()
                        if isinstance(v, dict) and v.get("dtype")), None)
    dtype_size = _DTYPE_BYTES.get(first_dtype)
    declared_widths = {
        _DTYPE_BYTES[v["dtype"]] for v in inputs_decl.values()
        if isinstance(v, dict) and v.get("dtype") in _DTYPE_BYTES
    }
    knob_names = set(shapes_doc.get("size_knobs", {}) or {})
    public = shapes_doc.get("public_shapes", []) or []

    try:
        run = _load_oracle_run(d)
    except Exception as e:
        return {"task": tid, "error": f"oracle load failed: {e}"}

    flags: list[str] = []
    warnings: list[str] = []
    records = []  # (sid, formula_val, out_b, in_b, measured)
    size_knobs = shapes_doc.get("size_knobs") or {}
    for sh in public:
        sid = sh["id"]
        # merge size_knobs defaults with the shape's overrides, exactly as the
        # cost-xcheck path does -- a knob like KW (conv kernel width) lives in
        # size_knobs and is not re-listed per shape.
        knobs = {**size_knobs, **(sh.get("knobs") or {})}
        try:
            ins = build_inputs(shapes_doc, sid, seed=0)
            outs = run(ins)
        except Exception as e:
            warnings.append(f"ORACLE_RUN_FAILED[{sid}]: {type(e).__name__}: {e}")
            continue
        in_b = _tensor_bytes(ins)
        # outputs that are not pass-through of an input key are genuine writes.
        out_tensors = {k: v for k, v in (outs or {}).items()
                       if torch.is_tensor(v) and k not in ins}
        out_b = _tensor_bytes(out_tensors)
        env = FormulaEnv(knobs, derived=({"dtype_size": dtype_size}
                                         if dtype_size is not None else None))
        try:
            formula_val = sum(env.evaluate(f) for f in cost_model.bytes.values())
        except Exception as e:
            warnings.append(f"FORMULA_EVAL_FAILED[{sid}]: {e}")
            continue
        measured = None
        for e in (cm_raw.get("xcheck_evidence") or []):
            if (e.get("shape_id") == sid and e.get("measured_bytes")
                    and not e.get("blocked")):
                measured = float(e["measured_bytes"])
                break
        records.append((sid, formula_val, out_b, in_b, measured))
    # Only judge MISSING on ASYMPTOTIC gate shapes (measured >= 0.5*max measured),
    # exactly as asymptotic_hbm_xcheck does -- a small shape's DRAM burst/launch
    # floor inflates measured far above the formula and must NOT be read as a
    # missing output term (the L3_18 P1 lesson).
    meas_vals = [r[4] for r in records if r[4]]
    max_meas = max(meas_vals) if meas_vals else 0.0
    gate = [r for r in records
            if r[4] is None or (max_meas > 0 and r[4] >= 0.5 * max_meas)]
    worst = None  # (ratio, sid, formula_val, out_b, in_b, measured)
    for sid, fv, ob, ib, measured in gate:
        if ob <= 0:
            continue
        ratio = fv / ob
        if worst is None or ratio < worst[0]:
            worst = (ratio, sid, fv, ob, ib, measured)
    if worst is not None:
        ratio, sid, fv, ob, ib, measured = worst
        detail = f"formula={fv:.0f} out_write={ob:.0f} rw_touch={ib + ob:.0f}"
        if ratio < 0.9:
            # Cross-check with ncu measured: a GENUINE missing HBM-write term
            # shows up as measured >> formula. If measured ~ formula, the oracle
            # output tensor simply exceeds real HBM write (on-chip reuse /
            # partial store) and the formula is fine -- not a missing term.
            if measured is not None and measured > fv * 1.5:
                flags.append(f"MISSING_OUTPUT_TERM[{sid}]: {detail}; "
                             f"measured={measured:.0f} >> formula CONFIRMS a real "
                             f"omitted HBM-write term")
            elif measured is None:
                flags.append(f"MISSING_OUTPUT_TERM?[{sid}]: {detail}; NO measured "
                             f"evidence to confirm/refute")
            # else: oracle output > formula but measured ~ formula -> formula OK
    # symbol audit (any shape's knobs are representative)
    merged0 = ({**size_knobs, **(public[0].get("knobs") or {})}
               if public else dict(size_knobs))
    try:
        sample_env = FormulaEnv(merged0, derived={"dtype_size": dtype_size or 2})
        known = set(sample_env.symbols) | {"dtype_size"}
    except Exception:
        known = set(merged0) | set(knob_names) | {"dtype_size"}
    for f in cost_model.bytes.values():
        for sym in set(re.findall(r"[A-Za-z_]\w*", f)):
            if sym not in known and sym not in _FORMULA_FUNCS:
                flags.append(f"UNKNOWN_SYMBOL: '{sym}' in bytes formula {f!r}")
    return {"task": tid, "status": cm_raw.get("status"),
            "dtype_size": dtype_size, "declared_widths": sorted(declared_widths),
            "flags": flags, "warnings": warnings}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tasks", nargs="*", help="task ids; default = all non-PASS")
    ap.add_argument("--task", action="append", dest="task_single", default=[],
                    help="single task id (repeatable); used by the daemon cert hook")
    ap.add_argument("--all", action="store_true", help="audit every task")
    args = ap.parse_args()

    picked = list(args.tasks or []) + list(args.task_single or [])
    if picked:
        tids = picked
    else:
        tids = []
        for f in sorted((REPO / "tasks").glob("level*/*/meta.yaml")):
            cm = (yaml.safe_load(f.read_text()).get("cost_model") or {})
            st = cm.get("status")
            if args.all or (st and st not in ("PASS", "SKIPPED_NON_ELIGIBLE")):
                tids.append(f.parent.name)

    n_flag = 0
    for tid in tids:
        r = audit_task(tid)
        if r.get("error"):
            print(f"[skip ] {tid}: {r['error']}")
            continue
        fl = r.get("flags") or []
        warn = r.get("warnings") or []
        if fl:
            n_flag += 1
            print(f"[FLAG ] {tid} (status={r['status']}):")
            for x in fl:
                print(f"          - {x}")
        elif warn:
            print(f"[warn ] {tid} (status={r['status']}): {warn[0]}")
        else:
            print(f"[clean] {tid} (status={r['status']})")
    print(f"\n{n_flag}/{len(tids)} task(s) flagged for formula review "
          f"(warnings = oracle-OOM/eval-env issues, not formula defects)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
