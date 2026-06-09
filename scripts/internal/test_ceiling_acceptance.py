#!/usr/bin/env python3
"""Fail-closed unit tests for the dtype-ceiling acceptance verdict.
Run: python3 scripts/internal/test_ceiling_acceptance.py"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "oracle"))
import run_oracle_equivalence_gpu as R

MODEB = "L3_22_chunk_gated_delta_rule_fwd_kernel_h_blockdim64"  # the ONLY Mode-B task

def mk(tu=None, mabs=None, mref=None, fk=None, bad=None, blocker="", cerr=False):
    sv = R.ShapeVerdict(shape_id="P4")
    sv.max_tol_units = tu or {}
    sv.max_abs_diff = mabs or {}
    sv.max_ref_abs = mref or {}
    sv.failing_keys = fk if fk is not None else (list((tu or mabs or {}).keys()))
    sv.bad_dtype_keys = bad or []
    sv.blocker_reason = blocker
    sv.compare_error = cerr
    return sv

cfg = R._ceiling_config()
fac = float(cfg["physical_factor"]); ulpB = float(cfg["physical_tasks"][MODEB]["ulp_rel"])
cases = []
def ck(name, sv_or_acc, exp, task="L1_12_rms_norm"):
    acc = sv_or_acc if isinstance(sv_or_acc, bool) else R._ceiling_verdict(task, sv_or_acc)[0]
    cases.append((name, acc == exp, acc, exp))

# Allowlist membership locks: Mode-B holds exactly L3_13; the rest are Mode-A.
ck("config: physical_tasks is exactly {L3_13}", set(cfg["physical_tasks"]) == {MODEB}, True)
ck("config: L2_03 is Mode-A (in tol_units_tasks)", "L3_02_chunk_state_fwd" in cfg["tol_units_tasks"], True)
ck("config: L2_03 NOT Mode-B", "L3_02_chunk_state_fwd" not in cfg["physical_tasks"], True)
ck("config: 18 Mode-A + 1 Mode-B = 19", len(cfg["tol_units_tasks"]) == 18 and len(cfg["physical_tasks"]) == 1, True)

# Mode-A
ck("modeA under cap + failing_key accept", mk(tu={"o":6.0}, fk=["o"]), True)
ck("modeA over cap reject", mk(tu={"o":9.0}, fk=["o"]), False)
ck("modeA NO failing_keys -> fail-closed", mk(tu={"o":6.0}, fk=[]), False)
ck("modeA normal tolerance-fail blocker text -> still ACCEPT", mk(tu={"o":6.0}, fk=["o"], blocker="o: tolerance fail (max_abs=..)"), True)
ck("modeA compare_error=True -> fail-closed", mk(tu={"o":6.0}, fk=["o"], cerr=True), False)
ck("modeA dtype mismatch -> fail-closed", mk(tu={"o":6.0}, fk=["o"], bad=["o"]), False)
ck("modeA nonfinite tol_units -> fail-closed", mk(tu={"o":float('inf')}, fk=["o"]), False)
# L2_03 lock: its real failing profile (tol_units 2.19, but max_abs 0.1454 / max|ref| 15 would
# FAIL the Mode-B physical bound 2*15*ulpB=0.117). Mode-A must ACCEPT on tol_units alone.
ck("L2_03 accepts via tol_units despite physical-failing max_abs", mk(tu={"states":2.19}, mabs={"states":0.1454}, mref={"states":15.0}, fk=["states"]), True, "L3_02_chunk_state_fwd")

# Mode-B (L3_13): bound = fac*mref*ulpB; tol_units cap does NOT apply.
ck("modeB within bound accept", mk(mabs={"output":0.2}, mref={"output":50.0}, fk=["output"]), True, MODEB)
ck("modeB over bound reject", mk(mabs={"output":0.5}, mref={"output":50.0}, fk=["output"]), False, MODEB)
ck("modeB zero scale -> fail-closed", mk(mabs={"output":0.2}, mref={"output":0.0}, fk=["output"]), False, MODEB)
ck("modeB nonfinite -> fail-closed", mk(mabs={"output":float('inf')}, mref={"output":50.0}, fk=["output"]), False, MODEB)
ck("modeB no failing_keys -> fail-closed", mk(mabs={"output":0.2}, mref={"output":50.0}, fk=[]), False, MODEB)
ck("modeB dtype mismatch -> fail-closed", mk(mabs={"output":0.2}, mref={"output":50.0}, fk=["output"], bad=["output"]), False, MODEB)
ck("modeB two keys, one over -> reject", mk(mabs={"a":0.2,"b":0.5}, mref={"a":50.0,"b":50.0}, fk=["a","b"]), False, MODEB)
# L3_13 lock: accepts via physical bound EVEN with tol_units 12.10 (> cap 8) -- proves it is
# judged by the physical fallback, not the Mode-A cap.
ck("L3_13 accepts via physical despite tol_units 12.10 > cap", mk(tu={"output":12.10}, mabs={"output":0.2}, mref={"output":50.0}, fk=["output"]), True, MODEB)

# Guardrails
ck("L2_23 not-listed reject (low tu)", mk(tu={"o":2.0}, fk=["o"]), False, "L2_23_fwd_kv_parallel")
ck("L2_23 not-listed reject (physical)", mk(mabs={"o":0.01}, mref={"o":50.0}, fk=["o"]), False, "L2_23_fwd_kv_parallel")
ck("unknown task reject", mk(tu={"o":1.0}, fk=["o"]), False, "L9_99_made_up")
# non-finite config -> fail-closed (mutate cached cfg)
R._CEILING_CFG = dict(cfg); R._CEILING_CFG["tol_units_cap"] = float("inf")
ck("modeA .inf cap -> fail-closed", mk(tu={"o":1e9}, fk=["o"]), False)
R._CEILING_CFG = dict(cfg); R._CEILING_CFG["physical_factor"] = float("nan")
ck("modeB .nan factor -> fail-closed", mk(mabs={"o":1e9}, mref={"o":50.0}, fk=["o"]), False, MODEB)
R._CEILING_CFG = cfg  # restore

ok = all(c[1] for c in cases)
for name, passed, got, exp in cases:
    print(f"  [{'OK' if passed else 'FAIL'}] {name}: got={got} exp={exp}")
print(f"\n>>> {sum(c[1] for c in cases)}/{len(cases)} PASS" + ("  ALL FAIL-CLOSED OK" if ok else "  <<< FAILURE"))
sys.exit(0 if ok else 1)
