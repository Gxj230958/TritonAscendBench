#!/usr/bin/env python3
"""Per-task, append-only shape-extension tool (Unit 2 of the shape-coverage-expansion plan).

Generates candidate new shapes for a task from ITS OWN family (the active dims its ``inputs:`` use),
validates each on the CPU (a real ``build_inputs`` + oracle dry-run + ``param_constraints`` check),
and appends only the survivors -- preserving the existing P1-P5/H1-H4 rows byte-for-byte.

CRITICAL (codex design review): a CPU-validated candidate is NOT proven kernel-valid. CPU validation
only proves the shape doc is *materializable* and *oracle-domain-valid*. The FROZEN Triton kernel can
still reject it (e.g. a feature-dim BLOCK_N cap). So every survivor is labelled ``pending_gpu`` -- it
is Unit 3 (the GPU S2/C2 re-verify) that finally PROMOTES or QUARANTINES it. This tool never claims a
candidate is "valid", only "CPU-valid, pending GPU confirmation". Cheap obvious caps (element / single
-dim) preflight out the clearly-too-big candidates so GPU time is not wasted.

Determinism: active knobs are read in shapes.yaml document order; generated values, candidates, and
survivors are sorted; P6.../H5... ids are assigned only after sorting. Same task + same code -> same
candidates (so ``--dry-run`` is reproducible).

dtype: a new shape row carries ONLY knobs, never a dtype -- each input's dtype stays whatever
``inputs[*].dtype`` declares (the source of truth; ``param_constraints.dtypes`` may not cover every
input). So this tool inherently preserves dtype and does NOT sweep it (that is Unit 4).

Append-only WRITE re-emits the whole doc as canonical compact v2 (the corpus is v2 -- generate_compact_shapes
+ write_v2): the pre-existing rows re-materialise deterministically (asserted an unchanged ordered prefix per
role) and only the new candidate rows are appended.

Run:
  python3 scripts/corpus/extend_task_shapes.py --task L2_01_layer_norm_fwd --dry-run
  python3 scripts/corpus/extend_task_shapes.py --task L2_01_layer_norm_fwd --apply --target-public 6 --target-hidden 4
"""
from __future__ import annotations

import argparse
import importlib.util as iu
import json
import math
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))

from ascendbench.input_spec import param_constraints as pc  # noqa: E402
from ascendbench.input_spec.compact_shapes import generate_compact_shapes  # noqa: E402
from ascendbench.input_spec.inputs import build_inputs  # noqa: E402
from scripts.corpus.v2_authoring import load_v2, v2_to_knob_spec, write_v2  # noqa: E402

GOLDEN_SEED = 0  # mirror precompute/runner determinism

# Cheap preflight caps (NOT a kernel model -- just obvious-too-big rejects so GPU time isn't wasted).
ELEMENT_CAP = 16 * 1024 * 1024   # max elements per materialised input tensor (~64MB fp32 / 32MB bf16)
DIM_CAP = 32768                  # max value of any single active knob (feature-dim sanity)


@dataclass
class Candidate:
    source: str                  # scale | diagonal | edge | borrow
    knobs: dict                  # active-knob overrides (the row's knobs)
    role: str = "public"         # public | hidden
    status: str = ""             # CPU_VALID_PENDING_GPU | DROPPED
    reason: str = ""             # drop reason / pending note
    assigned_id: str = ""


@dataclass
class TaskExtension:
    task_id: str
    active_knobs: list[str] = field(default_factory=list)
    existing_keys: set = field(default_factory=set)
    candidates: list[Candidate] = field(default_factory=list)
    oracle_cpu_runnable: bool = True
    skipped_reason: str = ""


# ── shapes.yaml structure helpers ────────────────────────────────────────────────────────────────

def _active_knobs_in_order(shapes_doc: dict) -> list[str]:
    """Active knobs = every knob name referenced by an input's ``dims``, ``dist`` (e.g.
    ``offsets:to=E*L`` / ``full:L``), ``derive``, or ``scalar_knob`` expression, in shapes.yaml
    document order. Only these drive tensor sizes/values; the rest are inert. ``dist`` MUST be scanned
    -- some knobs (e.g. a segment-length knob in a ``dist`` string) appear nowhere else, so omitting
    ``dist`` mis-classifies them as dead (the gap this fix closes)."""
    seen: list[str] = []
    size_knobs = list((shapes_doc.get("size_knobs") or {}).keys())

    def _add(name: str) -> None:
        if name in size_knobs and name not in seen:
            seen.append(name)

    for _name, spec in (shapes_doc.get("inputs") or {}).items():
        if not isinstance(spec, dict):
            continue
        for d in spec.get("dims") or []:
            # dims entries are knob names or simple expressions like "2*D"
            for tok in re.findall(r"[A-Za-z_]\w*", str(d)):
                _add(tok)
        for fld in ("scalar_knob", "derive"):
            v = spec.get(fld)
            if isinstance(v, str):
                for tok in re.findall(r"[A-Za-z_]\w*", v):
                    _add(tok)
        # dist may be a str ("offsets:to=E*L", "full:L", "randint:0:E") or a dict; str() renders both
        # and the _add filter keeps only real size-knob names (non-knob tokens like "offsets" are ignored).
        # NOTE: this is a regex scan, not semantic -- a ref arg (e.g. randn_inf_excl:<input>) whose input
        # name collides with a size-knob name would be over-counted as active. That is the SAFE direction
        # for the dead-knob report (over-counting active => never falsely flags a live knob as dead). If
        # this detector ever becomes a hard gate (Phase 2), switch to parse_dist_spec + scan only the
        # expression-bearing args.
        dist = spec.get("dist")
        if dist is not None:
            for tok in re.findall(r"[A-Za-z_]\w*", str(dist)):
                _add(tok)
    return seen


def _effective(shapes_doc: dict, row_knobs: dict) -> dict:
    knobs = dict(shapes_doc.get("size_knobs") or {})
    knobs.update(row_knobs)
    return knobs


def _active_key(active: list[str], effective_knobs: dict) -> tuple:
    """Canonical dedup key: the effective value of each active knob (inert knobs + formatting ignored)."""
    return tuple((k, effective_knobs.get(k)) for k in active)


def _existing_keys(shapes_doc: dict, active: list[str]) -> set:
    out = set()
    for grp in ("public_shapes", "hidden_shapes"):
        for s in shapes_doc.get(grp) or []:
            if isinstance(s, dict) and s.get("id"):
                out.add(_active_key(active, _effective(shapes_doc, s.get("knobs") or {})))
    return out


def _existing_active_values(shapes_doc: dict, active: list[str]) -> list[dict]:
    vals = []
    for grp in ("public_shapes", "hidden_shapes"):
        for s in shapes_doc.get(grp) or []:
            if isinstance(s, dict) and s.get("id"):
                eff = _effective(shapes_doc, s.get("knobs") or {})
                vals.append({k: eff.get(k) for k in active})
    return vals


# ── param-constraint bound extraction (best-effort; check_param_constraints is the real gate) ──────

def _knob_bounds(meta: dict, active: list[str]) -> dict:
    """Best-effort {knob: {"min": int|None, "mod": int|None}} parsed from param_constraints.dims.
    Only used to BIAS generation toward valid values; the authoritative gate is check_param_constraints."""
    bounds = {k: {"min": 1, "mod": None} for k in active}
    dims = ((meta.get("param_constraints") or {}).get("dims")) or []
    for d in dims:
        if not isinstance(d, str):
            continue
        m = re.match(r"\s*([A-Za-z_]\w*)\s*>=\s*(\d+)", d)
        if m and m.group(1) in bounds:
            bounds[m.group(1)]["min"] = max(bounds[m.group(1)]["min"], int(m.group(2)))
        m = re.match(r"\s*([A-Za-z_]\w*)\s*%\s*(\d+)\s*==\s*0", d)
        if m and m.group(1) in bounds:
            bounds[m.group(1)]["mod"] = int(m.group(2))
    return bounds


def _round_valid(v: int, b: dict) -> int:
    """Clamp to >= min and round UP to the nearest multiple of mod (if any)."""
    v = max(v, b.get("min") or 1)
    mod = b.get("mod")
    if mod and mod > 0 and v % mod != 0:
        v = ((v // mod) + 1) * mod
    return int(v)


# ── candidate generation (self-contained: scale + diagonal + edge) ─────────────────────────────────

def _is_pow2(n: int) -> bool:
    return n >= 1 and (n & (n - 1)) == 0


def _nonpow2_near(scale: int, b: dict) -> int:
    """A NON-power-of-2 value near 1.5x ``scale`` (the knob's existing max), respecting the knob's
    min + mod constraint. Non-pow2 sizes stress tile/block assumptions; scaling to the knob's own
    range keeps it realistic (e.g. max 8192 -> ~12288) instead of a fixed small constant."""
    v = _round_valid(int(scale * 1.5) or (b.get("min") or 1), b)
    # nudge off any accidental power of two (respect mod by stepping a full mod, else +1).
    step = b.get("mod") or 1
    while _is_pow2(v):
        v += step
    return v


_GRID_PRIMES = (7, 11, 13, 17, 19, 23, 29, 31, 37, 41)


def _coprime_stride(length: int, axis: int) -> int:
    """A grid stride coprime to ``length`` so ``(i*stride) % length`` cycles through every ladder
    index. Pick the first listed prime coprime to ``length`` (rotated by ``axis`` so distinct axes
    differ); fall back to 1, which is coprime to everything."""
    for t in range(len(_GRID_PRIMES)):
        p = _GRID_PRIMES[(axis + t) % len(_GRID_PRIMES)]
        if math.gcd(p, length) == 1:
            return p
    return 1


def _gen_self_candidates(active: list[str], existing: list[dict], bounds: dict) -> list[Candidate]:
    """scale (single-axis) + diagonal (small multi-axis) + edge, all under the caps."""
    cands: list[Candidate] = []
    if not active or not existing:
        return cands
    # per-knob existing max + a median base row. Index the UNIQUE-value list by its own length
    # (a constant knob has fewer unique values than rows -- indexing by len(existing) would crash).
    kmax = {k: max((e.get(k) or 1) for e in existing) for k in active}
    base = {}
    for k in active:
        vals = sorted({e.get(k) or 1 for e in existing})
        base[k] = _round_valid(vals[len(vals) // 2] if vals else kmax[k], bounds[k])

    ladders = {k: _value_ladder(bounds[k], kmax[k]) for k in active}

    # 1. SINGLE-AXIS SWEEP: vary ONE active knob across its full value ladder (others at base) -- so
    #    every active dim's whole range (small..large, pow2 + non-pow2 LLM dims) is covered.
    for k in active:
        for v in ladders[k]:
            row = dict(base)
            row[k] = v
            cands.append(Candidate(source="sweep", knobs=row, role="public"))

    # 2. MULTI-AXIS GRID SAMPLE: a low-discrepancy (coprime-stride) sweep across the ladder PRODUCT --
    #    covers INTERACTIONS (small-B/large-D, large-B/small-D, ...) without a blind full product, for
    #    broad shape-space coverage. Deterministic (no RNG). Each axis advances by a stride coprime to
    #    ITS OWN ladder length, so (i*stride) % len visits every value -- a shared factor would collapse
    #    the axis onto a sub-cycle (or a constant) and cluster the grid.
    n_grid = 30 if len(active) >= 2 else 0  # 1-knob tasks are fully covered by the single-axis sweep
    strides = {k: _coprime_stride(len(ladders[k]), j) for j, k in enumerate(active)}
    for i in range(n_grid):
        row = {}
        for j, k in enumerate(active):
            lad = ladders[k]
            row[k] = lad[(i * strides[k] + j) % len(lad)]
        cands.append(Candidate(source="grid", knobs=dict(row), role="public", reason=f"g{i}"))

    # 3. EDGES (-> hidden): all-min, all-max, aspect extremes (one dim min / another max), singleton.
    cands.append(Candidate(source="edge", knobs={k: ladders[k][0] for k in active},
                           role="hidden", reason="all_min"))
    cands.append(Candidate(source="edge", knobs={k: ladders[k][-1] for k in active},
                           role="hidden", reason="all_max"))
    if len(active) >= 2:
        for lo, hi in ((active[0], active[-1]), (active[-1], active[0])):
            row = dict(base)
            row[lo] = ladders[lo][0]
            row[hi] = ladders[hi][-1]
            cands.append(Candidate(source="edge", knobs=dict(row), role="hidden", reason=f"aspect_{lo}_lo"))
    sing = dict(base)
    sing[active[0]] = ladders[active[0]][0]
    cands.append(Candidate(source="edge", knobs=dict(sing), role="hidden", reason="singleton"))
    return cands


def _value_ladder(b: dict, kmax: int) -> list[int]:
    """A diverse, sorted ladder of candidate values for ONE active knob, spanning its min up to ~4x
    its existing max (under DIM_CAP): powers of two + realistic non-power-of-2 LLM dims + a few
    odd/scaled points, respecting the knob's min + mod constraint. Drives the single-axis sweep so a
    wide value range (not just 2 points) is covered per dim -- the basis for ~50-shape breadth."""
    mn = b.get("min") or 1
    hi = min(max(4 * int(kmax), mn * 8, 64), DIM_CAP)
    raw: set[int] = set()
    v = 1
    while v <= hi:
        if v >= mn:
            raw.add(v)
        v *= 2
    for x in (96, 192, 256, 384, 512, 640, 768, 1024, 1280, 1536, 2048, 2560, 3072, 3584,
              4096, 5120, 6144, 8192, 12288, 16384, 24576):
        if mn <= x <= hi:
            raw.add(x)
    # a few non-aligned / odd points + the existing-max neighbourhood
    for x in (mn, mn * 3, int(kmax) // 2 or mn, int(kmax), int(kmax) * 3 // 2):
        if mn <= x <= hi:
            raw.add(x)
    vals = sorted({_round_valid(int(x), b) for x in raw if x >= 1 and _round_valid(int(x), b) <= DIM_CAP})
    return vals or [_round_valid(mn, b)]


# ── NPUKernelBench borrow (realistic LLM dims) — ONLY ops with an unambiguous geometry mapping ─────
# codex residual #4: a hand-authored dim mapping is fragile, so borrow is restricted to the two
# overlap ops whose geometry maps cleanly (LayerNorm, BatchMatmul), each guarded by a per-case filter
# AND a post-build dim assert (the materialised tensor dims must match the borrowed case). The other
# overlap ops (SwiGLU 1-D, GELU 1-D + dominating residual, 2-D matmul) map ambiguously and rely on
# the self-contained scale/diagonal/edge generation instead. New rows carry only knobs (dtype stays
# whatever inputs[*].dtype declares -- borrow is a GEOMETRY source, not a dtype/semantic source).
NPUKB_ROOT = REPO.parent / "AscendOpGenAgent" / "benchmarks" / "NPUKernelBench"


def _npukb_cases(rel_json: str) -> list[list]:
    """Return [[tensor_shape, ...], ...] per case (tensor inputs only)."""
    p = NPUKB_ROOT / rel_json
    if not p.is_file():
        return []
    out = []
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            case = json.loads(line)
        except json.JSONDecodeError:
            continue
        tensors = [i.get("shape") for i in case.get("inputs", []) if i.get("type") == "tensor"]
        attrs = [i for i in case.get("inputs", []) if i.get("type") == "attr"]
        out.append((tensors, attrs))
    return out


def _borrow_layernorm(tensors, attrs):
    """LayerNorm x=[*, D] over the LAST dim -> {B: prod(leading), D: last}. Filter: normalized_shape
    is 1-D (single-element list) so the normalize axis is exactly the last dim."""
    if not tensors or not tensors[0] or len(tensors[0]) < 1:
        return None
    ns = next((a.get("value") for a in attrs if a.get("name") == "normalized_shape"), None)
    if isinstance(ns, list) and len(ns) != 1:
        return None  # multi-dim normalize -> does not map to TAB x=[B,D]
    x = tensors[0]
    b_prod = 1
    for d in x[:-1]:
        b_prod *= int(d)
    return ({"B": max(b_prod, 1), "D": int(x[-1])}, {"x_last": int(x[-1])})


def _borrow_bmm(tensors, attrs):
    """BatchMatmul a=[b,m,k] @ bt=[b,k,n] -> {B:b, L:m, D:k, N:n}. Filter: both 3-D and k matches."""
    if len(tensors) < 2 or not tensors[0] or not tensors[1]:
        return None
    a, bt = tensors[0], tensors[1]
    if len(a) != 3 or len(bt) != 3 or int(a[2]) != int(bt[1]):
        return None
    return ({"B": int(a[0]), "L": int(a[1]), "D": int(a[2]), "N": int(bt[2])},
            {"a_dims": [int(a[0]), int(a[1]), int(a[2])]})


_NPUKB_BORROW = {
    "L2_01_layer_norm_fwd": ("level1/10_LayerNorm.json", _borrow_layernorm),
    "L2_20_bmm": ("level3/1_BatchMatmul.json", _borrow_bmm),
}


def _gen_borrow_candidates(task_id: str, active: list[str], bounds: dict, cap: int = 6) -> list[Candidate]:
    """Borrowed candidates, sampled DIVERSELY across the size range (not file order, which clusters at
    the small cases). Dedup, sort by size proxy (product of active knobs), then even-sample up to cap."""
    spec = _NPUKB_BORROW.get(task_id)
    if not spec:
        return []
    rel_json, mapper = spec
    label = f"npukb:{rel_json.split('/')[-1].replace('.json', '')}"
    by_key: dict[tuple, dict] = {}
    for tensors, attrs in _npukb_cases(rel_json):
        mapped = mapper(tensors, attrs)
        if mapped is None:
            continue
        knobs, _expect = mapped
        want = [k for k in active if k in knobs]
        row = {k: _round_valid(int(knobs[k]), bounds[k]) for k in want}
        if len(row) != len(want):
            continue
        size = 1
        for v in row.values():
            size *= max(int(v), 1)
        by_key[tuple(sorted(row.items()))] = {"row": row, "size": size}
    items = sorted(by_key.values(), key=lambda d: (d["size"], tuple(sorted(d["row"].items()))))
    if len(items) > cap > 0:
        # even spread across the size-sorted list (small -> large), deterministic.
        idx = sorted({round(i * (len(items) - 1) / (cap - 1)) for i in range(cap)}) if cap > 1 else [0]
        items = [items[i] for i in idx]
    return [Candidate(source="borrow", knobs=dict(it["row"]), role="public", reason=label)
            for it in items]


def _passes_caps(active: list[str], shapes_doc: dict, row_knobs: dict) -> tuple[bool, str]:
    """Cheap obvious-too-big preflight: per-knob DIM_CAP + per-input ELEMENT_CAP (estimated)."""
    eff = _effective(shapes_doc, row_knobs)
    for k in active:
        if (eff.get(k) or 0) > DIM_CAP:
            return False, f"{k}={eff.get(k)} > DIM_CAP {DIM_CAP}"
    for name, spec in (shapes_doc.get("inputs") or {}).items():
        if not isinstance(spec, dict) or "dims" not in spec:
            continue
        n = 1
        ok = True
        for d in spec["dims"]:
            try:
                from ascendbench.input_spec.expr import eval_numeric
                n *= int(eval_numeric(str(d), eff))
            except Exception:
                ok = False
                break
        if ok and n > ELEMENT_CAP:
            return False, f"input {name} ~{n} elements > ELEMENT_CAP {ELEMENT_CAP}"
    return True, ""


# ── CPU validation (build_inputs + oracle dry-run + param_constraints) ─────────────────────────────

def _load_oracle(task_dir: Path):
    op = task_dir / "oracle_pytorch.py"
    if not op.exists():
        return None, "no oracle_pytorch.py"
    spec = iu.spec_from_file_location(f"oracle_{task_dir.name}", str(op))
    if spec is None or spec.loader is None:
        return None, "spec_from_file_location failed"
    mod = iu.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except Exception as e:
        msg = str(e)
        if "libdevice" in msg or "triton" in msg:
            return None, f"NEEDS_GPU_FOR_ORACLE: {e}"
        return None, f"oracle import: {e}"
    if not hasattr(mod, "run"):
        return None, "oracle has no run()"
    return mod, ""


def _cpu_validate(task_dir: Path, base_doc: dict, meta: dict, oracle_mod,
                  cand: Candidate, cand_id: str) -> tuple[bool, str]:
    """Build a trial doc with the candidate row appended and run the CPU gates. Returns
    (cpu_valid, reason). cpu_valid True == materializable + oracle-domain-valid + constraint-clean
    == the candidate is CPU_VALID_PENDING_GPU (NOT proven kernel-valid)."""
    import copy
    trial = copy.deepcopy(base_doc)
    role_key = "public_shapes" if cand.role == "public" else "hidden_shapes"
    trial.setdefault(role_key, [])
    trial[role_key] = [*trial[role_key], {"id": cand_id, "name": cand.source, "knobs": cand.knobs}]

    # constraint check over the WHOLE trial doc (the shared authoritative helper)
    res = pc.check_param_constraints(meta.get("param_constraints"), trial, tid=task_dir.name)
    if not res.ok:
        return False, f"param_constraints: {'; '.join(res.reasons)[:160]}"

    try:
        inputs = build_inputs(trial, cand_id, seed=GOLDEN_SEED)
    except Exception as e:
        return False, f"build_inputs: {type(e).__name__}: {e}"

    import torch
    inputs_cpu = {k: (v.cpu() if isinstance(v, torch.Tensor) else v) for k, v in inputs.items()}
    try:
        oracle_mod.run(inputs_cpu)
    except Exception as e:
        return False, f"oracle.run: {type(e).__name__}: {e}"
    return True, "CPU-valid; pending GPU kernel confirmation"


# ── canonical v2 append ──────────────────────────────────────────────────────────────────────────
# The v2 corpus is canonical, so extend re-emits the WHOLE doc through generate_compact_shapes + write_v2
# (the byte-preserving text splice is retired): the existing rows re-materialise deterministically and the
# new candidate rows are appended.

def _write_extended_v2(orig_v2: dict, base_doc: dict, pub_rows: list[dict], hid_rows: list[dict],
                       shapes_path: Path) -> dict:
    """Append the new knob rows to ``base_doc``, re-emit the whole doc as canonical v2, and write it.

    Fail-closed: a new id must be unique among the new rows AND must not collide with an existing one; every
    PRE-EXISTING row must re-materialise byte-identical (only the new rows are added). Returns the written v2.
    """
    new_ids = [r["id"] for r in pub_rows + hid_rows]
    if len(set(new_ids)) != len(new_ids):
        raise RuntimeError(f"duplicate id among new rows: {new_ids}")
    existing = {r["id"] for r in (orig_v2.get("public") or []) + (orig_v2.get("hidden") or [])}
    clash = sorted(set(new_ids) & existing)
    if clash:
        raise RuntimeError(f"new row id(s) collide with existing shapes: {clash}")

    merged = {**base_doc,
              "public_shapes": [*(base_doc.get("public_shapes") or []), *pub_rows],
              "hidden_shapes": [*(base_doc.get("hidden_shapes") or []), *hid_rows]}
    new_v2 = generate_compact_shapes(merged)

    # byte-preservation: every pre-existing row must remain an UNCHANGED, ORDERED PREFIX of its role (the new
    # rows are only appended). Catches a regressed generate_compact_shapes that dropped / reordered / mutated
    # an existing row -- a per-id check alone would miss a drop or a cross-section move.
    for role in ("public", "hidden"):
        orig_role = orig_v2.get(role) or []
        if (new_v2.get(role) or [])[:len(orig_role)] != orig_role:
            raise RuntimeError(
                f"byte-preservation FAILED: an existing {role} row was changed/dropped/reordered on extend")
    write_v2(new_v2, shapes_path)
    return new_v2


# ── meta + contract count update ───────────────────────────────────────────────────────────────────

def _update_meta_coverage(task_dir: Path, n_public: int, n_hidden: int) -> None:
    mp = task_dir / "meta.yaml"
    meta = yaml.safe_load(mp.read_text()) or {}
    rc = meta.setdefault("runner_checks", {})
    cov = rc.setdefault("shape_coverage", {})
    cov["public"] = n_public
    cov["hidden"] = n_hidden
    cov["total"] = n_public + n_hidden
    tmp = mp.with_suffix(".yaml.tmp")
    tmp.write_text(yaml.safe_dump(meta, sort_keys=False, allow_unicode=True))
    tmp.replace(mp)


# ── per-task driver ────────────────────────────────────────────────────────────────────────────────

def extend_task(task_dir: Path, target_public: int, target_hidden: int,
                apply: bool, no_borrow: bool = False) -> TaskExtension:
    tid = task_dir.name
    ext = TaskExtension(task_id=tid)
    shapes_path = task_dir / "shapes.yaml"
    v2_doc = load_v2(shapes_path)
    meta = yaml.safe_load((task_dir / "meta.yaml").read_text()) or {}

    # Reject struct inputs FIRST (fail-closed): struct fields can materialise tensors that scale with knobs,
    # which the element-cap preflight does not model, and v2_to_knob_spec cannot reconstruct struct losslessly.
    # No corpus task uses struct today.
    if any(inp.get("kind") == "struct"
           for r in (v2_doc.get("public") or []) + (v2_doc.get("hidden") or [])
           for inp in (r.get("inputs") or [])):
        ext.skipped_reason = "struct input present -- element cap does not model struct tensors (fail-closed)"
        return ext

    # reconstruct a knob-equivalent doc so the candidate generation + CPU validation below run UNCHANGED; the
    # final write re-emits the whole doc through generate_compact_shapes + write_v2 (canonical v2).
    base_doc = v2_to_knob_spec(v2_doc)

    active = _active_knobs_in_order(base_doc)
    ext.active_knobs = active
    if not active:
        ext.skipped_reason = "no active knobs (no inputs[].dims)"
        return ext

    oracle_mod, oerr = _load_oracle(task_dir)
    if oracle_mod is None:
        ext.oracle_cpu_runnable = False
        ext.skipped_reason = f"oracle not CPU-runnable: {oerr}"
        return ext

    bounds = _knob_bounds(meta, active)
    existing_vals = _existing_active_values(base_doc, active)
    ext.existing_keys = _existing_keys(base_doc, active)

    raw = _gen_self_candidates(active, existing_vals, bounds)
    if not no_borrow:
        # borrow candidates first so realistic LLM dims get priority for the public id slots, then
        # the structural scale/diagonal fill the rest. (Both go through the SAME CPU validation.)
        raw = [*_gen_borrow_candidates(tid, active, bounds), *raw]

    # dedup vs existing + vs each other (canonical active-knob key), preserving deterministic order.
    seen = set(ext.existing_keys)
    deduped: list[Candidate] = []
    for c in raw:
        eff = _effective(base_doc, c.knobs)
        key = _active_key(active, eff)
        if key in seen:
            continue
        ok, why = _passes_caps(active, base_doc, c.knobs)
        if not ok:
            c.status, c.reason = "DROPPED", f"cap: {why}"
            ext.candidates.append(c)
            continue
        seen.add(key)
        deduped.append(c)

    # sort survivors deterministically BEFORE id assignment: public first, then per-axis sweep before
    # grid before edge -- so the full-range single-axis coverage lands within the public budget before
    # the interaction grid fills the remainder (a clustered grid must not crowd out the sweep).
    src_rank = {"sweep": 0, "grid": 1, "edge": 2}
    deduped.sort(key=lambda c: (c.role != "public", src_rank.get(c.source, 9),
                                tuple(sorted((k, v) for k, v in c.knobs.items()))))

    # validate (CPU) up to target counts per role.
    next_pub = _next_id(base_doc, "public_shapes", "P")
    next_hid = _next_id(base_doc, "hidden_shapes", "H")
    n_pub = n_hid = 0
    for c in deduped:
        want = (c.role == "public" and n_pub < target_public) or \
               (c.role == "hidden" and n_hid < target_hidden)
        if not want:
            c.status, c.reason = "DROPPED", "target reached"
            ext.candidates.append(c)
            continue
        cand_id = f"P{next_pub + n_pub}" if c.role == "public" else f"H{next_hid + n_hid}"
        ok, why = _cpu_validate(task_dir, base_doc, meta, oracle_mod, c, cand_id)
        if ok:
            c.status, c.reason, c.assigned_id = "CPU_VALID_PENDING_GPU", why, cand_id
            if c.role == "public":
                n_pub += 1
            else:
                n_hid += 1
        else:
            c.status, c.reason = "DROPPED", why
        ext.candidates.append(c)

    survivors = [c for c in ext.candidates if c.status == "CPU_VALID_PENDING_GPU"]
    if apply and survivors:
        # the knob `name` is dropped (v2 carries none); the new rows carry only id + active-knob overrides.
        pub_rows = [{"id": c.assigned_id, "knobs": c.knobs} for c in survivors if c.role == "public"]
        hid_rows = [{"id": c.assigned_id, "knobs": c.knobs} for c in survivors if c.role == "hidden"]
        new_v2 = _write_extended_v2(v2_doc, base_doc, pub_rows, hid_rows, shapes_path)
        _update_meta_coverage(task_dir, len(new_v2["public"]), len(new_v2["hidden"]))
    return ext


def _next_id(shapes_doc: dict, grp: str, prefix: str) -> int:
    nums = [int(m.group(1)) for s in (shapes_doc.get(grp) or [])
            if isinstance(s, dict) and (m := re.match(rf"{prefix}(\d+)$", str(s.get("id") or "")))]
    return (max(nums) + 1) if nums else 1


# ── CLI / reporting ────────────────────────────────────────────────────────────────────────────────

def _task_dir(task_id: str) -> Path | None:
    for level in ("level1", "level2", "level3"):
        d = REPO / "tasks" / level / task_id
        if d.is_dir():
            return d
    return None


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    p.add_argument("--task", required=True, help="task id")
    p.add_argument("--dry-run", action="store_true", help="report candidates, do NOT write (default)")
    p.add_argument("--apply", action="store_true", help="append survivors (canonical v2 re-emit)")
    p.add_argument("--target-public", type=int, default=30)
    p.add_argument("--target-hidden", type=int, default=18)
    p.add_argument("--no-borrow", action="store_true", help="disable NPUKernelBench dim borrow")
    p.add_argument("--json-out", help="write the per-candidate report as JSON")
    args = p.parse_args(argv)

    td = _task_dir(args.task)
    if td is None:
        print(f"task {args.task} not found", file=sys.stderr)
        return 2
    ext = extend_task(td, args.target_public, args.target_hidden,
                      apply=args.apply and not args.dry_run, no_borrow=args.no_borrow)

    surv = [c for c in ext.candidates if c.status == "CPU_VALID_PENDING_GPU"]
    print(f"== {ext.task_id} ==")
    print(f"active knobs (doc order): {ext.active_knobs}")
    if ext.skipped_reason:
        print(f"SKIPPED: {ext.skipped_reason}")
        return 0
    print(f"survivors (CPU-valid, PENDING GPU confirmation): {len(surv)} "
          f"({sum(c.role=='public' for c in surv)} public + {sum(c.role=='hidden' for c in surv)} hidden)")
    for c in ext.candidates:
        tag = c.assigned_id or "-"
        print(f"  [{c.status:>22}] {c.source:8} {tag:4} {c.knobs}  {('-- ' + c.reason) if c.reason else ''}")
    if args.apply and not args.dry_run and surv:
        print(f"APPLIED: appended {len(surv)} rows to {td.name}/shapes.yaml (canonical v2 re-emit). "
              f"Task now correctly release-gate RED until Unit-3 GPU re-verify.")
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(
            {"task_id": ext.task_id, "active_knobs": ext.active_knobs,
             "candidates": [vars(c) for c in ext.candidates]}, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
