"""Deterministic auto-filler for shapes-incomplete BLOCKED tasks.

The resolver reports missing param names in its per-task ``reason`` field,
e.g.::

    BLOCKED(shapes-incomplete: missing param 'topk_ids' for entrypoint X)

This tool maps those names to default ``(dtype, dims, dist)`` tuples using
a hardcoded conventions table covering the ~30 most common ML idioms seen
in the bench corpus, and appends matching rows to each task's
``shapes.yaml`` ``inputs:`` block so the next resolver sweep's by-name
adapter can bind on them.

Why standalone (vs hooked into fix_orchestrator): keeps the blast radius
small. The orchestrator already routes shapes-incomplete tasks through
the ``patch-shapes`` decision + LLM; this tool front-runs the LLM for
the conventional names, leaving the LLM with only the genuinely novel
ones (which it actually needs to think about). Run between
``resolve`` and ``smoke``::

    python3 scripts/auto_shapes_filler.py             # dry-run
    python3 scripts/auto_shapes_filler.py --apply     # write changes
    python3 scripts/auto_shapes_filler.py --task L1_15_act_and_mul --apply

The tool is idempotent: it never overwrites an existing ``inputs[name]``
entry, only appends missing ones. Writes are atomic (temp + rename) so a
crash mid-write leaves shapes.yaml byte-identical. A convention is only
applied when the task's existing ``public_shapes`` knobs declare every
dim symbol the convention needs -- otherwise the row is skipped (the LLM
path will get it).

Cascade caveat: the resolver only reports the FIRST missing param per
sweep (the by-name adapter stops at the first ``ShapesIncompleteError``).
A task with N missing params therefore needs up to N
auto-filler + resolve iterations to resolve; the summary's
``fully-filled`` count reflects only this iteration's evidence, not the
final count. Workflow::

    resolve --force
    auto_shapes_filler --apply
    resolve --force        # picks up any newly-revealed missing params
    auto_shapes_filler --apply
    # repeat until summary shows no new fills
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import yaml

THIS = Path(__file__).resolve()
REPO_ROOT = THIS.parent.parent
TASKS_ROOT = REPO_ROOT / "tasks"
DEFAULT_CACHE = REPO_ROOT / "results" / "resolve_all.json"

# Conventions table: param_name -> (dtype, dims, dist_or_scalar_default).
#
# ``dims`` is either a tuple of dim-symbol strings (tensor rows) or the
# literal string ``"scalar"`` (scalar inputs). Composite symbols:
#   ``BN`` = B * N      (used by topk_ids: (batch*tokens, topk))
#   ``BP1`` = B + 1     (used by cu_seqlens, the prefix-sum form)
# Composite symbols are decomposed into their atomic bases (``B``, ``N``)
# when checking that the task's knobs declare all required dims.
#
# Each entry below appears >= 2 times in the resolve_all.json
# shapes-incomplete tasks (verified against the 2026-05-25 snapshot:
# 56 shapes-incomplete tasks, 30 distinct names with >=2 occurrences).
# Names appearing only once are deliberately omitted -- they are
# task-specific and belong on the LLM path.
_CONVENTIONS: dict[str, tuple[str | None, tuple[str, ...] | str, str | None]] = {
    # Generic activation tensors (float, B-N-D shape).
    "input":         ("float16", ("B", "N", "D"), "randn"),
    "input_emb":     ("float16", ("B", "N", "D"), "randn"),
    "input_tensor":  ("float16", ("B", "N", "D"), "randn"),
    "x":             ("float16", ("B", "N", "D"), "randn"),
    "hidden_states": ("float16", ("B", "N", "D"), "randn"),
    "recv_x":        ("float16", ("B", "N", "D"), "randn"),
    "residual":      ("float16", ("B", "N", "D"), "randn"),
    # GEMM operand matrices (lowercase ``a`` and uppercase ``A`` both seen).
    "a":             ("float16", ("M", "K"), "randn"),
    "A":             ("float16", ("M", "K"), "randn"),
    # Generic output tensors (kernel writes; no dist needed).
    "output":        ("float16", ("B", "N", "D"), None),
    "out":           ("float16", ("B", "N", "D"), None),
    "o":             ("float16", ("B", "N", "D"), None),
    "down_output":   ("float16", ("B", "N", "D"), None),
    "att_out":       ("float16", ("B", "H", "N", "D"), None),
    # MLP intermediates (gate+up projection output; (B*N, 2*I) is common).
    "gateup_input":  ("float16", ("BN", "I2"), "randn"),
    # Attention tensors.
    "q":             ("float16", ("B", "H", "N", "D"), "randn"),
    "k":             ("float16", ("B", "H", "N", "D"), "randn"),
    "v":             ("float16", ("B", "H", "N", "D"), "randn"),
    "key":           ("float16", ("B", "H", "N", "D"), "randn"),
    "value":         ("float16", ("B", "H", "N", "D"), "randn"),
    "logits":        ("float16", ("B", "V"), "randn"),
    # Masks / routing.
    "mask":          ("bool", ("B", "N"), None),
    "masked_m":      ("bool", ("B", "N"), None),
    "topk_ids":      ("int32", ("BN", "K"), "randint:0:E"),
    "topk_indices":  ("int32", ("BN", "K"), "randint:0:E"),
    # Sequence-length vectors.
    "seq_lens":         ("int32", ("B",), "randint:1:N"),
    "extend_seq_lens":  ("int32", ("B",), "randint:1:N"),
    "cu_seqlens":       ("int32", ("BP1",), None),
    # Scalars (dtype carried implicitly by yaml value; default chosen so
    # the resolver's by-name adapter binds successfully on the first
    # sweep -- the real value is task-specific and the LLM/operator
    # author will refine if the default is wrong).
    "softcap_const":  (None, "scalar", "1.0"),     # float
    # Triton block-size meta-params travel as a {M, N, K} family at the
    # host wrapper boundary; the resolver only reports the FIRST one
    # missing per sweep, so the other two will land in the next cache
    # iteration -- table them all together so cascade resolves in one
    # extra sweep instead of three.
    "BLOCK_SIZE_M":   (None, "scalar", "64"),      # int
    "BLOCK_SIZE_N":   (None, "scalar", "64"),      # int
    "BLOCK_SIZE_K":   (None, "scalar", "32"),      # int
    "max_num_tokens": (None, "scalar", "1024"),    # int
    "weight":         ("float16", ("M", "K"), "randn"),
    "x_scale":        ("float32", ("B",), "randn"),
    # Misc structured arrays that appear >= 2 times.
    "problem_sizes1": ("int32", ("E", "DIM3"), None),
    "conv_states":    ("float16", ("B", "D", "S"), "randn"),
    "kv_caches":      ("float16", ("BLOCKS", "PAGE", "H", "D"), "randn"),
    "quant_k_cache":  ("int8",    ("BLOCKS", "PAGE", "H", "D"), "randint:-128:127"),
}

_MISSING_PARAM_RE = re.compile(r"missing param '([^']+)'")

# Composite dim symbols decomposed to atomic bases. Only the atomic
# bases must exist in the task's knobs universe; composites are kept
# verbatim in the emitted YAML so a separate spec layer can resolve
# them later (B*N for BN, B+1 for BP1, etc.).
_COMPOSITE_BASES: dict[str, tuple[str, ...]] = {
    "BN":   ("B", "N"),
    "BP1":  ("B",),
    "I2":   ("I",),
    "DIM3": (),  # literal-3 placeholder; no symbol required
}


def _load_cache(path: Path) -> list[dict]:
    if not path.is_file():
        sys.exit(
            f"resolve cache not found: {path}\n"
            f"run `bash scripts/automation/runners/tab_quickstart.sh resolve --force` first"
        )
    try:
        data = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError) as exc:
        sys.exit(f"resolve cache unreadable: {exc}")
    return data.get("verdicts", []) or []


def _missing_params(verdict: dict) -> list[str]:
    if "shapes-incomplete" not in verdict.get("status", ""):
        return []
    reason = verdict.get("reason") or ""
    # Preserve discovery order; de-dup so a name reported on multiple
    # candidates only generates one entry.
    out: list[str] = []
    seen: set[str] = set()
    for m in _MISSING_PARAM_RE.findall(reason):
        if m not in seen:
            out.append(m)
            seen.add(m)
    return out


def _knobs_universe(shapes_doc: dict) -> set[str]:
    """Union of dim symbols declared by any public_shapes or
    hidden_shapes row's ``knobs`` map. This is the set of dim names the
    task already binds; conventions referencing dims outside this set
    cannot be auto-filled safely."""
    universe: set[str] = set()
    for section in ("public_shapes", "hidden_shapes"):
        for row in shapes_doc.get(section) or []:
            universe.update((row.get("knobs") or {}).keys())
    return universe


def _existing_inputs(shapes_doc: dict) -> set[str]:
    return set((shapes_doc.get("inputs") or {}).keys())


def _scalar_value(default: str | None):
    if default is None:
        return 0
    try:
        return int(default)
    except ValueError:
        pass
    try:
        return float(default)
    except ValueError:
        return default


def _build_entry(name: str, knobs: set[str]) -> dict | None:
    """Build the ``shapes.yaml inputs[name]`` entry from CONVENTIONS, or
    return ``None`` when the convention is missing OR when the required
    atomic dim bases are not all present in the task's knobs universe."""
    spec = _CONVENTIONS.get(name)
    if spec is None:
        return None
    dtype, dims, dist = spec
    if dims == "scalar":
        return {"scalar": _scalar_value(dist)}
    assert isinstance(dims, tuple)
    bases: set[str] = set()
    for d in dims:
        if d in _COMPOSITE_BASES:
            bases.update(_COMPOSITE_BASES[d])
        else:
            bases.add(d)
    if not bases.issubset(knobs):
        return None
    entry: dict = {"dims": list(dims)}
    if dtype is not None:
        entry["dtype"] = dtype
    if dist is not None:
        entry["dist"] = dist
    return entry


def _format_entry_text(name: str, entry: dict) -> str:
    """Render one ``inputs[name]: entry`` as YAML text indented to live
    under ``inputs:`` (2-space top, 4-space sub-keys -- matching the
    style used everywhere else in the bench's shapes.yaml files)."""
    lines = [f"  {name}:"]
    for k, v in entry.items():
        if isinstance(v, list):
            lines.append(f"    {k}:")
            for item in v:
                lines.append(f"    - {item}")
        else:
            lines.append(f"    {k}: {v}")
    return "\n".join(lines)


def process_task(task_dir: Path, missing: list[str], apply: bool) -> dict:
    """Try to fill ``missing`` param names in ``task_dir/shapes.yaml``.

    Returns a dict {task_id, added, skipped, already_present, changed,
    error}. ``changed`` is True iff at least one new entry was inserted.
    When ``apply`` is True and ``changed`` is True, the file is rewritten
    atomically (tmp + rename) -- a crash mid-write leaves the file
    byte-identical.

    Write strategy is TEXT-LEVEL append, not yaml round-trip: we parse
    the existing shapes.yaml with PyYAML only for validation +
    idempotency, then append the new entries as raw text to the bottom
    of the file. This preserves top-of-file comments and the file's
    existing key order, which a ``yaml.safe_dump`` round-trip would
    discard (PyYAML does not retain comments). The bench's shapes.yaml
    files always have ``inputs:`` as the final top-level block, so
    appending at EOF lands inside the inputs: block.
    """
    result = {
        "task_id": task_dir.name,
        "added": [],
        "skipped": [],
        "already_present": [],
        "changed": False,
        "error": None,
    }
    shapes_path = task_dir / "shapes.yaml"
    if not shapes_path.is_file():
        result["error"] = f"{shapes_path} missing"
        return result
    try:
        text = shapes_path.read_text()
        doc = yaml.safe_load(text)
    except (OSError, yaml.YAMLError) as exc:
        result["error"] = f"yaml load failed: {exc}"
        return result
    if not isinstance(doc, dict):
        result["error"] = "shapes.yaml top level is not a mapping"
        return result
    if "inputs" not in doc:
        result["error"] = "shapes.yaml has no inputs: block (cannot append)"
        return result

    knobs = _knobs_universe(doc)
    existing = _existing_inputs(doc)
    new_entries: list[tuple[str, dict]] = []

    for name in missing:
        if name in existing:
            result["already_present"].append(name)
            continue
        entry = _build_entry(name, knobs)
        if entry is None:
            result["skipped"].append(name)
            continue
        new_entries.append((name, entry))
        result["added"].append(name)

    result["changed"] = bool(new_entries)
    if apply and new_entries:
        appended = "\n".join(_format_entry_text(n, e) for n, e in new_entries)
        new_text = text.rstrip("\n") + "\n" + appended + "\n"
        # Sanity-check the result still parses and the appended names
        # land under inputs: before writing.
        try:
            new_doc = yaml.safe_load(new_text)
        except yaml.YAMLError as exc:
            result["error"] = f"appended text would corrupt yaml: {exc}"
            return result
        for n, _ in new_entries:
            if n not in (new_doc.get("inputs") or {}):
                result["error"] = (
                    f"append landed outside inputs: -- aborting "
                    f"(name={n}, likely inputs: not the last top-level block)"
                )
                return result
        tmp = shapes_path.with_suffix(".yaml.tmp")
        tmp.write_text(new_text)
        tmp.replace(shapes_path)
    return result


def _find_task_dir(task_id: str) -> Path | None:
    for level in ("level1", "level2", "level3"):
        cand = TASKS_ROOT / level / task_id
        if cand.is_dir():
            return cand
    return None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--cache", type=Path, default=DEFAULT_CACHE,
        help=f"path to resolve_all.json (default: {DEFAULT_CACHE})",
    )
    ap.add_argument(
        "--apply", action="store_true",
        help="write changes (default: dry-run, no file mutation)",
    )
    ap.add_argument("--task", help="restrict to one task id")
    args = ap.parse_args(argv)

    verdicts = _load_cache(args.cache)
    # ``fully_filled``: every missing param was auto-filled, so the next
    # resolver sweep should advance this task out of shapes-incomplete
    # (into PASS or a different verdict_class).
    # ``partial``: at least one param was auto-filled but others still
    # need the LLM -- those tasks stay shapes-incomplete on the next
    # sweep but the LLM prompt has fewer asks.
    n_seen = n_added = n_skipped = n_already = n_error = 0
    n_full = n_partial = 0

    for v in verdicts:
        tid = v["task_id"]
        if args.task and tid != args.task:
            continue
        missing = _missing_params(v)
        if not missing:
            continue
        td = _find_task_dir(tid)
        if td is None:
            print(
                f"[skip] {tid}: task dir not found under tasks/",
                file=sys.stderr,
            )
            continue
        n_seen += 1
        r = process_task(td, missing, args.apply)
        if r["error"]:
            n_error += 1
            print(f"[error] {tid}: {r['error']}", file=sys.stderr)
            continue
        n_added += len(r["added"])
        n_skipped += len(r["skipped"])
        n_already += len(r["already_present"])
        if r["changed"] and not r["skipped"]:
            n_full += 1
        elif r["changed"]:
            n_partial += 1
        if r["changed"] and not r["skipped"]:
            tag = "WROTE-FULL" if args.apply else "WOULD-FULL"
        elif r["changed"]:
            tag = "WROTE-PARTIAL" if args.apply else "WOULD-PARTIAL"
        elif r["already_present"] and not r["skipped"]:
            tag = "ALREADY-COMPLETE"
        else:
            tag = "NEEDS-LLM"
        line = f"[{tag}] {tid}: added={r['added'] or '-'}"
        if r["skipped"]:
            line += f" llm-handles={r['skipped']}"
        if r["already_present"]:
            line += f" already={r['already_present']}"
        print(line)

    print(
        f"\n--- summary ---\n"
        f"tasks-seen={n_seen} "
        f"fully-filled={n_full} (expected to resolve next sweep) "
        f"partially-filled={n_partial} (LLM still has reduced ask) "
        f"rows-added={n_added} rows-llm-handles={n_skipped} "
        f"rows-already-present={n_already} errors={n_error} "
        f"(mode={'APPLY' if args.apply else 'DRY-RUN'})"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
