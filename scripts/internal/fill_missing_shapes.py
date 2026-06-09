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

    python3 scripts/internal/fill_missing_shapes.py             # dry-run
    python3 scripts/internal/fill_missing_shapes.py --apply     # write changes
    python3 scripts/internal/fill_missing_shapes.py --task L2_03_act_and_mul --apply

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
REPO_ROOT = THIS.parent.parent.parent
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
    # Cascade-revealed names from second-iteration resolve sweeps after
    # an earlier filler pass. Each appeared in at least one task's
    # `missing param '<name>'` reason after the first pass landed.
    "chunk_size":         (None, "scalar", "64"),        # mamba SSD / chunk scan
    "kv_cache_dtype":     (None, "scalar", "auto"),      # paged attn dtype tag
    "mrope_interleaved":  (None, "scalar", "false"),     # mrope variant flag
    "stack_num":          (None, "scalar", "1"),         # sgemm lora stack
    "n_slices":           (None, "scalar", "3"),         # qkv lora slices
    "num_slices":         (None, "scalar", "1"),         # chunked sgmv shrink
    "softplus_beta":      (None, "scalar", "1.0"),       # gated delta rule update
    "query_start_loc":    ("int32",   ("BP1",), None),   # cu_seqlens-style offsets (uses _NAME_TO_DEFAULT_DIST if added)
    "target_logits":      ("float16", ("B", "V"), "randn"),  # rejection sampling target
    "sampled_token_ids":  ("int32",   ("B",), "randint:0:V"),
    "topk_weights":       ("float32", ("BN", "K"), "rand"),  # MoE routing weights
    # cu_chunk_seqlens deliberately NOT in conventions: chunk axis differs
    # per task (e.g. L2_02 chunks on B, L2_03 chunks on C), so a global
    # dims/cap rule would be wrong. Per-task patch via LLM or manual edit.
    "max_input_len":      (None, "scalar", "128"),       # attention max seq
    "max_seqlen":         (None, "scalar", "128"),
    "num_logprobs":       (None, "scalar", "5"),
    "num_local_experts":  (None, "scalar", "8"),
    "current_draft_step": (None, "scalar", "0"),
    "num_draft_tokens":   (None, "scalar", "4"),
    "split_k_iters":      (None, "scalar", "1"),
    "logits_ptr":         ("float16", ("B", "V"), "randn"),
    "full_logits":        ("float16", ("B", "V"), "randn"),
    "gateup_output":      ("float16", ("BN", "I2"), None),
    "weight1":            ("float16", ("D",), "randn"),
    "weight2":            ("float16", ("D",), "randn"),
    "conv_state":         ("float16", ("B", "D", "S"), "randn"),
    "draft_tokens":       ("int32",   ("B", "K"), "randint:0:V"),
    "token_ids":          ("int32",   ("B", "N"), "randint:0:V"),
    "all_token_ids":      ("int32",   ("B", "N"), "randint:0:V"),
    "expert_indices":     ("int32",   ("BN", "K"), "randint:0:E"),
    "recv_topk_ids":      ("int32",   ("BN", "K"), "randint:0:E"),
    "recv_x_scale":       ("float32", ("BN",), "randn"),
}

_MISSING_PARAM_RE = re.compile(r"missing param '([^']+)'")
# Each `missing param` mention is followed by `for entrypoint '<name>'` in
# the resolver's reason field. Capturing both lets us scope filler additions
# to the chosen candidate's signature -- otherwise the resolver's reason can
# include missing-params from SIBLING candidates that share a vendored file
# (e.g. L1_15 reported `output` from a sibling kernel, not from the
# act_and_mul_triton host wrapper that the task actually picks). Codex
# review caught five such false additions in B1's first batch.
_MISSING_FOR_ENTRY_RE = re.compile(
    r"missing param '([^']+)' for entrypoint '([^']+)'"
)

# Composite dim symbols: (required_atomic_bases, emitted_dim_expression).
# The atomic bases are checked against the task's knobs universe (skip
# the convention if any base is missing). The emitted expression is the
# arithmetic form ``ascendbench.input_spec.inputs._resolve_dims`` actually
# evaluates -- inputs.py's whitelist evaluator accepts ``B+1`` / ``B*N``
# but NOT a bare token like ``BP1``; emitting ``[BP1]`` directly would
# leave the dim unmaterializable.
_COMPOSITE_BASES: dict[str, tuple[tuple[str, ...], str]] = {
    "BN":   (("B", "N"), "B*N"),
    "BP1":  (("B",), "B+1"),
    "I2":   (("I",), "2*I"),
    "DIM3": ((), "3"),
}

# Names that need a special ``dist`` because random integers wouldn't
# satisfy the kernel's invariant (monotone prefix-sum, offsets, etc.).
# Without this, ``inputs.py`` defaults absent ``dist`` to ``randn`` and
# the int cast produces arbitrary integers -- the kernel's
# ``cu_seqlens[i+1] - cu_seqlens[i]`` slicing then gives nonsense
# prefix-sum names like cu_seqlens get an explicit ``offsets:`` distribution.
_NAME_TO_DEFAULT_DIST: dict[str, str] = {
    "cu_seqlens":   "offsets:to=B*L",
    "seg_indptr":   "offsets:2",
}

# Names whose declared dtype would silently be ignored by
# ``ascendbench.input_spec.inputs._make_tensor`` (the ``offsets`` /
# ``cumcount`` / ``arange`` / ``full`` branches return early as
# ``torch.int64`` -- they never reach the dtype-cast step). Authoring
# a ``dtype`` for those rows would lie in the spec, so the auto-filler
# emits no ``dtype`` field for them and lets the underlying generator's
# native return type stand. TODO: fix _make_tensor to honour declared
# dtype, then this gate can be removed.
_DIST_PREFIXES_IGNORE_DTYPE: tuple[str, ...] = (
    "offsets", "cumcount", "arange", "full",
)


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


_TASK_ID_PREFIX_RE = re.compile(r"^L\d+_\d+_")


def _intended_entrypoint_candidates(task_id: str) -> list[str]:
    """Derive plausible "intended entrypoint" names from a task_id.

    Task IDs follow ``L<level>_<idx>_<intended_name>[_kernel]``.
    Stripping the prefix yields the conceptual intended entrypoint
    (e.g. ``L1_20_create_flashinfer_kv_indices_triton`` ->
    ``create_flashinfer_kv_indices_triton``). Some vendored kernels
    keep a ``_kernel`` suffix on the actual @triton.jit function;
    we generate both the stripped form and the with-suffix form.
    """
    stripped = _TASK_ID_PREFIX_RE.sub("", task_id)
    cands = [stripped]
    if not stripped.endswith("_kernel"):
        cands.append(stripped + "_kernel")
    if stripped.endswith("_kernel"):
        cands.append(stripped[: -len("_kernel")])
    return cands


def _choose_entrypoint(task_id: str, reason: str, tried: list[str]) -> str | None:
    """Pick the entrypoint name that THIS task intends, given the
    resolver's reason (which can mention many sibling candidates).

    Substring matching is bidirectional so naming-convention drift
    between the task_id and the upstream function name does not
    silently skip the task. Examples:
      - L1_15: task_id ``act_and_mul`` -> intended ``act_and_mul`` -> the
        upstream wrapper is named ``act_and_mul_triton`` (task_id is a
        substring of the wrapper) -> chosen = ``act_and_mul_triton``.
      - L3_19: task_id ``chunk_kda_scaled_dot_kkt_fwd_kernel_intra_sub_inter``
        -> tried contains ``chunk_kda_scaled_dot_kkt_fwd`` (a prefix of
        the task_id) -> chosen = ``chunk_kda_scaled_dot_kkt_fwd``.
      - L1_32: task_id ``create_flashinfer_kv_indices_triton`` is not a
        substring of any name in tried, and no tried name is a substring
        of it -> chosen = None -> skip the task (no auto-fill is far
        safer than wrong auto-fill; a blanket ``tried[0]`` fallback would
        scope additions to a sibling kernel and produce false rows).

    Priority:
      1. Exact match: any tried candidate equals an intended form.
      2. Bidirectional substring match against tried (intended in tried[i]
         OR tried[i] in intended).
      3. Bidirectional substring match against entrypoint mentions in
         the reason (covers candidates that didn't make it into tried[]
         for some reason).
      4. None -- skip rather than guess wrong.
    """
    intended = _intended_entrypoint_candidates(task_id)
    intended_set = set(intended)

    # 1. exact
    for cand in tried:
        if cand in intended_set:
            return cand

    # 2. bidirectional substring against tried[].
    # Reject if MORE THAN ONE distinct candidate matches -- the chooser
    # cannot prove which one the task intends. Example: a task_id like
    # ``_quantize_k_cache_fast`` is a substring of both ``_quantize_k_cache_fast``
    # and ``_quantize_k_cache_fast_separate``; arbitrarily returning the first
    # is just as wrong as guessing.
    sub_matches: list[str] = []
    for cand in tried:
        for intended_form in intended:
            if intended_form in cand or cand in intended_form:
                if cand not in sub_matches:
                    sub_matches.append(cand)
                break
    if len(sub_matches) == 1:
        return sub_matches[0]
    if len(sub_matches) > 1:
        return None  # ambiguous

    # 3. bidirectional substring against reason mentions
    mentioned: list[str] = []
    for (_n, entry) in _MISSING_FOR_ENTRY_RE.findall(reason):
        if entry not in mentioned:
            mentioned.append(entry)
    reason_matches: list[str] = []
    for entry in mentioned:
        for intended_form in intended:
            if intended_form in entry or entry in intended_form:
                if entry not in reason_matches:
                    reason_matches.append(entry)
                break
    if len(reason_matches) == 1:
        return reason_matches[0]
    return None


def _missing_params(verdict: dict) -> list[str]:
    """Return the missing params for the candidate the task INTENDS to
    bind to, not every missing-param mention across all tried siblings.

    The resolver's reason field can mention `missing param 'X' for
    entrypoint 'Y'` multiple times -- once per candidate it tried. If
    we naively collect every X we end up adding shapes.yaml rows for
    sibling candidates that share the vendored kernel.py file but are
    NOT what the task picks, leading to false rows. Scope additions to
    the task's intended entrypoint via the chooser instead.

    Use the task_id-derived intended entrypoint when it can be matched
    against the reason; skip the task otherwise (the LLM patch-shapes
    path will handle it later -- "no auto-fill" beats "wrong auto-fill").
    """
    if "shapes-incomplete" not in verdict.get("status", ""):
        return []
    reason = verdict.get("reason") or ""
    tried = verdict.get("tried") or []
    chosen = _choose_entrypoint(verdict.get("task_id", ""), reason, tried)
    if chosen is None:
        return []
    out: list[str] = []
    seen: set[str] = set()
    for name, entry in _MISSING_FOR_ENTRY_RE.findall(reason):
        if entry == chosen and name not in seen:
            out.append(name)
            seen.add(name)
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
    atomic dim bases are not all present in the task's knobs universe.

    Composite dim symbols (``BN`` / ``BP1`` / ``I2`` / ``DIM3``) are
    decomposed two ways:
      - the atomic bases (``("B", "N")`` for ``BN``) are checked against
        the knobs universe;
      - the emitted ``dims`` entry uses the arithmetic expression
        (``B*N``) that ``ascendbench.input_spec.inputs._resolve_dims`` can
        actually evaluate. Bare composite tokens are not understood by
        the input materialiser, so the dim would stay unmaterializable.
    """
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
            bases.update(_COMPOSITE_BASES[d][0])
        else:
            bases.add(d)
    if not bases.issubset(knobs):
        return None
    # Emit arithmetic form for composites so inputs.py can evaluate them.
    emitted_dims = [
        _COMPOSITE_BASES[d][1] if d in _COMPOSITE_BASES else d
        for d in dims
    ]
    entry: dict = {"dims": emitted_dims}
    # Per-name dist override (cu_seqlens / seg_indptr need monotone forms);
    # otherwise use the convention's declared dist.
    if name in _NAME_TO_DEFAULT_DIST:
        chosen_dist = _NAME_TO_DEFAULT_DIST[name]
    else:
        chosen_dist = dist
    if chosen_dist is not None:
        entry["dist"] = chosen_dist
    # Only declare dtype when the underlying dist generator will honour
    # it. Distributions like ``offsets`` short-circuit to int64 and
    # ignore the declared dtype; emitting the field would be a
    # spec-vs-runtime lie.
    if dtype is not None and not (
        chosen_dist is not None
        and chosen_dist.split(":", 1)[0] in _DIST_PREFIXES_IGNORE_DTYPE
    ):
        entry["dtype"] = dtype
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
            # Nothing the convention table can bind for this task's chosen
            # entrypoint (either truly no missing params, or the chooser
            # could not match the intended entrypoint against the reason).
            # The LLM patch-shapes path handles it -- "no auto-fill" beats
            # "wrong auto-fill".
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
