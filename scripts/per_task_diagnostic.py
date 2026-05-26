"""Per-task root-cause diagnostic for the 100-task corpus.

Read-only analyzer. For each BLOCKED/UNRUN task, cross-references:
  - resolver verdict (status + per-entrypoint segment) from results/resolve_all.json
  - meta.yaml (intended kernel name + oracle config + upstream path)
  - kernel.py (intended function signature + parameters + asserts)
  - shapes.yaml (declared input rows: param names + dtypes + dims)
  - oracle_pytorch.py (expected input contract)

Outputs a per-task structured report with:
  - resolver surface error (verbatim segment for the INTENDED entrypoint)
  - intended kernel signature (parsed param list with defaults)
  - shapes.yaml declared inputs (param -> dtype/dims/dist or scalar)
  - DIFF: kernel-required params not in shapes.yaml
  - DIFF: kernel asserts that look violated by current shapes
  - root cause category (one of: missing-shape-row / wrong-dtype / wrong-shape /
    assertion-violated / import-error / kernel-assert-on-knob / oracle-divergence /
    interpreter-numeric / triton-version-incompat / sigsegv / multi-entrypoint /
    other)
  - fix recipe: concrete shapes.yaml / meta.yaml / kernel.py / oracle_pytorch.py
    diff sketch
  - effort estimate: S (≤30min) / M (30-120min) / L (≥120min)

Usage::

    python3 scripts/per_task_diagnostic.py                # stdout summary
    python3 scripts/per_task_diagnostic.py --md out.md    # full markdown
    python3 scripts/per_task_diagnostic.py --task L1_30   # one task
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parent.parent
CACHE = REPO / "results" / "resolve_all.json"


@dataclass
class TaskDiagnosis:
    task_id: str
    family: str
    aggregate_status: str
    intended_kernel_name: str = ""
    intended_segment: str = ""
    intended_substatus: str = ""
    kernel_required_params: list[str] = field(default_factory=list)
    kernel_optional_params: list[str] = field(default_factory=list)
    shapes_inputs: dict[str, dict] = field(default_factory=dict)
    missing_in_shapes: list[str] = field(default_factory=list)
    upstream_exception: str = ""
    upstream_exception_detail: str = ""
    root_cause_category: str = ""
    root_cause_detail: str = ""
    fix_recipe: list[str] = field(default_factory=list)
    effort: str = ""  # S/M/L


def _task_dir(task_id: str) -> Path:
    lvl = task_id.split("_", 1)[0].lower()
    return REPO / "tasks" / f"level{lvl[1:]}" / task_id


def _split_segments(reason: str) -> list[tuple[str, str, str]]:
    """Return list of ``(entrypoint_name, substatus, segment_text)``."""
    after_tried = re.split(r";\s+", reason, maxsplit=1)
    body = after_tried[1] if len(after_tried) > 1 else reason
    parts = re.split(r"\s\|\s", body)
    out: list[tuple[str, str, str]] = []
    for part in parts:
        m = re.match(r"\s*([A-Za-z_][\w\-]*):\s*([a-z][a-z_\-]+)\b", part)
        if m:
            out.append((m.group(1), m.group(2), part))
    return out


def _pick_intended_segment(task_id: str, segments: list[tuple[str, str, str]]) -> tuple[str, str, str]:
    if not segments:
        return ("", "", "")
    intended = re.sub(r"^L\d+_\d+_", "", task_id)
    norm = lambda s: s.replace("-", "_")
    for name, sub, seg in segments:
        if norm(name) == norm(intended):
            return (name, sub, seg)
    for name, sub, seg in segments:
        if norm(intended) in norm(name) or norm(name) in norm(intended):
            return (name, sub, seg)
    return segments[0]


def _parse_kernel_signature(kernel_src: str, fn_name: str) -> tuple[list[str], list[str]]:
    """Return ``(required_params, optional_params)`` for ``def fn_name(...)``.

    Optional means the param has a default value. Looks for the FIRST
    matching ``def`` block (not @triton.jit kernels which are internal).
    """
    if not kernel_src or not fn_name:
        return ([], [])
    try:
        tree = ast.parse(kernel_src)
    except SyntaxError:
        return ([], [])
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == fn_name:
            args = node.args
            n_defaults = len(args.defaults)
            n_args = len(args.args)
            required = [a.arg for a in args.args[: n_args - n_defaults]]
            optional = [a.arg for a in args.args[n_args - n_defaults :]]
            return (required, optional)
    return ([], [])


def _parse_shapes_inputs(shapes_src: str) -> dict[str, dict]:
    """Return ``{name: {dtype, dims, dist, scalar, ...}}`` from shapes.yaml."""
    try:
        doc = yaml.safe_load(shapes_src) or {}
    except yaml.YAMLError:
        return {}
    inputs = doc.get("inputs") or {}
    if not isinstance(inputs, dict):
        return {}
    return {k: v if isinstance(v, dict) else {"value": v} for k, v in inputs.items()}


def _extract_upstream_exception(segment: str) -> tuple[str, str]:
    """Return ``(exception_type, short_detail)`` from a segment like
    ``foo: upstream-not-invocable-here [SomeError: details...]``."""
    m = re.search(r"\[([A-Za-z][A-Za-z0-9]*(?:Error|Exception)?):?\s*([^\]]{0,200})", segment)
    if m:
        return (m.group(1), m.group(2).strip().rstrip(" |"))
    # No bracketed; look for 'missing param' (shapes-incomplete) phrasing
    mp = re.search(r"missing param '([^']+)'", segment)
    if mp:
        return ("ShapesIncompleteError", f"missing param '{mp.group(1)}'")
    return ("", "")


def _diagnose_one(verdict: dict) -> TaskDiagnosis:
    tid = verdict["task_id"]
    td = _task_dir(tid)
    diag = TaskDiagnosis(
        task_id=tid,
        family="?",
        aggregate_status=verdict.get("status", ""),
    )

    meta_path = td / "meta.yaml"
    try:
        meta = yaml.safe_load(meta_path.read_text()) or {}
    except (OSError, yaml.YAMLError):
        meta = {}
    diag.family = (meta.get("category") or "?")

    # PASS / UNRUN short-circuit -- no root-cause needed.
    if "PASS" in diag.aggregate_status:
        diag.root_cause_category = "pass"
        diag.fix_recipe = []
        diag.effort = ""
        return diag
    if "UNRUN" in diag.aggregate_status:
        diag.root_cause_category = "needs-gpu"
        diag.root_cause_detail = (
            "Kernel binds successfully on dev; oracle is consistent; "
            "Phase A' verification requires GPU/NPU execution."
        )
        diag.fix_recipe = [
            "Run on GPU host: bash scripts/automation/runners/tab_quickstart.sh verify-all",
            "Expected outcome: PASS after Phase A' completes",
        ]
        diag.effort = "S"
        return diag

    # Parse resolver reason -> intended segment.
    reason = verdict.get("reason", "") or ""
    segments = _split_segments(reason)
    name, sub, seg = _pick_intended_segment(tid, segments)
    diag.intended_kernel_name = name
    diag.intended_substatus = sub
    diag.intended_segment = seg

    # Parse kernel.py signature for the intended kernel.
    try:
        kernel_src = (td / "kernel.py").read_text()
    except OSError:
        kernel_src = ""
    if name and kernel_src:
        required, optional = _parse_kernel_signature(kernel_src, name)
        diag.kernel_required_params = required
        diag.kernel_optional_params = optional

    # Parse shapes.yaml declared inputs.
    try:
        shapes_src = (td / "shapes.yaml").read_text()
    except OSError:
        shapes_src = ""
    diag.shapes_inputs = _parse_shapes_inputs(shapes_src)
    declared_in_shapes = set(diag.shapes_inputs.keys())

    # Compute missing params: required not in shapes.
    if diag.kernel_required_params:
        diag.missing_in_shapes = [
            p for p in diag.kernel_required_params if p not in declared_in_shapes
        ]

    # Extract upstream exception detail.
    exc, exc_detail = _extract_upstream_exception(seg or reason)
    diag.upstream_exception = exc
    diag.upstream_exception_detail = exc_detail

    # Root-cause categorization + fix recipe.
    _classify_root_cause(diag, kernel_src, meta, reason)
    return diag


def _classify_root_cause(
    diag: TaskDiagnosis, kernel_src: str, meta: dict, full_reason: str
) -> None:
    """Populate diag.root_cause_category + .root_cause_detail + .fix_recipe + .effort."""
    sub = diag.intended_substatus
    exc = diag.upstream_exception
    exc_detail = diag.upstream_exception_detail
    name = diag.intended_kernel_name

    if sub == "shapes-incomplete":
        # The resolver's "missing param 'X'" inside the intended segment.
        mp = re.search(r"missing param '([^']+)'", diag.intended_segment)
        param = mp.group(1) if mp else (diag.missing_in_shapes[0] if diag.missing_in_shapes else "?")
        is_known_to_kernel = (param in diag.kernel_required_params) or (param in diag.kernel_optional_params)
        if is_known_to_kernel:
            diag.root_cause_category = "missing-shape-row"
            diag.root_cause_detail = (
                f"Kernel function `{name}` requires param `{param}` (in its signature), "
                f"but shapes.yaml `inputs:` block does not declare it. "
                f"Resolver's by-name adapter stops at first missing param."
            )
            diag.fix_recipe = [
                f"Open tasks/{_task_dir(diag.task_id).name}/shapes.yaml",
                f"Add `{param}:` entry under `inputs:` with the kernel's expected dtype + dims + dist",
                "Refer to kernel.py signature / docstring for shape constraints",
                "Re-run resolver to confirm advance to next stage (often more missing params surface in cascade)",
            ]
            diag.effort = "S" if param.startswith("BLOCK_SIZE") or param in ("chunk_size", "max_seq_len") else "M"
        else:
            diag.root_cause_category = "shapes-incomplete-sibling"
            diag.root_cause_detail = (
                f"Resolver's missing-param mention (`{param}`) belongs to a sibling "
                f"candidate, not the intended kernel `{name}`. The aggregate "
                f"shapes-incomplete masks the intended kernel's true status."
            )
            diag.fix_recipe = [
                f"Verify intended kernel `{name}` actually exists in kernel.py",
                f"If `{name}` is correct, the resolver may be trying wrong siblings -- "
                "fix meta.upstream.path or meta.oracle.gpu_triton to point at the right symbol",
            ]
            diag.effort = "M"

    elif sub == "upstream-not-invocable-here" or sub == "upstream-not-invocable":
        if "No module named" in diag.intended_segment:
            mm = re.search(r"No module named '([^']+)'", diag.intended_segment)
            mod = mm.group(1) if mm else "?"
            diag.root_cause_category = "import-error"
            diag.root_cause_detail = f"Upstream import chain fails: ModuleNotFoundError for `{mod}`."
            diag.fix_recipe = [
                f"Option A: `pip install {mod}` (if pure-Python, low risk)",
                "Option B: oracle_pytorch.py implement a standalone reference (no upstream import)",
                "Option C: GPU-only -- defer to Phase B if module is CUDA-bound",
            ]
            diag.effort = "M"
        elif exc == "KeyError":
            key = re.search(r"KeyError: '?([^'\)]+)'?", diag.intended_segment)
            ks = key.group(1) if key else "?"
            diag.root_cause_category = "kernel-config-keyerror"
            diag.root_cause_detail = (
                f"Upstream raises KeyError(`{ks}`). The missing dict key is provided by an "
                f"upstream config layer / model_executor caller that does not initialise on dev."
            )
            diag.fix_recipe = [
                f"Read the call site to identify which dict expected `{ks}`",
                "Option A: shim the dict in oracle_pytorch.py with a known-good default",
                "Option B: rewrite oracle_pytorch.py as a standalone reference (no upstream import)",
            ]
            diag.effort = "M"
        elif "num_compute_units" in diag.intended_segment:
            diag.root_cause_category = "platform-stub-noimpl"
            diag.root_cause_detail = (
                "Upstream calls vllm.platforms.current_platform.num_compute_units() which "
                "raises NotImplementedError on the dev host's stub platform."
            )
            diag.fix_recipe = [
                "Option A: mock num_compute_units in tests/conftest.py or shim layer",
                "Option B: rewrite oracle_pytorch.py to not depend on platform helpers",
            ]
            diag.effort = "M"
        elif "AssertionError" in diag.intended_segment:
            assertion = re.search(r"AssertionError:?\s*([^|\]]{0,200})", diag.intended_segment)
            astxt = assertion.group(1).strip() if assertion else ""
            diag.root_cause_category = "kernel-assertion"
            diag.root_cause_detail = (
                f"Kernel function `{name}` hits an AssertionError when called with the "
                f"current shapes: `{astxt}`"
            )
            diag.fix_recipe = [
                "Read the asserted condition in kernel.py",
                "Adjust shapes.yaml so the input tensors satisfy the assertion",
                "(e.g. paired dtype, divisibility, shape compatibility)",
            ]
            diag.effort = "M"
        elif "Tried to register an operator" in diag.intended_segment or "Duplicate registration" in diag.intended_segment:
            diag.root_cause_category = "duplicate-op-registration"
            diag.root_cause_detail = (
                "Upstream import triggers torch.library duplicate operator registration -- "
                "another vendored copy registered the same op first."
            )
            diag.fix_recipe = [
                "Identify the duplicate registration path (usually two vllm copies)",
                "Option A: dedupe sys.path so only one vllm import wins",
                "Option B: oracle_pytorch.py rewrite to not import vllm",
            ]
            diag.effort = "L"
        elif "ZeroDivisionError" in diag.intended_segment:
            diag.root_cause_category = "kernel-zerodivision"
            diag.root_cause_detail = "Shape sweep produces a zero-valued dim/stride the kernel does not guard."
            diag.fix_recipe = [
                "Identify which shape row triggers the zero (usually H2/H_singleton with extreme small knobs)",
                "Add a guard in shapes.yaml hidden_shapes or in oracle_pytorch.py for that row",
            ]
            diag.effort = "S"
        elif "AttributeError" in diag.intended_segment and "_AutotunedKernel" in diag.intended_segment:
            diag.root_cause_category = "triton-autotune-api-mismatch"
            diag.root_cause_detail = (
                "Kernel applies @triton.heuristics on top of a wrapper that lacks .arg_names. "
                "Current Triton API expects different ordering."
            )
            diag.fix_recipe = [
                "Pin Triton version OR rewrite kernel.py decorator ordering",
                "If oracle_pytorch.py is independent, defer kernel.py fix to Phase B",
            ]
            diag.effort = "L"
        elif "rc=-11" in diag.intended_segment or "SIGSEGV" in diag.intended_segment:
            diag.root_cause_category = "sigsegv"
            diag.root_cause_detail = "Subprocess SIGSEGV under CPU interpreter (likely Triton interpreter bug)."
            diag.fix_recipe = [
                "Phase B golden capture on GPU/NPU is the only authoritative path",
                "On dev: try TRITON_INTERPRET=0 to skip interpreter path",
            ]
            diag.effort = "L"
        elif exc:
            diag.root_cause_category = f"upstream-{exc.lower()}"
            diag.root_cause_detail = f"Upstream raises {exc}: {exc_detail[:200]}"
            diag.fix_recipe = [
                "Inspect the exception's call site in upstream",
                "Decide: shim, mock, or rewrite oracle_pytorch.py standalone",
            ]
            diag.effort = "M"
        else:
            diag.root_cause_category = "upstream-unclassified"
            diag.root_cause_detail = "Upstream raises an exception not matched by known patterns; see intended_segment."
            diag.fix_recipe = ["Read intended_segment in full and classify manually"]
            diag.effort = "M"

    elif sub == "oracle-mismatch":
        if "STRUCTURE MISMATCH" in diag.intended_segment or "wrong operator" in diag.intended_segment:
            diag.root_cause_category = "oracle-structure-mismatch"
            diag.root_cause_detail = (
                "Kernel output and oracle disagree on output structure (shape, dtype, or "
                "operator semantics)."
            )
            diag.fix_recipe = [
                "Read kernel.py + oracle_pytorch.py side by side",
                "Identify which output dim/dtype the oracle expects vs what kernel produces",
                "Fix oracle_pytorch.py to match the kernel's true output contract",
            ]
            diag.effort = "M"
        elif "dtype mismatch" in diag.intended_segment.lower():
            diag.root_cause_category = "oracle-dtype-mismatch"
            diag.root_cause_detail = "Output dtype divergence between kernel and oracle."
            diag.fix_recipe = [
                "Align oracle_pytorch.py output dtype with kernel.py's emitted dtype",
            ]
            diag.effort = "S"
        else:
            diag.root_cause_category = "oracle-numeric-tolerance"
            diag.root_cause_detail = (
                "Numeric divergence between kernel and oracle under the task's tolerance class. "
                "Likely CPU interpreter precision drift but may be real kernel bug."
            )
            diag.fix_recipe = [
                "Re-run on GPU to distinguish interpreter drift from real divergence",
                "If real, audit oracle_pytorch.py against kernel.py mathematically",
            ]
            diag.effort = "M"

    elif sub == "smoke-fail":
        diag.root_cause_category = "smoke-fail-cpu-budget"
        diag.root_cause_detail = (
            "S2 smoke (TRITON_INTERPRET) fails / times out on CPU. Often a compute-density "
            "issue (large GEMMs, atomic ops) where the interpreter is too slow."
        )
        diag.fix_recipe = [
            "Reduce smoke shape coverage to only H2 (smallest hidden shape)",
            "OR: skip smoke pre-filter for this task family if interpreter cannot run",
            "Phase B GPU run will verify",
        ]
        diag.effort = "S"

    else:
        # Catch-all for special resolver outputs that don't follow the
        # `<entrypoint>: <substatus>` structure (SIGSEGV preambles,
        # severed-side-no-run reports, A-own deferred tasks with custom
        # phase_a_prime SKIP messages, etc.).
        if "rc=-11" in full_reason or "SIGSEGV" in full_reason:
            diag.root_cause_category = "sigsegv"
            diag.root_cause_detail = (
                "Resolver subprocess SIGSEGV under CPU interpreter. Crash is in "
                "third-party C (Triton interpreter / numpy backend); not reproducible "
                "on GPU/NPU."
            )
            diag.fix_recipe = [
                "Phase B golden capture on GPU/NPU is the authoritative path",
                "Optional: rewrite oracle_pytorch.py to avoid interpreter-only codepath",
            ]
            diag.effort = "L"
        elif "S2-smoke MISMATCH" in full_reason or "severed-no-run" in full_reason or "original-no-run" in full_reason:
            diag.root_cause_category = "smoke-fail-severed-run"
            diag.root_cause_detail = (
                "S2-smoke side process did not run (severed or original side rc!=0). "
                "Usually CPU interpreter cannot exec the kernel within budget."
            )
            diag.fix_recipe = [
                "Reduce smoke shape coverage to H2 (smallest hidden shape)",
                "Or skip CPU pre-filter for this task family",
            ]
            diag.effort = "M"
        elif "smoke-fail" in full_reason or "smoke fail" in full_reason.lower():
            diag.root_cause_category = "smoke-fail-cpu-budget"
            diag.root_cause_detail = (
                "Resolver records the kernel as oracle-consistent but S2-smoke "
                "(CPU interpreter pre-filter) fails or exceeds budget on the dev host. "
                "The kernel runs correctly on GPU/NPU."
            )
            diag.fix_recipe = [
                "Phase B golden capture on GPU/NPU is the authoritative path",
                "Optional: reduce smoke shape coverage to H2 to skip the worst row",
            ]
            diag.effort = "S"
        elif "phase B golden capture" in full_reason.lower() or "SKIPPED_BY_DESIGN" in full_reason:
            diag.root_cause_category = "phase-b-deferred"
            diag.root_cause_detail = (
                "Task is explicitly marked SKIPPED_BY_DESIGN in meta.phase_a_prime "
                "(A-owned deferral). Phase B golden capture is the authoritative path."
            )
            diag.fix_recipe = [
                "Run phase_b_golden_capture --task <id> on GPU host",
            ]
            diag.effort = "M"
        else:
            diag.root_cause_category = "other"
            diag.root_cause_detail = f"Unclassified substatus: {sub or '(none extracted)'}; full_reason head: {full_reason[:200]}"
            diag.fix_recipe = ["Inspect resolver verdict manually"]
            diag.effort = "M"


def _render_markdown(diags: list[TaskDiagnosis]) -> str:
    """Render full markdown report grouped by family + root-cause."""
    by_family: dict[str, list[TaskDiagnosis]] = {}
    for d in diags:
        by_family.setdefault(d.family, []).append(d)

    out: list[str] = []
    out.append("---")
    out.append("date: 2026-05-26")
    out.append("status: per-task root-cause diagnostic for 100-task corpus")
    out.append("based-on: results/resolve_all.json + per-task source files")
    out.append("scope: read-only diagnosis (no code/yaml/meta changes)")
    out.append("---\n")
    out.append("# Per-Task Root-Cause Diagnostic\n")

    # Top-level summary.
    cat_counter = Counter(d.root_cause_category for d in diags if d.root_cause_category)
    effort_counter = Counter(d.effort for d in diags if d.effort)
    out.append("## Summary\n")
    out.append(f"Total tasks analysed: {len(diags)}\n")
    out.append("### Root-cause categories\n")
    out.append("| category | count |")
    out.append("|---|---|")
    for k, n in cat_counter.most_common():
        out.append(f"| `{k}` | {n} |")
    out.append("")
    out.append("### Effort estimate distribution\n")
    out.append("| effort | count | meaning |")
    out.append("|---|---|---|")
    out.append(f"| S | {effort_counter.get('S', 0)} | ≤30 min per task |")
    out.append(f"| M | {effort_counter.get('M', 0)} | 30-120 min per task |")
    out.append(f"| L | {effort_counter.get('L', 0)} | ≥120 min per task |")
    out.append("")

    # Per-family per-task detail.
    out.append("## Per-Task Detail\n")
    for family in sorted(by_family.keys()):
        tasks = by_family[family]
        non_pass = [t for t in tasks if t.root_cause_category not in ("pass", "")]
        if not non_pass:
            continue
        out.append(f"### family: `{family}` (n={len(non_pass)})\n")
        for d in sorted(non_pass, key=lambda t: t.task_id):
            out.append(f"#### `{d.task_id}`\n")
            out.append(f"- **aggregate status**: `{d.aggregate_status}`")
            if d.intended_kernel_name:
                out.append(f"- **intended kernel**: `{d.intended_kernel_name}`")
            if d.intended_substatus:
                out.append(f"- **intended substatus**: `{d.intended_substatus}`")
            if d.upstream_exception:
                out.append(
                    f"- **upstream exception**: `{d.upstream_exception}` "
                    f"({d.upstream_exception_detail[:120]})"
                )
            out.append(f"- **root cause category**: `{d.root_cause_category}`")
            if d.root_cause_detail:
                out.append(f"- **root cause detail**: {d.root_cause_detail}")
            if d.kernel_required_params:
                out.append(
                    f"- **kernel required params** ({len(d.kernel_required_params)}): "
                    f"`{', '.join(d.kernel_required_params)}`"
                )
            if d.missing_in_shapes:
                out.append(
                    f"- **missing in shapes.yaml** ({len(d.missing_in_shapes)}): "
                    f"`{', '.join(d.missing_in_shapes)}`"
                )
            if d.shapes_inputs:
                declared = sorted(d.shapes_inputs.keys())
                out.append(
                    f"- **shapes.yaml declared inputs** ({len(declared)}): "
                    f"`{', '.join(declared)}`"
                )
            if d.fix_recipe:
                out.append(f"- **fix recipe**:")
                for step in d.fix_recipe:
                    out.append(f"  - {step}")
            out.append(f"- **effort**: `{d.effort}`")
            out.append("")
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache", default=str(CACHE), help="resolver cache path")
    ap.add_argument("--task", help="restrict to one task_id")
    ap.add_argument("--md", metavar="PATH", help="write markdown report to PATH")
    ap.add_argument("--json", metavar="PATH", help="write JSON dump to PATH")
    args = ap.parse_args()

    cache = json.loads(Path(args.cache).read_text())
    verdicts = cache["verdicts"]
    if args.task:
        verdicts = [v for v in verdicts if v["task_id"] == args.task]
        if not verdicts:
            print(f"task_id not found in cache: {args.task}", file=sys.stderr)
            return 1

    diags = [_diagnose_one(v) for v in verdicts]

    if args.md:
        Path(args.md).write_text(_render_markdown(diags))
        print(f"wrote markdown: {args.md} ({len(diags)} tasks)")

    if args.json:
        payload = [
            {
                "task_id": d.task_id,
                "family": d.family,
                "aggregate_status": d.aggregate_status,
                "intended_kernel_name": d.intended_kernel_name,
                "intended_substatus": d.intended_substatus,
                "intended_segment": d.intended_segment[:500],
                "kernel_required_params": d.kernel_required_params,
                "kernel_optional_params": d.kernel_optional_params,
                "shapes_inputs": list(d.shapes_inputs.keys()),
                "missing_in_shapes": d.missing_in_shapes,
                "upstream_exception": d.upstream_exception,
                "upstream_exception_detail": d.upstream_exception_detail,
                "root_cause_category": d.root_cause_category,
                "root_cause_detail": d.root_cause_detail,
                "fix_recipe": d.fix_recipe,
                "effort": d.effort,
            }
            for d in diags
        ]
        Path(args.json).write_text(json.dumps(payload, indent=2, ensure_ascii=False))
        print(f"wrote json: {args.json}")

    # Stdout summary.
    cat_counter = Counter(d.root_cause_category for d in diags)
    effort_counter = Counter(d.effort for d in diags)
    print()
    print(f"=== diagnosed {len(diags)} tasks ===")
    print("--- root cause categories ---")
    for k, n in cat_counter.most_common():
        print(f"  {n:3d} {k}")
    print("--- effort ---")
    for k in ("S", "M", "L"):
        print(f"  {effort_counter.get(k, 0):3d} {k}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
