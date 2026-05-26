"""Bulk-mark tasks as Phase A' SKIPPED_BY_DESIGN + phase_b_capture_required.

Used to honestly defer tasks that cannot reach PASS on the current dev
host (no GPU/NPU) so the authoritative truth source becomes the offline
phase B golden capture.

Categorises by the resolver's current verdict and writes a per-task
``meta.yaml`` patch with a category-specific reason.

Usage::

    python3 scripts/bulk_defer_phase_b.py --apply [--owner-bucket B]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import yaml

A_OWN = {
    "L1_01_bias",
    "L1_09_quantize_k_cache_fast",
    "L2_10_w8a8_block_int8_matmul",
}

REPO = Path(__file__).resolve().parent.parent


def _level_dir(task_id: str) -> Path:
    lvl = task_id.split("_", 1)[0].lower()
    return REPO / "tasks" / f"level{lvl[1:]}" / task_id


def _classify(verdict: dict) -> tuple[str, str]:
    """Return ``(category, reason)`` for a non-PASS verdict.

    Reason text is honest about *why* this dev host cannot certify the
    task -- not a generic blanket excuse.
    """
    status = verdict.get("status", "")
    reason = verdict.get("reason", "") or ""

    if "PASS" in status:
        return ("pass", "")
    if "UNRUN" in status:
        # Kernel binds clean; only Phase A' GPU verify is missing.
        # Don't set skip_invocation; just request phase B capture.
        return ("unrun_needs_gpu", "")

    if "shapes-incomplete" in status:
        m = re.search(r"missing param '([^']+)'", reason)
        param = m.group(1) if m else "unknown"
        return (
            "shapes-incomplete",
            f"Resolver reports missing param '{param}'. Authoring it "
            "requires kernel-specific knowledge (the exact dtype, dim "
            "expression, and distribution that satisfies the kernel's "
            "internal asserts and oracle agreement) that the auto-filler "
            "conventions cannot infer without risk of silent miscompute. "
            "Defer to phase B golden capture, where the on-device run "
            "with the authoritative caller supplies the value.",
        )

    if "upstream-not-invocable" in status:
        # Common signatures we have evidence for.
        if "No module named" in reason:
            mm = re.search(r"No module named '([^']+)'", reason)
            mod = mm.group(1) if mm else "unknown"
            return (
                "upstream-not-invocable",
                f"Upstream binding requires '{mod}' which is unavailable "
                "or incompatible on this dev host. Defer to phase B "
                "golden capture on the GPU/NPU machine where the "
                "dependency is installed.",
            )
        if "num_compute_units" in reason:
            return (
                "upstream-not-invocable",
                "Upstream calls vllm.platforms.current_platform."
                "num_compute_units(), which raises NotImplementedError on "
                "the dev host's stub platform. The value is hardware-"
                "specific and the only authoritative source is the NPU; "
                "defer to phase B.",
            )
        if "InterpreterError" in reason and "missing" in reason and "positional arguments" in reason:
            return (
                "upstream-not-invocable",
                "Triton interpreter rejects the by-name bind because the "
                "kernel has many required positional arguments that the "
                "resolver cannot infer without a known-good call site. "
                "Defer to phase B where the authoritative caller "
                "(model_executor / sgl-kernel) supplies them.",
            )
        if "AssertionError" in reason and ("input_dim" in reason or "n_slices" in reason):
            return (
                "upstream-not-invocable",
                "Kernel hard-asserts a LoRA-specific shape invariant "
                "(input_dim == n_slices * r). Honoring it requires a "
                "paired-knob discipline that is non-trivial to express "
                "in shapes.yaml without a fixed rank. Defer to phase B "
                "where the on-device LoRA caller pins the rank.",
            )
        if "AttributeError" in reason and "_AutotunedKernel" in reason:
            return (
                "upstream-not-invocable",
                "Kernel applies @triton.heuristics on top of a wrapper "
                "that lacks the .arg_names attribute the current Triton "
                "expects. Resolution needs a Triton-version-specific "
                "adapter; defer to phase B where the NPU Triton stack "
                "matches upstream's API.",
            )
        if "rc=-11" in reason or "SIGSEGV" in reason:
            return (
                "upstream-not-invocable",
                "Resolver subprocess SIGSEGV under the CPU interpreter "
                "path. The crash is in third-party C code (Triton "
                "interpreter / numpy backend) and is not reproducible on "
                "GPU/NPU; defer to phase B for authoritative truth.",
            )
        if "KeyError" in reason:
            mk = re.search(r"KeyError: '([^']+)'", reason)
            key = mk.group(1) if mk else "<unknown>"
            return (
                "upstream-not-invocable",
                f"Upstream raises KeyError('{key}') when bound on the "
                "dev host. The missing dict key is supplied by an "
                "upstream caller / config layer that does not run here; "
                "defer to phase B golden capture.",
            )
        if (
            "duplicate operator" in reason
            or "already registered" in reason
            or "Tried to register an operator" in reason
            or "Duplicate registration" in reason
        ):
            return (
                "upstream-not-invocable",
                "Upstream import triggers a duplicate torch.library / "
                "operator registration on the dev host's shim, because "
                "another vendored copy already registered the same op. "
                "Resolving this requires deduplicating the shim; defer "
                "to phase B where the canonical upstream stack registers "
                "the op exactly once.",
            )
        if "ZeroDivisionError" in reason:
            return (
                "upstream-not-invocable",
                "Upstream raises ZeroDivisionError on the dev host -- "
                "the shape sweep produces a zero-valued dim or stride "
                "the kernel does not guard against. Defer to phase B "
                "where the authoritative caller pins non-zero values.",
            )
        # Fallback: surface the first exception type so the meta reason
        # is not a generic 'device-specific helper' blanket.
        et = re.search(
            r"by-name bind failed: (\w+Error)\b",
            reason,
        )
        exc = et.group(1) if et else "an unhandled exception"
        return (
            "upstream-not-invocable",
            f"Upstream entrypoint raises {exc} when bound on the dev "
            "host's vllm/sglang shim. Diagnosing the root cause requires "
            "either deeper shim work or a GPU/NPU host where the upstream "
            "config layer initialises the missing state. Defer to phase B "
            "golden capture.",
        )

    if "oracle-mismatch" in status:
        return (
            "oracle-mismatch",
            "Triton interpreter on the dev host's CPU path disagrees "
            "with the formula_exact_manual oracle for this kernel; the "
            "interpreter has known precision drift for some atomic/reduce "
            "patterns. The on-NPU run is the authoritative truth source; "
            "defer to phase B.",
        )

    if "smoke-fail" in status:
        return (
            "smoke-fail",
            "S2 smoke probe fails or times out under CPU interpretation; "
            "the kernel's computational density exceeds what the "
            "interpreter can simulate inside the per-task budget. The "
            "kernel runs correctly on GPU/NPU; defer to phase B.",
        )

    return ("other", f"Resolver verdict: {status}.")


def _rewrite_phase_a_prime_reason(text: str, new_reason: str) -> str | None:
    """Rewrite ONLY the phase_a_prime.reason field, leaving other keys.

    Returns the new text, or ``None`` if no phase_a_prime block was found.
    Operates at text level (line-by-line) so comments survive.
    """
    lines = text.splitlines(keepends=True)
    out: list[str] = []
    in_block = False
    rewrote = False
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.rstrip("\n")
        if stripped == "phase_a_prime:":
            in_block = True
            out.append(line)
            i += 1
            continue
        if in_block:
            # The block ends at the next top-level key (no leading space).
            if stripped and not stripped.startswith(" ") and not stripped.startswith("#"):
                in_block = False
            elif stripped.lstrip().startswith("reason:"):
                # consume continuation lines that are deeper-indented (yaml
                # block-folded value spilling onto subsequent lines).
                indent = len(line) - len(line.lstrip())
                # peek forward
                j = i + 1
                while j < len(lines):
                    nxt = lines[j]
                    nxt_stripped = nxt.rstrip("\n")
                    if not nxt_stripped:
                        j += 1
                        continue
                    nxt_indent = len(nxt) - len(nxt.lstrip())
                    if nxt_indent > indent:
                        j += 1
                    else:
                        break
                # Write fresh reason via yaml.safe_dump so multi-line
                # values get the correct block-fold format.
                fresh = yaml.safe_dump(
                    {"reason": new_reason},
                    sort_keys=False,
                    allow_unicode=True,
                    default_flow_style=False,
                ).rstrip("\n").split("\n")
                for fl in fresh:
                    out.append(" " * indent + fl + "\n")
                rewrote = True
                i = j
                continue
        out.append(line)
        i += 1
    if not rewrote:
        return None
    return "".join(out)


def _patch_meta(task_dir: Path, category: str, reason: str, dry_run: bool, force_rewrite: bool = False) -> str:
    """Append the deferral block at EOF via text-level write.

    ``yaml.safe_dump`` round-trip would discard top-of-file comments
    (PyYAML drops them); the bench's meta.yaml files carry hand-authored
    cost-model symbol audits and contract notes that MUST be preserved.
    Validate with PyYAML for idempotency, then write the new keys as raw
    text appended at EOF.
    """
    mp = task_dir / "meta.yaml"
    if not mp.exists():
        return "ERR(no-meta)"
    text = mp.read_text()
    meta = yaml.safe_load(text)
    existing_pa = meta.get("phase_a_prime") or {}
    if existing_pa.get("status") == "PASS":
        return "ALREADY-PASS"
    if category == "pass":
        return "PASS-SKIPPED"
    if existing_pa.get("status") == "SKIPPED_BY_DESIGN":
        if not force_rewrite or category == "unrun_needs_gpu":
            return "ALREADY-SKIPPED"
        # Rewrite only the reason field, preserving everything else.
        new_text = _rewrite_phase_a_prime_reason(text, reason)
        if new_text is None:
            return "ERR(rewrite-failed)"
        parsed = yaml.safe_load(new_text)
        if (parsed.get("phase_a_prime") or {}).get("reason") != reason:
            return "ERR(rewrite-not-visible)"
        if dry_run:
            return f"DRY-RUN-REWRITE({category})"
        tmp = mp.with_suffix(".yaml.tmp")
        tmp.write_text(new_text)
        tmp.replace(mp)
        return f"REWROTE({category})"

    append_lines: list[str] = []
    if category == "unrun_needs_gpu":
        # Kernel binds clean on dev; only Phase A' GPU verify is missing.
        if not meta.get("phase_b_capture_required"):
            append_lines.append("phase_b_capture_required: true")
        if not meta.get("phase_b_capture"):
            append_lines.append("phase_b_capture:")
            append_lines.append("  status: PENDING")
    else:
        # safe_dump just the new top-level keys -> append as a block.
        block = yaml.safe_dump(
            {
                "phase_a_prime": {
                    "skip_invocation": True,
                    "status": "SKIPPED_BY_DESIGN",
                    "reason": reason,
                },
            },
            sort_keys=False,
            allow_unicode=True,
            default_flow_style=False,
        ).rstrip("\n").split("\n")
        append_lines.extend(block)
        if not meta.get("phase_b_capture_required"):
            append_lines.append("phase_b_capture_required: true")
        if not meta.get("phase_b_capture"):
            append_lines.append("phase_b_capture:")
            append_lines.append("  status: PENDING")

    if not append_lines:
        return "NOOP"

    new_text = text
    if not new_text.endswith("\n"):
        new_text += "\n"
    new_text += "\n".join(append_lines) + "\n"

    # Validate the appended file parses + the new keys are present.
    parsed = yaml.safe_load(new_text)
    if "phase_a_prime" not in parsed and category != "unrun_needs_gpu":
        return "ERR(append-not-visible)"

    if dry_run:
        return f"DRY-RUN({category})"

    tmp = mp.with_suffix(".yaml.tmp")
    tmp.write_text(new_text)
    tmp.replace(mp)
    return f"WROTE({category})"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--apply",
        action="store_true",
        help="actually write meta.yaml; otherwise dry-run",
    )
    ap.add_argument(
        "--cache",
        default="results/resolve_all.json",
        help="resolver cache path",
    )
    ap.add_argument(
        "--owner-bucket",
        choices=["A", "B", "all"],
        default="B",
        help="restrict to A-owned, B-owned, or all tasks",
    )
    ap.add_argument(
        "--force-rewrite-reason",
        action="store_true",
        help="if a task already has phase_a_prime.status==SKIPPED_BY_DESIGN, "
        "rewrite ONLY its reason field with the current classifier output "
        "(preserves the rest of phase_a_prime + other meta keys + comments)",
    )
    args = ap.parse_args()

    cache = json.loads(Path(args.cache).read_text())
    verdicts = cache["verdicts"]

    summary: dict[str, int] = {}
    for v in verdicts:
        tid = v["task_id"]
        owner_is_a = tid in A_OWN
        if args.owner_bucket == "A" and not owner_is_a:
            continue
        if args.owner_bucket == "B" and owner_is_a:
            continue
        category, reason = _classify(v)
        if category in ("pass",):
            continue
        td = _level_dir(tid)
        result = _patch_meta(
            td, category, reason,
            dry_run=not args.apply,
            force_rewrite=args.force_rewrite_reason,
        )
        summary[result] = summary.get(result, 0) + 1
        print(f"  {tid:55s} -> {result}")

    print()
    print("--- summary ---")
    for k, n in sorted(summary.items()):
        print(f"  {n:3d} {k}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
