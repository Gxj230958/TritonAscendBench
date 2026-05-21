#!/usr/bin/env python3
"""generate_task_scaffolds.py — batch-create task directories.

Parses the L1/L2/L3 markdown tables in docs/npu/benchmark_select.md and
generates a scaffold under tasks/level{1,2,3}/<task_id>/ for every entry
that does not already exist on disk.

The scaffold passes scripts/validate_tasks.py and the dataset loader, but
each file carries explicit TODO markers. kernel.py / oracle_pytorch.py
must be filled in by hand (freeze the upstream GPU triton kernel) before
the task is graded.

Idempotent: directories that already exist are skipped.
"""
from __future__ import annotations

import argparse
import hashlib
import re
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
TASKS_ROOT = REPO_ROOT / "tasks"
SELECT_MD = REPO_ROOT.parent / "docs" / "npu" / "benchmark_select.md"

DIFFICULTY_MAP = {
    "低": "low",
    "中低": "medium",
    "中": "medium",
    "中高": "medium",
    "高": "high",
}

REPO_MAP = {
    "vLLM": "vLLM",
    "SGLang": "SGLang",
    "FBGEMM": "FBGEMM",
    "字节中台编译": "Bytedance_internal",
}

# Existing PoCs — directory names diverge from the default pattern.
# Keys are markdown IDs (e.g. "L1-01"), values are existing dir names.
KNOWN_POC_DIRS = {
    "L1-01": "L1_01_bias_kernel",
    "L1-12": "L1_12_rms_norm_kernel",
    "L1-16": "L1_16_silu_and_mul",
    "L2-10": "L2_10_w8a8_block_int8_matmul",
    "L3-04": "L3_04_fwd_kernel",
}

# Category heuristic: substring → category enum.
# Conservative — falls back to "other" if no match.
CATEGORY_RULES: list[tuple[str, str]] = [
    ("rms", "normalization"),
    ("layer_norm", "normalization"),
    ("layernorm", "normalization"),
    ("norm", "normalization"),
    ("softmax", "sampling"),
    ("sample", "sampling"),
    ("logprob", "sampling"),
    ("min_p", "sampling"),
    ("topk", "sampling"),
    ("top_k", "sampling"),
    ("temperature", "sampling"),
    ("bias", "sampling"),
    ("logit", "sampling"),
    ("penalt", "sampling"),
    ("gumbel", "sampling"),
    ("reject", "sampling"),
    ("grammar", "sampling"),
    ("bitmask", "sampling"),
    ("paged", "attention"),
    ("flash", "attention"),
    ("attention", "attention"),
    ("attn", "attention"),
    ("mla", "attention"),
    ("kv", "attention"),
    ("page", "attention"),
    ("rope", "attention"),
    ("mrope", "attention"),
    ("sdpa", "attention"),
    ("eagle", "spec_decode"),
    ("draft", "spec_decode"),
    ("moe", "moe"),
    ("expert", "moe"),
    ("router", "moe"),
    ("ep_", "moe"),
    ("deepep", "moe"),
    ("permute", "moe"),
    ("scatter", "moe"),
    ("gather", "moe"),
    ("matmul", "gemm"),
    ("gemm", "gemm"),
    ("bmm", "gemm"),
    ("lora", "gemm"),
    ("awq", "quantization"),
    ("dequant", "quantization"),
    ("quant", "quantization"),
    ("int8", "quantization"),
    ("fp8", "quantization"),
    ("w4a8", "quantization"),
    ("w8a8", "quantization"),
    ("silu", "elementwise"),
    ("gelu", "elementwise"),
    ("act_and", "elementwise"),
    ("softcap", "elementwise"),
    ("residual", "elementwise"),
    ("merge", "elementwise"),
    ("concat", "elementwise"),
    ("cast", "elementwise"),
    ("conv", "convolution"),
    ("mamba", "ssm"),
    ("chunk", "ssm"),
    ("scan", "ssm"),
    ("cumsum", "ssm"),
    ("recurrent", "ssm"),
    ("delta_rule", "ssm"),
    ("gdn", "ssm"),
    ("gla", "ssm"),
    ("kda", "ssm"),
    ("fla", "ssm"),
    ("linear_attn", "ssm"),
    ("state", "ssm"),
    ("tril", "ssm"),
    ("jagged", "attention"),
    ("page_table", "attention"),
    ("slot", "attention"),
    ("index", "attention"),
    ("block_table", "attention"),
    ("seg_indptr", "moe"),
    ("src2dst", "moe"),
]

# Tolerance class heuristic per category.
TOL_BY_CATEGORY = {
    "sampling": "sampling_rng",
    "attention": "attention_flash",
    "moe": "atomic_scatter",
    "gemm": "vector_reduce",
    "quantization": "vector_reduce",
    "normalization": "vector_reduce",
    "elementwise": "elementwise",
    "convolution": "vector_reduce",
    "ssm": "recurrent_chunk",
    "spec_decode": "sampling_rng",
    "other": "elementwise",
}


@dataclass
class RawRow:
    md_id: str            # "L1-01"
    level: str            # "L1"
    idx: int              # 1
    name: str             # "_bias_kernel"
    repo_raw: str         # "vLLM" / "vLLM、SGLang" / ...
    cv_vv: str            # "VV" / "CV" / "未标"
    rationale: str
    difficulty_raw: str
    source_path: str

    @property
    def task_id(self) -> str:
        if self.md_id in KNOWN_POC_DIRS:
            return KNOWN_POC_DIRS[self.md_id]
        # Default: L<level>_<NN>_<name with leading _ stripped>
        clean = self.name.lstrip("_")
        # Truncate very long names — keep readable.
        return f"{self.level}_{self.idx:02d}_{clean}"

    @property
    def repo(self) -> str:
        first = self.repo_raw.split("、")[0].strip()
        return REPO_MAP.get(first, "vLLM")

    @property
    def dtype_class(self) -> str:
        if self.cv_vv in ("CV", "VV"):
            return self.cv_vv
        return "VV"

    @property
    def difficulty(self) -> str:
        for key in ("中高", "中低", "高", "中", "低"):
            if key in self.difficulty_raw:
                return DIFFICULTY_MAP[key]
        return "medium"

    @property
    def category(self) -> str:
        lname = self.name.lower()
        for substr, cat in CATEGORY_RULES:
            if substr in lname:
                return cat
        return "other"

    @property
    def tolerance_class(self) -> str:
        return TOL_BY_CATEGORY.get(self.category, "elementwise")

    @property
    def is_appendix(self) -> bool:
        # L3-07 is the only appendix-only entry per v3.1.
        return self.md_id == "L3-07"


_TABLE_ROW_RE = re.compile(
    r"^\|\s*\*{0,2}(L[123]-\d{2})\*{0,2}\s*"   # md_id
    r"\|\s*`([^`]+)`\s*"                          # name
    r"\|\s*([^|]+?)\s*"                           # repo_raw
    r"\|\s*([^|]+?)\s*"                           # cv_vv
    r"\|\s*([^|]+?)\s*"                           # rationale
    r"\|\s*([^|]+?)\s*"                           # difficulty
    r"\|\s*([^|]+?)\s*\|"                         # source
)


def parse_rows(md_text: str) -> list[RawRow]:
    rows: list[RawRow] = []
    for line in md_text.splitlines():
        m = _TABLE_ROW_RE.match(line)
        if not m:
            continue
        md_id, name, repo_raw, cv_vv, rationale, difficulty, source = m.groups()
        level, idx_s = md_id.split("-")
        rows.append(RawRow(
            md_id=md_id, level=level, idx=int(idx_s),
            name=name.strip(), repo_raw=repo_raw.strip(),
            cv_vv=cv_vv.strip(), rationale=rationale.strip(),
            difficulty_raw=difficulty.strip(), source_path=source.strip(),
        ))
    return rows


# -------------------- file emitters --------------------

PLACEHOLDER_KERNEL = '''"""Task {task_id} — GPU triton reference (SCAFFOLD).

TODO(scaffold): replace this placeholder with the frozen upstream kernel.
Source: {source_path}
Upstream: {repo} — {name}
Rationale: {rationale}

Until the real kernel is frozen, this file holds a no-op @triton.jit
declaration so dataset.load_task and validate_tasks accept the directory.
"""

import triton
import triton.language as tl


@triton.jit
def {kernel_fn}(
    x_ptr,                                            # placeholder pointer
    BLOCK: tl.constexpr,
):
    """Placeholder. Replace with the frozen upstream GPU kernel body."""
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    v = tl.load(x_ptr + offs)
    tl.store(x_ptr + offs, v)
'''

PLACEHOLDER_ORACLE = '''"""oracle_pytorch.py — eager reference for {task_id} (SCAFFOLD).

TODO(scaffold): implement the PyTorch reference path that the runner
calls when oracle_simplified=simple_pytorch_ok / formula_exact_manual.
Until then, this stub raises so any accidental scoring run fails loudly.
"""
from __future__ import annotations


def run(*args, **kwargs):                                    # pragma: no cover
    raise NotImplementedError(
        "oracle_pytorch.run not yet implemented for {task_id}; "
        "see contract.md and freeze the upstream reference first."
    )
'''

PLACEHOLDER_CONTRACT = '''# Contract — {task_id} (SCAFFOLD)

> **TODO(scaffold)**: replace this contract with the frozen upstream
> kernel's behavioural contract. Until then, this file documents only
> the metadata that has been parsed from `docs/npu/benchmark_select.md`.

## Identity

- Benchmark id: `{task_id}` (markdown id `{md_id}`)
- Upstream: `{repo}` — `{name}`
- Source path: `{source_path}`
- CV/VV: `{cv_vv}` · NPU difficulty: `{difficulty}` · category: `{category}`

## Selection rationale

> {rationale}

(Verbatim from `docs/npu/benchmark_select.md`.)

## Required semantics

TODO(scaffold): describe the upstream kernel's per-tile semantics,
mutability of arguments, stride layout, workspace policy, and any
stage ordering invariants — mirror `tasks/level1/L1_01_bias_kernel/contract.md`.

## Tolerance class

`{tolerance_class}` (see `triton_ascend_bench/tolerance_policy.yaml`).

## Metamorphic checks

TODO(scaffold): list runner-enforced invariants per
`specs/canary_task_design.md`.

## Shape coverage

TODO(scaffold): enumerate at least 5 public + 4 hidden shapes when the
task is promoted out of scaffold state.

<!-- param_constraints mirror: auto-generated -->

## Constraints

> Mirror of `meta.yaml.param_constraints` — `meta.yaml` is the single authoritative source; this section is regenerated by `scripts/phase_c_docs.py`.

**Status:** `scaffold: true` — un-reconciled draft; the constraint subsections below are placeholders pending the WB.1 per-task reconcile.

### Knob predicates (`dims`)

_(none)_

### Input dtypes (`dtypes`)

_(none)_

### Input distributions (`dists`)

_(none)_

### Scalar / scalar_seq constraints (`scalars`)

_(none)_

### Tensor invariants (`tensor_invariants`)

_(none)_

<!-- /param_constraints mirror -->
'''

PLACEHOLDER_SHAPES = '''# Placeholder shape coverage for {task_id} (SCAFFOLD).
#
# TODO(scaffold): the `kernel.py` body is a no-op. Once the real kernel is
# frozen, replace these dummy shapes with the upstream coverage suite
# (Public ≥ 5, Hidden ≥ 4 per task_api_contract §6).

public_shapes:
  - id: P1
    name: scaffold_smoke
    num_tokens: 4
    dtype: float32
  - id: P2
    name: scaffold_medium
    num_tokens: 128
    dtype: float32
  - id: P3
    name: scaffold_large
    num_tokens: 1024
    dtype: float32

hidden_shapes:
  - id: H1
    name: scaffold_hidden_smoke
    num_tokens: 8
    dtype: float32
'''

def file_sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def render_meta(row: RawRow, kernel_sha: str) -> str:
    appendix_flag = "true" if row.is_appendix else "false"
    return f'''schema_version: "1.0"
benchmark_version: "0.1.0-scaffold"
task_id: {row.task_id}
level: "{row.level}"
idx: {row.idx}
name: "{row.name}"
repo: "{row.repo}"
dtype_class: "{row.dtype_class}"
category: "{row.category}"
npu_difficulty: "{row.difficulty}"

upstream:
  repo: "{'vllm-project/vllm' if row.repo == 'vLLM' else 'sgl-project/sglang' if row.repo == 'SGLang' else row.repo}"
  path: "{row.source_path}"
  commit_sha: "snapshot-no-git"
  file_sha256: "{kernel_sha}"
  function_hash: "{kernel_sha}"
  license: "Apache-2.0"
  notes: "scaffold — kernel.py is a placeholder; freeze upstream before grading."

interface:
  alias_map: {{}}
  mutability: "unspecified"
  stride_layout: "TODO(scaffold)"
  workspace_policy: "none"
  stream_sync_policy: "runner-managed"
  rng_state_policy: "deterministic"

oracle:
  pytorch: "oracle_pytorch.py::run"
  gpu_triton: "kernel.py::placeholder"

tolerance_class: "{row.tolerance_class}"

# cost_model — constant placeholder block (W1.2 made `cost_model` a
# required top-level key; this generator backfills it). The scalar form
# names a real peak-table compute class / resource so `validate_tasks`'
# schema + formula check passes. The constant `1` formulas reference no
# `size_knobs` (the scaffold shapes.yaml declares none yet); WB.1's
# per-task reconcile replaces them with the real FLOPs / bytes formulas.
cost_model:
  flops: "1"
  flops_class: vector
  bytes: "1"
  bytes_resource: hbm
  gpu_xcheck_tol: 0.05

# param_constraints — operator input contract (WA.1,
# specs/task_api_contract.md §5.2). `scaffold: true` is the single
# sentinel the validator / eval gate consult: the block is schema-shape
# valid but flags the task as not yet hand-reconciled (WB.1 backfills the
# five subsections). The contract.md mirror section is kept in sync.
param_constraints:
  scaffold: true
  dims: []
  dtypes: {{}}
  dists: {{}}
  scalars: {{}}
  tensor_invariants: []

oracle_simplified: "upstream_oracle_required"

runner_checks:
  expected_kernel_count_range: [1, 4]
  min_dram_bytes: 0
  min_flops: 0
  output_format: "safetensors"
  output_allocation: "candidate_inplace"
  static_checks:
    - ast_denylist
    - import_allowlist
    - constexpr_match
  metamorphic_checks: []
  shape_coverage:
    public: 3
    hidden: 1

is_canary: false
is_hidden_only: false
is_appendix: {appendix_flag}

hardware:
  npu: "910b"
  ub_budget_kb: 192
  fp8_required: false

provenance:
  added_in: "v0.1-scaffold"
  authors: ["triton-ascend-bench team"]
  last_modified: "2026-05-15"
  source_md_id: "{row.md_id}"
'''


# -------------------- main --------------------

def write_scaffold(row: RawRow, force: bool, dry_run: bool) -> str:
    level_dir = TASKS_ROOT / f"level{row.level[1]}"
    task_dir = level_dir / row.task_id

    if task_dir.exists() and not force:
        return "skip"

    # Sanitise kernel function identifier — strip leading underscores and
    # any "kernel" suffix duplication so the placeholder reads naturally.
    kernel_fn = row.name.lstrip("_") or "placeholder_kernel"
    kernel_text = PLACEHOLDER_KERNEL.format(
        task_id=row.task_id,
        source_path=row.source_path,
        repo=row.repo,
        name=row.name,
        rationale=row.rationale.replace('"', "'"),
        kernel_fn=kernel_fn,
    )
    kernel_bytes = kernel_text.encode("utf-8")
    kernel_sha = file_sha256(kernel_bytes)

    files = {
        "kernel.py": kernel_text,
        "oracle_pytorch.py": PLACEHOLDER_ORACLE.format(task_id=row.task_id),
        "contract.md": PLACEHOLDER_CONTRACT.format(
            task_id=row.task_id,
            md_id=row.md_id,
            repo=row.repo,
            name=row.name,
            source_path=row.source_path,
            cv_vv=row.cv_vv,
            difficulty=row.difficulty,
            category=row.category,
            rationale=row.rationale.replace('"', "'"),
            tolerance_class=row.tolerance_class,
        ),
        "shapes.yaml": PLACEHOLDER_SHAPES.format(task_id=row.task_id),
        "meta.yaml": render_meta(row, kernel_sha),
    }

    if dry_run:
        return f"would-create:{task_dir.relative_to(REPO_ROOT)}"

    task_dir.mkdir(parents=True, exist_ok=True)
    for fname, content in files.items():
        (task_dir / fname).write_text(content)
    return f"created:{task_dir.relative_to(REPO_ROOT)}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force", action="store_true",
                    help="Overwrite existing dirs (DANGEROUS).")
    args = ap.parse_args()

    if not SELECT_MD.is_file():
        print(f"missing: {SELECT_MD}", file=sys.stderr)
        return 1

    rows = parse_rows(SELECT_MD.read_text())
    if not rows:
        print("no rows parsed — check table regex", file=sys.stderr)
        return 1

    summary: dict[str, int] = {"created": 0, "skip": 0, "would-create": 0}
    for row in rows:
        result = write_scaffold(row, force=args.force, dry_run=args.dry_run)
        head = result.split(":", 1)[0]
        summary[head] = summary.get(head, 0) + 1
        print(f"{row.md_id:>6} → {row.task_id}: {result}")

    print()
    print(f"parsed rows: {len(rows)}")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
