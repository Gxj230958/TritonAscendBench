#!/usr/bin/env python3
"""phase_c_docs.py — scaffold-only generator for shapes.yaml + contract.md +
meta.yaml ``param_constraints``.

For every non-PoC working-reference task this generator emits, from the
oracle:

  shapes.yaml  — size-knob schema consumed by ascendbench.input_spec.inputs:
    derive each oracle input's {dims,dtype,dist} from the run() body
    keys + the helper docstring shape annotations, then emit ≥5 public
    + ≥4 hidden rows varying the knobs.

  contract.md  — the authoritative spec is the oracle docstring (math +
    I/O + ORACLE-SCOPE). Surface it plus meta (tolerance, upstream),
    a metamorphic-check stub, and the shape-coverage table. The
    ``param_constraints`` block of ``meta.yaml`` is also mirrored here as
    a human-readable constraint section, fenced by explicit markers and
    rewritten in place on every run.

For **all 100 tasks** (PoCs and stub-oracle tasks included) this generator
also seeds a ``scaffold: true`` ``param_constraints`` skeleton into
``meta.yaml`` — the operator input contract (WA.1 /
``docs/reference/task_api_contract.md §5.2``). The skeleton is merged into the
existing ``meta.yaml`` without disturbing any other field; a task whose
``param_constraints`` block already exists with ``scaffold: false`` (a
human-reconciled block) is left untouched (the meta-skip path).

The shapes.yaml / contract.md *body* is still skipped for the 5
hand-curated PoCs and the stub-oracle tasks (their oracle is a stub); the
``param_constraints`` seeding and the contract.md constraint-mirror refresh
are **not** skipped for them.
"""
from __future__ import annotations

import argparse
import ast
import re
import sys
from pathlib import Path
from typing import Any

import yaml
from loguru import logger

REPO = Path(__file__).resolve().parent.parent.parent
TASKS = REPO / "tasks"

SKIP_POC = {
    "L1_01_bias", "L1_12_rms_norm", "L2_04_silu_and_mul",
    "L2_17_w8a8_block_int8_matmul", "L3_16_fwd",
}

# Canonical size knobs. Symbolic dims in oracle docstrings are mapped
# onto these; unknown symbols fall back to D.
KNOB_DEFAULT = {"B": 2, "L": 8, "D": 16, "Dv": 16, "N": 16, "T": 8,
                "H": 3, "E": 4, "C": 8, "G": 4}
PUBLIC_ROWS = [
    ("P1", "tiny",   {"B": 1, "L": 4,   "D": 8,   "Dv": 8,  "N": 8,   "T": 4,  "H": 2, "E": 2, "C": 4,  "G": 2}),
    ("P2", "small",  {"B": 2, "L": 8,   "D": 16,  "Dv": 16, "N": 16,  "T": 8,  "H": 3, "E": 4, "C": 8,  "G": 4}),
    ("P3", "medium", {"B": 4, "L": 32,  "D": 64,  "Dv": 64, "N": 32,  "T": 16, "H": 4, "E": 8, "C": 16, "G": 8}),
    ("P4", "large",  {"B": 8, "L": 128, "D": 128, "Dv": 128,"N": 64,  "T": 32, "H": 8, "E": 8, "C": 32, "G": 8}),
    ("P5", "wide",   {"B": 2, "L": 8,   "D": 256, "Dv": 256,"N": 128, "T": 8,  "H": 2, "E": 4, "C": 8,  "G": 4}),
]
HIDDEN_ROWS = [
    ("H1", "h_odd",     {"B": 3, "L": 13, "D": 40,  "Dv": 40, "N": 24,  "T": 12, "H": 3, "E": 3, "C": 6,  "G": 3}),
    ("H2", "h_singleton",{"B": 1,"L": 1,  "D": 32,  "Dv": 32, "N": 16,  "T": 1,  "H": 1, "E": 1, "C": 2,  "G": 1}),
    ("H3", "h_deep",    {"B": 2, "L": 256,"D": 96,  "Dv": 96, "N": 48,  "T": 64, "H": 6, "E": 8, "C": 64, "G": 8}),
    ("H4", "h_wide_b",  {"B": 16,"L": 8,  "D": 64,  "Dv": 64, "N": 32,  "T": 8,  "H": 4, "E": 8, "C": 8,  "G": 4}),
]

# input-name → (dtype, dist) heuristics.
_INT_HINT = ("_id", "_ids", "idx", "mapping", "indices", "indptr",
             "src2dst", "reorder", "table", "slot", "perm", "ranks",
             "seg", "weight_indices", "positions", "req_id", "masked_m",
             "tokens_per_expert", "num_", "expert_num", "block_table",
             "cu_", "offsets", "logits_indices", "topk_ids", "seqused",
             "seq_lens", "last_token", "num_sampled", "num_rejected",
             "is_greedy", "expert_ids", "sorted_token")
_SCALAR_DEFAULTS = {
    "eps": 1.0e-5, "scale": 0.125, "cap": 30.0, "fp8_max": 448.0,
    "group_size": 8, "topk": 2, "block_size": 4, "num_experts": 4,
    "page_size": 4, "max_spec_len": 2, "n": 16, "k": 16,
    "num_toks": 8, "quant_group_size": 8, "dim": -1, "rotary_dim": 8,
    "head_size": 8, "out_rows": 8, "stack_num": 1, "moe_softcapping": 0.0,
    "beta": 1.0, "threshold": 20.0, "softplus_beta": 1.0,
    "softplus_threshold": 20.0, "softcap": 0.0, "out_dtype": "bf16",
    "approximate": "tanh", "act": "silu", "activation": "silu",
    "causal": True, "reverse": False, "dt_softplus": True,
    "apply_weight": True, "sliding_window": None,
}

# Symbolic-dim token → canonical knob.
_DIM_MAP = {
    "n": "B", "b": "B", "bt": "B", "batch": "B",
    "l": "L", "seqlen": "L", "s": "L", "t_": "L",
    "d": "D", "hd": "D", "head_size": "D", "k": "D", "p": "Dv",
    "dv": "Dv", "e": "Dv", "n_kvh": "H", "n_qh": "H", "h": "H",
    "n_heads": "H", "num_heads": "H", "nh": "H",
    "n_chunks": "C", "c": "C", "ctx": "C", "n_ctx": "C",
    "vocab": "N", "vocab_size": "N", "n_tokens": "B",
    "t": "T", "out": "N", "od": "N", "max_rank": "G", "rank": "G",
}


def _parse_run_keys(src: str) -> list[str]:
    keys: list[str] = []
    for m in re.finditer(r'inputs\["(\w+)"\]', src):
        if m.group(1) not in keys:
            keys.append(m.group(1))
    for m in re.finditer(r'inputs\.get\("(\w+)"', src):
        if m.group(1) not in keys:
            keys.append(m.group(1))
    return keys


def _helper_docstring(src: str) -> str:
    """Pick the helper whose docstring carries the richest `name: [dims]`
    shape annotation (not merely the first `_`-prefixed function — that
    misfires on files with a tiny `_rotate_half`-style utility first)."""
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return ""
    best = ""
    best_score = -1
    for n in ast.walk(tree):
        if isinstance(n, ast.FunctionDef) and n.name != "run":
            doc = ast.get_docstring(n) or ""
            score = doc.count(": [") + doc.count(",") + len(doc) // 200
            if score > best_score:
                best_score = score
                best = doc
    return best


def _shape_annotations(doc: str) -> dict[str, list[str]]:
    """Parse `name1,name2: [a,b,c]` segments → {name: [dim tokens]}."""
    out: dict[str, list[str]] = {}
    # normalise unicode commas/brackets
    text = doc.replace("，", ",")
    for m in re.finditer(r'([\w, /]+?):\s*\[([^\]]+)\]', text):
        names = [x.strip() for x in re.split(r'[ ,/]+', m.group(1)) if x.strip()]
        dims_raw = [d.strip() for d in m.group(2).split(",") if d.strip()]
        dims: list[str] = []
        for d in dims_raw:
            if d in ("...", ".."):
                continue
            dims.append(d)
        for nm in names:
            if nm.isidentifier() and nm not in out:
                out[nm] = dims
    return out


def _map_dim(tok: str) -> str:
    t = tok.strip().lower()
    if t.isdigit():
        return t
    base = re.sub(r'[^a-z_0-9]', '', t)
    if base in _DIM_MAP:
        return _DIM_MAP[base]
    if base in {"b", "l", "d", "dv", "n", "t", "h", "e", "c", "g"}:
        return base.upper()
    # arithmetic like 2*d / head_size//2
    arith = re.sub(r'[a-z_]+', lambda mm: _DIM_MAP.get(mm.group(0), "D"), t)
    if re.fullmatch(r'[0-9*+\-/ ]+', arith):
        return arith.replace(" ", "")
    return "D"


def _infer_input_spec(name: str, ann: dict[str, list[str]]) -> dict:
    lname = name.lower()
    # scalar?
    if name in _SCALAR_DEFAULTS:
        return {"scalar": _SCALAR_DEFAULTS[name]}
    if lname in _SCALAR_DEFAULTS:
        return {"scalar": _SCALAR_DEFAULTS[lname]}
    is_int = any(h in lname for h in _INT_HINT)
    dims_src = ann.get(name)
    if dims_src:
        dims = [_map_dim(d) for d in dims_src]
    else:
        dims = ["B", "D"]
    if is_int:
        # integer index tensor — keep values in a safe small range.
        return {"dims": dims, "dtype": "int64", "dist": "randint:0:2"}
    if "weight" in lname or lname in ("w", "g", "gamma"):
        return {"dims": dims, "dtype": "float32", "dist": "ones"}
    if lname in ("a", "cb") and len(dims) >= 2 and dims[-1] == dims[-2]:
        return {"dims": dims, "dtype": "float32", "dist": "tril"}
    if "prob" in lname:
        return {"dims": dims, "dtype": "float32", "dist": "softmax_rows"}
    return {"dims": dims, "dtype": "float32", "dist": "randn"}


# Per-task exact input specs (authored from the oracle helper code —
# the docstring shape parser cannot recover prose/math shape forms).
# dims are size-knob formulas; dist ∈ inputs._make_tensor vocabulary.
def _T(dims, dist="randn", dtype="float32"):  # noqa: N802 - fixture constructor API
    return {"dims": dims, "dtype": dtype, "dist": dist}


def _I(dims, dist="randint:0:2"):  # noqa: N802
    return {"dims": dims, "dtype": "int64", "dist": dist}


def _S(v):  # noqa: N802
    """Non-tensor literal input spec.

    A list literal is emitted under the ``scalar_seq`` key (WA.3 literal
    schema); a JSON scalar stays under ``scalar``. The sidecar / validator
    no longer accept a list under ``scalar``.
    """
    if isinstance(v, (list, tuple)):
        return {"scalar_seq": list(v)}
    return {"scalar": v}


def _SK(expr):  # noqa: N802
    """Scalar whose value equals a size-knob expression (e.g. num_toks==B)."""
    return {"scalar_knob": expr}


_LORA_SHRINK = {
    "x": _T(["B", "D"]), "weights": _T(["E", "G", "D"]),
    "seg_indptr": _I(["E+1"], "offsets:2"),
    "weight_indices": _I(["E"], "randint:0:E"),
    "lora_ranks": _I(["E"], "randint:1:G"),
}
_LORA_EXPAND = {
    "x": _T(["B", "G"]), "weights": _T(["E", "N", "G"]),
    "seg_indptr": _I(["E+1"], "offsets:2"),
    "weight_indices": _I(["E"], "randint:0:E"),
    "lora_ranks": _I(["E"], "randint:1:G"),
    "scalings": _T(["E"], "rand"),
}
_MERGE = {
    "prefix_output": _T(["B", "H", "D"]), "prefix_lse": _T(["B", "H"]),
    "suffix_output": _T(["B", "H", "D"]), "suffix_lse": _T(["B", "H"]),
}
_LIGHTNING = {
    "q": _T(["B", "H", "L", "D"]), "k": _T(["B", "H", "L", "D"]),
    "v": _T(["B", "H", "L", "D"]), "slope": _T(["H"], "rand"),
}
_MLA_STAGE1 = {
    "q": _T(["B", "H", "D"]), "k_buffer": _T(["C", "D"]),
    "v_buffer": _T(["C", "D"]), "scale": _S(0.125),
}
_BATCHED_EXPERT = {
    "a": _T(["E", "T", "D"]), "b": _T(["E", "N", "D"]),
    "expert_num_tokens": _I(["E"], "randint:0:T"),
}
_MOE_ROUTER = {
    "x": _T(["B", "D"]), "router_weight": _T(["8", "D"]),
    "topk": _S(2), "moe_softcapping": _S(0.0),
    "correction_bias": _T(["8"], "rand"),
}
_RECURRENT_GDR = {
    "q": _T(["B", "T", "H", "D"]), "k": _T(["B", "T", "H", "D"]),
    "v": _T(["B", "T", "H", "D"]), "g": _T(["B", "T", "H"], "rand"),
    "beta": _T(["B", "T", "H"], "rand"), "scale": _S(0.125),
}

SHAPE_OVERRIDES: dict[str, dict] = {
    # v3.2 errata substitutes
    "L1_18_act_quant": {
        "x": _T(["B", "D"]), "block_size": _S(8),
    },
    # ngroups == nheads (rep=1) — a valid Mamba2 config that keeps the
    # H % G constraint trivially satisfied across the knob grid.
    "L3_15_chunk_state_varlen": {
        "B": _T(["B*L", "H", "N"]), "x": _T(["B*L", "H", "Dv"]),
        "dt": _T(["B*L", "H"], "rand"),
        "dA_cumsum": _T(["B*L", "H"], "rand"),
        "cu_seqlens": _I(["B+1"], "offsets:to=B*L"),
    },
    "L2_32_correct_attn_cp_out": {
        "out": _T(["B", "H", "D"]), "lses": _T(["E", "B", "H"]),
        "cp_rank": _S(0), "is_lse_base_on_e": _S(True),
    },
    # sampling
    "L2_09_min_p": {
        "logits": _T(["B", "N"]),
        "expanded_idx_mapping": _I(["B"], "randint:0:E"),
        "min_p": _T(["E"], "rand"),
    },
    "L2_10_penalties_and_temperature": {
        "logits": _T(["B", "N"]),
        "expanded_idx_mapping": _I(["B"], "randint:0:E"),
        "output_token_counts": _T(["E", "N"], "rand"),
        "prompt_bin_mask": _I(["E", "N"], "randint:0:2"),
        "repetition_penalty": _T(["E"], "rand"),
        "frequency_penalty": _T(["E"], "rand"),
        "presence_penalty": _T(["E"], "rand"),
    },
    # total draft tokens = cu_num_draft_tokens[-1] = E*2; draft arrays
    # are sized by that total, per-request arrays by E.
    "L2_11_rejection_sample": {
        "draft_token_ids": _I(["E*2"], "randint:0:N"),
        "cu_num_draft_tokens": _I(["E"], "offsets:to=E*2"),
        "draft_probs": _T(["E*2", "N"], "softmax_rows"),
        "target_probs": _T(["E*2", "N"], "softmax_rows"),
        "bonus_token_ids": _I(["E"], "randint:0:N"),
        "recovered_token_ids": _I(["E*2"], "randint:0:N"),
        "uniform_probs": _T(["E*2"], "rand"),
        "is_greedy": _I(["E"], "randint:0:1"),
        "max_spec_len": _S(2),
    },
    "L1_16_apply_grammar_bitmask": {
        "logits": _T(["B", "N"]),
        "logits_indices": _I(["B"], "randint:0:E"),
        "bitmask": _I(["E", "N//32+1"], "randint:0:1024"),
    },
    # quantization
    "L1_07_dequantize_k_cache_fast": {
        "nope_q": _T(["B", "D"]), "nope_s": _T(["B", "D//8"], "rand"),
        "rope": _T(["B", "D"]), "group_size": _S(8),
    },
    "L1_08_dequantize_k_cache_paged": {
        "nope_q": _T(["C", "D"]), "nope_s": _T(["C", "D//8"], "rand"),
        "rope": _T(["C", "D"]),
        "page_table": _I(["B", "2"], "randint:0:C"), "group_size": _S(8),
    },
    "L1_10_per_token_group_quant_int8": {
        "x": _T(["B", "D"]), "group_size": _S(8), "eps": _S(1.0e-10),
    },
    "L1_11_awq_dequantize": {
        "qweight": _I(["N", "E"], "randint:0:65535"),
        "scales": _T(["N//8", "E*8"], "rand"),
        "zeros": _I(["N//8", "E"], "randint:0:65535"),
        "group_size": _S(8),
    },
    "L1_25_compute_problem_sizes_w4a8": {
        "masked_m": _I(["E"], "randint:0:T"), "n": _S(16),
        "k": _S(16), "num_experts": _SK("E"),
    },
    "L2_18_w8a8_triton_block_scaled_mm": {
        "A": _I(["B", "D"], "randint:0:8"),
        "B": _I(["D", "N"], "randint:0:8"),
        "As": _T(["B", "D//8"], "rand"),
        "Bs": _T(["D//8", "N//8"], "rand"),
        "group_n": _S(8), "group_k": _S(8),
    },
    "L2_30_silu_and_mul_post_quant": {
        "input": _T(["E", "T", "2*D"]),
        "masked_m": _I(["E"], "randint:0:T"),
        "quant_group_size": _S(8), "fp8_max": _S(448.0),
    },
    # attention / cache / index
    "L1_02_compute_slot_mappings": {
        "idx_mapping": _I(["B"], "randint:0:E"),
        "positions": _I(["B"], "randint:0:N"),
        "block_table": _I(["E", "N"], "randint:0:N"), "block_size": _S(4),
    },
    "L1_03_convert_req_index_to_global_index": {
        "req_id": _I(["B"], "randint:0:E"),
        "block_table": _I(["E", "N"], "randint:0:N"),
        "token_indices": _I(["E", "L"], "randint:0:N"), "block_size": _S(4),
    },
    "L1_05_gather_block_tables": {
        "block_table": _I(["E", "N"], "randint:0:N"),
        "idx_mapping": _I(["B"], "randint:0:E"),
    },
    "L3_04_qkv_lora_b": _LORA_EXPAND,
    "L3_19_fused_moe": {
        "a": _T(["B", "D"]), "b": _T(["E", "N", "D"]),
        "topk_weights": _T(["B", "1"], "rand"),
        "sorted_token_ids": _I(["E*4"], "randint:0:B"),
        "expert_ids": _I(["E"], "randint:0:E"),
        "block_size": _S(4), "apply_weight": _S(True),
    },
    "L3_20_fused_moe_kernel_gptq_awq": {
        "a": _T(["B", "D"]), "b": _T(["E", "N", "D"]),
        "topk_weights": _T(["B", "1"], "rand"),
        "sorted_token_ids": _I(["E*4"], "randint:0:B"),
        "expert_ids": _I(["E"], "randint:0:E"),
        "block_size": _S(4), "apply_weight": _S(True),
    },
    "L1_04_copy_page_indices": {
        "block_table": _I(["E", "N"], "randint:0:N"),
        "seq_lens": _I(["E"], "randint:1:N"), "page_size": _S(4),
    },
    "L1_06_reshape_and_cache_kernel_flash": {
        "key": _T(["B", "H", "D"]), "value": _T(["B", "H", "D"]),
        "key_cache": _T(["C", "T", "H", "D"]),
        "value_cache": _T(["C", "T", "H", "D"]),
        "slot_mapping": _I(["B"], "randint:0:C"),
    },
    "L1_14_merge_attn_states": _MERGE,
    "L1_15_merge_state": _MERGE,
    # head_size=16 → rotary_dim=16 → half=8 → mrope_section must sum to 8.
    "L1_19_triton_mrope_forward": {
        "q": _T(["B", "H*16"]), "k": _T(["B", "16"]),
        "cos": _T(["3", "B", "8"]), "sin": _T(["3", "B", "8"]),
        "mrope_section": _S([3, 3, 2]), "head_size": _S(16),
        "rotary_dim": _S(16),
    },
    "L1_20_create_flashinfer_kv_indices_triton": {
        "seq_lens": _I(["E"], "randint:1:N"),
        "block_table": _I(["E", "N"], "randint:0:N"), "page_size": _S(4),
    },
    "L1_21_create_flashmla_kv_indices_triton": {
        "seq_lens": _I(["E"], "randint:1:N"),
        "block_table": _I(["E", "N"], "randint:0:N"), "page_size": _S(4),
    },
    "L2_24_linear_attn_decode": {
        "q": _T(["B", "H", "D"]), "k": _T(["B", "H", "D"]),
        "v": _T(["B", "H", "D"]), "kv_caches": _T(["B", "H", "D", "D"]),
        "slope_rate": _T(["H"], "rand"),
    },
    "L2_23_fwd_kv_parallel": _LIGHTNING,
    "L2_24_fwd_kv_reduce": _LIGHTNING,
    "L3_11_fwd_grouped_kernel_stage1_rope": _MLA_STAGE1,
    "L3_18_fwd_grouped_kernel_stage1": _MLA_STAGE1,
    "L3_13_kernel_unified_attention_3d": {
        "q": _T(["E*L", "H", "D"]), "k": _T(["E*L", "H", "D"]),
        "v": _T(["E*L", "H", "D"]),
        "cu_seqlens_q": _I(["E+1"], "offsets:to=E*L"),
        "seqused_k": _I(["E"], "randint:1:L"), "scale": _S(0.125),
        "causal": _S(True),
    },
    "L3_14_kernel_paged_attention_2d": {
        "query": _T(["B", "H", "D"]),
        "key_cache": _T(["C", "T", "H", "D"]),
        "value_cache": _T(["C", "T", "H", "D"]),
        "block_table": _I(["B", "N"], "randint:0:C"),
        "seq_lens": _I(["B"], "randint:1:T"), "scale": _S(0.125),
    },
    "L3_17_fwd_kernel_alibi": {
        "q": _T(["H", "L", "D"]), "k": _T(["H", "L", "D"]),
        "v": _T(["H", "L", "D"]), "alibi_slopes": _T(["H"], "rand"),
        "scale": _S(0.125),
    },
    "L3_08_jagged_dense_flash_attention_fwd": {
        "Q": _T(["B", "H", "D"]), "K": _T(["B", "H", "D"]),
        "V": _T(["B", "H", "D"]),
        "jagged_offsets": _I(["E+1"], "offsets:2"), "scale": _S(0.125),
    },
    # gemm
    "L2_14_bmm_chunk_fwd": {
        "a": _T(["C*4", "H", "D"]), "b": _T(["C*4", "H", "D"]),
        "chunk_size": _S(4),
        "cu_chunk_seqlens": _I(["C+1"], "offsets:to=C*4"), "causal": _S(True),
    },
    "L3_03_chunked_lora_expand": _LORA_EXPAND,
    "L2_15_chunked_lora_shrink": _LORA_SHRINK,
    "L3_05_gate_up_lora_b": _LORA_EXPAND,
    "L2_16_sgemm_lora_a": _LORA_SHRINK,
    "L3_06_sgemm_lora_b": _LORA_EXPAND,
    "L2_19_awq_gemm": {
        "input": _T(["B", "N"]),
        "qweight": _I(["N", "E"], "randint:0:65535"),
        "scales": _T(["N//8", "E*8"], "rand"),
        "qzeros": _I(["N//8", "E"], "randint:0:65535"),
        "group_size": _S(8),
    },
    "L2_20_bmm": {"a": _T(["B", "L", "D"]), "b": _T(["B", "D", "N"])},
    "L2_21_matmul_kernel_persistent": {
        "a": _T(["B", "D"]), "b": _T(["D", "N"]), "bias": _T(["N"], "rand"),
    },
    "L3_21_expert_triton": _BATCHED_EXPERT,
    "L2_33_moe_mmk": _BATCHED_EXPERT,
    # moe
    "L1_23_compute_src2dst_triton": {
        "reorder_ids": _I(["B"], "arange"), "num_toks": _SK("B"),
    },
    "L2_13_silu_mul_fp8_quant_deep_gemm": {
        "y": _T(["E", "T", "2*D"]),
        "tokens_per_expert": _I(["E"], "randint:0:T"),
        "group_size": _S(8), "fp8_max": _S(448.0),
    },
    "L2_25_fwd_kernel_ep_gather": {
        "input_tensor": _T(["B", "D"]),
        "input_index": _I(["B"], "randint:0:B"),
    },
    "L2_26_fwd_kernel_ep_scatter_1": {
        "recv_x": _T(["B", "D"]),
        "dst_index": _I(["B"], "arange"), "out_rows": _SK("B"),
    },
    "L1_30_deepep_permute_triton": {
        "x": _T(["B", "D"]), "src2dst": _I(["B*2"], "arange"), "topk": _S(2),
    },
    "L1_31_deepep_post_reorder_triton": {
        "x": _T(["B*2", "D"]), "src2dst": _I(["B*2"], "arange"),
        "topk_weights": _T(["B", "2"], "rand"), "topk": _S(2),
    },
    "L2_27_fused_moe_router_cudacore": _MOE_ROUTER,
    "L3_10_fused_moe_router_tensorcore": _MOE_ROUTER,
    "L2_29_silu_mul_static_tensorwise_quant_triton_kernel_for_cutlass_moe": {
        "input": _T(["B", "D"]), "scale": _T(["1"], "rand"),
    },
    "L1_33_pre_reorder_triton_kernel_for_cutlass_moe": {
        "input": _T(["B", "D"]), "src2dst": _I(["B*2"], "arange"),
        "topk": _S(2),
    },
    "L1_34_post_reorder_triton_kernel_for_cutlass_moe": {
        "down_output": _T(["B*2", "D"]), "src2dst": _I(["B*2"], "arange"),
        "topk_weights": _T(["B", "2"], "rand"), "topk": _S(2),
    },
    # other
    "L3_29_recompute_w_u_fwd": {
        "k": _T(["B", "L", "D"]), "v": _T(["B", "L", "D"]),
        "beta": _T(["B", "L"], "rand"), "g_cumsum": _T(["B", "L"], "rand"),
        "A": _T(["B", "L", "L"]),
    },
    "L3_30_merge_16x16_to_32x32_inverse": {
        "A": _T(["B", "D", "D"], "tril"),
    },
    # ssm
    "L1_26_state_passing_fwd": {
        "states": _T(["C", "D", "D"]), "dA_cumsum": _T(["C"], "rand"),
    },
    "L1_27_chunk_cumsum_fwd": {
        "dt": _T(["B", "L"], "rand"), "A": _T(["B"], "rand"),
        "dt_softplus": _S(True),
    },
    "L3_02_chunk_state_fwd": {
        "B": _T(["C", "L", "N"]), "x": _T(["C", "L", "D"]),
        "dt": _T(["C", "L"], "rand"),
        "dA_cumsum": _T(["C", "L"], "rand"),
    },
    "L3_07_fused_sigmoid_gating_delta_rule_update": {
        "A_log": _T(["B", "H"]), "a": _T(["B", "H"]),
        "dt_bias": _T(["H"], "rand"), "q": _T(["B", "H", "D"]),
        "k": _T(["B", "H", "D"]), "v": _T(["B", "H", "D"]),
        "beta": _T(["B", "H"], "rand"), "state": _T(["B", "H", "D", "D"]),
    },
    "L3_22_chunk_gated_delta_rule_fwd_kernel_h_blockdim64": {
        "k": _T(["B", "T", "D"]), "w": _T(["B", "T", "D"]),
        "u": _T(["B", "T", "D"]), "g": _T(["B", "T"], "rand"),
    },
    "L3_23_fused_recurrent_gated_delta_rule_fwd": _RECURRENT_GDR,
    "L3_24_fused_recurrent_gated_delta_rule_update_fwd": _RECURRENT_GDR,
    "L3_25_chunk_fwd_kernel_o": {
        "q": _T(["C", "L", "D"]), "k": _T(["C", "L", "D"]),
        "v": _T(["C", "L", "D"]), "h": _T(["C", "D", "D"]),
        "g": _T(["C", "L"], "rand"), "scale": _S(0.125),
    },
    "L3_27_chunk_scaled_dot_kkt_fwd": {
        "k": _T(["C", "L", "D"]), "g": _T(["C", "L"], "rand"),
        "beta": _T(["C", "L"], "rand"),
    },
    "L3_28_chunk_kda_scaled_dot_kkt_fwd_kernel_intra_sub_inter": {
        "q": _T(["C", "L", "D"]), "k": _T(["C", "L", "D"]),
        "gk": _T(["C", "L"], "rand"), "beta": _T(["C", "L"], "rand"),
        "scale": _S(0.125),
    },
}


# --------------------- param_constraints scaffold ---------------------

# Marker pair fencing the contract.md constraint-mirror section. The
# generator always rewrites the text between these two markers from the
# current `meta.param_constraints`; everything outside is left untouched.
_MIRROR_BEGIN = "<!-- param_constraints mirror: auto-generated -->"
_MIRROR_END = "<!-- /param_constraints mirror -->"

# Heading text the legacy hand-written constraint sections used. When a
# contract.md has no mirror marker yet, a `## <heading>` block matching one
# of these is replaced in place by the marker-fenced mirror (so the file
# never carries two authoritative constraint sections).
_LEGACY_CONSTRAINT_HEADINGS = ("Constraints", "Constraint")


def _param_constraints_scaffold() -> dict[str, Any]:
    """Return a fresh ``scaffold: true`` ``param_constraints`` skeleton.

    The skeleton is schema-shape-valid (passes the WA.1 validator layer)
    with the five constraint subsections empty / defaulted. ``scaffold:
    true`` is the single sentinel the WA.4 validator / eval gate consult to
    flag the block as not yet hand-reconciled.

    Returns:
        A new ``param_constraints`` mapping with ``scaffold: true`` and the
        five empty constraint subsections.
    """
    return {
        "scaffold": True,
        "dims": [],
        "dtypes": {},
        "dists": {},
        "scalars": {},
        "tensor_invariants": [],
    }


def _is_reconciled(param_constraints: Any) -> bool:
    """True when a ``param_constraints`` block is human-reconciled.

    A block counts as reconciled — and therefore protected from the
    generator's merge-write — only when it already exists and carries an
    explicit ``scaffold == false``. A missing block, or one with ``scaffold
    == true`` (or a missing / non-boolean ``scaffold``), is treated as an
    un-reconciled draft eligible for (re-)seeding.

    Args:
        param_constraints: the raw ``meta.yaml.param_constraints`` value, or
            ``None`` when the block is absent.

    Returns:
        ``True`` when the block must be left untouched (meta-skip path).
    """
    if not isinstance(param_constraints, dict):
        return False
    return param_constraints.get("scaffold") is False


def _seed_meta_param_constraints(task_dir: Path, dry: bool) -> str:
    """Merge a ``scaffold: true`` ``param_constraints`` skeleton into meta.yaml.

    Only the ``param_constraints`` key is added / replaced; every other
    ``meta.yaml`` field is preserved verbatim. A task whose
    ``param_constraints`` block already exists with ``scaffold == false`` (a
    human-reconciled block) is left untouched — the meta-skip path.

    Args:
        task_dir: the task directory containing ``meta.yaml``.
        dry: when ``True`` no file is written; the would-be action is
            reported instead.

    Returns:
        A short status token — ``meta-seeded`` (skeleton merged in),
        ``meta-skip`` (reconciled block protected), or ``meta-dry-*``.
    """
    meta_path = task_dir / "meta.yaml"
    meta = yaml.safe_load(meta_path.read_text()) or {}
    existing = meta.get("param_constraints")
    if _is_reconciled(existing):
        return "meta-skip"
    if dry:
        return "meta-dry-seed"
    meta["param_constraints"] = _param_constraints_scaffold()
    meta_path.write_text(
        yaml.safe_dump(meta, sort_keys=False, allow_unicode=True))
    return "meta-seeded"


def _render_constraint_mirror(param_constraints: Any) -> str:
    """Render the ``param_constraints`` block as a human-readable section.

    The returned text is the marker-fenced contract.md constraint mirror —
    a pure derived rendering of ``meta.param_constraints`` (the single
    authoritative source). It always starts with :data:`_MIRROR_BEGIN` and
    ends with :data:`_MIRROR_END`.

    Args:
        param_constraints: the raw ``meta.yaml.param_constraints`` value, or
            ``None`` when the block is absent.

    Returns:
        The marker-fenced markdown section (no trailing newline).
    """
    lines: list[str] = [_MIRROR_BEGIN, "", "## Constraints", ""]
    lines.append(
        "> Mirror of `meta.yaml.param_constraints` — `meta.yaml` is the "
        "single authoritative source; this section is regenerated by "
        "`scripts/corpus/generate_shapes_and_contract.py`.")
    lines.append("")

    pc = param_constraints if isinstance(param_constraints, dict) else None
    if pc is None:
        lines.append("_No `param_constraints` block in `meta.yaml`._")
        lines.append("")
        lines.append(_MIRROR_END)
        return "\n".join(lines)

    scaffold = pc.get("scaffold")
    if scaffold is True:
        lines.append(
            "**Status:** `scaffold: true` — un-reconciled draft; the "
            "constraint subsections below are placeholders pending the "
            "WB.1 per-task reconcile.")
    elif scaffold is False:
        lines.append("**Status:** `scaffold: false` — hand-reconciled.")
    else:
        lines.append(f"**Status:** `scaffold: {scaffold!r}`.")
    lines.append("")

    dims = pc.get("dims")
    lines.append("### Knob predicates (`dims`)")
    lines.append("")
    if dims:
        for expr in dims:
            lines.append(f"- `{expr}`")
    else:
        lines.append("_(none)_")
    lines.append("")

    dtypes = pc.get("dtypes")
    lines.append("### Input dtypes (`dtypes`)")
    lines.append("")
    if dtypes:
        for name, dt in dtypes.items():
            lines.append(f"- `{name}`: `{dt}`")
    else:
        lines.append("_(none)_")
    lines.append("")

    dists = pc.get("dists")
    lines.append("### Input distributions (`dists`)")
    lines.append("")
    if dists:
        for name, spec in dists.items():
            kind = spec.get("kind") if isinstance(spec, dict) else spec
            args = spec.get("args") if isinstance(spec, dict) else None
            suffix = f" — args `{args}`" if args else ""
            lines.append(f"- `{name}`: `{kind}`{suffix}")
    else:
        lines.append("_(none)_")
    lines.append("")

    scalars = pc.get("scalars")
    lines.append("### Scalar / scalar_seq constraints (`scalars`)")
    lines.append("")
    if scalars:
        for name, rule in scalars.items():
            if isinstance(rule, dict) and "derive" in rule:
                lines.append(f"- `{name}`: derive `{rule['derive']}`")
            else:
                lines.append(f"- `{name}`: `{rule}`")
    else:
        lines.append("_(none)_")
    lines.append("")

    invariants = pc.get("tensor_invariants")
    lines.append("### Tensor invariants (`tensor_invariants`)")
    lines.append("")
    if invariants:
        for inv in invariants:
            if isinstance(inv, dict):
                args = inv.get("args") or {}
                lines.append(f"- `{inv.get('type')}` — args `{args}`")
            else:
                lines.append(f"- `{inv}`")
    else:
        lines.append("_(none)_")
    lines.append("")

    lines.append(_MIRROR_END)
    return "\n".join(lines)


def _strip_legacy_constraint_section(text: str) -> str:
    """Remove a legacy hand-written ``## Constraints`` section.

    Used only when a contract.md has no mirror marker yet: the first
    heading-2 block whose title is in :data:`_LEGACY_CONSTRAINT_HEADINGS` is
    deleted so the marker-fenced mirror can take its place as the file's one
    authoritative constraint section.

    Args:
        text: the full contract.md text.

    Returns:
        The text with any legacy constraint section removed.
    """
    lines = text.splitlines()
    out: list[str] = []
    i = 0
    n = len(lines)
    while i < n:
        line = lines[i]
        m = re.match(r"^##\s+(.+?)\s*$", line)
        if m and m.group(1) in _LEGACY_CONSTRAINT_HEADINGS:
            i += 1
            while i < n and not re.match(r"^##\s+", lines[i]):
                i += 1
            # Drop a single trailing blank line left before the next heading.
            while out and out[-1].strip() == "":
                out.pop()
            out.append("")
            continue
        out.append(line)
        i += 1
    return "\n".join(out)


def _apply_constraint_mirror(contract_text: str, mirror: str) -> str:
    """Splice the constraint mirror into a contract.md body.

    If the body already carries a marker-fenced mirror section, that section
    is rewritten in place. Otherwise the marker block is appended at end of
    file — after first removing any legacy hand-written constraint section so
    the file never holds two authoritative constraint sections.

    Args:
        contract_text: the full contract.md text (with or without a mirror).
        mirror: the freshly rendered marker-fenced mirror section.

    Returns:
        The updated contract.md text.
    """
    begin = contract_text.find(_MIRROR_BEGIN)
    end = contract_text.find(_MIRROR_END)
    if begin != -1 and end != -1 and end > begin:
        end += len(_MIRROR_END)
        return contract_text[:begin] + mirror + contract_text[end:]
    body = _strip_legacy_constraint_section(contract_text)
    body = body.rstrip("\n")
    return body + "\n\n" + mirror + "\n"


def _refresh_contract_mirror(task_dir: Path, dry: bool) -> str:
    """Rewrite the contract.md constraint mirror from current meta.yaml.

    Always runs — even for a meta-skipped (reconciled) task — so the
    contract.md mirror never lags behind ``meta.param_constraints``.

    Args:
        task_dir: the task directory.
        dry: when ``True`` no file is written.

    Returns:
        A short status token describing the action taken.
    """
    contract_path = task_dir / "contract.md"
    meta_path = task_dir / "meta.yaml"
    if not contract_path.is_file() or not meta_path.is_file():
        return "mirror-missing-files"
    meta = yaml.safe_load(meta_path.read_text()) or {}
    mirror = _render_constraint_mirror(meta.get("param_constraints"))
    old_text = contract_path.read_text()
    new_text = _apply_constraint_mirror(old_text, mirror)
    if new_text == old_text:
        return "mirror-unchanged"
    if dry:
        return "mirror-dry-rewrite"
    contract_path.write_text(new_text)
    return "mirror-rewritten"


def _emit_shapes_yaml(inputs_spec: dict) -> str:
    # The corpus is compact v2 (schema_version 2): canonical, comment-free, byte-pinned by
    # test_committed_v2_corpus_is_self_consistent. Scaffold a NEW task directly in v2 (knob template ->
    # generate_compact_shapes -> the single canonical writer) so it never diverges from the migrated format.
    # The "low-confidence dims default to [B, D]; reconcile when frozen" guidance moves to the contract.md.
    from ascendbench.input_spec.compact_shapes import generate_compact_shapes
    from scripts.corpus.migrate_to_v2 import to_yaml
    knob = {
        "size_knobs": KNOB_DEFAULT,
        "public_shapes": [{"id": i, "name": nm, "knobs": kb} for i, nm, kb in PUBLIC_ROWS],
        "hidden_shapes": [{"id": i, "name": nm, "knobs": kb} for i, nm, kb in HIDDEN_ROWS],
        "inputs": inputs_spec,
    }
    return to_yaml(generate_compact_shapes(knob))


def _emit_contract_md(task_id: str, meta: dict, oracle_doc: str,
                       helper_doc: str, run_keys: list[str],
                       scope_line: str) -> str:
    tol = meta.get("tolerance_class", "?")
    tier = meta.get("oracle_simplified", "?")
    cat = meta.get("category", "?")
    up = meta.get("upstream", {})
    return f"""# Contract — {task_id}

> Generated by scripts/corpus/generate_shapes_and_contract.py. The oracle docstring is the
> authoritative interim spec until a hand-written contract supersedes it.

## Identity

- Benchmark id: `{task_id}`
- Upstream: `{up.get('repo', '?')}` — `{meta.get('name', '?')}`
- Source path: `{up.get('path', '?')}`
- Category: `{cat}` · oracle tier: `{tier}` · tolerance: `{tol}`

## Required semantics

{oracle_doc.strip() or '(see oracle_pytorch.py)'}

### Reference math (oracle helper)

{helper_doc.strip() or '(see oracle_pytorch.py helper docstring)'}
{scope_line}
## I/O contract

`oracle_pytorch.run(inputs: dict) -> dict`. Inputs (keys consumed):
{chr(10).join(f'- `{k}`' for k in run_keys) or '- (none parsed)'}

Concrete per-shape tensors are built by
`ascendbench.input_spec.inputs.build_inputs(shapes.yaml, shape_id)`.

## Tolerance class

`{tol}` — see `ascendbench/policies/tolerance_policy.yaml`.

## Metamorphic checks

Candidate runner-enforced invariants to confirm per task: permutation
invariance, zero-input ⇒ zero-output, scale equivariance.

## Shape coverage

Public: {len(PUBLIC_ROWS)} (P1..P{len(PUBLIC_ROWS)}). Hidden: {len(HIDDEN_ROWS)} (H1..H{len(HIDDEN_ROWS)}).
Knob grid in `shapes.yaml`.
"""


def process_body(task_dir: Path, dry: bool) -> str:
    """Regenerate the shapes.yaml + contract.md *body* for a task.

    Skipped for the 5 hand-curated PoCs and stub-oracle tasks (their oracle
    is a stub). The ``param_constraints`` seeding and the contract.md
    constraint-mirror refresh are handled separately by
    :func:`process_param_constraints` and run for **all** tasks.

    Args:
        task_dir: the task directory.
        dry: when ``True`` no file is written.

    Returns:
        A short status token describing the action taken.
    """
    tid = task_dir.name
    if tid in SKIP_POC:
        return "skip-poc"
    meta = yaml.safe_load((task_dir / "meta.yaml").read_text())
    osrc = (task_dir / "oracle_pytorch.py").read_text()
    if "raise NotImplementedError" in osrc:
        return "skip-stub"

    run_keys = _parse_run_keys(osrc)
    helper_doc = _helper_docstring(osrc)
    ann = _shape_annotations(helper_doc)
    if tid in SHAPE_OVERRIDES:
        inputs_spec = SHAPE_OVERRIDES[tid]
    else:
        inputs_spec = {k: _infer_input_spec(k, ann) for k in run_keys}

    # PHASE-C-MANUAL: tasks whose input shapes carry a cross-input
    # constraint not expressible in the size-knob schema (documented,
    # not silently broken). L1_24's `cu_num_draft_tokens` uses a
    # non-indptr cumulative convention (cu[0]=count of req0, no leading
    # 0), which the generic `offsets` dist cannot produce; a bespoke
    # one-task dist is past the point of diminishing returns.
    _MANUAL = {"L2_11_rejection_sample"}  # noqa: N806 - fixed allowlist
    manual_note = (
        "\n# PHASE-C-MANUAL: cu_num_draft_tokens uses a non-indptr "
        "cumulative\n# convention; build_inputs cannot auto-generate a "
        "consistent set.\n# Hand-author inputs for this task before "
        "enabling it in a run.\n" if tid in _MANUAL else "")

    # module docstring (has ORACLE-SCOPE if present)
    try:
        mod_doc = ast.get_docstring(ast.parse(osrc)) or ""
    except SyntaxError:
        mod_doc = ""
    scope_line = ""
    for ln in mod_doc.splitlines():
        if ln.startswith("ORACLE-SCOPE"):
            scope_line = f"\n## Oracle scope\n\n> {ln.strip()}\n"

    shapes_txt = _emit_shapes_yaml(inputs_spec)   # canonical v2 (comment-free); scaffold guidance -> contract
    contract_txt = _emit_contract_md(
        tid, meta, mod_doc, helper_doc, run_keys, scope_line)
    contract_txt += (
        "\n## Phase-C scaffold\n\n> shapes.yaml was auto-generated from the oracle run() signature + docstring "
        "shape annotations. Low-confidence dims default to [B, D]; reconcile against the real contract when the "
        "kernel is frozen.\n")
    if manual_note:
        contract_txt += (
            "\n## Phase-C status\n\n> PHASE-C-MANUAL: inputs carry a "
            "cross-input constraint not expressible in the size-knob "
            "schema; hand-author inputs before enabling.\n")

    if dry:
        return f"would-write ({len(run_keys)} inputs)"
    (task_dir / "shapes.yaml").write_text(shapes_txt)
    (task_dir / "contract.md").write_text(contract_txt)
    if manual_note:
        # the v2 shapes.yaml carries no comment, so gate this un-reconciled MANUAL scaffold via the structured
        # meta marker -- is_score_eligible (release gate) + eval's `gradeable` both honor
        # release_scope.excluded. Replaces the old `PHASE-C-MANUAL` shapes.yaml-comment gate. Reconcile -> clear.
        # All four fields are required by validate_tasks._validate_release_scope when excluded is True.
        from datetime import datetime, timezone
        mp = task_dir / "meta.yaml"
        m = yaml.safe_load(mp.read_text()) or {}
        rs = m.setdefault("release_scope", {})
        if rs.get("excluded") is not True:
            rs.update({
                "excluded": True,
                "reason": "phase-c-manual: inputs carry a cross-input constraint not expressible in the "
                          "size-knob schema; hand-author + reconcile before enabling",
                "excluded_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "excluded_by": "generate_shapes_and_contract/phase-c-manual",
            })
            mp.write_text(yaml.safe_dump(m, sort_keys=False, allow_unicode=True))
    return f"wrote ({len(run_keys)} inputs)" + (" [MANUAL]" if manual_note else "")


def process_param_constraints(task_dir: Path, dry: bool) -> str:
    """Seed the ``param_constraints`` skeleton + refresh the contract mirror.

    Runs for **every** task — PoCs and stub-oracle tasks included — because
    ``param_constraints`` is a required top-level ``meta.yaml`` key. The
    ``meta.yaml`` write is meta-skipped for a human-reconciled block
    (``scaffold == false``); the contract.md mirror is **always** rewritten
    so it never lags behind ``meta.param_constraints``.

    Args:
        task_dir: the task directory.
        dry: when ``True`` no file is written.

    Returns:
        A combined status string ``"<meta-token>/<mirror-token>"``.
    """
    meta_result = _seed_meta_param_constraints(task_dir, dry)
    mirror_result = _refresh_contract_mirror(task_dir, dry)
    return f"{meta_result}/{mirror_result}"


def _iter_task_dirs(only: str | None = None):
    """Yield the real (non-canary, non-PoC-namespace) task directories.

    Walks ``level1`` / ``level2`` / ``level3`` in sorted order, skipping any
    underscore-prefixed directory (``tasks/_canary/`` is excluded this way —
    canary tasks are exempt from ``param_constraints``, see WA.1).

    Args:
        only: when given, yield only the task directory of that exact name.

    Yields:
        Each matching task directory as a :class:`~pathlib.Path`.
    """
    for level in ("level1", "level2", "level3"):
        for d in sorted((TASKS / level).iterdir()):
            if not d.is_dir() or d.name.startswith("_"):
                continue
            if only and d.name != only:
                continue
            yield d


def refresh_mirror_only(dry: bool, only: str | None = None) -> dict[str, int]:
    """Refresh **only** the contract.md ``param_constraints`` mirror sections.

    This is the ``--refresh-mirror-only`` mode. It runs the marker-fenced
    contract.md mirror rewrite (:func:`_refresh_contract_mirror`) for every
    real task and touches nothing else — in particular it never calls
    :func:`process_body` (which would rewrite ``shapes.yaml``) and never calls
    :func:`_seed_meta_param_constraints` (which would write ``meta.yaml``). It
    exists so a stale contract.md mirror can be re-synced to its already
    reconciled ``meta.param_constraints`` without disturbing the WB.1 knob-grid
    edits in ``shapes.yaml`` or any ``meta.yaml`` content.

    Because the rewrite is a pure derived rendering of
    ``meta.param_constraints``, re-running it on an already synced corpus is a
    byte-stable no-op (every task reports ``mirror-unchanged``).

    Args:
        dry: when ``True`` no file is written; the would-be action is
            reported instead.
        only: when given, restrict the refresh to that single task directory.

    Returns:
        A mapping of mirror status token → count across the processed tasks.
    """
    counts: dict[str, int] = {}
    for d in _iter_task_dirs(only):
        token = _refresh_contract_mirror(d, dry)
        counts[token] = counts.get(token, 0) + 1
        flag = {"mirror-rewritten": "✓", "mirror-unchanged": "·",
                "mirror-dry-rewrite": "·"}.get(token, "?")
        logger.info(f"  {flag} {d.name:55s} mirror={token}")
    logger.info(f"== contract-mirror refresh summary == {counts}")
    return counts


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--task")
    ap.add_argument(
        "--refresh-mirror-only", action="store_true",
        help="Only rewrite the contract.md param_constraints mirror section "
             "from the current meta.param_constraints; never touch "
             "shapes.yaml or meta.yaml.")
    args = ap.parse_args()
    if args.refresh_mirror_only:
        refresh_mirror_only(args.dry_run, args.task)
        return 0
    body_counts: dict[str, int] = {}
    pc_counts: dict[str, int] = {}
    for d in _iter_task_dirs(args.task):
        body = process_body(d, args.dry_run)
        body_head = body.split(" ", 1)[0]
        body_counts[body_head] = body_counts.get(body_head, 0) + 1
        # param_constraints seeding + contract-mirror refresh — runs for
        # every task (PoCs and stub-oracle tasks included).
        pc = process_param_constraints(d, args.dry_run)
        for token in pc.split("/"):
            pc_counts[token] = pc_counts.get(token, 0) + 1
        flag = {"wrote": "✓", "would-write": "·",
                "skip-poc": "·", "skip-stub": "—"}.get(body_head, "?")
        logger.info(f"  {flag} {d.name:55s} body={body}  pc={pc}")
    logger.info(f"== body summary == {body_counts}")
    logger.info(f"== param_constraints summary == {pc_counts}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
