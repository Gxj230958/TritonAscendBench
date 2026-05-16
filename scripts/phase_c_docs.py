#!/usr/bin/env python3
"""phase_c_docs.py — generate shapes.yaml + contract.md from oracles.

For every non-PoC working-reference task:

  shapes.yaml  — size-knob schema consumed by triton_ascend_bench.inputs:
    derive each oracle input's {dims,dtype,dist} from the run() body
    keys + the helper docstring shape annotations, then emit ≥5 public
    + ≥4 hidden rows varying the knobs.

  contract.md  — the authoritative spec is the oracle docstring (math +
    I/O + ORACLE-SCOPE). Surface it plus meta (tolerance, upstream),
    a metamorphic-check stub, and the shape-coverage table.

Skips the 5 hand-curated PoCs and the 3 errata-scaffold tasks (their
oracle is still a stub).
"""
from __future__ import annotations

import argparse
import ast
import re
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parent.parent
TASKS = REPO / "tasks"

SKIP_POC = {
    "L1_01_bias_kernel", "L1_12_rms_norm_kernel", "L1_16_silu_and_mul",
    "L2_10_w8a8_block_int8_matmul", "L3_04_fwd_kernel",
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
def _T(dims, dist="randn", dtype="float32"):
    return {"dims": dims, "dtype": dtype, "dist": dist}


def _I(dims, dist="randint:0:2"):
    return {"dims": dims, "dtype": "int64", "dist": dist}


def _S(v):
    return {"scalar": v}


def _SK(expr):
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
    # sampling
    "L1_22_min_p_kernel": {
        "logits": _T(["B", "N"]),
        "expanded_idx_mapping": _I(["B"], "randint:0:E"),
        "min_p": _T(["E"], "rand"),
    },
    "L1_23_penalties_and_temperature_kernel": {
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
    "L1_24_rejection_sample_kernel": {
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
    "L1_27_apply_grammar_bitmask_kernel": {
        "logits": _T(["B", "N"]),
        "logits_indices": _I(["B"], "randint:0:E"),
        "bitmask": _I(["E", "N//32+1"], "randint:0:1024"),
    },
    # quantization
    "L1_07_dequantize_k_cache_fast_kernel": {
        "nope_q": _T(["B", "D"]), "nope_s": _T(["B", "D//8"], "rand"),
        "rope": _T(["B", "D"]), "group_size": _S(8),
    },
    "L1_08_dequantize_k_cache_paged_kernel": {
        "nope_q": _T(["C", "D"]), "nope_s": _T(["C", "D//8"], "rand"),
        "rope": _T(["C", "D"]),
        "page_table": _I(["B", "2"], "randint:0:C"), "group_size": _S(8),
    },
    "L1_10_per_token_group_quant_int8": {
        "x": _T(["B", "D"]), "group_size": _S(8), "eps": _S(1.0e-10),
    },
    "L1_11_awq_dequantize_kernel": {
        "qweight": _I(["N", "E"], "randint:0:65535"),
        "scales": _T(["N//8", "E*8"], "rand"),
        "zeros": _I(["N//8", "E"], "randint:0:65535"),
        "group_size": _S(8),
    },
    "L1_37_compute_problem_sizes_w4a8_kernel": {
        "masked_m": _I(["E"], "randint:0:T"), "n": _S(16),
        "k": _S(16), "num_experts": _SK("E"),
    },
    "L2_11_w8a8_triton_block_scaled_mm": {
        "A": _I(["B", "D"], "randint:0:8"),
        "B": _I(["D", "N"], "randint:0:8"),
        "As": _T(["B", "D//8"], "rand"),
        "Bs": _T(["D//8", "N//8"], "rand"),
        "group_n": _S(8), "group_k": _S(8),
    },
    "L2_37_silu_and_mul_post_quant_kernel": {
        "input": _T(["E", "T", "2*D"]),
        "masked_m": _I(["E"], "randint:0:T"),
        "quant_group_size": _S(8), "fp8_max": _S(448.0),
    },
    # attention / cache / index
    "L1_02_compute_slot_mappings_kernel": {
        "idx_mapping": _I(["B"], "randint:0:E"),
        "positions": _I(["B"], "randint:0:N"),
        "block_table": _I(["E", "N"], "randint:0:N"), "block_size": _S(4),
    },
    "L1_03_convert_req_index_to_global_index_kernel": {
        "req_id": _I(["B"], "randint:0:E"),
        "block_table": _I(["E", "N"], "randint:0:N"),
        "token_indices": _I(["E", "L"], "randint:0:N"), "block_size": _S(4),
    },
    "L1_05_gather_block_tables_kernel": {
        "block_table": _I(["E", "N"], "randint:0:N"),
        "idx_mapping": _I(["B"], "randint:0:E"),
    },
    "L2_06_qkv_lora_b_kernel": _LORA_EXPAND,
    "L3_09_fused_moe_kernel": {
        "a": _T(["B", "D"]), "b": _T(["E", "N", "D"]),
        "topk_weights": _T(["B", "1"], "rand"),
        "sorted_token_ids": _I(["E*4"], "randint:0:B"),
        "expert_ids": _I(["E"], "randint:0:E"),
        "block_size": _S(4), "apply_weight": _S(True),
    },
    "L3_10_fused_moe_kernel_gptq_awq": {
        "a": _T(["B", "D"]), "b": _T(["E", "N", "D"]),
        "topk_weights": _T(["B", "1"], "rand"),
        "sorted_token_ids": _I(["E*4"], "randint:0:B"),
        "expert_ids": _I(["E"], "randint:0:E"),
        "block_size": _S(4), "apply_weight": _S(True),
    },
    "L1_04_copy_page_indices_kernel": {
        "block_table": _I(["E", "N"], "randint:0:N"),
        "seq_lens": _I(["E"], "randint:1:N"), "page_size": _S(4),
    },
    "L1_06_reshape_and_cache_kernel_flash": {
        "key": _T(["B", "H", "D"]), "value": _T(["B", "H", "D"]),
        "key_cache": _T(["C", "T", "H", "D"]),
        "value_cache": _T(["C", "T", "H", "D"]),
        "slot_mapping": _I(["B"], "randint:0:C"),
    },
    "L1_25_merge_attn_states_kernel": _MERGE,
    "L1_26_merge_state_kernel": _MERGE,
    # head_size=16 → rotary_dim=16 → half=8 → mrope_section must sum to 8.
    "L1_31_triton_mrope_forward": {
        "q": _T(["B", "H*16"]), "k": _T(["B", "16"]),
        "cos": _T(["3", "B", "8"]), "sin": _T(["3", "B", "8"]),
        "mrope_section": _S([3, 3, 2]), "head_size": _S(16),
        "rotary_dim": _S(16),
    },
    "L1_32_create_flashinfer_kv_indices_triton": {
        "seq_lens": _I(["E"], "randint:1:N"),
        "block_table": _I(["E", "N"], "randint:0:N"), "page_size": _S(4),
    },
    "L1_33_create_flashmla_kv_indices_triton": {
        "seq_lens": _I(["E"], "randint:1:N"),
        "block_table": _I(["E", "N"], "randint:0:N"), "page_size": _S(4),
    },
    "L2_22_linear_attn_decode_kernel": {
        "q": _T(["B", "H", "D"]), "k": _T(["B", "H", "D"]),
        "v": _T(["B", "H", "D"]), "kv_caches": _T(["B", "H", "D", "D"]),
        "slope_rate": _T(["H"], "rand"),
    },
    "L2_23_fwd_kv_parallel": _LIGHTNING,
    "L2_24_fwd_kv_reduce": _LIGHTNING,
    "L2_33_fwd_grouped_kernel_stage1_rope": _MLA_STAGE1,
    "L3_06_fwd_grouped_kernel_stage1": _MLA_STAGE1,
    "L3_01_kernel_unified_attention_3d": {
        "q": _T(["E*L", "H", "D"]), "k": _T(["E*L", "H", "D"]),
        "v": _T(["E*L", "H", "D"]),
        "cu_seqlens_q": _I(["E+1"], "offsets:to=E*L"),
        "seqused_k": _I(["E"], "randint:1:L"), "scale": _S(0.125),
        "causal": _S(True),
    },
    "L3_02_kernel_paged_attention_2d": {
        "query": _T(["B", "H", "D"]),
        "key_cache": _T(["C", "T", "H", "D"]),
        "value_cache": _T(["C", "T", "H", "D"]),
        "block_table": _I(["B", "N"], "randint:0:C"),
        "seq_lens": _I(["B"], "randint:1:T"), "scale": _S(0.125),
    },
    "L3_05_fwd_kernel_alibi": {
        "q": _T(["H", "L", "D"]), "k": _T(["H", "L", "D"]),
        "v": _T(["H", "L", "D"]), "alibi_slopes": _T(["H"], "rand"),
        "scale": _S(0.125),
    },
    "L3_08_jagged_dense_flash_attention_fwd_kernel": {
        "Q": _T(["B", "H", "D"]), "K": _T(["B", "H", "D"]),
        "V": _T(["B", "H", "D"]),
        "jagged_offsets": _I(["E+1"], "offsets:2"), "scale": _S(0.125),
    },
    # gemm
    "L2_01_bmm_chunk_fwd_kernel": {
        "a": _T(["C*4", "H", "D"]), "b": _T(["C*4", "H", "D"]),
        "chunk_size": _S(4),
        "cu_chunk_seqlens": _I(["C+1"], "offsets:to=C*4"), "causal": _S(True),
    },
    "L2_04_chunked_lora_expand_kernel": _LORA_EXPAND,
    "L2_05_chunked_lora_shrink_kernel": _LORA_SHRINK,
    "L2_07_gate_up_lora_b_kernel": _LORA_EXPAND,
    "L2_08_sgemm_lora_a_kernel": _LORA_SHRINK,
    "L2_09_sgemm_lora_b_kernel": _LORA_EXPAND,
    "L2_12_awq_gemm_kernel": {
        "input": _T(["B", "N"]),
        "qweight": _I(["N", "E"], "randint:0:65535"),
        "scales": _T(["N//8", "E*8"], "rand"),
        "qzeros": _I(["N//8", "E"], "randint:0:65535"),
        "group_size": _S(8),
    },
    "L2_13_bmm_kernel": {"a": _T(["B", "L", "D"]), "b": _T(["B", "D", "N"])},
    "L2_14_matmul_kernel_persistent": {
        "a": _T(["B", "D"]), "b": _T(["D", "N"]), "bias": _T(["N"], "rand"),
    },
    "L3_11_expert_triton_kernel": _BATCHED_EXPERT,
    "L3_12_moe_mmk": _BATCHED_EXPERT,
    # moe
    "L1_35_compute_src2dst_triton_kernel": {
        "reorder_ids": _I(["B"], "arange"), "num_toks": _SK("B"),
    },
    "L1_38_silu_mul_fp8_quant_deep_gemm": {
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
    "L2_27_deepep_permute_triton_kernel": {
        "x": _T(["B", "D"]), "src2dst": _I(["B*2"], "arange"), "topk": _S(2),
    },
    "L2_28_deepep_post_reorder_triton_kernel": {
        "x": _T(["B*2", "D"]), "src2dst": _I(["B*2"], "arange"),
        "topk_weights": _T(["B", "2"], "rand"), "topk": _S(2),
    },
    "L2_29_fused_moe_router_cudacore_kernel": _MOE_ROUTER,
    "L2_30_fused_moe_router_tensorcore_kernel": _MOE_ROUTER,
    "L2_34_silu_mul_static_tensorwise_quant_triton_kernel_for_cutlass_moe": {
        "input": _T(["B", "D"]), "scale": _T(["1"], "rand"),
    },
    "L2_35_pre_reorder_triton_kernel_for_cutlass_moe": {
        "input": _T(["B", "D"]), "src2dst": _I(["B*2"], "arange"),
        "topk": _S(2),
    },
    "L2_36_post_reorder_triton_kernel_for_cutlass_moe": {
        "down_output": _T(["B*2", "D"]), "src2dst": _I(["B*2"], "arange"),
        "topk_weights": _T(["B", "2"], "rand"), "topk": _S(2),
    },
    # other
    "L3_20_recompute_w_u_fwd_kernel": {
        "k": _T(["B", "L", "D"]), "v": _T(["B", "L", "D"]),
        "beta": _T(["B", "L"], "rand"), "g_cumsum": _T(["B", "L"], "rand"),
        "A": _T(["B", "L", "L"]),
    },
    "L3_23_merge_16x16_to_32x32_inverse_kernel": {
        "A": _T(["B", "D", "D"], "tril"),
    },
    # ssm
    "L1_39_state_passing_fwd_kernel": {
        "states": _T(["C", "D", "D"]), "dA_cumsum": _T(["C"], "rand"),
    },
    "L1_40_chunk_cumsum_fwd_kernel": {
        "dt": _T(["B", "L"], "rand"), "A": _T(["B"], "rand"),
        "dt_softplus": _S(True),
    },
    "L2_03_chunk_state_fwd_kernel": {
        "B": _T(["C", "L", "N"]), "x": _T(["C", "L", "D"]),
        "dt": _T(["C", "L"], "rand"),
        "dA_cumsum": _T(["C", "L"], "rand"),
    },
    "L2_19_fused_sigmoid_gating_delta_rule_update_kernel": {
        "A_log": _T(["B", "H"]), "a": _T(["B", "H"]),
        "dt_bias": _T(["H"], "rand"), "q": _T(["B", "H", "D"]),
        "k": _T(["B", "H", "D"]), "v": _T(["B", "H", "D"]),
        "beta": _T(["B", "H"], "rand"), "state": _T(["B", "H", "D", "D"]),
    },
    "L3_13_chunk_gated_delta_rule_fwd_kernel_h_blockdim64": {
        "k": _T(["B", "T", "D"]), "w": _T(["B", "T", "D"]),
        "u": _T(["B", "T", "D"]), "g": _T(["B", "T"], "rand"),
    },
    "L3_14_fused_recurrent_gated_delta_rule_fwd_kernel": _RECURRENT_GDR,
    "L3_15_fused_recurrent_gated_delta_rule_update_fwd_kernel": _RECURRENT_GDR,
    "L3_16_chunk_fwd_kernel_o": {
        "q": _T(["C", "L", "D"]), "k": _T(["C", "L", "D"]),
        "v": _T(["C", "L", "D"]), "h": _T(["C", "D", "D"]),
        "g": _T(["C", "L"], "rand"), "scale": _S(0.125),
    },
    "L3_18_chunk_scaled_dot_kkt_fwd_kernel": {
        "k": _T(["C", "L", "D"]), "g": _T(["C", "L"], "rand"),
        "beta": _T(["C", "L"], "rand"),
    },
    "L3_19_chunk_kda_scaled_dot_kkt_fwd_kernel_intra_sub_inter": {
        "q": _T(["C", "L", "D"]), "k": _T(["C", "L", "D"]),
        "gk": _T(["C", "L"], "rand"), "beta": _T(["C", "L"], "rand"),
        "scale": _S(0.125),
    },
}


def _emit_shapes_yaml(inputs_spec: dict) -> str:
    doc = {
        "size_knobs": KNOB_DEFAULT,
        "public_shapes": [
            {"id": i, "name": nm, "knobs": kb}
            for i, nm, kb in PUBLIC_ROWS
        ],
        "hidden_shapes": [
            {"id": i, "name": nm, "knobs": kb}
            for i, nm, kb in HIDDEN_ROWS
        ],
        "inputs": inputs_spec,
    }
    return ("# Auto-generated by scripts/phase_c_docs.py from the oracle\n"
            "# run() signature + docstring shape annotations. Size-knob\n"
            "# formulas; see triton_ascend_bench/inputs.build_inputs.\n"
            "# TODO(phase-c-review): low-confidence dims default to [B,D];\n"
            "# reconcile against the real contract when frozen.\n\n"
            + yaml.safe_dump(doc, sort_keys=False, allow_unicode=True))


def _emit_contract_md(task_id: str, meta: dict, oracle_doc: str,
                       helper_doc: str, run_keys: list[str],
                       scope_line: str) -> str:
    tol = meta.get("tolerance_class", "?")
    tier = meta.get("oracle_simplified", "?")
    cat = meta.get("category", "?")
    up = meta.get("upstream", {})
    return f"""# Contract — {task_id}

> Generated by scripts/phase_c_docs.py. The oracle docstring is the
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
`triton_ascend_bench.inputs.build_inputs(shapes.yaml, shape_id)`.

## Tolerance class

`{tol}` — see `triton_ascend_bench/tolerance_policy.yaml`.

## Metamorphic checks

TODO(phase-c-review): enumerate runner-enforced invariants
(`specs/canary_task_design.md`). Candidate generic checks: permutation
invariance, zero-input ⇒ zero-output, scale equivariance — confirm per
task before enabling.

## Shape coverage

Public: {len(PUBLIC_ROWS)} (P1..P{len(PUBLIC_ROWS)}). Hidden: {len(HIDDEN_ROWS)} (H1..H{len(HIDDEN_ROWS)}).
Knob grid in `shapes.yaml`.
"""


def process(task_dir: Path, dry: bool) -> str:
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
    _MANUAL = {"L1_24_rejection_sample_kernel"}
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

    shapes_txt = _emit_shapes_yaml(inputs_spec) + manual_note
    contract_txt = _emit_contract_md(
        tid, meta, mod_doc, helper_doc, run_keys, scope_line)
    if manual_note:
        contract_txt += (
            "\n## Phase-C status\n\n> PHASE-C-MANUAL: inputs carry a "
            "cross-input constraint not expressible in the size-knob "
            "schema; hand-author inputs before enabling.\n")

    if dry:
        return f"would-write ({len(run_keys)} inputs)"
    (task_dir / "shapes.yaml").write_text(shapes_txt)
    (task_dir / "contract.md").write_text(contract_txt)
    return f"wrote ({len(run_keys)} inputs)" + (" [MANUAL]" if manual_note else "")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--task")
    args = ap.parse_args()
    counts: dict[str, int] = {}
    for level in ("level1", "level2", "level3"):
        for d in sorted((TASKS / level).iterdir()):
            if not d.is_dir() or d.name.startswith("_"):
                continue
            if args.task and d.name != args.task:
                continue
            r = process(d, args.dry_run)
            head = r.split(" ", 1)[0]
            counts[head] = counts.get(head, 0) + 1
            flag = {"wrote": "✓", "would-write": "·",
                    "skip-poc": "·", "skip-stub": "—"}.get(head, "?")
            print(f"  {flag} {d.name:55s} {r}")
    print(f"\n== summary == {counts}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
