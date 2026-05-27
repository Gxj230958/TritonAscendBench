"""Unit tests for triton_ascend_bench.wrapper_contract."""
from __future__ import annotations

from pathlib import Path

import pytest

from triton_ascend_bench.wrapper_contract import (
    check_against_inputs,
    extract_wrapper_contract,
)


def _write(tmp_path: Path, src: str) -> Path:
    p = tmp_path / "kernel.py"
    p.write_text(src)
    return p


# ─────────────────────────────────────────────────────────────────────
# Direct tuple unpack
# ─────────────────────────────────────────────────────────────────────


def test_direct_unpack_4tuple(tmp_path):
    src = (
        "def wrap(k):\n"
        "    B, T, H, K = k.shape\n"
        "    return B + T + H + K\n"
    )
    kp = _write(tmp_path, src)
    c = extract_wrapper_contract(kp, "wrap")
    assert c.accepted_ranks == {"k": {4}}


def test_direct_unpack_3tuple(tmp_path):
    src = "def wrap(g):\n    B, T, D = g.shape\n    return 0\n"
    kp = _write(tmp_path, src)
    c = extract_wrapper_contract(kp, "wrap")
    assert c.accepted_ranks == {"g": {3}}


def test_branch_unpack_collects_set(tmp_path):
    """if/else with different unpack rank -> accepted_ranks is the union."""
    src = (
        "def wrap(g, head_first):\n"
        "    if head_first:\n"
        "        B, H, T = g.shape\n"
        "    else:\n"
        "        B, T, H = g.shape\n"
        "    return B + T + H\n"
    )
    kp = _write(tmp_path, src)
    c = extract_wrapper_contract(kp, "wrap")
    # Both branches unpack to 3-tuple -> single rank in the set
    assert c.accepted_ranks == {"g": {3}}


def test_multi_branch_different_ranks(tmp_path):
    src = (
        "def wrap(g):\n"
        "    if cond:\n"
        "        B, T, H, S = g.shape\n"
        "    else:\n"
        "        B, T, H = g.shape\n"
        "    return 0\n"
    )
    kp = _write(tmp_path, src)
    c = extract_wrapper_contract(kp, "wrap")
    assert c.accepted_ranks == {"g": {3, 4}}


# ─────────────────────────────────────────────────────────────────────
# Starred unpack (the codex-flagged pattern)
# ─────────────────────────────────────────────────────────────────────


def test_starred_unpack_extracts_param_rank(tmp_path):
    """``B, T, Hg, K, V = *k.shape, u.shape[-1]`` -> k.ndim == 4."""
    src = (
        "def wrap(k, u):\n"
        "    B, T, Hg, K, V = *k.shape, u.shape[-1]\n"
        "    return 0\n"
    )
    kp = _write(tmp_path, src)
    c = extract_wrapper_contract(kp, "wrap")
    assert c.accepted_ranks.get("k") == {4}, (
        f"expected k rank set {{4}}, got {c.accepted_ranks.get('k')}"
    )
    # u only has shape[-1] index -> min_rank 1
    assert c.min_ranks.get("u", 0) >= 1


def test_starred_unpack_three_extra(tmp_path):
    """``B, Hg, K, V, X, Y = *k.shape, u.shape[-1], v.shape[2]``
       -> k.ndim == 4."""
    src = (
        "def wrap(k, u, v):\n"
        "    B, Hg, K, V, X, Y = *k.shape, u.shape[-1], v.shape[2]\n"
        "    return 0\n"
    )
    kp = _write(tmp_path, src)
    c = extract_wrapper_contract(kp, "wrap")
    assert c.accepted_ranks.get("k") == {4}


# ─────────────────────────────────────────────────────────────────────
# Shape subscript -> min_rank only
# ─────────────────────────────────────────────────────────────────────


def test_shape_subscript_positive_index(tmp_path):
    src = "def wrap(x):\n    H = x.shape[3]\n    return H\n"
    kp = _write(tmp_path, src)
    c = extract_wrapper_contract(kp, "wrap")
    assert c.min_ranks["x"] == 4  # shape[3] -> ndim >= 4


def test_shape_subscript_negative_index(tmp_path):
    src = "def wrap(x):\n    K = x.shape[-1]\n    return K\n"
    kp = _write(tmp_path, src)
    c = extract_wrapper_contract(kp, "wrap")
    assert c.min_ranks["x"] == 1  # shape[-1] -> ndim >= 1


def test_shape_subscript_does_not_set_accepted_rank(tmp_path):
    """A bare ``.shape[N]`` access is a MIN constraint, not an EXACT
    accepted_rank. The checker should let any rank >= min pass."""
    src = "def wrap(x):\n    H = x.shape[2]\n    return H\n"
    kp = _write(tmp_path, src)
    c = extract_wrapper_contract(kp, "wrap")
    assert c.accepted_ranks.get("x", set()) == set()


# ─────────────────────────────────────────────────────────────────────
# check_against_inputs
# ─────────────────────────────────────────────────────────────────────


def test_check_passes_when_rank_in_accepted_set(tmp_path):
    src = "def wrap(k):\n    B, T, H, K = k.shape\n    return 0\n"
    kp = _write(tmp_path, src)
    c = extract_wrapper_contract(kp, "wrap")
    violations = check_against_inputs(c, {"k": 4})
    assert violations == []


def test_check_violates_when_rank_not_in_accepted_set(tmp_path):
    src = "def wrap(k):\n    B, T, H, K = k.shape\n    return 0\n"
    kp = _write(tmp_path, src)
    c = extract_wrapper_contract(kp, "wrap")
    violations = check_against_inputs(c, {"k": 3})
    assert len(violations) == 1
    assert violations[0].input_name == "k"
    assert violations[0].actual_rank == 3
    assert violations[0].expected_ranks == {4}


def test_check_violates_on_min_rank_only(tmp_path):
    """``x.shape[3]`` access requires ndim >= 4. Input rank 2 should fail."""
    src = "def wrap(x):\n    H = x.shape[3]\n    return H\n"
    kp = _write(tmp_path, src)
    c = extract_wrapper_contract(kp, "wrap")
    violations = check_against_inputs(c, {"x": 2})
    assert len(violations) == 1
    assert violations[0].expected_min_rank == 4


def test_check_silent_when_input_not_referenced(tmp_path):
    """An input the wrapper never .shape-uses is NOT a violation
    candidate -- absence of constraint != failure."""
    src = "def wrap(k, untouched):\n    B, T, H, K = k.shape\n    return 0\n"
    kp = _write(tmp_path, src)
    c = extract_wrapper_contract(kp, "wrap")
    # untouched has no constraint -- any rank is fine
    violations = check_against_inputs(c, {"k": 4, "untouched": 99})
    assert violations == []


# ─────────────────────────────────────────────────────────────────────
# Alias tracking (V9 codex review 2026-05-27)
# ─────────────────────────────────────────────────────────────────────


def test_simple_dtype_alias_unpacks_to_original_param(tmp_path):
    """``kf = k.to(fp32); B, T, D = kf.shape`` -> k accepted_ranks={3}."""
    src = (
        "import torch\n"
        "def wrap(k):\n"
        "    kf = k.to(torch.float32)\n"
        "    B, T, D = kf.shape\n"
        "    return B + T + D\n"
    )
    kp = _write(tmp_path, src)
    c = extract_wrapper_contract(kp, "wrap")
    assert c.accepted_ranks.get("k") == {3}, (
        f"alias k->kf->.shape unpack should credit k, got {dict(c.accepted_ranks)}"
    )


def test_chained_alias_methods_preserve_rank(tmp_path):
    """``a = x.to(fp32).contiguous(); B, T = a.shape`` is one alias hop +
    a rank-preserving method call. v1 only follows ONE hop -- assert
    that one-hop still works."""
    src = (
        "import torch\n"
        "def wrap(x):\n"
        "    a = x.detach()\n"
        "    B, T = a.shape\n"
        "    return B + T\n"
    )
    kp = _write(tmp_path, src)
    c = extract_wrapper_contract(kp, "wrap")
    assert c.accepted_ranks.get("x") == {2}


def test_genexp_alias_unpack_credits_all_params(tmp_path):
    """``kf, wf, uf = (t.to(fp32) for t in (k, w, u))`` -- the L3_13
    oracle pattern. Subsequent ``B,T,D = kf.shape`` must credit ``k``."""
    src = (
        "import torch\n"
        "def wrap(k, w, u):\n"
        "    kf, wf, uf = (t.to(torch.float32) for t in (k, w, u))\n"
        "    B, T, D = kf.shape\n"
        "    Du = uf.shape[-1]\n"
        "    return B + T + D + Du\n"
    )
    kp = _write(tmp_path, src)
    c = extract_wrapper_contract(kp, "wrap")
    assert c.accepted_ranks.get("k") == {3}
    # uf only has .shape[-1] -> min_rank 1 on u
    assert c.min_ranks.get("u", 0) >= 1


def test_lhs_starred_unpack_records_min_rank(tmp_path):
    """``*lead, n, _ = Af.shape`` -> A.ndim >= 2 (n + _ are fixed)."""
    src = (
        "import torch\n"
        "def wrap(A):\n"
        "    Af = A.to(torch.float32)\n"
        "    *lead, n, _ = Af.shape\n"
        "    return n\n"
    )
    kp = _write(tmp_path, src)
    c = extract_wrapper_contract(kp, "wrap")
    # *lead + 2 fixed slots -> min_rank 2 on the original param A
    assert c.min_ranks.get("A", 0) >= 2
    # No exact accepted_rank (starred LHS absorbs unknown count)
    assert "A" not in c.accepted_ranks


def test_l3_13_oracle_helper_exposes_k_rank_3():
    """End-to-end on real corpus: L3_13 oracle helper's alias chain
    must surface k.ndim==3. This is the failure mode codex flagged
    on 2026-05-27 -- if alias tracking is missing the helper returns
    an empty accepted_ranks and the GPU runner can't classify L3_13
    as ORACLE_KERNEL_LAYOUT_MISMATCH."""
    oracle = Path(
        "tasks/level3/L3_13_chunk_gated_delta_rule_fwd_kernel_h_blockdim64/oracle_pytorch.py"
    )
    if not oracle.is_file():
        pytest.skip("L3_13 oracle missing in this checkout")
    c = extract_wrapper_contract(oracle, "_chunk_gated_delta_rule_fwd_h")
    assert c is not None
    assert c.accepted_ranks.get("k") == {3}, (
        f"L3_13 oracle should expose k={{3}}, got {dict(c.accepted_ranks)}"
    )


# ─────────────────────────────────────────────────────────────────────
# Corpus integration -- 7 ValueError task classify correctly
# ─────────────────────────────────────────────────────────────────────


REAL_CORPUS_CASES = [
    # (task_id, key, wrapper_expected_rank, actual_rank)
    # L3_22 was patched 2026-05-27 (oracle rewritten as per-chunk
    # cumsum + shapes.yaml.g.dims aligned to [B,H,L,D]); it now
    # returns ALIGNED_READY and no longer belongs in the
    # rank-mismatch list.
    ("L3_13_chunk_gated_delta_rule_fwd_kernel_h_blockdim64", "k", 4, 3),
    ("L3_16_chunk_fwd_kernel_o", "q", 4, 3),
    ("L3_17_chunk_gla_fwd_kernel_o", "q", 4, 3),
    ("L3_18_chunk_scaled_dot_kkt_fwd", "k", 4, 3),
    ("L3_20_recompute_w_u_fwd", "k", 4, 3),
    ("L3_23_merge_16x16_to_32x32_inverse", "A", 4, 3),
]


@pytest.mark.parametrize(
    "task_id,key,expected_rank,actual_rank", REAL_CORPUS_CASES
)
def test_real_corpus_value_error_tasks_detect_rank_mismatch(
    task_id, key, expected_rank, actual_rank
):
    """All 7 L3 tasks that LAUNCH_BLOCKED with `not enough values to
    unpack` get statically flagged with the correct mismatch shape."""
    import sys
    import re
    import yaml
    sys.path.insert(0, "scripts")
    from oracle_equivalence_gpu_runner import _resolve_kernel_entrypoint
    from triton_ascend_bench.inputs import build_inputs

    td = Path(f"tasks/level3/{task_id}")
    if not td.is_dir():
        pytest.skip(f"corpus task {task_id} missing")
    meta = yaml.safe_load((td / "meta.yaml").read_text())
    src = (td / "oracle_pytorch.py").read_text()
    keys = sorted(
        set(re.findall(r'inputs\[["\']([^"\']+)["\']\]', src))
        | set(re.findall(r'inputs\.get\(["\']([^"\']+)["\']', src))
    )
    fn = _resolve_kernel_entrypoint(
        meta, td / "kernel.py", task_id, oracle_input_keys=keys
    )
    assert fn is not None
    c = extract_wrapper_contract(td / "kernel.py", fn)
    assert c is not None

    shapes = yaml.safe_load((td / "shapes.yaml").read_text())
    inputs = build_inputs(shapes, "P1", seed=0)
    input_ranks = {k: v.ndim for k, v in inputs.items() if hasattr(v, "ndim")}
    violations = check_against_inputs(c, input_ranks)

    assert violations, (
        f"{task_id}: expected wrapper_contract to detect rank mismatch"
    )
    bad = next((v for v in violations if v.input_name == key), None)
    assert bad is not None, (
        f"{task_id}: violation on {key!r} not found "
        f"(got {[v.input_name for v in violations]})"
    )
    assert bad.actual_rank == actual_rank, (
        f"{task_id}: actual_rank {bad.actual_rank} != {actual_rank}"
    )
    assert expected_rank in (bad.expected_ranks or {bad.expected_min_rank})
