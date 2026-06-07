import torch
import triton

# Authored launcher: drives the frozen in-module host wrapper
# `unified_attention` (vLLM unified varlen flash attention, kernel.py:519)
# EXACTLY as upstream does on its 2D (prefill) Triton path. The launcher
# never reimplements the attention numerics -- it only marshals the
# shapes.yaml varlen fixtures into the kernel's tensor contract and
# launches the graded `@triton.jit kernel_unified_attention` through the
# frozen wrapper.
#
# Why a launcher rather than binding `unified_attention` directly: the
# wrapper's strict by-name surface takes `out` / `max_seqlen_q` /
# `max_seqlen_k` / `window_size` / `block_table` / `softcap` /
# `q_descale` / `k_descale` / `v_descale` -- and, critically, its `k` / `v`
# are NOT the packed [E*L, H, D] tensors the fixtures (and the oracle)
# use: the wrapper reads a PAGED KV cache
#   key_cache / value_cache : [num_blocks, block_size, num_kv_heads, head_size]
# gathered through `block_table` (kernel.py:585 `block_size = v.shape[1]`,
# :250-279 the per-tile paged gather). The fixture `block_table` is a
# random [E, N] index tensor that does NOT lay the packed K/V into a cache,
# so binding the wrapper by name would feed it an inconsistent cache and a
# wrong gather. We build the paged cache ourselves and pin the wrapper's
# pure-2D prefill path -- the exact branch the oracle models.
#
# 2D vs 3D path. The wrapper takes the 2D path (writes the final result
# into `output_ptr`, kernel.py:384-398) whenever `max_seqlen_q > 1`
# (kernel.py:636). Every fixture sequence has query_len == L > 1
# (cu_seqlens_q is offsets:to=E*L over E sequences => each span is exactly
# L), so passing max_seqlen_q = L forces the 2D path and the 3D segm
# tensors are never touched. We pass them as None.
#
# Varlen -> paged cache marshalling. The fixtures pack E sequences of
# length L into q/k/v [E*L, H, D]; the oracle gathers, for sequence i,
# k[s : s+Lk] with s = i*L and Lk = seqused_k[i] (oracle_pytorch.py:65,78).
# We scatter each sequence's L KV positions across ceil(L / block_size)
# physical blocks of the [num_blocks, block_size, num_kv_heads, head_size]
# cache, indirected through a SHUFFLED block_table so logical block (i, b)
# maps to a non-contiguous physical id:
#   block_table[i, b] = perm[i*blocks_per_seq + b]   (a fixed permutation)
#   key_cache  [block_table[i,b], slot, h, d] == k[i*L + b*block_size + slot]
#   value_cache[block_table[i,b], slot, h, d] == v[i*L + b*block_size + slot]
#   seq_lens = seqused_k                       kernel reads first Lk positions
# This is a pure copy/permutation of the SAME logical values (no value
# change), so a gathered KV position p in [0, Lk) is block_table[i, p //
# block_size] slot (p % block_size) == packed k[i*L + p] == the oracle's
# k[s + p]: a wrong gather can never coincidentally match the oracle. The
# non-identity, multi-block layout forces the graded kernel to genuinely
# dereference block_table and walk the `seq_offset // BLOCK_SIZE` block index
# (kernel.py:250) -- a kernel that ignores the paged remapping reads the
# wrong KV and misses the oracle.
#
# Physical block_size. The kernel loads the block-table entry for EVERY tile
# lane UNMASKED -- `physical_block_idx = tl.load(block_tables_ptr +
# block_table_offset + seq_offset // BLOCK_SIZE)` (kernel.py:250-252) -- and
# only applies `tile_mask` to the subsequent K/V loads. The prefill
# TILE_SIZE is 32 (kernel.py:513 `_get_tile_size(is_prefill=True)`). We set
# block_size == TILE_SIZE so every lane in tile j shares the same logical
# block index `j*TILE_SIZE // block_size == j`, which is always a valid
# block_table column (j < blocks_per_seq for any active tile, since
# max_seq_prefix_len <= seq_len <= L <= blocks_per_seq*block_size). The last
# partial block is zero-padded; the masked KV load (`tile_mask = seq_offset
# < max_seq_prefix_len`) never reads the padding, so it has no numeric
# effect.
#
# Output. The wrapper writes the final [E*L, H, D] result into `out`. The
# oracle returns float32, so we allocate `out` as float32 and return it
# under the "output" key.

# Prefill tile width the wrapper selects via _get_tile_size(is_prefill=True)
# (kernel.py:513). The physical block_size equals this so each tile maps to
# exactly one logical block and the kernel's unmasked block-table index
# stays in range across the whole tile span.
PREFILL_TILE_SIZE = 32


def _next_multiple(n: int, m: int) -> int:
    return ((n + m - 1) // m) * m


def run(inputs: dict) -> dict:
    q = inputs["q"].contiguous()                 # [E*L, H, D]
    k = inputs["k"].contiguous()                 # [E*L, H, D]
    v = inputs["v"].contiguous()                 # [E*L, H, D]
    cu_seqlens_q = inputs["cu_seqlens_q"]         # [E+1] int64
    seqused_k = inputs["seqused_k"]               # [E]   int64
    scale = float(inputs["softmax_scale"])
    causal = bool(inputs.get("causal", True))

    device = q.device
    total_tokens, num_query_heads, head_size = q.shape
    num_kv_heads = k.shape[1]                      # Hq == Hk here (GQA group 1)
    num_seqs = int(seqused_k.shape[0])

    # Per-sequence span L: cu_seqlens_q is offsets:to=E*L over E sequences,
    # so every span is exactly L. Derive it from the cumulative offsets so
    # the launcher never assumes a fixed knob value.
    cu = cu_seqlens_q.to(torch.long)
    spans = (cu[1:] - cu[:-1]).tolist()
    L = max(spans) if spans else 0
    assert all(s == L for s in spans), (
        f"fixture expects equal per-sequence query spans, got {spans}"
    )

    # `kernel_unified_attention` / `unified_attention` are resolved from the
    # kernel module's namespace: the runner execs this launcher into
    # kernel.py's __dict__ (scripts/_pa_oracle_check_runner.py), so the
    # frozen wrapper is already in scope -- no import.

    # ---- pack the varlen [E*L, H, D] K/V into the paged cache layout ----
    # Genuine multi-block paging (NOT an identity column-0 cache): each
    # sequence's L KV positions are scattered across ceil(L / block_size)
    # physical blocks, and the block_table maps logical block -> a SHUFFLED,
    # non-contiguous physical block id drawn from a shared pool. This forces
    # the graded kernel to actually dereference `block_table`
    # (kernel.py:250 `physical_block_idx = tl.load(block_tables_ptr + ... +
    # seq_offset // BLOCK_SIZE)`) and walk the per-tile `seq_offset //
    # BLOCK_SIZE` block index -- a kernel that ignores the paged remapping or
    # only handles a single contiguous block would read the wrong KV and miss
    # the oracle. block_size == PREFILL_TILE_SIZE so each tile maps to exactly
    # one logical block (the unmasked block-table index stays in-range for the
    # whole tile span); block_size is a pow2 >= the tile so `seq_offset %
    # block_size` / `// block_size` partition cleanly.
    block_size = PREFILL_TILE_SIZE
    blocks_per_seq = _next_multiple(max(L, 1), block_size) // block_size
    num_blocks = num_seqs * blocks_per_seq

    # Deterministic non-identity physical-block assignment: a fixed-seed
    # permutation of the block pool so logical block (i, b) lands on a
    # non-contiguous physical id. Seeded locally (not from torch global RNG)
    # so the mapping is reproducible and never the identity for num_blocks>1.
    perm = torch.randperm(
        num_blocks, generator=torch.Generator(device="cpu").manual_seed(1234)
    ).to(device)
    block_table = perm.to(torch.int32).reshape(num_seqs, blocks_per_seq)

    key_cache = q.new_zeros((num_blocks, block_size, num_kv_heads, head_size))
    value_cache = q.new_zeros((num_blocks, block_size, num_kv_heads, head_size))
    for i in range(num_seqs):
        s = int(cu[i].item())
        e = int(cu[i + 1].item())
        ln = e - s
        for b in range(blocks_per_seq):
            lo = b * block_size
            hi = min(lo + block_size, ln)
            if hi <= lo:
                break
            phys = int(block_table[i, b].item())
            key_cache[phys, : hi - lo] = k[s + lo:s + hi]
            value_cache[phys, : hi - lo] = v[s + lo:s + hi]

    seq_lens_i = seqused_k.to(torch.int32)

    # Output: float32 to match the oracle's reference dtype.
    out = q.new_empty((total_tokens, num_query_heads, head_size),
                      dtype=torch.float32)

    # Pure-2D prefill controls: max_seqlen_q = L > 1 forces the 2D path
    # (kernel.py:636), window_size disables sliding-window, softcap off, no
    # ALiBi / sinks / fp8 / KV quant -- the exact branch the oracle models.
    # `unified_attention` is in scope: the runner execs this launcher into
    # kernel.py's __dict__ (scripts/_pa_oracle_check_runner.py:330).
    unified_attention(
        q=q,
        k=key_cache,
        v=value_cache,
        out=out,
        cu_seqlens_q=cu_seqlens_q.to(torch.int32),
        max_seqlen_q=L,
        seqused_k=seq_lens_i,
        max_seqlen_k=L,
        softmax_scale=scale,
        causal=causal,
        window_size=(-1, -1),
        block_table=block_table,
        softcap=0.0,
        q_descale=None,
        k_descale=None,
        v_descale=None,
    )
    return {"output": out}
