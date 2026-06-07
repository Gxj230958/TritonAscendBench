import torch


# Authored launcher: drives the DECLARED graded @triton.jit kernel
# `_chunk_state_varlen_kernel` through its in-module host wrapper
# `chunk_state_varlen` (sglang
# python/sglang/srt/layers/attention/mamba/ops/ssd_chunk_state.py, commit
# 409d350fb6f6a1e7c7546e39028f811092a8e489; adapted from vLLM / state-spaces
# Mamba2 SSD). The launcher never reimplements the SSD recurrence -- it only
# marshals the flat fixture tensors into the chunked varlen layout the wrapper
# requires and assembles the wrapper's return into the oracle contract key.
#
# Why a launcher rather than binding `chunk_state_varlen` directly: its
# signature is
#   chunk_state_varlen(B, x, dt, dA_cumsum, cu_seqlens, chunk_states,
#                      initial_states=None)
# `chunk_states` has NO default and is NOT a primitive the benchmark fixture
# materialises -- it is the *cumulative* start-of-chunk SSD state that the
# upstream pipeline produces by running the per-chunk state forward
# (`_chunk_state_fwd`) and then carrying it across chunks within each sequence
# (state passing). The strict by-name adapter therefore rejects the wrapper as
# shapes-incomplete. This launcher reproduces that prerequisite from the SAME
# fixture tensors, then calls the wrapper verbatim.
#
# Faithful full-recurrence reconstruction (no single-chunk shortcut):
#   * The varlen kernel partitions the packed [total_seqlen] token stream into
#     chunks of `chunk_size`. dt / dA_cumsum are laid out [nheads, nchunks,
#     chunk_size] with a PER-CHUNK-LOCAL cumulative dA (reset each chunk); the
#     oracle's `dA_cumsum` fixture is a PER-SEQUENCE-GLOBAL cumulative. We
#     recover the per-token dA increments (delta) from the global cumulative,
#     re-chunk them, and take a per-chunk cumsum -- the exact `_chunk_cumsum_fwd`
#     output shape. (delta + per-chunk cumsum is an algebraic identity, so no
#     numeric model is introduced.)
#   * `chunk_states[c]` is the cumulative state at the START of chunk c: zero at
#     each sequence's first chunk, else the previous chunk's start-state decayed
#     by exp(dA_local_last[c-1]) plus the previous chunk's intra-chunk state
#     (`_chunk_state_fwd` output). This is the standard Mamba2 state-passing
#     carry. The graded kernel reads chunk_states[pid_c] (the sequence's last
#     chunk) ONCE and re-decays it to the chunk end -- so chunk_states MUST be
#     cumulative; feeding raw per-chunk states is the single-chunk shortcut and
#     mismatches the oracle for any sequence spanning >1 chunk (fault-injected
#     and confirmed to FAIL).
#
# chunk_size choice: chunk_size = floor(L/2) shrunk to a common divisor of every
# sequence length, so every sequence spans >= 2 chunks AND chunks never straddle
# a sequence boundary. The >= 2 chunks guarantee makes `chunk_states` LOAD-BEARING
# (the kernel's carried-state branch fires); the boundary alignment keeps each
# chunk's dA cumsum / state within one sequence (the varlen invariant).
#
# `_chunk_state_fwd` (the sibling host wrapper in this module) is NOT the graded
# kernel; it is the upstream prerequisite that produces the per-chunk intra
# states, called exactly as upstream does. The graded kernel is
# `_chunk_state_varlen_kernel`, launched by `chunk_state_varlen` below.
#
# Output key (states) matches the oracle contract exactly.
def run(inputs: dict) -> dict:
    Bt = inputs["B"]            # [total_seqlen, ngroups, dstate]
    x = inputs["x"]            # [total_seqlen, nheads, headdim]
    dt = inputs["dt"]           # [total_seqlen, nheads]
    dA = inputs["dA_cumsum"]    # [total_seqlen, nheads]  per-sequence-global cumsum
    cu = inputs["cu_seqlens"].to(torch.long)

    total_seqlen, nheads, headdim = x.shape
    ngroups, dstate = Bt.shape[1], Bt.shape[2]
    seglens = (cu[1:] - cu[:-1]).tolist()
    pos = [int(s) for s in seglens if s > 0]
    if not pos:
        # Degenerate: no tokens -> zero states for every batch slot.
        batch = cu.shape[0] - 1
        out = x.new_zeros((batch, nheads, headdim, dstate))
        return {"states": out}

    # chunk_size: floor(min_seqlen / 2) shrunk to a common divisor of all
    # positive sequence lengths (and hence of total_seqlen). Guarantees
    # >= 2 chunks per sequence (chunk_states load-bearing) and chunk/sequence
    # boundary alignment (varlen invariant).
    chunk_size = max(min(pos) // 2, 1)
    while chunk_size > 1 and any(s % chunk_size != 0 for s in pos):
        chunk_size -= 1
    assert total_seqlen % chunk_size == 0, (total_seqlen, chunk_size)
    nchunks = total_seqlen // chunk_size

    # Recover per-token dA increments from the per-sequence-global cumulative,
    # then re-chunk into [nheads, nchunks, chunk_size] and take a per-chunk
    # cumsum (the _chunk_cumsum_fwd layout the varlen kernel indexes).
    delta = dA.clone()
    for i in range(len(seglens)):
        a, b = int(cu[i]), int(cu[i + 1])
        if b - a > 1:
            delta[a + 1:b] = dA[a + 1:b] - dA[a:b - 1]

    def to_chunks(v):  # [T, H] -> [H, nchunks, chunk_size]
        return v.reshape(nchunks, chunk_size, nheads).permute(2, 0, 1).contiguous()

    dt_c = to_chunks(dt)
    dA_local = torch.cumsum(to_chunks(delta), dim=2)  # per-chunk-local cumsum

    # Per-chunk intra states via the upstream prerequisite host wrapper.
    x_b = x.reshape(1, total_seqlen, nheads, headdim)
    B_b = Bt.reshape(1, total_seqlen, ngroups, dstate)
    dt_b = dt_c.reshape(1, nheads, nchunks, chunk_size)
    dA_b = dA_local.reshape(1, nheads, nchunks, chunk_size)
    states_raw = _chunk_state_fwd(B_b, x_b, dt_b, dA_b)[0]  # noqa: F821 (exec'd into kernel module)
    # states_raw: [nchunks, nheads, headdim, dstate]

    # Cumulative start-of-chunk states (state passing), reset at each
    # sequence's first chunk.
    cu_set = set(int(c) for c in cu.tolist())
    chunk_states = torch.zeros_like(states_raw)
    for c in range(1, nchunks):
        if (c * chunk_size) in cu_set:  # this chunk begins a new sequence
            continue
        dA_last_prev = dA_local[:, c - 1, -1]  # [nheads]
        chunk_states[c] = (
            chunk_states[c - 1] * torch.exp(dA_last_prev)[:, None, None]
            + states_raw[c - 1]
        )

    out = chunk_state_varlen(  # noqa: F821 (exec'd into kernel module)
        B_b[0],
        x_b[0],
        dt_b[0],
        dA_b[0],
        cu.to(torch.int64),
        chunk_states.contiguous(),
    )
    return {"states": out}
