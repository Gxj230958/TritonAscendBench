import torch


# Authored launcher: drives the in-module host wrapper `_chunk_cumsum_fwd`
# (which launches the graded @triton.jit `_chunk_cumsum_fwd_kernel`) exactly
# as the upstream Mamba2 SSD call site does
# (vllm/model_executor/layers/mamba/ops/ssd_combined.py::_chunk_scan_combined_fwd,
# vLLM commit 18f6bf5a214fd51d0cb08fcf1b59edc5629e7c19).
#
# Why a launcher rather than the host wrapper alone: the graded entrypoint
# `_chunk_cumsum_fwd` needs a `cu_chunk_seqlens` metadata tensor that the
# upstream pipeline derives from the request layout, not a primitive the
# benchmark fixture materialises directly. The launcher reproduces upstream's
# `compute_varlen_chunk_metadata` (mamba2_attn.py) for the fixture's
# contiguous token stream, then calls the wrapper verbatim. The sibling
# `_chunk_state_fwd` in this module is NOT graded and is never called here.
#
# `build_cu_chunk_seqlens` is byte-identical to the copy in oracle_pytorch.py;
# both derive the partition from the SAME (seqlen, chunk_size), so they cannot
# diverge -- any drift would make the correct kernel mismatch the oracle and
# BLOCK the task (fail-closed), never a silent pass.
def build_cu_chunk_seqlens(seqlen: int, chunk_size: int) -> list[int]:
    cs = int(chunk_size)
    pattern = [cs + cs // 2, cs // 2, 2 * cs, (cs * 3) // 4, cs - 1]
    seq_lens: list[int] = []
    rem = int(seqlen)
    i = 0
    while rem > 0:
        length = pattern[i % len(pattern)] or rem
        length = min(length, rem)
        seq_lens.append(length)
        rem -= length
        i += 1
    chunk_lens: list[int] = []
    pos = 0
    for length in seq_lens:
        end = pos + length
        p = pos
        while p < end:
            room = cs - (p % cs)
            take = min(room, end - p)
            chunk_lens.append(take)
            p += take
        pos = end
    cu = [0]
    for c in chunk_lens:
        cu.append(cu[-1] + c)
    assert cu[-1] == int(seqlen), (cu[-1], seqlen)
    assert all(0 < c <= cs for c in chunk_lens), chunk_lens
    return cu


def _pin_cumsum_autotune_config() -> None:
    """Make the @triton.autotune launch config DETERMINISTIC for the strict
    severed-vs-original bit-exact gate.

    ``_chunk_cumsum_fwd_kernel`` (frozen kernel.py) is ``@triton.autotune``'d
    over BLOCK_SIZE_H. Triton's Autotuner selects the config by TIMING
    (``cache[key] = min(timings, ...)``), which is non-deterministic across the
    two independent S2 subprocesses (severed vs original). A different
    BLOCK_SIZE_H gives a different ``[BLOCK_SIZE_H, chunk_size]`` tile layout,
    and that changes how ``tl.cumsum(..., axis=1)`` lowers its scan reduction
    -- float add is non-associative, so a reassociated scan flips a ULP in
    ``dA_cumsum`` (observed at the large shape only; the elementwise ``dt_out``
    stays bit-stable, which pinpoints the reduction as the sole diverging op).

    Collapsing the autotuner to a SINGLE config (one of the kernel's own
    candidates) makes ``len(configs) <= 1``, so the tuner skips benchmarking
    and runs ``configs[0]`` directly -- BOTH sides reduce identically. This
    runs in the SAME launcher on both sides (original exec's it into kernel.py;
    the severed staged candidate appends it), so the pin is symmetric. It
    changes neither the math nor the candidate set.
    """
    kfn = globals().get("_chunk_cumsum_fwd_kernel")
    if kfn is None or not hasattr(kfn, "configs"):
        return  # not an autotuned kernel -> nothing to pin
    try:
        import triton as _triton
    except ImportError:
        return
    # BLOCK_SIZE_H=64 is verbatim one of kernel.py's autotune configs; the
    # head axis is masked (offs_h < nheads) so it is valid for every shape.
    cfg = _triton.Config({"BLOCK_SIZE_H": 64})
    # Two wrapper types reach here:
    #   * real triton Autotuner (GPU path, TRITON_INTERPRET unset + active
    #     driver): plain settable ``.configs`` + a ``.cache`` dict. This is
    #     where the timing-based race lives and where the pin matters.
    #   * the interpreter-fallback _AutotunedKernel (no-driver / interpret
    #     smoke): ``.configs`` is a read-only property over ``._configs`` and
    #     the launch reads ``._first_kwargs`` (built from ``_configs[0]``). It
    #     is ALREADY deterministic, but pin it to the SAME config so both
    #     paths select identically and the assignment never raises.
    try:
        kfn.configs = [cfg]
        if hasattr(kfn, "cache"):
            kfn.cache = {}
        # Drop any stale best_config so the single-config path is taken.
        if hasattr(kfn, "best_config"):
            kfn.best_config = cfg
    except AttributeError:
        if hasattr(kfn, "_configs"):
            kfn._configs = [cfg]
            akw = getattr(cfg, "all_kwargs", None)
            if callable(akw):
                kfn._first_kwargs = dict(akw())


def run(inputs: dict) -> dict:
    _pin_cumsum_autotune_config()
    dt = inputs["dt"]
    A = inputs["A"]
    chunk_size = int(inputs["chunk_size"])
    dt_softplus = bool(inputs.get("dt_softplus", False))

    seqlen = int(dt.shape[0])
    cu_list = build_cu_chunk_seqlens(seqlen, chunk_size)
    cu_chunk_seqlens = torch.tensor(cu_list, dtype=torch.int32, device=dt.device)

    dl = inputs.get("dt_limit")
    dt_limit = tuple(dl) if dl is not None else (0.0, float("inf"))

    dA_cumsum, dt_out = _chunk_cumsum_fwd(  # noqa: F821 (exec'd into kernel module)
        dt,
        A,
        chunk_size,
        cu_chunk_seqlens,
        dt_bias=inputs.get("dt_bias"),
        dt_softplus=dt_softplus,
        dt_limit=dt_limit,
    )
    return {"dt_out": dt_out, "dA_cumsum": dA_cumsum}
