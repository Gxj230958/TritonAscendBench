import torch
import triton


# Authored jit-only launcher: drives the in-module @triton.jit
# `_gumbel_sample_kernel` exactly as the upstream host wrapper
# `gumbel_sample` does (vLLM, commit 18f6bf5,
# vllm/v1/worker/gpu/sample/gumbel.py).
#
# Why a launcher rather than the host wrapper: the module imports
# `from vllm.triton_utils import tl, triton`; on a host with no active
# Triton driver the real upstream re-export hands out a TritonPlaceholder
# (no-op .jit), so calling `gumbel_sample` would not run the kernel. The
# bench injects the real-triton shim into `vllm.triton_utils` and this
# launcher drives the frozen @triton.jit `_gumbel_sample_kernel` directly,
# then reproduces the wrapper's pure-PyTorch post-reduction (argmax over the
# per-block partials + gather). The @triton.jit kernel IS the unit evaluated;
# the host-side block reduction is the same code the upstream wrapper runs.
#
# Kernel contract (matches the re-authored oracle_pytorch). For each token
# `t` and vocab block `b` the @triton.jit kernel computes, with
#   req = expanded_idx_mapping[t]; temp = temperature[req]:
#     logits_f32 = logits[t, block].to(f32)           # block = b*BS + arange
#     if temp != 0 and APPLY_TEMPERATURE: logits /= temp
#     logits_f64 = logits.to(f64)
#     if temp != 0:                                    # gumbel-max sampling
#         gseed = philox.randint(seed[req], pos[t])
#         u     = fp64 uniform from 64 philox bits at offset `block`
#         logits_f64 += -log(-log(u))                  # gumbel noise
#     value, idx = masked_argmax(logits_f64)           # tie -> first index
#     local_argmax[t, b] = b*BS + idx; local_max[t, b] = value
# then the wrapper reduces: sampled[t] = local_argmax[t, argmax_b local_max[t]].
#
# Load-bearing inputs: `temperature` selects greedy (==0) vs gumbel (>0) and
# scales the logits; `seed`/`pos` seed the per-(token,vocab) philox noise;
# `expanded_idx_mapping` routes each token to its request's temp+seed. A
# kernel ignoring any of these returns a different `sampled` (fault-injection
# proven). Output key: "sampled" (the argmax token id per token).
def run(inputs: dict) -> dict:
    logits = inputs["logits"]
    expanded_idx_mapping = inputs["expanded_idx_mapping"]
    temperature = inputs["temperature"]
    seed = inputs["seed"]
    pos = inputs["pos"]
    apply_temperature = bool(inputs.get("apply_temperature", True))

    assert logits.ndim == 2, "logits must be (num_tokens, vocab_size)"
    num_tokens, vocab_size = logits.shape
    assert expanded_idx_mapping.numel() == num_tokens, (
        "expanded_idx_mapping must have one request index per token"
    )
    assert pos.numel() == num_tokens, "pos must have one position per token"

    BLOCK_SIZE = 1024
    num_blocks = triton.cdiv(vocab_size, BLOCK_SIZE)
    local_argmax = logits.new_empty(num_tokens, num_blocks, dtype=torch.int64)
    local_max = logits.new_empty(num_tokens, num_blocks, dtype=torch.float64)

    _gumbel_sample_kernel[(num_tokens, num_blocks)](
        local_argmax,
        local_argmax.stride(0),
        local_max,
        local_max.stride(0),
        None,  # output_processed_logits (not part of the sampling contract)
        0,     # processed_logits_stride
        None,  # output_processed_logits_col
        logits,
        logits.stride(0),
        expanded_idx_mapping,
        seed,
        pos,
        temperature,
        vocab_size,
        BLOCK_SIZE=BLOCK_SIZE,
        APPLY_TEMPERATURE=apply_temperature,
    )

    # Upstream wrapper post-reduction (pure PyTorch, int64 for indexing):
    # pick the winning block per token, then gather its token id.
    max_block_idx = local_max.argmax(dim=-1, keepdim=True)
    sampled = local_argmax.gather(dim=-1, index=max_block_idx).view(-1)
    return {"sampled": sampled}
