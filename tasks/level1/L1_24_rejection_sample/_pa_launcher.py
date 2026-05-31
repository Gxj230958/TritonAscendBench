import torch


# Authored launcher: drives the in-module @triton.jit
# `rejection_random_sample_kernel` exactly as the upstream host call site does
# (rejection_sample(), vLLM vllm/v1/sample/rejection_sampler.py, commit
# 18f6bf5a214fd51d0cb08fcf1b59edc5629e7c19, lines 501-516).
#
# Why a launcher rather than the host wrapper: the wrapper `rejection_sample`
# is bound to the full `RejectionSampler.forward` pipeline, which pulls in
# `SamplingMetadata`, the bonus `Sampler`, logits processors, top-k/top-p, the
# recovered-token sampler kernel, and the greedy kernel -- none of which is the
# unit under test. The fixture supplies the already-materialised tensors the
# random kernel consumes (draft_token_ids / cu_num_draft_tokens / draft_probs /
# target_probs / bonus_token_ids / recovered_token_ids / uniform_probs /
# is_greedy / max_spec_len), so the bare @triton.jit is driven directly.
#
# Kernel semantics (matches oracle_pytorch): one program per request. Greedy
# requests early-exit (their row keeps the -1 sentinel -- the upstream greedy
# kernel, not this one, handles them). For a non-greedy request the program
# walks its draft tokens, accepting draft token x while
# `draft_prob > 0 and target_prob/draft_prob >= uniform_prob`; on the first
# rejection it emits the pre-computed recovered token and stops; if every draft
# token is accepted it appends the bonus token. The output buffer is allocated
# int32 (consistent with SamplerOutput.sampled_token_ids) and prefilled with
# PLACEHOLDER_TOKEN_ID (-1). Output key matches the oracle contract:
# output_token_ids.
PLACEHOLDER_TOKEN_ID = -1


def run(inputs: dict) -> dict:
    draft_token_ids = inputs["draft_token_ids"].contiguous()
    cu_num_draft_tokens = inputs["cu_num_draft_tokens"].contiguous()
    draft_probs = inputs["draft_probs"].contiguous()
    target_probs = inputs["target_probs"].contiguous()
    bonus_token_ids = inputs["bonus_token_ids"].contiguous()
    recovered_token_ids = inputs["recovered_token_ids"].contiguous()
    uniform_probs = inputs["uniform_probs"].contiguous()
    is_greedy = inputs["is_greedy"].contiguous()
    max_spec_len = int(inputs["max_spec_len"])

    batch_size = cu_num_draft_tokens.shape[0]
    vocab_size = target_probs.shape[-1]
    device = target_probs.device

    output_token_ids = torch.full(
        (batch_size, max_spec_len + 1),
        PLACEHOLDER_TOKEN_ID,
        dtype=torch.int32,  # Consistent with SamplerOutput.sampled_token_ids.
        device=device,
    )

    # NO_DRAFT_PROBS=False: real draft_probs are supplied; SYNTHETIC_MODE=False:
    # acceptance uses the real probability ratio (no synthetic rate table), so
    # synthetic_conditional_rates is None.
    rejection_random_sample_kernel[(batch_size,)](  # noqa: F821
        output_token_ids,
        cu_num_draft_tokens,
        draft_token_ids,
        draft_probs,
        target_probs,
        bonus_token_ids,
        recovered_token_ids,
        uniform_probs,
        is_greedy,
        max_spec_len,
        vocab_size,
        None,  # synthetic_conditional_rates
        NO_DRAFT_PROBS=False,
        SYNTHETIC_MODE=False,
    )
    return {"output_token_ids": output_token_ids}
