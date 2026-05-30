# Authored jit-only launcher (U5): drives _apply_grammar_bitmask_kernel from the
# upstream call site (StructuredOutputsWorker.apply_grammar_bitmask). The kernel
# masks logits in place; the oracle's output key is "logits".
def run(inputs: dict) -> dict:
    logits = inputs["logits"]
    logits_indices = inputs["logits_indices"]
    bitmask = inputs["bitmask"]
    num_masks = bitmask.shape[0]
    vocab_size = logits.shape[-1]
    BLOCK_SIZE = 8192
    grid = (num_masks, triton.cdiv(vocab_size, BLOCK_SIZE))
    _apply_grammar_bitmask_kernel[grid](
        logits,
        logits.stride(0),
        logits_indices,
        bitmask,
        bitmask.stride(0),
        vocab_size,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return {"logits": logits}
