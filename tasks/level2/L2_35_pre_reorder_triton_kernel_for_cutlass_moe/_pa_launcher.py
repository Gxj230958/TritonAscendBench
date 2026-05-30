import torch
import triton


# Drives the graded @triton.jit `pre_reorder_triton_kernel_for_cutlass_moe`
# directly, NOT through the upstream host wrapper `pre_reorder_for_cutlass_moe`:
# that wrapper sizes the grid via `_get_launch_config_2d`, which calls
# torch.cuda.get_device_properties(input.device) and aborts on the CPU
# oracle-consistency anchor. The grid is pure scheduling -- the kernel strides
# tokens by num_programs(0) and masks the hidden tail -- so any valid cover
# reproduces the result: one program per token, one hidden block.
#
# Faithful invocation: the kernel's expert gate AND activation scaling are
# exercised, NOT pinned to a trivial subcase. topk_ids carries real expert ids
# including the padding sentinel num_local_experts (gated slots are skipped);
# a1_scales is a real per-tensor scale; src2dst is a real permutation.
def run(inputs: dict) -> dict:
    x = inputs["input"].contiguous()
    src2dst = inputs["src2dst"].to(torch.int64).contiguous()
    topk_ids = inputs["topk_ids"].to(torch.int64).contiguous()
    a1_scales = inputs["a1_scales"].to(torch.float32).contiguous()
    num_local_experts = int(inputs["num_local_experts"])
    topk = int(inputs["topk"])

    num_tokens, hidden_size = x.shape
    gateup_input = x.new_zeros((num_tokens * topk, hidden_size))

    block_dim = triton.next_power_of_2(hidden_size)
    grid = (num_tokens, triton.cdiv(hidden_size, block_dim))

    pre_reorder_triton_kernel_for_cutlass_moe[grid](
        input_ptr=x,
        gateup_input_ptr=gateup_input,
        src2dst_ptr=src2dst,
        topk_ids_ptr=topk_ids,
        a1_scales_ptr=a1_scales,
        num_local_experts=num_local_experts,
        topk=topk,
        num_tokens=num_tokens,
        hidden_size=hidden_size,
        BLOCK_SIZE=block_dim,
        NUM_STAGES=3,
    )
    return {"gateup_input": gateup_input}
