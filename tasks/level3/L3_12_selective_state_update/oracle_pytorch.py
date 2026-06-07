"""Oracle reference for L3_12_selective_state_update (_selective_scan_update_kernel).

Tier: formula_exact_manual
Tolerance class: recurrent_chunk
Category: ssm

Independent eager reference for vLLM's Mamba2 selective-state single-step
DECODE update. For each batch b and head h (group g = h // (nheads/ngroups)):

  dt    = softplus(dt + dt_bias)                       # [dim]
  dA    = exp(A * dt[:, None])                         # [dim, dstate]
  dB    = B[g][None, :] * dt[:, None]                  # [dim, dstate]
  state = state * dA + dB * x[:, None]                 # [dim, dstate]  (in place)
  out   = sum(state * C[g][None, :], dim=dstate)       # [dim]
  out  += x * D                                        # D-skip
  out  *= z * sigmoid(z)                               # SiLU z-gate

`state` [B, H, dim, dstate] and the ssm output `out` [B, H, dim] are produced;
the kernel mutates `state` in place. The recurrence runs in fp32 (matching the
kernel's `.to(tl.float32)` loads). No upstream code is copied.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

F32 = torch.float32


def run(inputs: dict) -> dict:
    state = inputs["state"]                  # [B, H, dim, dstate]
    x = inputs["x"]                          # [B, H, dim]
    dt = inputs["dt"]                        # [B, H, dim]
    A = inputs["A"]                          # [H, dim, dstate]
    B = inputs["B"]                          # [B, G, dstate]
    C = inputs["C"]                          # [B, G, dstate]
    D = inputs["D"]                          # [H, dim]
    z = inputs["z"]                          # [B, H, dim]
    dt_bias = inputs["dt_bias"]              # [H, dim]

    Bsz, H, dim, dstate = state.shape
    ngroups = B.shape[1]
    ratio = H // ngroups
    head_to_group = torch.arange(H, device=state.device) // ratio

    dtf = F.softplus(dt.to(F32) + dt_bias.to(F32).unsqueeze(0))     # [B, H, dim]
    dA = torch.exp(A.to(F32).unsqueeze(0) * dtf.unsqueeze(-1))      # [B, H, dim, dstate]
    Bh = B.to(F32)[:, head_to_group, :]                            # [B, H, dstate]
    Ch = C.to(F32)[:, head_to_group, :]                            # [B, H, dstate]
    dB = Bh.unsqueeze(2) * dtf.unsqueeze(-1)                       # [B, H, dim, dstate]
    new_state = state.to(F32) * dA + dB * x.to(F32).unsqueeze(-1)  # [B, H, dim, dstate]

    out = (new_state * Ch.unsqueeze(2)).sum(dim=-1)                # [B, H, dim]
    out = out + x.to(F32) * D.to(F32).unsqueeze(0)
    out = out * (z.to(F32) * torch.sigmoid(z.to(F32)))

    # candidate mutates `state` in place (fp32); mirror for mutated-input
    # tracking. The ssm output buffer is fp32 (see launcher).
    inputs["state"].copy_(new_state.to(state.dtype))
    return {"out": out.to(torch.float32)}
