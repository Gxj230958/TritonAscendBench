"""Identity-trap oracle for canary C1_identity_trap.

This is INTENTIONALLY an identity: `run()` returns `logits` untouched. It is
NOT a correct logit-bias reference — it is the canary trap.

How the trap works (W2.5 / DESIGN-V2 §6.8):

* The canary is bound (via `meta.yaml.source_task_id`) to the real task
  `L1_01_bias_kernel`. The runner reuses the submission's `solution.py` for
  L1_01 to run this canary.
* An HONEST L1_01 kernel genuinely applies the additive logit bias, so its
  output DIFFERS from this identity oracle → it does NOT "pass" the canary.
  `meta.yaml.expected_honest_pass` is therefore `false`.
* An identity-cheating kernel (one that returns its `logits` input unchanged
  to game elementwise tolerance) MATCHES this identity oracle → it "passes"
  the canary. `check_canaries` flags `got != expected_honest_pass` as a
  triggered finding, escalating the whole submission to `suspicious`.

The canary is never score-eligible: it lives under `tasks/_canary/`, which
`dataset.iter_all_tasks()` skips, and `meta.yaml` declares `is_canary: true`.
"""
from __future__ import annotations

import torch


def run(inputs: dict) -> dict:
    """Identity trap entrypoint — returns `logits` unchanged.

    Standard oracle contract (`oracle_pytorch.py::run`): a real oracle would
    apply the three-stage logit transform here. This one does not — that is
    the point. A submission whose kernel output matches this identity is
    behaving as an identity hack.
    """
    logits: torch.Tensor = inputs["logits"]
    return {"logits": logits.clone()}
