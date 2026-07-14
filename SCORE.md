# TAB — current corpus status

This is the top-line status of the task corpus. It matches the
[README](README.md) §Status; that section is authoritative.

## Headline

**83 verified + 18 ceiling-accepted, of 101 tasks** (0 blocked today).

- **Verified** — the extracted GPU-Triton kernel is bit-exact severed from its
  cited upstream commit **and** its output matches an independent PyTorch
  oracle within the task's tolerance.
- **Ceiling-accepted** — a documented large-shape precision relaxation on an
  allowlisted set; distinct from Verified, never a silent pass.
- **Blocked** — reserved for a task where *upstream itself* diverges from a
  correct reference. **None today.**

Every scored `kernel.py` is byte-identical to its cited upstream commit
(`meta.upstream.file_sha256` / `phase_a_prime.standalone_sha256`); the status
moves **only** by re-running the unmodified frozen tasks, never by patching a
kernel we judged buggy. That is what makes it independently auditable.

## Performance

Roofline efficiency requires real Ascend 910B hardware and is **not yet wired
end-to-end**. Until then NPU latency is reported as raw numbers only and is
**out of the release gate**. Nothing is reported as perfect that is not.

## Upstream-fidelity principle (why "blocked" exists)

When a frozen upstream kernel carries a real defect, a correct oracle
*necessarily* diverges from it. The bench's rule is **exclusion-with-evidence,
never a silent fix or hide**: such a task is documented and kept out of the
scored denominator rather than patched to pass. Refusing to silently correct
upstream is what makes any discovered defect credible. See
[reference/reward_hack_defense.md](docs/reference/reward_hack_defense.md) and
[reference/correctness_model.md](docs/reference/correctness_model.md).

## See also

- [README](README.md) §Status — authoritative status
- [docs/reference/scoring_spec.md](docs/reference/scoring_spec.md) — how the score is computed
- [docs/reference/tolerance_policy.md](docs/reference/tolerance_policy.md) — verified vs ceiling-accepted
