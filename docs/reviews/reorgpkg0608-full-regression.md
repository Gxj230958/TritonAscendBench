# Full GPU regression -- run reorgpkg0608

**Overall: GREEN** | commit `6685992` | 5/5 PASS

| Stage | Result | Batch | Summary | Regressions |
|-------|--------|-------|---------|-------------|
| unit | PASS | `-` | 0 failed, 0 errors, 0 skipped, 2277 passed (baseline 0) | - |
| s2 | PASS | `bs2reorgpkg0608` | 101/101 severance PASS (cert-stage also attempted cost_model_xcheck + golden) | - |
| c2 | PASS | `bc2reorgpkg0608` | 101/101 PASS, 0 non-PASS (0 NEW vs bc2apply) | - |
| gpu-rank | PASS | `brankreorgpkg0608` | 101/101 ranked, 0 timing-missing, 0 flips vs brankall6 | - |
| release-gate | PASS | `-` | release-gate OK: all 101 release-ready (score-eligible AND C2-aligned) task(s) release_status == PASS (phase_a_prime.sta | - |

**Cleanup:** reverted 101 run-churned meta, dropped 0 stale untracked tasks (none)

Fresh batch-id per stage (preflight refuses a pre-existing results dir); a stage is PASS only on rc==0 + full task coverage. C2 verdicts read from results/<batch>/r10/.
