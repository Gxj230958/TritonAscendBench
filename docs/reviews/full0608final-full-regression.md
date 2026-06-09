# Full GPU regression -- run full0608final

**Overall: GREEN** | commit `120fc44` | 5/5 PASS

| Stage | Result | Batch | Summary | Regressions |
|-------|--------|-------|---------|-------------|
| unit | PASS | `-` | 0 failed, 0 errors, 0 skipped, 2279 passed (baseline 121) | - |
| c2 | PASS | `bc2full0608final` | 101/101 PASS, 0 non-PASS (0 NEW vs bc2apply) | - |
| s2 | PASS | `bs2full0608final` | 101/101 severance PASS (cert-stage also attempted cost_model_xcheck + golden) | - |
| gpu-rank | PASS | `brankfull0608final` | 101/101 ranked, 0 timing-missing, 0 flips vs brankall6 | - |
| release-gate | PASS | `-` | release-gate OK: all 101 release-ready (score-eligible AND C2-aligned) task(s) release_status == PASS (phase_a_prime.sta | - |

**Cleanup:** reverted 101 run-churned meta, dropped 0 stale untracked tasks (none)

Fresh batch-id per stage (preflight refuses a pre-existing results dir); a stage is PASS only on rc==0 + full task coverage. C2 verdicts read from results/<batch>/r10/.
