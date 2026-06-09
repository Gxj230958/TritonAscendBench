# Full GPU regression -- run frzfinal

**Overall: GREEN** | commit `1b18e22` | 4/4 PASS

| Stage | Result | Batch | Summary | Regressions |
|-------|--------|-------|---------|-------------|
| c2 | PASS | `bc2frzfinal` | 101/101 PASS, 0 non-PASS (0 NEW vs bc2apply) | - |
| s2 | PASS | `bs2frzfinal` | 101/101 severance PASS (cert-stage also attempted cost_model_xcheck + golden) | - |
| gpu-rank | PASS | `brankfrzfinal` | 101/101 ranked, 0 timing-missing, 0 flips vs brankall6 | - |
| release-gate | PASS | `-` | release-gate OK: all 101 release-ready (score-eligible AND C2-aligned) task(s) release_status == PASS (phase_a_prime.sta | - |

**Cleanup:** reverted 101 run-churned meta, dropped 0 stale untracked tasks (none)

Fresh batch-id per stage (preflight refuses a pre-existing results dir); a stage is PASS only on rc==0 + full task coverage. C2 verdicts read from results/<batch>/r10/.
