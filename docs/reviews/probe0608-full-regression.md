# Full GPU regression -- run probe0608

**Overall: GREEN** | commit `94e5ddb` | 3/3 PASS

| Stage | Result | Batch | Summary | Regressions |
|-------|--------|-------|---------|-------------|
| c2 | PASS | `bc2probe0608` | 3/3 PASS, 0 non-PASS (0 NEW vs bc2apply) | - |
| s2 | PASS | `bs2probe0608` | 3/3 severance PASS (cert-stage also attempted cost_model_xcheck + golden) | - |
| gpu-rank | PASS | `brankprobe0608` | 3/3 ranked, 0 timing-missing, 0 flips vs brankall6 | - |

**Cleanup:** reverted 3 run-churned meta, dropped 0 stale untracked tasks (none)

Fresh batch-id per stage (preflight refuses a pre-existing results dir); a stage is PASS only on rc==0 + full task coverage. C2 verdicts read from results/<batch>/r10/.
