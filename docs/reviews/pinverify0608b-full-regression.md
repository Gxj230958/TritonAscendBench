# Full GPU regression -- run pinverify0608b

**Overall: GREEN** | commit `ed1ddeb` | 1/1 PASS

| Stage | Result | Batch | Summary | Regressions |
|-------|--------|-------|---------|-------------|
| s2 | PASS | `bs2pinverify0608b` | 1/1 severance PASS (cert-stage also attempted cost_model_xcheck + golden) | - |

**Cleanup:** reverted 1 run-churned meta, dropped 0 stale untracked tasks (none)

Fresh batch-id per stage (preflight refuses a pre-existing results dir); a stage is PASS only on rc==0 + full task coverage. C2 verdicts read from results/<batch>/r10/.
