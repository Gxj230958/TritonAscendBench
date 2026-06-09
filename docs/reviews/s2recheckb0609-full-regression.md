# Full GPU regression -- run s2recheckb0609

**Overall: RED** | commit `0a79249` | 0/1 PASS

| Stage | Result | Batch | Summary | Regressions |
|-------|--------|-------|---------|-------------|
| s2 | FAIL | `bs2s2recheckb0609` | orchestrator rc=2 (DEAD/timeout/refused) | orchestrator rc=2 (DEAD/timeout/refused) |

**Cleanup:** no run-introduced churn

Fresh batch-id per stage (preflight refuses a pre-existing results dir); a stage is PASS only on rc==0 + full task coverage. C2 verdicts read from results/<batch>/r10/.
