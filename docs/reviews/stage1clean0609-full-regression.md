# Full GPU regression -- run stage1clean0609

**Overall: RED** | commit `0a79249` | 4/5 PASS

| Stage | Result | Batch | Summary | Regressions |
|-------|--------|-------|---------|-------------|
| unit | PASS | `-` | 0 failed, 0 errors, 0 skipped, 2277 passed (baseline 0) | - |
| s2 | FAIL | `bs2stage1clean0609` | 100/101 severance PASS (cert-stage also attempted cost_model_xcheck + golden) | L3_30_merge_16x16_to_32x32_inverse: not PASS |
| c2 | PASS | `bc2stage1clean0609` | 101/101 PASS, 0 non-PASS (0 NEW vs bc2reorgfull0609) | - |
| gpu-rank | PASS | `brankstage1clean0609` | 101/101 ranked, 0 timing-missing, 0 flips vs brankreorgfull0609 | - |
| release-gate | PASS | `-` | release-gate OK: all 101 release-ready (score-eligible AND C2-aligned) task(s) release_status == PASS (phase_a_prime.sta | - |

**Cleanup:** reverted 100 run-churned meta, dropped 0 stale untracked tasks (none)

Fresh batch-id per stage (preflight refuses a pre-existing results dir); a stage is PASS only on rc==0 + full task coverage. C2 verdicts read from results/<batch>/r10/.
