# Full GPU regression -- run renum0608

**Overall: RED** | commit `94e5ddb` | 3/4 PASS

| Stage | Result | Batch | Summary | Regressions |
|-------|--------|-------|---------|-------------|
| c2 | PASS | `bc2renum0608` | 101/101 PASS, 0 non-PASS (0 NEW vs bc2apply) | - |
| s2 | FAIL | `bs2renum0608` | 100/101 severance PASS (cert-stage also attempted cost_model_xcheck + golden) | L3_27_chunk_scaled_dot_kkt_fwd: not PASS |
| gpu-rank | PASS | `brankrenum0608` | 101/101 ranked, 0 timing-missing, 0 flips vs brankall6 | - |
| release-gate | PASS | `-` | release-gate OK: all 101 release-ready (score-eligible AND C2-aligned) task(s) release_status == PASS (phase_a_prime.sta | - |

**Cleanup:** reverted 101 run-churned meta, dropped 0 stale untracked tasks (none)

Fresh batch-id per stage (preflight refuses a pre-existing results dir); a stage is PASS only on rc==0 + full task coverage. C2 verdicts read from results/<batch>/r10/.
