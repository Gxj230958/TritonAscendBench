# 提交限流 + Sybil 检测

> English: [../../reference/submission_throttling.md](../../reference/submission_throttling.md)

## 速率限制（按任务、按 org 身份）

| Mode | Limit |
|------|-------|
| `self-reported`（v0.1） | **每周每个 (org_identity, task_id) 限 3 次提交** |
| `verified`（v1.0） | **每周每个 (org_identity, task_id) 限 2 次提交**（verified 运行开销大） |
| 初次接入（org 的首次提交） | 1 次免费提交，之后按标准限额 |

## Org 身份推导

每次提交的 `org_identity` = 以下内容的哈希：
- GPG 签名密钥指纹（首选——最强） 或
- 邮箱域名（回退——例如 `@example-university.edu`） 或
- IP 块（最后回退——IPv4 用 `/24`，IPv6 用 `/64`）

## Sybil 检测

多个"不同"的 org_identity 表现出协同模式：

| Signal | Threshold | Action |
|--------|----------|--------|
| 同一 Docker digest 横跨 N+ 个"不同"提交者 | N ≥ 3 | 标记交由人工审计 |
| 同一源代码哈希横跨"不同"提交者 | N ≥ 2 | 标记 |
| 提交时间突发（来自不同 IP 且间隔 < 60s） | 3+ | 限流 + 记录日志 |
| 同一 `runtime_env/Dockerfile` SHA 横跨多个 org | N ≥ 5 | 审计 |
| 被拒后在隐藏 shape 上出现可疑的渐进式迭代 | 启发式 | 阻断 + 人工复核 |

Sybil 标记 → 提交仅保留在 `self-reported` 榜单上，不进入 verified 排名。该 org 将受到更慢的限流约束。

## 借失败探测（Probe-Through-Failure）防御

如果提交者耗尽本周额度，但其重新提交集中聚集在隐藏 shape 的边界附近（这是探测行为的统计特征），则封禁该提交者 30 天。

每个 epoch（约 6 个月）轮换隐藏对抗集，以限制累积的信息泄露。

## 隐藏结果日志脱敏

发布 verified 结果 JSON 时：
- 公开日志：`{task_id, aggregate_pass_rate, public_shape_timings, container_digest}`
- 隐去内容：`hidden_shape_dimensions`、`hidden_shape_failure_inputs`、`expected_vs_actual_tensor_data`

隐藏的失败原因会做聚合：
```
"L3_04 hidden cases: 5 passed, 2 numerical_diff_exceeded, 1 nan_inf_inconsistency"
```

而**不是**：
```
"L3_04 hidden case [seq=12345, hidden=4096]: out_c[0,5] = 1.234 vs golden = 1.245"  # never published
```

## 额度重置

- 周窗口：每周一 00:00 UTC（滑动 7 天窗口）
- 手动重置：org-admin 通过 ops API 操作（有审计记录）

## 存储

由 verified-runner 排行榜 / broker 基础设施管理的提交者额度数据库（SQLite 或 Postgres）——属于未来工作，参见 `docs/DESIGN.md §6`。
