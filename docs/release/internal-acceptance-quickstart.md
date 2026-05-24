# Acceptance Gate Quickstart(三机模式,LLM 在 C 机)

> **INTERNAL** — 文件名 `internal-*.md` 已被 `.gitattributes export-ignore`
> 排除,不进 release tarball;仅供 bench 团队 operator 跟着跑。
>
> 完整规范参考 `docs/release/internal-acceptance-runbook.md`(§1-§7),
> 本文档是面向具体操作的 step-by-step 速跑指引。

## TL;DR — 一键封装(推荐)

不想手动跑每条命令?用 `tab_quickstart.sh` 封装,只需在 conf 文件里填三个变量,
然后一条命令一个子命令:

```bash
# 1. 一次性配置(只填一遍,gitignored 防意外 commit)
cp scripts/automation/runners/tab_quickstart.conf.example ./tab_quickstart.conf
$EDITOR ./tab_quickstart.conf
# 改 TAB_GPFS_ROOT / TAB_C_HOST / TAB_C_CLI_CMD 三个必填变量为你的实际值

# 2. 跑各步(每条独立可重跑;子命令名见 `bash ... help`)
bash scripts/automation/runners/tab_quickstart.sh setup       # 验环境
bash scripts/automation/runners/tab_quickstart.sh verify-c    # 验 C 机 SSH + CLI
bash scripts/automation/runners/tab_quickstart.sh resolve     # 产 resolve log
bash scripts/automation/runners/tab_quickstart.sh smoke       # Phase 1 (~30 min)
bash scripts/automation/runners/tab_quickstart.sh full        # Phase all (~2-4 hr)
bash scripts/automation/runners/tab_quickstart.sh status      # 看进度 + 报告
bash scripts/automation/runners/tab_quickstart.sh clean       # 重新跑前清理
```

变量优先级 `env > conf file > defaults`,也可以临时 `TAB_GPFS_ROOT=/x bash ... status` 覆盖。
封装内部走的就是下面"手动逐 step"章节里同样的 5 个 runner 脚本,只是把参数收敛到一处。

> conf 路径优先级:`$TAB_CONFIG` env > `./tab_quickstart.conf`(repo-local,**推荐**)> `~/.tab-quickstart.conf`(老路径,fallback)

> 如需自定义每条命令的细节(`--c-port` / `--health-path` / Tier 2 daemon 等),
> 看下面的"手动逐 step"章节,直接调用 `run_acceptance_gate.sh` / `run_machine_a_daemon.sh` 等。

## 三机角色

| 机 | 角色 | 跑什么 |
|---|---|---|
| **A** | GPU 机 | bench(`phase_a_prime.py --verify`)+ 可选 daemon |
| **B** | CPU 机(orchestrator 主循环) | `fix_orchestrator.py` + LLM transport(SSH tunnel 或 SSH CLI) |
| **C** | LLM 机 | HTTP 服务 **或** CLI(已登,SSH 调) |

> 如果只有 1 台 GPU 机(没单独 CPU 机),把 A 和 B 合并:
> orchestrator 在 GPU 机本地跑,SSH 到 C 拿 LLM。

## LLM transport 两种模式(选一个)

bench 支持两种 LLM 接入方式,根据 C 机实际情况选:

| Transport | 适用场景 | C 机要求 |
|---|---|---|
| **http**(默认) | C 机起 OpenAI-compatible HTTP 服务(vllm / sglang / 自研) | HTTP `/v1/chat/completions` + `/v1/models` |
| **ssh-cli** | C 机只有 CLI(已登,用户 quota),没起 HTTP 服务 | CLI(任意名,接受 prompt 作为 argv 一部分) + SSH 通 |

> **http 模式**:runner 自动起 `ssh -L` tunnel(`--c-host` / `--c-port` 指向 C 机),orchestrator 通过 `http://localhost:<local-port>/v1/chat/completions` 调 LLM
> **ssh-cli 模式**:B → C 直接 `ssh <c-host> '<cli> exec <prompt>'`,bench 取 stdout 当 reply;**不需要** port-forward,**不需要** HTTP 服务

## Step 0:三机配置 worksheet(开跑前填好)

```
[A 机]  GPU host:                 ______________ (本机,或 ssh 目标)
        bench repo 路径:           /data1/code/agentic-rl/kernel/triton-ascend-bench
        共享存储路径(可本地):       /tmp/tab-shared

[B 机]  orchestrator 跑在哪:        ______________ (跟 A 同机就填同样)

[C 机]  LLM SSH 目标:              ______________ (user@host)
        LLM HTTP 端口:             ______________ (e.g. 8080)
        LLM endpoint health path:  /v1/models       (OpenAI 默认)

SSH 免密:B → C 必须能 `ssh -o BatchMode=yes <c-host> 'echo OK'` 输出 OK
```

如果 SSH 免密没配:

```bash
ssh-copy-id <c-host>
# 验证:
ssh -o BatchMode=yes -o ConnectTimeout=5 <c-host> 'echo OK'
```

## Step 1:pull 最新 main(在 A 机)

```bash
cd /data1/code/agentic-rl/kernel/triton-ascend-bench
git pull origin main
git log --oneline | head -10
```

应**至少**含这些 commit(顺序最新先;未来可能更多):

```
feat: add SSH subprocess transport for remote-CLI LLM
feat: add three-machine runner scripts + acceptance gate
feat: add oracle-mismatch + upstream-not-invocable verdict pipelines
feat: add fix_orchestrator with shapes-incomplete pipeline
feat: implement M0 automation skeleton + release boundary
docs: add three-machine automation design + impl plan
```

> 实际 hash 会变;只要看到 **"SSH subprocess transport"** 出现就说明拉到最新自动化栈
> 自检命令:`git log --oneline | grep -cE "SSH subprocess|three-machine|verdict pipelines|fix_orchestrator|M0 automation|automation design"` 应输出 ≥ 6

## Step 2:setup A 机环境(一次性)

```bash
bash scripts/automation/runners/setup_machine_a_gpu.sh \
  --gpfs-root /tmp/tab-shared
```

预期最后输出:

```
[OK] machine A ready: GPU=<name> mem=<>GB GPFS=/tmp/tab-shared
```

**常见失败**:

| 错误 | 修法 |
|---|---|
| `torch < 2.9.1` | `pip install torch==2.9.1+cu126 triton==3.5.1 -i <index>` |
| `CUDA not available` | `nvidia-smi` 验证 + `nvidia-container-toolkit` 装好 |
| `GPFS not writable` | `--gpfs-root` 改成可写路径(本地 `/tmp/tab-shared` 最简) |
| `pip install` 网络问题 | 加 `--skip-install` 跳过(假设 deps 已装) |

## Step 3:配 LLM transport(http 或 ssh-cli)

### 选项 A:http 模式(C 机起 HTTP 服务)

**前提**:
- C 机已起 OpenAI-compatible HTTP 服务(vllm / sglang / 自研),端口 `<c-port>`
- B 机 → C 机 SSH 免密通

**不需要手动起 tunnel** — runner 会按 `--c-host <c-ssh-host> --c-port <c-port>`
自动起 SSH tunnel + 健康探测 + cleanup。

**验证 SSH 通**(可选 sanity check,在 C 机本地探 LLM 端口):

```bash
ssh -o BatchMode=yes -o ConnectTimeout=5 <c-ssh-host> \
  "curl -s -o /dev/null -w '%{http_code}\n' http://localhost:<c-port>/v1/models"
```

应输出 `200`(或 `401` / `404` — 都算 service 起着)。

**如果失败**:
- C 机 LLM 服务没起 → 先去 C 机起 LLM
- C 机 LLM 端口不是 worksheet 填的那个 → `ssh <c-ssh-host> 'netstat -tlnp | grep LISTEN'`
- SSH key 没配 → 回 Step 0 配 `ssh-copy-id`

### 选项 B:ssh-cli 模式(C 机只有 CLI,无 HTTP 服务)

**前提**:
- C 机已装某个 CLI(代号 `<your-cli>`,如 OpenAI Codex CLI / 其他登录式 LLM CLI)
- C 机已登 CLI(用户的 quota)
- B 机 → C 机 SSH 免密通(`ssh <c-host> 'echo OK'` 输出 OK)
- **不需要** `ssh -L` tunnel
- **不需要** C 机起任何 HTTP service

**验证 CLI 通**(C 机上跑一次 `--version`):

```bash
ssh -o BatchMode=yes -o ConnectTimeout=5 <c-host> '<your-cli> --version'
```

应输出 CLI 版本号。如果失败:
- CLI 没装 → 去 C 机装好
- CLI 没登 → 在 C 机交互式登录(`<your-cli> login` 等)
- SSH 不通 → 回 Step 0 配 `ssh-copy-id`

**重要的 `{prompt}` placeholder 规则**:

后面 `--llm-cli-cmd` 参数传 CLI template,**必须**含字面 `{prompt}`,且**不要**在它周围加引号:

| 写法 | 正误 |
|---|---|
| `'<your-cli> exec {prompt}'` | 正确(bench 自己 shlex.quote 加引号) |
| `'<your-cli> exec "{prompt}"'` | 错误(双重引号 = 把内层引号当字面字符) |
| `'<your-cli> exec'` | 错误(缺 `{prompt}` placeholder) |

## Step 4:产 resolve log(BLOCKED 分类,~5-10 分钟)

如果之前没跑过 resolver,或 `pa_resolve.log` 太旧:

```bash
python3 scripts/resolve_upstream_entrypoints.py 2>&1 \
  | tee /tmp/pa_resolve.log
```

跑完确认:

```bash
wc -l /tmp/pa_resolve.log              # 至少几百行
grep -c shapes-incomplete /tmp/pa_resolve.log   # 应该 ~78
grep -c oracle-mismatch /tmp/pa_resolve.log     # 应该 ~17
grep -c upstream-not-invocable /tmp/pa_resolve.log  # 应该 ~2
```

如果分布偏太大,看 `/tmp/pa_resolve.log` 头尾(`head -30 /tmp/pa_resolve.log` + `tail -30 /tmp/pa_resolve.log`)排查环境问题。

## Step 5:Phase 1 smoke(3 个 shapes-incomplete,~15-30 分钟)

### 选项 A 命令(http 模式,接 Step 3 选项 A)

```bash
bash scripts/automation/runners/run_acceptance_gate.sh \
  --gpfs-root /tmp/tab-shared \
  --c-host <c-ssh-host> \
  --c-port <c-port> \
  --health-path /v1/models \
  --resolve-log /tmp/pa_resolve.log \
  --state-path /tmp/tab-shared/state/triage_state.json \
  --phase 1
```

> `<c-ssh-host>` 是 C 机真实 SSH 目标(`user@host` 或 `~/.ssh/config` alias)
> `<c-port>` 是 C 机本地 LLM HTTP 端口
> runner 自动 SSH tunnel 到 C 机,orchestrator 通过
> `http://localhost:<local-port>/v1/chat/completions` 调 LLM(`--local-port` 默认 8080)

### 选项 B 命令(ssh-cli 模式,接 Step 3 选项 B)

```bash
bash scripts/automation/runners/run_acceptance_gate.sh \
  --gpfs-root /tmp/tab-shared \
  --llm-transport ssh-cli \
  --llm-ssh-host <c-ssh-host> \
  --llm-cli-cmd '<your-cli> exec {prompt}' \
  --resolve-log /tmp/pa_resolve.log \
  --state-path /tmp/tab-shared/state/triage_state.json \
  --phase 1
```

> ssh-cli 模式**不需要** `--c-port` / `--local-port` / `--health-path`(没 HTTP)
> `<c-ssh-host>` 是 Step 0 worksheet 里 C 机 SSH 目标(`user@host` 或 `~/.ssh/config` alias)
> `<your-cli>` 是 C 机上实际装的 CLI 名(替成你的);**注意** `{prompt}` 周围**没有引号**

**预期**:
- 跑 ~15-30 分钟(取决于 LLM 速度)
- 真调 LLM 3-9 次(每 task 1-3 次,根据 retry)
- 真跑 GPU verify
- 产 `docs/release/internal-acceptance-report-<YYYYMMDD>.md` Phase 1 章节
- Budget stanza 含 LLM token 用量 + 估算 cost

**PASS 判定**:
- ≥1 task 状态 `DONE`(LLM 真修出有效 patch)
- 0 task 卡 `ATTEMPTING_NEXT_ROUND`(retries 没 exhaust)
- 0 task 直接 raise(都 reach 某终态)

**FAIL 排查**:

| 现象 | 排查路径 |
|---|---|
| `LLM endpoint not reachable` | 看 runner 日志里 SSH tunnel 起动行 + curl probe 输出;tunnel 由 runner 管理,失败时检查 `--c-host` / `--c-port` 是否正确 |
| `--resolve-log missing or empty` | 回 Step 4 重产 |
| 所有 task `UNFIXABLE_BY_LLM` | LLM 太弱 / prompt 走偏;看 `/tmp/tab-shared/logs/orchestrator-*.log` 里 LLM 返回原文 |
| GPU 报错 OOM | 单独跑 `python3 scripts/phase_a_prime.py --verify --task <t>` 复现 + 减 BLOCK_SIZE |

## Step 6:Phase 2-4 全跑(Phase 1 PASS 后,~2-4 小时)

### 选项 A 命令(http 模式)

```bash
bash scripts/automation/runners/run_acceptance_gate.sh \
  --gpfs-root /tmp/tab-shared \
  --c-host <c-ssh-host> \
  --c-port <c-port> \
  --health-path /v1/models \
  --resolve-log /tmp/pa_resolve.log \
  --state-path /tmp/tab-shared/state/triage_state.json \
  --phase all
```

### 选项 B 命令(ssh-cli 模式)

```bash
bash scripts/automation/runners/run_acceptance_gate.sh \
  --gpfs-root /tmp/tab-shared \
  --llm-transport ssh-cli \
  --llm-ssh-host <c-ssh-host> \
  --llm-cli-cmd '<your-cli> exec {prompt}' \
  --resolve-log /tmp/pa_resolve.log \
  --state-path /tmp/tab-shared/state/triage_state.json \
  --phase all
```

**Phase 分解**:

| Phase | 范围 | 估时 |
|---|---|---|
| 1 | 3 个 shapes-incomplete(同 Step 5) | 15-30 min |
| 2 | 5 shapes + 5 oracle-mismatch + ≤2 upstream-not-invocable | 30-60 min |
| 3 | 全 101 task 语料 | 1.5-3 hr |
| 4 | 5 task Tier 1 vs Tier 2(自动 git worktree 隔离) | 10-20 min |

**产出**:

- `docs/release/internal-acceptance-report-<YYYYMMDD>.md`
  4 phase 全 PASS/FAIL + budget(LLM cost / token / call 数)
- `docs/triage_backlog/backlog-report-<YYYYMMDD>.{md,json}`
  按 state 分类列 non-DONE tasks
- `docs/triage_backlog/summary-metrics-<YYYYMMDD>.{md,json}`
  4 个独立 metric:reverify_nonblocked / phase_a_promoted / golden_ready / manual_backlog

## Step 7:cleanup

```bash
# runner 退出时自动清理 SSH tunnel,无需手动 kill。
# 若 runner 被强杀(SIGKILL)留下 orphan tunnel,可手动清:
# ps -ef | grep '[s]sh -N -L' | awk '{print $2}' | xargs -r kill

# 可选:清理 state(下次重跑前)
rm -rf /tmp/tab-shared/state-phase*
rm -rf /tmp/tab-shared/state/triage_state.json
```

## 三机分离模式(可选,Phase 4 更纯)

如果你确实有独立的 B 机(CPU)和 C 机(LLM)且共享 GPFS:

**A 机起 daemon**(长期运行):

```bash
bash scripts/automation/runners/run_machine_a_daemon.sh \
  --gpfs-root <shared-gpfs-path>
```

**B 机起 orchestrator + tunnel + 跑 acceptance gate**:

```bash
bash scripts/automation/runners/setup_machine_b_cpu.sh \
  --gpfs-root <shared-gpfs-path> \
  --a-host <a-ssh> \
  --c-host <c-ssh>

# 跑 acceptance gate(自动管理 tunnel,不需 Step 3 手动 ssh -L):
bash scripts/automation/runners/run_acceptance_gate.sh \
  --gpfs-root <shared-gpfs-path> \
  --c-host <c-ssh-host> \
  --c-port <c-port> \
  --health-path /v1/models \
  --resolve-log <shared-gpfs>/pa_resolve.log \
  --state-path <shared-gpfs>/state/triage_state.json \
  --phase all
```

`<shared-gpfs-path>` 必须 A + B 都能读写(NFS / GPFS mount)。

## Acceptance criteria(详 `internal-acceptance-runbook.md` §7)

| Phase | PASS 判定 |
|---|---|
| 1 | ≥1 task DONE,0 task 卡 ATTEMPTING_NEXT_ROUND |
| 2 | 三 verdict 闭环正常;defer-to-golden task → BACKLOG_PENDING_GOLDEN |
| 3 | backlog_report + summary_metrics 都非空;4 metric 计算值合理 |
| 4 | 5 task Tier 1 vs Tier 2 verdict_class + state 一致 |

4 phase 全 PASS = 三机自动化 acceptance gate 通过 = M1.11 + M2.9 + M3.3 + M4.3 真跑完成。

## 真跑前最后 checklist

- [ ] `git pull origin main` 后 `git log --oneline | head -1` 含 **"SSH subprocess transport"**(或更新的 commit;意指拉到含 ssh-cli transport 的最新自动化栈)
- [ ] `nvidia-smi` 看到 GPU
- [ ] `python3 -c "import torch; print(torch.__version__)"` 是 2.9.1+
- [ ] C 机 LLM 服务起着 + B 机 → C 机 SSH 免密通
- [ ] http 模式:`ssh <c-ssh-host> "curl -s -o /dev/null -w '%{http_code}\n' http://localhost:<c-port>/v1/models"` 返 200/401/404(runner 起 tunnel,无需手动)
- [ ] `pa_resolve.log` 有合理分布(78/17/2/4)

全 ✓ → 跑 Step 5。
