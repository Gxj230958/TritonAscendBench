I have all facts verified. Producing the plan now.

---

# Strategy-C 全量重编号迁移方案（triton-ascend-bench 公开发布版）

> 状态：**待 codex 签字**（执行前）。本方案由三份穷尽 blast-radius 审计 + dev 侧事实校验驱动，所有脚本以 `/tmp/renumber_map.json`（101 条全量）与 `/tmp/renumber_changed.json`（87 条 old→new）为单一事实源。
> 已校验事实（dev 侧）：map 101 条 / 改名 87 / 不变 14 / 跨层移动 38；新分布 **L1=36 · L2=35 · L3=30** 三层全部连续 `1..N`；磁盘旧目录 level1=40/level2=39/level3=22=101 全部存在；**git mv 路径零碰撞**（每个 new 全路径都不与任何 old 全路径相等，new 全 id 唯一）；dev 侧 `golden/` 子目录多为空（真 golden bytes 只在 gpfs `sealed_golden/`，由 Step g 迁移；空目录 git 不跟踪也无妨，不影响 golden 解析）；`golden_store.py:131` 确认 `sealed_golden/<task_id>/sealed_<shape>.safetensors`——task_id 仅作路径段。

---

## 1. 目标与最终态

把语料重排成 **干净、连续、可公开发布** 的布局：`L1_01..L1_36 / L2_01..L2_35 / L3_01..L3_30`，每层无空洞、idx 连续、`dir 名 == meta.task_id == 所在 levelN`（三者严格一致）。

"completely fine" 的验收定义（缺一不可）：

1. **0 dead id**：87 个 old 全 id 在仓库活跃面（`tests/ scripts/` + 活跃 `tasks/` + 公开文档 `README.md/SCORE.md/docs/kernel-select.md` + 两个 code-consumed YAML + resolve 缓存）grep 命中为 **0**（历史 dated 文档与 `_vendored/` 除外，见 §6）。
2. **三向一致**：`scripts/validate_tasks.py` 0 error（证明 dir / `meta.task_id` / contract.md short_id 对齐），`scripts/validate_corpus_consistency.py` 0 violation（证明每个 `CEILING_ACCEPTED` 任务的 **new** id 都在迁移后的 ceiling allowlist 上）。
3. **dev unit 0-fail / 0-skip**：`pytest tests/unit_tests -q` 全绿，且 disk-glob 测试**真正执行**（不得因目录消失静默 skip）；23 个 pending unit fix 一并落地（见 §4）。
4. **GPU 全流程绿**：daemon 真机对 **renamed 101** 跑通 C1/C2/S2/cost/golden/release-gate/gpu-rank，golden 在 **new key** 下解析成功，无 ceiling/floor 评分回归。
5. **文档对齐**：`docs/kernel-select.md §7` 按新 level/idx 重排，counts → 101 / `L1 36·L2 35·L3 30`；`README.md`/`SCORE.md` 的 per-level 分布与 prose id 全部更新。

---

## 2. 迁移总览与不变量

### 2.1 改名 / 稳定切分

- **87 改名**（`renumber_changed.json` 全集），其中 **38 跨层移动**（`moved:true`，如 `L2_16_fused_rmsnorm→L1_28`、`L3_07_correct_attn_cp_out→L2_32`、`L2_18_fused_gdn_gating→L1_29`）。
- **14 不变**：`L1_01..L1_12`、`L2_25_fwd_kernel_ep_gather`、`L2_26_fwd_kernel_ep_scatter_1`——脚本一律 `if old==new: continue`，绝不触碰。

### 2.2 让全程安全的**唯一核心不变量**

> **Golden 字节不变，只重命名存储 KEY，绝不 GPU 重采。**

证据链：`golden_store.py:131` 的 sealed 路径 `gpfs/sealed_golden/<task_id>/sealed_<shape_id>.safetensors` 中 task_id **仅作目录段**；`meta.yaml provenance.golden.golden_sha256` 是 safetensors 字节内容哈希，**provenance 里没有 task_id 字段**。因此：dev 侧 `tasks/<id>/golden/` 随 `git mv` 自动跟走；gpfs 侧只需 `mv sealed_golden/<old> sealed_golden/<new>`，sha256 保持有效。**任何 golden 都不需要 GPU 重新 seal/capture。** 这把整个迁移降级为"key/path 簿记 + 文本改写"。

### 2.3 需要 / 不需要 GPU

| 不需要 GPU（dev / daemon-storage 侧，纯 git + 文件 mv + 文本改写） | 需要 GPU（仅最终回归一次） |
|---|---|
| `git mv` 87 dir；改 87 份 `meta.yaml`（task_id/level/idx）；改 87 份 `contract.md`；map 驱动改写 `tests/ scripts/` 全部 old-id 引用；改 `ceiling_acceptance.yaml`/`floor_dominated_acceptance.yaml`；改 `README/SCORE/kernel-select`；gpfs `sealed_golden/<old>→<new>` 17 个目录 mv（存储 rename，无算力）；regenerate `resolve_all.json`（静态 AST 解析，无 GPU）；gpfs 树 `--delete` 同步 | daemon 真机对 renamed 101 跑 **C1/C2 oracle-equiv + S2/phase_a_prime + cost_model xcheck + golden verify + release-gate + gpu-rank**——唯一目的是确认 golden 在 new key 下解析 + 无 ceiling/floor 评分回归（按 MEMORY「Full regression = whole GPU pipeline」要求，全 101 真机跑） |

---

## 3. 逐步迁移序列（rollback-safe，全脚本化）

> 全部在**专用 migration 分支**上执行（`git switch -c renumber-public-release`），commit 前每一步可 `git restore`/`git reset --hard` 回滚。每步给出脚本 + 验证。脚本一律以 map 为源、fail-closed（断言不符即 `assert` 崩，绝不静默跳过）。

### Step (a) — 冻结映射 + 快照

```bash
cd /data1/code/agentic-rl/kernel/triton-ascend-bench
mkdir -p scripts/renumber
cp /tmp/renumber_map.json     scripts/renumber/renumber_map.json
cp /tmp/renumber_changed.json scripts/renumber/renumber_changed.json
# 入仓冻结，使迁移可复现 / 可审计。后续脚本一律读这两份（不再读 /tmp，防漂移）
git add scripts/renumber && git status --porcelain scripts/renumber
```

校验：`python3 -c "import json;m=json.load(open('scripts/renumber/renumber_map.json'));assert len(m)==101 and sum(k!=v['new'] for k,v in m.items())==87"`。

### Step (b) — `git mv` 87 个 dir（保留 git 历史 + golden/ 子目录随迁）

```bash
python3 - <<'PY'
import json, os, subprocess
m = json.load(open('scripts/renumber/renumber_map.json'))
LD = {'L1':'level1','L2':'level2','L3':'level3'}
moved = 0
for old, v in m.items():
    if old == v['new']:
        continue
    op = f"tasks/{LD[v['old_level']]}/{old}"
    np = f"tasks/{LD[v['new_level']]}/{v['new']}"
    assert os.path.isdir(op), f"missing old dir {op}"
    assert not os.path.exists(np), f"target already exists {np}"  # 零碰撞已校验
    subprocess.run(["git", "mv", op, np], check=True)
    moved += 1
print("git mv count:", moved)
PY
```

为什么安全：dev 侧已校验 **0 个 new 全路径等于任一 old 全路径**，故单次顺序 mv 无中间碰撞；`assert not exists(np)` 是兜底。`git mv` 保留历史（`git log --follow` 可追），`golden/`、`r10_pyref/`、`kernel.py`、`oracle_pytorch.py`、`shapes.yaml` 等子文件全部随迁——这一步**自动修复**所有「按 dir/glob 定位」的消费者：public golden（`golden_store.py:84-107`）、manifest、census、planning-gate、`cost_model`（`meta.yaml` 内联随迁）。

校验：`git diff --cached --name-status -M | grep -c '^R'`（应 ≈ 87 dir × 每 dir 文件数，全部 `R` 重命名）；抽查历史 `git log --follow --oneline -- tasks/level1/L1_13_fused_softcap/kernel.py | head`。

### Step (c) — 改写 87 份 `meta.yaml`（task_id + idx + level，先断言旧值）

```bash
python3 - <<'PY'
import json, re, os
m = json.load(open('scripts/renumber/renumber_map.json'))
LD = {'L1':'level1','L2':'level2','L3':'level3'}
n = 0
for old, v in m.items():
    if old == v['new']:
        continue
    new, nl = v['new'], v['new_level']
    nidx = int(new.split('_')[1])
    p = f"tasks/{LD[nl]}/{new}/meta.yaml"   # dir 已在 (b) 移到 new 路径
    s = open(p).read()
    # 前置不变量断言：dir 已动但内容仍是 OLD（半迁移防御）
    assert re.search(rf'(?m)^task_id:\s*{re.escape(old)}\s*$', s), f"{p}: task_id != {old}"
    # 行内锚定改写，绝不 yaml round-trip（会重排 key、毁掉 cost_model 注释格式）
    s = re.sub(r'(?m)^task_id:\s*.*$', f'task_id: {new}', s, count=1)
    s = re.sub(r'(?m)^level:\s*.*$',   f'level: {nl}',    s, count=1)
    s = re.sub(r'(?m)^idx:\s*.*$',     f'idx: {nidx}',    s, count=1)
    open(p, 'w').write(s)
    n += 1
print("meta.yaml rewritten:", n)
PY
```

依据审计：`meta.yaml:3 task_id`（`validate_tasks.py:71` 必填 + `:563` 派生 short_id 跨核 contract.md）、`meta.yaml level`（`validate_tasks.py:1173` ∈ VALID_LEVELS + `dataset.py:129-134` 运行时 `TaskDef.level` 来源——**没有任何检查比对 meta.level 与物理 dir**，半迁移会静默错分 scoring，故必须改）、`meta.yaml idx`（非硬校验但清洁语料要求 `idx=int(new.split('_')[1])`）。`name:` **不改**（= kernel 函数名，与 id 无关）。注意 `^task_id:`/`^level:`/`^idx:` 行首锚定只命中顶层 key，嵌套 `upstream.repo` 等不受影响（meta 顶层 task_id/level/idx 行号因任务而异，故用字段锚定而非行号）。**幂等**：重跑时 old 值已不在 → 断言失败提示已迁移，不会双改。

### Step (d) — 改写 87 份 `contract.md`（每任务局部、3 处全 id + bare short）

```bash
python3 - <<'PY'
import json, re
m = json.load(open('scripts/renumber/renumber_map.json'))
LD = {'L1':'level1','L2':'level2','L3':'level3'}
n = 0
for old, v in m.items():
    if old == v['new']:
        continue
    new = v['new']
    olds = '_'.join(old.split('_')[:2]); news = '_'.join(new.split('_')[:2])
    p = f"tasks/{LD[v['new_level']]}/{new}/contract.md"
    s = open(p).read()
    s = s.replace(old, new)  # 全 id：line 1 '# Contract — <id>', 8 '- Benchmark id: `<id>`', 15 'Oracle reference for <id>.'
    # bare short-id（如 'L1_18' 单独出现），词边界防 L1_180 / 不碰已替换的新全 id
    s = re.sub(rf'(?<![A-Za-z0-9_]){re.escape(olds)}(?![0-9A-Za-z_])', news, s)
    open(p, 'w').write(s)
    n += 1
print("contract.md rewritten:", n)
PY
```

依据 `contract.md:1,8,15`（dev 实测：renamed task 三行均为**全 id**）+ `validate_tasks.py:562-568`（要求 `meta.task_id` 的 short_id 出现在 contract.md，否则硬 FAIL）。**严禁** `phase_c_docs.py process_body`——它会 clobber `shapes.yaml`（MEMORY「phase_c_docs clobbers shapes」）；`--refresh-mirror-only` 只改 param_constraints 镜像，改不到这三行。故用上面的局部文本替换。

### Step (e) — map 驱动一次性改写 `tests/ scripts/` + 两个 code-consumed YAML（全 id token）

```bash
python3 - <<'PY'
import json, re, pathlib
ch = json.load(open('scripts/renumber/renumber_changed.json'))
keys = sorted(ch, key=len, reverse=True)  # longest-first，彻底防前缀互吃
pat = re.compile(r'(?<![A-Za-z0-9_])(' + '|'.join(map(re.escape, keys)) + r')(?![A-Za-z0-9_])')
EXTS = {'.py', '.sh', '.md', '.js', '.json', '.yaml', '.yml'}
EXCL = {'_vendored', '3rdparty', '__pycache__', '.git'}
roots = [pathlib.Path('tests'), pathlib.Path('scripts')]
extra = [pathlib.Path('triton_ascend_bench/ceiling_acceptance.yaml'),
         pathlib.Path('triton_ascend_bench/floor_dominated_acceptance.yaml')]
changed_files = 0
def rewrite(p):
    global changed_files
    t = p.read_text()
    nt = pat.sub(lambda mm: ch[mm.group(1)], t)
    if nt != t:
        p.write_text(nt); changed_files += 1; print("  ", p)
for r in roots:
    for p in r.rglob('*'):
        if p.suffix in EXTS and not any(d in p.parts for d in EXCL):
            rewrite(p)
for p in extra:
    rewrite(p)
print("files rewritten:", changed_files)
PY
```

这一遍同时修掉审计列出的**全部 6 个 load-bearing 注册表**与 ~21 个测试文件 + 全部注释/docstring/README + 两个 YAML allowlist：

- `scripts/phase_b_oracle.py:2654-2764` **WORKING_REFS**（~78 个 renamed key，dev 实测 `dict[str, tuple[str,str]]` 全 id 字面 key）、`:2777-2830` **ORACLE_SCOPE_DOWNGRADE**、`:42-44` **SKIP_TASKS**（`L1_16_silu_and_mul→L2_04`、`L2_10_w8a8_block_int8_matmul→L2_17`、`L3_04_fwd→L3_16`）。
- `scripts/phase_c_docs.py:265-545` **SHAPE_OVERRIDES**（~50 key）、`:945` **_MANUAL**（`L1_24_rejection_sample→L2_11`）、`:50-51` **SKIP_POC**。
- `scripts/phase_a_freeze.py:55-59` **SKIP_TASKS**；`scripts/generate_task_scaffolds.py:46-50` **KNOWN_POC_DIRS**（codex#4：其 **keys 是 markdown id**「L1-16」**非 task_id**，只 **values** 是 full task_id —— 全局 full-id 替换正确地只改 values；markdown-id keys 保留指向历史 selection doc，不动）。
- 测试 load_task/parametrize/path 字面：`test_kernel_resolver.py`、`test_r10_multi_output_order.py`、`test_launch_blocked_fixes.py`、`test_phase_a_prime_s2_soundness.py`、`test_pa_fix1_*`、`test_oracles.py`、`test_dataset.py`、`test_phase_b_golden_capture.py`、`test_phase_a_cost_xcheck.py`、`test_golden_store.py`、`test_phase_c_generators_wa5.py`、`test_resolve_upstream_entrypoints.py`、`test_phase_a_prime_rewrite.py`、`test_wrapper_contract.py`、`test_r10a_wrapper_contract_classification.py`、`test_static_checker.py`、`test_cli_w1_4.py`、`test_eval_w4_3.py`、`test_param_constraints_wa4.py`、`test_validate_tasks_param_constraints.py`、`test_r2_review_fixes.py`、`test_manifest.py`、`test_upstream_shims.py`。
- `scripts/_test_ceiling_acceptance.py` 的 MODEB / `L2_03_chunk_state_fwd` 字面 **与** `ceiling_acceptance.yaml`（`tol_units_tasks` 18/19 改名 + `physical_tasks: L3_13_…→L3_22_…`）**锁步**同遍改写——dev 实测该 YAML 当前列的就是 old id（`L1_26_merge_state`/`L2_01_bmm_chunk_fwd`/`L2_02_chunk_scan_fwd`/`L2_03_chunk_state_fwd`/`L2_17_layer_norm_gated_fwd`/`L2_19_…`/`L3_03_…`/`L3_04_fwd`/`L3_09_fused_moe`/`L3_14/15/16`/`L3_20`/物理 key `L3_13_…`）。
- `floor_dominated_acceptance.yaml`（`phase_a_cost_xcheck.py:148` 消费）5/6 list 改名：`L1_28_pack_bitmatrix→L2_12`、`L1_35_compute_src2dst_triton→L1_23`、`L1_37_compute_problem_sizes_w4a8→L1_25`、`L1_24_rejection_sample→L2_11`、`L1_29_prepare_eagle_inputs→L1_17`（`L1_04` 不变）。
- 全部非破坏注释/docstring/README（`phase_a_prime.py` 多处、`phase_a_cost_xcheck.py`、`validate_tasks.py`、`oracle_equivalence_gpu_runner.py`、`auto_shapes_filler.py`、`_c2_sweep_analyze.py`、`oracle_pyref_precompute.py`、`operator_slice.py`、`multi_launched_manifest.py`、`per_task_diagnostic.py`、`loop/_wf_authoritative_route.js`、`loop/README.md`、`gpu_pipeline/README.md`）——`renumber_changed.json` 只含**全 id**，故注释里的短 id（如 `L1_13`、`L3_02`）不会被这遍命中；这些是 cosmetic，单独在 §6「可选清洁」里处理。

边界 `(?<![A-Za-z0-9_])…(?![A-Za-z0-9_])` + longest-first 保证只命中完整全 id token、前缀互吃不会发生（`L3_15_..._fwd` 不会误吃 `L3_15_..._update_fwd`）。**不含 `tasks/`**（meta/contract 已由 c/d 专门处理，避免双改）。

校验：`git grep -nIE` 全 87 old-id（见 §5 dead-id 守卫脚本）在 `tests/ scripts/` 应 0；`python3 -c "import ast,scripts...; ..."` 或直接 `python3 -c "import importlib.util as u;[u.spec_from_file_location('m',f) for f in ['scripts/phase_b_oracle.py','scripts/phase_c_docs.py']]"` 语法自检。

### Step (f) — 修跨层移动的 `tasks/levelN/<id>` 字面 level 段

```bash
python3 - <<'PY'
import json, re, pathlib
m = json.load(open('scripts/renumber/renumber_map.json'))
LD = {'L1':'level1','L2':'level2','L3':'level3'}
correct = {v['new']: LD[v['new_level']] for v in m.values()}   # new_id -> 正确 level 目录
pat = re.compile(r'(tasks/)(level[123])(/)(L[123]_[0-9]+_[A-Za-z0-9_]+)')
def fix(mm):
    pre, lvl, sl, tid = mm.groups()
    want = correct.get(tid)
    return f"{pre}{want}{sl}{tid}" if want and want != lvl else mm.group(0)
n = 0
for root in ('tests', 'scripts'):
    for p in pathlib.Path(root).rglob('*'):
        if p.suffix in {'.py', '.sh', '.md', '.js'} and '__pycache__' not in p.parts:
            t = p.read_text(); nt = pat.sub(fix, t)
            if nt != t:
                p.write_text(nt); n += 1; print("  ", p)
print("level-segment files fixed:", n)
PY
```

依据审计 33 个 `moved:true` 跨层硬路径：`test_kernel_resolver.py`（`L1_20_log_softmax→level2/L2_07`、`L1_17_gelu_and_mul→level2/L2_05`、`L3_07_correct_attn_cp_out→level2/L2_32`）、`test_dataset.py`（`L1_16` level1→level2）、`test_r10a_wrapper_contract_classification.py`（`L3_22_chunk_local_cumsum_vector→level2/L2_35`、`L3_21_solve_tril_16x16→level2/L2_34`）。Step (e) 已把 id 文本改成 new，本步把残留的错 level 段机械纠正（new_id 的正确 level 来自 map）。

### Step (g) — gpfs `sealed_golden` key rename（daemon/storage 侧，幂等 + 可逆）

> 通过 tunnel 在 daemon B 跳板投递到 daemon A 执行（MEMORY「loop topology」：dev 只能 ssh 到 B，daemon 在 A，探活/作业只能投 mailbox）。**先 quiesce daemon**（① 无 in-flight job；② **drain/inspect queued `loop_mailbox/`** —— adversarial：排队中的 job 可能携 old id，必须清空或确认无 renamed-task job，否则 mid-migration 命中旧 key）再做，作为一个维护窗口（g→regen→tree-sync 原子化）。

```bash
# forward 脚本（daemon 上跑；纯存储 mv，无 GPU）
python3 - <<'PY'
import json, os, subprocess
m = json.load(open('scripts/renumber/renumber_map.json'))
root = os.path.join(os.environ['TAB_GPFS_ROOT'], 'sealed_golden')
n = 0
for old, v in m.items():
    if old == v['new']:
        continue
    src, dst = os.path.join(root, old), os.path.join(root, v['new'])
    if os.path.isdir(src):                       # 仅对实际 sealed 的任务（17 个）生效
        assert not os.path.exists(dst), dst
        subprocess.run(['mv', src, dst], check=True)   # plain mv：gpfs 非 git 树
        n += 1; print("sealed", old, "->", v['new'])
print("sealed dirs renamed:", n)   # 期望 17
PY
```

涉及 17 个 sealed renamed（审计列：`L1_13→L2_01, L1_14→L2_02, L1_15→L2_03, L1_21→L2_08, L1_25→L1_14, L1_26→L1_15, L2_10→L2_17, L2_17→L2_23, L2_19→L3_07, L2_33→L3_11, L2_34→L2_29, L2_37→L2_30, L3_01→L3_13, L3_02→L3_14, L3_03→L3_15, L3_04→L3_16, L3_09→L3_19`；`L2_25_fwd_kernel_ep_gather` sealed 但不变）。脚本直接扫 `isdir(src)`，自动只命中实际 sealed 的，**无需手维护 17 清单**。

**可逆**：保留 inverse 脚本（`for new→old: mv`）。**验证**：rename 后对每个迁移任务读 `meta.yaml provenance.golden.golden_sha256`，并对 `sealed_golden/<new>/sealed_*.safetensors` 重算 sha256 比对——必须相等（content 与 task_id 无关，§2.2）。

### Step (h) — 再生 `resolve_all.json`（root + scratch）+ gpfs 树 `--delete` 同步

```bash
# (h1) 再生 resolve 缓存（静态 AST，无 GPU；可在 dev 或 daemon 跑后随码同步）
rm -f resolve_all.json results/resolve_all.json
python3 scripts/resolve_upstream_entrypoints.py --refresh-cache   # 从 new dir 名重新派生 verdicts[].task_id, 写 root + results/ 缓存(codex-r3: results/resolve_all.json 现有 105 old-id hits, 必须再生)
# (h2) gpfs 树同步 + 清掉残留 old dir（防 daemon 跑 stale 重复）
#   codex-r4 caveat: orchestrator 无 --delete flag(:601 parser 无该选项, cmd_sync_code :153 不传 delete=True)
#   —— 必须手动两步: 先在 daemon 删 87 个旧 dir, 再 rsync 新树
ssh daemon 'for o in $(cat /path/renumber_old_ids.txt); do rm -rf $TAB_GPFS_ROOT/.../tasks/level*/$o; done'
python3 scripts/loop/orchestrator.py sync-code   # rsync 新树(update 模式); 旧 dir 已由上面 rm 清掉
```

依据：`kernel_resolver.py:289-291` 运行时按 `v.get('task_id')==task_id` 匹配 resolve 缓存——stale old id 会让 renamed 任务 cache-miss、静默退化到 fallback 启发式（`kernel_resolver.py:6` docstring）。再生是唯一干净修法（content 与 task_id 无关，re-key 也行但 regen 更正）。`orchestrator.py:127-154` 确认 `cmd_sync_code` 用 `update=True` 且**不传 `--delete`**（`_rsync` 仅在 `delete=True` 时 append `--delete`，dev 实测 `:109`），故必须显式删，否则触发 MEMORY「sync staleness」陷阱。

### Step (i) — 公开文档 regen

- `docs/kernel-select.md`：手改 `:5` counts → `101 ... L1 36 · L2 35 · L3 30`；`§7` 用**只读** generator 从 `meta.yaml` 重排（按 new level 重分区、按 new idx 重排序、重写首列 task_id），**禁用 `phase_c_docs.py`**（会 clobber shapes.yaml）。`:9-10` 聚合 repo/VV-CV 计数不变，留。**（dry-run 实证：§7 首列是 `<task_id>_kernel`，old→new 若走文本替换**须 substring**(catch `_kernel` 后缀) —— token-boundary 会漏全部 84 行；从 meta 重生成 §7 则天然规避。dry-run 加这一步后全仓 dead-id guard = 0。）**
- `README.md`：`:140` 分布 → `(L1=36, L2=35, L3=30)`；prose id 由 Step (e) 已覆盖（README 在 e 的 docs 范围内则已改，否则在此手 remap：`L1_24→L2_11, L1_21→L2_08, L1_30→L1_18, L3_03→L3_15, L3_07→L2_32`）。`:29/36/48` 的 `L1_12_rms_norm_kernel` 示例与 Tasks-101 badge 不动。
- `SCORE.md`：`L2_33→L3_11_fwd_grouped_kernel_stage1_rope`；**消歧** `L2_23/L2_24` 显式标注为 `tasks/_deprecated`/`tasks/_corrected` 的 `fwd_kv_*`（避免与 new 活跃 `L2_23/L2_24` 撞名）；`:46-47` 指向 `_corrected/_deprecated` 的结果路径不动。

> README/SCORE 的简单 prose id remap 可直接纳入 Step (e) 的 roots（把 `README.md`、`SCORE.md` 加进 `extra`），但 **counts 行与 SCORE 消歧手工做**；kernel-select §7 结构性重排必须独立 generator。

---

## 3-bis. adversarial 收口（critical/major 补全）

> Step (e) 明确「不含 tasks/」以避开 meta/contract 双改，但这漏了 task **内容文件** 的 dead id（adversarial critical）。以下 3 步补齐，且处理 SHA-pin / pyproject / solutions / short-id。

### Step (j) — content 文件 dead id 改写 + kernel.py SHA 重算（critical）

dev 实测：`kernel.py:1` header `"""Task <old_id>_kernel — frozen upstream...`、`oracle_pytorch.py:1` `"""Oracle reference for <old_id>_kernel.`、部分 `_pa_launcher.py` 跨任务注释（如 `L3_07` launcher 引用 `L2_05/L3_14/L3_20`），都含 old id。且 **kernel.py 是 SHA-pinned**（`meta.phase_a_prime.standalone_sha256 == kernel_sha256 == sha256(kernel.py)`，dev 实测全等），改 header 必须同步重算这两个 sha，否则 S2 fail-closed。

> **不违反 frozen**：header `Task <id>` 是 freeze 时 bench 加的标注行（`Frozen by: phase_a_freeze.py`），改它不动任何 upstream 逻辑字节；`upstream.file_sha256`（vendored snapshot hash，独立字段）**保持不变**。

```python
import json, re, hashlib, pathlib
m = json.load(open('scripts/renumber/renumber_map.json'))
LD = {'L1':'level1','L2':'level2','L3':'level3'}
ch = {k: v['new'] for k, v in m.items() if k != v['new']}
# 全局 87-id 词边界替换（longest-first），覆盖 当前 task header + 跨任务注释
# codex-r3: substring(无词边界) —— header 是 "Task <old>_kernel"，词边界 lookahead 被 _kernel 的 `_` 挡掉会漏 112 个 header；old full-id 互不为子串(codex 验证 (a))故 substring 替换安全
pat = re.compile('|'.join(map(re.escape, sorted(ch, key=len, reverse=True))))
CONTENT = ('kernel.py', 'oracle_pytorch.py', '_pa_launcher.py')
nsha = 0
for old, v in m.items():
    if old == v['new']: continue
    d = pathlib.Path(f"tasks/{LD[v['new_level']]}/{v['new']}")
    for fn in CONTENT:
        fp = d / fn
        if fp.exists():
            t = fp.read_text(); nt = pat.sub(lambda mm: ch[mm.group(1)], t)
            if nt != t: fp.write_text(nt)
    # 重算 kernel.py sha, 行内锚定更新 meta（不 round-trip）
    kp = d / 'kernel.py'
    if kp.exists():
        new_sha = hashlib.sha256(kp.read_bytes()).hexdigest()
        mp = d / 'meta.yaml'; s = mp.read_text()
        s = re.sub(r'(?m)^(\s*standalone_sha256:\s*).*$', lambda mm: mm.group(1)+new_sha, s)
        s = re.sub(r'(?m)^(\s*kernel_sha256:\s*).*$',     lambda mm: mm.group(1)+new_sha, s)
        mp.write_text(s); nsha += 1
print("content rewritten + sha recomputed:", nsha)
```
注意先跑 (b)/(c)（dir 已 mv 到 new、meta 已是 new id）再跑 (j)。验证：`git grep -nIE '<87 old-id>' tasks/` = 0；逐题 `sha256(kernel.py) == meta.standalone_sha256 == meta.kernel_sha256`。全量回归的 S2 会用 new sha 重新 verify（pass）。

### Step (k) — pyproject / _frozen_shim / solutions（major）

```bash
# (k1) 顶层公开包文件 + frozen_shim: 加进 Step(e) 同款全局替换 roots
#   pyproject.toml:80-81 注释含 L1_19/L1_24 等; tasks/_frozen_shim/__init__.py:581.. 含 6+ old id
python3 - <<'PY'
import json, re, pathlib
ch = json.load(open('scripts/renumber/renumber_changed.json'))
pat = re.compile(r'(?<![A-Za-z0-9_])(' + '|'.join(map(re.escape, sorted(ch, key=len, reverse=True))) + r')(?![A-Za-z0-9_])')
import glob
# codex#2: _frozen_shim 全 .py(__init__/modular_kernel/envs), 不只 __init__
files = ['pyproject.toml', 'README.md', 'SCORE.md'] + sorted(glob.glob('tasks/_frozen_shim/*.py'))
for f in files:
    p = pathlib.Path(f); t = p.read_text(); nt = pat.sub(lambda mm: ch[mm.group(1)], t)
    if nt != t: p.write_text(nt); print("rewrote", f)
PY
# (k2) solutions/ 若存在 old-id 目录则 git mv（CI --solutions-root baseline 随迁）
python3 - <<'PY'
import json, pathlib, subprocess, os
m = json.load(open('scripts/renumber/renumber_map.json'))
sol = pathlib.Path('solutions')
if sol.is_dir():
    for old, v in m.items():
        if old == v['new']: continue
        od = sol / old
        if od.is_dir():
            nd = sol / v['new']
            assert not nd.exists(), nd
            # solutions/ 可能 untracked: 用 plain mv（git mv 对 untracked 报错）
            (subprocess.run(['git','mv',str(od),str(nd)]) if subprocess.run(['git','ls-files','--error-unmatch',str(od)],capture_output=True).returncode==0 else os.rename(od,nd))
            print("solutions", old, "->", v['new'])
else:
    print("no solutions/ dir (skip)")
PY
```
注：CI `ci.yml:182 --solutions-root /workspace/solutions` 是 CI runner 的外部 baseline，与本仓 `solutions/` 不同源 —— CI 侧 baseline 的 id 对齐需在 CI 维护窗口单独做（§7 codex gate 列为显式确认项）。

### Step (l) — bare short-id 残留（minor，定点）

Step (e) 只替换**全 id**。审计点名 `scripts/phase_a_cost_xcheck.py:60,768` 有 bare 短 id `'L1_21'`（其全 id `L1_21_gumbel_sample→L2_08`，但 bare `L1_21` 现在是**别的题**的新前缀，语义已变，必须定点改为 `L2_08`）。bare short-id **不能**全局正则（会撞新前缀），逐处人工核对 + codex 复核：
```bash
git grep -nE "(?<![A-Za-z0-9_])L[123]_[0-9]{1,2}(?![0-9A-Za-z_])" scripts/ tests/ triton_ascend_bench/ | grep -v _vendored
# 对每个 bare 命中: 查它的 OLD 全 id -> map -> 改成 new 前缀; 若该 bare 已是某 new 前缀则跳过
```

### Step (m) — active meta 残留 old id 清理 + `_vendored` 文件 rename（codex 终审补，critical）

codex 抓出 (c)/(j) 仍漏的 **active meta 字段**：① `phase_a_prime.s2_gpu_run_id`（格式 `s2-<old_id>-<ts>`，87 题全含）；② `upstream.vendored_snapshot`（`tasks/_vendored/<old>_kernel_<commit>.py`，84 题含 old 前缀）+ `tasks/_vendored/` 99 个 snapshot 文件名本身含 old id。为达「0 dead id active surface」，**rename `_vendored` 文件并同步 meta 路径**（比豁免更干净，公开发布无遗留）。

```python
import json, pathlib, subprocess
m = json.load(open('scripts/renumber/renumber_map.json'))
LD = {'L1':'level1','L2':'level2','L3':'level3'}
vd = pathlib.Path('tasks/_vendored')
# m1: git mv _vendored snapshot 文件 <old>_kernel_<commit>.py -> <new>_kernel_<commit>.py
for old, v in m.items():
    if old == v['new']: continue
    for f in vd.glob(f'{old}_*.py'):   # codex-r2: 不限 _kernel —— 真文件含 <old>_<op>_<commit>.py(如 L1_28_pack_bitmatrix_*.py)，{old}_kernel_* 只覆盖 59/84
        subprocess.run(['git','mv', str(f), str(vd / f.name.replace(old, v['new'], 1))], check=True)
# m2: meta 内 s2_gpu_run_id 的 <old> + vendored_snapshot 的 <old> 前缀 -> new
#     task_id 行已由 Step c 改成 new，故全文 replace 只命中这两处残留 old id
import yaml
for old, v in m.items():
    if old == v['new']: continue
    mp = pathlib.Path(f"tasks/{LD[v['new_level']]}/{v['new']}/meta.yaml")
    s = mp.read_text(); ns = s.replace(old, v['new'])
    if ns != s: mp.write_text(ns)
    # m3(codex-r2): assert rewritten vendored_snapshot 路径真实存在，防 25 个 dangling
    vs = (yaml.safe_load(ns).get('upstream') or {}).get('vendored_snapshot')
    if vs: assert pathlib.Path(vs).exists(), f"dangling vendored_snapshot: {vs}"
print("meta s2/vendored cleaned + _vendored renamed + all vendored_snapshot paths verified")
```
**`source_md_id` 政策（codex 必修#5）**：`source_md_id`（如 `L1-13`，hyphen 格式）是**历史 selection 编号**，指向已冻结为历史记录的 `benchmark_select.md`，provenance-only / 代码不消费 —— **明确保留不迁移**（它不是 task_id；迁移反而割裂与历史 selection doc 的对应关系）。rename 后 §5 guard 不再 SKIP `_vendored/`（验证其文件名也 0 old id）。

---

## 4. 与 23 个 pending unit fix 的关系

用户决策：**先做 renumber 迁移，23 个 pending unit fix 折叠进最终回归，不作前置依赖**。落地方式：

- 23 个 fix 全部在**同一 migration 分支**上，于 **Step (d)/(e)/(f) 之后、§5 pytest 之前**应用，与改名后的目录/引用一次性收敛，最终回归一次跑全。
- **task_id-bearing 的重叠面 = 审计列出的 ~21 个 breaking 测试文件**（§3 Step e 清单）。凡 23 个 fix 中触及这些文件者，必须在改名脚本跑完后的**新 id 语境**下改（否则会与刚替换的 new id 冲突，或重新引入 dead old id）。重点点名：`test_kernel_resolver.py`、`test_wrapper_contract.py`、`test_r10a_wrapper_contract_classification.py`、`test_dataset.py`、`test_golden_store.py`、`test_phase_b_golden_capture.py`、`test_resolve_upstream_entrypoints.py`、`test_cli_w1_4.py`、`test_phase_a_prime_s2_soundness.py`——这些既是改名 blast-radius 又是高频 pending-fix 区。
- 顺序保证：先跑 §3 全部脚本 → 再 apply 23 fix → 跑 §5 dead-id 守卫（确认 fix 没引回 old id）→ 跑 pytest。任何 fix 若新增 task_id 字面，必须用 **new** id。

> 我没有 23 个 fix 的逐条清单。执行时第一步先 `git log`/backlog 取出 23 条，逐条标注是否落在上述 21 文件；落在的与改名同分支同 commit-group 处理，不落的独立小 commit。这点请在执行启动时与用户对齐清单。

---

## 5. 验证矩阵（每步后 + 总闸）

| 阶段 | 命令 | 通过判据 |
|---|---|---|
| (b) 后 git sanity | `git diff --cached --name-status -M \| grep -c '^R'`；`git log --follow --oneline -- tasks/level1/L1_13_fused_softcap/kernel.py \| head` | 重命名条目 ≈ 87 dir 的文件数，全 `R`；history 可 follow |
| (c) 后 meta 分布 | 下方 `meta_dist.py` | level 计数 `L1=36/L2=35/L3=30`；每层 idx 严格 `1..N` 连续；dir==task_id==levelN 三向一致 |
| (e/f) 后 0 dead id | 下方 `dead_id_guard.py` | **0 命中**（活跃面）；非 0 直接 `sys.exit(1)` |
| 总闸 1 | `python3 scripts/validate_tasks.py` | 0 error（证 dir/meta.task_id/contract short_id 对齐） |
| 总闸 2 | `python3 scripts/validate_corpus_consistency.py` | 0 violation（证每个 CEILING_ACCEPTED 的 new id 在迁移后 ceiling allowlist） |
| 总闸 3 | `python3 scripts/_test_ceiling_acceptance.py`；`pytest tests/unit_tests -q` | 全过；**0 failed 且 0 skipped**（disk-glob 测试真跑，不静默 skip） |
| 总闸 4 | run-suite `--level` 池计数 | L1/L2/L3 池 = **36/35/30** |
| GPU 总闸 | daemon 真机对 renamed 101 跑 C1/C2 oracle-equiv + S2/phase_a_prime + cost_model xcheck + golden verify + release-gate + gpu-rank | 全绿；**golden 在 new key 下解析成功**；无 ceiling/floor 评分回归（无 renamed ceiling 任务被打成 strict FAIL） |

```bash
# meta_dist.py — 分布 + idx 连续 + 三向一致
python3 - <<'PY'
import pathlib, yaml
from collections import defaultdict
LD = {'level1':'L1','level2':'L2','level3':'L3'}
seen = defaultdict(list)
bad = []
for p in pathlib.Path('tasks').glob('level*/*/meta.yaml'):
    d = yaml.safe_load(p.read_text())
    dirname = p.parent.name; phys = LD[p.parent.parent.name]
    if d['task_id'] != dirname: bad.append(f"task_id {d['task_id']} != dir {dirname}")
    if d['level'] != phys:      bad.append(f"{dirname}: meta.level {d['level']} != phys {phys}")
    seen[d['level']].append(int(d['idx']))
for lvl in ('L1','L2','L3'):
    nums = sorted(seen[lvl])
    print(lvl, "count", len(nums), "contiguous", nums == list(range(1, len(nums)+1)))
assert dict((k, len(v)) for k, v in seen.items()) == {'L1':36,'L2':35,'L3':30}, seen
assert not bad, bad
print("OK 36/35/30, idx contiguous, 3-way consistent")
PY
```

```bash
# dead_id_guard.py — 全 87 old-id 在活跃面 0 命中；CI/pre-commit 闸
python3 - <<'PY'
import json, re, pathlib, sys
ch = json.load(open('scripts/renumber/renumber_changed.json'))
# substring(无词边界, codex#3): catch token + 嵌入式 <old>_kernel(vendored_snapshot 路径/文件名).
# old full-id 互不为子串(op/编号不同), 故 substring 无误报。
pat = re.compile('|'.join(map(re.escape, sorted(ch, key=len, reverse=True))))
SKIP = ('.git/', '.humanize/', '.codex/', '.agents/', 'external/', 'infra/', 'specs/',
        'tasks/_deprecated/', 'tasks/_corrected/',
        'docs/reviews/', 'docs/triage_backlog/', 'docs/plans/', 'docs/brainstorms/',
        'docs/benchmark_select.md', 'docs/npu/', '__pycache__/', 'results/',
        'loop_mailbox/', 'scripts/renumber/')
# 排除：工具产物(.humanize 2954 hits/.codex/.agents)、deprecated/corrected 历史题、
# dated 文档、scratch(results/loop_mailbox)、映射自身。
# _vendored 不再 SKIP(Step m 已 rename 文件名)；tasks/ 活跃题 NOT skipped(Step j/m 已改 header+s2+vendored)
# —— guard 必须验证 tasks/(含 _vendored 文件名 + meta vendored_snapshot/s2_gpu_run_id) 也 0 dead id。
hits = []
for p in pathlib.Path('.').rglob('*'):
    if not p.is_file() or p.suffix not in {'.py','.sh','.md','.js','.yaml','.yml','.json','.toml'}:
        continue
    s = str(p)
    if any(k in s for k in SKIP) and not s.endswith('resolve_all.json'):
        continue   # codex-r3: results/ scratch 跳过，但 root + results/resolve_all.json 是 code-consumed active 缓存(phase_a_prime:101)，必须验证其再生后 0 old id
    if pat.search(s):   # codex-r2: 扫文件路径名本身 —— catch stale 改漏的 <old>_*.py 文件名(line 内容扫不到)
        hits.append(f"{p}: <path-name carries old id>")
    for i, line in enumerate(p.read_text(errors='ignore').splitlines(), 1):
        if pat.search(line):
            hits.append(f"{p}:{i}: {line.strip()[:110]}")
print(len(hits), "dead old-id hits")
print('\n'.join(hits[:60]))
sys.exit(1 if hits else 0)
PY
```

> dead-id 守卫的排除集 = 历史 dated 文档（`docs/reviews|triage_backlog|plans|brainstorms`、`benchmark_select.md`、`docs/npu/`，按设计引用 old id）+ 工具产物（`.humanize/.codex/.agents`）+ `_deprecated/_corrected`（历史题）+ `results/loop_mailbox/`（scratch）+ `scripts/renumber/*.json`（映射自身）。**`_vendored/` 不再排除**（Step m 已 rename 其文件名，守卫须验证它也 0 old id）。守卫同时扫**文件路径名 + 行内容**（codex-r2），做成 release gate / pre-commit hook。

---

## 6. 风险与回滚

| 项 | 可逆性 | 失败模式 / 缓解 |
|---|---|---|
| Step (b)-(f)（git mv + meta/contract/tests/scripts/docs/yaml 改写） | **完全可逆**：commit 前 `git reset --hard` / `git restore .`；migration 分支隔离 | 脚本 fail-closed（断言旧值/目标不存在），半迁移会崩而非静默 |
| Step (g) **gpfs sealed_golden key rename**（daemon 侧、非 git） | **唯一外部/不可 git 回滚项**——必须脚本化 + 备 inverse 脚本 + 先验证 | ① 先 **quiesce daemon**（无 in-flight job）再 mv，g→h 原子维护窗口；② mv 后**逐任务比对 golden_sha256 == 重算 sha**（§3g）才放行；③ inverse 脚本随时回滚（`for new→old: mv`） |
| daemon job 命中**未改名 key** | — | `golden_store.locate_sealed` 返回不存在路径 → golden 缺失 → C2/release-gate **fail-closed 报缺 golden**（不会静默冒充 PASS，符合 MEMORY「C2 是真 correctness gate」）。缓解：g/h 在维护窗口内完成、tree-sync `--delete` 清 stale dir、重启前用一次 golden-locate 干跑探针确认 new key 命中 |
| resolve_all stale | 可逆（再生/re-key） | cache-miss → 退化 fallback（非崩）；h1 再生修复 |
| gpfs 残留 old dir | 可逆（重 sync） | daemon 跑 stale 重复（MEMORY sync-staleness）；h2 `--delete` 强制清 |

回滚总策略：dev 侧一切走 git 分支，未 merge 前零成本回退；gpfs 侧（g/h）是唯一需要"显式可逆 + 先验证后依赖"的外部状态，inverse 脚本 + sha 校验 + 维护窗口三重兜底。

---

## 7. codex 监督 gate

按 CLAUDE.md「Mandatory codex review」「Codex 调用只用 humanize skill」「No premature commits」「Approval required for risky operations（git push / gpfs 改动）」：

1. **执行前**：本方案（连同 §3 全部脚本）先经 `humanize:ask-codex`（模型 `gpt-5.5:xhigh`）深度 review，重点审查：(a) git mv 零碰撞论证与幂等性；(b) meta/contract 行内锚定改写不破坏格式、short_id 校验不被绕过；(c) Step (e) 正则边界 + longest-first 无前缀互吃、不双改 tasks/；(d) golden key-rename-not-recapture 不变量是否真的 task_id-free；(e) gpfs g/h 维护窗口与 fail-closed 行为。按「Review external feedback critically」逐条评审采纳，迭代至 **0 FAIL** 才执行。
2. **实现后**：把 §3 全脚本运行的 **diff + dead-id 守卫输出 + `validate_tasks`/`validate_corpus_consistency`/pytest 摘要** 交 codex 二次 review（post-implementation），确认无遗漏 old id、无评分回归隐患。
3. **push gate**：dev 侧全绿 + codex 0 FAIL 后，**展示结果、等用户明确说「提交/push」** 才 commit + push（push 是 A1 risky operation，必须用户确认）。
4. **gpfs gate**：Step (g)/(h) 改 gpfs 是外部状态变更，执行前单独向用户确认维护窗口。
5. **push 后**：按 MEMORY「Auto sync after push」立即 `orchestrator.py sync-code --delete` 把 renumbered 树 rsync 到 gpfs，再触发 daemon GPU 全量回归（§5 GPU 总闸）。

---

### 关键文件清单（绝对路径）

- 映射源：`/tmp/renumber_map.json`、`/tmp/renumber_changed.json` →（Step a 冻结）`/data1/code/agentic-rl/kernel/triton-ascend-bench/scripts/renumber/`
- 注册表：`/data1/code/agentic-rl/kernel/triton-ascend-bench/scripts/phase_b_oracle.py`（`:42 SKIP_TASKS`、`:2654 WORKING_REFS`、`:2777 ORACLE_SCOPE_DOWNGRADE`）、`scripts/phase_c_docs.py`（`:50 SKIP_POC`、`:265 SHAPE_OVERRIDES`、`:945 _MANUAL`）、`scripts/phase_a_freeze.py:55`、`scripts/generate_task_scaffolds.py:46`
- code-consumed YAML：`triton_ascend_bench/ceiling_acceptance.yaml`、`triton_ascend_bench/floor_dominated_acceptance.yaml`
- golden/resolve：`triton_ascend_bench/golden_store.py:131`、`triton_ascend_bench/kernel_resolver.py:289`、`resolve_all.json`、`results/resolve_all.json`、`scripts/resolve_upstream_entrypoints.py`、`scripts/loop/orchestrator.py:127`（sync-code）
- 校验闸：`scripts/validate_tasks.py:562-568`、`scripts/validate_corpus_consistency.py:103`、`scripts/_test_ceiling_acceptance.py`
- 文档：`docs/kernel-select.md:5,79+`、`README.md:140,142,157,158`、`SCORE.md:19-52`