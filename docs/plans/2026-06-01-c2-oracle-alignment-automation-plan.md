# C2(PyTorch oracle 精度对齐)全量自动化实施计划

date: 2026-06-01
status: design codex-supervised (gpt-5.5:xhigh),进入实现阶段
core-principle: **所有精度对齐必须 GPU 实跑得到,不是理论推导/自嗨;tolerance 锁死行业标准绝不为刷率放大;不改 vendored;上游 bug 诚实 BLOCKED 不掩盖;denominator 锁死 100 不剔除**

## 0. 已确认的真实状态(实测,非推断)

- C1(severed-vs-original S2 bit-exact)= 另一 session 成熟,batch bfull100g(6-01)98 PASS / 2 BLOCKED
- C2 当前:meta.yaml golden VERIFIED 仅 10;R10.a census(6-01 dev)= 40 ALIGNED_READY / 40 SIGNATURE_MISMATCH / 19 NEEDS_BRIDGE / 2 WRAPPER_CONTRACT
- 机器:dev(无 GPU/gpfs)→ ssh 反向隧道 → 中间机 B(gpfs,无 GPU)→ GPU A 机(daemon_a,只能投 mailbox 触达)

## 1. 载体决策(codex confirmed)

**C2 载体 = R10 pipeline**(`oracle_pyref_precompute` R10.a → `oracle_equivalence_gpu_runner` R10.b → `apply_oracle_equivalence_verdict` R10.c),**不是 resolve**。
- resolve 只写 `oracle.gpu_triton` entrypoint(CPU interpret anchor),产 **0** max_rel / 0 golden
- R10.b 才在真 GPU 上跑 kernel vs oracle ref,产 per-output max_abs/max_rel(`oracle_equivalence_gpu_runner.py:587`,正确用 `compare(...).passed` 非 truthiness)
- R10.c 是 `provenance.golden` 的唯一写者

## 2. codex 监督发现的 3 个 R10-vs-S2 harness 不一致(全量前必修,否则假 FAILED 污染精度数据)

| # | 不一致 | 真后果 | 修复 |
|---|--------|--------|------|
| H1 | R10 resolver 只认 kernel.py AST 非 JIT def,**不支持 `_pa_launcher.py::run` fallback**(`kernel_resolver.py:250`);C1/S2 loader 有(`_pa_subproc_runner.py:591`) | placeholder+launcher 任务(L1_15/L2_24 C1 PASS)在 R10 选错/block | R10 resolver 加 launcher::run fallback |
| H2 | R10.b 全程 `build_inputs(seed=GOLDEN_SEED)` 重建输入(`:311,:458`),**不 load** R10.a 存的 `inputs_*.safetensors`(只 load oracle ref `:425`) | 跨机(dev-CPU ref vs GPU kernel)/torch 版本/struct 输入 drift → **正确 kernel 假 FAILED** | 最稳:R10.a+R10.b 同机(daemon 内串联);或 R10.b load saved inputs + hash 校验,不一致 BLOCKED 不比较 |
| H3 | R10.b **不读** `meta.interface.output_projection/inplace_outputs/return_plus_inplace/param_name_map`,但 oracle anchor 读(`_pa_oracle_check_runner.py:416`) | structure mismatch 假阳;"补 meta.interface 修 R10"当前**无效** | R10.b 读 meta.interface 做 arity/projection 对齐 |

## 3. daemon/orchestrator 改动(加全自动 C2 通道)

- daemon_a.py:403 白名单:加 `oracle-equiv`,且改 **fail-closed**(未知 action 报错,不静默降级 verify)
- daemon_a.py `_run_verify_one`:加 `elif action=='oracle-equiv'` 分支。**最稳设计:在 GPU 机内先 R10.a precompute 再 R10.b**(消除 H2 跨机 drift),输出 `<results>/<batch>/r10/<task>.json`
- daemon_a.py verdict-parse:加分支读 `r10/<task>.json` 的 TaskVerdict(overall + per-shape max_abs/max_rel),R10 写结构化 JSON 不是 'verify <task>:' 行
- orchestrator.py:加 `cmd_run_oracle`(照抄 run-resolve 模板,action='oracle-equiv',**不带 pull-meta**)+ 注册 subparser
- 注意:daemon self-update 按 mtime re-exec 但 **argv 不变** → `--per-task-timeout` 不会自动更新,GPU kernel build+多 shape 慢需 ≥1800s,要重启 daemon(GPU A 机操作,需协调)

## 4. 三态诚实分类(GPU 实跑结果驱动,无 fudging)

1. **ALIGNED**(唯一算 C2 通过):R10.b overall=VERIFIED on real CUDA,每输出 tensor 在 tolerance_class(行业标准,**绝不放大**)内 → 写 golden_source=oracle_pytorch_verified + promoted_to_truth + max_abs/max_rel 证据
2. **NEEDS-INVESTIGATION**(oracle-mismatch:值超 tolerance 但 structure/dtype 匹配):查 (a) GPU 复确认 (b) 手审 oracle vs 上游数学 (c) oracle 错→修 oracle 重跑→ALIGNED (d) kernel 真 diverge→class 3。**改 oracle 前先 codex 验 compare()/ToleranceClass API**(A6 memory:compare 返回 ComparisonResult,取 .passed)
3. **BLOCKED**(oracle 确认对但 kernel 仍 diverge=上游 bug,或无 faithful shapes 能 bind):诚实标 reason,不 promote,**留在分母**(baseline-no-subset-escape)

## 5. 执行顺序(每步 GPU 实跑验证,不自嗨)

1. ✅ STEP 1 done:R10.a census = 40 ALIGNED_READY(C2 第一批可跑分母)
2. 修 H1/H2/H3 harness 一致性(dev 代码)→ **codex review 到 0 FAIL**
3. 改 daemon+orchestrator oracle-equiv action(fail-closed)→ codex review cross-file protocol → 本地 synthetic result.json 验 parse
4. **小批 canary GPU 实跑**(placeholder+launcher / multi-output / inplace / 普通各代表)→ 实跑确认 harness 对、无假 FAILED、action 字段==oracle-equiv 未降级
5. 全量 oracle-equiv batch(fresh batch_id)GPU 实跑 → collect
6. R10.c apply 写回 meta golden(max_abs/max_rel 证据)
7. 攻 gap:(A) shapes.yaml faithful 补参数(SIGNATURE_MISMATCH 40)(C) meta.interface(structure)→ 仅重跑新 eligible 子集 → 每个 codex co-review
8. NEEDS-INVESTIGATION 手审(L2_08/L2_16 等)→ oracle-bug 修 / kernel-bug BLOCKED
9. 交付:三态 verdict 表(分母 100)+ max_abs/max_rel 证据 + BLOCKED reason

## 6. 现实可达(codex:60-80 偏乐观)

- GPU baseline(40 ALIGNED 跑 R10.b,无 gap 修):VERIFIED ~15-25(env-artifact 在 CUDA 上消失)
- 加 gap A(shapes)+ C(interface):~40-60(NEEDS_BRIDGE 19 是语义桥大桶,不只补 shapes)
- irreducible floor:~2 wrong-operator(L2_08/L2_16)+ optional-dep 缺失(flashinfer)task → 诚实 BLOCKED 留分母
- **不承诺 100/100**;交付 = 自动化 + 实跑 + 三态表 + 证据

## 7. GPU A 机协调点(我够不到,需 D session/用户)

- daemon 重启 with `--per-task-timeout>=1800`(self-update 不更新 argv)
- 全量 GPU 实跑(改完 daemon code,self-update 加载新代码后,通过 orchestrator run-oracle 投 mailbox)

## 8. 红线(zero-tolerance)

- tolerance 锁死行业标准 per-class,**绝不 task-by-task 放大刷率**
- 不改 vendored vllm/sglang
- faithful 输入分布,补 shapes 参数不用 placeholder(否则 oracle 比对无意义=假 VERIFIED)
- compare() 取 .passed 不用 truthiness(否则全 false-positive)
- dev verdict 非权威(无 CUDA/interpret),只认 GPU overall=VERIFIED

## 9. 进度 + codex 监督结论(2026-06-01)

### 已收敛(codex PASS,可 commit)
- **H2 input-faithfulness guard**(6 轮 codex 迭代):R10.b 用 split_inputs fail-closed 校验完整 payload(tensor 字节 + 非 tensor type-strict);drift → INPUT_DRIFT(绝不假 FAILED)。配套:R10.a 存 sidecar、provenance enum、apply golden_source、bootstrap 缺 sidecar 重跑、contract.md。实跑验证:正常不误伤 + drift 检测。

### daemon oracle-equiv action 第一版 = codex FAIL 5 条(WIP,勿 ship)
1. [confirmed 高危] `bash -c` 串联破坏 timeout:subprocess.run timeout 只杀 bash,不杀 R10.a/R10.b Python 孙进程 → 旧进程占 GPU/晚写 JSON 污染后续。修:改 python wrapper 同进程 import 调用(无孙进程),或 process-group + killpg。
2. [confirmed 高危] `R10.a && R10.b` 挡不住业务失败:oracle_pyref_precompute 即使 SIGNATURE_MISMATCH/RUNTIME_ERROR 也 return 0 且不清旧 r10_pyref → && 仍跑 R10.b 用旧/半新 artifact 出假 verdict。修:wrapper 检查 R10.a audit.status==ALIGNED_READY 才跑 R10.b + 清旧 ref。
3. [confirmed 高危] R10.b 仍不等价 C1 harness(= 未做的 H1/H3):kernel_resolver 无 `_pa_launcher.py::run` fallback;R10.b 不读 meta.interface(output_projection/inplace_outputs/return_plus_inplace)→ launcher/multi-output/inplace 任务假 LAUNCH_BLOCKED/FAILED。
4. [confirmed 中危] R10.c provenance hash 不闭环:apply 的 golden_sha256 来自 dev 本地 r10_pyref,非 daemon GPU 端同机 R10.a 生成的 ref。修:collect 拉回 GPU 端 r10_pyref,或 verdict JSON 记 GPU 端 ref hash 供 apply 用。
5. [likely 运维] `--per-task-timeout=1200` 对 R10.a+R10.b 串联偏小且 self-update 不更新 argv → 跑前需重启 daemon 调大(GPU A 机操作)。

### 推荐修复落地顺序(下一步)
- 写 `scripts/loop/oracle_equiv_one.py`:同进程 import R10.a audit(status gate + 清旧 ref)+ R10.b verify_task,daemon 调它单 subprocess(解 1+2)
- R10.b 加 launcher::run fallback + meta.interface 读取(解 3 = H1+H3)
- collect/apply provenance 闭环(解 4)
- daemon 重启 --per-task-timeout≥1800(解 5,GPU A 机协调)
- 每项 codex review → 小批 canary GPU 实跑 → 全量

### 续 (autonomous, codex-supervised)
- **H1 (codex FAIL #3 part 1) DONE + codex PASS on convention**: R10.b `_verify_one_task`
  加 launcher fallback: `is_launcher` 从 meta.oracle.gpu_triton 判定(`_pa_launcher.py::run`
  或 `::run`+存在 `_pa_launcher.py`,fail-closed,覆盖 L1_27 这类 meta 写 kernel.py::run 但
  真 run 在 launcher 的情况),launcher 走 dict-in 调用 `kernel_fn(inputs_gpu)`。codex 确认
  调用约定与 C1 `_pa_oracle_check_runner._check_one_shape` 的 `fn(dict(kernel_inputs))` 一致。
- **H3 fail-closed guard DONE**(codex 揪出的假 VERIFIED 风险):新增 `_mutated_input_keys`
  (从 meta.interface.mutability + oracle `inputs[X].copy_(` 推断 candidate_inplace 的 mutated
  key);overall=VERIFIED 后,若 mutated key 不在 ref_keys_seen → 降级 `INPLACE_UNCOMPARED`
  (honest BLOCK,不假 VERIFIED)。验证:L2_41(state)/L2_22(kv_caches) 触发拦截;L2_40(返回
  q,k,v)/L2_20(返回 conv_state) 不误伤。
- **下一步**:完整 H3(R10.a 存 mutated input + R10.b 合并 mutated input 进 out_dict 比对,
  让 in-place 题能真 VERIFIED)→ #4 provenance 闭环 → canary GPU → 全量 batch。

### 续2 (codex FAIL #3 完整收敛, autonomous)
- **完整 H3 DONE**: R10.a run-前 always-clone 快照(修潜在 H2 timing bug)+ 存 mutated 输出(skip 已返回的 key);R10.b 合并 mutated 输入(override candidate 侧)。in-place 题(L2_41 state / L2_22 kv_caches / L2_20 conv_state / L2_40 q,k,v)现可真比对。
- **codex REFINE 全修**: (b) R10.a override→skip(L2_40/L2_20/L1_25 oracle 返回不 mutate,ref 保留返回值;verified ref[q]≠inputs[q]);(c) `_mutated_input_keys` 解析 return_plus_inplace dict/inplace_outputs list;(d) R10.a audit 加 launcher bypass(L2_39/L2_40 SIGNATURE_MISMATCH→ALIGNED_READY,~30 launcher 任务可审)。
- **注释**: 改简洁工业标准,去过程噪音(旧的 V9/round-2 噪音注释待整体 commit 前清)。
- **pytest**: test_r10_multi_output_order 的 2 failures = L1_36 oracle 重构 stale test,改前就存在(git-stash 确认),与本改动无关。
- **就绪度**: R10.a/R10.b harness 对 launcher + in-place 都正确(无假 LAUNCH_BLOCKED/VERIFIED/DRIFT/SIGNATURE_MISMATCH)。下一步 = sync-code → canary GPU 小批实跑(daemon per-task-timeout 协调点未解,需 GPU-A 重启)→ 全量。
