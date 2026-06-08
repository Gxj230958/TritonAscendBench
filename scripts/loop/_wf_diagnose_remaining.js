export const meta = {
  name: 'diagnose-remaining',
  description: 'Classify every non-passing task by root cause; apply per-task meta fixes (output_projection/param_name_map/tolerance); flag severance-leak + structural',
  phases: [{ title: 'Diagnose', detail: 'one agent per task: classify root cause + apply per-task fix where a known lever applies' }],
}

const ROOT = '/data1/code/agentic-rl/kernel/triton-ascend-bench'
const RL = '/inspire/hdd/project/qianghuaxuexi/liuliming-liuliming/kernel/loop_mailbox'
const TASKS = Array.isArray(args) ? args : JSON.parse(args)

const SCHEMA = {
  type: 'object',
  additionalProperties: false,
  properties: {
    task_id: { type: 'string' },
    graded_entrypoint: { type: 'string' },
    blocker_category: { type: 'string', enum: [
      'A_severance_leak', 'B_output_projection', 'C_param_name_map', 'D_tolerance',
      'E_structural_layout_mismatch', 'E_bare_jit_placeholder', 'E_wrong_candidate',
      'E_config_dict', 'E_sigsegv_needs_gpu', 'F_shapes_multilayer', 'other',
    ] },
    fixable: { type: 'boolean' },
    fix_applied: { type: 'boolean' },
    fix_summary: { type: 'string' },
    severance_imports: { type: 'array', items: { type: 'string' } },
    structural_reason: { type: 'string' },
    confidence: { type: 'string', enum: ['high', 'medium', 'low'] },
  },
  required: ['task_id', 'blocker_category', 'fixable', 'fix_applied', 'confidence'],
}

function prompt(t) {
  return `Diagnose a non-passing triton-ascend-bench task and, IF a known per-task lever applies, FIX it. Goal: get it to a GPU bit-exact PASS or honestly classify why not. WORK IN ${ROOT}.

TASK: ${t.id} (${t.level}).

STEP 1 — Read LOCAL (use Read): tasks/${t.level}/${t.id}/{meta.yaml, oracle_pytorch.py, kernel.py, shapes.yaml}, and triton_ascend_bench/frozen/snapshots/${t.id}_*.py if present. Optionally fetch the task's most recent daemon blocker:
  ssh -p 2222 -o BatchMode=yes -o StrictHostKeyChecking=no root@127.0.0.1 'for b in $(ls -t ${RL}/results/ 2>/dev/null); do f=${RL}/results/$b/${t.id}.json; if [ -f "$f" ]; then grep -ao "reason:[^\\"]*" "$f" | head -1; break; fi; done'

STEP 2 — Identify the GRADED operator: meta.oracle.pytorch (= oracle_pytorch.py::run) defines the true semantics; meta.oracle.gpu_triton names the kernel entrypoint. Find the host-wrapper / kernel that computes the SAME operation as oracle_pytorch.py::run on the SAME logical inputs.

STEP 3 — Classify the ROOT blocker into EXACTLY ONE category:
  A_severance_leak: the intended entrypoint IS the graded operator and is oracle-consistent, but the severed standalone leaks a module-level vllm/sglang import used ONLY by SIBLING code (a setup/state class, type hint, output container), NOT the graded numeric path. -> set severance_imports = the exact leaking import line(s). DO NOT edit shared infra; just identify (the operator owner applies the shim).
  B_output_projection: the wrapper returns a SUPERSET tuple/list and only some indices are the oracle contract (e.g. returns (y, mean, rstd) but oracle wants y). -> APPLY meta.interface.output_projection: {arity: <len>, map: {<oracle_key>: <index>}}.
  C_param_name_map: the wrapper requires an input that IS an existing shapes.yaml tensor under a DIFFERENT name. -> APPLY meta.interface.param_name_map: {<wrapper_param>: <shapes_key>}.
  D_tolerance: the candidate IS provably the same operator as the oracle but oracle-mismatches by a ROUNDING-scale numeric diff (fp8/bf16/accumulation order). -> only if clearly a tolerance gap, set the correct tolerance_class in meta (conservative: do NOT relax tolerance to mask a real wrong-operator).
  E_structural_layout_mismatch: oracle and kernel need INCOMPATIBLE tensor ranks/layouts for the SAME shared input dict (e.g. oracle rank-3 [n,L,D] vs kernel rank-4 [B,T,H,K]); the bench feeds one input dict to both, so no shape reconciles them.
  E_bare_jit_placeholder: graded gpu_triton is a bare @triton.jit with ::placeholder oracle and NO faithful host-wrapper (cannot auto-invoke).
  E_wrong_candidate: NO entrypoint in kernel.py computes the oracle's operation (resolver only finds unrelated wrappers).
  E_config_dict: the entrypoint requires a non-tensor config object (dict/dataclass/constexpr tuning) the binder cannot synthesize.
  E_sigsegv_needs_gpu: prior run crashed rc=-11 (SIGSEGV) -- needs live GPU debug, not source-fixable here.
  F_shapes_multilayer: needs MULTIPLE interdependent missing input params (Mamba/MoE chunk state, cu_seqlens, etc.) -- deep per-kernel authoring.

STEP 4 — For B/C/D ONLY: APPLY the meta.yaml edit (own file, parallel-safe). Match the existing YAML style; add under a top-level \`interface:\` key (create it if absent) or set tolerance_class. Then validate: \`cd ${ROOT} && python3 -c "import yaml; yaml.safe_load(open('tasks/${t.level}/${t.id}/meta.yaml')); print('YAML OK')"\`. For A: identify only. For E/F: no edit.

Be rigorous + honest: do NOT fake a fixable label. A B/C/D fix must be strongly evidenced by the wrapper signature + oracle semantics + kernel indexing.

Return ONLY StructuredOutput: task_id, graded_entrypoint, blocker_category, fixable (true only for A/B/C/D with a concrete fix path), fix_applied (true only if you edited meta.yaml for B/C/D), fix_summary (what you applied or the precise lever), severance_imports (for A), structural_reason (for E/F), confidence.`
}

phase('Diagnose')
log(`Diagnosing + fixing-where-tractable ${TASKS.length} non-passing tasks`)

const results = await parallel(
  TASKS.map((t) => () => agent(prompt(t), { label: `diag:${t.id}`, phase: 'Diagnose', schema: SCHEMA }))
)

const clean = results.filter(Boolean)
const by = (c) => clean.filter((x) => x.blocker_category === c)
const applied = clean.filter((x) => x.fix_applied)
return {
  total: TASKS.length,
  fix_applied: applied.map((x) => ({ task: x.task_id, cat: x.blocker_category, summary: x.fix_summary, conf: x.confidence })),
  severance_leak: by('A_severance_leak').map((x) => ({ task: x.task_id, ep: x.graded_entrypoint, imports: x.severance_imports })),
  output_projection: by('B_output_projection').map((x) => x.task_id),
  param_name_map: by('C_param_name_map').map((x) => x.task_id),
  tolerance: by('D_tolerance').map((x) => x.task_id),
  structural: clean.filter((x) => x.blocker_category.startsWith('E_')).map((x) => ({ task: x.task_id, cat: x.blocker_category, why: x.structural_reason })),
  shapes_multilayer: by('F_shapes_multilayer').map((x) => x.task_id),
  fixable_count: clean.filter((x) => x.fixable).length,
}