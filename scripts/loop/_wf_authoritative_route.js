export const meta = {
  name: 'authoritative-route',
  description: 'Per-task authoritative classify (bench classify_task) + route to the right lever + apply the tractable per-task fix; adversarial-review applied fixes for upstream faithfulness',
  phases: [
    { title: 'Route', detail: 'classify_task + blocker -> apply per-task fix (launcher/shapes/param_name_map/output_projection/oracle re-author) or classify structural' },
    { title: 'Review', detail: 'adversarial check: applied fix faithful to frozen upstream, not fake' },
  ],
}

const ROOT = '/data1/code/agentic-rl/kernel/triton-ascend-bench'
const TASKS = Array.isArray(args) ? args : JSON.parse(args)

const ROUTE_SCHEMA = {
  type: 'object',
  additionalProperties: false,
  properties: {
    task_id: { type: 'string' },
    authoritative_type: { type: 'string', enum: ['jit_only', 'host_wrapper', 'mixed', 'unknown'] },
    lever: { type: 'string', enum: [
      'launcher', 'param_name_map', 'shapes_fix', 'output_projection', 'oracle_reauthor',
      'severance_identify', 'structural_uva_pointer', 'structural_config_dict',
      'structural_layout_mismatch', 'structural_bare_jit_incompatible', 'structural_other', 'already_passing',
    ] },
    fix_applied: { type: 'boolean' },
    fix_files: { type: 'array', items: { type: 'string' } },
    fix_summary: { type: 'string' },
    faithfulness_basis: { type: 'string' },
    severance_imports: { type: 'array', items: { type: 'string' } },
    structural_reason: { type: 'string' },
    confidence: { type: 'string', enum: ['high', 'medium', 'low'] },
  },
  required: ['task_id', 'authoritative_type', 'lever', 'fix_applied', 'confidence'],
}

const REVIEW_SCHEMA = {
  type: 'object',
  additionalProperties: false,
  properties: {
    task_id: { type: 'string' },
    faithful: { type: 'boolean' },
    issues: { type: 'string' },
    verdict: { type: 'string', enum: ['ship', 'needs-fix', 'non-tractable'] },
  },
  required: ['task_id', 'faithful', 'verdict'],
}

function routePrompt(t) {
  return `Authoritatively classify a non-passing triton-ascend-bench task and, where a per-task lever applies, APPLY the fix. GROUND TRUTH = the frozen upstream kernel.py (specs/task_api_contract.md: oracle_pytorch.py must align to upstream, it is NOT an independent truth source). WORK IN ${ROOT}.

TASK: ${t.id} (${t.level}).

STEP 1 -- AUTHORITATIVE TYPE. Run the bench's own classifier (do not guess):
  cd ${ROOT} && python3 -c "import importlib.util,sys; sys.path.insert(0,'.'); s=importlib.util.spec_from_file_location('pap','scripts/phase_a_prime.py'); p=importlib.util.module_from_spec(s); sys.modules['pap']=p; s.loader.exec_module(p); from pathlib import Path; td=Path('tasks/${t.level}/${t.id}'); c=p.classify_task(td.name,(td/'kernel.py').read_text()); print('diagnostics:', c.diagnostics)"
  -> "jit-only" diagnostic => no host-wrapper (jit_only). Else host_wrapper/mixed.

STEP 2 -- READ: tasks/${t.level}/${t.id}/{meta.yaml (oracle.gpu_triton/pytorch, interface, param_constraints, output_allocation), oracle_pytorch.py (the reference), kernel.py (the frozen upstream + its host call site / how the @triton.jit is launched), shapes.yaml}, ascendbench/frozen/snapshots/${t.id}_*.py. Optionally the task's last daemon blocker via ssh -p "$LOOP_SSH_PORT" -o BatchMode=yes -o StrictHostKeyChecking=no "$LOOP_SSH_HOST".

STEP 3 -- ROUTE to ONE lever (and APPLY for the per-task-tractable ones):
  * launcher (jit_only with DENSE oracle-compatible inputs): author tasks/${t.level}/${t.id}/_pa_launcher.py defining run(inputs)->dict that derives grid/constexpr/strides from the upstream call site and drives the in-module @triton.jit; return the oracle's exact output dict key(s). Match the upstream invocation EXACTLY. If the oracle/shapes are also wrong vs upstream, fix them too (oracle_reauthor below).
  * param_name_map: a host-wrapper needs an input that IS an existing shapes tensor under a different name -> add meta.interface.param_name_map {wrapper_param: shapes_key}.
  * shapes_fix: a required input is missing or wrong-rank/dtype vs how the entrypoint indexes it -> add/fix it in shapes.yaml (dims as size_knobs, dtype, dist).
  * output_projection: the wrapper returns a SUPERSET tuple, only some indices are the oracle contract -> meta.interface.output_projection {arity, map}.
  * oracle_reauthor: oracle_pytorch.py computes a DIFFERENT operation/layout/index convention than the frozen kernel (the oracle is wrong). Re-author it as an INDEPENDENT pure-PyTorch reference that EXACTLY matches the upstream kernel's math/contract (NEVER call the kernel; never trivially echo inputs). Fix shapes.yaml/meta dtypes+invariants to the upstream contract too. (This is the L1_27 pattern.)
  * severance_identify (DO NOT edit shared infra): the entrypoint is oracle-consistent but the severed standalone leaks a module-level vllm/sglang import used only by SIBLING code -> set severance_imports = the exact import line(s); the operator owner applies the shim serially.
  * structural_uva_pointer: kernel needs raw data_ptr()/UVA pointer-array inputs the dense oracle cannot supply -> no fix.
  * structural_config_dict: entrypoint needs a non-tensor config object -> no fix.
  * structural_layout_mismatch: oracle and kernel need INCOMPATIBLE ranks/layouts from the single shared input dict AND no oracle re-author reconciles them (e.g. chunk-state granularity differs) -> no fix.
  * structural_bare_jit_incompatible / structural_other: explain.

STEP 4 -- APPLY (per-task files only: _pa_launcher.py / shapes.yaml / meta.yaml / oracle_pytorch.py -- all parallel-safe). After editing, validate: python3 -c "import yaml,ast; yaml.safe_load(open('tasks/${t.level}/${t.id}/meta.yaml')); yaml.safe_load(open('tasks/${t.level}/${t.id}/shapes.yaml')); ast.parse(open('tasks/${t.level}/${t.id}/oracle_pytorch.py').read()); print('OK')". For severance_identify and structural_* do NOT edit.

ANTI-FAKE (critical): every applied fix must be FAITHFUL to the frozen upstream kernel. An oracle re-author must be independent PyTorch that reproduces the kernel's true math; a launcher must mirror the upstream call site exactly. Never relax a tolerance or echo the kernel to force a match. If you cannot make it faithful, classify structural.

Return ONLY StructuredOutput: task_id, authoritative_type, lever, fix_applied (did you edit per-task files), fix_files, fix_summary, faithfulness_basis (the upstream evidence your fix matches), severance_imports, structural_reason, confidence (high only if the fix is strongly evidenced by the upstream).`
}

function reviewPrompt(t, r) {
  return `Adversarially verify an APPLIED fix for triton-ascend-bench task ${t.id} in ${ROOT}. GROUND TRUTH = the frozen upstream kernel.py.

A prior agent applied lever "${r.lever}" (files: ${(r.fix_files || []).join(', ')}). Its faithfulness basis: ${r.faithfulness_basis || '(none)'}. Summary: ${r.fix_summary || '(none)'}.

Read tasks/${t.level}/${t.id}/{kernel.py, oracle_pytorch.py, shapes.yaml, meta.yaml, _pa_launcher.py if present} and ascendbench/frozen/snapshots/${t.id}_*.py. Try to BREAK the fix:
- oracle_reauthor: is the new oracle INDEPENDENT pure-PyTorch (does NOT import/call the kernel) AND does it reproduce the kernel's exact math/index/layout convention? Any case (rank, dtype, indexing direction, masking convention, accumulation) where it diverges from the frozen kernel? Does it secretly echo inputs?
- launcher: does run() mirror the upstream call site exactly (grid, constexpr, strides, arg order, in-place vs out, output key)? Any wrong constant?
- param_name_map/shapes_fix/output_projection: do the dims/dtypes/maps match how the kernel actually indexes the tensors and the oracle's contract?
Confirm the fix cannot produce a wrong-but-successful comparison (fake PASS). If the applied fix is faithful and parallel-safe, verdict=ship; if it has a real defect, needs-fix; if the task was misrouted and is actually structural, non-tractable.
Return ONLY StructuredOutput: task_id, faithful, issues (specific, with kernel/oracle line evidence), verdict.`
}

phase('Route')
log(`Authoritatively routing + fixing ${TASKS.length} tasks`)

const results = await pipeline(
  TASKS,
  (t) => agent(routePrompt(t), { label: `route:${t.id}`, phase: 'Route', schema: ROUTE_SCHEMA }),
  (r, t) => {
    if (!r) return null
    if (!r.fix_applied) return { route: r, review: null }
    return agent(reviewPrompt(t, r), { label: `review:${t.id}`, phase: 'Review', schema: REVIEW_SCHEMA })
      .then((rv) => ({ route: r, review: rv }))
  }
)

const clean = results.filter(Boolean)
const applied = clean.filter((x) => x.route.fix_applied)
const ship = applied.filter((x) => x.review && x.review.verdict === 'ship')
const needsFix = applied.filter((x) => x.review && x.review.verdict === 'needs-fix')
const sev = clean.filter((x) => x.route.lever === 'severance_identify')
const structural = clean.filter((x) => x.route.lever.startsWith('structural_'))
return {
  total: TASKS.length,
  ship_ready: ship.map((x) => ({ task: x.route.task_id, lever: x.route.lever, summary: x.route.fix_summary })),
  needs_fix: needsFix.map((x) => ({ task: x.route.task_id, lever: x.route.lever, issues: x.review.issues })),
  severance_identify: sev.map((x) => ({ task: x.route.task_id, imports: x.route.severance_imports })),
  structural: structural.map((x) => ({ task: x.route.task_id, lever: x.route.lever, why: x.route.structural_reason })),
  applied_count: applied.length,
  ship_count: ship.length,
}
