"""triton-ascend-bench (TAB) — GPU triton → Ascend 910B triton migration benchmark.

Public surface:
    - dataset.load_task(task_id) → TaskDef
    - eval.run(task_def, solution_path, backend) → EvalResult
    - scoring.fastp_native(measured_ns, baseline_ns) → float
    - manifest.SolutionManifest

See specs/ for the protocol and tasks/ for task definitions.
"""

__version__ = "0.1.0"
__all__ = [
    "dataset",
    "eval",
    "scoring",
    "manifest",
    "tolerance",
    "static_checker",
    "adversarial",
    "timing",
    "backends",
]
