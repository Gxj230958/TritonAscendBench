# shapes.yaml readability — research + optimization proposal

Status: PROPOSAL (no code changed). Trigger: knob names (`B`/`L`/`D`/`N`/`T`/`H`/`E`/`C`/`G`/`Dv`…)
are cryptic and a reader cannot tell what they mean or which ones a given operator actually uses.

## 1. What the format looks like today

```yaml
# Operator-family knob grid: values reflect the attention family's real axes.
size_knobs:        # per-task DEFAULTS (one value per knob)
  B: 8
  L: 512
  D: 128
  ...
public_shapes:
- id: P1
  name: tiny
  knobs: {B: 2, L: 64, D: 32, ...}   # per-shape overrides of size_knobs
inputs:            # the REAL semantic anchor: knobs -> tensor shapes
  q: {dims: [E*L, H, D], dtype: bfloat16, dist: randn}
  ...
```

The *meaning* of a knob lives only implicitly in `inputs.<name>.dims` (e.g. `q` is `[E*L, H, D]`, so
`E`,`L`,`H`,`D` mean "num-seqs × seqlen, heads, head-dim"). There is **no glossary anywhere** —
`contract.md` lists tensors but never defines the knobs.

## 2. The two concrete readability problems (data, scored corpus = 101 tasks; 102 incl. `_canary`)

1. **Undocumented single-letter names.** 35 distinct knob names. A shared core of 10 (`B H T D N G E C L Dv`,
   each in 82–96 tasks) plus 25 task-specific (`K V CAUSAL S P BT APPW KW SOFTCAP …`, 1–6 tasks each).
   None are documented; the same letter is not guaranteed to mean the same thing across families.
2. **One-size-fits-all template bloat.** `size_knobs` is a generic ~9-knob grid stamped onto every
   task regardless of what it uses. **Median 60% of a task's declared knobs are truly dead** (avg 9.0
   declared, 5.4 never referenced). "Referenced" = the union of: input `dims`, input `dist`
   expressions (e.g. `offsets:to=E*L`, `full:L`), `derive`/`scalar_knob`, `param_constraints`
   predicates/value-exprs, and `cost_model` formulas — codex re-derived the same 60% median; of the
   non-dead knobs only ~2 are cost-formula-only and ~3 are param-constraint-only. E.g. a `bias` add
   carries `L H E C G` it never uses (`L1_01_bias` is 7/10 dead); `L3_07` uses 3/10, `L3_09` 3/11.
   Only ~19 tasks are already lean. **This is the bigger problem**: even knowing what each letter
   means, the reader can't tell the 3 real knobs from the 6 boilerplate ones.

## 3. Consumers / blast radius (what reads the knobs)

| Consumer | Effect of a change |
|---|---|
| `shape_manifest_sha256` (`canonical_manifest`) hashes `size_knobs` + `inputs` + per-shape `knobs` | **Any** prune/rename of `size_knobs` → hash changes → invalidates every S2/C2 PASS → **full GPU re-grade (~2–3 h)** |
| `param_constraints.py` builds a synthetic `size_knobs` row + per-row overlay | a knob referenced by a constraint predicate (`B>=1`) must not be pruned |
| `cost_model` formulas (`validate_tasks.py`: "universe is size_knobs keys + per-row knobs") | a knob referenced by a cost formula must not be pruned |
| `build_inputs` materialises from `inputs[].dims` | unaffected by pruning a knob no input references (safe) |
| `extend_task_shapes.py` generates over the active knobs (those in `inputs`) | unaffected by pruning dead knobs |
| **comments / new non-hashed top-level fields** | **NOT in the manifest → free, no re-grade** |

Key lever: **documentation (comment legend / glossary / a non-hashed field) is free; pruning or
renaming costs a full re-grade.**

## 4. Options

| # | Option | Readability gain | Cost / blast radius |
|---|--------|------------------|---------------------|
| A | **Central glossary** `docs/reference/shape_knobs.md` defining the 10 core knobs (+ pointer for task-specific ones) | meaning of the shared vocab | free, ~0 risk |
| B | **Per-task legend comment** at the top of each `shapes.yaml` naming ONLY the knobs that task uses, 1-word each (`# B=batch  E=num_seqs  L=seqlen  H=heads  D=head_dim  N=max_blocks`) | "what + which apply here" | free (comment, not hashed); generated from `inputs` so it can't rot |
| C | **Prune dead knobs** from `size_knobs` + every shape row (the 5.4/9 median) | files show only real knobs | manifest change → **full re-grade**; must exclude knobs used by param_constraints/cost_model |
| D | **Rename to descriptive** (`B`→`batch`, `D`→`head_dim`) everywhere | maximal | rewrites `size_knobs`+`knobs`+`inputs.dims`+`param_constraints`+`cost_model`+kernel `param_name_map`; manifest change → full re-grade; very high risk |

## 5. Recommendation (staged hybrid — revised after codex review)

- **Phase 1 — free, do now (A + structured B).** Two pieces, neither touches the manifest hash → **no
  re-grade**:
  - `docs/reference/shape_knobs.md` — a "common conventions, NOT authority" glossary of the 10 core
    knobs (the inputs section stays the per-task authority).
  - A structured, non-hashed **`knob_doc:` map** in each `shapes.yaml` (confirmed: not in
    `canonical_manifest`, not rejected by the validator) giving a one-word gloss per knob the task
    actually uses, **plus** a validator check that `knob_doc.keys()` equals the full referenced-knob
    set. A `knob_doc:` field is chosen over a comment because the patch writer (`file_blocks.py`
    parse+re-dump) does **not** preserve comments — a comment legend would silently rot/disappear.
    The referenced-knob scan MUST include `dist`/`derive`/`scalar_knob` expressions, not just `dims`
    (codex found 6 tasks with knobs used only in `dist`); ambiguous gloss words are authored, not
    guessed.
- **Phase 2 — gated, optional (C).** Prune the truly-dead knobs (excluding any referenced by
  `param_constraints` or `cost_model`). Genuinely leans the files but changes the manifest → must be
  followed by a full S2/C2/golden re-grade (~2–3 h). Do only if the user wants the files minimal.
- **Defer D (rename), don't reject outright.** Correction: the kernel `param_name_map` maps wrapper
  param names → input keys, **not** shape knobs, so renaming knobs does not touch it (my earlier
  blast-radius was overstated). A full rename is still a real cost (rewrites every `dims`/`dist`/cost
  formula + manifest + full re-grade) and descriptive names *lengthen* the dims math (`E*L` →
  `num_seqs*seqlen`), so the glossary+`knob_doc` delivers most of the human-readability at near-zero
  cost. Keep rename as a candidate only if a re-grade window is already planned and minimal files are
  wanted over terse math.

## 6. Side finding (separate from this proposal)

`extend_task_shapes.py`'s active-knob detector (~line 79) scans `dims` but **ignores `dist`
expressions**, so a knob used only inside a `dist` string is not swept during generation. Worth a
follow-up fix independent of readability.

## 7. Resolved review questions

1. Legend form → **structured `knob_doc:` field + validator**, not a comment (comments are dropped by
   the YAML rewrite path; a `knob_doc:` key is free of the manifest and accepted by the validator).
2. "Truly dead" must exclude knobs referenced by `inputs` (dims+dist+derive+scalar_knob) ∪
   `param_constraints` ∪ `cost_model`. No other consumer reads `size_knobs` keys directly (checked:
   `param_name_map` is input-keys, not knobs).
3. Glossary central (`docs/reference/`), per-task `knob_doc:` in `shapes.yaml` (closest to the data,
   machine-validated).
4. Gloss is **not** reliably derivable from `dims` alone → scan all expression sites + author
   overrides for the ambiguous ones.
