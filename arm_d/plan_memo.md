# Arm D — Plan Memo

**Status:** design / pre-implementation
**Authored:** 2026-05-01
**Companion:** `research_memo.md` (literature positioning and thesis)

---

## 0. One-paragraph summary

Arm D is a **breadth-first ideation pipeline** for the v2 invented-world
benchmark. For each instance: (1) generate a pool of N candidate
`hidden_rule_fn` Python sources via persona-jittered LLM sampling
(`engine/claude_generator.py`) plus a small canonical-template seed
(`engine/template_seeds.py`); (2) score every candidate on the
instance's `train_obs` using a deterministic `engine/scorer.py`; (3)
take the top-K by `(train_acc, -mdl)`; (4) if no candidate scores ≥ ε
on train, run a deterministic AST-level mutation + crossover pass
(`engine/mutator.py`, `engine/crossover.py`) and re-score; (5) submit
the highest-scoring survivor (after a strict linter pass on the
emitted source). Claude Code is the **generator**, never the
evaluator: it writes Python source variants given the train_obs and
nothing else, and the deterministic scoring + selection chooses the
winner. The MVP runs end-to-end with a mock generator
(`engine/mock_generator.py`) so the pipeline is exercised without
needing live `claude -p` calls; the live generator is a thin module
documented as **P-D2**.

---

## 1. Scope and non-goals

### 1.1 In scope (MVP, P-D0 + P-D1)

- All three families end-to-end with the MVP pipeline.
- Mock generator (`engine/mock_generator.py`) returning a fixed pool
  of canned candidates drawn from the C-induce template library plus
  family-appropriate variations; lets `python arm_d/run_arm_d.py` run
  end-to-end with no external dependency.
- Deterministic scorer + selector + linter.
- AST-level mutation pass (operator swaps, constant nudges, neighbour
  offset swaps).
- Source-level crossover pass (branch-tree recombination).
- Result file `arm_d/results_v2.json` matching the schema of
  `arm_b/results_v2_merged.json`.
- `FAIL_TIMEOUT` / `FAIL_NO_SUBMIT` / `FAIL_NO_HYPOTHESIS` /
  `FAIL_EXCEPTION` status reporting per the v2-P6 convention
  (`arm_b/SPEC_v2_p6.md`).

### 1.2 Out of scope (deferred)

- **P-D2** — Live `claude -p` generator. The wiring is sketched in
  `orchestrator/claude_runner.py` (header-only) but the MVP does not
  invoke real Claude calls. Reason: makes the MVP runnable in any
  environment without claude-code installed; the same scoring +
  selection pipeline operates on either source of candidates.
- **P-D3** — Multi-generation evolutionary loop (FunSearch-style
  islands with cross-island migration). MVP runs at most 2
  generations: initial pool → mutation/crossover pass → submit. A
  third or fourth generation could improve hard CA results but is
  bounded compute, not principled.
- **P-D4** — Fine-tune-on-survivors (the SOAR step). Out of scope:
  no model-fine-tune in the project plan.
- **Cross-instance transfer.** Each instance is solved
  independently; no surviving hypothesis from instance A is reused
  on instance B. v2 is single-instance abductive; cross-instance
  transfer would conflate it with library learning à la DreamCoder.
- **Direct comparison against AutomataGPT or other CA-specialised
  models.** Out of scope; flagged for v3.

### 1.3 Methodology constraints inherited from `worlds/SPEC.md`

- §1.4: "No self-assessment prose." Arm D never emits
  `novelty_justification`, `abduction_notes`, or any field describing
  its own reasoning. Only `hidden_rule_fn` Python source is
  submitted. Validated by `engine/linter.py` (rejects any candidate
  containing those literal field names or equivalent reasoning-prose
  strings).
- §3 family-specific intervention APIs are the *only* permitted
  world-mutation surface. Arm D in MVP does not call `intervene` at
  all (passive observation only). When the live generator is wired
  in (P-D2), the prompt explicitly forbids referring to the
  intervention API.
- §5.5: Verdict file must include verbatim final
  `executable_hypothesis`, per-instance accuracy, and per-instance
  scoring-process paragraphs where accuracy deviates ≥ 0.2 from arm
  mean.
- The submission must be `executable_hypothesis` Python source
  *only*. Verified by lint at submission time.

---

## 2. Pipeline (single instance, MVP)

```
worlds/instances/world_NNN.json
        │
        ▼
┌──────────────────────────────┐
│ 1. Generator                 │   produces N candidate sources.
│    (mock OR claude -p)       │   N defaults to 80 for SEQ/PT, 200 for CA.
└──────────────────────────────┘
        │
        ▼
┌──────────────────────────────┐
│ 2. Linter                    │   reject syntax errors, banned imports,
│    + Compile gate            │   reasoning-prose tokens, intervention literals.
└──────────────────────────────┘
        │
        ▼
┌──────────────────────────────┐
│ 3. Scorer (train)            │   for each candidate, compute
│                              │   train_acc = mean(predicted == actual on train_obs).
│                              │   per-call compile + exec; soft-timeout 1s/call.
└──────────────────────────────┘
        │
        ▼  (if any candidate train_acc ≥ ε_done = 1.0)         (else)
        ├──────────────────────┐                                  │
        ▼                      │                                  ▼
┌──────────────────────────┐   │                ┌──────────────────────────┐
│ 4a. Selector             │   │                │ 4b. Mutator + Crossover  │
│   pick top by            │   │                │   AST-level edit pass on │
│   (train_acc, -mdl)      │   │                │   top-K survivors;       │
│                          │   │                │   re-score; re-select.   │
└──────────────┬───────────┘   │                └──────────────┬───────────┘
               │               │                               │
               │               │                               ▼
               │               │                ┌──────────────────────────┐
               │               │                │ 4c. Optional 1× re-prompt│
               │               │                │     (LLM, if available;  │
               │               │                │      no-op in mock)      │
               │               │                └──────────────┬───────────┘
               │               │                               │
               ▼               │                               ▼
┌──────────────────────────────────────────────────────────────────────┐
│ 5. Final lint + harness.submit_hypothesis(survivor.source)           │
│    accuracy on test_obs is the final result.                         │
└──────────────────────────────────────────────────────────────────────┘
```

The selector is mechanical (no LLM in the loop). The optional re-prompt
in 4c is conditional on `--use-claude` (default off in MVP).

---

## 3. Generator details

### 3.1 Mock generator (`engine/mock_generator.py`)

For the MVP, the mock generator returns a fixed pool drawn from:

- **C-induce template library**, expanded to ~30 templates per family
  (the `controls.py` C-induce module covers ~12 templates total; the
  mock pool expands this to cover diagonal/knight/von-Neumann
  neighbourhoods, multiple thresholds, multiple distance metrics, etc.).
- **Random parameter draws** within each template (e.g. 5 random
  threshold values per template, 6 random offset combinations per
  neighbourhood-shape template).
- **Family-specific seed templates** that encode the worked-example
  rule shape from `worlds/SPEC.md` §4 (so the SEQ family pool always
  contains a "branch on prime(n) / parity(n) → XOR or sum" template,
  even if the parameters are wrong).

The mock pool sizes (per instance):
- CA: ~150 candidates (largest because the schema space is biggest).
- PT: ~80 candidates (4 distance metrics × 2 seek/flee × 5 axis rules
  × 2 type-asymmetries).
- SEQ: ~80 candidates (5 guard predicates × 4 RHS forms × 4 moduli).

Total pool size across the 17 instances: ~1700 source variants.
Per-instance scoring cost: ~80 × ~2 ms per candidate = < 200 ms.

The mock pool is **deterministic given the instance ID**: the same
instance always produces the same pool. This is the reproducibility
win over a stochastic LLM generator at the cost of obvious coverage
gaps for instances whose true rule is not in any template.

### 3.2 Claude generator (`orchestrator/claude_runner.py`, P-D2)

When wired in, the live generator:

1. Loads the instance's `train_obs`, `intervention_api` (for
   reference only — *not* passed to the prompt body), `primitive_glossary`.
2. For each persona in `PERSONAS = ["careful", "creative",
   "empirical", "minimalist", "pattern-wide", "pattern-narrow"]`,
   issues a `claude -p` call at temperature 0.9 with prompt template
   `prompts/generate_<persona>.md`.
3. Each call requests M variants (default M = 14, so 6 personas × 14
   = 84 variants per instance). The prompt explicitly says: "Output
   ONLY a JSON list of {source: string} objects, no commentary."
4. Parses the response, splits into individual sources, returns to
   the pipeline.

The Claude generator is *additive* to the mock pool, not replacement:
when `--use-claude` is set, the mock pool is generated as well and
both pools are concatenated before scoring. This guarantees the
pipeline produces results even if the Claude calls all time out (a
realistic failure mode per `arm_b/SPEC_v2_p6.md`).

Subprocess wiring inherits Arm B's mitigations:
- 900 s turn-1 timeout, 600 s subsequent.
- 3 retry attempts with backoff [0, 30, 60].
- `start_new_session=True`, `os.killpg` on timeout.
- Per-instance hard cap 1800 s.
- `--include-partial-messages`, `--output-format stream-json`,
  `--dangerously-skip-permissions`, `--model claude-sonnet-4-6`,
  cwd `/tmp` to avoid loading project-level CLAUDE.md.

### 3.3 Persona library

Six personas, each implemented as a different system-prompt prefix
(file = `orchestrator/prompts/generate_<persona>.md`):

| Persona | Style hint in prompt | Intended diversity contribution |
|---------|----------------------|---------------------------------|
| careful | "Write the most parsimonious rule that explains every train_obs exactly. Prefer short over clever." | Pulls toward minimal-MDL hypotheses |
| creative | "Invent an unusual but coherent rule. Try unexpected operator combinations (XOR, mod, parity, argmin)." | Pulls toward novel branch structures |
| empirical | "Look at the train_obs as data: what summary statistic predicts each next_state? Use that statistic." | Pulls toward feature-extraction-style rules |
| minimalist | "Write the rule in ≤ 10 lines of Python. No helper functions. No imports beyond `copy`." | Length cap forces compositional choices |
| pattern-wide | "The rule may use a wide neighbourhood (knight-move, 6-cell-radius, position-dependent)." | Coverage for hard CA invention criteria |
| pattern-narrow | "The rule depends on at most 4 cells / particles / elements. Find the smallest local mask that suffices." | Coverage for the easy/medium cases |

The personas are intentionally orthogonal. `careful + minimalist` and
`creative + pattern-wide` are the two extremes; the four middle
personas span the design space.

### 3.4 Prompt structure

Common prompt body (per persona, per instance):

```
You are a Python programmer. Below is an empirical observation set:
a list of (state, next_state) pairs from an unknown deterministic rule.
Your task: write Python source for `def hidden_rule_fn(state) -> next_state`
that, when run on each `state`, produces the corresponding `next_state`.

You may NOT call any external API. You may NOT refer to the words
'intervene', 'experiment', 'gather more data', or any reasoning-prose
field. Your output is the function source ONLY.

train_obs:
<JSON-serialized train_obs, full text>

primitive_glossary (informational only — names are arbitrary):
<JSON glossary>

Output FORMAT (strict): a JSON list of M = <M> objects:
[{"source": "def hidden_rule_fn(state):\n    ..."}, ...]

<persona-specific style hint>

Investigation seed: <random integer> (use to break ties across calls)
```

The investigation seed forces non-deterministic completions across
otherwise-identical calls (Greenblatt's persona-jitter trick at the
prompt level).

---

## 4. Scoring details

### 4.1 The scorer (`engine/scorer.py`)

```python
def score_candidate(source: str, train_obs: list[dict],
                    timeout_s: float = 1.0) -> tuple[float, dict]:
    """
    Compile and run candidate source against each train_obs.
    Returns (train_acc in [0,1], meta dict with per-obs hits/misses, exec time).
    On any exception during compile or per-call exec, that observation
    counts as 0; total exception ⇒ train_acc = 0.0.
    """
```

Per-candidate timeout is 1 s (cumulative across all train_obs of an
instance). Exceptions are caught silently and logged in the meta
dict. The intent is that hostile or buggy candidates do not crash the
pipeline.

### 4.2 The selector (`engine/selector.py`)

Sorting key: `(-train_acc, mdl(source), source_hash_for_stability)`.
- `train_acc` is the primary criterion (higher = better).
- `mdl(source)` is `len(source) + entropy_penalty(source)` where
  the entropy penalty discourages obvious bloat (long variable
  names, redundant comments).
- `source_hash` provides deterministic tiebreak.

When multiple candidates tie at `train_acc = 1.0` (perfect on
train), the selector also runs an internal-validation pass on a
held-out slice (last 2 of 12 train_obs are "validation only" and not
shown to the scorer when selecting between perfect-train candidates).
Selected candidate has highest validation accuracy among the
train-perfect set. (See research_memo §5.4.)

### 4.3 The linter (`engine/linter.py`)

Rejects any candidate whose source:
- Fails to compile.
- Does not define `hidden_rule_fn` at module scope.
- Imports anything beyond `copy`, `math`, `itertools`, `functools`.
- Contains the literal strings `intervene`, `interaction_log`,
  `novelty_justification`, `abduction_notes`, `refinement_steps` as
  identifiers or strings.
- Contains a literal substring matching any action name from any v2
  instance's `intervention_api` (i.e. the names `set_cell`,
  `flip_row`, `inject_pattern`, `spawn`, `remove`, `set_velocity`,
  `change_type`, `set_element`, `insert`, `apply_perturbation`).
- Exceeds 200 lines or 6000 chars (bloat cap; well above any v2
  reference solution).

Linter rejection is non-fatal at pool stage (the rejected candidate
is dropped); only at the `submit_hypothesis` gate is rejection
fatal, in which case the next-best surviving candidate is tried, and
if the entire pool is rejected, `FAIL_NO_HYPOTHESIS` is recorded.

### 4.4 The mutator (`engine/mutator.py`)

AST-level edits applied to a candidate source. Each edit produces a
new candidate; a single source spawns up to ~25 mutated variants.

| Mutation | Example | Coverage |
|----------|---------|----------|
| Comparison swap | `==` ↔ `>=` ↔ `>` ↔ `!=` | Branch threshold drift |
| Arithmetic op swap | `+` ↔ `-` ↔ `^` | Wrong operator in RHS |
| Constant nudge | `n=2` → `n in {1, 3, 4}` | Off-by-one in branch consts |
| Modulus nudge | `% 7` → `% 5, % 11, % 13` | Wrong modulus |
| Neighbour offset swap | von-Neumann ↔ Moore ↔ diagonal | Wrong neighbourhood |
| Branch reorder | reorder `if`/`elif` chain | Branch precedence |

The mutator does NOT use an LLM; all edits are deterministic AST
manipulations from a fixed catalogue. This bounds compute and keeps
the "no LLM in the inner selection loop" property.

### 4.5 The crossover module (`engine/crossover.py`)

For each pair of top-K candidates from different personas (or
different template families in the mock case): take the `if/elif`
branch tree of one and replace branches with branches from the
other. Validates the result compiles before adding to pool. Up to
~K² crossbred candidates per pass (K small, default K=8, so ≤56
crosses).

### 4.6 The submitter (`engine/submitter.py`)

Final gate before `harness.submit_hypothesis`:
1. Re-lint the chosen survivor.
2. Re-score the chosen survivor on train_obs as a sanity check.
3. Submit. The harness scores it on test_obs; result returned.
4. Log: chosen source, train_acc, test_acc, pool_size,
   mutation_pass_used, crossover_pass_used.

---

## 5. Module layout

```
arm_d/
├── plan_memo.md                    # this file
├── research_memo.md                # companion lit memo
├── requirements.txt                # pure stdlib for MVP; pydantic optional
├── engine/
│   ├── __init__.py
│   ├── scorer.py                   # train-acc + MDL + per-candidate exec sandbox
│   ├── selector.py                 # top-K sort key + held-out validation
│   ├── linter.py                   # syntax + ban-list + import gate + size cap
│   ├── mutator.py                  # AST-level mutation catalogue
│   ├── crossover.py                # branch-tree recombination
│   ├── mock_generator.py           # MVP candidate pool (template lib + variations)
│   ├── template_seeds.py           # family-specific canonical templates
│   ├── submitter.py                # final lint + harness.submit gate
│   └── statuses.py                 # FAIL_TIMEOUT / FAIL_NO_SUBMIT / etc. constants
├── orchestrator/                   # P-D2 (live generator) — header-only in MVP
│   ├── __init__.py
│   ├── claude_runner.py            # claude -p subprocess wrapper (sketch)
│   └── prompts/
│       ├── generate_careful.md
│       ├── generate_creative.md
│       ├── generate_empirical.md
│       ├── generate_minimalist.md
│       ├── generate_pattern_wide.md
│       └── generate_pattern_narrow.md
├── run_arm_d.py                    # top-level driver: iterates instances
├── results_v2.json                 # produced by run_arm_d.py
└── run_log_v2.txt
```

Estimated MVP LoC: ~1100 lines Python + ~150 lines prompts. (Comparable
to Arm C MVP estimate.)

---

## 6. Integration with shared infrastructure

### 6.1 Result schema

`arm_d/results_v2.json` matches `arm_b/results_v2_merged.json`
columns + Arm-D-specific `engine_meta`:

```json
[
  {
    "instance_id": "world_ca_001",
    "family": "cellular_automata",
    "difficulty": "medium",
    "accuracy": 0.83,
    "hypothesis_source": "<verbatim Python>",
    "hypothesis_status": "submitted | FAIL_TIMEOUT | FAIL_NO_SUBMIT | FAIL_NO_HYPOTHESIS | FAIL_EXCEPTION",
    "n_interventions": 0,
    "n_turns": 1,
    "fallback": false,
    "engine_meta": {
      "pool_size": 152,
      "n_lint_rejected": 8,
      "best_train_acc": 1.0,
      "n_at_top": 3,
      "mutation_pass_used": false,
      "crossover_pass_used": false,
      "selected_persona": "mock_template_seed:ca_diagonal_tetrad",
      "wall_s": 0.43
    }
  }
]
```

The `engine_meta` block is Arm-D-specific informational data; verdict
tooling tolerates extra fields (verified against
`arm_b/results_v2_merged.json` column set).

### 6.2 Reuse of `harness.py` and `controls.py`

No changes. `harness.WorldHarness(instance)` is constructed by
`run_arm_d.py`; the engine consumes `harness.get_train_obs()` (the
*only* harness call before submission); the engine then calls
`harness.submit_hypothesis(survivor.source)`. **MVP makes zero
`harness.intervene` calls.**

### 6.3 Failure modes mirrored on Arm B

Per `arm_b/SPEC_v2_p6.md` §3.2, fail loud — never identity-fallback
silently. Arm D-specific failure statuses:

- `FAIL_NO_HYPOTHESIS` — entire pool failed lint at submit time, no
  survivor available.
- `FAIL_NO_SUBMIT` — pipeline reached submit gate but
  `submit_hypothesis` raised (should not happen in MVP since the
  scorer and submitter both validate the source).
- `FAIL_EXCEPTION` — uncaught exception in pipeline; recorded with
  `error` field.
- `FAIL_TIMEOUT` — instance hard cap (1800 s) exceeded. In MVP this
  only fires if the mock pool is exceptionally large; the live
  generator inherits this gate from Arm B.

`FAIL_*` instances are excluded from the `mean_acc_valid_only`
headline number and reported alongside `mean_acc_fail_as_zero` per
the v2-P6 convention.

### 6.4 Verdict file

`verdicts/v2_arm_d.md` follows the format of `v2_arm_b.md` exactly:
verdict header, 5-axis rubric table, per-family breakdown vs
controls, per-instance table, per-instance scoring-process paragraph
for any instance whose accuracy deviates ≥ 0.2 from Arm D mean (per
`worlds/SPEC.md` §5.5). The verdict file is **not** generated by the
MVP runner — it is hand-written after the run completes (mirrors
`v2_arm_b.md` workflow).

---

## 7. Phased rollout

### P-D0 — Scaffolding (target: this commit)

- Create directory structure per §5.
- `requirements.txt`: pure stdlib (no pysat, no networkx, no numpy).
- Stub all engine modules with type signatures + minimal logic.
- Mock generator returns the worked-example exact source for the
  three SPEC §4 instances and template-family-appropriate variants
  for the rest.
- `run_arm_d.py` produces a valid `results_v2.json` for all 17
  instances with at least the schema correct. Smoke-test: pipeline
  end-to-end without raising.

### P-D1 — Pool quality and selection (target: 1–2 days)

- Expand `template_seeds.py` to cover every family rule structure
  observed in `worlds/gen.py` (verified by grep against the 17
  instance hidden_rule_fn sources — without copying them, just
  matching their *structural shape*: branch counts, neighbourhood
  shapes, modulus values).
- Implement `mutator.py` with the 6-category AST edit catalogue.
- Implement `crossover.py` with branch-tree recombination.
- Hook `selector.py` validation slice (last 2 train_obs).
- Verify the pipeline end-to-end on all 17 instances and record
  baseline `mean_acc`.
- **Acceptance gate (P-D1)**: Arm D `mean_acc ≥ C-induce + 0.10` =
  ≥ 0.227 across the 17 instances. Per-family target: PT mean ≥
  0.30 (beats Arm B substantively), SEQ mean ≥ 0.80 (matches Arm
  B).

### P-D2 — Live Claude generator (target: 2–3 days, post-MVP)

- Implement `orchestrator/claude_runner.py` with subprocess
  wiring inherited from Arm B.
- Write the 6 persona prompts in `orchestrator/prompts/`.
- Add `--use-claude` flag to `run_arm_d.py`. When set, append the
  Claude pool to the mock pool before scoring.
- Re-run the benchmark with `--use-claude` and compare against the
  P-D1 baseline. Expected gain: +0.10 to +0.20 on CA family
  (where the mock pool's coverage is weakest).
- **Acceptance gate (P-D2)**: Arm D `mean_acc` increases by ≥ 0.05
  over P-D1 baseline OR CA family mean ≥ 0.30 (whichever is
  stricter).

### P-D3 — Multi-generation evolutionary loop (target: 2 days,
deferred unless P-D2 misses APPROVE)

- Add a second generation: take all top-K survivors from
  generation 1, re-prompt Claude with them as in-context examples,
  generate 50 more variants, re-score.
- Add cross-island migration if multiple personas have produced
  near-tied survivors.
- **Acceptance gate (P-D3)**: bumps `mean_acc` by ≥ 0.03 over
  P-D2 OR fixes one previously-failing family.

### P-D4 — Verdict + write-up

- `verdicts/v2_arm_d.md` per `worlds/SPEC.md` §5.5.
- Update root `README.md` with Arm D entry.

---

## 8. Success criteria

The shipping bar for Arm D, in priority order:

1. **APPROVE per `worlds/SPEC.md` §5.3**: `mean_acc ≥ 0.7` AND beats
   C-random by > 0.1 AND beats C-induce by > 0.1 across the
   17-instance benchmark. *(Stretch goal; achieving this would put
   Arm D at the level of Arm B's SEQ-only success applied to the
   full benchmark.)*
2. **Beat C-induce on at least one family** (the realistic bar from
   the user's brief): mean_acc(family) > C-induce(family) + 0.10
   for at least one of {CA, PT, SEQ}.
   - C-induce CA = 0.000 → bar is 0.10 (very low, but non-trivial
     given Arms A and B both score 0).
   - C-induce PT = 0.067 → bar is 0.167.
   - C-induce SEQ = 0.306 → bar is 0.406.
3. **Methodology cleanness**: zero `harness.intervene` calls in the
   MVP runner output; zero silent identity-fallback submissions
   (all failures use the `FAIL_*` status set); verdict file
   populated per §5.5.
4. **Reproducibility**: `python arm_d/run_arm_d.py --seed 42` with
   the mock generator produces deterministic results across runs.
   The live generator is stochastic (LLM sampling) and reproducibility
   is a best-effort property only when `--use-claude` is set.

If criterion 2 fails on every family, the architecture is not
contributing meaningful value and Arm D is considered REJECT-ed.
The expectation is that SEQ should pass criterion 2 trivially
(template-rich rule space), PT should pass with the live generator
(diverse distance metrics), and CA is the discriminating
hard-instance test.

---

## 9. Risk register

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| Mock pool has zero coverage for a CA invention class (e.g. 6-cell-radius 1D rule) | High | Acceptable for MVP; documented as expected failure on `world_ca_004`. P-D2 live generator should cover via `pattern-wide` persona. |
| Mutator AST edits produce uncompilable Python | Low | Each edit is wrapped in `compile()` validation; failures are silently dropped, not propagated. |
| Crossover produces semantic nonsense (e.g. branches that reference variables defined in the other branch's parent function) | Medium | Crossover only swaps top-level `if/elif/else` branches within the same function body, not across function definitions; variable-scope check via simple AST walker before adding to pool. |
| Mock-pool's "perfect-on-train" candidate is actually wrong on test (overfits the train slice) | Medium | The validation-slice tiebreak in `selector.py` mitigates; only an issue if validation slice is not informative (e.g. SEQ length ≤ 4 may not hit any branch boundary). |
| MDL tiebreak prefers a clever-short-but-wrong rule over a verbose-but-right one | Medium | Validation-slice gate is the primary defence; MDL is only used among tied candidates. |
| Live generator (P-D2) all-times-out (Arm B's plague) | High | Mock pool is always run regardless; even with 0/6 personas returning, the mock pool produces a survivor for selection. Worst case: Arm D = mock-pool-only baseline. |
| Greenblatt-style scaling assumption fails — N=200 is insufficient | Medium | If P-D1 misses the family-level acceptance gate, scale N up before adding personas. Diagnostic: log `n_at_top` (number of pool members tied at the best train_acc); if low and best_train_acc < 1.0, increase N. |
| Compute cost at full N × 6 personas exceeds project budget | Low | The mock generator is free; P-D2 with 6 personas × 14 candidates × 17 instances = 1428 `claude -p` calls, comparable to a single Arm B re-run. Within budget per Arm B's track record. |
| Linter rejects every candidate (rare but possible if persona prompts drift toward verbose explanations) | Low | The mock pool is hand-curated and lint-clean by construction; the live generator's prompts explicitly require source-only output; a final fallback to the C-induce template library is wired in via `engine/template_seeds.py`'s `EMERGENCY_FALLBACK` set. |

---

## 10. Open questions for review

These need an explicit answer before P-D2 implementation starts (none
block P-D0/P-D1):

1. **Should the MVP optionally call `harness.intervene` when the entire
   pool fails train, as Option B in `research_memo.md` §4?** The
   research memo argues no (preserves "ideation, not abduction"
   character); the user's brief leaves it open. Recommend NO for the
   MVP and revisit only if P-D2 with the live generator still misses
   criterion 2 on PT or CA.

2. **Should the validation-slice tiebreak fall back to MDL when
   validation is uninformative (e.g. all candidates score 1.0 on
   validation as well)?** Recommend YES; current implementation does
   this in `selector.py`.

3. **Should the live generator (P-D2) be wired to use the Anthropic
   SDK rather than `claude -p` to avoid the cold-start hang Arm B
   suffered?** Recommend keeping `claude -p` for methodology parity
   with Arm B; the cold-start hang is bounded by the timeout and
   retry budget, and the mock pool covers the worst case.

4. **Cross-instance pool reuse.** A surviving hypothesis from
   `world_seq_001` is structurally similar to one that may help on
   `world_seq_003`. Should the MVP attempt cross-instance template
   reuse? Recommend NO (single-instance is the v2 design); flag for
   v3 / library-learning work.

5. **Is the MDL definition sensitive to formatting (whitespace,
   comments)?** Currently `mdl(source) = len(source) +
   entropy_penalty`. Whitespace differs across personas. Recommend
   using `len(ast.dump(ast.parse(source)))` as the primary MDL
   measure and falling back to `len(source.replace(' ', ''))` if AST
   dump fails. Implementing this in `engine/scorer.py::mdl()`.
