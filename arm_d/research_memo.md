# Arm D — Research Memo

**Status:** design / pre-implementation
**Authored:** 2026-05-01
**Companion:** `plan_memo.md` (concrete implementation plan)

---

## 1. Thesis

Arm D is the **breadth-first ideation** counterpart to Arm C's depth-first
symbolic search and Arm B's single-shot careful skill. The defining bet is:
**when the hypothesis space is well-typed and the verifier is cheap, scale of
sampling beats depth of reasoning.** Concretely, instead of producing one
careful `hidden_rule_fn` per instance, Arm D generates a large *pool* of
candidate `hidden_rule_fn` Python sources (hundreds to low thousands per
instance), scores each on `train_obs` via the harness's own grading function
as a fitness oracle, takes the top-K, applies crossover/mutation when no
candidate scores 1.0 on train, and submits the survivor with highest train
accuracy and smallest source MDL as tiebreak.

This is operationally distinct from the abductive arms in three ways:

1. **Hypotheses are sampled, not reasoned.** Arm B (skill) and Arm C
   (CEGIS) commit to a single working hypothesis and refine it under
   careful local edits. Arm D commits to *no* hypothesis until the pool
   is generated and scored. The model's uncertainty is preserved as
   sample diversity rather than collapsed into a single best guess.

2. **The harness is a verifier, not a reasoning peer.** Arms B/C call
   `intervene` to gather evidence that disambiguates rival explanations.
   Arm D treats `train_obs` as the fitness function and uses
   `intervene` only when the entire pool fails train (the "no candidate
   ≥ ε on train" branch — see §4 below). The bulk of compute is
   spent generating and re-running candidate Python, not deliberating.

3. **Selection is mechanical, not deliberative.** The winner is whichever
   candidate maximises a fixed objective `(train_acc, -mdl(source))`.
   No LLM is in the loop for picking the survivor; the LLM's role ends
   at sampling and (optional) crossover proposal.

These three properties place Arm D in a different lineage from B and C:
the **program-search / sample-and-filter / evolutionary search** family,
not the **careful-single-hypothesis / abductive-refinement** family.

---

## 2. Why ideation is a distinct angle, not a different solver

The user's framing — "ideation rather than abductive reasoning" — maps
onto a well-documented split in the AI-for-program-induction literature.

### 2.1 The split

| Axis | Abductive arms (B, C) | Ideation arm (D) |
|------|-----------------------|------------------|
| Number of hypotheses considered per instance | 1, refined | 10² – 10³, filtered |
| Compute spent on inference | Long single chain | Short many chains |
| Role of the model | Reasoner | Generator |
| Role of the harness | Evidence source for refinement | Cheap fitness oracle |
| Failure mode | Premature commitment to wrong rule | Coverage gaps in the pool |
| Strength | Gets parsimonious rules right when prior is correct | Survives wrong priors via sample diversity |
| Closest precedent | DreamCoder, AlphaProof | Greenblatt-50%, FunSearch, SOAR |

### 2.2 Why both can win on different instances

This is empirically what the ARC literature finds. ARC-AGI-2 (Chollet et
al., 2025, arXiv:2505.11831) reports that the highest-scoring 2025 entries
*combine* a program-search bulk pass with a smaller symbolic-refinement
pass, and the two contribute additively rather than redundantly: instances
where the bulk pass produces a candidate within edit-distance 2 of correct
get refined by the symbolic pass, while instances where bulk pass produces
nothing usable fall back to deeper search. The reverse — bulk-only or
symbolic-only — leaves money on the table. This vindicates running both
Arm C and Arm D against the v2 benchmark; an APPROVE on D doesn't make C
redundant, and vice versa.

For v2 specifically, the families divide the strengths cleanly:

- **CA family** (where Arms A, B both score 0.000): likely Arm-C territory
  — these rules sit in a clean schema lattice (neighbourhood × per-state
  transition table) that exhaustive enumeration can hit. Arm D's
  ideation may also hit them but requires the candidate generator's prior
  to cover diagonal/knight neighbourhoods, which is a strong demand on
  the LLM sampler.
- **Particle family** (Arm B 0.333, mostly fallbacks): likely Arm-D
  territory — the rule schema "type-conditional move toward/away from
  nearest particle by some distance metric" is a 4-axis combinatorial
  space (distance metric × seek/flee × target type × axis-choice rule)
  that an LLM can produce variants for cheaply.
- **Pattern/sequence family** (Arm B 1.000, Arm A 0.833): likely
  ideation+filter is sufficient — Arm B's 6/6 solves used near-zero
  intervention, so the rule space is small and the LLM's prior is good;
  scaling to a pool of 50 candidates and picking the train-perfect one is
  almost certainly going to match or exceed Arm B's careful single-shot.

The honest expectation is therefore: Arm D matches Arm B on SEQ, beats
Arm B on PT, and may or may not beat Arm C on CA — the experimental
question.

---

## 3. Lineage and closest precedents

### 3.1 Greenblatt 2024 — "50% on ARC-AGI by sampling thousands of programs"

Ryan Greenblatt's June 2024 Redwood Research blog post is the canonical
modern demonstration of the ideation pattern. Procedure: prompt GPT-4o to
generate 8000 distinct Python programs per ARC task, score each on the
provided train pairs, keep the perfect-on-train ones, select the most
common output on the test pair as the answer. This achieved ~50% on the
ARC-AGI public set (then-SOTA, in front of every fine-tuned approach).

Key Greenblatt observations Arm D inherits:

- **Sample diversity matters more than per-sample quality.** Performance
  scales with `log(n_samples)` up to ~10⁴; reasoning effort per sample
  has diminishing returns past ~30 s of CoT.
- **Verification is essentially free.** Each ARC train pair is ~30 cells
  of grid; running a Python program on it costs sub-millisecond. The
  bottleneck is generation, not verification. v2 has the same property:
  `train_obs` is ≤12 grids of ≤16 cells, or ≤12 sequences of ≤8 ints,
  or ≤12 particle lists of ≤4 particles. Per-candidate scoring is cheap.
- **Persona/style jitter increases diversity at no quality cost.**
  Greenblatt prompts variants like "you are a careful programmer", "you
  are a creative programmer", "you are a methodical programmer" to break
  the model out of ruts. Arm D adopts this directly (§5 of plan_memo.md).

### 3.2 FunSearch (DeepMind, *Nature* 2024)

Romera-Paredes et al. (*Nature* 625, 2024). LLM (PaLM-2 / Codey)
proposes Python function variants, an island-based evolutionary loop
selects survivors by a problem-specific fitness function (e.g. cap-set
size), winners are fed back into the prompt as exemplars for the next
generation. FunSearch discovered new mathematical constructions for the
cap-set and online bin-packing problems — i.e. *generated functions
better than any known to the model*.

Key FunSearch primitives Arm D adopts:

- **Island model.** Multiple sub-populations evolve in parallel; periodic
  migration of best survivors prevents premature convergence. For v2
  this is overkill at MVP scale but cleanly motivates the
  "generate batches of N candidates with different personas" step.
- **Programs-as-individuals.** The genome *is* the Python source. No
  encoding/decoding indirection. Arm D inherits this directly.
- **Fitness-then-LLM-mutation cycle.** After selecting survivors, the
  next prompt includes the top-3 survivors as in-context examples and
  asks for variants. This is the "cross-pollination from prior winners"
  step in the user's design brief.

What Arm D differs on: FunSearch runs for 10⁵+ generations on a single
problem to discover a single global optimum. Arm D runs for 1–3
generations per instance and stops as soon as a candidate hits perfect
train accuracy. The v2 benchmark is many small problems, not one huge
one.

### 3.3 ARC Prize 2024 / 2025 — SOAR and CompressARC

Chollet, Knoop, Kamradt & Landers (2024, arXiv:2412.04604) catalogued
the field; ARC-AGI-2 (2025, arXiv:2505.11831) updated it after the 2025
prize round. The two top program-search architectures are:

- **SOAR (ARC Prize 2025 winner)** — evolutionary self-improvement over
  generated programs. Each iteration: sample a batch of programs from a
  fine-tuned LLM, score on train, keep top-K, fine-tune the LLM on the
  surviving sources for the next round. Terminal step: select the
  highest-train-scoring survivor. Arm D's MVP omits the fine-tune step
  (no time, no GPU, no fine-tunable model in scope) but inherits the
  rest of the architecture.

- **CompressARC** — single-task MDL. For each task, search the space of
  programs and choose the one minimising `compression_cost(program) +
  reconstruction_error(program, train)`. This is the "MDL tiebreak"
  step in Arm D's selection rule.

### 3.4 Brameier-Banzhaf linear genetic programming + neuroevolution

Brameier & Banzhaf's *Linear Genetic Programming* (2007) is the canonical
text for evolutionary program induction over a fixed instruction set.
Modern incarnations (e.g. NEAT for neural net topology, CGP for
combinational circuits) all share the inner loop: random initial
population, fitness scoring, crossover + mutation, tournament selection.

Arm D's MVP uses LLM sampling in place of random initial population
(a much stronger prior than uniform) and substring-level crossover in
place of operator-level (LLMs produce coherent function bodies, not
expression trees, so source-level recombination respects program
structure better).

The deeper LGP/CGP literature on **introns and bloat** is directly
relevant: when LLM sampling produces verbose hypotheses, the MDL tiebreak
and a length cap on submitted source act as bloat control.

### 3.5 AlphaCode — sample-and-filter at programming-competition scale

Li et al. (DeepMind, *Science* 2022). For competitive programming
problems: sample ~10⁶ Python solutions per problem, filter by execution
on the example I/O pairs, cluster surviving solutions, submit cluster
representatives. AlphaCode reached top-54% on Codeforces.

Two AlphaCode lessons Arm D adopts:

- **Test-time-only scaling.** No fine-tuning required to get the
  sampling boost; an off-the-shelf LM with high-temperature sampling is
  sufficient. v2 with `claude -p` at temperature ~0.9 across many
  prompts is the analogue.
- **Filter aggressively before clustering.** AlphaCode rejects any
  solution that fails any example. v2 analogue: train-accuracy < ε
  filter before considering anything else. The discarded population is
  large but the residual is much higher quality than picking from the
  raw samples.

### 3.6 LLM-as-creative-generator literature

Two recent threads matter:

- **Eureka** (Ma et al., NVIDIA, ICLR 2024). LLM proposes reward
  functions, RL trains agents on them, fitness loop selects the reward
  that produces the best policy. The key insight: LLMs are *good* at
  proposing structured candidates in restricted spaces but *bad* at
  evaluating them, so make the LLM the proposer and a deterministic
  process the evaluator. Arm D applies the same split.

- **OpenEvolve / AutoML-Zero successors** (Real et al., Google, 2020;
  active 2024–2026). Evolve ML pipelines or reward functions by
  iterative LLM-driven mutation. Stronger than pure random search by
  4–10×, weaker than hand-designed pipelines but generalises better.

These lines vindicate the broader "LLMs are creative generators, not
careful evaluators" claim that Arm D is built on.

---

## 4. The "no candidate scores > 0 on train" branch — the pivotal design decision

Per the user's brief, the hardest question for an ideation arm on v2 is:
**what to do on instances where the entire candidate pool fails train?**
This is the realistic outcome on hard CA instances (`world_ca_002`
knight-move 4-state, `world_ca_004` 1D 6-cell-radius, `world_ca_006`
position-dependent neighbourhood). Three approaches were considered:

### 4.1 Option A — "retry with mutation"

Bigger pool, more personas, longer chain-of-thought per sample. This is
the FunSearch path: keep generating until a survivor emerges. **Cost is
unbounded; quality plateau is hard.** Greenblatt's data shows the
log-scaling holds out to 8000 samples then flattens; doubling that
doesn't double success rate. For v2 with ~17 instances and a budget of
hundreds of samples per instance, more budget is unlikely to flip a
genuinely-uncovered hypothesis class into coverage.

### 4.2 Option B — "augment with intervention-derived data"

When the train pool is exhausted with all candidates failing, run a few
`intervene` calls with diagnostic states (e.g. uniform grid, isolated
single non-zero cell, opposing pair) to expand the effective `train_obs`,
then re-sample. **This re-introduces the abductive-evidence-gathering
loop**, which makes Arm D look like a slow Arm B. The user's brief
explicitly warns against this collapse.

### 4.3 Option C — "cross-breed surviving partials + structural mutation" (chosen)

When no candidate scores > ε on train (default ε = 0.0, i.e. zero
candidates are even partially correct), do **not** intervene. Instead:

1. Take the top-K candidates by `train_acc` even if they're 0 (so K
   sources from the original pool, sorted lexically by `mdl(source)`).
2. Run a **structural-mutation pass**: for each candidate, generate
   small Python edits — swap a comparison operator, swap a neighbour
   offset, swap an arithmetic operator, replace a constant with a
   nearby value. Implemented deterministically by AST manipulation in
   `engine/mutator.py`, not by the LLM.
3. Run a **crossover pass**: for each pair of candidates from different
   "personas" in the original pool, take their `if/elif` branches and
   swap them, producing a new candidate that combines both branch
   structures.
4. Re-score the mutated + crossbred pool against `train_obs`.
5. If still no candidate ≥ ε, optionally make ONE final LLM call with
   the top-3 sources as context and an explicit "rethink the
   neighbourhood / metric / branch structure" instruction (the
   FunSearch-style refinement). Submit the best of the resulting pool
   regardless of whether it scores > 0.

This is **Option C: deterministic mutation + crossover + at most one LLM
re-prompt**. It preserves Arm D's character (sampling-dominated, not
intervention-driven) and bounds compute. The contrast with Arm B is
clean: Arm B uses the harness's intervention API to gather *new
evidence*; Arm D uses internal mutation to expand *coverage of the
hypothesis space already implied by train_obs*.

The cost: on instances where the rule is genuinely outside the LLM's
prior + the deterministic mutation reach (likely the hardest CA
instances), Arm D will submit a low-train-accuracy hypothesis that
scores ~0 on test. This is the price of methodological cleanliness:
keeping Arm D recognisably "ideation" rather than letting it bleed into
Arm B's territory.

---

## 5. Failure modes the literature warns about (and Arm D's mitigations)

### 5.1 Pool homogeneity / mode collapse

Default LLM sampling at temperature 0 produces near-identical samples;
even at temperature 0.9 the same seed prompt can lead to mostly-similar
hypotheses. Greenblatt addressed this with persona variants. Arm D
adopts the same: ~6 persona templates ("careful", "creative",
"empirical", "minimalist", "pattern-match-wide", "pattern-match-narrow")
plus a per-call random integer "investigation seed" baked into the
prompt to break determinism. See `plan_memo.md` §3.

### 5.2 Pool quality collapse on hard instances

If every candidate scores 0 on train, mean-of-pool is uninformative.
Arm D's mitigation: the linter rejects degenerate candidates (e.g.
`return state` literal, `return [[0]*n for _ in state]` literal) before
they enter the pool; the survivor selection rule treats `0.0` train
accuracy as a tie and breaks by MDL, which prefers shorter sources;
the `FAIL_NO_HYPOTHESIS` status is reserved for the case where the
linter rejects every candidate AND the deterministic-mutation pass
produces nothing parseable. (Arm D distinguishes this from
`FAIL_TIMEOUT` and `FAIL_NO_SUBMIT` per the v2-P6 convention.)

### 5.3 Smuggling abductive reasoning back through the persona prompt

If the persona prompt contains "use the `intervene` tool to gather
evidence", Arm D becomes Arm B. Mitigation: the persona prompts are
locked to *passive* observation of `train_obs` only; the prompt
explicitly forbids the model from referring to interventions, and the
runner does not pass the `intervention_api` into the prompt body.
Linter check: reject any candidate source that references string
literals from the `intervention_api`. This is the methodological hard
gate that distinguishes Arm D from Arm B at the prompt level.

### 5.4 Source MDL doesn't pick the right rule

MDL is a heuristic. The shortest perfect-on-train hypothesis is not
necessarily the test-correct one — overfitting via cleverness is
possible (Arm B's `world_seq_005` solution is a 7-line piecewise rule
that happens to be exactly right; a 4-line `(state[-1] - state[0]) %
13` would also be perfect on train and would be MDL-preferred but
wrong on test). Mitigation: when ≥2 candidates tie at perfect train
accuracy and MDL difference is < 30%, run all of them on a
held-out-from-train internal validation slice (the last 2 of the 12
train_obs are held out for selection only, not for sampling-time
scoring). This is borrowed from AlphaCode's clustering step.

### 5.5 Greenblatt-style scaling assumption may not hold for v2

Greenblatt found 50% on ARC at 8000 samples; v2 may have a different
scaling curve. The MVP defaults to N=200 candidates per instance with
a per-instance budget cap of 600 s. If results show Arm D plateaus
below APPROVE at N=200, the open question is whether scaling to
N=1000 helps (cheap to test) or whether the bottleneck is prior
coverage (in which case more N doesn't help and the persona library
needs expansion).

---

## 6. Citations to lead with

For the abstract / executive summary of any Arm D write-up:

1. **Greenblatt (2024)** — "Getting 50% (SoTA) on ARC-AGI with GPT-4o."
   Redwood Research blog. *The single closest precedent and the
   architectural template Arm D inherits.*
2. **Romera-Paredes et al. (2024)** — FunSearch. *Nature* 625, 468–475.
   *The gold-standard demonstration that LLM-proposer + symbolic-fitness
   + evolutionary loop discovers genuinely-novel programs.*
3. **Chollet, Knoop, Kamradt, Landers (2024)** + Chollet et al. (2025)
   — ARC Prize 2024/2025 technical reports. arXiv:2412.04604,
   arXiv:2505.11831. *Empirical justification that program-search
   architectures dominate the ARC leaderboard, providing the strongest
   independent evidence for the ideation angle.*

Engineering-precedent stack to cite when defending implementation choices:
**SOAR** (evolutionary loop without fine-tuning),
**CompressARC** (MDL tiebreak),
**AlphaCode** (sample-filter-cluster pipeline at programming-contest scale),
**Eureka** (LLM-proposer + deterministic-evaluator split),
**Brameier-Banzhaf** (introns/bloat control via length cap).

Direct baselines to benchmark against in any Arm D write-up:
**Greenblatt 50%** (the architectural ancestor), **Arm B** (the
careful-single-hypothesis comparison point), **Arm C** (the symbolic
combinatorial-search comparison point), **C-induce** (the
template-enumeration-only baseline that Arm D must clear by > 0.1 to
APPROVE).

---

## 7. References (full list)

- Brameier & Banzhaf (2007). *Linear Genetic Programming*. Springer.
- Chollet, Knoop, Kamradt, Landers (2024). ARC Prize 2024: Technical
  Report. arXiv:2412.04604.
- Chollet et al. (2025). ARC-AGI-2: A New Challenge for Frontier AI
  Reasoning Systems. arXiv:2505.11831.
- Greenblatt (2024). Getting 50% (SoTA) on ARC-AGI with GPT-4o.
  Redwood Research blog, June 2024.
- Hodel (2024). The ARC-DSL: A Domain-Specific Language for ARC. GitHub
  michaelhodel/arc-dsl, MIT, 2024.
- Li et al. (DeepMind, 2022). AlphaCode: Competition-Level Code
  Generation. *Science* 378(6624).
- Ma, Liang, Zhang et al. (2024). Eureka: Human-Level Reward Design via
  Coding Large Language Models. ICLR 2024.
- Real et al. (2020). AutoML-Zero: Evolving Machine Learning Algorithms
  From Scratch. ICML 2020.
- Romera-Paredes et al. / DeepMind (2024). Mathematical discoveries from
  program search with large language models (FunSearch). *Nature* 625.
- ARC Prize 2025 (2025). Official results: SOAR (1st place,
  evolutionary self-improvement), CompressARC (2nd place, single-task
  MDL). Documented in arXiv:2505.11831 and the ARC Prize 2025
  leaderboard.
- Zahavy, T. (2026). LLMs can't jump. PhilSci Archive 28024 — the
  project's framing paper. *(Note: Arm D is the "anti-Zahavy" arm in a
  specific sense — if Zahavy is right that LLMs cannot generate
  genuinely novel hypotheses, then ideation should fail too, since it
  is sampling from the same prior. Arm D is therefore the cleanest test
  of the Zahavy claim: it strips away clever-prompting and inference
  technique and asks "is the prior even capable of producing the right
  hypothesis with sufficient sampling".)*
