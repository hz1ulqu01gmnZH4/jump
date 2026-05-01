# Arm C — Research Memo

**Status:** design / pre-implementation
**Authored:** 2026-05-01
**Companion:** `plan_memo.md` (concrete implementation plan)

---

## 1. Thesis

Arm C is a **symbolic, graph-structured abductive solver** for the v2
invented-world benchmark in which **Claude Code is reduced to an orchestrator and
code-emitter** rather than a hypothesis-proposer. The hypothesis space is
explored by an external typed-property-graph engine that performs operations
LLMs are demonstrably bad at (substructure isomorphism, exact anti-unification
across thousands of contexts, exhaustive enumeration over a parameterised rule
schema, CEGIS).

Three load-bearing claims, each with empirical support in the literature:

1. **LLMs cannot natively perform the operations Arm C externalises.**
   Dai et al. (2024) benchmark LLMs on graph-pattern comprehension and find
   that subgraph-isomorphism / motif-counting performance collapses past small
   graphs even with sophisticated prompting (arXiv:2410.05298). Burtsev (2024)
   shows transformers plateau on multi-step CA prediction and fail at *rule
   extraction* even when one-step forecasting works (arXiv:2412.01417). These
   are exactly the operations Arm A and Arm B were tasked with on the v2 CA
   family — and on which they scored 0.000 mean (`verdicts/v2_arm_b.md`).

2. **Symbolic meta-search beats brute enumeration on the kind of rule learning
   v2 requires.** Rule, Piantadosi, Cropper, Ellis, Nye & Tenenbaum (2024,
   *Nature Communications*) demonstrate that search over *programs that revise
   programs* fits human one-shot rule learning across 100 algorithmically-rich
   rules with orders-of-magnitude less search than naive enumeration. The v2
   CA / particle / sequence rules sit squarely in this regime: small primitive
   alphabets, multi-branch compositional rules, no real-world recall to
   exploit.

3. **The "LLM-proposer + symbolic-verifier" architecture is the gold-standard
   precedent.** AlphaProof (DeepMind 2025, *Nature*) reached IMO silver-medal
   performance using a Lean-tactic proposer LLM steering an AlphaZero-style
   symbolic search; the LLM is *never* in the inner reasoning loop. Arm C
   adopts the same split for inductive abduction: Claude Code proposes/edits
   the symbolic engine's primitives and inspects its outputs; the engine alone
   inducts rules from data.

This positions Arm C as **orthogonal**, not incremental, to Arms A and B. The
prediction is that Arm C should sharply outperform A and B on the CA family
(where structural reasoning dominates and Arms A/B both score 0.000), match
them on sequences (where Arm B already scores 1.000 — the symbolic ceiling),
and substantially exceed C-induce on all families (which is the formal
APPROVE bar per `worlds/SPEC.md` §5.3).

---

## 2. What "create connections LLMs can't do" means, sharply

The user's framing — "build connections LLMs can't make" — is best operationalised as
**guaranteed global consistency under exact combinatorial search within a
declaratively-bounded hypothesis class**. Concretely:

- **Exact anti-unification** across all `(cell, neighbourhood-signature) ↦
  next-state` tuples in the training observations. For a 4×4 toroidal grid
  with 12 train_obs, that is 12 × 16 = 192 typed contexts per CA instance;
  exact least-general-generalisation (Plotkin 1970, modernised by Cropper)
  partitions them into equivalence classes and detects conflicts deterministically.
  An LLM scanning prose tokens cannot enforce this.

- **Subgraph isomorphism with exact answer.** When Arm C asks "are the
  property-bag signatures at cells (0,0) and (2,2) identical under the
  diagonal-tetrad mask?" the answer is yes/no with no tolerance. Dai et al.
  show LLMs guess on this even at scale.

- **Counterexample-guided inductive synthesis (CEGIS) with bounded
  completeness.** Within a declared schema lattice (e.g. all transition
  tables `(current_state, count_vector) → next_state` over a fixed
  neighbourhood family), Arm C either returns a consistent rule or proves
  none exists. This is a soundness/completeness guarantee no LLM offers.

- **Discriminative intervention by SAT.** Given two surviving candidate rules
  `h1` and `h2`, generate the smallest input on which they disagree by SAT
  (or exhaustive enumeration over the bounded grid). LLMs cannot reliably
  search disagreement sets.

The honest version of "graph-based" is therefore: **the typed property graph
is the data structure that makes anti-unification, isomorphism, and
counterexample search efficient**, not a magical reasoning substrate.
Graph-of-Thoughts (Besta et al., AAAI 2024, arXiv:2308.09687) uses
"graph" as a metaphor for LLM-internal scratchpad topology; Arm C is the
opposite — the graph lives outside the LLM and the LLM only steers the
engine that operates on it.

---

## 3. Closely-related work and how Arm C differs

### 3.1 Program induction lineage

| Work | What it does | How Arm C differs |
|------|--------------|-------------------|
| **DreamCoder** (Ellis et al., PLDI 2021) | Wake-sleep loop grows a library of program abstractions; neural recognition guides search; E-graph extracts shared subcomponents | Arm C does *not* learn cross-task libraries — single-instance abduction is the v2 task. But the E-graph refactor step is directly the equivalence-class machinery Arm C uses per-instance |
| **HYSYNTH** (Barke, Kalyan, Polikarpova et al., NeurIPS 2024, arXiv:2405.15880) | LLM completions distilled into a task-specific PCFG that biases bottom-up symbolic search | Cleanest precedent for the orchestrator pattern: Claude Code emits priors over candidate rule schemas; the symbolic enumerator does the actual search and scoring |
| **CodeIt** (Butt et al., ICML 2024, arXiv:2402.04858) | Programming-by-example on ARC via sampling + hindsight-relabeled replay in a small DSL | Arm C's intervention loop is the deterministic analogue of CodeIt's hindsight replay — instead of relabelling sampled programs, Arm C asks the harness for ground truth via `intervene` |
| **Symbolic metaprogram search** (Rule et al., *Nature Comm.* 2024) | Search over programs-that-revise-programs fits human one-shot rule learning on 100 rules | The most direct precedent for the v2 task class. Validates Arm C's bet that symbolic search is sufficient where pure neural is not |
| **Bayesian Program Learning** (Lake, Salakhutdinov, Tenenbaum, *Science* 2015) | Hierarchical generative programs over strokes for one-shot Omniglot | Foundational reference for "concepts as programs over typed primitives" — the philosophical backbone of v2 invented-world primitives |

### 3.2 Structure mapping

The Structure-Mapping Engine (SME — Falkenhainer, Forbus, Gentner 1989) is
the canonical algorithm for finding maximal structurally-consistent mappings
between two relational graphs under systematicity constraints. Arm C's
cross-observation alignment step (matching the property graph at `state` to
the property graph at `next_state` and deriving local rewrite rules) is
direct SME application; Petersen et al.'s YARN (2025, arXiv:2603.29997)
shows the modern hybrid form (LLM extracts predicates, SME aligns) and
validates the orchestrator split Arm C uses.

### 3.3 ILP / ALP backends

Cropper & Morel's **Popper** (2021) is the modern reference for inductive
logic programming with SAT/ASP failure constraints; it supports recursion,
lists, numbers, and produces textually-minimal programs. Hocquette et al.
(2025, arXiv:2508.06263) add symmetry-breaking that cuts solve time by
~200× on hard tasks — non-optional for Arm C's interactive loop.
**FOLD-RM** (Wang, Shakerin, Gupta 2022, arXiv:2202.06913) is a lightweight
alternative when the problem reduces to multi-class node classification.
**NeurASP** (Yang, Ishay, Lee 2020/2023) provides the cleanest framework if
Arm C needs probabilistic edges later.

The honest assessment: Arm C **is** an ILP system in disguise. The choice
between (i) calling Popper as a tool and (ii) writing a bespoke solver comes
down to numerics: standard ILP struggles with `argmin`/`argmax`, modular
arithmetic, and integer-weighted neighbour sums unless heavy background
predicates are pre-supplied. v2 worlds use exactly those constructs (the
`d_p == 2` exact-count branch in `world_ca_001`; the `(state[n-1] +
state[n-2] + 3) % 7` branch in `world_seq_001`; the
`max(|dx|, |dy|) + type-mismatch` chromodistance in `world_pt_001`).
Arm C's MVP therefore implements a bespoke solver with: (i) fixed typed
feature vocabularies, (ii) a SAT/ILP back end for guard/weight selection,
(iii) CEGIS for pruning. We borrow Popper's metarule and MDL ideas without
the LP-style bias machinery.

### 3.4 KG-augmented LLM reasoning

GraphRAG (Edge et al., Microsoft 2024, arXiv:2404.16130), Graph-of-Thoughts
(Besta et al., AAAI 2024, arXiv:2308.09687), and the broader KG+LLM
literature (Frontiers 2025 survey) all build a graph and then *let the LLM
query it via natural language*. They are silent on whether the LLM can
actually exploit graph structure; Dai et al. (2024) show it largely cannot.
Arm C inverts the relationship: the graph is queried by deterministic
symbolic operations and the LLM only sees aggregate diagnostics
(equivalence-class conflict counts, MDL scores, CEGIS counterexamples in
abstract feature space). This is the mechanism by which Arm C avoids the
Dai-et-al. failure mode.

### 3.5 ARC-AGI lineage

ARC is the closest existing benchmark to v2 in spirit. Greenblatt's 50%
result (2024) samples ~8000 Python programs per task and revises on
feedback; ARC Prize 2024 (Chollet et al., arXiv:2412.04604) catalogues the
field as primarily program-search-centric; ARC-AGI-2 (2025,
arXiv:2505.11831) confirms the SOTA still tops out around 24% on the
private set even with 2025 LLMs. SOAR and CompressARC (ARC Prize 2025
results) are evolutionary self-improvement and single-task MDL respectively
— both program-search architectures, both vindicating the symbolic split.
**The v2 invented-world benchmark is, in effect, ARC with formal
intervention APIs and predictive scoring on held-out observations** — and
Arm C is the v2 analogue of an ARC symbolic solver.

### 3.6 LLM-as-orchestrator patterns

ReAct (Yao et al., ICLR 2023, arXiv:2210.03629) is the foundational
think-act-observe interleave. Voyager (Wang et al., 2023, arXiv:2305.16291)
grows a skill library of executable code via curriculum + iterative
self-verification. AlphaProof (DeepMind, *Nature* 2025) is the strongest
precedent: LLM proposes Lean tactics, AlphaZero-style MCTS searches the
formal proof tree, the verifier is the symbolic ground truth.

Arm C's loop is ReAct over a symbolic graph backend: observe graph
diagnostics → decide next operation (refine mask, run CEGIS, request
intervention) → execute via the engine → re-inspect. The Voyager skill
library maps to a stash of vetted graph operations / rule templates that
Claude can compose (deferred past the MVP). AlphaProof is the precedent we
cite when defending the architecture against "why not just prompt the LLM
better".

### 3.7 CA rule induction specifically

Springer & Kenyon (2020, arXiv:2012.02179) recover CA local rules from
sparse temporal snapshots via neural search over rule tables. Burtsev
(2024, arXiv:2412.01417) shows transformer plateau on multi-step CA
behaviour. **AutomataGPT** (Burtsev et al., 2025, arXiv:2506.17333) is the
current SOTA: 98.5% one-step forecast and up to 96% rule reconstruction on
held-out 2D *binary* CA. Arm C should beat AutomataGPT on the v2 CA family
because (i) the v2 alphabet is 3-state with type-asymmetric transitions
(outside AutomataGPT's binary training distribution), and (ii) the v2
rules use specific-count and parity branches that exhaustive
neighbourhood-template enumeration captures cleanly while learned-from-data
transformers must approximate.

---

## 4. Failure modes the literature warns about

1. **Property-bag oversummarisation** (gpt5 design review, 2026-05-01).
   Diagonal- and knight-neighbourhood rules are often *position-sensitive*,
   not just count-sensitive; weighted neighbourhoods especially. Pure
   count-vector features will produce mysterious equivalence-class
   conflicts. Mitigation: feature basis must include linear forms over
   position-specific neighbour one-hots with small integer weights, not
   just counts.

2. **Schema-lattice combinatorial explosion.** Naive enumeration over all
   feature-mask subsets is intractable. Mitigation: greedy forward
   selection guided by MDL or a small decision tree, not full lattice
   enumeration. Hash-by-active-feature-mask to build equivalence classes
   incrementally rather than pairwise lgg (Hocquette et al. 2025 confirms
   the order-of-magnitude impact).

3. **Particle representation pitfall.** Anchoring graph nodes to particle
   IDs creates a permutation problem across timesteps. Mitigation: make
   the world grid-centric — nodes are `(cell, t, type-occupancy)` and the
   target is a local move field (displacement to a target cell). This is
   the only known representation under which the v2 particle rule
   `world_pt_001` (ZEX seeks nearest same-type, GLORP flees nearest
   cross-type) admits a clean local-rule encoding.

4. **Sequence family is not really graph-shaped.** The honest framing is
   CEGIS over a tiny DSL of guarded recurrences (guards = index
   predicates; right-hand sides = modular affine forms and XORs). The
   "graph" is the dependency graph of the synthesised program, not an
   observation graph. Arm C should not pretend otherwise.

5. **LLM smuggling pattern-match through the orchestrator interface.**
   If Claude Code can read raw `train_obs` it will inevitably guess rules
   from token-pattern recall. Mitigation: hard isolation — the orchestrator
   sees only aggregate statistics (conflict counts per mask, MDL scores,
   confusion matrices, CEGIS counterexamples in abstract feature-space
   coordinates). Concrete grids are routed through an anonymising
   serializer with random per-run primitive permutation. Linter rejects
   literal magic constants in emitted code. See `plan_memo.md` §5 for
   the enforcement protocol.

---

## 5. Citations to lead with

For the abstract / executive summary of any Arm C write-up:

1. **Dai, Tang, Wu et al. (2024)** — LLMs cannot reliably do graph-pattern
   tasks. arXiv:2410.05298. *The empirical justification for the entire arm.*
2. **Rule, Piantadosi, Cropper, Ellis, Nye, Tenenbaum (2024)** — Symbolic
   metaprogram search wins on rule learning. *Nature Communications* 15.
   *The closest existing precedent to the v2 task class.*
3. **AlphaProof team / DeepMind (2025)** — LLM proposer + symbolic verifier
   reaches IMO silver. *Nature* 2025. *The gold-standard architecture
   precedent.*

Engineering-precedent stack to cite when defending implementation choices:
**HYSYNTH** (LLM seeds symbolic search), **Voyager** (skill library),
**ReAct** (orchestrator loop), **Popper + symmetry-breaking** (ILP backend
when pulled in), **SME** (cross-observation alignment).

Direct baselines to benchmark against in any Arm C write-up:
**AutomataGPT** for CA tasks, **Greenblatt 50%** for ARC-style program
search, **GraphRAG / GoT** for naive LLM-on-graph baselines.

---

## 6. References (full list)

- Barke, Kalyan, Polikarpova et al. (2024). HYSYNTH: Context-Free LLM Approximation for Guiding Program Synthesis. NeurIPS 2024. arXiv:2405.15880.
- Besta, Blach, Kubicek et al. (2024). Graph of Thoughts. AAAI 2024. arXiv:2308.09687.
- Burtsev (2024). Learning Elementary Cellular Automata with Transformers. arXiv:2412.01417.
- Burtsev et al. (2025). AutomataGPT: Forecasting and Ruleset Inference for 2D Cellular Automata. arXiv:2506.17333.
- Butt, Chocron, Gruver et al. (2024). CodeIt: Self-Improving Language Models with Prioritized Hindsight Replay. ICML 2024. arXiv:2402.04858.
- Chollet, Knoop, Kamradt, Landers (2024). ARC Prize 2024: Technical Report. arXiv:2412.04604.
- Chollet et al. (2025). ARC-AGI-2: A New Challenge for Frontier AI Reasoning Systems. arXiv:2505.11831.
- Cropper & Morel (2021). Learning programs by learning from failures (Popper). Machine Learning 110.
- Dai, Tang, Wu et al. (2024). How Do Large Language Models Understand Graph Patterns? arXiv:2410.05298.
- DeepMind AlphaProof team (2025). Olympiad-level formal mathematical reasoning with reinforcement learning. *Nature*.
- Edge, Trinh, Cheng et al. / Microsoft (2024). GraphRAG: From Local to Global. arXiv:2404.16130.
- Ellis, Wong, Nye et al. (2021). DreamCoder. PLDI 2021.
- Falkenhainer, Forbus, Gentner (1989). The Structure-Mapping Engine. *Artificial Intelligence* 41(1).
- Hocquette, Niskanen, Järvisalo, Cropper (2025). Symmetry Breaking for Inductive Logic Programming. arXiv:2508.06263.
- Greenblatt (2024). Getting 50% (SoTA) on ARC-AGI with GPT-4o. Redwood Research blog.
- Lake, Salakhutdinov, Tenenbaum (2015). Human-level concept learning through probabilistic program induction. *Science* 350(6266).
- Petersen, Forbus, Gentner et al. (2025). YARN: Enhancing Structural Mapping with LLM-derived Abstractions. arXiv:2603.29997.
- Rule, Piantadosi, Cropper, Ellis, Nye, Tenenbaum (2024). Symbolic metaprogram search improves learning efficiency and explains rule learning in humans. *Nature Communications* 15.
- Springer & Kenyon (2020). Reconstructing cellular automata rules from observations at nonconsecutive times. arXiv:2012.02179.
- Wang, Shakerin, Gupta (2022). FOLD-RM. arXiv:2202.06913.
- Wang, Xie, Jiang et al. (2023). Voyager. arXiv:2305.16291.
- Yang, Ishay, Lee (2020/2023). NeurASP.
- Yao, Zhao, Yu et al. (2023). ReAct. ICLR 2023. arXiv:2210.03629.
- Zahavy, T. (2026). LLMs can't jump. PhilSci Archive 28024 — the project's framing paper.
