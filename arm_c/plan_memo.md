# Arm C — Plan Memo

**Status:** design / pre-implementation
**Authored:** 2026-05-01
**Companion:** `research_memo.md` (literature positioning and thesis)

---

## 0. One-paragraph summary

Arm C is a typed-property-graph + CEGIS solver for the v2 invented-world
benchmark. Each observation is decomposed into a graph whose nodes carry
property bags drawn from a frozen, family-specific feature vocabulary; rule
abduction proceeds by exact anti-unification across nodes (collapsing
property-bag-equivalent contexts into local rule fragments), MDL-scored
schema selection, and SAT-based discriminative-intervention generation
against the shared `harness.intervene` API. **Claude Code is the
orchestrator and code-emitter only**: it reads aggregate diagnostics from
the engine, decides which feature mask to refine or which CEGIS branch to
explore next, and writes the final hypothesis module in the shared
submission format. It never sees raw `train_obs`. The MVP targets the
cellular_automata family (where Arms A and B both score 0.000) and the
APPROVE bar of `mean_acc ≥ 0.7` against C-induce + C-random
(`worlds/SPEC.md` §5.3).

---

## 1. Scope and non-goals

### 1.1 In scope (MVP, P-C0)

- CA family (6 instances): full pipeline.
- Particle and sequence families: stub solvers that produce identity-rule
  baselines so the harness call shape is exercised end-to-end. Real
  solvers are P-C2 / P-C3 (post-MVP).
- Integration with the existing `harness.WorldHarness` and
  `worlds.gen.run_instance` API. No changes to `worlds/`, `harness.py`, or
  `controls.py`.
- Results file `arm_c/results_v2.json` matching the shape of
  `arm_b/results_v2_merged.json` exactly (so verdict tooling reuses).

### 1.2 Out of scope (deferred to P-C2+)

- Particle-system solver (requires grid-centric move-field representation —
  see §3.2 below; spec only).
- Sequence-puzzle solver (requires CEGIS over a guarded-recurrence DSL —
  spec only).
- Cross-instance library learning (à la DreamCoder). v2 is single-instance;
  no library is necessary.
- Switching to Popper or other off-the-shelf ILP backends. The bespoke
  solver is justified by numerics (`argmin`, `argmax`, modular arithmetic,
  integer-weighted neighbour sums) and the smallness of v2 worlds. See
  research_memo.md §3.3 for the trade-off; revisit if MVP solver hits
  scaling limits on harder CA instances.
- Claude-Code-as-skill packaging (à la `arm_b/skill/jump.md`). Arm C uses
  `claude -p` only as a code-runner around the symbolic engine, not as the
  reasoner.

### 1.3 Methodology constraints inherited from `worlds/SPEC.md`

- §1.4: "No self-assessment prose" — Arm C never emits `novelty_justification`,
  `abduction_notes`, or any field describing its own reasoning.
- §3 family-specific intervention APIs are the *only* permitted world-mutation
  surface.
- §5.5: Verdict file must include verbatim final `executable_hypothesis`,
  per-instance accuracy, and per-instance scoring-process paragraphs where
  accuracy deviates ≥ 0.2 from arm mean.

---

## 2. Pipeline (CA family, MVP)

```
worlds/instances/world_ca_NNN.json
        │
        ▼
┌──────────────────────────────┐
│ 1. Property Extractor        │   uses ONLY train_obs + intervention_api;
│    (per-family module)       │   produces typed property graph G_obs.
└──────────────────────────────┘
        │
        ▼
┌──────────────────────────────┐
│ 2. Schema Lattice Search     │   greedy MDL-driven feature-mask selection.
│    + Anti-unification        │   produces equivalence classes per mask.
└──────────────────────────────┘
        │
        ▼  (if conflicts remain)         (if conflict-free + small MDL)
        ├──────────────────────┐                          │
        ▼                      │                          ▼
┌──────────────────────────┐   │            ┌──────────────────────────┐
│ 3. Discriminative        │   │            │ 5. Hypothesis Compiler   │
│    Intervention (CEGIS)  │   │            │    feature-mask + class  │
│    SAT-search smallest   │   │            │    table  →  Python rule │
│    grid where ≥2 surviv. │   │            │    function source       │
│    candidates disagree   │   │            └──────────────┬───────────┘
└──────────────┬───────────┘   │                           │
               │               │                           ▼
               ▼               │            ┌──────────────────────────┐
┌──────────────────────────┐   │            │ 6. Submission            │
│ 4. harness.intervene()   │   │            │ harness.submit_hypothesis│
│    record outcome,       │   │            │ scored on test_obs       │
│    refresh G_obs, loop   │───┘            └──────────────────────────┘
└──────────────────────────┘
```

The orchestrator (Claude Code) sits *to the side* of this pipeline and
decides between (3) and (5) at each step based on diagnostics emitted by
(2). Diagnostics are aggregate only — see §5.

---

## 3. Family-specific representations

### 3.1 CA family (`world_ca_*`)

**Nodes.** One node per `(observation_index, row, col)` tuple. Total node
count for a 4×4 grid with 12 train_obs = 192 nodes per instance.

**Property bag** (frozen feature vocabulary):

- `cur` — current cell state (one of `{0, 1, 2}`).
- `next` — observed next-state (the prediction target; not a feature for
  the antecedent of a rule).
- For each candidate neighbourhood mask N ∈ M_cand (see below):
  - `count_N[s]` for each state `s` — count of state `s` in N.
  - `parity_N[s]` — `count_N[s] % 2`.
  - `presence_N[s]` — `count_N[s] > 0`.
- Position-specific neighbour features (for non-rotation-invariant rules):
  one-hot `pos_N[i] == s` for `i` in N's ordering.
- Cell coordinate parity: `(row % 2, col % 2)` — captures alternating-parity
  rules.

**Candidate neighbourhood masks `M_cand`** — the union covers every
`worlds/SPEC.md` §3a invention criterion:

- Moore-8 (full 8 neighbours).
- Von Neumann-4 (N, S, E, W).
- Diagonal-tetrad (NE, NW, SE, SW). [matches `world_ca_001`]
- Knight-move-8.
- Weighted Moore (each neighbour with weight in `{-2..2}`; integer).
- Alternating-parity (different mask depending on `(row + col) % 2`).

This is intentionally larger than any single rule needs. The schema-lattice
search prunes.

**Anti-unification.** For a chosen feature mask `m ⊆` property-bag fields,
group nodes by their projection onto `m`. Each group is an equivalence
class. If all members of a class have the same `next` value, the class
yields a deterministic local rule:

```
IF (cur, count_diag[ZORK], count_diag[PLOON], count_diag[QUAV]) == (0, ., 2, .)
THEN next = PLOON
```

(matches the `d_p == 2 → ZORK→PLOON` branch of `world_ca_001`.)

If a class has conflicting `next` values, the mask is too coarse → expand
`m` (greedy: add the feature that maximally reduces conflict count by
single-step lookahead, MDL-tiebreak).

**Hypothesis cost (MDL).** For mask `m` with `K` resulting equivalence
classes, cost is `|m| * α + log2(K) * β + conflicts(m) * γ`, with
`α=1, β=2, γ=10`. (γ much larger so conflict-free masks always win when
they exist.) Greedy forward selection from `m = {cur}` until either
conflicts hit zero or feature-mask size hits a cap (default 6 fields).

**SAT-based discriminative intervention.** When ≥2 candidate hypotheses
remain after schema search, generate the smallest 4×4 grid `s` such that
`h1(s) ≠ h2(s)` on at least one cell. This is a small SAT instance
(48 boolean vars for 16 cells × 3 states one-hot, plus per-cell
inequality clause). Use `pysat` (Glucose backend). Submit the grid via
`harness.intervene("inject_pattern", state=blank, row=0, col=0,
pattern=s)` and record the true `next_state`. Update G_obs and re-run
schema search.

### 3.2 Particle family (`world_pt_*`) — spec only, P-C2

Per gpt5 design review (2026-05-01): **anchor nodes to the grid, not
particle IDs** to dodge the permutation problem. Nodes are `(t, x, y)`
plus a `type-occupancy` field (one of `{empty, ZEX, GLORP, ...}`).
Target is a **local move field**: for each occupied cell, the displacement
to the target cell at `t+1`. Features per occupied cell:

- Type at cell.
- Nearest-same-type direction vector (precomputed for 8 candidate
  distance metrics including `chromodistance`).
- Nearest-cross-type direction vector.
- Larger-axis flag.
- Lexicographic tiebreak (x, y) of nearest neighbour.

This converts the type-asymmetric, permutation-laden particle dynamics
into the same local-rule shape as CA. Anti-unification then works as in
§3.1.

### 3.3 Sequence family (`world_seq_*`) — spec only, P-C3

Drop graph framing (per research_memo.md §4.4). Use CEGIS over a tiny DSL
of guarded recurrences:

```
guards    := even(n) | odd(n) | prime(n) | composite(n) | (n mod k == j)
rhs_terms := state[n-1] | state[n-2] | const | (rhs_term op rhs_term)
op        := + | * | XOR | mod
```

Search procedure: enumerate guard partitions of `range(2, max_n)`, for
each partition fit a candidate `rhs` to each cell using SAT/SMT over Z_7
(or whichever modulus the train_obs evidence). Return the MDL-minimal
consistent program. The "graph" here is just the dependency graph of the
synthesised program — a derived artefact, not the inductive scaffold.

---

## 4. Module layout

```
arm_c/
├── plan_memo.md              # this file
├── research_memo.md          # companion lit memo
├── requirements.txt          # pysat, networkx, numpy, pydantic
├── engine/
│   ├── __init__.py
│   ├── property_graph.py     # node, edge, property-bag types; G_obs builder
│   ├── ca_features.py        # frozen feature vocabulary for CA family
│   ├── pt_features.py        # stub for P-C2
│   ├── seq_features.py       # stub for P-C3
│   ├── anti_unify.py         # equivalence-class hash + conflict detection
│   ├── schema_search.py      # greedy MDL forward selection
│   ├── compiler.py           # equivalence-class table → Python rule source
│   ├── cegis.py              # SAT-based smallest-disagreement grid
│   └── stub_solvers.py       # PT/SEQ identity-baseline submitters for MVP
├── orchestrator/
│   ├── __init__.py
│   ├── claude_runner.py      # claude -p subprocess wrapper (mirrors arm_b/run_arm_b_v2.py)
│   ├── diagnostics.py        # aggregate-only views of engine state
│   ├── prompts/
│   │   ├── ca_orchestrate.md # the orchestrator prompt (NO raw obs)
│   │   └── compile.md        # final hypothesis-source emission prompt
│   └── linter.py             # rejects emitted code with magic constants
├── run_arm_c.py              # top-level: iterates instances, calls engine + orchestrator
├── results_v2.json           # produced by run_arm_c.py
└── run_log_v2.txt
```

Total estimated LoC for MVP: ~1200 lines Python + ~200 lines orchestrator
prompts. (Comparable to `arm_b/run_arm_b_v2.py` ~400 LoC + its skill.)

---

## 5. Claude-Code-as-orchestrator: isolation protocol

This is the **load-bearing methodology constraint** that distinguishes Arm
C from Arm B (which exposes `train_obs` directly to Claude via the skill).
Without it, Arm C silently degrades into Arm B with extra steps.

### 5.1 What the orchestrator may see

Per turn, the orchestrator receives a JSON blob containing only:

- Instance metadata: `id`, `family`, `difficulty`, `len(train_obs)`,
  grid dimensions.
- Engine diagnostics:
  - `current_mask` (the active feature mask, e.g.
    `["cur", "count_diag[PLOON]"]`).
  - `n_classes` (number of equivalence classes under `current_mask`).
  - `n_conflicts` (classes with mixed `next` values).
  - `mdl_score` (current cost).
  - `top_5_candidate_extensions` — for each candidate feature to add,
    the projected `n_conflicts` after adding it (computed by the engine,
    not the LLM).
  - `surviving_hypotheses` — count, MDL scores; **not their source**.
  - `last_intervention_outcome` — when CEGIS produced a counterexample
    grid, the orchestrator sees only `(intervention_id,
    refuted_hypothesis_indices)`, never the grid contents.

### 5.2 What the orchestrator may decide

The orchestrator chooses one of a small action set:

- `extend_mask(feature_index)` — pick a feature to add from the top-5 list.
- `run_cegis()` — request the engine generate a discriminative
  intervention.
- `compile_hypothesis(survivor_index)` — finalise; the engine compiles
  the equivalence-class table into Python source via the deterministic
  `engine.compiler.compile_class_table()` function.
- `give_up()` — submit identity baseline and record `FAIL_NO_HYPOTHESIS`.

The orchestrator does **not** write the hypothesis source itself. The
hypothesis is generated mechanically from the equivalence-class table.

### 5.3 Code-emission gates (final hypothesis only)

Even the compiled hypothesis is fed to a linter before submission:

- Reject string literals matching primitive names (`ZORK`, `PLOON`, etc.)
  unless they appear in a generated `STATES = [...]` constant produced by
  `compile_class_table`.
- Reject any integer literal not present in: `{0, 1, 2}` (state values),
  the feature-vocabulary bounds (counts up to 8), the modulus (3 for CA
  next-state).
- Reject `import` of anything beyond `copy`.

Lint failures → re-emit from the equivalence-class table without LLM
involvement.

### 5.4 Data isolation enforcement

`engine/property_graph.py` exposes no `train_obs` accessor to the
orchestrator. The orchestrator runs in a subprocess with `cwd` set to a
temporary directory containing only the diagnostics JSON. The
`harness.WorldHarness` is held by the engine, not the orchestrator. There
is no path by which a raw observation can reach the LLM.

This is the strongest version of the user's "create connections LLMs can't
do" framing: connections drawn by the engine from the actual data,
diagnostics summarised by the engine for the LLM, and the LLM only steers
which symbolic operation runs next.

---

## 6. Integration with shared infrastructure

### 6.1 Result schema

`arm_c/results_v2.json` matches `arm_b/results_v2_merged.json` exactly:

```json
[
  {
    "instance_id": "world_ca_001",
    "family": "cellular_automata",
    "difficulty": "medium",
    "accuracy": 0.83,
    "hypothesis_source": "<verbatim Python>",
    "hypothesis_status": "submitted | FAIL_TIMEOUT | FAIL_NO_SUBMIT | FAIL_NO_HYPOTHESIS",
    "n_interventions": 4,
    "n_turns": 6,
    "fallback": false,
    "engine_meta": {
      "final_mask": ["cur", "count_diag[PLOON]", "count_diag[QUAV]"],
      "n_equivalence_classes": 9,
      "mdl_score": 23,
      "cegis_calls": 4
    }
  }
]
```

The `engine_meta` block is Arm-C-specific informational data — verdict
tooling tolerates extra fields (verified against `arm_b/results_v2.json`).

### 6.2 Reuse of `harness.py` and `controls.py`

No changes. `harness.WorldHarness(instance)` is constructed by the engine,
not the orchestrator; `harness.intervene(...)` is called by the engine's
CEGIS module; `harness.submit_hypothesis(...)` is called by the engine's
compiler module after lint passes.

### 6.3 Failure modes mirrored on Arm B

Per `arm_b/SPEC_v2_p6.md` §3.2, fail loud — never identity-fallback
silently. Arm C-specific failure statuses:

- `FAIL_NO_HYPOTHESIS` — schema search exhausted without conflict-free
  mask under the cap; engine refused to compile.
- `FAIL_LINT` — compiler emitted code that the linter rejected and
  re-emission also failed.
- `FAIL_TIMEOUT` (per-instance cap = 1800 s, mirrored from Arm B).

`FAIL_*` instances are excluded from the `mean_acc_valid_only` headline
number and reported alongside `mean_acc_fail_as_zero` per the v2-P6
convention.

### 6.4 Verdict file

`verdicts/v2_arm_c.md` follows the format of `v2_arm_b.md` exactly:
verdict header, 5-axis rubric table, per-family breakdown vs controls,
per-instance table, per-instance scoring-process paragraph for any
instance whose accuracy deviates ≥ 0.2 from Arm C mean (per
`worlds/SPEC.md` §5.5).

---

## 7. Phased rollout

### P-C0 — Scaffolding (target: 1 day)

- Create directory structure per §4.
- `requirements.txt`: `pysat`, `networkx`, `numpy`, `pydantic`.
- Stub `engine/stub_solvers.py` returns identity baseline for all 17
  instances; verify `run_arm_c.py` produces a valid `results_v2.json`
  with `hypothesis_status="FAIL_NO_HYPOTHESIS"`.
- Smoke-test verdict tooling against the stub output.

### P-C1 — CA family solver (target: 3–5 days)

- `engine/ca_features.py` — frozen feature vocabulary (§3.1).
- `engine/property_graph.py` + `engine/anti_unify.py` — node builder,
  hash-by-mask equivalence-class detector.
- `engine/schema_search.py` — greedy MDL forward selection.
- `engine/compiler.py` — class-table → Python source.
- `engine/cegis.py` — pysat-backed smallest-disagreement grid generator.
- `orchestrator/claude_runner.py` — adapted from `arm_b/run_arm_b_v2.py`
  with raw-obs scrubbed from prompts.
- `orchestrator/prompts/ca_orchestrate.md` — turn template.
- `orchestrator/linter.py` — magic-constant + import gate.
- `run_arm_c.py` — CA-only iteration; PT and SEQ left at stub.
- **Acceptance gate (P-C1)**: Arm C CA `mean_acc ≥ 0.5` and `> C-induce
  CA + 0.1` (i.e. ≥ 0.10, since C-induce CA = 0.000). If gate fails,
  diagnose and patch the engine before unblocking P-C2.

### P-C2 — Particle family solver (target: 3 days, post-MVP)

- `engine/pt_features.py` (grid-centric move-field per §3.2).
- Re-use schema_search, anti_unify, compiler.
- CEGIS adapted to particle interventions
  (`spawn`/`remove`/`change_type`).
- **Acceptance gate (P-C2)**: PT `mean_acc ≥ 0.4` (Arm B PT = 0.333 with
  high failure rate; Arm C target is to clear it on substance).

### P-C3 — Sequence family solver (target: 3 days, post-MVP)

- `engine/seq_features.py` — replaced by CEGIS DSL per §3.3.
- SMT backend (z3 over Z_7) for guard partition + RHS fitting.
- **Acceptance gate (P-C3)**: SEQ `mean_acc ≥ 0.9` (Arm B SEQ = 1.000 is
  the symbolic ceiling; Arm C should approach it).

### P-C4 — Verdict + write-up

- `verdicts/v2_arm_c.md` per `worlds/SPEC.md` §5.5.
- Update root `README.md` with Arm C entry.

---

## 8. Success criteria

The shipping bar for Arm C, in priority order:

1. **APPROVE per `worlds/SPEC.md` §5.3**: `mean_acc ≥ 0.7` AND beats
   C-random by > 0.1 AND beats C-induce by > 0.1 across the 17-instance
   benchmark.
2. **CA-family decisive win**: Arm C CA `mean_acc ≥ 0.5` while Arm A CA =
   0.000 and Arm B CA = 0.000. This is the headline result; the entire
   Arm C thesis lives or dies here.
3. **Methodology cleanness**: zero instances of raw `train_obs` reaching
   the orchestrator (verifiable by grepping the orchestrator prompt log
   for any state literal); zero silent-fallback submissions; verdict
   file fully populated per §5.5.
4. **Reproducibility**: `python run_arm_c.py --seed 42` produces
   deterministic results across runs (the engine is deterministic; the
   orchestrator is the only stochastic component, and its decisions are
   small enough that re-runs should produce ≥80% identical hypotheses).

If criterion 2 fails, do not proceed to P-C2/P-C3 — diagnose first.
If criterion 1 fails but 2 succeeds, the result is still publishable as
"symbolic graph-based abduction beats direct LLM on structural rule
families; remains below APPROVE on numeric/recurrence families".

---

## 9. Risk register

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| Schema-lattice combinatorial explosion on `world_ca_002`/`world_ca_004` (hard) | Medium | Greedy forward selection + MDL cap; if exhausted at cap-6 features without conflict-free mask, fall back to position-specific one-hot features (larger but more expressive) |
| CEGIS produces grids the harness can't apply (e.g. `inject_pattern` doesn't cover the disagreement region) | Low | Pre-validate generated grid against the instance's `intervention_api` action surface; if no action covers the disagreement, fall back to enumerating `set_cell` sequences |
| Orchestrator LLM still smuggles pattern-match through "extend_mask(feature_index)" choices | Medium | Top-5 list is engine-ranked by MDL-projected gain; tie-break by feature lexical order; orchestrator's choice has a maximum impact bounded by the engine's ranking. Validate by comparing Arm C with random-orchestrator vs LLM-orchestrator: if results are identical, the LLM contributes nothing (good — no smuggling); if LLM-orchestrator wins by > 0.05, investigate |
| `claude -p` cold-start hangs (Arm B's plague — see `arm_b/SPEC_v2_p6.md`) | High | Inherit Arm B's mitigations: 900s turn-1 timeout, 600s subsequent, 3 retries, instance cap 1800s, `FAIL_TIMEOUT` recording |
| Bespoke ILP-style solver under-performs Popper on hard instances | Low | If MVP gate fails on a specific instance pattern, slot in Popper (with Hocquette-2025 symmetry-breaking) for that family. Documented as P-C5 contingency, not MVP scope |
| `pysat` not installable in user environment | Low | Fall back to brute-force enumeration over 4×4 grids (3^16 ≈ 43M states; tractable but slower); document in `requirements.txt` |
| Particle move-field representation breaks on instances with type-creation/deletion (none in v2 currently, but possible in future generators) | Low | Not an MVP concern; flag in P-C2 implementation |

---

## 10. Open questions for review

These need an explicit answer before P-C1 implementation starts:

1. **Should Arm C use `claude -p` (Arm B-style) or the Anthropic SDK
   (`arm_a/driver.py`-style)?** Per `arm_b/SPEC_v2_p6.md` §2, the project
   prefers `claude -p` for subscription-cost reasons. Arm C's
   orchestrator is small enough that either works; recommend matching
   Arm B for methodology parity.
2. **Random-orchestrator control.** Should we run an additional control
   `C-orchestrator-random` that replaces Claude with random feature-mask
   choices? This is the cleanest way to demonstrate the LLM is (or is
   not) contributing real value vs the engine alone. Recommend yes;
   include in P-C1.
3. **Feature-vocabulary lock.** Arm C's bias is encoded in the feature
   vocabulary. Should the vocabulary be frozen *before any results are
   seen* (publish-then-run) to avoid overfitting? Recommend yes; commit
   `engine/ca_features.py` before any results are recorded; treat
   vocabulary edits after that as a methodology violation requiring a
   re-run on a fresh world set.
4. **Comparison against AutomataGPT**
   (Burtsev et al. 2025, arXiv:2506.17333). Out-of-scope for v2 (binary
   CA only) but a tempting follow-up benchmark. Defer to v3 planning.
