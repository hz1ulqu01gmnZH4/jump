# RDR — Research Design Requirements: Intervention as Anti-Homogenization

**Status:** draft v0.1 — direction confirmed by User 2026-05-23, design (HOW) deferred to designer
**Owner:** Director
**Companion specs:** `worlds/SPEC.md` (v2 benchmark), `README.md` (project framing)
**Supersedes:** nothing — this is an additive research direction (provisional **Arm E** / harness measurement extension)

---

## 0. One-paragraph summary

The v2 benchmark already measures whether an agent can produce a predictively
accurate executable hypothesis for an invented world it has never seen. This
RDR adds the question *behind* that score: **when an agent looks successful, is
the success a grounded inference from the world, or an artifact of
homogenization** — the same prior-driven answer recurring across attempts and
across models regardless of what the world actually is? We hold that genuine
abductive competence (with external tools permitted) must be *responsive to the
world*, and that a reproducible-across-models answer at fixed surface context is
a **red flag**, not a success signal. The proposed mechanism that resolves both
problems at once is **intervention**: world feedback is the only source of
divergence that is external to the shared LLM prior. The deliverable is an
experimental design that can measure, on the existing harness, whether
intervention simultaneously (a) raises grounded agreement-on-truth and (b)
lowers prior-driven agreement-independent-of-truth.

---

## 1. Motivation

### 1.1 What we are NOT doing
- We are **not** trying to disprove Zahavy (2026). The working assumption is
  accepted: an LLM, by its nature, does not make the generative leap unaided.
- **Using external tools / symbolic machinery is explicitly in-bounds.** The
  "jump" may be carried by the agent+tool system, not the bare model.

### 1.2 The actual target: homogenization
A success score on the v2 benchmark is ambiguous. Two very different phenomena
both produce a high, reproducible accuracy:

1. **Grounding** — the world is (near-)deterministic, so the *true* rule is a
   single target; agents that genuinely inferred it converge on it. Reproducible
   **because they all found the truth.**
2. **Homogenization** — the agents share a pretraining/RLHF prior, so the same
   context elicits the same answer across attempts and across models, *whether
   or not it matches the world.* Reproducible **because they share a prior.**

Both look like "same input → same output." Standard science treats reproducibility
as the gold standard; **for a claim about creative/abductive capacity it is a red
flag**, because it cannot be told apart from the prior-mode collapse.

### 1.3 Why this is a real failure mode, not a hypothetical
- **Pseudo-randomness collapse:** an LLM asked to "pick a random word/number"
  returns a sharply peaked distribution (the 47/42 effect), recurring across
  models given the same prompt. The output is *independent of the world it
  claims to sample from.*
- **Idea-homogenization (Si et al., Stanford 2024):** LLM-generated research
  ideas were judged more novel than experts', but the distinct-idea pool
  saturates — high cross-sample and cross-model duplication. The literature can
  observe *H* (homogeneous agreement) but **cannot separate it from grounding**,
  because research ideas have no held-out ground truth.

**The v2 invented-world benchmark has held-out `test_obs` (ground truth).** This
is precisely what lets us separate grounding from homogenization — the
contribution this project can make that the idea-generation literature cannot.

---

## 2. Goals and non-goals

### 2.1 Goals (G)
- **G1.** Define a measurable construct that separates *grounded agreement*
  (convergence on the world's truth) from *homogenized agreement* (convergence
  on a shared prior, independent of the world).
- **G2.** Establish whether **intervention** is the operation that flips an
  agent population from the homogenized regime to the grounded regime.
- **G3.** Produce a falsifiable prediction and the conditions under which it is
  rejected (honest null result is an acceptable, publishable outcome).
- **G4.** Reuse the existing harness, scoring, and three world families with the
  minimum new machinery.

### 2.2 Non-goals
- Disproving / defending Zahavy (see §1.1).
- Banning external tools or symbolic solvers.
- Cross-instance library learning / transfer (out of scope; revisit later).
- Any change to the v2 scoring of predictive accuracy itself.

---

## 3. Core hypotheses

- **H-MAIN (intervention × grounding interaction).** Turning intervention ON
  *raises* grounded agreement **G** and *lowers* homogenized agreement **H**;
  turning it OFF does the reverse. The single intervention manipulation moves
  both birds — this is the "shoots multiple birds from the air at once" claim,
  stated as a measurable interaction effect.
- **H-RESP (world-responsiveness).** Under matched surface context, an
  intervening agent's hypothesis **covaries with the hidden rule**; a
  non-intervening agent's hypothesis is **invariant to the hidden rule** (it
  tracks the prior, not the world).
- **H-NULL (the falsifier of our own thesis).** If, with intervention OFF,
  agreement already covaries with the hidden rule (G high without intervention),
  then intervention is *not* the load-bearing mechanism and H-MAIN is rejected.

---

## 4. Key constructs and requirements

### 4.1 Functional, not textual, agreement (**REQ-1**)
Agreement between two hypotheses MUST be measured by **behavioral equivalence**
— identical predictions on a shared probe set — not by source-text or prose
similarity. Two different-looking functions that predict identically are the
same hypothesis; two similar-looking functions that predict differently are not.
*Open (designer): exact probe-set construction and the equivalence/clustering
metric (e.g. prediction-signature hashing, agreement rate on a held-out probe
grid).*

### 4.2 H / G decomposition (**REQ-2**)
For a fixed world and a population of runs (across seeds and across models),
define two scalars:
- **G (grounded agreement):** the component of cross-run agreement that is
  *aligned with `hidden_rule_fn`* — i.e. agreement *and* correctness on `test_obs`.
- **H (homogenized agreement):** the component of cross-run agreement that is
  *not aligned with the truth* — runs agree with each other but not with the
  world.

A credible jump requires **G > H**. High H with low G is the homogenization
artifact (the "47" regime). *Open (designer): the exact estimator decomposing
total agreement into G and H, and its confidence interval / null model.*

### 4.3 Minimal-pair worlds (**REQ-3**)
To dissociate prior from grounding, the benchmark MUST include **minimal pairs**:
two (or more) instances with **identical public surface** — same
`primitive_glossary`, same observation format, same `intervention_api`,
statistically indistinguishable `train_obs` framing — but **different
`hidden_rule_fn`**. Then:
- prior-driven agents give the *same* hypothesis to both members of a pair
  (invariant → homogenization);
- grounded agents give *different* hypotheses that each track their own rule
  (covariant → grounding).

This is the central new generation requirement on top of `worlds/SPEC.md` §3.
*Open (designer): how many pairs per family, and how to certify two surfaces are
"matched" without leaking the rule.*

### 4.4 Cross-model run matrix (**REQ-4**)
Each instance MUST be run as a matrix of
**{model} × {intervention ON, OFF} × {N seeds}**, with raw runs logged so H, G,
and world-responsiveness can be recomputed offline. Minimum model set: the two
already wired (Qwen / Arm A, Claude / Arm B); ≥1 additional model strengthens the
cross-model homogenization claim. *Open (designer): which additional model(s),
and minimum N for adequate power on the interaction effect.*

### 4.5 Reproducibility-as-red-flag reporting (**REQ-5**)
The verdict / report MUST present cross-model reproducibility **explicitly as a
diagnostic axis, not a success axis.** A result of "all models agree" must be
reported with its H/G split. A high-accuracy result with high H and low G MUST
be flagged as *suspected homogenization*, not recorded as a jump.

---

## 5. Success criteria

The direction is **validated** if, with adequate statistical power:
- **SC1.** The intervention × {H, G} interaction (H-MAIN) is observed and
  survives across ≥2 models and ≥2 families: intervention ON shows higher G and
  lower H than intervention OFF.
- **SC2.** World-responsiveness (H-RESP) holds on the minimal pairs: intervening
  agents' hypotheses covary with `hidden_rule_fn`; non-intervening agents' do not.
- **SC3.** At least one family exhibits the homogenization regime under
  intervention OFF (high H, low G) — i.e. the artifact is real and detectable on
  this benchmark, otherwise the instrument is not sensitive.

The direction yields an **honest null** (also a deliverable, not a failure) if:
- **NC1.** G is already high without intervention (H-NULL) — intervention is not
  load-bearing; or
- **NC2.** H and G cannot be separated by the chosen estimator with usable
  confidence — the benchmark cannot adjudicate the question and REQ-2 needs
  redesign.

---

## 6. Scope of work (requirements, not implementation)

Built on the existing harness; the new pieces are deliberately minimal:

| # | New capability | Builds on | Status |
|---|----------------|-----------|--------|
| W1 | Minimal-pair world generation (REQ-3) | `worlds/gen.py`, SPEC §3 | design needed |
| W2 | Functional-agreement + H/G estimator (REQ-1, REQ-2) | `harness.py` scoring | design needed |
| W3 | Cross-model × intervention × seed run matrix + logging (REQ-4) | Arm A/B drivers, ledger | design needed |
| W4 | Verdict template with red-flag reporting (REQ-5) | `verdicts/`, SPEC §6 | design needed |

No change to: predictive-accuracy scoring, the invented-primitive constraint, or
the `executable_hypothesis` submission format.

---

## 7. Open design questions (for designer — opus)

1. **Functional signature (REQ-1):** probe-set construction and the
   equivalence/clustering metric. Held-out probe grid vs. reuse of `test_obs`?
2. **H/G estimator (REQ-2):** the exact decomposition of total cross-run
   agreement into grounded vs. homogenized components, with a null model and CI.
3. **Minimal-pair certification (REQ-3):** how many pairs per family; how to
   prove two public surfaces are matched without leaking `hidden_rule`.
4. **Model set & power (REQ-4):** which third model; minimum N seeds for the
   interaction effect in SC1.
5. **Intervention OFF condition:** passive observation only, or
   observation + a fixed non-adaptive action schedule, as the control?
6. **Relationship to existing arms:** is this a new Arm E, or a measurement
   layer wrapping Arms A–D? (Affects W3 logging shape.)

---

## 8. Decision log

- **2026-05-23 — CONFIRMED by User (direction):** Goal is *not* to disprove
  Zahavy; external tools are acceptable. The research target is **homogenization**
  — apparent success that is over-reproducible across attempts and models is
  suspect — and **intervention** is hypothesized to address jump-grounding and
  anti-homogenization simultaneously. Rationale (User): pseudo-random-word and
  research-idea-homogenization examples show LLM outputs collapse to a shared
  prior independent of context; a real test must show the answer comes from the
  world, not the prior.
- **2026-05-23 — Director (implementation-layer):** Constructs framed as
  functional (behavioral) agreement, H/G decomposition, minimal-pair worlds, and
  a cross-model × intervention run matrix, reusing the existing harness. These
  are measurement-form choices; the open HOW (§7) is delegated to designer and
  remains unconfirmed until designed and User-reviewed.
- **2026-05-23 — designer (opus) delivered `DESIGN_RDR_armE.md`; reviewer (opus)
  APPROVE.** §7 Q1–Q5 resolved at implementation layer (Q1 held-out probe grid
  n=100; Q2 chance-model H/G decomposition with bootstrap null + δ/Δδ CIs; Q3
  3 shared-input-state pairs/family = 18 instances; Q4 add GPT-4o-class third
  model, N=15 seeds (floor 10); Q5 passive=primary OFF null + fixed-schedule
  secondary arm). These are Director-autonomous implementation choices.
- **2026-05-23 — CONFIRMED by User (Q6, direction):** Q6 resolved as **Option A
  — New Arm E** (a single uniform agent loop with model adapters {Qwen, Claude,
  GPT-4o-class} and the {ON / passive-OFF / [fixed]} switch as a parameter),
  *not* the measurement-layer wrapping of Arms A–D. Rationale (User accepted
  Director recommendation): the scientific value rests on a clean, balanced
  model × condition interaction; wrapping Arms A–D would confound *model* with
  *arm-protocol* (Qwen-in-Arm-A vs Claude-in-Arm-B are different loops/prompts)
  and leave the ON/OFF switch partly undefined (Arm D passive-by-design, Arm B
  intervention-driven), yielding a ragged, protocol-confounded matrix. W3 loop
  shape is therefore balanced-by-construction. W1/W2/W4 were already
  implementer-ready; W3 is now unblocked.
- **Open caveat (non-blocking, reviewer note):** power analysis (power≈0.85 @
  d=0.8) pools all 18 worlds; per-family (6 worlds/family) detection power is
  lower than the pooled figure, so SC1's "survives across ≥2 families" bar is
  optimistic at the family level. To be reflected in the W3 run plan.
