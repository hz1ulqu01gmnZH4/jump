---
status: completed
project: jump
document: DESIGN_RDR_armE
---

# DESIGN_RDR_armE — Implementable decisions for the anti-homogenization measurement (RDR §7)

**Owner:** designer (opus)  ·  **Companion:** `RDR.md` (requirements), `worlds/SPEC.md` (v2 benchmark), `harness.py`, `controls.py`
**Scope:** resolve the 6 open design questions in `RDR.md §7` to the point where an implementer can build W1–W4 (`RDR.md §6`) without further design discussion. **Q6 is a User Approval Gate — presented, not decided.**

This document is decisions only. It reuses `WorldHarness`, `run_agent_on_instance`, `run_instance`, and the three controls unchanged. No change is made to predictive-accuracy scoring, the `executable_hypothesis` format, the invented-primitive constraint, or the 3-family structure beyond what REQ-3 requires.

---

## Q1: Functional signature (REQ-1) — probe-set construction & equivalence metric

**Options:**
- **Option A — reuse `test_obs` as the probe set.** Measure agreement on the same observations used for accuracy.
- **Option B — held-out probe grid, separate from `test_obs`, sampled from the same input distribution.** Agreement on the probe grid; correctness on `test_obs`.
- **Option C — synthesise probes by intervention-style state construction** (uniform grids, isolated single-cell perturbations, opposing pairs).

**Recommendation: Option B**, with a deterministic, per-world probe grid `P_w` of fixed size; equivalence measured by **pairwise prediction-agreement rate**, with **prediction-signature hashing** as the exact-match (threshold = 1.0) special case; clustering by an agreement threshold.

**Rationale:** REQ-1 demands *behavioral* equivalence, and REQ-2 needs agreement and correctness to be measurable as **separable** quantities. If agreement is measured on `test_obs` (Option A), "two runs agree" and "two runs are both correct" become statistically entangled — exactly the confound REQ-2 must avoid, because the H/G split needs an agreement signal that is *not definitionally* a correctness signal. A separate probe grid (Option B) decouples them: agreement is a property of the hypotheses' behavior over a broad input space, while correctness stays anchored to `hidden_rule_fn` on `test_obs`. Option C alone under-covers the input space and biases toward intervention-favorable inputs; its constructed states are folded into B as a *stratum* rather than the whole grid.

**Implementation sketch:**
```python
# worlds/probes.py  (new, ~40 lines; reuses family state-samplers from worlds/gen.py)
def build_probe_grid(instance: dict, n: int, seed: int) -> list:
    """Return n input states sampled from the instance family's state distribution.
    CA: random rows×cols grids over the instance alphabet (incl. uniform + single-defect strata).
    particle_system: random type/position configs at the instance's particle count (+ isolated-pair stratum).
    pattern_puzzle: random valid seed lists extended to varied lengths (to exercise index-parity/prime branches).
    Deterministic in (instance.id, seed). NEVER calls hidden_rule_fn."""

def prediction_signature(hyp_fn, probes: list) -> tuple:
    """Canonicalize hyp_fn(state) via json.dumps(sort_keys=True) per probe; on exception emit sentinel '<ERR>'.
    Return a tuple of canonical strings (hashable signature)."""

def agreement_rate(sig_a: tuple, sig_b: tuple) -> float:
    """fraction of probes where canonical outputs are byte-equal."""

def cluster_runs(sigs: dict[run_id, tuple], theta: float = 0.95) -> list[set]:
    """Connected-components clustering: edge (r,r') iff agreement_rate >= theta. theta=1.0 == exact signature hash."""
```
**Minimum probe-set size.** To make "different hypotheses land in different clusters" reliable, the grid must let genuinely-different rules disagree on ≥1 probe with high probability, and must estimate an agreement proportion with SE small enough to separate θ=0.95 from 1.0. SE of a proportion is `sqrt(p(1-p)/n)`; `n = 100` gives SE ≤ 0.05. **Recommend `n = 100` probes per world (floor 50)**, stratified: 60% random, 20% uniform/degenerate, 20% single-defect/opposing-pair. Larger `n` is cheap (verification is sub-millisecond per `RDR`/Arm-D evidence) so 100 is the default, not a ceiling.

---

## Q2: H/G estimator (REQ-2) — decomposition, null model, CI

**Definitions.** Fix a world `w` and a population of runs `R` (across seeds × models within one intervention condition). For run `r` compute the signature `σ_r` on probe grid `P_w`, and the **truth signature** `t = signature(hidden_rule_fn, P_w)`. Let `g_r = agreement_rate(σ_r, t)` ∈ [0,1] (closeness-to-truth on the probe grid; this is predictive accuracy generalised off `test_obs`). Let `a(r,r') = agreement_rate(σ_r, σ_{r'})`.

**The decomposition.** Total cross-run agreement is `Ā = mean_{r<r'} a(r,r')`. Under the null of **no homogenization**, two runs' errors are *conditionally independent given their closeness-to-truth*, so their expected agreement is the chance-model value:

```
Ê(r,r') = g_r · g_{r'}                          # both right on a probe → agree
        + (1 - g_r)(1 - g_{r'}) · κ̄             # both wrong → agree only by coincidence
```
where `κ̄` = the probe-averaged probability that two **independent** wrong answers coincide. **`κ` is estimated from an independent reference population, NOT from the run population `R` being analyzed** (see next sub-section). The earlier spec ("estimate `κ` empirically from the observed distribution of wrong outputs per probe") is **withdrawn as circular** — see the bug note below.

### κ estimation — independent reference population (revises the circular estimator)

**Why the original estimator is wrong (circular).** Estimating `κ` from the very `sigs` being decomposed makes the estimator *blind to homogenization*. Under a homogenized population (all runs return the same wrong answer), the observed wrong-output diversity collapses, so any `κ` read from `R` is inflated toward agreement. Then `Ê → a`, the residual `H = max(a − Ê, 0) → 0`, and `δ = G − H` flips **positive** — exactly the failure GAP2 caught (`eval/demo_hg.py`, commit `b5c4a42`): `G=0.887  H=0.113  δ=+0.774` on a fully-homogenized world. A per-cell alphabet shortcut (`κ = 1/(k−1)`, e.g. `1/2` for a 3-value cell) is equally invalid: a prediction signature is a **whole-grid** `json.dumps(next_state)`, so the relevant output space for a 4×4 3-value CA is `3¹⁶ ≈ 43M`, and two independent wrong grids coincide with probability `~1/43M`, not `0.5`. Per-cell κ over-states `Ê`, under-states `H`, and again misses homogenization.

**Definition restored.** `κ` must estimate `P(two independent wrong outputs coincide on a probe)`. "Independent" means *generated without knowledge of the test population or its instances*. The only sound source is therefore a reference population drawn outside `R`.

**Options.**
- **Option A — C-random reference, empirical per-probe coincidence (recommended).** Draw `N_ref` independent random hypotheses by reusing `c_random_agent` / the family random-rule samplers (`controls.py`, `worlds/gen._*_random_state`). Run each on the *same* probe grid `P_w` (never on `test_obs`, never on `hidden_rule_fn`) to get reference signatures `{ρ_j}`. For each probe `i`, take the reference outputs that **disagree with truth** (`ρ_j[i] ≠ t[i]`) and compute the empirical pairwise coincidence `κ_i = (#equal wrong-pairs)/(#wrong-pairs)`. The decomposition uses the scalar `κ̄ = mean_i κ_i`. Random hypotheses are independent of `R` by construction, so `κ_i` measures genuine accidental coincidence — including *natural* per-probe collisions (e.g. degenerate uniform probes where many rules emit the same grid), which a global constant cannot capture.
- **Option B — family rule-bank reference.** Use the family's paired rule bank (the B-member rules from `generate_minimal_pairs()`) as the reference population. More structured than random, but the bank is tiny (≈3 rules/family) → too few wrong-pairs per probe to estimate `κ_i` stably, and "plausible" rules cluster, *understating* coincidence diversity. Rejected as primary; usable only as a supplementary stratum.
- **Option C — analytic cardinality from the reference.** For each probe, count distinct reference outputs `m_eff,i` and set `κ_i = 1/(m_eff,i − 1)`. Grounds `m_eff` in an independent population (not `R`), so it is non-circular, but it assumes a *uniform* distribution over the `m_eff` outputs and so under-states coincidence when the wrong-output distribution is peaked. **Adopted as the per-probe fallback** when probe `i` has `< 2` reference wrong-outputs (so the empirical pairwise rate of Option A is undefined): `κ_i = 1/max(m_eff,i − 1, 1)`.

**Recommendation: Option A** (empirical per-probe coincidence from a C-random reference), with **Option C as the per-probe fallback** for sparse probes. **`[Director決定待ち]`** — this changes *what `κ` measures* (test-population diversity → independent-reference coincidence), a direction-level decision. `N_ref`, the sweep grid, and exact signatures below are implementation details (CONFIRMED).

**Rationale.** (a) *Non-circular:* the reference signatures are never the `sigs` under analysis (Hard-constraint 1), so homogenization in `R` cannot leak into `κ`. (b) *Right scale for whole-grid JSON outputs:* coincidence is measured directly on full `json.dumps(next_state)` signatures over `P_w`, so the `~1/43M` true scale is recovered empirically rather than mis-set by a per-cell alphabet. (c) *Tractable:* `N_ref` random rules × ~100 probes × sub-ms verification (`RDR`/Arm-D evidence) ⟹ well under the 5 s/world budget (Hard-constraint 2); `N_ref = 100` (floor 50) gives ≥ ~1.2 k wrong-pairs/probe on high-entropy probes. (d) *Validity preserved:* `κ̄` is computed **once per world** and passed *unchanged* into the observed decomposition, the null model, and every bootstrap resample, so `G/H/δ/Ā` keep their definitions and the only thing that changed is the (now independent) constant they share.

**Null-model correction (same independence principle).** The parametric null (below) previously drew synthetic wrong outputs from the **test population's** empirical wrong distribution — also circular. It now draws from the **reference population's** per-probe wrong-output distribution `{ρ_j[i] : ρ_j[i] ≠ t[i]}`, keeping the null independent of homogenization in `R`. `E[H₀] ≈ 0` still holds under true conditional independence.

**κ-sensitivity check (mandatory, all options).** Because `δ`'s sign can depend on `κ̄`, every world is re-decomposed over a κ-sweep spanning **≥ 10×**: `κ ∈ {0, κ̂/10, κ̂, 10·κ̂, κ_hi}` clamped to `[0, 0.5]`, where `κ̂` is the reference estimate and `κ_hi = 1/(k−1)` is the deliberately-inflated per-cell upper bound. Report the `sign(δ)` (equivalently `G>H` vs `H>G`) at each grid point. **If the classification is constant across the whole sweep → robust; if `sign(δ)` flips anywhere in the range → flag the world `AMBIGUOUS` and exclude it from the SC1 interaction.** This diagnostic is REQUIRED in the W4 verdict template (a `κ-robust? [Y/N]` column per world).

Define the two scalar estimators:
```
G := mean_{r<r'} Ê(r,r')               # agreement explained by shared closeness to truth (GROUNDED)
H := mean_{r<r'} max(a(r,r') - Ê(r,r'), 0)   # excess agreement beyond truth-explained (HOMOGENIZED)
```
So `Ā ≈ G + H`. **G is high when runs agree because they all found the truth; H is high when runs agree with each other but not with the world** (the "47" regime). A credible jump requires `G > H` (REQ-2). This satisfies §1.2: both "grounding" and "homogenization" produce high `Ā`, and only the `Ê`-residual `H` distinguishes them.

**Null model.** Parametric bootstrap under conditional independence: for each run `r` synthesise an output on each probe that equals truth with probability `g_r` and otherwise draws a wrong value independently from the empirical wrong-output distribution. Compute `H` on the synthetic population; repeat `B = 1000` times → null distribution `H₀`. Under the null, `E[H₀] ≈ 0` (positive only from finite-sample noise). The observed `H` is significant if it exceeds the `(1-α)` quantile of `H₀`.

**CI / test.** Bootstrap over runs (resample `R` with replacement, `B = 1000`) to get percentile CIs on `G`, `H`, and `δ = G - H`. Report `H > G` / `G > H` via the sign and CI of `δ`. The SC1 interaction is the second difference `Δδ = δ_ON - δ_OFF`, with CI from the same run-bootstrap applied per condition (paired by world).

**Implementation sketch:**
```python
# eval/hg.py (~120 lines, pure-numeric; reuses controls.c_random_agent + worlds.probes; no harness change)

def estimate_kappa_reference(instance: dict, probes: list, truth_sig: tuple,
                             n_ref: int = 100, seed: int = 0) -> dict:
    """Estimate per-probe wrong-output coincidence from an INDEPENDENT reference
    population — never reads the test `sigs`.
      1. Draw n_ref random hypotheses via c_random_agent / family random samplers
         (seed = f'{instance.id}:kref:{seed}:{j}'), one rule each.
      2. ref_sigs[j] = prediction_signature(rule_j, probes)         # same P_w as the test
      3. For each probe i: W = [ref_sigs[j][i] for j if ref_sigs[j][i] != truth_sig[i]]
           kappa_i = (#equal pairs in W)/(C(|W|,2))      if |W| >= 2   (Option A)
                   = 1 / max(len(set(W_all_i)) - 1, 1)   if |W| <  2   (Option C fallback)
      4. kappa_bar = mean_i kappa_i
    Returns {'kappa': kappa_bar, 'kappa_i': [...], 'ref_wrong_dist': [W_i,...], 'n_ref': n_ref}.
    MUST complete < 5 s/world (n_ref * len(probes) sub-ms verifications)."""

def hg_decompose(sigs: dict, truth_sig: tuple, kappa: float) -> dict:
    """kappa is now REQUIRED (scalar κ̄ from estimate_kappa_reference); no internal
    estimation, no fallback default — raise if None (fail loud).
    G,H,δ,Ā definitions unchanged. returns {'G','H','delta','Abar','g','kappa','n_pairs'}."""

def hg_null_pvalue(sigs: dict, truth_sig: tuple, kappa: float,
                   ref_wrong_dist: list, B=1000, seed=0) -> float:
    """Parametric null under conditional independence. Synthetic wrong draws come from
    ref_wrong_dist[i] (the REFERENCE population), NOT from the test sigs. Each null
    population is scored with the SAME fixed `kappa`. p = P(H_null >= H_obs)."""

def hg_bootstrap_ci(sigs, truth_sig, kappa: float, B=1000, seed=0) -> dict:
    """Resample runs with replacement; pass the fixed reference `kappa` into every
    hg_decompose call. 95% percentile CIs on G, H, delta. (approach unchanged)"""

def hg_kappa_sensitivity(sigs, truth_sig, kappa_hat: float, k_alphabet: int) -> dict:
    """Re-decompose over kappa in {0, κ̂/10, κ̂, 10·κ̂, 1/(k-1)} clamped to [0,0.5].
    Returns {'grid':[...], 'sign_delta':[...], 'kappa_robust': bool}.
    kappa_robust = all sign(delta) identical across the grid; else world is AMBIGUOUS."""
```

**Caller wiring (per world).** Build `P_w` (Q1) → `truth_sig` → `kref = estimate_kappa_reference(...)` **once** → pass `kref['kappa']` to `hg_decompose`, `hg_bootstrap_ci`, and (with `kref['ref_wrong_dist']`) `hg_null_pvalue`, then run `hg_kappa_sensitivity`. The reference population is regenerated per world and discarded; it is the *only* κ source.
**Falls through to NC2** (honest null) if the bootstrap CI on `δ` straddles 0 at every world — REQ-2 cannot adjudicate and needs redesign, exactly as `RDR §5` specifies.

---

## Q3: Minimal-pair certification (REQ-3) — count & matching proof

**Options for count:** (A) 2 pairs/family (min for a paired sign test), (B) **3 pairs/family**, (C) 4 pairs/family (hits the §3 cap of 8 with no room for singletons).

**Recommendation: 3 matched pairs per family = 6 instances/family = 18 instances total** (inside the SPEC §3 `15–20` band and the `≤8` per-family cap). Construct pairs by the **shared-input-states** method below; certification is by **construction-invariant equality of the input-state column plus identical public surface**.

**Rationale:** 3 pairs gives `H-RESP` a within-family paired comparison with `n = 9` pair-instances across families — enough for a paired sign/Wilcoxon test at the family level while staying within the benchmark size band. The shared-input-states construction makes the surface-match certificate *exact and trivially auditable* rather than a fragile statistical near-match.

**Construction (the key idea).** Build member B from member A by holding the generator config fixed (same `primitive_glossary`, same `intervention_api`, same family/difficulty, same grid/particle/sequence shape, **same set of input states**) and swapping only `hidden_rule_fn` to a different rule from the family's rule bank. Regenerate B's `train_obs`/`test_obs` by executing B's rule on **the identical `state` values used in A**. Result: the `state` column is byte-identical across the pair; only `next_state` differs — and `next_state` difference is precisely the hidden rule, the thing a grounded agent must track and a homogenized agent ignores.

**Certification criterion ("identical public surface"), evaluator-side only:**
```python
# worlds/minimal_pairs.py (new, ~60 lines)
def certify_pair(A: dict, B: dict) -> bool:
    assert A['family'] == B['family'] and A['difficulty'] == B['difficulty']
    assert A['primitive_glossary'] == B['primitive_glossary']
    assert A['intervention_api'] == B['intervention_api']
    states_A = [o['state'] for o in A['train_obs']]
    states_B = [o['state'] for o in B['train_obs']]
    assert states_A == states_B                       # identical input-state column (exact)
    assert _shape_signature(A) == _shape_signature(B) # grid dims / particle count / seq-length profile
    # NOTE: next_state intentionally EXCLUDED from the match test — that is where the rule lives.
    return True
```
For the (rare) case where exact state-sharing is undesirable, fall back to a **surface-only two-sample test**: train a logistic classifier on *state-marginal features only* (value histograms, shape stats — never `next_state`) to predict member A vs B; certify matched iff AUC ≤ 0.55 (permutation `p > 0.05`).

**No leakage to the agent.** The agent is run via `run_agent_on_instance(instance, ...)` on **one member at a time**; it receives only that member's public fields. The pair linkage lives in evaluator-only metadata (`pair_id`), never in the instance dict the agent sees. `hidden_rule_fn` is already evaluator-only inside `WorldHarness` (executed in `self._hidden_ns`, never returned by any tool except the existing C-retrieval control). `certify_pair` runs offline at benchmark-build time; its inputs are the full instances but its *logic touches only public surface + the input-state column*, so certification cannot itself become a leak channel.

---

## Q4: Model set & statistical power (REQ-4)

**Candidate third models:**
- **GPT-4o-class (OpenAI), via the available `openrouter` MCP** — frontier, *independent pretraining lineage* from Qwen (Alibaba) and Claude (Anthropic).
- **Gemini-class (Google), via OpenRouter** — also independent-lab frontier; natural backup.
- **An open mid-size model (Llama-3.x-70B / DeepSeek)** — cheapest, but materially weaker.

**Recommendation: add a GPT-4o-class model from OpenAI (OpenRouter), with Gemini as the documented backup; reject the open mid-size model as the *primary* third.**

**Rationale:** the cross-model homogenization claim (REQ-4, SC1) is *strongest* when the converging models come from maximally independent pretraining pipelines — if Qwen, Claude, and GPT all collapse to the same prior-driven hypothesis under intervention OFF, shared-prior homogenization is far more compelling than if two same-family checkpoints agree. An under-powered open model would confound "homogenization" with "incompetence" (low `g_r` for the wrong reason), polluting both `H` and `G`. OpenRouter is already wired in this environment, minimizing the W3 integration delta.

**Power analysis (rough).** Unit of analysis for SC1 = world (instance), with `G`/`H` estimated per `(model × intervention × world)` cell from `N` seed runs. The interaction is the within-world paired contrast `Δδ = δ_ON - δ_OFF`, tested across the 18 worlds (worlds as the replication unit), per model and pooled.

- *Worlds (replications):* a paired t/Wilcoxon over 18 worlds detects a **large** effect (Cohen's `d ≈ 0.8`) at `α = 0.05`, two-sided, with **power ≈ 0.85**; `d = 1.0` → power ≈ 0.97. A large effect is the right design assumption given Arm A/B score **0.000 on CA** (`arm_c/research_memo.md §1`) — the hoped intervention lift is not subtle. 18 worlds is therefore adequate; the binding constraint is seed `N`, not world count.
- *Seeds per cell (estimation stability of G/H):* `H`/`G` are aggregates over `N(N-1)/2` run-pairs. To keep the per-cell SE of an agreement proportion ≤ 0.05 you want ≥ ~45 pairs → **`N ≥ 10`**. **Recommend `N = 15`** (105 pairs/cell, SE ≈ 0.03) where budget allows; **hard floor `N = 10`**.
- *Total run matrix:* `models(3) × intervention(2, or 3 if Q5 Option B added) × worlds(18) × seeds(15) = 1620` runs (`= 1080` at `N = 10`; `= 2430` if the Q5 fixed-schedule arm is included at the 2-condition→3-condition expansion). All raw runs logged (REQ-4) so `H`, `G`, world-responsiveness are recomputable offline.

---

## Q5: Intervention OFF condition (REQ-2, SC1, SC3)

**Options:**
- **Option A — passive observation only.** Agent receives `train_obs`, never touches `intervention_api`, submits a hypothesis.
- **Option B — fixed non-adaptive action schedule.** Agent issues a pre-declared, observation-independent set of intervention actions (e.g. 3 fixed `intervene` calls with canned diagnostic states), then submits — extra data, but **no adaptivity**.

**Recommendation: primary OFF = Option A (passive). Add Option B as a *secondary* "active-but-non-adaptive" control where budget permits.**

**Rationale & tradeoffs.** H-MAIN attributes the regime flip to *world feedback used adaptively* to escape the shared prior. The maximal, conceptually-cleanest contrast to adaptive intervention is *no intervention channel at all* — Option A — which is also the direct operationalization of "prior-only vs world-informed" in `RDR §0`. Its **interpretive risk** is a single confound: ON gets *more data* than A, so a skeptic can attribute any lift to data quantity rather than abductive querying. Option B closes exactly that gap: it grants the same *quantity* of extra observations without *adaptive selection*, so:
- `ON − A` measures the **total** intervention effect (channel present vs absent),
- `ON − B` isolates **adaptivity** (same data budget, adaptive vs fixed),
- `B − A` isolates **data quantity** (extra fixed observations vs none).

Option B alone is a worse primary null: a fixed schedule still partially probes the world and can leak grounding (raising `G` under "OFF"), which would spuriously trigger the H-NULL falsifier (`RDR §3`). Therefore A is the clean null that makes SC3 (a family showing high-H/low-G under OFF) detectable, while B is the *mechanism-disambiguating* third arm, not the null. Implementation: both are thin agent wrappers around `run_agent_on_instance`; Option A simply never calls `harness.intervene`; Option B issues a per-family constant action list independent of observations.

---

## Q6: Relationship to existing arms — **[User決定待ち]**

> **This is a User Approval Gate. Both options are presented with tradeoffs; neither is confirmed. The User decides.**

**Option A — New Arm E (separate experiment).** A single uniform agent loop with its own model adapters (Qwen, Claude, +GPT/Gemini), one controlled prompt scaffold, one intervention budget, and the `{ON, A_passive, [B_fixed]}` switch as a parameter.
- *Pros:* the only varying factors are `model` and `intervention condition` → a **balanced, fully-crossed run matrix** by construction, so the SC1 interaction is cleanly attributable. Maximal isolation; matches REQ-4's "matrix" framing directly.
- *Cons:* largest implementation delta — a new agent loop plus 3 model adapters and the OFF/ON protocol.
- *W3 logging shape:* one uniform record schema, balanced across every `(world × pair_id × model × condition × seed)` cell:
  ```
  run_record = {world_id, pair_id, family, difficulty, model, condition,
                seed, hypothesis_source, probe_signature, accuracy_test, g_r,
                n_interventions, interaction_log}
  ```
  Fully crossed ⇒ `hg_decompose`/`Δδ` computable with no alignment step.

**Option B — Measurement layer wrapping Arms A–D.** Add probe-grid evaluation, signature logging, minimal-pair worlds, and post-hoc `H/G` computation onto the *existing* Arm A/D drivers.
- *Pros:* smallest delta — reuse current runs; mostly new offline analysis (`eval/hg.py`, `worlds/probes.py`, `worlds/minimal_pairs.py`) plus a logging hook.
- *Cons:* inherits arm-specific design differences — Arm A (Qwen) and Arm B/D run *different agent loops, prompts, and intervention-usage patterns*. A cross-arm "model" comparison then conflates **model** with **arm-protocol** (Qwen-in-Arm-A vs Claude-in-Arm-B is not a clean model contrast). The ON/OFF intervention switch may not exist symmetrically in each arm (Arm D is passive-by-design; Arm B is intervention-driven), so the interaction may be **partly undefined**.
- *W3 logging shape:* heterogeneous per-arm schemas normalized post-hoc into the common `run_record` view; the matrix is **ragged** (arms ran different instance subsets / seed counts / conditions), needing alignment and likely targeted re-runs to fill empty cells before `Δδ` is estimable. The `model × condition` interaction is **confounded with arm-protocol** and must be reported with that caveat.

**Decision implication summary:** Option A buys a clean, balanced interaction estimate at the cost of a new loop; Option B buys a small delta at the cost of a ragged, protocol-confounded matrix. The choice directly determines whether W3 logging is *balanced-by-construction* (A) or *ragged-and-realigned* (B). **Awaiting User decision before W3 is specified further.**

---

## Build order (W1–W4, per RDR §6) — ready to implement except where Q6 gates W3

- **W1 (REQ-3):** `worlds/minimal_pairs.py` + extend `worlds/gen.py` rule bank to 3 paired rules/family via the shared-input-states construction; `certify_pair` as the gate. → **18 instances (3 pairs × 3 families).**
- **W2 (REQ-1, REQ-2):** `worlds/probes.py` (probe grids + signatures) and `eval/hg.py` (`estimate_kappa_reference` + G/H, null, CI, κ-sensitivity). Pure additions; no `harness.py` change. κ comes only from the independent reference population (Q2).
- **W3 (REQ-4):** run matrix `3 models × {ON, passive(A), [fixed(B)]} × 18 worlds × N(10–15) seeds`, raw-logged. **Exact loop shape gated on Q6.**
- **W4 (REQ-5):** verdict template extending `verdicts/` — accuracy table augmented with per-world `G`, `H`, `δ`, `H>G?` flag **and the mandatory `κ-robust? [Y/N]` column from the Q2 sensitivity sweep** (worlds flagged `AMBIGUOUS` are excluded from the SC1 interaction); cross-model reproducibility printed as a **diagnostic axis** with mandatory `suspected homogenization` flag whenever accuracy is high but `H > G`.

**Completion check:** every REQ-1…REQ-5 item maps to a concrete file/estimator above; an implementer can begin W1, W2, W4 immediately and W3 once the User resolves Q6.
