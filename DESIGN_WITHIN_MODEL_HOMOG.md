# Within-Model Homogenization Experiment Protocol (W3 / Arm E)

**Research question.** Does a single LLM, run in N independent sessions on the same world,
collapse to the *same wrong* hypothesis (the "47 effect" — high cross-run agreement that is
**not** explained by both runs being close to truth), and does the active-intervention
condition (`ON`) break that homogenization relative to passive observation?

**Primary endpoint.** `Δδ = δ_ON − δ_passive`, aggregated across worlds, where
`δ = G − H` from `eval/hg.py` (reference-κ estimator, Option A, frozen since GAP2). A
*positive* `Δδ` means intervention shifts a cell away from homogenization (`H>G`, `δ<0`)
toward grounding (`G>H`, `δ>0`).

This protocol is for the **within-model** stage on the local Qwen MoE. The estimator
(`eval/hg.py`), `harness.py`, and `worlds/` are **frozen** (constraints 2 & 5).

---

## 1. World set and N

**Decision: Option B — the 9 A-member worlds** (3 families × 3 pairs, A member only).
`generate_minimal_pairs()` yields 18 instances; filter `pair_member == 'A'` → 9 worlds
spanning `cellular_automata`, `particle_system`, `pattern_puzzle` × 3 difficulty pairs.

**Rationale.**
- Worlds are the *replication unit* for the `Δδ` paired test. 9 worlds is the minimum that
  gives a usable one-sided Wilcoxon (n=9 → exact p can reach <0.05; see Appendix).
- The B member is a *minimal-pair twin* matched on public surface (`certify_pair`); it adds
  almost no independent information for the **within-model** `δ` (same family/difficulty,
  identical train input column). Spending budget on B members instead buys higher N — which
  this protocol does. Cross-arm/minimal-pair contrasts are a W4 concern, not within-model H.
- This halves the matrix vs. all-18, keeping us inside the 8 h cap (constraint 4).

**N (independent sessions per cell): N = 5.**
- Pairs per cell = `N(N−1)/2 = 10` ≥ the N≥3 (pairs≥3) minimum, comfortably.
- `seeds = [0,1,2,3,4]`. Each seed sets the llama-server sampling RNG so sessions diverge;
  identical seed + temp=0 would force H=1 trivially (see §2), so temp>0 + distinct seeds is
  mandatory for an honest H.

**Primary matrix size:** `9 worlds × 2 conditions × 5 seeds = 90 sessions`.

**Wall-time gate (executable, required before committing to N=5).**
Run a **timing pilot** first: 1 world × 2 conditions × 5 seeds = 10 sessions. Let `T_med`
= median session wall time.
- If `90 × T_med ≤ 7.0 h` → proceed with N=5, all 9 worlds.
- Else if `72 × T_med ≤ 7.0 h` → drop to **N=4** (`9×2×4 = 72`, pairs=6).
- Else → drop world count to 6 (2 pairs/family) at N=5 (`60` sessions).
Record the chosen `(n_worlds, N)` in the run report. This keeps constraint 4 hard-satisfied
rather than hoped-for. The timing pilot sessions are reusable as real data (same settings).

The **temperature sweep (§2) is explicitly OUTSIDE the 8 h primary budget** (constraint 4
scopes the cap to "the primary matrix (not the sweep)") and is run as a separate, smaller job.

---

## 2. Sampling specification

**Greedy is disallowed.** temp=0 ⇒ all N sessions identical ⇒ `H=1` by construction and the
question is undefined. Stochastic decoding between sessions is the regime of interest.

**Primary temperature for H measurement: `temp = 1.0`** with `top_k = 20`, `top_p = 0.95`
(the Arm A / Qwen3 thinking-mode defaults). All 90 primary sessions use this single setting.
Rationale: it matches the Arm A operating point (so within-model H is comparable to Arm A
behaviour) and is the highest-diversity recommended setting → the *hardest* regime in which
to still observe homogenization (a conservative, strong test of the 47 effect).

**Temperature sweep (robustness, separate job):**
- Temps: **{0.3, 0.7, 1.0}** (1.0 reuses the primary-matrix cells for that world — do not
  re-run). top_k/top_p held fixed at (20, 0.95).
- Worlds: the **single best-grounded world** = the world with the highest
  `max(G_ON, G_passive)` in the primary matrix (guarantees there is grounding signal to
  perturb). Optionally a second world from a different family if budget remains.
- Cells: `1 world × 2 conditions × {0.3, 0.7} × N=5 = 20 new sessions` (+ the reused 1.0
  cells). ~20 × T_med ≈ 1.5–3 h, run after the primary matrix.
- **Only `temp` is varied.** top_k and top_p are held fixed: varying three coupled diversity
  knobs confounds the H trend and multiplies the matrix. State this explicitly in the report.

**Tool-calling stability.** Arm A (`arm_a/run_arm_a_v3.py`) runs Qwen3.6-35B-A3B in native
thinking mode with `max_tokens=32768`, `tool_choice="auto"`, and a **stuck-reasoning
watchdog** (`STUCK_REASONING_CAP_S = 600`). No temp-dependent tool failure is documented
there, but malformed-JSON tool args are a known failure: Arm A catches `json.JSONDecodeError`
and substitutes an empty args dict (logged), never crashing the loop. The Arm E adapter MUST
reuse both the watchdog and the args-parse fallback (§6). If the sweep shows a tool-call
failure rate >10% of sessions at temp=1.0, footnote it; do not silently drop those sessions
(they become `__fallback__` signatures, see §3).

**Sweep presentation:** a line plot `H(temp)` and `G(temp)` per condition (Figure 2, §5).
Expectation: `H` decreases with temp (more diversity → less convergence); the 47 effect is
most visible at low temp. The ON−passive gap should persist or widen at low temp.

---

## 3. H-MAIN calculation spec

**Per `(world_id, condition, temp)` cell** (frozen `eval/hg.py` API):

```python
# sigs: one signature per session, computed by run_matrix from the submitted hypothesis
sigs = {f'seed_{k}': prediction_signature(hyp_fn_k, probes) for k in range(N)}
# probes + truth_sig + kappa precomputed ONCE per world (reused across condition/temp/seed)
probes    = build_probe_grid(inst, n=100, seed=0)
truth_sig = prediction_signature(truth_fn, probes)
kref      = estimate_kappa_reference(inst, probes, truth_sig, n_ref=100, seed=0)
kappa     = kref['kappa']

result = hg_decompose(sigs, truth_sig, kappa)            # -> G, H, delta, g, n_pairs
sens   = hg_kappa_sensitivity(sigs, truth_sig, kappa, k_alphabet=k_w)   # k_w PER WORLD
ci     = hg_bootstrap_ci(sigs, truth_sig, kappa, B=1000, seed=0)        # per-cell δ CI
pval   = hg_null_pvalue(sigs, truth_sig, kappa, kref['ref_wrong_dist'], B=1000, seed=0)
```

**`k_w` (per-world alphabet cardinality) — fixes a bug.** `run_matrix.py` currently hardcodes
`k_alphabet=3`. Replace with a per-world estimate from the reference pool:
```python
k_w = max(1 + len({v for j in range(n_probes) for v in kref['ref_wrong_dist'][j]}) // n_probes, 2)
```
Simpler and sufficient: `k_w = max over probes i of |set(ref_wrong_dist[i]) ∪ {truth_sig[i]}|`,
clamped to ≥2. This sets `κ_hi = 1/(k_w−1)` correctly per world for the sensitivity sweep.

**Per-world contrast:** `Δδ_w = δ_ON,w − δ_passive,w` at the primary temp (1.0).

**Aggregation across worlds (pre-registered):**
1. **Eligibility filter** (apply BEFORE the test, see §4): keep world `w` iff *both* its ON
   and passive cells are non-`RANDOM` and both are `kappa_robust == True` (not `AMBIGUOUS`).
2. **Primary test:** one-sided **Wilcoxon signed-rank** on `{Δδ_w : w eligible}`,
   H0: median Δδ = 0, H1: median Δδ > 0 (intervention reduces homogenization). Use
   `scipy.stats.wilcoxon(deltas, alternative='greater')`.
3. **Point + interval:** report `mean Δδ` and a **paired bootstrap 95% CI** — resample
   eligible worlds with replacement (B=2000), recompute `mean Δδ`, take 2.5/97.5 percentiles.
4. **Effect size:** report **both** matched-rank-biserial correlation `r_rb` (matched to
   Wilcoxon) and **Cohen's `d_z`** = `mean(Δδ) / sd(Δδ)` (paired).
5. **Secondary:** repeat the Wilcoxon on the *full* 9-world set (no eligibility filter) as a
   sensitivity check; report both. Primary inference = eligible subset.

**Low-G worlds in the test:** worlds with `max(G_ON,G_passive) < 0.3` are *flagged WEAK* and
excluded from the primary Wilcoxon (they enter the secondary full-set test and the §4 rubric).
This prevents "incompetence" from masquerading as homogenization signal.

**What a positive result looks like (continuous, not a hard threshold):**
- Passive cells show `δ_passive < 0` (H>G) for a majority of eligible worlds → the 47 effect
  exists.
- `Δδ_w > 0` for a majority, Wilcoxon one-sided `p < 0.05`, bootstrap CI of `mean Δδ`
  excludes 0, and `d_z` ≥ ~0.5.
- Per-world "homogenization-broken" call: `δ_passive < 0` AND the per-world Δδ_w bootstrap CI
  (from `hg_bootstrap_ci` differenced across conditions) excludes 0.

---

## 4. Low-G interpretation rubric

The local Qwen MoE is mid-range; `g_r` is not always high. Classify **each `(world,
condition)` cell** from `(G, H, δ, kappa_robust)`:

| Class | Criteria | Use in H-MAIN |
|---|---|---|
| **RANDOM** | `G < 0.15` AND `H < 0.15` | Discard cell; if either condition RANDOM, world excluded from primary test |
| **AMBIGUOUS** | `kappa_robust == False` (sign δ flips over κ-sweep) | Excluded from SC1 / primary test (RDR Q2 mandate) |
| **HOMOGENIZED** | `δ < 0` (H>G) robustly AND `H ≥ 0.30` | Core signal; expected in passive |
| **GROUNDED** | `δ > 0` (G>H) robustly AND `G ≥ 0.30` | Expected in ON if intervention helps |
| **WEAK** | `0.15 ≤ G < 0.30`, not clearly HOMOGENIZED | Flag; secondary analysis only |

**Disambiguation rules (mapping to the H-MAIN verdict):**

1. **"Model too weak" vs "world too hard".** Look across worlds *for this model*:
   - `G` low in **all** worlds (both conditions) → **model-capability ceiling**: H-MAIN is
     *inconclusive* for this model; report the ceiling, do not claim homogenization absence.
   - `G` low only in **specific** worlds while high elsewhere → those worlds are **too hard**;
     flag per-world and exclude from the primary test (WEAK/RANDOM), keep the rest.
2. **G low in passive, ↑ in ON** → direct evidence intervention aids grounding → **supports
   H-MAIN** (report as ΔG>0 alongside Δδ).
3. **H high in passive, ↓ in ON even if G stays low** → intervention breaks wrong-answer
   convergence → **supports H-MAIN** (this is captured by Δδ>0 driven by H falling).
4. **G and H both near 0 (RANDOM)** → no signal → **discard** the cell; if both conditions of
   a world are RANDOM, drop the world from the primary test and list it under "no signal".

The verdict integrates: primary test on eligible worlds + an explicit count of
`{HOMOGENIZED, GROUNDED, WEAK, RANDOM, AMBIGUOUS}` cells per condition, and the model-ceiling
check (rule 1) gating whether a null `Δδ` is "intervention doesn't help" vs "model can't ground".

---

## 5. Expected output (tables + figures)

The analyse script (`arm_e/analyse.py`, §6) prints/saves all of the following.

**Table 1 — per-cell results** (one row per `(world_id, condition, temp)`):
```
world_id | family | condition | temp | N | G | H | δ | κ | κ_hi | κ_robust | n_pairs | class | mean g_r
```
`κ_robust` = `sens['kappa_robust']`; `class` = §4 label. Sort by family, world, condition.

**Table 2 — per-world Δδ (primary temp):**
```
world_id | family | δ_passive | δ_ON | Δδ_w | Δδ_w 95% CI | eligible? | per-world call
```
Footer: `n_eligible`, `mean Δδ (eligible)`, paired bootstrap 95% CI, Wilcoxon p (one-sided),
`r_rb`, `d_z`; and the same line for the full-9 secondary set.

**Figure 1 — paired δ slope plot (primary temp).** One panel; x-axis = {passive, ON};
each eligible world = a line connecting its `δ_passive` → `δ_ON` (slope-graph / dumbbell).
Horizontal reference line at `δ=0` (G=H boundary). Color HOMOGENIZED-start worlds distinctly.
A rising population = intervention breaks homogenization. Save `arm_e/figs/fig1_delta_paired.png`.
matplotlib: for each world plot `ax.plot([0,1],[d_pa,d_on], marker='o')`; `ax.axhline(0)`;
`ax.set_xticks([0,1],['passive','ON'])`; `ax.set_ylabel('δ = G − H')`.

**Figure 2 — temperature sweep** (representative world). Two lines (`H`, `G`) vs
`temp ∈ {0.3,0.7,1.0}`, one subplot per condition (passive | ON), shared y. Shows H falling
as temp rises and the ON−passive separation. Save `arm_e/figs/fig2_temp_sweep.png`.

**Figure 3 (optional but recommended) — G vs H scatter**, one point per cell, colored by
condition, with the `G=H` diagonal. Cells below the diagonal (H>G) are homogenized; arrows
passive→ON per world visualize the §3 contrast. Save `arm_e/figs/fig3_GH_scatter.png`.

**Key number block (printed prominently):**
```
H-MAIN (within-model, Qwen3.6-35B-A3B, temp=1.0):
  eligible worlds:        n / 9   (excluded: <list with reason>)
  mean Δδ:                <val>   95% CI [<lo>, <hi>]
  Wilcoxon (one-sided):   p = <val>
  effect size:            d_z = <val>,  r_rb = <val>
  model-ceiling check:    <PASS: grounding present | CEILING: G low in all worlds>
  verdict:                <SUPPORTS / NULL / INCONCLUSIVE-CEILING>
```

---

## 6. Implementation spec

**`arm_e/agent.py` — add a real local-Qwen adapter** (mirror `arm_a/run_arm_a_v3.py`):
- Add `qwen_local` to `ADAPTER_REGISTRY`. The adapter signature must accept the sampling
  params; extend `get_adapter(model_id, seed, temp, top_k, top_p)` and the registry factory.
- Inside: `from openai import OpenAI; client = OpenAI(base_url="http://127.0.0.1:8080/v1",
  api_key="local")`. Call `client.chat.completions.create(model=actual_model,
  messages=messages, tools=TOOLS_SCHEMA_OPENAI, tool_choice="auto", max_tokens=32768,
  temperature=temp, top_p=top_p, seed=seed, timeout=600.0, extra_body={"top_k": top_k})`.
  (`top_k` is not a standard OpenAI field → pass via `extra_body`; llama-server honours it.)
- `actual_model` from `probe_llama_server()` (copy from arm_a) — fail loudly if server down.
- Translate the existing `TOOLS_SCHEMA` (Anthropic-style `input_schema`) into OpenAI
  `{"type":"function","function":{"name","description","parameters"}}` form (one helper).
- Parse `resp.choices[0].message.tool_calls`; on `json.JSONDecodeError` for `tc.function.
  arguments`, substitute `{}` and log (Arm A behaviour). Map to the dicts
  `{"name","input","id"}` that `_execute_tool` already consumes.
- Reuse Arm A's **stuck-reasoning watchdog** (600 s cap) and the `no_tool_call` break.
- Keep `cost_usd = 0.0` (local). `prompt_tokens`/`completion_tokens` from `resp.usage`.
- The agent loop, condition gating (`passive` blocks `intervene`), and `submit_hypothesis`
  scoring in `run_arm_e_agent` are **already correct** — only the adapter is new. Thread
  `temp/top_k/top_p` from `run_arm_e_agent(...)` kwargs into `get_adapter`.

**`arm_e/run_matrix.py` — changes:**
- Add `temps: list` param. `build_matrix` → 5-tuple `(world, model, condition, temp, seed)`.
- `cell_key = (world_id, model_id, condition, temp)`; add `'temp'` to `run_record`.
- **Fix `k_alphabet`:** compute `k_w` per world (§3 formula) from `kref['ref_wrong_dist']`;
  pass to `hg_kappa_sensitivity`. Remove the hardcoded `3`.
- Add `hg_bootstrap_ci` and `hg_null_pvalue` calls in the per-cell fill-in loop; store
  `delta_ci`, `pval` on the records.
- Call `probe_llama_server()` once at start (loud fail if unreachable).
- **Per-model batching (cross-model readiness):** iterate `models` as the OUTERMOST loop so
  all cells of one model finish before the next. CLI: `--models qwen_local`,
  `--worlds <ids>`, `--conditions ON passive`, `--temps 1.0`, `--seeds 0 1 2 3 4`.
- Primary run: `--worlds <9 A-members> --temps 1.0 --seeds 0..4`.
  Sweep run (separate): `--worlds <best world> --temps 0.3 0.7 --seeds 0..4`.

**`arm_e/analyse.py` — add `within_model_report(log_path)`:**
- Build Table 1, Table 2 (per §5), apply the §4 classifier, run the eligibility filter,
  `scipy.stats.wilcoxon(..., alternative='greater')`, paired bootstrap CI, `d_z`, `r_rb`.
- Generate Figures 1–3 with matplotlib (`Agg` backend; save to `arm_e/figs/`).
- Print the Key-number block. Keep the existing `analyse()` for pipeline health.

**Run commands** (constraint 1 & 4): all via `uv run python -m arm_e.run_matrix ...` from
`~/projects/jump`; no paid API. Timing pilot first (§1), then primary matrix, then sweep,
then `uv run python -c "from arm_e.analyse import within_model_report; within_model_report('<log>')"`.

### Cross-model extension note (design-mention only — not this task's scope)
Because `cell_key` already carries `model_id`, the within-model H-MAIN logic needs **no
redesign** for cross-model. Procedure: (1) run the full matrix with `--models qwen_local`;
(2) **swap the llama-server** to `gemma-4-26B-A4B-it-MXFP4_MOE.gguf` (RTX 5090 32 GB holds
only one large model — sequential, not concurrent); (3) re-run with `--models gemma_local`
(a second adapter identical to `qwen_local` save the `actual_model` string). Each model's
within-model `Δδ` is computed independently with this exact protocol. The *cross-model* H
(pooling sigs across models within a `(world, condition)` cell) is a strictly additive
grouping in `analyse.py`, deferred to W-cross. No `gemma_local` 3rd lineage / extra download
is in scope.

---

## Appendix: Statistical power sketch

**Replication units.** Primary test = paired Wilcoxon over worlds; `n_eligible ≤ 9` (likely
6–8 after RANDOM/AMBIGUOUS/WEAK exclusions). Within each cell, `δ` is estimated from
`pairs = N(N−1)/2 = 10` (N=5) run-pairs; per-cell δ SE is reported via `hg_bootstrap_ci`.

**Wilcoxon power (worlds as units).** One-sided signed-rank at α=0.05:
- n=9: smallest p attainable ≈ 0.002 (all same sign) → can detect a *consistent* effect; for
  a large paired effect (`d_z ≈ 1.0`) power ≈ 0.8; for medium (`d_z ≈ 0.5`) power ≈ 0.4–0.5.
- n=6: only a near-unanimous sign pattern reaches p<0.05 → detects only large, consistent Δδ.

**Implication.** This within-model stage is **pilot-scale**: it is powered to detect a
*strong, consistent* anti-homogenization effect (the regime the 47 effect predicts), not a
subtle one. A null result is reported as "no strong within-model effect at this N", and the
**model-ceiling check** (§4 rule 1) gates whether a null means "intervention ineffective" vs
"model cannot ground". Power is augmented in the cross-model stage (more model×world cells)
and by the temp sweep (low temp amplifies H). Per-cell N=5 keeps δ noise bounded
(bootstrap CI reported) while staying inside the 8 h wall-clock cap.

**Budget arithmetic.** Primary = 90 sessions; at the task's stated 5–10 min/session the
timing gate (§1) forces N or world-count down if 90·T_med would exceed 7 h, so constraint 4
holds by construction. Sweep (~20 sessions) runs outside the primary cap.
