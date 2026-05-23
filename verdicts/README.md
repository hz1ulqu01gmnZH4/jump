# verdicts/

- `v2_arm_a.md` — Arm A (Qwen3.6-35B) verdict
- `v2_arm_b.md` — Arm B (Claude Sonnet 4.6 via `claude -p`) verdict
- `TEMPLATE_armE.md` — Template for Arm E (anti-homogenization) verdicts

## Using TEMPLATE_armE.md

1. Copy to `v2_armE_<condition>.md` (e.g. `v2_armE_ON.md`, `v2_armE_passive.md`)
2. Fill in the per-world table with real G/H/δ values from `eval.hg.hg_decompose()`
3. Mark `suspected_homogenization = True/False` per world (True when accuracy > 0.5 AND H > G)
4. Fill cross-model Δδ table once all 3 models have completed their runs
5. Score rubric axis (f) based on whether H-MAIN criterion is met

## Fields

- **G** (Grounded): mean expected cross-run agreement explained by shared closeness to truth.
  Two runs both getting the right answer → they agree; G counts this as grounding.
  Formally: `G = mean_{r<r'} Ê(r,r')` where `Ê = g_r·g_{r'} + (1-g_r)(1-g_{r'})·κ`.

- **H** (Homogenized): mean excess cross-run agreement beyond what truth explains.
  Two runs agreeing with each other but not with the world → homogenization.
  Formally: `H = mean_{r<r'} max(a(r,r') - Ê(r,r'), 0)`.

- **δ = G − H**: positive → run population is grounded; negative → homogenized.
  CI excludes 0 → statistically significant grounding or homogenization signal.

- **Δδ** (SC1 interaction): `δ_ON − δ_OFF`; positive means intervention reduced
  homogenization (more grounding under ON than passive). This is the H-MAIN effect.
  Computed per model pair and per family; CI from run-bootstrap (B=1000).

- **κ** (kappa): probability two independent wrong outputs coincide by chance.
  Estimated from the empirical wrong-output distribution per probe; used in `Ê`.

- **suspected_homogenization**: True when accuracy > 0.5 AND H > G for a world.
  Signals the "47 regime": models converge on a shared wrong hypothesis. A high
  accuracy result with suspected_homogenization=True must NOT be recorded as a jump.

## H-MAIN criterion (SC1)

H-MAIN is supported when, across the full run matrix:
- Δδ > 0 (intervention ON raises grounding vs OFF baseline)
- CI (95%) on Δδ excludes 0
- This holds for ≥2 model pairs (e.g. Qwen/Claude, Qwen/GPT-4o)
- AND ≥2 world families (cellular_automata, particle_system, pattern_puzzle)

Failing this criterion is an honest null result (NC1 or NC2 in RDR §5), not a failure of the
experiment — it means intervention is not the load-bearing mechanism, or the estimator lacks
power.

## Relationship to RDR.md

- REQ-2 → G/H decomposition (from `eval.hg.hg_decompose()`)
- REQ-5 → mandatory `suspected_homogenization` flag + axis (f) in every Arm E verdict
- SC1 → cross-model Δδ table
- NC2 → honest null condition (report if CI straddles 0 everywhere)
