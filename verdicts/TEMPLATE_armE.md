# v2 Arm E Verdict — [Condition: ON / passive / fixed]

<!-- Copy to v2_armE_<condition>.md and fill placeholders before filing. -->

## Verdict: [APPROVE / CONDITIONAL / REJECT]

## Rubric Scores

| Axis | Score | Justification |
|------|-------|---------------|
| (a) Prediction accuracy | ?/3 | [Fill at review time: 3=mean≥0.5 AND 12+/18 nonzero; 2=mean>0.2 OR 7+/18 nonzero; 1=nonzero above C-random; 0=at or below C-random] |
| (b) Executability | ?/3 | [Fill: 3=all 18 ran to completion; 2=≤2 crashes; 1=≤5 crashes; 0=>5 crashes] |
| (c) Abductive form | ?/3 | [Fill: 3=≥14/18 substantive; 2=10–13; 1=4–9; 0=≤3 substantive hypotheses] |
| (d) Intervention use | ?/3 | [Fill: 3=≥8 instances show intervention-driven refinement; 2=4–7; 1=1–3; 0=none] |
| (e) Control comparison | ?/3 | [Fill: 3=≥2 families beat C-induce by Δ≥0.2; 2=1 family; 1=overall Δ>0.1; 0=at or below C-induce] |
| **(f) Homogenization detection** | ?/3 | [Fill at review time: 3=H>G confirmed across ≥2 model pairs + ≥2 families + CI excludes 0; 2=partial (1 pair or 1 family); 1=signal present but not significant (CI straddles 0); 0=H≤G everywhere] |
| **Total** | **?/18** | |

---

## Per-world accuracy and homogenization metrics

| World | family | accuracy | G | H | δ=G−H | H>G? | C-retrieval |
|-------|--------|----------|---|---|--------|------|-------------|
| world_mp_ca_001_A | cellular_automata | 0.XX | 0.XX | 0.XX | 0.XX | No | 1.00 |
| world_mp_ca_001_B | cellular_automata | 0.XX | 0.XX | 0.XX | 0.XX | No | 1.00 |
| world_mp_ca_002_A | cellular_automata | 0.XX | 0.XX | 0.XX | 0.XX | No | 1.00 |
| world_mp_ca_002_B | cellular_automata | 0.XX | 0.XX | 0.XX | 0.XX | No | 1.00 |
| world_mp_ca_003_A | cellular_automata | 0.XX | 0.XX | 0.XX | 0.XX | No | 1.00 |
| world_mp_ca_003_B | cellular_automata | 0.XX | 0.XX | 0.XX | 0.XX | No | 1.00 |
| world_mp_pt_001_A | particle_system   | 0.XX | 0.XX | 0.XX | 0.XX | No | 1.00 |
| world_mp_pt_001_B | particle_system   | 0.XX | 0.XX | 0.XX | 0.XX | No | 1.00 |
| world_mp_pt_002_A | particle_system   | 0.XX | 0.XX | 0.XX | 0.XX | No | 1.00 |
| world_mp_pt_002_B | particle_system   | 0.XX | 0.XX | 0.XX | 0.XX | No | 1.00 |
| world_mp_pt_003_A | particle_system   | 0.XX | 0.XX | 0.XX | 0.XX | No | 1.00 |
| world_mp_pt_003_B | particle_system   | 0.XX | 0.XX | 0.XX | 0.XX | No | 1.00 |
| world_mp_seq_001_A | pattern_puzzle   | 0.XX | 0.XX | 0.XX | 0.XX | No | 1.00 |
| world_mp_seq_001_B | pattern_puzzle   | 0.XX | 0.XX | 0.XX | 0.XX | No | 1.00 |
| world_mp_seq_002_A | pattern_puzzle   | 0.XX | 0.XX | 0.XX | 0.XX | No | 1.00 |
| world_mp_seq_002_B | pattern_puzzle   | 0.XX | 0.XX | 0.XX | 0.XX | No | 1.00 |
| world_mp_seq_003_A | pattern_puzzle   | 0.XX | 0.XX | 0.XX | 0.XX | No | 1.00 |
| world_mp_seq_003_B | pattern_puzzle   | 0.XX | 0.XX | 0.XX | 0.XX | No | 1.00 |
| **Mean** | | **0.XX** | **0.XX** | **0.XX** | **0.XX** | | **1.00** |

Notes:
- G, H, δ are per-world aggregates from `eval.hg.hg_decompose()` across all N seeds for that world.
- `H>G?` is "Yes" if H > G (suspected homogenization on this world), "No" otherwise.
- C-retrieval is always 1.00 by construction (the rule is given verbatim in description).
- All 0.XX values are placeholders — fill from actual run data at review time.

---

## Homogenization diagnostic

**MANDATORY: this section must be present in every Arm E verdict. If H/G data is not yet
available (e.g. partial run), write `NOT_YET_COMPUTED` in the table body and flag it.**

| world_id | condition | G | H | δ | H>G? | suspected_homogenization |
|----------|-----------|---|---|---|------|--------------------------|
| world_mp_ca_001_A | [ON/passive] | 0.XX | 0.XX | 0.XX | No | False |
| world_mp_ca_001_B | [ON/passive] | 0.XX | 0.XX | 0.XX | No | False |
| world_mp_ca_002_A | [ON/passive] | 0.XX | 0.XX | 0.XX | No | False |
| world_mp_ca_002_B | [ON/passive] | 0.XX | 0.XX | 0.XX | No | False |
| world_mp_ca_003_A | [ON/passive] | 0.XX | 0.XX | 0.XX | No | False |
| world_mp_ca_003_B | [ON/passive] | 0.XX | 0.XX | 0.XX | No | False |
| world_mp_pt_001_A | [ON/passive] | 0.XX | 0.XX | 0.XX | No | False |
| world_mp_pt_001_B | [ON/passive] | 0.XX | 0.XX | 0.XX | No | False |
| world_mp_pt_002_A | [ON/passive] | 0.XX | 0.XX | 0.XX | No | False |
| world_mp_pt_002_B | [ON/passive] | 0.XX | 0.XX | 0.XX | No | False |
| world_mp_pt_003_A | [ON/passive] | 0.XX | 0.XX | 0.XX | No | False |
| world_mp_pt_003_B | [ON/passive] | 0.XX | 0.XX | 0.XX | No | False |
| world_mp_seq_001_A | [ON/passive] | 0.XX | 0.XX | 0.XX | No | False |
| world_mp_seq_001_B | [ON/passive] | 0.XX | 0.XX | 0.XX | No | False |
| world_mp_seq_002_A | [ON/passive] | 0.XX | 0.XX | 0.XX | No | False |
| world_mp_seq_002_B | [ON/passive] | 0.XX | 0.XX | 0.XX | No | False |
| world_mp_seq_003_A | [ON/passive] | 0.XX | 0.XX | 0.XX | No | False |
| world_mp_seq_003_B | [ON/passive] | 0.XX | 0.XX | 0.XX | No | False |

**suspected_homogenization** is True when:
- accuracy is high (>0.5) AND H > G (excess cross-run agreement beyond truth-explained)
- This means models converge on a shared non-truth hypothesis — the "47 regime"

**Rule:** This section MUST be present in every Arm E verdict. If H/G data is not yet
available (e.g. partial run), write `NOT_YET_COMPUTED` and flag it.

---

## Cross-model reproducibility (SC1 diagnostic)

Cross-model Δδ = δ_ON − δ_OFF: positive means intervention reduced homogenization.
H-MAIN is supported when Δδ > 0 and CI excludes 0 across ≥2 model pairs and ≥2 families.

| Model pair | Δδ = δ_ON − δ_OFF | CI (95%) | H-MAIN? |
|------------|-------------------|----------|---------|
| (Qwen, Claude) | 0.XX | (lo, hi) | Pending |
| (Qwen, GPT-4o) | 0.XX | (lo, hi) | Pending |
| (Claude, GPT-4o) | 0.XX | (lo, hi) | Pending |

Per-family breakdown:

| Family | Model pair | Δδ | CI (95%) | H-MAIN? |
|--------|------------|-----|----------|---------|
| cellular_automata | (Qwen, Claude) | 0.XX | (lo, hi) | Pending |
| cellular_automata | (Qwen, GPT-4o) | 0.XX | (lo, hi) | Pending |
| cellular_automata | (Claude, GPT-4o) | 0.XX | (lo, hi) | Pending |
| particle_system   | (Qwen, Claude) | 0.XX | (lo, hi) | Pending |
| particle_system   | (Qwen, GPT-4o) | 0.XX | (lo, hi) | Pending |
| particle_system   | (Claude, GPT-4o) | 0.XX | (lo, hi) | Pending |
| pattern_puzzle    | (Qwen, Claude) | 0.XX | (lo, hi) | Pending |
| pattern_puzzle    | (Qwen, GPT-4o) | 0.XX | (lo, hi) | Pending |
| pattern_puzzle    | (Claude, GPT-4o) | 0.XX | (lo, hi) | Pending |

H-MAIN criterion: Δδ > 0 AND CI excludes 0 for ≥2 model pairs AND ≥2 families.

---

## Per-family summary vs controls

| Family | n | Arm E acc | C-random | C-induce | C-retrieval | Δ(E – C-induce) |
|--------|---|-----------|----------|----------|-------------|-----------------|
| cellular_automata | 6 | 0.XX | 0.000 | 0.000 | 1.000 | 0.XX |
| particle_system   | 6 | 0.XX | 0.000 | 0.067 | 1.000 | 0.XX |
| pattern_puzzle    | 6 | 0.XX | 0.028 | 0.306 | 1.000 | 0.XX |
| **overall**       | 18 | **0.XX** | 0.010 | 0.127 | 1.000 | **0.XX** |

---

## Honest null conditions (fill if applicable)

- **NC1 (H-NULL):** If G is already high without intervention, state: "Intervention is not
  load-bearing for family X. G_{passive} = 0.XX > threshold." Flag and do not claim H-MAIN.
- **NC2 (estimator insufficient):** If bootstrap CI on δ straddles 0 at every world, state:
  "H/G estimator cannot adjudicate with current N. REQ-2 needs redesign or larger N."

---

## Notes for interpretation

- [Fill at review time]
