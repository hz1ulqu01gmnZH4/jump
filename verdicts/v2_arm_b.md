# v2 Arm B Review — Claude Sonnet 4.6 via `claude -p` + jump_v2 skill

## Verdict: CONDITIONAL — APPROVE the SEQ signal; 10/17 of the run is invalid (harness timeouts)

**Headline warning:** per `arm_b/run_log_v2.txt`, 10 of the 17 instances failed with
`Command '[...claude -p ...]' timed out after 300 seconds` on turn 1 (both the initial call
and its retry). The driver then wrote `return state` via the `if not submitted` fallback at
`arm_b/run_arm_b_v2.py:227-230`. **The 6 CA "fallbacks" and 4 of the 5 PT "fallbacks" do not
reflect Claude's abductive behaviour on those instances — the agent never produced a
hypothesis for them.** The 0.451 headline mean is therefore a mixture of 7 measured
instances (all substantive) and 10 infrastructure-failure floors.

## Rubric Scores (v2 5-axis rubric, max 15)
| Axis | Score | Justification |
|------|-------|---------------|
| (a) Prediction accuracy | 2/3 | Mean acc = 0.451; nonzero instances = 9/17 (pt_001, pt_003, pt_004, seq_001–006). Clears the "mean > 0.2 OR 7+/17 nonzero" bar for 2, but not "mean ≥ 0.5 AND 12+/17 nonzero" for 3. |
| (b) Executability | 3/3 | 17/17 executable. Fallback `return state` is syntactically valid Python that runs to completion; the 10 zeros are wrong-prediction, not crash. |
| (c) Abductive form | 1/3 | 7/17 instances (pt_001 + all 6 SEQ) contain substantive rule structure. The other 10 are verbatim `return state` fallbacks inserted by the driver after subprocess timeout — they use **none** of the invented vocabulary. Falls in the 4–7 substantive bucket (= 1). |
| (d) Intervention use | 1/3 | Mean interventions across the 7 non-fallback instances = (0+1+0+0+0+7+0)/7 = **1.14**. 5 of 7 wins used 0 interventions — Claude solved the SEQ rules directly from `get_train_obs`. Only seq_005 (7 interventions) and seq_001 (1) show intervention-driven refinement. Falls in the 1–2 bucket. |
| (e) Control comparison | 3/3 | Overall Δ vs C-induce = +0.324. Per-family: CA Δ=0, **PT Δ = +0.267** (Arm B 0.333 vs C-induce-PT 0.067), **SEQ Δ = +0.694** (Arm B 1.000 vs C-induce-SEQ 0.306). Two families clear Δ ≥ 0.2 — the 3/3 bar. (Caveat: Arm B remains far below C-retrieval = 1.000 overall, driven by the CA=0 result.) |
| **Total** | **10/15** | SEQ is a clean, genuine 1.000. CA and 4/5 PT numbers are driver-timeout artefacts, not measurement. |

## Per-family breakdown vs controls

| Family | n | Arm B mean | Arm A mean | C-random | C-induce | C-retrieval | Δ(B – C-induce) |
|--------|---|-----------|-----------|----------|----------|-------------|-----------------|
| cellular_automata | 6 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 |
| particle_system   | 5 | 0.333 | 0.167 | 0.000 | 0.067 | 1.000 | +0.267 |
| pattern_puzzle    | 6 | 1.000 | 0.833 | 0.028 | 0.306 | 1.000 | +0.694 |
| **overall**       | 17 | **0.451** | 0.343 | 0.010 | 0.127 | 1.000 | +0.324 |

C-induce per-family means computed from `controls_baseline.json`:
`CA = 0/6 = 0.000`, `PT = (0+0+0.333+0+0)/5 = 0.067`, `SEQ = (0.167+0+0+1.0+0.167+0.5)/6 = 0.306`.

## Per-instance table

| instance_id | family | Arm B acc | C-induce | n_int | turns | fallback | notes |
|-------------|--------|-----------|----------|-------|-------|----------|-------|
| world_ca_001 | CA | 0.000 | 0.000 | 0 | 2 | **Y** | timeout turn 1 (+retry) |
| world_ca_002 | CA | 0.000 | 0.000 | 0 | 2 | **Y** | timeout turn 1 (+retry) |
| world_ca_003 | CA | 0.000 | 0.000 | 0 | 2 | **Y** | timeout turn 1 (+retry) |
| world_ca_004 | CA | 0.000 | 0.000 | 0 | 2 | **Y** | timeout turn 1 (+retry) |
| world_ca_005 | CA | 0.000 | 0.000 | 0 | 2 | **Y** | timeout turn 1 (+retry) |
| world_ca_006 | CA | 0.000 | 0.000 | 0 | 2 | **Y** | timeout turn 1 (+retry) |
| world_pt_001 | PT | **1.000** | 0.000 | 0 | 2 | N | **substantive hypothesis; 0-intervention win** |
| world_pt_002 | PT | 0.000 | 0.000 | 0 | 2 | **Y** | timeout turn 1 (+retry) |
| world_pt_003 | PT | 0.333 | 0.333 | 0 | 2 | **Y** | timeout; `return state` coincidentally matches 1/3 (= C-induce) |
| world_pt_004 | PT | 0.333 | 0.000 | 0 | 2 | **Y** | timeout; `return state` coincidentally matches 1/3 |
| world_pt_005 | PT | 0.000 | 0.000 | 0 | 2 | **Y** | timeout turn 1 (+retry) |
| world_seq_001 | SEQ | 1.000 | 0.167 | 1 | 3 | N | prime-indexed XOR-vs-sum branching rule |
| world_seq_002 | SEQ | 1.000 | 0.000 | 0 | 2 | N | 3-row grid rule, no intervention needed |
| world_seq_003 | SEQ | 1.000 | 0.000 | 0 | 2 | N | `digit_sum(n) mod 3` branch; 0 intervention |
| world_seq_004 | SEQ | 1.000 | 1.000 | 0 | 2 | N | **template-solvable — C-induce also 1.000** |
| world_seq_005 | SEQ | 1.000 | 0.167 | 7 | 10 | N | `(state[-1] - state[0]) mod 13` via sweeps |
| world_seq_006 | SEQ | 1.000 | 0.500 | 0 | 2 | N | perfect-square / mod-3 branched rule |

## Verbatim hypothesis excerpts

### Excerpt 1 — SEQ success (world_seq_003, acc=1.000, n_int=0)
```python
def hidden_rule_fn(state):
    n = len(state)
    digit_sum = sum(int(d) for d in str(n))
    last = state[-1]
    second_last = state[-2]
    mod = digit_sum % 3
    if mod == 0:
        new_val = (last + second_last) % 11
    elif mod == 1:
        new_val = (last ^ second_last) % 11
    else:
        new_val = (last * second_last) % 11
    return state + [new_val]
```
Uses the instance's invented vocabulary structurally: `digit_sum(n)` is the hidden rule's
`splid-sum` (the "zark-index clock"), and the three-branch selection over
`splid % 3 ∈ {add, XOR, multiply} mod 11` exactly matches the hidden rule. The agent
produced this from `get_train_obs` alone, with zero interventions. C-induce scored 0 here,
so this is not a template-enumeration win — it is genuine compositional rule construction
from observation. It is marginally *simpler* than Arm A's Qwen solution for this instance
(fewer comments, identical structure).

### Excerpt 2 — PT success (world_pt_001, acc=1.000, n_int=0)
```python
def hidden_rule_fn(state):
    def signed_delta(from_pos, to_pos, size=8):
        d = (to_pos - from_pos) % size
        if d > size // 2:
            d -= size
        return d

    def toroidal_dist(p, q):
        dx = abs(signed_delta(p['x'], q['x']))
        dy = abs(signed_delta(p['y'], q['y']))
        penalty = 0 if p['type'] == q['type'] else 1
        return max(dx, dy) + penalty

    def find_nearest(particle, candidates, target_type):
        others = [c for c in candidates if c['type'] == target_type and c is not particle]
        if not others:
            return None
        return min(others, key=lambda c: (toroidal_dist(particle, c), c['x'], c['y']))
    # ... step_toward, flee_from, and the main loop: ZEX seeks nearest ZEX,
    # GLORP flees nearest ZEX by larger-|delta| axis (x-axis on tie).
```
This is an independent reconstruction of the full `chromodistance` metric (Chebyshev +
type-mismatch penalty), the same-type-seek / cross-type-flee dichotomy, and the axis-of-
larger-|delta| flee direction — all derived from 12 `train_obs` with **zero** interventions.
Arm A's corresponding hypothesis scored only 0.500 on this instance; Arm B's hits 1.000.
This is the cleanest individual piece of evidence in the run that Claude Sonnet 4.6 can
recover a non-standard type-asymmetric particle rule from passive observation of 12 pairs.

### Excerpt 3 — CA fallback (world_ca_005, acc=0.000, n_int=0, fallback=True)
```python
def hidden_rule_fn(state): return state
```
This is **not** Claude's hypothesis. Per `arm_b/run_log_v2.txt` lines 22–26, both the
initial and retry calls to `claude -p` on world_ca_005 timed out after 300 s. The fallback
at `arm_b/run_arm_b_v2.py:227-230` then wrote `return state` via `harness.submit_hypothesis`
with no interventions. Treating this as "Claude submitted a trivial hypothesis" is a
misreading of the data — the agent never got to respond. For comparison, Arm A's
world_ca_005 hypothesis was a 15-line structurally coherent row-parity-selective von
Neumann rule (reproduced in `verdicts/v2_arm_a.md` Excerpt 2); it was wrong but principled.
Arm B produced no hypothesis at all for this instance.

## CA fallback analysis — **root cause is a harness bug, not capability**

Every one of the 10 fallback entries in `results_v2.json` is explained by a single
subprocess-timeout pattern in the run log:

```
world_ca_001 (cellular_automata, medium)...
    [retry turn 1]: Command '['env', '-u', 'CLAUDECODE', '-u', 'ANTHROPIC_API_KEY',
                              'claude', '-p', '--dangerously-skip-permissions',
                              '--output-format', 'stream-json', '--verbose']'
                   timed out after 300 seconds
    [fail turn 1]: Command '[...]' timed out after 300 seconds
 acc=0.000 n_int=0 turns=2
```

This occurred for world_ca_001..006 (all 6 CA), world_pt_002, world_pt_003, world_pt_004,
world_pt_005 — exactly the 10 "fallback" rows in `results_v2.json`. In each case:
- The first `claude -p` subprocess (cwd `/tmp`, no CLAUDE.md loaded) did not produce a
  `result` JSON event within 300 s.
- The driver at `arm_b/run_arm_b_v2.py:194-201` caught the timeout, slept 15 s, retried once.
- The retry also timed out at 300 s.
- The driver `break`ed out of the turn loop with `submitted=False`.
- Line 227 (`if not submitted: harness.submit_hypothesis("def hidden_rule_fn(state): return state")`) wrote the fallback.

Why the diagnosis is **harness bug, not agent-output-broken**:

1. **Same model and skill succeeded on SEQ.** seq_001–006 all completed normally with a
   measured round-trip (some with `n_turns=2`, seq_005 with `n_turns=10` and a mid-run
   timeout on turn 8 that was followed by a valid submit). If the subprocess were
   fundamentally broken, SEQ would also have failed.

2. **pt_001 completed normally with a full 2 kB Python hypothesis.** Particle instances are
   not inherently too large for the 300 s budget — at least one did complete.

3. **The 300 s budget and `--output-format stream-json`.** CA prompts include the full
   `intervention_api` + `primitive_glossary` + (when in iteration) long multi-grid
   `train_obs` (4×4 grids times 12 observations is ~1.5 kB JSON per instance). Combined
   with the v2 skill body and a long reasoning budget, the first turn can exceed 300 s of
   real time on these instances while Claude is still streaming reasoning. The
   `subprocess.run(..., timeout=timeout)` call kills the child at exactly 300 s, discarding
   any partial output even if a final tool call would have arrived shortly.

4. **No Claude 400/401/5xx errors in the log.** If auth or model-side failures were at
   play, they would surface as `claude -p rc=N: stderr` lines (the `RuntimeError` branch at
   `run_arm_b_v2.py:84`). None appear — every failure line is "timed out after 300 seconds".

5. **`cwd="/tmp"` does not help.** The driver sets `cwd="/tmp"` explicitly at line 81 to
   avoid loading `tmux-agents/CLAUDE.md`. This is correct mitigation but insufficient — the
   bottleneck is in-model generation, not prompt ingestion.

**Fix directions for v2-P6** (not blockers for this review, but required before the
headline number is usable):
- Raise the per-call timeout to 600–900 s (the CA prompt + hypothesis generation genuinely
  needs more than 5 minutes of wall clock in some cases).
- Or switch Arm B from `claude -p` subprocess to the Anthropic Python SDK (`driver.py`
  already has this path) so streaming can be consumed as it arrives without a hard clock.
- Alternatively, keep the subprocess approach but use `--max-turns` / shorter reasoning
  budgets for CA-only to force earlier hypothesis submission.
- **Any re-run must exclude the fallback rows from the "mean accuracy" headline** or report
  two numbers: `measured = mean over 7 substantive` and `incl_fallback = mean over 17`.
  Conflating the two mixes a capability measurement with an infrastructure floor.

## Cross-arm comparison (Arm B vs Arm A)

| Family | Arm A acc | Arm A n_int (mean) | Arm B acc | Arm B n_int (mean, non-fallback) | Capability gap |
|--------|-----------|--------------------|-----------|----------------------------------|----------------|
| CA | 0.000 | ~7 per instance | 0.000 | N/A (all timeouts) | **Uninterpretable** — Arm A produced 6 real hypotheses and they were all wrong; Arm B produced 0 real hypotheses. The shared 0.000 is not the same kind of zero. |
| PT | 0.167 | ~7 per instance | 0.333 | 0.0 (1 instance) | Arm B +0.167, driven entirely by pt_001 going 0.5→1.0. But Arm B only measured 1 of 5 PT instances, so the family mean is dominated by timeout floors. |
| SEQ | 0.833 | ~7 per instance | 1.000 | 1.14 | Arm B +0.167. seq_002 is the single gain (Arm A miss → Arm B hit). 5 of 6 SEQ wins used 0 interventions; the "jump" here is closer to stronger passive few-shot induction than to intervention-driven abduction. |

Key qualitative difference: **Arm A used its intervention budget (7 per instance across all
17 instances); Arm B used interventions only when forced (mean 1.14 across the 7 real
runs, 5 of 7 at zero).** For instances both arms solved (SEQ 001, 003, 004, 005, 006), Arm
B reached the correct rule from training observations alone while Arm A reasoned forward
from interventions. This is a different mode of solution — stronger one-shot induction vs
evidence-gathering abduction. Whether this counts as a "jump" in Zahavy's sense depends on
what one is trying to measure:
- If the claim is "the agent is more sample-efficient than a non-LLM induction baseline",
  Arm B strongly supports this (1.000 SEQ with near-zero intervention use vs C-induce 0.306).
- If the claim is "the agent does action-controllable world modelling", Arm B supports
  this only on seq_005 (7 targeted interventions for coefficient isolation) and, vacuously,
  pt_001 (no interventions needed). The other 5 SEQ wins are passive few-shot.

The Arm A result, by contrast, has *more* interventions per correct SEQ answer and
therefore provides clearer evidence of the Zahavy criterion being exercised — even though
Arm A's mean accuracy is lower.

## HITL signal

- **Beats C-random** (0.010): yes, +0.441.
- **Beats C-induce** (0.127): yes, +0.324. (APPROVE threshold under SPEC.md §5.3 primary
  metric would normally fire — but see caveat below.)
- **Matches/exceeds C-retrieval** (1.000): **no**, –0.549. Arm B is limited by the CA=0
  result, which is a harness artefact; however, even if every CA instance had been measured
  cleanly, Arm A's full CA run scored 0.000 with 7-intervention budget, so there is no
  prior evidence that fixing the timeout would make Arm B's CA nonzero.
- **Family-level Zahavy "jump" claim:** only SEQ clearly clears the bar (Δ=+0.694 vs
  C-induce, 5 clean wins out of 6 excluding the template-solvable seq_004). But note 5 of
  those 5 wins used ≤1 intervention, weakening the action-controllability interpretation.
- **Per SPEC.md §5.3** (`APPROVE = mean_accuracy ≥ 0.7 AND beats C-random AND beats
  C-induce`), Arm B's 0.451 does **not** meet the APPROVE mean threshold. It would fall
  into **BORDERLINE** (mean ≥ 0.5 AND beats C-random but not C-induce) if mean ≥ 0.5, but
  0.451 < 0.5, so it is also below BORDERLINE. Strictly, Arm B maps to **REJECT** on the
  SPEC.md primary rubric with the current run. The 5-axis rubric (10/15 = 0.67) and the
  recognition that CA+4PT are infrastructure failures together push me to CONDITIONAL.

**Recommendation:** CONDITIONAL APPROVE, scoped to the SEQ result and the pt_001
intervention-free reconstruction. Do not quote the 0.451 overall figure without the
fallback caveat. Rerun the 10 timed-out instances with a 600–900 s budget or the SDK path
before Arm B is used in any cross-arm or Zahavy-claim argument. After rerun:
- If CA goes to nonzero: real capability lift over Arm A. Worth a separate v3 study.
- If CA stays 0 with real hypotheses: same CA failure as Arm A — capability-bounded, not
  harness-bounded. The Arm B overall would then be `(0 + real_PT + 1.000*6) / 17` with
  re-measured PT.

## Notes for interpretation (not verdict blockers)

1. `world_seq_004` remains template-solvable (Arm B 1.0, C-induce 1.0). Exclude from
   "abduction wins" counts for both arms.
2. pt_003/pt_004 accuracy of 0.333 is a **null-hypothesis coincidence** — `return state`
   happens to match 1/3 of the test observations because the hidden rules there leave some
   particles stationary when their trigger condition is absent. Do not credit Arm B for
   these two cells; they are zero capability signal.
3. Arm B's SEQ 1.000 with 5/6 zero-intervention solutions is the strongest piece of
   evidence in either arm's run that the v2 SEQ family is solvable by strong passive
   induction, not only by intervention. This is a methodological finding worth flagging:
   the SEQ family may not discriminate "abductive jump" from "high-variance
   template-matching" as sharply as the v2 design intended. A SEQ-v2.1 with deliberate
   observation paucity (fewer train_obs, longer state vectors) would tighten this.
