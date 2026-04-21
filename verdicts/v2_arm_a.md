# v2 Arm A Review — Qwen3.6-35B

## Verdict: APPROVE (with noted family-level limitations)

## Rubric Scores
| Axis | Score | Justification |
|------|-------|---------------|
| (a) Prediction accuracy | 1/3 | Mean acc = 0.343 (<0.4) and nonzero rate = 7/17 = 41% (<50%). Fails both thresholds for a 2, but is far above C-random (0.010) and C-induce (0.127). |
| (b) Executability | 3/3 | All 17/17 hypotheses ran to completion and returned a value of the correct shape. `world_ca_002` is a trivial `return state` fallback but is syntactically valid; the scoring is 0.0 because it is wrong, not because it crashed. |
| (c) Abductive form | 3/3 | 16/17 hypotheses contain non-trivial rule structure; the single exception is `ca_002` (fallback flag=True). Successful SEQ hypotheses explicitly engage the instance's invented vocabulary (`ploon-XOR`, `grolp-sum`, `splid-sum`, `blorf-add`, `A-strand`/`B-strand`, `warp-class`, `far-skip`, `thrax-square`, `half-xor`, `shift-merge`) and derive the rule structurally from observations + interventions rather than returning lookup tables. |
| (d) Intervention use | 2/3 | Every instance made 7–8 interventions (≈77–80% of turn budget). In SEQ the intervention data is visibly reused in hypothesis comments (e.g., seq_001 cites `[1,5]->[1,5,4]: new=4`; seq_005 does per-variable sweeps with `state[0]=1 → 7, state[0]=0 → 8`). For CA, interventions were made but did not steer the model to a correct neighborhood rule in any of 6 instances — limited impact on hypothesis quality. Does not clear the "≥8 instances" bar for 3/3. |
| (e) Control comparison | 2/3 | Overall Arm A 0.343 vs C-induce 0.127 (Δ=+0.216, ≥0.1). Per family Δ: CA 0.000, PT +0.100, SEQ +0.527. Only SEQ exceeds C-induce by >0.2, so the "≥2 families" bar for 3/3 is not met. On `seq_004`, C-induce also scores 1.0 — so 1 of the 5 SEQ wins is template-solvable and cannot be claimed as abduction-specific evidence. |
| **Total** | **11/15** | Meaningful abductive signal on SEQ; genuine zero on CA; partial on PT. |

## SEQ Analysis

Per-instance assessment of the 6 pattern_puzzle instances:

- **world_seq_001 (acc=1.000; C-induce=0.167):** The hypothesis derives a prime-indexed branching rule (`new_index` prime ⇒ XOR mod 7, else `(a+b+3)%7`) by first trying XOR, observing it fails at n=4, proposing `grolp-sum`, and unifying the two. The reasoning cites specific intervention observations. This is genuine hypothesis revision under evidence, not retrieval — no canonical mathematical sequence has this prime-indexed branching.
- **world_seq_002 (acc=0.000; C-induce=0.000):** Legitimate hard case. Both Arm A and C-induce fail; Arm A proposes a plausible grid-based row/column rule but at the wrong granularity. Not a fallback (928 chars of actual rule code).
- **world_seq_003 (acc=1.000; C-induce=0.000):** Uses `splid = digit_sum(n)` as a clock, branches over `splid % 3` between add/XOR/multiply mod 11. The invented names (`splid-sum`, `blorf-add`, `blorf-XOR`, `blorf-mul`) are engaged compositionally. C-induce scored 0, so this is not template-retrieval.
- **world_seq_004 (acc=1.000; C-induce=1.000):** **Template-solvable** — C-induce also scores 1.0. Arm A's hypothesis uses correct A-strand/B-strand logic, but this instance cannot be claimed as abduction-specific evidence because an enumerate-templates baseline already wins.
- **world_seq_005 (acc=1.000; C-induce=0.167):** The hypothesis reduces to `new = (state[-1] - state[0]) % 13`. The annotated derivation shows systematic variable-by-variable intervention sweeps (holding two of `state[0..2]` fixed and varying the third) to isolate each coefficient. This is a textbook example of using interventions for causal attribution, and the final rule is simple enough to verify directly.
- **world_seq_006 (acc=1.000; C-induce=0.500):** Agent enumerates three candidate primitives (`half-xor`, `index-sum`, `shift-merge`), tests all of them against training pairs, finds `shift-merge = ((last<<1)&0xF) | (second & 1)` matches all 11 training examples, and submits it. Clean ablation over a candidate space.

**Net SEQ signal:** 4 of 6 SEQ perfect scores (001, 003, 005, 006) meaningfully exceed C-induce; seq_004 is a tie; seq_002 is a legitimate failure. The four "clean" wins show rule structure being *constructed* from observations + interventions, with the invented vocabulary used as honest slot-fillers rather than as decorations on memorised output.

### Verbatim excerpt 1 — SEQ success (world_seq_003, acc=1.000)
```python
def hidden_rule_fn(state):
    n = len(state)
    # splid-sum = sum of decimal digits of index n
    splid = sum(int(d) for d in str(n))
    mod = splid % 3
    last = state[-1]
    second_last = state[-2]
    if mod == 0:
        # blorf-add
        new_val = (last + second_last) % 11
    elif mod == 1:
        # blorf-XOR
        new_val = (last ^ second_last) % 11
    else:  # blorf-mul
        new_val = (last * second_last) % 11
    return state + [new_val]
```
This is a 6-line rule expressed directly in the instance's invented vocabulary. The compositional use of `splid-sum` as a branching clock is not recoverable from surface frequency statistics — C-induce scored 0 here.

## CA Analysis

All 6 `cellular_automata` instances scored 0.000. Failure mode diagnosis:

- **ca_001 (diagonal-tetrad count rule):** Plausible-but-wrong hypothesis (threshold-count over {0,1,2} diagonal neighbors). Executable, uses the invented `PLOON/QUAV/ZORK` naming. Wrong.
- **ca_002:** `return state` fallback (fallback flag=True). Genuine agent failure on this instance.
- **ca_003 (von-Neumann quad, distinct-count branching):** Plausible, executable, but wrong prediction.
- **ca_004 (hexadic 1D neighborhood, current+sum state-table):** Substantive hypothesis with a full state-transition table; wrong.
- **ca_005 (row-parity axis-selective N/S vs E/W neighbors):** Plausible 2-cell structural rule; wrong.
- **ca_006 (coord-class-dependent neighborhood):** Most sophisticated — switches neighborhood by `(r+c)%3`; wrong.

The CA failure is **not** a scoring artifact: C-retrieval scores 1.0 on all 6, so the rules are learnable from the description. It is also not an agent-output-broken failure: 5/6 CA hypotheses are substantive, structured Python that attempt a neighborhood rule. The failure signature is:

- 7 interventions per instance is likely **insufficient** to disambiguate a multi-valued (k=3) CA state transition over a 9-cell (or larger) neighborhood. The rule space is combinatorially large; the agent proposes one plausible neighborhood + count rule per instance and never revisits after a single failing hypothesis probe.
- There is no evidence of CA-specific shortcut exploitation that C-induce could match — C-induce also scores 0.0 on all 6.

So CA=0.000 is evidence of **insufficient exploration budget / planning depth for large discrete rule spaces**, not evidence that the agent is pretending to abduce. It is a real limit on the abduction capability being measured, but it does not contaminate the SEQ signal.

### Verbatim excerpt 2 — CA failure (world_ca_005, acc=0.000)
```python
def hidden_rule_fn(state):
    grid = [row[:] for row in state]
    rows = len(grid); cols = len(grid[0])
    next_grid = [[0]*cols for _ in range(rows)]
    for r in range(rows):
        for c in range(cols):
            cell = grid[r][c]
            if cell == 0:
                next_grid[r][c] = 0
                continue
            if r % 2 == 0:
                left = grid[r][(c-1) % cols]; right = grid[r][(c+1) % cols]
                neighbors = [left, right]
            else:
                up = grid[(r-1) % rows][c]; down = grid[(r+1) % rows][c]
                neighbors = [up, down]
            non_zero_count = sum(1 for n in neighbors if n != 0)
            if non_zero_count == 0:
                next_grid[r][c] = cell
            elif non_zero_count == 1:
                next_grid[r][c] = 2
            elif non_zero_count == 2:
                next_grid[r][c] = 0
    return next_grid
```
Structurally coherent (row-parity-selective von-Neumann axis, nonzero-count branching) but predicts none of the held-out observations. This is what honest abductive failure looks like — not a crash, not a fallback, a wrong but principled guess.

## PT Analysis

- **world_pt_001 (acc=0.500):** Implements a `chromodistance` metric (Chebyshev + cross-type penalty) plus nearest-partner selection, with ZEX moving toward other ZEX and GLORP moving *away* from ZEX. Captures two-type differential-attraction structure. Structurally correct shape but the specific same/other interaction is incomplete.
- **world_pt_002 (acc=0.000):** Proposes an anisotropic `grolp_dist = |dx| + 2|dy|` for cross-type and `|dx|*|dy|+1` for same-type, plus step-toward logic. Genuine hypothesis, wrong rule. Code is 5k chars, non-trivial.
- **world_pt_003 (acc=0.333; C-induce=0.333):** The comments reveal the agent reasoning through swap vs move candidates and settling on "move (-1,-1) mod 8 if any opposite-type within Chebyshev 1". Partially correct — tied with C-induce, so the gain over induce here is zero.
- **world_pt_004 (acc=0.000):** Proposes `PLEKT` moves diagonally +1,+1 and `WOVVE` moves based on nearest-PLEKT Chebyshev distance. Structurally separates the two types; wrong rule.
- **world_pt_005 (acc=0.000):** Plausible but wrong hypothesis (1.5k chars, non-fallback).

PT success is genuine but thin: only pt_001 gives real signal above C-induce (Δ=+0.5), and the 0.167 family mean barely clears C-induce's 0.067.

## Key Finding

**Arm A demonstrates genuine abductive capability on sequence-rule worlds:** 4 of 6 SEQ instances yield executable hypotheses that use the instance's invented primitive vocabulary compositionally (not decoratively), predict held-out observations perfectly, and meaningfully exceed the C-induce template-search baseline — behavior consistent with Zahavy's criterion of generating novel axioms from an action-controllable world model. **However, the capability does not generalise to large discrete state-transition spaces:** all 6 cellular-automata instances fail (0.000) despite C-retrieval confirming the rules are in principle learnable, indicating the positive result is family-bounded and budget-sensitive, not a universal abduction competence.

## Blocker for APPROVE

None — verdict is APPROVE. Notes for interpretation (not blockers):

1. Headline numbers should be reported per-family, not as a single mean. The 0.343 overall mean obscures a 0.833 SEQ result bracketed by a 0.000 CA result.
2. `world_seq_004` (Arm A 1.0, C-induce 1.0) should be excluded from "abduction wins" counts when making the Zahavy-criterion claim, as it is template-solvable.
3. The CA failure suggests the abduction-capability claim should be scoped to rule spaces where 7–8 interventions can plausibly disambiguate; a follow-up with an expanded turn/intervention budget on CA would clarify whether the failure is capability or budget.
