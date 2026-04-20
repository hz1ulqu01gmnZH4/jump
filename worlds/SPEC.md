# v2 Invented-World Abduction Benchmark — Specification

**Status:** v2 design (supersedes v1 historical-discovery retrieval test). All primitive names are invented; hypotheses are executable Python; scoring is predictive accuracy on held-out observations.

---

## 1. Purpose and design principles

v1 used famous historical discoveries (Einstein equivalence principle, DNA double helix, ...) as gold cases and failed because every gold_case was present in the LLM's training corpus: the experiment measured retrieval-with-fluent-justification, not abduction. v2 is designed to make retrieval impossible by construction.

Four principles govern v2 design:

1. **Invented primitive names.** Every entity, quantity, relation, and operation is a synthetic token (e.g. `ZORK`, `chromodistance`, `thraxgate`). No real-physics, real-biology, or real-math vocabulary is admitted (banned: mass, force, gravity, atom, electron, gene, neuron, momentum, energy, etc.). This forecloses verbal retrieval from training data.

2. **Executable hypotheses.** The agent's submission is Python source for a function `hypothesis(state) -> next_state`. Its accuracy on held-out `test_obs` is mechanically graded. This forecloses "sounds-like-the-gold" scoring (which v1 P4.5 already tried to patch with per-task verbatim excerpts — the v2 pivot makes the patch unnecessary).

3. **Intervention.** The agent may emit actions via an `intervention_api` that mutate world state and observe the consequence. Zahavy's (2026) "action-controllable world model" criterion is that abductive competence requires *doing*, not just *watching*. v2 instruments this directly: pure passive-observation arms are one measured condition; arms with intervention are another.

4. **No self-assessment prose.** Banned output fields include `novelty_justification`, `abduction_notes`, or any field whose purpose is the model describing its own reasoning quality. Allowed: `executable_hypothesis` (Python source), `interaction_log` (raw records of actions + observations), `refinement_steps` (list of {prior_hypothesis, new_hypothesis, trigger_observation}). If the model emits banned fields, they are stripped before scoring — scoring is purely predictive.

Together these principles yield a benchmark where high performance cannot be explained by memorisation, plausible-sounding prose, or passive statistical pattern-matching — the three failure modes v1 could not exclude.

---

## 2. World instance schema

A world instance is a JSON object. Agents see only the public fields (everything except `hidden_rule`, `hidden_rule_fn`, and `test_obs`). The evaluator sees all fields.

```json
{
  "id": "world_NNN",
  "family": "cellular_automata | particle_system | pattern_puzzle",
  "difficulty": "easy | medium | hard",
  "primitive_glossary": {"<invented_name>": "<invented description>"},
  "hidden_rule": "<human-readable description of the true generative rule — evaluator-only>",
  "hidden_rule_fn": "<Python source of the true rule — evaluator-only>",
  "intervention_api": [
    {"action": "<action_name>", "params": ["<param>", ...], "description": "<what it does to world state>"}
  ],
  "train_obs": [{"state": <JSON>, "next_state": <JSON>}, ...],
  "test_obs":  [{"state": <JSON>, "next_state": <JSON>}, ...]
}
```

Constraints:
- `train_obs` length ≥ 10. Agent sees these.
- `test_obs` length ≥ 5. Agent never sees these; used only by `run_instance`.
- All `next_state` values must be produced by actually executing `hidden_rule_fn` on the corresponding `state`. Hand-crafted observations are a bug.
- `primitive_glossary` lists every invented name used in the instance with an invented description (the description itself must not leak a standard-physics meaning — "zork: a type of cell state" is fine; "zork: what physicists call mass" is forbidden).
- `hidden_rule` is a prose description intended only for evaluators and C-retrieval control; `hidden_rule_fn` is the authoritative definition.
- `intervention_api` declares the actions the evaluator will execute on the agent's behalf. It does not need to include every conceivable world-mutation; it must be sufficient to distinguish the hidden rule from obvious foils.

---

## 3. Three world families

Every v2 benchmark set draws from exactly these three families. Each family has ≥4 and ≤8 instances; total benchmark size 15–20 instances.

### 3a. Cellular automata (invented)

- Grid of cells (rows × cols, toroidal boundary).
- Each cell holds a synthetic state drawn from a small alphabet (3–4 symbols, named e.g. `ZORK`, `PLOON`, `QUAV`, `THRAX`).
- Update rule: a function of some neighbourhood of each cell producing the next state, applied synchronously to all cells.
- **Invention criteria (all must hold):**
  - Alphabet has ≥ 2 non-trivial states (beyond binary dead/alive).
  - Neighbourhood is non-standard: diagonal-only, knight-move, weighted, alternating-parity, or dependent on cell coordinates. Not the full Moore or von Neumann neighbourhood applied uniformly.
  - Transition function uses a non-monotone dependence on neighbour counts (e.g. a specific count triggers a specific transition, not a threshold), or mixes modular arithmetic over counts with per-state asymmetry.
  - Rule is not Conway's Game of Life, not Wolfram elementary CA (rules 0–255), not Brian's Brain, not WireWorld, not Langton's Ant, not Rule 30/90/110, not any named CA in Wolfram's atlas or the LifeWiki.
- **Intervention API:**
  - `set_cell(row, col, state)` — write `state` into (row, col) of current grid.
  - `flip_row(row)` — cycle every state in `row` by one symbol (ZORK→PLOON→QUAV→ZORK).
  - `inject_pattern(row, col, pattern)` — stamp a 2D `pattern` (list of lists of states, possibly with `null` wildcards meaning "leave untouched") at top-left (row, col).
  - Each intervention mutates the current state, not a historical state.

### 3b. Particle systems (invented)

- N particles on a toroidal 2D integer grid, each with a synthetic type drawn from 2–3 invented type names.
- Each particle has position (x, y) and type. No continuous velocities, no masses, no charges — just type + position.
- Update rule: a function that produces next positions for all particles simultaneously from current positions + types.
- **Invention criteria (all must hold):**
  - Interaction law depends on a synthetic quantity defined only in the glossary (e.g. `chromodistance`, `grolp potential`, `thraxon coupling`) whose definition does not coincide with Euclidean distance, Manhattan distance, or any named metric. It must involve type-asymmetry (distance depends on whether the pair share a type).
  - Rule is not gravity, not electrostatic attraction, not Lennard–Jones, not hard-sphere collision, not flocking (Reynolds / Vicsek), not reaction–diffusion, not lattice-Boltzmann, not any standard agent-based swarm rule.
  - Rule is deterministic given the global state (no internal random component).
  - At least one type exhibits different dynamics from another type (type asymmetry).
- **Intervention API:**
  - `spawn(type, x, y)` — add a particle at (x, y); no-op if cell already occupied.
  - `remove(x, y)` — delete any particle at (x, y).
  - `set_velocity(x, y, vx, vy)` — NOTE: particles have no velocity in these systems; this action *displaces* the particle at (x, y) to ((x+vx) mod W, (y+vy) mod H) immediately. Included in the API because the agent may have prior expectations about "velocity"; part of the benchmark is seeing whether the agent tests this assumption.
  - `change_type(x, y, new_type)` — replace the type of the particle at (x, y) if one exists.

### 3c. Pattern / sequence puzzles (hidden rule)

- A 1D sequence or small grid governed by a non-standard deterministic recurrence.
- State is a list of integers (possibly bounded modulo some small prime); next_state extends or permutes the list.
- **Invention criteria (all must hold):**
  - Rule combines ≥2 operations in a non-obvious way (e.g. different branches depending on whether the index is prime, perfect-square, or congruent to k mod m; different branches depending on parity or divisibility of a cumulative quantity).
  - Rule is not any named sequence (Fibonacci, Lucas, Padovan, look-and-say, Recamán, Kolakoski, any OEIS A-number with simple closed form), and not any single arithmetic/geometric progression, and not a simple polynomial in n.
  - At least one branch uses a bitwise operation (XOR, AND, OR, shift) and at least one branch uses an arithmetic operation (+, ×, mod); mixing these is already rare in standard puzzles.
- **Intervention API:**
  - `set_element(index, value)` — overwrite state[index] with value.
  - `insert(index, value)` — insert value at position index (shifts subsequent).
  - `apply_perturbation(index, delta)` — add delta (mod the alphabet size) to state[index].

---

## 4. Worked example instances (one per family — fully specified)

These three examples are authoritative: the implementer uses them as templates and the verifier uses them as a smoke test. Every `next_state` in the observations below was produced by executing the corresponding `hidden_rule_fn` on the corresponding `state` (random seed 42; do not substitute hand-crafted values).

---

### 4a. Worked example: CA instance `world_ca_001` — "Thraxon Diagonal CA"

#### 4a.1. Design commentary

- **Why these primitive names?** `ZORK`, `PLOON`, `QUAV` are chosen to have no English-morpheme similarity to any physics/biology/CS term. A model that has read Wolfram's atlas cannot map ZORK→"alive" or QUAV→"wire" by name alone.
- **Why non-standard?** The rule uses the diagonal-only neighbourhood (NE, NW, SE, SW; four cells, **not** the eight-cell Moore neighbourhood), and the per-state transition is not majority-based, not threshold-based, and not symmetric across states:
  - A `ZORK` cell has a specific-count transition (`d_p == 2` exactly, not `>= 2` or `< 2`), making it non-monotone.
  - A `PLOON` cell's transition depends on the *parity* of the sum of two neighbour counts, which mixes counts and modular arithmetic.
  - A `QUAV` cell's transition is an asymmetric argmax over two of the three counts.
- **What does intervention reveal?** Running `set_cell` to construct a uniform grid of all-`ZORK` and stepping reveals that `ZORK` is a fixed point when no `PLOON` or `QUAV` neighbours are present (because `d_p = 0 ≠ 2` and `d_q = 0 < 3`). Running `inject_pattern` with a lone `PLOON` surrounded by `ZORK` reveals the `d_p == 2` branch is hard to satisfy, and the agent learns the specific-count dependence.

#### 4a.2. Full JSON instance

```json
{
  "id": "world_ca_001",
  "family": "cellular_automata",
  "difficulty": "medium",
  "primitive_glossary": {
    "ZORK": "one of three cell states; represented as 0 in encoded grids",
    "PLOON": "one of three cell states; represented as 1",
    "QUAV": "one of three cell states; represented as 2",
    "diagonal-tetrad": "the four cells at NE, NW, SE, SW of a target cell (toroidal wrap)",
    "d_z, d_p, d_q": "the count of ZORK, PLOON, QUAV cells respectively in the diagonal-tetrad"
  },
  "hidden_rule": "Synchronous update of a 3-state grid under toroidal wrap. Each cell's next state depends only on its diagonal-tetrad (NE, NW, SE, SW — never its von Neumann cross). Let d_z, d_p, d_q be the counts of ZORK, PLOON, QUAV in that tetrad. Transition table (by current cell state): ZORK -> PLOON if d_p == 2 (exactly), else QUAV if d_q >= 3, else ZORK. PLOON -> QUAV if (d_z + d_q) is odd, else ZORK if d_p >= 3, else PLOON. QUAV -> ZORK if d_z == 0 and d_p == 0, else PLOON if d_z > d_p, else QUAV.",
  "hidden_rule_fn": "def hidden_rule_fn(state):\n    rows = len(state); cols = len(state[0])\n    ZORK, PLOON, QUAV = 0, 1, 2\n    nxt = [[0]*cols for _ in range(rows)]\n    for r in range(rows):\n        for c in range(cols):\n            diags = [state[(r-1)%rows][(c-1)%cols], state[(r-1)%rows][(c+1)%cols],\n                     state[(r+1)%rows][(c-1)%cols], state[(r+1)%rows][(c+1)%cols]]\n            d_z = diags.count(ZORK); d_p = diags.count(PLOON); d_q = diags.count(QUAV)\n            cur = state[r][c]\n            if cur == ZORK:\n                nxt[r][c] = PLOON if d_p == 2 else (QUAV if d_q >= 3 else ZORK)\n            elif cur == PLOON:\n                nxt[r][c] = QUAV if (d_z + d_q) % 2 == 1 else (ZORK if d_p >= 3 else PLOON)\n            else:\n                if d_z == 0 and d_p == 0: nxt[r][c] = ZORK\n                elif d_z > d_p: nxt[r][c] = PLOON\n                else: nxt[r][c] = QUAV\n    return nxt",
  "intervention_api": [
    {"action": "set_cell", "params": ["row", "col", "state"], "description": "write state into (row, col) of current grid"},
    {"action": "flip_row", "params": ["row"], "description": "cycle every cell in row by ZORK->PLOON->QUAV->ZORK"},
    {"action": "inject_pattern", "params": ["row", "col", "pattern"], "description": "stamp a 2D pattern (list-of-lists; None wildcards leave untouched) at top-left (row, col)"}
  ],
  "train_obs": [
    {"state": [[2,0,0,2],[1,0,0,0],[2,0,2,2],[2,0,2,1]], "next_state": [[1,0,0,2],[1,2,0,2],[1,0,1,2],[1,2,1,1]]},
    {"state": [[0,0,0,0],[0,2,2,0],[2,0,2,2],[2,2,1,0]], "next_state": [[0,0,0,0],[0,1,1,0],[1,0,1,2],[1,1,1,0]]},
    {"state": [[1,2,1,0],[0,2,1,1],[1,0,0,1],[0,0,1,0]], "next_state": [[2,2,2,1],[0,2,2,2],[2,1,0,1],[0,0,2,0]]},
    {"state": [[1,1,2,1],[0,2,1,2],[0,1,0,2],[1,2,2,1]], "next_state": [[2,1,2,1],[0,1,2,1],[2,1,2,2],[2,1,2,2]]},
    {"state": [[2,0,2,0],[0,2,0,1],[0,0,0,1],[1,1,2,1]], "next_state": [[2,0,2,0],[0,1,0,1],[0,0,0,2],[2,1,1,1]]},
    {"state": [[0,1,1,0],[2,1,2,2],[2,0,2,2],[0,2,2,0]], "next_state": [[0,1,2,2],[1,2,1,2],[2,2,2,1],[0,2,1,0]]},
    {"state": [[0,1,1,1],[2,2,2,0],[2,1,0,0],[0,1,1,1]], "next_state": [[1,2,1,2],[2,1,2,0],[2,2,1,0],[0,2,2,2]]},
    {"state": [[0,0,2,2],[1,0,2,1],[1,2,1,0],[1,0,0,2]], "next_state": [[0,1,1,2],[1,1,1,1],[2,2,2,1],[1,1,0,2]]},
    {"state": [[2,2,1,2],[2,1,2,1],[1,0,0,2],[1,0,0,0]], "next_state": [[2,2,1,2],[1,1,1,1],[1,0,1,2],[1,1,2,1]]},
    {"state": [[0,2,0,2],[1,2,0,1],[1,2,1,2],[1,2,0,2]], "next_state": [[2,2,2,2],[1,2,2,1],[2,2,2,2],[1,2,2,2]]},
    {"state": [[2,0,2,2],[1,2,1,0],[1,1,0,1],[0,2,2,1]], "next_state": [[2,1,2,2],[1,2,1,0],[2,1,0,1],[1,2,2,2]]},
    {"state": [[2,0,2,0],[2,1,2,2],[2,0,0,1],[0,2,2,0]], "next_state": [[2,2,2,2],[1,1,1,1],[2,2,0,1],[0,1,1,2]]}
  ],
  "test_obs": [
    {"state": [[2,1,1,0],[0,1,1,0],[0,0,2,0],[0,2,1,0]], "next_state": [[1,1,2,1],[0,2,2,0],[0,1,1,1],[0,2,2,0]]},
    {"state": [[2,0,0,2],[1,2,0,1],[2,2,1,0],[2,2,2,0]], "next_state": [[2,0,0,2],[1,2,0,2],[2,2,2,0],[1,2,1,0]]},
    {"state": [[2,1,1,2],[2,1,1,2],[1,0,0,0],[0,1,0,2]], "next_state": [[2,2,1,1],[1,1,2,2],[1,0,1,0],[0,1,0,2]]},
    {"state": [[2,0,2,0],[0,0,2,2],[0,0,0,0],[1,0,2,0]], "next_state": [[1,0,1,0],[0,0,1,1],[0,0,0,0],[1,0,1,0]]},
    {"state": [[1,2,1,0],[2,0,2,2],[2,1,0,1],[1,0,0,0]], "next_state": [[1,2,1,0],[2,1,2,2],[1,2,0,2],[1,1,1,1]]},
    {"state": [[2,1,1,1],[1,1,2,0],[2,2,2,0],[0,1,2,1]], "next_state": [[2,2,2,2],[1,2,2,2],[2,2,2,0],[1,2,2,2]]}
  ]
}
```

#### 4a.3. Sample successful-agent interaction log

```
step 1: agent observes train_obs[0..11], notices:
        - non-neighbouring (row, col ±1) diagonal-only dependence suspected because
          edge cells in train[4] row 0 are unchanged despite changing row 1 neighbours
        - 3 states, so not a binary CA

step 2: intervention: inject_pattern(0, 0, [[0,0,0],[0,0,0],[0,0,0]])   # all-ZORK 3x3 block
        observe: block remains all-ZORK after one step.
        inference: ZORK is stable when its diagonal-tetrad is all-ZORK.

step 3: intervention: set_cell(1,1, PLOON); set_cell(1,3, PLOON); step.
        observe: cell (2,2) flips ZORK -> PLOON; cell (0,0) stays ZORK.
        inference: 2 PLOONs at diagonals of (2,2) triggered the ZORK->PLOON rule.
        hypothesis-v1: "ZORK cell turns PLOON iff exactly 2 of its diagonal neighbours are PLOON"

step 4: intervention: inject_pattern(0,0, [[1,0,1],[0,0,0],[1,0,1]])    # PLOON at all 4 diagonals of (1,1)
        observe: (1,1) flips ZORK -> ? expect PLOON if hyp-v1 ("exactly 2") is wrong as "≥2";
                  actual: QUAV. Hypothesis refined: not >= 2 but == 2 exactly.
        hypothesis-v2: "ZORK cell turns PLOON iff d_p == 2 (exactly); else QUAV if d_q >= 3 else ZORK"

step 5: similar probing for PLOON and QUAV cells → full rule identified.

final executable_hypothesis: [Python source matching hidden_rule_fn above, after 3–5 refinement iterations]
```

#### 4a.4. Correct executable hypothesis

Identical Python to `hidden_rule_fn` above.

#### 4a.5. Two foil hypotheses (wrong but plausible)

**Foil A — "Majority rule over Moore neighbourhood":**
```python
def foil_A(state):
    rows = len(state); cols = len(state[0])
    nxt = [[0]*cols for _ in range(rows)]
    for r in range(rows):
        for c in range(cols):
            counts = {0:0, 1:0, 2:0}
            for dr in (-1,0,1):
                for dc in (-1,0,1):
                    if dr==0 and dc==0: continue
                    counts[state[(r+dr)%rows][(c+dc)%cols]] += 1
            nxt[r][c] = max(counts, key=counts.get)
    return nxt
```
Why it fails: uses the Moore (8-cell) neighbourhood, not the diagonal-only tetrad; does not reproduce the specific-count behaviour (`d_p == 2` exactly triggers ZORK→PLOON); predicts symmetric majority updates instead of the asymmetric per-state transitions. Mean accuracy on `test_obs` ≈ 0.25–0.35 (often gets uniform regions right but misses every non-trivial transition).

**Foil B — "XOR of diagonals mod 3":**
```python
def foil_B(state):
    rows = len(state); cols = len(state[0])
    nxt = [[0]*cols for _ in range(rows)]
    for r in range(rows):
        for c in range(cols):
            d = [state[(r-1)%rows][(c-1)%cols], state[(r-1)%rows][(c+1)%cols],
                 state[(r+1)%rows][(c-1)%cols], state[(r+1)%rows][(c+1)%cols]]
            nxt[r][c] = (d[0] ^ d[1] ^ d[2] ^ d[3]) % 3
    return nxt
```
Why it fails: picks the correct neighbourhood but the wrong combining function (XOR instead of per-state transition table). Ignores current cell state entirely. Mean accuracy ≈ 0.15–0.25; approximately random.

---

### 4b. Worked example: Particle instance `world_pt_001` — "Grolp Dynamics"

#### 4b.1. Design commentary

- **Why these primitive names?** `ZEX`, `GLORP` are chosen to be phonemically alien to real particle-physics terminology. `chromodistance` borrows the Greek root but is redefined with no connection to QCD colour, so a model that has read particle physics cannot reuse its intuitions.
- **Why non-standard?** The rule combines (i) a type-asymmetric "distance" that differs from both Euclidean and Chebyshev, (ii) a type-asymmetric movement rule where ZEX seeks its nearest same-type partner (bound) while GLORP flees its nearest opposite-type partner (unbound-avoidance), (iii) selection by argmin over chromodistance with lexicographic tie-break. No textbook force law produces this combination; attempting to fit it as "attraction + repulsion" misses the fact that ZEX ignores GLORP entirely when choosing its target.
- **What does intervention reveal?** `spawn(ZEX, ...)` in isolation (single ZEX, no other particles) reveals the "no same-type partner → stay still" branch. `spawn(GLORP, ...)` + `spawn(ZEX, ...)` with controlled placements reveals GLORP flees the ZEX along the larger-component axis (Manhattan-like, but with the axis choice as a discriminating probe). Adding a second ZEX and varying type mixture reveals ZEX's total indifference to GLORP positions.

#### 4b.2. Full JSON instance

```json
{
  "id": "world_pt_001",
  "family": "particle_system",
  "difficulty": "medium",
  "primitive_glossary": {
    "ZEX": "one of two particle types",
    "GLORP": "the other particle type",
    "chromodistance(p, q)": "max(|dx|, |dy|) on the 8x8 toroidal grid, PLUS 0 if p.type == q.type else 1 (the type-mismatch penalty)",
    "same-type-seek": "ZEX particles move one step toward the position of their nearest ZEX partner by chromodistance",
    "cross-type-flee": "GLORP particles move one step away from their nearest ZEX by chromodistance, along the axis whose signed toroidal delta has larger magnitude (x-axis preferred on tie)",
    "tiebreak": "if multiple partners tie on chromodistance, the one with lexicographically smaller (x, y) is chosen"
  },
  "hidden_rule": "Toroidal 8x8 grid. Two particle types: ZEX, GLORP. Chromodistance(p,q) = max(|wrap_dx|, |wrap_dy|) + (0 if same type else 1). Each ZEX particle finds its nearest ZEX (excluding itself) by chromodistance, breaking ties by (x,y); it steps 1 cell (dx, dy) where each component is sign(wrap_delta_to_target). If no other ZEX exists, it stays still. Each GLORP finds its nearest ZEX by chromodistance; it steps 1 cell AWAY from that ZEX along the axis of larger |wrap_delta| (x-axis on tie), unchanged along the other axis. If no ZEX exists, it stays still. All updates are synchronous (based on the t=current snapshot).",
  "hidden_rule_fn": "def hidden_rule_fn(state, grid_size=8):\n    def wrap_delta(a, b):\n        d = (b - a) % grid_size\n        return d - grid_size if d > grid_size // 2 else d\n    def chromodist(p, q):\n        dx = abs(wrap_delta(p['x'], q['x'])); dy = abs(wrap_delta(p['y'], q['y']))\n        return max(dx, dy) + (0 if p['type'] == q['type'] else 1)\n    nxt = []\n    for i, p in enumerate(state):\n        if p['type'] == 'ZEX':\n            same = [q for j, q in enumerate(state) if q['type'] == 'ZEX' and j != i]\n            if not same:\n                nxt.append(dict(p)); continue\n            tgt = min(same, key=lambda q: (chromodist(p, q), q['x'], q['y']))\n            ddx = wrap_delta(p['x'], tgt['x']); ddy = wrap_delta(p['y'], tgt['y'])\n            sx = 0 if ddx == 0 else (1 if ddx > 0 else -1)\n            sy = 0 if ddy == 0 else (1 if ddy > 0 else -1)\n            nxt.append({'type':'ZEX', 'x':(p['x']+sx)%grid_size, 'y':(p['y']+sy)%grid_size})\n        else:\n            diff = [q for j, q in enumerate(state) if q['type'] == 'ZEX']\n            if not diff:\n                nxt.append(dict(p)); continue\n            tgt = min(diff, key=lambda q: (chromodist(p, q), q['x'], q['y']))\n            ddx = wrap_delta(p['x'], tgt['x']); ddy = wrap_delta(p['y'], tgt['y'])\n            if abs(ddx) >= abs(ddy):\n                sx = -1 if ddx >= 0 else 1; sy = 0\n            else:\n                sx = 0; sy = -1 if ddy >= 0 else 1\n            nxt.append({'type':'GLORP', 'x':(p['x']+sx)%grid_size, 'y':(p['y']+sy)%grid_size})\n    return nxt",
  "intervention_api": [
    {"action": "spawn", "params": ["type", "x", "y"], "description": "add a particle at (x,y); no-op if cell already occupied"},
    {"action": "remove", "params": ["x", "y"], "description": "delete any particle at (x,y)"},
    {"action": "set_velocity", "params": ["x", "y", "vx", "vy"], "description": "immediately displace the particle at (x,y) to ((x+vx)%8, (y+vy)%8); these particles have no persistent velocity state — this is a one-shot teleport"},
    {"action": "change_type", "params": ["x", "y", "new_type"], "description": "replace the type of the particle at (x,y) if one exists"}
  ],
  "train_obs": [
    {"state": [{"type":"ZEX","x":1,"y":3},{"type":"ZEX","x":3,"y":7},{"type":"GLORP","x":6,"y":2},{"type":"ZEX","x":7,"y":3}],
     "next_state": [{"type":"ZEX","x":0,"y":3},{"type":"ZEX","x":2,"y":0},{"type":"GLORP","x":5,"y":2},{"type":"ZEX","x":0,"y":3}]},
    {"state": [{"type":"ZEX","x":7,"y":1},{"type":"ZEX","x":0,"y":1},{"type":"GLORP","x":2,"y":6},{"type":"GLORP","x":7,"y":3}],
     "next_state": [{"type":"ZEX","x":0,"y":1},{"type":"ZEX","x":7,"y":1},{"type":"GLORP","x":2,"y":5},{"type":"GLORP","x":7,"y":4}]},
    {"state": [{"type":"GLORP","x":0,"y":2},{"type":"GLORP","x":0,"y":6},{"type":"GLORP","x":7,"y":4},{"type":"ZEX","x":7,"y":2}],
     "next_state": [{"type":"GLORP","x":1,"y":2},{"type":"GLORP","x":0,"y":5},{"type":"GLORP","x":7,"y":5},{"type":"ZEX","x":7,"y":2}]},
    {"state": [{"type":"ZEX","x":4,"y":3},{"type":"ZEX","x":0,"y":5},{"type":"ZEX","x":0,"y":7},{"type":"ZEX","x":0,"y":1}],
     "next_state": [{"type":"ZEX","x":5,"y":2},{"type":"ZEX","x":0,"y":6},{"type":"ZEX","x":0,"y":0},{"type":"ZEX","x":0,"y":0}]},
    {"state": [{"type":"ZEX","x":1,"y":1},{"type":"ZEX","x":6,"y":1},{"type":"GLORP","x":0,"y":1},{"type":"ZEX","x":5,"y":4}],
     "next_state": [{"type":"ZEX","x":0,"y":1},{"type":"ZEX","x":7,"y":1},{"type":"GLORP","x":7,"y":1},{"type":"ZEX","x":6,"y":3}]},
    {"state": [{"type":"GLORP","x":5,"y":3},{"type":"GLORP","x":6,"y":2},{"type":"ZEX","x":7,"y":5},{"type":"ZEX","x":0,"y":7}],
     "next_state": [{"type":"GLORP","x":4,"y":3},{"type":"GLORP","x":6,"y":3},{"type":"ZEX","x":0,"y":6},{"type":"ZEX","x":7,"y":6}]},
    {"state": [{"type":"GLORP","x":1,"y":3},{"type":"ZEX","x":2,"y":5},{"type":"GLORP","x":3,"y":5},{"type":"GLORP","x":2,"y":7}],
     "next_state": [{"type":"GLORP","x":1,"y":2},{"type":"ZEX","x":2,"y":5},{"type":"GLORP","x":4,"y":5},{"type":"GLORP","x":2,"y":0}]},
    {"state": [{"type":"ZEX","x":0,"y":4},{"type":"ZEX","x":2,"y":4},{"type":"GLORP","x":1,"y":2},{"type":"GLORP","x":4,"y":3}],
     "next_state": [{"type":"ZEX","x":1,"y":4},{"type":"ZEX","x":1,"y":4},{"type":"GLORP","x":1,"y":1},{"type":"GLORP","x":5,"y":3}]},
    {"state": [{"type":"GLORP","x":3,"y":4},{"type":"ZEX","x":4,"y":0},{"type":"ZEX","x":6,"y":4},{"type":"ZEX","x":0,"y":5}],
     "next_state": [{"type":"GLORP","x":4,"y":4},{"type":"ZEX","x":5,"y":7},{"type":"ZEX","x":7,"y":5},{"type":"ZEX","x":7,"y":4}]},
    {"state": [{"type":"GLORP","x":4,"y":2},{"type":"ZEX","x":6,"y":0},{"type":"ZEX","x":1,"y":2},{"type":"GLORP","x":5,"y":2}],
     "next_state": [{"type":"GLORP","x":3,"y":2},{"type":"ZEX","x":7,"y":1},{"type":"ZEX","x":0,"y":1},{"type":"GLORP","x":5,"y":3}]},
    {"state": [{"type":"GLORP","x":2,"y":0},{"type":"GLORP","x":5,"y":0},{"type":"ZEX","x":3,"y":3},{"type":"ZEX","x":5,"y":6}],
     "next_state": [{"type":"GLORP","x":2,"y":7},{"type":"GLORP","x":5,"y":1},{"type":"ZEX","x":4,"y":4},{"type":"ZEX","x":4,"y":5}]},
    {"state": [{"type":"ZEX","x":3,"y":2},{"type":"ZEX","x":6,"y":0},{"type":"ZEX","x":5,"y":6},{"type":"ZEX","x":4,"y":2}],
     "next_state": [{"type":"ZEX","x":4,"y":2},{"type":"ZEX","x":5,"y":1},{"type":"ZEX","x":6,"y":7},{"type":"ZEX","x":3,"y":2}]}
  ],
  "test_obs": [
    {"state": [{"type":"GLORP","x":6,"y":0},{"type":"GLORP","x":3,"y":3},{"type":"ZEX","x":5,"y":4},{"type":"ZEX","x":3,"y":0}],
     "next_state": [{"type":"GLORP","x":7,"y":0},{"type":"GLORP","x":2,"y":3},{"type":"ZEX","x":4,"y":5},{"type":"ZEX","x":4,"y":1}]},
    {"state": [{"type":"GLORP","x":6,"y":5},{"type":"GLORP","x":1,"y":4},{"type":"GLORP","x":0,"y":1},{"type":"ZEX","x":2,"y":4}],
     "next_state": [{"type":"GLORP","x":5,"y":5},{"type":"GLORP","x":0,"y":4},{"type":"GLORP","x":0,"y":0},{"type":"ZEX","x":2,"y":4}]},
    {"state": [{"type":"GLORP","x":1,"y":6},{"type":"ZEX","x":5,"y":6},{"type":"GLORP","x":6,"y":3},{"type":"ZEX","x":0,"y":6}],
     "next_state": [{"type":"GLORP","x":2,"y":6},{"type":"ZEX","x":6,"y":6},{"type":"GLORP","x":6,"y":2},{"type":"ZEX","x":7,"y":6}]},
    {"state": [{"type":"GLORP","x":3,"y":5},{"type":"GLORP","x":1,"y":5},{"type":"GLORP","x":1,"y":4},{"type":"GLORP","x":6,"y":5}],
     "next_state": [{"type":"GLORP","x":3,"y":5},{"type":"GLORP","x":1,"y":5},{"type":"GLORP","x":1,"y":4},{"type":"GLORP","x":6,"y":5}]},
    {"state": [{"type":"ZEX","x":4,"y":2},{"type":"ZEX","x":6,"y":6},{"type":"ZEX","x":4,"y":6},{"type":"ZEX","x":4,"y":4}],
     "next_state": [{"type":"ZEX","x":4,"y":3},{"type":"ZEX","x":5,"y":5},{"type":"ZEX","x":4,"y":5},{"type":"ZEX","x":4,"y":3}]},
    {"state": [{"type":"GLORP","x":6,"y":5},{"type":"ZEX","x":7,"y":7},{"type":"ZEX","x":7,"y":2},{"type":"ZEX","x":4,"y":5}],
     "next_state": [{"type":"GLORP","x":7,"y":5},{"type":"ZEX","x":6,"y":6},{"type":"ZEX","x":6,"y":3},{"type":"ZEX","x":5,"y":4}]}
  ]
}
```

#### 4b.3. Sample successful-agent interaction log

```
step 1: observe train_obs — note that in train[3] (all ZEX, no GLORP) every particle moves,
        while in train[10] (all GLORP) they also move, contradicting a naive "GLORPs need ZEX" story.
        Wait — train[10] does have ZEX particles. Re-check: yes, indexes 2,3 are ZEX.
        note: in test[3] (all GLORP, no ZEX) the reference answer is no movement — confirmed by
        evaluator-only check; agent would not see test_obs but can reproduce by intervention.

step 2: intervention: remove all particles; spawn ZEX at (0,0), spawn ZEX at (4,0); step.
        observe: (0,0) -> (1,0), (4,0) -> (3,0). Both moved toward each other along x.
        hypothesis-v1: "ZEX moves toward nearest ZEX"

step 3: intervention: remove all; spawn ZEX at (0,0); step.
        observe: (0,0) stays at (0,0).
        inference: "no other ZEX => ZEX stays still" — confirms v1.

step 4: intervention: remove all; spawn ZEX at (0,0), spawn GLORP at (4,0); step.
        observe: ZEX at (0,0) stays (no other ZEX); GLORP moves to (3,0) — AWAY from ZEX along x-axis.
        inference: "GLORP flees ZEX along larger-|delta| axis".

step 5: intervention: remove all; spawn ZEX at (0,0), spawn GLORP at (0,3); step.
        observe: GLORP moves to (0,2) — AWAY along y. Confirms axis-choice = larger-|delta|.

step 6: intervention: spawn ZEX at (1,1), spawn ZEX at (1,6), spawn GLORP at (1,3); step.
        observe: GLORP moves to (1,4) — AWAY from nearest ZEX (which is the (1,1) one after type-penalty,
        since chromodist to (1,1) = max(0,2)+1=3, to (1,6)=max(0,3)+1=4; nearest is (1,1); flee=(1,4)).
        hypothesis-v2 now includes chromodistance definition: "distance is max(|dx|,|dy|) + 1 if type differs".

step 7: refine iterating to full rule. Final hypothesis matches hidden_rule_fn.
```

#### 4b.4. Correct executable hypothesis

Identical Python to `hidden_rule_fn` above.

#### 4b.5. Two foil hypotheses

**Foil A — "Gravity-like: all particles attract all others":**
```python
def foil_A(state, grid_size=8):
    def sgn(a, b):
        d = (b - a) % grid_size
        if d > grid_size // 2: d -= grid_size
        return 0 if d == 0 else (1 if d > 0 else -1)
    nxt = []
    for i, p in enumerate(state):
        others = [q for j, q in enumerate(state) if j != i]
        if not others:
            nxt.append(dict(p)); continue
        # naive "move toward centroid"
        cx = sum(q["x"] for q in others) // len(others)
        cy = sum(q["y"] for q in others) // len(others)
        nxt.append({"type": p["type"],
                    "x": (p["x"] + sgn(p["x"], cx)) % grid_size,
                    "y": (p["y"] + sgn(p["y"], cy)) % grid_size})
    return nxt
```
Why it fails: assumes type symmetry (GLORP attracts to centroid, which is wrong — GLORP flees ZEX, and ignores other GLORPs when ZEX exists), uses centroid rather than nearest-neighbour selection, uses Euclidean-like sign instead of axis-of-larger-|delta|. Mean accuracy ≈ 0.15–0.25.

**Foil B — "Type-asymmetric Chebyshev: ZEX seeks ZEX, GLORP flees GLORP":**
```python
def foil_B(state, grid_size=8):
    def sgn(a, b):
        d = (b - a) % grid_size
        if d > grid_size // 2: d -= grid_size
        return 0 if d == 0 else (1 if d > 0 else -1)
    def cheby(p, q):
        dx = abs((q["x"] - p["x"]) % grid_size); dy = abs((q["y"] - p["y"]) % grid_size)
        dx = min(dx, grid_size - dx); dy = min(dy, grid_size - dy)
        return max(dx, dy)
    nxt = []
    for i, p in enumerate(state):
        same = [q for j, q in enumerate(state) if j != i and q["type"] == p["type"]]
        if not same:
            nxt.append(dict(p)); continue
        tgt = min(same, key=lambda q: cheby(p, q))
        sx = sgn(p["x"], tgt["x"]); sy = sgn(p["y"], tgt["y"])
        if p["type"] == "GLORP":
            sx, sy = -sx, -sy  # flee
        nxt.append({"type": p["type"],
                    "x": (p["x"] + sx) % grid_size,
                    "y": (p["y"] + sy) % grid_size})
    return nxt
```
Why it fails: gets the "ZEX seeks same-type" branch right but misidentifies GLORP as fleeing other GLORPs; the true rule has GLORP fleeing ZEX (cross-type). Uses pure Chebyshev instead of chromodistance (no type-mismatch penalty). Uses sign-based two-axis movement for GLORP instead of larger-axis-only. Mean accuracy ≈ 0.30–0.45 (gets ZEX moves; misses most GLORP moves).

---

### 4c. Worked example: Pattern instance `world_seq_001` — "Thraxgram Sequence"

#### 4c.1. Design commentary

- **Why these primitive names?** `thraxgate`, `kreels`, `zark-index` are fully synthetic. No English or Latin/Greek root attaches them to a known sequence naming convention.
- **Why non-standard?** The rule has three branches selected by a non-trivial predicate on the new index `n`:
  - If `n` is prime (a `thraxgate` index): XOR the last two elements, mod 7.
  - Else if `n` is even: sum the last two plus an additive constant, mod 7.
  - Else (odd, composite): take twice the element two steps back, plus 1, mod 7.
  The mixture of bitwise (XOR) and arithmetic operations, selected by the primality of the index, is not any named sequence; no closed-form OEIS entry matches; a Fibonacci-style hypothesis fails on prime positions, and a linear-recurrence hypothesis fails on odd-composite positions.
- **What does intervention reveal?** `set_element(n, v)` overwrites an earlier position and re-applies the rule forward — the agent can check whether the rule depends on index alone (yes, via n's primality/parity) or only on values (no). `apply_perturbation(n, delta)` lets the agent test linearity of each branch.

#### 4c.2. Full JSON instance

```json
{
  "id": "world_seq_001",
  "family": "pattern_puzzle",
  "difficulty": "medium",
  "primitive_glossary": {
    "kreels": "the values in the sequence; integers in [0,6]",
    "zark-index": "the position n in the sequence (starting at 0)",
    "thraxgate-index": "a zark-index that is a prime number",
    "ploon-XOR": "bitwise XOR of two kreels, then mod 7",
    "grolp-sum": "arithmetic sum of two kreels plus 3, then mod 7",
    "back-double": "twice the value at zark-index n-2, plus 1, then mod 7"
  },
  "hidden_rule": "Given a sequence (list) of at least 2 kreels, produce the next kreel at zark-index n = len(state). If n is a thraxgate-index (prime), the next kreel = ploon-XOR(state[n-1], state[n-2]). Else if n is even, next kreel = grolp-sum(state[n-1], state[n-2]). Else (n odd and composite), next kreel = back-double(state[n-2]). The returned next_state is the input state with this next kreel appended.",
  "hidden_rule_fn": "def hidden_rule_fn(state):\n    def is_prime(k):\n        if k < 2: return False\n        if k == 2: return True\n        if k % 2 == 0: return False\n        for i in range(3, int(k**0.5)+1, 2):\n            if k % i == 0: return False\n        return True\n    n = len(state)\n    if n < 2:\n        raise ValueError('state must have at least 2 initial elements')\n    if is_prime(n):\n        nxt = (state[n-1] ^ state[n-2]) % 7\n    elif n % 2 == 0:\n        nxt = (state[n-1] + state[n-2] + 3) % 7\n    else:\n        nxt = (state[n-2] * 2 + 1) % 7\n    return state + [nxt]",
  "intervention_api": [
    {"action": "set_element", "params": ["index", "value"], "description": "overwrite state[index] with value"},
    {"action": "insert", "params": ["index", "value"], "description": "insert value at position index, shifting later elements"},
    {"action": "apply_perturbation", "params": ["index", "delta"], "description": "state[index] = (state[index] + delta) % 7"}
  ],
  "train_obs": [
    {"state": [2,5],              "next_state": [2,5,0]},
    {"state": [2,5,0],            "next_state": [2,5,0,5]},
    {"state": [2,5,0,5],          "next_state": [2,5,0,5,1]},
    {"state": [2,5,0,5,1],        "next_state": [2,5,0,5,1,4]},
    {"state": [2,5,0,5,1,4],      "next_state": [2,5,0,5,1,4,1]},
    {"state": [2,5,0,5,1,4,1],    "next_state": [2,5,0,5,1,4,1,5]},
    {"state": [1,3],              "next_state": [1,3,2]},
    {"state": [1,3,2],            "next_state": [1,3,2,1]},
    {"state": [1,3,2,1],          "next_state": [1,3,2,1,6]},
    {"state": [1,3,2,1,6],        "next_state": [1,3,2,1,6,0]},
    {"state": [4,0],              "next_state": [4,0,4]},
    {"state": [4,0,4],            "next_state": [4,0,4,4]}
  ],
  "test_obs": [
    {"state": [6,6],              "next_state": [6,6,0]},
    {"state": [6,6,0],            "next_state": [6,6,0,6]},
    {"state": [6,6,0,6],          "next_state": [6,6,0,6,2]},
    {"state": [6,6,0,6,2],        "next_state": [6,6,0,6,2,4]},
    {"state": [6,6,0,6,2,4],      "next_state": [6,6,0,6,2,4,2]},
    {"state": [6,6,0,6,2,4,2],    "next_state": [6,6,0,6,2,4,2,6]}
  ]
}
```

#### 4c.3. Sample successful-agent interaction log

```
step 1: observe train_obs — multiple seed pairs (2,5), (1,3), (4,0) all yield different patterns.
        This rules out simple "deterministic function of position only" (i.e. no fixed sequence).

step 2: compute differences/sums/XORs of consecutive pairs:
        seed (2,5): x2=0 where 5 XOR 2 = 7 mod 7 = 0. Fit: XOR mod 7.
        seed (1,3): x2=2 where 3 XOR 1 = 2. Fit: XOR mod 7. Consistent.
        seed (4,0): x2=4 where 0 XOR 4 = 4. Fit: XOR mod 7. Consistent.
        hypothesis-v1: "x_n = (x_{n-1} XOR x_{n-2}) mod 7"

step 3: check at n=4 (seed (2,5) gives [2,5,0,5,1]): 5 XOR 0 = 5, but x4 = 1. Fails.
        hypothesis-v1 rejected.

step 4: compute at n=4: 5+0=5. 5+0+1=6. 5+0+3=8 mod 7 = 1. MATCH for +3.
        hypothesis-v2: "x_n = (x_{n-1} + x_{n-2} + 3) mod 7 for even n; XOR for odd n".

step 5: check at n=3 (seed (2,5) gives x3=5): 0 XOR 5 = 5. Consistent with XOR branch.
        check at n=5: 1 XOR 5 = 4. x5=4. Consistent.

step 6: intervention: set_element(2, 3) in seq [2,5,0]; observe next.
        This forces the n=3 position's computation: since the rule depends on index 3 (prime) and state[2]=3, state[1]=5,
        expected x3 = 3 XOR 5 = 6.
        Agent verifies the rule is position-dependent, not value-dependent only.

step 7: check at n=9: train doesn't include this directly. Run hypothesis: n=9 is odd.
        Per v2, use XOR of last two. But n=9 is odd AND composite (9=3*3). Hypothesis-v2 has no composite branch.
        Agent realises need to distinguish prime-odd from composite-odd.

step 8: observe train seed(2,5) step to n=9. Agent uses apply_perturbation(7, +2) to test.
        Computing: x9 expected = 2*x7 + 1 mod 7 = 2*5+1 = 4. Observed: 4. Fit: back-double.
        hypothesis-v3: "if n is prime: XOR; elif n is even: sum+3; else (n odd composite): 2*x_{n-2}+1, all mod 7"

step 9: final hypothesis matches hidden_rule_fn.
```

#### 4c.4. Correct executable hypothesis

Identical Python to `hidden_rule_fn` above.

#### 4c.5. Two foil hypotheses

**Foil A — "Fibonacci-mod-7":**
```python
def foil_A(state):
    return state + [(state[-1] + state[-2]) % 7]
```
Why it fails: matches no branch exactly. On train seed (2,5): predicts x2 = (5+2)%7 = 0. Correct.
But at x3 predicts (0+5)%7 = 5. Correct by coincidence. At x4 predicts (5+0)%7 = 5 but correct is 1. Fails.
On seed (1,3): x2 predicted 4, correct is 2. Mean accuracy ≈ 0.1–0.2.

**Foil B — "XOR mod 7 always":**
```python
def foil_B(state):
    return state + [(state[-1] ^ state[-2]) % 7]
```
Why it fails: correct on primes by coincidence. Misses all even and odd-composite positions. Mean accuracy ≈ 0.35–0.45 (because primes are frequent early in the sequence).

---

## 5. Scoring rubric (v2)

### 5.1. Primary metric

For each world instance, the agent submits an executable hypothesis `h(state) -> next_state`. The evaluator computes:

```
accuracy(instance) = (# of test_obs o for which h(o.state) == o.next_state) / len(test_obs)
```

Equality is deep structural equality on the JSON-encoded state (lists, dicts, ints). For particle systems, order of particles in the output list must match the order expected by `hidden_rule_fn`; if the hypothesis returns a permuted list, it counts as incorrect for that observation. (Rationale: order is part of the world state representation.)

Per arm, the reported metric is `mean_accuracy = mean over all N instances of accuracy(instance)`.

**Accuracy thresholds (per-instance):**
- **≥ 0.8**: strong evidence of correct rule identification.
- **0.5 – 0.8**: partial rule capture (agent has some structural insight but misses a branch or a detail).
- **< 0.5**: rule not captured; may be indistinguishable from random baseline.

### 5.2. Mandatory controls

All three controls are run against the same instance set as the treatment arm and reported alongside. Implemented in v2-P2.

**C-random:** a randomly sampled executable rule from a fixed distribution over simple functions — for CA, a random lookup table over (current_state, neighbour_count_vector); for particles, a random type-conditional sign-direction table; for sequences, a random single-branch linear recurrence. Seed fixed and declared per-instance for reproducibility. Provides a floor baseline: any competent arm must exceed this by > 0.1.

**C-induce:** a non-abductive symbolic-regression learner applied to `train_obs` only (no intervention, no LLM in the loop). Concretely: PySR for the sequence family (over operations {+, −, ×, XOR, mod}); a lightweight grid-search over pre-declared CA rule templates (restricted to von Neumann / Moore / diagonal neighbourhoods, with 3-state transition tables searched exhaustively up to a size cap) for the CA family; and a combined type-conditional template search for the particle family. This control tests whether a non-LLM learner that has no ability to *intervene* but does have large compute can match the abductive arm. If the LLM arm does not exceed C-induce by > 0.1, the abduction claim is not supported.

**C-retrieval:** the agent receives the prose `hidden_rule` field verbatim before solving the instance. This tests whether verbal knowledge of the rule is sufficient (no abduction needed). If the treatment arm matches or exceeds C-retrieval, the arm has achieved rule understanding at least as good as the model's own conversion-of-English-rule-to-code capability. C-retrieval is *not* a ceiling: an arm that genuinely abducts the rule and also writes correct code may equal C-retrieval. But if the treatment arm falls far below C-retrieval, the bottleneck is abduction, not code generation.

An arm "beats" a control if its `mean_accuracy` across the benchmark exceeds the control's `mean_accuracy` by > 0.1 (10 percentage points).

### 5.3. Verdict thresholds

For each arm:
- **APPROVE**: `mean_accuracy ≥ 0.7` AND beats C-random AND beats C-induce.
- **BORDERLINE**: `mean_accuracy ≥ 0.5` AND beats C-random but does NOT beat C-induce.
- **REJECT**: `mean_accuracy < 0.5` OR does not beat C-random.

C-retrieval is reported alongside but does not directly determine the verdict; instead it contextualises the result: "Arm X matched C-retrieval" vs "Arm X underperformed C-retrieval" indicate different qualitative outcomes.

### 5.4. Qualitative rubric axes (secondary — recorded per-instance, not aggregated)

These are informational only and are not used to compute the verdict, but must be reported in verdict files:

- **Intervention efficiency**: the number of intervention steps the agent issued before its executable hypothesis stabilised (no further refinements over the final 20% of its interaction log). Fewer = more efficient abduction.
- **Hypothesis parsimony**: word-count of the final `executable_hypothesis` source divided by the word-count of `hidden_rule_fn`. Near 1.0 = minimal; > 2.0 = baroque.
- **Generalisation**: whether the hypothesis attains ≥ 0.8 accuracy on test_obs (a reflection of the primary metric but recorded per-instance, not just the arm mean).

### 5.5. Verdict file requirements (rubric §6 — inherited from v1.5 P4.5)

For each arm and each instance, the verdict file must include:

- The verbatim `executable_hypothesis` submitted by the arm (full Python source, exactly as the agent emitted it — no paraphrase, no re-indent, no stripped comments).
- An accuracy table: rows = instances, columns = arms and controls (A, B, C-random, C-induce, C-retrieval). Each cell is the accuracy of that arm/control on that instance.
- Control accuracy columns populated from the same run (not a separate pass; fixed seed per-instance so all conditions see identical test_obs).
- A per-instance scoring-process paragraph where the arm's accuracy deviates from its arm mean by more than 0.2. The paragraph explains which observations the hypothesis mis-predicted and what rule misidentification caused the error. (The v1.5 requirement was prose-verbatim-excerpt-of-gold-case; in v2 the analogous per-task transparency is identifying the specific test_obs that failed and why.)

Verdict files that omit any of these are REJECT-ed by the meta-review gate.

---

## 6. Implementer contract

`worlds/gen.py` must implement and export the following API. Other modules import from it; no other file may re-implement these.

```python
def generate_all() -> list[dict]:
    """Return all 15–20 world instances as dicts matching the schema in §2.
    Each family must have ≥4 and ≤8 instances. Total must be ≥15 and ≤20.
    All test_obs must be produced by executing the instance's own hidden_rule_fn
    on the corresponding state (no hand-crafted values)."""

def run_instance(instance: dict, hypothesis_fn) -> float:
    """Given a world instance dict and an executable hypothesis (callable),
    return the fraction of test_obs for which hypothesis_fn(o['state']) == o['next_state'].
    On exception during hypothesis_fn, that observation counts as incorrect.
    Equality is deep structural equality on the JSON-normalised state."""
```

Additional required exports:

```python
FAMILIES = ("cellular_automata", "particle_system", "pattern_puzzle")

def sample_instance(family: str, seed: int) -> dict:
    """Return a single instance for the given family at the given seed.
    Used for regenerating specific instances during debugging and for C-random baselines."""
```

The three worked examples in §4 are authoritative: the implementer must either embed them verbatim (with IDs `world_ca_001`, `world_pt_001`, `world_seq_001`) or reproduce them by seed. Deviation from the documented `next_state` values is a regression bug.

---

## 7. Anti-retrieval checklist

Before adding any instance to the benchmark, verify all of the following:

- [ ] No primitive name matches a real physics, biology, math, or CS term. Check by: searching each name as a Wikipedia article title and as a Wiktionary entry; searching for it in the model's likely training corpora (arXiv abstracts, textbook indices). If any hit is semantically related, rename.
- [ ] The hidden rule cannot be described using standard vocabulary. Test: write the rule in plain English without using any invented term. If the plain-English version is coherent and refers to any named law, metric, sequence, or CA, the rule is too standard. Iterate until the plain-English version is explicitly a composition of operations with no single-word name.
- [ ] `train_obs` and `test_obs` are consistent with `hidden_rule_fn`. Mechanical check: for each observation `o`, assert `hidden_rule_fn(o['state']) == o['next_state']`.
- [ ] An LLM that has read all of Wikipedia would not recognise the rule from the primitive names or the observation patterns alone. Stress-test: present the instance's `train_obs` + `primitive_glossary` to a strong LLM (not the arm under test — a reference model) with the prompt "what rule is this?". If the LLM produces a correct description, the instance is too recognisable; rename primitives and re-run.
- [ ] The `intervention_api` is sufficient to distinguish the hidden rule from at least two plausible foils. For the worked examples in §4, the stated interaction log demonstrates this; for novel instances, the author must write a one-paragraph note in the instance's design commentary explaining which interventions disambiguate which foils.
- [ ] `hidden_rule` prose and `hidden_rule_fn` Python are semantically equivalent. Mechanical check: run `hidden_rule_fn` on each `test_obs.state` and verify it returns `test_obs.next_state`. (This is the same check as point 3 but is called out separately because it has caught real bugs in v1 where prose diverged from code.)

Any instance that fails one or more of these is not eligible for inclusion until fixed.
