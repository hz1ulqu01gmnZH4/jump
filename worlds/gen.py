"""
worlds/gen.py — v2 Invented-World Abduction Benchmark instance generator.

Exports:
  FAMILIES        : list[str]
  generate_all()  : list[dict]
  run_instance()  : float
  sample_instance(): dict
"""

import random
import copy
from typing import Callable

FAMILIES = ["cellular_automata", "particle_system", "pattern_puzzle"]


# ---------------------------------------------------------------------------
# run_instance
# ---------------------------------------------------------------------------

def run_instance(instance: dict, hypothesis_fn: Callable) -> float:
    correct = 0
    for obs in instance["test_obs"]:
        try:
            predicted = hypothesis_fn(obs["state"])
            if predicted == obs["next_state"]:
                correct += 1
        except Exception:
            pass
    return correct / len(instance["test_obs"])


# ---------------------------------------------------------------------------
# Helpers: generate (state, next_state) pairs by running rule functions
# ---------------------------------------------------------------------------

def _ca_random_state(rows, cols, alphabet, rng):
    return [[rng.choice(alphabet) for _ in range(cols)] for _ in range(rows)]


def _pt_random_state(n_particles, types, grid_size, rng):
    positions = set()
    particles = []
    while len(particles) < n_particles:
        x = rng.randint(0, grid_size - 1)
        y = rng.randint(0, grid_size - 1)
        if (x, y) not in positions:
            positions.add((x, y))
            particles.append({"type": rng.choice(types), "x": x, "y": y})
    return particles


def _seq_random_seed(length, mod, rng):
    return [rng.randint(0, mod - 1) for _ in range(length)]


def _make_obs_ca(rule_fn, rows, cols, alphabet, n_train, n_test, seed):
    rng = random.Random(seed)
    obs = []
    while len(obs) < n_train + n_test:
        s = _ca_random_state(rows, cols, alphabet, rng)
        ns = rule_fn(s)
        obs.append({"state": s, "next_state": ns})
    return obs[:n_train], obs[n_train:]


def _make_obs_pt(rule_fn, n_particles, types, grid_size, n_train, n_test, seed):
    rng = random.Random(seed)
    obs = []
    while len(obs) < n_train + n_test:
        s = _pt_random_state(n_particles, types, grid_size, rng)
        ns = rule_fn(s)
        obs.append({"state": s, "next_state": ns})
    return obs[:n_train], obs[n_train:]


def _make_obs_seq(rule_fn, seed_len, mod, n_train, n_test, seed):
    rng = random.Random(seed)
    obs = []
    attempts = 0
    while len(obs) < n_train + n_test and attempts < 10000:
        attempts += 1
        s = _seq_random_seed(seed_len, mod, rng)
        try:
            ns = rule_fn(s)
        except Exception:
            continue
        obs.append({"state": s, "next_state": ns})
    return obs[:n_train], obs[n_train:]


# ---------------------------------------------------------------------------
# ========== CELLULAR AUTOMATA ==========
# ---------------------------------------------------------------------------

# --- world_ca_001: Thraxon Diagonal CA (verbatim from SPEC §4a) ---
def _make_world_ca_001():
    fn_src = (
        "def hidden_rule_fn(state):\n"
        "    rows = len(state); cols = len(state[0])\n"
        "    ZORK, PLOON, QUAV = 0, 1, 2\n"
        "    nxt = [[0]*cols for _ in range(rows)]\n"
        "    for r in range(rows):\n"
        "        for c in range(cols):\n"
        "            diags = [state[(r-1)%rows][(c-1)%cols], state[(r-1)%rows][(c+1)%cols],\n"
        "                     state[(r+1)%rows][(c-1)%cols], state[(r+1)%rows][(c+1)%cols]]\n"
        "            d_z = diags.count(ZORK); d_p = diags.count(PLOON); d_q = diags.count(QUAV)\n"
        "            cur = state[r][c]\n"
        "            if cur == ZORK:\n"
        "                nxt[r][c] = PLOON if d_p == 2 else (QUAV if d_q >= 3 else ZORK)\n"
        "            elif cur == PLOON:\n"
        "                nxt[r][c] = QUAV if (d_z + d_q) % 2 == 1 else (ZORK if d_p >= 3 else PLOON)\n"
        "            else:\n"
        "                if d_z == 0 and d_p == 0: nxt[r][c] = ZORK\n"
        "                elif d_z > d_p: nxt[r][c] = PLOON\n"
        "                else: nxt[r][c] = QUAV\n"
        "    return nxt"
    )
    train_obs = [
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
        {"state": [[2,0,2,0],[2,1,2,2],[2,0,0,1],[0,2,2,0]], "next_state": [[2,2,2,2],[1,1,1,1],[2,2,0,1],[0,1,1,2]]},
    ]
    test_obs = [
        {"state": [[2,1,1,0],[0,1,1,0],[0,0,2,0],[0,2,1,0]], "next_state": [[1,1,2,1],[0,2,2,0],[0,1,1,1],[0,2,2,0]]},
        {"state": [[2,0,0,2],[1,2,0,1],[2,2,1,0],[2,2,2,0]], "next_state": [[2,0,0,2],[1,2,0,2],[2,2,2,0],[1,2,1,0]]},
        {"state": [[2,1,1,2],[2,1,1,2],[1,0,0,0],[0,1,0,2]], "next_state": [[2,2,1,1],[1,1,2,2],[1,0,1,0],[0,1,0,2]]},
        {"state": [[2,0,2,0],[0,0,2,2],[0,0,0,0],[1,0,2,0]], "next_state": [[1,0,1,0],[0,0,1,1],[0,0,0,0],[1,0,1,0]]},
        {"state": [[1,2,1,0],[2,0,2,2],[2,1,0,1],[1,0,0,0]], "next_state": [[1,2,1,0],[2,1,2,2],[1,2,0,2],[1,1,1,1]]},
        {"state": [[2,1,1,1],[1,1,2,0],[2,2,2,0],[0,1,2,1]], "next_state": [[2,2,2,2],[1,2,2,2],[2,2,2,0],[1,2,2,2]]},
    ]
    return {
        "id": "world_ca_001",
        "family": "cellular_automata",
        "difficulty": "medium",
        "primitive_glossary": {
            "ZORK": "one of three cell states; represented as 0 in encoded grids",
            "PLOON": "one of three cell states; represented as 1",
            "QUAV": "one of three cell states; represented as 2",
            "diagonal-tetrad": "the four cells at NE, NW, SE, SW of a target cell (toroidal wrap)",
            "d_z, d_p, d_q": "the count of ZORK, PLOON, QUAV cells respectively in the diagonal-tetrad",
        },
        "hidden_rule": (
            "Synchronous update of a 3-state grid under toroidal wrap. Each cell's next state depends only on its "
            "diagonal-tetrad (NE, NW, SE, SW — never its von Neumann cross). Let d_z, d_p, d_q be the counts of "
            "ZORK, PLOON, QUAV in that tetrad. Transition table (by current cell state): "
            "ZORK -> PLOON if d_p == 2 (exactly), else QUAV if d_q >= 3, else ZORK. "
            "PLOON -> QUAV if (d_z + d_q) is odd, else ZORK if d_p >= 3, else PLOON. "
            "QUAV -> ZORK if d_z == 0 and d_p == 0, else PLOON if d_z > d_p, else QUAV."
        ),
        "hidden_rule_fn": fn_src,
        "intervention_api": [
            {"action": "set_cell", "params": ["row", "col", "state"], "description": "write state into (row, col) of current grid"},
            {"action": "flip_row", "params": ["row"], "description": "cycle every cell in row by ZORK->PLOON->QUAV->ZORK"},
            {"action": "inject_pattern", "params": ["row", "col", "pattern"], "description": "stamp a 2D pattern (list-of-lists; None wildcards leave untouched) at top-left (row, col)"},
        ],
        "train_obs": train_obs,
        "test_obs": test_obs,
    }


# --- world_ca_002: 4-state CA with knight-move neighbourhood ---
def _make_world_ca_002(seed=200):
    """
    4-state CA on 5x5 grid. Neighbourhood: knight-move (8 cells at ±1,±2 and ±2,±1).
    States: VORN=0, SKEL=1, BRIX=2, FLUM=3.
    Rule by current state:
      VORN -> SKEL if count(BRIX in knight) == 3, else FLUM if count(VORN in knight) >= 5, else VORN.
      SKEL -> BRIX if (count(FLUM) + count(BRIX)) % 3 == 0, else VORN if count(SKEL) >= 4, else SKEL.
      BRIX -> FLUM if count(VORN) > count(BRIX), else SKEL if count(SKEL) == 2, else BRIX.
      FLUM -> VORN if count(FLUM) == 0, else BRIX if count(BRIX) >= 3, else FLUM.
    """
    fn_src = (
        "def hidden_rule_fn(state):\n"
        "    rows = len(state); cols = len(state[0])\n"
        "    V, S, B, F = 0, 1, 2, 3\n"
        "    nxt = [[0]*cols for _ in range(rows)]\n"
        "    knight_offsets = [(-2,-1),(-2,1),(-1,-2),(-1,2),(1,-2),(1,2),(2,-1),(2,1)]\n"
        "    for r in range(rows):\n"
        "        for c in range(cols):\n"
        "            nb = [state[(r+dr)%rows][(c+dc)%cols] for dr,dc in knight_offsets]\n"
        "            cv = nb.count(V); cs = nb.count(S); cb = nb.count(B); cf = nb.count(F)\n"
        "            cur = state[r][c]\n"
        "            if cur == V:\n"
        "                if cb == 3: nxt[r][c] = S\n"
        "                elif cv >= 5: nxt[r][c] = F\n"
        "                else: nxt[r][c] = V\n"
        "            elif cur == S:\n"
        "                if (cf + cb) % 3 == 0: nxt[r][c] = B\n"
        "                elif cs >= 4: nxt[r][c] = V\n"
        "                else: nxt[r][c] = S\n"
        "            elif cur == B:\n"
        "                if cv > cb: nxt[r][c] = F\n"
        "                elif cs == 2: nxt[r][c] = S\n"
        "                else: nxt[r][c] = B\n"
        "            else:  # FLUM\n"
        "                if cf == 0: nxt[r][c] = V\n"
        "                elif cb >= 3: nxt[r][c] = B\n"
        "                else: nxt[r][c] = F\n"
        "    return nxt"
    )
    ns = {}
    exec(fn_src, ns)
    rule_fn = ns["hidden_rule_fn"]
    train_obs, test_obs = _make_obs_ca(rule_fn, 5, 5, [0, 1, 2, 3], 12, 6, seed)
    return {
        "id": "world_ca_002",
        "family": "cellular_automata",
        "difficulty": "hard",
        "primitive_glossary": {
            "VORN": "first of four cell states (value 0)",
            "SKEL": "second cell state (value 1)",
            "BRIX": "third cell state (value 2)",
            "FLUM": "fourth cell state (value 3)",
            "knight-octad": "the eight cells reachable by chess knight-moves from a target cell (±1,±2 and ±2,±1), toroidal",
            "cv, cs, cb, cf": "counts of VORN, SKEL, BRIX, FLUM respectively in the knight-octad",
        },
        "hidden_rule": (
            "Synchronous 4-state CA on 5x5 toroidal grid. Neighbourhood is the knight-octad (8 cells at ±1,±2 and ±2,±1). "
            "Let cv,cs,cb,cf be counts of VORN,SKEL,BRIX,FLUM in the knight-octad. "
            "VORN -> SKEL if cb==3, FLUM if cv>=5, else VORN. "
            "SKEL -> BRIX if (cf+cb)%3==0, VORN if cs>=4, else SKEL. "
            "BRIX -> FLUM if cv>cb, SKEL if cs==2, else BRIX. "
            "FLUM -> VORN if cf==0, BRIX if cb>=3, else FLUM."
        ),
        "hidden_rule_fn": fn_src,
        "intervention_api": [
            {"action": "set_cell", "params": ["row", "col", "state"], "description": "write state into (row, col) of current grid"},
            {"action": "flip_row", "params": ["row"], "description": "cycle every cell in row: VORN->SKEL->BRIX->FLUM->VORN"},
            {"action": "inject_pattern", "params": ["row", "col", "pattern"], "description": "stamp 2D pattern at top-left (row,col); None wildcards leave cell untouched"},
        ],
        "train_obs": train_obs,
        "test_obs": test_obs,
    }


# --- world_ca_003: 3-state CA where rule depends on count of DISTINCT neighbour states ---
def _make_world_ca_003(seed=300):
    """
    3-state CA on 4x6 grid. Von Neumann neighbourhood (N,S,E,W — 4 cells).
    States: GREEL=0, TARV=1, MUXON=2.
    Rule depends on the number of DISTINCT states in the neighbourhood (1, 2, or 3),
    combined with the current state:
      Let D = len({nb states}) = distinct count.
      GREEL: -> TARV if D==1 and nb[0]==TARV, else MUXON if D==3, else GREEL.
      TARV:  -> GREEL if D==2 and count(GREEL)>count(TARV), else MUXON if D==1 and nb[0]==MUXON, else TARV.
      MUXON: -> GREEL if D==3, else TARV if D==2 and count(TARV) >= 2, else MUXON.
    """
    fn_src = (
        "def hidden_rule_fn(state):\n"
        "    rows = len(state); cols = len(state[0])\n"
        "    G, T, M = 0, 1, 2\n"
        "    nxt = [[0]*cols for _ in range(rows)]\n"
        "    von_neumann = [(-1,0),(1,0),(0,-1),(0,1)]\n"
        "    for r in range(rows):\n"
        "        for c in range(cols):\n"
        "            nb = [state[(r+dr)%rows][(c+dc)%cols] for dr,dc in von_neumann]\n"
        "            D = len(set(nb))\n"
        "            cg = nb.count(G); ct = nb.count(T)\n"
        "            cur = state[r][c]\n"
        "            if cur == G:\n"
        "                if D == 1 and nb[0] == T: nxt[r][c] = T\n"
        "                elif D == 3: nxt[r][c] = M\n"
        "                else: nxt[r][c] = G\n"
        "            elif cur == T:\n"
        "                if D == 2 and cg > ct: nxt[r][c] = G\n"
        "                elif D == 1 and nb[0] == M: nxt[r][c] = M\n"
        "                else: nxt[r][c] = T\n"
        "            else:  # MUXON\n"
        "                if D == 3: nxt[r][c] = G\n"
        "                elif D == 2 and ct >= 2: nxt[r][c] = T\n"
        "                else: nxt[r][c] = M\n"
        "    return nxt"
    )
    ns = {}
    exec(fn_src, ns)
    rule_fn = ns["hidden_rule_fn"]
    train_obs, test_obs = _make_obs_ca(rule_fn, 4, 6, [0, 1, 2], 12, 6, seed)
    return {
        "id": "world_ca_003",
        "family": "cellular_automata",
        "difficulty": "medium",
        "primitive_glossary": {
            "GREEL": "first of three cell states (value 0)",
            "TARV": "second cell state (value 1)",
            "MUXON": "third cell state (value 2)",
            "von-quad": "the four cells N, S, E, W of a target cell (toroidal wrap)",
            "D": "the count of DISTINCT states present in the von-quad (1, 2, or 3)",
        },
        "hidden_rule": (
            "Synchronous 3-state CA on 4x6 toroidal grid. Neighbourhood is the von-quad (N,S,E,W). "
            "Let D = number of distinct states in the von-quad. "
            "GREEL -> TARV if D==1 and all neighbours are TARV; MUXON if D==3; else GREEL. "
            "TARV -> GREEL if D==2 and count(GREEL)>count(TARV); MUXON if D==1 and all are MUXON; else TARV. "
            "MUXON -> GREEL if D==3; TARV if D==2 and count(TARV)>=2; else MUXON."
        ),
        "hidden_rule_fn": fn_src,
        "intervention_api": [
            {"action": "set_cell", "params": ["row", "col", "state"], "description": "write state into (row, col) of current grid"},
            {"action": "flip_row", "params": ["row"], "description": "cycle every cell in row: GREEL->TARV->MUXON->GREEL"},
            {"action": "inject_pattern", "params": ["row", "col", "pattern"], "description": "stamp 2D pattern at top-left (row,col); None wildcards leave cell untouched"},
        ],
        "train_obs": train_obs,
        "test_obs": test_obs,
    }


# --- world_ca_004: 1D range-3 CA with 3 states ---
def _make_world_ca_004(seed=400):
    """
    1D CA, represented as a single row (1x12 grid, cols=12, rows=1).
    Neighbourhood: range-3 (6 neighbours: offsets -3,-2,-1,+1,+2,+3), toroidal.
    States: KRIV=0, SPEL=1, DORN=2.
    Rule:
      Let s = sum of neighbour values mod 3.
      KRIV -> SPEL if s==1, DORN if s==2, else KRIV.
      SPEL -> DORN if (count(KRIV) + count(DORN)) % 2 == 1, else KRIV if count(SPEL) == 3, else SPEL.
      DORN -> KRIV if (count(KRIV) XOR count(DORN)) > 2, else SPEL if s == 0, else DORN.
      (XOR here means bitwise ^ on integer counts)
    """
    fn_src = (
        "def hidden_rule_fn(state):\n"
        "    cols = len(state[0])\n"
        "    K, S, D = 0, 1, 2\n"
        "    nxt = [[0]*cols]\n"
        "    offsets = [-3,-2,-1,1,2,3]\n"
        "    for c in range(cols):\n"
        "        nb = [state[0][(c+o)%cols] for o in offsets]\n"
        "        ck = nb.count(K); cs = nb.count(S); cd = nb.count(D)\n"
        "        s = sum(nb) % 3\n"
        "        cur = state[0][c]\n"
        "        if cur == K:\n"
        "            if s == 1: nxt[0][c] = S\n"
        "            elif s == 2: nxt[0][c] = D\n"
        "            else: nxt[0][c] = K\n"
        "        elif cur == S:\n"
        "            if (ck + cd) % 2 == 1: nxt[0][c] = D\n"
        "            elif cs == 3: nxt[0][c] = K\n"
        "            else: nxt[0][c] = S\n"
        "        else:  # DORN\n"
        "            if (ck ^ cd) > 2: nxt[0][c] = K\n"
        "            elif s == 0: nxt[0][c] = S\n"
        "            else: nxt[0][c] = D\n"
        "    return nxt"
    )
    ns = {}
    exec(fn_src, ns)
    rule_fn = ns["hidden_rule_fn"]
    train_obs, test_obs = _make_obs_ca(rule_fn, 1, 12, [0, 1, 2], 12, 6, seed)
    return {
        "id": "world_ca_004",
        "family": "cellular_automata",
        "difficulty": "hard",
        "primitive_glossary": {
            "KRIV": "first of three states in the 1D strip (value 0)",
            "SPEL": "second state (value 1)",
            "DORN": "third state (value 2)",
            "range-hexad": "the six cells at offsets -3,-2,-1,+1,+2,+3 from a target cell (toroidal)",
            "s": "sum of range-hexad values mod 3",
            "ck, cs, cd": "counts of KRIV, SPEL, DORN in the range-hexad",
        },
        "hidden_rule": (
            "Synchronous 3-state 1D CA (1x12 strip, toroidal). Neighbourhood: range-hexad (offsets ±1,±2,±3). "
            "Let s = sum(neighbours) mod 3. "
            "KRIV -> SPEL if s==1, DORN if s==2, else KRIV. "
            "SPEL -> DORN if (ck+cd)%2==1, KRIV if cs==3, else SPEL. "
            "DORN -> KRIV if (ck XOR cd)>2, SPEL if s==0, else DORN. "
            "(XOR is integer bitwise ^ on the counts.)"
        ),
        "hidden_rule_fn": fn_src,
        "intervention_api": [
            {"action": "set_cell", "params": ["row", "col", "state"], "description": "write state into (row, col) of current strip (row is always 0)"},
            {"action": "flip_row", "params": ["row"], "description": "cycle every cell: KRIV->SPEL->DORN->KRIV (row must be 0)"},
            {"action": "inject_pattern", "params": ["row", "col", "pattern"], "description": "stamp 1D pattern [[s1,s2,...]] at offset col in strip"},
        ],
        "train_obs": train_obs,
        "test_obs": test_obs,
    }


# --- world_ca_005: 3-state CA, alternate-row parity neighbourhood ---
def _make_world_ca_005(seed=500):
    """
    3-state CA on 5x5 toroidal grid.
    Neighbourhood: for even rows use E/W neighbours (offsets (0,±1)); for odd rows use N/S (offsets (±1,0)).
    States: WULT=0, PRAX=1, ZYMO=2.
    Rule:
      For even row:
        nb = [E, W] (2 neighbours)
        WULT -> PRAX if nb[0]==PRAX and nb[1]==PRAX, else ZYMO if nb[0]!=nb[1], else WULT.
        PRAX -> ZYMO if nb[0]==ZYMO or nb[1]==ZYMO, else WULT if nb[0]==nb[1]==WULT, else PRAX.
        ZYMO -> WULT if nb[0]==nb[1], else PRAX if nb[0]==PRAX or nb[1]==PRAX, else ZYMO.
      For odd row:
        nb = [N, S] (2 neighbours)
        Same rule as even row above (applied to N/S instead of E/W).
    """
    fn_src = (
        "def hidden_rule_fn(state):\n"
        "    rows = len(state); cols = len(state[0])\n"
        "    W, P, Z = 0, 1, 2\n"
        "    nxt = [[0]*cols for _ in range(rows)]\n"
        "    for r in range(rows):\n"
        "        for c in range(cols):\n"
        "            if r % 2 == 0:\n"
        "                nb = [state[r][(c+1)%cols], state[r][(c-1)%cols]]\n"
        "            else:\n"
        "                nb = [state[(r-1)%rows][c], state[(r+1)%rows][c]]\n"
        "            cur = state[r][c]\n"
        "            if cur == W:\n"
        "                if nb[0]==P and nb[1]==P: nxt[r][c] = P\n"
        "                elif nb[0]!=nb[1]: nxt[r][c] = Z\n"
        "                else: nxt[r][c] = W\n"
        "            elif cur == P:\n"
        "                if nb[0]==Z or nb[1]==Z: nxt[r][c] = Z\n"
        "                elif nb[0]==W and nb[1]==W: nxt[r][c] = W\n"
        "                else: nxt[r][c] = P\n"
        "            else:  # ZYMO\n"
        "                if nb[0]==nb[1]: nxt[r][c] = W\n"
        "                elif nb[0]==P or nb[1]==P: nxt[r][c] = P\n"
        "                else: nxt[r][c] = Z\n"
        "    return nxt"
    )
    ns = {}
    exec(fn_src, ns)
    rule_fn = ns["hidden_rule_fn"]
    train_obs, test_obs = _make_obs_ca(rule_fn, 5, 5, [0, 1, 2], 12, 6, seed)
    return {
        "id": "world_ca_005",
        "family": "cellular_automata",
        "difficulty": "easy",
        "primitive_glossary": {
            "WULT": "first of three cell states (value 0)",
            "PRAX": "second cell state (value 1)",
            "ZYMO": "third cell state (value 2)",
            "row-parity split": "even rows use their E/W pair as neighbourhood; odd rows use their N/S pair",
            "diad": "the two-cell neighbourhood used in a given row (either E/W or N/S pair)",
        },
        "hidden_rule": (
            "Synchronous 3-state CA on 5x5 toroidal grid. Neighbourhood is parity-split: "
            "even rows use E/W diad; odd rows use N/S diad. Let nb=[nb0,nb1]. "
            "WULT -> PRAX if both PRAX; ZYMO if nb0!=nb1; else WULT. "
            "PRAX -> ZYMO if any ZYMO; WULT if both WULT; else PRAX. "
            "ZYMO -> WULT if nb0==nb1; PRAX if any PRAX; else ZYMO."
        ),
        "hidden_rule_fn": fn_src,
        "intervention_api": [
            {"action": "set_cell", "params": ["row", "col", "state"], "description": "write state into (row, col) of current grid"},
            {"action": "flip_row", "params": ["row"], "description": "cycle every cell in row: WULT->PRAX->ZYMO->WULT"},
            {"action": "inject_pattern", "params": ["row", "col", "pattern"], "description": "stamp 2D pattern at top-left (row,col); None wildcards leave cell untouched"},
        ],
        "train_obs": train_obs,
        "test_obs": test_obs,
    }


# --- world_ca_006: 3-state CA, coordinate-modular neighbourhood selection ---
def _make_world_ca_006(seed=600):
    """
    3-state CA on 4x4 toroidal grid.
    Neighbourhood depends on (r+c) mod 3:
      0 -> diagonal tetrad (NE, NW, SE, SW)
      1 -> von Neumann quad (N,S,E,W)
      2 -> just cell itself is only input (so count is current only; always self-stable unless coord changes)
         Actually: use the 4 cells at distance 2 along axes: (r±2, c), (r, c±2)
    States: FLEB=0, ZORQ=1, PUND=2.
    Rule:
      f0 = fleb count, f1 = zorq count, f2 = pund count in neighbourhood.
      FLEB -> ZORQ if f1 > f2, PUND if f2 > f1, else FLEB.
      ZORQ -> FLEB if f0 >= 3, PUND if f1 == 0, else ZORQ.
      PUND -> FLEB if (f0 + f1) % 2 == 0 and f2 == 0, else ZORQ if f0 < f2, else PUND.
    """
    fn_src = (
        "def hidden_rule_fn(state):\n"
        "    rows = len(state); cols = len(state[0])\n"
        "    FL, ZO, PU = 0, 1, 2\n"
        "    nxt = [[0]*cols for _ in range(rows)]\n"
        "    diag = [(-1,-1),(-1,1),(1,-1),(1,1)]\n"
        "    vn = [(-1,0),(1,0),(0,-1),(0,1)]\n"
        "    axial2 = [(-2,0),(2,0),(0,-2),(0,2)]\n"
        "    for r in range(rows):\n"
        "        for c in range(cols):\n"
        "            m = (r + c) % 3\n"
        "            if m == 0: offsets = diag\n"
        "            elif m == 1: offsets = vn\n"
        "            else: offsets = axial2\n"
        "            nb = [state[(r+dr)%rows][(c+dc)%cols] for dr,dc in offsets]\n"
        "            f0 = nb.count(FL); f1 = nb.count(ZO); f2 = nb.count(PU)\n"
        "            cur = state[r][c]\n"
        "            if cur == FL:\n"
        "                if f1 > f2: nxt[r][c] = ZO\n"
        "                elif f2 > f1: nxt[r][c] = PU\n"
        "                else: nxt[r][c] = FL\n"
        "            elif cur == ZO:\n"
        "                if f0 >= 3: nxt[r][c] = FL\n"
        "                elif f1 == 0: nxt[r][c] = PU\n"
        "                else: nxt[r][c] = ZO\n"
        "            else:  # PUND\n"
        "                if (f0 + f1) % 2 == 0 and f2 == 0: nxt[r][c] = FL\n"
        "                elif f0 < f2: nxt[r][c] = ZO\n"
        "                else: nxt[r][c] = PU\n"
        "    return nxt"
    )
    ns = {}
    exec(fn_src, ns)
    rule_fn = ns["hidden_rule_fn"]
    train_obs, test_obs = _make_obs_ca(rule_fn, 4, 4, [0, 1, 2], 12, 6, seed)
    return {
        "id": "world_ca_006",
        "family": "cellular_automata",
        "difficulty": "hard",
        "primitive_glossary": {
            "FLEB": "first of three cell states (value 0)",
            "ZORQ": "second cell state (value 1)",
            "PUND": "third cell state (value 2)",
            "coord-class": "(row + col) mod 3 — determines which neighbourhood class a cell uses",
            "class-0 neighbourhood": "diagonal tetrad (NE,NW,SE,SW) — used when coord-class==0",
            "class-1 neighbourhood": "von Neumann quad (N,S,E,W) — used when coord-class==1",
            "class-2 neighbourhood": "axial-2 quad (cells at distance 2 along N,S,E,W axes) — used when coord-class==2",
            "f0, f1, f2": "counts of FLEB, ZORQ, PUND in the cell's neighbourhood",
        },
        "hidden_rule": (
            "Synchronous 3-state CA on 4x4 toroidal grid. Each cell's neighbourhood is selected by (r+c) mod 3: "
            "class 0 uses diagonal tetrad; class 1 uses von Neumann quad; class 2 uses axial-2 quad (±2 along axes). "
            "FLEB -> ZORQ if f1>f2, PUND if f2>f1, else FLEB. "
            "ZORQ -> FLEB if f0>=3, PUND if f1==0, else ZORQ. "
            "PUND -> FLEB if (f0+f1)%2==0 and f2==0, else ZORQ if f0<f2, else PUND."
        ),
        "hidden_rule_fn": fn_src,
        "intervention_api": [
            {"action": "set_cell", "params": ["row", "col", "state"], "description": "write state into (row, col) of current grid"},
            {"action": "flip_row", "params": ["row"], "description": "cycle every cell in row: FLEB->ZORQ->PUND->FLEB"},
            {"action": "inject_pattern", "params": ["row", "col", "pattern"], "description": "stamp 2D pattern at top-left (row,col); None wildcards leave cell untouched"},
        ],
        "train_obs": train_obs,
        "test_obs": test_obs,
    }


# ---------------------------------------------------------------------------
# ========== PARTICLE SYSTEMS ==========
# ---------------------------------------------------------------------------

# --- world_pt_001: Grolp Dynamics (verbatim from SPEC §4b) ---
def _make_world_pt_001():
    fn_src = (
        "def hidden_rule_fn(state, grid_size=8):\n"
        "    def wrap_delta(a, b):\n"
        "        d = (b - a) % grid_size\n"
        "        return d - grid_size if d > grid_size // 2 else d\n"
        "    def chromodist(p, q):\n"
        "        dx = abs(wrap_delta(p['x'], q['x'])); dy = abs(wrap_delta(p['y'], q['y']))\n"
        "        return max(dx, dy) + (0 if p['type'] == q['type'] else 1)\n"
        "    nxt = []\n"
        "    for i, p in enumerate(state):\n"
        "        if p['type'] == 'ZEX':\n"
        "            same = [q for j, q in enumerate(state) if q['type'] == 'ZEX' and j != i]\n"
        "            if not same:\n"
        "                nxt.append(dict(p)); continue\n"
        "            tgt = min(same, key=lambda q: (chromodist(p, q), q['x'], q['y']))\n"
        "            ddx = wrap_delta(p['x'], tgt['x']); ddy = wrap_delta(p['y'], tgt['y'])\n"
        "            sx = 0 if ddx == 0 else (1 if ddx > 0 else -1)\n"
        "            sy = 0 if ddy == 0 else (1 if ddy > 0 else -1)\n"
        "            nxt.append({'type':'ZEX', 'x':(p['x']+sx)%grid_size, 'y':(p['y']+sy)%grid_size})\n"
        "        else:\n"
        "            diff = [q for j, q in enumerate(state) if q['type'] == 'ZEX']\n"
        "            if not diff:\n"
        "                nxt.append(dict(p)); continue\n"
        "            tgt = min(diff, key=lambda q: (chromodist(p, q), q['x'], q['y']))\n"
        "            ddx = wrap_delta(p['x'], tgt['x']); ddy = wrap_delta(p['y'], tgt['y'])\n"
        "            if abs(ddx) >= abs(ddy):\n"
        "                sx = -1 if ddx >= 0 else 1; sy = 0\n"
        "            else:\n"
        "                sx = 0; sy = -1 if ddy >= 0 else 1\n"
        "            nxt.append({'type':'GLORP', 'x':(p['x']+sx)%grid_size, 'y':(p['y']+sy)%grid_size})\n"
        "    return nxt"
    )
    train_obs = [
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
         "next_state": [{"type":"ZEX","x":4,"y":2},{"type":"ZEX","x":5,"y":1},{"type":"ZEX","x":6,"y":7},{"type":"ZEX","x":3,"y":2}]},
    ]
    test_obs = [
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
         "next_state": [{"type":"GLORP","x":7,"y":5},{"type":"ZEX","x":6,"y":6},{"type":"ZEX","x":6,"y":3},{"type":"ZEX","x":5,"y":4}]},
    ]
    return {
        "id": "world_pt_001",
        "family": "particle_system",
        "difficulty": "medium",
        "primitive_glossary": {
            "ZEX": "one of two particle types",
            "GLORP": "the other particle type",
            "chromodistance(p, q)": "max(|dx|, |dy|) on the 8x8 toroidal grid, PLUS 0 if p.type == q.type else 1 (the type-mismatch penalty)",
            "same-type-seek": "ZEX particles move one step toward the position of their nearest ZEX partner by chromodistance",
            "cross-type-flee": "GLORP particles move one step away from their nearest ZEX by chromodistance, along the axis whose signed toroidal delta has larger magnitude (x-axis preferred on tie)",
            "tiebreak": "if multiple partners tie on chromodistance, the one with lexicographically smaller (x, y) is chosen",
        },
        "hidden_rule": (
            "Toroidal 8x8 grid. Two particle types: ZEX, GLORP. Chromodistance(p,q) = max(|wrap_dx|, |wrap_dy|) + (0 if same type else 1). "
            "Each ZEX particle finds its nearest ZEX (excluding itself) by chromodistance, breaking ties by (x,y); it steps 1 cell (dx, dy) where each component is sign(wrap_delta_to_target). "
            "If no other ZEX exists, it stays still. Each GLORP finds its nearest ZEX by chromodistance; it steps 1 cell AWAY from that ZEX along the axis of larger |wrap_delta| (x-axis on tie), unchanged along the other axis. "
            "If no ZEX exists, it stays still. All updates are synchronous (based on the t=current snapshot)."
        ),
        "hidden_rule_fn": fn_src,
        "intervention_api": [
            {"action": "spawn", "params": ["type", "x", "y"], "description": "add a particle at (x,y); no-op if cell already occupied"},
            {"action": "remove", "params": ["x", "y"], "description": "delete any particle at (x,y)"},
            {"action": "set_velocity", "params": ["x", "y", "vx", "vy"], "description": "immediately displace the particle at (x,y) to ((x+vx)%8, (y+vy)%8); these particles have no persistent velocity state — this is a one-shot teleport"},
            {"action": "change_type", "params": ["x", "y", "new_type"], "description": "replace the type of the particle at (x,y) if one exists"},
        ],
        "train_obs": train_obs,
        "test_obs": test_obs,
    }


# --- world_pt_002: 3-type parity-grolp system ---
def _make_world_pt_002(seed=700):
    """
    3 types: VRIX, SPONT, KRAUL on 8x8 toroidal grid.
    Synthetic distance: grolp_dist(p,q) = |dx| + 2*|dy| if p.type != q.type else |dx|*|dy| + 1
    (type-mismatch uses weighted-Manhattan; same type uses a product-based distance)
    Rule:
      VRIX: move toward nearest SPONT by grolp_dist. If no SPONT, stay still.
      SPONT: move away from nearest KRAUL along y-axis (away = sign(-wrap_dy)), x unchanged.
             If no KRAUL, stay still.
      KRAUL: move toward nearest VRIX by grolp_dist. If no VRIX, stay still.
    Tiebreak for all: lexicographic (x,y).
    """
    fn_src = (
        "def hidden_rule_fn(state, grid_size=8):\n"
        "    def wrap_delta(a, b):\n"
        "        d = (b - a) % grid_size\n"
        "        return d - grid_size if d > grid_size // 2 else d\n"
        "    def grolp_dist(p, q):\n"
        "        dx = abs(wrap_delta(p['x'], q['x']))\n"
        "        dy = abs(wrap_delta(p['y'], q['y']))\n"
        "        if p['type'] != q['type']:\n"
        "            return dx + 2 * dy\n"
        "        else:\n"
        "            return dx * dy + 1\n"
        "    nxt = []\n"
        "    for i, p in enumerate(state):\n"
        "        t = p['type']\n"
        "        if t == 'VRIX':\n"
        "            targets = [q for q in state if q['type'] == 'SPONT']\n"
        "            if not targets:\n"
        "                nxt.append(dict(p)); continue\n"
        "            tgt = min(targets, key=lambda q: (grolp_dist(p,q), q['x'], q['y']))\n"
        "            ddx = wrap_delta(p['x'], tgt['x'])\n"
        "            ddy = wrap_delta(p['y'], tgt['y'])\n"
        "            sx = 0 if ddx==0 else (1 if ddx>0 else -1)\n"
        "            sy = 0 if ddy==0 else (1 if ddy>0 else -1)\n"
        "            nxt.append({'type':'VRIX','x':(p['x']+sx)%grid_size,'y':(p['y']+sy)%grid_size})\n"
        "        elif t == 'SPONT':\n"
        "            targets = [q for q in state if q['type'] == 'KRAUL']\n"
        "            if not targets:\n"
        "                nxt.append(dict(p)); continue\n"
        "            tgt = min(targets, key=lambda q: (grolp_dist(p,q), q['x'], q['y']))\n"
        "            ddy = wrap_delta(p['y'], tgt['y'])\n"
        "            sy = 0 if ddy==0 else (-1 if ddy>0 else 1)\n"
        "            nxt.append({'type':'SPONT','x':p['x'],'y':(p['y']+sy)%grid_size})\n"
        "        else:  # KRAUL\n"
        "            targets = [q for q in state if q['type'] == 'VRIX']\n"
        "            if not targets:\n"
        "                nxt.append(dict(p)); continue\n"
        "            tgt = min(targets, key=lambda q: (grolp_dist(p,q), q['x'], q['y']))\n"
        "            ddx = wrap_delta(p['x'], tgt['x'])\n"
        "            ddy = wrap_delta(p['y'], tgt['y'])\n"
        "            sx = 0 if ddx==0 else (1 if ddx>0 else -1)\n"
        "            sy = 0 if ddy==0 else (1 if ddy>0 else -1)\n"
        "            nxt.append({'type':'KRAUL','x':(p['x']+sx)%grid_size,'y':(p['y']+sy)%grid_size})\n"
        "    return nxt"
    )
    ns = {}
    exec(fn_src, ns)
    rule_fn = ns["hidden_rule_fn"]
    train_obs, test_obs = _make_obs_pt(rule_fn, 5, ["VRIX","SPONT","KRAUL"], 8, 12, 6, seed)
    return {
        "id": "world_pt_002",
        "family": "particle_system",
        "difficulty": "medium",
        "primitive_glossary": {
            "VRIX": "first of three particle types",
            "SPONT": "second particle type",
            "KRAUL": "third particle type",
            "grolp_dist(p,q)": "if types differ: |wrap_dx| + 2*|wrap_dy|; if same type: |wrap_dx|*|wrap_dy| + 1; on 8x8 toroidal grid",
        },
        "hidden_rule": (
            "Toroidal 8x8 grid, three types: VRIX, SPONT, KRAUL. grolp_dist depends on type-match. "
            "VRIX moves toward nearest SPONT (by grolp_dist, tiebreak (x,y)), one diagonal step. "
            "SPONT moves away from nearest KRAUL along y-axis only (x unchanged). "
            "KRAUL moves toward nearest VRIX, one diagonal step. "
            "If the target type is absent, particle stays still. All synchronous."
        ),
        "hidden_rule_fn": fn_src,
        "intervention_api": [
            {"action": "spawn", "params": ["type", "x", "y"], "description": "add a particle at (x,y); no-op if occupied"},
            {"action": "remove", "params": ["x", "y"], "description": "delete particle at (x,y)"},
            {"action": "set_velocity", "params": ["x", "y", "vx", "vy"], "description": "displace particle at (x,y) to ((x+vx)%8, (y+vy)%8) immediately"},
            {"action": "change_type", "params": ["x", "y", "new_type"], "description": "replace the type of the particle at (x,y)"},
        ],
        "train_obs": train_obs,
        "test_obs": test_obs,
    }


# --- world_pt_003: parity-based interaction strength ---
def _make_world_pt_003(seed=800):
    """
    2 types: BLINX, TORVE on 8x8 toroidal grid.
    Synthetic quantity: splurn(p,q) = (p.x + p.y + q.x + q.y) mod 2   (parity of coordinate sum)
    Rule:
      BLINX: find nearest TORVE by Chebyshev distance (max(|dx|,|dy|)).
        If splurn(BLINX,TORVE)==0: move toward TORVE (1 step, sign each axis).
        If splurn==1: move away (negate sign).
        If no TORVE: stay.
      TORVE: find nearest BLINX.
        If splurn(TORVE,BLINX)==0: move toward BLINX.
        If splurn==1: stay still.
        If no BLINX: stay.
    Tiebreak: lex (x,y).
    """
    fn_src = (
        "def hidden_rule_fn(state, grid_size=8):\n"
        "    def wrap_delta(a, b):\n"
        "        d = (b - a) % grid_size\n"
        "        return d - grid_size if d > grid_size // 2 else d\n"
        "    def cheby(p, q):\n"
        "        return max(abs(wrap_delta(p['x'],q['x'])), abs(wrap_delta(p['y'],q['y'])))\n"
        "    def splurn(p, q):\n"
        "        return (p['x'] + p['y'] + q['x'] + q['y']) % 2\n"
        "    nxt = []\n"
        "    for p in state:\n"
        "        if p['type'] == 'BLINX':\n"
        "            targets = [q for q in state if q['type'] == 'TORVE']\n"
        "            if not targets:\n"
        "                nxt.append(dict(p)); continue\n"
        "            tgt = min(targets, key=lambda q: (cheby(p,q), q['x'], q['y']))\n"
        "            ddx = wrap_delta(p['x'], tgt['x'])\n"
        "            ddy = wrap_delta(p['y'], tgt['y'])\n"
        "            sx = 0 if ddx==0 else (1 if ddx>0 else -1)\n"
        "            sy = 0 if ddy==0 else (1 if ddy>0 else -1)\n"
        "            if splurn(p, tgt) == 1:\n"
        "                sx, sy = -sx, -sy\n"
        "            nxt.append({'type':'BLINX','x':(p['x']+sx)%grid_size,'y':(p['y']+sy)%grid_size})\n"
        "        else:  # TORVE\n"
        "            targets = [q for q in state if q['type'] == 'BLINX']\n"
        "            if not targets:\n"
        "                nxt.append(dict(p)); continue\n"
        "            tgt = min(targets, key=lambda q: (cheby(p,q), q['x'], q['y']))\n"
        "            if splurn(p, tgt) == 0:\n"
        "                ddx = wrap_delta(p['x'], tgt['x'])\n"
        "                ddy = wrap_delta(p['y'], tgt['y'])\n"
        "                sx = 0 if ddx==0 else (1 if ddx>0 else -1)\n"
        "                sy = 0 if ddy==0 else (1 if ddy>0 else -1)\n"
        "                nxt.append({'type':'TORVE','x':(p['x']+sx)%grid_size,'y':(p['y']+sy)%grid_size})\n"
        "            else:\n"
        "                nxt.append(dict(p))\n"
        "    return nxt"
    )
    ns = {}
    exec(fn_src, ns)
    rule_fn = ns["hidden_rule_fn"]
    train_obs, test_obs = _make_obs_pt(rule_fn, 4, ["BLINX","TORVE"], 8, 12, 6, seed)
    return {
        "id": "world_pt_003",
        "family": "particle_system",
        "difficulty": "hard",
        "primitive_glossary": {
            "BLINX": "first of two particle types",
            "TORVE": "second particle type",
            "splurn(p,q)": "(p.x + p.y + q.x + q.y) mod 2 — parity of total coordinate sum; 0 = even, 1 = odd",
            "Chebyshev distance": "max(|wrap_dx|, |wrap_dy|) on 8x8 toroidal grid",
        },
        "hidden_rule": (
            "Toroidal 8x8 grid, two types. BLINX seeks or flees nearest TORVE depending on splurn: "
            "even splurn -> toward (diagonal step); odd splurn -> away (negated). "
            "TORVE: even splurn with nearest BLINX -> move toward; odd splurn -> stay. "
            "All use Chebyshev distance, lex (x,y) tiebreak. Synchronous."
        ),
        "hidden_rule_fn": fn_src,
        "intervention_api": [
            {"action": "spawn", "params": ["type", "x", "y"], "description": "add a particle at (x,y); no-op if occupied"},
            {"action": "remove", "params": ["x", "y"], "description": "delete particle at (x,y)"},
            {"action": "set_velocity", "params": ["x", "y", "vx", "vy"], "description": "displace particle at (x,y) to ((x+vx)%8,(y+vy)%8) immediately"},
            {"action": "change_type", "params": ["x", "y", "new_type"], "description": "replace the type of the particle at (x,y)"},
        ],
        "train_obs": train_obs,
        "test_obs": test_obs,
    }


# --- world_pt_004: predator-prey style with spawning ---
def _make_world_pt_004(seed=900):
    """
    2 types: PLEKT (predator) and WOVVE (prey) on 8x8 toroidal grid.
    Distance: wovve_dist(p,q) = |dx|^2 + |dy| (non-symmetric in axes — squared x + linear y)
              but only used for target selection; symmetric.
    Rule:
      PLEKT: moves toward nearest WOVVE by wovve_dist (diagonal step). If no WOVVE, stay.
      WOVVE: if any PLEKT is at Chebyshev distance 1 (adjacent), it moves away from it along
             the axis of largest |delta| (x preferred on tie). Otherwise moves toward nearest
             WOVVE (same-type clustering) by Chebyshev, or stays if no other WOVVE.
    Spawning rule: WOVVE spawns a new WOVVE at the cell diametrically opposite on the grid
      ((x+4)%8, (y+4)%8) whenever there is NO PLEKT within Chebyshev distance 3. New particle
      added at the END of the state list. But if target cell is occupied, skip spawn.
    Tiebreak: lex (x,y).
    """
    fn_src = (
        "def hidden_rule_fn(state, grid_size=8):\n"
        "    def wrap_delta(a, b):\n"
        "        d = (b - a) % grid_size\n"
        "        return d - grid_size if d > grid_size // 2 else d\n"
        "    def cheby(p, q):\n"
        "        return max(abs(wrap_delta(p['x'],q['x'])), abs(wrap_delta(p['y'],q['y'])))\n"
        "    def wovve_dist(p, q):\n"
        "        dx = abs(wrap_delta(p['x'],q['x']))\n"
        "        dy = abs(wrap_delta(p['y'],q['y']))\n"
        "        return dx*dx + dy\n"
        "    occupied = {(p['x'],p['y']) for p in state}\n"
        "    nxt = []\n"
        "    spawns = []\n"
        "    for p in state:\n"
        "        if p['type'] == 'PLEKT':\n"
        "            targets = [q for q in state if q['type'] == 'WOVVE']\n"
        "            if not targets:\n"
        "                nxt.append(dict(p)); continue\n"
        "            tgt = min(targets, key=lambda q: (wovve_dist(p,q), q['x'], q['y']))\n"
        "            ddx = wrap_delta(p['x'],tgt['x']); ddy = wrap_delta(p['y'],tgt['y'])\n"
        "            sx = 0 if ddx==0 else (1 if ddx>0 else -1)\n"
        "            sy = 0 if ddy==0 else (1 if ddy>0 else -1)\n"
        "            nxt.append({'type':'PLEKT','x':(p['x']+sx)%grid_size,'y':(p['y']+sy)%grid_size})\n"
        "        else:  # WOVVE\n"
        "            plekts = [q for q in state if q['type'] == 'PLEKT']\n"
        "            near_plekt = [q for q in plekts if cheby(p,q) <= 1]\n"
        "            if near_plekt:\n"
        "                threat = min(near_plekt, key=lambda q: (cheby(p,q), q['x'], q['y']))\n"
        "                ddx = wrap_delta(p['x'],threat['x']); ddy = wrap_delta(p['y'],threat['y'])\n"
        "                if abs(ddx) >= abs(ddy):\n"
        "                    sx = -1 if ddx>=0 else 1; sy = 0\n"
        "                else:\n"
        "                    sx = 0; sy = -1 if ddy>=0 else 1\n"
        "                nxt.append({'type':'WOVVE','x':(p['x']+sx)%grid_size,'y':(p['y']+sy)%grid_size})\n"
        "            else:\n"
        "                others = [q for q in state if q['type']=='WOVVE' and (q['x'],q['y'])!=(p['x'],p['y'])]\n"
        "                if others:\n"
        "                    tgt2 = min(others, key=lambda q: (cheby(p,q), q['x'], q['y']))\n"
        "                    ddx = wrap_delta(p['x'],tgt2['x']); ddy = wrap_delta(p['y'],tgt2['y'])\n"
        "                    sx = 0 if ddx==0 else (1 if ddx>0 else -1)\n"
        "                    sy = 0 if ddy==0 else (1 if ddy>0 else -1)\n"
        "                    nxt.append({'type':'WOVVE','x':(p['x']+sx)%grid_size,'y':(p['y']+sy)%grid_size})\n"
        "                else:\n"
        "                    nxt.append(dict(p))\n"
        "                # spawn check\n"
        "                plekts_all = [q for q in state if q['type']=='PLEKT']\n"
        "                safe = all(cheby(p,q) > 3 for q in plekts_all) if plekts_all else True\n"
        "                if safe:\n"
        "                    nx2 = (p['x']+4)%grid_size; ny2 = (p['y']+4)%grid_size\n"
        "                    if (nx2, ny2) not in occupied:\n"
        "                        spawns.append({'type':'WOVVE','x':nx2,'y':ny2})\n"
        "                        occupied.add((nx2,ny2))\n"
        "    return nxt + spawns"
    )
    ns = {}
    exec(fn_src, ns)
    rule_fn = ns["hidden_rule_fn"]
    rng = random.Random(seed)
    train_obs = []
    test_obs = []
    # Generate with mixed particle counts to get diverse scenarios
    for target_list, n_needed in [(train_obs, 12), (test_obs, 6)]:
        attempts = 0
        while len(target_list) < n_needed and attempts < 10000:
            attempts += 1
            n = rng.randint(3, 6)
            s = _pt_random_state(n, ["PLEKT", "WOVVE"], 8, rng)
            try:
                ns_state = rule_fn(s)
            except Exception:
                continue
            target_list.append({"state": s, "next_state": ns_state})
    return {
        "id": "world_pt_004",
        "family": "particle_system",
        "difficulty": "hard",
        "primitive_glossary": {
            "PLEKT": "predator particle type",
            "WOVVE": "prey particle type",
            "wovve_dist(p,q)": "|wrap_dx|^2 + |wrap_dy| — squared horizontal component plus linear vertical",
            "Chebyshev distance": "max(|wrap_dx|, |wrap_dy|) on 8x8 toroidal grid",
            "anti-pole": "the cell at ((x+4)%8, (y+4)%8) — diametrically opposite on the 8x8 grid",
        },
        "hidden_rule": (
            "Toroidal 8x8 grid, two types. PLEKT moves toward nearest WOVVE by wovve_dist (diagonal step). "
            "WOVVE: if any PLEKT within Chebyshev-1, flee along larger-|delta| axis; else cluster toward nearest WOVVE. "
            "WOVVE also spawns a copy at its anti-pole if no PLEKT is within Chebyshev-3 and the anti-pole is unoccupied. "
            "Spawned particles appended at end of state list. Synchronous (based on current snapshot)."
        ),
        "hidden_rule_fn": fn_src,
        "intervention_api": [
            {"action": "spawn", "params": ["type", "x", "y"], "description": "add a particle at (x,y); no-op if occupied"},
            {"action": "remove", "params": ["x", "y"], "description": "delete particle at (x,y)"},
            {"action": "set_velocity", "params": ["x", "y", "vx", "vy"], "description": "displace particle at (x,y) to ((x+vx)%8,(y+vy)%8) immediately"},
            {"action": "change_type", "params": ["x", "y", "new_type"], "description": "replace the type of the particle at (x,y)"},
        ],
        "train_obs": train_obs,
        "test_obs": test_obs,
    }


# --- world_pt_005: type-A repels B along one axis, C attracts A by chromodistance ---
def _make_world_pt_005(seed=1000):
    """
    3 types: VEXON, BRANT, CLORF on 8x8 toroidal grid.
    krelon_dist(p,q) = |dx|^2 + |dy|^2 + (0 if same type else |dx - dy|)
    Rule:
      VEXON: repelled by nearest BRANT — move away (negate diagonal step). No BRANT -> stay.
      BRANT: attracted to nearest CLORF (diagonal step toward). No CLORF -> stay.
      CLORF: attracted to nearest VEXON (diagonal step toward). No VEXON -> stay.
    Tiebreak: lex (x,y). Distance metric for all: krelon_dist.
    """
    fn_src = (
        "def hidden_rule_fn(state, grid_size=8):\n"
        "    def wrap_delta(a, b):\n"
        "        d = (b - a) % grid_size\n"
        "        return d - grid_size if d > grid_size // 2 else d\n"
        "    def krelon_dist(p, q):\n"
        "        dx = abs(wrap_delta(p['x'], q['x']))\n"
        "        dy = abs(wrap_delta(p['y'], q['y']))\n"
        "        return dx*dx + dy*dy + (0 if p['type']==q['type'] else abs(dx-dy))\n"
        "    def step_toward(p, tgt):\n"
        "        ddx = wrap_delta(p['x'], tgt['x'])\n"
        "        ddy = wrap_delta(p['y'], tgt['y'])\n"
        "        sx = 0 if ddx==0 else (1 if ddx>0 else -1)\n"
        "        sy = 0 if ddy==0 else (1 if ddy>0 else -1)\n"
        "        return sx, sy\n"
        "    nxt = []\n"
        "    for p in state:\n"
        "        t = p['type']\n"
        "        if t == 'VEXON':\n"
        "            targets = [q for q in state if q['type']=='BRANT']\n"
        "            if not targets:\n"
        "                nxt.append(dict(p)); continue\n"
        "            tgt = min(targets, key=lambda q: (krelon_dist(p,q), q['x'],q['y']))\n"
        "            sx, sy = step_toward(p, tgt)\n"
        "            sx, sy = -sx, -sy  # repel = move away\n"
        "            nxt.append({'type':'VEXON','x':(p['x']+sx)%grid_size,'y':(p['y']+sy)%grid_size})\n"
        "        elif t == 'BRANT':\n"
        "            targets = [q for q in state if q['type']=='CLORF']\n"
        "            if not targets:\n"
        "                nxt.append(dict(p)); continue\n"
        "            tgt = min(targets, key=lambda q: (krelon_dist(p,q), q['x'],q['y']))\n"
        "            sx, sy = step_toward(p, tgt)\n"
        "            nxt.append({'type':'BRANT','x':(p['x']+sx)%grid_size,'y':(p['y']+sy)%grid_size})\n"
        "        else:  # CLORF\n"
        "            targets = [q for q in state if q['type']=='VEXON']\n"
        "            if not targets:\n"
        "                nxt.append(dict(p)); continue\n"
        "            tgt = min(targets, key=lambda q: (krelon_dist(p,q), q['x'],q['y']))\n"
        "            sx, sy = step_toward(p, tgt)\n"
        "            nxt.append({'type':'CLORF','x':(p['x']+sx)%grid_size,'y':(p['y']+sy)%grid_size})\n"
        "    return nxt"
    )
    ns = {}
    exec(fn_src, ns)
    rule_fn = ns["hidden_rule_fn"]
    train_obs, test_obs = _make_obs_pt(rule_fn, 5, ["VEXON","BRANT","CLORF"], 8, 12, 6, seed)
    return {
        "id": "world_pt_005",
        "family": "particle_system",
        "difficulty": "medium",
        "primitive_glossary": {
            "VEXON": "first of three particle types",
            "BRANT": "second particle type",
            "CLORF": "third particle type",
            "krelon_dist(p,q)": "|dx|^2 + |dy|^2 + (0 if same type else |dx-dy|) on 8x8 toroidal grid",
        },
        "hidden_rule": (
            "Toroidal 8x8, three types, krelon_dist metric. "
            "VEXON repels (moves away from) nearest BRANT. "
            "BRANT attracts toward nearest CLORF. "
            "CLORF attracts toward nearest VEXON. "
            "Diagonal steps, lex (x,y) tiebreak. Synchronous."
        ),
        "hidden_rule_fn": fn_src,
        "intervention_api": [
            {"action": "spawn", "params": ["type", "x", "y"], "description": "add a particle at (x,y); no-op if occupied"},
            {"action": "remove", "params": ["x", "y"], "description": "delete particle at (x,y)"},
            {"action": "set_velocity", "params": ["x", "y", "vx", "vy"], "description": "displace particle at (x,y) to ((x+vx)%8,(y+vy)%8) immediately"},
            {"action": "change_type", "params": ["x", "y", "new_type"], "description": "replace the type of the particle at (x,y)"},
        ],
        "train_obs": train_obs,
        "test_obs": test_obs,
    }


# ---------------------------------------------------------------------------
# ========== SEQUENCE / PATTERN PUZZLES ==========
# ---------------------------------------------------------------------------

# --- world_seq_001: Thraxgram Sequence (verbatim from SPEC §4c) ---
def _make_world_seq_001():
    fn_src = (
        "def hidden_rule_fn(state):\n"
        "    def is_prime(k):\n"
        "        if k < 2: return False\n"
        "        if k == 2: return True\n"
        "        if k % 2 == 0: return False\n"
        "        for i in range(3, int(k**0.5)+1, 2):\n"
        "            if k % i == 0: return False\n"
        "        return True\n"
        "    n = len(state)\n"
        "    if n < 2:\n"
        "        raise ValueError('state must have at least 2 initial elements')\n"
        "    if is_prime(n):\n"
        "        nxt = (state[n-1] ^ state[n-2]) % 7\n"
        "    elif n % 2 == 0:\n"
        "        nxt = (state[n-1] + state[n-2] + 3) % 7\n"
        "    else:\n"
        "        nxt = (state[n-2] * 2 + 1) % 7\n"
        "    return state + [nxt]"
    )
    train_obs = [
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
        {"state": [4,0,4],            "next_state": [4,0,4,4]},
    ]
    test_obs = [
        {"state": [6,6],              "next_state": [6,6,0]},
        {"state": [6,6,0],            "next_state": [6,6,0,6]},
        {"state": [6,6,0,6],          "next_state": [6,6,0,6,2]},
        {"state": [6,6,0,6,2],        "next_state": [6,6,0,6,2,4]},
        {"state": [6,6,0,6,2,4],      "next_state": [6,6,0,6,2,4,2]},
        {"state": [6,6,0,6,2,4,2],    "next_state": [6,6,0,6,2,4,2,6]},
    ]
    return {
        "id": "world_seq_001",
        "family": "pattern_puzzle",
        "difficulty": "medium",
        "primitive_glossary": {
            "kreels": "the values in the sequence; integers in [0,6]",
            "zark-index": "the position n in the sequence (starting at 0)",
            "thraxgate-index": "a zark-index that is a prime number",
            "ploon-XOR": "bitwise XOR of two kreels, then mod 7",
            "grolp-sum": "arithmetic sum of two kreels plus 3, then mod 7",
            "back-double": "twice the value at zark-index n-2, plus 1, then mod 7",
        },
        "hidden_rule": (
            "Given a sequence (list) of at least 2 kreels, produce the next kreel at zark-index n = len(state). "
            "If n is a thraxgate-index (prime), the next kreel = ploon-XOR(state[n-1], state[n-2]). "
            "Else if n is even, next kreel = grolp-sum(state[n-1], state[n-2]). "
            "Else (n odd and composite), next kreel = back-double(state[n-2]). "
            "The returned next_state is the input state with this next kreel appended."
        ),
        "hidden_rule_fn": fn_src,
        "intervention_api": [
            {"action": "set_element", "params": ["index", "value"], "description": "overwrite state[index] with value"},
            {"action": "insert", "params": ["index", "value"], "description": "insert value at position index, shifting later elements"},
            {"action": "apply_perturbation", "params": ["index", "delta"], "description": "state[index] = (state[index] + delta) % 7"},
        ],
        "train_obs": train_obs,
        "test_obs": test_obs,
    }


# --- world_seq_002: row-parity 2D grid update ---
def _make_world_seq_002(seed=1100):
    """
    2D grid (3x5) of integers mod 5. Represented as a flat list (row-major: [row0_col0, row0_col1, ...]).
    Update: for each cell (r,c) with value v:
      if r is even: new_v = (v + grid[r][(c+1)%5]) % 5
      if r is odd:  new_v = (v XOR grid[(r+1)%3][c]) % 5
    All updates synchronous.
    State is a list of 15 ints.
    """
    fn_src = (
        "def hidden_rule_fn(state):\n"
        "    rows, cols = 3, 5\n"
        "    grid = [state[r*cols:(r+1)*cols] for r in range(rows)]\n"
        "    nxt = []\n"
        "    for r in range(rows):\n"
        "        for c in range(cols):\n"
        "            v = grid[r][c]\n"
        "            if r % 2 == 0:\n"
        "                nv = (v + grid[r][(c+1)%cols]) % 5\n"
        "            else:\n"
        "                nv = (v ^ grid[(r+1)%rows][c]) % 5\n"
        "            nxt.append(nv)\n"
        "    return nxt"
    )
    ns = {}
    exec(fn_src, ns)
    rule_fn = ns["hidden_rule_fn"]
    rng = random.Random(seed)
    train_obs = []
    test_obs = []
    for target, n in [(train_obs, 12), (test_obs, 6)]:
        for _ in range(n):
            s = [rng.randint(0, 4) for _ in range(15)]
            target.append({"state": s, "next_state": rule_fn(s)})
    return {
        "id": "world_seq_002",
        "family": "pattern_puzzle",
        "difficulty": "easy",
        "primitive_glossary": {
            "sprivals": "integers mod 5 filling a 3x5 grid; listed row-major",
            "plex-index": "position in the flat list; (r,c) = (pos//5, pos%5)",
            "even-row-mix": "for even rows: new value = (v + right-neighbour) mod 5",
            "odd-row-mix": "for odd rows: new value = (v XOR cell-below) mod 5",
        },
        "hidden_rule": (
            "A 3x5 grid of sprivals (integers mod 5), stored flat row-major. "
            "Even rows: each cell updates to (v + right-neighbour) mod 5 (toroidal columns). "
            "Odd rows: each cell updates to (v XOR cell-below) mod 5 (toroidal rows). "
            "All updates synchronous."
        ),
        "hidden_rule_fn": fn_src,
        "intervention_api": [
            {"action": "set_element", "params": ["index", "value"], "description": "overwrite flat-list position index with value (0..4)"},
            {"action": "insert", "params": ["index", "value"], "description": "insert value at flat-list position index (not normally useful for fixed-size grid)"},
            {"action": "apply_perturbation", "params": ["index", "delta"], "description": "state[index] = (state[index] + delta) % 5"},
        ],
        "train_obs": train_obs,
        "test_obs": test_obs,
    }


# --- world_seq_003: digit-sum branch ---
def _make_world_seq_003(seed=1200):
    """
    Sequence of integers mod 11. State is a list of at least 2 ints.
    At index n = len(state):
      Let S = sum of digits of n (e.g. n=12 -> S=3).
      If S % 3 == 0: append (state[-1] + state[-2]) % 11
      If S % 3 == 1: append (state[-1] XOR state[-2]) % 11
      If S % 3 == 2: append (state[-1] * state[-2]) % 11
    """
    fn_src = (
        "def hidden_rule_fn(state):\n"
        "    def digit_sum(n):\n"
        "        return sum(int(d) for d in str(n))\n"
        "    n = len(state)\n"
        "    S = digit_sum(n)\n"
        "    if S % 3 == 0:\n"
        "        nxt = (state[-1] + state[-2]) % 11\n"
        "    elif S % 3 == 1:\n"
        "        nxt = (state[-1] ^ state[-2]) % 11\n"
        "    else:\n"
        "        nxt = (state[-1] * state[-2]) % 11\n"
        "    return state + [nxt]"
    )
    ns = {}
    exec(fn_src, ns)
    rule_fn = ns["hidden_rule_fn"]
    train_obs, test_obs = _make_obs_seq(rule_fn, 2, 11, 12, 6, seed)
    return {
        "id": "world_seq_003",
        "family": "pattern_puzzle",
        "difficulty": "medium",
        "primitive_glossary": {
            "luxels": "integers mod 11 in the sequence",
            "noxel-index": "the position n (starting at 0); the index whose digit-sum governs the next step",
            "splid-sum": "the sum of decimal digits of the noxel-index n",
            "blorf-add": "append (last + second-last) mod 11 — used when splid-sum mod 3 == 0",
            "blorf-XOR": "append (last XOR second-last) mod 11 — used when splid-sum mod 3 == 1",
            "blorf-mul": "append (last * second-last) mod 11 — used when splid-sum mod 3 == 2",
        },
        "hidden_rule": (
            "Sequence of luxels (mod 11). At each step appending position n = len(state): "
            "compute splid-sum S = digit_sum(n). "
            "If S%3==0: blorf-add. If S%3==1: blorf-XOR. If S%3==2: blorf-mul. "
            "Result appended to state."
        ),
        "hidden_rule_fn": fn_src,
        "intervention_api": [
            {"action": "set_element", "params": ["index", "value"], "description": "overwrite state[index] with value (0..10)"},
            {"action": "insert", "params": ["index", "value"], "description": "insert value at position index"},
            {"action": "apply_perturbation", "params": ["index", "delta"], "description": "state[index] = (state[index] + delta) % 11"},
        ],
        "train_obs": train_obs,
        "test_obs": test_obs,
    }


# --- world_seq_004: two interleaved sequences that cross-influence ---
def _make_world_seq_004(seed=1300):
    """
    State is a list of even length. Two interleaved sub-sequences: A = state[0::2], B = state[1::2].
    At each step, ONE element is appended to produce next_state of length len(state)+1.
    The appended element's role alternates: if len(state) is even, next goes to A (appended at even position);
    if odd, goes to B.
    Let La = len(A), Lb = len(B) in the CURRENT state.
    If appending to A (even len(state)):
      new_a = (A[-1] + B[-1]) % 8 if B else A[-1]
    If appending to B (odd len(state)):
      new_b = (A[-1] XOR B[-1] XOR La) % 8
    Result: state + [new_element]
    State must have at least 2 elements (one of each).
    """
    fn_src = (
        "def hidden_rule_fn(state):\n"
        "    n = len(state)\n"
        "    if n < 2:\n"
        "        raise ValueError('state needs at least 2 elements')\n"
        "    A = state[0::2]  # indices 0,2,4,...\n"
        "    B = state[1::2]  # indices 1,3,5,...\n"
        "    La = len(A)\n"
        "    if n % 2 == 0:  # append to A\n"
        "        new_val = (A[-1] + B[-1]) % 8 if B else A[-1]\n"
        "    else:  # append to B\n"
        "        new_val = (A[-1] ^ B[-1] ^ La) % 8\n"
        "    return state + [new_val]"
    )
    ns = {}
    exec(fn_src, ns)
    rule_fn = ns["hidden_rule_fn"]
    train_obs, test_obs = _make_obs_seq(rule_fn, 2, 8, 12, 6, seed)
    return {
        "id": "world_seq_004",
        "family": "pattern_puzzle",
        "difficulty": "medium",
        "primitive_glossary": {
            "jrels": "integers mod 8 in the interleaved sequence",
            "A-strand": "even-indexed jrels: positions 0,2,4,...",
            "B-strand": "odd-indexed jrels: positions 1,3,5,...",
            "strand-sum": "(last-A + last-B) mod 8 — used when appending to A-strand",
            "strand-XOR": "(last-A XOR last-B XOR len-A) mod 8 — used when appending to B-strand",
        },
        "hidden_rule": (
            "Two interleaved strands A (even positions) and B (odd positions). "
            "Append one jrel per step, alternating strands. "
            "If appending to A: new = (A[-1] + B[-1]) mod 8. "
            "If appending to B: new = (A[-1] XOR B[-1] XOR len(A)) mod 8. "
            "len(A) is counted from the CURRENT state (before appending)."
        ),
        "hidden_rule_fn": fn_src,
        "intervention_api": [
            {"action": "set_element", "params": ["index", "value"], "description": "overwrite state[index] with value (0..7)"},
            {"action": "insert", "params": ["index", "value"], "description": "insert value at position index"},
            {"action": "apply_perturbation", "params": ["index", "delta"], "description": "state[index] = (state[index] + delta) % 8"},
        ],
        "train_obs": train_obs,
        "test_obs": test_obs,
    }


# --- world_seq_005: index-congruence multi-branch ---
def _make_world_seq_005(seed=1400):
    """
    Sequence of ints mod 13. State is a list of at least 3 elements.
    At position n = len(state), compute r = n % 4:
      r==0: append (state[-1] + state[-3]) % 13
      r==1: append (state[-1] XOR state[-2] XOR state[-3]) % 13
      r==2: append (state[-2] * 3 + state[-1]) % 13
      r==3: append (state[-1] - state[-3]) % 13
    """
    fn_src = (
        "def hidden_rule_fn(state):\n"
        "    if len(state) < 3:\n"
        "        raise ValueError('need at least 3 elements')\n"
        "    n = len(state)\n"
        "    r = n % 4\n"
        "    if r == 0:\n"
        "        nxt = (state[-1] + state[-3]) % 13\n"
        "    elif r == 1:\n"
        "        nxt = (state[-1] ^ state[-2] ^ state[-3]) % 13\n"
        "    elif r == 2:\n"
        "        nxt = (state[-2] * 3 + state[-1]) % 13\n"
        "    else:\n"
        "        nxt = (state[-1] - state[-3]) % 13\n"
        "    return state + [nxt]"
    )
    ns = {}
    exec(fn_src, ns)
    rule_fn = ns["hidden_rule_fn"]
    train_obs, test_obs = _make_obs_seq(rule_fn, 3, 13, 12, 6, seed)
    return {
        "id": "world_seq_005",
        "family": "pattern_puzzle",
        "difficulty": "easy",
        "primitive_glossary": {
            "luxons": "integers mod 13 in the sequence",
            "warp-index": "the appending position n = len(state)",
            "warp-class": "n mod 4 — determines which blending rule applies",
            "far-skip": "state[-3]: the luxon three positions back",
        },
        "hidden_rule": (
            "Sequence of luxons (mod 13). At each step n = len(state), warp-class = n mod 4. "
            "class 0: append (last + far-skip) mod 13. "
            "class 1: append (last XOR second XOR far-skip) mod 13. "
            "class 2: append (second * 3 + last) mod 13. "
            "class 3: append (last - far-skip) mod 13."
        ),
        "hidden_rule_fn": fn_src,
        "intervention_api": [
            {"action": "set_element", "params": ["index", "value"], "description": "overwrite state[index] with value (0..12)"},
            {"action": "insert", "params": ["index", "value"], "description": "insert value at position index"},
            {"action": "apply_perturbation", "params": ["index", "delta"], "description": "state[index] = (state[index] + delta) % 13"},
        ],
        "train_obs": train_obs,
        "test_obs": test_obs,
    }


# --- world_seq_006: bitshift + arithmetic hybrid ---
def _make_world_seq_006(seed=1500):
    """
    Sequence of ints mod 16. State at least 2 elements.
    At position n = len(state):
      Let v = state[-1], w = state[-2].
      If n is a perfect square: append (v >> 1) ^ w) % 16
      Elif n % 3 == 0: append (v + w + n) % 16
      Else: append ((v << 1) & 0xF) | (w & 1)   (keep in 4 bits)
    """
    fn_src = (
        "def hidden_rule_fn(state):\n"
        "    import math\n"
        "    def is_perfect_square(k):\n"
        "        if k < 0: return False\n"
        "        r = int(math.isqrt(k))\n"
        "        return r * r == k\n"
        "    n = len(state)\n"
        "    v = state[-1]; w = state[-2]\n"
        "    if is_perfect_square(n):\n"
        "        nxt = ((v >> 1) ^ w) % 16\n"
        "    elif n % 3 == 0:\n"
        "        nxt = (v + w + n) % 16\n"
        "    else:\n"
        "        nxt = ((v << 1) & 0xF) | (w & 1)\n"
        "    return state + [nxt]"
    )
    ns = {}
    exec(fn_src, ns)
    rule_fn = ns["hidden_rule_fn"]
    train_obs, test_obs = _make_obs_seq(rule_fn, 2, 16, 12, 6, seed)
    return {
        "id": "world_seq_006",
        "family": "pattern_puzzle",
        "difficulty": "hard",
        "primitive_glossary": {
            "nibrels": "4-bit integers (0..15) in the sequence",
            "thrax-square": "a position index that is a perfect square (0, 1, 4, 9, 16, ...)",
            "half-xor": "(last >> 1) XOR second: right-shift last by 1 then XOR with second — used at thrax-square positions",
            "index-sum": "(last + second + n) mod 16 — used when n is divisible by 3 (non-thrax-square)",
            "shift-merge": "((last << 1) & 0xF) | (second & 1): left-shift last into 4 bits, then OR with parity bit of second",
        },
        "hidden_rule": (
            "Sequence of nibrels (mod 16). At position n = len(state): "
            "if n is a perfect square: half-xor. "
            "elif n divisible by 3: index-sum. "
            "else: shift-merge. "
            "Priority: perfect-square check first."
        ),
        "hidden_rule_fn": fn_src,
        "intervention_api": [
            {"action": "set_element", "params": ["index", "value"], "description": "overwrite state[index] with value (0..15)"},
            {"action": "insert", "params": ["index", "value"], "description": "insert value at position index"},
            {"action": "apply_perturbation", "params": ["index", "delta"], "description": "state[index] = (state[index] + delta) % 16"},
        ],
        "train_obs": train_obs,
        "test_obs": test_obs,
    }


# ---------------------------------------------------------------------------
# generate_all and sample_instance
# ---------------------------------------------------------------------------

def generate_all() -> list:
    instances = [
        # Cellular automata (6 instances)
        _make_world_ca_001(),
        _make_world_ca_002(),
        _make_world_ca_003(),
        _make_world_ca_004(),
        _make_world_ca_005(),
        _make_world_ca_006(),
        # Particle systems (5 instances)
        _make_world_pt_001(),
        _make_world_pt_002(),
        _make_world_pt_003(),
        _make_world_pt_004(),
        _make_world_pt_005(),
        # Sequence / pattern puzzles (6 instances)
        _make_world_seq_001(),
        _make_world_seq_002(),
        _make_world_seq_003(),
        _make_world_seq_004(),
        _make_world_seq_005(),
        _make_world_seq_006(),
    ]
    return instances


_FAMILY_GENERATORS = {
    "cellular_automata": [
        (100, _make_world_ca_001),
        (200, _make_world_ca_002),
        (300, _make_world_ca_003),
        (400, _make_world_ca_004),
        (500, _make_world_ca_005),
        (600, _make_world_ca_006),
    ],
    "particle_system": [
        (700, _make_world_pt_001),
        (800, _make_world_pt_002),
        (900, _make_world_pt_003),
        (1000, _make_world_pt_004),
        (1100, _make_world_pt_005),
    ],
    "pattern_puzzle": [
        (1200, _make_world_seq_001),
        (1300, _make_world_seq_002),
        (1400, _make_world_seq_003),
        (1500, _make_world_seq_004),
        (1600, _make_world_seq_005),
        (1700, _make_world_seq_006),
    ],
}


def sample_instance(family: str, seed: int) -> dict:
    if family not in _FAMILY_GENERATORS:
        raise ValueError(f"Unknown family: {family!r}. Must be one of {FAMILIES}")
    gens = _FAMILY_GENERATORS[family]
    idx = seed % len(gens)
    _, gen_fn = gens[idx]
    return gen_fn()
