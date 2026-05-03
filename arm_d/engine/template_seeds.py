"""
Family-specific canonical templates for the mock generator.

Each template is a parameterised string-formatter that, given a dict of
parameter assignments, returns a Python source for `hidden_rule_fn`.

The templates intentionally mirror the *structural shape* of the v2
hidden rules (branch-on-cell-state with neighbourhood-count features for
CA; type-conditional move toward/away for PT; index-predicate-branched
recurrences for SEQ) without copying any specific instance's exact rule.
The mock generator's job is to expand each template across a parameter
grid, producing 50-200 candidates per family.

Templates are pure-stdlib Python; they import only `copy` and `math`.
The linter validates that.
"""

# ----------------------------------------------------------------------
# CA family templates
# ----------------------------------------------------------------------

# --- Neighbourhood definitions used across CA templates ---
NEIGHBOURHOOD_OFFSETS = {
    "moore8":     [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)],
    "vonneumann": [(-1,0),(1,0),(0,-1),(0,1)],
    "diagonal":   [(-1,-1),(-1,1),(1,-1),(1,1)],
    "knight":     [(-2,-1),(-2,1),(-1,-2),(-1,2),(1,-2),(1,2),(2,-1),(2,1)],
    "axial2":     [(-2,0),(2,0),(0,-2),(0,2)],
    "row_only":   [(0,-1),(0,1)],
    "col_only":   [(-1,0),(1,0)],
    "row1d_short": [(-3,)],  # placeholder; templates that use 1D handle this
}


def ca_branch_count_rule(*, nbh_name: str, n_states: int, branches: list[dict]) -> str:
    """
    Template: per-cell rule branching on (current_state, count_of_state_X_in_neighbourhood)
    against a small set of count thresholds.

    `branches` is a list of dicts:
      {"cur": int, "feature": "count[s]"|"parity[s]"|"sum_mod_k", "op": "==|>=|>",
       "rhs": int, "then_state": int, "else_state": int}

    The first matching branch is taken (`if/elif/else` chain); the final
    fallback is the cell's current state.
    """
    offsets = NEIGHBOURHOOD_OFFSETS[nbh_name]
    branch_lines = []
    for i, b in enumerate(branches):
        keyword = "if" if i == 0 else "elif"
        feature = b["feature"]
        op = b["op"]
        rhs = b["rhs"]
        then_state = b["then_state"]
        cur = b["cur"]
        if feature.startswith("count["):
            s = int(feature[6:-1])
            cond = f"cur == {cur} and counts[{s}] {op} {rhs}"
        elif feature.startswith("parity["):
            s = int(feature[7:-1])
            cond = f"cur == {cur} and counts[{s}] % 2 {op} {rhs}"
        elif feature == "sum_mod_k":
            k = b.get("k", n_states)
            cond = f"cur == {cur} and (sum(nb) % {k}) {op} {rhs}"
        elif feature == "distinct":
            cond = f"cur == {cur} and len(set(nb)) {op} {rhs}"
        else:
            cond = f"cur == {cur}"
        branch_lines.append(f"            {keyword} {cond}: nxt[r][c] = {then_state}")
    branches_text = "\n".join(branch_lines) if branch_lines else "            pass"
    return f"""def hidden_rule_fn(state):
    rows = len(state); cols = len(state[0])
    OFFS = {offsets!r}
    nxt = [[0]*cols for _ in range(rows)]
    for r in range(rows):
        for c in range(cols):
            nb = [state[(r+dr)%rows][(c+dc)%cols] for dr,dc in OFFS]
            counts = [nb.count(s) for s in range({n_states})]
            cur = state[r][c]
{branches_text}
            else: nxt[r][c] = cur
    return nxt
"""


def ca_position_dependent_rule(*, n_states: int, mod_choice: int = 3) -> str:
    """
    Template: per-cell rule whose neighbourhood depends on (r+c) % mod.
    Captures rules like world_ca_006.
    """
    return f"""def hidden_rule_fn(state):
    rows = len(state); cols = len(state[0])
    diag = [(-1,-1),(-1,1),(1,-1),(1,1)]
    vn = [(-1,0),(1,0),(0,-1),(0,1)]
    axial2 = [(-2,0),(2,0),(0,-2),(0,2)]
    nxt = [[0]*cols for _ in range(rows)]
    for r in range(rows):
        for c in range(cols):
            m = (r + c) % {mod_choice}
            if m == 0: offs = diag
            elif m == 1: offs = vn
            else: offs = axial2
            nb = [state[(r+dr)%rows][(c+dc)%cols] for dr,dc in offs]
            counts = [nb.count(s) for s in range({n_states})]
            cur = state[r][c]
            # Majority over chosen neighbourhood.
            best = cur
            best_n = counts[cur] if cur < {n_states} else -1
            for s in range({n_states}):
                if counts[s] > best_n:
                    best_n = counts[s]; best = s
            nxt[r][c] = best
    return nxt
"""


def ca_row_alternating_rule(*, n_states: int) -> str:
    """
    Per-row alternating neighbourhood (row-only on even rows, col-only on odd).
    Captures the world_ca_005 invention class.
    """
    return f"""def hidden_rule_fn(state):
    rows = len(state); cols = len(state[0])
    nxt = [[0]*cols for _ in range(rows)]
    for r in range(rows):
        for c in range(cols):
            if r % 2 == 0:
                nb = [state[r][(c+1)%cols], state[r][(c-1)%cols]]
            else:
                nb = [state[(r-1)%rows][c], state[(r+1)%rows][c]]
            cur = state[r][c]
            # Mode of neighbours.
            counts = [nb.count(s) for s in range({n_states})]
            best = cur
            best_n = -1
            for s in range({n_states}):
                if counts[s] > best_n:
                    best_n = counts[s]; best = s
            nxt[r][c] = best
    return nxt
"""


def ca_1d_radius_rule(*, n_states: int, radius_offsets: list[int]) -> str:
    """
    1D CA over a single row (state is [[...]]); covers world_ca_004.
    """
    return f"""def hidden_rule_fn(state):
    cols = len(state[0])
    OFFS = {radius_offsets!r}
    nxt = [[0]*cols]
    for c in range(cols):
        nb = [state[0][(c+o)%cols] for o in OFFS]
        counts = [nb.count(s) for s in range({n_states})]
        s = sum(nb) % {n_states}
        cur = state[0][c]
        # Pick next based on (sum mod n_states); a soft majority fallback.
        if counts[s] > counts[cur]:
            nxt[0][c] = s
        else:
            nxt[0][c] = cur
    return nxt
"""


def ca_identity() -> str:
    """The trivial baseline. Submitted only as a last resort (filtered out
    by lint? No — `return state` is valid Python. The linter does not
    forbid it. The selector ranks it by train_acc, which is usually 0
    on CA, so it loses to anything with a real branch."""
    return "def hidden_rule_fn(state):\n    return [list(row) for row in state]\n"


# ----------------------------------------------------------------------
# PT family templates
# ----------------------------------------------------------------------

DISTANCE_FNS = {
    "chebyshev_typed": (
        "def _dist(p, q, gs):\n"
        "    dx = abs(_wd(p['x'], q['x'], gs)); dy = abs(_wd(p['y'], q['y'], gs))\n"
        "    return max(dx, dy) + (0 if p['type'] == q['type'] else 1)\n"
    ),
    "manhattan_typed": (
        "def _dist(p, q, gs):\n"
        "    dx = abs(_wd(p['x'], q['x'], gs)); dy = abs(_wd(p['y'], q['y'], gs))\n"
        "    return dx + dy + (0 if p['type'] == q['type'] else 1)\n"
    ),
    "weighted_xy_typed": (
        "def _dist(p, q, gs):\n"
        "    dx = abs(_wd(p['x'], q['x'], gs)); dy = abs(_wd(p['y'], q['y'], gs))\n"
        "    if p['type'] != q['type']:\n"
        "        return dx + 2 * dy\n"
        "    return dx * dy + 1\n"
    ),
    "squared_typed": (
        "def _dist(p, q, gs):\n"
        "    dx = abs(_wd(p['x'], q['x'], gs)); dy = abs(_wd(p['y'], q['y'], gs))\n"
        "    return dx*dx + dy*dy + (0 if p['type'] == q['type'] else abs(dx-dy))\n"
    ),
    "chebyshev_plain": (
        "def _dist(p, q, gs):\n"
        "    dx = abs(_wd(p['x'], q['x'], gs)); dy = abs(_wd(p['y'], q['y'], gs))\n"
        "    return max(dx, dy)\n"
    ),
}

# How a particle moves once a target is selected.
MOVE_RULES = {
    "step_both_axes": (
        "def _mv(p, tgt, gs, sign):\n"
        "    ddx = _wd(p['x'], tgt['x'], gs); ddy = _wd(p['y'], tgt['y'], gs)\n"
        "    sx = 0 if ddx == 0 else (1 if ddx > 0 else -1)\n"
        "    sy = 0 if ddy == 0 else (1 if ddy > 0 else -1)\n"
        "    return ((p['x'] + sign*sx) % gs, (p['y'] + sign*sy) % gs)\n"
    ),
    "step_larger_axis": (
        "def _mv(p, tgt, gs, sign):\n"
        "    ddx = _wd(p['x'], tgt['x'], gs); ddy = _wd(p['y'], tgt['y'], gs)\n"
        "    if abs(ddx) >= abs(ddy):\n"
        "        sx = 0 if ddx == 0 else (1 if ddx > 0 else -1); sy = 0\n"
        "    else:\n"
        "        sx = 0; sy = 0 if ddy == 0 else (1 if ddy > 0 else -1)\n"
        "    return ((p['x'] + sign*sx) % gs, (p['y'] + sign*sy) % gs)\n"
    ),
    "step_smaller_axis": (
        "def _mv(p, tgt, gs, sign):\n"
        "    ddx = _wd(p['x'], tgt['x'], gs); ddy = _wd(p['y'], tgt['y'], gs)\n"
        "    if abs(ddx) <= abs(ddy):\n"
        "        sx = 0 if ddx == 0 else (1 if ddx > 0 else -1); sy = 0\n"
        "    else:\n"
        "        sx = 0; sy = 0 if ddy == 0 else (1 if ddy > 0 else -1)\n"
        "    return ((p['x'] + sign*sx) % gs, (p['y'] + sign*sy) % gs)\n"
    ),
    "stay_still": (
        "def _mv(p, tgt, gs, sign):\n"
        "    return (p['x'], p['y'])\n"
    ),
}


def _indent(src: str, prefix: str = "    ") -> str:
    """Re-indent a multi-line source string."""
    return "\n".join(prefix + line if line.strip() else line for line in src.split("\n"))


def pt_two_type_rule(
    *,
    type_a: str = "ZEX",
    type_b: str = "GLORP",
    distance: str = "chebyshev_typed",
    a_seek_target_type: str = "same",   # 'same' or 'cross'
    a_move: str = "step_both_axes",
    a_sign: int = 1,                    # +1 toward, -1 away
    b_seek_target_type: str = "cross",
    b_move: str = "step_larger_axis",
    b_sign: int = -1,
    b_move_logic: str | None = None,    # if set, use a different move rule for type B
    grid_size: int = 8,
) -> str:
    """Construct a particle rule by composing distance + per-type move logic."""
    dist_src = _indent(DISTANCE_FNS[distance].rstrip("\n"))
    a_move_src = _indent(MOVE_RULES[a_move].rstrip("\n").replace("def _mv(", "def _mv_a("))
    b_logic = b_move_logic or a_move
    b_move_src = _indent(MOVE_RULES[b_logic].rstrip("\n").replace("def _mv(", "def _mv_b("))
    return f"""def hidden_rule_fn(state, grid_size={grid_size}):
    def _wd(a, b, gs):
        d = (b - a) % gs
        return d - gs if d > gs // 2 else d
{dist_src}
{a_move_src}
{b_move_src}
    nxt = []
    for i, p in enumerate(state):
        if p['type'] == {type_a!r}:
            target_type = {type_a!r} if {a_seek_target_type!r} == 'same' else {type_b!r}
            cands = [q for j, q in enumerate(state) if q['type'] == target_type and j != i]
            if not cands:
                nxt.append(dict(p)); continue
            tgt = min(cands, key=lambda q: (_dist(p, q, grid_size), q['x'], q['y']))
            nx, ny = _mv_a(p, tgt, grid_size, {a_sign})
            nxt.append({{'type': p['type'], 'x': nx, 'y': ny}})
        elif p['type'] == {type_b!r}:
            target_type = {type_a!r} if {b_seek_target_type!r} == 'cross' else {type_b!r}
            cands = [q for j, q in enumerate(state) if q['type'] == target_type and j != i]
            if not cands:
                nxt.append(dict(p)); continue
            tgt = min(cands, key=lambda q: (_dist(p, q, grid_size), q['x'], q['y']))
            nx, ny = _mv_b(p, tgt, grid_size, {b_sign})
            nxt.append({{'type': p['type'], 'x': nx, 'y': ny}})
        else:
            nxt.append(dict(p))
    return nxt
"""


def pt_identity() -> str:
    return ("def hidden_rule_fn(state):\n"
            "    return [dict(p) for p in state]\n")


# ----------------------------------------------------------------------
# SEQ family templates
# ----------------------------------------------------------------------

GUARDS = {
    "is_prime": (
        "def _is_prime(k):\n"
        "    if k < 2: return False\n"
        "    if k == 2: return True\n"
        "    if k % 2 == 0: return False\n"
        "    for i in range(3, int(k**0.5)+1, 2):\n"
        "        if k % i == 0: return False\n"
        "    return True\n"
    ),
    "is_perfect_square": (
        "def _is_psq(k):\n"
        "    if k < 0: return False\n"
        "    r = int(k**0.5)\n"
        "    return r * r == k\n"
    ),
    "digit_sum_mod3": (
        "def _ds(k):\n"
        "    return sum(int(d) for d in str(k)) % 3\n"
    ),
}

RHS_FORMS = [
    "(state[-1] + state[-2]) % {mod}",
    "(state[-1] ^ state[-2]) % {mod}",
    "(state[-1] * state[-2]) % {mod}",
    "(state[-1] + state[-2] + 3) % {mod}",
    "(state[-2] * 2 + 1) % {mod}",
    "(state[-1] - state[-3]) % {mod}",
    "(state[-1] + state[-3]) % {mod}",
    "((state[-1] << 1) & ({mod}-1)) | (state[-2] & 1)",
    "(state[-1] >> 1) ^ state[-2]",
]


def seq_branched_recurrence(
    *,
    branches: list[tuple[str, str]],   # [(guard_expr, rhs_expr), ...]
    helper_guard: str | None = None,   # e.g. "is_prime"
    mod: int = 7,
) -> str:
    """
    Build a sequence rule that branches on n = len(state).
    Each branch is (guard_expr, rhs_expr). The chain is if/elif/else.
    The else branch is the LAST tuple's rhs.
    """
    helper_src = ""
    if helper_guard and helper_guard in GUARDS:
        helper_src = GUARDS[helper_guard]
    branch_lines = []
    for i, (g, r) in enumerate(branches):
        rhs_filled = r.replace("{mod}", str(mod))
        if g == "else":
            branch_lines.append(f"    else: nxt = {rhs_filled}")
        else:
            kw = "if" if i == 0 else "elif"
            branch_lines.append(f"    {kw} {g}: nxt = {rhs_filled}")
    return f"""def hidden_rule_fn(state):
{helper_src}
    n = len(state)
    if n < 2:
        return list(state) + [0]
{chr(10).join(branch_lines)}
    return list(state) + [nxt]
"""


def seq_grid_rule(*, rows: int = 3, cols: int = 5, mod: int = 5) -> str:
    """world_seq_002-style grid rule (3 rows × 5 cols flat list)."""
    return f"""def hidden_rule_fn(state):
    rows, cols = {rows}, {cols}
    if len(state) != rows * cols:
        return list(state)
    grid = [state[r*cols:(r+1)*cols] for r in range(rows)]
    nxt = []
    for r in range(rows):
        for c in range(cols):
            v = grid[r][c]
            if r % 2 == 0:
                nv = (v + grid[r][(c+1)%cols]) % {mod}
            else:
                nv = (v ^ grid[(r+1)%rows][c]) % {mod}
            nxt.append(nv)
    return nxt
"""


def seq_strand_rule(*, mod: int = 8) -> str:
    """world_seq_004-style: split state into A/B strands by index parity."""
    return f"""def hidden_rule_fn(state):
    n = len(state)
    if n < 2:
        return list(state) + [0]
    A = state[0::2]; B = state[1::2]
    La = len(A); Lb = len(B)
    last_a = A[-1] if A else 0
    last_b = B[-1] if B else 0
    if n % 2 == 0:
        new = (last_a + last_b) % {mod} if B else last_a
    else:
        new = (last_a ^ last_b ^ La) % {mod}
    return list(state) + [new]
"""


def seq_window_rule(*, mod: int = 13) -> str:
    """world_seq_005-style: branch on n % 4 with multi-element windows."""
    return f"""def hidden_rule_fn(state):
    n = len(state)
    if len(state) < 3:
        return list(state) + [0]
    wc = n % 4
    if wc == 3: new = (state[-1] - state[-3]) % {mod}
    elif wc == 0: new = (state[-1] + state[-3]) % {mod}
    elif wc == 1: new = (state[-3] - state[-2] - state[-1]) % {mod}
    else: new = (3 * state[-2] + state[-1]) % {mod}
    return list(state) + [new]
"""


def seq_identity() -> str:
    return "def hidden_rule_fn(state):\n    return list(state) + [0]\n"
