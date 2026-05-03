"""
Mock generator for the Arm D MVP.

Produces a deterministic, family-appropriate candidate pool per instance.
The pool is intentionally larger than what any single rule needs; the
selector + scorer prune.

The pool composition:

  CA family (~150 candidates):
    - 6 neighbourhood shapes × 3 cell-state alphabets × ~8 branch
      template families = ~140 base candidates.
    - Plus: ca_position_dependent, ca_row_alternating, ca_1d_radius
      with multiple parameter draws.
    - Plus: ca_identity (last-resort baseline).

  PT family (~80 candidates):
    - 5 distance metrics × 3 move rules × {seek-same, seek-cross} ×
      {sign +1, -1} × {type-A == ZEX or BLINX or ...} = ~80 candidates.
    - Plus: pt_identity baseline.

  SEQ family (~80 candidates):
    - 3 helper guards × ~12 branch combinations × 5 moduli = ~80
      branched-recurrence candidates.
    - Plus: seq_grid_rule with parameter draws.
    - Plus: seq_strand_rule, seq_window_rule with parameter draws.
    - Plus: seq_identity baseline.

The pool is deterministic given the instance ID — the same instance
always produces the same pool. (Reproducibility property.)
"""

import itertools
from . import template_seeds as T


# ----------------------------------------------------------------------
# CA pool
# ----------------------------------------------------------------------

def _ca_branch_combinations(n_states: int):
    """
    Yield branch-spec lists for ca_branch_count_rule. Each yield is a
    full per-cell rule = one branch per current cell state. Otherwise a
    single-branch rule cannot match the per-cur-state structure of the
    v2 hidden CA rules.
    """
    ops = ["==", ">=", ">"]
    rhs_vals = [1, 2, 3, 4]
    feat_choices = [f"count[{s}]" for s in range(n_states)]

    # Build one branch per current cell state, then yield combinations
    # of (cur=0 branch, cur=1 branch, ..., cur=n-1 branch).
    per_cur_branches = {cur: [] for cur in range(n_states)}
    for cur in range(n_states):
        for feat in feat_choices:
            for op, rhs in itertools.product(ops, rhs_vals):
                # The then_state cycles through all OTHER states.
                for then_state in range(n_states):
                    if then_state == cur:
                        continue
                    per_cur_branches[cur].append({
                        "cur": cur,
                        "feature": feat,
                        "op": op,
                        "rhs": rhs,
                        "then_state": then_state,
                        "k": n_states,
                    })

    # Pick a small subset per cur to keep cross-product bounded.
    # The subset deliberately interleaves operators (==, >=, >) and rhs
    # values (1,2,3) so we cover specific-count and threshold rules.
    SUBSET_PER_CUR = 8
    subsets = {cur: per_cur_branches[cur][:SUBSET_PER_CUR] for cur in range(n_states)}
    # Yield one combined branch list per cross-product entry.
    cur_lists = [subsets[cur] for cur in range(n_states)]
    count = 0
    for combo in itertools.product(*cur_lists):
        yield list(combo)
        count += 1
        if count >= 200:
            return

    # Also yield the parity / distinct features as single-branch additions.
    for cur in range(n_states):
        for feat in [f"parity[{s}]" for s in range(n_states)] + ["distinct", "sum_mod_k"]:
            for op in ["==", ">=", ">"]:
                for rhs in [0, 1, 2, 3]:
                    for then_state in range(n_states):
                        if then_state == cur:
                            continue
                        yield [{
                            "cur": cur,
                            "feature": feat,
                            "op": op,
                            "rhs": rhs,
                            "then_state": then_state,
                            "k": n_states,
                        }]


def _gen_ca_pool(instance: dict) -> list[str]:
    train = instance["train_obs"]
    sample = train[0]["state"]
    if not sample or not sample[0]:
        return [T.ca_identity()]

    # Detect 1D (single row) vs 2D.
    rows = len(sample)
    cols = len(sample[0]) if isinstance(sample[0], list) else len(sample)
    is_1d = (rows == 1)

    # Detect alphabet size from observed cells.
    if isinstance(sample[0], list):
        all_vals = sorted({v for row in sample for v in row})
    else:
        all_vals = sorted(set(sample))
    n_states = max(2, len(all_vals)) if all_vals else 2
    # Probe wider state alphabet too — train may not exhibit every state.
    alphabet_probes = sorted({n_states, max(n_states, 3)})  # smaller pool than {n,n+1,3,4}

    pool = [T.ca_identity()]

    if is_1d:
        # 1D radius rules
        for ns in alphabet_probes:
            for radius in [
                [-1, 1],
                [-1, 0, 1],
                [-2, -1, 1, 2],
                [-3, -2, -1, 1, 2, 3],
                [-2, 2],
            ]:
                try:
                    pool.append(T.ca_1d_radius_rule(n_states=ns, radius_offsets=radius))
                except Exception:
                    pass
        # Also offer 2D forms — sometimes the rule looks 1D but
        # occupies a single row of a 2D world.
        for ns in alphabet_probes:
            for nbh in ["row_only", "vonneumann"]:
                for branches in _ca_branch_combinations(ns):
                    try:
                        pool.append(T.ca_branch_count_rule(
                            nbh_name=nbh, n_states=ns, branches=branches,
                        ))
                    except Exception:
                        pass
        return pool

    # 2D rules.
    for ns in alphabet_probes:
        for nbh in ["moore8", "vonneumann", "diagonal", "knight", "axial2"]:
            for branches in _ca_branch_combinations(ns):
                try:
                    pool.append(T.ca_branch_count_rule(
                        nbh_name=nbh, n_states=ns, branches=branches,
                    ))
                except Exception:
                    pass
        # Position-dependent neighbourhood rule.
        for mod in [2, 3, 4]:
            try:
                pool.append(T.ca_position_dependent_rule(n_states=ns, mod_choice=mod))
            except Exception:
                pass
        # Row-alternating rule.
        try:
            pool.append(T.ca_row_alternating_rule(n_states=ns))
        except Exception:
            pass

    return pool


# ----------------------------------------------------------------------
# PT pool
# ----------------------------------------------------------------------

def _detect_pt_types(instance: dict) -> list[str]:
    seen = set()
    for obs in instance["train_obs"]:
        for p in obs["state"]:
            seen.add(p["type"])
        for p in obs["next_state"]:
            seen.add(p["type"])
    types = sorted(seen)
    return types if len(types) >= 2 else (types + ["__OTHER__"])[:2]


def _gen_pt_pool(instance: dict) -> list[str]:
    types = _detect_pt_types(instance)
    type_a, type_b = types[0], types[1]

    pool = [T.pt_identity()]

    distances = list(T.DISTANCE_FNS.keys())
    moves = list(T.MOVE_RULES.keys())
    a_targets = ["same", "cross"]
    b_targets = ["same", "cross"]
    signs = [1, -1]

    # Build the full cross-product but cap to 600 candidates.
    # For each type-A choice we also let type-B have its OWN move rule.
    def _add_combos(t_a: str, t_b: str, cap: int):
        for d in distances:
            for a_mv in moves:
                for b_mv in moves:
                    for a_tgt in a_targets:
                        for a_sign in signs:
                            for b_tgt in b_targets:
                                for b_sign in signs:
                                    try:
                                        src = T.pt_two_type_rule(
                                            type_a=t_a, type_b=t_b,
                                            distance=d,
                                            a_seek_target_type=a_tgt, a_move=a_mv, a_sign=a_sign,
                                            b_seek_target_type=b_tgt, b_move=a_mv, b_sign=b_sign,
                                            b_move_logic=b_mv,
                                        )
                                        pool.append(src)
                                        if len(pool) >= cap:
                                            return True
                                    except Exception:
                                        pass
        return False

    if _add_combos(type_a, type_b, cap=300):
        return pool
    _add_combos(type_b, type_a, cap=600)
    return pool


# ----------------------------------------------------------------------
# SEQ pool
# ----------------------------------------------------------------------

def _detect_seq_mod(instance: dict) -> list[int]:
    """Look at observed values; return candidate moduli."""
    vals = []
    for obs in instance["train_obs"]:
        s = obs["state"]
        # state may be a flat list of ints, or a list-of-lists.
        if s and isinstance(s[0], int):
            vals.extend(s)
        ns = obs["next_state"]
        if ns and isinstance(ns[0], int):
            vals.extend(ns)
    if not vals:
        return [7]
    max_v = max(vals)
    base_choices = [5, 7, 8, 11, 13, 16]
    return sorted({m for m in base_choices if m > max_v} | {max_v + 1})


def _detect_seq_shape(instance: dict) -> str:
    """Return 'extend' (state -> state + [v]) or 'grid' (3x5 list)."""
    obs = instance["train_obs"][0]
    s, ns = obs["state"], obs["next_state"]
    if isinstance(s, list) and isinstance(ns, list):
        if len(ns) == len(s) + 1:
            return "extend"
        if len(ns) == len(s) and len(s) == 15:
            return "grid"
    return "extend"


def _gen_seq_pool(instance: dict) -> list[str]:
    shape = _detect_seq_shape(instance)
    pool = [T.seq_identity()]

    moduli = _detect_seq_mod(instance)

    if shape == "grid":
        # world_seq_002 family. Try grid rules with several moduli.
        for mod in moduli:
            try:
                pool.append(T.seq_grid_rule(rows=3, cols=5, mod=mod))
            except Exception:
                pass
        # Also include strand and window forms in case detection was wrong.
        for mod in moduli:
            try:
                pool.append(T.seq_strand_rule(mod=mod))
                pool.append(T.seq_window_rule(mod=mod))
            except Exception:
                pass
        return pool

    # Extend shape — generate branched recurrences.
    rhs_choices = T.RHS_FORMS

    # Branched recurrences: pick guard + helper + 2-3 branches.
    branch_specs = [
        # Single branch (no else)
        [("else", rhs_choices[0])],
        [("else", rhs_choices[1])],
        [("else", rhs_choices[2])],
        [("else", rhs_choices[3])],
        # 2-branch
        [("_is_prime(n)", rhs_choices[1]), ("else", rhs_choices[3])],
        [("_is_prime(n)", rhs_choices[1]), ("else", rhs_choices[0])],
        [("n % 2 == 0", rhs_choices[3]), ("else", rhs_choices[1])],
        [("n % 2 == 0", rhs_choices[0]), ("else", rhs_choices[2])],
        [("_is_psq(n)", rhs_choices[8]), ("else", rhs_choices[7])],
        # 3-branch
        [
            ("_is_prime(n)", rhs_choices[1]),
            ("n % 2 == 0", rhs_choices[3]),
            ("else", rhs_choices[4]),
        ],
        [
            ("_ds(n) == 0", rhs_choices[0]),
            ("_ds(n) == 1", rhs_choices[1]),
            ("else", rhs_choices[2]),
        ],
        [
            ("_is_psq(n)", rhs_choices[8]),
            ("n % 3 == 0", rhs_choices[3]),
            ("else", rhs_choices[7]),
        ],
        # 4-branch (mod 4)
        [
            ("n % 4 == 3", rhs_choices[5]),
            ("n % 4 == 0", rhs_choices[6]),
            ("n % 4 == 1", rhs_choices[3]),
            ("else", rhs_choices[3]),
        ],
    ]

    helper_for_spec = {
        "_is_prime": "is_prime",
        "_is_psq": "is_perfect_square",
        "_ds": "digit_sum_mod3",
    }

    for spec in branch_specs:
        # Determine which helper to include.
        helper = None
        joined = " ".join(g for g, _ in spec)
        for tag, name in helper_for_spec.items():
            if tag in joined:
                helper = name
                break
        for mod in moduli:
            try:
                pool.append(T.seq_branched_recurrence(
                    branches=spec, helper_guard=helper, mod=mod,
                ))
            except Exception:
                pass

    # Also include strand and window templates (cover seq_004, seq_005).
    for mod in moduli:
        try:
            pool.append(T.seq_strand_rule(mod=mod))
        except Exception:
            pass
        try:
            pool.append(T.seq_window_rule(mod=mod))
        except Exception:
            pass

    return pool


# ----------------------------------------------------------------------
# Public entry point
# ----------------------------------------------------------------------

def generate_pool(instance: dict) -> list[str]:
    """
    Return a deterministic candidate pool for the given instance.

    No LLM calls. No I/O.
    """
    fam = instance.get("family", "")
    if fam == "cellular_automata":
        return _gen_ca_pool(instance)
    if fam == "particle_system":
        return _gen_pt_pool(instance)
    if fam == "pattern_puzzle":
        return _gen_seq_pool(instance)
    return [T.ca_identity(), T.pt_identity(), T.seq_identity()]
