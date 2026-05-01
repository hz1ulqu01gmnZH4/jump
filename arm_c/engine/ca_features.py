"""
Frozen feature vocabulary for the cellular_automata family.

Each cell-at-time-t becomes a node with a property bag. The vocabulary is
intentionally larger than any single rule needs — schema_search picks the
minimal mask that makes the (features -> next) function deterministic.

Neighbourhood library covers every v2-CA invention criterion in worlds/SPEC.md
section 3a (and the two not-listed ones from world_ca_004 and world_ca_006).
"""

from __future__ import annotations
from typing import Iterable


def _wrap(r: int, c: int, rows: int, cols: int) -> tuple[int, int]:
    return r % rows, c % cols


# ── Neighbourhood definitions ────────────────────────────────────────────────
# Each is a list of (dr, dc) offsets. None of them include the centre cell.

NEIGHBOURHOODS_2D: dict[str, list[tuple[int, int]]] = {
    "moore8":      [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if (dr, dc) != (0, 0)],
    "vonNeumann4": [(-1, 0), (1, 0), (0, -1), (0, 1)],
    "diag4":       [(-1, -1), (-1, 1), (1, -1), (1, 1)],
    "knight8":     [(-2, -1), (-2, 1), (-1, -2), (-1, 2),
                    (1, -2), (1, 2), (2, -1), (2, 1)],
    "axial2_4":    [(-2, 0), (2, 0), (0, -2), (0, 2)],
    "ew_diad2":    [(0, -1), (0, 1)],
    "ns_diad2":    [(-1, 0), (1, 0)],
}

# 1D neighbourhoods (used when the grid has 1 row).
NEIGHBOURHOODS_1D: dict[str, list[int]] = {
    "range1":  [-1, 1],
    "range2":  [-2, -1, 1, 2],
    "range3":  [-3, -2, -1, 1, 2, 3],
    "range4":  [-4, -3, -2, -1, 1, 2, 3, 4],
}


def _counts_2d(state: list[list[int]], r: int, c: int,
               offsets: list[tuple[int, int]], n_states: int) -> tuple[int, ...]:
    rows = len(state)
    cols = len(state[0])
    counts = [0] * n_states
    for dr, dc in offsets:
        v = state[(r + dr) % rows][(c + dc) % cols]
        if 0 <= v < n_states:
            counts[v] += 1
    return tuple(counts)


def _counts_1d(state: list[int], i: int, offsets: list[int],
               n_states: int) -> tuple[int, ...]:
    n = len(state)
    counts = [0] * n_states
    for d in offsets:
        v = state[(i + d) % n]
        if 0 <= v < n_states:
            counts[v] += 1
    return tuple(counts)


def _alphabet(state) -> list[int]:
    """Return the sorted list of cell-state values seen in `state`."""
    if isinstance(state[0], list):
        vals = {v for row in state for v in row}
    else:
        vals = set(state)
    return sorted(vals)


def n_states_for(train_obs: list[dict]) -> int:
    """Maximum state-value seen across all train obs, +1. Determines alphabet size."""
    seen = set()
    for o in train_obs:
        s = o["state"]
        if isinstance(s[0], list):
            for row in s:
                seen.update(row)
        else:
            seen.update(s)
        ns = o["next_state"]
        if isinstance(ns[0], list):
            for row in ns:
                seen.update(row)
        else:
            seen.update(ns)
    return max(seen) + 1


# ── Node extraction ──────────────────────────────────────────────────────────

def extract_nodes(train_obs: list[dict]) -> list[dict]:
    """
    For each (obs_idx, r, c) build a property-bag node.

    Property bag fields (always present):
      cur                            : int
      next                           : int   (target)
      row_parity, col_parity         : 0|1
      rcsum_mod2, rcsum_mod3         : 0|1 / 0|1|2
      For each 2D neighbourhood N in NEIGHBOURHOODS_2D:
        count_<N>_<s>                : int    (number of state s in N)
      For each 1D neighbourhood N in NEIGHBOURHOODS_1D (only emitted if grid is 1xN):
        count1d_<N>_<s>              : int

    The number of fields per node is bounded; greedy schema search picks
    a small subset.
    """
    n_states = n_states_for(train_obs)
    nodes: list[dict] = []
    sample_state = train_obs[0]["state"]
    is_1d = isinstance(sample_state[0], int) or (
        isinstance(sample_state, list) and isinstance(sample_state[0], list)
        and len(sample_state) == 1
    )

    for obs_idx, obs in enumerate(train_obs):
        s = obs["state"]
        ns = obs["next_state"]
        # Normalise 1D input to a 1-row grid for uniform handling.
        if isinstance(s[0], int):
            s = [s]
            ns = [ns]
        rows, cols = len(s), len(s[0])

        for r in range(rows):
            for c in range(cols):
                node: dict = {
                    "_obs": obs_idx,
                    "_pos": (r, c),
                    "cur": s[r][c],
                    "next": ns[r][c],
                    "row_parity": r % 2,
                    "col_parity": c % 2,
                    "rcsum_mod2": (r + c) % 2,
                    "rcsum_mod3": (r + c) % 3,
                }
                # 2D neighbourhood counts — always emitted (cheap, constant).
                for nname, offsets in NEIGHBOURHOODS_2D.items():
                    counts = _counts_2d(s, r, c, offsets, n_states)
                    for sval, cnt in enumerate(counts):
                        node[f"count_{nname}_{sval}"] = cnt
                # 1D neighbourhood counts — only if grid is genuinely 1D.
                if rows == 1:
                    for nname, offsets1d in NEIGHBOURHOODS_1D.items():
                        if 2 * max(abs(o) for o in offsets1d) >= cols:
                            continue  # neighbourhood larger than grid: skip
                        counts = _counts_1d(s[0], c, offsets1d, n_states)
                        for sval, cnt in enumerate(counts):
                            node[f"count1d_{nname}_{sval}"] = cnt
                nodes.append(node)
    return nodes


def feature_names_for(nodes: list[dict]) -> list[str]:
    """All non-meta, non-target keys present in the first node."""
    if not nodes:
        return []
    return [k for k in nodes[0].keys() if not k.startswith("_") and k != "next"]
