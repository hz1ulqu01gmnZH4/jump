"""
Mechanical compilation: (mask, lookup_table) -> Python source for hidden_rule_fn.

The compiler is deterministic; no LLM involvement. The orchestrator calls
this directly with the schema-search output. Generated source uses ONLY:
  - the feature-extraction functions inlined,
  - a literal dict for the lookup table,
  - default-fallback to `cur` for missing entries.
"""

from __future__ import annotations
from .ca_features import NEIGHBOURHOODS_2D, NEIGHBOURHOODS_1D


# Source fragments for each feature kind. Each returns a Python expression
# of type int that, given (state, r, c, rows, cols, n_states), yields the
# feature value at cell (r, c).
def _feature_expr(feat: str) -> str:
    if feat == "cur":
        return "state[r][c]"
    if feat == "row_parity":
        return "(r % 2)"
    if feat == "col_parity":
        return "(c % 2)"
    if feat == "rcsum_mod2":
        return "((r + c) % 2)"
    if feat == "rcsum_mod3":
        return "((r + c) % 3)"
    if feat.startswith("count_"):
        # count_<N>_<s>
        rest = feat[len("count_"):]
        # Find longest matching neighbourhood key (names contain underscores).
        nname = max((n for n in NEIGHBOURHOODS_2D if rest.startswith(n + "_")),
                    key=len, default=None)
        if nname is None:
            raise ValueError(f"Unknown 2D neighbourhood in feature: {feat}")
        sval = int(rest[len(nname) + 1:])
        offsets = NEIGHBOURHOODS_2D[nname]
        offsets_lit = repr(offsets)
        return (f"sum(1 for dr, dc in {offsets_lit} "
                f"if state[(r + dr) % rows][(c + dc) % cols] == {sval})")
    if feat.startswith("count1d_"):
        rest = feat[len("count1d_"):]
        nname = max((n for n in NEIGHBOURHOODS_1D if rest.startswith(n + "_")),
                    key=len, default=None)
        if nname is None:
            raise ValueError(f"Unknown 1D neighbourhood in feature: {feat}")
        sval = int(rest[len(nname) + 1:])
        offsets = NEIGHBOURHOODS_1D[nname]
        offsets_lit = repr(offsets)
        # 1D: state has a single row at index 0.
        return (f"sum(1 for d in {offsets_lit} "
                f"if state[0][(c + d) % cols] == {sval})")
    raise ValueError(f"Unknown feature: {feat}")


def compile_to_python(mask: tuple[str, ...],
                      lookup: dict[tuple, int],
                      majority_per_cur: dict[int, int]) -> str:
    """
    Emit Python source for `hidden_rule_fn(state) -> next_state`.

    Args:
      mask: tuple of feature names selected by schema_search.
      lookup: (feature-projection-tuple) -> next_state value.
      majority_per_cur: fallback for missing lookup entries; maps current-state
        value to the most common next-state observed for that current.

    The emitted function:
      - Detects 1D vs 2D state shape and normalises to a 1-row 2D grid for 1D.
      - Computes the mask features at each cell.
      - Looks up next-state; on miss, falls back to L1-nearest training key
        sharing the same `cur` (if `cur` is in the mask), then to
        majority_per_cur[cur].
      - Returns a 2D grid (or flattens to 1D if input was 1D).
    """
    feature_exprs = [_feature_expr(f) for f in mask]
    feature_block = ", ".join(feature_exprs)
    cur_idx = mask.index("cur") if "cur" in mask else -1

    # Build per-cur key buckets so the nearest-neighbour scan is per-state.
    by_cur: dict[int, list[tuple[tuple, int]]] = {}
    for key, nxt_val in lookup.items():
        cur_v = key[cur_idx] if cur_idx >= 0 else -1
        by_cur.setdefault(cur_v, []).append((key, nxt_val))

    src_lines = [
        "def hidden_rule_fn(state):",
        "    _LOOKUP = " + repr(dict(lookup)),
        "    _BY_CUR = " + repr(by_cur),
        "    _FALLBACK = " + repr(dict(majority_per_cur)),
        f"    _CUR_IDX = {cur_idx}",
        "    _is_1d = isinstance(state[0], int)",
        "    if _is_1d:",
        "        state = [list(state)]",
        "    rows = len(state)",
        "    cols = len(state[0])",
        "    nxt = [[0] * cols for _ in range(rows)]",
        "    for r in range(rows):",
        "        for c in range(cols):",
        f"            key = ({feature_block},)",
        "            cur = state[r][c]",
        "            v = _LOOKUP.get(key)",
        "            if v is None:",
        "                bucket = _BY_CUR.get(cur, []) if _CUR_IDX >= 0 else []",
        "                if not bucket:",
        "                    bucket = [(k, nv) for kk in _BY_CUR.values() for (k, nv) in kk]",
        "                if bucket:",
        "                    best_d = None; best_v = None",
        "                    for k2, nv in bucket:",
        "                        d = sum(abs(a - b) for a, b in zip(key, k2)",
        "                                if isinstance(a, int) and isinstance(b, int))",
        "                        if best_d is None or d < best_d:",
        "                            best_d = d; best_v = nv",
        "                    v = best_v",
        "                if v is None:",
        "                    v = _FALLBACK.get(cur, cur)",
        "            nxt[r][c] = v",
        "    if _is_1d:",
        "        return nxt[0]",
        "    return nxt",
    ]
    return "\n".join(src_lines) + "\n"


def majority_per_cur(nodes: list[dict]) -> dict[int, int]:
    """For each current-state value, the most-common next-state observed."""
    from collections import Counter
    by_cur: dict[int, Counter] = {}
    for n in nodes:
        by_cur.setdefault(n["cur"], Counter())[n["next"]] += 1
    return {cur: ctr.most_common(1)[0][0] for cur, ctr in by_cur.items()}
