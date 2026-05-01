"""
Beam search over feature masks with leave-one-obs-out cross-validation.

Pure-greedy hill-climbing on conflict-reduction takes wrong turns on
parity-split rules (e.g. world_ca_005) because a feature can reduce
training conflicts via spurious correlation rather than true causal
dependence. We address this by:

  (1) keeping `beam_width` candidate masks alive at each expansion step,
  (2) scoring leaves by CV accuracy: build the lookup from N-1 train
      grids, evaluate on the held-out grid, average.

Returns (best_mask, conflicts_remaining, mdl_score).
"""

from __future__ import annotations
from .anti_unify import group, conflicts, conflicted_node_count, lookup_table


def mdl_cost(nodes: list[dict], mask: tuple[str, ...]) -> float:
    """
    A simple MDL: |mask| * 1 + log2(n_classes) * 2 + conflicted_nodes * 10.
    Conflicts dominate so the search prefers any conflict-free mask over
    any conflicted one.
    """
    g = group(nodes, mask)
    n_classes = len(g)
    conf_nodes = conflicted_node_count(nodes, mask)
    import math
    return len(mask) * 1.0 + (math.log2(max(n_classes, 1))) * 2.0 + conf_nodes * 10.0


def greedy_search(
    nodes: list[dict],
    candidate_features: list[str],
    seed_mask: tuple[str, ...] = ("cur",),
    cap_size: int = 8,
) -> tuple[tuple[str, ...], int, float]:
    """
    Pure greedy: maximally reduce conflicted-node count at each step.
    Used as a fallback / sanity check; beam_search_cv is preferred.
    """
    mask: tuple[str, ...] = tuple(seed_mask)
    pool = [f for f in candidate_features if f not in mask]
    best_score = mdl_cost(nodes, mask)
    best_conf = conflicts(nodes, mask)

    while len(mask) < cap_size and best_conf > 0:
        cur_conf_nodes = conflicted_node_count(nodes, mask)
        candidates = []
        for f in pool:
            new_mask = mask + (f,)
            cn = conflicted_node_count(nodes, new_mask)
            ncl = len(group(nodes, new_mask))
            candidates.append((cn, ncl, f))
        candidates.sort(key=lambda t: (t[0], t[1], t[2]))
        best = candidates[0]
        if best[0] >= cur_conf_nodes:
            break
        mask = mask + (best[2],)
        pool.remove(best[2])
        best_conf = conflicts(nodes, mask)
        best_score = mdl_cost(nodes, mask)
    return mask, best_conf, best_score


def _cv_score(nodes: list[dict], mask: tuple[str, ...]) -> float:
    """
    Leave-one-obs-grid-out cross-validation score.
    For each held-out obs index, build lookup from the rest, predict the
    held-out cells; return overall cell-level accuracy.

    With k-NN fallback (always finds *some* nearest training key sharing
    `cur`), this rewards masks that actually generalise across grids.
    """
    obs_indices = sorted({n["_obs"] for n in nodes})
    if len(obs_indices) < 2:
        return 0.0
    cur_idx = mask.index("cur") if "cur" in mask else -1
    correct = 0
    total = 0
    for held in obs_indices:
        train = [n for n in nodes if n["_obs"] != held]
        test = [n for n in nodes if n["_obs"] == held]
        table = lookup_table(train, mask)
        # Per-cur buckets for L1-NN fallback (mirrors the compiler's runtime path).
        by_cur: dict = {}
        for k, v in table.items():
            cv = k[cur_idx] if cur_idx >= 0 else -1
            by_cur.setdefault(cv, []).append((k, v))
        for n in test:
            key = tuple(n[f] for f in mask)
            v = table.get(key)
            if v is None:
                cur_v = key[cur_idx] if cur_idx >= 0 else -1
                bucket = by_cur.get(cur_v) or [
                    (k2, v2) for kk in by_cur.values() for (k2, v2) in kk]
                if bucket:
                    best_d = None
                    best_v = None
                    for k2, v2 in bucket:
                        d = sum(abs(a - b) for a, b in zip(key, k2)
                                if isinstance(a, int) and isinstance(b, int))
                        if best_d is None or d < best_d:
                            best_d = d
                            best_v = v2
                    v = best_v
            if v is None:
                v = n["cur"]
            if v == n["next"]:
                correct += 1
            total += 1
    return correct / total if total else 0.0


def beam_search_cv(
    nodes: list[dict],
    candidate_features: list[str],
    seed_mask: tuple[str, ...] = ("cur",),
    cap_size: int = 6,
    beam_width: int = 5,
) -> tuple[tuple[str, ...], int, float]:
    """
    Beam search guided by CV accuracy.

    At each step, expand each beam-mask with each remaining candidate feature,
    keep the top `beam_width` by CV score (tiebreak: fewer features, then
    smaller class count, then lex name order).

    Stop expanding when the best beam is conflict-free OR cap reached OR
    no expansion improves CV score.
    """
    seed = tuple(seed_mask)
    seed_score = _cv_score(nodes, seed)
    seed_classes = len(group(nodes, seed))
    beam: list[tuple[float, int, int, tuple[str, ...]]] = [
        (seed_score, len(seed), seed_classes, seed)
    ]
    best_overall = beam[0]

    for _step in range(cap_size - len(seed)):
        next_beam: list[tuple[float, int, int, tuple[str, ...]]] = []
        for _, _, _, mask in beam:
            pool = [f for f in candidate_features if f not in mask]
            for f in pool:
                m2 = mask + (f,)
                s = _cv_score(nodes, m2)
                ncl = len(group(nodes, m2))
                next_beam.append((s, len(m2), ncl, m2))
        if not next_beam:
            break
        # Higher CV is better; smaller mask better; smaller class count better.
        next_beam.sort(key=lambda t: (-t[0], t[1], t[2], t[3]))
        beam = next_beam[:beam_width]
        # Track best-seen including a small bonus for smallness when tied on CV.
        cand = beam[0]
        if (cand[0], -cand[1], -cand[2]) > (best_overall[0], -best_overall[1], -best_overall[2]):
            best_overall = cand
        # Early stop: best beam is conflict-free and adding a feature would
        # not improve CV more than 1 cell.
        best_mask = beam[0][3]
        if conflicts(nodes, best_mask) == 0:
            # Try one more layer; if it doesn't help, stop.
            break
    best_mask = best_overall[3]
    return best_mask, conflicts(nodes, best_mask), mdl_cost(nodes, best_mask)


def rank_extensions(
    nodes: list[dict],
    mask: tuple[str, ...],
    candidate_features: list[str],
    top_k: int = 5,
) -> list[tuple[str, int, int]]:
    """
    Return the top-k candidate features by projected conflict reduction.
    Diagnostic output for the orchestrator (no raw obs leaked).
    Each tuple is (feature_name, conflict_nodes_after, n_classes_after).
    """
    rows = []
    for f in candidate_features:
        if f in mask:
            continue
        new_mask = mask + (f,)
        cn = conflicted_node_count(nodes, new_mask)
        ncl = len(group(nodes, new_mask))
        rows.append((f, cn, ncl))
    rows.sort(key=lambda t: (t[1], t[2], t[0]))
    return rows[:top_k]
