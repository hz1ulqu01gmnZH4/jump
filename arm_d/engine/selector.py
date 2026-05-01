"""
Selector: pick the survivor from a scored pool.

Sort key (descending preference):
  1. train_acc   — primary; higher is better.
  2. validation_acc on a held-out slice — only used to break ties at the
     top of the train_acc ranking.
  3. mdl         — lower is better (Occam tiebreak among validation-tied).
  4. source_hash — deterministic tiebreak.

The validation slice = the LAST `n_val` train_obs (default 2 of ≥10),
removed from the slice the scorer sees during pool ranking. This is
implemented in `run_arm_d.py` by passing the train_obs explicitly to
the scorer (early-train) and the selector (late-train as validation).
"""
from .scorer import score_candidate, mdl, source_hash


def split_train_val(train_obs: list, n_val: int = 2):
    """Return (train_for_scoring, val_for_tiebreak)."""
    if len(train_obs) <= n_val + 2:
        # Tiny train set; don't hold any out — ranking-only mode.
        return train_obs, []
    return train_obs[:-n_val], train_obs[-n_val:]


def select(records: list[dict], val_obs: list) -> dict:
    """
    Pick the survivor from a sorted records list.

    `records` must be sorted by train_acc desc (the scorer.score_pool output is).
    If multiple records tie at the top train_acc, run them on val_obs and
    pick the highest val_acc (mdl asc tiebreak, hash asc final).

    Returns the selected record dict (with extra keys: val_acc).
    """
    if not records:
        raise ValueError("select(): empty records list")

    top_acc = records[0]["train_acc"]
    tied = [r for r in records if abs(r["train_acc"] - top_acc) < 1e-9]

    # Tiebreak via validation slice (if any).
    if val_obs and len(tied) > 1:
        for r in tied:
            v_acc, _ = score_candidate(r["source"], val_obs)
            r["val_acc"] = v_acc
        tied.sort(key=lambda r: (-r["val_acc"], r["mdl"], r["hash"]))
    else:
        for r in tied:
            r["val_acc"] = None

    return tied[0]


def select_topk(records: list[dict], k: int) -> list[dict]:
    """Return the top-K records by the score_pool sort order (no validation)."""
    return records[: max(1, k)]
