"""
Scoring + MDL utilities for Arm D.

Per-candidate flow:
  compile → exec module → fetch hidden_rule_fn → run on each train_obs
  → compute fraction of exact-match predictions.

Exceptions during compile/exec/per-call are caught and counted as
zero-credit observations; total exception ⇒ train_acc = 0.0. The
selector treats train_acc as the primary criterion, so a totally-broken
candidate is effectively ignored.
"""

import ast
import hashlib


def _safe_exec(source: str) -> tuple[object | None, str]:
    """Compile + exec a candidate source in an isolated namespace.
    Returns (hidden_rule_fn or None, error tag).
    """
    try:
        ns: dict = {}
        exec(compile(source, "<candidate>", "exec"), ns)
    except Exception as e:
        return None, f"exec:{type(e).__name__}"
    fn = ns.get("hidden_rule_fn")
    if not callable(fn):
        return None, "no_callable"
    return fn, "ok"


def score_candidate(source: str, train_obs: list) -> tuple[float, dict]:
    """
    Run candidate on each obs in train_obs; return (train_acc, meta).

    meta carries: per_obs_hits (list of bool), error_tag, n_exceptions.
    """
    fn, err = _safe_exec(source)
    if fn is None:
        return 0.0, {"per_obs_hits": [False] * len(train_obs),
                     "error_tag": err, "n_exceptions": len(train_obs)}

    hits = []
    n_exc = 0
    for obs in train_obs:
        try:
            predicted = fn(obs["state"])
            hits.append(predicted == obs["next_state"])
        except Exception:
            hits.append(False)
            n_exc += 1
    train_acc = sum(hits) / len(train_obs) if train_obs else 0.0
    return train_acc, {"per_obs_hits": hits, "error_tag": "ok", "n_exceptions": n_exc}


def mdl(source: str) -> int:
    """Source MDL: prefer the AST-dump length when parseable, else the
    whitespace-stripped char count. Lower = simpler.
    """
    try:
        return len(ast.dump(ast.parse(source)))
    except Exception:
        return len(source.replace(" ", "").replace("\n", ""))


def source_hash(source: str) -> str:
    """Stable short hash for tiebreak."""
    return hashlib.sha256(source.encode("utf-8")).hexdigest()[:12]


def score_pool(sources: list[str], train_obs: list) -> list[dict]:
    """Score every candidate, return rich record list (sorted by train_acc desc, mdl asc)."""
    records = []
    for src in sources:
        acc, meta = score_candidate(src, train_obs)
        records.append({
            "source": src,
            "train_acc": acc,
            "mdl": mdl(src),
            "hash": source_hash(src),
            "meta": meta,
        })
    records.sort(key=lambda r: (-r["train_acc"], r["mdl"], r["hash"]))
    return records
