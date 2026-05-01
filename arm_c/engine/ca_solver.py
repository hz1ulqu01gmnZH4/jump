"""
End-to-end CA family solver.

Pipeline:
  1. Extract property-graph nodes from train_obs (ca_features.extract_nodes).
  2. Greedy MDL forward selection picks a feature mask making the
     (mask -> next) function deterministic on training (schema_search).
  3. Build the lookup table from clean equivalence classes (anti_unify).
  4. Compile to Python source (compiler.compile_to_python).
  5. Submit via harness.submit_hypothesis.

CEGIS / interventions are deferred (P-C1c). The MVP relies on training
coverage being sufficient — for v2 CA worlds with 12 train_obs * 16 cells
= 192 nodes, that holds for most rules.
"""

from __future__ import annotations
from .ca_features import extract_nodes, feature_names_for
from .schema_search import beam_search_cv, rank_extensions
from .anti_unify import lookup_table, conflicts
from .compiler import compile_to_python, majority_per_cur


def solve_ca(harness) -> dict:
    """
    Solve a single CA instance end-to-end. Returns engine_meta diagnostics.
    Side effect: harness.submit_hypothesis is called.
    """
    train = harness.get_train_obs()
    nodes = extract_nodes(train)
    feat_pool = feature_names_for(nodes)

    mask, n_conflicts, mdl = beam_search_cv(
        nodes, candidate_features=feat_pool, seed_mask=("cur",),
        cap_size=6, beam_width=5,
    )

    table = lookup_table(nodes, mask)
    fallback = majority_per_cur(nodes)
    src = compile_to_python(mask, table, fallback)

    # Diagnostics for the caller (no raw obs leaked).
    diag = {
        "final_mask": list(mask),
        "n_equivalence_classes_clean": len(table),
        "n_remaining_conflicts": n_conflicts,
        "mdl_score": round(mdl, 3),
        "n_features_in_pool": len(feat_pool),
        "cegis_calls": 0,
        "top_5_extensions_at_stop": [
            (f, cn, ncl) for (f, cn, ncl) in
            rank_extensions(nodes, mask, feat_pool, top_k=5)
        ],
    }

    harness.submit_hypothesis(src)
    return diag
