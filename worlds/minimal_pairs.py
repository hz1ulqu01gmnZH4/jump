"""worlds/minimal_pairs.py — Minimal-pair construction and certification for the anti-homogenization benchmark."""


def _shape_signature(instance: dict) -> tuple:
    """Return a hashable tuple capturing grid/particle/sequence shape, not content."""
    family = instance.get('family', '')
    train_obs = instance.get('train_obs', [])
    test_obs = instance.get('test_obs', [])
    n_train = len(train_obs)
    n_test = len(test_obs)
    if not train_obs:
        return (family, n_train, n_test)
    s0 = train_obs[0]['state']
    if family == 'cellular_automata':
        rows = len(s0)
        cols = len(s0[0]) if s0 else 0
        return (family, rows, cols, n_train, n_test)
    elif family == 'particle_system':
        n0 = len(s0)
        return (family, n0, n_train, n_test)
    elif family == 'pattern_puzzle':
        lengths = tuple(len(o['state']) for o in train_obs + test_obs)
        return (family, lengths)
    return (family, n_train, n_test)


def certify_pair(A: dict, B: dict) -> bool:
    """Assert all public-surface invariants of a minimal pair. Returns True or raises AssertionError."""
    assert A['family'] == B['family'], (
        f"family mismatch: {A['family']!r} vs {B['family']!r}"
    )
    assert A['difficulty'] == B['difficulty'], (
        f"difficulty mismatch: {A['difficulty']!r} vs {B['difficulty']!r}"
    )
    assert A['primitive_glossary'] == B['primitive_glossary'], (
        "primitive_glossary mismatch"
    )
    assert A['intervention_api'] == B['intervention_api'], (
        "intervention_api mismatch"
    )
    states_A = [o['state'] for o in A['train_obs']]
    states_B = [o['state'] for o in B['train_obs']]
    assert states_A == states_B, (
        "train_obs input-state column is not byte-identical across pair members"
    )
    assert _shape_signature(A) == _shape_signature(B), (
        f"shape_signature mismatch: {_shape_signature(A)} vs {_shape_signature(B)}"
    )
    return True


def generate_minimal_pairs() -> list:
    """Return all 18 instances (9 pairs × 2 members each).

    Each instance has 'pair_id' and 'pair_member' ('A' or 'B') as evaluator metadata.
    certify_pair(A, B) is called inside the gen module to gate generation.
    """
    from worlds.gen import generate_minimal_pairs as _gen
    return _gen()
