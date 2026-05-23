"""H/G homogenization estimator: decomposes cross-run agreement into grounded (G) vs homogenized (H)."""
import math
import random
from itertools import combinations

from worlds.probes import agreement_rate, prediction_signature


def _make_random_rule_src(instance: dict, seed: int) -> str:
    fam = instance['family']
    train = instance['train_obs']
    sample_state = train[0]['state']
    if fam == 'cellular_automata':
        if isinstance(sample_state[0], list):
            states = sorted({v for row in sample_state for v in row})
        else:
            states = sorted(set(sample_state))
        return (
            f"def hidden_rule_fn(state):\n"
            f"    import random\n"
            f"    rng = random.Random({seed})\n"
            f"    states = {states!r}\n"
            "    if isinstance(state[0], list):\n"
            "        return [[rng.choice(states) for _ in row] for row in state]\n"
            "    return [rng.choice(states) for _ in state]\n"
        )
    elif fam == 'particle_system':
        return (
            "def hidden_rule_fn(state):\n"
            "    import copy, random\n"
            f"    rng = random.Random({seed})\n"
            "    result = copy.deepcopy(state)\n"
            "    for p in result:\n"
            "        p['x'] = (p['x'] + rng.randint(-1,1)) % 8\n"
            "        p['y'] = (p['y'] + rng.randint(-1,1)) % 8\n"
            "    return result\n"
        )
    else:  # pattern_puzzle
        return (
            "def hidden_rule_fn(state):\n"
            f"    import random\n"
            f"    rng = random.Random({seed})\n"
            "    m = max(state) + 1 if state else 7\n"
            "    return list(state) + [rng.randint(0, m - 1)]\n"
        )


def estimate_kappa_reference(instance: dict, probes: list, truth_sig: tuple,
                             n_ref: int = 100, seed: int = 0) -> dict:
    """Estimate per-probe wrong-output coincidence from an INDEPENDENT reference population.

    Generates n_ref random rule functions (never uses the test sigs) and computes
    empirical pairwise coincidence among wrong outputs per probe.

    Returns {'kappa': float, 'kappa_i': list, 'ref_wrong_dist': list, 'n_ref': int}.
    """
    ref_sigs = []
    for j in range(n_ref):
        seed_j = seed * 10000 + j
        src = _make_random_rule_src(instance, seed_j)
        ns = {}
        exec(src, ns)
        rule_fn = ns['hidden_rule_fn']
        ref_sigs.append(prediction_signature(rule_fn, probes))

    n_probes = len(truth_sig)
    kappa_i = []
    ref_wrong_dist = []

    for i in range(n_probes):
        W = [ref_sigs[j][i] for j in range(n_ref) if ref_sigs[j][i] != truth_sig[i]]
        ref_wrong_dist.append(W)
        if len(W) >= 2:
            pairs = list(combinations(W, 2))
            n_equal = sum(1 for a, b in pairs if a == b)
            kappa_i.append(n_equal / max(len(pairs), 1))
        else:
            # Option C fallback: analytic cardinality
            distinct = set(W)
            kappa_i.append(1.0 / max(len(distinct) - 1, 1) if distinct else 0.0)

    kappa_bar = sum(kappa_i) / n_probes if n_probes > 0 else 0.0

    return {
        'kappa': kappa_bar,
        'kappa_i': kappa_i,
        'ref_wrong_dist': ref_wrong_dist,
        'n_ref': n_ref,
    }


def hg_decompose(sigs: dict, truth_sig: tuple, kappa: float) -> dict:
    """Compute G, H, delta, Abar for a population of runs.

    sigs: {run_id: signature_tuple}  — one signature per run
    truth_sig: signature of the ground-truth rule
    kappa: REQUIRED — probability two independent wrong outputs coincide.
            Must come from estimate_kappa_reference(). Raises ValueError if None.

    Returns dict: {'G': float, 'H': float, 'delta': float, 'Abar': float,
                   'g': {run_id: float}, 'kappa': float, 'n_pairs': int}
    """
    if kappa is None:
        raise ValueError("kappa is required; call estimate_kappa_reference() first")

    run_ids = list(sigs.keys())
    n_runs = len(run_ids)

    g = {r: agreement_rate(sigs[r], truth_sig) for r in run_ids}

    if n_runs < 2:
        return {'G': 0.0, 'H': 0.0, 'delta': 0.0, 'Abar': 0.0,
                'g': g, 'kappa': kappa, 'n_pairs': 0}

    G_sum = H_sum = A_sum = 0.0
    n_pairs = 0

    for i in range(n_runs):
        for j in range(i + 1, n_runs):
            r1, r2 = run_ids[i], run_ids[j]
            g1, g2 = g[r1], g[r2]

            e_hat = g1 * g2 + (1 - g1) * (1 - g2) * kappa
            a = agreement_rate(sigs[r1], sigs[r2])

            G_sum += e_hat
            H_sum += max(a - e_hat, 0.0)
            A_sum += a
            n_pairs += 1

    G = G_sum / n_pairs
    H = H_sum / n_pairs
    Abar = A_sum / n_pairs
    delta = G - H

    return {'G': G, 'H': H, 'delta': delta, 'Abar': Abar,
            'g': g, 'kappa': kappa, 'n_pairs': n_pairs}


def hg_null_pvalue(sigs: dict, truth_sig: tuple, kappa: float,
                   ref_wrong_dist: list, B: int = 1000, seed: int = 0) -> float:
    """Parametric bootstrap p-value for H > 0 under conditional independence.

    Wrong outputs drawn from ref_wrong_dist (reference population), NOT from test sigs.
    p = fraction of null resamples where H_null >= H_observed.
    """
    rng = random.Random(seed)
    run_ids = list(sigs.keys())
    n_probes = len(truth_sig)

    result_obs = hg_decompose(sigs, truth_sig, kappa)
    H_obs = result_obs['H']
    g = result_obs['g']

    H_null_vals = []
    for _ in range(B):
        null_sigs = {}
        for r in run_ids:
            sig_null = []
            for i in range(n_probes):
                if rng.random() < g[r]:
                    sig_null.append(truth_sig[i])
                else:
                    wrongs = ref_wrong_dist[i]
                    sig_null.append(rng.choice(wrongs) if wrongs else truth_sig[i])
            null_sigs[r] = tuple(sig_null)
        H_null_vals.append(hg_decompose(null_sigs, truth_sig, kappa)['H'])

    return sum(1 for h in H_null_vals if h >= H_obs) / B


def hg_bootstrap_ci(sigs: dict, truth_sig: tuple, kappa: float, B: int = 1000,
                    seed: int = 0, alpha: float = 0.05) -> dict:
    """Percentile bootstrap CIs on G, H, delta by resampling runs with replacement.

    kappa: REQUIRED — fixed reference kappa from estimate_kappa_reference().
    Returns dict: {'G_ci': (lo, hi), 'H_ci': (lo, hi), 'delta_ci': (lo, hi)}.
    """
    rng = random.Random(seed)
    run_ids = list(sigs.keys())
    n_runs = len(run_ids)

    G_vals, H_vals, delta_vals = [], [], []

    for _ in range(B):
        sampled = [rng.choice(run_ids) for _ in range(n_runs)]
        resampled = {f'_b{k}': sigs[r] for k, r in enumerate(sampled)}
        res = hg_decompose(resampled, truth_sig, kappa)
        G_vals.append(res['G'])
        H_vals.append(res['H'])
        delta_vals.append(res['delta'])

    G_vals.sort()
    H_vals.sort()
    delta_vals.sort()

    lo = int(math.floor(alpha / 2 * B))
    hi = min(int(math.ceil((1 - alpha / 2) * B)) - 1, B - 1)

    return {
        'G_ci': (G_vals[lo], G_vals[hi]),
        'H_ci': (H_vals[lo], H_vals[hi]),
        'delta_ci': (delta_vals[lo], delta_vals[hi]),
    }


def hg_kappa_sensitivity(sigs: dict, truth_sig: tuple, kappa_hat: float,
                         k_alphabet: int) -> dict:
    """Re-decompose over kappa sweep to check robustness of sign(delta).

    Sweep: kappa in {0, kappa_hat/10, kappa_hat, 10*kappa_hat, 1/(k-1)} clamped to [0, 0.5].
    Returns {'grid': list, 'sign_delta': list, 'kappa_robust': bool}.
    kappa_robust = True iff all sign(delta) are identical across the grid.
    If not robust -> world is AMBIGUOUS.
    """
    kappa_hi = 1.0 / max(k_alphabet - 1, 1)
    grid = [0.0, kappa_hat / 10.0, kappa_hat, min(kappa_hat * 10.0, 0.5), kappa_hi]
    grid = [max(0.0, min(0.5, k)) for k in grid]

    sign_delta = []
    for k in grid:
        res = hg_decompose(sigs, truth_sig, k)
        sign_delta.append(1 if res['delta'] > 0 else (-1 if res['delta'] < 0 else 0))

    kappa_robust = len(set(sign_delta)) == 1
    return {'grid': grid, 'sign_delta': sign_delta, 'kappa_robust': kappa_robust}
