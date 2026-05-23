"""H/G homogenization estimator: decomposes cross-run agreement into grounded (G) vs homogenized (H)."""
import math
import random
from collections import Counter

from worlds.probes import agreement_rate


def _estimate_kappa(sigs: dict, truth_sig: tuple) -> float:
    """Estimate probability two independent wrong answers coincide.
    Per probe: collect wrong outputs from all runs where they disagree with truth.
    kappa = mean over probes of (coinciding wrong-output pairs / total wrong pairs).
    Fallback when no wrong pairs: 1 / (m_eff - 1) where m_eff = distinct outputs on that probe.
    """
    run_ids = list(sigs.keys())
    n_probes = len(truth_sig)
    kappa_vals = []

    for i in range(n_probes):
        wrong_outputs = [sigs[r][i] for r in run_ids if sigs[r][i] != truth_sig[i]]
        n_wrong = len(wrong_outputs)
        n_wrong_pairs = n_wrong * (n_wrong - 1) // 2

        if n_wrong_pairs == 0:
            all_outputs = [sigs[r][i] for r in run_ids]
            m_eff = max(2, len(set(all_outputs)))
            kappa_vals.append(1.0 / (m_eff - 1))
        else:
            counts = Counter(wrong_outputs)
            n_coinciding = sum(c * (c - 1) // 2 for c in counts.values())
            kappa_vals.append(n_coinciding / n_wrong_pairs)

    return sum(kappa_vals) / len(kappa_vals) if kappa_vals else 0.0


def hg_decompose(sigs: dict, truth_sig: tuple, kappa: float = None) -> dict:
    """Compute G, H, delta, Abar for a population of runs.

    sigs: {run_id: signature_tuple}  — one signature per run
    truth_sig: signature of the ground-truth rule
    kappa: probability two independent wrong outputs coincide (estimated if None)

    Returns dict: {'G': float, 'H': float, 'delta': float, 'Abar': float,
                   'g': {run_id: float}, 'kappa': float, 'n_pairs': int}
    """
    run_ids = list(sigs.keys())
    n_runs = len(run_ids)

    g = {r: agreement_rate(sigs[r], truth_sig) for r in run_ids}

    if n_runs < 2:
        return {'G': 0.0, 'H': 0.0, 'delta': 0.0, 'Abar': 0.0,
                'g': g, 'kappa': kappa if kappa is not None else 0.0, 'n_pairs': 0}

    if kappa is None:
        kappa = _estimate_kappa(sigs, truth_sig)

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


def hg_bootstrap_ci(sigs: dict, truth_sig: tuple, B: int = 1000, seed: int = 0,
                    alpha: float = 0.05) -> dict:
    """Percentile bootstrap CIs on G, H, delta by resampling runs with replacement.
    Returns dict: {'G_ci': (lo, hi), 'H_ci': (lo, hi), 'delta_ci': (lo, hi)}.
    """
    rng = random.Random(seed)
    run_ids = list(sigs.keys())
    n_runs = len(run_ids)

    G_vals, H_vals, delta_vals = [], [], []

    for _ in range(B):
        sampled = [rng.choice(run_ids) for _ in range(n_runs)]
        resampled = {f'_b{k}': sigs[r] for k, r in enumerate(sampled)}
        res = hg_decompose(resampled, truth_sig)
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


def hg_null_pvalue(sigs: dict, truth_sig: tuple, B: int = 1000, seed: int = 0) -> float:
    """Parametric bootstrap p-value for H > 0 under conditional independence.
    Synthesize null populations: run r gets correct answer with prob g_r, otherwise
    draws independently from the empirical wrong-output distribution.
    p = fraction of null resamples where H_null >= H_observed.
    """
    rng = random.Random(seed)
    run_ids = list(sigs.keys())
    n_probes = len(truth_sig)

    result_obs = hg_decompose(sigs, truth_sig)
    H_obs = result_obs['H']
    g = result_obs['g']

    wrong_dist = []
    for i in range(n_probes):
        wrongs = [sigs[r][i] for r in run_ids if sigs[r][i] != truth_sig[i]]
        wrong_dist.append(wrongs)

    H_null_vals = []
    for _ in range(B):
        null_sigs = {}
        for r in run_ids:
            sig_null = []
            for i in range(n_probes):
                if rng.random() < g[r]:
                    sig_null.append(truth_sig[i])
                else:
                    wrongs = wrong_dist[i]
                    sig_null.append(rng.choice(wrongs) if wrongs else '<NULL_WRONG>')
            null_sigs[r] = tuple(sig_null)
        H_null_vals.append(hg_decompose(null_sigs, truth_sig)['H'])

    return sum(1 for h in H_null_vals if h >= H_obs) / B
