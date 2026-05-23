#!/usr/bin/env python3
"""
eval/demo_hg.py — GAP2 re-validation: hg.py correctly separates:
  (a) GROUNDED population: each run independently converges to truth
  (b) HOMOGENIZED population: all runs converge to a shared non-truth answer (100%)
  (c) PARTIAL HOMOGENIZATION: p=0.3/0.5/0.7 partial coincidence regimes
  (d) κ-sensitivity: checks robustness of sign(delta) across kappa sweep

Run from ~/projects/jump: uv run python eval/demo_hg.py
"""
import sys
import random
sys.path.insert(0, '/home/ak/projects/jump')

from worlds.gen import generate_all
from worlds.probes import build_probe_grid, prediction_signature, agreement_rate
from eval.hg import (
    hg_decompose, hg_null_pvalue, hg_bootstrap_ci,
    hg_kappa_sensitivity, estimate_kappa_reference,
)


def make_partial_homog_sigs(shared_wrong_sig, truth_sig, n_runs=5, p=0.5, seed=42):
    """Each run: at each probe, with prob p use shared_wrong_sig[i],
    otherwise use a unique sentinel (guaranteed non-coinciding wrong value).
    p=1.0 -> full homogenization."""
    rng = random.Random(seed)
    sigs = {}
    n_probes = len(truth_sig)
    for run_idx in range(n_runs):
        sig = []
        for i in range(n_probes):
            if rng.random() < p:
                sig.append(shared_wrong_sig[i])
            else:
                # Unique sentinel per (run_idx, probe i) -> no coincidence with other runs
                sig.append(('__wrong__', run_idx, i))
        sigs[f'partial_{run_idx}'] = tuple(sig)
    return sigs


def make_demo():
    instances = generate_all()
    inst = instances[0]  # world_ca_001 (CA family, deterministic)

    probes = build_probe_grid(inst, n=100, seed=0)

    ns = {}
    exec(inst['hidden_rule_fn'], ns)
    truth_fn = ns['hidden_rule_fn']
    truth_sig = prediction_signature(truth_fn, probes)

    # Estimate kappa ONCE from independent reference population
    kref = estimate_kappa_reference(inst, probes, truth_sig, n_ref=100, seed=0)
    kappa = kref['kappa']
    ref_wrong_dist = kref['ref_wrong_dist']
    print(f"kappa estimate: {kappa:.6f}  (n_ref={kref['n_ref']})")

    # ------------------------------------------------------------------ #
    # (a) GROUNDED population: 5 runs, each independently close to truth  #
    # ------------------------------------------------------------------ #
    grounded_sigs = {}
    for i in range(5):
        rng = random.Random(i * 100)
        def grounded_fn(state, _rng=rng, _tf=truth_fn):
            import copy
            result = _tf(copy.deepcopy(state))
            if isinstance(result, list) and isinstance(result[0], list):
                for r in result:
                    for c in range(len(r)):
                        if _rng.random() < 0.15:
                            r[c] = (r[c] + 1) % 3
            elif isinstance(result, list):
                for j in range(len(result)):
                    if _rng.random() < 0.15:
                        result[j] = (result[j] + 1) % max(3, max(result) + 1 if result else 3)
            return result
        grounded_sigs[f'run_{i}'] = prediction_signature(grounded_fn, probes)

    result_g = hg_decompose(grounded_sigs, truth_sig, kappa)
    pval_g = hg_null_pvalue(grounded_sigs, truth_sig, kappa, ref_wrong_dist, B=200, seed=42)
    print("\n=== (a) GROUNDED population ===")
    print(f"  G={result_g['G']:.3f}  H={result_g['H']:.3f}  delta={result_g['delta']:.3f}  Abar={result_g['Abar']:.3f}")
    print(f"  null p-value for H: {pval_g:.4f}")
    print(f"  Expected: delta > 0")
    assert result_g['delta'] > 0, f"GROUNDED: expected delta>0, got {result_g['delta']:.3f}"

    # ------------------------------------------------------------------ #
    # (b) HOMOGENIZED 100%: all runs agree on a shared WRONG answer        #
    # ------------------------------------------------------------------ #
    def shared_wrong_fn(state):
        if isinstance(state[0], list):
            return [[(v + 1) % 3 for v in row] for row in state]
        return [(v + 1) % max(3, max(state) + 1) for v in state]

    shared_wrong_sig = prediction_signature(shared_wrong_fn, probes)
    homog_sigs = {f'run_{i}': shared_wrong_sig for i in range(5)}

    result_h = hg_decompose(homog_sigs, truth_sig, kappa)
    pval_h = hg_null_pvalue(homog_sigs, truth_sig, kappa, ref_wrong_dist, B=200, seed=42)
    print("\n=== (b) HOMOGENIZED 100% ===")
    print(f"  G={result_h['G']:.3f}  H={result_h['H']:.3f}  delta={result_h['delta']:.3f}  Abar={result_h['Abar']:.3f}")
    print(f"  null p-value for H: {pval_h:.4f}")
    print(f"  Expected: H>=G, delta<=0")
    assert result_h['H'] >= result_h['G'], \
        f"HOMOG: expected H>=G, got G={result_h['G']:.3f} H={result_h['H']:.3f}"
    assert result_h['delta'] <= 0, \
        f"HOMOG: expected delta<=0, got {result_h['delta']:.3f}"

    # ------------------------------------------------------------------ #
    # (c) PARTIAL HOMOGENIZATION: p=0.3, 0.5, 0.7                         #
    # ------------------------------------------------------------------ #
    for p in [0.3, 0.5, 0.7]:
        ph_sigs = make_partial_homog_sigs(
            shared_wrong_sig, truth_sig, n_runs=5, p=p, seed=42 + int(p * 100)
        )
        result_p = hg_decompose(ph_sigs, truth_sig, kappa)
        pval_p = hg_null_pvalue(ph_sigs, truth_sig, kappa, ref_wrong_dist, B=200, seed=42)
        print(f"\n=== (c) PARTIAL HOMOGENIZATION p={p} ===")
        print(f"  G={result_p['G']:.3f}  H={result_p['H']:.3f}  delta={result_p['delta']:.3f}  Abar={result_p['Abar']:.3f}")
        print(f"  null p-value for H: {pval_p:.4f}")
        print(f"  Expected: H>=G, delta<=0  (~{int(p*100)}% probe coincidence among runs)")
        assert result_p['H'] >= result_p['G'], \
            f"PARTIAL p={p}: expected H>=G, got H={result_p['H']:.3f} G={result_p['G']:.3f}"
        assert result_p['delta'] <= 0, \
            f"PARTIAL p={p}: expected delta<=0, got {result_p['delta']:.3f}"

    # ------------------------------------------------------------------ #
    # (d) κ-sensitivity on homogenized case (k_alphabet=3 for 3-value CA) #
    # ------------------------------------------------------------------ #
    sens = hg_kappa_sensitivity(homog_sigs, truth_sig, kappa, k_alphabet=3)
    print(f"\n=== (d) κ-sensitivity (homogenized) ===")
    print(f"  grid: {[f'{k:.4f}' for k in sens['grid']]}")
    print(f"  sign_delta: {sens['sign_delta']}")
    print(f"  kappa_robust: {sens['kappa_robust']}")

    print("\nGAP2 DEMO PASSED: estimator correctly separates all regimes ✓")


if __name__ == '__main__':
    make_demo()
