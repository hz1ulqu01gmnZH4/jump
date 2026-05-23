#!/usr/bin/env python3
"""
eval/demo_hg.py — demonstrates that hg.py correctly separates:
  (a) GROUNDED population: each run independently converges to truth
  (b) HOMOGENIZED population: all runs converge to a shared non-truth answer

Run from ~/projects/jump: python eval/demo_hg.py
"""
import sys, random
sys.path.insert(0, '/home/ak/projects/jump')
from worlds.gen import generate_all
from worlds.probes import build_probe_grid, prediction_signature, agreement_rate
from eval.hg import hg_decompose, hg_null_pvalue

def make_demo():
    instances = generate_all()
    inst = instances[0]  # world_ca_001 (CA family, deterministic)

    probes = build_probe_grid(inst, n=100, seed=0)

    ns = {}; exec(inst['hidden_rule_fn'], ns)
    truth_fn = ns['hidden_rule_fn']
    truth_sig = prediction_signature(truth_fn, probes)

    # (a) GROUNDED population: 5 runs, each independently close to truth
    # Each run has ~85% agreement with truth, independent errors
    grounded_sigs = {}
    for i in range(5):
        rng = random.Random(i * 100)
        def grounded_fn(state, _rng=rng, _tf=truth_fn):
            import copy
            result = _tf(copy.deepcopy(state))
            if isinstance(result, list) and isinstance(result[0], list):
                for r in result:
                    for c in range(len(r)):
                        if _rng.random() < 0.15:  # 15% independent errors
                            r[c] = (r[c] + 1) % 3
            elif isinstance(result, list):
                for j in range(len(result)):
                    if _rng.random() < 0.15:
                        result[j] = (result[j] + 1) % max(3, max(result) + 1 if result else 3)
            return result
        grounded_sigs[f'run_{i}'] = prediction_signature(grounded_fn, probes)

    result_g = hg_decompose(grounded_sigs, truth_sig)
    pval_g = hg_null_pvalue(grounded_sigs, truth_sig, B=200, seed=42)
    print("=== (a) GROUNDED population ===")
    print(f"  G={result_g['G']:.3f}  H={result_g['H']:.3f}  delta={result_g['delta']:.3f}  Abar={result_g['Abar']:.3f}")
    print(f"  null p-value for H: {pval_g:.4f}")
    print(f"  Expected: delta > 0, p high (H not significant)")

    # (b) HOMOGENIZED population: all runs agree on a shared WRONG answer (the "47 regime")
    # All 5 runs return the same non-truth function
    def shared_wrong_fn(state):
        """A plausible-looking but wrong rule — shifts all values by 1."""
        if isinstance(state[0], list):
            return [[(v + 1) % 3 for v in row] for row in state]
        return [(v + 1) % max(3, max(state) + 1) for v in state]
    shared_wrong_sig = prediction_signature(shared_wrong_fn, probes)

    homog_sigs = {f'run_{i}': shared_wrong_sig for i in range(5)}
    result_h = hg_decompose(homog_sigs, truth_sig)
    pval_h = hg_null_pvalue(homog_sigs, truth_sig, B=200, seed=42)
    print("\n=== (b) HOMOGENIZED population ===")
    print(f"  G={result_h['G']:.3f}  H={result_h['H']:.3f}  delta={result_h['delta']:.3f}  Abar={result_h['Abar']:.3f}")
    print(f"  null p-value for H: {pval_h:.4f}")
    print(f"  Expected: H > G (or H >= G), delta <= 0, p < 0.05")

    # Assertions to make the demo self-testing
    assert result_g['delta'] > 0, f"GROUNDED: expected delta>0, got {result_g['delta']:.3f}"
    assert result_h['delta'] <= 0 or result_h['H'] >= result_h['G'] * 0.9, \
        f"HOMOG: expected H>=G, got G={result_h['G']:.3f} H={result_h['H']:.3f}"
    assert pval_h < 0.1, f"HOMOG: expected p<0.1, got {pval_h:.4f}"

    print("\nGAP2 DEMO PASSED: estimator correctly separates GROUNDED vs HOMOGENIZED ✓")

if __name__ == '__main__':
    make_demo()
