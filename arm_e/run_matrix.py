"""W3 run matrix generator and JSONL logger."""
import argparse, json, os, sys, time
from pathlib import Path
from datetime import datetime
from collections import defaultdict

sys.path.insert(0, '/home/ak/projects/jump')

from worlds.minimal_pairs import generate_minimal_pairs
from worlds.probes import build_probe_grid, prediction_signature
from harness import WorldHarness
from eval.hg import (estimate_kappa_reference, hg_decompose, hg_kappa_sensitivity,
                     hg_bootstrap_ci, hg_null_pvalue)
from arm_e.agent import run_arm_e_agent, probe_llama_server


def _compute_k_w(kref: dict) -> int:
    """Compute per-world alphabet cardinality from reference wrong distribution."""
    ref_wrong_dist = kref['ref_wrong_dist']
    truth_sig_vals = []  # we need truth per probe; compute from kref context
    # k_w: max over probes of |set(ref_wrong_dist[i]) ∪ {truth_sig[i]}|
    # We approximate: truth contributes 1 extra distinct value per probe
    k_w = max(
        max((len(set(probe_dist)) + 1) for probe_dist in ref_wrong_dist if probe_dist),
        2
    )
    return k_w


def build_matrix(worlds: list, models: list, conditions: list, temps: list, seeds: list) -> list:
    """Return list of all (world_id, model_id, condition, temp, seed) 5-tuples."""
    return [
        (w, m, c, t, s)
        for w in worlds
        for m in models
        for c in conditions
        for t in temps
        for s in seeds
    ]


def run_matrix(worlds: list, models: list, conditions: list, seeds: list,
               output_dir: str = 'arm_e/runs', temps: list = None) -> tuple:
    """
    Run a fully-crossed matrix of Arm E agent episodes.

    Returns (records: list[dict], log_path: str).
    """
    if temps is None:
        temps = [1.0]

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    log_path = Path(output_dir) / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"

    # Probe server once — fail loudly if unreachable
    actual_model_name = probe_llama_server()

    all_instances = {inst['id']: inst for inst in generate_minimal_pairs()}
    matrix = build_matrix(worlds, models, conditions, temps, seeds)
    total = len(matrix)
    print(f"[run_matrix] {total} runs → {log_path}", flush=True)

    # Pre-compute kappa once per world
    kappa_cache = {}
    probe_cache = {}
    truth_cache = {}
    ref_wrong_cache = {}
    kref_cache = {}
    k_w_cache = {}

    for world_id in worlds:
        inst = all_instances[world_id]
        probes = build_probe_grid(inst, n=100, seed=0)
        ns = {}; exec(inst['hidden_rule_fn'], ns)
        truth_fn = ns['hidden_rule_fn']
        truth_sig = prediction_signature(truth_fn, probes)
        kref = estimate_kappa_reference(inst, probes, truth_sig, n_ref=100, seed=0)
        kappa_cache[world_id] = kref['kappa']
        probe_cache[world_id] = probes
        truth_cache[world_id] = truth_sig
        ref_wrong_cache[world_id] = kref['ref_wrong_dist']
        kref_cache[world_id] = kref

        # Per-world alphabet cardinality (fixes hardcoded k_alphabet=3 bug)
        k_w = max(
            max(
                len(set(probe_dist) | {truth_sig[j]})
                for j, probe_dist in enumerate(kref['ref_wrong_dist'])
            ),
            2
        )
        k_w_cache[world_id] = k_w

    records = []
    n_done = 0

    # Group by (world, model, condition, temp) for within-cell hg analysis
    cell_sigs = defaultdict(dict)

    for world_id, model_id, condition, temp, seed in matrix:
        inst = all_instances[world_id]
        harness = WorldHarness(inst)
        kappa = kappa_cache[world_id]
        truth_sig = truth_cache[world_id]
        probes = probe_cache[world_id]

        n_done += 1
        print(f"  [{n_done}/{total}] {world_id} | {model_id} | {condition} | temp={temp} | seed={seed}",
              flush=True)

        agent_result = run_arm_e_agent(harness, model_id, condition, seed,
                                       temp=temp, top_k=20, top_p=0.95)

        # Compute prediction signature from hypothesis
        sig = None
        if agent_result.get('hypothesis_src'):
            try:
                ns2 = {}
                exec(agent_result['hypothesis_src'], ns2)
                hyp_fn = ns2['hidden_rule_fn']
                sig = prediction_signature(hyp_fn, probes)
            except Exception:
                pass
        if sig is None:
            sig = tuple(('__fallback__', model_id, condition, temp, seed, i)
                        for i in range(len(truth_sig)))

        cell_key = (world_id, model_id, condition, temp)
        cell_sigs[cell_key][f'seed_{seed}'] = sig

        run_record = {
            # Identity
            'world_id': world_id,
            'pair_id': inst['pair_id'],
            'pair_member': inst.get('pair_member'),
            'family': inst['family'],
            'difficulty': inst['difficulty'],
            'model': model_id,
            'condition': condition,
            'temp': temp,
            'seed': seed,
            # Agent outcome
            'accuracy': agent_result['accuracy'],
            'n_interventions': agent_result['n_interventions'],
            'n_turns': agent_result['n_turns'],
            'condition_intervene_called': agent_result['condition_intervene_called'],
            'hypothesis_src': agent_result.get('hypothesis_src') or '',
            'hypothesis_src_len': len(agent_result.get('hypothesis_src') or ''),
            # hg metrics (filled in after all seeds for this cell are done)
            'kappa': kappa,
            'k_w': k_w_cache[world_id],
            'G': None, 'H': None, 'delta': None, 'kappa_robust': None,
            'delta_ci_lo': None, 'delta_ci_hi': None, 'hg_pval': None,
            # Cost / perf
            'wall_time_s': agent_result['wall_time_s'],
            'cost_usd': agent_result['cost_usd'],
            'prompt_tokens': agent_result['prompt_tokens'],
            'completion_tokens': agent_result['completion_tokens'],
            # Metadata
            'error': agent_result['error'],
            'timestamp': datetime.now().isoformat(),
        }

        with open(log_path, 'a') as f:
            f.write(json.dumps(run_record) + '\n')
        records.append(run_record)

        acc_str = f"{agent_result['accuracy']:.3f}" if agent_result['accuracy'] is not None else "None"
        print(f"    acc={acc_str} n_int={agent_result['n_interventions']} "
              f"wall={agent_result['wall_time_s']:.1f}s "
              f"err={agent_result['error'] or '-'}", flush=True)

    # Fill in hg metrics for each cell (all seeds collected)
    print("\n[run_matrix] Computing hg metrics per cell...", flush=True)
    updated_records = []
    for r in records:
        cell_key = (r['world_id'], r['model'], r['condition'], r['temp'])
        sigs = cell_sigs[cell_key]
        truth_sig = truth_cache[r['world_id']]
        kappa = r['kappa']
        k_w = r['k_w']
        ref_wrong_dist = ref_wrong_cache[r['world_id']]
        if len(sigs) >= 2:
            try:
                hg_result = hg_decompose(sigs, truth_sig, kappa)
                sens = hg_kappa_sensitivity(sigs, truth_sig, kappa, k_alphabet=k_w)
                ci = hg_bootstrap_ci(sigs, truth_sig, kappa, B=1000, seed=0)
                pval = hg_null_pvalue(sigs, truth_sig, kappa, ref_wrong_dist, B=1000, seed=0)
                r['G'] = round(hg_result['G'], 4)
                r['H'] = round(hg_result['H'], 4)
                r['delta'] = round(hg_result['delta'], 4)
                r['kappa_robust'] = sens['kappa_robust']
                r['delta_ci_lo'] = round(ci['delta_ci'][0], 4)
                r['delta_ci_hi'] = round(ci['delta_ci'][1], 4)
                r['hg_pval'] = round(pval, 4)
            except Exception as e:
                r['error'] = (r['error'] or '') + f' hg_error: {e}'
        updated_records.append(r)

    # Rewrite log with updated hg fields
    with open(log_path, 'w') as f:
        for r in updated_records:
            f.write(json.dumps(r) + '\n')

    print(f"[run_matrix] Done. {len(records)} records in {log_path}", flush=True)
    return updated_records, str(log_path)


def main():
    parser = argparse.ArgumentParser(description="W3 run matrix executor")
    parser.add_argument('--models', nargs='+', default=['stub'])
    parser.add_argument('--worlds', nargs='+', default=None)
    parser.add_argument('--conditions', nargs='+', default=['ON', 'passive'])
    parser.add_argument('--temps', nargs='+', type=float, default=[1.0])
    parser.add_argument('--seeds', nargs='+', type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument('--output-dir', default='arm_e/runs')
    args = parser.parse_args()

    if args.worlds is None:
        # Default: all A-member worlds
        from worlds.minimal_pairs import generate_minimal_pairs
        all_pairs = generate_minimal_pairs()
        args.worlds = [p['id'] for p in all_pairs if p.get('pair_member') == 'A']

    records, log_path = run_matrix(
        worlds=args.worlds,
        models=args.models,
        conditions=args.conditions,
        seeds=args.seeds,
        output_dir=args.output_dir,
        temps=args.temps,
    )
    print(f"\nLog: {log_path}")
    print(f"Records: {len(records)}")


if __name__ == '__main__':
    main()
