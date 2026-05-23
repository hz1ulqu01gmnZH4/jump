"""W3 run matrix generator and JSONL logger."""
import json, os, time, sys
from pathlib import Path
from datetime import datetime
from collections import defaultdict

sys.path.insert(0, '/home/ak/projects/jump')

from worlds.minimal_pairs import generate_minimal_pairs
from worlds.probes import build_probe_grid, prediction_signature
from harness import WorldHarness
from eval.hg import estimate_kappa_reference, hg_decompose, hg_kappa_sensitivity
from arm_e.agent import run_arm_e_agent


def build_matrix(worlds: list, models: list, conditions: list, seeds: list) -> list:
    """Return list of all (world_id, model_id, condition, seed) tuples."""
    return [
        (w, m, c, s)
        for w in worlds
        for m in models
        for c in conditions
        for s in seeds
    ]


def run_matrix(worlds: list, models: list, conditions: list, seeds: list,
               output_dir: str = 'arm_e/runs') -> tuple:
    """
    Run a fully-crossed matrix of Arm E agent episodes.

    Returns (records: list[dict], log_path: str).
    Each record = one run_record dict with all fields from the run_contract schema.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    log_path = Path(output_dir) / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"

    all_instances = {inst['id']: inst for inst in generate_minimal_pairs()}
    matrix = build_matrix(worlds, models, conditions, seeds)
    total = len(matrix)
    print(f"[run_matrix] {total} runs → {log_path}", flush=True)

    # Pre-compute kappa once per world (expensive, reused across all models/conditions/seeds)
    kappa_cache = {}
    probe_cache = {}
    truth_cache = {}
    ref_wrong_cache = {}

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

    records = []
    n_done = 0

    # Group by (world, model, condition) for within-cell hg analysis
    cell_sigs = defaultdict(dict)

    for world_id, model_id, condition, seed in matrix:
        inst = all_instances[world_id]
        harness = WorldHarness(inst)
        kappa = kappa_cache[world_id]
        truth_sig = truth_cache[world_id]
        probes = probe_cache[world_id]

        n_done += 1
        print(f"  [{n_done}/{total}] {world_id} | {model_id} | {condition} | seed={seed}",
              flush=True)

        agent_result = run_arm_e_agent(harness, model_id, condition, seed)

        # Compute prediction signature from hypothesis (for hg analysis)
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
            # Unique fallback per run (guaranteed wrong, non-coinciding)
            sig = tuple(('__fallback__', model_id, condition, seed, i)
                        for i in range(len(truth_sig)))

        cell_key = (world_id, model_id, condition)
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
            'seed': seed,
            # Agent outcome
            'accuracy': agent_result['accuracy'],
            'n_interventions': agent_result['n_interventions'],
            'n_turns': agent_result['n_turns'],
            'condition_intervene_called': agent_result['condition_intervene_called'],
            'hypothesis_src_len': len(agent_result.get('hypothesis_src') or ''),
            # hg metrics (filled in after all seeds for this cell are done)
            'kappa': kappa,
            'G': None, 'H': None, 'delta': None, 'kappa_robust': None,
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
              f"cost=${agent_result['cost_usd'] or 0:.5f} "
              f"err={agent_result['error'] or '-'}", flush=True)

    # Fill in hg metrics for each cell (all seeds collected)
    print("\n[run_matrix] Computing hg metrics per cell...", flush=True)
    # Re-read log and update G/H/delta/kappa_robust
    updated_records = []
    for r in records:
        cell_key = (r['world_id'], r['model'], r['condition'])
        sigs = cell_sigs[cell_key]
        truth_sig = truth_cache[r['world_id']]
        kappa = r['kappa']
        if len(sigs) >= 2:
            try:
                hg_result = hg_decompose(sigs, truth_sig, kappa)
                sens = hg_kappa_sensitivity(sigs, truth_sig, kappa, k_alphabet=3)
                r['G'] = round(hg_result['G'], 4)
                r['H'] = round(hg_result['H'], 4)
                r['delta'] = round(hg_result['delta'], 4)
                r['kappa_robust'] = sens['kappa_robust']
            except Exception as e:
                r['error'] = (r['error'] or '') + f' hg_error: {e}'
        updated_records.append(r)

    # Rewrite log with updated hg fields
    with open(log_path, 'w') as f:
        for r in updated_records:
            f.write(json.dumps(r) + '\n')

    print(f"[run_matrix] Done. {len(records)} records in {log_path}", flush=True)
    return updated_records, str(log_path)
