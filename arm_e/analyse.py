"""Analyse run log: pipeline health, condition checks, hg summary, cost."""
import json, sys
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, '/home/ak/projects/jump')


def analyse(log_path: str, label: str = ""):
    records = [json.loads(l) for l in Path(log_path).read_text().splitlines() if l.strip()]
    print(f"\n{'='*60}")
    print(f"ANALYSE{' — '+label if label else ''}: {len(records)} runs")
    print(f"{'='*60}\n")

    # (1) Adapter health
    print("=== (1) ADAPTER HEALTH ===")
    by_model = defaultdict(list)
    for r in records:
        by_model[r['model']].append(r)
    for model, runs in sorted(by_model.items()):
        errors = [r for r in runs if r.get('error')]
        with_acc = [r for r in runs if r.get('accuracy') is not None]
        avg_acc = (sum(r['accuracy'] for r in with_acc) / len(with_acc)) if with_acc else None
        acc_str = f"{avg_acc:.3f}" if avg_acc is not None else "N/A"
        print(f"  {model}: {len(runs)} runs | {len(errors)} errors | avg_acc={acc_str}")
        if errors:
            print(f"    errors: {[r['error'][:60] for r in errors[:3]]}")

    # (2) Log schema completeness
    print("\n=== (2) LOG SCHEMA ===")
    required = ['world_id','pair_id','family','model','condition','seed',
                'accuracy','kappa','cost_usd','prompt_tokens','completion_tokens',
                'G','H','delta','kappa_robust']
    missing_fields = defaultdict(list)
    for r in records:
        for f in required:
            if f not in r:
                missing_fields[f].append(r.get('world_id','?'))
    if missing_fields:
        for f, worlds in missing_fields.items():
            print(f"  MISSING field '{f}' in {len(worlds)} records")
    else:
        print(f"  All {len(required)} required fields present in all {len(records)} records ✓")

    # (3) hg pipeline
    print("\n=== (3) HG PIPELINE ===")
    with_hg = [r for r in records if r.get('G') is not None]
    print(f"  hg computed: {len(with_hg)}/{len(records)}")
    if with_hg:
        avg_G = sum(r['G'] for r in with_hg) / len(with_hg)
        avg_H = sum(r['H'] for r in with_hg) / len(with_hg)
        avg_d = sum(r['delta'] for r in with_hg) / len(with_hg)
        n_robust = sum(1 for r in with_hg if r.get('kappa_robust'))
        print(f"  avg G={avg_G:.3f} H={avg_H:.3f} δ={avg_d:.3f} | kappa_robust: {n_robust}/{len(with_hg)}")

    # (4) Condition discipline
    print("\n=== (4) CONDITION DISCIPLINE ===")
    passive = [r for r in records if r['condition'] == 'passive']
    passive_viol = [r for r in passive if r.get('condition_intervene_called')]
    on_runs = [r for r in records if r['condition'] == 'ON']
    on_intervened = [r for r in on_runs if r.get('n_interventions', 0) > 0]
    print(f"  passive: {len(passive)} runs | violations (intervene called): {len(passive_viol)}")
    print(f"  ON:      {len(on_runs)} runs | with interventions: {len(on_intervened)}")
    if passive_viol:
        print(f"  VIOLATION DETAIL: {[r['world_id']+'/'+r['model'] for r in passive_viol[:3]]}")

    # (5) Cost summary
    print("\n=== (5) COST SUMMARY ===")
    with_cost = [r for r in records if r.get('cost_usd') is not None]
    if with_cost:
        total = sum(r['cost_usd'] for r in with_cost)
        avg = total / len(with_cost)
        print(f"  Runs with cost: {len(with_cost)}/{len(records)}")
        print(f"  Total: ${total:.4f} | avg/run: ${avg:.5f}")
        extrap_1080 = avg * 1080
        print(f"  EXTRAPOLATED 1080-run full matrix: ${extrap_1080:.2f}")
    else:
        print(f"  No cost_usd data (stub runs: cost=0.0 expected)")
        stub_tokens = sum(r.get('prompt_tokens', 0) + r.get('completion_tokens', 0) for r in records)
        print(f"  Total stub tokens recorded: {stub_tokens}")

    # (6) Δδ preview (within-model homogenization signal)
    print("\n=== (6) Δδ PREVIEW (ON vs passive) ===")
    worlds = set(r['world_id'] for r in records)
    models = set(r['model'] for r in records)
    for world in sorted(worlds)[:3]:  # show first 3 worlds
        for model in sorted(models):
            on_r = [r for r in records if r['world_id']==world and r['model']==model
                    and r['condition']=='ON' and r.get('delta') is not None]
            pa_r = [r for r in records if r['world_id']==world and r['model']==model
                    and r['condition']=='passive' and r.get('delta') is not None]
            if on_r and pa_r:
                # Use first record's delta (all seeds in cell share same G/H/delta)
                d_on = on_r[0]['delta']
                d_pa = pa_r[0]['delta']
                dd = d_on - d_pa
                print(f"  {world[:20]} | {model}: δ_ON={d_on:.3f} δ_passive={d_pa:.3f} Δδ={dd:+.3f}")

    print(f"\n{'='*60}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*60}\n")
