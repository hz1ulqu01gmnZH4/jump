#!/usr/bin/env bash
# Wait for all 20 runs to complete, then compile results

REPO=/home/ak/tmux-agents/projects/jump/repo
LOG="$REPO/arm_a/e3_logs/master.log"

echo "Waiting for experiment to complete..."

until grep -q "E3 variance baseline complete" "$LOG"; do
    sleep 5
done

echo "Experiment complete! Compiling results..."

# Compile results using Python
python3 << 'EOF'
import json, glob, os

repo = '/home/ak/tmux-agents/projects/jump/repo'
log_dir = f'{repo}/arm_a/e3_logs'

def get_metrics(prefix, run_idx):
    f = f'{repo}/arm_a/results_v3_e3_{prefix}_run{run_idx}.json'
    jf = f'{log_dir}/v3_progress_{prefix}_run{run_idx}.jsonl'
    
    with open(f) as fp:
        r = json.load(fp)[0]
    
    first_t = "N/A"
    if os.path.exists(jf):
        with open(jf) as fp2:
            events = [json.loads(l) for l in fp2]
        intvs = [e for e in events if e.get('event') == 'tool_use' and 'intervene' in e.get('name','')]
        if intvs:
            first_t = f"{intvs[0]['t']:.1f}s"
    
    return {
        'status': r['hypothesis_status'],
        'accuracy': r['accuracy'],
        'n_int': r['n_interventions'],
        'wall_s': r['wall_s'],
        'first_intervene_t': first_t
    }

# Collect all metrics
results = {'ca': [], 'seq': [], 'pt': []}
for prefix in ['ca', 'seq', 'pt']:
    n = 10 if prefix == 'ca' else 5
    for i in range(1, n+1):
        results[prefix].append(get_metrics(prefix, i))

# Write compiled JSON
with open(f'{repo}/arm_a/e3_compiled_results.json', 'w') as f:
    json.dump(results, f, indent=2)
    
print("Compiled results written to arm_a/e3_compiled_results.json")
print("CA:", json.dumps(results['ca'], indent=2))
print("SEQ:", json.dumps(results['seq'], indent=2))
print("PT:", json.dumps(results['pt'], indent=2))
EOF
