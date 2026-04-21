"""Quick smoke test — run all controls on all instances, print results."""
import sys
sys.path.insert(0, ".")
from worlds.gen import generate_all
from controls import run_controls

instances = generate_all()
print(f"Testing {len(instances)} instances...")
results = run_controls(instances)

import json
from pathlib import Path
Path("controls_baseline.json").write_text(json.dumps(results, indent=2))
print("\nControls baseline written to controls_baseline.json")

import statistics
for ctrl in ["C-random", "C-induce", "C-retrieval"]:
    scores = [results[iid][ctrl] for iid in results]
    print(f"{ctrl}: mean={statistics.mean(scores):.3f} min={min(scores):.2f} max={max(scores):.2f}")

# Assertions
retrieval_scores = [results[iid]["C-retrieval"] for iid in results]
failures = [(iid, results[iid]["C-retrieval"]) for iid in results if results[iid]["C-retrieval"] < 1.0]
if failures:
    print("\nFAIL: C-retrieval < 1.0 on:", failures)
    sys.exit(1)
else:
    print("\nPASS: C-retrieval = 1.0 on all instances")

random_mean = statistics.mean(results[iid]["C-random"] for iid in results)
if random_mean >= 0.3:
    print(f"WARN: C-random mean={random_mean:.3f} >= 0.3 (floor validation weak)")
else:
    print(f"PASS: C-random mean={random_mean:.3f} < 0.3 (floor confirmed)")
