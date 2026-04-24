#!/usr/bin/env python3
"""Merge v2-P4 + v2-P6 re-run into final Arm B results."""
import json, statistics
from pathlib import Path

ROOT = Path("arm_b")
v2_path = ROOT / "results_v2.json"
p6_path = ROOT / "results_v2_p6.json"

if not v2_path.exists():
    raise FileNotFoundError(f"{v2_path} not found")
if not p6_path.exists():
    raise FileNotFoundError(f"{p6_path} not found")

v2 = {r["instance_id"]: r for r in json.loads(v2_path.read_text())}
p6 = {r["instance_id"]: r for r in json.loads(p6_path.read_text())}

merged = []
replaced_ids = []
kept_ids = []

for iid in sorted(v2.keys()):
    if iid in p6:
        merged.append(p6[iid])
        replaced_ids.append(iid)
    else:
        merged.append(v2[iid])
        kept_ids.append(iid)

(ROOT / "results_v2_merged.json").write_text(json.dumps(merged, indent=2))

print("=== Merge summary ===")
print(f"Replaced:  {len(replaced_ids)} entries from results_v2_p6.json")
print(f"Kept:      {len(kept_ids)} entries from results_v2.json")

scored = [r for r in merged if r.get("accuracy") is not None]
n_fail = len(merged) - len(scored)
mean_valid = statistics.mean(r["accuracy"] for r in scored) if scored else float("nan")
mean_pessim = statistics.mean(r.get("accuracy") or 0.0 for r in merged)

print(f"\n=== Accuracy (mean_valid = accuracy is not None) ===")
print(f"mean_valid:   {mean_valid:.3f}  (N={len(scored)} / {len(merged)})")
print(f"mean_pessim:  {mean_pessim:.3f}  (N={len(merged)}, FAIL -> 0.0)")
print(f"mean_v2_orig: 0.451  (N=17, baseline with silent fallbacks)")

print(f"\n=== Per-family ===")
for fam in ["cellular_automata", "particle_system", "pattern_puzzle"]:
    fam_all = [r for r in merged if r["family"] == fam]
    fam_scored = [r["accuracy"] for r in fam_all if r.get("accuracy") is not None]
    fam_pessim = statistics.mean(r.get("accuracy") or 0.0 for r in fam_all) if fam_all else float("nan")
    fam_valid = statistics.mean(fam_scored) if fam_scored else float("nan")
    print(
        f"{fam:20s}: n={len(fam_all)}  valid={len(fam_scored)}"
        f"  mean_valid={fam_valid:.3f}  mean_pessim={fam_pessim:.3f}"
    )

statuses = [
    r.get("hypothesis_status", "submitted" if not r.get("fallback") else "FAIL_LEGACY")
    for r in merged
]
print(f"\n=== FAIL counts ===")
print(f"FAIL_TIMEOUT:   {statuses.count('FAIL_TIMEOUT')}")
print(f"FAIL_NO_SUBMIT: {statuses.count('FAIL_NO_SUBMIT')}")
print(f"FAIL_EXCEPTION: {statuses.count('FAIL_EXCEPTION')}")

print(f"\nArm A reference: 0.343 over 17")
print(f"Output written to {ROOT / 'results_v2_merged.json'} ({len(merged)} entries)")
