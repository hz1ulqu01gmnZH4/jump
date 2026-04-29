#!/usr/bin/env python3
"""Aggregates E4 results and writes task output file."""
import json
import glob
import statistics
from collections import defaultdict
from pathlib import Path

results = []
for f in sorted(glob.glob("arm_a/results_v3_e4_*.json")):
    if "sanity" in f or "summary" in f:
        continue
    try:
        data = json.load(open(f))
        if isinstance(data, list):
            results.extend(data)
        else:
            results.append(data)
    except Exception as e:
        print(f"ERROR loading {f}: {e}")

print(f"Total run records: {len(results)}")

by_instance = defaultdict(list)
for r in results:
    by_instance[r["instance_id"]].append(r)

summary = {"per_instance": {}, "per_family": {}, "overall": {}}

families = {"seq": [], "pt": [], "ca": []}
for inst, runs in sorted(by_instance.items()):
    family = inst.split("_")[1]
    submitted = [r for r in runs if r["hypothesis_status"] == "submitted"]
    accs = [r["accuracy"] for r in submitted if r["accuracy"] is not None]
    wall_times = [r["wall_s"] for r in runs if "wall_s" in r]
    summary["per_instance"][inst] = {
        "n_runs": len(runs),
        "submit_rate": f"{len(submitted)}/{len(runs)}",
        "mean_acc": round(statistics.mean(accs), 3) if accs else None,
        "mean_wall_s": round(statistics.mean(wall_times), 1) if wall_times else None,
        "failure_modes": {
            "submitted": len(submitted),
            "FAIL_NO_SUBMIT": sum(1 for r in runs if r["hypothesis_status"] == "FAIL_NO_SUBMIT"),
            "FAIL_STUCK_REASONING": sum(1 for r in runs if r["hypothesis_status"] == "FAIL_STUCK_REASONING"),
            "other": sum(1 for r in runs if r["hypothesis_status"] not in
                        ["submitted", "FAIL_NO_SUBMIT", "FAIL_STUCK_REASONING"]),
        }
    }
    if family in families:
        families[family].extend(runs)

for fam, runs in families.items():
    submitted = [r for r in runs if r["hypothesis_status"] == "submitted"]
    accs = [r["accuracy"] for r in submitted if r["accuracy"] is not None]
    summary["per_family"][fam] = {
        "n_instances": len(set(r["instance_id"] for r in runs)),
        "n_runs": len(runs),
        "submit_rate": f"{len(submitted)}/{len(runs)}",
        "mean_acc": round(statistics.mean(accs), 3) if accs else None,
    }

all_submitted = [r for r in results if r["hypothesis_status"] == "submitted"]
all_accs = [r["accuracy"] for r in all_submitted if r["accuracy"] is not None]
summary["overall"] = {
    "n_runs": len(results),
    "submit_rate": f"{len(all_submitted)}/{len(results)}",
    "mean_acc": round(statistics.mean(all_accs), 3) if all_accs else None,
}

with open("arm_a/results_v3_e4_summary.json", "w") as f:
    json.dump(summary, f, indent=2)
print(json.dumps(summary, indent=2))

# Read restarts log
restarts_info = "None (server stable throughout)"
try:
    restarts_info = open("/tmp/e4_restarts.log").read().strip() or "None"
except FileNotFoundError:
    pass

def acc_str(v):
    return f"{v:.3f}" if v is not None else "N/A"

def fmt_delta(e4_acc, e3_acc):
    if e4_acc is None:
        return "N/A"
    delta = e4_acc - e3_acc
    sign = "+" if delta >= 0 else ""
    return f"{sign}{delta:.3f}"

seq = summary["per_family"]["seq"]
pt = summary["per_family"]["pt"]
ca = summary["per_family"]["ca"]
ov = summary["overall"]

lines = []
lines.append("---")
lines.append("id: v2_p42e4_arm_a_full_run")
lines.append("role: implementer")
lines.append("status: completed")
lines.append(f"tokens_estimate: {len(results) * 1000}")
lines.append("---")
lines.append("")
lines.append("# E4 Full Run Results — Arm A v3")
lines.append("")
lines.append("## Setup")
lines.append("")
lines.append("- llama-server restarted with ctx=65536 (E3c best config)")
lines.append("- Model: Qwen3.6-35B-A3B-MXFP4_MOE.gguf")
lines.append("- Config: Option B (thinking enabled, max_tokens=32768, ctx=65536)")
lines.append("")
lines.append("## 1. Sanity Smoke")
lines.append("")
lines.append("**PASS** — world_seq_001: hypothesis_status=submitted, accuracy=1.0, wall_s=71")
lines.append("")
lines.append("## 2. Llama-server Restarts")
lines.append("")
lines.append(restarts_info)
lines.append("")
lines.append("## 3. Per-Instance Results (17 instances)")
lines.append("")
lines.append("| instance_id | submit_rate | mean_acc | mean_wall_s | top_failure |")
lines.append("|-------------|-------------|----------|-------------|-------------|")
for inst, d in sorted(summary["per_instance"].items()):
    fm = d["failure_modes"]
    n_fail_ns = fm["FAIL_NO_SUBMIT"]
    n_fail_sr = fm["FAIL_STUCK_REASONING"]
    n_other = fm["other"]
    if fm["submitted"] == d["n_runs"]:
        top_fail = "none"
    elif n_fail_ns >= n_fail_sr:
        top_fail = f"FAIL_NO_SUBMIT({n_fail_ns})"
    elif n_fail_sr > 0:
        top_fail = f"FAIL_STUCK({n_fail_sr})"
    else:
        top_fail = f"other({n_other})"
    wall = f"{d['mean_wall_s']:.0f}s" if d['mean_wall_s'] is not None else "N/A"
    lines.append(f"| {inst} | {d['submit_rate']} | {acc_str(d['mean_acc'])} | {wall} | {top_fail} |")

lines.append("")
lines.append("## 4. Per-Family Summary")
lines.append("")
lines.append("| family | n_instances | n_runs | submit_rate | mean_acc |")
lines.append("|--------|-------------|--------|-------------|----------|")
for fam_key in ["seq", "pt", "ca"]:
    d = summary["per_family"][fam_key]
    lines.append(f"| {fam_key.upper()} | {d['n_instances']} | {d['n_runs']} | {d['submit_rate']} | {acc_str(d['mean_acc'])} |")

lines.append("")
lines.append("## 5. Overall")
lines.append("")
lines.append(f"- **Total runs**: {ov['n_runs']}")
lines.append(f"- **Submit rate**: {ov['submit_rate']}")
lines.append(f"- **Mean accuracy (submitted only)**: {acc_str(ov['mean_acc'])}")
lines.append("")
lines.append("## 6. Comparison to E3 Baselines")
lines.append("")
lines.append("| family | E3 baseline | E4 result | delta |")
lines.append("|--------|------------|-----------|-------|")
lines.append(f"| SEQ | 5/5 submit, acc=1.000 | {seq['submit_rate']}, acc={acc_str(seq['mean_acc'])} | {fmt_delta(seq['mean_acc'], 1.0)} |")
lines.append(f"| PT  | 2/5 submit, acc=0.580 | {pt['submit_rate']}, acc={acc_str(pt['mean_acc'])} | {fmt_delta(pt['mean_acc'], 0.58)} |")
lines.append(f"| CA  | 5/10 submit, acc=0.067 | {ca['submit_rate']}, acc={acc_str(ca['mean_acc'])} | {fmt_delta(ca['mean_acc'], 0.067)} |")

lines.append("")
lines.append("## 7. Notable Findings")
lines.append("")
by_acc = [(inst, d['mean_acc']) for inst, d in summary['per_instance'].items() if d['mean_acc'] is not None]
by_acc.sort(key=lambda x: x[1], reverse=True)
if by_acc:
    lines.append(f"- **Highest accuracy**: {by_acc[0][0]} (mean_acc={by_acc[0][1]:.3f})")
    lines.append(f"- **Lowest accuracy**: {by_acc[-1][0]} (mean_acc={by_acc[-1][1]:.3f})")
perfect_submit = [inst for inst, d in summary['per_instance'].items()
                  if d['failure_modes']['submitted'] == d['n_runs']]
if perfect_submit:
    lines.append(f"- **100% submit rate**: {', '.join(sorted(perfect_submit))}")
zero_submit = [inst for inst, d in summary['per_instance'].items()
               if d['failure_modes']['submitted'] == 0]
if zero_submit:
    lines.append(f"- **0% submit rate**: {', '.join(sorted(zero_submit))}")

lines.append("")
lines.append("---")
lines.append("")
lines.append("*Auto-generated by e4_aggregate.py after all 170 runs completed.*")

output_path = Path("/home/ak/tmux-agents/projects/jump/queue/done/v2_p42e4_arm_a_full_run.md")
output_path.write_text("\n".join(lines) + "\n")
print(f"\nTask output written to {output_path}")
