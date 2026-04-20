#!/usr/bin/env python3
"""
Arm A driver: sends abductive-jump eval tasks to local llama-server.
Usage: uv run python arm_a/driver.py --eval eval/tasks.jsonl --out results/arm_a/
"""
import argparse, json, sys
from pathlib import Path
import requests

LLAMA_SERVER = "http://localhost:8080/v1/chat/completions"
MODEL = "Qwen3.6-35B-A3B-MXFP4_MOE.gguf"

def run(eval_path: Path, out_dir: Path) -> None:
    tasks = [json.loads(l) for l in eval_path.read_text().splitlines() if l.strip()]
    out_dir.mkdir(parents=True, exist_ok=True)
    for task in tasks:
        # Real prompt — will be refined in P2
        messages = [
            {"role": "system", "content": "You are a scientific reasoner. Given a Rule and a Result, generate the most parsimonious novel explanatory Case (axiom) that would make the Result follow from the Rule. This is Peircean abduction — you must propose a genuinely new hypothesis, not pattern-match or recombine known facts."},
            {"role": "user", "content": f"Rule: {task['rule']}\nResult: {task['result']}\n\nPropose the Case (novel explanatory axiom):"}
        ]
        resp = requests.post(LLAMA_SERVER, json={
            "model": MODEL,
            "messages": messages,
            "temperature": 0.7,
            "max_tokens": 512,
        }, timeout=120)
        resp.raise_for_status()
        output = resp.json()["choices"][0]["message"]["content"]
        result = {"task_id": task["id"], "rule": task["rule"], "result": task["result"], "arm_a_output": output}
        (out_dir / f"{task['id']}.json").write_text(json.dumps(result, indent=2))
        print(f"[OK] {task['id']}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    run(args.eval, args.out)
