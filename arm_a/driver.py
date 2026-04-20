#!/usr/bin/env python3
"""
Arm A driver: sends abductive-jump eval tasks to local llama-server.
Outputs must conform to eval/output_schema.json.
Usage: uv run python arm_a/driver.py --eval eval/tasks.jsonl --out results/arm_a/
"""
import argparse, json, sys
from pathlib import Path
import requests

LLAMA_SERVER = "http://localhost:8080/v1/chat/completions"
MODEL = "Qwen3.6-35B-A3B-MXFP4_MOE.gguf"

SYSTEM = """You are a scientific reasoner performing Peircean abduction.

Given a Rule (general law/principle) and a Result (observed phenomenon), generate the most parsimonious novel explanatory Case (hypothesis/axiom) using this exact reasoning protocol:

1. CANDIDATE SURVEY: List 3-5 possible Cases silently. Label each:
   - "retrieval": most obvious/textbook answer
   - "historical_decoy": a competing hypothesis that was historically proposed but wrong
   - "abductive": a structurally novel hypothesis you are generating fresh

2. FILTER: Score candidates on parsimony (single mechanism vs ad-hoc patches), unifying scope (explains phenomena beyond the given Rule+Result), novelty (generated inference vs name-dropping), testability.

3. OUTPUT: Respond ONLY with a JSON object matching this schema exactly:
{
  "task_id": "<from input>",
  "hypothesis": "<1-3 sentences: the novel Case. DO NOT name the discoverer or theory by name>",
  "mechanism": "<1-3 sentences: how Case + Rule → Result causally>",
  "observations_explained": ["<phenomenon 1 beyond the given pair>", "<phenomenon 2>"],
  "parsimony_justification": "<1-2 sentences: why this is simplest; what it avoids>",
  "novelty_justification": "<1-2 sentences: what inferential step was taken vs retrieved>",
  "candidate_survey": [
    {"label": "retrieval", "description": "<...>"},
    {"label": "historical_decoy", "description": "<...>"},
    {"label": "abductive", "description": "<...>"}
  ]
}

Produce ONLY the JSON. No prose before or after. No discoverer names in hypothesis field."""


def call_llm(task: dict, temperature: float = 0.7, extra_instruction: str = "") -> dict:
    user_content = f"task_id: {task['id']}\nRule: {task['rule']}\nResult: {task['result']}\n\nPropose the Case:"
    if extra_instruction:
        user_content += f"\n\n{extra_instruction}"
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": user_content}
    ]
    resp = requests.post(LLAMA_SERVER, json={
        "model": MODEL,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": 1024,
        "chat_template_kwargs": {"enable_thinking": False},
    }, timeout=180)
    resp.raise_for_status()
    raw = resp.json()["choices"][0]["message"]["content"].strip()
    # Strip markdown fences if model wraps output
    if raw.startswith("```"):
        lines = raw.splitlines()
        raw = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
    return json.loads(raw)


def run(eval_path: Path, out_dir: Path) -> None:
    tasks = [json.loads(l) for l in eval_path.read_text().splitlines() if l.strip()]
    out_dir.mkdir(parents=True, exist_ok=True)
    required = ["task_id", "hypothesis", "mechanism", "observations_explained",
                "parsimony_justification", "novelty_justification"]
    json_errors = []
    for task in tasks:
        try:
            output = call_llm(task)
            missing = [f for f in required if f not in output]
            if missing:
                raise ValueError(f"Schema violation: missing fields {missing}")
            if not isinstance(output.get("observations_explained"), list) or len(output["observations_explained"]) < 1:
                raise ValueError("observations_explained must be a non-empty list")
            output["task_id"] = task["id"]  # enforce correct task_id
            (out_dir / f"{task['id']}.json").write_text(json.dumps(output, indent=2))
            print(f"[OK] {task['id']}: {output['hypothesis'][:80]}")
        except json.JSONDecodeError as e:
            json_errors.append(task)
            err = {"task_id": task["id"], "error": f"JSON parse failed: {e}", "raw": ""}
            (out_dir / f"{task['id']}.json").write_text(json.dumps(err, indent=2))
            print(f"[JSON-ERR] {task['id']}: {e}", file=sys.stderr)
        except Exception as e:
            err = {"task_id": task["id"], "error": str(e)}
            (out_dir / f"{task['id']}.json").write_text(json.dumps(err, indent=2))
            print(f"[ERR] {task['id']}: {e}", file=sys.stderr)

    # Retry JSON parse errors with lower temperature
    if json_errors:
        print(f"\nRetrying {len(json_errors)} JSON-parse failures at temperature=0.3...", file=sys.stderr)
        for task in json_errors:
            try:
                output = call_llm(task, temperature=0.3,
                                  extra_instruction="IMPORTANT: Output ONLY the JSON object. No other text.")
                missing = [f for f in required if f not in output]
                if missing:
                    raise ValueError(f"Schema violation: missing fields {missing}")
                if not isinstance(output.get("observations_explained"), list) or len(output["observations_explained"]) < 1:
                    raise ValueError("observations_explained must be a non-empty list")
                output["task_id"] = task["id"]
                (out_dir / f"{task['id']}.json").write_text(json.dumps(output, indent=2))
                print(f"[RETRY-OK] {task['id']}: {output['hypothesis'][:80]}")
            except Exception as e:
                err = {"task_id": task["id"], "error": f"Retry failed: {e}"}
                (out_dir / f"{task['id']}.json").write_text(json.dumps(err, indent=2))
                print(f"[RETRY-ERR] {task['id']}: {e}", file=sys.stderr)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    run(args.eval, args.out)
