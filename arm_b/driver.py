#!/usr/bin/env python3
"""
Arm B driver: invokes the jump Claude Code skill on each eval task.
Uses Anthropic SDK if ANTHROPIC_API_KEY is set, otherwise falls back to
OpenRouter (OPENROUTER_API_KEY) with OpenAI-compatible interface.
Outputs conform to eval/output_schema.json.
Usage: uv run python arm_b/driver.py --eval eval/tasks.jsonl --out results/arm_b/
"""
import argparse, json, os, sys
from pathlib import Path

SKILL_FILE = Path.home() / ".claude" / "skills" / "jump.md"
MODEL_ANTHROPIC = "claude-sonnet-4-6"
MODEL_OPENROUTER = "anthropic/claude-sonnet-4-6"
OPENROUTER_BASE = "https://openrouter.ai/api/v1"


def load_skill() -> str:
    if not SKILL_FILE.exists():
        raise RuntimeError(f"Skill not found: {SKILL_FILE}")
    content = SKILL_FILE.read_text()
    lines = content.splitlines()
    if lines and lines[0].strip() == "---":
        end = next((i for i, l in enumerate(lines[1:], 1) if l.strip() == "---"), None)
        if end:
            content = "\n".join(lines[end + 1:]).strip()
    return content


def build_messages(skill: str, task: dict) -> tuple[str, list]:
    system = f"""You are performing Peircean abduction using the following skill protocol:

{skill}

IMPORTANT: Output ONLY a valid JSON object. No prose, no markdown fences. No discoverer names in the hypothesis field."""
    user = f"task_id: {task['id']}\nRule: {task['rule']}\nResult: {task['result']}\n\nPropose the Case:"
    return system, [{"role": "user", "content": user}]


def strip_fences(raw: str) -> str:
    raw = raw.strip()
    if raw.startswith("```"):
        lines = raw.splitlines()
        end = len(lines) - 1 if lines[-1].strip() == "```" else len(lines)
        raw = "\n".join(lines[1:end])
    return raw.strip()


def call_anthropic(skill: str, task: dict) -> dict:
    import anthropic
    client = anthropic.Anthropic()
    system, messages = build_messages(skill, task)
    msg = client.messages.create(
        model=MODEL_ANTHROPIC,
        max_tokens=2048,
        system=system,
        messages=messages,
    )
    raw = strip_fences(msg.content[0].text)
    return json.loads(raw)


def call_openrouter(skill: str, task: dict) -> dict:
    from openai import OpenAI
    api_key = os.environ["OPENROUTER_API_KEY"]
    client = OpenAI(api_key=api_key, base_url=OPENROUTER_BASE)
    system, messages = build_messages(skill, task)
    full_messages = [{"role": "system", "content": system}] + messages
    resp = client.chat.completions.create(
        model=MODEL_OPENROUTER,
        max_tokens=2048,
        messages=full_messages,
    )
    raw = strip_fences(resp.choices[0].message.content)
    return json.loads(raw)


def call_arm_b(skill: str, task: dict) -> dict:
    if os.environ.get("ANTHROPIC_API_KEY"):
        return call_anthropic(skill, task)
    elif os.environ.get("OPENROUTER_API_KEY"):
        return call_openrouter(skill, task)
    else:
        raise RuntimeError("Neither ANTHROPIC_API_KEY nor OPENROUTER_API_KEY is set")


def run(eval_path: Path, out_dir: Path) -> None:
    skill = load_skill()
    tasks = [json.loads(l) for l in eval_path.read_text().splitlines() if l.strip()]
    out_dir.mkdir(parents=True, exist_ok=True)
    required = [
        "task_id", "hypothesis", "mechanism", "observations_explained",
        "parsimony_justification", "novelty_justification",
    ]
    for task in tasks:
        try:
            output = call_arm_b(skill, task)
            missing = [f for f in required if f not in output]
            if missing:
                raise ValueError(f"Schema violation: missing {missing}")
            if not isinstance(output.get("observations_explained"), list) or len(output["observations_explained"]) < 1:
                raise ValueError("observations_explained must be non-empty list")
            output["task_id"] = task["id"]
            (out_dir / f"{task['id']}.json").write_text(json.dumps(output, indent=2))
            print(f"[OK] {task['id']}: {output['hypothesis'][:80]}")
        except json.JSONDecodeError as e:
            err = {"task_id": task["id"], "error": f"JSON parse failed: {e}"}
            (out_dir / f"{task['id']}.json").write_text(json.dumps(err, indent=2))
            print(f"[JSON-ERR] {task['id']}: {e}", file=sys.stderr)
        except Exception as e:
            err = {"task_id": task["id"], "error": str(e)}
            (out_dir / f"{task['id']}.json").write_text(json.dumps(err, indent=2))
            print(f"[ERR] {task['id']}: {e}", file=sys.stderr)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    run(args.eval, args.out)
