#!/usr/bin/env python3
"""
Arm B v2: claude -p subprocess + jump_v2 skill.
Runs all 17 v2 instances with MAX_TURNS=12 (same as Arm A for fair comparison).

NOTE: --resume hangs in nested claude process context. We use fresh calls each
turn with accumulated interaction history baked into the prompt instead.
"""
import subprocess, json, sys, re, time, statistics
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

CLAUDE_CMD = [
    "env", "-u", "CLAUDECODE", "-u", "ANTHROPIC_API_KEY",
    "claude", "-p",
    "--dangerously-skip-permissions",
    "--output-format", "stream-json", "--verbose",
]
SKILL = Path.home() / ".claude" / "skills" / "jump_v2.md"
MAX_TURNS = 12


def load_skill_body() -> str:
    text = SKILL.read_text()
    lines = text.splitlines()
    if lines and lines[0].strip() == "---":
        end = next((i for i, l in enumerate(lines[1:], 1) if l.strip() == "---"), None)
        if end:
            return "\n".join(lines[end + 1:]).strip()
    return text


def extract_json_tools(text: str) -> list:
    """Extract {"tool": ...} objects from text output."""
    calls = []
    seen = set()

    # Pass 1: line-by-line
    for line in text.splitlines():
        line = line.strip()
        if '"tool"' not in line or not line.startswith("{"):
            continue
        try:
            obj = json.loads(line)
            if "tool" in obj:
                key = json.dumps(obj, sort_keys=True)
                if key not in seen:
                    seen.add(key)
                    name = obj.pop("tool")
                    calls.append({"name": name, "input": obj})
        except json.JSONDecodeError:
            pass

    if calls:
        return calls

    # Pass 2: regex for multi-line JSON blocks
    for m in re.finditer(r'\{[^{}]*"tool"[^{}]*\}', text, re.DOTALL):
        try:
            obj = json.loads(m.group())
            if "tool" in obj:
                key = json.dumps(obj, sort_keys=True)
                if key not in seen:
                    seen.add(key)
                    name = obj.pop("tool")
                    calls.append({"name": name, "input": obj})
        except json.JSONDecodeError:
            pass

    return calls


def run_claude(prompt: str, timeout: int = 300) -> tuple[list, str]:
    """Fresh claude -p call (no --resume). Returns (tool_calls, final_text).
    Runs from /tmp to avoid loading tmux-agents/CLAUDE.md (huge system prompt).
    """
    proc = subprocess.run(
        CLAUDE_CMD, input=prompt, capture_output=True, text=True,
        timeout=timeout, cwd="/tmp"
    )

    if proc.returncode != 0:
        raise RuntimeError(f"claude -p rc={proc.returncode}: {proc.stderr[:400]}")

    tool_calls = []
    final_text = ""
    assistant_texts = []

    for line in proc.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            ev = json.loads(line)
        except json.JSONDecodeError:
            continue

        t = ev.get("type", "")
        if t == "assistant":
            for block in ev.get("message", {}).get("content", []):
                if block.get("type") == "text":
                    assistant_texts.append(block["text"])
                elif block.get("type") == "tool_use":
                    tool_calls.append({"name": block["name"], "input": block.get("input", {})})
        elif t == "result" and ev.get("subtype") == "success":
            final_text = ev.get("result", "")

    all_text = "\n".join(assistant_texts) or final_text
    if not tool_calls and all_text:
        tool_calls = extract_json_tools(all_text)

    return tool_calls, final_text


def build_prompt(skill_body: str, instance: dict, history: list, turn: int) -> str:
    ctx = {
        "id": instance["id"],
        "family": instance["family"],
        "difficulty": instance["difficulty"],
        "intervention_api": instance["intervention_api"],
        "primitive_glossary": instance["primitive_glossary"],
    }

    prompt = skill_body + f"\n\n---\nWorld instance: {json.dumps(ctx)}\n\n"

    if not history:
        prompt += (
            "Begin by calling get_train_obs to see the training observations.\n"
            "Output ONLY one JSON object on its own line:\n"
            '{"tool": "get_train_obs"}'
        )
    else:
        prompt += "## Interaction history so far:\n"
        for entry in history:
            prompt += f"\n### Tool: {entry['tool']}\n"
            if entry.get("args"):
                prompt += f"Args: {json.dumps(entry['args'])}\n"
            prompt += f"Result: {json.dumps(entry['result'])}\n"

        turns_left = MAX_TURNS - turn - 1
        if turns_left <= 1:
            prompt += (
                f"\n\nFINAL TURN — you MUST submit your hypothesis now.\n"
                "Output ONLY (replace ... with real Python):\n"
                '{"tool": "submit_hypothesis", "hypothesis_source": "def hidden_rule_fn(state):\\n    ..."}'
            )
        elif turns_left <= 3:
            prompt += (
                f"\n\n{turns_left} turns left. Submit your hypothesis or do one more intervention.\n"
                "Output ONLY one JSON object on its own line."
            )
        else:
            prompt += (
                f"\n\n{turns_left} turns remaining. Continue investigating or submit.\n"
                "Output ONLY one JSON object on its own line."
            )

    return prompt


def dispatch(harness, name: str, args: dict):
    if name == "get_train_obs":
        return harness.get_train_obs()
    elif name == "intervene":
        action = args.get("action")
        state = args.get("state")
        if action is None or state is None:
            return {"error": "intervene requires 'action' and 'state'"}
        kwargs = {k: v for k, v in args.items() if k not in ("action", "state")}
        return harness.intervene(action, state=state, **kwargs)
    elif name == "submit_hypothesis":
        src = args.get("hypothesis_source", "")
        return harness.submit_hypothesis(src)
    else:
        return {"error": f"Unknown tool: {name}"}


def run_arm_b_on_instance(instance: dict) -> dict:
    from harness import WorldHarness

    skill_body = load_skill_body()
    harness = WorldHarness(instance)
    history = []  # [{tool, args, result}, ...]
    submitted = False
    n_turns = 0

    for turn in range(MAX_TURNS):
        n_turns = turn + 1
        prompt = build_prompt(skill_body, instance, history, turn)

        try:
            tool_calls, _ = run_claude(prompt)
        except Exception as e:
            print(f"\n    [retry turn {turn}]: {e}", flush=True)
            time.sleep(15)
            try:
                tool_calls, _ = run_claude(prompt)
            except Exception as e2:
                print(f"\n    [fail turn {turn}]: {e2}", flush=True)
                break

        if not tool_calls:
            turns_left = MAX_TURNS - turn - 1
            if turns_left <= 2:
                # Force submit on next turn via history entry
                history.append({
                    "tool": "system",
                    "args": {},
                    "result": "No tool call detected. You MUST submit_hypothesis on the next turn."
                })
            continue

        for tc in tool_calls:
            try:
                res = dispatch(harness, tc["name"], tc["input"])
                history.append({"tool": tc["name"], "args": tc["input"], "result": res})
                if tc["name"] == "submit_hypothesis" and isinstance(res, dict) and "accuracy" in res:
                    submitted = True
            except Exception as e:
                history.append({"tool": tc["name"], "args": tc["input"], "result": {"error": str(e)}})

        if submitted:
            break

    if not submitted:
        try:
            harness.submit_hypothesis("def hidden_rule_fn(state): return state")
        except Exception:
            pass

    log = harness.get_log()
    subs = [e for e in log if e["tool"] == "submit_hypothesis"]
    return {
        "instance_id": instance["id"],
        "family": instance["family"],
        "difficulty": instance["difficulty"],
        "accuracy": subs[-1]["accuracy"] if subs else 0.0,
        "hypothesis_source": subs[-1].get("hypothesis_source", "") if subs else "",
        "n_interventions": sum(1 for e in log if e["tool"] == "intervene"),
        "n_turns": n_turns,
        "fallback": not submitted,
    }


def main():
    from worlds.gen import generate_all

    if not SKILL.exists():
        raise RuntimeError(f"v2 jump skill not found at {SKILL}. Create it first.")

    instances = generate_all()
    print(f"Arm B (claude -p + jump_v2 skill) — {len(instances)} instances, MAX_TURNS={MAX_TURNS}")

    results = []
    for inst in instances:
        print(f"  {inst['id']} ({inst['family']}, {inst['difficulty']})...", end="", flush=True)
        try:
            r = run_arm_b_on_instance(inst)
        except Exception as e:
            r = {
                "instance_id": inst["id"],
                "family": inst["family"],
                "difficulty": inst["difficulty"],
                "accuracy": 0.0,
                "hypothesis_source": f"# ERROR: {e}",
                "n_interventions": 0,
                "n_turns": 0,
                "fallback": True,
            }
        results.append(r)
        print(f" acc={r['accuracy']:.3f} n_int={r['n_interventions']} turns={r['n_turns']}", flush=True)

    Path("arm_b/results_v2.json").write_text(json.dumps(results, indent=2))

    mean_acc = statistics.mean(r["accuracy"] for r in results)
    arm_a_mean = 0.343
    print(f"\n=== Arm B mean accuracy: {mean_acc:.3f} (Arm A: {arm_a_mean:.3f}) ===")
    print(f"Controls: C-random=0.010, C-induce=0.127, C-retrieval=1.000")

    arm_a_fam = {
        "cellular_automata": 0.000,
        "particle_system": 0.167,
        "pattern_puzzle": 0.833,
    }
    for fam in ["cellular_automata", "particle_system", "pattern_puzzle"]:
        fam_accs = [r["accuracy"] for r in results if r["family"] == fam]
        if fam_accs:
            print(
                f"  {fam}: Arm B={statistics.mean(fam_accs):.3f}"
                f"  Arm A={arm_a_fam.get(fam, '?'):.3f}"
            )


if __name__ == "__main__":
    main()
