#!/usr/bin/env python3
"""
Arm B v2: claude -p subprocess + jump_v2 skill.
Runs all 17 v2 instances with MAX_TURNS=12 (same as Arm A for fair comparison).

NOTE: --resume hangs in nested claude process context. We use fresh calls each
turn with accumulated interaction history baked into the prompt instead.
"""
import argparse, subprocess, json, sys, re, time, statistics
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
INSTANCE_CAP_SEC = 1800
TIMEOUT_TURN1 = 900
TIMEOUT_TURN_N = 600
RETRY_BACKOFFS = [0, 30, 60]  # 3 attempts total


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


def run_claude(prompt: str, timeout: int = TIMEOUT_TURN1) -> tuple[list, str]:
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
    timeout_turn = None
    timeout_attempts = 0
    n_turns = 0
    instance_start = time.monotonic()

    for turn in range(MAX_TURNS):
        n_turns = turn + 1

        # Per-instance hard cap
        if time.monotonic() - instance_start > INSTANCE_CAP_SEC:
            timeout_turn = turn
            print(f"\n    [instance cap exceeded @ turn {turn}]", flush=True)
            break

        prompt = build_prompt(skill_body, instance, history, turn)
        turn_timeout = TIMEOUT_TURN1 if turn == 0 else TIMEOUT_TURN_N

        tool_calls = None
        last_err = None
        for attempt, delay in enumerate(RETRY_BACKOFFS):
            if delay:
                time.sleep(delay)
            try:
                tool_calls, _ = run_claude(prompt, timeout=turn_timeout)
                break
            except subprocess.TimeoutExpired as e:
                timeout_attempts += 1
                last_err = e
                print(f"\n    [timeout turn={turn} attempt={attempt+1}/3 after {turn_timeout}s]",
                      flush=True)
            except Exception as e:
                last_err = e
                print(f"\n    [error turn={turn} attempt={attempt+1}/3]: {e}", flush=True)

        if tool_calls is None:
            # All attempts failed — record FAIL_TIMEOUT, abort instance. No identity fallback.
            timeout_turn = turn
            print(
                f"\n    [FAIL_TIMEOUT instance={instance['id']} turn={timeout_turn} "
                f"attempts={timeout_attempts} last_err={last_err!r}]",
                flush=True
            )
            break

        if not tool_calls:
            turns_left = MAX_TURNS - turn - 1
            if turns_left <= 2:
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

    # Build result — NO IDENTITY FALLBACK.
    log = harness.get_log()
    subs = [e for e in log if e["tool"] == "submit_hypothesis"]

    if submitted:
        return {
            "instance_id": instance["id"],
            "family": instance["family"],
            "difficulty": instance["difficulty"],
            "accuracy": subs[-1]["accuracy"],
            "hypothesis_source": subs[-1].get("hypothesis_source", ""),
            "hypothesis_status": "submitted",
            "n_interventions": sum(1 for e in log if e["tool"] == "intervene"),
            "n_turns": n_turns,
            "fallback": False,
        }
    else:
        status = "FAIL_TIMEOUT" if timeout_turn is not None else "FAIL_NO_SUBMIT"
        return {
            "instance_id": instance["id"],
            "family": instance["family"],
            "difficulty": instance["difficulty"],
            "accuracy": None,  # excluded from mean_valid
            "hypothesis_source": None,
            "hypothesis_status": status,
            "n_interventions": sum(1 for e in log if e["tool"] == "intervene"),
            "n_turns": n_turns,
            "fallback": True,
            "timeout_turn": timeout_turn,
            "timeout_attempts": timeout_attempts,
        }


def main():
    parser = argparse.ArgumentParser(description="Arm B v2 runner")
    parser.add_argument(
        "--only", nargs="+", metavar="INSTANCE_ID",
        help="Run only the specified instance IDs (space-separated)"
    )
    args = parser.parse_args()

    from worlds.gen import generate_all

    if not SKILL.exists():
        raise RuntimeError(f"v2 jump skill not found at {SKILL}. Create it first.")

    all_instances = generate_all()

    if args.only:
        only_set = set(args.only)
        instances = [i for i in all_instances if i["id"] in only_set]
        missing = only_set - {i["id"] for i in instances}
        if missing:
            raise ValueError(f"Unknown instance IDs: {sorted(missing)}")
        output_path = Path("arm_b/results_v2_p6.json")
    else:
        instances = all_instances
        output_path = Path("arm_b/results_v2.json")

    print(
        f"Arm B (claude -p + jump_v2 skill) — {len(instances)} instances, "
        f"MAX_TURNS={MAX_TURNS}, output={output_path}"
    )

    results = []
    fail_timeout_count = 0
    ESCALATION_THRESHOLD = 3

    for inst in instances:
        print(f"  {inst['id']} ({inst['family']}, {inst['difficulty']})...", end="", flush=True)
        try:
            r = run_arm_b_on_instance(inst)
        except Exception as e:
            r = {
                "instance_id": inst["id"],
                "family": inst["family"],
                "difficulty": inst["difficulty"],
                "accuracy": None,
                "hypothesis_source": None,
                "hypothesis_status": "FAIL_EXCEPTION",
                "n_interventions": 0,
                "n_turns": 0,
                "fallback": True,
                "error": str(e),
            }

        results.append(r)

        status = r.get("hypothesis_status", "submitted")
        acc_str = f"{r['accuracy']:.3f}" if r["accuracy"] is not None else "None"
        print(
            f" acc={acc_str} n_int={r['n_interventions']} turns={r['n_turns']} [{status}]",
            flush=True
        )

        # Machine-parseable log tag
        if status == "FAIL_TIMEOUT":
            total_wait = r.get("timeout_attempts", 0) * (TIMEOUT_TURN1 + TIMEOUT_TURN_N)
            print(
                f"[FAIL_TIMEOUT instance={r['instance_id']} turn={r.get('timeout_turn')} "
                f"attempts={r.get('timeout_attempts')} total_wait_s={total_wait}]",
                flush=True
            )
            fail_timeout_count += 1
        elif status == "FAIL_NO_SUBMIT":
            print(f"[FAIL_NO_SUBMIT instance={r['instance_id']} turn={r['n_turns']}]", flush=True)
        elif status == "FAIL_EXCEPTION":
            print(f"[FAIL_EXCEPTION instance={r['instance_id']} err={r.get('error', '')}]", flush=True)
        else:
            print(f"[OK instance={r['instance_id']} turns={r['n_turns']} acc={acc_str}]", flush=True)

        # Escalation check — stop if too many FAIL_TIMEOUT (structural hang, not fluke)
        if fail_timeout_count >= ESCALATION_THRESHOLD:
            print(
                f"\n[ESCALATION] {fail_timeout_count} FAIL_TIMEOUT instances detected. "
                f"Stopping run. Completed {len(results)}/{len(instances)} instances. "
                f"Structural hang suspected — investigate root cause before retrying.",
                flush=True
            )
            break

    # Write completed results regardless of escalation
    output_path.write_text(json.dumps(results, indent=2))
    print(f"\nResults written to {output_path} ({len(results)} entries)")

    # Summary statistics
    scored = [r for r in results if r.get("accuracy") is not None]
    n_excluded = len(results) - len(scored)
    mean_acc = statistics.mean(r["accuracy"] for r in scored) if scored else float("nan")
    print(
        f"\n=== Arm B mean accuracy: {mean_acc:.3f} over {len(scored)}/{len(results)} "
        f"(excluded {n_excluded} FAIL_*) ==="
    )
    if not args.only:
        arm_a_mean = 0.343
        print(f"Controls: C-random=0.010, C-induce=0.127, C-retrieval=1.000")
        print(f"Arm A reference: {arm_a_mean:.3f}")

        arm_a_fam = {
            "cellular_automata": 0.000,
            "particle_system": 0.167,
            "pattern_puzzle": 0.833,
        }
        for fam in ["cellular_automata", "particle_system", "pattern_puzzle"]:
            fam_results = [r for r in results if r["family"] == fam]
            fam_scored = [r["accuracy"] for r in fam_results if r.get("accuracy") is not None]
            if fam_scored:
                print(
                    f"  {fam}: Arm B={statistics.mean(fam_scored):.3f} "
                    f"(n={len(fam_scored)}/{len(fam_results)})  "
                    f"Arm A={arm_a_fam.get(fam, '?'):.3f}"
                )
            else:
                print(f"  {fam}: Arm B=(all FAIL)  Arm A={arm_a_fam.get(fam, '?'):.3f}")

    statuses = [r.get("hypothesis_status", "submitted") for r in results]
    print(
        f"FAIL counts: timeout={statuses.count('FAIL_TIMEOUT')} "
        f"no_submit={statuses.count('FAIL_NO_SUBMIT')} "
        f"exception={statuses.count('FAIL_EXCEPTION')}"
    )

    if fail_timeout_count >= ESCALATION_THRESHOLD:
        print(
            f"\nESCALATION: {fail_timeout_count} FAIL_TIMEOUT — re-run blocked. "
            f"Investigate claude -p hang before v2-P6c."
        )


if __name__ == "__main__":
    main()
