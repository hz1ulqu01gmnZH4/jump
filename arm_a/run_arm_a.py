"""
Arm A: Qwen3.6-35B-A3B-MXFP4_MOE agent loop via llama-server at localhost:8080.
Runs on all 17 v2 instances, observes→intervenes→hypothesises→submits.
"""

import json
import sys
import time
import requests
from functools import partial

# Force unbuffered output
print = partial(print, flush=True)
from pathlib import Path

# Add repo root to path
repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))

from worlds.gen import generate_all
from harness import WorldHarness

# ── Tool specs ───────────────────────────────────────────────────────────────

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_train_obs",
            "description": "Get the training observations (state → next_state pairs). Call this first to understand the world.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "intervene",
            "description": "Apply an action to a world state and observe the result. Use this to probe the hidden rule.",
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "description": "The action name from the world's intervention_api",
                    },
                    "state": {
                        "description": "The current world state to modify (must match the state format from train_obs)"
                    },
                    "kwargs": {
                        "type": "object",
                        "description": "Additional action parameters (row, col, value, etc.)",
                    },
                },
                "required": ["action", "state"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "submit_hypothesis",
            "description": "Submit your final executable hypothesis. Python source defining hidden_rule_fn(state) -> next_state. This ends the interaction.",
            "parameters": {
                "type": "object",
                "properties": {
                    "hypothesis_source": {
                        "type": "string",
                        "description": "Python source code defining hidden_rule_fn(state) -> next_state. Must be a complete, runnable function.",
                    }
                },
                "required": ["hypothesis_source"],
            },
        },
    },
]

SYSTEM_PROMPT = """You are a scientific agent probing an unknown world to discover its hidden rule.

The world has a set of states and transitions. Your task is to:
1. Observe training examples (state → next_state pairs)
2. Intervene: apply actions to probe states and observe results
3. Formulate a hypothesis about the hidden rule
4. Submit an executable Python function: hidden_rule_fn(state) -> next_state

IMPORTANT:
- You must call submit_hypothesis exactly once, with valid Python source code
- The function must be named hidden_rule_fn
- It takes a state (same format as in train_obs) and returns the next state
- Test your hypothesis mentally before submitting
- The world uses invented primitive names — do not rely on domain knowledge
- Aim for exact correctness on held-out test observations

Available tools:
- get_train_obs: see the training data
- intervene: probe the world with interventions
- submit_hypothesis: submit your Python hypothesis (call once only)"""

LLAMA_URL = "http://localhost:8080/v1/chat/completions"
MAX_TURNS = 12
FALLBACK_HYPOTHESIS = "def hidden_rule_fn(state):\n    return state"


# ── LLM call ────────────────────────────────────────────────────────────────

def call_llm(messages: list, tools: list, max_tokens: int = 1024, tool_choice: str = "required") -> dict:
    payload = {
        "model": "local",
        "messages": messages,
        "tools": tools,
        "tool_choice": tool_choice,
        "max_tokens": max_tokens,
        "temperature": 0.3,
    }
    for attempt in range(2):
        try:
            resp = requests.post(LLAMA_URL, json=payload, timeout=180)
            resp.raise_for_status()
            choice = resp.json()["choices"][0]
            msg = choice["message"]
            msg["finish_reason"] = choice.get("finish_reason", "unknown")
            return msg
        except Exception as e:
            if attempt == 0:
                print(f"    LLM call failed (attempt 1): {e}, retrying in 5s...")
                time.sleep(5)
            else:
                raise


# ── Tool dispatch ────────────────────────────────────────────────────────────

def dispatch_tool(harness: WorldHarness, name: str, args: dict) -> dict:
    if name == "get_train_obs":
        return harness.get_train_obs()
    elif name == "intervene":
        action = args["action"]
        state = args["state"]
        kwargs = {k: v for k, v in args.get("kwargs", {}).items() if k != "state"}
        return harness.intervene(action, state=state, **kwargs)
    elif name == "submit_hypothesis":
        return harness.submit_hypothesis(args["hypothesis_source"])
    else:
        raise ValueError(f"Unknown tool: {name}")


# ── Agent loop per instance ──────────────────────────────────────────────────

def run_agent_on_instance(instance: dict) -> dict:
    harness = WorldHarness(instance)

    # Build context — no hidden_rule fields
    instance_context = {
        "id": instance["id"],
        "family": instance["family"],
        "difficulty": instance["difficulty"],
        "intervention_api": instance["intervention_api"],
        "primitive_glossary": instance["primitive_glossary"],
    }

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"World instance: {json.dumps(instance_context)}\n\n"
                "Begin by calling get_train_obs to see the training data. /nothink"
            ),
        },
    ]

    submitted = False
    for turn in range(MAX_TURNS):
        # After turn 9, force submit if not yet done
        forcing_submit = turn >= 9 and not submitted
        if forcing_submit:
            tools_this_turn = [t for t in TOOLS if t["function"]["name"] == "submit_hypothesis"]
            max_tokens_this_turn = 4096  # hypothesis can be long
        else:
            tools_this_turn = TOOLS
            max_tokens_this_turn = 2048  # enough thinking + tool args

        try:
            msg = call_llm(messages, tools_this_turn, max_tokens=max_tokens_this_turn, tool_choice="required")
        except Exception as e:
            print(f"    LLM error on turn {turn}: {e}")
            break

        messages.append(msg)

        tool_calls = msg.get("tool_calls")
        if not tool_calls:
            # No tool call — unusual with tool_choice=required; log and nudge
            content_preview = (msg.get("content") or "")[:60]
            reasoning_preview = (msg.get("reasoning_content") or "")[:60]
            print(f"    turn {turn}: NO_TOOL_CALL finish_reason={msg.get('finish_reason','?')} content={repr(content_preview)} reasoning={repr(reasoning_preview)}")
            if not submitted:
                messages.append({
                    "role": "user",
                    "content": "Please submit your hypothesis now using submit_hypothesis. /nothink",
                })
            continue

        stop_after = False
        for tc in tool_calls:
            name = tc["function"]["name"]
            raw_args = tc["function"]["arguments"]
            try:
                args = json.loads(raw_args)
            except json.JSONDecodeError as e:
                # Model sometimes emits literal newlines in JSON strings — fix and retry
                fixed = raw_args.replace('\n', '\\n').replace('\r', '\\r').replace('\t', '\\t')
                try:
                    args = json.loads(fixed)
                except json.JSONDecodeError:
                    print(f"    turn {turn}: {name} ARGS_PARSE_ERR: {e}")
                    result = {"error": f"Invalid JSON in arguments: {e}"}
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc["id"],
                        "content": json.dumps(result),
                    })
                    continue

            try:
                result = dispatch_tool(harness, name, args)
                if name == "submit_hypothesis":
                    print(f"    turn {turn}: submit_hypothesis → acc={result.get('accuracy',0):.3f}")
                else:
                    print(f"    turn {turn}: {name}")
            except Exception as e:
                result = {"error": str(e)}
                print(f"    turn {turn}: {name} ERROR: {str(e)[:80]}")

            messages.append({
                "role": "tool",
                "tool_call_id": tc["id"],
                "content": json.dumps(result),
            })

            if name == "submit_hypothesis" and "error" not in result:
                submitted = True
                stop_after = True

        if stop_after:
            break

    # Fallback: agent never submitted
    if not submitted:
        print(f"    Fallback submit for {instance['id']}")
        try:
            harness.submit_hypothesis(FALLBACK_HYPOTHESIS)
        except Exception:
            pass  # already submitted somehow

    log = harness.get_log()
    subs = [e for e in log if e["tool"] == "submit_hypothesis"]

    return {
        "instance_id": instance["id"],
        "family": instance["family"],
        "difficulty": instance["difficulty"],
        "accuracy": subs[-1]["accuracy"] if subs else 0.0,
        "hypothesis_source": subs[-1].get("hypothesis_source", "") if subs else "",
        "n_interventions": sum(1 for e in log if e["tool"] == "intervene"),
        "n_turns": len([m for m in messages if m.get("role") == "assistant"]),
        "fallback": not submitted,
    }


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    out_dir = Path(__file__).parent
    results_path = out_dir / "results_v2.json"

    instances = generate_all()
    print(f"Loaded {len(instances)} instances")

    results = []
    for i, inst in enumerate(instances):
        iid = inst["id"]
        fam = inst["family"]
        diff = inst["difficulty"]
        print(f"\n[{i+1:02d}/{len(instances)}] {iid} ({fam}, {diff})")
        t0 = time.time()

        try:
            r = run_agent_on_instance(inst)
        except Exception as e:
            print(f"    INSTANCE FAILED: {e}")
            r = {
                "instance_id": iid,
                "family": fam,
                "difficulty": diff,
                "accuracy": 0.0,
                "hypothesis_source": "",
                "n_interventions": 0,
                "n_turns": 0,
                "fallback": True,
                "error": str(e),
            }

        elapsed = time.time() - t0
        print(f"    acc={r['accuracy']:.3f}  interventions={r['n_interventions']}  turns={r['n_turns']}  {elapsed:.0f}s")
        results.append(r)

        # Save incrementally
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)

    # ── Summary ──
    print("\n" + "=" * 70)
    print(f"{'Instance':<20} {'Family':<20} {'Diff':<10} {'Acc':>6}")
    print("-" * 70)
    for r in results:
        print(f"{r['instance_id']:<20} {r['family']:<20} {r['difficulty']:<10} {r['accuracy']:>6.3f}")

    accs = [r["accuracy"] for r in results]
    mean_acc = sum(accs) / len(accs)
    print("-" * 70)
    print(f"{'MEAN':<20} {'':20} {'':10} {mean_acc:>6.3f}")

    # By family
    print()
    for fam in ["cellular_automata", "particle_system", "pattern_puzzle"]:
        fam_accs = [r["accuracy"] for r in results if r["family"] == fam]
        if fam_accs:
            print(f"  {fam}: mean={sum(fam_accs)/len(fam_accs):.3f}  n={len(fam_accs)}")

    # By difficulty
    print()
    for diff in sorted({r["difficulty"] for r in results}):
        diff_accs = [r["accuracy"] for r in results if r["difficulty"] == diff]
        if diff_accs:
            print(f"  {diff}: mean={sum(diff_accs)/len(diff_accs):.3f}  n={len(diff_accs)}")

    # vs controls
    print()
    print(f"  C-random  = 0.010")
    print(f"  C-induce  = 0.127")
    print(f"  C-retrieval = 1.000")
    print(f"  Arm A mean  = {mean_acc:.3f}")
    print("=" * 70)

    print(f"\nResults written to {results_path}")
    return mean_acc


if __name__ == "__main__":
    main()
