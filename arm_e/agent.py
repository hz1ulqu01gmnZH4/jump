"""arm_e/agent.py — unified Arm E agent loop with configurable model adapters."""
import json, time, copy, os, sys, random
sys.path.insert(0, '/home/ak/projects/jump')


# ── Model adapter registry ─────────────────────────────────────────────────
# Each adapter is a callable: adapter(messages, tools_schema) -> (tool_calls, text, usage_dict)
# usage_dict: {'prompt_tokens': int, 'completion_tokens': int, 'cost_usd': float|None}

def _stub_adapter(messages, tools_schema, seed=0):
    """Stub adapter: returns a deterministic submit_hypothesis after 2 fake intervene calls.
    Used for infrastructure testing — no real API call."""
    rng = random.Random(seed + len(messages))
    # Detect passive condition from initial user message content
    is_passive = any(
        '[PASSIVE' in (m.get('content', '') or '')
        for m in messages if m.get('role') == 'user'
    )
    # Simulate: first 2 calls = intervene (ON only), then submit_hypothesis
    call_index = (len(messages) - 1) // 2  # approximate turn count
    if call_index < 2 and not is_passive:
        # Fake intervene call
        tool_calls = [{"name": "intervene", "input": {"action": "STUB_ACTION", "state": []}, "id": f"stub_{call_index}"}]
    else:
        # Fake submit_hypothesis
        src = (
            "def hidden_rule_fn(state):\n"
            "    # STUB: returns state unchanged\n"
            "    import copy\n"
            "    return copy.deepcopy(state)\n"
        )
        tool_calls = [{"name": "submit_hypothesis", "input": {"source": src}, "id": "stub_submit"}]
    usage = {"prompt_tokens": 100, "completion_tokens": 50, "cost_usd": 0.0}
    return tool_calls, ["stub text"], usage


# Registry: model_id -> adapter factory (returns bound adapter callable)
# Real adapters will be added here after model list is confirmed.
ADAPTER_REGISTRY = {
    "stub": lambda seed=0: (lambda msgs, tools: _stub_adapter(msgs, tools, seed=seed)),
}


def get_adapter(model_id: str, seed: int = 0):
    """Return an adapter callable for the given model_id."""
    if model_id not in ADAPTER_REGISTRY:
        raise ValueError(f"Unknown model_id '{model_id}'. Available: {list(ADAPTER_REGISTRY.keys())}")
    return ADAPTER_REGISTRY[model_id](seed=seed)


# ── Tool execution ─────────────────────────────────────────────────────────

def _execute_tool(harness, tc: dict, condition: str, result: dict) -> dict:
    """Execute a parsed tool call against WorldHarness. Returns output dict."""
    name = tc['name']
    inp = tc['input']

    if name == 'intervene':
        if condition == 'passive':
            result['condition_intervene_called'] = True
            return {'output': {'error': 'intervene not allowed in passive condition'}}
        result['n_interventions'] += 1
        result['condition_intervene_called'] = True
        try:
            out = harness.intervene(**inp)
            return {'output': out}
        except Exception as e:
            return {'output': {'error': str(e)}}

    elif name == 'submit_hypothesis':
        src = inp.get('source', '')
        result['hypothesis_src'] = src
        try:
            score_result = harness.submit_hypothesis(src)
            acc = score_result.get('accuracy', 0.0) if isinstance(score_result, dict) else float(score_result)
            return {'submitted': True, 'accuracy': acc, 'source': src, 'output': score_result}
        except Exception as e:
            return {'submitted': True, 'accuracy': 0.0, 'source': src,
                    'output': {'error': str(e)}}
    else:
        return {'output': {'error': f'unknown tool: {name}'}}


# ── Main agent run function ────────────────────────────────────────────────

TOOLS_SCHEMA = [
    {
        "name": "intervene",
        "description": "Apply an action to a state; returns {modified_state, next_state}.",
        "input_schema": {
            "type": "object",
            "required": ["action", "state"],
            "properties": {
                "action": {"type": "string"},
                "state": {"description": "The state to modify"},
                "row": {"type": "integer"}, "col": {"type": "integer"},
                "value": {}, "x": {"type": "integer"}, "y": {"type": "integer"},
                "type": {"type": "string"}, "index": {"type": "integer"},
                "vx": {"type": "integer"}, "vy": {"type": "integer"},
                "pattern": {"type": "array"}, "delta": {"type": "integer"},
                "new_type": {"type": "string"}, "mod": {"type": "integer"},
            }
        }
    },
    {
        "name": "submit_hypothesis",
        "description": "Submit your final rule as Python source defining hidden_rule_fn(state)->next_state.",
        "input_schema": {
            "type": "object",
            "required": ["source"],
            "properties": {"source": {"type": "string"}}
        }
    }
]


def run_arm_e_agent(harness, model_id: str, condition: str, seed: int,
                    max_turns: int = 8) -> dict:
    """
    Run one Arm E agent episode.

    model_id: key in ADAPTER_REGISTRY (e.g. 'stub', or real model IDs once confirmed)
    condition: 'ON' (intervention allowed) or 'passive' (no intervene)
    seed: integer seed for reproducibility

    Returns:
    {
        'accuracy': float|None,
        'n_interventions': int,
        'hypothesis_src': str|None,
        'n_turns': int,
        'cost_usd': float|None,
        'prompt_tokens': int,
        'completion_tokens': int,
        'wall_time_s': float,
        'condition_intervene_called': bool,
        'error': str|None,
    }
    """
    t0 = time.monotonic()
    adapter = get_adapter(model_id, seed=seed)

    train_obs = harness.get_train_obs()
    initial_content = (
        f"Training observations (seed={seed}, condition={condition}):\n\n"
        f"{json.dumps(train_obs, indent=2)}\n\n"
        f"intervention_api: {json.dumps(harness.instance['intervention_api'])}\n"
        f"primitive_glossary: {json.dumps(harness.instance['primitive_glossary'])}\n\n"
        "Discover the hidden rule and call submit_hypothesis."
    )
    if condition == 'passive':
        initial_content += "\n\n[PASSIVE: do NOT call intervene. Submit from training observations only.]"

    messages = [{"role": "user", "content": initial_content}]

    result = {
        'accuracy': None,
        'n_interventions': 0,
        'hypothesis_src': None,
        'n_turns': 0,
        'cost_usd': None,
        'prompt_tokens': 0,
        'completion_tokens': 0,
        'wall_time_s': 0.0,
        'condition_intervene_called': False,
        'error': None,
    }

    try:
        for turn in range(max_turns):
            result['n_turns'] = turn + 1

            tool_calls, text_blocks, usage = adapter(messages, TOOLS_SCHEMA)

            result['prompt_tokens'] += usage.get('prompt_tokens', 0)
            result['completion_tokens'] += usage.get('completion_tokens', 0)
            if usage.get('cost_usd') is not None:
                result['cost_usd'] = (result['cost_usd'] or 0.0) + usage['cost_usd']

            if not tool_calls:
                result['error'] = f'turn {turn}: adapter returned no tool_calls'
                break

            # Simulate assistant message in conversation
            messages.append({"role": "assistant", "content": text_blocks[0] if text_blocks else ""})

            submitted = False
            for tc in tool_calls:
                tool_result = _execute_tool(harness, tc, condition, result)
                if tool_result.get('submitted'):
                    result['accuracy'] = tool_result['accuracy']
                    result['hypothesis_src'] = tool_result['source']
                    submitted = True
                # Add tool result to messages for next turn
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.get('id', f'tc_{turn}'),
                    "content": json.dumps(tool_result.get('output', 'ok')),
                })

            if submitted:
                break

    except Exception as e:
        result['error'] = str(e)

    result['wall_time_s'] = time.monotonic() - t0
    return result
