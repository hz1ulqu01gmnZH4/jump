"""arm_e/agent.py — unified Arm E agent loop with configurable model adapters."""
import json, time, copy, os, sys, random, threading
sys.path.insert(0, '/home/ak/projects/jump')

LLAMA_BASE_URL = "http://127.0.0.1:8080/v1"
STUCK_REASONING_CAP_S = 600
STUCK_WATCHDOG_POLL_S = 5

# Cached actual model name (set on first probe, reused thereafter)
_ACTUAL_MODEL_CACHE: str | None = None


class StuckReasoningError(Exception):
    pass


# ── Model adapter registry ─────────────────────────────────────────────────
# Each adapter is a callable: adapter(messages, tools_schema) -> (tool_calls, text, usage_dict)
# usage_dict: {'prompt_tokens': int, 'completion_tokens': int, 'cost_usd': float|None}

def _stub_adapter(messages, tools_schema, seed=0):
    """Stub adapter: returns a deterministic submit_hypothesis after 2 fake intervene calls."""
    rng = random.Random(seed + len(messages))
    is_passive = any(
        '[PASSIVE' in (m.get('content', '') or '')
        for m in messages if m.get('role') == 'user'
    )
    call_index = (len(messages) - 1) // 2
    if call_index < 2 and not is_passive:
        tool_calls = [{"name": "intervene", "input": {"action": "STUB_ACTION", "state": []}, "id": f"stub_{call_index}"}]
    else:
        src = (
            "def hidden_rule_fn(state):\n"
            "    import copy\n"
            "    return copy.deepcopy(state)\n"
        )
        tool_calls = [{"name": "submit_hypothesis", "input": {"source": src}, "id": "stub_submit"}]
    usage = {"prompt_tokens": 100, "completion_tokens": 50, "cost_usd": 0.0}
    return tool_calls, ["stub text"], usage


def probe_llama_server(use_cache: bool = True) -> str:
    """Probe llama-server /v1/models and return actual model name. Fail loudly if not reachable."""
    global _ACTUAL_MODEL_CACHE
    if use_cache and _ACTUAL_MODEL_CACHE is not None:
        return _ACTUAL_MODEL_CACHE
    import urllib.request, urllib.error
    try:
        with urllib.request.urlopen(f"{LLAMA_BASE_URL}/models", timeout=10) as resp:
            data = json.loads(resp.read())
        models = data.get("data", [])
        if not models:
            raise RuntimeError("llama-server /v1/models returned empty model list")
        model_id = models[0]["id"]
        print(f"[llama-server] model confirmed: {model_id}", flush=True)
        _ACTUAL_MODEL_CACHE = model_id
        return model_id
    except urllib.error.URLError as e:
        raise RuntimeError(
            f"llama-server not reachable at {LLAMA_BASE_URL}: {e}\n"
            "Start with: llama-server -m /home/ak/Qwen3.6-35B-A3B-MXFP4_MOE.gguf --host 127.0.0.1 --port 8080"
        ) from e


def _tools_to_openai(tools_schema: list) -> list:
    return [{"type": "function", "function": {
        "name": t["name"],
        "description": t["description"],
        "parameters": t["input_schema"],
    }} for t in tools_schema]


def _qwen_local_adapter(messages, tools_schema, seed=0, temp=1.0, top_k=20, top_p=0.95):
    """Real local Qwen adapter using OpenAI SDK sync client against llama-server."""
    from openai import OpenAI

    actual_model = probe_llama_server()
    client = OpenAI(base_url=LLAMA_BASE_URL, api_key="local")
    openai_tools = _tools_to_openai(tools_schema)

    # Stuck-reasoning watchdog state
    last_tool_time = [time.monotonic()]
    tool_count = [0]
    abort_event = threading.Event()
    stuck_error = [None]

    def watchdog():
        while not abort_event.is_set():
            time.sleep(STUCK_WATCHDOG_POLL_S)
            if abort_event.is_set():
                return
            since = time.monotonic() - last_tool_time[0]
            if since > STUCK_REASONING_CAP_S and tool_count[0] >= 1:
                stuck_error[0] = StuckReasoningError(
                    f"stuck reasoning: {since:.1f}s without tool_use after {tool_count[0]} calls"
                )
                abort_event.set()
                return

    wt = threading.Thread(target=watchdog, daemon=True)
    wt.start()

    total_prompt = 0
    total_completion = 0

    try:
        # Single-turn call: messages already contains the full conversation
        if abort_event.is_set():
            raise stuck_error[0] or StuckReasoningError("aborted before call")

        try:
            resp = client.chat.completions.create(
                model=actual_model,
                messages=messages,
                tools=openai_tools,
                tool_choice="auto",
                max_tokens=32768,
                temperature=temp,
                top_p=top_p,
                seed=seed,
                timeout=600.0,
                extra_body={"top_k": top_k},
            )
        except Exception as e:
            print(f"[qwen_local] LLM call error: {e}", file=sys.stderr, flush=True)
            raise

        if resp.usage:
            total_prompt += resp.usage.prompt_tokens or 0
            total_completion += resp.usage.completion_tokens or 0

        msg = resp.choices[0].message
        tool_calls_raw = msg.tool_calls

        if not tool_calls_raw:
            usage = {"prompt_tokens": total_prompt, "completion_tokens": total_completion, "cost_usd": 0.0}
            return [], [msg.content or ""], usage

        tool_count[0] += len(tool_calls_raw)
        last_tool_time[0] = time.monotonic()

        parsed = []
        for tc in tool_calls_raw:
            try:
                args = json.loads(tc.function.arguments)
            except json.JSONDecodeError as e:
                print(f"[qwen_local] args parse error for {tc.function.name}: {e}", file=sys.stderr, flush=True)
                args = {}
            parsed.append({"name": tc.function.name, "input": args, "id": tc.id})

        usage = {"prompt_tokens": total_prompt, "completion_tokens": total_completion, "cost_usd": 0.0}
        return parsed, [msg.content or ""], usage

    finally:
        abort_event.set()
        wt.join(timeout=1.0)


# Registry: model_id -> adapter factory (returns bound adapter callable)
ADAPTER_REGISTRY = {
    "stub": lambda seed=0, temp=1.0, top_k=20, top_p=0.95: (
        lambda msgs, tools: _stub_adapter(msgs, tools, seed=seed)
    ),
    "qwen_local": lambda seed=0, temp=1.0, top_k=20, top_p=0.95: (
        lambda msgs, tools: _qwen_local_adapter(msgs, tools, seed=seed, temp=temp, top_k=top_k, top_p=top_p)
    ),
}


def get_adapter(model_id: str, seed: int = 0, temp: float = 1.0, top_k: int = 20, top_p: float = 0.95):
    """Return an adapter callable for the given model_id."""
    if model_id not in ADAPTER_REGISTRY:
        raise ValueError(f"Unknown model_id '{model_id}'. Available: {list(ADAPTER_REGISTRY.keys())}")
    return ADAPTER_REGISTRY[model_id](seed=seed, temp=temp, top_k=top_k, top_p=top_p)


# ── Tool execution ─────────────────────────────────────────────────────────

def _execute_tool(harness, tc: dict, condition: str, result: dict) -> dict:
    """Execute a parsed tool call against WorldHarness. Returns output dict."""
    name = tc['name']
    inp = tc['input']

    if name == 'intervene':
        if condition == 'passive':
            # Belt-and-suspenders: structural gating should prevent this
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
                    max_turns: int = 8, temp: float = 1.0, top_k: int = 20,
                    top_p: float = 0.95) -> dict:
    """
    Run one Arm E agent episode.

    model_id: key in ADAPTER_REGISTRY
    condition: 'ON' (intervention allowed) or 'passive' (no intervene)
    seed: integer seed for reproducibility
    """
    t0 = time.monotonic()
    adapter = get_adapter(model_id, seed=seed, temp=temp, top_k=top_k, top_p=top_p)

    # Structural passive gating: passive condition never sees intervene tool
    if condition == 'passive':
        tools_for_adapter = [t for t in TOOLS_SCHEMA if t['name'] == 'submit_hypothesis']
    else:
        tools_for_adapter = TOOLS_SCHEMA

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

            tool_calls, text_blocks, usage = adapter(messages, tools_for_adapter)

            result['prompt_tokens'] += usage.get('prompt_tokens', 0)
            result['completion_tokens'] += usage.get('completion_tokens', 0)
            if usage.get('cost_usd') is not None:
                result['cost_usd'] = (result['cost_usd'] or 0.0) + usage['cost_usd']

            if not tool_calls:
                result['error'] = usage.get('error') or f'turn {turn}: adapter returned no tool_calls'
                break

            # Build assistant message dict with tool_calls if available
            assistant_msg: dict = {"role": "assistant", "content": text_blocks[0] if text_blocks else ""}
            # For OpenAI-style tool calls, also attach tool_calls to the assistant message
            if tool_calls and model_id != 'stub':
                assistant_msg["tool_calls"] = [
                    {
                        "id": tc.get('id', f'tc_{turn}_{i}'),
                        "type": "function",
                        "function": {"name": tc['name'], "arguments": json.dumps(tc['input'])},
                    }
                    for i, tc in enumerate(tool_calls)
                ]
            messages.append(assistant_msg)

            submitted = False
            for tc in tool_calls:
                tool_result = _execute_tool(harness, tc, condition, result)
                if tool_result.get('submitted'):
                    result['accuracy'] = tool_result['accuracy']
                    result['hypothesis_src'] = tool_result['source']
                    submitted = True
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.get('id', f'tc_{turn}'),
                    "content": json.dumps(tool_result.get('output', 'ok')),
                })

            if submitted:
                break

    except StuckReasoningError as e:
        result['error'] = str(e)
    except Exception as e:
        result['error'] = str(e)

    result['wall_time_s'] = time.monotonic() - t0
    return result
