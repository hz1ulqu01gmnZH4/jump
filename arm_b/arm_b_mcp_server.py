#!/usr/bin/env python3
"""
Arm B v3 MCP server — stdio transport, one process per instance.
Receives JUMP_INSTANCE_ID and JUMP_RESULTS_PATH from environment.
"""
import asyncio, json, os, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import anyio
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp import types

# Module-level state (single instance, set in main() before server starts)
_harness = None
_instance = None
_results_path = None
_start_time = None

server = Server("jump-world")

TOOLS = [
    types.Tool(
        name="get_train_obs",
        description=(
            "Return the instance's training observations as a list of "
            "{state, next_state} pairs, plus intervention_api, primitive_glossary, "
            "family, and difficulty for this world. Call this first."
        ),
        inputSchema={
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        },
    ),
    types.Tool(
        name="intervene",
        description=(
            "Apply an action to a state and return the world's one-step transition. "
            "'action' must be one of the actions listed in the intervention_api "
            "from get_train_obs."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "action":   {"type": "string"},
                "state":    {"description": "The state to modify. Must match the shape returned by get_train_obs."},
                "row":      {"type": "integer"},
                "col":      {"type": "integer"},
                "value":    {},
                "pattern":  {"type": "array"},
                "x":        {"type": "integer"},
                "y":        {"type": "integer"},
                "type":     {"type": "string"},
                "vx":       {"type": "integer"},
                "vy":       {"type": "integer"},
                "new_type": {"type": "string"},
                "index":    {"type": "integer"},
                "delta":    {"type": "integer"},
                "mod":      {"type": "integer"},
            },
            "required": ["action", "state"],
            "additionalProperties": False,
        },
    ),
    types.Tool(
        name="submit_hypothesis",
        description=(
            "Submit the final hypothesis as Python source defining "
            "hidden_rule_fn(state) -> next_state. Scored against held-out test "
            "observations. Call exactly once per instance."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "hypothesis_source": {"type": "string"},
            },
            "required": ["hypothesis_source"],
            "additionalProperties": False,
        },
    ),
]


@server.list_tools()
async def list_tools() -> list[types.Tool]:
    return TOOLS


@server.call_tool()
async def call_tool(name: str, arguments: dict):
    global _harness, _instance, _results_path, _start_time

    if _harness is None:
        raise RuntimeError("Server not initialized — JUMP_INSTANCE_ID not set")

    if name == "get_train_obs":
        obs = _harness.get_train_obs()
        result = {
            "observations": obs,
            "intervention_api": _instance["intervention_api"],
            "primitive_glossary": _instance["primitive_glossary"],
            "family": _instance["family"],
            "difficulty": _instance["difficulty"],
        }
        return [types.TextContent(type="text", text=json.dumps(result))]

    elif name == "intervene":
        action = arguments.get("action")
        state = arguments.get("state")
        if action is None:
            return [types.TextContent(type="text", text=json.dumps({"error": "missing required argument: action"}))]
        if state is None:
            return [types.TextContent(type="text", text=json.dumps({"error": "missing required argument: state"}))]
        kwargs = {k: v for k, v in arguments.items() if k not in ("action", "state")}
        kwargs["state"] = state
        try:
            result = _harness.intervene(action, **kwargs)
        except ValueError as e:
            return [types.TextContent(type="text", text=json.dumps({"error": str(e)}))]
        return [types.TextContent(type="text", text=json.dumps(result))]

    elif name == "submit_hypothesis":
        hypothesis_source = arguments.get("hypothesis_source", "")
        try:
            result = _harness.submit_hypothesis(hypothesis_source)
        except (RuntimeError, ValueError, SyntaxError) as e:
            return [types.TextContent(type="text", text=json.dumps({"error": str(e)}))]

        # Write result to results file (atomic: tmp + rename)
        log = _harness.get_log()
        payload = {
            "instance_id": _instance["id"],
            "family": _instance["family"],
            "difficulty": _instance["difficulty"],
            "accuracy": result["accuracy"],
            "hypothesis_source": hypothesis_source,
            "interaction_log": log,
            "submitted": True,
            "t_end": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        if _results_path is not None:
            tmp_path = _results_path.with_suffix(".tmp")
            tmp_path.write_text(json.dumps(payload, indent=2))
            tmp_path.rename(_results_path)

        return [types.TextContent(type="text", text=json.dumps(result))]

    else:
        return [types.TextContent(type="text", text=json.dumps({"error": f"Unknown tool: {name}"}))]


async def async_main():
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options(),
        )


def main():
    global _harness, _instance, _results_path, _start_time

    instance_id = os.environ.get("JUMP_INSTANCE_ID")
    results_path_str = os.environ.get("JUMP_RESULTS_PATH")

    if not instance_id:
        print("FATAL: JUMP_INSTANCE_ID not set in environment", file=sys.stderr)
        sys.exit(1)
    if not results_path_str:
        print("FATAL: JUMP_RESULTS_PATH not set in environment", file=sys.stderr)
        sys.exit(1)

    _results_path = Path(results_path_str)
    _results_path.parent.mkdir(parents=True, exist_ok=True)
    _start_time = time.monotonic()

    from worlds.gen import generate_all
    from harness import WorldHarness

    all_instances = generate_all()
    matches = [i for i in all_instances if i["id"] == instance_id]
    if not matches:
        print(f"FATAL: instance_id '{instance_id}' not found in registry", file=sys.stderr)
        sys.exit(1)

    _instance = matches[0]
    _harness = WorldHarness(_instance)

    anyio.run(async_main)


if __name__ == "__main__":
    main()
