#!/usr/bin/env python3
"""
Arm A v3 harness: Path B — in-process OpenAI SDK + MCP client agentic loop.
Mirrors run_arm_b_v3.py structure with llama-server (OpenAI-compat) instead of claude.

Tool-calling failure diagnosis (REQ-2):
  Symptom: n_interventions=0, accuracy=null after full run
  Diagnosis: inspect raw llama-server response — curl http://127.0.0.1:8080/v1/chat/completions
             with tools list and check "tool_calls" field in response JSON.
  Fallback: if Qwen3.6-35B-A3B-MXFP4_MOE tool-calls are broken, try
            Qwen3.6-27B-Q6_K.gguf (also on disk at /home/ak/Qwen3.6-27B-Q6_K.gguf).
"""
import argparse, asyncio, json, os, statistics, sys, threading, time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from openai import AsyncOpenAI
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from arm_b.common.prompts import DEADLINE_NUDGE  # byte-identical with Arm B

LLAMA_BASE_URL = "http://127.0.0.1:8080/v1"
SKILL = Path.home() / ".claude" / "skills" / "jump_v3.md"
SKILL_FALLBACK = Path(__file__).parent.parent / "arm_b" / "jump_v3_skill.md"
MCP_RESULTS_DIR = Path("arm_a/.mcp_results")
REPO_ROOT = str(Path(__file__).parent.parent)

INSTANCE_HARD_CAP = 7200
RETRY_ON_TIMEOUT = 1
STUCK_REASONING_CAP_S = 600
STUCK_WATCHDOG_POLL_S = 5


class StuckReasoningError(Exception):
    def __init__(self, wall_s, since_last_tool_s, tool_use_count):
        self.wall_s = wall_s
        self.since_last_tool_s = since_last_tool_s
        self.tool_use_count = tool_use_count
        super().__init__(
            f"stuck reasoning: {since_last_tool_s:.1f}s without tool_use "
            f"after {tool_use_count} tool calls (cap={STUCK_REASONING_CAP_S}s)"
        )


@dataclass
class AgentState:
    last_tool_t: float = 0.0
    tool_count: int = 0
    intervene_count: int = 0
    abort_event: threading.Event = field(default_factory=threading.Event)
    lock: threading.Lock = field(default_factory=threading.Lock)
    # Set by watchdog when it triggers stuck-reasoning kill
    stuck_triggered: bool = False
    stuck_since_s: float = 0.0
    stuck_count_at_trigger: int = 0


def load_skill_body() -> str:
    skill_path = SKILL if SKILL.exists() else SKILL_FALLBACK
    text = skill_path.read_text()
    lines = text.splitlines()
    if lines and lines[0].strip() == "---":
        end = next((i for i, l in enumerate(lines[1:], 1) if l.strip() == "---"), None)
        if end:
            return "\n".join(lines[end + 1:]).strip()
    return text


def mcp_tools_to_openai(tools) -> list:
    """Translate MCP tool list to OpenAI function-calling schema.
    Prefix mcp__jump-world__ mirrors Arm B's tool namespace from claude MCP."""
    return [
        {
            "type": "function",
            "function": {
                "name": f"mcp__jump-world__{t.name}",
                "description": t.description,
                "parameters": t.inputSchema,
            },
        }
        for t in tools
    ]


def _append_progress(progress_file, record: dict) -> None:
    if progress_file is not None:
        progress_file.write(json.dumps(record) + "\n")
        progress_file.flush()


def _log_tool_event(progress_file, event_type: str, t: float, **kwargs) -> None:
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {event_type}: " + " ".join(f"{k}={v}" for k, v in kwargs.items()),
          file=sys.stderr, flush=True)
    _append_progress(progress_file, {"t": round(t, 2), "event": event_type, **kwargs})


def _build_assistant_dict(msg) -> dict:
    """Convert OpenAI ChatCompletionMessage to dict safe for messages list."""
    d: dict = {"role": "assistant"}
    if msg.content:
        d["content"] = msg.content
    if msg.tool_calls:
        d["tool_calls"] = [
            {
                "id": tc.id,
                "type": "function",
                "function": {"name": tc.function.name, "arguments": tc.function.arguments},
            }
            for tc in msg.tool_calls
        ]
    return d


def _watchdog(state: AgentState, t0: float, progress_file) -> None:
    while not state.abort_event.is_set():
        time.sleep(STUCK_WATCHDOG_POLL_S)
        if state.abort_event.is_set():
            return
        with state.lock:
            since = (time.monotonic() - t0) - state.last_tool_t
            count = state.tool_count
        if since > STUCK_REASONING_CAP_S and count >= 1:
            elapsed = time.monotonic() - t0
            _append_progress(progress_file, {
                "t": round(elapsed, 2),
                "event": "stuck_reasoning_kill",
                "since_last_tool_s": round(since, 2),
                "tool_use_count": count,
                "cap_s": STUCK_REASONING_CAP_S,
            })
            with state.lock:
                state.stuck_triggered = True
                state.stuck_since_s = since
                state.stuck_count_at_trigger = count
            state.abort_event.set()
            return


async def _run_instance_async(
    instance: dict,
    actual_model: str,
    state: AgentState,
    t0: float,
    progress_file,
) -> dict:
    results_path = MCP_RESULTS_DIR / f"{instance['id']}.json"
    if results_path.exists():
        results_path.unlink()

    params = StdioServerParameters(
        command="uv",
        args=["run", "python", "arm_a/arm_a_mcp_server.py"],
        cwd=REPO_ROOT,
        env={
            **os.environ,
            "JUMP_INSTANCE_ID": instance["id"],
            "JUMP_RESULTS_PATH": str(results_path),
        },
    )

    client = AsyncOpenAI(base_url=LLAMA_BASE_URL, api_key="local")

    async with stdio_client(params) as (r, w):
        async with ClientSession(r, w) as session:
            await session.initialize()
            tools = (await session.list_tools()).tools
            openai_tools = mcp_tools_to_openai(tools)

            skill_body = load_skill_body()
            messages = [
                {"role": "system", "content": skill_body + "\n\n" + DEADLINE_NUDGE},
                {"role": "user", "content": "Begin by calling get_train_obs."},
            ]

            submit_called = False

            while not state.abort_event.is_set() and not submit_called:
                # OpenAI SDK requires extra_body to pass chat_template_kwargs to llama-server.
                # The "/nothink" string in user messages does NOT reach the Jinja template.
                # The old arm_a/driver.py had the correct implementation using raw requests;
                # this is the OpenAI-SDK-compatible equivalent.
                try:
                    resp = await client.chat.completions.create(
                        model=actual_model,
                        messages=messages,
                        tools=openai_tools,
                        tool_choice="auto",
                        max_tokens=16384,
                        timeout=300.0,
                        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
                    )
                except Exception as e:
                    ts = datetime.now().strftime("%H:%M:%S")
                    print(f"[{ts}] LLM call error: {e}", file=sys.stderr, flush=True)
                    _append_progress(progress_file, {
                        "t": round(time.monotonic() - t0, 2),
                        "event": "llm_error",
                        "error": str(e),
                    })
                    break

                msg = resp.choices[0].message
                messages.append(_build_assistant_dict(msg))

                ts = datetime.now().strftime("%H:%M:%S")
                if not msg.tool_calls:
                    print(f"[{ts}] no tool_calls — finish_reason={resp.choices[0].finish_reason}",
                          file=sys.stderr, flush=True)
                    _append_progress(progress_file, {
                        "t": round(time.monotonic() - t0, 2),
                        "event": "no_tool_call",
                        "finish_reason": resp.choices[0].finish_reason,
                    })
                    break

                for tc in msg.tool_calls:
                    full_name = tc.function.name
                    mcp_name = full_name.removeprefix("mcp__jump-world__")
                    try:
                        args = json.loads(tc.function.arguments)
                    except json.JSONDecodeError as e:
                        args = {}
                        print(f"[{ts}] args parse error for {full_name}: {e}", file=sys.stderr, flush=True)

                    elapsed = time.monotonic() - t0
                    _log_tool_event(progress_file, "tool_use", t=elapsed,
                                    name=full_name, input_keys=list(args.keys()))
                    with state.lock:
                        state.last_tool_t = elapsed
                        state.tool_count += 1
                        if mcp_name == "intervene":
                            state.intervene_count += 1

                    try:
                        result = await session.call_tool(mcp_name, args)
                        content = result.content[0].text if result.content else "{}"
                    except Exception as e:
                        content = json.dumps({"error": str(e)})
                        print(f"[{ts}] MCP call error {mcp_name}: {e}", file=sys.stderr, flush=True)

                    elapsed2 = time.monotonic() - t0
                    _log_tool_event(progress_file, "tool_result", t=elapsed2,
                                    id=tc.id, content_len=len(content))
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": content,
                    })

                    if mcp_name == "submit_hypothesis":
                        submit_called = True

    wall_s = time.monotonic() - t0
    state.abort_event.set()  # stop watchdog

    if state.stuck_triggered and not submit_called:
        with state.lock:
            raise StuckReasoningError(wall_s, state.stuck_since_s, state.stuck_count_at_trigger)

    # Primary: MCP-side persisted file
    if results_path.exists():
        try:
            payload = json.loads(results_path.read_text())
            return {
                "instance_id": instance["id"],
                "family": instance["family"],
                "difficulty": instance["difficulty"],
                "accuracy": payload.get("accuracy"),
                "hypothesis_source": payload.get("hypothesis_source"),
                "hypothesis_status": "submitted",
                "n_interventions": state.intervene_count,
                "n_turns": None,
                "fallback": False,
                "capture_path": "mcp_file",
                "wall_s": wall_s,
                "model": actual_model,
                "effort": None,
                "arm": "A",
                "version": "v3",
            }
        except (json.JSONDecodeError, KeyError) as e:
            print(f"[WARN] MCP results file parse failed: {e}", flush=True)

    # Fallback: FAIL_NO_SUBMIT
    return {
        "instance_id": instance["id"],
        "family": instance["family"],
        "difficulty": instance["difficulty"],
        "accuracy": None,
        "hypothesis_source": None,
        "hypothesis_status": "FAIL_NO_SUBMIT",
        "n_interventions": state.intervene_count,
        "n_turns": None,
        "fallback": True,
        "capture_path": None,
        "wall_s": wall_s,
        "model": actual_model,
        "effort": None,
        "arm": "A",
        "version": "v3",
    }


def run_one_instance(instance: dict, actual_model: str) -> dict:
    MCP_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    state = AgentState()
    t0 = time.monotonic()

    progress_path = Path(f"arm_a/v3_progress_{instance['id']}.jsonl")
    progress_file = progress_path.open("w")

    watchdog_thread = threading.Thread(
        target=_watchdog, args=(state, t0, progress_file), daemon=True
    )
    watchdog_thread.start()

    try:
        for attempt in range(1 + RETRY_ON_TIMEOUT):
            t_start = time.time()
            try:
                result = asyncio.run(
                    _run_instance_async(instance, actual_model, state, t0, progress_file)
                )
                t_end = time.time()
                wall_s = result.get("wall_s", time.monotonic() - t0)
                print(
                    f"[ATTEMPT instance={instance['id']} attempt={attempt+1}/{1+RETRY_ON_TIMEOUT} "
                    f"wall_s={wall_s:.1f} "
                    f"start={time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime(t_start))} "
                    f"end={time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime(t_end))}]",
                    flush=True,
                )
                return result
            except StuckReasoningError as e:
                print(
                    f"[STUCK_REASONING instance={instance['id']} attempt={attempt+1} "
                    f"wall_s={e.wall_s:.1f} since_last_tool_s={e.since_last_tool_s:.1f} "
                    f"tool_use_count={e.tool_use_count}]",
                    flush=True,
                )
                return {
                    "instance_id": instance["id"],
                    "family": instance["family"],
                    "difficulty": instance["difficulty"],
                    "accuracy": None,
                    "hypothesis_source": None,
                    "hypothesis_status": "FAIL_STUCK_REASONING",
                    "n_interventions": state.intervene_count,
                    "n_turns": None,
                    "fallback": True,
                    "capture_path": None,
                    "wall_s": e.wall_s,
                    "stuck_since_last_tool_s": e.since_last_tool_s,
                    "stuck_tool_use_count": e.tool_use_count,
                    "model": actual_model,
                    "effort": None,
                    "arm": "A",
                    "version": "v3",
                }
            except asyncio.TimeoutError:
                print(
                    f"[TIMEOUT instance={instance['id']} attempt={attempt+1}/{1+RETRY_ON_TIMEOUT} "
                    f"wall_s={INSTANCE_HARD_CAP}]",
                    flush=True,
                )
                if attempt >= RETRY_ON_TIMEOUT:
                    return {
                        "instance_id": instance["id"],
                        "family": instance["family"],
                        "difficulty": instance["difficulty"],
                        "accuracy": None,
                        "hypothesis_source": None,
                        "hypothesis_status": "FAIL_TIMEOUT_RETRY",
                        "n_interventions": state.intervene_count,
                        "n_turns": None,
                        "fallback": True,
                        "capture_path": None,
                        "wall_s": INSTANCE_HARD_CAP * (attempt + 1),
                        "model": actual_model,
                        "effort": None,
                        "arm": "A",
                        "version": "v3",
                    }
                continue
    finally:
        state.abort_event.set()
        progress_file.close()

    # Should not reach here
    return {
        "instance_id": instance["id"],
        "family": instance["family"],
        "difficulty": instance["difficulty"],
        "accuracy": None,
        "hypothesis_source": None,
        "hypothesis_status": "FAIL_EXCEPTION",
        "n_interventions": state.intervene_count,
        "n_turns": None,
        "fallback": True,
        "capture_path": None,
        "wall_s": time.monotonic() - t0,
        "model": actual_model,
        "effort": None,
        "arm": "A",
        "version": "v3",
    }


def probe_llama_server() -> str:
    """Probe llama-server /v1/models and return actual model name. Fail loudly if not reachable."""
    import urllib.request, urllib.error
    try:
        with urllib.request.urlopen(f"{LLAMA_BASE_URL}/models", timeout=10) as resp:
            data = json.loads(resp.read())
        models = data.get("data", [])
        if not models:
            raise RuntimeError("llama-server /v1/models returned empty model list")
        model_id = models[0]["id"]
        print(f"[llama-server] model confirmed: {model_id}", flush=True)
        return model_id
    except urllib.error.URLError as e:
        raise RuntimeError(
            f"llama-server not reachable at {LLAMA_BASE_URL}: {e}\n"
            "Start it with: llama-server -m /home/ak/Qwen3.6-35B-A3B-MXFP4_MOE.gguf "
            "--host 127.0.0.1 --port 8080 --ctx-size 32768 --n-gpu-layers 0 "
            "--jinja --chat-template-file /home/ak/llama.cpp/models/templates/Qwen-Qwen3-0.6B.jinja"
        ) from e


def main():
    parser = argparse.ArgumentParser(description="Arm A v3 runner (Path B: OpenAI SDK + MCP client)")
    parser.add_argument("--only", nargs="+", metavar="INSTANCE_ID")
    parser.add_argument("--output", default=None)
    # Internal override for forced-stuck test (REQ check 5)
    parser.add_argument("--stuck-cap-override", type=float, default=None,
                        help="Override STUCK_REASONING_CAP_S for testing (e.g. 5)")
    args = parser.parse_args()

    if args.stuck_cap_override is not None:
        global STUCK_REASONING_CAP_S
        STUCK_REASONING_CAP_S = args.stuck_cap_override
        print(f"[WARN] STUCK_REASONING_CAP_S overridden to {STUCK_REASONING_CAP_S}s", flush=True)

    actual_model = probe_llama_server()

    skill_path = SKILL if SKILL.exists() else SKILL_FALLBACK
    if not skill_path.exists():
        raise RuntimeError(f"jump_v3 skill not found at {SKILL} or {SKILL_FALLBACK}")

    from worlds.gen import generate_all
    all_instances = generate_all()

    if args.only:
        only_set = set(args.only)
        instances = [i for i in all_instances if i["id"] in only_set]
        missing = only_set - {i["id"] for i in instances}
        if missing:
            raise ValueError(f"Unknown instance IDs: {sorted(missing)}")
    else:
        instances = all_instances

    output_path = Path(args.output) if args.output else Path("arm_a/results_v3.json")

    print(
        f"Arm A v3 (Path B: OpenAI+MCP) — {len(instances)} instances, "
        f"model={actual_model}, output={output_path}",
        flush=True,
    )

    results = []
    family_fail_counts = {"cellular_automata": 0, "particle_system": 0, "pattern_puzzle": 0}
    skipped_families = set()
    ESCALATION_THRESHOLD_PER_FAMILY = 3

    for inst in instances:
        if inst["family"] in skipped_families:
            continue
        print(f"  {inst['id']} ({inst['family']}, {inst['difficulty']})...", end="", flush=True)
        try:
            r = run_one_instance(inst, actual_model)
        except Exception as e:
            r = {
                "instance_id": inst["id"],
                "family": inst["family"],
                "difficulty": inst["difficulty"],
                "accuracy": None,
                "hypothesis_source": None,
                "hypothesis_status": "FAIL_EXCEPTION",
                "n_interventions": 0,
                "n_turns": None,
                "fallback": True,
                "capture_path": None,
                "wall_s": 0,
                "model": actual_model,
                "effort": None,
                "arm": "A",
                "version": "v3",
                "error": str(e),
            }

        results.append(r)

        status = r.get("hypothesis_status", "submitted")
        acc_str = f"{r['accuracy']:.3f}" if r["accuracy"] is not None else "None"
        print(
            f" acc={acc_str} n_int={r['n_interventions']} [{status}] "
            f"wall_s={r.get('wall_s', 0):.0f} capture={r.get('capture_path', 'none')}",
            flush=True,
        )

        if status in ("FAIL_TIMEOUT", "FAIL_TIMEOUT_RETRY"):
            print(f"[FAIL_TIMEOUT instance={r['instance_id']}]", flush=True)
            fam = r.get("family", "")
            if fam in family_fail_counts:
                family_fail_counts[fam] += 1
                if family_fail_counts[fam] >= ESCALATION_THRESHOLD_PER_FAMILY:
                    skipped_families.add(fam)
                    print(f"[FAMILY_ESCALATION family={fam}]", flush=True)
                    if fam == "pattern_puzzle":
                        print(f"[GLOBAL_STOP] pattern_puzzle escalation — stopping.", flush=True)
                        break
        elif status == "FAIL_NO_SUBMIT":
            print(f"[FAIL_NO_SUBMIT instance={r['instance_id']}]", flush=True)
        elif status == "FAIL_EXCEPTION":
            print(f"[FAIL_EXCEPTION instance={r['instance_id']} err={r.get('error', '')}]", flush=True)
        elif status == "FAIL_STUCK_REASONING":
            print(
                f"[FAIL_STUCK_REASONING instance={r['instance_id']} "
                f"wall_s={r.get('wall_s', 0):.0f} "
                f"since_last_tool_s={r.get('stuck_since_last_tool_s', 0):.0f}]",
                flush=True,
            )
        else:
            print(f"[OK instance={r['instance_id']} acc={acc_str} capture={r.get('capture_path')}]", flush=True)

    output_path.write_text(json.dumps(results, indent=2))
    print(f"\nResults written to {output_path} ({len(results)} entries)", flush=True)

    scored = [r for r in results if r.get("accuracy") is not None]
    n_excluded = len(results) - len(scored)
    mean_acc = statistics.mean(r["accuracy"] for r in scored) if scored else float("nan")
    print(
        f"\n=== Arm A v3 mean accuracy: {mean_acc:.3f} over {len(scored)}/{len(results)} "
        f"(excluded {n_excluded} FAIL_*) ===",
        flush=True,
    )

    statuses = [r.get("hypothesis_status", "submitted") for r in results]
    print(
        f"FAIL counts: timeout={statuses.count('FAIL_TIMEOUT') + statuses.count('FAIL_TIMEOUT_RETRY')} "
        f"no_submit={statuses.count('FAIL_NO_SUBMIT')} "
        f"exception={statuses.count('FAIL_EXCEPTION')} "
        f"stuck_reasoning={statuses.count('FAIL_STUCK_REASONING')}",
        flush=True,
    )


if __name__ == "__main__":
    main()
