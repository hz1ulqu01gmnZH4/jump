#!/usr/bin/env python3
"""
Arm B v3 harness: per-instance single claude -p call with MCP tools.
No MAX_TURNS — claude's native tool loop handles multi-turn reasoning.
"""
import argparse, json, os, signal, statistics, subprocess, sys, threading, time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

CLAUDE_CMD = [
    "env", "-u", "CLAUDECODE", "-u", "ANTHROPIC_API_KEY",
    "claude", "-p",
    "--dangerously-skip-permissions",
    "--output-format", "stream-json", "--include-partial-messages", "--verbose",
    "--model", "claude-opus-4-7",
    "--mcp-config", "arm_b/mcp_config_instance.json",
    "--allowed-tools",
        "mcp__jump-world__get_train_obs,"
        "mcp__jump-world__intervene,"
        "mcp__jump-world__submit_hypothesis",
]

SKILL = Path.home() / ".claude" / "skills" / "jump_v3.md"
SKILL_FALLBACK = Path(__file__).parent / "jump_v3_skill.md"
MCP_RESULTS_DIR = Path("arm_b/.mcp_results")
MCP_CONFIG_INSTANCE = Path("arm_b/mcp_config_instance.json")
INSTANCE_HARD_CAP = 7200
RETRY_ON_TIMEOUT = 1
STUCK_REASONING_CAP_S = 300       # seconds since last tool_use before kill
STUCK_WATCHDOG_POLL_S = 5         # watchdog loop cadence


class StuckReasoningError(Exception):
    def __init__(self, wall_s, since_last_tool_s, tool_use_count):
        self.wall_s = wall_s
        self.since_last_tool_s = since_last_tool_s
        self.tool_use_count = tool_use_count
        super().__init__(
            f"stuck reasoning: {since_last_tool_s:.1f}s without tool_use "
            f"after {tool_use_count} tool calls (cap={STUCK_REASONING_CAP_S}s)"
        )


def load_skill_body() -> str:
    skill_path = SKILL if SKILL.exists() else SKILL_FALLBACK
    text = skill_path.read_text()
    lines = text.splitlines()
    if lines and lines[0].strip() == "---":
        end = next((i for i, l in enumerate(lines[1:], 1) if l.strip() == "---"), None)
        if end:
            return "\n".join(lines[end + 1:]).strip()
    return text


DEADLINE_NUDGE = """## Deadline policy (harness-enforced)

You operate under a hard turn budget. Endless silent reasoning is killed as
FAIL_STUCK_REASONING with zero credit and no retry — worse than any submission.

- After `get_train_obs`, your FIRST subsequent action MUST be a call to
  `intervene`. Do not finalize, submit, or continue internal analysis before
  making that probe. Even a basic valid action breaks the analysis loop.
- After your first `intervene`, submit if you have a supported hypothesis or
  continue probing. A weak submission with uncertainty noted in a Python
  comment is strictly better than silence.
- Do not loop in private analysis between tool calls. Externalise reasoning
  through tool calls, not internal monologue.
"""


def build_prompt(skill_body: str) -> str:
    return skill_body + "\n\n" + DEADLINE_NUDGE + "\nBegin by calling get_train_obs."


def write_instance_mcp_config(instance_id: str, results_path: Path) -> None:
    """Write a per-instance mcp_config with literal env values (no substitution)."""
    config = {
        "mcpServers": {
            "jump-world": {
                "command": "uv",
                "args": ["run", "python", "arm_b/arm_b_mcp_server.py"],
                "cwd": "/home/ak/tmux-agents/projects/jump/repo",
                "env": {
                    "JUMP_INSTANCE_ID": instance_id,
                    "JUMP_RESULTS_PATH": str(results_path),
                },
            }
        }
    }
    MCP_CONFIG_INSTANCE.write_text(json.dumps(config, indent=2))


def run_subprocess(cmd, prompt, timeout, env, cwd, progress_file=None):
    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd=cwd,
        env=env,
        start_new_session=True,
    )
    t0 = time.monotonic()
    stdout_lines = []
    stderr_lines = []
    last_tool_t = 0.0
    last_tool_count = 0
    state_lock = threading.Lock()
    stuck_flag = threading.Event()

    def _append_progress(record):
        if progress_file is not None:
            progress_file.write(json.dumps(record) + "\n")
            progress_file.flush()

    def _log_event(ev, elapsed):
        nonlocal last_tool_t, last_tool_count
        ts = datetime.now().strftime('%H:%M:%S')
        ev_type = ev.get("type", "")
        if ev_type == "tool_use":
            name = ev.get("name", "")
            input_keys = list(ev.get("input", {}).keys())
            print(f"[{ts}] tool_use: name={name} input_keys={input_keys}", file=sys.stderr, flush=True)
            _append_progress({"t": round(elapsed, 2), "event": "tool_use", "name": name, "input": ev.get("input", {})})
            with state_lock:
                last_tool_t = elapsed
                last_tool_count += 1
        elif ev_type == "tool_result":
            tool_use_id = ev.get("tool_use_id", "")
            content_len = len(str(ev.get("content", "")))
            print(f"[{ts}] tool_result: tool_use_id={tool_use_id} content_len={content_len}", file=sys.stderr, flush=True)
            _append_progress({"t": round(elapsed, 2), "event": "tool_result", "id": tool_use_id, "content_len": content_len})
        elif ev_type in ("text_delta", "content_block_delta"):
            delta = ev.get("delta", {})
            text = delta.get("text", "") if isinstance(delta, dict) else ""
            if text:
                print(f"[{ts}] text_delta: {text[:80]}", file=sys.stderr, flush=True)
                _append_progress({"t": round(elapsed, 2), "event": "text", "text": text[:200]})
        elif ev_type == "message_start":
            print(f"[{ts}] message_start", file=sys.stderr, flush=True)
        elif ev_type in ("message_stop", "message_delta"):
            delta = ev.get("delta", {})
            stop_reason = (delta.get("stop_reason", "") if isinstance(delta, dict) else "") or ev.get("stop_reason", "")
            print(f"[{ts}] message_stop stop_reason={stop_reason}", file=sys.stderr, flush=True)
            _append_progress({"t": round(elapsed, 2), "event": "message_stop", "stop_reason": stop_reason})
        elif ev_type == "assistant":
            for block in ev.get("message", {}).get("content", []):
                btype = block.get("type", "")
                if btype == "tool_use":
                    name = block.get("name", "")
                    input_keys = list(block.get("input", {}).keys())
                    print(f"[{ts}] tool_use: name={name} input_keys={input_keys}", file=sys.stderr, flush=True)
                    _append_progress({"t": round(elapsed, 2), "event": "tool_use", "name": name, "input": block.get("input", {})})
                    with state_lock:
                        last_tool_t = elapsed
                        last_tool_count += 1
                elif btype == "tool_result":
                    tid = block.get("tool_use_id", "")
                    clen = len(str(block.get("content", "")))
                    print(f"[{ts}] tool_result: tool_use_id={tid} content_len={clen}", file=sys.stderr, flush=True)
                    _append_progress({"t": round(elapsed, 2), "event": "tool_result", "id": tid, "content_len": clen})
        else:
            if ev_type:
                print(f"[{ts}] event: {ev_type}", file=sys.stderr, flush=True)

    def _read_stdout():
        for line in proc.stdout:
            stdout_lines.append(line)
            stripped = line.strip()
            if not stripped:
                continue
            try:
                ev = json.loads(stripped)
            except json.JSONDecodeError:
                continue
            _log_event(ev, time.monotonic() - t0)

    def _read_stderr():
        for line in proc.stderr:
            stderr_lines.append(line)

    def _watchdog():
        POLL_S = 5.0
        while not stuck_flag.is_set():
            time.sleep(POLL_S)
            if proc.poll() is not None:
                return
            with state_lock:
                since = (time.monotonic() - t0) - last_tool_t
                count = last_tool_count
            if since > STUCK_REASONING_CAP_S and count >= 1:
                stuck_flag.set()
                elapsed = time.monotonic() - t0
                _append_progress({
                    "t": round(elapsed, 2),
                    "event": "stuck_reasoning_kill",
                    "since_last_tool_s": round(since, 2),
                    "tool_use_count": count,
                    "cap_s": STUCK_REASONING_CAP_S,
                })
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    proc.kill()
                return

    try:
        proc.stdin.write(prompt)
        proc.stdin.close()
    except BrokenPipeError:
        pass

    stdout_thread = threading.Thread(target=_read_stdout, daemon=True)
    stderr_thread = threading.Thread(target=_read_stderr, daemon=True)
    stdout_thread.start()
    stderr_thread.start()
    watchdog_thread = threading.Thread(target=_watchdog, daemon=True)
    watchdog_thread.start()

    try:
        stdout_thread.join(timeout=timeout)
        if stuck_flag.is_set():
            stderr_thread.join(timeout=5)
            wall_s = time.monotonic() - t0
            with state_lock:
                since = wall_s - last_tool_t
                count = last_tool_count
            raise StuckReasoningError(wall_s, since, count)
        if stdout_thread.is_alive():
            elapsed = time.monotonic() - t0
            _append_progress({"t": round(elapsed, 2), "event": "timeout_sigkill"})
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                proc.kill()
            stdout_thread.join(timeout=5)
            stderr_thread.join(timeout=5)
            wall_s = time.monotonic() - t0
            raise subprocess.TimeoutExpired(cmd, timeout)
        stderr_thread.join(timeout=30)
        proc.wait()
        wall_s = time.monotonic() - t0
        return "".join(stdout_lines), "".join(stderr_lines), proc.returncode, wall_s
    except subprocess.TimeoutExpired:
        raise


def parse_tool_uses(stdout: str, name: str) -> list[dict]:
    """Extract tool_use blocks from stream-json output matching the given name."""
    results = []
    for line in stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            ev = json.loads(line)
        except json.JSONDecodeError:
            continue
        if ev.get("type") == "assistant":
            for block in ev.get("message", {}).get("content", []):
                if block.get("type") == "tool_use" and block.get("name") == name:
                    results.append(block)
    return results


def rescore_from_hypothesis(instance: dict, hypothesis_source: str) -> float:
    from harness import WorldHarness
    h = WorldHarness(instance)
    result = h.submit_hypothesis(hypothesis_source)
    return result["accuracy"]


def count_interventions(stdout: str) -> int:
    """Count intervene tool_use calls in stream-json output."""
    return len(parse_tool_uses(stdout, "mcp__jump-world__intervene"))


def run_one_instance(instance: dict, model: str = "claude-opus-4-7", effort: str = "medium") -> dict:
    MCP_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results_path = MCP_RESULTS_DIR / f"{instance['id']}.json"
    if results_path.exists():
        results_path.unlink()

    write_instance_mcp_config(instance["id"], results_path)

    env = os.environ.copy()
    env.pop("CLAUDECODE", None)
    env.pop("ANTHROPIC_API_KEY", None)

    prompt = build_prompt(load_skill_body())

    cmd = list(CLAUDE_CMD)
    # Replace model if overridden
    if model != "claude-opus-4-7":
        try:
            idx = cmd.index("claude-opus-4-7")
            cmd[idx] = model
        except ValueError:
            pass
    # inject effort at runtime
    cmd = cmd + ["--effort", effort]

    progress_path = Path(f"arm_b/v3_progress_{instance['id']}.jsonl")
    progress_file = progress_path.open("w")
    try:
        for attempt in range(1 + RETRY_ON_TIMEOUT):
            t_start = time.time()
            try:
                stdout, stderr, rc, wall_s = run_subprocess(
                    cmd, prompt,
                    timeout=INSTANCE_HARD_CAP,
                    env=env,
                    cwd="/home/ak/tmux-agents/projects/jump/repo",
                    progress_file=progress_file,
                )
                t_end = time.time()
                t_start_iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t_start))
                t_end_iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t_end))
                print(
                    f"[ATTEMPT instance={instance['id']} attempt={attempt+1}/{1+RETRY_ON_TIMEOUT} "
                    f"rc={rc} wall_s={wall_s:.1f} start={t_start_iso} end={t_end_iso}]",
                    flush=True,
                )
                break
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
                    "n_interventions": 0,
                    "n_turns": None,
                    "fallback": True,
                    "capture_path": None,
                    "wall_s": e.wall_s,
                    "stuck_since_last_tool_s": e.since_last_tool_s,
                    "stuck_tool_use_count": e.tool_use_count,
                    "model": model,
                    "effort": effort,
                    "arm": "B",
                    "version": "v3",
                }
            except subprocess.TimeoutExpired:
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
                        "n_interventions": 0,
                        "n_turns": None,
                        "fallback": True,
                        "capture_path": None,
                        "wall_s": INSTANCE_HARD_CAP * (attempt + 1),
                        "model": model,
                        "effort": effort,
                        "arm": "B",
                        "version": "v3",
                    }
                continue
    finally:
        progress_file.close()

    n_interventions = count_interventions(stdout)

    # §4.1 Primary: MCP-side persisted file
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
                "n_interventions": n_interventions,
                "n_turns": None,
                "fallback": False,
                "capture_path": "mcp_file",
                "wall_s": wall_s,
                "model": model,
                "effort": effort,
                "arm": "B",
                "version": "v3",
            }
        except (json.JSONDecodeError, KeyError) as e:
            print(f"[WARN] MCP results file parse failed: {e}", flush=True)

    # §4.2 Fallback: stream-json tool_use parse
    submit_calls = parse_tool_uses(stdout, "mcp__jump-world__submit_hypothesis")
    if submit_calls:
        hyp = submit_calls[-1].get("input", {}).get("hypothesis_source", "")
        try:
            acc = rescore_from_hypothesis(instance, hyp)
            return {
                "instance_id": instance["id"],
                "family": instance["family"],
                "difficulty": instance["difficulty"],
                "accuracy": acc,
                "hypothesis_source": hyp,
                "hypothesis_status": "submitted",
                "n_interventions": n_interventions,
                "n_turns": None,
                "fallback": True,
                "capture_path": "stream_parse_fallback",
                "wall_s": wall_s,
                "model": model,
                "effort": effort,
                "arm": "B",
                "version": "v3",
            }
        except Exception as e:
            print(f"[WARN] Fallback rescore failed: {e}", flush=True)

    # §4.3 FAIL_NO_SUBMIT
    return {
        "instance_id": instance["id"],
        "family": instance["family"],
        "difficulty": instance["difficulty"],
        "accuracy": None,
        "hypothesis_source": None,
        "hypothesis_status": "FAIL_NO_SUBMIT",
        "n_interventions": n_interventions,
        "n_turns": None,
        "fallback": True,
        "capture_path": None,
        "wall_s": wall_s,
        "model": model,
        "effort": effort,
        "arm": "B",
        "version": "v3",
    }


def main():
    parser = argparse.ArgumentParser(description="Arm B v3 runner")
    parser.add_argument("--only", nargs="+", metavar="INSTANCE_ID")
    parser.add_argument("--model", default="claude-opus-4-7")
    parser.add_argument("--effort", default="medium",
                        choices=["low", "medium", "high", "xhigh", "max"])
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    from worlds.gen import generate_all

    skill_path = SKILL if SKILL.exists() else SKILL_FALLBACK
    if not skill_path.exists():
        raise RuntimeError(f"jump_v3 skill not found at {SKILL} or {SKILL_FALLBACK}")

    all_instances = generate_all()

    if args.only:
        only_set = set(args.only)
        instances = [i for i in all_instances if i["id"] in only_set]
        missing = only_set - {i["id"] for i in instances}
        if missing:
            raise ValueError(f"Unknown instance IDs: {sorted(missing)}")
        output_path = Path(args.output) if args.output else Path("arm_b/results_v3.json")
    else:
        instances = all_instances
        output_path = Path(args.output) if args.output else Path("arm_b/results_v3.json")

    print(
        f"Arm B v3 (MCP) — {len(instances)} instances, "
        f"model={args.model}, output={output_path}",
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
            r = run_one_instance(inst, model=args.model, effort=args.effort)
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
                "model": args.model,
                "effort": args.effort,
                "arm": "B",
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
        f"\n=== Arm B v3 mean accuracy: {mean_acc:.3f} over {len(scored)}/{len(results)} "
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

    if skipped_families:
        print(f"ESCALATION: skipped_families={skipped_families}", flush=True)


if __name__ == "__main__":
    main()
