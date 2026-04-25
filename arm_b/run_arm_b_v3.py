#!/usr/bin/env python3
"""
Arm B v3 harness: per-instance single claude -p call with MCP tools.
No MAX_TURNS — claude's native tool loop handles multi-turn reasoning.
"""
import argparse, json, os, signal, statistics, subprocess, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

CLAUDE_CMD = [
    "env", "-u", "CLAUDECODE", "-u", "ANTHROPIC_API_KEY",
    "claude", "-p",
    "--dangerously-skip-permissions",
    "--effort", "medium",
    "--output-format", "stream-json", "--include-partial-messages", "--verbose",
    "--model", "claude-sonnet-4-6",
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


def load_skill_body() -> str:
    skill_path = SKILL if SKILL.exists() else SKILL_FALLBACK
    text = skill_path.read_text()
    lines = text.splitlines()
    if lines and lines[0].strip() == "---":
        end = next((i for i, l in enumerate(lines[1:], 1) if l.strip() == "---"), None)
        if end:
            return "\n".join(lines[end + 1:]).strip()
    return text


def build_prompt(skill_body: str) -> str:
    return skill_body + "\n\nBegin by calling get_train_obs."


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


def run_subprocess(cmd, prompt, timeout, env, cwd):
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
    try:
        stdout, stderr = proc.communicate(input=prompt, timeout=timeout)
        wall_s = time.monotonic() - t0
        return stdout, stderr, proc.returncode, wall_s
    except subprocess.TimeoutExpired:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            proc.kill()
        try:
            proc.communicate(timeout=10)
        except subprocess.TimeoutExpired:
            pass
        wall_s = time.monotonic() - t0
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


def run_one_instance(instance: dict, model: str = "claude-sonnet-4-6") -> dict:
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
    if model != "claude-sonnet-4-6":
        try:
            idx = cmd.index("claude-sonnet-4-6")
            cmd[idx] = model
        except ValueError:
            pass

    for attempt in range(1 + RETRY_ON_TIMEOUT):
        t_start = time.time()
        try:
            stdout, stderr, rc, wall_s = run_subprocess(
                cmd, prompt,
                timeout=INSTANCE_HARD_CAP,
                env=env,
                cwd="/home/ak/tmux-agents/projects/jump/repo",
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
                    "effort": "medium",
                    "arm": "B",
                    "version": "v3",
                }
            continue

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
                "effort": "medium",
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
                "effort": "medium",
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
        "effort": "medium",
        "arm": "B",
        "version": "v3",
    }


def main():
    parser = argparse.ArgumentParser(description="Arm B v3 runner")
    parser.add_argument("--only", nargs="+", metavar="INSTANCE_ID")
    parser.add_argument("--model", default="claude-sonnet-4-6")
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
            r = run_one_instance(inst, model=args.model)
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
                "effort": "medium",
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
        f"exception={statuses.count('FAIL_EXCEPTION')}",
        flush=True,
    )

    if skipped_families:
        print(f"ESCALATION: skipped_families={skipped_families}", flush=True)


if __name__ == "__main__":
    main()
