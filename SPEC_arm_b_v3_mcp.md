# Arm B v3 — MCP-based Redesign

**Status:** Design only. No implementation.
**Replaces:** `arm_b/run_arm_b_v2.py` (kept untouched for reproducibility).
**Motivation:** v2's fake-JSON tool protocol forced Sonnet/Opus to burn their full 64k/128k thinking budgets on protocol-bridging meta-reasoning. Real MCP tool registration eliminates that friction.

---

## 0. Summary of Architectural Decisions

| # | Decision | Choice | One-line rationale |
|---|----------|--------|--------------------|
| 1 | Server granularity | **A — one server process per instance** | stdio lifecycle is tied to the claude subprocess; per-instance isolation is free. |
| 2 | Transport | **stdio** | No separate server daemon to supervise; subprocess lifetime matches claude's. |
| 3 | Tool allowlist | **A — ONLY the three `mcp__jump-world__*` tools** | Arm B measures abduction on unseen worlds; Bash/Read could retrieve the hidden rule from `worlds/gen.py`, invalidating the measurement. |
| 4 | Session lifecycle | **B-prime — per-instance single `claude -p` call (no `--resume`)** | Claude handles multi-turn reasoning natively when given real tools; `--resume` is documented as hanging in subprocess contexts. |
| 5 | Result capture | **C — dual: stream-json parse + MCP-side results file** | Parsing fails → fall back to server-written JSON; server-side truth is authoritative. |

---

## 1. MCP Server Architecture (`arm_b/arm_b_mcp_server.py`)

### 1.1 Granularity: one server per instance (Option A)

Each instance run = one `claude -p` subprocess = one stdio MCP server subprocess (spawned by claude from `mcp_config.json`). When claude exits, the server exits. No cross-instance state leakage. No registry to maintain.

**Why not Option B (long-running server with `instance_id`)?**
stdio transport already ties server lifetime to claude's. Adding an `instance_id` parameter would require an HTTP/SSE daemon, a separate supervisor, registry cleanup, and failure recovery. Option A gets isolation for free. The extra startup cost (~hundreds of ms for a Python process) is negligible compared to a multi-minute instance run.

### 1.2 State held by the server

```python
# module-level state, single instance
_STATE = {
    "instance": None,        # dict: the world instance loaded at startup
    "harness": None,         # WorldHarness instance (holds interaction_log, submitted flag)
    "results_path": None,    # Path to write {accuracy, hypothesis_source, log} on submit
    "start_time": None,
}
```

The server is initialised at startup from two environment variables:

- `JUMP_INSTANCE_ID` — which instance to load (e.g. `world_ca_001`)
- `JUMP_RESULTS_PATH` — where to persist the final submit_hypothesis result

At startup the server calls `worlds.gen.generate_all()`, finds the instance by id, constructs a `WorldHarness`, and registers three MCP tools. No shared state; no cross-call serialization needed.

### 1.3 Tool schemas

All three tools share one pattern: they dispatch to the existing `WorldHarness` and return pure JSON. No `instance_id` parameter (Option A).

#### 1.3.1 `get_train_obs`

```json
{
  "name": "get_train_obs",
  "description": "Return the instance's training observations as a list of {state, next_state} pairs.",
  "inputSchema": {
    "type": "object",
    "properties": {},
    "additionalProperties": false
  }
}
```

Returns (as MCP text content; claude receives it as a JSON string):

```json
{
  "observations": [
    {"state": <state-shape>, "next_state": <state-shape>},
    ...
  ],
  "intervention_api": [
    {"action": "...", "params": [...], "description": "..."},
    ...
  ],
  "primitive_glossary": {"...": "..."},
  "family": "cellular_automata | particle_system | pattern_puzzle",
  "difficulty": "easy | medium | hard"
}
```

Note: `intervention_api`, `primitive_glossary`, `family`, `difficulty` are returned here (not in the system prompt) so the tool schema stays family-agnostic and claude's context starts empty beyond the skill preamble. This also prevents having to encode schema variance across families into the tool itself.

#### 1.3.2 `intervene`

Because the action vocabulary varies per family (see `harness._apply_action`), the tool input accepts a discriminated union by `action`. Using oneOf keeps the contract explicit to claude without baking one schema per family into three tools.

```json
{
  "name": "intervene",
  "description": "Apply an action to a state and return the world's one-step transition. 'action' must be one of the actions listed in the intervention_api from get_train_obs.",
  "inputSchema": {
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
      "mod":      {"type": "integer"}
    },
    "required": ["action", "state"],
    "additionalProperties": false
  }
}
```

Returns:

```json
{
  "modified_state": <state-shape>,
  "next_state":    <state-shape>
}
```

Errors (e.g. unknown action, missing state) are returned as MCP `isError: true` responses with a short error message — no silent fallback. `harness.intervene()` already raises `ValueError` on unknown actions; the server surfaces that.

#### 1.3.3 `submit_hypothesis`

```json
{
  "name": "submit_hypothesis",
  "description": "Submit the final hypothesis as Python source defining hidden_rule_fn(state) -> next_state. Scored against held-out test observations. Call exactly once per instance.",
  "inputSchema": {
    "type": "object",
    "properties": {
      "hypothesis_source": {"type": "string"}
    },
    "required": ["hypothesis_source"],
    "additionalProperties": false
  }
}
```

Returns:

```json
{
  "accuracy": 0.833,
  "test_obs_count": 6,
  "correct": 5
}
```

**Side effect (critical for result capture):** when `submit_hypothesis` succeeds, the server writes a JSON payload to `$JUMP_RESULTS_PATH`:

```json
{
  "instance_id": "world_ca_001",
  "family": "cellular_automata",
  "difficulty": "hard",
  "accuracy": 0.833,
  "hypothesis_source": "def hidden_rule_fn(state): ...",
  "interaction_log": [...],
  "submitted": true,
  "t_end": "2026-04-25T..."
}
```

If claude exits without submitting, the file does not exist. The harness checks for its absence to classify `FAIL_NO_SUBMIT`.

### 1.4 Sandboxing `submit_hypothesis`

The server runs `exec(hypothesis_source, ns)` exactly as `harness.submit_hypothesis` does today. This is a **known security posture inherited from v2** — the benchmark assumes trusted model output. v3 does not change it, but documents it (see §7 Risk 5). If later hardening is desired, wrap `exec` in a subprocess with a CPU/time rlimit. Out of scope for v3.

### 1.5 Server file skeleton (pseudocode, not implementation)

```text
# arm_b/arm_b_mcp_server.py  (pseudocode)
#
# from mcp.server.stdio import stdio_server
# from mcp.server import Server
# server = Server("jump-world")
#
# @server.list_tools()        -> returns the three schemas above
# @server.call_tool()         -> dispatches on name, calls _harness.get_train_obs() / intervene(...) / submit_hypothesis(...)
#
# On submit_hypothesis success: write_text(JUMP_RESULTS_PATH, json.dumps(payload))
#
# main() reads JUMP_INSTANCE_ID and JUMP_RESULTS_PATH from os.environ,
#        builds harness, runs stdio_server(server) loop.
```

Dependencies: `mcp` Python SDK (`mcp>=1.0`). Added to `arm_b/requirements.txt`.

---

## 2. Claude Invocation Design (`run_arm_b_v3.py`)

### 2.1 MCP config file — `arm_b/mcp_config.json`

```json
{
  "mcpServers": {
    "jump-world": {
      "command": "uv",
      "args": ["run", "python", "arm_b/arm_b_mcp_server.py"],
      "cwd": "/home/ak/tmux-agents/projects/jump/repo",
      "env": {
        "JUMP_INSTANCE_ID": "${JUMP_INSTANCE_ID}",
        "JUMP_RESULTS_PATH": "${JUMP_RESULTS_PATH}"
      }
    }
  }
}
```

The harness sets `JUMP_INSTANCE_ID` and `JUMP_RESULTS_PATH` in the claude subprocess's env before spawning; claude forwards them into the MCP server subprocess via this config.

### 2.2 CLAUDE_CMD (full v3 command line)

```python
CLAUDE_CMD = [
    "env", "-u", "CLAUDECODE", "-u", "ANTHROPIC_API_KEY",
    "claude", "-p",
    "--dangerously-skip-permissions",
    "--effort", "medium",
    "--output-format", "stream-json",
    "--include-partial-messages",
    "--verbose",
    "--model", "claude-sonnet-4-6",   # parameterised; opus-4-7 for Opus runs
    "--mcp-config", "arm_b/mcp_config.json",
    "--allowed-tools",
        "mcp__jump-world__get_train_obs,"
        "mcp__jump-world__intervene,"
        "mcp__jump-world__submit_hypothesis",
    # Note: NO --disallowed-tools; --allowed-tools is an exclusive allowlist in claude -p.
]
```

### 2.3 Tool allowlist rationale (Option A — only the three jump tools)

**Decision:** `--allowed-tools mcp__jump-world__get_train_obs,mcp__jump-world__intervene,mcp__jump-world__submit_hypothesis`

The guiding principle is **experimental validity**, not engineering convenience:

- Arm B measures *abduction on unseen worlds*. The hidden rule is defined in `worlds/gen.py`. If claude can call `Read` on that file or `Bash` with `cat` / `grep`, it can retrieve the rule verbatim — turning an abduction experiment into a retrieval experiment. That is exactly what the control condition `C-retrieval=1.000` already isolates; adding retrieval leakage to Arm B would contaminate the headline number.
- Allowing `Read` "just for self-introspection of the skill file" sounds benign but is unnecessary: the skill is delivered in the system prompt. If claude is confused about the protocol, the fix is to improve the prompt, not to hand it a side-channel.
- Option C (unrestricted) is ruled out for the same reason.

**Cost of strict allowlist:** claude cannot write scratchpad code to test hypotheses privately. That is acceptable — testing hypotheses is exactly what `intervene` is for.

### 2.4 `--effort medium`

Kept from v2-P9 research finding. The thinking-exhaustion failure mode is addressed by removing the protocol meta-reasoning tax, not by raising the budget further; `medium` is sufficient once the friction is gone and keeps cost bounded.

### 2.5 `--output-format stream-json --include-partial-messages --verbose`

Retained from v2 so the harness can parse `tool_use` blocks (§4) and the existing pretty-printer in `scripts/dispatch.sh` continues to render output for debugging.

### 2.6 Per-instance environment

```python
os.environ["JUMP_INSTANCE_ID"]  = instance["id"]
os.environ["JUMP_RESULTS_PATH"] = str(tmp_results_path)
```

`tmp_results_path` is a per-instance temp file (e.g. `arm_b/.mcp_results/<instance_id>.json`). The harness reads it after claude exits (§4).

---

## 3. Skill Prompt — `arm_b/jump_v3_skill.md`

Full content (dropped into the skill store at `~/.claude/skills/jump_v3.md`, or passed inline via a single-shot prompt; preferred delivery is `~/.claude/skills/jump_v3.md` symlinked from `arm_b/jump_v3_skill.md` for in-repo ownership).

```markdown
---
name: jump_v3
description: v3 abductive-jump skill — discover the hidden rule of an unknown world using three MCP tools.
version: "3.0"
---

# jump_v3 — World Abduction

You are probing an unknown world to discover its hidden transition rule. The world uses invented vocabulary. Do NOT import external domain knowledge; discover the rule from observations and interventions alone.

## Tools available to you

Three MCP tools, namespace `mcp__jump-world__*`, are provided. Call them as normal tool calls — no JSON-in-text protocol.

- **get_train_obs** — returns training observations (state → next_state pairs) plus the `intervention_api`, `primitive_glossary`, `family`, and `difficulty` for this world. Call this first.
- **intervene** — apply an action from the `intervention_api` to a state; returns `{modified_state, next_state}`. Use this to test causal hypotheses.
- **submit_hypothesis** — submit your final rule as Python source defining `hidden_rule_fn(state) -> next_state`. Call exactly once per instance. Returns accuracy on held-out test observations.

## Workflow

1. Call `get_train_obs`. Inspect state shapes, symbol sets, and transitions across pairs.
2. Form a candidate rule from the patterns.
3. Call `intervene` to probe: modify a state in a way that discriminates between hypotheses, then compare the returned `next_state` to your prediction.
4. Refine the rule. Repeat step 3 until confident.
5. Call `submit_hypothesis` with the complete Python source.

## Hypothesis constraints

- Must define a function named exactly `hidden_rule_fn`.
- Signature: `hidden_rule_fn(state) -> next_state`. The `state` shape matches `train_obs[0]["state"]`.
- Python stdlib only — no `numpy`, no external imports.
- No hardcoded lookup of the training pairs; the rule must generalise to unseen states.

## Abductive principles

- The best rule explains ALL training observations, not just most.
- The best rule predicts intervention results correctly.
- Prefer the simplest rule that fits (Occam).
- Use the world's invented vocabulary structurally, not as labels.

Do not guess. Do not submit until interventions support the rule.
```

Key differences from v2:

- No JSON-line protocol, no format examples — claude already knows how to call MCP tools.
- Removed instructions about outputting tool calls on their own line, with no surrounding text, etc. That friction is what exhausted the thinking budget.
- Metadata (`intervention_api`, `primitive_glossary`, `family`, `difficulty`) is delivered by `get_train_obs`, not the prompt. Keeps the skill prompt world-agnostic; the first tool call grounds claude in the instance.
- Shorter: ~30 lines vs. v2's ~60 lines.

---

## 4. Result Capture Mechanism (Option C — dual redundancy)

The harness reads results in this priority order:

### 4.1 Primary: MCP-side persisted file

After claude exits:

```python
results_path = Path(f"arm_b/.mcp_results/{instance_id}.json")
if results_path.exists():
    payload = json.loads(results_path.read_text())
    # payload has: accuracy, hypothesis_source, interaction_log, submitted=True
    return _build_ok_result(instance, payload)
```

This is authoritative because the server computes accuracy directly by calling the real `harness.submit_hypothesis()`. No parsing risk.

### 4.2 Fallback: stream-json tool_use parse

If the MCP results file is missing (e.g. claude submitted but the server died before flushing — unlikely with stdio since the server writes before responding), parse `stdout` for `tool_use` blocks:

```python
for ev in parse_stream_json(stdout):
    if ev.type == "assistant":
        for block in ev.message.content:
            if block.type == "tool_use" and block.name == "mcp__jump-world__submit_hypothesis":
                hypothesis_source = block.input["hypothesis_source"]
                # Need to run scoring ourselves in this fallback path.
```

In fallback mode the harness re-instantiates a `WorldHarness` for the instance and calls `submit_hypothesis(hypothesis_source)` directly to get the accuracy. This duplicates scoring in the harness process, which is safe — `run_instance` is pure.

### 4.3 FAIL_NO_SUBMIT path

If both the MCP file is missing AND no `submit_hypothesis` tool_use block is found in stream-json output → `hypothesis_status = "FAIL_NO_SUBMIT"`, `accuracy = None`, excluded from the mean (same semantics as v2).

### 4.4 Why dual

- The stream-json parse alone could miss the call if output is truncated on timeout SIGKILL.
- The file alone could miss if the server process is SIGKILLed mid-write (non-atomic). Mitigation: write to `tmp_path`, then `os.rename()` to final path for atomicity.
- The fallback also gives the harness a non-silent diagnostic: "server crashed but claude did submit", logged separately.

### 4.5 `results_v3.json` entry schema

Per-instance, compatible with v2's shape (for the merge/review step):

```json
{
  "instance_id": "world_ca_001",
  "family": "cellular_automata",
  "difficulty": "hard",
  "accuracy": 0.833,
  "hypothesis_source": "def hidden_rule_fn(state): ...",
  "hypothesis_status": "submitted",
  "n_interventions": 4,
  "n_turns": null,
  "fallback": false,
  "capture_path": "mcp_file",
  "wall_s": 412.7,
  "model": "claude-sonnet-4-6",
  "effort": "medium",
  "arm": "B",
  "version": "v3"
}
```

New fields vs v2:

- `capture_path`: `"mcp_file" | "stream_parse_fallback"` — which branch of §4 produced the result.
- `wall_s`, `model`, `effort`, `arm`, `version` — explicit provenance for the merge step.
- `n_turns` is `null` because v3 is not turn-based in the same sense (see §5). Kept in schema for merge compatibility.

Failure entries mirror v2's shape (`accuracy: null`, `hypothesis_status: "FAIL_*"`).

---

## 5. Session Lifecycle — Option B-prime: per-instance single `claude -p` call

### 5.1 What runs

For each instance the harness runs **one** `claude -p` subprocess, with the MCP server subprocess spawned by claude from `mcp_config.json`. Claude performs its entire multi-step abductive loop inside that one call, invoking `get_train_obs`, then any number of `intervene` calls, then `submit_hypothesis`. When `submit_hypothesis` returns, claude's natural turn-taking ends and the subprocess exits.

This differs from v2's "fresh call per turn with reconstructed history prompt". With real tools, claude's own context *is* the history — no prompt reconstruction, no JSON replay. This is the single biggest simplification in v3.

### 5.2 Why not Option A (`--resume`)?

`--resume` is documented in `arm_b/run_arm_b_v2.py:6` as hanging in subprocess contexts. Not re-litigated.

### 5.3 Why not Option B (per-turn fresh claude)?

Per-turn fresh claude requires reconstructing interaction history in the prompt (v2's `build_prompt`). That reconstruction was cheap when the "history" was a few text-JSON lines; with real `tool_use`/`tool_result` blocks, faithfully replaying them across subprocess boundaries is work that gains nothing, because a single `claude -p` call handles multi-turn natively.

It also re-introduces the exact failure mode v3 is designed to eliminate: each fresh turn pays the protocol-orientation cost again.

### 5.4 Why not Option C (phase-gated sessions)?

Adds complexity (three phase prompts, phase-handoff logic) with no clear scientific benefit. Claude's own reasoning already naturally phases: observe → hypothesize → test → refine → submit.

### 5.5 Timeouts

Per-instance wall-clock cap: `INSTANCE_HARD_CAP = 7200` (same as v2). Single timeout for the single subprocess:

```python
proc.communicate(timeout=INSTANCE_HARD_CAP)
```

No per-turn timeouts. If claude is stuck in a pathological loop, the instance cap ends it.

Retries: one retry on timeout, logged as `FAIL_TIMEOUT_RETRY`; no second retry. Per-family escalation (3x FAIL_TIMEOUT → skip family; pattern_puzzle 3x → global stop) is kept from v2.

### 5.6 `run_arm_b_v3.py` pseudocode

```python
# arm_b/run_arm_b_v3.py  (pseudocode — NOT implementation)

CLAUDE_CMD = [...see §2.2...]
MCP_RESULTS_DIR = Path("arm_b/.mcp_results")
SKILL = Path.home() / ".claude" / "skills" / "jump_v3.md"
INSTANCE_HARD_CAP = 7200
RETRY_ON_TIMEOUT = 1

def build_prompt(skill_body: str) -> str:
    """v3 prompt is trivial — tools carry the protocol."""
    return skill_body + "\n\nBegin by calling get_train_obs."

def run_one_instance(instance: dict) -> dict:
    MCP_RESULTS_DIR.mkdir(exist_ok=True)
    results_path = MCP_RESULTS_DIR / f"{instance['id']}.json"
    if results_path.exists(): results_path.unlink()

    env = os.environ.copy()
    env["JUMP_INSTANCE_ID"]  = instance["id"]
    env["JUMP_RESULTS_PATH"] = str(results_path)
    env.pop("CLAUDECODE", None); env.pop("ANTHROPIC_API_KEY", None)

    prompt = build_prompt(load_skill_body(SKILL))

    for attempt in range(1 + RETRY_ON_TIMEOUT):
        t0 = time.monotonic()
        try:
            stdout, stderr, rc, wall_s = run_subprocess(
                CLAUDE_CMD, prompt,
                timeout=INSTANCE_HARD_CAP, env=env,
                cwd="/home/ak/tmux-agents/projects/jump/repo",
            )
            break
        except subprocess.TimeoutExpired:
            continue
    else:
        return _fail_timeout_result(instance)

    # §4.1 primary capture
    if results_path.exists():
        payload = json.loads(results_path.read_text())
        return _ok_result(instance, payload, capture="mcp_file", wall_s=wall_s)

    # §4.2 fallback capture
    tool_uses = parse_tool_uses(stdout, name="mcp__jump-world__submit_hypothesis")
    if tool_uses:
        hyp = tool_uses[-1]["input"]["hypothesis_source"]
        acc = rescore_from_hypothesis(instance, hyp)
        return _ok_result(instance, {"accuracy": acc, "hypothesis_source": hyp},
                          capture="stream_parse_fallback", wall_s=wall_s)

    return _fail_no_submit_result(instance, wall_s)

def main():
    # parse --only, --model, --output; iterate instances; write results_v3.json;
    # per-family escalation + pattern_puzzle global-stop as in v2.
    ...
```

---

## 6. Migration Plan — File Layout

```
arm_b/
  run_arm_b_v2.py            ← UNTOUCHED
  run_arm_b_v3.py            ← NEW harness (§5.6)
  arm_b_mcp_server.py        ← NEW MCP server (§1)
  mcp_config.json            ← NEW MCP config (§2.1)
  jump_v3_skill.md           ← NEW skill source-of-truth (symlinked to ~/.claude/skills/jump_v3.md)
  requirements.txt           ← UPDATED to add mcp>=1.0
  .mcp_results/              ← NEW (gitignored) per-instance result drop
  results_v2.json            ← UNTOUCHED
  results_v3.json            ← NEW
  results_v3_merged.json     ← NEW — v2 ∪ v3 for arm-vs-arm review
  merge_v3.py                ← NEW (trivial — extends merge_v2_p6.py pattern)
```

### v2 vs v3 comparison

| Aspect | v2 | v3 |
|--------|----|----|
| Tool registration | None (prose + JSON-in-text) | MCP (`mcp__jump-world__*`) |
| Thinking-exhaustion risk | High — protocol meta-reasoning | Low — direct tool calls |
| Session model | Fresh `claude -p` per turn, history replay in prompt | **Per-instance single `claude -p` call (§5)** |
| Transport | subprocess pipes (text) | **MCP over stdio (§1.1–1.2)** |
| Result capture | Regex parse of text output | **MCP-side JSON file + stream-json parse fallback (§4)** |
| Tool allowlist | N/A (no real tools) | **Only 3 jump tools (§2.3)** — Bash/Read blocked |
| Metadata delivery | In prompt | In `get_train_obs` response |
| Skill length | ~60 lines | ~30 lines |
| `--effort` | medium | medium (unchanged) |

### Validation strategy (before running full 17 instances)

1. Smoke: `world_ca_001` only, Sonnet, `--effort medium`. Expect non-null accuracy, `capture_path=mcp_file`.
2. Sanity: re-run an instance Arm A solved easily (e.g. a pattern_puzzle easy). v3 should now match/approach Arm A there.
3. Full 17-instance run, then merge with v2 results for arm-vs-arm.

---

## 7. Risks and Open Questions

| # | Risk | Mitigation / resolution |
|---|------|-------------------------|
| 1 | **stdio MCP server subprocess lifetime** — does claude spawn/tear down the server cleanly? What if server state were needed across claude calls? | Resolved by §5.1 per-instance single-call design: server lifetime = claude lifetime = instance lifetime. No cross-call state problem. |
| 2 | **Tool schema vs. skill prompt conflict** | Schema is authoritative; prompt describes semantics only, never JSON shape. §3 skill deliberately omits format examples. If a mismatch appears in testing, fix the schema or the prompt, never both independently. |
| 3 | **`claude -p` + `--mcp-config` + `--effort medium` + `--output-format stream-json` compatibility** — this exact combination may not be validated | Smoke-test as step 1 of §6 validation. If stream-json does not emit `tool_use` blocks for MCP tools (they do for built-in tools in v2 runs), the fallback capture path would regress; the MCP-side file (§4.1) remains authoritative regardless. |
| 4 | **World state serialization between tool calls** | Not an issue: state is held entirely in the server's in-process `WorldHarness`. Each `intervene` call passes `state` as an argument (as v2 does) — it is not persisted by the server between calls. The server only persists `interaction_log` and the submitted-flag. |
| 5 | **`submit_hypothesis` exec security** | Inherited v2 posture: trusted model output, no sandbox. Documented here; out of scope. If later required, run `exec` in a subprocess with `resource.setrlimit(RLIMIT_CPU, ...)`. |
| 6 | **Experimental validity — does the tool schema leak structural information?** | Yes, a little: the schema enumerates candidate parameter names (`row`, `col`, `pattern`, `x`, `y`, ...). This is the *same* information v2 puts in `intervention_api`, which is already provided to claude. Net leakage = zero vs v2. The new thing claude gets is that these are *tools it can actually call*, which is the point of v3. No validity regression. |
| 7 | **Tool-call budget** — claude may call `intervene` dozens of times; each call's `state` payload is grids up to 8×8 | Token cost per call is small (~100–300 tokens). The full budget is per-model context (200k Sonnet / 200k Opus 4.7) — even 50 interventions stay well under 20k tokens of tool traffic. |
| 8 | **Retry-on-timeout and server restart** — on retry the server subprocess is fresh, so prior interventions are lost | Intentional. A retry is a fresh attempt by design; losing speculative state is correct semantics. The `.mcp_results/{id}.json` is deleted at start of each attempt (§5.6). |
| 9 | **Arm A vs Arm B comparability** — v2 used MAX_TURNS=12 per v2 runner for parity with Arm A | v3 has no "turns" concept. For comparability, compare on wall-clock and `n_interventions` instead. If Arm A's turn-limit is a binding budget, cap `n_interventions` in v3 to match (e.g. 12). Decision: **do NOT cap in v3**; document as a v3↔Arm A methodological difference in the review step. v2↔v3 comparison within Arm B is unaffected. |

---

## 8. Open implementation questions (for the implementer task)

These are questions the implementer should verify *before* writing code, not design decisions:

- Exact `mcp` Python SDK version pinned — test that `Server.list_tools`/`call_tool` decorators yield a schema that the Claude Code `claude -p` client accepts.
- The precise `--allowed-tools` spelling for MCP-namespaced tools (`mcp__<server>__<tool>` vs other separators) — confirm in a smoke test.
- Whether `--mcp-config` path is resolved relative to CWD or the process — set CWD deterministically (§2.1 `cwd` key already pins it).
- Whether stream-json emits `tool_use` events for MCP tools identically to built-in tools — determines §4.2 fallback viability.
