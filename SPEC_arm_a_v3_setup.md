# SPEC: Arm A v3 Setup

Status: design (no code changes yet). Implementer task to follow.
Goal: Symmetric harness for Arm A using a local open-weights model that
matches Arm B v3 C₆ feature parity (watchdog 600 s, DEADLINE_NUDGE, submit
mandate, MCP tool calling, output JSON schema).

Inputs surveyed:
- `arm_b/run_arm_b_v3.py` (template)
- `arm_b/jump_v3_skill.md` (shared skill prompt)
- `arm_b/arm_b_mcp_server.py` (stdio MCP server, per-instance process)
- `SPEC_arm_b_v3_c6_prompt.md` (C₆ mandate + cap=600)
- `arm_b/results_v3_c6_smoke.json` (FAIL_STUCK on world_ca_001)
- `arm_b/results_v3_c5_seq_regression.json` (acc=1.0 on world_seq_001 baseline)

---

## §1 — Environment Inventory (verified, not guessed)

Commands run on host (2026-04-26):

| Probe | Result | Notes |
|---|---|---|
| `which opencode` | `/usr/sbin/opencode` | installed |
| `opencode --version` | `1.14.20` | recent |
| `opencode run --help` | supports `--format json`, `--model provider/model`, `--dangerously-skip-permissions`, `--prompt`, stdin message | non-interactive batch OK |
| `opencode mcp --help` | `add`, `list`, `auth`, `logout`, `debug` | MCP first-class |
| `~/.config/opencode/opencode.json` | already configured `llama.cpp` provider via `@ai-sdk/openai-compatible` against `http://127.0.0.1:8080/v1`; one model alias `nemotron3` registered | OpenAI-compat path established |
| `which llama-server` | `/usr/sbin/llama-server` | works (CPU + BLAS + OpenBLAS detected) |
| `which ollama` | `/usr/sbin/ollama` | present but not running |
| Local GGUFs | `/home/ak/Qwen3.6-27B-Q6_K.gguf` (21 GB), `/home/ak/Qwen3.6-35B-A3B-MXFP4_MOE.gguf` (20 GB), `/home/ak/gemma-4-26B-A4B-it-MXFP4_MOE.gguf`, `/home/ak/NVIDIA-Nemotron-3-Super-120B-A12B-MXFP4_MOE-*.gguf` (3-shard) | **No `Qwen3-235B-A22B` GGUF on disk** |
| `df -h /home` | 79 GB free of 1 TB (92 % used) | **Insufficient for Qwen3-235B-A22B at MXFP4** (≈120 GB) without freeing ≥50 GB first |
| `mcp` Python lib | `mcp 1.27.0` in `repo/.venv` | client API available |
| `openai` Python lib | not installed | needed for Path B |

### Model substitution decision

Task names `Qwen3-235B-A22B` but it is **not available locally** and disk is
tight. Two options for the implementer phase:

1. **Free disk + download** `Qwen3-235B-A22B-Instruct-MXFP4_MOE.gguf` (~120 GB)
   from HuggingFace into `/home/ak/`. Required if the experiment plan demands
   that exact model.
2. **Substitute** `Qwen3.6-35B-A3B-MXFP4_MOE.gguf` (already on disk) as the
   stand-in. Same Qwen3.x family, same MoE-MXFP4 quant, fits in RAM, no
   download. Acceptable when the goal is symmetric-harness validation and
   high-N runs rather than capability ceiling. **Recommended for Phase 32e1
   smoke; revisit before the production run.**

Either choice is decoupled from harness design — only the `--model` flag and
llama-server `-m` path differ.

---

## §2 — Path A: opencode-based

### Feasibility verdict — VIABLE but UNDESIRABLE

opencode supports every functional requirement on paper:

- **Batch / non-interactive**: `opencode run --prompt "<text>" --format json --dangerously-skip-permissions --model llama.cpp/qwen3` emits a stream of JSON events to stdout; exit code reflects success.
- **MCP tool calling**: `opencode mcp add` registers the jump-world stdio server; tools appear with native names (no JSON-in-text protocol). Existing `mcpServers` block in `arm_b/mcp_config_instance.json` translatable 1:1.
- **System prompt injection**: opencode supports `--agent <name>` referencing a project-level agent file, where `system` content is the contents of `jump_v3_skill.md` plus the C₆ `DEADLINE_NUDGE`.
- **Local model**: existing `~/.config/opencode/opencode.json` already wires `llama.cpp` provider with OpenAI-compat baseURL `http://127.0.0.1:8080/v1`; add a `qwen3` model entry.

### Setup commands (Path A reference)

```bash
# 1. Start llama-server (Qwen3.6-35B-A3B-MXFP4_MOE substitute)
llama-server -m /home/ak/Qwen3.6-35B-A3B-MXFP4_MOE.gguf \
  --host 127.0.0.1 --port 8080 \
  --ctx-size 65536 --n-gpu-layers 0 \
  --jinja --chat-template-file /home/ak/llama.cpp/models/templates/Qwen3.5-4B.jinja &

# 2. Add model entry to ~/.config/opencode/opencode.json
#    "models": { "qwen3": { "name": "qwen3", "limit": {"context": 65536, "output": 16384}}}

# 3. Register MCP server (opencode reads mcpServers from project config)
opencode mcp add  # interactive; or write project-level opencode.json with mcpServers

# 4. Per-instance: write env-substituted mcp_config_instance.json (same as Arm B)
# 5. Spawn:
opencode run \
  --model llama.cpp/qwen3 \
  --format json \
  --dangerously-skip-permissions \
  --agent jump_v3_arm_a \
  --prompt "$(cat prompt.txt)"
```

### How requirements map onto opencode

| Requirement | opencode path |
|---|---|
| `DEADLINE_NUDGE` injection | concat into agent system prompt at write time |
| Watchdog 600 s | external `subprocess.Popen` + identical thread-based watchdog as Arm B; SIGKILL the `opencode run` process group on stuck-reasoning. opencode is just another subprocess. |
| Stream parsing | `--format json` emits per-event JSON lines; structurally similar to claude stream-json but **schema differs** (event types: `message`, `tool-call`, `tool-result`, `step-finish`). Need a Path-A-specific `parse_tool_uses` adapter. |
| Per-instance MCP env | opencode reads `mcpServers` from project config; need to overwrite the project config per-instance (or use stdio inheritance via env vars). |
| Output JSON schema | wrap opencode result extraction in same `run_one_instance` shell, return identical dict. |

### Compatibility risks

1. **Event-schema drift**: opencode's JSON event types are versioned with the
   binary; an upgrade can break `parse_tool_uses`. We do not control the
   schema and there is no `--schema-version` pin. **Severity: medium**, since
   the parse-shape regression is exactly the C₅→C₆ failure mode we just
   fixed for Arm B.
2. **Tool-call shape with OpenAI-compat providers**: ai-sdk's
   openai-compatible provider relies on the model emitting OpenAI-style
   `tool_calls` JSON. Qwen3.x with `--jinja` chat templates *does* emit
   `<tool_call>` blocks, but reliability across families (CA / PT / SEQ
   prompts) under abductive reasoning is **untested on this stack**. Path B
   has the same risk but with one fewer translation layer.
3. **Project-config global state**: opencode reads project-scoped config from
   a single file; concurrent instances would race. Mitigated by serial
   execution (Arm B is also serial), or by `--dir <per-instance-tmpdir>`.
4. **Hidden retry / fallback policies**: opencode TUI/agent loop has its own
   error handling; inspecting the source of truth requires reading the bun
   binary's behaviour. We prefer a harness where every behaviour is in our
   tree.
5. **No `--effort` analogue**: Qwen3 doesn't expose effort levels via OpenAI
   API; `--variant` is provider-specific and would need plumbing through the
   llama-cpp provider plugin. Drop the `effort` field for Arm A or hardcode
   it to `null` in output JSON.

### Verdict

Functionally feasible. **Rejected** as primary path on grounds (1) and (4):
opaque, version-fragile, and adds an indirection we don't need given that the
underlying APIs (OpenAI-compat HTTP + MCP stdio) are directly callable from
Python with the same `mcp` library Arm B already uses on the server side.

---

## §3 — Path B: Custom (OpenAI SDK + MCP client)

### Architecture — mirror `run_arm_b_v3.py`

Replace the `claude -p` subprocess with an in-process Python agentic loop:

```
┌────── arm_a/run_arm_a_v3.py (one process per instance) ──────────────────┐
│                                                                            │
│  ┌─ MCP stdio client (mcp.client.stdio.stdio_client) ──┐                  │
│  │ spawns: uv run python arm_a/arm_a_mcp_server.py     │                  │
│  │ env: JUMP_INSTANCE_ID, JUMP_RESULTS_PATH            │                  │
│  └──────────────────────────────────────────────────────┘                  │
│                            │ ↑                                              │
│                  list_tools, call_tool                                       │
│                            │ ↑                                              │
│  ┌─ Agentic loop ───────────────────────────────────────┐                  │
│  │ messages = [system, user_with_skill+nudge]           │                  │
│  │ while True:                                          │                  │
│  │   resp = openai.chat.completions.create(             │                  │
│  │     base_url="http://127.0.0.1:8080/v1",             │                  │
│  │     model="qwen3", messages=messages,                │                  │
│  │     tools=mcp_tools_as_openai_schema,                │                  │
│  │     tool_choice="auto", stream=True)                 │                  │
│  │   if tool_calls: dispatch via MCP client; append     │                  │
│  │     tool_result; loop                                │                  │
│  │   else: break                                        │                  │
│  └──────────────────────────────────────────────────────┘                  │
│                            ↑                                                │
│  ┌─ Watchdog thread (mirrors run_arm_b_v3.py) ──────────┐                  │
│  │ poll every 5 s; if (now - last_tool_t) > 600 and     │                  │
│  │ tool_count >= 1 → set abort flag, kill llama HTTP    │                  │
│  │ via cancel token + close MCP session                 │                  │
│  └──────────────────────────────────────────────────────┘                  │
└────────────────────────────────────────────────────────────────────────────┘
```

### llama.cpp server startup

```bash
llama-server \
  -m /home/ak/Qwen3.6-35B-A3B-MXFP4_MOE.gguf \
  --host 127.0.0.1 --port 8080 \
  --ctx-size 65536 \
  --n-gpu-layers 0 \
  --jinja \
  --chat-template-file /home/ak/llama.cpp/models/templates/Qwen3.5-4B.jinja \
  --parallel 1 --cont-batching \
  --metrics
```

Verification: `curl -s http://127.0.0.1:8080/v1/models | jq .` must return a
non-empty list. Implementer should add a startup probe to `run_arm_a_v3.py`
that fails loudly if the server is not reachable (no fallback to a different
backend).

### Key code shapes

#### 3.1 MCP client + tool schema export

```python
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

async def open_mcp_session(instance_id, results_path):
    params = StdioServerParameters(
        command="uv",
        args=["run", "python", "arm_a/arm_a_mcp_server.py"],
        cwd="/home/ak/projects/jump",
        env={
            **os.environ,
            "JUMP_INSTANCE_ID": instance_id,
            "JUMP_RESULTS_PATH": str(results_path),
        },
    )
    async with stdio_client(params) as (r, w):
        async with ClientSession(r, w) as session:
            await session.initialize()
            tools = (await session.list_tools()).tools
            yield session, tools
```

`arm_a/arm_a_mcp_server.py` is a verbatim copy of `arm_b_mcp_server.py` except
for the `Server("jump-world")` name, which **must remain `jump-world`** so
that downstream parsing keys (`mcp__jump-world__intervene`) match Arm B
counters and analysis scripts.

#### 3.2 OpenAI tool-schema translation

```python
def mcp_tools_to_openai(tools):
    return [{
        "type": "function",
        "function": {
            "name": f"mcp__jump-world__{t.name}",  # match Arm B namespacing
            "description": t.description,
            "parameters": t.inputSchema,
        }
    } for t in tools]
```

The `mcp__jump-world__` prefix is artificial here (Arm B inherits it from
claude's MCP namespacing) — we keep it so `parse_tool_uses(stdout, "mcp__jump-world__intervene")`-equivalent counters reuse the same string keys. The
local function-name → MCP-name translation is one line in the dispatcher.

#### 3.3 Agentic loop with watchdog

```python
def run_one_instance(instance, model="qwen3"):
    state = AgentState()  # last_tool_t, tool_count, abort_event
    watchdog = threading.Thread(target=_watchdog_loop, args=(state,), daemon=True)
    watchdog.start()
    t0 = time.monotonic()

    messages = [
        {"role": "system", "content": load_skill_body() + "\n\n" + DEADLINE_NUDGE},
        {"role": "user",   "content": "Begin by calling get_train_obs."},
    ]

    submit_called = False
    while not state.abort_event.is_set() and not submit_called:
        resp = client.chat.completions.create(
            model=model, messages=messages,
            tools=openai_tools, tool_choice="auto",
            timeout=120, max_tokens=4096,
        )
        msg = resp.choices[0].message
        messages.append(msg.model_dump(exclude_none=True))
        if not msg.tool_calls:
            break
        for tc in msg.tool_calls:
            mcp_name = tc.function.name.removeprefix("mcp__jump-world__")
            args = json.loads(tc.function.arguments)
            log_event("tool_use", name=tc.function.name, input=args, t=time.monotonic()-t0)
            with state.lock:
                state.last_tool_t = time.monotonic() - t0
                state.tool_count += 1
            result = run_async(session.call_tool(mcp_name, args))
            content = result.content[0].text
            log_event("tool_result", id=tc.id, content_len=len(content), t=time.monotonic()-t0)
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": content})
            if mcp_name == "submit_hypothesis":
                submit_called = True

    if state.abort_event.is_set():
        raise StuckReasoningError(time.monotonic()-t0,
            (time.monotonic()-t0)-state.last_tool_t, state.tool_count)
    return collect_result(instance, state, t0)
```

The watchdog mirrors `run_arm_b_v3.py:187–210` line-for-line, with one
substitution: instead of `os.killpg`, set `state.abort_event` (the loop
checks it before each `client.chat.completions.create`) AND close the openai
HTTP connection so the in-flight request returns. If llama-server is in the
middle of generation, the openai client's `timeout=` will surface an
exception; that is the kill signal.

#### 3.4 Output JSON — identical schema to Arm B

Same keys: `instance_id, family, difficulty, accuracy, hypothesis_source,
hypothesis_status, n_interventions, n_turns, fallback, capture_path, wall_s,
stuck_since_last_tool_s, stuck_tool_use_count, model, effort, arm, version`.

Differences:
- `arm = "A"`
- `version = "v3"`
- `effort = None` (Qwen3 has no effort knob; document in §5)
- `model` = whatever `--model` resolved to (e.g. `qwen3-35b-a3b-mxfp4` or
  `qwen3-235b-a22b`)

#### 3.5 `parse_tool_uses` analogue

We don't parse a stream-json log — the dispatcher counted tool calls in
real time. `n_interventions` is read directly from the in-memory log we
persist to `arm_a/v3_progress_<instance_id>.jsonl` (same format as Arm B's
progress file). Submit-mandate enforcement is harness-side: see §5 row.

### Dependencies to install

```bash
cd /home/ak/projects/jump
uv pip install openai>=1.40.0   # mcp 1.27.0 already present
```

`anthropic` not needed; `mcp` already installed.

### Estimated implementation hours (Path B)

| Block | Hours |
|---|---|
| `arm_a/arm_a_mcp_server.py` (copy + verify) | 0.5 |
| `arm_a/run_arm_a_v3.py` skeleton (mirror Arm B structure) | 2 |
| MCP-client integration + tool-schema translation | 1.5 |
| OpenAI-compat agentic loop + tool dispatch | 2 |
| Watchdog port + StuckReasoningError + progress JSONL | 1 |
| Output schema + result extraction (`run_arm_b_v3.py:400–467` parity) | 1 |
| llama-server startup probe + config | 0.5 |
| Smoke test on `world_seq_001` (matches C₅ baseline) | 1.5 |
| **Total** | **10 hours** |

---

## §4 — Selection Decision

### Verdict: **Path B (custom OpenAI SDK + MCP client).**

### Rationale

1. **Code reuse is structural, not literal.** Arm B's `run_arm_b_v3.py` is
   the right template — agentic loop, watchdog, progress JSONL, results
   schema, MCP per-instance env. Path B reproduces this structure verbatim
   with a different LLM client. Path A introduces a third party (opencode)
   between us and the structure we already understand.
2. **Watchdog controllability.** Path B owns the agentic loop; the abort
   flag stops the loop on the next iteration without external killpg. Path A
   relies on SIGKILL of an opencode subprocess and depends on opencode not
   swallowing the signal during MCP teardown.
3. **Schema stability.** Path A's parse-shape depends on opencode's
   `--format json` event grammar across upgrades. Arm B already lost a day
   to this exact failure (`parse_tool_uses` shape-1-only bug fixed in C₆).
   Path B has zero schema-evolution risk: we control the dispatcher.
4. **Maintenance.** Path B sits next to `run_arm_b_v3.py`; future fixes (e.g.
   another mandate revision) port one-to-one. Single mental model for the
   reviewer.
5. **Path A does not save time.** Both paths must adapt the prompt, write
   per-instance MCP env, port the watchdog, translate tool schemas, and
   match output JSON. opencode replaces ~80 lines of Python with ~80 lines
   of opencode-glue + a separately-versioned binary dependency.
6. **Cost shape unchanged.** Both paths run free against local
   llama-server; the choice does not affect the high-N economic argument.

### When to revisit Path A

If a future task requires multi-agent coordination, session forking, or
human-in-the-loop intervention (opencode TUI features), Path A becomes
attractive. Phase 32e1 needs none of these.

### Estimated total: **10 hours** for chosen path (Path B).

---

## §5 — Feature Parity Map

| # | Arm B v3 C₆ feature | Where (Arm B) | Arm A v3 implementation strategy |
|---|---|---|---|
| 1 | `STUCK_REASONING_CAP_S = 600` watchdog | `run_arm_b_v3.py:31, 187–210` | Same constant; same threading.Thread polling every 5 s; on trigger set `abort_event` + close openai HTTP client; raise `StuckReasoningError` from outer loop. |
| 2 | `DEADLINE_NUDGE` verbatim | `run_arm_b_v3.py:57–72` | Import the same string from a shared module (`arm_b.deadline_nudge`) OR copy verbatim into `run_arm_a_v3.py`; **must be byte-identical** so cross-arm comparisons aren't confounded by prompt drift. |
| 3 | Submit mandate (after 5 intervenes → MUST submit) | Skill prompt + nudge | Prompt-side enforcement only; identical text. Harness does NOT auto-inject — symmetry with Arm B preserved. |
| 4 | `parse_tool_uses` dual-shape | `run_arm_b_v3.py:253–282` | Not needed: dispatcher counts in real time and writes the canonical record. `n_interventions` comes from the in-memory counter, not stdout parsing. |
| 5 | `n_interventions, wall_s, hypothesis_status` in output JSON | `run_arm_b_v3.py:398–467` | Same keys; populated from the in-memory `AgentState` at result-collection time. |
| 6 | `hypothesis_source` verbatim in output JSON | `run_arm_b_v3.py:404–419` | Same path: MCP-side persisted file (`results_path.read_text()`) is **primary** source of truth; in-memory dispatcher capture is **fallback**. |
| 7 | Accuracy via `WorldHarness.submit_hypothesis` | MCP server writes accuracy; rescore fallback at `run_arm_b_v3.py:285–289` | Identical: MCP server logic is reused unchanged. Fallback path uses `harness.WorldHarness` directly when MCP file missing. |
| 8 | MCP tools: `mcp__jump-world__{get_train_obs,intervene,submit_hypothesis}` | `arm_b/arm_b_mcp_server.py` | Reuse the **same** server file by symlink or copy as `arm_a/arm_a_mcp_server.py`. Server name string must stay `jump-world` so namespaced keys match across arms. |

### Intentional non-parity (documented)

| Field | Arm B | Arm A | Reason |
|---|---|---|---|
| `effort` | `low/medium/high/xhigh/max` | `null` | Qwen3 OpenAI-compat surface has no reasoning-effort knob. Setting to `null` rather than echoing `"high"` avoids a misleading metadata claim. |
| `RETRY_ON_TIMEOUT = 1` | enabled | enabled | Same retry semantics; identical constant. |
| Tool-naming prefix | `mcp__jump-world__` from claude | `mcp__jump-world__` synthesised | Cosmetic; preserves cross-arm string keys. |

---

## §6 — Reviewer Checklist

Five executable checks for the reviewer once Implementer ships:

1. `python3 -c "from arm_a.run_arm_a_v3 import DEADLINE_NUDGE; from arm_b.run_arm_b_v3 import DEADLINE_NUDGE as B; assert DEADLINE_NUDGE == B"` — **DEADLINE_NUDGE byte-identical** between arms.
2. `grep -E "STUCK_REASONING_CAP_S\s*=\s*600" arm_a/run_arm_a_v3.py` returns one match — **watchdog cap is 600 s**.
3. Run `cd /home/ak/projects/jump && llama-server -m … &` then `python3 arm_a/run_arm_a_v3.py --only world_seq_001 --output arm_a/results_v3_smoke_seq.json`; the resulting JSON must contain `arm == "A"`, `version == "v3"`, `accuracy` non-null, `n_interventions >= 1`, `hypothesis_status == "submitted"` — **smoke parity with Arm B C₅ SEQ baseline**.
4. `jq 'keys' arm_a/results_v3_smoke_seq.json[0]` and `jq 'keys' arm_b/results_v3_c5_seq_regression.json[0]` produce identical key sets except `effort` may be `null` in Arm A — **output schema parity**.
5. Trigger watchdog on purpose: `python3 arm_a/run_arm_a_v3.py --only world_ca_001 --max-intervenes 0 …` (or comparable forced-stuck path); verify `hypothesis_status == "FAIL_STUCK_REASONING"`, `stuck_since_last_tool_s` populated, exit code 0 (instance recorded, not crashed) — **failure-path symmetry**.

---

## §7 — Open Items for Implementer (not blocking design)

- Decide whether to download Qwen3-235B-A22B-MXFP4 (~120 GB; must free
  ≥50 GB) or run smoke on Qwen3.6-35B-A3B-MXFP4. Default to the latter for
  Phase 32e1 unless Director vetoes.
- Confirm Qwen3 jinja template renders OpenAI `tools` parameter correctly
  with `--jinja`; if tool-call emission is unreliable, fall back to
  `tools=[]` + JSON-grammar-constrained prompt (last resort).
- Decide on shared-module location for `DEADLINE_NUDGE` (currently inline in
  Arm B). A `repo/common/prompts.py` import target avoids future drift.
