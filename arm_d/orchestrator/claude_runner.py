"""
P-D2 (post-MVP) — live `claude -p` candidate generator.

This module is a thin wiring sketch. The MVP does NOT call into it;
the pipeline runs against `engine/mock_generator.py` only. When P-D2
is implemented, `run_arm_d.py --use-claude` will call
`generate_pool_via_claude(instance)` to append to the mock pool.

Inherits Arm B's subprocess hardening:
  - 900 s turn-1, 600 s subsequent timeout
  - 3 retry attempts with backoff [0, 30, 60]
  - start_new_session=True, os.killpg on timeout
  - cwd=/tmp to avoid CLAUDE.md ingestion
  - --include-partial-messages, --output-format stream-json
"""
import json
import os
import signal
import subprocess
import time
from pathlib import Path

CLAUDE_CMD = [
    "env", "-u", "CLAUDECODE", "-u", "ANTHROPIC_API_KEY",
    "claude", "-p",
    "--dangerously-skip-permissions",
    "--output-format", "stream-json", "--include-partial-messages", "--verbose",
    "--model", "claude-sonnet-4-6",
]

PROMPTS_DIR = Path(__file__).parent / "prompts"

PERSONAS = [
    "careful", "creative", "empirical",
    "minimalist", "pattern_wide", "pattern_narrow",
]

TIMEOUT_TURN1 = 900
RETRY_BACKOFFS = [0, 30, 60]
M_PER_PERSONA = 14  # default; overridden by run_arm_d.py


def load_persona_prompt(persona: str) -> str:
    """Load a persona prompt template from disk. Falls back to a built-in if missing."""
    f = PROMPTS_DIR / f"generate_{persona}.md"
    if f.exists():
        return f.read_text()
    return _DEFAULT_PERSONA_PROMPT.replace("{{PERSONA_HINT}}", persona)


_DEFAULT_PERSONA_PROMPT = """You are a Python programmer. Below is an empirical observation set:
a list of (state, next_state) pairs from an unknown deterministic rule.
Your task: write Python source for `def hidden_rule_fn(state) -> next_state`
that, when run on each `state`, produces the corresponding `next_state`.

You may NOT call any external API. You may NOT refer to the words
'intervene', 'experiment', 'gather more data', or any reasoning-prose
field. Your output is the function source ONLY.

Style: {{PERSONA_HINT}}

train_obs:
{{TRAIN_OBS_JSON}}

primitive_glossary (informational only — names are arbitrary):
{{GLOSSARY_JSON}}

Output FORMAT (strict): a JSON list of {{M}} objects:
[{"source": "def hidden_rule_fn(state):\\n    ..."}, ...]

Investigation seed: {{SEED}}
"""


def _run_claude_subprocess(prompt: str, timeout: int) -> tuple[str, str, int]:
    """Run claude -p with proper process group cleanup on timeout.
    Returns (stdout, stderr, returncode). Raises subprocess.TimeoutExpired on hard timeout.
    """
    proc = subprocess.Popen(
        CLAUDE_CMD,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd="/tmp",
        start_new_session=True,
    )
    try:
        stdout, stderr = proc.communicate(input=prompt, timeout=timeout)
        return stdout, stderr, proc.returncode
    except subprocess.TimeoutExpired:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            proc.kill()
        try:
            proc.communicate(timeout=10)
        except subprocess.TimeoutExpired:
            pass
        raise


def _extract_sources_from_stream_json(stdout: str) -> list[str]:
    """Walk the stream-json output and extract any JSON list of {source: ...}."""
    sources = []
    seen = set()
    assistant_texts = []
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
                if block.get("type") == "text":
                    assistant_texts.append(block["text"])
        elif ev.get("type") == "result" and ev.get("subtype") == "success":
            assistant_texts.append(ev.get("result", ""))
    for text in assistant_texts:
        # Look for a JSON array
        try:
            parsed = json.loads(text)
            if isinstance(parsed, list):
                for entry in parsed:
                    if isinstance(entry, dict) and "source" in entry:
                        s = entry["source"]
                        if isinstance(s, str) and s not in seen:
                            seen.add(s)
                            sources.append(s)
        except json.JSONDecodeError:
            # Try to extract triple-backtick JSON blocks
            import re
            for m in re.finditer(r"```(?:json)?\s*(\[.*?\])\s*```", text, re.DOTALL):
                try:
                    parsed = json.loads(m.group(1))
                    for entry in parsed:
                        if isinstance(entry, dict) and "source" in entry:
                            s = entry["source"]
                            if isinstance(s, str) and s not in seen:
                                seen.add(s)
                                sources.append(s)
                except json.JSONDecodeError:
                    pass
    return sources


def generate_pool_via_claude(
    instance: dict,
    personas: list[str] | None = None,
    m_per_persona: int = M_PER_PERSONA,
    seed_base: int = 42,
) -> list[str]:
    """
    Issue one `claude -p` call per persona; concatenate the parsed sources.

    NOT CALLED IN MVP — the run_arm_d.py driver gates this behind --use-claude.
    """
    personas = personas or PERSONAS
    glossary = instance.get("primitive_glossary", {})
    train_obs = instance["train_obs"]

    out = []
    for i, persona in enumerate(personas):
        template = load_persona_prompt(persona)
        prompt = (
            template
            .replace("{{TRAIN_OBS_JSON}}", json.dumps(train_obs))
            .replace("{{GLOSSARY_JSON}}", json.dumps(glossary))
            .replace("{{M}}", str(m_per_persona))
            .replace("{{SEED}}", str(seed_base + i * 1009))
        )
        last_err = None
        stdout = ""
        for attempt, delay in enumerate(RETRY_BACKOFFS):
            if delay:
                time.sleep(delay)
            try:
                stdout, _stderr, rc = _run_claude_subprocess(prompt, TIMEOUT_TURN1)
                if rc != 0:
                    last_err = f"rc={rc}"
                    continue
                break
            except subprocess.TimeoutExpired as e:
                last_err = "TIMEOUT"
                continue
            except Exception as e:
                last_err = type(e).__name__
                continue
        if not stdout:
            # Persona failed entirely — skip
            continue
        out.extend(_extract_sources_from_stream_json(stdout))

    return out
