# SPEC — arm_b v3 Phase C₆ prompt + harness fixes

Status: design (no code changes yet). Implementer task to follow.
Inputs: C₅ smoke result on `world_ca_001` (Opus 4.7, effort=high).
- 13 tool_use events logged (1 ToolSearch, 1 get_train_obs, 11 intervene)
- first intervene at t=161s; last intervene at t=551s
- 304s silence after last intervene → watchdog FAIL_STUCK_REASONING at 855s
- `submit_hypothesis` never called
- `n_interventions=0` in results JSON despite 11 intervene events in progress JSONL

---

## §1 — Submit-side mandate

### Bullet to add to DEADLINE_NUDGE (quote-ready)

> After 5 successful `intervene` calls (a "successful" call = received a
> `tool_result`, regardless of action validity), your NEXT action MUST be
> `submit_hypothesis` with your best current rule. Stating uncertainty in
> Python comments is permitted; continued probing without submission is not.

### "Successful" definition

A `intervene` call is **successful** iff the model receives a `tool_result`
block referencing that call's `tool_use_id`, **regardless of whether the
action was semantically valid in the world** (e.g. an out-of-bounds row or
unknown action that returns a structured error message still counts; the only
failures excluded are MCP transport errors that produce no tool_result at
all).

This definition is **self-monitorable** by the model: tool_result blocks are
visible in its own context window after each tool call. The model does not
need to inspect harness internals — it counts the tool_results it has seen.

Rationale: a stricter "valid action only" definition would force the model to
distinguish action-grammar errors from world-rule violations, which is exactly
the analytical task the deadline policy is trying to compress.

### Count vs time tracking — resolution

Director's spec proposed "5 successful intervenes OR 240 seconds elapsed".
**Drop the 240s clause.** The model has no wall-clock access (no system time
tool; no harness-injected timestamps in messages). Asking it to self-monitor
elapsed seconds invites hallucinated estimates and silent non-compliance.

Pure-count criterion ("5 successful intervenes") is unambiguous, deterministic,
and self-verifiable from the model's own context. Acting earlier is permitted
and the bullet on "weak submission > silence" already biases toward early
submission.

### Conflict check vs first-action mandate

First-action mandate (revised, see §3): "before any hypothesis, FIRST tool
call after get_train_obs MUST be intervene".
Submit mandate: "after 5 successful intervenes, NEXT action MUST be submit".

**No conflict.** The two mandates apply to disjoint action positions:
- Position 1 (immediately after get_train_obs): forced to `intervene`.
- Positions 2-5: free (intervene or submit).
- Position 6 (after 5 successful intervenes): forced to `submit_hypothesis`.

Edge case: a model could submit at position 2 (only 1 intervene) — that is
**permitted** by both mandates (first-action satisfied; submit-mandate not yet
triggered). Early submit is the desired behavior, not a violation.

---

## §2 — Watchdog cap

`STUCK_REASONING_CAP_S`: **300 → 600**.

### Conflict check (run_arm_b_v3.py)

| Constant | Value | Interaction with 600s cap |
|---|---|---|
| `INSTANCE_HARD_CAP` | 7200 | 600 ≪ 7200, no conflict |
| `RETRY_ON_TIMEOUT` | 1 | Worst case: 2 attempts × ~600s stuck-kill = 1200s, well under 7200s |
| `STUCK_WATCHDOG_POLL_S` | 5 | Poll cadence unchanged; granularity already fine |
| outer driver loop | per-instance | No assumption on per-attempt duration |

No timing assumption depends on the 300s value. The watchdog message
(`StuckReasoningError.__str__`) auto-formats `cap=` from the constant, so the
new value propagates to logs without further edits.

---

## §3 — First-action mandate revision

### Recommended phrasing (quote-ready)

> Before forming or articulating any hypothesis, your FIRST tool call after
> `get_train_obs` MUST be `intervene`. Treat the first probe as a reflex,
> not a conclusion.

### Justification vs alternatives

| Candidate | Verdict |
|---|---|
| "Within 30 seconds of receiving get_train_obs, you MUST call intervene." | **Reject.** Model has no wall-clock; "30 seconds" is not self-monitorable. C₅ shows the model spent 161s thinking before its first intervene — it cannot observe that it has done so. |
| Current C₅ wording: "your FIRST subsequent action MUST be a call to intervene." | **Reject (status quo).** "Subsequent" was interpreted as "eventually after thinking" — failed in C₅. |
| "Before forming any hypothesis, FIRST call MUST be intervene. Treat as reflex, not conclusion." | **Adopt.** Intent-based, anchored on a cognitive boundary the model controls (its own act of hypothesis-formation), and the "reflex not conclusion" framing directly counters the failure mode of analyse-first-then-probe. |

The intent-based form converts an unobservable physical constraint
("seconds") into an observable cognitive one ("have I started forming a
hypothesis?"). The model can self-check the latter; it cannot self-check the
former.

---

## §4 — `n_interventions` root cause + fix proposal

### Root cause

File: `arm_b/run_arm_b_v3.py`, function `parse_tool_uses` at lines **251-266**.

```python
def parse_tool_uses(stdout: str, name: str) -> list[dict]:
    ...
    for line in stdout.splitlines():
        ev = json.loads(line)
        if ev.get("type") == "assistant":           # ← only this branch
            for block in ev.get("message", {}).get("content", []):
                if block.get("type") == "tool_use" and block.get("name") == name:
                    results.append(block)
    return results
```

The function inspects **only** stream-json events with top-level
`type == "assistant"`, walking the `message.content` blocks for nested
`tool_use` blocks.

However, `_log_event` (lines 119-167) handles **two** event shapes from the
`claude --output-format stream-json --include-partial-messages` output:

1. **Partial-stream events** with top-level `type == "tool_use"` directly
   carrying `name` and `input` (handled at line 123).
2. **Consolidated assistant messages** with `type == "assistant"` whose
   `message.content[*].type == "tool_use"` (handled at line 149).

In the C₅ run, the 13 tool_use rows in `v3_progress_world_ca_001.jsonl` came
through path (1) — the partial-stream `type == "tool_use"` events — because
`_log_event` increments `last_tool_count` for both shapes (lines 130, 159).
The watchdog therefore correctly counted ≥1 tool calls.

`parse_tool_uses` only handles shape (2). When the run was watchdog-killed
mid-turn, no consolidated assistant message containing the 11 intervene
tool_uses was ever flushed to stdout — only the partial-stream form, which
`parse_tool_uses` ignores. Hence `n_interventions = 0`.

### Minimal fix (pseudocode for implementer)

Extend `parse_tool_uses` to match both shapes and de-duplicate by
`tool_use.id` (when present) so a future flush of both forms is not
double-counted:

```python
def parse_tool_uses(stdout: str, name: str) -> list[dict]:
    results = []
    seen_ids = set()
    for line in stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            ev = json.loads(line)
        except json.JSONDecodeError:
            continue
        # Shape 1: top-level partial-stream tool_use event
        if ev.get("type") == "tool_use" and ev.get("name") == name:
            tid = ev.get("id")
            if tid is None or tid not in seen_ids:
                if tid is not None:
                    seen_ids.add(tid)
                results.append(ev)
            continue
        # Shape 2: consolidated assistant message with content blocks
        if ev.get("type") == "assistant":
            for block in ev.get("message", {}).get("content", []):
                if block.get("type") == "tool_use" and block.get("name") == name:
                    tid = block.get("id")
                    if tid is None or tid not in seen_ids:
                        if tid is not None:
                            seen_ids.add(tid)
                        results.append(block)
    return results
```

This affects both `count_interventions` (line 276) and the
`submit_hypothesis` fallback parser (line 409) — both gain correct counts
without further edits.

---

## §5 — Updated DEADLINE_NUDGE (full text, quote-ready)

```
## Deadline policy (harness-enforced)

You operate under a hard turn budget. Endless silent reasoning is killed as
FAIL_STUCK_REASONING with zero credit and no retry — worse than any submission.

- Before forming or articulating any hypothesis, your FIRST tool call after
  `get_train_obs` MUST be `intervene`. Treat the first probe as a reflex,
  not a conclusion.
- After 5 successful `intervene` calls (a "successful" call = received a
  `tool_result`, regardless of action validity), your NEXT action MUST be
  `submit_hypothesis` with your best current rule. Stating uncertainty in
  Python comments is permitted; continued probing without submission is not.
- A weak submission with uncertainty noted in comments is strictly better
  than silence or extended internal analysis.
- Externalise reasoning through tool calls, not internal monologue.
```

Line count: 15 (4 header/preamble + 11 bulleted body). Within ≤18 budget.

---

## §6 — Reviewer checklist (executable pass/fail)

1. **NUDGE length.** `awk 'NR>=start && NR<=end' run_arm_b_v3.py` over the new
   `DEADLINE_NUDGE` string body shows ≤ 18 lines between the opening and
   closing `"""`.
2. **Submit mandate is unconditional.** The submit-mandate bullet contains no
   "if confident", "if you have a hypothesis", "consider", or "may" qualifier
   on the trigger. Trigger phrasing is exactly count-based ("5 successful
   intervene calls"), not time-based ("seconds", "minutes").
3. **First-action mandate is intent-based.** The first-action bullet contains
   no numeric time qualifier (no "30 seconds", no "first 60s") and is anchored
   on hypothesis-formation, not wall clock.
4. **Watchdog cap.** `grep -E '^STUCK_REASONING_CAP_S\s*=' run_arm_b_v3.py`
   returns exactly `STUCK_REASONING_CAP_S = 600` (with whatever comment).
5. **`parse_tool_uses` handles both event shapes.** The function body
   contains a branch for `ev.get("type") == "tool_use"` AND a branch for
   `ev.get("type") == "assistant"`, with a shared dedup set keyed on
   tool_use `id`. Smoke-test fixture: feed the existing
   `v3_progress_world_ca_001.jsonl` (after wrapping each row as the
   stream-json shape if needed) through `count_interventions` and assert
   result == 11.
