# SPEC — Arm B v3 Phase C₂ Modifications

Author: designer (v2_p16c2_design)
Target file: `arm_b/run_arm_b_v3.py` (primary), `arm_b/jump_v3_skill.md` (untouched)
Trigger: Phase C smoke FAIL_NO_SUBMIT on `world_ca_001` (wall=1600.9s, n_int=0,
2 tool_uses then 2164 partial-stream events with **no** `message_stop` and no further
tool call).

## §1 Turn cap spec

### 1.1 Unit: wall-clock seconds since last tool_use

Justified by the failure evidence:

```
$ grep -oE "event: [a-z_]+" arm_b/run_log_v3_smoke_ca.txt | sort | uniq -c
   2164 event: stream_event
      7 event: system
      2 event: user
      1 event: result
      1 event: rate_limit_event
```

Zero `message_stop` events. Zero finalized `assistant` content blocks after the second
tool_use. Therefore counting "assistant turns" or `message_stop`s would never fire — the
model stayed in mid-turn extended-thinking partial-stream output for ~26 minutes. The
only continuously-monotonic observable signal is wall-clock time since the last
`tool_use` event we logged in `_log_event`.

Counting `tool_result` is equivalent (every tool_use yields a tool_result), but tool_use
is what `_log_event` already timestamps in `_append_progress`, so reuse that hook.

### 1.2 Threshold: `STUCK_REASONING_CAP_S = 300` (5 min)

- Healthy SEQ run gaps between consecutive tool_uses (from
  `v3_progress_world_seq_001.jsonl`): 10→49→58→65→74s, max gap = **39s**.
- 300s gives ~7.7× headroom over the largest observed healthy gap.
- For CA (no healthy reference yet), 300s of *zero* tool calls still bounds reasoning
  generously: extended thinking that warrants a full 5 minutes without committing to a
  probe is, by Phase C₂'s definition, stuck.
- Detects the failure at t ≈ 9.37 + 300 = **309s** instead of 1600s actual (5.2×
  faster), satisfying the success criterion.

### 1.3 Architecture: dedicated watchdog thread (Option B)

`_read_stdout` cannot self-monitor inactivity — it blocks on
`for line in proc.stdout:` and runs only when bytes arrive. (Notably, in the failure
case it ran constantly because partial deltas streamed; a busy reader can still be
"stuck" by our definition.) A separate watchdog must observe an externally-updated
timestamp.

**Shared state inside `run_subprocess`:**

```python
last_tool_t      : float                      # monotonic seconds (relative to t0); init 0.0
state_lock       : threading.Lock             # guards last_tool_t writes
stuck_flag       : threading.Event            # set when watchdog kills proc
last_tool_count  : int                        # for stuck-event progress payload
```

**`_read_stdout` change:** when a `tool_use` event is observed (both in the
`type=="tool_use"` branch and the nested `assistant`→`tool_use` branch), update:

```python
with state_lock:
    last_tool_t = elapsed
    last_tool_count += 1
```

**Watchdog loop (new thread, daemon=True, started after stdin write):**

```python
def _watchdog():
    POLL_S = 5.0
    while not stuck_flag.is_set():
        time.sleep(POLL_S)
        if proc.poll() is not None:
            return                            # process exited normally
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
```

**Note `count >= 1`:** the cap only arms after the first tool_use. If the model never
calls *any* tool (e.g. authentication failure), the existing `INSTANCE_HARD_CAP=7200`
catches it. The cap is specifically for "started exploring, then froze," matching the
Phase C failure mode.

### 1.4 Result-dict schema for FAIL_STUCK_REASONING

`run_subprocess` raises a new exception `StuckReasoningError(elapsed: float, since_last_tool_s: float, tool_use_count: int)` after `stuck_thread.join`/`proc.wait` if `stuck_flag.is_set()`. Define at module scope:

```python
class StuckReasoningError(Exception):
    def __init__(self, wall_s, since_last_tool_s, tool_use_count):
        self.wall_s = wall_s
        self.since_last_tool_s = since_last_tool_s
        self.tool_use_count = tool_use_count
```

`run_one_instance` catches it inside the attempt loop and returns:

```python
{
    "instance_id": instance["id"],
    "family": instance["family"],
    "difficulty": instance["difficulty"],
    "accuracy": None,
    "hypothesis_source": None,
    "hypothesis_status": "FAIL_STUCK_REASONING",
    "n_interventions": 0,                        # by definition: no intervene since cap arms after get_train_obs
    "n_turns": None,
    "fallback": True,
    "capture_path": None,
    "wall_s": e.wall_s,
    "stuck_since_last_tool_s": e.since_last_tool_s,
    "stuck_tool_use_count": e.tool_use_count,
    "model": model,
    "effort": "medium",
    "arm": "B",
    "version": "v3",
}
```

Progress JSONL: the `stuck_reasoning_kill` record (written by the watchdog before the
SIGKILL) is the in-band marker. Distinguishable from the existing `timeout_sigkill`
record.

### 1.5 Retry interaction

`FAIL_STUCK_REASONING` MUST NOT trigger retry. Implementation: `run_one_instance`
catches `StuckReasoningError` *outside* the existing
`except subprocess.TimeoutExpired` block and returns immediately without `continue`-ing
the `for attempt in range(...)` loop. Justification matches user's spec — retries are
for transient network issues; stuck reasoning is a deterministic model-state condition
that re-rolling the same prompt would reproduce.

```python
for attempt in range(1 + RETRY_ON_TIMEOUT):
    try:
        stdout, stderr, rc, wall_s = run_subprocess(...)
        break
    except StuckReasoningError as e:
        # NO retry, NO continue
        return {... "hypothesis_status": "FAIL_STUCK_REASONING" ...}
    except subprocess.TimeoutExpired:
        if attempt >= RETRY_ON_TIMEOUT: return {... FAIL_TIMEOUT_RETRY ...}
        continue
```

---

## §2 Prompt nudge spec

### 2.1 Placement: appended in `build_prompt()` (Option B)

Justification:

| Concern | A (skill file) | **B (build_prompt)** | C (--append-system-prompt) |
|---|---|---|---|
| Versioning | Skill file mixes prompt + harness policy | **Skill stays canonical; harness owns deadline** | Same as B but split across CLI |
| Isolation per phase | Edit/revert touches the skill | **Single-file change, easy revert** | CLI string scattered in code |
| Reproducibility | Future runs use whichever skill version is on disk | **Skill content in repo is unchanged; phase-C₂ behavior is in driver code** | Same as B |
| Risk of leak across runs | Yes (other harnesses read the skill) | **No** | No |

Phase C₂ is a harness-policy fix, not a skill-content fix. Keep `jump_v3_skill.md`
untouched so the underlying world-abduction skill remains a stable reference document.

### 2.2 Exact nudge text

To be appended *verbatim* (no reformatting) at the end of `skill_body` inside
`build_prompt`. Quoted block:

```
## Deadline policy (harness-enforced)

You operate under a hard turn budget. Endless silent reasoning will be killed as
FAIL_STUCK_REASONING with zero credit and no retry — a worse outcome than any
submission.

- You MUST call `intervene` at least once before calling `submit_hypothesis`. A
  hypothesis without at least one intervention is unacceptable.
- If, after a handful of interventions, your confidence is low, submit your best
  current hypothesis anyway. Note your uncertainty in a Python comment inside
  `hypothesis_source`. A weak submission with stated caveats is strictly better
  than no submission.
- Do not loop in private analysis between tool calls. Commit to a probe, read the
  result, refine — externalise reasoning through tool calls, not internal monologue.
```

**Leak audit:** the text mentions no world family, no rule structure (deterministic vs
stochastic, lookup vs arithmetic, neighbourhood radius, etc.), and no test-set
properties. It only describes harness policy. Reviewer can verify by `grep` against
`worlds/gen.py` rule-name tokens — none should appear.

### 2.3 Position in final prompt

Final prompt structure (ordering in `build_prompt`):

```
{skill_body}                                     ← unchanged jump_v3_skill.md content
                                                   (ends "...interventions support the rule.")
\n\n
## Deadline policy (harness-enforced)            ← new nudge (§2.2 text)
...
\n\n
Begin by calling get_train_obs.                  ← unchanged final imperative
```

Rationale: the skill's abductive principles establish *what* to do; the deadline
policy then sets *operational constraints*; finally the start trigger fires. Placing
the policy after skill content lets it override the skill's
"Do not submit until interventions support the rule" with the more permissive
"submit a weak hypothesis rather than nothing" without editing the skill.

`build_prompt` becomes:

```python
DEADLINE_NUDGE = """## Deadline policy (harness-enforced)

You operate under a hard turn budget. ...
... externalise reasoning through tool calls, not internal monologue.
"""

def build_prompt(skill_body: str) -> str:
    return skill_body + "\n\n" + DEADLINE_NUDGE + "\nBegin by calling get_train_obs."
```

---

## §3 Re-smoke gate criteria

Phase C₂ smoke = same single instance `world_ca_001`, fresh `claude -p`, with §1+§2
applied.

| Verdict | hypothesis_status | n_interventions | accuracy | wall_s |
|---|---|---|---|---|
| **PASS** | `submitted` | ≥ 1 | any value (incl. 0.0) | ≤ 600 |
| **MARGINAL** | `FAIL_STUCK_REASONING` | (= 0 by definition) | None | ≤ STUCK_CAP × 1.1 ≈ 330 |
| **FAIL** | `FAIL_NO_SUBMIT` | any | None | any |
| **FAIL** | `FAIL_TIMEOUT` / `FAIL_TIMEOUT_RETRY` | any | None | ≥ 7200 |
| **FAIL** | `FAIL_EXCEPTION` | any | None | any |

Verdicts are mutually exclusive (each `hypothesis_status` value appears in exactly one
row) and exhaustive (covers all five values produced by `run_one_instance`).

`submitted` with n_int = 0 cannot occur after §2 lands, because the nudge mandates
intervene-first; if it does, treat as **FAIL** (nudge ineffective + protocol violation).

### 3.1 Why these thresholds

- **PASS** ignores accuracy. Phase C₂ is a *plumbing* test: did the cap+nudge enable
  the model to traverse get_train_obs → intervene → submit_hypothesis? Solving CA on
  the first medium-difficulty world is not the bar.
- **MARGINAL** = the cap fired correctly. Plumbing for the failure path is verified;
  the model's inability to commit on this specific world is a separate signal.
- **FAIL_NO_SUBMIT** post-fix means the cap did not fire — bug in §1 implementation.
  The hard cap (7200s) catching a frozen run also indicates the watchdog never armed.

### 3.2 Escalation trigger

| Outcome | Action |
|---|---|
| PASS | Proceed to Phase D as designed. |
| MARGINAL (FAIL_STUCK_REASONING after nudge on `world_ca_001`) | **Escalate to Director.** Rationale: cap+nudge demonstrably installed (the cap fired and the model received the intervene-first policy), yet the model still produced no probe in 300s on a 4×4 3-state CA. This is evidence of an abduction-inability mode for this family/difficulty under medium effort, not a harness bug. Director should weigh: (a) raise `--effort` to high and re-smoke, (b) downgrade smoke instance to `easy`, (c) accept the signal and adjust the Arm B research question. |
| FAIL: FAIL_NO_SUBMIT or FAIL_TIMEOUT* on smoke | **Re-implement.** §1 watchdog has a bug — wall-clock cap should have fired before either of these states is reachable. Manager re-dispatches implementer with the watchdog log showing why kill didn't trigger. Do NOT escalate to Director. |
| FAIL_EXCEPTION | Read traceback; fix and re-run. Not an escalation. |

A single MARGINAL on `world_ca_001` is sufficient escalation evidence; do not silently
re-roll on a different CA seed before notifying Director, because the seed change would
confound the abduction-inability signal with seed variance.

---

## §4 Implementation checklist

All edits are in `arm_b/run_arm_b_v3.py` unless noted. Line numbers are approximate
(based on current file at HEAD).

1. **Add module constants** near the top (after `RETRY_ON_TIMEOUT = 1`, ~line 31):
   ```python
   STUCK_REASONING_CAP_S = 300       # seconds since last tool_use before kill
   STUCK_WATCHDOG_POLL_S = 5         # watchdog loop cadence
   ```

2. **Add module-scope exception class** (after constants, before `load_skill_body`):
   ```python
   class StuckReasoningError(Exception):
       def __init__(self, wall_s, since_last_tool_s, tool_use_count):
           self.wall_s = wall_s
           self.since_last_tool_s = since_last_tool_s
           self.tool_use_count = tool_use_count
           super().__init__(
               f"stuck reasoning: {since_last_tool_s:.1f}s without tool_use "
               f"after {tool_use_count} tool calls (cap={STUCK_REASONING_CAP_S}s)"
           )
   ```

3. **Add `DEADLINE_NUDGE` constant** (above `build_prompt`, ~line 44):
   - Contents: the literal text from §2.2 of this spec.

4. **Modify `build_prompt`** (line 45–46):
   ```python
   def build_prompt(skill_body: str) -> str:
       return skill_body + "\n\n" + DEADLINE_NUDGE + "\nBegin by calling get_train_obs."
   ```

5. **Modify `run_subprocess`** (lines 67–175):
   - After `stdout_lines = []` / `stderr_lines = []` (~line 80), add shared state:
     ```python
     last_tool_t = 0.0
     last_tool_count = 0
     state_lock = threading.Lock()
     stuck_flag = threading.Event()
     ```
   - Inside `_log_event`, on every branch that records a `tool_use` (both the
     top-level `ev_type == "tool_use"` branch ~line 90 AND the nested
     `assistant`-message tool_use branch ~line 116), after `_append_progress(...)`
     add:
     ```python
     nonlocal last_tool_t, last_tool_count
     with state_lock:
         last_tool_t = elapsed
         last_tool_count += 1
     ```
     (Declare `nonlocal` at the top of `_log_event`.)
   - Define `_watchdog()` (the function from §1.3) inside `run_subprocess` after
     `_read_stderr`.
   - After `stdout_thread.start()` and `stderr_thread.start()` (~line 155), add:
     ```python
     watchdog_thread = threading.Thread(target=_watchdog, daemon=True)
     watchdog_thread.start()
     ```
   - In the `try` block after `stdout_thread.join(timeout=timeout)`, before
     `stderr_thread.join`, add:
     ```python
     if stuck_flag.is_set():
         stderr_thread.join(timeout=5)
         wall_s = time.monotonic() - t0
         since = wall_s - last_tool_t
         raise StuckReasoningError(wall_s, since, last_tool_count)
     ```

6. **Modify `run_one_instance` retry loop** (lines 234–277):
   - Add `except StuckReasoningError as e:` between the `try`/`break` and the
     existing `except subprocess.TimeoutExpired`:
     ```python
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
             "effort": "medium",
             "arm": "B",
             "version": "v3",
         }
     ```
   - Critically: this `return` is inside the `try:` body — `progress_file.close()`
     in the `finally` still runs.

7. **Update `main()` summary** (lines 456–462):
   - Add `stuck_count = statuses.count("FAIL_STUCK_REASONING")` and include it
     in the `FAIL counts:` print.
   - Optionally treat `FAIL_STUCK_REASONING` like other failures in the
     `family_fail_counts` escalation path (~lines 426–436), or leave it out
     deliberately (recommended: leave out — Phase C₂ wants this signal visible
     per-instance, not silenced via family escalation).

8. **Smoke command** (no code change — for reviewer/implementer to execute after
   1–7):
   ```bash
   cd /home/ak/tmux-agents/projects/jump/repo
   uv run python arm_b/run_arm_b_v3.py --only world_ca_001 \
       --output arm_b/results_v3_c2_smoke.json \
       2>&1 | tee arm_b/run_log_v3_c2_smoke.txt
   ```
   Verify `arm_b/v3_progress_world_ca_001.jsonl` contains a `stuck_reasoning_kill`
   record at `t ≈ 309–330` IF the model fails to commit, OR a
   `mcp__jump-world__intervene` followed by `mcp__jump-world__submit_hypothesis`
   record IF the nudge worked.

9. **Commit** (single commit, message body includes phase tag):
   ```
   arm_b v3 phase C2: turn cap watchdog + intervene-first nudge

   - Add STUCK_REASONING_CAP_S=300 watchdog killing subprocess if no tool_use
     in 300s after first tool call
   - Add deadline-policy nudge to build_prompt requiring intervene before submit
   - New result status FAIL_STUCK_REASONING (no retry)
   - Skill file jump_v3_skill.md unchanged
   ```

10. **Reviewer verification** (no code change — reviewer checklist):
    - Threshold: `STUCK_REASONING_CAP_S == 300` ✓ (would have caught Phase C
      failure at t≈309s vs 1600s observed).
    - Nudge text: no world/family/rule tokens — `grep -i "cellular\|particle\|pattern\|automaton\|prime\|xor" arm_b/run_arm_b_v3.py` returns no matches in the new constant.
    - Retry isolation: `FAIL_STUCK_REASONING` returns from inside the `for attempt` loop without `continue`.
    - Skill file untouched: `git diff HEAD~1 -- arm_b/jump_v3_skill.md` is empty.
