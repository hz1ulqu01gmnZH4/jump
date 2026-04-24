# SPEC v2-P6: Arm B timeout fix & fallback elimination

**Status**: design (pre-implementation)
**Supersedes**: silent fallback behaviour in `arm_b/run_arm_b_v2.py` (lines 194-201, 226-230, 242, 261-270)
**Authored**: 2026-04-24 (designer)

## 1. Problem summary

`arm_b/results_v2.json` reports `mean_acc=0.451` on 17 v2 instances, but **10/17 instances
(59%)** never executed the agent — they received a silent identity-function fallback
(`def hidden_rule_fn(state): return state`) because `claude -p` timed out at 300 s.
Current code (lines 226-230) suppresses the timeout into `fallback: True` + `accuracy:
0.0/0.333/...` (whatever score the identity function happens to achieve). This
violates the project's no-fallback policy and inflates/deflates the reported mean.

### 1.1 Timeout distribution (from `run_log_v2.txt`)

| Instance          | Turn of timeout | n_turns | Agent submitted? |
|-------------------|----------------:|--------:|:----------------:|
| world_ca_001      | 1 (retry fail)  | 2       | No — identity FB |
| world_ca_002      | 1               | 2       | No — identity FB |
| world_ca_003      | 1               | 2       | No — identity FB |
| world_ca_004      | 1               | 2       | No — identity FB |
| world_ca_005      | 1               | 2       | No — identity FB |
| world_ca_006      | 1               | 2       | No — identity FB |
| world_pt_002      | 1               | 2       | No — identity FB |
| world_pt_003      | 1               | 2       | No — identity FB |
| world_pt_004      | 1               | 2       | No — identity FB |
| world_pt_005      | 1               | 2       | No — identity FB |
| world_seq_005     | 8 (mid-run)     | 10      | Yes (acc=1.0)    |

**Pattern**: **9/10** failures occur at turn 1 — the very first `claude -p` call with no
interaction history. The only mid-run timeout (`seq_005` at turn 8) had already
submitted successfully, so its accuracy is valid. All 6/6 cellular_automata and
4/5 particle_system instances failed at turn 1; pattern_puzzle mostly succeeded.

The 300 s turn-1 hang is consistent with known `claude -p` silent-hang behavior
(GH issue #29543) and/or MCP-server / skill-load cold-start stalls — not model
latency (the prompt is small: ~2.3 KB skill + ~1–2 KB world context). The bias
toward CA/PT is likely a secondary effect of non-trivial `intervention_api` /
`primitive_glossary` payload causing different tokenization or tool-init paths,
but the root cause is the hang, not compute time.

## 2. Recommendation: **Path A (increase timeout + retries + hard failure)**

**Not Path B (SDK switch).** Reasons:

1. **Subscription auth compatibility.** The current run uses
   `env -u ANTHROPIC_API_KEY claude -p …` to run under the Claude Code
   subscription. The Anthropic Python SDK (`anthropic.Anthropic()`) requires
   an `ANTHROPIC_API_KEY` env var. Switching to `arm_b/driver.py`-style SDK
   calls would silently change the auth backend and cost model, which is a
   methodology change on top of a timeout fix.
2. **Methodology parity with v2-P4.** The point of v2-P6c is a **re-run**, not
   a re-design. Changing the inference backend mid-experiment contaminates the
   v2-P4 vs v2-P6 comparison (different model routing, different tool-use
   surface — SDK doesn't have the Claude Code tool-use loop or MCP).
3. **Tool-use loop.** v2 requires interactive `get_train_obs` / `intervene`
   / `submit_hypothesis` calls. The current driver embeds this as a custom
   history-in-prompt loop; replicating it via SDK is feasible but non-trivial
   and would need its own review.
4. **`driver.py` is for a different eval.** `arm_b/driver.py` runs a single-
   shot JSON-schema abduction task (`eval/tasks.jsonl`), not the v2
   interactive harness. Reusing it requires a substantial rewrite, not a drop-
   in replacement.
5. **Cost.** `claude -p` under subscription is sunk-cost; SDK calls are
   metered and may not be approved.

Path A is a ~10-line diff. Path B is a new file.

## 3. Path A implementation spec

### 3.1 Timeout & retry parameters

| Parameter           | Current | New   | Rationale                              |
|---------------------|--------:|------:|----------------------------------------|
| `timeout` (turn 1)  | 300 s   | **900 s** | Cold-start + MCP init can exceed 5 min |
| `timeout` (turn ≥2) | 300 s   | **600 s** | Warm cache, shorter prompts            |
| Max retries         | 1       | **2**     | Total 3 attempts per turn              |
| Retry backoff       | 15 s    | **30 s, 60 s** | Exponential; avoid thundering herd |

Upper-bound wall time if all retries fire: 17 instances × 12 turns × (900+600+600)s
= far more than acceptable. **Hard per-instance cap**: 1800 s. If exceeded,
abort instance and record `FAIL_TIMEOUT`. Typical successful run is <60 s/instance
(see seq_* timings).

### 3.2 `FAIL_TIMEOUT` marker (no-fallback compliance)

Replace the current silent fallback with a loud failure. **Do not call
`harness.submit_hypothesis` with an identity function.** Instead, record
diagnostic state and exclude from the accuracy mean.

```python
# arm_b/run_arm_b_v2.py — replace lines 193-201, 226-230, 242

def run_arm_b_on_instance(instance: dict) -> dict:
    from harness import WorldHarness

    skill_body = load_skill_body()
    harness = WorldHarness(instance)
    history = []
    submitted = False
    timeout_turn = None         # NEW: which turn fatally timed out
    timeout_attempts = 0        # NEW: cumulative retry count
    n_turns = 0
    instance_start = time.monotonic()
    INSTANCE_CAP_SEC = 1800

    for turn in range(MAX_TURNS):
        n_turns = turn + 1
        if time.monotonic() - instance_start > INSTANCE_CAP_SEC:
            timeout_turn = turn
            print(f"\n    [FAIL_TIMEOUT instance cap @ turn {turn}]", flush=True)
            break

        prompt = build_prompt(skill_body, instance, history, turn)
        turn_timeout = 900 if turn == 0 else 600
        backoffs = [0, 30, 60]  # 3 attempts

        tool_calls = None
        last_err = None
        for attempt, delay in enumerate(backoffs):
            if delay:
                time.sleep(delay)
            try:
                tool_calls, _ = run_claude(prompt, timeout=turn_timeout)
                break
            except subprocess.TimeoutExpired as e:
                timeout_attempts += 1
                last_err = e
                print(f"\n    [timeout turn {turn} attempt {attempt+1}/3 "
                      f"after {turn_timeout}s]", flush=True)
            except Exception as e:
                last_err = e
                print(f"\n    [error turn {turn} attempt {attempt+1}/3]: {e}",
                      flush=True)

        if tool_calls is None:
            # All attempts failed — record FAIL_TIMEOUT, abort instance, no fallback.
            timeout_turn = turn
            print(f"\n    [FAIL_TIMEOUT turn {turn}, last_err={last_err!r}]",
                  flush=True)
            break

        # ... existing tool dispatch loop (lines 203-224) unchanged ...

    # Build result — NO IDENTITY FALLBACK.
    log = harness.get_log()
    subs = [e for e in log if e["tool"] == "submit_hypothesis"]

    if submitted:
        return {
            "instance_id": instance["id"],
            "family": instance["family"],
            "difficulty": instance["difficulty"],
            "accuracy": subs[-1]["accuracy"],
            "hypothesis_source": subs[-1].get("hypothesis_source", ""),
            "n_interventions": sum(1 for e in log if e["tool"] == "intervene"),
            "n_turns": n_turns,
            "fallback": False,
            "hypothesis_status": "submitted",
        }
    else:
        # Agent never submitted — either timed out or exhausted MAX_TURNS without submit.
        status = "FAIL_TIMEOUT" if timeout_turn is not None else "FAIL_NO_SUBMIT"
        return {
            "instance_id": instance["id"],
            "family": instance["family"],
            "difficulty": instance["difficulty"],
            "accuracy": None,               # excluded from mean
            "hypothesis_source": None,
            "n_interventions": sum(1 for e in log if e["tool"] == "intervene"),
            "n_turns": n_turns,
            "fallback": True,
            "hypothesis_status": status,
            "timeout_turn": timeout_turn,
            "timeout_attempts": timeout_attempts,
        }
```

And update `main()` exception handler (lines 258-270) similarly — on exception
set `accuracy: None`, `hypothesis_status: "FAIL_EXCEPTION"`, `error: str(e)`.

**Aggregation rule** (replace line 276):
```python
scored = [r for r in results if r["accuracy"] is not None]
n_excluded = len(results) - len(scored)
mean_acc = statistics.mean(r["accuracy"] for r in scored) if scored else float("nan")
print(f"\n=== Arm B mean accuracy: {mean_acc:.3f} over {len(scored)}/{len(results)} "
      f"(excluded {n_excluded} FAIL_*) ===")
```

### 3.3 run_log format

Prefix all timeout/failure lines with a machine-parseable tag so v2-P6c
post-hoc analysis can grep them:

```
[FAIL_TIMEOUT instance=world_ca_001 turn=0 attempts=3 total_wait_s=2580]
[FAIL_NO_SUBMIT instance=world_ca_004 turn=12]
[FAIL_EXCEPTION instance=world_ca_006 err=RuntimeError(...)]
[OK instance=world_pt_001 turns=2 acc=1.000]
```

Emit at instance end in `main()`. Keep existing per-turn progress messages.

## 4. v2-P6c merge spec (`arm_b/merge_v2_p6.py`)

### 4.1 Inputs
- `arm_b/results_v2.json` — 17 entries from v2-P4 (contains 10 silent-fallback bogus scores + 7 valid).
- `arm_b/results_v2_p6.json` — **10 entries** from re-run. Re-run set is the exact
  10 instances where `results_v2.json[i].fallback == True`:
  `world_ca_001..006`, `world_pt_002`, `world_pt_003`, `world_pt_004`, `world_pt_005`.
  *Note*: `world_seq_005` was `fallback=False` (submitted before its mid-run
  timeout) and is **not** re-run.

### 4.2 Merge rule
For each instance_id:
1. If present in `results_v2_p6.json`, take the p6 entry (even if p6 entry is
   `FAIL_TIMEOUT` — p6 is authoritative because it ran with the new timeout
   policy and honest error recording).
2. Otherwise, take the v2 entry.
3. For v2 entries carried over, clear the `fallback` field only if v2 had
   `fallback=False` (the 7 valid instances); the re-run set will always come
   from p6 and carry p6's status fields.

### 4.3 Output (`arm_b/results_v2_merged.json`)

Same schema as `results_v2.json` plus `hypothesis_status` field. Exactly 17
entries, keyed by `instance_id`.

### 4.4 Reported statistics

`merge_v2_p6.py` must print **three** numbers, not one:

```
Arm B merged (v2 + v2-P6 re-run, n_instances=17):
  mean_acc_valid_only   = X.XXX over K scored  (honest: excludes FAIL_*)
  mean_acc_fail_as_zero = Y.YYY over 17        (pessimistic: FAIL_* scored 0)
  mean_acc_v2_original  = 0.451 over 17         (baseline for comparison)

Family breakdown (valid-only, n_scored in each):
  cellular_automata : 0.XXX over N/6
  particle_system   : 0.XXX over N/5
  pattern_puzzle    : 0.XXX over N/6

FAIL counts: timeout=T no_submit=N exception=E
Arm A reference: 0.343 over 17
```

The **valid-only mean** is the headline number for the scientific claim; the
**fail-as-zero mean** is a conservative lower bound for the reviewer gate.
Reporting only one of them would misrepresent the data.

### 4.5 Skeleton

```python
#!/usr/bin/env python3
"""Merge v2-P4 + v2-P6 re-run into final Arm B results."""
import json, statistics
from pathlib import Path

ROOT = Path("arm_b")
v2  = {r["instance_id"]: r for r in json.loads((ROOT / "results_v2.json").read_text())}
p6  = {r["instance_id"]: r for r in json.loads((ROOT / "results_v2_p6.json").read_text())}

merged = []
for iid in sorted(v2.keys()):
    entry = p6.get(iid, v2[iid])
    merged.append(entry)

(ROOT / "results_v2_merged.json").write_text(json.dumps(merged, indent=2))

scored = [r for r in merged if r.get("accuracy") is not None]
n_fail = len(merged) - len(scored)
mean_valid = statistics.mean(r["accuracy"] for r in scored) if scored else float("nan")
mean_pessim = statistics.mean(r.get("accuracy") or 0.0 for r in merged)

print(f"Arm B merged (n={len(merged)}):")
print(f"  mean_acc_valid_only   = {mean_valid:.3f} over {len(scored)} scored")
print(f"  mean_acc_fail_as_zero = {mean_pessim:.3f} over {len(merged)}")
print(f"  mean_acc_v2_original  = 0.451 over 17")

for fam in ["cellular_automata", "particle_system", "pattern_puzzle"]:
    fs = [r["accuracy"] for r in scored if r["family"] == fam]
    fn = sum(1 for r in merged if r["family"] == fam)
    if fs:
        print(f"  {fam:18s}: {statistics.mean(fs):.3f} over {len(fs)}/{fn}")
    else:
        print(f"  {fam:18s}: (all FAIL)    over 0/{fn}")

statuses = [r.get("hypothesis_status", "submitted" if not r.get("fallback") else "FAIL_LEGACY")
            for r in merged]
print(f"FAIL counts: "
      f"timeout={statuses.count('FAIL_TIMEOUT')} "
      f"no_submit={statuses.count('FAIL_NO_SUBMIT')} "
      f"exception={statuses.count('FAIL_EXCEPTION')}")
```

## 5. Success criteria (for implementer)

1. `arm_b/run_arm_b_v2.py` patched: timeout→900/600, retries→3, identity
   fallback removed, `FAIL_TIMEOUT` / `FAIL_NO_SUBMIT` / `FAIL_EXCEPTION`
   statuses recorded, instance-cap enforced.
2. Re-run the 10 fallback instances (not all 17) — add CLI flag
   `--only world_ca_001,world_ca_002,...` or similar. Output to
   `arm_b/results_v2_p6.json`.
3. `arm_b/merge_v2_p6.py` produces `results_v2_merged.json` and prints all
   three means + family breakdown + fail counts.
4. `arm_b/run_log_v2_p6.txt` contains machine-parseable `[FAIL_*]` lines.
5. If the re-run still produces ≥3 `FAIL_TIMEOUT` instances, **stop and
   escalate** — do not reduce timeouts or reintroduce fallbacks; the hang is
   structural and needs Path B (SDK) as a follow-up.

## 6. Non-goals / out of scope

- Switching to Anthropic SDK (`arm_b/driver.py` style) — deferred pending v2-P6
  re-run outcome.
- Modifying `worlds/`, `harness.py`, or `skill/jump_v2.md`.
- Changing `MAX_TURNS` or scoring rubric.
- Parallelizing instance execution.

## 7. Risk register

| Risk                                            | Mitigation                                      |
|-------------------------------------------------|-------------------------------------------------|
| 900 s × 3 retries × 17 instances = 12.75 h worst-case | `INSTANCE_CAP_SEC=1800` hard-aborts; typical runs <60 s/instance based on v2-P4 OK cases |
| `claude -p` hang is not time-bounded at all     | Instance cap + exponential backoff; escalate if hang rate >30% after fix |
| v2-P6 re-run gets *different* acc for successful instances (stochasticity) | Re-run ONLY the 10 failed instances; do not overwrite the 7 valid v2 results |
| API rate limits during burst                    | 30 s / 60 s backoff spreads retries; sequential run (no parallelism) |
| `valid-only mean` biases toward easy instances if hard ones keep timing out | Report `fail-as-zero` alongside; reviewer can judge |
