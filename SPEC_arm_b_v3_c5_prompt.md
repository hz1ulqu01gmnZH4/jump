# SPEC — Arm B v3 C5 prompt redesign

Eliminates the skill/nudge contradiction that produced FAIL_STUCK_REASONING in
every C₂–C₄ smoke run. Replaces the absolute "do not submit" prohibition with a
goal-statement consistent with the harness's deadline policy, makes the
first-action probe unconditional, and adds minimal worked `intervene` call
shapes — one per world family — without leaking rule structure.

Affected files (implementer scope, not this designer's):
- `arm_b/jump_v3_skill.md`
- `arm_b/run_arm_b_v3.py` (DEADLINE_NUDGE constant only)

---

## §1 — Contradiction fix

**Chosen replacement** for the skill body's final line:

> ~~Do not guess. Do not submit until interventions support the rule.~~
> **Your goal is the most accurate hypothesis you can produce, not silence.**

**Justification.** The old line is an *absolute* prohibition ("Do not submit
until…"). When `n_interventions=0`, the model interpreted the nudge's
conditional "after a handful of interventions… submit anyway" as inapplicable
and fell back to the prohibition, producing analysis-only stalls.

The new line preserves the abductive intent (chase the right rule, do not
guess) but expresses it as a **goal**, not a gate. It is consistent with the
NUDGE's first-action mandate (you must probe before you can submit) and with
its "weak submission > silence" rule. No clause in the new tail conflicts with
any clause in the new NUDGE.

The two alternative options were rejected because:
- "Prefer probing to guessing; submit your best-supported hypothesis before
  the session ends." duplicates information that now lives in the NUDGE
  (deadline + submission-on-uncertainty), increasing length without payoff.
- Removing the line entirely silently drops the abductive framing; the model
  could read the workflow as a tool-call ritual with no quality bar.

**Diff (skill body, last line):**

```diff
- Do not guess. Do not submit until interventions support the rule.
+ Your goal is the most accurate hypothesis you can produce, not silence.
```

---

## §2 — First-action mandate

**Placement.** Goes in `DEADLINE_NUDGE` (harness-enforced policy is the right
section for hard requirements). It is the **first bullet** of the new NUDGE,
ensuring it is read before the conditional submission rule.

**Exact new text** (quote-ready, replaces the previous bullet about "at least
one intervention before `submit_hypothesis`"):

> - After `get_train_obs`, your FIRST subsequent action MUST be a call to
>   `intervene`. Do not finalize, submit, or continue internal analysis before
>   making that probe.

**How it replaces the conditional clause.** The old bullet read:

> If, after a handful of interventions, your confidence is low, submit your
> best current hypothesis anyway.

The phrase "after a handful of interventions" is removed. The replacement
bullet (see §4, bullet 3) reads:

> After your first `intervene`, submit if you have a supported hypothesis or
> continue probing. A weak submission with uncertainty noted in a Python
> comment is strictly better than silence.

This makes the submission-under-uncertainty rule fire immediately after the
first probe — no "handful" gating, no implicit licence to keep analysing.

---

## §3 — Worked examples

Three minimal `intervene` call shapes, one per family. The values shown are
placeholders chosen to exercise the field set, not to recommend a probe.

| Family | Example call (form only) |
|--------|--------------------------|
| CA  | `{"action": "set_cell",    "state": <any train state>, "row": 0, "col": 0, "value": "<any glossary symbol>"}` |
| PT  | `{"action": "spawn",       "state": <any train state>, "type": "<any glossary type>", "x": 0, "y": 0}` |
| SEQ | `{"action": "set_element", "state": <any train state>, "index": 0, "value": <any glossary value>}` |

All three actions (`set_cell`, `spawn`, `set_element`) come from
`intervention_api` entries the model already receives via `get_train_obs`;
naming them in the prompt is not a leak.

**Per-example leak audit:**

- **CA — `set_cell`:** uses fields `row`, `col`, `value` (all listed as safe).
  No tokens from {diagonal, neighbor, count, parity, threshold, totalistic,
  majority, distance, velocity, force, mass}. No mention of neighborhoods,
  cell adjacency, or state counting. **No leak tokens found.**
- **PT — `spawn`:** uses fields `type`, `x`, `y`. The forbidden PT action
  `set_velocity` is **not** used (avoiding the token "velocity"). No mention
  of particle interactions or collision. **No leak tokens found.**
- **SEQ — `set_element`:** uses fields `index`, `value`. The action
  `apply_perturbation` (which references modular arithmetic in its own
  description) is **not** used, so no modular hint is introduced.
  **No leak tokens found.**

All placeholders (`<any train state>`, `<any glossary symbol>`,
`<any glossary type>`, `<any glossary value>`) reference *categories* of
values the model already has from `get_train_obs`. No specific cell value,
state, or vocabulary item from `world_ca_001` (or any other instance) is
named.

---

## §4 — Updated DEADLINE_NUDGE (full text, quote-ready)

Paste this verbatim as the value of `DEADLINE_NUDGE` in
`arm_b/run_arm_b_v3.py` (replacing the current triple-quoted string). 11
content lines, ≤14.

```text
## Deadline policy (harness-enforced)

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
```

---

## §5 — Updated skill body tail (full text, quote-ready)

Replaces the current `## Abductive principles` section and the final
prohibition line in `arm_b/jump_v3_skill.md` (everything from line 34 to the
end of file). Net change: **+5 lines** vs. current (≤+8 cap).

```markdown
## Abductive principles

- The best rule explains ALL training observations, not just most.
- The best rule predicts intervention results correctly.
- Prefer the simplest rule that fits (Occam).
- Use the world's invented vocabulary structurally, not as labels.

Your goal is the most accurate hypothesis you can produce, not silence.

## Worked `intervene` call shapes (illustrative form, not probe strategy)
- CA:  {"action": "set_cell", "state": <any train state>, "row": 0, "col": 0, "value": "<any glossary symbol>"}
- PT:  {"action": "spawn", "state": <any train state>, "type": "<any glossary type>", "x": 0, "y": 0}
- SEQ: {"action": "set_element", "state": <any train state>, "index": 0, "value": <any glossary value>}
```

Worked examples section: **4 lines total** (header + 3 bullets) — within the
≤6-line cap.

---

## §6 — Reviewer checklist

For an opus reviewer to APPROVE/REQUEST_CHANGES.

1. **Contradiction check.** Read §5 (new skill tail) and §4 (new NUDGE) back
   to back. Confirm: no clause in the skill tail forbids what the NUDGE
   permits, and no clause in the NUDGE permits what the skill tail forbids.
   Specifically, the old "Do not submit until interventions support the rule"
   line is gone, and nothing replacing it acts as an absolute submission
   gate. ✅ if consistent.
2. **First-action mandate present and unconditional?** §4 bullet 1 must
   require a call to `intervene` immediately after `get_train_obs` with no
   "after N", "if confidence", "when ready", or other conditional phrasing.
   ✅ if unconditional.
3. **Three worked examples?** §5 must contain exactly three `intervene` call
   shapes labeled `CA`, `PT`, `SEQ`. ✅ if all three present and each shows a
   valid action name from that family's `intervention_api`.
4. **Leak audit.** Grep §3, §4, §5 for any of:
   `diagonal`, `neighbor`, `count`, `parity`, `threshold`, `totalistic`,
   `majority`, `distance`, `velocity`, `force`, `mass`, `synchronous`,
   `toroidal`, `diagonal-tetrad`, `d_z`, `d_p`, `d_q`. Also check no example
   uses a specific symbol/value/state drawn from a known instance. ✅ if all
   absent.
5. **Length budget.**
   - DEADLINE_NUDGE (§4): count content lines (excluding the surrounding
     triple-quote markers). Must be ≤14. (Current draft: 11.)
   - Worked examples section (§5, the `## Worked …` block): must be ≤6 lines.
     (Current draft: 4.)
   - Skill body net change vs. current: must be ≤ +8 lines. (Current draft:
     +5: removed 1 prohibition line; added 1 goal line + 1 blank + 1 header
     + 3 example bullets = +6 added, net +5.)
   ✅ if all three within budget.

If items 1–5 all pass, APPROVE. Otherwise REQUEST_CHANGES naming the
failing item.
