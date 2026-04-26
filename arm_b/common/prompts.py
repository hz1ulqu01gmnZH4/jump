DEADLINE_NUDGE = """## Deadline policy (harness-enforced)

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
"""
