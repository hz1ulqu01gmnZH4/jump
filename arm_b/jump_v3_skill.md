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
