You are a creative Python programmer. Below is an empirical observation
set: a list of (state, next_state) pairs from an unknown deterministic
rule. Write Python source for `def hidden_rule_fn(state) -> next_state`
that, when run on each `state`, produces the corresponding `next_state`
exactly.

You may NOT call any external API. You may NOT refer to the words
'intervene', 'experiment', 'gather more data', 'novelty_justification',
'abduction_notes', 'refinement_steps', 'candidate_survey'. Your output
is the function source ONLY — no commentary, no markdown explanation
outside the JSON list.

Style: try UNUSUAL but coherent rule structures. Mix bitwise (XOR, AND,
shift) with arithmetic (+, *, mod). Use parity, primality,
perfect-square indices. Try non-obvious neighbourhood shapes (knight,
diagonal-only, alternating-row).

train_obs:
{{TRAIN_OBS_JSON}}

primitive_glossary (informational only — names are arbitrary):
{{GLOSSARY_JSON}}

Output FORMAT (strict): a JSON list of {{M}} objects:
[{"source": "def hidden_rule_fn(state):\n    ..."}, ...]

Investigation seed: {{SEED}}
