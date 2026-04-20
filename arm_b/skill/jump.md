---
name: jump
description: Peircean abductive reasoning — given Rule + Result, generate a genuinely novel Case (explanatory axiom). Produces schema-conforming JSON output for the jump benchmark.
version: "1.0"
---

# jump — Abductive Reasoning Skill

You perform **Peircean abduction**: given a Rule (general principle) and a Result (observation), generate the most parsimonious novel **Case** (explanatory axiom) that makes Result follow from Rule.

## Protocol

### Step 1 — Candidate Survey
List 3-5 candidate Cases:
- **retrieval**: the most obvious textbook answer
- **historical_decoy**: a serious historical competitor that was proposed but rejected
- **abductive**: a structurally novel hypothesis you are generating as a fresh inference

### Step 2 — Abductive Filter
Score each candidate:
1. Parsimony (1–5): fewest new entities?
2. Unifying scope (1–5): explains phenomena BEYOND this Rule+Result?
3. Novelty (1–5): structural leap vs retrieval?
4. Testability (1–5): generates new predictions?

### Step 3 — Output
Produce ONLY a JSON object. No prose. No discoverer names in hypothesis.

{
  "task_id": "<from input>",
  "hypothesis": "<1-3 sentences: novel Case. NO discoverer/theory name>",
  "mechanism": "<1-3 sentences: causal path Case+Rule+Result>",
  "observations_explained": ["<phenomenon beyond the pair>", "..."],
  "parsimony_justification": "<why simplest; what it avoids>",
  "novelty_justification": "<what inferential step taken vs retrieved>",
  "candidate_survey": [
    {"label": "retrieval", "description": "..."},
    {"label": "historical_decoy", "description": "..."},
    {"label": "abductive", "description": "..."}
  ]
}

## Input expected
task_id: <id>
Rule: <general law>
Result: <observation>
