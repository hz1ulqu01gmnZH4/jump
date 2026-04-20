# jump: Empirical Test of LLM Abductive Reasoning

## Citation

Zahavy, T. "LLMs can't jump." PhilSci Archive 28024 (Jan 2026). https://philsci-archive.pitt.edu/28024/

## What is Abduction?

Peirce distinguished three forms of inference. Deduction derives necessary consequences from axioms. Induction generalises from observations to probable rules. **Abduction** (Peirce's "retroduction") is the third: given a Rule and a surprising Result, abduction proposes the *Case* — a novel explanatory hypothesis that, if true, would make the Result follow from the Rule. It is the logic of discovery, not of justification. The Case is not derivable from the evidence; it is a creative leap, the origin of new scientific ideas.

Zahavy (2026) claims that current LLMs cannot perform genuine abduction — they pattern-match and recombine known facts rather than generating truly novel explanatory axioms. This project empirically tests that claim across two evaluation arms.

## Arms

| Arm | System | Model | Method |
|-----|--------|-------|--------|
| A | llama-server (local) | Qwen3.6-35B-A3B-MXFP4_MOE | Direct prompt via OpenAI-compatible API |
| B | Claude Code skill | claude-sonnet-4-6 | Structured skill with evaluation harness |

## How to Run

**Arm A:**
```bash
cd arm_a
uv run python driver.py --eval ../eval/tasks.jsonl --out ../results/arm_a/
```

**Arm B:**
```bash
claude-code --skill jump
```

## Status

| Component | Status |
|-----------|--------|
| P0: Scaffold | ✓ done |
| P1: Eval tasks | pending |
| P2: Arm A driver | pending |
| P3: Arm B skill | pending |
| P4: Verdicts / scoring | pending |
