# Abductive Jump Benchmark — Scoring Rubric

This rubric governs how both Arm A and Arm B outputs are scored on the abductive
reasoning benchmark. Each task is scored across five axes. **Maximum = 10, Minimum = −1.**

Scores are assigned INDEPENDENTLY per arm, per task, BEFORE any cross-arm comparison.

---

## 1. Scoring Axes

### (a) gold-match (0–3) — Semantic equivalence to the held-out `gold_case`

Does the response propose the same structural/causal insight as the reference
`gold_case` in the task record? Judge on mechanism, not phrasing.

| Score | Criterion |
|-------|-----------|
| 3 | Proposes the same structural insight as gold_case (core mechanism identical, even if phrased differently) |
| 2 | Proposes a closely related hypothesis that captures the key mechanism partially |
| 1 | Partially overlaps with gold_case but misses the central insight |
| 0 | No overlap; proposes something orthogonal or wrong |

### (b) parsimony (0–2) — Single mechanism vs. ad-hoc patches

Does one unifying mechanism do the explanatory work, or is the answer a
patchwork of auxiliary assumptions?

| Score | Criterion |
|-------|-----------|
| 2 | One unifying mechanism explains the result; no auxiliary assumptions needed |
| 1 | Mostly parsimonious but requires one additional assumption beyond the rule |
| 0 | Patchwork of multiple ad-hoc explanations; or circular reasoning |

### (c) unifying_scope (0–2) — Coverage beyond the given Rule+Result pair

Does the proposed Case extend to other phenomena, or is it bespoke to this
single data point?

| Score | Criterion |
|-------|-----------|
| 2 | The proposed Case would explain multiple phenomena outside the given Rule+Result |
| 1 | The Case fits the given pair but barely extends beyond it |
| 0 | The Case is tailored only to this specific Rule+Result and has no broader explanatory scope |

### (d) novelty_vs_retrieval (0–2) — Generated inference vs. name-dropping

Is the hypothesis **constructed** step-by-step from the given Rule+Result, or
is it pulled from memory by naming a theory or discoverer?

| Score | Criterion |
|-------|-----------|
| 2 | The response *constructs* the hypothesis step-by-step, names no theory/discoverer, and the reasoning process is visible |
| 1 | The response arrives at the right answer but the path reads as direct retrieval (e.g., immediately names the theory or discoverer) |
| 0 | The response simply states the historical answer by name with no inferential work shown |

**Automatic (d) = 0** if the output does NOT conform to `eval/output_schema.json`
(no inferential work is visible to score).

### (e) decoy_trap (−1 if triggered) — Penalty for adopting a listed decoy verbatim

Did the response adopt one of the task's `decoy_cases` substantially unchanged?

| Score | Criterion |
|-------|-----------|
| −1 | Response proposes one of the `decoy_cases` from the task record substantially unchanged |
| 0 | No penalty (response avoids decoy traps or proposes a novel alternative) |

---

## 2. Scoring Formula

```
Total = (a) + (b) + (c) + (d) + (e)
Maximum = 10   Minimum = −1
```

---

## 3. Worked Example — task_007 (Aromaticity)

> **Rule:** When carbon atoms form a closed ring, the molecule's reactivity pattern depends on the total number of valence electrons available for sharing between ring atoms. Compounds with certain electron counts show exceptional thermal stability and prefer substitution reactions over addition reactions.
>
> **Result:** A six-carbon compound isolated from coal tar forms a ring, resists addition reactions (unlike alkenes), and yields a single mono-substitution product regardless of which carbon is attacked first, indicating all six carbons are chemically equivalent.

**Gold case:** Planar hexagonal ring with a delocalised 6π aromatic electron
system (cyclic conjugation satisfying Hückel's 4n+2 rule for n=1), in which all
C–C bonds are equivalent (bond order 1.5), explaining both exceptional stability
and electrophilic substitution preference.

### Response A — gold-level

```
## Candidate Survey
- Retrieval candidate: alternating single/double bonds (Kekulé)
- Historical decoy: prismane cage structure (Ladenburg 1869)
- Abductive candidate: all valence electrons delocalized uniformly around the ring →
  bond-order equalisation → stability from electron delocalisation energy

## Chosen Case
The six carbons contribute one p-orbital electron each to a continuous cyclic π system.
All 6π electrons are collectively shared, making all bonds equivalent (order ~1.5) and
preventing localisation of double bonds. This explains substitution preference (the π
density is uniformly distributed; an incoming electrophile displaces H rather than
adding, preserving the delocalised system) and the single mono-substitution product.

## Abductive Justification
- Parsimony: one mechanism — cyclic delocalisation — handles stability + substitution + symmetry.
- Unifying scope: predicts ring current (NMR downfield shift for aromatic H), explains
  thermochemical stability excess, applies to pyridine, naphthalene, thiophene.
- Novelty vs retrieval: constructed the electron-counting argument from the ring geometry
  without naming 'aromaticity' or 'Hückel'.
- Novel prediction: a 4-electron ring (cyclobutadiene) should be *anti-stabilised* —
  the reverse of the pattern.
```

| Axis | Score | Reasoning |
|------|-------|-----------|
| (a) gold-match | 3 | Captures cyclic π delocalisation, bond-order equalisation, and stability — mechanism identical to gold |
| (b) parsimony | 2 | Single mechanism (cyclic delocalisation) explains all three observations |
| (c) unifying_scope | 2 | Predicts ring currents, extends to pyridine/naphthalene/thiophene, and makes a novel prediction about cyclobutadiene |
| (d) novelty_vs_retrieval | 2 | Explicitly avoids the word "aromaticity" and "Hückel"; builds the argument from ring geometry |
| (e) decoy_trap | 0 | Does not adopt the Kekulé alternation or prismane decoys |
| **Total** | **9/10** | |

### Response B — decoy trap

```
The result is best explained by Kekulé's alternating single-double bond structure,
where the three double bonds rapidly oscillate, creating apparent equivalence.
This is aromaticity as described by Faraday and confirmed by X-ray diffraction.
```

| Axis | Score | Reasoning |
|------|-------|-----------|
| (a) gold-match | 1 | Mentions equivalence via oscillation but misses delocalisation; partial overlap |
| (b) parsimony | 1 | Relies on an auxiliary "rapid oscillation" assumption on top of Kekulé structure |
| (c) unifying_scope | 0 | Does not extend to other aromatics, ring currents, or stability excess |
| (d) novelty_vs_retrieval | 0 | Names Kekulé and Faraday; no inferential work shown |
| (e) decoy_trap | −1 | Kekulé bond-shift tautomerism is the listed decoy, adopted essentially verbatim |
| **Total** | **1/10** | |

---

## 4. Aggregate Scoring

After all tasks are scored for an arm, compute:

- **Mean score** across all tasks
- **Count of tasks scoring ≥ 8/10** (near-gold responses)

### Verdict Thresholds

| Condition | Verdict |
|-----------|---------|
| Mean ≥ 6.0/10 **OR** at least 3 tasks ≥ 8/10 | **APPROVE** |
| Mean between 4.0 and 6.0 (and fewer than 3 tasks ≥ 8/10) | **BORDERLINE — request iteration** |
| Mean < 4.0/10 | **REJECT** |

---

## 5. Reviewer Protocol

The reviewer MUST follow this order:

1. **Score Arm A INDEPENDENTLY.** For each task, read only the Arm A output and
   the task's `gold_case` + `decoy_cases`. Assign (a)–(e) and compute the total.
   Do NOT look at Arm B's output or scores yet.
2. **Score Arm B INDEPENDENTLY** in the same manner.
3. **Compare arms.** Compute per-arm mean and ≥8/10 counts. Apply the verdict
   thresholds above.
4. **Write the verdict file** at `verdicts/{reviewer_id}.md` with:
   - A per-task score table (columns: task_id, arm, a, b, c, d, e, total, one-line justification)
   - Per-arm aggregate statistics (mean, median, count ≥ 8, count ≤ 2)
   - Final **APPROVE / REJECT / BORDERLINE** decision per arm
   - A head-to-head comparison section: which arm wins on which axes, and any
     systematic differences observed
   - Any schema-non-conformance notes (triggering automatic (d) = 0)

Reviewer bias note: if the reviewer knows which arm is which (A = baseline, B =
intervention), they must still score tasks in the ORDER specified above and
justify each axis score in one line so their calibration is auditable.
