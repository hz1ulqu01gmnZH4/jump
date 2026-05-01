"""
Arm D linter — methodology gate for candidate Python sources.

Rejects candidates that:
  - Fail to compile.
  - Do not define `hidden_rule_fn` at module scope.
  - Import banned modules.
  - Reference reasoning-prose / abductive-self-assessment field names
    (the v2 §1.4 ban on `novelty_justification` / `abduction_notes` /
    `refinement_steps` / `interaction_log`).
  - Reference any v2 intervention-API action name as a string literal
    (preserves the "ideation, not abduction" character; the engine in
    MVP does not call intervene at all, but this gate also stops
    smuggled abductive reasoning if the live generator drifts).
  - Exceed the bloat cap (200 lines or 6000 chars).

Lint failure is non-fatal at the pool stage (drop and continue). It is
fatal at the submitter stage; the submitter retries the next-best
surviving candidate before declaring FAIL_NO_HYPOTHESIS.
"""
import ast


ALLOWED_IMPORTS = {"copy", "math", "itertools", "functools"}

# v2 §1.4 banned field names (the prose-reasoning trap from v1)
BANNED_PROSE_TOKENS = {
    "novelty_justification",
    "abduction_notes",
    "refinement_steps",
    "interaction_log",
    "candidate_survey",
    "parsimony_justification",
}

# v2 §3 intervention-API action names — Arm D MVP must not reference these
BANNED_INTERVENTION_TOKENS = {
    "set_cell",
    "flip_row",
    "inject_pattern",
    "spawn",
    "remove_particle",  # `remove` itself is too common in Python; we forbid the v2-specific form
    "set_velocity",
    "change_type",
    "set_element",
    "apply_perturbation",
    "intervene",
}

MAX_LINES = 200
MAX_CHARS = 6000


def lint(source: str) -> tuple[bool, str]:
    """
    Validate a candidate hypothesis source.

    Returns (ok, reason). When ok is False, reason is a short tag
    suitable for logging (no newlines).
    """
    if not isinstance(source, str) or not source.strip():
        return False, "empty"

    if len(source) > MAX_CHARS:
        return False, f"size_chars>{MAX_CHARS}"
    if source.count("\n") + 1 > MAX_LINES:
        return False, f"size_lines>{MAX_LINES}"

    # Compile gate
    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        return False, f"syntax:{e.msg[:40]}"
    try:
        compile(source, "<lint>", "exec")
    except Exception as e:
        return False, f"compile:{type(e).__name__}"

    # hidden_rule_fn must be defined at module scope.
    has_fn = False
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "hidden_rule_fn":
            has_fn = True
            break
    if not has_fn:
        return False, "no_hidden_rule_fn"

    # Import gate.
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for n in node.names:
                root = n.name.split(".")[0]
                if root not in ALLOWED_IMPORTS:
                    return False, f"import:{root}"
        elif isinstance(node, ast.ImportFrom):
            root = (node.module or "").split(".")[0]
            if root not in ALLOWED_IMPORTS:
                return False, f"importfrom:{root}"

    # Prose / reasoning-field token gate (string literals + identifier names).
    text_lower = source.lower()
    for tok in BANNED_PROSE_TOKENS:
        if tok in text_lower:
            return False, f"prose:{tok}"

    # Intervention-API token gate. Walk strings + identifiers.
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            v = node.value
            for tok in BANNED_INTERVENTION_TOKENS:
                if tok in v:
                    return False, f"intervention_str:{tok}"
        elif isinstance(node, ast.Name):
            if node.id in BANNED_INTERVENTION_TOKENS:
                return False, f"intervention_id:{node.id}"
        elif isinstance(node, ast.Attribute):
            if node.attr in BANNED_INTERVENTION_TOKENS:
                return False, f"intervention_attr:{node.attr}"

    return True, "ok"


def lint_with_drop(sources: list[str]) -> tuple[list[str], list[tuple[str, str]]]:
    """
    Convenience: filter a pool, returning (kept, dropped) where dropped is
    a list of (truncated_src_prefix, reason) for diagnostics.
    """
    kept, dropped = [], []
    for src in sources:
        ok, reason = lint(src)
        if ok:
            kept.append(src)
        else:
            dropped.append((src[:60].replace("\n", "\\n"), reason))
    return kept, dropped
