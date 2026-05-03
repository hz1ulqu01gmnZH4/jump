"""
Identity-baseline submitters for non-MVP families (P-C2 / P-C3).

These return `def hidden_rule_fn(state): return state` so the harness call
shape is exercised end-to-end while the real solvers are deferred.
Marked with hypothesis_status="STUB_NOT_IMPLEMENTED" by run_arm_c.py.
"""

IDENTITY_SRC = "def hidden_rule_fn(state):\n    return state\n"


def stub_submit(harness) -> None:
    harness.submit_hypothesis(IDENTITY_SRC)
