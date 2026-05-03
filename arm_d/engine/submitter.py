"""
Final gate: re-lint the chosen survivor, sanity-check on train_obs, and
submit to the harness.

The submitter is allowed to fall back through the pool: if the top
survivor fails the final lint or the harness raises during submit, it
tries the next-best candidate. If the entire pool is exhausted,
FAIL_NO_HYPOTHESIS is reported.
"""
from .linter import lint
from .scorer import score_candidate
from . import statuses


def submit_with_fallback(harness, ranked_pool: list[dict], train_obs: list) -> dict:
    """
    Try to submit each ranked candidate in order until one succeeds.

    Returns a dict:
      {"status": "submitted"|"FAIL_NO_HYPOTHESIS",
       "selected": <record dict or None>,
       "test_acc": float or None,
       "tries": int,
       "lint_failures": [reason, ...]}
    """
    lint_failures = []
    tries = 0
    for rec in ranked_pool:
        tries += 1
        ok, reason = lint(rec["source"])
        if not ok:
            lint_failures.append(reason)
            continue
        # Sanity-check: re-score on full train.
        train_acc, _ = score_candidate(rec["source"], train_obs)
        rec["resubmit_train_acc"] = train_acc
        # Attempt submission.
        try:
            result = harness.submit_hypothesis(rec["source"])
            return {
                "status": statuses.SUBMITTED,
                "selected": rec,
                "test_acc": result.get("accuracy"),
                "tries": tries,
                "lint_failures": lint_failures,
            }
        except Exception as e:
            # Submission raised — log and try next.
            lint_failures.append(f"submit_exc:{type(e).__name__}:{str(e)[:60]}")
            continue

    return {
        "status": statuses.FAIL_NO_HYPOTHESIS,
        "selected": None,
        "test_acc": None,
        "tries": tries,
        "lint_failures": lint_failures,
    }
