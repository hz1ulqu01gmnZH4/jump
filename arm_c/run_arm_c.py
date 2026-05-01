#!/usr/bin/env python3
"""
Arm C top-level driver.

Iterates v2 instances, dispatches family-specific solver, writes
results_v2.json. CA family uses the symbolic engine; PT and SEQ use
identity stubs (P-C2 / P-C3 deferred per plan_memo.md §1.2).
"""

from __future__ import annotations
import argparse, json, statistics, sys, time, traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from harness import WorldHarness  # noqa: E402
from worlds.gen import generate_all  # noqa: E402

from engine.ca_solver import solve_ca  # noqa: E402
from engine.stub_solvers import IDENTITY_SRC  # noqa: E402


PER_INSTANCE_CAP_SEC = 1800


def run_one(instance: dict) -> dict:
    """Solve one instance; never silent-fallback."""
    harness = WorldHarness(instance)
    fam = instance["family"]
    t0 = time.monotonic()
    engine_meta: dict = {}
    status = "submitted"
    err_repr = None
    try:
        if fam == "cellular_automata":
            engine_meta = solve_ca(harness)
        else:
            # PT / SEQ stubs (P-C2 / P-C3). Submit identity baseline so
            # the harness call shape is exercised end-to-end. Marked.
            harness.submit_hypothesis(IDENTITY_SRC)
            status = "STUB_NOT_IMPLEMENTED"
    except Exception as e:
        err_repr = repr(e)
        status = "FAIL_EXCEPTION"
        traceback.print_exc()

    elapsed = time.monotonic() - t0
    log = harness.get_log()
    subs = [e for e in log if e["tool"] == "submit_hypothesis"]
    interventions = sum(1 for e in log if e["tool"] == "intervene")

    if subs:
        accuracy = subs[-1]["accuracy"]
        hypothesis_source = subs[-1]["hypothesis_source"]
    else:
        accuracy = None
        hypothesis_source = None
        if status == "submitted":
            status = "FAIL_NO_SUBMIT"

    out = {
        "instance_id": instance["id"],
        "family": fam,
        "difficulty": instance["difficulty"],
        "accuracy": accuracy,
        "hypothesis_source": hypothesis_source,
        "hypothesis_status": status,
        "n_interventions": interventions,
        "n_turns": 1,  # MVP is single-turn (no iterative orchestrator).
        "fallback": status not in ("submitted",),
        "elapsed_s": round(elapsed, 3),
    }
    if err_repr:
        out["error"] = err_repr
    if engine_meta:
        out["engine_meta"] = engine_meta
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="",
                    help="comma-separated instance_ids to run (default: all)")
    ap.add_argument("--out", default="arm_c/results_v2.json",
                    help="output path (relative to repo root)")
    args = ap.parse_args()

    only = set(s.strip() for s in args.only.split(",") if s.strip())
    instances = generate_all()
    if only:
        instances = [i for i in instances if i["id"] in only]
        if not instances:
            print(f"No instances matched --only={args.only}", file=sys.stderr)
            return 2

    results: list[dict] = []
    for inst in instances:
        print(f"[run] {inst['id']:18s} fam={inst['family']:18s} "
              f"diff={inst['difficulty']:6s}", end="  ", flush=True)
        r = run_one(inst)
        acc_str = f"{r['accuracy']:.3f}" if r["accuracy"] is not None else "  -  "
        print(f"acc={acc_str}  status={r['hypothesis_status']:22s} "
              f"t={r['elapsed_s']:.2f}s", flush=True)
        results.append(r)

    out_path = REPO_ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))

    # Summary stats.
    scored = [r for r in results if r["accuracy"] is not None
              and r["hypothesis_status"] != "STUB_NOT_IMPLEMENTED"]
    stubs = [r for r in results if r["hypothesis_status"] == "STUB_NOT_IMPLEMENTED"]
    fails = [r for r in results if r["hypothesis_status"].startswith("FAIL")]

    print()
    print("=" * 70)
    print(f"Arm C — {len(results)} instances")
    if scored:
        m_valid = statistics.mean(r["accuracy"] for r in scored)
        print(f"  mean_acc_valid_only   = {m_valid:.3f} over {len(scored)} scored "
              f"(excludes {len(stubs)} STUB and {len(fails)} FAIL)")
    m_pessim = statistics.mean(r["accuracy"] or 0.0 for r in results)
    print(f"  mean_acc_fail_as_zero = {m_pessim:.3f} over {len(results)} "
          f"(STUB/FAIL count as 0)")
    for fam in ("cellular_automata", "particle_system", "pattern_puzzle"):
        fr = [r for r in results if r["family"] == fam]
        fr_scored = [r for r in fr if r["accuracy"] is not None
                     and r["hypothesis_status"] != "STUB_NOT_IMPLEMENTED"]
        if fr_scored:
            print(f"  {fam:18s}: {statistics.mean(r['accuracy'] for r in fr_scored):.3f} "
                  f"over {len(fr_scored)}/{len(fr)} (engine-solved)")
        else:
            print(f"  {fam:18s}: stub/fail-only over 0/{len(fr)}")
    print(f"  STUB count = {len(stubs)}   FAIL count = {len(fails)}")
    print(f"  results -> {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
