#!/usr/bin/env python3
"""
Arm D — breadth-first ideation pipeline.

For each v2 instance:
  1. Generate a candidate pool via engine.mock_generator (and, when
     `--use-claude` is set, append the live Claude pool from
     orchestrator.claude_runner).
  2. Lint the pool; drop syntactic / methodology violators.
  3. Score each survivor on train_obs; rank by (train_acc, -mdl).
  4. If no candidate scores ≥ ε on train, run AST-level mutation +
     branch-tree crossover and re-score.
  5. Submit the best surviving candidate (with fallback through the
     pool on lint or submission errors).

Result schema mirrors arm_b/results_v2_merged.json with an extra
`engine_meta` block. Failure statuses follow the v2-P6 convention:
FAIL_TIMEOUT / FAIL_NO_SUBMIT / FAIL_NO_HYPOTHESIS / FAIL_EXCEPTION.

Usage:
  python arm_d/run_arm_d.py                   # full mock-pool run on all 17
  python arm_d/run_arm_d.py --only world_ca_001 world_pt_001
  python arm_d/run_arm_d.py --use-claude      # P-D2 (NOT IMPLEMENTED in MVP)
"""
import argparse
import json
import statistics
import sys
import time
from pathlib import Path

# Make the project root importable.
sys.path.insert(0, str(Path(__file__).parent.parent))

from harness import WorldHarness  # noqa: E402
from worlds.gen import generate_all  # noqa: E402

from arm_d.engine import (  # noqa: E402
    crossover, linter, mock_generator, mutator,
    scorer, selector, statuses, submitter,
)


INSTANCE_HARD_CAP_S = 1800
EPSILON_FORCE_REFINE = 0.0  # no candidate strictly above this on train ⇒ trigger mutation pass
TOPK_FOR_MUTATION = 8
TOPK_FOR_CROSSOVER = 6


def run_arm_d_on_instance(instance: dict, use_claude: bool = False, verbose: bool = True) -> dict:
    """Run the full ideation pipeline on a single instance."""
    t_start = time.monotonic()
    harness = WorldHarness(instance)

    # Step 0: pull train_obs from the harness (the only allowed pre-submit call).
    full_train = harness.get_train_obs()

    # Hold out the last 2 train_obs as a tiebreak validation slice.
    train_for_scoring, val_obs = selector.split_train_val(full_train, n_val=2)

    # Step 1: generate the pool.
    pool = list(mock_generator.generate_pool(instance))
    n_mock = len(pool)
    n_claude = 0
    if use_claude:
        try:
            from arm_d.orchestrator import claude_runner
            claude_pool = claude_runner.generate_pool_via_claude(instance)
            pool.extend(claude_pool)
            n_claude = len(claude_pool)
        except Exception as e:
            if verbose:
                print(f"  [claude generator failed: {e}]", flush=True)

    # Step 2: lint
    pool, dropped = linter.lint_with_drop(pool)
    n_lint_rejected = len(dropped)
    if not pool:
        # Entire pool failed lint — record FAIL_NO_HYPOTHESIS.
        return _build_result(
            instance, status=statuses.FAIL_NO_HYPOTHESIS,
            test_acc=None, source=None, t_start=t_start,
            engine_meta={
                "pool_size": n_mock + n_claude,
                "n_mock": n_mock,
                "n_claude": n_claude,
                "n_lint_rejected": n_lint_rejected,
                "best_train_acc": 0.0,
                "n_at_top": 0,
                "mutation_pass_used": False,
                "crossover_pass_used": False,
                "selected_persona": "none",
            },
        )

    # Step 3: score on train_for_scoring
    records = scorer.score_pool(pool, train_for_scoring)
    best_train_acc = records[0]["train_acc"]
    n_at_top = sum(1 for r in records if abs(r["train_acc"] - best_train_acc) < 1e-9)

    mutation_used = False
    crossover_used = False

    # Step 4: if best ≤ EPSILON_FORCE_REFINE, trigger mutation + crossover refinement.
    if best_train_acc <= EPSILON_FORCE_REFINE:
        # Mutation pass on top-K
        topk_sources = [r["source"] for r in records[:TOPK_FOR_MUTATION]]
        mutated = mutator.mutate_pool(topk_sources, per_source=12, max_total=200)
        if mutated:
            mutation_used = True
            mutated, _ = linter.lint_with_drop(mutated)
            extra_records = scorer.score_pool(mutated, train_for_scoring)
            records = sorted(records + extra_records,
                             key=lambda r: (-r["train_acc"], r["mdl"], r["hash"]))
            best_train_acc = records[0]["train_acc"]
            n_at_top = sum(1 for r in records if abs(r["train_acc"] - best_train_acc) < 1e-9)

        # Crossover pass on top-K (post-mutation if any).
        topk_sources = [r["source"] for r in records[:TOPK_FOR_CROSSOVER]]
        crossed = crossover.crossover_pool(topk_sources, max_total=120)
        if crossed:
            crossover_used = True
            crossed, _ = linter.lint_with_drop(crossed)
            extra_records = scorer.score_pool(crossed, train_for_scoring)
            records = sorted(records + extra_records,
                             key=lambda r: (-r["train_acc"], r["mdl"], r["hash"]))
            best_train_acc = records[0]["train_acc"]
            n_at_top = sum(1 for r in records if abs(r["train_acc"] - best_train_acc) < 1e-9)

    # Step 5: select survivor (tiebreak via val_obs if multiple at top)
    survivor = selector.select(records, val_obs)
    selected_persona = "mock_pool"  # MVP marker; live generator would set per-call

    # Hard cap
    if time.monotonic() - t_start > INSTANCE_HARD_CAP_S:
        return _build_result(
            instance, status=statuses.FAIL_TIMEOUT,
            test_acc=None, source=None, t_start=t_start,
            engine_meta={
                "pool_size": len(pool) + (len(mutated) if mutation_used else 0),
                "n_mock": n_mock,
                "n_claude": n_claude,
                "n_lint_rejected": n_lint_rejected,
                "best_train_acc": best_train_acc,
                "n_at_top": n_at_top,
                "mutation_pass_used": mutation_used,
                "crossover_pass_used": crossover_used,
                "selected_persona": "timeout",
            },
        )

    # Step 6: submit (with fallback through ranked pool).
    submit_result = submitter.submit_with_fallback(harness, records, full_train)
    test_acc = submit_result["test_acc"]
    status = submit_result["status"]
    final_source = submit_result["selected"]["source"] if submit_result["selected"] else None

    return _build_result(
        instance, status=status,
        test_acc=test_acc, source=final_source, t_start=t_start,
        engine_meta={
            "pool_size": len(records),
            "n_mock": n_mock,
            "n_claude": n_claude,
            "n_lint_rejected": n_lint_rejected,
            "best_train_acc": best_train_acc,
            "n_at_top": n_at_top,
            "mutation_pass_used": mutation_used,
            "crossover_pass_used": crossover_used,
            "selected_persona": selected_persona,
            "selected_train_acc": survivor["train_acc"],
            "selected_val_acc": survivor.get("val_acc"),
            "selected_mdl": survivor["mdl"],
            "submit_tries": submit_result["tries"],
        },
    )


def _build_result(instance: dict, *, status: str, test_acc, source,
                  t_start: float, engine_meta: dict) -> dict:
    wall_s = time.monotonic() - t_start
    engine_meta["wall_s"] = round(wall_s, 3)
    return {
        "instance_id": instance["id"],
        "family": instance["family"],
        "difficulty": instance["difficulty"],
        "accuracy": test_acc,
        "hypothesis_source": source,
        "hypothesis_status": status,
        "n_interventions": 0,    # Arm D MVP makes ZERO intervene calls
        "n_turns": 1,            # one ideation pass per instance
        "fallback": status in statuses.ALL_FAIL,
        "engine_meta": engine_meta,
    }


def main():
    parser = argparse.ArgumentParser(description="Arm D ideation runner")
    parser.add_argument(
        "--only", nargs="+", metavar="INSTANCE_ID",
        help="Run only the specified instance IDs (space-separated)."
    )
    parser.add_argument(
        "--use-claude", action="store_true",
        help="Append a live `claude -p` candidate pool (P-D2; not in MVP)."
    )
    parser.add_argument(
        "--out", type=str, default=None,
        help="Override output path (defaults to arm_d/results_v2.json or _p<run>.json with --only)."
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Suppress per-instance progress lines."
    )
    args = parser.parse_args()

    all_instances = generate_all()
    if args.only:
        only_set = set(args.only)
        instances = [i for i in all_instances if i["id"] in only_set]
        missing = only_set - {i["id"] for i in instances}
        if missing:
            raise ValueError(f"Unknown instance IDs: {sorted(missing)}")
        default_out = Path("arm_d/results_v2_partial.json")
    else:
        instances = all_instances
        default_out = Path("arm_d/results_v2.json")

    output_path = Path(args.out) if args.out else default_out
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(
        f"Arm D (mock generator{', + claude' if args.use_claude else ''}) — "
        f"{len(instances)} instances, output={output_path}"
    )

    results = []
    for inst in instances:
        if not args.quiet:
            print(f"  {inst['id']} ({inst['family']}, {inst['difficulty']})...",
                  end="", flush=True)
        try:
            r = run_arm_d_on_instance(inst, use_claude=args.use_claude, verbose=not args.quiet)
        except Exception as e:
            r = _build_result(
                inst, status=statuses.FAIL_EXCEPTION,
                test_acc=None, source=None, t_start=time.monotonic(),
                engine_meta={"error": f"{type(e).__name__}:{str(e)[:120]}"},
            )

        results.append(r)
        if not args.quiet:
            acc_str = f"{r['accuracy']:.3f}" if r["accuracy"] is not None else "None"
            meta = r.get("engine_meta", {})
            print(
                f" acc={acc_str} pool={meta.get('pool_size','?')} "
                f"top_train={meta.get('best_train_acc','?'):.2f} "
                f"mut={meta.get('mutation_pass_used', False)} "
                f"xover={meta.get('crossover_pass_used', False)} "
                f"wall={meta.get('wall_s','?'):.2f}s [{r['hypothesis_status']}]",
                flush=True,
            )

    output_path.write_text(json.dumps(results, indent=2))
    print(f"\nResults written to {output_path} ({len(results)} entries)")

    # Summary statistics
    scored = [r for r in results if r.get("accuracy") is not None]
    n_excluded = len(results) - len(scored)
    mean_acc = statistics.mean(r["accuracy"] for r in scored) if scored else float("nan")
    print(
        f"\n=== Arm D mean accuracy: {mean_acc:.3f} over {len(scored)}/{len(results)} "
        f"(excluded {n_excluded} FAIL_*) ==="
    )
    if not args.only:
        print("Reference: C-random=0.010, C-induce=0.127, C-retrieval=1.000, "
              "Arm A=0.343, Arm B=0.451 (with timeout caveats).")
        for fam in ["cellular_automata", "particle_system", "pattern_puzzle"]:
            fam_results = [r for r in results if r["family"] == fam]
            fam_scored = [r["accuracy"] for r in fam_results if r.get("accuracy") is not None]
            if fam_scored:
                print(f"  {fam}: Arm D={statistics.mean(fam_scored):.3f} "
                      f"(n={len(fam_scored)}/{len(fam_results)})")
            else:
                print(f"  {fam}: Arm D=(all FAIL)")

    statuses_count = [r.get("hypothesis_status") for r in results]
    print(
        f"FAIL counts: timeout={statuses_count.count('FAIL_TIMEOUT')} "
        f"no_submit={statuses_count.count('FAIL_NO_SUBMIT')} "
        f"no_hypothesis={statuses_count.count('FAIL_NO_HYPOTHESIS')} "
        f"exception={statuses_count.count('FAIL_EXCEPTION')}"
    )


if __name__ == "__main__":
    main()
