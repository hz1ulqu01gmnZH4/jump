"""
Three mandatory control arms for the v2 abductive-jump benchmark.
  C-random:    random executable rule (floor baseline)
  C-induce:    symbolic induction via template grid search (non-abductive learner)
  C-retrieval: agent given gold rule description before interacting (no abduction needed)
"""
import random, itertools, copy
from worlds.gen import run_instance


# ── C-random ─────────────────────────────────────────────────────────
def c_random_agent(harness, seed: int = 0) -> None:
    """Submits a random rule from a small fixed distribution."""
    fam = harness.instance["family"]
    train = harness.get_train_obs()
    sample_state = train[0]["state"]

    if fam == "cellular_automata":
        # Derive actual cell states from train_obs (primitive_glossary keys are labels, not states)
        if isinstance(sample_state[0], list):
            states = sorted({v for row in sample_state for v in row})
        else:
            states = sorted(set(sample_state))
        src = (
            "def hidden_rule_fn(state):\n"
            f"    states = {states!r}\n"
            f"    rng = __import__('random').Random({seed})\n"
            "    if isinstance(state[0], list):\n"
            "        return [[rng.choice(states) for _ in row] for row in state]\n"
            "    return [rng.choice(states) for _ in state]\n"
        )
    elif fam == "particle_system":
        src = (
            "def hidden_rule_fn(state):\n"
            "    import copy, random\n"
            f"    rng = random.Random({seed})\n"
            "    result = copy.deepcopy(state)\n"
            "    for p in result:\n"
            "        p['x'] = (p['x'] + rng.randint(-1,1)) % 8\n"
            "        p['y'] = (p['y'] + rng.randint(-1,1)) % 8\n"
            "    return result\n"
        )
    else:  # pattern_puzzle — append a random element
        src = (
            "def hidden_rule_fn(state):\n"
            f"    rng = __import__('random').Random({seed})\n"
            "    m = max(state) + 1 if state else 7\n"
            "    return list(state) + [rng.randint(0, m - 1)]\n"
        )
    harness.submit_hypothesis(src)


# ── C-induce ─────────────────────────────────────────────────────────
def c_induce_agent(harness) -> None:
    """
    Lightweight symbolic regression: tries a template library of parameterised
    rules and picks the one with best accuracy on train_obs.
    Not a full SR solver — tests ~200 templates in O(n) time.
    """
    train = harness.get_train_obs()
    fam = harness.instance["family"]
    best_src, best_acc = None, -1.0

    if fam == "pattern_puzzle":
        # Template: (a*s[-1] + b*s[-2] + c) % m — appends one new element
        for a, b, c, m in itertools.product([0, 1, 2], [0, 1, 2], [0, 1, 2, 3], [5, 7, 8, 11, 13, 16]):
            src = (
                f"def hidden_rule_fn(state):\n"
                f"    v1 = state[-1] if len(state) >= 1 else 0\n"
                f"    v2 = state[-2] if len(state) >= 2 else 0\n"
                f"    return list(state) + [({a}*v1 + {b}*v2 + {c}) % {m}]\n"
            )
            acc = _eval_on_train(src, train)
            if acc > best_acc:
                best_acc, best_src = acc, src

    elif fam == "cellular_automata":
        # Derive actual integer cell states from train_obs
        sample = train[0]["state"]
        if isinstance(sample[0], list):
            all_vals = sorted({v for row in sample for v in row})
        else:
            all_vals = sorted(set(sample))

        if len(all_vals) >= 2:
            for threshold in range(1, 5):
                src = (
                    f"def hidden_rule_fn(state):\n"
                    f"    vals = {all_vals!r}\n"
                    f"    if not isinstance(state[0], list):\n"
                    f"        state = [list(state)]\n"
                    f"    rows = len(state); cols = len(state[0])\n"
                    f"    result = []\n"
                    f"    for r in range(rows):\n"
                    f"        row = []\n"
                    f"        for c in range(cols):\n"
                    f"            nbrs = [state[(r+dr)%rows][(c+dc)%cols] for dr in [-1,0,1] for dc in [-1,0,1] if (dr,dc)!=(0,0)]\n"
                    f"            cnt = sum(1 for n in nbrs if n==vals[0])\n"
                    f"            row.append(vals[0] if cnt > {threshold} else vals[1 % len(vals)])\n"
                    f"        result.append(row)\n"
                    f"    return result\n"
                )
                acc = _eval_on_train(src, train)
                if acc > best_acc:
                    best_acc, best_src = acc, src

    elif fam == "particle_system":
        for dx, dy in itertools.product([-1, 0, 1], [-1, 0, 1]):
            src = (
                f"def hidden_rule_fn(state):\n"
                f"    import copy\n"
                f"    result = copy.deepcopy(state)\n"
                f"    for p in result:\n"
                f"        p['x'] = (p['x'] + {dx}) % 8\n"
                f"        p['y'] = (p['y'] + {dy}) % 8\n"
                f"    return result\n"
            )
            acc = _eval_on_train(src, train)
            if acc > best_acc:
                best_acc, best_src = acc, src

    if best_src is None:
        best_src = "def hidden_rule_fn(state): return state"
    harness.submit_hypothesis(best_src)


def _eval_on_train(src: str, train_obs: list) -> float:
    try:
        ns = {}
        exec(src, ns)
        fn = ns["hidden_rule_fn"]
        correct = sum(1 for o in train_obs if fn(o["state"]) == o["next_state"])
        return correct / len(train_obs)
    except Exception:
        return 0.0


# ── C-retrieval ───────────────────────────────────────────────────────
def c_retrieval_agent(harness) -> None:
    """
    Submits the ground-truth hidden_rule_fn directly (ceiling — no abduction needed).
    """
    harness.submit_hypothesis(harness.instance["hidden_rule_fn"])


# ── Run all controls on all instances ────────────────────────────────
def run_controls(instances: list) -> dict:
    """Returns {instance_id: {C-random: acc, C-induce: acc, C-retrieval: acc}}"""
    from harness import WorldHarness
    results = {}
    for inst in instances:
        row = {}
        for name, agent in [
            ("C-random",    lambda h: c_random_agent(h, seed=42)),
            ("C-induce",    c_induce_agent),
            ("C-retrieval", c_retrieval_agent),
        ]:
            h = WorldHarness(inst)
            agent(h)
            log = h.get_log()
            subs = [e for e in log if e["tool"] == "submit_hypothesis"]
            row[name] = subs[-1]["accuracy"] if subs else 0.0
        results[inst["id"]] = row
        print(f"{inst['id']}: C-random={row['C-random']:.2f} C-induce={row['C-induce']:.2f} C-retrieval={row['C-retrieval']:.2f}")
    return results
