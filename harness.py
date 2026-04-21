"""
v2 abductive-jump harness.
Wraps a world instance, exposes tool-call API to agents,
scores submitted executable hypotheses against held-out test_obs.
"""

import json, sys, copy, time
from pathlib import Path
from worlds.gen import generate_all, run_instance


class WorldHarness:
    def __init__(self, instance: dict):
        self.instance = instance
        self.interaction_log = []
        self.hypothesis_submitted = False
        self._hidden_ns = {}
        exec(instance["hidden_rule_fn"], self._hidden_ns)
        self._true_rule = self._hidden_ns["hidden_rule_fn"]

    # ── Tool: get_train_obs ──────────────────────────────────────────
    def get_train_obs(self) -> list:
        """Return training observations (state → next_state pairs). Agent-visible."""
        self.interaction_log.append({"tool": "get_train_obs", "t": time.monotonic()})
        return self.instance["train_obs"]

    # ── Tool: intervene ─────────────────────────────────────────────
    def intervene(self, action: str, **kwargs) -> dict:
        """
        Apply an action from the instance's intervention_api to a given state,
        then advance the world one step using the true rule.
        kwargs must include 'state' (the current world state to modify).
        Returns {"modified_state": ..., "next_state": ...}.
        Fails loud if action not in intervention_api.
        """
        valid_actions = {a["action"] for a in self.instance["intervention_api"]}
        if action not in valid_actions:
            raise ValueError(f"Unknown action '{action}'. Valid: {valid_actions}")
        state = copy.deepcopy(kwargs.get("state"))
        if state is None:
            raise ValueError("intervene() requires 'state' kwarg")
        modified = self._apply_action(action, state, kwargs)
        next_state = self._true_rule(modified)
        self.interaction_log.append({
            "tool": "intervene", "action": action,
            "kwargs": {k: v for k, v in kwargs.items() if k != "state"},
            "next_state_preview": str(next_state)[:80],
            "t": time.monotonic()
        })
        return {"modified_state": modified, "next_state": next_state}

    def _apply_action(self, action: str, state, kwargs: dict):
        """Minimal action dispatcher — applies action to state in-place copy."""
        fam = self.instance["family"]
        if fam == "cellular_automata":
            if action == "set_cell":
                r, c, v = kwargs["row"], kwargs["col"], kwargs["value"]
                state[r][c] = v
            elif action == "flip_row":
                row = kwargs["row"]
                fam_states = list(self.instance["primitive_glossary"].keys())
                # Only use keys that are single-symbol cell states (integers or short strings)
                # Use the alphabet from the first train_obs to determine valid cell states
                sample = state[row][0] if state[row] else None
                if isinstance(sample, int):
                    # Find max value used — states are 0..N-1
                    all_vals = sorted({v for r in state for v in r})
                    n = len(all_vals)
                    state[row] = [(v + 1) % n for v in state[row]]
                else:
                    # String states
                    for i, c in enumerate(state[row]):
                        if c in fam_states:
                            state[row][i] = fam_states[(fam_states.index(c) + 1) % len(fam_states)]
            elif action == "inject_pattern":
                r, c = kwargs["row"], kwargs["col"]
                rows = len(state)
                cols = len(state[0]) if state else 0
                for dr, pat_row in enumerate(kwargs["pattern"]):
                    for dc, val in enumerate(pat_row):
                        if val is not None:
                            nr = (r + dr) % rows
                            nc = (c + dc) % cols
                            state[nr][nc] = val
        elif fam == "particle_system":
            if action == "spawn":
                # No-op if cell already occupied
                occupied = {(p["x"], p["y"]) for p in state}
                if (kwargs["x"], kwargs["y"]) not in occupied:
                    state.append({"type": kwargs["type"], "x": kwargs["x"], "y": kwargs["y"]})
            elif action == "remove":
                state[:] = [p for p in state if not (p["x"] == kwargs["x"] and p["y"] == kwargs["y"])]
            elif action == "set_velocity":
                grid_size = 8
                for p in state:
                    if p["x"] == kwargs["x"] and p["y"] == kwargs["y"]:
                        p["x"] = (p["x"] + kwargs.get("vx", 0)) % grid_size
                        p["y"] = (p["y"] + kwargs.get("vy", 0)) % grid_size
            elif action == "change_type":
                for p in state:
                    if p["x"] == kwargs["x"] and p["y"] == kwargs["y"]:
                        p["type"] = kwargs["new_type"]
        elif fam == "pattern_puzzle":
            if action == "set_element":
                state[kwargs["index"]] = kwargs["value"]
            elif action == "insert":
                state.insert(kwargs["index"], kwargs["value"])
            elif action == "apply_perturbation":
                # Determine modulus from context — use 7 as default (spec uses mod 7 for seqs)
                mod = kwargs.get("mod", 7)
                state[kwargs["index"]] = (state[kwargs["index"]] + kwargs["delta"]) % mod
        return state

    # ── Tool: submit_hypothesis ──────────────────────────────────────
    def submit_hypothesis(self, hypothesis_source: str) -> dict:
        """
        Accept a Python source string defining hidden_rule_fn(state) -> next_state.
        Compile, run against test_obs, return accuracy.
        Fails loud on syntax error — does NOT silently return 0.
        """
        if self.hypothesis_submitted:
            raise RuntimeError("Hypothesis already submitted for this instance.")
        compile(hypothesis_source, "<hypothesis>", "exec")
        ns = {}
        exec(hypothesis_source, ns)
        if "hidden_rule_fn" not in ns:
            raise ValueError("Hypothesis source must define a function named 'hidden_rule_fn'.")
        hyp_fn = ns["hidden_rule_fn"]
        accuracy = run_instance(self.instance, hyp_fn)
        self.hypothesis_submitted = True
        self.interaction_log.append({
            "tool": "submit_hypothesis",
            "hypothesis_source": hypothesis_source,
            "accuracy": accuracy,
            "t": time.monotonic()
        })
        return {
            "accuracy": accuracy,
            "test_obs_count": len(self.instance["test_obs"]),
            "correct": int(accuracy * len(self.instance["test_obs"]))
        }

    def get_log(self) -> list:
        return self.interaction_log


def run_agent_on_instance(instance: dict, agent_fn) -> dict:
    """
    Run agent_fn(harness) on a world instance.
    agent_fn receives a WorldHarness and must call submit_hypothesis() exactly once.
    Returns: {accuracy, interaction_log, instance_id, hypothesis_source}
    """
    harness = WorldHarness(instance)
    agent_fn(harness)
    log = harness.get_log()
    submissions = [e for e in log if e["tool"] == "submit_hypothesis"]
    if not submissions:
        raise RuntimeError(f"Agent did not submit a hypothesis for {instance['id']}")
    return {
        "instance_id": instance["id"],
        "family": instance["family"],
        "difficulty": instance["difficulty"],
        "accuracy": submissions[-1]["accuracy"],
        "hypothesis_source": submissions[-1]["hypothesis_source"],
        "interaction_log": log,
        "n_interventions": sum(1 for e in log if e["tool"] == "intervene"),
    }
