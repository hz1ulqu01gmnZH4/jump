"""
Smoke test: end-to-end on world_ca_001 must score >0 and produce
syntactically valid Python source.
"""

from __future__ import annotations
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "arm_c"))

from harness import WorldHarness
from worlds.gen import generate_all
from engine.ca_solver import solve_ca


def test_ca_001_end_to_end():
    inst = next(i for i in generate_all() if i["id"] == "world_ca_001")
    h = WorldHarness(inst)
    diag = solve_ca(h)

    log = h.get_log()
    subs = [e for e in log if e["tool"] == "submit_hypothesis"]
    assert len(subs) == 1, "must submit exactly once"

    src = subs[-1]["hypothesis_source"]
    compile(src, "<smoke-hypothesis>", "exec")  # syntactic validity

    # Diagonal-tetrad is the ground truth for ca_001; the engine must
    # have selected at least one diag4 feature and beat the C-induce floor (0.0).
    assert any("diag4" in f for f in diag["final_mask"]), \
        f"expected diag4 in mask, got {diag['final_mask']}"
    assert subs[-1]["accuracy"] > 0.5, \
        f"expected >0.5 on ca_001, got {subs[-1]['accuracy']}"


def test_all_ca_beats_baseline():
    """Mean across all 6 CA instances must beat C-induce (0.000) by > 0.1."""
    accs = []
    for inst in generate_all():
        if inst["family"] != "cellular_automata":
            continue
        h = WorldHarness(inst)
        solve_ca(h)
        log = h.get_log()
        subs = [e for e in log if e["tool"] == "submit_hypothesis"]
        accs.append(subs[-1]["accuracy"])
    mean = sum(accs) / len(accs)
    assert mean > 0.1, f"CA mean {mean:.3f} did not beat C-induce baseline by 0.1"


if __name__ == "__main__":
    test_ca_001_end_to_end()
    print("test_ca_001_end_to_end: PASS")
    test_all_ca_beats_baseline()
    print("test_all_ca_beats_baseline: PASS")
