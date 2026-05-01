"""
Anti-unification by hash-on-mask.

Group nodes by their projection onto a feature mask; an equivalence class is
"clean" if all members share the same `next` value, "conflicted" otherwise.
This is the core operation Arm C externalises from the LLM.
"""

from __future__ import annotations
from collections import defaultdict


def project(node: dict, mask: tuple[str, ...]) -> tuple:
    return tuple(node[f] for f in mask)


def group(nodes: list[dict], mask: tuple[str, ...]) -> dict[tuple, list[dict]]:
    g: dict[tuple, list[dict]] = defaultdict(list)
    for n in nodes:
        g[project(n, mask)].append(n)
    return g


def conflicts(nodes: list[dict], mask: tuple[str, ...]) -> int:
    """Count of equivalence classes whose `next` is multi-valued."""
    g = group(nodes, mask)
    return sum(1 for members in g.values()
               if len({m["next"] for m in members}) > 1)


def conflicted_node_count(nodes: list[dict], mask: tuple[str, ...]) -> int:
    """Sum of nodes living in conflicted classes (a finer-grained signal)."""
    g = group(nodes, mask)
    total = 0
    for members in g.values():
        if len({m["next"] for m in members}) > 1:
            total += len(members)
    return total


def lookup_table(nodes: list[dict], mask: tuple[str, ...]) -> dict[tuple, int]:
    """
    Build a (mask-projection -> next) table from clean equivalence classes.
    Conflicted classes are dropped (caller must ensure mask makes them clean
    before relying on the table).
    """
    g = group(nodes, mask)
    table: dict[tuple, int] = {}
    for key, members in g.items():
        nexts = {m["next"] for m in members}
        if len(nexts) == 1:
            table[key] = next(iter(nexts))
    return table
