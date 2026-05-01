"""
Branch-tree crossover. Take two parent sources defining hidden_rule_fn;
swap one parent's `if/elif/else` chain with the other's — yielding a
hybrid candidate.

This is heavier than pure mutation because branches reference variables
defined in the surrounding scope (e.g. `cur`, `counts`, `nb`); for the
crossover to be valid, both parents must use compatible variable
conventions. Our templates do — the CA branch templates always define
`cur`, `counts`, `nb` in the same way; the SEQ branch templates always
reference `state`, `n`. The crossover module checks compilability via
linter and discards invalid splices.
"""
import ast
import copy

from .linter import lint


def _find_topmost_if(fn_node: ast.FunctionDef) -> tuple[int | None, ast.If | None]:
    """Find the index in fn_node.body of the outermost If chain (if any).
    Returns (index, node) or (None, None).
    """
    for i, stmt in enumerate(fn_node.body):
        if isinstance(stmt, ast.If):
            return i, stmt
    return None, None


def _find_innermost_if(fn_node: ast.FunctionDef) -> tuple[ast.AST | None, int | None, ast.If | None]:
    """Find the deepest If statement nested inside loops/functions for CA-style
    rules whose branch chain lives inside `for r in range(rows): for c in range(cols):`.
    Returns (parent_body_list, index_in_parent, if_node) or (None, None, None).
    """
    found = [None, None, None]

    def walk(parent_body, ctx_kind):
        for i, stmt in enumerate(parent_body):
            if isinstance(stmt, ast.If) and ctx_kind == "loop":
                found[0] = parent_body
                found[1] = i
                found[2] = stmt
                return True
            if isinstance(stmt, ast.For):
                if walk(stmt.body, "loop"):
                    return True
            elif isinstance(stmt, ast.While):
                if walk(stmt.body, "loop"):
                    return True
            elif isinstance(stmt, ast.If):
                if walk(stmt.body, ctx_kind):
                    return True
                if walk(stmt.orelse, ctx_kind):
                    return True
        return False

    walk(fn_node.body, "top")
    return found[0], found[1], found[2]


def crossover(src_a: str, src_b: str) -> list[str]:
    """
    Produce up to 2 hybrid candidates: A's outer structure with B's branch
    tree spliced in, and vice versa.
    """
    out = []
    try:
        tree_a = ast.parse(src_a)
        tree_b = ast.parse(src_b)
    except SyntaxError:
        return out

    fn_a = next((n for n in tree_a.body if isinstance(n, ast.FunctionDef)
                 and n.name == "hidden_rule_fn"), None)
    fn_b = next((n for n in tree_b.body if isinstance(n, ast.FunctionDef)
                 and n.name == "hidden_rule_fn"), None)
    if fn_a is None or fn_b is None:
        return out

    # Try innermost-loop-If swap first (the CA case).
    pba, ia, ifa = _find_innermost_if(fn_a)
    pbb, ib, ifb = _find_innermost_if(fn_b)
    if ifa is not None and ifb is not None:
        # A-with-B-branches
        try:
            tree_a2 = ast.parse(src_a)
            pba2, ia2, _ = _find_innermost_if(
                next(n for n in tree_a2.body if isinstance(n, ast.FunctionDef)
                     and n.name == "hidden_rule_fn")
            )
            if pba2 is not None and ia2 is not None:
                pba2[ia2] = copy.deepcopy(ifb)
                ast.fix_missing_locations(tree_a2)
                src_new = ast.unparse(tree_a2)
                if lint(src_new)[0]:
                    out.append(src_new)
        except Exception:
            pass
        # B-with-A-branches
        try:
            tree_b2 = ast.parse(src_b)
            pbb2, ib2, _ = _find_innermost_if(
                next(n for n in tree_b2.body if isinstance(n, ast.FunctionDef)
                     and n.name == "hidden_rule_fn")
            )
            if pbb2 is not None and ib2 is not None:
                pbb2[ib2] = copy.deepcopy(ifa)
                ast.fix_missing_locations(tree_b2)
                src_new = ast.unparse(tree_b2)
                if lint(src_new)[0]:
                    out.append(src_new)
        except Exception:
            pass
        if out:
            return out

    # Fall back to topmost-If swap (the SEQ case).
    ia_top, ifa_top = _find_topmost_if(fn_a)
    ib_top, ifb_top = _find_topmost_if(fn_b)
    if ifa_top is None or ifb_top is None:
        return out

    try:
        tree_a2 = ast.parse(src_a)
        fn_a2 = next(n for n in tree_a2.body if isinstance(n, ast.FunctionDef)
                     and n.name == "hidden_rule_fn")
        idx, _ = _find_topmost_if(fn_a2)
        if idx is not None:
            fn_a2.body[idx] = copy.deepcopy(ifb_top)
            ast.fix_missing_locations(tree_a2)
            src_new = ast.unparse(tree_a2)
            if lint(src_new)[0]:
                out.append(src_new)
    except Exception:
        pass

    try:
        tree_b2 = ast.parse(src_b)
        fn_b2 = next(n for n in tree_b2.body if isinstance(n, ast.FunctionDef)
                     and n.name == "hidden_rule_fn")
        idx, _ = _find_topmost_if(fn_b2)
        if idx is not None:
            fn_b2.body[idx] = copy.deepcopy(ifa_top)
            ast.fix_missing_locations(tree_b2)
            src_new = ast.unparse(tree_b2)
            if lint(src_new)[0]:
                out.append(src_new)
    except Exception:
        pass

    return out


def crossover_pool(top_k: list[str], max_total: int = 100) -> list[str]:
    """For each pair in top_k, produce hybrids; dedupe and cap."""
    out = []
    seen = set()
    for i in range(len(top_k)):
        for j in range(len(top_k)):
            if i == j:
                continue
            for src in crossover(top_k[i], top_k[j]):
                if src not in seen:
                    seen.add(src)
                    out.append(src)
                    if len(out) >= max_total:
                        return out
    return out
