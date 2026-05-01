"""
AST-level mutator. Applied to top-K survivors when no candidate scores
> ε on train. Each input source can spawn ~25 mutated variants.

All edits are deterministic; no LLM. Each variant is round-tripped
through `compile()` and the linter — uncompilable variants are dropped.
"""
import ast
from .linter import lint


# Comparison swap catalogue
COMPARE_SWAPS = {
    ast.Eq:    [ast.NotEq, ast.GtE, ast.Gt, ast.LtE, ast.Lt],
    ast.NotEq: [ast.Eq],
    ast.Gt:    [ast.GtE, ast.Eq, ast.Lt],
    ast.GtE:   [ast.Gt, ast.Eq],
    ast.Lt:    [ast.LtE, ast.Eq, ast.Gt],
    ast.LtE:   [ast.Lt, ast.Eq],
}

# Binary operator swap catalogue
BINOP_SWAPS = {
    ast.Add:     [ast.Sub, ast.BitXor, ast.Mult],
    ast.Sub:     [ast.Add, ast.BitXor],
    ast.BitXor:  [ast.Add, ast.BitOr, ast.BitAnd],
    ast.Mult:    [ast.Add, ast.BitXor],
    ast.Mod:     [ast.FloorDiv],
    ast.BitOr:   [ast.BitXor, ast.BitAnd],
    ast.BitAnd:  [ast.BitOr, ast.BitXor],
    ast.LShift:  [ast.RShift],
    ast.RShift:  [ast.LShift],
}

# Constant nudges: integer literals get nudged ±1, ±2 plus mod-7/mod-11/mod-13 swaps
def _nudge_int(v: int) -> list[int]:
    candidates = {v - 2, v - 1, v + 1, v + 2}
    if v in {3, 5, 7, 8, 11, 13, 16}:
        # Modulus-shaped constants get neighbour-modulus suggestions.
        candidates |= {5, 7, 8, 11, 13, 16} - {v}
    return [c for c in candidates if c >= 0]


# Neighbourhood offset swap: detect 2-tuple offsets in a list literal
NEIGHBOURHOOD_PRESETS = [
    [(-1, 0), (1, 0), (0, -1), (0, 1)],                                              # vonneumann
    [(-1, -1), (-1, 1), (1, -1), (1, 1)],                                            # diagonal
    [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)],          # moore
    [(-2, -1), (-2, 1), (-1, -2), (-1, 2), (1, -2), (1, 2), (2, -1), (2, 1)],        # knight
    [(-2, 0), (2, 0), (0, -2), (0, 2)],                                              # axial2
]


# ----------------------------------------------------------------------
# Mutation passes
# ----------------------------------------------------------------------

class _CompareSwap(ast.NodeTransformer):
    def __init__(self, target_idx: int, new_op_cls):
        self.target_idx = target_idx
        self.new_op_cls = new_op_cls
        self.idx = -1

    def visit_Compare(self, node):
        self.generic_visit(node)
        for i, op in enumerate(node.ops):
            self.idx += 1
            if self.idx == self.target_idx:
                node.ops[i] = self.new_op_cls()
        return node


class _BinOpSwap(ast.NodeTransformer):
    def __init__(self, target_idx: int, new_op_cls):
        self.target_idx = target_idx
        self.new_op_cls = new_op_cls
        self.idx = -1

    def visit_BinOp(self, node):
        self.generic_visit(node)
        self.idx += 1
        if self.idx == self.target_idx:
            node.op = self.new_op_cls()
        return node


class _ConstNudge(ast.NodeTransformer):
    def __init__(self, target_idx: int, new_value):
        self.target_idx = target_idx
        self.new_value = new_value
        self.idx = -1

    def visit_Constant(self, node):
        if isinstance(node.value, int) and not isinstance(node.value, bool):
            self.idx += 1
            if self.idx == self.target_idx:
                node.value = self.new_value
        return node


class _OffsetListSwap(ast.NodeTransformer):
    """
    Replace a list-of-2-tuples literal with one of the NEIGHBOURHOOD_PRESETS.
    Heuristic: only swap lists where every element is a Tuple of two ints.
    """
    def __init__(self, target_idx: int, new_offsets):
        self.target_idx = target_idx
        self.new_offsets = new_offsets
        self.idx = -1

    def visit_List(self, node):
        self.generic_visit(node)
        # Validate shape
        if all(isinstance(e, ast.Tuple) and len(e.elts) == 2
               and all(isinstance(x, ast.Constant) and isinstance(x.value, int) for x in e.elts)
               for e in node.elts):
            self.idx += 1
            if self.idx == self.target_idx:
                # Build replacement node
                new_elts = []
                for dr, dc in self.new_offsets:
                    new_elts.append(ast.Tuple(
                        elts=[ast.Constant(value=dr), ast.Constant(value=dc)],
                        ctx=ast.Load(),
                    ))
                return ast.List(elts=new_elts, ctx=ast.Load())
        return node


def _safe_unparse(tree) -> str | None:
    try:
        ast.fix_missing_locations(tree)
        return ast.unparse(tree)
    except Exception:
        return None


def _count_nodes(tree, predicate) -> int:
    return sum(1 for n in ast.walk(tree) if predicate(n))


def mutate(source: str, max_per_source: int = 25) -> list[str]:
    """
    Generate up to `max_per_source` mutated variants of `source`.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []

    out = []

    # 1. Compare-op swaps
    n_compares = _count_nodes(tree, lambda n: isinstance(n, ast.Compare))
    for i in range(n_compares):
        # Find the i-th compare's op type
        idx = -1
        target_op_cls = None
        for n in ast.walk(tree):
            if isinstance(n, ast.Compare):
                for op in n.ops:
                    idx += 1
                    if idx == i:
                        target_op_cls = type(op)
                        break
                if target_op_cls is not None and idx == i:
                    break
        if target_op_cls is None:
            continue
        for new_cls in COMPARE_SWAPS.get(target_op_cls, []):
            tree2 = ast.parse(source)
            tree2 = _CompareSwap(i, new_cls).visit(tree2)
            src2 = _safe_unparse(tree2)
            if src2 and lint(src2)[0]:
                out.append(src2)
                if len(out) >= max_per_source:
                    return out

    # 2. BinOp swaps
    n_binops = _count_nodes(tree, lambda n: isinstance(n, ast.BinOp))
    for i in range(n_binops):
        # Find the i-th binop's op type
        idx = -1
        target_op_cls = None
        for n in ast.walk(tree):
            if isinstance(n, ast.BinOp):
                idx += 1
                if idx == i:
                    target_op_cls = type(n.op)
                    break
        if target_op_cls is None:
            continue
        for new_cls in BINOP_SWAPS.get(target_op_cls, []):
            tree2 = ast.parse(source)
            tree2 = _BinOpSwap(i, new_cls).visit(tree2)
            src2 = _safe_unparse(tree2)
            if src2 and lint(src2)[0]:
                out.append(src2)
                if len(out) >= max_per_source:
                    return out

    # 3. Const nudges
    n_consts = _count_nodes(tree,
                            lambda n: isinstance(n, ast.Constant)
                            and isinstance(n.value, int)
                            and not isinstance(n.value, bool))
    for i in range(n_consts):
        # Find the i-th int constant
        idx = -1
        target_value = None
        for n in ast.walk(tree):
            if isinstance(n, ast.Constant) and isinstance(n.value, int) and not isinstance(n.value, bool):
                idx += 1
                if idx == i:
                    target_value = n.value
                    break
        if target_value is None:
            continue
        for new_val in _nudge_int(target_value):
            tree2 = ast.parse(source)
            tree2 = _ConstNudge(i, new_val).visit(tree2)
            src2 = _safe_unparse(tree2)
            if src2 and lint(src2)[0]:
                out.append(src2)
                if len(out) >= max_per_source:
                    return out

    # 4. Neighbourhood-offset list swaps
    # Count list-of-int-tuples literals
    list_idx = -1
    swap_targets = []
    for n in ast.walk(tree):
        if isinstance(n, ast.List) and all(
                isinstance(e, ast.Tuple) and len(e.elts) == 2
                and all(isinstance(x, ast.Constant) and isinstance(x.value, int) for x in e.elts)
                for e in n.elts):
            list_idx += 1
            swap_targets.append(list_idx)
    for i in swap_targets:
        for preset in NEIGHBOURHOOD_PRESETS:
            tree2 = ast.parse(source)
            tree2 = _OffsetListSwap(i, preset).visit(tree2)
            src2 = _safe_unparse(tree2)
            if src2 and lint(src2)[0]:
                out.append(src2)
                if len(out) >= max_per_source:
                    return out

    return out


def mutate_pool(sources: list[str], per_source: int = 10, max_total: int = 200) -> list[str]:
    """Convenience: mutate each source and return a flat deduped list."""
    out = []
    seen = set()
    for src in sources:
        for v in mutate(src, max_per_source=per_source):
            if v not in seen:
                seen.add(v)
                out.append(v)
                if len(out) >= max_total:
                    return out
    return out
