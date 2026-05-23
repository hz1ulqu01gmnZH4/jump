"""Probe-grid construction and equivalence metrics for the anti-homogenization benchmark."""
import copy
import json
import random

from worlds.gen import _ca_random_state, _pt_random_state, _seq_random_seed


def _ca_alphabet(instance):
    values = set()
    for obs in instance['train_obs']:
        for row in obs['state']:
            values.update(row)
    return sorted(values)


def _pt_types(instance):
    return sorted({p['type'] for obs in instance['train_obs'] for p in obs['state']})


def _pt_grid_size(instance):
    max_coord = max(
        max(p['x'], p['y'])
        for obs in instance['train_obs']
        for p in obs['state']
    )
    return max(8, max_coord + 1)


def _seq_params(instance):
    all_vals = set()
    for obs in instance['train_obs']:
        all_vals.update(obs['state'])
        ns = obs['next_state']
        if isinstance(ns, list):
            all_vals.update(ns)
    mod = max(all_vals) + 1
    seed_len = min(len(obs['state']) for obs in instance['train_obs'])
    return seed_len, mod


def build_probe_grid(instance: dict, n: int = 100, seed: int = 0) -> list:
    """Return n input states sampled from the instance family's state distribution.

    Stratification:
    - 60% random states (family-appropriate random sampler)
    - 20% uniform/degenerate states (all same value)
    - 20% single-defect states (one cell different from a random baseline)

    Never calls hidden_rule_fn. Deterministic in (instance['id'], seed).
    """
    rng = random.Random(f"{instance['id']}:{seed}")
    n_random = int(n * 0.6)
    n_uniform = int(n * 0.2)
    n_defect = n - n_random - n_uniform

    family = instance['family']
    probes = []

    if family == 'cellular_automata':
        state0 = instance['train_obs'][0]['state']
        rows = len(state0)
        cols = len(state0[0])
        alphabet = _ca_alphabet(instance)

        for _ in range(n_random):
            probes.append(_ca_random_state(rows, cols, alphabet, rng))

        for i in range(n_uniform):
            val = alphabet[i % len(alphabet)]
            probes.append([[val] * cols for _ in range(rows)])

        for _ in range(n_defect):
            base = _ca_random_state(rows, cols, alphabet, rng)
            r = rng.randint(0, rows - 1)
            c = rng.randint(0, cols - 1)
            cur = base[r][c]
            others = [v for v in alphabet if v != cur]
            defect = copy.deepcopy(base)
            if others:
                defect[r][c] = rng.choice(others)
            probes.append(defect)

    elif family == 'particle_system':
        n_particles = len(instance['train_obs'][0]['state'])
        types = _pt_types(instance)
        grid_size = _pt_grid_size(instance)

        for _ in range(n_random):
            probes.append(_pt_random_state(n_particles, types, grid_size, rng))

        for i in range(n_uniform):
            t = types[i % len(types)]
            particles = [
                {'type': t, 'x': j % grid_size, 'y': (j // grid_size) % grid_size}
                for j in range(n_particles)
            ]
            probes.append(particles)

        for _ in range(n_defect):
            base = _pt_random_state(n_particles, types, grid_size, rng)
            defect = copy.deepcopy(base)
            idx = rng.randint(0, n_particles - 1)
            defect[idx]['x'] = (defect[idx]['x'] + 1) % grid_size
            probes.append(defect)

    elif family == 'pattern_puzzle':
        seed_len, mod = _seq_params(instance)

        for _ in range(n_random):
            probes.append(_seq_random_seed(seed_len, mod, rng))

        for i in range(n_uniform):
            probes.append([i % mod] * seed_len)

        for _ in range(n_defect):
            base = _seq_random_seed(seed_len, mod, rng)
            defect = list(base)
            if seed_len > 0:
                idx = rng.randint(0, seed_len - 1)
                defect[idx] = (defect[idx] + 1 + rng.randint(0, mod - 2)) % mod
            probes.append(defect)

    else:
        raise ValueError(f"Unknown family: {family!r}")

    return probes


def prediction_signature(hyp_fn, probes: list) -> tuple:
    """Apply hyp_fn to each probe state; canonicalize output via json.dumps(sort_keys=True).
    On exception, emit sentinel '<ERR>'.
    Return a tuple of canonical strings (hashable, one per probe)."""
    result = []
    for state in probes:
        try:
            output = hyp_fn(state)
            result.append(json.dumps(output, sort_keys=True))
        except Exception:
            result.append('<ERR>')
    return tuple(result)


def agreement_rate(sig_a: tuple, sig_b: tuple) -> float:
    """Fraction of probes where canonical outputs are byte-equal."""
    assert len(sig_a) == len(sig_b)
    return sum(a == b for a, b in zip(sig_a, sig_b)) / len(sig_a)


def cluster_runs(sigs: dict, theta: float = 0.95) -> list:
    """Connected-components clustering over run signatures.
    sigs: {run_id: signature_tuple}
    theta: minimum agreement_rate to add an edge.
    Returns list of sets of run_ids."""
    run_ids = list(sigs.keys())
    adj = {r: set() for r in run_ids}
    for i in range(len(run_ids)):
        for j in range(i + 1, len(run_ids)):
            r1, r2 = run_ids[i], run_ids[j]
            if agreement_rate(sigs[r1], sigs[r2]) >= theta:
                adj[r1].add(r2)
                adj[r2].add(r1)

    visited = set()
    components = []
    for r in run_ids:
        if r not in visited:
            component = set()
            queue = [r]
            while queue:
                node = queue.pop()
                if node not in visited:
                    visited.add(node)
                    component.add(node)
                    queue.extend(adj[node] - visited)
            components.append(component)
    return components
