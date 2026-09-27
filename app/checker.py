"""Modal mu-calculus evaluation over finite transition systems.

Fixpoints are computed by naive iteration on the finite powerset
lattice: μ starts from the empty set, ν from the full state set, and
each approximant is recorded so callers can inspect the stabilisation
evidence. Guardedness/positivity (enforced by app.formula) guarantees
monotonicity, hence termination within |S| iterations.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from .formula import And, Box, Dia, Mu, Not, Nu, Or, Prop, Var

_SPLIT = re.compile(r"(\d+)")


def state_sort_key(state: str) -> str:
    """Natural-sort key so that e.g. s2 orders before s10."""
    return _SPLIT.sub(lambda m: m.group(0).zfill(12), state)


@dataclass(frozen=True)
class Model:
    states: tuple[str, ...]
    props: dict[str, frozenset]
    succ: dict[str, frozenset]
    edges: dict[str, tuple]

    @classmethod
    def from_spec(cls, locations, propositions, transitions):
        """Build a model from plain data.

        transitions: iterable of (transition_id, source, target) triples.
        """
        states = tuple(sorted(locations, key=state_sort_key))
        props = {s: frozenset(propositions.get(s, ())) for s in states}
        nxt = {s: set() for s in states}
        edges = {s: [] for s in states}
        for tid, src, dst in transitions:
            nxt[src].add(dst)
            edges[src].append((tid, dst))
        return cls(
            states=states,
            props=props,
            succ={s: frozenset(targets) for s, targets in nxt.items()},
            edges={s: tuple(sorted(ts)) for s, ts in edges.items()},
        )


def evaluate(ast, model: Model):
    """Evaluate a formula globally; return (satisfaction_set, iterations).

    iterations is a chronological list of approximant records:
        {"binder", "op" ("mu"|"nu"), "round", "step", "states"}
    where "round" distinguishes re-evaluations of the same binder under
    an enclosing fixpoint.
    """
    iterations: list[dict] = []
    rounds: dict[str, int] = {}
    universe = frozenset(model.states)
    sat = _eval(ast, {}, model, universe, iterations, rounds)
    return sat, iterations


def _record(iterations, node, round_no: int, step: int, states) -> None:
    iterations.append(
        {
            "binder": node.var,
            "op": "mu" if isinstance(node, Mu) else "nu",
            "round": round_no,
            "step": step,
            "states": sorted(states, key=state_sort_key),
        }
    )


def _eval(node, env, model: Model, universe, iterations, rounds):
    if isinstance(node, Prop):
        return frozenset(s for s in model.states if node.name in model.props[s])
    if isinstance(node, Var):
        return env[node.name]
    if isinstance(node, Not):
        return universe - _eval(node.child, env, model, universe, iterations, rounds)
    if isinstance(node, And):
        return _eval(node.left, env, model, universe, iterations, rounds) & _eval(
            node.right, env, model, universe, iterations, rounds
        )
    if isinstance(node, Or):
        return _eval(node.left, env, model, universe, iterations, rounds) | _eval(
            node.right, env, model, universe, iterations, rounds
        )
    if isinstance(node, Dia):
        inner = _eval(node.child, env, model, universe, iterations, rounds)
        return frozenset(s for s in model.states if model.succ[s] & inner)
    if isinstance(node, Box):
        inner = _eval(node.child, env, model, universe, iterations, rounds)
        return frozenset(s for s in model.states if model.succ[s] <= inner)
    if isinstance(node, (Mu, Nu)):
        round_no = rounds.get(node.var, 0) + 1
        rounds[node.var] = round_no
        current = frozenset() if isinstance(node, Mu) else universe
        step = 0
        _record(iterations, node, round_no, step, current)
        while True:
            # Copy the environment: nested bindings never leak outward.
            inner_env = dict(env)
            inner_env[node.var] = current
            nxt = _eval(node.body, inner_env, model, universe, iterations, rounds)
            step += 1
            _record(iterations, node, round_no, step, nxt)
            if nxt == current:
                return nxt
            current = nxt
    raise TypeError(f"unknown syntax node {node!r}")
