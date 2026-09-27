"""Stable occurrence identifiers for parsed formulas.

Every *syntactic occurrence* of a formula receives a stable position id
(``p0``, ``p1`` ...) by deterministic pre-order traversal of the AST
(parentheses do not occupy positions).  Positions survive freezing and
recomputation: re-parsing the same formula yields identical ids.

The metadata table also records, for each variable occurrence, the
statically resolved binder position (nearest enclosing binder of that
name), so evidence edges can be checked against the binding structure
without trusting the evaluator.
"""
from __future__ import annotations

from dataclasses import dataclass

from .formula import And, Dia, Box, Mu, Nu, Not, Or, Prop, Var

NODE_KINDS = {
    Prop: "prop",
    Var: "var",
    Not: "not",
    And: "and",
    Or: "or",
    Dia: "dia",
    Box: "box",
    Mu: "mu",
    Nu: "nu",
}


def _children(node):
    if isinstance(node, (Not, Dia, Box)):
        return (node.child,)
    if isinstance(node, (And, Or)):
        return (node.left, node.right)
    if isinstance(node, (Mu, Nu)):
        return (node.body,)
    return ()


@dataclass(frozen=True)
class PositionTable:
    """Result of labelling: node identity -> pos, pos -> metadata."""

    node_pos: dict
    meta: dict
    order: tuple

    def root_pos(self) -> str:
        return self.order[0]


def label_positions(ast) -> PositionTable:
    """Assign stable pre-order occurrence ids to every AST node."""
    node_pos: dict = {}
    meta: dict = {}
    order: list[str] = []
    counter = 0

    def walk(node, parent, binders):
        nonlocal counter
        pos = f"p{counter}"
        counter += 1
        kind = NODE_KINDS.get(type(node))
        if kind is None:
            raise TypeError(f"unknown syntax node {node!r}")
        node_pos[id(node)] = pos
        order.append(pos)

        child_bindings = binders
        entry = {"kind": kind, "parent": parent, "children": []}
        if isinstance(node, Prop):
            entry["name"] = node.name
        elif isinstance(node, Var):
            entry["name"] = node.name
            for binder_pos, binder_name in reversed(binders):
                if binder_name == node.name:
                    entry["binder_pos"] = binder_pos
                    break
        elif isinstance(node, (Mu, Nu)):
            entry["name"] = node.var
            child_bindings = binders + ((pos, node.var),)

        child_posses = []
        for child in _children(node):
            child_posses.append(walk(child, pos, child_bindings))
        entry["children"] = child_posses
        meta[pos] = entry
        return pos

    walk(ast, None, ())
    return PositionTable(node_pos=node_pos, meta=meta, order=tuple(order))


def positions_to_json(table: PositionTable) -> list:
    """Serializable position catalogue, in stable pre-order."""
    out = []
    for pos in table.order:
        m = table.meta[pos]
        entry = {
            "pos": pos,
            "kind": m["kind"],
            "parent": m["parent"],
            "children": list(m["children"]),
        }
        if "name" in m:
            entry["name"] = m["name"]
        if "binder_pos" in m:
            entry["binder_pos"] = m["binder_pos"]
        out.append(entry)
    return out


def positions_from_json(entries: list) -> dict:
    """Rebuild the metadata index from a frozen catalogue."""
    meta = {}
    for e in entries:
        meta[e["pos"]] = {
            "kind": e["kind"],
            "parent": e["parent"],
            "children": list(e["children"]),
            **({"name": e["name"]} if "name" in e else {}),
            **({"binder_pos": e["binder_pos"]} if "binder_pos" in e else {}),
        }
    return meta
