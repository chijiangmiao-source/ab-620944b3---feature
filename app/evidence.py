"""Fixpoint dependency-evidence audits.

An audit freezes a review (spec, formula, initial location and the
original conclusion), assigns a stable identifier to every syntactic
occurrence of the formula, and builds a shareable directed dependency
evidence graph for the initial location: a satisfaction proof when the
conclusion holds, otherwise a refutation of the opposite polarity —
never a forged satisfaction DAG.

Evidence nodes are keyed by (position, location, polarity, binding
environment, round) and their ids are content-addressed, so identical
claims share one node and the graph stays acyclic except where
stable-cycle binders close loops:

* proposition, boolean and modal nodes reference their children per
  location, and modal references carry the witnessing transition ids;
* well-founded binders (mu being proved, nu being refuted) unfold only
  into strictly earlier approximation rounds;
* stable-cycle binders (nu being proved, mu being refuted) may close
  cycles, but only inside their stable approximant;
* every node carries its binding environment (binder position -> round),
  so nested fixpoint environments never leak into each other.

verify_audit() independently re-checks a stored audit on every read:
evidence edges, location propositions, transition endpoints, binding
environments, approximation rounds and the root conclusion, plus a
recomputation of the frozen spec against the frozen conclusion.
"""
from __future__ import annotations

import hashlib
import json

from .checker import Model, evaluate, state_sort_key
from .formula import And, Box, Dia, FormulaError, Mu, Not, Nu, Or, Prop, Var, parse

SAT = "sat"
UNSAT = "unsat"
_FLIP = {SAT: UNSAT, UNSAT: SAT}
# Binder/polarity pairs whose unfolding must descend to strictly earlier
# approximation rounds; the complementary pairs close cycles at the
# stable approximant instead.
_WELL_FOUNDED = {("mu", SAT), ("nu", UNSAT)}

MAX_ERRORS = 50


class EvidenceError(ValueError):
    """Raised when dependency evidence cannot be closed into a proof."""


def position_id(idx: int) -> str:
    return f"f{idx}"


def _is_round(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


# --- stable occurrence positions -------------------------------------------


class Positions:
    """Pre-order numbering of every syntactic occurrence of a formula."""

    def __init__(self, ast):
        self.nodes: list = []
        self.parent: list = []
        self.enclosing: list = []  # idx -> tuple of binder indices strictly above
        self.binder_of: dict = {}  # var occurrence idx -> binder idx
        self.children: dict = {}   # idx -> {"left"/"right"/"child"/"body": idx}
        self._walk(ast, None, (), {})

    def _walk(self, node, parent, enclosing, scope):
        idx = len(self.nodes)
        self.nodes.append(node)
        self.parent.append(parent)
        self.enclosing.append(enclosing)
        if isinstance(node, (Mu, Nu)):
            self.children[idx] = {"body": idx + 1}
            self._walk(node.body, idx, enclosing + (idx,), {**scope, node.var: idx})
        elif isinstance(node, Var):
            self.binder_of[idx] = scope[node.name]  # parser guarantees binding
        elif isinstance(node, (Not, Dia, Box)):
            self.children[idx] = {"child": idx + 1}
            self._walk(node.child, idx, enclosing, scope)
        elif isinstance(node, (And, Or)):
            left = idx + 1
            self._walk(node.left, idx, enclosing, scope)
            right = len(self.nodes)
            self._walk(node.right, idx, enclosing, scope)
            self.children[idx] = {"left": left, "right": right}
        elif not isinstance(node, Prop):
            raise EvidenceError(f"unknown syntax node {node!r}")

    def kind(self, idx: int) -> str:
        node = self.nodes[idx]
        if isinstance(node, Prop):
            return "prop"
        if isinstance(node, Var):
            return "var"
        if isinstance(node, Not):
            return "not"
        if isinstance(node, And):
            return "and"
        if isinstance(node, Or):
            return "or"
        if isinstance(node, Dia):
            return "dia"
        if isinstance(node, Box):
            return "box"
        if isinstance(node, Mu):
            return "mu"
        if isinstance(node, Nu):
            return "nu"
        raise EvidenceError(f"unknown syntax node {node!r}")

    def op(self, idx: int) -> str:
        return "mu" if isinstance(self.nodes[idx], Mu) else "nu"


def position_table(pos: Positions) -> list:
    """Stable, shareable description of every syntactic occurrence."""
    table = []
    for idx, node in enumerate(pos.nodes):
        entry = {
            "id": position_id(idx),
            "kind": pos.kind(idx),
            "parent": None if pos.parent[idx] is None else position_id(pos.parent[idx]),
            "enclosing": [position_id(c) for c in pos.enclosing[idx]],
        }
        if isinstance(node, Prop):
            entry["name"] = node.name
        elif isinstance(node, Var):
            entry["name"] = node.name
            entry["binder"] = position_id(pos.binder_of[idx])
        elif isinstance(node, (Mu, Nu)):
            entry["var"] = node.var
            entry["body"] = position_id(pos.children[idx]["body"])
        elif isinstance(node, (Not, Dia, Box)):
            entry["child"] = position_id(pos.children[idx]["child"])
        elif isinstance(node, (And, Or)):
            entry["left"] = position_id(pos.children[idx]["left"])
            entry["right"] = position_id(pos.children[idx]["right"])
        table.append(entry)
    return table


def evidence_node_id(idx, location, polarity, env_items, rnd) -> str:
    """Content-addressed node id: identical claims share one identifier,
    which keeps the evidence shareable across audits."""
    canon = json.dumps(
        {"p": idx, "l": location, "s": polarity, "r": rnd,
         "e": [[c, r] for c, r in env_items]},
        sort_keys=True,
    )
    return "e" + hashlib.sha256(canon.encode("utf-8")).hexdigest()[:24]


# --- semantics with round-indexed environments ------------------------------


class Semantics:
    """Evaluator exposing per-binder approximant sequences.

    Rounds environments map binder positions to the approximant index at
    which the binder's variable is currently interpreted.
    """

    def __init__(self, model: Model, pos: Positions):
        self.model = model
        self.pos = pos
        self.universe = frozenset(model.states)
        self._eval_cache = {}
        self._approx_sets_cache = {}
        self._approx_rounds_cache = {}

    @staticmethod
    def _sets_key(env_sets):
        return tuple(sorted(env_sets.items()))

    def eval_at(self, idx, env_sets):
        key = (idx, self._sets_key(env_sets))
        if key in self._eval_cache:
            return self._eval_cache[key]
        node = self.pos.nodes[idx]
        model = self.model
        if isinstance(node, Prop):
            value = frozenset(s for s in model.states if node.name in model.props[s])
        elif isinstance(node, Var):
            value = env_sets[self.pos.binder_of[idx]]
        elif isinstance(node, Not):
            value = self.universe - self.eval_at(self.pos.children[idx]["child"], env_sets)
        elif isinstance(node, And):
            ch = self.pos.children[idx]
            value = self.eval_at(ch["left"], env_sets) & self.eval_at(ch["right"], env_sets)
        elif isinstance(node, Or):
            ch = self.pos.children[idx]
            value = self.eval_at(ch["left"], env_sets) | self.eval_at(ch["right"], env_sets)
        elif isinstance(node, Dia):
            inner = self.eval_at(self.pos.children[idx]["child"], env_sets)
            value = frozenset(s for s in model.states if model.succ[s] & inner)
        elif isinstance(node, Box):
            inner = self.eval_at(self.pos.children[idx]["child"], env_sets)
            value = frozenset(s for s in model.states if model.succ[s] <= inner)
        elif isinstance(node, (Mu, Nu)):
            value = self.approx_from_sets(idx, env_sets)[-1]
        else:
            raise EvidenceError(f"unknown syntax node {node!r}")
        self._eval_cache[key] = value
        return value

    def approx_from_sets(self, binder, env_sets):
        """Approximant sequence [X^0 .. X^m] with X^m == X^(m-1)."""
        key = (binder, self._sets_key(env_sets))
        seq = self._approx_sets_cache.get(key)
        if seq is None:
            node = self.pos.nodes[binder]
            seq = [frozenset() if isinstance(node, Mu) else self.universe]
            while True:
                inner = dict(env_sets)
                inner[binder] = seq[-1]
                nxt = self.eval_at(self.pos.children[binder]["body"], inner)
                seq.append(nxt)
                if nxt == seq[-2]:
                    break
            self._approx_sets_cache[key] = seq
        return seq

    def approximants(self, binder, rounds_env):
        """Approximants of `binder` under the enclosing binders' rounds."""
        enclosing = self.pos.enclosing[binder]
        key = (binder, tuple(rounds_env[c] for c in enclosing))
        seq = self._approx_rounds_cache.get(key)
        if seq is None:
            env_sets = {}
            for c in enclosing:
                env_sets[c] = self.approximants(c, rounds_env)[rounds_env[c]]
            seq = self.approx_from_sets(binder, env_sets)
            self._approx_rounds_cache[key] = seq
        return seq

    def env_sets(self, idx, rounds_env):
        """Materialise the set-valued environment for a position."""
        return {
            c: self.approximants(c, rounds_env)[rounds_env[c]]
            for c in self.pos.enclosing[idx]
        }


# --- evidence construction ---------------------------------------------------


class EvidenceBuilder:
    """Builds the directed dependency evidence for one initial location."""

    def __init__(self, model: Model, pos: Positions, transitions):
        self.model = model
        self.pos = pos
        self.sem = Semantics(model, pos)
        self.transitions_between = {}
        for t in transitions:
            self.transitions_between.setdefault((t["source"], t["target"]), []).append(t["id"])
        for ids in self.transitions_between.values():
            ids.sort()
        self.nodes = {}
        self.order = []

    def _env_items(self, idx, rounds_env):
        return tuple((c, rounds_env[c]) for c in self.pos.enclosing[idx])

    def _register(self, idx, location, polarity, rounds_env, rnd):
        env_items = self._env_items(idx, rounds_env)
        key = (idx, location, polarity, env_items, rnd)
        node = self.nodes.get(key)
        if node is not None:
            return node["id"], node, False
        nid = evidence_node_id(idx, location, polarity, env_items, rnd)
        node = {
            "id": nid,
            "position": position_id(idx),
            "location": location,
            "polarity": polarity,
            "round": rnd,
            "env": {position_id(c): r for c, r in env_items},
            "refs": [],
        }
        self.nodes[key] = node
        self.order.append(key)
        return nid, node, True

    def _claim_holds(self, idx, location, polarity, rounds_env, rnd=None):
        node = self.pos.nodes[idx]
        if isinstance(node, (Mu, Nu)):
            seq = self.sem.approximants(idx, rounds_env)
            holds = location in seq[-1 if rnd is None else rnd]
        elif isinstance(node, Var):
            binder = self.pos.binder_of[idx]
            seq = self.sem.approximants(binder, rounds_env)
            holds = location in seq[rounds_env[binder]]
        else:
            holds = location in self.sem.eval_at(idx, self.sem.env_sets(idx, rounds_env))
        return holds == (polarity == SAT)

    def build(self, initial):
        if initial not in self.sem.universe:
            raise EvidenceError(f"initial location '{initial}' is not in the model")
        polarity = SAT if initial in self.sem.eval_at(0, {}) else UNSAT
        root_id = self._expand(0, initial, polarity, {})
        return root_id, polarity

    def _expand(self, idx, location, polarity, rounds_env):
        node = self.pos.nodes[idx]
        if isinstance(node, (Mu, Nu)):
            return self._enter_binder(idx, location, polarity, rounds_env)
        nid, rec, fresh = self._register(idx, location, polarity, rounds_env, None)
        if not fresh:
            return nid
        if not self._claim_holds(idx, location, polarity, rounds_env):
            raise EvidenceError(
                f"claim at position {position_id(idx)} / location '{location}' "
                f"does not hold with polarity {polarity}"
            )
        if isinstance(node, Prop):
            pass  # leaf: justified by the frozen location propositions
        elif isinstance(node, Var):
            binder = self.pos.binder_of[idx]
            target = self._expand_binder(binder, location, polarity,
                                         rounds_env[binder], rounds_env)
            rec["refs"].append({"role": "binder", "to": target})
        elif isinstance(node, Not):
            child = self.pos.children[idx]["child"]
            rec["refs"].append(
                {"role": "child", "to": self._expand(child, location, _FLIP[polarity], rounds_env)}
            )
        elif isinstance(node, (And, Or)):
            self._expand_boolean(idx, location, polarity, rounds_env, rec)
        elif isinstance(node, (Dia, Box)):
            self._expand_modal(idx, location, polarity, rounds_env, rec)
        else:
            raise EvidenceError(f"unknown syntax node {node!r}")
        return nid

    def _expand_boolean(self, idx, location, polarity, rounds_env, rec):
        ch = self.pos.children[idx]
        need_both = isinstance(self.pos.nodes[idx], And) == (polarity == SAT)
        sides = (("left", ch["left"]), ("right", ch["right"]))
        if need_both:
            for role, cidx in sides:
                if not self._claim_holds(cidx, location, polarity, rounds_env):
                    raise EvidenceError("boolean child claim fails; evidence cannot close")
                rec["refs"].append(
                    {"role": role, "to": self._expand(cidx, location, polarity, rounds_env)}
                )
        else:
            for role, cidx in sides:
                if self._claim_holds(cidx, location, polarity, rounds_env):
                    rec["refs"].append(
                        {"role": role, "to": self._expand(cidx, location, polarity, rounds_env)}
                    )
                    return
            raise EvidenceError("no witnessing boolean child; evidence cannot close")

    def _expand_modal(self, idx, location, polarity, rounds_env, rec):
        child = self.pos.children[idx]["child"]
        cover_all = isinstance(self.pos.nodes[idx], Box) == (polarity == SAT)
        successors = sorted(self.model.succ[location], key=state_sort_key)
        if cover_all:
            for nxt in successors:
                if not self._claim_holds(child, nxt, polarity, rounds_env):
                    raise EvidenceError("successor claim fails; evidence cannot close")
                rec["refs"].append({
                    "role": "succ",
                    "to": self._expand(child, nxt, polarity, rounds_env),
                    "transitions": self.transitions_between[(location, nxt)],
                })
        else:
            for nxt in successors:
                if self._claim_holds(child, nxt, polarity, rounds_env):
                    rec["refs"].append({
                        "role": "succ",
                        "to": self._expand(child, nxt, polarity, rounds_env),
                        "transitions": self.transitions_between[(location, nxt)],
                    })
                    return
            raise EvidenceError("no witnessing successor; evidence cannot close")

    def _enter_binder(self, idx, location, polarity, rounds_env):
        op = self.pos.op(idx)
        seq = self.sem.approximants(idx, rounds_env)
        if (op, polarity) in _WELL_FOUNDED:
            want = op == "mu"  # mu/sat: first round containing; nu/unsat: first missing
            rnd = None
            for i in range(1, len(seq)):
                if (location in seq[i]) == want:
                    rnd = i
                    break
            if rnd is None:
                raise EvidenceError("no witnessing approximant; evidence cannot close")
        else:
            rnd = len(seq) - 1  # stable approximant
        return self._expand_binder(idx, location, polarity, rnd, rounds_env)

    def _expand_binder(self, idx, location, polarity, rnd, rounds_env):
        # The binder node's own environment is scoped to its statically
        # enclosing binders: inner bindings from the referencing site never
        # leak into it.
        scoped = {c: rounds_env[c] for c in self.pos.enclosing[idx]}
        nid, rec, fresh = self._register(idx, location, polarity, scoped, rnd)
        if not fresh:
            return nid
        op = self.pos.op(idx)
        seq = self.sem.approximants(idx, rounds_env)
        stable = len(seq) - 1
        if not 0 <= rnd <= stable:
            raise EvidenceError(f"round {rnd} outside approximant range 0..{stable}")
        if (location in seq[rnd]) != (polarity == SAT):
            raise EvidenceError("binder claim does not hold at its round")
        if (op, polarity) in _WELL_FOUNDED:
            if rnd < 1:
                raise EvidenceError("well-founded binder needs a round >= 1")
            body_round = rnd - 1  # strictly earlier approximant round
        else:
            if rnd != stable:
                raise EvidenceError("cycle binder must sit at the stable approximant")
            body_round = rnd  # cycles close inside the stable approximant
        body_env = dict(scoped)
        body_env[idx] = body_round
        rec["refs"].append({
            "role": "body",
            "to": self._expand(self.pos.children[idx]["body"], location, polarity, body_env),
        })
        return nid


def build_audit(spec, result):
    """Freeze a review into a dependency-evidence audit (not yet persisted)."""
    ast = parse(spec["formula"])
    pos = Positions(ast)
    model = Model.from_spec(
        spec["locations"],
        spec["propositions"],
        [(t["id"], t["source"], t["target"]) for t in spec["transitions"]],
    )
    builder = EvidenceBuilder(model, pos, spec["transitions"])
    root_id, polarity = builder.build(spec["initial"])
    return {
        "frozen": {"spec": spec, "result": result},
        "positions": position_table(pos),
        "polarity": polarity,
        "root": root_id,
        "evidence": [builder.nodes[key] for key in builder.order],
    }


# --- independent verification ------------------------------------------------


class _EvidenceVerifier:
    def __init__(self, pos, model, transitions, checks, fail):
        self.pos = pos
        self.model = model
        self.sem = Semantics(model, pos)
        self.checks = checks
        self.fail = fail
        self.nodes_by_id = {}
        self.transitions_between = {}
        for t in transitions:
            self.transitions_between.setdefault((t["source"], t["target"]), []).append(t["id"])
        for ids in self.transitions_between.values():
            ids.sort()
        self.pos_index = {position_id(i): i for i in range(len(pos.nodes))}

    def check_nodes(self, evidence):
        for node in evidence:
            if not isinstance(node, dict):
                self.fail("evidence node is not an object")
                continue
            nid = node.get("id")
            if nid in self.nodes_by_id:
                self.fail(f"duplicate evidence node id {nid!r}")
            self.nodes_by_id[nid] = node
        for node in evidence:
            if isinstance(node, dict):
                self.checks["nodes"] += 1
                try:
                    self._check_node(node)
                except Exception as exc:  # a corrupt node must not crash a read
                    self.fail(f"node {node.get('id')!r}: verification error: {exc}")
        return self.nodes_by_id

    def _check_node(self, node):
        nid = node.get("id")
        where = f"node {nid!r}"
        pid = node.get("position")
        idx = self.pos_index.get(pid)
        if idx is None:
            self.fail(f"{where}: unknown formula position {pid!r}")
            return
        location = node.get("location")
        if location not in self.model.states:
            self.fail(f"{where}: undeclared location {location!r}")
            return
        polarity = node.get("polarity")
        if polarity not in (SAT, UNSAT):
            self.fail(f"{where}: invalid polarity {polarity!r}")
            return
        env = node.get("env")
        if not isinstance(env, dict):
            self.fail(f"{where}: binding environment is not an object")
            return
        expected_keys = {position_id(c) for c in self.pos.enclosing[idx]}
        if set(env) != expected_keys:
            self.fail(
                f"{where}: binding environment {sorted(env)} does not match the "
                f"enclosing binders {sorted(expected_keys)}"
            )
            return
        rounds_env = {}
        for key, value in env.items():
            if not _is_round(value):
                self.fail(f"{where}: round for binder {key!r} is not an integer")
                return
            rounds_env[self.pos_index[key]] = value
        for c, r in rounds_env.items():
            seq = self.sem.approximants(c, rounds_env)
            if not 0 <= r < len(seq):
                self.fail(
                    f"{where}: round {r} for binder {position_id(c)} is outside "
                    f"the approximant range 0..{len(seq) - 1}"
                )
        env_items = tuple((c, rounds_env[c]) for c in self.pos.enclosing[idx])
        rnd = node.get("round")
        if nid != evidence_node_id(idx, location, polarity, env_items, rnd):
            self.fail(f"{where}: id is not the content-addressed evidence id")
        refs = node.get("refs")
        if not isinstance(refs, list):
            self.fail(f"{where}: refs is not a list")
            return
        self.checks["edges"] += len(refs)
        ast_node = self.pos.nodes[idx]
        if isinstance(ast_node, (Mu, Nu)):
            if not _is_round(rnd):
                self.fail(f"{where}: binder node must carry an integer round")
                return
            self._check_binder(node, idx, location, polarity, rounds_env, rnd, refs)
            return
        if rnd is not None:
            self.fail(f"{where}: non-binder node must not carry a round")
        if isinstance(ast_node, Prop):
            self._check_prop(node, ast_node, location, polarity, refs)
        elif isinstance(ast_node, Var):
            self._check_var(node, idx, location, polarity, rounds_env, refs)
        elif isinstance(ast_node, Not):
            self._check_semantic_claim(node, idx, location, polarity, rounds_env)
            child = self.pos.children[idx]["child"]
            self._expect_refs(
                node, refs, [("child", child, location, _FLIP[polarity], env, None, None)]
            )
        elif isinstance(ast_node, (And, Or)):
            self._check_semantic_claim(node, idx, location, polarity, rounds_env)
            self._check_boolean(node, idx, ast_node, location, polarity, env, refs)
        elif isinstance(ast_node, (Dia, Box)):
            self._check_semantic_claim(node, idx, location, polarity, rounds_env)
            self._check_modal(node, idx, ast_node, location, polarity, env, refs)
        else:
            self.fail(f"{where}: unknown syntax node kind")

    def _check_semantic_claim(self, node, idx, location, polarity, rounds_env):
        holds = location in self.sem.eval_at(idx, self.sem.env_sets(idx, rounds_env))
        if holds != (polarity == SAT):
            self.fail(
                f"node {node['id']!r}: claim does not hold under its binding environment"
            )

    def _check_prop(self, node, ast_node, location, polarity, refs):
        self.checks["proposition_leaves"] += 1
        if refs:
            self.fail(f"node {node['id']!r}: a proposition leaf must not reference nodes")
        holds = ast_node.name in self.model.props[location]
        if holds != (polarity == SAT):
            self.fail(
                f"node {node['id']!r}: proposition {ast_node.name!r} contradicts the "
                f"frozen propositions of location {location!r}"
            )

    def _check_var(self, node, idx, location, polarity, rounds_env, refs):
        binder = self.pos.binder_of[idx]
        rnd = rounds_env[binder]
        seq = self.sem.approximants(binder, rounds_env)
        if not 0 <= rnd < len(seq):
            self.fail(f"node {node['id']!r}: variable round {rnd} outside approximant range")
            return
        self.checks["round_constraints"] += 1
        if (location in seq[rnd]) != (polarity == SAT):
            self.fail(f"node {node['id']!r}: variable claim does not hold at round {rnd}")
        expected_env = {position_id(c): rounds_env[c] for c in self.pos.enclosing[binder]}
        self._expect_refs(
            node, refs, [("binder", binder, location, polarity, expected_env, None, rnd)]
        )

    def _check_binder(self, node, idx, location, polarity, rounds_env, rnd, refs):
        nid = node["id"]
        op = "mu" if isinstance(self.pos.nodes[idx], Mu) else "nu"
        seq = self.sem.approximants(idx, rounds_env)
        stable = len(seq) - 1
        if not 0 <= rnd <= stable:
            self.fail(f"node {nid!r}: round {rnd} outside approximant range 0..{stable}")
            return
        self.checks["round_constraints"] += 1
        if (location in seq[rnd]) != (polarity == SAT):
            self.fail(f"node {nid!r}: binder claim does not hold at round {rnd}")
        if (op, polarity) in _WELL_FOUNDED:
            if rnd < 1:
                self.fail(f"node {nid!r}: a {op}/{polarity} binder must unfold from a round >= 1")
                return
            body_round = rnd - 1  # strictly earlier approximant round
        else:
            if rnd != stable:
                self.fail(
                    f"node {nid!r}: a {op}/{polarity} binder may only cycle at the "
                    f"stable round {stable}"
                )
            body_round = rnd  # cycles close inside the stable approximant
        body_env = dict(node["env"])
        body_env[position_id(idx)] = body_round
        body = self.pos.children[idx]["body"]
        self._expect_refs(node, refs, [("body", body, location, polarity, body_env, None, None)])

    def _check_boolean(self, node, idx, ast_node, location, polarity, env, refs):
        ch = self.pos.children[idx]
        need_both = isinstance(ast_node, And) == (polarity == SAT)
        if need_both:
            self._expect_refs(node, refs, [
                ("left", ch["left"], location, polarity, env, None, None),
                ("right", ch["right"], location, polarity, env, None, None),
            ])
            return
        if len(refs) != 1 or not isinstance(refs[0], dict) or refs[0].get("role") not in ("left", "right"):
            self.fail(f"node {node['id']!r}: expected exactly one witnessing boolean reference")
            return
        role = refs[0]["role"]
        self._expect_refs(node, refs, [(role, ch[role], location, polarity, env, None, None)])

    def _check_modal(self, node, idx, ast_node, location, polarity, env, refs):
        child = self.pos.children[idx]["child"]
        cover_all = isinstance(ast_node, Box) == (polarity == SAT)
        successors = sorted(self.model.succ[location], key=state_sort_key)
        if cover_all:
            expected = [
                ("succ", child, nxt, polarity, env,
                 self.transitions_between.get((location, nxt), []), None)
                for nxt in successors
            ]
            self._expect_refs(node, refs, expected)
            self.checks["transition_endpoints"] += len(expected)
            return
        if len(refs) != 1 or not isinstance(refs[0], dict) or refs[0].get("role") != "succ":
            self.fail(f"node {node['id']!r}: expected exactly one witnessing successor")
            return
        target = self.nodes_by_id.get(refs[0].get("to"))
        tloc = target.get("location") if isinstance(target, dict) else None
        transitions = self.transitions_between.get((location, tloc))
        if not transitions:
            self.fail(
                f"node {node['id']!r}: witness {tloc!r} is not a successor of {location!r}"
            )
            return
        self._expect_refs(node, refs, [("succ", child, tloc, polarity, env, transitions, None)])
        self.checks["transition_endpoints"] += 1

    def _expect_refs(self, node, refs, expected):
        nid = node["id"]
        if len(refs) != len(expected):
            roles = ", ".join(e[0] for e in expected)
            self.fail(
                f"node {nid!r}: expected {len(expected)} reference(s) ({roles}), "
                f"found {len(refs)}"
            )
            return
        for ref, (role, tidx, tloc, tpol, tenv, ttrans, trnd) in zip(refs, expected):
            if not isinstance(ref, dict):
                self.fail(f"node {nid!r}: malformed reference")
                return
            if ref.get("role") != role:
                self.fail(
                    f"node {nid!r}: expected a {role!r} reference, found {ref.get('role')!r}"
                )
                return
            target = self.nodes_by_id.get(ref.get("to"))
            if target is None:
                self.fail(
                    f"node {nid!r}: reference {ref.get('to')!r} does not resolve to "
                    f"an evidence node"
                )
                return
            if target.get("position") != position_id(tidx):
                self.fail(
                    f"node {nid!r}: reference points at position "
                    f"{target.get('position')!r}, expected {position_id(tidx)!r}"
                )
            if target.get("location") != tloc:
                self.fail(
                    f"node {nid!r}: reference points at location "
                    f"{target.get('location')!r}, expected {tloc!r}"
                )
            if target.get("polarity") != tpol:
                self.fail(f"node {nid!r}: reference polarity mismatch")
            if target.get("env") != tenv:
                self.fail(
                    f"node {nid!r}: reference crosses binding environments "
                    f"(expected env {tenv}, found {target.get('env')})"
                )
            if trnd is not None and target.get("round") != trnd:
                self.fail(
                    f"node {nid!r}: reference points at round {target.get('round')}, "
                    f"expected {trnd}"
                )
            recorded = ref.get("transitions")
            if ttrans is None:
                if recorded is not None:
                    self.fail(f"node {nid!r}: unexpected transition record on a non-modal reference")
            elif recorded != ttrans:
                self.fail(
                    f"node {nid!r}: transition endpoints {recorded} do not match the "
                    f"frozen transitions {ttrans}"
                )


def verify_audit(record):
    """Independently verify an audit record; return a verification report."""
    errors: list[str] = []
    checks = {
        "positions": 0,
        "nodes": 0,
        "edges": 0,
        "proposition_leaves": 0,
        "transition_endpoints": 0,
        "round_constraints": 0,
    }

    def fail(message):
        if len(errors) < MAX_ERRORS:
            errors.append(message)

    frozen = record.get("frozen") or {}
    spec = frozen.get("spec") or {}
    result = frozen.get("result") or {}
    locations = spec.get("locations") or []
    transitions = spec.get("transitions") or []
    propositions = spec.get("propositions") or {}
    initial = spec.get("initial")

    # Frozen spec structural integrity.
    if not locations or len(set(locations)) != len(locations):
        fail("frozen locations are missing or not unique")
    if initial not in locations:
        fail("frozen initial location is not a declared location")
    transition_ids = [t.get("id") for t in transitions if isinstance(t, dict)]
    if len(transition_ids) != len(transitions) or len(set(transition_ids)) != len(transition_ids):
        fail("frozen transition identifiers are missing or not unique")
    for t in transitions:
        if not isinstance(t, dict) or t.get("source") not in locations or t.get("target") not in locations:
            fail(f"frozen transition {t!r} is not between declared locations")
    for loc in propositions:
        if loc not in locations:
            fail(f"frozen propositions reference unknown location {loc!r}")

    # Re-parse the frozen formula and rebuild the stable position table.
    try:
        ast = parse(spec.get("formula") or "")
    except FormulaError as exc:
        fail(f"frozen formula does not parse: {exc}")
        return {"verified": False, "errors": errors, "checks": checks}
    pos = Positions(ast)
    expected_positions = position_table(pos)
    checks["positions"] = len(expected_positions)
    if record.get("positions") != expected_positions:
        fail("position table does not match the frozen formula")

    # Recompute the conclusion from the frozen spec.
    try:
        model = Model.from_spec(
            locations,
            propositions,
            [(t["id"], t["source"], t["target"]) for t in transitions],
        )
        sat, iterations = evaluate(ast, model)
    except Exception as exc:  # a corrupt frozen spec must not crash a read
        fail(f"frozen spec cannot be re-evaluated: {exc}")
        return {"verified": False, "errors": errors, "checks": checks}
    if result.get("satisfaction_set") != sorted(sat, key=state_sort_key):
        fail("frozen satisfaction set differs from recomputation")
    initial_satisfied = initial in sat
    if result.get("initial_satisfied") is not initial_satisfied:
        fail("frozen conclusion differs from recomputation")
    if result.get("iterations") != iterations:
        fail("frozen fixpoint iterations differ from recomputation")
    if result.get("formula") != spec.get("formula") or result.get("initial") != initial:
        fail("frozen result header differs from the frozen spec")

    expected_polarity = SAT if initial_satisfied else UNSAT
    if record.get("polarity") != expected_polarity:
        fail("audit polarity does not match the frozen conclusion")

    # Evidence graph: edges, propositions, endpoints, environments, rounds.
    evidence = record.get("evidence")
    if not isinstance(evidence, list):
        fail("evidence is not a list")
        evidence = []
    verifier = _EvidenceVerifier(pos, model, transitions, checks, fail)
    nodes_by_id = verifier.check_nodes(evidence)

    # Root conclusion.
    root = nodes_by_id.get(record.get("root"))
    if root is None:
        fail("root node is missing from the evidence")
    else:
        if root.get("position") != position_id(0):
            fail("root node is not at the formula root position")
        if root.get("location") != initial:
            fail("root node is not at the frozen initial location")
        if root.get("polarity") != expected_polarity:
            fail("root polarity does not match the frozen conclusion")
        if root.get("env") != {}:
            fail("root node must have an empty binding environment")

    return {"verified": not errors, "errors": errors, "checks": checks}
