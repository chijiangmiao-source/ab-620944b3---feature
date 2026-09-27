"""Construct and independently verify position-indexed dependency evidence.

The instrumented evaluator re-runs the global fixpoint semantics from
``app.checker`` while recording one *context* per evaluation of every
syntactic occurrence: its stable position id, the truth set at that
occurrence, the approximant frame in force, and the child contexts it
depends on.  Fixpoint binders additionally record one approximant
context per iteration step, with the entering binding frame and the
body context that produced the step.

From these frozen facts a directed proof graph is built for the initial
location:

* proposition / boolean / modal nodes reference exact position ids and
  identified transitions;
* μ variable references always cite the strictly preceding approximant
  step (finite unfolding from the empty set);
* a positive ν variable evaluated inside the stable iteration adds one
  admissible *fold* edge closing on the stable approximant
  (coinductive closure); ν falsity unfolds finitely and never folds;
* an unsatisfied initial position yields a proof of the opposite
  polarity, never a forged satisfaction DAG.

``verify_bundle`` trusts nothing the producer did: it re-parses the
frozen formula, rebuilds the position catalogue, re-derives every
frozen truth set from the frozen model, checks every approximant chain
and binding frame (nested environments included), and then validates
every edge of the proof graph and the root conclusion.
"""
from __future__ import annotations

from .formula import And, Dia, Box, Mu, Nu, Not, Or, Prop, Var, parse
from .positions import PositionTable, label_positions, positions_from_json, positions_to_json
from .checker import state_sort_key


class EvidenceError(ValueError):
    """Evidence could not be built, frozen, or independently closed."""


# --- instrumented evaluation ----------------------------------------------


def _sorted_states(states):
    return sorted(states, key=state_sort_key)


def evaluate_with_evidence(ast, table, model):
    """Evaluate ``ast`` recording every occurrence context.

    Returns ``(sat, iterations, facts)`` where ``iterations`` matches
    ``app.checker.evaluate`` chronology exactly and ``facts`` is the
    JSON-serialisable frozen evidence base.
    """
    contexts: list[dict] = []
    by_cid: dict[str, dict] = {}
    approx_cids: dict[tuple, str] = {}
    rounds: dict[str, int] = {}
    counter = 0

    def new_cid() -> str:
        nonlocal counter
        cid = f"c{counter}"
        counter += 1
        return cid

    def put(ctx: dict) -> str:
        contexts.append(ctx)
        by_cid[ctx["cid"]] = ctx
        return ctx["cid"]

    def frame_json(frame) -> list:
        return [[bpos, r, step] for bpos, r, step in frame]

    def add_eval(pos, states, frame, children) -> str:
        return put(
            {
                "cid": new_cid(),
                "kind": "eval",
                "pos": pos,
                "states": _sorted_states(states),
                "frame": frame_json(frame),
                "children": list(children),
            }
        )

    def add_approx(node, pos, rnd, step, states, entry_frame, body_cid) -> str:
        cid = new_cid()
        put(
            {
                "cid": cid,
                "kind": "approx",
                "pos": pos,
                "op": "mu" if isinstance(node, Mu) else "nu",
                "binder": node.var,
                "round": rnd,
                "step": step,
                "states": _sorted_states(states),
                "frame": frame_json(entry_frame),
                "body_cid": body_cid,
            }
        )
        approx_cids[(pos, rnd, step)] = cid
        return cid

    universe = frozenset(model.states)

    def ev(node, frame):
        pos = table.node_pos[id(node)]

        if isinstance(node, Prop):
            states = frozenset(s for s in model.states if node.name in model.props[s])
            return add_eval(pos, states, frame, [])

        if isinstance(node, Var):
            bpos = table.meta[pos]["binder_pos"]
            rnd, fstep = _frame_lookup(frame, bpos)
            bound = by_cid[approx_cids[(bpos, rnd, fstep)]]
            return add_eval(pos, frozenset(bound["states"]), frame, [])

        if isinstance(node, Not):
            child = ev(node.child, frame)
            states = universe - frozenset(by_cid[child]["states"])
            return add_eval(pos, states, frame, [child])

        if isinstance(node, And):
            left = ev(node.left, frame)
            right = ev(node.right, frame)
            states = frozenset(by_cid[left]["states"]) & frozenset(by_cid[right]["states"])
            return add_eval(pos, states, frame, [left, right])

        if isinstance(node, Or):
            left = ev(node.left, frame)
            right = ev(node.right, frame)
            states = frozenset(by_cid[left]["states"]) | frozenset(by_cid[right]["states"])
            return add_eval(pos, states, frame, [left, right])

        if isinstance(node, Dia):
            child = ev(node.child, frame)
            inner = frozenset(by_cid[child]["states"])
            states = frozenset(s for s in model.states if model.succ[s] & inner)
            return add_eval(pos, states, frame, [child])

        if isinstance(node, Box):
            child = ev(node.child, frame)
            inner = frozenset(by_cid[child]["states"])
            states = frozenset(s for s in model.states if model.succ[s] <= inner)
            return add_eval(pos, states, frame, [child])

        if isinstance(node, (Mu, Nu)):
            rnd = rounds.get(pos, 0) + 1
            rounds[pos] = rnd
            current = frozenset() if isinstance(node, Mu) else universe
            step = 0
            add_approx(node, pos, rnd, step, current, frame, None)
            while True:
                body_frame = frame + [(pos, rnd, step)]
                body_cid = ev(node.body, body_frame)
                nxt = frozenset(by_cid[body_cid]["states"])
                step += 1
                last_cid = add_approx(node, pos, rnd, step, nxt, frame, body_cid)
                if nxt == current:
                    return last_cid
                current = nxt

        raise TypeError(f"unknown syntax node {node!r}")

    root_cid = ev(ast, [])
    sat = frozenset(by_cid[root_cid]["states"])

    chain_end: dict[tuple, int] = {}
    for c in contexts:
        if c["kind"] == "approx":
            key = (c["pos"], c["round"])
            chain_end[key] = max(chain_end.get(key, 0), c["step"])
    stable = [[bpos, rnd, last] for (bpos, rnd), last in sorted(chain_end.items())]

    iterations = [
        {
            "binder": c["binder"],
            "op": c["op"],
            "round": c["round"],
            "step": c["step"],
            "states": list(c["states"]),
        }
        for c in contexts
        if c["kind"] == "approx"
    ]
    facts = {"root_cid": root_cid, "contexts": contexts, "stable": stable}
    return sat, iterations, facts


def _frame_lookup(frame, binder_pos):
    for bpos, rnd, step in reversed(frame):
        if bpos == binder_pos:
            return rnd, step
    raise EvidenceError(f"no binding frame for position {binder_pos}")


# --- proof graph construction ---------------------------------------------


def build_proof(facts: dict, positions: list, model, initial: str, polarity: str) -> dict:
    """Construct the directed proof graph for the initial location.

    ``polarity`` "+" affirms the root at the initial location; "-" is a
    rejection proof of the opposite polarity.  It must match the frozen
    truth.  Nodes are discovered on demand, so the graph contains only
    the edges needed to close the root claim.
    """
    meta = positions_from_json(positions)
    by_cid = {c["cid"]: c for c in facts["contexts"]}
    approx_step: dict[tuple, dict] = {}
    body_of: dict[tuple, str] = {}
    for c in facts["contexts"]:
        if c["kind"] == "approx":
            approx_step[(c["pos"], c["round"], c["step"])] = c
            if c["body_cid"] is not None:
                body_of[(c["pos"], c["round"], c["step"])] = c["body_cid"]
    stable = {(bpos, r): n for bpos, r, n in facts["stable"]}
    universe = set(model.states)
    outgoing = {s: list(model.edges[s]) for s in model.states}

    nodes: list[dict] = []
    edges: list[dict] = []
    key_to_id: dict = {}
    queue: list[tuple] = []

    def intern(key) -> str:
        if key not in key_to_id:
            key_to_id[key] = f"n{len(key_to_id)}"
            queue.append(key)
        return key_to_id[key]

    root_ctx = by_cid[facts["root_cid"]]
    if root_ctx["kind"] == "approx":
        root_key = (
            "A", root_ctx["pos"], root_ctx["round"], root_ctx["step"],
            initial, polarity,
        )
    else:
        root_key = ("E", root_ctx["cid"], initial, polarity)
    intern(root_key)

    while queue:
        key = queue.pop(0)
        nid = key_to_id[key]
        if key[0] == "E":
            _, cid, state, pol = key
            node = _build_eval_node(
                nid, cid, state, pol, meta, by_cid, approx_step, stable, universe,
                model, outgoing, intern, edges,
            )
        else:
            _, bpos, rnd, step, state, pol = key
            node = _build_approx_node(
                nid, bpos, rnd, step, state, pol, approx_step, body_of, universe,
                by_cid, intern, edges,
            )
        nodes.append(node)

    return {
        "root": key_to_id[root_key],
        "nodes": sorted(nodes, key=lambda n: int(n["id"][1:])),
        "edges": edges,
    }


def _flip(pol):
    return "-" if pol == "+" else "+"


def _assert_polarity(nid, states, state, pol):
    if (state in set(states)) != (pol == "+"):
        raise EvidenceError(f"node {nid}: polarity {pol} contradicts frozen truth")


def _build_eval_node(nid, cid, state, pol, meta, by_cid, approx_step, stable,
                     universe, model, outgoing, intern, edges):
    ctx = by_cid.get(cid)
    if ctx is None or ctx["kind"] != "eval":
        raise EvidenceError(f"node {nid}: context {cid} is not an evaluation context")
    if state not in universe:
        raise EvidenceError(f"node {nid}: state {state!r} is not declared")
    _assert_polarity(nid, ctx["states"], state, pol)
    pos = ctx["pos"]
    pkind = meta[pos]["kind"]
    node = {
        "id": nid, "type": pkind, "pos": pos, "cid": cid,
        "state": state, "polarity": pol,
    }
    children = ctx["children"]

    if pkind == "prop":
        node["name"] = meta[pos]["name"]
        holds = node["name"] in model.props[state]
        if holds != (pol == "+"):
            raise EvidenceError(
                f"node {nid}: proposition {node['name']!r} at {state} "
                f"does not anchor polarity {pol}"
            )

    elif pkind == "var":
        node["name"] = meta[pos]["name"]
        bpos = meta[pos]["binder_pos"]
        rnd, frame_step = _frame_entry(ctx["frame"], bpos)
        target_step, fold = _resolve_var_step(
            bpos, rnd, frame_step, stable, approx_step, meta, pol,
        )
        edge = {
            "source": nid,
            "target": intern(("A", bpos, rnd, target_step, state, pol)),
        }
        if fold:
            edge["fold"] = True
        edges.append(edge)

    elif pkind == "not":
        edges.append(
            {"source": nid,
             "target": intern(_child_key(by_cid, children[0], state, _flip(pol)))}
        )

    elif pkind in ("and", "or"):
        _boolean_edges(nid, pkind, pol, state, children, by_cid, intern, edges)

    elif pkind in ("dia", "box"):
        _modal_edges(nid, pkind, pol, state, children[0], outgoing, by_cid,
                     intern, edges)

    else:
        raise EvidenceError(f"node {nid}: unexpected evaluation kind {pkind}")

    return node


def _child_key(by_cid, cid, state, pol):
    """Interning key for a child occurrence: binder children are their
    stabilised approximant contexts, everything else an evaluation."""
    ctx = by_cid[cid]
    if ctx["kind"] == "approx":
        return ("A", ctx["pos"], ctx["round"], ctx["step"], state, pol)
    return ("E", cid, state, pol)


def _build_approx_node(nid, bpos, rnd, step, state, pol, approx_step, body_of,
                       universe, by_cid, intern, edges):
    approx = approx_step.get((bpos, rnd, step))
    if approx is None:
        raise EvidenceError(
            f"node {nid}: approximant {bpos} round {rnd} step {step} missing"
        )
    if state not in universe:
        raise EvidenceError(f"node {nid}: state {state!r} is not declared")
    _assert_polarity(nid, approx["states"], state, pol)
    node = {
        "id": nid,
        "type": "approx",
        "pos": bpos,
        "op": approx["op"],
        "binder": approx["binder"],
        "round": rnd,
        "step": step,
        "state": state,
        "polarity": pol,
    }
    if step > 0:
        body_cid = body_of.get((bpos, rnd, step))
        if body_cid is None:
            raise EvidenceError(f"node {nid}: step {step} has no body context")
        edges.append(
            {"source": nid,
             "target": intern(_child_key(by_cid, body_cid, state, pol))}
        )
    return node


def _frame_entry(frame, bpos):
    for fbpos, rnd, step in reversed(frame):
        if fbpos == bpos:
            return rnd, step
    raise EvidenceError(f"binding frame does not contain {bpos}")


def _resolve_var_step(bpos, rnd, frame_step, stable, approx_step, meta, pol):
    """Approximant a variable occurrence may cite.

    The default is the exact approximant bound in the occurrence's
    frame, which is always the strictly preceding iteration step.  A
    positive ν occurrence evaluated inside the final iteration may fold
    onto the stable approximant; μ never folds and nothing folds under
    negative use.
    """
    n_stable = stable.get((bpos, rnd))
    if (
        meta[bpos]["kind"] == "nu"
        and pol == "+"
        and n_stable is not None
        and frame_step == n_stable - 1
        and list(approx_step[(bpos, rnd, n_stable)]["states"])
        == list(approx_step[(bpos, rnd, n_stable - 1)]["states"])
    ):
        return n_stable, True
    return frame_step, False


def _boolean_edges(nid, pkind, pol, state, children, by_cid, intern, edges):
    left, right = children
    in_left = state in set(by_cid[left]["states"])
    in_right = state in set(by_cid[right]["states"])

    if pkind == "and":
        if pol == "+":
            if not (in_left and in_right):
                raise EvidenceError(f"node {nid}: conjunction cannot anchor both sides")
            chosen = [(left, "+"), (right, "+")]
        else:
            if in_left and in_right:
                raise EvidenceError(f"node {nid}: conjunction holds, cannot refute")
            chosen = [(left if not in_left else right, "-")]
    else:
        if pol == "+":
            if not (in_left or in_right):
                raise EvidenceError(f"node {nid}: disjunction fails, cannot affirm")
            chosen = [(left if in_left else right, "+")]
        else:
            if in_left or in_right:
                raise EvidenceError(f"node {nid}: disjunction holds, cannot refute")
            chosen = [(left, "-"), (right, "-")]

    for child_cid, cpol in chosen:
        edges.append(
            {"source": nid, "target": intern(_child_key(by_cid, child_cid, state, cpol))}
        )


def _modal_edges(nid, pkind, pol, state, child_cid, outgoing, by_cid, intern, edges):
    inner = set(by_cid[child_cid]["states"])
    transitions = outgoing.get(state, [])

    def add(tid, dst, cpol):
        edges.append(
            {
                "source": nid,
                "target": intern(_child_key(by_cid, child_cid, dst, cpol)),
                "transition": tid,
            }
        )

    if pkind == "dia":
        if pol == "+":
            witness = next(((t, d) for t, d in transitions if d in inner), None)
            if witness is None:
                raise EvidenceError(f"node {nid}: <> has no witnessing transition")
            add(witness[0], witness[1], "+")
        else:
            inward = next(((t, d) for t, d in transitions if d in inner), None)
            if inward is not None:
                raise EvidenceError(
                    f"node {nid}: <> refuted but transition {inward[0]} leads inward"
                )
            for tid, dst in transitions:
                add(tid, dst, "-")
    else:
        if pol == "+":
            escaping = next(((t, d) for t, d in transitions if d not in inner), None)
            if escaping is not None:
                raise EvidenceError(
                    f"node {nid}: [] affirmed but transition {escaping[0]} escapes"
                )
            for tid, dst in transitions:
                add(tid, dst, "+")
        else:
            witness = next(((t, d) for t, d in transitions if d not in inner), None)
            if witness is None:
                raise EvidenceError(f"node {nid}: [] has no escaping transition")
            add(witness[0], witness[1], "-")


# --- independent verification ---------------------------------------------


def verify_bundle(*, proof, facts, positions, model, initial, polarity,
                  formula_text) -> None:
    """Independently verify a frozen audit bundle.

    Raises EvidenceError on the first defect; returns silently when the
    position catalogue, every frozen truth set, all approximant chains
    and binding frames, every proof edge and the root conclusion hold.
    """
    # 1. The frozen formula re-parses and re-labels identically.
    fresh = label_positions(parse(formula_text))
    frozen_meta = positions_from_json(positions)
    frozen_table = PositionTable(
        node_pos={}, meta=frozen_meta,
        order=tuple(e["pos"] for e in positions),
    )
    if positions_to_json(frozen_table) != positions_to_json(fresh):
        raise EvidenceError("frozen position catalogue does not match the frozen formula")
    meta = fresh.meta

    # 2. Index the frozen base and validate it structurally.
    by_cid: dict[str, dict] = {}
    for c in facts.get("contexts", []):
        cid = c.get("cid")
        if not isinstance(cid, str) or cid in by_cid:
            raise EvidenceError("missing or duplicate context id")
        by_cid[cid] = c
    if facts.get("root_cid") not in by_cid:
        raise EvidenceError("root context is missing from the frozen base")

    approx_step: dict[tuple, dict] = {}
    for c in by_cid.values():
        if c.get("kind") == "approx":
            key = (c["pos"], c["round"], c["step"])
            if key in approx_step:
                raise EvidenceError(f"duplicate approximant {key}")
            approx_step[key] = c
    stable = {(bpos, r): n for bpos, r, n in facts.get("stable", [])}

    _check_contexts(meta, by_cid, approx_step, facts, model, stable)

    # 3. The proof graph itself.
    _verify_graph(
        proof, meta, by_cid, approx_step, stable, model, initial, polarity, facts,
        root_pos=positions[0]["pos"],
    )


def _ancestor_binders(meta, pos):
    """Lexically enclosing binder positions, outermost first."""
    chain = []
    p = meta[pos]["parent"]
    while p is not None:
        if meta[p]["kind"] in ("mu", "nu"):
            chain.append(p)
        p = meta[p]["parent"]
    return list(reversed(chain))


def _check_contexts(meta, by_cid, approx_step, facts, model, stable):
    universe = set(model.states)

    # 2a. approximant chains are contiguous, seeded correctly, stabilise,
    #     and every body runs under the strictly preceding approximant.
    chain_max: dict[tuple, int] = {}
    for c in by_cid.values():
        if c.get("kind") != "approx":
            continue
        bpos = c.get("pos")
        if bpos not in meta or meta[bpos]["kind"] not in ("mu", "nu"):
            raise EvidenceError(f"approximant {c.get('cid')} sits at a non-binder position")
        if c.get("binder") != meta[bpos].get("name"):
            raise EvidenceError(f"approximant at {bpos} carries the wrong binder name")
        if c.get("op") != meta[bpos]["kind"]:
            raise EvidenceError(f"approximant at {bpos} disagrees with its operator")
        key = (bpos, c["round"])
        chain_max[key] = max(chain_max.get(key, 0), c["step"])
        if [e[0] for e in c.get("frame", [])] != _ancestor_binders(meta, bpos):
            raise EvidenceError(f"approximant at {bpos} entered through a foreign frame")
        for fbpos, frnd, fstep in c.get("frame", []):
            if (fbpos, frnd, fstep) not in approx_step:
                raise EvidenceError(f"approximant at {bpos} names a missing frame entry")

    for (bpos, rnd), n in stable.items():
        if chain_max.get((bpos, rnd)) != n:
            raise EvidenceError(
                f"stable table for {bpos} round {rnd} is {n}, "
                f"chains end at {chain_max.get((bpos, rnd))}"
            )

    for (bpos, rnd), last in sorted(chain_max.items()):
        kind = meta[bpos]["kind"]
        steps = {
            a["step"]
            for a in by_cid.values()
            if a.get("kind") == "approx" and a["pos"] == bpos and a["round"] == rnd
        }
        if steps != set(range(last + 1)) or last < 1:
            raise EvidenceError(
                f"approximant chain {bpos} round {rnd} is not a 0..k iteration"
            )
        seed = approx_step[(bpos, rnd, 0)]
        expected_seed = [] if kind == "mu" else _sorted_states(universe)
        if list(seed["states"]) != expected_seed:
            raise EvidenceError(
                f"approximant {bpos} round {rnd} seed is not the "
                f"{'empty' if kind == 'mu' else 'full'} set"
            )
        for step in range(1, last + 1):
            a = approx_step[(bpos, rnd, step)]
            body = by_cid.get(a.get("body_cid"))
            if body is None or body.get("kind") not in ("eval", "approx"):
                raise EvidenceError(f"approximant {bpos}/{rnd}/{step} body is missing")
            body_pos = meta[bpos]["children"][0]
            if body.get("pos") != body_pos:
                raise EvidenceError(
                    f"approximant {bpos}/{rnd}/{step} body is not the static body occurrence"
                )
            if body["kind"] == "approx":
                # the body is itself a binder: it must be cited at its own
                # stabilised step, entered under this iteration's frame
                if meta[body_pos]["kind"] not in ("mu", "nu"):
                    raise EvidenceError(f"body at {body_pos} is a non-binder approximant")
                if body["step"] != stable[(body["pos"], body["round"])]:
                    raise EvidenceError(
                        f"binder body at {body_pos} is not its stable approximant"
                    )
            if list(a["states"]) != list(body["states"]):
                raise EvidenceError(
                    f"approximant {bpos}/{rnd}/{step} disagrees with its body truth"
                )
            expect_frame = a["frame"] + [[bpos, rnd, step - 1]]
            if body.get("frame") != expect_frame:
                raise EvidenceError(
                    f"body of {bpos}/{rnd}/{step} is bound to {body.get('frame')}, "
                    f"expected {expect_frame} — nested environments must not cross-talk"
                )
        if list(approx_step[(bpos, rnd, last)]["states"]) != list(
            approx_step[(bpos, rnd, last - 1)]["states"]
        ):
            raise EvidenceError(f"chain {bpos} round {rnd} did not stabilise")

    # 2b. every evaluation context is structurally sound and its truth
    #     re-derived from the frozen model and child contexts.
    arity = {"prop": 0, "var": 0, "not": 1, "dia": 1, "box": 1, "and": 2, "or": 2}
    for ctx in by_cid.values():
        if ctx.get("kind") != "eval":
            continue
        pos = ctx.get("pos")
        if pos not in meta:
            raise EvidenceError(f"context {ctx.get('cid')} names an unknown position")
        kind = meta[pos]["kind"]
        if kind in ("mu", "nu"):
            raise EvidenceError(f"binder position {pos} must be an approximant context")
        frame = ctx.get("frame", [])
        if [e[0] for e in frame] != _ancestor_binders(meta, pos):
            raise EvidenceError(
                f"context at {pos} runs under the wrong binding frame {frame}"
            )
        for fbpos, frnd, fstep in frame:
            if (fbpos, frnd, fstep) not in approx_step:
                raise EvidenceError(f"context at {pos} names a missing frame approximant")
        child_ids = ctx.get("children", [])
        if len(child_ids) != arity[kind]:
            raise EvidenceError(f"context at {pos} has the wrong child arity")
        expected_posses = meta[pos]["children"]
        children = []
        for i, chid in enumerate(child_ids):
            ch = by_cid.get(chid)
            if ch is None:
                raise EvidenceError(f"context at {pos} has a dangling child reference")
            if ch.get("pos") != expected_posses[i]:
                raise EvidenceError(
                    f"context at {pos} child {i} is not the static child occurrence"
                )
            if ch["kind"] == "eval":
                # an ordinary occurrence shares the parent frame
                if ch.get("frame") != frame:
                    raise EvidenceError(
                        f"child occurrence of {pos} changed frame — binding cross-talk"
                    )
            elif ch["kind"] == "approx":
                # a binder child: the referenced approximant must have been
                # entered under the same frame and be the stabilised one
                if meta[ch["pos"]]["kind"] not in ("mu", "nu"):
                    raise EvidenceError(f"child at {ch['pos']} is a non-binder approximant")
                if ch.get("frame") != frame:
                    raise EvidenceError(
                        f"binder child at {ch['pos']} entered through a foreign frame"
                    )
                if ch["step"] != stable[(ch["pos"], ch["round"])]:
                    raise EvidenceError(
                        f"binder child at {ch['pos']} is not its stable approximant"
                    )
            else:
                raise EvidenceError(f"context at {pos} child has an unknown kind")
            children.append(ch)
        states = set(ctx.get("states", []))
        if any(s not in universe for s in states):
            raise EvidenceError(f"context at {pos} names undeclared states")

        if kind == "prop":
            expect = {s for s in model.states if meta[pos]["name"] in model.props[s]}
        elif kind == "var":
            bpos = meta[pos].get("binder_pos")
            if bpos is None:
                raise EvidenceError(f"variable at {pos} has no resolved binder")
            bound = next((e for e in reversed(frame) if e[0] == bpos), None)
            if bound is None:
                raise EvidenceError(f"variable at {pos} is not bound in its frame")
            expect = set(approx_step[tuple(bound)]["states"])
        elif kind == "not":
            expect = set(universe) - set(children[0]["states"])
        elif kind == "and":
            expect = set(children[0]["states"]) & set(children[1]["states"])
        elif kind == "or":
            expect = set(children[0]["states"]) | set(children[1]["states"])
        elif kind == "dia":
            inner = set(children[0]["states"])
            expect = {s for s in model.states if model.succ[s] & inner}
        else:  # box
            inner = set(children[0]["states"])
            expect = {s for s in model.states if model.succ[s] <= inner}

        if states != expect:
            raise EvidenceError(
                f"frozen truth at {ctx['cid']} re-derives as "
                f"{_sorted_states(expect)}, not {_sorted_states(states)}"
            )

    root = by_cid[facts["root_cid"]]
    if root.get("kind") == "approx":
        rkey = (root["pos"], root["round"])
        if root["step"] != stable.get(rkey):
            raise EvidenceError("root approximant is not the stabilised one")
    elif root.get("kind") != "eval":
        raise EvidenceError("root context has an unknown kind")


def _verify_graph(proof, meta, by_cid, approx_step, stable, model, initial,
                  polarity, facts, root_pos):
    nodes = proof.get("nodes")
    edges = proof.get("edges")
    if not isinstance(nodes, list) or not nodes:
        raise EvidenceError("proof has no nodes")
    if not isinstance(edges, list):
        raise EvidenceError("proof edges must be a list")

    node_by_id: dict[str, dict] = {}
    for n in nodes:
        nid = n.get("id")
        if not isinstance(nid, str) or nid in node_by_id:
            raise EvidenceError("duplicate or missing node id")
        node_by_id[nid] = n

    edges_by_src: dict[str, list[dict]] = {}
    seen = set()
    for e in edges:
        src, dst = e.get("source"), e.get("target")
        if src not in node_by_id or dst not in node_by_id:
            raise EvidenceError(f"edge {src}->{dst} references an unknown node")
        token = (src, dst, e.get("transition"), bool(e.get("fold")))
        if token in seen:
            raise EvidenceError(f"duplicate edge {token}")
        seen.add(token)
        edges_by_src.setdefault(src, []).append(e)

    transition_index = {
        tid: (src, dst)
        for src in model.states for tid, dst in model.edges[src]
    }
    out_ids = {s: sorted(tid for tid, _ in model.edges[s]) for s in model.states}

    for n in nodes:
        _verify_node(
            n, node_by_id, edges_by_src, meta, by_cid, approx_step, stable,
            model, transition_index, out_ids,
        )

    _verify_well_founded(node_by_id, edges)
    _verify_root(proof, node_by_id, meta, facts, by_cid, approx_step, stable,
                 initial, polarity, edges_by_src, root_pos)


def _truth_of(n, by_cid, approx_step):
    if n["type"] == "approx":
        return set(approx_step[(n["pos"], n["round"], n["step"])]["states"])
    return set(by_cid[n["cid"]]["states"])


def _verify_node(n, node_by_id, edges_by_src, meta, by_cid, approx_step, stable,
                 model, transition_index, out_ids):
    nid = n.get("id")
    ntype = n.get("type")
    pos = n.get("pos")
    pol = n.get("polarity")
    state = n.get("state")
    outs = edges_by_src.get(nid, [])

    if pos not in meta:
        raise EvidenceError(f"node {nid}: unknown position {pos}")
    if ntype != "approx" and meta[pos]["kind"] != ntype:
        raise EvidenceError(f"node {nid}: type {ntype} disagrees with position kind")
    if state not in set(model.states):
        raise EvidenceError(f"node {nid}: undeclared state {state!r}")
    if pol not in ("+", "-"):
        raise EvidenceError(f"node {nid}: polarity must be + or -")
    if ntype != "approx" and n.get("cid") not in by_cid:
        raise EvidenceError(f"node {nid}: cites a missing evaluation context")
    if ntype != "approx" and by_cid[n["cid"]]["pos"] != pos:
        raise EvidenceError(f"node {nid}: context is for another position")
    if (state in _truth_of(n, by_cid, approx_step)) != (pol == "+"):
        raise EvidenceError(f"node {nid}: polarity contradicts frozen truth")

    if ntype == "prop":
        _require_leaf(nid, outs)
        if n.get("name") != meta[pos]["name"]:
            raise EvidenceError(f"node {nid}: proposition name mismatch")
        holds = meta[pos]["name"] in model.props[state]
        if holds != (pol == "+"):
            raise EvidenceError(f"node {nid}: proposition anchor at {state} is wrong")

    elif ntype == "var":
        _verify_var(nid, n, outs, node_by_id, meta, by_cid, approx_step, stable)

    elif ntype == "not":
        if len(outs) != 1 or "transition" in outs[0]:
            raise EvidenceError(f"node {nid}: negation needs exactly one plain edge")
        tgt = node_by_id[outs[0]["target"]]
        if tgt["state"] != state or tgt["polarity"] != ("-" if pol == "+" else "+"):
            raise EvidenceError(f"node {nid}: negation edge must flip polarity")
        _node_matches_child(tgt, by_cid[by_cid[n["cid"]]["children"][0]], nid)

    elif ntype in ("and", "or"):
        _verify_boolean(nid, ntype, pol, state, outs, n, node_by_id, by_cid)

    elif ntype in ("dia", "box"):
        _verify_modal(nid, ntype, pol, state, outs, n, node_by_id, by_cid,
                      transition_index, out_ids)

    elif ntype == "approx":
        _verify_approx(nid, n, outs, node_by_id, meta, by_cid, approx_step)

    else:
        raise EvidenceError(f"node {nid}: unknown node type {ntype}")


def _require_leaf(nid, outs):
    if outs:
        raise EvidenceError(f"node {nid}: leaf must have no outgoing edges")


def _node_matches_child(node, child_ctx, nid):
    """An edge target must witness exactly the frozen child occurrence:
    an eval node cites the child context; an approx node cites the
    child's stabilised approximant (for binder-occurrence children)."""
    if child_ctx["kind"] == "eval":
        if node["type"] == "approx" or node.get("cid") != child_ctx["cid"]:
            raise EvidenceError(
                f"node {nid}: edge does not target the frozen child occurrence"
            )
    else:
        if node["type"] != "approx" or (
            node["pos"], node["round"], node["step"]
        ) != (child_ctx["pos"], child_ctx["round"], child_ctx["step"]):
            raise EvidenceError(
                f"node {nid}: edge does not target the binder child's stable approximant"
            )


def _verify_var(nid, n, outs, node_by_id, meta, by_cid, approx_step, stable):
    pos = n["pos"]
    pol = n["polarity"]
    if len(outs) != 1:
        raise EvidenceError(f"node {nid}: variable needs exactly one edge")
    edge = outs[0]
    if "transition" in edge:
        raise EvidenceError(f"node {nid}: variable edge must not name a transition")
    if n.get("name") != meta[pos]["name"]:
        raise EvidenceError(f"node {nid}: variable name mismatch")
    bpos = meta[pos].get("binder_pos")
    if bpos is None:
        raise EvidenceError(f"node {nid}: variable has no statically resolved binder")
    _verify_guard_path(meta, pos, bpos)

    ctx = by_cid[n["cid"]]
    bound = next(e for e in reversed(ctx["frame"]) if e[0] == bpos)
    rnd, frame_step = bound[1], bound[2]
    tgt = node_by_id[edge["target"]]
    if tgt["type"] != "approx" or (tgt["pos"], tgt["round"]) != (bpos, rnd):
        raise EvidenceError(f"node {nid}: variable must cite its bound approximant")
    if tgt["state"] != n["state"] or tgt["polarity"] != pol:
        raise EvidenceError(f"node {nid}: variable edge state/polarity drift")

    fold = bool(edge.get("fold"))
    if fold:
        n_stable = stable.get((bpos, rnd))
        if meta[bpos]["kind"] != "nu" or pol != "+":
            raise EvidenceError(f"node {nid}: only positive ν references may fold")
        if n_stable is None or tgt["step"] != n_stable:
            raise EvidenceError(f"node {nid}: fold must close on the stable approximant")
        if frame_step != n_stable - 1:
            raise EvidenceError(f"node {nid}: fold originates outside the stable step")
    else:
        if tgt["step"] != frame_step:
            raise EvidenceError(
                f"node {nid}: reference must cite the frame's exact approximant step "
                f"{frame_step}, got {tgt['step']}"
            )
        if meta[bpos]["kind"] == "mu" and tgt["step"] > frame_step:
            raise EvidenceError(f"node {nid}: μ may only cite strictly earlier steps")


def _verify_guard_path(meta, var_pos, binder_pos):
    p = meta[var_pos]["parent"]
    guarded = False
    while p is not None:
        if meta[p]["kind"] in ("dia", "box"):
            guarded = True
        if p == binder_pos:
            if not guarded:
                raise EvidenceError(
                    f"variable at {var_pos} reaches {binder_pos} without a modality"
                )
            return
        p = meta[p]["parent"]
    raise EvidenceError(f"variable at {var_pos} is not under binder {binder_pos}")


def _verify_boolean(nid, ntype, pol, state, outs, n, node_by_id, by_cid):
    ctx = by_cid[n["cid"]]
    children = [by_cid[c] for c in ctx["children"]]
    in_left = state in set(children[0]["states"])
    in_right = state in set(children[1]["states"])
    if pol == "+":
        expected = 2 if ntype == "and" else 1
    else:
        expected = 1 if ntype == "and" else 2
    if len(outs) != expected:
        raise EvidenceError(f"node {nid}: {ntype} {pol} requires {expected} edge(s)")

    matched = set()
    for i, e in enumerate(outs):
        if "transition" in e:
            raise EvidenceError(f"node {nid}: boolean edge must not name a transition")
        tgt = node_by_id[e["target"]]
        if tgt["state"] != state or tgt["polarity"] != pol:
            raise EvidenceError(f"node {nid}: boolean edge state/polarity drift")
        child_index = None
        for j, child_ctx in enumerate(children):
            try:
                _node_matches_child(tgt, child_ctx, nid)
            except EvidenceError:
                continue
            child_index = j
            break
        if child_index is None:
            raise EvidenceError(f"node {nid}: edge leaves the frozen child positions")
        matched.add(child_index)
        side_truth = state in set(children[child_index]["states"])
        if ntype == "and" and pol == "-" and side_truth:
            raise EvidenceError(f"node {nid}: and- must cite a failing side")
        if ntype == "or" and pol == "+" and not side_truth:
            raise EvidenceError(f"node {nid}: or+ must cite a holding side")
    if expected == 2 and matched != {0, 1}:
        raise EvidenceError(f"node {nid}: both children must be cited")
    # polarity anchor re-derived from the children
    holds = (in_left and in_right) if ntype == "and" else (in_left or in_right)
    if holds != (pol == "+"):
        raise EvidenceError(f"node {nid}: boolean polarity contradicts child truths")


def _verify_modal(nid, ntype, pol, state, outs, n, node_by_id, by_cid,
                  transition_index, out_ids):
    ctx = by_cid[n["cid"]]
    child_ctx = by_cid[ctx["children"][0]]
    inner = set(child_ctx["states"])
    cited = []
    for e in outs:
        tid = e.get("transition")
        if tid is None:
            raise EvidenceError(f"node {nid}: modal edge must name a transition")
        if tid not in transition_index:
            raise EvidenceError(f"node {nid}: unknown transition {tid!r}")
        src, dst = transition_index[tid]
        if src != state:
            raise EvidenceError(f"node {nid}: transition {tid} leaves {src}, not {state}")
        tgt = node_by_id[e["target"]]
        _node_matches_child(tgt, child_ctx, nid)
        if tgt["state"] != dst:
            raise EvidenceError(
                f"node {nid}: edge endpoint state {tgt['state']} != transition target {dst}"
            )
        cited.append((tid, dst))

    must_cover_all = (ntype == "dia" and pol == "-") or (ntype == "box" and pol == "+")
    if must_cover_all:
        if sorted(t for t, _ in cited) != out_ids[state]:
            raise EvidenceError(
                f"node {nid}: must cite every outgoing transition exactly once"
            )
    elif len(cited) != 1:
        raise EvidenceError(f"node {nid}: needs exactly one witnessing edge")

    for tid, dst in cited:
        inward = dst in inner
        if ntype == "dia":
            if pol == "+" and not inward:
                raise EvidenceError(f"node {nid}: <> witness {tid} does not enter the set")
            if pol == "-" and inward:
                raise EvidenceError(f"node {nid}: <> edge {tid} leads inward")
        else:
            if pol == "+" and not inward:
                raise EvidenceError(f"node {nid}: [] edge {tid} leaves the set")
            if pol == "-" and inward:
                raise EvidenceError(f"node {nid}: [] refutation witness {tid} stays inward")


def _verify_approx(nid, n, outs, node_by_id, meta, by_cid, approx_step):
    bpos, rnd, step = n["pos"], n["round"], n["step"]
    a = approx_step.get((bpos, rnd, step))
    if a is None:
        raise EvidenceError(f"node {nid}: approximant is not in the frozen base")
    if n.get("op") != a["op"] or n.get("binder") != a["binder"]:
        raise EvidenceError(f"node {nid}: approximant header mismatch")
    if step == 0:
        if outs:
            raise EvidenceError(f"node {nid}: seed approximant must have no edges")
        truth = n["state"] in set(a["states"])
        if a["op"] == "nu" and not truth:
            raise EvidenceError(f"node {nid}: ν seed must be the full universe")
        if a["op"] == "mu" and truth:
            raise EvidenceError(f"node {nid}: μ seed must be empty")
        return
    if len(outs) != 1:
        raise EvidenceError(f"node {nid}: approximant step needs exactly one body edge")
    tgt = node_by_id[outs[0]["target"]]
    _node_matches_child(tgt, by_cid[a["body_cid"]], nid)
    if tgt["state"] != n["state"] or tgt["polarity"] != n["polarity"]:
        raise EvidenceError(f"node {nid}: body edge state/polarity drift")


def _verify_well_founded(node_by_id, edges):
    """Non-fold edges must form a DAG; a fold edge may only go from a
    variable node to an approximant node (admissibility of the fold is
    checked at the variable node)."""
    fold_pairs = {(e["source"], e["target"]) for e in edges if e.get("fold")}
    adj: dict[str, set] = {}
    for e in edges:
        if (e["source"], e["target"]) in fold_pairs:
            continue
        adj.setdefault(e["source"], set()).add(e["target"])

    # iterative three-colour DFS (proof graphs can be deep)
    color: dict[str, int] = {}
    for start in node_by_id:
        if color.get(start, 0) != 0:
            continue
        stack = [(start, iter(adj.get(start, ())))]
        color[start] = 1
        while stack:
            v, it = stack[-1]
            advanced = False
            for w in it:
                if color.get(w) == 1:
                    raise EvidenceError(f"non-fold evidence cycle through {v}->{w}")
                if color.get(w, 0) == 0:
                    color[w] = 1
                    stack.append((w, iter(adj.get(w, ()))))
                    advanced = True
                    break
            if not advanced:
                color[v] = 2
                stack.pop()

    for src, dst in fold_pairs:
        s, t = node_by_id[src], node_by_id[dst]
        if s["type"] != "var" or t["type"] != "approx":
            raise EvidenceError("fold edges may only run from a variable to an approximant")


def _verify_root(proof, node_by_id, meta, facts, by_cid, approx_step, stable,
                 initial, polarity, edges_by_src, root_pos):
    root_id = proof.get("root")
    root = node_by_id.get(root_id)
    if root is None:
        raise EvidenceError("proof root does not exist")
    if root["pos"] != root_pos:
        raise EvidenceError("root node does not sit at the root formula position")
    if root["state"] != initial:
        raise EvidenceError("root node is not anchored at the initial location")
    if root["polarity"] != polarity:
        raise EvidenceError("root polarity does not match the frozen conclusion")

    root_ctx = by_cid[facts["root_cid"]]
    if root_ctx["kind"] == "approx":
        if (root["pos"], root["round"], root["step"]) != (
            root_ctx["pos"], root_ctx["round"], root_ctx["step"]
        ):
            raise EvidenceError("root node is not the frozen root approximant")
        truth_set = set(root_ctx["states"])
    else:
        if root.get("cid") != facts["root_cid"]:
            raise EvidenceError("root node does not evaluate the frozen root context")
        truth_set = set(root_ctx["states"])
    if (initial in truth_set) != (polarity == "+"):
        raise EvidenceError("root conclusion contradicts the frozen satisfaction set")

    # every node must be reachable from the root — no detached forged pieces
    seen = {root_id}
    stack = [root_id]
    while stack:
        for e in edges_by_src.get(stack.pop(), ()):
            if e["target"] not in seen:
                seen.add(e["target"])
                stack.append(e["target"])
    if seen != set(node_by_id):
        raise EvidenceError("proof contains nodes unreachable from the root")
