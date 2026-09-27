"""Dependency-evidence audits: positions, proof graphs, verification.

Covers the acceptance scenarios (ν self-loop fold, μ reachability
unfolding, dangerous-transition rejection proof) and adversarial cases:
forged satisfaction DAGs, rewired edges, μ references to later
approximants, ν folds away from stability, tampered truth sets, binding
frame cross-talk, freeze/recompute inconsistency and missing sources.
"""
import copy
import pytest
from fastapi.testclient import TestClient

from app.audit import AuditFreezeError, produce_audit, reverify_audit
from app.checker import Model, evaluate, state_sort_key
from app.evidence import EvidenceError, build_proof, verify_bundle
from app.formula import parse
from app.main import create_app
from app.positions import label_positions, positions_to_json
from app.store import ReviewStore


SAFETY = "νX.(safe & []X)"
REACH = "μX.(goal | <>X)"


def make_record(locations, initial, props, transitions, formula):
    model = Model.from_spec(locations, props, transitions)
    ast = parse(formula)
    sat, iterations = evaluate(ast, model)
    ordered = sorted(locations, key=state_sort_key)
    spec = {
        "locations": ordered,
        "initial": initial,
        "propositions": {loc: sorted(set(props.get(loc, []))) for loc in ordered},
        "transitions": [
            {"id": tid, "source": src, "target": dst} for tid, src, dst in transitions
        ],
        "formula": formula,
    }
    result = {
        "formula": formula,
        "initial": initial,
        "initial_satisfied": initial in sat,
        "satisfaction_set": sorted(sat, key=state_sort_key),
        "iterations": iterations,
        "converged": True,
    }
    return {"spec": spec, "result": result}, model


def bundle_for(locations, initial, props, transitions, formula):
    record, model = make_record(locations, initial, props, transitions, formula)
    bundle = produce_audit(record)
    return bundle, model, record


# --- position labelling ---------------------------------------------------


def test_positions_are_stable_per_occurrence():
    ast = parse("νX.(safe & []X)")
    t1 = positions_to_json(label_positions(ast))
    t2 = positions_to_json(label_positions(parse("νX.(safe & []X)")))
    assert t1 == t2  # re-parse of the same text yields identical ids
    kinds = [(p["pos"], p["kind"]) for p in t1]
    assert kinds == [
        ("p0", "nu"),
        ("p1", "and"),
        ("p2", "prop"),
        ("p3", "box"),
        ("p4", "var"),
    ]
    var = t1[-1]
    assert var["binder_pos"] == "p0"
    # parent/child wiring is consistent
    by_pos = {p["pos"]: p for p in t1}
    assert by_pos["p4"]["parent"] == "p3"
    assert by_pos["p0"]["children"] == ["p1"]


def test_two_siblings_get_distinct_positions():
    table = positions_to_json(label_positions(parse("(νX.(safe & []X)) | safe")))
    props = [p["pos"] for p in table if p["kind"] == "prop"]
    assert len(props) == 2 and props[0] != props[1]


# --- acceptance scenarios -------------------------------------------------


def test_nu_self_loop_audit_folds_at_stable_approximant():
    bundle, model, _ = bundle_for(
        ["s1", "s2"], "s1",
        {"s1": ["safe"], "s2": ["safe"]},
        [("t1", "s1", "s1"), ("t2", "s1", "s2"), ("t3", "s2", "s2")],
        SAFETY,
    )
    assert bundle["polarity"] == "+"
    proof = bundle["proof"]
    folds = [e for e in proof["edges"] if e.get("fold")]
    assert folds, "the guarded self-loop must close via a ν fold"
    # every fold targets the single stable approximant (round 1, step 1)
    for e in folds:
        tgt = next(n for n in proof["nodes"] if n["id"] == e["target"])
        assert tgt["type"] == "approx" and tgt["op"] == "nu"
        assert (tgt["round"], tgt["step"]) == (1, 1)
    # the fold reaches back to the root conclusion itself
    root = next(n for n in proof["nodes"] if n["id"] == proof["root"])
    assert root["type"] == "approx" and root["state"] == "s1"
    assert any(e["target"] == proof["root"] for e in folds)
    reverify_audit(bundle)


def test_mu_reachability_audit_unfolds_without_folds():
    bundle, _, _ = bundle_for(
        ["s1", "s2", "s3"], "s1", {"s3": ["goal"]},
        [("a", "s1", "s2"), ("b", "s2", "s3"), ("c", "s3", "s3")],
        REACH,
    )
    assert bundle["polarity"] == "+"
    proof = bundle["proof"]
    assert not [e for e in proof["edges"] if e.get("fold")]
    # every variable edge points at the strictly earlier approximant
    # step bound in its evaluation frame
    stable = {(b, r): s for b, r, s in bundle["facts"]["stable"]}
    for n in proof["nodes"]:
        if n["type"] != "var":
            continue
        (edge,) = [e for e in proof["edges"] if e["source"] == n["id"]]
        tgt = next(t for t in proof["nodes"] if t["id"] == edge["target"])
        assert tgt["type"] == "approx"
        # variable contexts during step k bind step k-1
        ctx = next(c for c in bundle["facts"]["contexts"] if c["cid"] == n["cid"])
        bound = next(e for e in reversed(ctx["frame"]) if e[0] == tgt["pos"])
        assert tgt["step"] == bound[2]
        assert tgt["step"] < stable[(tgt["pos"], tgt["round"])]
    reverify_audit(bundle)


def test_dangerous_transition_yields_opposite_polarity_rejection_proof():
    bundle, _, _ = bundle_for(
        ["s1", "s2", "s3"], "s1", {"s1": ["safe"], "s2": ["safe"]},
        [
            ("t1", "s1", "s2"),
            ("t2", "s2", "s2"),
            ("t3", "s1", "s3"),
            ("t4", "s3", "s3"),
        ],
        SAFETY,
    )
    assert bundle["conclusion"]["initial_satisfied"] is False
    assert bundle["polarity"] == "-"
    proof = bundle["proof"]
    assert not [e for e in proof["edges"] if e.get("fold")]
    root = next(n for n in proof["nodes"] if n["id"] == proof["root"])
    assert root["polarity"] == "-"
    # the refutation names the dangerous transition t3
    modal = next(n for n in proof["nodes"] if n["type"] == "box")
    edges = [e for e in proof["edges"] if e["source"] == modal["id"]]
    assert edges[0]["transition"] == "t3"
    reverify_audit(bundle)


def test_nested_binders_do_not_cross_talk():
    bundle, _, _ = bundle_for(
        ["a", "b"], "a", {"a": ["p"]},
        [("t1", "a", "b"), ("t2", "b", "b")],
        "νX.(<>X & μY.(p | <>Y))",
    )
    assert bundle["polarity"] == "-"
    # the inner μ chain was re-entered on every outer ν round and each
    # stabilised independently; the frozen stable table records that
    chains = {(b, r): s for b, r, s in bundle["facts"]["stable"]}
    assert len(chains) >= 3
    reverify_audit(bundle)


def test_sibling_fixpoint_audit():
    bundle, _, _ = bundle_for(
        ["a", "b", "c", "d"], "a",
        {"a": ["safe"], "b": ["safe"], "c": ["goal"]},
        [("t1", "a", "b"), ("t2", "b", "b"), ("t3", "c", "c"), ("t4", "d", "c")],
        "(νX.(safe & []X)) | (μY.(goal | <>Y))",
    )
    assert bundle["polarity"] == "+"
    reverify_audit(bundle)


def test_binder_directly_under_binder_audit():
    # μX.νY.…: the approximant body of the outer binder is the inner
    # binder's own stable approximant, not an evaluation context
    bundle, _, _ = bundle_for(
        ["s1", "s2", "s3"], "s1",
        {"s1": ["safe"], "s2": ["safe"], "s3": ["safe", "goal"]},
        [("t1", "s1", "s2"), ("t2", "s2", "s3"), ("t3", "s3", "s3")],
        "μX.νY.((goal & <>X) | (safe & <>Y))",
    )
    assert bundle["polarity"] == "+"
    reverify_audit(bundle)


def test_alternating_nesting_audit():
    bundle, _, _ = bundle_for(
        ["s1", "s2", "s3"], "s1",
        {"s1": ["safe"], "s2": ["safe"], "s3": ["safe", "goal"]},
        [("t1", "s1", "s2"), ("t2", "s2", "s3"), ("t3", "s3", "s3")],
        "νX.(safe & [](μY.(goal | <>Y) & []X))",
    )
    assert bundle["polarity"] == "+"
    reverify_audit(bundle)


# --- adversarial: tampering must be detected -----------------------------


def _verify(bundle):
    spec = bundle["frozen"]["spec"]
    model = Model.from_spec(
        spec["locations"],
        spec["propositions"],
        [(t["id"], t["source"], t["target"]) for t in spec["transitions"]],
    )
    verify_bundle(
        proof=bundle["proof"], facts=bundle["facts"], positions=bundle["positions"],
        model=model, initial=bundle["conclusion"]["initial"],
        polarity=bundle["polarity"], formula_text=spec["formula"],
    )


def test_forging_positive_dag_for_unsatisfied_position_fails():
    bundle, model, record = bundle_for(
        ["s1", "s2", "s3"], "s1", {"s1": ["safe"], "s2": ["safe"]},
        [
            ("t1", "s1", "s2"), ("t2", "s2", "s2"),
            ("t3", "s1", "s3"), ("t4", "s3", "s3"),
        ],
        SAFETY,
    )
    # try to build a satisfaction proof against the same frozen facts
    with pytest.raises(EvidenceError):
        verify_bundle(
            proof=build_proof(bundle["facts"], bundle["positions"], model, "s1", "+"),
            facts=bundle["facts"], positions=bundle["positions"], model=model,
            initial="s1", polarity="+", formula_text=SAFETY,
        )
    # and merely relabelling the rejection root is caught at the root
    forged = copy.deepcopy(bundle)
    forged["polarity"] = "+"
    forged["proof"]["nodes"][0]["polarity"] = "+"
    with pytest.raises(EvidenceError):
        _verify(forged)


def test_rewiring_edge_to_non_child_position_fails():
    bundle, _, _ = bundle_for(
        ["s1", "s2"], "s1", {"s1": ["safe"], "s2": ["safe"]},
        [("t1", "s1", "s1"), ("t2", "s1", "s2"), ("t3", "s2", "s2")],
        SAFETY,
    )
    tampered = copy.deepcopy(bundle)
    # swap a modal edge's transition for one that does not exist
    box = next(n for n in tampered["proof"]["nodes"] if n["type"] == "box")
    edge = next(e for e in tampered["proof"]["edges"] if e["source"] == box["id"])
    edge["transition"] = "t-missing"
    with pytest.raises(EvidenceError):
        _verify(tampered)


def test_mu_reference_to_later_approximant_fails():
    bundle, _, _ = bundle_for(
        ["s1", "s2", "s3"], "s1", {"s3": ["goal"]},
        [("a", "s1", "s2"), ("b", "s2", "s3"), ("c", "s3", "s3")],
        REACH,
    )
    tampered = copy.deepcopy(bundle)
    var = next(n for n in tampered["proof"]["nodes"] if n["type"] == "var")
    edge = next(e for e in tampered["proof"]["edges"] if e["source"] == var["id"])
    tgt = next(n for n in tampered["proof"]["nodes"] if n["id"] == edge["target"])
    tgt["step"] = 4  # beyond the strictly earlier bound approximant
    with pytest.raises(EvidenceError):
        _verify(tampered)


def test_nu_fold_away_from_stability_fails():
    bundle, _, _ = bundle_for(
        ["s1", "s2"], "s1", {"s1": ["safe"], "s2": ["safe"]},
        [("t1", "s1", "s1"), ("t2", "s1", "s2"), ("t3", "s2", "s2")],
        SAFETY,
    )
    tampered = copy.deepcopy(bundle)
    fold = next(e for e in tampered["proof"]["edges"] if e.get("fold"))
    tgt = next(n for n in tampered["proof"]["nodes"] if n["id"] == fold["target"])
    tgt["step"] = 0  # seed approximant is not the stable one
    with pytest.raises(EvidenceError):
        _verify(tampered)


def test_tampered_frozen_truth_set_fails_rederivation():
    bundle, _, _ = bundle_for(
        ["s1", "s2"], "s1", {"s1": ["safe"], "s2": ["safe"]},
        [("t1", "s1", "s1"), ("t2", "s1", "s2"), ("t3", "s2", "s2")],
        SAFETY,
    )
    tampered = copy.deepcopy(bundle)
    # pretend an unsafe state was safe inside the frozen proposition base
    prop_ctx = next(
        c for c in tampered["facts"]["contexts"]
        if c["kind"] == "eval" and c["pos"] == "p2"
    )
    prop_ctx["states"] = ["s1", "s2", "ghost"]
    with pytest.raises(EvidenceError):
        _verify(tampered)


def test_tampered_transition_endpoint_fails():
    bundle, _, _ = bundle_for(
        ["s1", "s2"], "s1", {"s1": ["safe"], "s2": ["safe"]},
        [("t1", "s1", "s1"), ("t2", "s1", "s2"), ("t3", "s2", "s2")],
        SAFETY,
    )
    tampered = copy.deepcopy(bundle)
    tampered["frozen"]["spec"]["transitions"][0]["target"] = "s2"
    with pytest.raises(EvidenceError):
        _verify(tampered)


def test_position_catalogue_tamper_fails():
    bundle, _, _ = bundle_for(
        ["s1", "s2"], "s1", {"s1": ["safe"], "s2": ["safe"]},
        [("t1", "s1", "s1"), ("t2", "s1", "s2"), ("t3", "s2", "s2")],
        SAFETY,
    )
    tampered = copy.deepcopy(bundle)
    tampered["positions"][2]["kind"] = "var"
    with pytest.raises(EvidenceError):
        _verify(tampered)


def test_binding_frame_cross_talk_fails():
    bundle, _, _ = bundle_for(
        ["a", "b"], "a", {"a": ["p"]},
        [("t1", "a", "b"), ("t2", "b", "b")],
        "νX.(<>X & μY.(p | <>Y))",
    )
    tampered = copy.deepcopy(bundle)
    # corrupt an inner body frame: bump the round it was bound in,
    # simulating nested binding-environment leakage
    approx_ctx = next(
        c for c in tampered["facts"]["contexts"]
        if c["kind"] == "approx" and c["body_cid"] and c["frame"]
    )
    body = next(
        c for c in tampered["facts"]["contexts"]
        if c["cid"] == approx_ctx["body_cid"]
    )
    body["frame"][-1][1] += 1
    with pytest.raises(EvidenceError):
        _verify(tampered)


def test_unreachable_forged_node_fails():
    bundle, _, _ = bundle_for(
        ["s1", "s2"], "s1", {"s1": ["safe"], "s2": ["safe"]},
        [("t1", "s1", "s1"), ("t2", "s1", "s2"), ("t3", "s2", "s2")],
        SAFETY,
    )
    tampered = copy.deepcopy(bundle)
    tampered["proof"]["nodes"].append(
        {"id": "nX", "type": "prop", "pos": "p2", "cid":
         next(c["cid"] for c in tampered["facts"]["contexts"] if c["pos"] == "p2"),
         "state": "s1", "polarity": "+"}
    )
    with pytest.raises(EvidenceError):
        _verify(tampered)


def test_non_fold_cycle_fails():
    bundle, _, _ = bundle_for(
        ["s1", "s2"], "s1", {"s1": ["safe"], "s2": ["safe"]},
        [("t1", "s1", "s1"), ("t2", "s1", "s2"), ("t3", "s2", "s2")],
        SAFETY,
    )
    tampered = copy.deepcopy(bundle)
    prop = next(n for n in tampered["proof"]["nodes"] if n["type"] == "prop")
    root_id = tampered["proof"]["root"]
    tampered["proof"]["edges"].append({"source": prop["id"], "target": root_id})
    with pytest.raises(EvidenceError):
        _verify(tampered)


# --- freeze/recompute -----------------------------------------------------


def test_freeze_recompute_inconsistency_aborts():
    record, _ = make_record(
        ["s1", "s2"], "s1", {"s1": ["safe"], "s2": ["safe"]},
        [("t1", "s1", "s1"), ("t2", "s1", "s2"), ("t3", "s2", "s2")],
        SAFETY,
    )
    # corrupt the stored conclusion after the fact
    record["result"]["satisfaction_set"] = ["s1"]
    record["result"]["initial_satisfied"] = False
    with pytest.raises(AuditFreezeError):
        produce_audit(record)


# --- HTTP surface ---------------------------------------------------------


@pytest.fixture()
def client(tmp_path):
    store = ReviewStore(str(tmp_path / "reviews.db"))
    yield TestClient(create_app(store)), store, tmp_path
    store.close()


def _post_safety_review(client, dangerous=False):
    c, _, _ = client
    body = {
        "locations": ["s1", "s2", "s3"] if dangerous else ["s1", "s2"],
        "initial": "s1",
        "propositions": (
            {"s1": ["safe"], "s2": ["safe"]}
            if dangerous else {"s1": ["safe"], "s2": ["safe"]}
        ),
        "transitions": (
            [
                {"id": "t1", "source": "s1", "target": "s2"},
                {"id": "t2", "source": "s2", "target": "s2"},
                {"id": "t3", "source": "s1", "target": "s3"},
                {"id": "t4", "source": "s3", "target": "s3"},
            ]
            if dangerous
            else [
                {"id": "t1", "source": "s1", "target": "s1"},
                {"id": "t2", "source": "s1", "target": "s2"},
                {"id": "t3", "source": "s2", "target": "s2"},
            ]
        ),
        "formula": SAFETY,
    }
    resp = c.post("/reviews", json=body)
    assert resp.status_code == 201
    return resp.json()["id"]


def test_audit_create_and_read_http(client):
    c, _, _ = client
    rid = _post_safety_review(client)
    created = c.post(f"/reviews/{rid}/audits")
    assert created.status_code == 201, created.text
    body = created.json()
    assert body["review_id"] == rid
    assert body["polarity"] == "+"
    assert body["conclusion"]["initial_satisfied"] is True

    fetched = c.get(f"/audits/{body['id']}")
    assert fetched.status_code == 200
    bundle = fetched.json()
    assert bundle["review_id"] == rid
    assert bundle["frozen"]["spec"]["formula"] == SAFETY
    assert bundle["proof"]["root"]
    assert any(e.get("fold") for e in bundle["proof"]["edges"])
    # positions freeze every syntactic occurrence
    assert [p["kind"] for p in bundle["positions"]] == [
        "nu", "and", "prop", "box", "var",
    ]


def test_rejection_audit_http(client):
    c, _, _ = client
    rid = _post_safety_review(client, dangerous=True)
    created = c.post(f"/reviews/{rid}/audits")
    assert created.status_code == 201
    assert created.json()["polarity"] == "-"
    fetched = c.get(f"/audits/{created.json()['id']}")
    assert fetched.status_code == 200
    root = next(
        n for n in fetched.json()["proof"]["nodes"]
        if n["id"] == fetched.json()["proof"]["root"]
    )
    assert root["polarity"] == "-"


def test_audit_missing_source_is_404_and_writes_nothing(client):
    c, store, tmp_path = client
    resp = c.post("/reviews/" + "0" * 32 + "/audits")
    assert resp.status_code == 404
    assert "id" not in resp.json()
    # no audit row exists
    import sqlite3

    conn = sqlite3.connect(str(tmp_path / "reviews.db"))
    assert conn.execute("SELECT COUNT(*) FROM audits").fetchone()[0] == 0
    conn.close()


def test_unknown_audit_is_404(client):
    c, _, _ = client
    assert c.get("/audits/" + "0" * 32).status_code == 404


def test_tampered_persisted_audit_is_not_served(client):
    c, store, _ = client
    rid = _post_safety_review(client)
    audit_id = c.post(f"/reviews/{rid}/audits").json()["id"]
    # corrupt the persisted payload directly
    import json
    import sqlite3

    conn = store._conn
    row = conn.execute("SELECT payload FROM audits WHERE id = ?", (audit_id,)).fetchone()
    payload = json.loads(row[0])
    for node in payload["proof"]["nodes"]:
        if node["type"] == "prop":
            node["name"] = "tampered"
    conn.execute(
        "UPDATE audits SET payload = ? WHERE id = ?",
        (json.dumps(payload, ensure_ascii=False), audit_id),
    )
    conn.commit()
    assert c.get(f"/audits/{audit_id}").status_code == 409


def test_review_endpoints_still_serve_normative_evidence(client):
    c, _, _ = client
    rid = _post_safety_review(client)
    review = c.get(f"/reviews/{rid}").json()
    assert review["result"]["satisfaction_set"] == ["s1", "s2"]
    assert [i["states"] for i in review["result"]["iterations"]] == [
        ["s1", "s2"], ["s1", "s2"],
    ]
