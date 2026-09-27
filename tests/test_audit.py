"""Audit tests: frozen snapshots, stable positions, dependency evidence."""
import copy
import json
import sqlite3

import pytest
from fastapi.testclient import TestClient

from app.evidence import build_audit, verify_audit
from app.main import create_app
from app.store import ReviewStore

SAFETY_LOOPS = {
    "locations": ["s1", "s2"],
    "initial": "s1",
    "propositions": {"s1": ["safe"], "s2": ["safe"]},
    "transitions": [
        {"id": "t1", "source": "s1", "target": "s1"},
        {"id": "t2", "source": "s1", "target": "s2"},
        {"id": "t3", "source": "s2", "target": "s2"},
    ],
    "formula": "νX.(safe & []X)",
}

REACH = {
    "locations": ["s1", "s2", "s3"],
    "initial": "s1",
    "propositions": {"s3": ["goal"]},
    "transitions": [
        {"id": "a", "source": "s1", "target": "s2"},
        {"id": "b", "source": "s2", "target": "s3"},
        {"id": "c", "source": "s3", "target": "s3"},
    ],
    "formula": "μX.(goal | <>X)",
}

DANGER = {
    "locations": ["s1", "s2", "s3"],
    "initial": "s1",
    "propositions": {"s1": ["safe"], "s2": ["safe"]},
    "transitions": [
        {"id": "t1", "source": "s1", "target": "s2"},
        {"id": "t2", "source": "s2", "target": "s2"},
        {"id": "t3", "source": "s1", "target": "s3"},  # danger: s3 not safe
        {"id": "t4", "source": "s3", "target": "s3"},
    ],
    "formula": "νX.(safe & []X)",
}

NESTED_SAT = {
    "locations": ["a", "b"],
    "initial": "a",
    "propositions": {"a": ["p"]},
    "transitions": [
        {"id": "u1", "source": "a", "target": "b"},
        {"id": "u2", "source": "b", "target": "b"},
        {"id": "u3", "source": "b", "target": "a"},
    ],
    "formula": "νX.(<>X & μY.(p | <>Y))",
}

NESTED_UNSAT = {
    "locations": ["a", "b"],
    "initial": "a",
    "propositions": {"a": ["p"]},
    "transitions": [
        {"id": "u1", "source": "a", "target": "b"},
        {"id": "u2", "source": "b", "target": "b"},
    ],
    "formula": "νX.(<>X & μY.(p | <>Y))",
}


@pytest.fixture()
def client(tmp_path):
    store = ReviewStore(str(tmp_path / "reviews.db"))
    yield TestClient(create_app(store))
    store.close()


def make_review(client, body):
    resp = client.post("/reviews", json=body)
    assert resp.status_code == 201, resp.text
    return resp.json()["id"]


def make_audit(client, review_id):
    resp = client.post(f"/reviews/{review_id}/audits")
    assert resp.status_code == 201, resp.text
    return resp.json()


def get_audit(client, audit_id):
    resp = client.get(f"/audits/{audit_id}")
    assert resp.status_code == 200, resp.text
    return resp.json()


def pos_map(audit):
    return {p["id"]: p for p in audit["positions"]}


def node_map(audit):
    return {n["id"]: n for n in audit["evidence"]}


# --- positions and freezing ---------------------------------------------------


def test_positions_are_stable_per_occurrence(client):
    rid = make_review(client, SAFETY_LOOPS)
    audit = get_audit(client, make_audit(client, rid)["id"])
    assert audit["positions"] == [
        {"id": "f0", "kind": "nu", "parent": None, "enclosing": [],
         "var": "X", "body": "f1"},
        {"id": "f1", "kind": "and", "parent": "f0", "enclosing": ["f0"],
         "left": "f2", "right": "f3"},
        {"id": "f2", "kind": "prop", "parent": "f1", "enclosing": ["f0"],
         "name": "safe"},
        {"id": "f3", "kind": "box", "parent": "f1", "enclosing": ["f0"],
         "child": "f4"},
        {"id": "f4", "kind": "var", "parent": "f3", "enclosing": ["f0"],
         "name": "X", "binder": "f0"},
    ]


def test_audit_freezes_spec_formula_initial_and_conclusion(client):
    rid = make_review(client, SAFETY_LOOPS)
    review = client.get(f"/reviews/{rid}").json()
    audit = get_audit(client, make_audit(client, rid)["id"])
    assert audit["frozen"]["spec"] == review["spec"]
    assert audit["frozen"]["result"] == review["result"]
    assert audit["frozen"]["spec"]["formula"] == SAFETY_LOOPS["formula"]
    assert audit["frozen"]["spec"]["initial"] == "s1"
    assert audit["frozen"]["result"]["initial_satisfied"] is True


def test_audits_are_shareable_content_addressed(client):
    rid = make_review(client, SAFETY_LOOPS)
    first = get_audit(client, make_audit(client, rid)["id"])
    second = get_audit(client, make_audit(client, rid)["id"])
    assert first["id"] != second["id"]  # new audit id per request
    assert first["root"] == second["root"]
    assert [n["id"] for n in first["evidence"]] == [n["id"] for n in second["evidence"]]


# --- satisfaction proofs -------------------------------------------------------


def test_nu_self_loop_proof_cycles_inside_stable_approximant(client):
    rid = make_review(client, SAFETY_LOOPS)
    created = make_audit(client, rid)
    assert created["polarity"] == "sat"
    assert created["verified"] is True
    audit = get_audit(client, created["id"])
    assert audit["verification"]["verified"] is True
    assert audit["verification"]["errors"] == []

    positions, nodes = pos_map(audit), node_map(audit)
    root = nodes[audit["root"]]
    assert root["position"] == "f0"
    assert root["location"] == "s1"
    assert root["polarity"] == "sat"
    assert root["env"] == {}
    assert root["round"] == 1  # ν converges after one full approximant

    binder_nodes = [n for n in audit["evidence"] if positions[n["position"]]["kind"] == "nu"]
    assert {n["round"] for n in binder_nodes} == {1}  # stable approximant only

    var_nodes = [n for n in audit["evidence"] if positions[n["position"]]["kind"] == "var"]
    assert var_nodes, "expected variable occurrence nodes"
    for var in var_nodes:
        assert var["env"] == {"f0": 1}
        (ref,) = var["refs"]
        assert ref["role"] == "binder"
        assert nodes[ref["to"]]["round"] == 1  # cycle closes at the stable round
    # the self loop reaches back to the root binder itself
    assert any(ref["to"] == audit["root"] for var in var_nodes for ref in var["refs"])


def test_mu_proof_references_strictly_earlier_rounds(client):
    rid = make_review(client, REACH)
    audit = get_audit(client, make_audit(client, rid)["id"])
    assert audit["polarity"] == "sat"
    assert audit["verification"]["verified"] is True

    positions, nodes = pos_map(audit), node_map(audit)
    root = nodes[audit["root"]]
    assert positions[root["position"]]["kind"] == "mu"
    assert root["round"] == 3  # s1 enters the μ approximants at round 3

    binder_nodes = [n for n in audit["evidence"] if positions[n["position"]]["kind"] == "mu"]
    assert sorted(n["round"] for n in binder_nodes) == [1, 2, 3]
    for binder in binder_nodes:
        (ref,) = binder["refs"]
        assert ref["role"] == "body"
        body = nodes[ref["to"]]
        # μ unfolds only into the strictly earlier approximant round
        assert body["env"][root["position"]] == binder["round"] - 1

    var_nodes = [n for n in audit["evidence"] if positions[n["position"]]["kind"] == "var"]
    for var in var_nodes:
        (ref,) = var["refs"]
        assert nodes[ref["to"]]["round"] == var["env"][root["position"]]


def test_modal_references_carry_transition_endpoints(client):
    rid = make_review(client, SAFETY_LOOPS)
    audit = get_audit(client, make_audit(client, rid)["id"])
    positions, nodes = pos_map(audit), node_map(audit)
    box_nodes = [n for n in audit["evidence"] if positions[n["position"]]["kind"] == "box"]
    assert box_nodes
    for box in box_nodes:
        for ref in box["refs"]:
            assert ref["role"] == "succ"
            target = nodes[ref["to"]]
            for tid in ref["transitions"]:
                transition = next(
                    t for t in audit["frozen"]["spec"]["transitions"] if t["id"] == tid
                )
                assert transition["source"] == box["location"]
                assert transition["target"] == target["location"]


# --- opposite-polarity refutations ----------------------------------------------


def test_unsat_conclusion_yields_opposite_polarity_refutation(client):
    rid = make_review(client, DANGER)
    assert client.get(f"/reviews/{rid}").json()["result"]["initial_satisfied"] is False
    created = make_audit(client, rid)
    assert created["polarity"] == "unsat"  # refutation, not a forged sat DAG
    audit = get_audit(client, created["id"])
    assert audit["verification"]["verified"] is True

    positions, nodes = pos_map(audit), node_map(audit)
    root = nodes[audit["root"]]
    assert root["polarity"] == "unsat"
    assert root["location"] == "s1"
    assert root["round"] == 2  # s1 leaves the ν approximants at round 2
    # no satisfaction claim is forged anywhere in this evidence
    assert all(n["polarity"] == "unsat" for n in audit["evidence"])

    binder_nodes = [n for n in audit["evidence"] if positions[n["position"]]["kind"] == "nu"]
    assert sorted(n["round"] for n in binder_nodes) == [1, 2]
    for binder in binder_nodes:
        (ref,) = binder["refs"]
        body = nodes[ref["to"]]
        # refuting ν is well-founded: strictly earlier rounds only
        assert body["env"][root["position"]] == binder["round"] - 1


def test_nested_refutation_closes_mu_cycle_only_at_stable_round(client):
    rid = make_review(client, NESTED_UNSAT)
    assert client.get(f"/reviews/{rid}").json()["result"]["initial_satisfied"] is False
    audit = get_audit(client, make_audit(client, rid)["id"])
    assert audit["polarity"] == "unsat"
    assert audit["verification"]["verified"] is True

    positions, nodes = pos_map(audit), node_map(audit)
    mu_nodes = [n for n in audit["evidence"] if positions[n["position"]]["kind"] == "mu"]
    assert mu_nodes
    for mu in mu_nodes:
        assert mu["env"] == {"f0": 0}
        assert mu["round"] == 2  # stable round of the dualised μ
    var_y = [n for n in audit["evidence"] if positions[n["position"]]["kind"] == "var"
             and positions[n["position"]]["name"] == "Y"]
    assert var_y
    for var in var_y:
        assert var["env"] == {"f0": 0, "f4": 2}
        (ref,) = var["refs"]
        assert nodes[ref["to"]]["round"] == 2  # cycle stays inside the stable approximant


# --- nested binding environments -------------------------------------------------


def test_nested_binding_environments_do_not_cross(client):
    rid = make_review(client, NESTED_SAT)
    audit = get_audit(client, make_audit(client, rid)["id"])
    assert audit["polarity"] == "sat"
    assert audit["verification"]["verified"] is True

    positions, nodes = pos_map(audit), node_map(audit)
    for node in audit["evidence"]:
        kind = positions[node["position"]]["kind"]
        if kind == "nu":
            assert node["env"] == {} and node["round"] == 1
        elif kind == "mu":
            assert node["env"] == {"f0": 1}  # inner μ lives under the outer ν round
            assert node["round"] in (1, 2)
        elif kind == "var":
            binder = positions[node["position"]]["binder"]
            assert node["env"][binder] >= 0
            assert set(node["env"]) == set(positions[node["position"]]["enclosing"])
    # the outer ν cycle still closes back at the root
    var_x = [n for n in audit["evidence"] if positions[n["position"]]["kind"] == "var"
             and positions[n["position"]]["name"] == "X"]
    assert any(ref["to"] == audit["root"] for var in var_x for ref in var["refs"])


# --- rejection paths: nothing readable is written ---------------------------------


def test_audit_of_missing_review_is_404(client):
    resp = client.post(f"/reviews/{'0' * 32}/audits")
    assert resp.status_code == 404
    assert "id" not in resp.json()


def test_unknown_audit_id_is_404(client):
    assert client.get(f"/audits/{'0' * 32}").status_code == 404


def test_frozen_recompute_mismatch_409_and_nothing_written(tmp_path):
    db = str(tmp_path / "reviews.db")
    store = ReviewStore(db)
    client = TestClient(create_app(store))
    rid = make_review(client, SAFETY_LOOPS)
    store.close()

    # tamper with the stored conclusion behind the service's back
    conn = sqlite3.connect(db)
    payload = json.loads(
        conn.execute("SELECT payload FROM reviews WHERE id = ?", (rid,)).fetchone()[0]
    )
    payload["result"]["satisfaction_set"] = ["bogus"]
    conn.execute("UPDATE reviews SET payload = ? WHERE id = ?",
                 (json.dumps(payload, ensure_ascii=False), rid))
    conn.commit()
    conn.close()

    store = ReviewStore(db)
    client = TestClient(create_app(store))
    resp = client.post(f"/reviews/{rid}/audits")
    assert resp.status_code == 409
    assert "id" not in resp.json()
    count = store._conn.execute("SELECT COUNT(*) FROM audits").fetchone()[0]
    assert count == 0  # no readable audit was written
    store.close()


# --- independent verification on read ----------------------------------------------


def _tampered(audit, mutate):
    clone = copy.deepcopy(audit)
    mutate(clone)
    return clone


def test_verifier_accepts_json_round_tripped_audit(client):
    rid = make_review(client, SAFETY_LOOPS)
    audit = get_audit(client, make_audit(client, rid)["id"])
    clone = json.loads(json.dumps(audit, ensure_ascii=False))
    assert verify_audit(clone)["verified"] is True


@pytest.mark.parametrize(
    "mutate",
    [
        lambda a: a.update({"polarity": "unsat"}),  # forged polarity
        lambda a: a["evidence"][0].update({"location": "s2"}),  # moved node
        lambda a: a["evidence"][1]["refs"].pop(),  # missing edge
        lambda a: a["evidence"][0].update({"round": 7}),  # bogus round
        lambda a: a["evidence"][2].update({"env": {"f0": 0}}),  # wrong environment
        lambda a: a["frozen"]["result"].update({"initial_satisfied": False}),
        lambda a: a["positions"][0].update({"var": "Z"}),  # position table drift
        lambda a: a.update({"root": a["evidence"][1]["id"]}),  # wrong root
    ],
)
def test_verifier_rejects_tampered_audits(client, mutate):
    rid = make_review(client, SAFETY_LOOPS)
    audit = get_audit(client, make_audit(client, rid)["id"])
    report = verify_audit(_tampered(audit, mutate))
    assert report["verified"] is False
    assert report["errors"]


def test_forged_sat_dag_for_unsat_source_is_rejected(client):
    rid = make_review(client, DANGER)
    audit = get_audit(client, make_audit(client, rid)["id"])

    def forge(a):
        a["polarity"] = "sat"  # pretend the dangerous system is safe
        for node in a["evidence"]:
            node["polarity"] = "sat"

    report = verify_audit(_tampered(audit, forge))
    assert report["verified"] is False
    assert report["errors"]


def test_read_reverifies_and_flags_tampered_stored_audit(tmp_path):
    db = str(tmp_path / "reviews.db")
    store = ReviewStore(db)
    client = TestClient(create_app(store))
    rid = make_review(client, SAFETY_LOOPS)
    aid = make_audit(client, rid)["id"]

    payload = json.loads(
        store._conn.execute("SELECT payload FROM audits WHERE id = ?", (aid,)).fetchone()[0]
    )
    payload["evidence"][0]["location"] = "s2"  # corrupt the stored evidence
    store._conn.execute("UPDATE audits SET payload = ? WHERE id = ?",
                        (json.dumps(payload, ensure_ascii=False), aid))
    store._conn.commit()

    resp = client.get(f"/audits/{aid}")
    assert resp.status_code == 200
    assert resp.json()["verification"]["verified"] is False
    assert resp.json()["verification"]["errors"]
    store.close()


# --- review endpoints keep working -------------------------------------------------


def test_reviews_still_create_and_read(client):
    rid = make_review(client, SAFETY_LOOPS)
    body = client.get(f"/reviews/{rid}").json()
    assert body["result"]["initial_satisfied"] is True
    assert body["result"]["satisfaction_set"] == ["s1", "s2"]
    assert client.get(f"/reviews/{'0' * 32}").status_code == 404
