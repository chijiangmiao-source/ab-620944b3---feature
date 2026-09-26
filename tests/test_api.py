"""API-level tests: creation, persistence, rejection without id."""
import pytest
from fastapi.testclient import TestClient

from app.main import create_app
from app.store import ReviewStore

VALID = {
    "locations": ["s1", "s2"],
    "initial": "s1",
    "propositions": {"s1": ["safe"], "s2": ["safe"]},
    "transitions": [
        {"id": "t1", "source": "s1", "target": "s2"},
        {"id": "t2", "source": "s2", "target": "s2"},
    ],
    "formula": "νX.(safe & []X)",
}


@pytest.fixture()
def client(tmp_path):
    store = ReviewStore(str(tmp_path / "reviews.db"))
    yield TestClient(create_app(store))
    store.close()


def test_health(client):
    assert client.get("/health").json() == {"status": "ok"}


def test_create_and_read_back(client):
    created = client.post("/reviews", json=VALID)
    assert created.status_code == 201
    rid = created.json()["id"]
    assert created.json()["result"]["initial_satisfied"] is True

    fetched = client.get(f"/reviews/{rid}")
    assert fetched.status_code == 200
    body = fetched.json()
    assert body["id"] == rid
    assert body["result"]["satisfaction_set"] == ["s1", "s2"]
    assert [i["states"] for i in body["result"]["iterations"]] == [
        ["s1", "s2"],
        ["s1", "s2"],
    ]
    assert body["spec"]["formula"] == VALID["formula"]


def test_results_persist_across_store_reopen(tmp_path):
    db = str(tmp_path / "reviews.db")
    first = TestClient(create_app(ReviewStore(db)))
    rid = first.post("/reviews", json=VALID).json()["id"]
    second = TestClient(create_app(ReviewStore(db)))
    assert second.get(f"/reviews/{rid}").status_code == 200


def test_unknown_id_is_404(client):
    assert client.get("/reviews/" + "0" * 32).status_code == 404


@pytest.mark.parametrize(
    "patch",
    [
        {"locations": ["only"]},  # too few locations
        {"locations": [f"l{i}" for i in range(25)]},  # too many locations
        {"locations": ["s1", "s1"]},  # duplicate location identifier
        {"initial": "ghost"},  # unknown initial location
        {"transitions": [{"id": "t1", "source": "s1", "target": "ghost"}]},  # dangling
        {
            "transitions": [
                {"id": "t1", "source": "s1", "target": "s2"},
                {"id": "t1", "source": "s2", "target": "s1"},
            ]
        },  # duplicate transition identifier
        {"propositions": {"ghost": ["safe"]}},  # unknown location in propositions
        {"formula": "μX.(goal | <>Y)"},  # unbound variable
        {"formula": "μX.(<>X) & νX.(safe & []X)"},  # duplicate binder name
        {"formula": "μX.(goal | X)"},  # unguarded variable
        {"formula": "νX.(safe & []X) trailing"},  # syntax residue
        {"formula": "μX.(goal | <>X"},  # unbalanced
        {"formula": ""},  # empty formula
    ],
)
def test_invalid_specs_rejected_without_id(client, patch):
    body = {**VALID, **patch}
    resp = client.post("/reviews", json=body)
    assert resp.status_code == 422
    assert "id" not in resp.json()
