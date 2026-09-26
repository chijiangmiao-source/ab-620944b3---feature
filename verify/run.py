"""One-shot acceptance run for the fixpoint review service.

Steps: unit tests (μ expansion / ν convergence), build check, then
HTTP smoke against the running app covering the acceptance scenarios.
Exits 0 when every step passes, 1 otherwise — designed to be run as a
single-shot Compose service (`docker compose up --exit-code-from verify`).
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request

APP_URL = os.environ.get("APP_URL", "http://127.0.0.1:8000").rstrip("/")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SAFETY_FORMULA = "νX.(safe & []X)"
REACH_FORMULA = "μX.(goal | <>X)"

STEPS = []
FAILURES = []


def step(fn):
    STEPS.append(fn)
    return fn


def http(method, path, body=None, expect=200):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(
        APP_URL + path,
        data=data,
        method=method,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            status, payload = resp.status, resp.read()
    except urllib.error.HTTPError as exc:
        status, payload = exc.code, exc.read()
    if status != expect:
        raise AssertionError(
            f"{method} {path}: expected HTTP {expect}, got {status}: {payload[:300]!r}"
        )
    return json.loads(payload) if payload else {}


# --- code tests and build check -------------------------------------------


@step
def unit_tests_mu_expansion_and_nu_convergence():
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "tests", "-q"],
        cwd=ROOT,
        env={**os.environ, "DATA_DIR": "/tmp/verify-data"},
    )
    assert proc.returncode == 0, "pytest reported failures"


@step
def build_check_compiles_and_imports():
    proc = subprocess.run(
        [sys.executable, "-m", "compileall", "-q", "app", "verify", "tests"], cwd=ROOT
    )
    assert proc.returncode == 0, "compileall failed"
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import app.main, app.formula, app.checker, app.store",
        ],
        cwd=ROOT,
        check=True,
        env={**os.environ, "DATA_DIR": "/tmp/verify-data"},
    )


# --- HTTP smoke ------------------------------------------------------------


@step
def wait_for_app_health():
    deadline = time.time() + 60
    while time.time() < deadline:
        try:
            http("GET", "/health", expect=200)
            return
        except Exception:
            time.sleep(1)
    raise AssertionError(f"app at {APP_URL} did not become healthy in time")


@step
def smoke_nu_safety_self_loops_satisfied():
    body = {
        "locations": ["s1", "s2"],
        "initial": "s1",
        "propositions": {"s1": ["safe"], "s2": ["safe"]},
        "transitions": [
            {"id": "t1", "source": "s1", "target": "s1"},
            {"id": "t2", "source": "s1", "target": "s2"},
            {"id": "t3", "source": "s2", "target": "s2"},
        ],
        "formula": SAFETY_FORMULA,
    }
    created = http("POST", "/reviews", body, expect=201)
    result = created["result"]
    assert result["initial_satisfied"] is True, result
    assert result["satisfaction_set"] == ["s1", "s2"], result
    seq = [i["states"] for i in result["iterations"]]
    assert seq == [["s1", "s2"], ["s1", "s2"]], seq  # ν starts full, stable at once
    fetched = http("GET", f"/reviews/{created['id']}")
    assert fetched["result"] == result, "read-back evidence differs"
    assert fetched["spec"]["formula"] == SAFETY_FORMULA


@step
def smoke_mu_reachability_extends_to_initial():
    body = {
        "locations": ["s1", "s2", "s3"],
        "initial": "s1",
        "propositions": {"s3": ["goal"]},
        "transitions": [
            {"id": "a", "source": "s1", "target": "s2"},
            {"id": "b", "source": "s2", "target": "s3"},
            {"id": "c", "source": "s3", "target": "s3"},
        ],
        "formula": REACH_FORMULA,
    }
    created = http("POST", "/reviews", body, expect=201)
    result = created["result"]
    assert result["initial_satisfied"] is True, result
    assert result["satisfaction_set"] == ["s1", "s2", "s3"], result
    seq = [i["states"] for i in result["iterations"]]
    assert seq == [[], ["s3"], ["s2", "s3"], ["s1", "s2", "s3"], ["s1", "s2", "s3"]], seq
    fetched = http("GET", f"/reviews/{created['id']}")
    assert fetched["result"]["iterations"] == result["iterations"]


@step
def smoke_dangerous_transition_breaks_safety():
    body = {
        "locations": ["s1", "s2", "s3"],
        "initial": "s1",
        "propositions": {"s1": ["safe"], "s2": ["safe"]},
        "transitions": [
            {"id": "t1", "source": "s1", "target": "s2"},
            {"id": "t2", "source": "s2", "target": "s2"},
            {"id": "t3", "source": "s1", "target": "s3"},  # danger: s3 not safe
            {"id": "t4", "source": "s3", "target": "s3"},
        ],
        "formula": SAFETY_FORMULA,
    }
    created = http("POST", "/reviews", body, expect=201)
    result = created["result"]
    assert result["initial_satisfied"] is False, result
    assert result["satisfaction_set"] == ["s2"], result
    seq = [i["states"] for i in result["iterations"]]
    assert seq == [["s1", "s2", "s3"], ["s1", "s2"], ["s2"], ["s2"]], seq
    fetched = http("GET", f"/reviews/{created['id']}")
    assert fetched["result"]["initial_satisfied"] is False
    assert fetched["result"]["satisfaction_set"] == ["s2"]


@step
def smoke_invalid_specs_rejected_without_readable_id():
    base = {
        "locations": ["s1", "s2"],
        "initial": "s1",
        "propositions": {"s1": ["safe"], "s2": ["safe"]},
        "transitions": [{"id": "t1", "source": "s1", "target": "s2"}],
        "formula": SAFETY_FORMULA,
    }
    rejections = [
        {"locations": ["only"]},  # below minimum
        {"locations": [f"l{i}" for i in range(25)]},  # above maximum
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
        {"formula": "μX.!<>X"},  # variable under negation
        {"formula": SAFETY_FORMULA + " trailing"},  # syntax residue
        {"formula": "μX.(goal | <>X"},  # unbalanced
        {"formula": ""},  # empty formula
    ]
    for patch in rejections:
        resp = http("POST", "/reviews", {**base, **patch}, expect=422)
        assert "id" not in resp, f"rejected spec leaked an id: {patch}"


@step
def smoke_unknown_review_id_is_404():
    http("GET", "/reviews/" + "0" * 32, expect=404)


def main() -> int:
    print(f"acceptance target: {APP_URL}", flush=True)
    for fn in STEPS:
        name = fn.__name__.replace("_", " ")
        try:
            fn()
        except Exception as exc:  # report and continue with remaining steps
            FAILURES.append(name)
            print(f"[FAIL] {name}: {exc}", flush=True)
        else:
            print(f"[PASS] {name}", flush=True)
    if FAILURES:
        print(f"ACCEPTANCE FAILED ({len(FAILURES)} step(s)): {', '.join(FAILURES)}", flush=True)
        return 1
    print("ACCEPTANCE PASSED", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
