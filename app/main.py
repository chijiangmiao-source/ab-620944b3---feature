"""HTTP API for the fixpoint review service.

POST /reviews  — submit a review (locations, initial location, location
                 propositions, identified directed transitions, formula).
                 Invalid submissions are rejected with 422 and never
                 produce a readable id.
GET  /reviews/{id} — read back the conclusion and the normative
                 evidence (sorted satisfaction set, every fixpoint
                 iteration set).
POST /reviews/{id}/audits — freeze the review and build a fixpoint
                 dependency-evidence audit for the initial location
                 (satisfaction proof, or opposite-polarity refutation
                 when the conclusion does not hold).  A missing source
                 (404), a frozen recomputation that disagrees with the
                 stored conclusion (409), or evidence that cannot close
                 (500) never produces a readable audit id.
GET  /audits/{id} — read the audit (frozen snapshot, stable occurrence
                 positions, dependency evidence); every read
                 independently re-verifies evidence edges, location
                 propositions, transition endpoints, binding
                 environments, approximation rounds and the root
                 conclusion, and reports the outcome.
GET  /health   — liveness probe.
"""
from __future__ import annotations

import os

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .checker import Model, evaluate, state_sort_key
from .evidence import EvidenceError, build_audit, verify_audit
from .formula import FormulaError, parse
from .store import ReviewStore

MIN_LOCATIONS = 2
MAX_LOCATIONS = 24


class SpecError(ValueError):
    """Raised when a review specification violates the acceptance rules."""


class TransitionIn(BaseModel):
    id: str
    source: str
    target: str


class ReviewIn(BaseModel):
    locations: list[str]
    initial: str
    propositions: dict[str, list[str]] = Field(default_factory=dict)
    transitions: list[TransitionIn] = Field(default_factory=list)
    formula: str


def _normalize(spec: ReviewIn):
    """Validate the request; return (normalized_spec, model, ast)."""
    locations = spec.locations
    if not MIN_LOCATIONS <= len(locations) <= MAX_LOCATIONS:
        raise SpecError(
            f"need between {MIN_LOCATIONS} and {MAX_LOCATIONS} locations, "
            f"got {len(locations)}"
        )
    if any(not isinstance(loc, str) or not loc.strip() for loc in locations):
        raise SpecError("location identifiers must be non-empty strings")
    if len(set(locations)) != len(locations):
        raise SpecError("duplicate location identifier")
    if spec.initial not in locations:
        raise SpecError(f"initial location '{spec.initial}' is not a declared location")

    propositions: dict[str, list[str]] = {}
    for loc, props in spec.propositions.items():
        if loc not in locations:
            raise SpecError(f"propositions reference unknown location '{loc}'")
        if any(not isinstance(p, str) or not p.strip() for p in props):
            raise SpecError(f"propositions of location '{loc}' must be non-empty strings")
        propositions[loc] = sorted(set(props))

    transition_ids = [t.id for t in spec.transitions]
    if any(not t.id.strip() for t in spec.transitions):
        raise SpecError("transition identifiers must be non-empty strings")
    if len(set(transition_ids)) != len(transition_ids):
        raise SpecError("duplicate transition identifier")
    for t in spec.transitions:
        if t.source not in locations or t.target not in locations:
            raise SpecError(
                f"dangling transition '{t.id}': "
                f"'{t.source}' -> '{t.target}' is not between declared locations"
            )

    ast = parse(spec.formula)  # raises FormulaError on any ill-formedness

    ordered = sorted(locations, key=state_sort_key)
    model = Model.from_spec(
        ordered,
        propositions,
        [(t.id, t.source, t.target) for t in spec.transitions],
    )
    normalized = {
        "locations": ordered,
        "initial": spec.initial,
        "propositions": {loc: propositions.get(loc, []) for loc in ordered},
        "transitions": [
            {"id": t.id, "source": t.source, "target": t.target}
            for t in spec.transitions
        ],
        "formula": spec.formula,
    }
    return normalized, model, ast


def create_app(store: ReviewStore) -> FastAPI:
    app = FastAPI(title="Fixpoint Review Service", version="1.1.0")

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.post("/reviews", status_code=201)
    def create_review(spec: ReviewIn):
        try:
            normalized, model, ast = _normalize(spec)
        except (SpecError, FormulaError) as exc:
            # Rejected: no record is persisted, no readable id is issued.
            raise HTTPException(status_code=422, detail=str(exc))
        sat, iterations = evaluate(ast, model)
        result = {
            "formula": spec.formula,
            "initial": spec.initial,
            "initial_satisfied": spec.initial in sat,
            "satisfaction_set": sorted(sat, key=state_sort_key),
            "iterations": iterations,
            "converged": True,
        }
        review_id = store.save({"spec": normalized, "result": result})
        return {"id": review_id, "result": result}

    @app.get("/reviews/{review_id}")
    def get_review(review_id: str):
        record = store.get(review_id)
        if record is None:
            raise HTTPException(status_code=404, detail="review not found")
        return record

    @app.post("/reviews/{review_id}/audits", status_code=201)
    def create_audit(review_id: str):
        record = store.get(review_id)
        if record is None:
            # Source missing: no readable audit may be issued.
            raise HTTPException(status_code=404, detail="review not found")
        spec, result = record["spec"], record["result"]
        try:
            ast = parse(spec["formula"])
            model = Model.from_spec(
                spec["locations"],
                spec["propositions"],
                [(t["id"], t["source"], t["target"]) for t in spec["transitions"]],
            )
            sat, iterations = evaluate(ast, model)
        except Exception as exc:
            raise HTTPException(
                status_code=409, detail=f"frozen source cannot be re-evaluated: {exc}"
            )
        consistent = (
            sorted(sat, key=state_sort_key) == result["satisfaction_set"]
            and (spec["initial"] in sat) == result["initial_satisfied"]
            and iterations == result["iterations"]
        )
        if not consistent:
            # Frozen recomputation disagrees with the stored conclusion:
            # refuse to mint a readable audit.
            raise HTTPException(
                status_code=409,
                detail="frozen recomputation is inconsistent with the stored conclusion",
            )
        try:
            audit = build_audit(spec, result)
        except EvidenceError as exc:
            # Evidence cannot close: nothing readable is persisted.
            raise HTTPException(status_code=500, detail=f"evidence cannot close: {exc}")
        verification = verify_audit(audit)
        if not verification["verified"]:
            raise HTTPException(
                status_code=500,
                detail="evidence cannot close: " + "; ".join(verification["errors"][:3]),
            )
        audit_id = store.save_audit(review_id, audit)
        return {
            "id": audit_id,
            "review_id": review_id,
            "polarity": audit["polarity"],
            "root": audit["root"],
            "verified": verification["verified"],
            "checks": verification["checks"],
        }

    @app.get("/audits/{audit_id}")
    def get_audit(audit_id: str):
        record = store.get_audit(audit_id)
        if record is None:
            raise HTTPException(status_code=404, detail="audit not found")
        # Independent re-verification on every read.
        verification = verify_audit(record)
        frozen = record.get("frozen") or {}
        source = store.get(record["review_id"])
        if source is not None and (
            source["spec"] != frozen.get("spec")
            or source["result"] != frozen.get("result")
        ):
            verification["verified"] = False
            verification["errors"].append(
                "frozen snapshot differs from the current source review"
            )
        return {**record, "verification": verification}

    return app


def _default_store() -> ReviewStore:
    data_dir = os.environ.get("DATA_DIR", "./data")
    return ReviewStore(os.path.join(data_dir, "reviews.db"))


app = create_app(_default_store())
