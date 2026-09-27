"""HTTP API for the fixpoint review service.

POST /reviews  — submit a review (locations, initial location, location
                 propositions, identified directed transitions, formula).
                 Invalid submissions are rejected with 422 and never
                 produce a readable id.
GET  /reviews/{id} — read back the conclusion and the normative
                 evidence (sorted satisfaction set, every fixpoint
                 iteration set).
POST /reviews/{id}/audits — launch a stable-point dependency-evidence
                 audit against a persisted review.  The source procedure
                 and conclusion are frozen, re-evaluated for consistency,
                 and a position-indexed proof graph is built and
                 independently verified before a new audit id is issued.
                 A missing source yields 404; a freeze/recompute
                 mismatch yields 409; evidence that cannot close is
                 never persisted.
GET  /audits/{id} — read back the frozen bundle and the directed proof
                 graph after re-running the independent verification;
                 evidence that no longer closes is not served (409).
GET  /health   — liveness probe.
"""
from __future__ import annotations

import os

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .audit import AuditError, AuditFreezeError, produce_audit, reverify_audit
from .checker import Model, evaluate, state_sort_key
from .evidence import EvidenceError
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
    app = FastAPI(title="Fixpoint Review Service", version="1.0.0")

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
            # The source review does not exist: no audit is persisted.
            raise HTTPException(status_code=404, detail="review not found")
        try:
            bundle = produce_audit(record)
        except AuditFreezeError as exc:
            raise HTTPException(status_code=409, detail=str(exc))
        except (AuditError, EvidenceError, FormulaError, KeyError, TypeError,
                ValueError) as exc:
            # Evidence could not be closed (including a tampered frozen
            # source): nothing readable is written.
            raise HTTPException(status_code=500, detail=str(exc))
        audit_id = store.save_audit(review_id, bundle)
        return {
            "id": audit_id,
            "review_id": review_id,
            "conclusion": bundle["conclusion"],
            "polarity": bundle["polarity"],
        }

    @app.get("/audits/{audit_id}")
    def get_audit(audit_id: str):
        record = store.get_audit(audit_id)
        if record is None:
            raise HTTPException(status_code=404, detail="audit not found")
        try:
            reverify_audit(record)
        except (EvidenceError, KeyError, TypeError) as exc:
            # Stored evidence no longer closes under independent
            # verification: it is not served as a readable audit.
            raise HTTPException(
                status_code=409, detail=f"audit evidence does not verify: {exc}"
            )
        return record

    return app


def _default_store() -> ReviewStore:
    data_dir = os.environ.get("DATA_DIR", "./data")
    return ReviewStore(os.path.join(data_dir, "reviews.db"))


app = create_app(_default_store())
