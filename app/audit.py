"""Stable-point dependency-evidence audits.

An audit is launched against an already persisted review id.  The
service *freezes* the source procedure (location set, identified
transitions, location propositions, formula, initial location) and the
original conclusion, then:

1. re-parses the frozen formula and labels every syntactic occurrence
   with a stable position id;
2. independently re-runs the global fixpoint evaluation from the frozen
   procedure and demands bit-identical satisfaction set and approximant
   iterations (a freeze/recompute mismatch aborts the audit);
3. records the position-indexed evidence base, builds the directed
   proof graph for the initial location (opposite-polarity rejection
   proof when the position does not satisfy) and independently
   verifies the whole bundle before any audit id is issued.

Nothing is persisted when the source is missing or the evidence fails
to close.
"""
from __future__ import annotations

from .checker import Model, evaluate, state_sort_key
from .evidence import (
    EvidenceError,
    build_proof,
    evaluate_with_evidence,
    verify_bundle,
)
from .formula import parse
from .positions import label_positions, positions_to_json


class AuditError(Exception):
    """Audit could not be produced; no readable audit is persisted."""


class AuditFreezeError(AuditError):
    """The frozen procedure and the original conclusion disagree."""


def _build_model(spec: dict) -> Model:
    return Model.from_spec(
        list(spec["locations"]),
        {loc: list(props) for loc, props in spec["propositions"].items()},
        [(t["id"], t["source"], t["target"]) for t in spec["transitions"]],
    )


def _normalise_result(result: dict) -> dict:
    return {
        "initial_satisfied": bool(result["initial_satisfied"]),
        "satisfaction_set": sorted(result["satisfaction_set"], key=state_sort_key),
        "iterations": [
            {
                "binder": i["binder"],
                "op": i["op"],
                "round": i["round"],
                "step": i["step"],
                "states": sorted(i["states"], key=state_sort_key),
            }
            for i in result["iterations"]
        ],
    }


def produce_audit(record: dict) -> dict:
    """Build a fully verified audit bundle from a frozen review record."""
    spec = record.get("spec")
    result = record.get("result")
    if not isinstance(spec, dict) or not isinstance(result, dict):
        raise AuditError("source review is missing its frozen procedure or conclusion")

    formula_text = spec["formula"]
    initial = spec["initial"]
    model = _build_model(spec)
    ast = parse(formula_text)  # frozen formula must stay well-formed

    # Freeze/recompute: the original conclusion must reproduce exactly.
    sat, iterations = evaluate(ast, model)
    recomputed = _normalise_result(
        {
            "initial_satisfied": initial in sat,
            "satisfaction_set": sorted(sat, key=state_sort_key),
            "iterations": iterations,
        }
    )
    original = _normalise_result(result)
    if recomputed != original:
        raise AuditFreezeError(
            "frozen procedure re-evaluates to a different conclusion than the one stored"
        )

    table = label_positions(ast)
    positions = positions_to_json(table)
    ev_sat, ev_iterations, facts = evaluate_with_evidence(ast, table, model)
    if frozenset(ev_sat) != frozenset(sat):
        raise AuditError("instrumented satisfaction set disagrees with the frozen review")
    if _normalise_result(
        {
            "initial_satisfied": initial in ev_sat,
            "satisfaction_set": ev_sat,
            "iterations": ev_iterations,
        }
    )["iterations"] != original["iterations"]:
        raise AuditError("instrumented approximant iterations disagree with the frozen review")

    polarity = "+" if original["initial_satisfied"] else "-"
    proof = build_proof(facts, positions, model, initial, polarity)

    # Independent gate: re-parse, re-derive every truth set, re-check
    # every edge, frame, approximant and the root conclusion.
    verify_bundle(
        proof=proof,
        facts=facts,
        positions=positions,
        model=model,
        initial=initial,
        polarity=polarity,
        formula_text=formula_text,
    )

    return {
        "frozen": {"spec": spec, "result": original},
        "positions": positions,
        "facts": facts,
        "proof": proof,
        "polarity": polarity,
        "conclusion": {
            "initial": initial,
            "initial_satisfied": original["initial_satisfied"],
        },
    }


def reverify_audit(bundle: dict) -> None:
    """Re-run the independent verification on a persisted audit.

    Raises EvidenceError when any edge, position claim, transition
    endpoint, binding frame, approximant round/step or the root
    conclusion no longer closes.
    """
    spec = bundle["frozen"]["spec"]
    verify_bundle(
        proof=bundle["proof"],
        facts=bundle["facts"],
        positions=bundle["positions"],
        model=_build_model(spec),
        initial=bundle["conclusion"]["initial"],
        polarity=bundle["polarity"],
        formula_text=spec["formula"],
    )
    expected_satisfied = bundle["polarity"] == "+"
    if bundle["conclusion"]["initial_satisfied"] != expected_satisfied:
        raise EvidenceError("audit polarity contradicts its frozen conclusion")
