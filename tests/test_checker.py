"""Unit tests for fixpoint evaluation: μ expansion and ν convergence."""
import pytest

from app.checker import Model, evaluate
from app.formula import (
    And,
    Dia,
    FormulaError,
    Mu,
    Or,
    Prop,
    Var,
    parse,
)


def model(locations, props, transitions):
    return Model.from_spec(locations, props, transitions)


def test_mu_expands_from_empty_until_stable():
    m = model(
        ["s1", "s2", "s3"],
        {"s3": ["goal"]},
        [("a", "s1", "s2"), ("b", "s2", "s3"), ("c", "s3", "s3")],
    )
    sat, iters = evaluate(parse("μX.(goal | <>X)"), m)
    assert sat == frozenset({"s1", "s2", "s3"})
    assert [i["states"] for i in iters] == [
        [],
        ["s3"],
        ["s2", "s3"],
        ["s1", "s2", "s3"],
        ["s1", "s2", "s3"],  # final approximant repeats: stable
    ]
    assert all(i["op"] == "mu" and i["binder"] == "X" for i in iters)
    assert [i["step"] for i in iters] == [0, 1, 2, 3, 4]


def test_nu_converges_from_universe_on_safe_self_loops():
    m = model(
        ["s1", "s2"],
        {"s1": ["safe"], "s2": ["safe"]},
        [("t1", "s1", "s1"), ("t2", "s1", "s2"), ("t3", "s2", "s2")],
    )
    sat, iters = evaluate(parse("νX.(safe & []X)"), m)
    assert sat == frozenset({"s1", "s2"})
    assert [i["states"] for i in iters] == [["s1", "s2"], ["s1", "s2"]]
    assert all(i["op"] == "nu" for i in iters)


def test_nu_shrinks_away_from_dangerous_transition():
    m = model(
        ["s1", "s2", "s3"],
        {"s1": ["safe"], "s2": ["safe"]},
        [
            ("t1", "s1", "s2"),
            ("t2", "s2", "s2"),
            ("t3", "s1", "s3"),  # danger: s3 is not safe
            ("t4", "s3", "s3"),
        ],
    )
    sat, iters = evaluate(parse("νX.(safe & []X)"), m)
    assert sat == frozenset({"s2"})
    assert "s1" not in sat
    assert [i["states"] for i in iters] == [
        ["s1", "s2", "s3"],
        ["s1", "s2"],
        ["s2"],
        ["s2"],
    ]


def test_sibling_binders_keep_separate_environments():
    m = model(
        ["a", "b", "c", "d"],
        {"a": ["safe"], "b": ["safe"], "c": ["goal"]},
        [("t1", "a", "b"), ("t2", "b", "b"), ("t3", "c", "c"), ("t4", "d", "c")],
    )
    sat, iters = evaluate(parse("(νX.(safe & []X)) | (μY.(goal | <>Y))"), m)
    # νX.(safe & []X) = {a, b}; μY.(goal | <>Y) = {c, d}
    assert sat == frozenset({"a", "b", "c", "d"})
    assert {i["binder"] for i in iters} == {"X", "Y"}
    assert [i["states"] for i in iters if i["binder"] == "X"][-1] == ["a", "b"]
    assert [i["states"] for i in iters if i["binder"] == "Y"][-1] == ["c", "d"]


def test_inner_fixpoint_reevaluated_each_outer_round_without_leaking():
    m = model(["a", "b"], {"a": ["p"]}, [("t1", "a", "b"), ("t2", "b", "b")])
    sat, iters = evaluate(parse("νX.(<>X & μY.(p | <>Y))"), m)
    assert sat == frozenset()
    y_rounds: dict[int, list] = {}
    for i in iters:
        if i["binder"] == "Y":
            y_rounds.setdefault(i["round"], []).append(i["states"])
    # outer ν iterates {a,b} -> {a} -> ∅ -> ∅, re-entering μY three times
    assert len(y_rounds) == 3
    for states in y_rounds.values():
        assert states[0] == [] and states[-1] == ["a"]


def test_precedence_and_binder_scope():
    assert parse("a | b & c") == Or(Prop("a"), And(Prop("b"), Prop("c")))
    # binder scope extends as far right as possible
    assert parse("μX.p | <>X") == Mu("X", Or(Prop("p"), Dia(Var("X"))))


def test_guarded_nested_occurrence_accepted():
    parse("μX.(goal | νY.(<>X & []Y))")
    parse("νX.(safe & []X)")
    parse("μX.(goal | <>X)")


@pytest.mark.parametrize(
    "text",
    [
        "μX.(goal | <>X) junk",  # syntax residue
        "μX.(goal | <>X",  # unbalanced parenthesis
        "",  # empty
        "safe &",  # dangling operator
        "μX.(goal | <>Y)",  # unbound variable
        "μX.(<>X) & νX.(safe & []X)",  # duplicate binder name
        "μX.(goal | X)",  # unguarded variable
        "μX.!<>X",  # variable under negation (non-monotone)
        "μx.(goal | <>x)",  # binder must introduce an uppercase variable
        "safe @ sound",  # illegal character
    ],
)
def test_rejected_formulas(text):
    with pytest.raises(FormulaError):
        parse(text)
