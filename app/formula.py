"""Parser and well-formedness checks for the modal mu-calculus fragment.

Grammar (binders extend as far right as possible):

    formula := or
    or      := and ("|" and)*
    and     := unary ("&" unary)*
    unary   := "!" unary | "<>" unary | "[]" unary
             | "μ" VAR "." formula | "ν" VAR "." formula
             | "(" formula ")" | PROP | VAR

Well-formedness rules enforced here:
  * no syntax residue: the whole input must be consumed;
  * every variable occurrence is bound (nearest binder semantics);
  * binder names are unique within a formula;
  * every bound-variable occurrence is guarded by a modal operator
    (<> or []) between the binder and the occurrence;
  * bound variables occur only positively (never under an odd number
    of negations), so fixpoint iteration is monotone and terminates.
"""
from __future__ import annotations

import re
from dataclasses import dataclass


class FormulaError(ValueError):
    """Raised when a formula is syntactically or semantically ill-formed."""


# --- AST -------------------------------------------------------------------


@dataclass(frozen=True)
class Prop:
    name: str


@dataclass(frozen=True)
class Var:
    name: str


@dataclass(frozen=True)
class Not:
    child: object


@dataclass(frozen=True)
class And:
    left: object
    right: object


@dataclass(frozen=True)
class Or:
    left: object
    right: object


@dataclass(frozen=True)
class Dia:
    child: object


@dataclass(frozen=True)
class Box:
    child: object


@dataclass(frozen=True)
class Mu:
    var: str
    body: object


@dataclass(frozen=True)
class Nu:
    var: str
    body: object


# --- lexer -----------------------------------------------------------------

_TOKEN_RE = re.compile(
    r"(?P<ws>\s+)"
    r"|(?P<dia><>)"
    r"|(?P<box>\[\])"
    r"|(?P<mu>μ)"
    r"|(?P<nu>ν)"
    r"|(?P<not>!)"
    r"|(?P<and>&)"
    r"|(?P<or>\|)"
    r"|(?P<lparen>\()"
    r"|(?P<rparen>\))"
    r"|(?P<dot>\.)"
    r"|(?P<ident>[A-Za-z_][A-Za-z0-9_]*)"
)


def _is_variable(name: str) -> bool:
    return name[0].isupper()


def _tokenize(text: str) -> list[tuple[str, str]]:
    tokens: list[tuple[str, str]] = []
    pos = 0
    while pos < len(text):
        match = _TOKEN_RE.match(text, pos)
        if match is None:
            raise FormulaError(f"unexpected character {text[pos]!r} at column {pos}")
        pos = match.end()
        kind = match.lastgroup
        if kind == "ws":
            continue
        tokens.append((kind, match.group(0)))
    return tokens


# --- parser ----------------------------------------------------------------


class _Parser:
    def __init__(self, tokens: list[tuple[str, str]]):
        self._tokens = tokens
        self._pos = 0

    def peek(self):
        if self._pos < len(self._tokens):
            return self._tokens[self._pos]
        return None

    def advance(self):
        token = self.peek()
        if token is None:
            raise FormulaError("unexpected end of formula")
        self._pos += 1
        return token

    def expect(self, kind: str):
        token = self.advance()
        if token[0] != kind:
            raise FormulaError(f"expected {kind}, found {token[1]!r}")
        return token

    def parse_or(self):
        node = self.parse_and()
        while self.peek() is not None and self.peek()[0] == "or":
            self.advance()
            node = Or(node, self.parse_and())
        return node

    def parse_and(self):
        node = self.parse_unary()
        while self.peek() is not None and self.peek()[0] == "and":
            self.advance()
            node = And(node, self.parse_unary())
        return node

    def parse_unary(self):
        token = self.peek()
        if token is None:
            raise FormulaError("unexpected end of formula")
        kind, text = token
        if kind == "not":
            self.advance()
            return Not(self.parse_unary())
        if kind == "dia":
            self.advance()
            return Dia(self.parse_unary())
        if kind == "box":
            self.advance()
            return Box(self.parse_unary())
        if kind in ("mu", "nu"):
            self.advance()
            name = self.expect("ident")[1]
            if not _is_variable(name):
                raise FormulaError(
                    f"binder must introduce a variable (uppercase identifier), got {name!r}"
                )
            self.expect("dot")
            body = self.parse_or()  # binder scope extends as far right as possible
            return Mu(name, body) if kind == "mu" else Nu(name, body)
        if kind == "lparen":
            self.advance()
            node = self.parse_or()
            self.expect("rparen")
            return node
        if kind == "ident":
            self.advance()
            return Var(text) if _is_variable(text) else Prop(text)
        raise FormulaError(f"unexpected token {text!r}")


def parse(text: str):
    """Parse and validate a formula, returning its AST."""
    if not isinstance(text, str) or not text.strip():
        raise FormulaError("formula must be a non-empty string")
    parser = _Parser(_tokenize(text))
    node = parser.parse_or()
    rest = parser.peek()
    if rest is not None:
        raise FormulaError(f"syntax residue: trailing token {rest[1]!r}")
    validate(node)
    return node


# --- validation ------------------------------------------------------------


def validate(node) -> None:
    binders: list = []
    _validate_scope(node, (), set(), binders)
    for binder in binders:
        _check_guarded_positive(binder.body, binder.var, modal=False, negated=False)


def _validate_scope(node, bound: tuple, names: set, binders: list) -> None:
    if isinstance(node, (Mu, Nu)):
        if node.var in names:
            raise FormulaError(f"duplicate binder name '{node.var}'")
        names.add(node.var)
        binders.append(node)
        _validate_scope(node.body, bound + (node.var,), names, binders)
    elif isinstance(node, Var):
        if node.name not in bound:
            raise FormulaError(f"unbound variable '{node.name}'")
    elif isinstance(node, (Not, Dia, Box)):
        _validate_scope(node.child, bound, names, binders)
    elif isinstance(node, (And, Or)):
        _validate_scope(node.left, bound, names, binders)
        _validate_scope(node.right, bound, names, binders)
    elif isinstance(node, Prop):
        return
    else:
        raise FormulaError(f"unknown syntax node {node!r}")


def _check_guarded_positive(node, var: str, modal: bool, negated: bool) -> None:
    """Every occurrence of `var` must sit under a modality and an even
    number of negations (counted from its binder). Nested binders of
    other variables do not reset these conditions."""
    if isinstance(node, Var):
        if node.name == var:
            if negated:
                raise FormulaError(
                    f"variable '{var}' occurs under negation; fixpoint would not be monotone"
                )
            if not modal:
                raise FormulaError(
                    f"variable '{var}' must be guarded by a modal operator (<> or [])"
                )
    elif isinstance(node, (Dia, Box)):
        _check_guarded_positive(node.child, var, True, negated)
    elif isinstance(node, Not):
        _check_guarded_positive(node.child, var, modal, not negated)
    elif isinstance(node, (And, Or)):
        _check_guarded_positive(node.left, var, modal, negated)
        _check_guarded_positive(node.right, var, modal, negated)
    elif isinstance(node, (Mu, Nu)):
        _check_guarded_positive(node.body, var, modal, negated)
