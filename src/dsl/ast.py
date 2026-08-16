from __future__ import annotations

from dataclasses import dataclass
from typing import Union


@dataclass(frozen=True)
class Number:
    value: float


@dataclass(frozen=True)
class Field:
    name: str


@dataclass(frozen=True)
class Name:
    value: str


@dataclass(frozen=True)
class Call:
    operator: str
    args: tuple["Expr", ...]


Expr = Union[Number, Field, Name, Call]


@dataclass(frozen=True)
class ParsedExpression:
    assignments: tuple[tuple[str, Expr], ...]
    output: Expr
    node_count: int
    fields: frozenset[str]
    operators: frozenset[str]
