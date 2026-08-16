from __future__ import annotations

import ast as py_ast
import re

from dsl.ast import Call, Expr, Field, Name, Number, ParsedExpression


ALLOWED_FIELDS = frozenset(
    {
        "open", "high", "low", "close", "volume", "market_cap", "dollar_volume", "ret",
        "funda_pe", "funda_pb", "funda_div_yield", "funda_days_since_disclosure",
        "funda_days_since_quarter_start", "funda_net_income", "funda_book_equity",
        "funda_shares_out",
    }
)

OPERATOR_ARITY = {
    "TS_MEAN": 2, "TS_SUM": 2, "TS_MIN": 2, "TS_MAX": 2, "TS_STD": 2, "TS_RANK": 2,
    "DELTA": 2, "DELAY": 2, "TS_CORR": 3,
    "CS_RANK": 1, "CS_ZSCORE": 1, "CS_WINSORIZE": 3, "CS_NEUTRALIZE": 2, "CS_BUCKET": 2,
    "ADD": 2, "SUBTRACT": 2, "MULTIPLY": 2, "DIVIDE": 2, "ABS": 1, "SIGN": 1,
    "LOG": 1, "POW": 2,
    "GT": 2, "LT": 2, "IF_THEN_ELSE": 3,
    "TS_SINCE": 1, "TS_COUNT": 2,
}

INTEGER_LITERAL_ARGS = {
    "TS_MEAN": (1,), "TS_SUM": (1,), "TS_MIN": (1,), "TS_MAX": (1,), "TS_STD": (1,),
    "TS_RANK": (1,), "DELTA": (1,), "DELAY": (1,), "TS_CORR": (2,),
    "CS_BUCKET": (1,), "POW": (1,), "TS_COUNT": (1,),
}

POSITIVE_INTEGER_LITERAL_ARGS = {
    "TS_MEAN": (1,), "TS_SUM": (1,), "TS_MIN": (1,), "TS_MAX": (1,), "TS_STD": (1,),
    "TS_RANK": (1,), "DELTA": (1,), "TS_CORR": (2,), "CS_BUCKET": (1,), "TS_COUNT": (1,),
}

FIELD_RE = re.compile(r"\$([A-Za-z_][A-Za-z0-9_]*)")


class ParseError(ValueError):
    pass


def parse_expression(text: str, max_nodes: int = 50) -> ParsedExpression:
    assignments: list[tuple[str, Expr]] = []
    defined: set[str] = set()
    output: Expr | None = None
    for raw in [line.strip() for line in text.splitlines() if line.strip()]:
        if "=" in raw:
            name, expr_text = [part.strip() for part in raw.split("=", 1)]
            if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
                raise ParseError(f"bad assignment name: {name}")
            expr = _parse_expr(expr_text, defined)
            assignments.append((name, expr))
            defined.add(name)
        else:
            if output is not None:
                raise ParseError("only the final bare expression may define output")
            output = _parse_expr(raw, defined)
    if output is None:
        raise ParseError("missing final output expression")
    nodes = sum(_node_count(expr) for _, expr in assignments) + _node_count(output)
    if nodes > max_nodes:
        raise ParseError(f"expression has {nodes} AST nodes, exceeding cap {max_nodes}")
    fields: set[str] = set()
    operators: set[str] = set()
    for _, expr in assignments:
        _collect(expr, fields, operators)
    _collect(output, fields, operators)
    return ParsedExpression(tuple(assignments), output, nodes, frozenset(fields), frozenset(operators))


def _parse_expr(text: str, defined: set[str]) -> Expr:
    rewritten = FIELD_RE.sub(lambda m: f'FIELD("{m.group(1)}")', text)
    try:
        node = py_ast.parse(rewritten, mode="eval").body
    except SyntaxError as exc:
        raise ParseError(f"syntax error: {exc.msg}") from exc
    return _convert(node, defined)


def _convert(node: py_ast.AST, defined: set[str]) -> Expr:
    if isinstance(node, py_ast.Constant) and isinstance(node.value, (int, float)):
        return Number(float(node.value))
    if isinstance(node, py_ast.UnaryOp) and isinstance(node.op, py_ast.USub):
        child = _convert(node.operand, defined)
        if isinstance(child, Number):
            return Number(-child.value)
    if isinstance(node, py_ast.Name):
        if node.id not in defined:
            raise ParseError(f"undefined name reference: {node.id}")
        return Name(node.id)
    if isinstance(node, py_ast.Call) and isinstance(node.func, py_ast.Name):
        op = node.func.id
        if op == "FIELD":
            if len(node.args) != 1 or not isinstance(node.args[0], py_ast.Constant):
                raise ParseError("invalid field reference")
            field = str(node.args[0].value)
            if field not in ALLOWED_FIELDS:
                raise ParseError(f"unknown field: ${field}")
            return Field(field)
        if op not in OPERATOR_ARITY:
            raise ParseError(f"unknown operator: {op}")
        expected = OPERATOR_ARITY[op]
        if len(node.args) != expected:
            raise ParseError(f"{op} expects {expected} arguments, got {len(node.args)}")
        args = tuple(_convert(arg, defined) for arg in node.args)
        _validate_literal_args(op, args)
        return Call(op, args)
    raise ParseError("unsupported expression syntax")


def _validate_literal_args(op: str, args: tuple[Expr, ...]) -> None:
    for position in INTEGER_LITERAL_ARGS.get(op, ()):
        arg = args[position]
        if not isinstance(arg, Number) or int(arg.value) != arg.value:
            raise ParseError(f"{op} argument {position + 1} must be an integer numeric literal")
        if position in POSITIVE_INTEGER_LITERAL_ARGS.get(op, ()) and arg.value <= 0:
            raise ParseError(f"{op} argument {position + 1} must be a positive integer literal")
        if op == "DELAY" and position == 1 and arg.value < 0:
            raise ParseError("DELAY argument 2 must be a non-negative integer literal")


def _node_count(expr: Expr) -> int:
    if isinstance(expr, Call):
        return 1 + sum(_node_count(arg) for arg in expr.args)
    return 1


def _collect(expr: Expr, fields: set[str], operators: set[str]) -> None:
    if isinstance(expr, Field):
        fields.add(expr.name)
    elif isinstance(expr, Call):
        operators.add(expr.operator)
        for arg in expr.args:
            _collect(arg, fields, operators)
