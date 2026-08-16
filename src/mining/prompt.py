from __future__ import annotations

from dsl.parser import (
    ALLOWED_FIELDS,
    INTEGER_LITERAL_ARGS,
    OPERATOR_ARITY,
    POSITIVE_INTEGER_LITERAL_ARGS,
)
from evaluation.seeds import SEED_FACTORS

# Informational one-line descriptions for the $fields. The field NAMES themselves are driven by
# ALLOWED_FIELDS (the parser's validation set) so the prompt can never drift from what the parser
# accepts; the descriptions are purely illustrative and have no parsing effect.
FIELD_DESCRIPTIONS: dict[str, str] = {
    "open": "opening price",
    "high": "daily high price",
    "low": "daily low price",
    "close": "closing price",
    "volume": "trading volume",
    "market_cap": "market capitalization",
    "dollar_volume": "price x volume",
    "ret": "daily return",
    "funda_pe": "price-to-earnings ratio (point-in-time)",
    "funda_pb": "price-to-book ratio (point-in-time)",
    "funda_div_yield": "dividend yield (point-in-time)",
    "funda_days_since_disclosure": "trading days since the latest disclosure",
    "funda_days_since_quarter_start": "trading days since the start of the quarter",
    "funda_net_income": "raw net income (point-in-time)",
    "funda_book_equity": "raw book equity (point-in-time)",
    "funda_shares_out": "raw shares outstanding (point-in-time)",
}

# Argument-role labels per operator. Operator NAMES and ARITIES are driven by OPERATOR_ARITY (the
# parser's operator table) and validated against it at build time, so the two can never drift.
OPERATOR_SIGNATURES: dict[str, tuple[str, ...]] = {
    # 9 time-series operators
    "TS_MEAN": ("x", "window"),
    "TS_SUM": ("x", "window"),
    "TS_MIN": ("x", "window"),
    "TS_MAX": ("x", "window"),
    "TS_STD": ("x", "window"),
    "TS_RANK": ("x", "window"),
    "DELTA": ("x", "period"),
    "DELAY": ("x", "lag"),
    "TS_CORR": ("x", "y", "window"),
    # 5 cross-sectional operators
    "CS_RANK": ("x",),
    "CS_ZSCORE": ("x",),
    "CS_WINSORIZE": ("x", "lo_pct", "hi_pct"),
    "CS_NEUTRALIZE": ("x", "by"),
    "CS_BUCKET": ("x", "n_buckets"),
    # 8 arithmetic operators
    "ADD": ("a", "b"),
    "SUBTRACT": ("a", "b"),
    "MULTIPLY": ("a", "b"),
    "DIVIDE": ("a", "b"),
    "ABS": ("a",),
    "SIGN": ("a",),
    "LOG": ("a",),
    "POW": ("a", "exponent"),
    # 3 conditional/comparison operators
    "GT": ("a", "b"),
    "LT": ("a", "b"),
    "IF_THEN_ELSE": ("condition", "then_value", "else_value"),
    # 2 event operators
    "TS_SINCE": ("event",),
    "TS_COUNT": ("condition", "window"),
}

# Operators whose given argument must be an integer numeric literal (windows/periods/ranks), and
# the subset that must additionally be positive. DELAY's lag is integer but may be zero (handled
# as non-negative by the parser). All derived from the parser's own literal-argument tables.
INTEGER_LITERAL_NOTE = (
    "Integer-literal arguments (must be whole numbers, not $fields or sub-expressions): "
    "the window/period of TS_MEAN, TS_SUM, TS_MIN, TS_MAX, TS_STD, TS_RANK, DELTA, TS_CORR and "
    "TS_COUNT, the n_buckets of CS_BUCKET, and the exponent of POW must be POSITIVE integers; "
    "the lag of DELAY must be a non-negative integer."
)


def _validated_operator_signatures() -> dict[str, tuple[str, ...]]:
    if set(OPERATOR_SIGNATURES) != set(OPERATOR_ARITY):
        missing = set(OPERATOR_ARITY) - set(OPERATOR_SIGNATURES)
        extra = set(OPERATOR_SIGNATURES) - set(OPERATOR_ARITY)
        raise AssertionError(
            f"operator signature table out of sync with parser: missing={sorted(missing)} extra={sorted(extra)}"
        )
    for op, args in OPERATOR_SIGNATURES.items():
        if len(args) != OPERATOR_ARITY[op]:
            raise AssertionError(
                f"signature for {op} has {len(args)} args but parser arity is {OPERATOR_ARITY[op]}"
            )
    return OPERATOR_SIGNATURES


def build_grammar_prompt() -> str:
    """
    Builds the mining system prompt from the same source of truth the parser validates against:
    the DSL's operator table (OPERATOR_ARITY) and allowed-field set (ALLOWED_FIELDS), plus the
    already-implemented seed factors as worked examples. Nothing here is hand-copied prose that
    could drift out of sync with what parse_expression actually accepts.
    """
    signatures = _validated_operator_signatures()

    field_lines = "\n".join(
        f"- ${name}: {FIELD_DESCRIPTIONS.get(name, '')}"
        for name in sorted(ALLOWED_FIELDS)
    )
    op_lines = "\n".join(
        f"- {op}({', '.join(signatures[op])})" for op in sorted(OPERATOR_ARITY)
    )
    example_lines = "\n".join(f"- {seed.name}: {seed.expression}" for seed in SEED_FACTORS)

    return (
        "You are an alpha-mining engine. Propose alpha expressions in the fixed alpha-mining DSL.\n"
        "\n"
        "AVAILABLE $FIELDS (use exactly these $field names; no other $field is valid):\n"
        f"{field_lines}\n"
        "\n"
        f"AVAILABLE OPERATORS (use exactly these operator names; no operator outside this list is valid):\n"
        f"{op_lines}\n"
        "\n"
        f"{INTEGER_LITERAL_NOTE}\n"
        "\n"
        "SYNTAX: expressions use Python-like call syntax OPERATOR(arg1, arg2). Fields are written "
        "with a leading $ (e.g. $close). A proposal is a single expression, or one or more named "
        "assignments (name = expr) followed by a final bare output expression.\n"
        "\n"
        "WORKED EXAMPLES (already-implemented seed factors, all syntactically valid):\n"
        f"{example_lines}\n"
        "\n"
        "Return only JSON with required keys rationale and expression. "
        "The expression must use the fixed alpha-mining grammar and valid $fields only."
    )
