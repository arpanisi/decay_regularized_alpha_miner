from __future__ import annotations

import pytest

from dsl.evaluator import evaluate
from dsl.parser import ParseError, parse_expression


def test_seed_style_expression_parses_and_evaluates(make_panel) -> None:
    panel = make_panel
    parsed = parse_expression("CS_RANK(MULTIPLY(TS_STD($ret, 20), -1))")
    values = evaluate(parsed, panel)
    assert parsed.node_count == 6
    assert "ret" in parsed.fields
    assert values.notna().sum() > 0


def test_unknown_field_rejected_before_evaluation() -> None:
    with pytest.raises(ParseError, match="unknown field"):
        parse_expression("CS_RANK($label_fwd_10d)")


def test_complexity_cap_rejected() -> None:
    expr = "$close"
    for _ in range(51):
        expr = f"ABS({expr})"
    with pytest.raises(ParseError, match="exceeding cap"):
        parse_expression(expr, max_nodes=50)


def test_event_expression_is_ordinary_field(make_panel) -> None:
    parsed = parse_expression("IF_THEN_ELSE(LT($funda_days_since_disclosure, 5), CS_RANK(DELTA($funda_net_income, 60)), 0)")
    values = evaluate(parsed, make_panel)
    assert values.notna().sum() > 0


def test_window_arguments_rejected_at_parse_time() -> None:
    with pytest.raises(ParseError, match="integer numeric literal"):
        parse_expression("TS_MEAN($close, $volume)")
    with pytest.raises(ParseError, match="positive integer"):
        parse_expression("DELTA($close, 0)")
    with pytest.raises(ParseError, match="non-negative integer"):
        parse_expression("DELAY($close, -1)")


def test_reparse_reevaluate_is_identical(make_panel) -> None:
    expression = "x = DELTA($close, 5)\nCS_RANK(x)"
    first = parse_expression(expression)
    second = parse_expression(expression)
    assert first.node_count == second.node_count
    left = evaluate(first, make_panel)
    right = evaluate(second, make_panel)
    assert left.equals(right)


def test_ts_corr_zero_variance_returns_nan(make_panel) -> None:
    values = evaluate(parse_expression("TS_CORR($volume, $close, 5)"), make_panel)
    assert values.isna().all()


def test_cross_sectional_zscore_zero_variance_returns_nan(make_panel) -> None:
    values = evaluate(parse_expression("CS_ZSCORE($volume)"), make_panel)
    assert values.isna().all()


def test_bucket_and_neutralize_are_cross_sectional(make_panel) -> None:
    bucket = evaluate(parse_expression("CS_BUCKET($market_cap, 3)"), make_panel)
    first_day = bucket.loc[(bucket.index.get_level_values("datetime")[0], slice(None))]
    assert set(first_day.dropna().tolist()) == {1.0, 2.0, 3.0}
    neutralized = evaluate(
        parse_expression("CS_NEUTRALIZE($close, CS_BUCKET($market_cap, 3))"),
        make_panel,
    )
    assert neutralized.dropna().abs().max() == 0.0
