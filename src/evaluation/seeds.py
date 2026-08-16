from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SeedFactor:
    name: str
    expression: str
    rationale: str


SEED_FACTORS = (
    SeedFactor(
        "Momentum",
        "CS_RANK(DELTA($close, 126))",
        "6-month price momentum, cross-sectionally ranked.",
    ),
    SeedFactor(
        "Reversal",
        "CS_RANK(MULTIPLY(DELTA($close, 5), -1))",
        "Short-term reversal.",
    ),
    SeedFactor(
        "Low-volatility",
        "CS_RANK(MULTIPLY(TS_STD($ret, 20), -1))",
        "Inverse trailing realized volatility.",
    ),
    SeedFactor(
        "Value",
        "CS_NEUTRALIZE(CS_RANK(DIVIDE(1, $funda_pe)), CS_BUCKET($market_cap, 10))",
        "Cheap-earnings-relative-to-price, market-cap-neutralized.",
    ),
    SeedFactor(
        "Post-disclosure drift",
        "IF_THEN_ELSE(LT($funda_days_since_disclosure, 5), CS_RANK(DELTA($funda_net_income, 60)), 0)",
        "Post-disclosure drift using recent disclosure timing and changes in net income.",
    ),
)
