from __future__ import annotations

import pandas as pd

from data.panel import build_panel


def test_point_in_time_fundamental_effective_after_filing_date() -> None:
    crsp = pd.DataFrame(
        {
            "dlycaldt": ["2020-01-02", "2020-01-03", "2020-01-06"],
            "ticker": ["AAA", "AAA", "AAA"],
            "dlyopen": [10, 10, 10],
            "dlyhigh": [11, 11, 11],
            "dlylow": [9, 9, 9],
            "dlyclose": [10, 10, 10],
            "dlyvol": [100, 100, 100],
            "dlycap": [1000, 1000, 1000],
            "dlyret": [0, 0, 0],
        }
    )
    facts = pd.DataFrame(
        {
            "instrument": ["AAA", "AAA", "AAA"],
            "filed": ["2020-01-03", "2020-01-03", "2020-01-03"],
            "field": ["funda_net_income", "funda_book_equity", "funda_shares_out"],
            "value": [2.0, 5.0, 1.0],
        }
    )
    panel = build_panel(crsp, facts)
    assert pd.isna(panel.loc[(pd.Timestamp("2020-01-03"), "AAA"), "funda_net_income"])
    assert panel.loc[(pd.Timestamp("2020-01-06"), "AAA"), "funda_net_income"] == 2.0
    assert panel.loc[(pd.Timestamp("2020-01-06"), "AAA"), "funda_days_since_disclosure"] == 0.0
    assert panel.loc[(pd.Timestamp("2020-01-02"), "AAA"), "funda_days_since_quarter_start"] == 0.0
    assert panel.loc[(pd.Timestamp("2020-01-03"), "AAA"), "funda_days_since_quarter_start"] == 1.0
    assert panel.loc[(pd.Timestamp("2020-01-06"), "AAA"), "funda_days_since_quarter_start"] == 2.0


def test_dividend_yield_uses_trailing_four_disclosure_values() -> None:
    dates = pd.date_range("2020-01-02", periods=8, freq="B")
    crsp = pd.DataFrame(
        {
            "dlycaldt": dates,
            "ticker": ["AAA"] * len(dates),
            "dlyopen": [10] * len(dates),
            "dlyhigh": [11] * len(dates),
            "dlylow": [9] * len(dates),
            "dlyclose": [10] * len(dates),
            "dlyvol": [100] * len(dates),
            "dlycap": [1000] * len(dates),
            "dlyret": [0] * len(dates),
        }
    )
    facts = pd.DataFrame(
        {
            "instrument": ["AAA"] * 8,
            "filed": ["2020-01-02", "2020-01-03", "2020-01-06", "2020-01-07", "2020-01-08", "2020-01-02", "2020-01-02", "2020-01-02"],
            "field": ["funda_div_per_share"] * 5 + ["funda_net_income", "funda_book_equity", "funda_shares_out"],
            "value": [1.0, 2.0, 3.0, 4.0, 5.0, 2.0, 5.0, 1.0],
        }
    )
    panel = build_panel(crsp, facts)
    assert panel.loc[(pd.Timestamp("2020-01-09"), "AAA"), "funda_div_yield"] == 1.4


def test_select_static_universe_filters_share_codes() -> None:
    from data.panel import select_static_universe
    crsp = pd.DataFrame(
        {
            "dlycaldt": ["2015-01-02", "2015-01-02", "2015-01-02"],
            "ticker": ["AAA", "BBB", "CCC"],
            "shrcd": [10, 11, 31],
            "dlycap": [3000, 2000, 5000],
        }
    )
    universe = select_static_universe(crsp, asof_date="2015-01-01", end_date="2015-01-02", universe_size=10)
    assert "AAA" in universe
    assert "BBB" in universe
    assert "CCC" not in universe
