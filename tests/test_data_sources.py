from __future__ import annotations

import pandas as pd

from data.panel import select_static_universe
from data.sec_facts import parse_companyfacts


def test_select_static_universe_prefers_largest_continuous_common_stocks() -> None:
    rows = []
    for dt in pd.date_range("2015-01-02", periods=3, freq="B"):
        rows.extend(
            [
                {"dlycaldt": dt, "ticker": "AAA", "shrcd": 10, "dlycap": 300.0},
                {"dlycaldt": dt, "ticker": "BBB", "shrcd": 11, "dlycap": 200.0},
            ]
        )
    rows.append({"dlycaldt": "2015-01-02", "ticker": "CCC", "shrcd": 10, "dlycap": 500.0})
    universe = select_static_universe(pd.DataFrame(rows), "2015-01-01", "2015-01-06", 2)
    assert universe == ["AAA", "BBB"]


def test_parse_companyfacts_maps_required_tags_to_panel_fields() -> None:
    payload = {
        "facts": {
            "us-gaap": {
                "NetIncomeLoss": {"units": {"USD": [{"filed": "2020-02-01", "val": 10.0}]}},
                "CommonStockSharesOutstanding": {
                    "units": {"shares": [{"filed": "2020-02-01", "val": 5.0}]}
                },
            }
        }
    }
    facts = parse_companyfacts("AAA", payload)
    assert set(facts["field"]) == {"funda_net_income", "funda_shares_out"}
    assert set(facts["instrument"]) == {"AAA"}
