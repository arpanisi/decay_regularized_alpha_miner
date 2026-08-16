from __future__ import annotations

from typing import Any

import pandas as pd


TAG_TO_FIELD = {
    "NetIncomeLoss": "funda_net_income",
    "StockholdersEquity": "funda_book_equity",
    "CommonStockSharesOutstanding": "funda_shares_out",
    "EntityCommonStockSharesOutstanding": "funda_shares_out",
    "Revenues": "funda_revenue",
    "Assets": "funda_assets",
    "Liabilities": "funda_liabilities",
    "CommonStockDividendsPerShareDeclared": "funda_div_per_share",
}


def parse_companyfacts(instrument: str, payload: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    facts = payload.get("facts", {}).get("us-gaap", {})
    for tag, field in TAG_TO_FIELD.items():
        tag_payload = facts.get(tag)
        if not tag_payload:
            continue
        units = tag_payload.get("units", {})
        for observations in units.values():
            for obs in observations:
                if "filed" not in obs or "val" not in obs:
                    continue
                rows.append(
                    {
                        "instrument": instrument,
                        "filed": obs["filed"],
                        "field": field,
                        "value": obs["val"],
                    }
                )
    if not rows:
        return pd.DataFrame(columns=["instrument", "filed", "field", "value"])
    frame = pd.DataFrame(rows)
    frame["filed"] = pd.to_datetime(frame["filed"])
    frame = frame.sort_values(["instrument", "field", "filed"])
    return frame.drop_duplicates(["instrument", "field", "filed"], keep="last")
