from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd


PRICE_COLUMNS = {
    "dlyopen": "open",
    "dlyhigh": "high",
    "dlylow": "low",
    "dlyclose": "close",
    "dlyvol": "volume",
    "dlycap": "market_cap",
    "dlyret": "ret",
}

FUNDAMENTAL_COLUMNS = [
    "funda_net_income",
    "funda_book_equity",
    "funda_shares_out",
    "funda_revenue",
    "funda_assets",
    "funda_liabilities",
    "funda_div_per_share",
]

REQUIRED_CRSP_COLUMNS = {"dlycaldt", "ticker", *PRICE_COLUMNS.keys()}


@dataclass(frozen=True)
class DateSplit:
    train_start: str = "2015-01-01"
    train_end: str = "2020-12-31"
    validation_start: str = "2021-01-01"
    validation_end: str = "2023-12-31"


def normalize_crsp_frame(frame: pd.DataFrame) -> pd.DataFrame:
    missing = REQUIRED_CRSP_COLUMNS.difference(frame.columns)
    if missing:
        raise ValueError(f"CRSP frame missing columns: {sorted(missing)}")
    data = frame.rename(columns=PRICE_COLUMNS).copy()
    data["datetime"] = pd.to_datetime(data["dlycaldt"])
    data["instrument"] = data["ticker"].astype(str)
    data["dollar_volume"] = data["close"] * data["volume"]
    data = data.sort_values(["datetime", "instrument", "market_cap"], ascending=[True, True, False])
    data = data.drop_duplicates(["datetime", "instrument"], keep="first")
    data = data.set_index(["datetime", "instrument"]).sort_index()
    data["label_fwd_10d"] = (
        data.groupby(level="instrument")["close"].shift(-10) / data["close"] - 1.0
    )
    return data


def select_static_universe(
    crsp_frame: pd.DataFrame,
    asof_date: str = "2015-01-01",
    end_date: str = "2023-12-31",
    universe_size: int = 2750,
) -> list[str]:
    data = crsp_frame.copy()
    data["dlycaldt"] = pd.to_datetime(data["dlycaldt"])
    if "shrcd" in data.columns:
        data = data[data["shrcd"].isin([10, 11])]
    start = pd.Timestamp(asof_date)
    end = pd.Timestamp(end_date)
    trading_days = pd.Index(sorted(data.loc[data["dlycaldt"] >= start, "dlycaldt"].unique()))
    if trading_days.empty:
        raise ValueError("no trading day on or after universe as-of date")
    selection_day = trading_days[0]
    coverage_days = set(pd.Index(sorted(data.loc[(data["dlycaldt"] >= start) & (data["dlycaldt"] <= end), "dlycaldt"].unique())))
    counts = data[(data["dlycaldt"] >= start) & (data["dlycaldt"] <= end)].groupby("ticker")["dlycaldt"].nunique()
    continuous = set(counts[counts == len(coverage_days)].index)
    ranked = (
        data[data["dlycaldt"] == selection_day]
        .sort_values("dlycap", ascending=False)
        ["ticker"]
        .astype(str)
        .drop_duplicates()
    )
    selected = [ticker for ticker in ranked if ticker in continuous]
    if len(selected) < universe_size:
        selected = list(ranked.head(universe_size))
    return selected[:universe_size]


def expand_point_in_time_fundamentals(
    trading_index: pd.MultiIndex,
    facts: pd.DataFrame,
) -> pd.DataFrame:
    if not {"instrument", "filed", "field", "value"}.issubset(facts.columns):
        raise ValueError("facts must contain instrument, filed, field, value")
    dates = pd.Series(trading_index.get_level_values("datetime").unique()).sort_values()
    instruments = trading_index.get_level_values("instrument").unique()
    output = pd.DataFrame(index=trading_index, columns=FUNDAMENTAL_COLUMNS, dtype=float)
    effective_dates: dict[tuple[str, str], pd.Timestamp] = {}
    for instrument in instruments:
        inst_dates = dates
        inst_facts = facts[facts["instrument"] == instrument].copy()
        if inst_facts.empty:
            continue
        inst_facts["filed"] = pd.to_datetime(inst_facts["filed"])
        for field in FUNDAMENTAL_COLUMNS:
            field_facts = inst_facts[inst_facts["field"] == field].sort_values("filed")
            if field_facts.empty:
                continue
            series = pd.Series(index=inst_dates, dtype=float)
            for row in field_facts.itertuples(index=False):
                later = inst_dates[inst_dates > row.filed]
                if later.empty:
                    continue
                effective = later.iloc[0]
                series.loc[effective] = float(row.value)
                effective_dates[(instrument, field)] = effective
            series = series.ffill()
            idx = pd.MultiIndex.from_product([inst_dates, [instrument]], names=["datetime", "instrument"])
            output.loc[idx, field] = series.to_numpy()
    return output


def add_valuation_and_event_fields(panel: pd.DataFrame, facts: pd.DataFrame) -> pd.DataFrame:
    out = panel.copy()
    out["funda_pe"] = out["close"] / (out["funda_net_income"] / out["funda_shares_out"])
    out["funda_pb"] = out["close"] / (out["funda_book_equity"] / out["funda_shares_out"])
    out["funda_div_yield"] = _trailing_four_disclosure_dividends(out.index, facts) / out["close"]
    out["funda_days_since_quarter_start"] = _trading_days_since_quarter_start(out.index)
    out["funda_days_since_disclosure"] = _days_since_disclosure(out.index, facts)
    return out.replace([np.inf, -np.inf], np.nan)


def build_panel(crsp_frame: pd.DataFrame, facts: pd.DataFrame) -> pd.DataFrame:
    panel = normalize_crsp_frame(crsp_frame)
    expanded = expand_point_in_time_fundamentals(panel.index, facts)
    panel = panel.join(expanded)
    return add_valuation_and_event_fields(panel, facts)


def _trading_days_since_quarter_start(index: pd.MultiIndex) -> np.ndarray:
    result = pd.Series(np.nan, index=index, dtype=float)
    for instrument in index.get_level_values("instrument").unique():
        inst_dates = pd.Index(index.get_level_values("datetime")[index.get_level_values("instrument") == instrument])
        quarter_counts: dict[date, int] = {}
        values = []
        for dt in inst_dates:
            q_month = ((dt.month - 1) // 3) * 3 + 1
            q_start = date(dt.year, q_month, 1)
            count = quarter_counts.get(q_start, 0)
            values.append(float(count))
            quarter_counts[q_start] = count + 1
        result.loc[pd.MultiIndex.from_product([inst_dates, [instrument]], names=index.names)] = values
    return result.to_numpy()


def _trailing_four_disclosure_dividends(index: pd.MultiIndex, facts: pd.DataFrame) -> pd.Series:
    result = pd.Series(np.nan, index=index, dtype=float)
    if facts.empty:
        return result
    divs = facts[facts["field"] == "funda_div_per_share"].copy()
    if divs.empty:
        return result
    divs["filed"] = pd.to_datetime(divs["filed"])
    for instrument, group in divs.groupby("instrument"):
        inst_dates = pd.Index(index.get_level_values("datetime")[index.get_level_values("instrument") == instrument])
        events: list[tuple[pd.Timestamp, float]] = []
        for row in group.sort_values("filed").itertuples(index=False):
            later = inst_dates[inst_dates > row.filed]
            if not later.empty:
                events.append((later[0], float(row.value)))
        active: list[float] = []
        values = []
        event_pos = 0
        for dt in inst_dates:
            while event_pos < len(events) and events[event_pos][0] == dt:
                active.append(events[event_pos][1])
                active = active[-4:]
                event_pos += 1
            values.append(float(sum(active)) if active else np.nan)
        result.loc[pd.MultiIndex.from_product([inst_dates, [instrument]], names=index.names)] = values
    return result


def _days_since_disclosure(index: pd.MultiIndex, facts: pd.DataFrame) -> np.ndarray:
    result = pd.Series(np.nan, index=index, dtype=float)
    if facts.empty:
        return result.to_numpy()
    filed = facts[["instrument", "filed"]].drop_duplicates().copy()
    filed["filed"] = pd.to_datetime(filed["filed"])
    for instrument, group in filed.groupby("instrument"):
        inst_dates = pd.Index(index.get_level_values("datetime")[index.get_level_values("instrument") == instrument])
        effective = []
        for filed_dt in sorted(group["filed"]):
            later = inst_dates[inst_dates > filed_dt]
            if not later.empty:
                effective.append(later[0])
        last: pd.Timestamp | None = None
        counter = np.nan
        for dt in inst_dates:
            if dt in effective:
                last = dt
                counter = 0.0
            elif last is not None:
                counter += 1.0
            result.loc[(dt, instrument)] = counter
    return result.to_numpy()
