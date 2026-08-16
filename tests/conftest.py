from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def make_panel() -> pd.DataFrame:
    dates = pd.date_range("2020-01-01", periods=140, freq="B")
    instruments = ["A", "B", "C"]
    index = pd.MultiIndex.from_product([dates, instruments], names=["datetime", "instrument"])
    panel = pd.DataFrame(index=index)
    base = np.tile(np.array([10.0, 20.0, 30.0]), len(dates))
    trend = np.repeat(np.arange(len(dates), dtype=float), len(instruments))
    panel["open"] = base + trend
    panel["high"] = panel["open"] + 1
    panel["low"] = panel["open"] - 1
    panel["close"] = panel["open"] + 0.5
    panel["volume"] = 1000.0
    panel["market_cap"] = panel["close"] * 1_000_000
    panel["dollar_volume"] = panel["close"] * panel["volume"]
    panel["ret"] = panel.groupby(level="instrument")["close"].pct_change()
    panel["label_fwd_10d"] = panel.groupby(level="instrument")["close"].shift(-10) / panel["close"] - 1
    panel["funda_net_income"] = np.tile([1.0, 2.0, 3.0], len(dates))
    panel["funda_book_equity"] = np.tile([5.0, 6.0, 7.0], len(dates))
    panel["funda_shares_out"] = 1.0
    panel["funda_pe"] = panel["close"] / panel["funda_net_income"]
    panel["funda_pb"] = panel["close"] / panel["funda_book_equity"]
    panel["funda_div_yield"] = 0.01
    panel["funda_days_since_disclosure"] = np.repeat(np.arange(len(dates), dtype=float) % 20, len(instruments))
    panel["funda_days_since_quarter_start"] = np.repeat(np.arange(len(dates), dtype=float), len(instruments))
    return panel
